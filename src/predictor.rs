//! Public static-image SAM2 predictor.

use std::{collections::BTreeMap, path::Path, time::Instant};

use candle_core::{DType, Device, Result, Tensor};
use rayon::prelude::*;

use crate::{
    BackboneOutput, ImageEncoder, ImageEncoderProfile, ImageGeometry, MaskDecoder,
    MaskDecoderConfig, MaskDecoderOutput, MaskDecoderProfile, PromptEncoder, Sam2Base, Sam2Config,
    Sam2FrameOutput, Sam2ImageTransforms, Sam2Variant, VideoInferenceState,
    ops::{
        MaskRle, batch_ranges, build_all_layer_point_grids, calculate_stability_score,
        generate_crop_boxes, restore_video_masks, uncrop_rle,
    },
};

/// Counts the two thresholded areas needed by the AMG stability score while
/// bilinearly restoring a mask, without materializing the restored masks.
/// The interpolation arithmetic deliberately mirrors Candle's CPU
/// `upsample_bilinear2d(..., false)` implementation so the fused path makes
/// the same threshold decisions as the composed operations.
#[derive(Debug)]
struct CpuBilinearStabilityCounts {
    output_height: usize,
    output_width: usize,
    high_threshold: f32,
    low_threshold: f32,
}

impl candle_core::CustomOp1 for CpuBilinearStabilityCounts {
    fn name(&self) -> &'static str {
        "sam2-cpu-bilinear-stability-counts-f32"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let (masks, channels, input_height, input_width) = layout.shape().dims4()?;
        if channels != 1 || masks == 0 || input_height == 0 || input_width == 0 {
            candle_core::bail!("bilinear stability counts require non-empty one-channel masks")
        }
        if self.output_height == 0 || self.output_width == 0 {
            candle_core::bail!("bilinear stability counts require a non-empty output")
        }
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("bilinear stability counts require contiguous masks")
        };
        let input = &storage.as_slice::<f32>()?[start..end];

        let height_scale = input_height as f64 / self.output_height as f64;
        let width_scale = input_width as f64 / self.output_width as f64;
        let height_indices = (0..self.output_height)
            .map(|output_y| {
                let source = (height_scale * (output_y as f64 + 0.5) - 0.5).max(0.0);
                let low = source.floor() as usize;
                (
                    low,
                    (low + 1).min(input_height - 1),
                    (source - low as f64).clamp(0.0, 1.0),
                )
            })
            .collect::<Vec<_>>();
        let width_indices = (0..self.output_width)
            .map(|output_x| {
                let source = (width_scale * (output_x as f64 + 0.5) - 0.5).max(0.0);
                let low = source.floor() as usize;
                (
                    low,
                    (low + 1).min(input_width - 1),
                    (source - low as f64).clamp(0.0, 1.0),
                )
            })
            .collect::<Vec<_>>();

        let pixels_per_mask = input_height * input_width;
        let mut counts = vec![0u32; masks * 2];
        counts
            .par_chunks_mut(2)
            .enumerate()
            .for_each(|(mask, counts)| {
                let input = &input[mask * pixels_per_mask..(mask + 1) * pixels_per_mask];
                let mut high = 0u32;
                let mut low = 0u32;
                for &(y0, y1, weight_y) in &height_indices {
                    let row0 = y0 * input_width;
                    let row1 = y1 * input_width;
                    for &(x0, x1, weight_x) in &width_indices {
                        let v00 = input[row0 + x0] as f64;
                        let v10 = input[row0 + x1] as f64;
                        let v01 = input[row1 + x0] as f64;
                        let v11 = input[row1 + x1] as f64;
                        let top = v00 * (1.0 - weight_x) + v10 * weight_x;
                        let bottom = v01 * (1.0 - weight_x) + v11 * weight_x;
                        // Candle converts the interpolated f64 back to the
                        // input dtype before the following threshold op.
                        let value = (top * (1.0 - weight_y) + bottom * weight_y) as f32;
                        high += u32::from(value > self.high_threshold);
                        low += u32::from(value > self.low_threshold);
                    }
                }
                counts[0] = high;
                counts[1] = low;
            });
        Ok((
            candle_core::CpuStorage::U32(counts),
            candle_core::Shape::from_dims(&[masks, 2]),
        ))
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn sam2_threshold_pack_u32_f32(
        input: *const f32,
        output: *mut u32,
        pixels_per_mask: u64,
        words_per_mask: u64,
        output_words: u64,
        threshold: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    fn sam2_bilinear_stability_counts_f32(
        input: *const f32,
        counts: *mut u32,
        masks: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        high_threshold: f32,
        low_threshold: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaBilinearStabilityCounts {
    output_height: usize,
    output_width: usize,
    high_threshold: f32,
    low_threshold: f32,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for CudaBilinearStabilityCounts {
    fn name(&self) -> &'static str {
        "sam2-bilinear-stability-counts-f32"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("bilinear stability counts are only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        if storage.dtype() != DType::F32 {
            candle_core::bail!("bilinear stability counts require f32")
        }
        let (masks, channels, input_height, input_width) = layout.shape().dims4()?;
        if channels != 1 || masks == 0 {
            candle_core::bail!("bilinear stability counts require non-empty one-channel masks")
        }
        let Some((start, _)) = layout.contiguous_offsets() else {
            candle_core::bail!("bilinear stability counts require contiguous masks")
        };
        let device = storage.device();
        let stream = device.cuda_stream();
        let input = storage.as_cuda_slice::<f32>()?.slice(start..);
        // SAFETY: the kernel writes two counters for every validated mask.
        let mut output = unsafe { device.alloc::<u32>(masks * 2)? };
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_bilinear_stability_counts_f32(
                input_ptr as *const f32,
                output_ptr as *mut u32,
                masks as i32,
                input_height as i32,
                input_width as i32,
                self.output_height as i32,
                self.output_width as i32,
                self.high_threshold,
                self.low_threshold,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("bilinear stability CUDA launch failed with status {status}")
        }
        drop((input_guard, output_guard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(output),
                device: device.clone(),
            },
            candle_core::Shape::from_dims(&[masks, 2]),
        ))
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaThresholdPack {
    threshold: f32,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for CudaThresholdPack {
    fn name(&self) -> &'static str {
        "sam2-threshold-pack-u32-f32"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("threshold packing is only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        if storage.dtype() != DType::F32 {
            candle_core::bail!("threshold packing requires f32")
        }
        let (masks, channels, height, width) = layout.shape().dims4()?;
        if channels != 1 {
            candle_core::bail!("threshold packing requires one channel")
        }
        let Some((start, _)) = layout.contiguous_offsets() else {
            candle_core::bail!("threshold packing requires contiguous masks")
        };
        let pixels = height * width;
        let words = pixels.div_ceil(32);
        let output_words = masks * words;
        let output_shape = candle_core::Shape::from_dims(&[masks, words]);
        let dev = storage.device();
        let stream = dev.cuda_stream();
        let input = storage.as_cuda_slice::<f32>()?.slice(start..);
        // SAFETY: the kernel writes each word in the validated output shape.
        let mut output = unsafe { dev.alloc::<u32>(output_words)? };
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_threshold_pack_u32_f32(
                input_ptr as *const f32,
                output_ptr as *mut u32,
                pixels as u64,
                words as u64,
                output_words as u64,
                self.threshold,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("threshold-pack CUDA launch failed with status {status}")
        }
        drop((input_guard, output_guard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::U32(output),
                device: dev.clone(),
            },
            output_shape,
        ))
    }
}

/// A point in pixel coordinates in the model's 1024x1024 input space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointPrompt {
    pub x: f32,
    pub y: f32,
    pub positive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxPrompt {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

#[derive(Clone, Debug)]
pub struct Sam2ImagePredictor {
    device: Device,
    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    no_mem_embed: Tensor,
    transforms: Sam2ImageTransforms,
    cached_images: Option<CachedImages>,
}

#[derive(Clone, Debug)]
struct CachedImages {
    features: BackboneOutput,
    geometries: Vec<crate::ImageGeometry>,
    is_batch: bool,
}

/// Detached encoding of one preprocessed image. Model-family training and persistent-cache policy
/// stay outside sam2-rs; this value only exposes the faithful translated encoder boundary.
#[derive(Clone, Debug)]
pub struct EncodedImage {
    features: BackboneOutput,
    geometry: crate::ImageGeometry,
}

impl EncodedImage {
    pub fn estimated_bytes(&self) -> u64 {
        let bytes = |tensor: &Tensor| {
            tensor
                .elem_count()
                .saturating_mul(tensor.dtype().size_in_bytes()) as u64
        };
        bytes(&self.features.vision_features)
            .saturating_add(self.features.vision_pos_enc.iter().map(bytes).sum::<u64>())
            .saturating_add(self.features.backbone_fpn.iter().map(bytes).sum::<u64>())
    }
}

/// Tensor-native prompt batch for cached image prediction. Coordinates and
/// boxes may be normalized from original pixels or supplied directly in model
/// coordinates. Leading batch dimensions allow several prompt sets to reuse
/// one cached image embedding.
#[derive(Clone, Debug, Default)]
pub struct ImagePromptBatch {
    pub point_coords: Option<Tensor>,
    pub point_labels: Option<Tensor>,
    pub boxes: Option<Tensor>,
    pub mask_input: Option<Tensor>,
    pub normalize_coords: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePredictorConfig {
    pub mask_threshold: f64,
    pub max_hole_area: usize,
    pub max_sprinkle_area: usize,
    pub decoder: MaskDecoderConfig,
}

impl Default for ImagePredictorConfig {
    fn default() -> Self {
        Self {
            mask_threshold: 0.,
            max_hole_area: 0,
            max_sprinkle_area: 0,
            decoder: MaskDecoderConfig::default(),
        }
    }
}

/// Image-predictor result with both native low-resolution logits and logits
/// restored to the caller's original RGB pixel dimensions.
#[derive(Clone, Debug)]
pub struct ImagePrediction {
    pub low_res: MaskDecoderOutput,
    pub masks: Tensor,
}

/// Video/z-slice result with logits restored to the original RGB dimensions.
#[derive(Clone, Debug)]
pub struct VideoFramePrediction {
    pub frame: Sam2FrameOutput,
    pub masks: Tensor,
}

#[derive(Clone, Debug)]
pub struct AutomaticMask {
    pub segmentation: AutomaticMaskSegmentation,
    pub bbox: [f32; 4],
    pub area: usize,
    pub predicted_iou: f32,
    pub point: [f32; 2],
    pub stability_score: f32,
    pub crop_box: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AutomaticMaskSegmentation {
    Binary {
        width: usize,
        height: usize,
        row_major: Vec<bool>,
    },
    UncompressedRle(MaskRle),
    CocoRle {
        width: usize,
        height: usize,
        counts: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutomaticMaskOutputMode {
    BinaryMask,
    UncompressedRle,
    CocoRle,
}

#[derive(Clone, Debug)]
pub struct AutomaticMaskConfig {
    pub points_per_side: Option<usize>,
    pub point_grids: Option<Vec<Vec<[f32; 2]>>>,
    pub points_per_batch: usize,
    pub pred_iou_thresh: f32,
    pub stability_score_thresh: f32,
    pub stability_score_offset: f32,
    pub mask_threshold: f32,
    pub box_nms_thresh: f32,
    pub crop_n_layers: usize,
    pub crop_nms_thresh: f32,
    pub crop_overlap_ratio: f32,
    pub crop_n_points_downscale_factor: usize,
    pub min_mask_region_area: usize,
    pub output_mode: AutomaticMaskOutputMode,
    pub use_m2m: bool,
    pub multimask_output: bool,
}

impl Default for AutomaticMaskConfig {
    fn default() -> Self {
        Self {
            points_per_side: Some(32),
            point_grids: None,
            points_per_batch: 64,
            pred_iou_thresh: 0.8,
            stability_score_thresh: 0.95,
            stability_score_offset: 1.,
            mask_threshold: 0.,
            box_nms_thresh: 0.7,
            crop_n_layers: 0,
            crop_nms_thresh: 0.7,
            crop_overlap_ratio: 512. / 1500.,
            crop_n_points_downscale_factor: 1,
            min_mask_region_area: 0,
            output_mode: AutomaticMaskOutputMode::BinaryMask,
            use_m2m: false,
            multimask_output: true,
        }
    }
}

/// Full-image grid automatic mask generation backed by a cached Hiera image embedding.
#[derive(Clone, Debug)]
pub struct Sam2AutomaticMaskGenerator {
    predictor: Sam2ImagePredictor,
    config: AutomaticMaskConfig,
}

/// Stateful convenience wrapper for temporal video or ordered z-slice
/// inference. It owns the typed memory bank, while [`Sam2Base`] remains
/// available for applications that need to manage several object states.
#[derive(Clone, Debug)]
pub struct Sam2VideoPredictor {
    model: Sam2Base,
    state: VideoInferenceState,
}

/// Object-keyed temporal tracker. Each object owns an independent SAM2 memory
/// bank while sharing one checkpoint-backed model instance, making object IDs
/// explicit for video and z-slice applications.
#[derive(Clone, Debug)]
pub struct Sam2MultiObjectVideoPredictor {
    model: Sam2Base,
    states: BTreeMap<u32, VideoInferenceState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoPredictorConfig {
    pub fill_hole_area: usize,
    pub non_overlapping_masks: bool,
    pub clear_non_conditioning_memory_around_input: bool,
    pub add_all_corrections_as_conditioning: bool,
}

impl Default for VideoPredictorConfig {
    fn default() -> Self {
        Self {
            fill_hole_area: 8,
            non_overlapping_masks: false,
            clear_non_conditioning_memory_around_input: false,
            add_all_corrections_as_conditioning: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InteractiveVideoSession {
    frames: Vec<Tensor>,
    state: VideoInferenceState,
    model_states: BTreeMap<u32, VideoInferenceState>,
    outputs: BTreeMap<u32, BTreeMap<usize, StoredVideoOutput>>,
}

/// Minimal per-object/frame output retained for later interactive edits.
/// High-resolution masks and conditioned image features are intentionally not
/// kept for every propagated frame; the reference state stores only the
/// low-resolution prediction and temporal-memory inputs needed for reuse.
#[derive(Clone, Debug)]
pub struct StoredVideoOutput {
    pub low_res_masks: Tensor,
    pub object_score_logits: Tensor,
    pub object_pointer: Tensor,
}

impl InteractiveVideoSession {
    pub fn state(&self) -> &VideoInferenceState {
        &self.state
    }

    pub fn output(&self, object_id: u32, frame_index: usize) -> Option<&StoredVideoOutput> {
        self.outputs
            .get(&object_id)
            .and_then(|outputs| outputs.get(&frame_index))
    }
}

#[derive(Clone, Debug)]
pub struct InteractiveVideoOutput {
    pub frame_index: usize,
    pub object_ids: Vec<u32>,
    pub masks: Tensor,
}

/// Synchronized broad-stage attribution for one video propagation. This is
/// intentionally separate from the hot path so ordinary inference does not
/// read clocks or add per-frame synchronizations.
#[derive(Clone, Debug, Default)]
pub struct VideoPropagationProfile {
    pub preflight_ms: f64,
    pub image_encoder_ms: Vec<f64>,
    pub temporal_and_heads_ms: Vec<f64>,
    pub restore_masks_ms: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RemoveObjectOutput {
    pub object_ids: Vec<u32>,
    pub updated_frames: Vec<InteractiveVideoOutput>,
}

#[derive(Clone, Debug)]
pub struct Sam2InteractiveVideoPredictor {
    model: Sam2Base,
    config: VideoPredictorConfig,
}

impl Sam2VideoPredictor {
    /// Downloads the published checkpoint through the Hugging Face cache and
    /// constructs the predictor on CPU.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained(variant: Sam2Variant) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1(variant, path)
    }

    pub fn load_sam2_1(variant: Sam2Variant, path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            model: Sam2Base::load_sam2_1_with_decoder_config(
                variant,
                path,
                MaskDecoderConfig::default(),
            )?,
            state: VideoInferenceState::new(),
        })
    }

    pub fn load_sam2_1_hiera_tiny(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_sam2_1(Sam2Variant::HieraTiny, path)
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn state(&self) -> &VideoInferenceState {
        &self.state
    }

    /// Adds an interacted conditioning frame. `frame_index` can be a video
    /// timestamp or a z-slice index.
    pub fn add_conditioning_frame(
        &mut self,
        image: &Tensor,
        frame_index: usize,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
    ) -> Result<Sam2FrameOutput> {
        self.model.track(
            image,
            frame_index,
            true,
            &mut self.state,
            points,
            bounding_box,
            multimask_output,
            false,
        )
    }

    /// Offline-sequence variant of [`Self::add_conditioning_frame`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_conditioning_frame_with_num_frames(
        &mut self,
        image: &Tensor,
        frame_index: usize,
        num_frames: usize,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
    ) -> Result<Sam2FrameOutput> {
        self.model.track_with_num_frames(
            image,
            frame_index,
            true,
            &mut self.state,
            points,
            bounding_box,
            multimask_output,
            false,
            Some(num_frames),
        )
    }

    /// Adds a binary `B×1×1024×1024` annotation mask as a conditioning frame.
    /// This is the native counterpart of SAM2's direct mask-input workflow.
    pub fn add_mask_conditioning_frame(
        &mut self,
        image: &Tensor,
        mask: &Tensor,
        frame_index: usize,
        num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
        self.model.track_mask(
            image,
            mask,
            frame_index,
            true,
            &mut self.state,
            false,
            num_frames,
        )
    }

    /// Propagates one subsequent frame. Set `reverse` when traversing a video
    /// or z-stack toward decreasing indices.
    pub fn propagate(
        &mut self,
        image: &Tensor,
        frame_index: usize,
        reverse: bool,
    ) -> Result<Sam2FrameOutput> {
        self.model.track(
            image,
            frame_index,
            false,
            &mut self.state,
            &[],
            None,
            false,
            reverse,
        )
    }

    /// Offline-sequence variant of [`Self::propagate`].
    pub fn propagate_with_num_frames(
        &mut self,
        image: &Tensor,
        frame_index: usize,
        num_frames: usize,
        reverse: bool,
    ) -> Result<Sam2FrameOutput> {
        self.model.track_with_num_frames(
            image,
            frame_index,
            false,
            &mut self.state,
            &[],
            None,
            false,
            reverse,
            Some(num_frames),
        )
    }

    /// RGB8/original-coordinate counterpart of [`Self::add_conditioning_frame`].
    pub fn add_conditioning_rgb8(
        &mut self,
        rgb_hwc: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
    ) -> Result<VideoFramePrediction> {
        let transforms = Sam2ImageTransforms::default();
        let prepared = transforms.prepare_rgb8(rgb_hwc, width, height, &Device::Cpu)?;
        let points: Vec<_> = points
            .iter()
            .map(|point| {
                let (x, y) = prepared.geometry.model_point(point.x, point.y);
                PointPrompt {
                    x,
                    y,
                    positive: point.positive,
                }
            })
            .collect();
        let bounding_box = bounding_box.map(|bounds| {
            let (x0, y0) = prepared.geometry.model_point(bounds.x0, bounds.y0);
            let (x1, y1) = prepared.geometry.model_point(bounds.x1, bounds.y1);
            BoxPrompt { x0, y0, x1, y1 }
        });
        let frame = self.add_conditioning_frame(
            &prepared.tensor,
            frame_index,
            &points,
            bounding_box,
            multimask_output,
        )?;
        let masks = transforms.restore_masks(&frame.low_res_masks, prepared.geometry)?;
        Ok(VideoFramePrediction { frame, masks })
    }

    /// RGB8 counterpart of [`Self::propagate`], including original-resolution
    /// output restoration for video frames or z-slices.
    pub fn propagate_rgb8(
        &mut self,
        rgb_hwc: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
        reverse: bool,
    ) -> Result<VideoFramePrediction> {
        let transforms = Sam2ImageTransforms::default();
        let prepared = transforms.prepare_rgb8(rgb_hwc, width, height, &Device::Cpu)?;
        let frame = self.propagate(&prepared.tensor, frame_index, reverse)?;
        let masks = transforms.restore_masks(&frame.low_res_masks, prepared.geometry)?;
        Ok(VideoFramePrediction { frame, masks })
    }
}

impl Sam2MultiObjectVideoPredictor {
    /// Downloads the published checkpoint through the Hugging Face cache and
    /// constructs the predictor on CPU.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained(variant: Sam2Variant) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1(variant, path)
    }

    pub fn load_sam2_1(variant: Sam2Variant, path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            model: Sam2Base::load_sam2_1(variant, path)?,
            states: BTreeMap::new(),
        })
    }

    pub fn load_sam2_1_hiera_tiny(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_sam2_1(Sam2Variant::HieraTiny, path)
    }

    pub fn object_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.states.keys().copied()
    }

    pub fn remove_object(&mut self, object_id: u32) -> Option<VideoInferenceState> {
        self.states.remove(&object_id)
    }

    pub fn add_conditioning_frame(
        &mut self,
        object_id: u32,
        image: &Tensor,
        frame_index: usize,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
        num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
        let state = self.states.entry(object_id).or_default();
        self.model.track_with_num_frames(
            image,
            frame_index,
            true,
            state,
            points,
            bounding_box,
            multimask_output,
            false,
            num_frames,
        )
    }

    /// Propagates all currently known objects on the same frame.
    pub fn propagate_all(
        &mut self,
        image: &Tensor,
        frame_index: usize,
        reverse: bool,
        num_frames: Option<usize>,
    ) -> Result<Vec<(u32, Sam2FrameOutput)>> {
        let mut outputs = Vec::with_capacity(self.states.len());
        for (&object_id, state) in &mut self.states {
            outputs.push((
                object_id,
                self.model.track_with_num_frames(
                    image,
                    frame_index,
                    false,
                    state,
                    &[],
                    None,
                    false,
                    reverse,
                    num_frames,
                )?,
            ));
        }
        Ok(outputs)
    }
}

impl Sam2InteractiveVideoPredictor {
    /// Downloads the published checkpoint through the Hugging Face cache and
    /// constructs the editable video predictor on CPU.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained(variant: Sam2Variant, config: VideoPredictorConfig) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1(variant, path, config)
    }

    /// Downloads the published checkpoint and constructs the editable video
    /// predictor on `device`.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained_on_device(
        variant: Sam2Variant,
        device: &Device,
        config: VideoPredictorConfig,
    ) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1_on_device(variant, path, device, config)
    }

    pub fn load_sam2_1(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        config: VideoPredictorConfig,
    ) -> Result<Self> {
        Self::load_sam2_1_on_device(variant, path, &Device::Cpu, config)
    }

    pub fn load_sam2_1_on_device(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        device: &Device,
        config: VideoPredictorConfig,
    ) -> Result<Self> {
        Ok(Self {
            model: Sam2Base::load_sam2_1_on_device_with_decoder_config(
                variant,
                path,
                device,
                MaskDecoderConfig::default(),
            )?,
            config,
        })
    }

    pub fn init_state(&self, images: &[(&[u8], usize, usize)]) -> Result<InteractiveVideoSession> {
        let Some(&(_, width, height)) = images.first() else {
            candle_core::bail!("video must contain at least one frame")
        };
        if images
            .iter()
            .any(|(_, frame_width, frame_height)| *frame_width != width || *frame_height != height)
        {
            candle_core::bail!("all video frames must have the same dimensions")
        }
        let transforms = Sam2ImageTransforms::default();
        let device = self.model.device();
        let profile = std::env::var_os("SAM2_PROFILE_VIDEO_STAGES").is_some();
        if profile {
            device.synchronize()?;
        }
        let transform_start = Instant::now();
        let (frames_batch, _) = transforms.forward_batch(images, device)?;
        if profile {
            device.synchronize()?;
            eprintln!(
                "SAM2_VIDEO_INIT_RS stage=transform frames={} ms={:.6}",
                images.len(),
                transform_start.elapsed().as_secs_f64() * 1000.
            );
        }
        let frames = (0..images.len())
            .map(|index| frames_batch.narrow(0, index, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut session = InteractiveVideoSession {
            frames,
            state: VideoInferenceState::with_video(width, height, images.len())?,
            model_states: BTreeMap::new(),
            outputs: BTreeMap::new(),
        };
        // Match the reference predictor's init_state boundary: warm the
        // backbone and retain frame zero for the first interaction.
        let feature_start = Instant::now();
        let _ = self.image_feature(&mut session, 0)?;
        if profile {
            device.synchronize()?;
            eprintln!(
                "SAM2_VIDEO_INIT_RS stage=frame_zero_feature frames={} ms={:.6}",
                images.len(),
                feature_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(session)
    }

    /// Initializes a session from already decoded owned frames.
    #[cfg(feature = "image-decoding")]
    pub fn init_state_from_decoded_video(
        &self,
        video: &crate::DecodedVideo,
    ) -> Result<InteractiveVideoSession> {
        self.init_state(&video.inputs())
    }

    /// Consumes decoded frames, releasing each RGB allocation immediately
    /// after its normalized model tensor has been prepared.
    #[cfg(feature = "image-decoding")]
    pub fn init_state_from_owned_video(
        &self,
        video: crate::DecodedVideo,
    ) -> Result<InteractiveVideoSession> {
        let width = video.original_width();
        let height = video.original_height();
        let decoded = video.into_frames();
        let num_frames = decoded.len();
        let transforms = Sam2ImageTransforms::default();
        let device = self.model.device();
        let frames = decoded
            .into_iter()
            .map(|frame| {
                let (pixels, frame_width, frame_height) = frame.into_parts();
                Ok(transforms
                    .prepare_rgb8(&pixels, frame_width, frame_height, device)?
                    .tensor)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut session = InteractiveVideoSession {
            frames,
            state: VideoInferenceState::with_video(width, height, num_frames)?,
            model_states: BTreeMap::new(),
            outputs: BTreeMap::new(),
        };
        let _ = self.image_feature(&mut session, 0)?;
        Ok(session)
    }

    /// Decodes a numerically named JPEG directory and initializes a session.
    #[cfg(feature = "image-decoding")]
    pub fn init_state_from_jpeg_directory(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<InteractiveVideoSession> {
        let video = crate::load_jpeg_directory(path)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.init_state_from_owned_video(video)
    }

    /// Decodes an encoded video with FFmpeg and initializes a session.
    #[cfg(feature = "video-decoding")]
    pub fn init_state_from_video(&self, path: impl AsRef<Path>) -> Result<InteractiveVideoSession> {
        let video = crate::decode_video(path)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.init_state_from_owned_video(video)
    }

    fn memory(output: &Sam2FrameOutput) -> crate::MemoryFrame {
        crate::MemoryFrame {
            memory_features: output.memory_features.clone(),
            memory_pos_enc: output.memory_pos_enc.clone(),
            object_pointer: output.object_pointer.clone(),
            low_res_mask_logits: output.low_res_masks.clone(),
        }
    }

    fn stored_output(output: &Sam2FrameOutput) -> StoredVideoOutput {
        StoredVideoOutput {
            low_res_masks: output.low_res_masks.clone(),
            object_score_logits: output.object_score_logits.clone(),
            object_pointer: output.object_pointer.clone(),
        }
    }

    fn image_feature(
        &self,
        session: &mut InteractiveVideoSession,
        frame_index: usize,
    ) -> Result<BackboneOutput> {
        if let Some(feature) = session.state.cached_feature(frame_index) {
            return Ok(feature.clone());
        }
        let feature = self.model.encode_frame(&session.frames[frame_index])?;
        session.state.cache_feature(frame_index, feature.clone());
        Ok(feature)
    }

    fn postprocess_output(&self, mut output: Sam2FrameOutput) -> Result<Sam2FrameOutput> {
        if self.config.fill_hole_area > 0 {
            output.low_res_masks = crate::ops::fill_holes_in_mask_scores(
                &output.low_res_masks,
                self.config.fill_hole_area,
            )?;
        }
        Ok(output)
    }

    fn stored_prompts(
        session: &InteractiveVideoSession,
        object_id: u32,
        frame_index: usize,
    ) -> Result<(Vec<PointPrompt>, Option<BoxPrompt>)> {
        let prompt = session
            .state
            .object(object_id)
            .and_then(|object| object.point_inputs.get(&frame_index))
            .ok_or_else(|| candle_core::Error::Msg("point prompt is missing".into()))?;
        let coordinates = prompt
            .coordinates
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        let labels = prompt.labels.to_device(&Device::Cpu)?.to_vec2::<i64>()?;
        let mut points = Vec::new();
        let mut bounding_box = None;
        let mut index = 0;
        if labels[0].len() >= 2 && labels[0][0] == 2 && labels[0][1] == 3 {
            bounding_box = Some(BoxPrompt {
                x0: coordinates[0][0][0],
                y0: coordinates[0][0][1],
                x1: coordinates[0][1][0],
                y1: coordinates[0][1][1],
            });
            index = 2;
        }
        for (&label, coordinate) in labels[0][index..].iter().zip(&coordinates[0][index..]) {
            points.push(PointPrompt {
                x: coordinate[0],
                y: coordinate[1],
                positive: label == 1,
            });
        }
        Ok((points, bounding_box))
    }

    fn output_masks(
        &self,
        session: &InteractiveVideoSession,
        frame_index: usize,
        conditioning: bool,
    ) -> Result<InteractiveVideoOutput> {
        let (width, height) = session.state.video_dimensions().unwrap();
        let logits = session.state.consolidated_mask_logits(
            frame_index,
            conditioning,
            256,
            256,
            self.model.device(),
        )?;
        Ok(InteractiveVideoOutput {
            frame_index,
            object_ids: session.state.object_ids().to_vec(),
            masks: restore_video_masks(&logits, height, width, self.config.non_overlapping_masks)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_new_points_or_box(
        &self,
        session: &mut InteractiveVideoSession,
        frame_index: usize,
        object_id: u32,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        clear_old_points: bool,
    ) -> Result<InteractiveVideoOutput> {
        if frame_index >= session.frames.len() || (points.is_empty() && bounding_box.is_none()) {
            candle_core::bail!("invalid frame or empty interaction")
        }
        if bounding_box.is_some() && !clear_old_points {
            candle_core::bail!("a box must clear older points")
        }
        let (width, height) = session.state.video_dimensions().unwrap();
        let mut coordinates = Vec::new();
        let mut labels = Vec::new();
        if let Some(bounds) = bounding_box {
            coordinates.extend([
                bounds.x0 * 1024. / width as f32,
                bounds.y0 * 1024. / height as f32,
                bounds.x1 * 1024. / width as f32,
                bounds.y1 * 1024. / height as f32,
            ]);
            labels.extend([2i64, 3]);
        }
        for point in points {
            coordinates.extend([
                point.x * 1024. / width as f32,
                point.y * 1024. / height as f32,
            ]);
            labels.push(i64::from(point.positive));
        }
        let count = labels.len();
        session.state.add_points(
            object_id,
            frame_index,
            &Tensor::from_vec(coordinates, (1, count, 2), &Device::Cpu)?,
            &Tensor::from_vec(labels, (1, count), &Device::Cpu)?,
            clear_old_points,
        )?;
        let object = session.state.object(object_id).unwrap();
        let is_initial = !object.tracked_frames.contains_key(&frame_index);
        let reverse = object
            .tracked_frames
            .get(&frame_index)
            .is_some_and(|frame| frame.reverse);
        let conditioning = is_initial || self.config.add_all_corrections_as_conditioning;
        let (stored_points, stored_box) = Self::stored_prompts(session, object_id, frame_index)?;
        let previous = session
            .outputs
            .get(&object_id)
            .and_then(|outputs| outputs.get(&frame_index))
            .map(|output| output.low_res_masks.clamp(-32., 32.))
            .transpose()?;
        let feature = self.image_feature(session, frame_index)?;
        let output = self.postprocess_output(self.model.track_backbone_with_deferred_memory(
            &feature,
            frame_index,
            conditioning,
            session.model_states.entry(object_id).or_default(),
            &stored_points,
            stored_box,
            previous.as_ref(),
            true,
            // Published SAM2.1 uses multimask output for both initial prompts
            // and tracked-frame corrections when there are zero or one point
            // labels (boxes contribute two labels and therefore disable it).
            count <= 1,
            reverse,
            Some(session.frames.len()),
        )?)?;
        session
            .outputs
            .entry(object_id)
            .or_default()
            .insert(frame_index, Self::stored_output(&output));
        let object = session.state.object_mut(object_id);
        if conditioning {
            object
                .temporary_conditioning_outputs
                .insert(frame_index, Self::memory(&output));
        } else {
            object
                .temporary_non_conditioning_outputs
                .insert(frame_index, Self::memory(&output));
        }
        self.output_masks(session, frame_index, conditioning)
    }

    pub fn add_new_points(
        &self,
        session: &mut InteractiveVideoSession,
        frame_index: usize,
        object_id: u32,
        points: &[PointPrompt],
        clear_old_points: bool,
    ) -> Result<InteractiveVideoOutput> {
        self.add_new_points_or_box(
            session,
            frame_index,
            object_id,
            points,
            None,
            clear_old_points,
        )
    }

    pub fn add_new_mask(
        &self,
        session: &mut InteractiveVideoSession,
        frame_index: usize,
        object_id: u32,
        mask: &Tensor,
    ) -> Result<InteractiveVideoOutput> {
        if frame_index >= session.frames.len() {
            candle_core::bail!("frame is outside the video")
        }
        let (width, height) = session.state.video_dimensions().unwrap();
        let mask = match mask.rank() {
            2 => mask.unsqueeze(0)?.unsqueeze(0)?,
            3 => mask.unsqueeze(0)?,
            4 => mask.clone(),
            _ => candle_core::bail!("mask must be HxW, 1xHxW, or 1x1xHxW"),
        };
        if mask.dim(mask.rank() - 2)? != height || mask.dim(mask.rank() - 1)? != width {
            candle_core::bail!("mask dimensions do not match the video")
        }
        let model_mask = mask.to_dtype(DType::F32)?.to_device(self.model.device())?;
        let model_mask = self
            .model
            .antialiased_bilinear_resize(&model_mask, 1024, 1024)?
            .ge(0.5)?
            .to_dtype(DType::F32)?;
        session.state.add_mask(object_id, frame_index, mask);
        let feature = self.image_feature(session, frame_index)?;
        let output =
            self.postprocess_output(self.model.track_mask_from_backbone_deferred_memory(
                &feature,
                &model_mask,
                frame_index,
                true,
                session.model_states.entry(object_id).or_default(),
            )?)?;
        session
            .outputs
            .entry(object_id)
            .or_default()
            .insert(frame_index, Self::stored_output(&output));
        session
            .state
            .object_mut(object_id)
            .temporary_conditioning_outputs
            .insert(frame_index, Self::memory(&output));
        self.output_masks(session, frame_index, true)
    }

    pub fn propagate_in_video_preflight(
        &self,
        session: &mut InteractiveVideoSession,
    ) -> Result<()> {
        session.state.validate_for_propagation()?;
        let object_ids = session.state.object_ids().to_vec();
        for object_id in object_ids {
            let frames = {
                let object = session.state.object(object_id).unwrap();
                object
                    .temporary_conditioning_outputs
                    .keys()
                    .chain(object.temporary_non_conditioning_outputs.keys())
                    .copied()
                    .collect::<Vec<_>>()
            };
            for frame_index in frames {
                let feature = self.image_feature(session, frame_index)?;
                let cached_memory_pos = session
                    .state
                    .mask_memory_position_encoding()
                    .and_then(|positions| positions.first())
                    .cloned();
                let output = session
                    .outputs
                    .get_mut(&object_id)
                    .and_then(|outputs| outputs.get_mut(&frame_index))
                    .ok_or_else(|| candle_core::Error::Msg("temporary output is missing".into()))?;
                let mut memory = self.model.reencode_memory_from_backbone(
                    &feature,
                    &output.low_res_masks,
                    &output.object_score_logits,
                    &output.object_pointer,
                    true,
                )?;
                // Upstream SAM2VideoPredictor stores temporal features as
                // BF16 on CPU as well as CUDA. Attention promotes only the
                // selected frames back to the active compute dtype.
                memory.memory_features = memory.memory_features.to_dtype(DType::BF16)?;
                if let Some(position) = cached_memory_pos.as_ref() {
                    memory.memory_pos_enc = position.clone();
                }
                if cached_memory_pos.is_none() {
                    session
                        .state
                        .set_mask_memory_position_encoding(vec![memory.memory_pos_enc.clone()]);
                }
                let model_state = session.model_states.get_mut(&object_id).unwrap();
                let conditioning = session
                    .state
                    .object(object_id)
                    .unwrap()
                    .temporary_conditioning_outputs
                    .contains_key(&frame_index);
                if conditioning {
                    model_state.insert_conditioning(frame_index, memory.clone());
                    session
                        .state
                        .object_mut(object_id)
                        .temporary_conditioning_outputs
                        .insert(frame_index, memory);
                } else {
                    model_state.insert_propagated(frame_index, memory.clone());
                    session
                        .state
                        .object_mut(object_id)
                        .temporary_non_conditioning_outputs
                        .insert(frame_index, memory);
                }
            }
        }
        let mut conditioning = Vec::new();
        let mut non_conditioning = Vec::new();
        for object_id in session.state.object_ids() {
            let object = session.state.object(*object_id).unwrap();
            conditioning.extend(object.temporary_conditioning_outputs.keys().copied());
            non_conditioning.extend(object.temporary_non_conditioning_outputs.keys().copied());
        }
        conditioning.sort_unstable();
        conditioning.dedup();
        non_conditioning.sort_unstable();
        non_conditioning.dedup();
        for frame in non_conditioning {
            session.state.commit_temporary_frame(frame, false);
        }
        for frame in conditioning {
            session.state.commit_temporary_frame(frame, true);
            if self.config.clear_non_conditioning_memory_around_input {
                session.state.clear_non_conditioning_around(frame, 7);
            }
        }
        Ok(())
    }

    pub fn propagate_in_video(
        &self,
        session: &mut InteractiveVideoSession,
        start_frame: Option<usize>,
        maximum_frames: Option<usize>,
        reverse: bool,
    ) -> Result<Vec<InteractiveVideoOutput>> {
        self.propagate_in_video_preflight(session)?;
        let order = session
            .state
            .propagation_order(start_frame, maximum_frames, reverse)?;
        let object_ids = session.state.object_ids().to_vec();
        let mut results = Vec::with_capacity(order.len());
        for frame_index in order {
            let feature = self.image_feature(session, frame_index)?;
            let mut masks = Vec::with_capacity(object_ids.len());
            for &object_id in &object_ids {
                let existing_conditioning = session
                    .state
                    .object(object_id)
                    .unwrap()
                    .conditioning_outputs
                    .contains_key(&frame_index);
                let low_res_masks = if existing_conditioning {
                    session.outputs[&object_id][&frame_index]
                        .low_res_masks
                        .clone()
                } else {
                    let output =
                        self.postprocess_output(self.model.track_backbone_with_compact_memory(
                            &feature,
                            frame_index,
                            false,
                            session.model_states.get_mut(&object_id).unwrap(),
                            &[],
                            None,
                            None,
                            false,
                            // SAM2.1 enables multimask tracking for zero or one
                            // points, including ordinary propagated frames. The
                            // model then keeps the candidate with the best IoU
                            // score for the mask, pointer, and memory encoding.
                            true,
                            reverse,
                            Some(session.frames.len()),
                        )?)?;
                    session
                        .outputs
                        .entry(object_id)
                        .or_default()
                        .insert(frame_index, Self::stored_output(&output));
                    session
                        .state
                        .object_mut(object_id)
                        .non_conditioning_outputs
                        .insert(frame_index, Self::memory(&output));
                    output.low_res_masks
                };
                session.state.mark_tracked(object_id, frame_index, reverse);
                masks.push(low_res_masks);
            }
            let refs = masks.iter().collect::<Vec<_>>();
            let (width, height) = session.state.video_dimensions().unwrap();
            results.push(InteractiveVideoOutput {
                frame_index,
                object_ids: object_ids.clone(),
                masks: restore_video_masks(
                    &Tensor::cat(&refs, 0)?,
                    height,
                    width,
                    self.config.non_overlapping_masks,
                )?,
            });
        }
        Ok(results)
    }

    /// Synchronized companion to [`Self::propagate_in_video`] for benchmark
    /// attribution. Its additional synchronization makes its total unsuitable
    /// as the headline latency measurement.
    pub fn propagate_in_video_profiled(
        &self,
        session: &mut InteractiveVideoSession,
        start_frame: Option<usize>,
        maximum_frames: Option<usize>,
        reverse: bool,
    ) -> Result<(Vec<InteractiveVideoOutput>, VideoPropagationProfile)> {
        let mut profile = VideoPropagationProfile::default();
        self.model.device().synchronize()?;
        let start = Instant::now();
        self.propagate_in_video_preflight(session)?;
        self.model.device().synchronize()?;
        profile.preflight_ms = start.elapsed().as_secs_f64() * 1000.;
        let order = session
            .state
            .propagation_order(start_frame, maximum_frames, reverse)?;
        let object_ids = session.state.object_ids().to_vec();
        let mut results = Vec::with_capacity(order.len());
        for frame_index in order {
            let start = Instant::now();
            let feature = self.image_feature(session, frame_index)?;
            self.model.device().synchronize()?;
            profile
                .image_encoder_ms
                .push(start.elapsed().as_secs_f64() * 1000.);

            let start = Instant::now();
            let mut masks = Vec::with_capacity(object_ids.len());
            for &object_id in &object_ids {
                let existing_conditioning = session
                    .state
                    .object(object_id)
                    .unwrap()
                    .conditioning_outputs
                    .contains_key(&frame_index);
                let low_res_masks = if existing_conditioning {
                    session.outputs[&object_id][&frame_index]
                        .low_res_masks
                        .clone()
                } else {
                    let output =
                        self.postprocess_output(self.model.track_backbone_with_compact_memory(
                            &feature,
                            frame_index,
                            false,
                            session.model_states.get_mut(&object_id).unwrap(),
                            &[],
                            None,
                            None,
                            false,
                            true,
                            reverse,
                            Some(session.frames.len()),
                        )?)?;
                    session
                        .outputs
                        .entry(object_id)
                        .or_default()
                        .insert(frame_index, Self::stored_output(&output));
                    session
                        .state
                        .object_mut(object_id)
                        .non_conditioning_outputs
                        .insert(frame_index, Self::memory(&output));
                    output.low_res_masks
                };
                session.state.mark_tracked(object_id, frame_index, reverse);
                masks.push(low_res_masks);
            }
            self.model.device().synchronize()?;
            profile
                .temporal_and_heads_ms
                .push(start.elapsed().as_secs_f64() * 1000.);

            let start = Instant::now();
            let refs = masks.iter().collect::<Vec<_>>();
            let (width, height) = session.state.video_dimensions().unwrap();
            let masks = restore_video_masks(
                &Tensor::cat(&refs, 0)?,
                height,
                width,
                self.config.non_overlapping_masks,
            )?;
            self.model.device().synchronize()?;
            profile
                .restore_masks_ms
                .push(start.elapsed().as_secs_f64() * 1000.);
            results.push(InteractiveVideoOutput {
                frame_index,
                object_ids: object_ids.clone(),
                masks,
            });
        }
        Ok((results, profile))
    }

    pub fn clear_all_prompts_in_frame(
        &self,
        session: &mut InteractiveVideoSession,
        frame_index: usize,
        object_id: u32,
    ) -> Result<InteractiveVideoOutput> {
        session.state.clear_prompts_in_frame(object_id, frame_index);
        self.output_masks(session, frame_index, false)
    }

    pub fn remove_object(&self, session: &mut InteractiveVideoSession, object_id: u32) -> bool {
        session.model_states.remove(&object_id);
        session.outputs.remove(&object_id);
        session.state.remove_object(object_id).is_some()
    }

    /// Removes an object and returns the frames whose consolidated output may
    /// have changed because that object had an input there. This is the typed
    /// counterpart of the reference predictor's `(obj_ids, updated_frames)`
    /// return value; [`Self::remove_object`] remains as the lightweight legacy
    /// convenience method.
    pub fn remove_object_with_outputs(
        &self,
        session: &mut InteractiveVideoSession,
        object_id: u32,
    ) -> Result<RemoveObjectOutput> {
        let Some(object) = session.state.object(object_id) else {
            return Ok(RemoveObjectOutput {
                object_ids: session.state.object_ids().to_vec(),
                updated_frames: Vec::new(),
            });
        };
        let mut affected_frames = object
            .point_inputs
            .keys()
            .chain(object.mask_inputs.keys())
            .copied()
            .collect::<Vec<_>>();
        affected_frames.sort_unstable();
        affected_frames.dedup();

        if session.state.object_count() == 1 {
            self.reset_state(session);
            return Ok(RemoveObjectOutput {
                object_ids: Vec::new(),
                updated_frames: Vec::new(),
            });
        }

        session.model_states.remove(&object_id);
        session.outputs.remove(&object_id);
        session.state.remove_object(object_id);
        let mut updated_frames = Vec::with_capacity(affected_frames.len());
        for frame_index in affected_frames {
            let conditioning = session.state.object_ids().iter().any(|&remaining_id| {
                session.state.object(remaining_id).is_some_and(|object| {
                    object
                        .temporary_conditioning_outputs
                        .contains_key(&frame_index)
                })
            });
            updated_frames.push(self.output_masks(session, frame_index, conditioning)?);
        }
        Ok(RemoveObjectOutput {
            object_ids: session.state.object_ids().to_vec(),
            updated_frames,
        })
    }

    pub fn reset_state(&self, session: &mut InteractiveVideoSession) {
        let (width, height) = session.state.video_dimensions().unwrap();
        let frames = session.frames.len();
        session.state = VideoInferenceState::with_video(width, height, frames).unwrap();
        session.model_states.clear();
        session.outputs.clear();
    }
}

impl Sam2ImagePredictor {
    /// Downloads the published checkpoint through the Hugging Face cache and
    /// constructs the image predictor on CPU.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained(variant: Sam2Variant) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1(variant, path)
    }

    /// Downloads the published checkpoint and constructs it on `device`.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained_on_device(variant: Sam2Variant, device: &Device) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1_on_device(variant, path, device)
    }

    /// Loads a converted safetensors state dict. The model is initialized only
    /// after its checkpoint keys have been checked by `CheckpointIndex`.
    pub fn load_sam2_1_hiera_tiny(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_sam2_1(Sam2Variant::HieraTiny, path)
    }

    /// Loads any published SAM2.1 Hiera checkpoint converted to safetensors.
    /// The same static predictor path is shared by Tiny, Small, Base-Plus, and
    /// Large; only the typed architecture configuration changes.
    pub fn load_sam2_1(variant: Sam2Variant, path: impl AsRef<Path>) -> Result<Self> {
        Self::load_sam2_1_on_device(variant, path, &Device::Cpu)
    }

    /// Loads a converted checkpoint onto `device`. Keeping device choice at
    /// this boundary makes CPU and CUDA inference run the identical model path.
    pub fn load_sam2_1_on_device(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        Self::load_sam2_1_on_device_with_config(
            variant,
            path,
            device,
            ImagePredictorConfig::default(),
        )
    }

    /// Load frozen encoder and prompt weights from a complete parent checkpoint while replacing
    /// only the mask decoder with a validated decoder-only child artifact. This keeps fine-tuned
    /// revisions explicit and avoids rewriting or duplicating their immutable parent checkpoint.
    pub fn load_sam2_1_with_mask_decoder_on_device(
        variant: Sam2Variant,
        parent_checkpoint: impl AsRef<Path>,
        decoder_checkpoint: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let parent_checkpoint = parent_checkpoint.as_ref();
        let decoder_checkpoint = decoder_checkpoint.as_ref();
        crate::checkpoint::validate_sam2_1(parent_checkpoint, variant)?;
        crate::checkpoint::validate_sam2_1_mask_decoder(decoder_checkpoint, variant)?;
        if device.is_cpu() {
            crate::cpu_allocator::configure_cpu_allocator_for_inference();
        }
        crate::cuda_allocator::configure_cuda_allocator_for_inference(device)?;
        let parent = crate::checkpoint::load_var_builder(parent_checkpoint, DType::F32, device)?;
        let decoder = crate::checkpoint::load_var_builder(decoder_checkpoint, DType::F32, device)?;
        let config = Sam2Config::sam2_1(variant);
        let predictor_config = ImagePredictorConfig::default();
        Ok(Self {
            device: device.clone(),
            image_encoder: ImageEncoder::load(&config, parent.pp("image_encoder"))?,
            prompt_encoder: PromptEncoder::load(&config, parent.pp("sam_prompt_encoder"))?,
            mask_decoder: MaskDecoder::load_with_config(
                decoder.pp("sam_mask_decoder"),
                predictor_config.decoder,
            )?,
            no_mem_embed: parent.get((1, 1, 256), "no_mem_embed")?,
            transforms: Sam2ImageTransforms::with_postprocessing(
                config.image_size,
                predictor_config.mask_threshold,
                predictor_config.max_hole_area,
                predictor_config.max_sprinkle_area,
            ),
            cached_images: None,
        })
    }

    pub fn load_sam2_1_on_device_with_config(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        device: &Device,
        predictor_config: ImagePredictorConfig,
    ) -> Result<Self> {
        if device.is_cpu() {
            crate::cpu_allocator::configure_cpu_allocator_for_inference();
        }
        crate::cuda_allocator::configure_cuda_allocator_for_inference(device)?;
        crate::checkpoint::validate_sam2_1(path.as_ref(), variant)?;
        let vb = crate::checkpoint::load_var_builder(path, DType::F32, device)?;
        let config = Sam2Config::sam2_1(variant);
        Ok(Self {
            device: device.clone(),
            image_encoder: ImageEncoder::load(&config, vb.pp("image_encoder"))?,
            prompt_encoder: PromptEncoder::load(&config, vb.pp("sam_prompt_encoder"))?,
            mask_decoder: MaskDecoder::load_with_config(
                vb.pp("sam_mask_decoder"),
                predictor_config.decoder,
            )?,
            no_mem_embed: vb.get((1, 1, 256), "no_mem_embed")?,
            transforms: Sam2ImageTransforms::with_postprocessing(
                config.image_size,
                predictor_config.mask_threshold,
                predictor_config.max_hole_area,
                predictor_config.max_sprinkle_area,
            ),
            cached_images: None,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn is_image_set(&self) -> bool {
        self.cached_images.is_some()
    }

    pub fn is_batch(&self) -> bool {
        self.cached_images
            .as_ref()
            .is_some_and(|cache| cache.is_batch)
    }

    pub fn reset_predictor(&mut self) {
        self.cached_images = None;
    }

    /// Caches one RGB8 image embedding for repeated prompt decoding.
    pub fn set_image(&mut self, rgb_hwc: &[u8], width: usize, height: usize) -> Result<()> {
        let encoded = self.encode_rgb8(rgb_hwc, width, height)?;
        self.set_encoded_image(&encoded);
        Ok(())
    }

    pub fn encode_rgb8(&self, rgb_hwc: &[u8], width: usize, height: usize) -> Result<EncodedImage> {
        let prepared = self
            .transforms
            .prepare_rgb8(rgb_hwc, width, height, &self.device)?;
        let features = self.encode_image(&prepared.tensor)?;
        Ok(EncodedImage {
            features,
            geometry: prepared.geometry,
        })
    }

    pub fn set_encoded_image(&mut self, encoded: &EncodedImage) {
        self.reset_predictor();
        self.cached_images = Some(CachedImages {
            features: encoded.features.clone(),
            geometries: vec![encoded.geometry],
            is_batch: false,
        });
    }

    /// Decodes an image file and caches its image embedding.
    #[cfg(feature = "image-decoding")]
    pub fn set_image_path(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let image = crate::decode_image(path)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.set_image(image.pixels(), image.width(), image.height())
    }

    /// Encodes all RGB8 images in one backbone invocation and retains the
    /// batched features for `predict_batch`.
    pub fn set_image_batch(&mut self, images: &[(&[u8], usize, usize)]) -> Result<()> {
        self.reset_predictor();
        let (input, geometries) = self.transforms.forward_batch(images, &self.device)?;
        let features = self.encode_image(&input)?;
        self.cached_images = Some(CachedImages {
            features,
            geometries,
            is_batch: true,
        });
        Ok(())
    }

    fn cached_features(
        &self,
        image_index: usize,
    ) -> Result<(BackboneOutput, crate::ImageGeometry)> {
        let cache = self
            .cached_images
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("set an image before prediction".into()))?;
        if image_index >= cache.geometries.len() {
            candle_core::bail!("cached image index {image_index} is out of range")
        }
        let select = |tensor: &Tensor| tensor.narrow(0, image_index, 1);
        Ok((
            BackboneOutput {
                vision_features: select(&cache.features.vision_features)?,
                vision_pos_enc: cache
                    .features
                    .vision_pos_enc
                    .iter()
                    .map(select)
                    .collect::<Result<Vec<_>>>()?,
                backbone_fpn: cache
                    .features
                    .backbone_fpn
                    .iter()
                    .map(select)
                    .collect::<Result<Vec<_>>>()?,
                sam_high_res_features: cache
                    .features
                    .sam_high_res_features
                    .as_ref()
                    .map(|features| features.iter().map(select).collect::<Result<Vec<_>>>())
                    .transpose()?,
            },
            cache.geometries[image_index],
        ))
    }

    fn prepare_cached_prompts(
        &self,
        prompts: &ImagePromptBatch,
        geometry: crate::ImageGeometry,
    ) -> Result<(
        Option<Tensor>,
        Option<Tensor>,
        Option<Tensor>,
        Option<Tensor>,
    )> {
        let mut coordinates = prompts
            .point_coords
            .as_ref()
            .map(|value| value.to_device(&self.device))
            .transpose()?;
        let mut labels = prompts
            .point_labels
            .as_ref()
            .map(|value| value.to_device(&self.device))
            .transpose()?;
        if coordinates.is_some() != labels.is_some() {
            candle_core::bail!("point coordinates and labels must be supplied together")
        }
        if let Some(value) = coordinates.take() {
            let value = self.transforms.transform_coords(
                &value,
                prompts.normalize_coords,
                Some((geometry.original_height, geometry.original_width)),
            )?;
            coordinates = Some(if value.rank() == 2 {
                value.unsqueeze(0)?
            } else {
                value
            });
            if labels.as_ref().is_some_and(|value| value.rank() == 1) {
                labels = Some(labels.take().unwrap().unsqueeze(0)?);
            }
        }
        let boxes = prompts
            .boxes
            .as_ref()
            .map(|value| -> Result<Tensor> {
                let value = self.transforms.transform_boxes(
                    &value.to_device(&self.device)?,
                    prompts.normalize_coords,
                    Some((geometry.original_height, geometry.original_width)),
                )?;
                Ok(if value.rank() == 2 {
                    value.unsqueeze(0)?
                } else {
                    value
                })
            })
            .transpose()?;
        let mask = prompts
            .mask_input
            .as_ref()
            .map(|value| -> Result<Tensor> {
                let value = value.to_dtype(DType::F32)?.to_device(&self.device)?;
                Ok(if value.rank() == 3 {
                    value.unsqueeze(0)?
                } else {
                    value
                })
            })
            .transpose()?;
        Ok((coordinates, labels, boxes, mask))
    }

    /// Prepare detached upstream encoder and prompt outputs for a Blockflow-owned decoder
    /// training step. This exposes translated data flow, without adding losses or optimizer
    /// policy to sam2-rs.
    pub fn frozen_decoder_inputs(
        &self,
        image_index: usize,
        prompts: &ImagePromptBatch,
    ) -> Result<crate::FrozenDecoderInputs> {
        let (features, geometry) = self.cached_features(image_index)?;
        let (coordinates, labels, boxes, mask_input) =
            self.prepare_cached_prompts(prompts, geometry)?;
        let (coordinates, labels) = match (coordinates, labels, boxes) {
            (coordinates, labels, Some(boxes)) => {
                let batch = boxes.dim(0)?;
                let box_coords = boxes.reshape((batch, 2, 2))?;
                let box_labels = Tensor::new(&[2i64, 3], &self.device)?
                    .unsqueeze(0)?
                    .broadcast_as((batch, 2))?;
                match (coordinates, labels) {
                    (Some(coordinates), Some(labels)) => (
                        Some(Tensor::cat(&[&box_coords, &coordinates], 1)?),
                        Some(Tensor::cat(&[&box_labels, &labels], 1)?),
                    ),
                    (None, None) => (Some(box_coords), Some(box_labels)),
                    _ => candle_core::bail!("point coordinates and labels must match"),
                }
            }
            (Some(coordinates), Some(labels), None) => (Some(coordinates), Some(labels)),
            (None, None, None) => (None, None),
            _ => candle_core::bail!("point coordinates and labels must match"),
        };
        let prompt_embeddings = self.prompt_encoder.forward(
            coordinates.as_ref().zip(labels.as_ref()),
            None,
            mask_input.as_ref(),
        )?;
        let high_res_features: [Tensor; 2] = features.backbone_fpn[..2]
            .iter()
            .map(Tensor::detach)
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| {
                candle_core::Error::Msg("SAM2 encoder returned the wrong FPN count".into())
            })?;
        Ok(crate::FrozenDecoderInputs {
            image_embeddings: features.vision_features.detach(),
            image_pe: self.prompt_encoder.get_dense_pe()?.detach(),
            sparse_prompt_embeddings: prompt_embeddings.sparse_embeddings.detach(),
            dense_prompt_embeddings: prompt_embeddings.dense_embeddings.detach(),
            high_res_features,
            geometry,
        })
    }

    fn decode_prepared_cached(
        &self,
        features: &BackboneOutput,
        point_coords: Option<&Tensor>,
        point_labels: Option<&Tensor>,
        boxes: Option<&Tensor>,
        mask_input: Option<&Tensor>,
        multimask_output: bool,
    ) -> Result<MaskDecoderOutput> {
        let (coordinates, labels) = match (point_coords, point_labels, boxes) {
            (coordinates, labels, Some(boxes)) => {
                let batch = boxes.dim(0)?;
                let box_coords = boxes.reshape((batch, 2, 2))?;
                let box_labels = Tensor::new(&[2i64, 3], &self.device)?
                    .unsqueeze(0)?
                    .broadcast_as((batch, 2))?;
                match (coordinates, labels) {
                    (Some(coordinates), Some(labels)) => (
                        Some(Tensor::cat(&[&box_coords, coordinates], 1)?),
                        Some(Tensor::cat(&[&box_labels, labels], 1)?),
                    ),
                    (None, None) => (Some(box_coords), Some(box_labels)),
                    _ => candle_core::bail!("point coordinates and labels must match"),
                }
            }
            (Some(coordinates), Some(labels), None) => {
                (Some(coordinates.clone()), Some(labels.clone()))
            }
            (None, None, None) => (None, None),
            _ => candle_core::bail!("point coordinates and labels must match"),
        };
        let profile = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
        if profile {
            self.device.synchronize()?;
        }
        let prompt_start = Instant::now();
        let prompt_embeddings = self.prompt_encoder.forward(
            coordinates.as_ref().zip(labels.as_ref()),
            None,
            mask_input,
        )?;
        if profile {
            self.device.synchronize()?;
            eprintln!(
                "SAM2_IMAGE_STAGE stage=prompt_encoder mask={} batch={} ms={:.6}",
                mask_input.is_some(),
                prompt_embeddings.sparse_embeddings.dim(0)?,
                prompt_start.elapsed().as_secs_f64() * 1000.
            );
        }
        let dense_pe = self.prompt_encoder.get_dense_pe()?;
        if let Some(high_res) = &features.sam_high_res_features {
            self.mask_decoder.forward_preprojected(
                &features.vision_features,
                &dense_pe,
                &prompt_embeddings.sparse_embeddings,
                &prompt_embeddings.dense_embeddings,
                multimask_output,
                high_res,
            )
        } else {
            self.mask_decoder.forward(
                &features.vision_features,
                &dense_pe,
                &prompt_embeddings.sparse_embeddings,
                &prompt_embeddings.dense_embeddings,
                multimask_output,
                &features.backbone_fpn[..2],
            )
        }
    }

    fn predict_cached_index(
        &self,
        image_index: usize,
        prompts: &ImagePromptBatch,
        multimask_output: bool,
        return_logits: bool,
    ) -> Result<ImagePrediction> {
        let (mut low_res, geometry) =
            self.decode_cached_index_raw(image_index, prompts, multimask_output)?;
        let masks = self.transforms.postprocess_masks(
            &low_res.masks,
            geometry.original_height,
            geometry.original_width,
        )?;
        low_res.masks = low_res.masks.clamp(-32., 32.)?;
        let masks = if return_logits {
            masks
        } else {
            masks.gt(self.transforms.mask_threshold())?
        };
        Ok(ImagePrediction { low_res, masks })
    }

    /// Runs cached prompt preparation and the decoder without restoring masks
    /// to the source image. AMG mask-to-mask refinement consumes only these
    /// native 256×256 logits, so restoring all initial candidates would create
    /// a large result that is immediately discarded.
    fn decode_cached_index_raw(
        &self,
        image_index: usize,
        prompts: &ImagePromptBatch,
        multimask_output: bool,
    ) -> Result<(MaskDecoderOutput, ImageGeometry)> {
        let (features, geometry) = self.cached_features(image_index)?;
        let (coordinates, labels, boxes, mask) = self.prepare_cached_prompts(prompts, geometry)?;
        let low_res = self.decode_prepared_cached(
            &features,
            coordinates.as_ref(),
            labels.as_ref(),
            boxes.as_ref(),
            mask.as_ref(),
            multimask_output,
        )?;
        Ok((low_res, geometry))
    }

    /// Predicts from a previously cached single image without rerunning Hiera.
    pub fn predict_cached(
        &self,
        prompts: &ImagePromptBatch,
        multimask_output: bool,
        return_logits: bool,
    ) -> Result<ImagePrediction> {
        let cache = self
            .cached_images
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("set an image before prediction".into()))?;
        if cache.is_batch {
            candle_core::bail!("use predict_batch for a cached image batch")
        }
        self.predict_cached_index(0, prompts, multimask_output, return_logits)
    }

    /// Decode ordinary pixel-coordinate prompts against the currently cached image embedding.
    ///
    /// This is the allocation boundary needed by viewers and workflow engines: callers retain
    /// plain Rust prompt data and do not need a direct Candle dependency merely to construct the
    /// four small tensors accepted by [`Self::predict_cached`]. `mask_input`, when supplied, is a
    /// single low-resolution logit plane in row-major order.
    pub fn predict_cached_prompts(
        &self,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        mask_input: Option<(&[f32], usize, usize)>,
        multimask_output: bool,
        return_logits: bool,
    ) -> Result<ImagePrediction> {
        let point_coords = if points.is_empty() {
            None
        } else {
            Some(Tensor::from_vec(
                points
                    .iter()
                    .flat_map(|point| [point.x, point.y])
                    .collect::<Vec<_>>(),
                (points.len(), 2),
                &self.device,
            )?)
        };
        let point_labels = if points.is_empty() {
            None
        } else {
            Some(Tensor::from_vec(
                points
                    .iter()
                    .map(|point| if point.positive { 1_i64 } else { 0_i64 })
                    .collect::<Vec<_>>(),
                points.len(),
                &self.device,
            )?)
        };
        let boxes = bounding_box
            .map(|bounds| {
                Tensor::from_vec(
                    vec![bounds.x0, bounds.y0, bounds.x1, bounds.y1],
                    (1, 4),
                    &self.device,
                )
            })
            .transpose()?;
        let mask_input = mask_input
            .map(|(values, height, width)| {
                if values.len() != height.saturating_mul(width) {
                    candle_core::bail!(
                        "mask input has {} values, expected {height}x{width}",
                        values.len()
                    )
                }
                Tensor::from_vec(values.to_vec(), (1, height, width), &self.device)
            })
            .transpose()?;
        self.predict_cached(
            &ImagePromptBatch {
                point_coords,
                point_labels,
                boxes,
                mask_input,
                normalize_coords: true,
            },
            multimask_output,
            return_logits,
        )
    }

    /// Decodes one prompt batch per cached image. The backbone is not rerun.
    pub fn predict_batch(
        &self,
        prompts: &[ImagePromptBatch],
        multimask_output: bool,
        return_logits: bool,
    ) -> Result<Vec<ImagePrediction>> {
        let cache = self.cached_images.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("set an image batch before prediction".into())
        })?;
        if !cache.is_batch {
            candle_core::bail!("predict_batch requires set_image_batch")
        }
        if prompts.len() != cache.geometries.len() {
            candle_core::bail!("expected one prompt batch per cached image")
        }
        prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| {
                self.predict_cached_index(index, prompt, multimask_output, return_logits)
            })
            .collect()
    }

    pub fn get_image_embedding(&self) -> Result<&Tensor> {
        self.cached_images
            .as_ref()
            .map(|cache| &cache.features.vision_features)
            .ok_or_else(|| {
                candle_core::Error::Msg("set an image before requesting embedding".into())
            })
    }

    /// Predicts low-resolution (256x256) logits. `image` must be NCHW float
    /// data after SAM2 square resize and RGB normalization, normally `[1,3,1024,1024]`.
    pub fn predict(
        &self,
        image: &Tensor,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
    ) -> Result<MaskDecoderOutput> {
        let features = self.encode_image(image)?;
        self.decode_image_features(&features, points, bounding_box, None, multimask_output)
    }

    /// Encodes an already-normalized image once. Reuse the returned features
    /// for multiple prompt decodes and for component-level performance probes.
    pub fn encode_image(&self, image: &Tensor) -> Result<BackboneOutput> {
        let mut features = if std::env::var_os("SAM2_PROFILE_IMAGE_STAGES").is_some() {
            let (features, profile) = self.image_encoder.forward_profiled(image)?;
            eprintln!(
                "SAM2_IMAGE_ENCODER_RS hiera_setup_ms={:.6} hiera_blocks_ms={:?} \
                 norm1_ms={:?} shortcut_ms={:?} shortcut_pool_ms={:?} partition_ms={:?} \
                 attention_ms={:?} qkv_ms={:?} qkv_views_ms={:?} \
                 query_pool_ms={:?} query_reshape_ms={:?} query_pool_core_ms={:?} \
                 attention_kernel_ms={:?} attention_projection_ms={:?} \
                 attention_cleanup_ms={:?} attention_cleanup_attended_ms={:?} \
                 attention_cleanup_qkv_ms={:?} unpartition_ms={:?} residual_add_ms={:?} \
                 norm2_ms={:?} mlp_ms={:?} mlp_expansion_ms={:?} \
                 mlp_activation_ms={:?} mlp_projection_ms={:?} fpn_ms={:.6} \
                 fpn_lateral_ms={:?} fpn_top_down_ms={:?} fpn_position_ms={:?}",
                profile.hiera_setup_ms,
                profile.hiera_block_ms,
                profile.hiera_norm1_ms,
                profile.hiera_shortcut_ms,
                profile.hiera_shortcut_pool_ms,
                profile.hiera_partition_ms,
                profile.hiera_attention_ms,
                profile.hiera_attention_qkv_ms,
                profile.hiera_attention_qkv_views_ms,
                profile.hiera_attention_query_pool_ms,
                profile.hiera_attention_query_reshape_ms,
                profile.hiera_attention_query_pool_core_ms,
                profile.hiera_attention_kernel_ms,
                profile.hiera_attention_projection_ms,
                profile.hiera_attention_cleanup_ms,
                profile.hiera_attention_cleanup_attended_ms,
                profile.hiera_attention_cleanup_qkv_ms,
                profile.hiera_unpartition_ms,
                profile.hiera_residual_add_ms,
                profile.hiera_norm2_ms,
                profile.hiera_mlp_ms,
                profile.hiera_mlp_expansion_ms,
                profile.hiera_mlp_activation_ms,
                profile.hiera_mlp_projection_ms,
                profile.fpn_ms,
                profile.fpn_lateral_ms,
                profile.fpn_top_down_ms,
                profile.fpn_position_ms,
            );
            features
        } else {
            self.image_encoder.forward(image)?
        };
        features.vision_features = features
            .vision_features
            .broadcast_add(&self.no_mem_embed.reshape((1, 256usize, 1usize, 1usize))?)?;
        // The upstream image predictor caches both high-resolution skip
        // projections with the image embedding. This matters most on CUDA,
        // where repeated prompt decoding must not relaunch the two convolutions.
        // Preserve the CPU attribution escape hatch used by existing probes.
        if !self.device.is_cpu()
            || std::env::var_os("SAM2_CPU_PREDICTOR_PREPROJECT_SKIPS")
                .is_none_or(|value| value != "0")
        {
            let profile_skip = std::env::var_os("SAM2_PROFILE_IMAGE_STAGES").is_some();
            let profile_start = profile_skip.then(std::time::Instant::now);
            features.sam_high_res_features = Some(
                self.mask_decoder
                    .project_high_res_features(&features.backbone_fpn[..2])?,
            );
            if let Some(start) = profile_start {
                self.device.synchronize()?;
                eprintln!(
                    "SAM2_IMAGE_ENCODER_RS skip_projection_ms={:.6}",
                    start.elapsed().as_secs_f64() * 1000.
                );
            }
        }
        Ok(features)
    }

    /// Encodes an image with synchronized Hiera/FPN timings for optimization work.
    pub fn encode_image_profiled(
        &self,
        image: &Tensor,
    ) -> Result<(BackboneOutput, ImageEncoderProfile)> {
        let (mut features, profile) = self.image_encoder.forward_profiled(image)?;
        if !self.device.is_cpu()
            || std::env::var_os("SAM2_CPU_PREDICTOR_PREPROJECT_SKIPS")
                .is_none_or(|value| value != "0")
        {
            features.sam_high_res_features = Some(
                self.mask_decoder
                    .project_high_res_features(&features.backbone_fpn[..2])?,
            );
        }
        Ok((features, profile))
    }

    /// As [`Self::predict`], with an optional low-resolution mask-logit prompt
    /// shaped `B×1×256×256`. This enables iterative click/mask refinement.
    pub fn predict_with_mask(
        &self,
        image: &Tensor,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        mask_input: Option<&Tensor>,
        multimask_output: bool,
    ) -> Result<MaskDecoderOutput> {
        let features = self.encode_image(image)?;
        self.decode_image_features(
            &features,
            points,
            bounding_box,
            mask_input,
            multimask_output,
        )
    }

    /// Decodes prompts from cached image features without re-running Hiera.
    pub fn decode_image_features(
        &self,
        features: &BackboneOutput,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        mask_input: Option<&Tensor>,
        multimask_output: bool,
    ) -> Result<MaskDecoderOutput> {
        let point_tensor = (!points.is_empty())
            .then(|| {
                Tensor::from_vec(
                    points
                        .iter()
                        .flat_map(|point| [point.x, point.y])
                        .collect::<Vec<_>>(),
                    (1, points.len(), 2usize),
                    &self.device,
                )
            })
            .transpose()?;
        let label_tensor = (!points.is_empty())
            .then(|| {
                Tensor::from_vec(
                    points
                        .iter()
                        .map(|point| i64::from(point.positive))
                        .collect::<Vec<_>>(),
                    (1, points.len()),
                    &self.device,
                )
            })
            .transpose()?;
        let box_tensor = bounding_box
            .map(|bounds| {
                Tensor::from_vec(
                    vec![bounds.x0, bounds.y0, bounds.x1, bounds.y1],
                    (1, 4usize),
                    &self.device,
                )
            })
            .transpose()?;
        let prompts = self.prompt_encoder.forward(
            point_tensor.as_ref().zip(label_tensor.as_ref()),
            box_tensor.as_ref(),
            mask_input,
        )?;
        let dense_pe = self.prompt_encoder.get_dense_pe()?;
        if let Some(high_res_features) = &features.sam_high_res_features {
            self.mask_decoder.forward_preprojected(
                &features.vision_features,
                &dense_pe,
                &prompts.sparse_embeddings,
                &prompts.dense_embeddings,
                multimask_output,
                high_res_features,
            )
        } else {
            self.mask_decoder.forward(
                &features.vision_features,
                &dense_pe,
                &prompts.sparse_embeddings,
                &prompts.dense_embeddings,
                multimask_output,
                &features.backbone_fpn[..2],
            )
        }
    }

    /// Decodes several prompt sets against one already encoded image. Point
    /// coordinates and boxes are in the model's 1024×1024 coordinate space;
    /// the leading dimension is the prompt batch.
    pub fn decode_image_features_batch(
        &self,
        features: &BackboneOutput,
        prompts: &ImagePromptBatch,
        multimask_output: bool,
    ) -> Result<MaskDecoderOutput> {
        if prompts.normalize_coords {
            candle_core::bail!(
                "decode_image_features_batch expects model-space coordinates; use predict_cached for original-space prompts"
            )
        }
        let coordinates = prompts
            .point_coords
            .as_ref()
            .map(|value| value.to_device(&self.device))
            .transpose()?;
        let labels = prompts
            .point_labels
            .as_ref()
            .map(|value| value.to_device(&self.device))
            .transpose()?;
        if coordinates.is_some() != labels.is_some() {
            candle_core::bail!("point coordinates and labels must be supplied together")
        }
        let boxes = prompts
            .boxes
            .as_ref()
            .map(|value| value.to_device(&self.device))
            .transpose()?;
        let mask = prompts
            .mask_input
            .as_ref()
            .map(|value| value.to_dtype(DType::F32)?.to_device(&self.device))
            .transpose()?;
        self.decode_prepared_cached(
            features,
            coordinates.as_ref(),
            labels.as_ref(),
            boxes.as_ref(),
            mask.as_ref(),
            multimask_output,
        )
    }

    /// Decoder-only companion to [`Self::decode_image_features`] for stable
    /// CPU performance investigation.
    pub fn decode_image_features_profiled(
        &self,
        features: &BackboneOutput,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        mask_input: Option<&Tensor>,
        multimask_output: bool,
    ) -> Result<(MaskDecoderOutput, MaskDecoderProfile)> {
        let prompt_start = Instant::now();
        let point_tensor = (!points.is_empty())
            .then(|| {
                Tensor::from_vec(
                    points.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                    (1, points.len(), 2usize),
                    &self.device,
                )
            })
            .transpose()?;
        let label_tensor = (!points.is_empty())
            .then(|| {
                Tensor::from_vec(
                    points
                        .iter()
                        .map(|p| i64::from(p.positive))
                        .collect::<Vec<_>>(),
                    (1, points.len()),
                    &self.device,
                )
            })
            .transpose()?;
        let box_tensor = bounding_box
            .map(|b| Tensor::from_vec(vec![b.x0, b.y0, b.x1, b.y1], (1, 4usize), &self.device))
            .transpose()?;
        let prompts = self.prompt_encoder.forward(
            point_tensor.as_ref().zip(label_tensor.as_ref()),
            box_tensor.as_ref(),
            mask_input,
        )?;
        let prompt_encoder_ms = prompt_start.elapsed().as_secs_f64() * 1000.;
        let dense_pe_start = Instant::now();
        let dense_pe = self.prompt_encoder.get_dense_pe()?;
        let dense_pe_ms = dense_pe_start.elapsed().as_secs_f64() * 1000.;
        let (output, mut profile) = if let Some(high_res_features) = &features.sam_high_res_features
        {
            self.mask_decoder.forward_profiled_preprojected(
                &features.vision_features,
                &dense_pe,
                &prompts.sparse_embeddings,
                &prompts.dense_embeddings,
                multimask_output,
                high_res_features,
            )?
        } else {
            self.mask_decoder.forward_profiled(
                &features.vision_features,
                &dense_pe,
                &prompts.sparse_embeddings,
                &prompts.dense_embeddings,
                multimask_output,
                &features.backbone_fpn[..2],
            )?
        };
        profile.prompt_encoder_ms = prompt_encoder_ms;
        profile.dense_pe_ms = dense_pe_ms;
        Ok((output, profile))
    }

    /// Convenience API for interleaved RGB8 pixels in original image
    /// coordinates. Points and boxes use that same original coordinate system.
    pub fn predict_rgb8(
        &self,
        rgb_hwc: &[u8],
        width: usize,
        height: usize,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
    ) -> Result<ImagePrediction> {
        let transforms = self.transforms;
        let prepared = transforms.prepare_rgb8(rgb_hwc, width, height, &self.device)?;
        let points: Vec<_> = points
            .iter()
            .map(|point| {
                let (x, y) = prepared.geometry.model_point(point.x, point.y);
                PointPrompt {
                    x,
                    y,
                    positive: point.positive,
                }
            })
            .collect();
        let bounding_box = bounding_box.map(|bounds| {
            let (x0, y0) = prepared.geometry.model_point(bounds.x0, bounds.y0);
            let (x1, y1) = prepared.geometry.model_point(bounds.x1, bounds.y1);
            BoxPrompt { x0, y0, x1, y1 }
        });
        let low_res = self.predict(&prepared.tensor, &points, bounding_box, multimask_output)?;
        let masks = transforms.postprocess_masks(
            &low_res.masks,
            prepared.geometry.original_height,
            prepared.geometry.original_width,
        )?;
        Ok(ImagePrediction { low_res, masks })
    }
}

impl Sam2AutomaticMaskGenerator {
    /// Downloads the published checkpoint through the Hugging Face cache and
    /// constructs the automatic mask generator on CPU.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained(variant: Sam2Variant, config: AutomaticMaskConfig) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1(variant, path, config)
    }

    /// Downloads the published checkpoint and constructs the automatic mask
    /// generator on `device`.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained_on_device(
        variant: Sam2Variant,
        device: &Device,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        let path = crate::download_checkpoint(variant)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        Self::load_sam2_1_on_device(variant, path, device, config)
    }

    pub fn load_sam2_1(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        Self::load_sam2_1_on_device(variant, path, &Device::Cpu, config)
    }

    /// Loads the generator onto `device`. Automatic-mask metadata and RLE
    /// construction remain on the host, while image and prompt inference use
    /// the selected backend.
    pub fn load_sam2_1_on_device(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        device: &Device,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            predictor: Sam2ImagePredictor::load_sam2_1_on_device_with_config(
                variant,
                path,
                device,
                ImagePredictorConfig {
                    max_hole_area: config.min_mask_region_area,
                    max_sprinkle_area: config.min_mask_region_area,
                    ..Default::default()
                },
            )?,
            config,
        })
    }

    pub fn load_sam2_1_hiera_tiny(
        path: impl AsRef<Path>,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        Self::load_sam2_1(Sam2Variant::HieraTiny, path, config)
    }

    pub fn generate_rgb8(
        &self,
        rgb_hwc: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<AutomaticMask>> {
        self.config.validate()?;
        if rgb_hwc.len() != width * height * 3 {
            candle_core::bail!("expected {} interleaved RGB bytes", width * height * 3)
        }
        let grids = self.config.point_grids()?;
        let (crop_boxes, crop_layers) = generate_crop_boxes(
            height,
            width,
            self.config.crop_n_layers,
            self.config.crop_overlap_ratio as f64,
        );
        let mut predictor = self.predictor.clone();
        let mut candidates = Vec::new();
        for (crop, layer) in crop_boxes.iter().copied().zip(crop_layers) {
            candidates.extend(self.process_crop(
                &mut predictor,
                rgb_hwc,
                width,
                height,
                crop,
                &grids[layer],
            )?);
        }
        if crop_boxes.len() > 1 {
            candidates =
                filter_candidate_nms(candidates, self.config.crop_nms_thresh, |candidate| {
                    let width = candidate.crop_box[2] - candidate.crop_box[0];
                    let height = candidate.crop_box[3] - candidate.crop_box[1];
                    1. / (width * height).max(1.)
                });
        }
        Ok(candidates
            .into_iter()
            .map(|candidate| candidate.finish(self.config.output_mode))
            .collect::<Result<Vec<_>>>()?)
    }

    /// Decodes an image file and generates masks for it.
    #[cfg(feature = "image-decoding")]
    pub fn generate_path(&self, path: impl AsRef<Path>) -> Result<Vec<AutomaticMask>> {
        let image = crate::decode_image(path)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        self.generate_rgb8(image.pixels(), image.width(), image.height())
    }

    fn process_crop(
        &self,
        predictor: &mut Sam2ImagePredictor,
        image: &[u8],
        original_width: usize,
        original_height: usize,
        crop: [usize; 4],
        grid: &[[f32; 2]],
    ) -> Result<Vec<AutomaticMaskCandidate>> {
        let profile = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
        let [x0, y0, x1, y1] = crop;
        let crop_width = x1 - x0;
        let crop_height = y1 - y0;
        let mut pixels = Vec::with_capacity(crop_width * crop_height * 3);
        for y in y0..y1 {
            let start = (y * original_width + x0) * 3;
            pixels.extend_from_slice(&image[start..start + crop_width * 3]);
        }
        if profile {
            predictor.device().synchronize()?;
        }
        let encode_start = Instant::now();
        predictor.set_image(&pixels, crop_width, crop_height)?;
        if profile {
            predictor.device().synchronize()?;
            eprintln!(
                "SAM2_AMG_STAGE_RS stage=image_encoder ms={:.6}",
                encode_start.elapsed().as_secs_f64() * 1000.
            );
        }
        let points = grid
            .iter()
            .map(|point| [point[0] * crop_width as f32, point[1] * crop_height as f32])
            .collect::<Vec<_>>();
        let mut candidates = Vec::new();
        for range in batch_ranges(points.len(), self.config.points_per_batch)? {
            candidates.extend(self.process_point_batch(
                predictor,
                &points[range],
                crop,
                original_width,
                original_height,
            )?);
        }
        predictor.reset_predictor();
        Ok(filter_candidate_nms(
            candidates,
            self.config.box_nms_thresh,
            |candidate| candidate.predicted_iou,
        ))
    }

    fn process_point_batch(
        &self,
        predictor: &Sam2ImagePredictor,
        points: &[[f32; 2]],
        crop: [usize; 4],
        original_width: usize,
        original_height: usize,
    ) -> Result<Vec<AutomaticMaskCandidate>> {
        let profile = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
        let point_values = points.iter().flatten().copied().collect::<Vec<_>>();
        let point_tensor =
            Tensor::from_vec(point_values, (points.len(), 1, 2), predictor.device())?;
        let labels = Tensor::ones((points.len(), 1), DType::I64, predictor.device())?;
        if profile {
            predictor.device().synchronize()?;
        }
        let initial_start = Instant::now();
        let prompts = ImagePromptBatch {
            point_coords: Some(point_tensor.clone()),
            point_labels: Some(labels.clone()),
            normalize_coords: true,
            ..Default::default()
        };
        let (initial_low_res, initial_geometry) =
            predictor.decode_cached_index_raw(0, &prompts, self.config.multimask_output)?;
        if std::env::var_os("SAM2_PROFILE_A2_BOUNDARY").is_some() {
            static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let batch_index = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 16;
            if !self.config.use_m2m
                && batch_index == 2
                && initial_low_res.masks.dim(0).is_ok_and(|count| count > 18)
            {
                let restored = predictor.transforms.postprocess_masks(
                    &initial_low_res.masks,
                    initial_geometry.original_height,
                    initial_geometry.original_width,
                )?;
                let logits = restored
                    .narrow(0, 18, 1)?
                    .narrow(2, 79, 1)?
                    .narrow(3, 495, 1)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let ious = initial_low_res
                    .iou_predictions
                    .narrow(0, 18, 1)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let low_res = initial_low_res
                    .masks
                    .narrow(0, 18, 1)?
                    .narrow(1, 1, 1)?
                    .narrow(2, 37, 2)?
                    .narrow(3, 158, 2)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                // Multimask output index 1 is internal mask token 2. Capture
                // both 32-wide projection operands at the four contributing
                // native mask pixels to match the Python A2 diagnostic.
                let mask_token2 = initial_low_res
                    .mask_tokens
                    .narrow(0, 18, 1)?
                    .narrow(1, 1, 1)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let hyper_token2 = initial_low_res
                    .hyper_in
                    .narrow(0, 18, 1)?
                    .narrow(1, 2, 1)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                let upscaled = initial_low_res
                    .upscaled_embedding
                    .narrow(0, 18, 1)?
                    .narrow(2, 37, 2)?
                    .narrow(3, 158, 2)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                eprintln!(
                    "SAM2_A2_BOUNDARY_RS batch={batch_index} logits={logits:?} \
                     ious={ious:?} low_res_y37_38_x158_159={low_res:?} \
                     mask_token2={mask_token2:?} \
                     hyper_token2={hyper_token2:?} \
                     upscaled_y37_38_x158_159={upscaled:?}"
                );
            }
        }
        if profile {
            predictor.device().synchronize()?;
            eprintln!(
                "SAM2_AMG_STAGE_RS stage=initial_decoder ms={:.6}",
                initial_start.elapsed().as_secs_f64() * 1000.
            );
        }
        let (masks, ious, stability_scores, point_indices) = if self.config.use_m2m {
            // Match the public predictor's low-resolution return contract
            // before feeding these logits back as a refinement prompt. Move
            // out only the masks so the unused upscaled embedding, token
            // projections, and score tensors drop before refinement starts.
            let MaskDecoderOutput {
                masks: initial_low_res_masks,
                iou_predictions,
                mask_tokens,
                object_score_logits,
                upscaled_embedding,
                hyper_in,
            } = initial_low_res;
            drop((
                iou_predictions,
                mask_tokens,
                object_score_logits,
                upscaled_embedding,
                hyper_in,
            ));
            let initial_low_res_masks = initial_low_res_masks.clamp(-32., 32.)?;
            let masks_per_point = initial_low_res_masks.dim(1)?;
            let count = points.len() * masks_per_point;
            let repeated_points = point_tensor
                .broadcast_as((points.len(), masks_per_point, 2))?
                .reshape((count, 1, 2))?;
            let low_res_masks = initial_low_res_masks.reshape((count, 1, 256, 256))?;
            // Preserve the upstream batch bound: one initial point can yield
            // three candidates, so refining all candidates in one call would
            // silently triple the configured decoder batch and its GPU working
            // set.
            let mut refined_masks = Vec::new();
            let mut kept_ious = Vec::new();
            let mut kept_stability = Vec::new();
            let mut kept_point_indices = Vec::new();
            let mut refine_ms = 0.;
            for range in batch_ranges(count, self.config.points_per_batch)? {
                let batch_len = range.end - range.start;
                if profile {
                    predictor.device().synchronize()?;
                }
                let refine_start = Instant::now();
                let (refined, geometry) = predictor.decode_cached_index_raw(
                    0,
                    &ImagePromptBatch {
                        point_coords: Some(repeated_points.narrow(0, range.start, batch_len)?),
                        point_labels: Some(Tensor::ones(
                            (batch_len, 1),
                            DType::I64,
                            predictor.device(),
                        )?),
                        mask_input: Some(low_res_masks.narrow(0, range.start, batch_len)?),
                        normalize_coords: true,
                        ..Default::default()
                    },
                    false,
                )?;
                if profile {
                    predictor.device().synchronize()?;
                    refine_ms += refine_start.elapsed().as_secs_f64() * 1000.;
                }
                let batch_point_indices = (range.start..range.end)
                    .map(|index| index / masks_per_point)
                    .collect::<Vec<_>>();
                let MaskDecoderOutput {
                    masks: refined_low_res_masks,
                    iou_predictions,
                    mask_tokens,
                    object_score_logits,
                    upscaled_embedding,
                    hyper_in,
                } = refined;
                drop((
                    mask_tokens,
                    object_score_logits,
                    upscaled_embedding,
                    hyper_in,
                ));
                let (masks, ious, stability, point_indices) = filter_amg_low_res_quality_on_device(
                    refined_low_res_masks,
                    iou_predictions,
                    batch_point_indices,
                    geometry,
                    &predictor.transforms,
                    &self.config,
                )?;
                refined_masks.push(masks);
                kept_ious.extend(ious);
                kept_stability.extend(stability);
                kept_point_indices.extend(point_indices);
            }
            if profile {
                eprintln!("SAM2_AMG_STAGE_RS stage=refine_decoder ms={:.6}", refine_ms);
            }
            let mask_refs = refined_masks.iter().collect::<Vec<_>>();
            (
                Tensor::cat(&mask_refs, 0)?,
                kept_ious,
                kept_stability,
                kept_point_indices,
            )
        } else {
            let masks_per_point = initial_low_res.masks.dim(1)?;
            let count = points.len() * masks_per_point;
            filter_amg_low_res_quality_on_device(
                initial_low_res.masks.reshape((count, 1, 256, 256))?,
                initial_low_res.iou_predictions.reshape(count)?,
                (0..count)
                    .map(|index| index / masks_per_point)
                    .collect::<Vec<_>>(),
                initial_geometry,
                &predictor.transforms,
                &self.config,
            )?
        };
        if profile {
            predictor.device().synchronize()?;
        }
        let postprocess_start = Instant::now();
        let mask_width = crop[2] - crop[0];
        let mask_height = crop[3] - crop[1];
        let pixels_per_mask = mask_width * mask_height;
        let transfer_start = Instant::now();
        #[cfg(feature = "cuda")]
        let packed_masks = if masks.device().is_cuda()
            && masks.dtype() == DType::F32
            && std::env::var_os("SAM2_CUDA_PACK_AMG_MASKS").is_none_or(|value| value != "0")
        {
            Some(
                masks
                    .apply_op1_no_bwd(&CudaThresholdPack {
                        threshold: self.config.mask_threshold as f32,
                    })?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let packed_masks: Option<Vec<u32>> = None;
        let byte_masks = if packed_masks.is_none() {
            masks
                .gt(self.config.mask_threshold as f64)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u8>()?
        } else {
            Vec::new()
        };
        if profile {
            eprintln!(
                "SAM2_AMG_STAGE_RS stage=threshold_transfer ms={:.6}",
                transfer_start.elapsed().as_secs_f64() * 1000.
            );
        }
        let words_per_mask = pixels_per_mask.div_ceil(32);
        let metadata_start = Instant::now();
        let candidates = (0..ious.len())
            .into_par_iter()
            .map(|index| -> Result<Option<AutomaticMaskCandidate>> {
                let metadata = if let Some(packed_masks) = &packed_masks {
                    let mask = &packed_masks[index * words_per_mask..(index + 1) * words_per_mask];
                    packed_binary_mask_metadata_flat(mask, mask_width, mask_height)
                } else {
                    let mask = &byte_masks[index * pixels_per_mask..(index + 1) * pixels_per_mask];
                    binary_mask_metadata_flat(mask, mask_width, mask_height)
                };
                let Some((local_rle, local_box, _)) = metadata else {
                    return Ok(None);
                };
                if box_near_crop_edge(local_box, crop, original_width, original_height, 20.) {
                    return Ok(None);
                }
                let rle = uncrop_rle(&local_rle, crop, original_height, original_width)?;
                let global_box = [
                    local_box[0] + crop[0] as f32,
                    local_box[1] + crop[1] as f32,
                    local_box[2] + crop[0] as f32,
                    local_box[3] + crop[1] as f32,
                ];
                let point = points[point_indices[index]];
                Ok(Some(AutomaticMaskCandidate {
                    rle,
                    box_xyxy: global_box,
                    crop_box: crop.map(|value| value as f32),
                    predicted_iou: ious[index],
                    point: [point[0] + crop[0] as f32, point[1] + crop[1] as f32],
                    stability_score: stability_scores[index],
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        let output = candidates.into_iter().flatten().collect::<Vec<_>>();
        if profile {
            eprintln!(
                "SAM2_AMG_STAGE_RS stage=metadata_rle ms={:.6}",
                metadata_start.elapsed().as_secs_f64() * 1000.
            );
            eprintln!(
                "SAM2_AMG_STAGE_RS stage=transfer_filter_rle ms={:.6}",
                postprocess_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(output)
    }
}

/// Applies the AMG IoU gate while masks are still at the decoder's native
/// resolution, then restores only surviving masks before computing the
/// full-resolution stability score. IoU is independent of mask restoration,
/// and postprocessing operates independently per mask, so this preserves the
/// reference ordering and values while avoiding discarded interpolations.
fn filter_amg_low_res_quality_on_device(
    low_res_masks: Tensor,
    ious: Tensor,
    point_indices: Vec<usize>,
    geometry: ImageGeometry,
    transforms: &Sam2ImageTransforms,
    config: &AutomaticMaskConfig,
) -> Result<(Tensor, Vec<f32>, Vec<f32>, Vec<usize>)> {
    let profile = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
    let profile_device = low_res_masks.device().clone();
    if profile {
        profile_device.synchronize()?;
    }
    let mut profile_start = Instant::now();
    macro_rules! profile_quality_stage {
        ($name:expr) => {
            if profile {
                profile_device.synchronize()?;
                eprintln!(
                    "SAM2_AMG_QUALITY_STAGE stage={} ms={:.6}",
                    $name,
                    profile_start.elapsed().as_secs_f64() * 1000.
                );
                profile_start = Instant::now();
            }
        };
    }
    let host_ious = ious
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    if host_ious.len() != point_indices.len() {
        candle_core::bail!("AMG mask, score, and point counts differ")
    }
    let keep_iou = host_ious
        .iter()
        .enumerate()
        .filter_map(|(index, &iou)| {
            (config.pred_iou_thresh <= 0. || iou > config.pred_iou_thresh).then_some(index)
        })
        .collect::<Vec<_>>();
    profile_quality_stage!("iou_transfer_and_filter");
    if keep_iou.is_empty() {
        if std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some() {
            eprintln!(
                "SAM2_AMG_STAGE_RS stage=quality_counts input={} iou=0 final=0",
                point_indices.len()
            );
        }
        return Ok((
            Tensor::zeros(
                (0, 1, geometry.original_height, geometry.original_width),
                DType::F32,
                low_res_masks.device(),
            )?,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
    }
    let selected_low_res = if keep_iou.len() == point_indices.len() {
        low_res_masks
    } else {
        let indices = Tensor::from_vec(
            keep_iou
                .iter()
                .map(|&index| index as u32)
                .collect::<Vec<_>>(),
            keep_iou.len(),
            low_res_masks.device(),
        )?;
        low_res_masks.index_select(&indices, 0)?
    };
    profile_quality_stage!("select_iou_masks");
    let direct_stability = if selected_low_res.device().is_cpu()
        && selected_low_res.dtype() == DType::F32
        && !transforms.has_mask_cleanup()
        && std::env::var_os("SAM2_CPU_FUSED_AMG_STABILITY").is_none_or(|value| value != "0")
    {
        let counts = selected_low_res
            .apply_op1_no_bwd(&CpuBilinearStabilityCounts {
                output_height: geometry.original_height,
                output_width: geometry.original_width,
                high_threshold: config.mask_threshold + config.stability_score_offset,
                low_threshold: config.mask_threshold - config.stability_score_offset,
            })?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<u32>()?;
        Some(
            counts
                .chunks_exact(2)
                .map(|pair| pair[0] as f32 / pair[1] as f32)
                .collect::<Vec<_>>(),
        )
    } else {
        #[cfg(feature = "cuda")]
        {
            if selected_low_res.device().is_cuda()
                && !transforms.has_mask_cleanup()
                && std::env::var_os("SAM2_CUDA_FUSED_AMG_STABILITY")
                    .is_none_or(|value| value != "0")
            {
                let counts = selected_low_res
                    .apply_op1_no_bwd(&CudaBilinearStabilityCounts {
                        output_height: geometry.original_height,
                        output_width: geometry.original_width,
                        high_threshold: config.mask_threshold + config.stability_score_offset,
                        low_threshold: config.mask_threshold - config.stability_score_offset,
                    })?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?;
                Some(
                    counts
                        .chunks_exact(2)
                        .map(|pair| pair[0] as f32 / pair[1] as f32)
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    };
    if direct_stability.is_some() {
        profile_quality_stage!("direct_stability_counts");
    }
    let (restored, stability) = if let Some(stability) = direct_stability {
        (None, stability)
    } else {
        let restored = transforms.postprocess_masks(
            &selected_low_res,
            geometry.original_height,
            geometry.original_width,
        )?;
        profile_quality_stage!("restore_iou_masks");
        let stability = calculate_stability_score(
            &restored,
            config.mask_threshold as f64,
            config.stability_score_offset as f64,
        )?;
        profile_quality_stage!("calculate_stability");
        let stability = stability
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        profile_quality_stage!("stability_transfer");
        (Some(restored), stability)
    };
    if stability.len() != keep_iou.len() {
        candle_core::bail!("AMG mask, score, and point counts differ")
    }
    let keep_stability = stability
        .iter()
        .enumerate()
        .filter_map(|(index, &score)| {
            (config.stability_score_thresh <= 0. || score >= config.stability_score_thresh)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    profile_quality_stage!("filter_stability");
    if std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some() {
        eprintln!(
            "SAM2_AMG_STAGE_RS stage=quality_counts input={} iou={} final={}",
            point_indices.len(),
            keep_iou.len(),
            keep_stability.len()
        );
    }
    let selected_masks = if keep_stability.is_empty() {
        Tensor::zeros(
            (0, 1, geometry.original_height, geometry.original_width),
            DType::F32,
            selected_low_res.device(),
        )?
    } else if let Some(restored) = restored {
        if keep_stability.len() == keep_iou.len() {
            restored
        } else {
            let indices = Tensor::from_vec(
                keep_stability
                    .iter()
                    .map(|&index| index as u32)
                    .collect::<Vec<_>>(),
                keep_stability.len(),
                restored.device(),
            )?;
            restored.index_select(&indices, 0)?
        }
    } else {
        let indices = Tensor::from_vec(
            keep_stability
                .iter()
                .map(|&index| index as u32)
                .collect::<Vec<_>>(),
            keep_stability.len(),
            selected_low_res.device(),
        )?;
        let final_low_res = if keep_stability.len() == keep_iou.len() {
            selected_low_res
        } else {
            selected_low_res.index_select(&indices, 0)?
        };
        transforms.postprocess_masks(
            &final_low_res,
            geometry.original_height,
            geometry.original_width,
        )?
    };
    if profile {
        profile_device.synchronize()?;
        eprintln!(
            "SAM2_AMG_QUALITY_STAGE stage=select_final_masks ms={:.6}",
            profile_start.elapsed().as_secs_f64() * 1000.
        );
    }
    Ok((
        selected_masks,
        keep_stability
            .iter()
            .map(|&index| host_ious[keep_iou[index]])
            .collect(),
        keep_stability
            .iter()
            .map(|&index| stability[index])
            .collect(),
        keep_stability
            .iter()
            .map(|&index| point_indices[keep_iou[index]])
            .collect(),
    ))
}

impl AutomaticMaskConfig {
    fn validate(&self) -> Result<()> {
        if self.points_per_side.is_some() == self.point_grids.is_some() {
            candle_core::bail!("set exactly one of points_per_side and point_grids")
        }
        if self.points_per_batch == 0 || self.crop_n_points_downscale_factor == 0 {
            candle_core::bail!("AMG batch and grid scale values must be positive")
        }
        if let Some(grids) = &self.point_grids
            && grids.len() != self.crop_n_layers + 1
        {
            candle_core::bail!("point_grids must contain one grid per crop layer")
        }
        Ok(())
    }

    fn point_grids(&self) -> Result<Vec<Vec<[f32; 2]>>> {
        match (&self.point_grids, self.points_per_side) {
            (Some(grids), None) => Ok(grids.clone()),
            (None, Some(points)) => build_all_layer_point_grids(
                points,
                self.crop_n_layers,
                self.crop_n_points_downscale_factor,
            ),
            _ => candle_core::bail!("set exactly one AMG point-grid source"),
        }
    }
}

#[derive(Clone, Debug)]
struct AutomaticMaskCandidate {
    rle: MaskRle,
    box_xyxy: [f32; 4],
    crop_box: [f32; 4],
    predicted_iou: f32,
    point: [f32; 2],
    stability_score: f32,
}

impl AutomaticMaskCandidate {
    fn finish(self, mode: AutomaticMaskOutputMode) -> Result<AutomaticMask> {
        let area = self.rle.area();
        let segmentation = match mode {
            AutomaticMaskOutputMode::BinaryMask => AutomaticMaskSegmentation::Binary {
                width: self.rle.width,
                height: self.rle.height,
                row_major: self.rle.to_row_major_mask()?,
            },
            AutomaticMaskOutputMode::UncompressedRle => {
                AutomaticMaskSegmentation::UncompressedRle(self.rle.clone())
            }
            AutomaticMaskOutputMode::CocoRle => AutomaticMaskSegmentation::CocoRle {
                width: self.rle.width,
                height: self.rle.height,
                counts: self.rle.to_coco_counts()?,
            },
        };
        Ok(AutomaticMask {
            segmentation,
            bbox: [
                self.box_xyxy[0],
                self.box_xyxy[1],
                self.box_xyxy[2] - self.box_xyxy[0],
                self.box_xyxy[3] - self.box_xyxy[1],
            ],
            area,
            predicted_iou: self.predicted_iou,
            point: self.point,
            stability_score: self.stability_score,
            crop_box: [
                self.crop_box[0],
                self.crop_box[1],
                self.crop_box[2] - self.crop_box[0],
                self.crop_box[3] - self.crop_box[1],
            ],
        })
    }
}

fn binary_mask_metadata_flat(
    mask: &[u8],
    width: usize,
    height: usize,
) -> Option<(MaskRle, [f32; 4], usize)> {
    let mut area = 0usize;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut counts = Vec::new();
    let mut current = false;
    let mut run = 0usize;
    for x in 0..width {
        for y in 0..height {
            let value = mask[y * width + x];
            let foreground = value != 0;
            if foreground {
                area += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
            if foreground == current {
                run += 1;
            } else {
                counts.push(run);
                run = 1;
                current = foreground;
            }
        }
    }
    counts.push(run);
    if area == 0 {
        return None;
    }
    Some((
        MaskRle {
            width,
            height,
            counts,
        },
        [min_x as f32, min_y as f32, max_x as f32, max_y as f32],
        area,
    ))
}

fn packed_binary_mask_metadata_flat(
    mask: &[u32],
    width: usize,
    height: usize,
) -> Option<(MaskRle, [f32; 4], usize)> {
    let pixels = width * height;
    if mask.len() != pixels.div_ceil(32) {
        return None;
    }
    let mut area = 0usize;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut counts = Vec::new();
    let mut current = false;
    let mut run = 0usize;
    for x in 0..width {
        for y in 0..height {
            let pixel = y * width + x;
            let foreground = (mask[pixel / 32] >> (pixel % 32)) & 1 != 0;
            if foreground {
                area += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
            if foreground == current {
                run += 1;
            } else {
                counts.push(run);
                run = 1;
                current = foreground;
            }
        }
    }
    counts.push(run);
    if area == 0 {
        return None;
    }
    Some((
        MaskRle {
            width,
            height,
            counts,
        },
        [min_x as f32, min_y as f32, max_x as f32, max_y as f32],
        area,
    ))
}

fn box_iou_xyxy(a: [f32; 4], b: [f32; 4]) -> f32 {
    let right = a[2].min(b[2]);
    let bottom = a[3].min(b[3]);
    let width = (right - a[0].max(b[0])).max(0.);
    let height = (bottom - a[1].max(b[1])).max(0.);
    let intersection = width * height;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    intersection / (area_a + area_b - intersection)
}

fn filter_candidate_nms(
    mut proposals: Vec<AutomaticMaskCandidate>,
    threshold: f32,
    score: impl Fn(&AutomaticMaskCandidate) -> f32,
) -> Vec<AutomaticMaskCandidate> {
    proposals.sort_by(|a, b| score(b).total_cmp(&score(a)));
    let mut selected: Vec<AutomaticMaskCandidate> = Vec::new();
    'candidate: for proposal in proposals {
        for retained in &selected {
            if box_iou_xyxy(proposal.box_xyxy, retained.box_xyxy) > threshold {
                continue 'candidate;
            }
        }
        selected.push(proposal);
    }
    selected
}

fn box_near_crop_edge(
    local: [f32; 4],
    crop: [usize; 4],
    original_width: usize,
    original_height: usize,
    tolerance: f32,
) -> bool {
    let global = [
        local[0] + crop[0] as f32,
        local[1] + crop[1] as f32,
        local[2] + crop[0] as f32,
        local[3] + crop[1] as f32,
    ];
    let crop = crop.map(|value| value as f32);
    let original = [0., 0., original_width as f32, original_height as f32];
    (0..4).any(|index| {
        (global[index] - crop[index]).abs() <= tolerance
            && (global[index] - original[index]).abs() > tolerance
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_cpu_bilinear_stability_counts_match_composed_ops() -> Result<()> {
        let input_height = 5;
        let input_width = 7;
        let output_height = 13;
        let output_width = 19;
        let values = (0..3 * input_height * input_width)
            .map(|index| ((index * 37 % 101) as f32 - 50.) / 31.)
            .collect::<Vec<_>>();
        let masks = Tensor::from_vec(values, (3, 1, input_height, input_width), &Device::Cpu)?;
        let high_threshold = 0.23f32;
        let low_threshold = -0.17f32;
        let expected = masks
            .upsample_bilinear2d(output_height, output_width, false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .chunks_exact(output_height * output_width)
            .map(|mask| {
                mask.iter().fold([0u32; 2], |mut counts, &value| {
                    counts[0] += u32::from(value > high_threshold);
                    counts[1] += u32::from(value > low_threshold);
                    counts
                })
            })
            .collect::<Vec<_>>();
        let actual = masks
            .apply_op1_no_bwd(&CpuBilinearStabilityCounts {
                output_height,
                output_width,
                high_threshold,
                low_threshold,
            })?
            .to_vec2::<u32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn automatic_mask_metadata_is_coco_column_major_and_has_tight_box() {
        let mask = vec![0, 1, 1, 0, 0, 1];
        let (rle, bbox, area) = binary_mask_metadata_flat(&mask, 3, 2).unwrap();
        assert_eq!(area, 3);
        assert_eq!(bbox, [1., 0., 2., 1.]);
        assert_eq!(rle.counts, vec![2, 1, 1, 2]);
    }

    #[test]
    fn packed_automatic_mask_metadata_matches_byte_masks_across_word_boundaries() {
        let width = 19;
        let height = 3;
        let mask = (0..width * height)
            .map(|index| u8::from(index % 7 == 1 || index % 11 == 4))
            .collect::<Vec<_>>();
        let mut packed = vec![0u32; mask.len().div_ceil(32)];
        for (index, &value) in mask.iter().enumerate() {
            packed[index / 32] |= u32::from(value != 0) << (index % 32);
        }
        let byte_metadata = binary_mask_metadata_flat(&mask, width, height).unwrap();
        let packed_metadata = packed_binary_mask_metadata_flat(&packed, width, height).unwrap();
        assert_eq!(packed_metadata, byte_metadata);
    }

    #[test]
    fn nms_iou_uses_xyxy_boxes() {
        assert!((box_iou_xyxy([0., 0., 2., 2.], [1., 1., 3., 3.]) - 1. / 7.).abs() < 1e-6);
    }

    #[test]
    fn crop_boxes_match_the_square_pyramid_and_uncrop_preserves_coordinates() {
        let (crops, _) = generate_crop_boxes(6, 8, 1, 0.);
        assert_eq!(crops.len(), 5);
        let local = MaskRle {
            width: 2,
            height: 2,
            counts: vec![1, 1, 2],
        };
        let full = uncrop_rle(&local, [3, 2, 5, 4], 6, 8).unwrap();
        assert_eq!(full.width, 8);
        assert_eq!(full.height, 6);
        assert_eq!(full.counts.iter().sum::<usize>(), 48);
    }

    #[test]
    fn automatic_mask_output_modes_share_the_same_canonical_rle() {
        let rle = MaskRle::from_row_major_mask(&[false, true, true, false], 2, 2).unwrap();
        let candidate = AutomaticMaskCandidate {
            rle: rle.clone(),
            box_xyxy: [0., 0., 1., 1.],
            crop_box: [0., 0., 2., 2.],
            predicted_iou: 1.,
            point: [0.5, 0.5],
            stability_score: 1.,
        };
        let binary = candidate
            .clone()
            .finish(AutomaticMaskOutputMode::BinaryMask)
            .unwrap();
        assert!(matches!(
            binary.segmentation,
            AutomaticMaskSegmentation::Binary { row_major, .. }
                if row_major == vec![false, true, true, false]
        ));
        let uncompressed = candidate
            .clone()
            .finish(AutomaticMaskOutputMode::UncompressedRle)
            .unwrap();
        assert!(matches!(
            uncompressed.segmentation,
            AutomaticMaskSegmentation::UncompressedRle(value) if value == rle
        ));
        let coco = candidate.finish(AutomaticMaskOutputMode::CocoRle).unwrap();
        assert!(matches!(
            coco.segmentation,
            AutomaticMaskSegmentation::CocoRle { counts, .. }
                if counts == rle.to_coco_counts().unwrap()
        ));
    }
}
