//! Public static-image SAM2 predictor.

use std::{collections::BTreeMap, path::Path, time::Instant};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    BackboneOutput, ImageEncoder, ImageEncoderProfile, MaskDecoder, MaskDecoderConfig,
    MaskDecoderOutput, MaskDecoderProfile, PromptEncoder, Sam2Base, Sam2Config, Sam2FrameOutput,
    Sam2ImageTransforms, Sam2Variant, VideoInferenceState,
    ops::{
        MaskRle, batch_ranges, build_all_layer_point_grids, generate_crop_boxes,
        restore_video_masks, uncrop_rle,
    },
};

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
    outputs: BTreeMap<u32, BTreeMap<usize, Sam2FrameOutput>>,
}

impl InteractiveVideoSession {
    pub fn state(&self) -> &VideoInferenceState {
        &self.state
    }

    pub fn output(&self, object_id: u32, frame_index: usize) -> Option<&Sam2FrameOutput> {
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
        let frames = images
            .iter()
            .map(|&(pixels, frame_width, frame_height)| {
                Ok(transforms
                    .prepare_rgb8(pixels, frame_width, frame_height, device)?
                    .tensor)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(InteractiveVideoSession {
            frames,
            state: VideoInferenceState::with_video(width, height, images.len())?,
            model_states: BTreeMap::new(),
            outputs: BTreeMap::new(),
        })
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
        Ok(InteractiveVideoSession {
            frames,
            state: VideoInferenceState::with_video(width, height, num_frames)?,
            model_states: BTreeMap::new(),
            outputs: BTreeMap::new(),
        })
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
        let output = self.postprocess_output(self.model.track_backbone_with_mask_prompt(
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
            .insert(frame_index, output.clone());
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
        let output = self.postprocess_output(self.model.track_mask_from_backbone(
            &feature,
            &model_mask,
            frame_index,
            true,
            session.model_states.entry(object_id).or_default(),
            false,
            Some(session.frames.len()),
        )?)?;
        session
            .outputs
            .entry(object_id)
            .or_default()
            .insert(frame_index, output.clone());
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
                // The reference compacts temporal memory to bfloat16 on the
                // storage device. Convert back to f32 for Candle's homogeneous
                // attention ops while preserving the same quantized values.
                memory.memory_features = memory
                    .memory_features
                    .to_dtype(DType::BF16)?
                    .to_dtype(DType::F32)?;
                if let Some(position) = cached_memory_pos.as_ref() {
                    memory.memory_pos_enc = position.clone();
                }
                output.memory_features = memory.memory_features.clone();
                output.memory_pos_enc = memory.memory_pos_enc.clone();
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
                let output = if existing_conditioning {
                    session.outputs[&object_id][&frame_index].clone()
                } else {
                    let output =
                        self.postprocess_output(self.model.track_backbone_with_mask_prompt(
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
                        .insert(frame_index, output.clone());
                    session
                        .state
                        .object_mut(object_id)
                        .non_conditioning_outputs
                        .insert(frame_index, Self::memory(&output));
                    output
                };
                session.state.mark_tracked(object_id, frame_index, reverse);
                masks.push(output.low_res_masks);
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

    pub fn load_sam2_1_on_device_with_config(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        device: &Device,
        predictor_config: ImagePredictorConfig,
    ) -> Result<Self> {
        if device.is_cpu() {
            crate::cpu_allocator::configure_cpu_allocator_for_inference();
        }
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
        self.reset_predictor();
        let prepared = self
            .transforms
            .prepare_rgb8(rgb_hwc, width, height, &self.device)?;
        let features = self.encode_image(&prepared.tensor)?;
        self.cached_images = Some(CachedImages {
            features,
            geometries: vec![prepared.geometry],
            is_batch: false,
        });
        Ok(())
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
        let prompt_embeddings = self.prompt_encoder.forward(
            coordinates.as_ref().zip(labels.as_ref()),
            None,
            mask_input,
        )?;
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
        let (features, geometry) = self.cached_features(image_index)?;
        let (coordinates, labels, boxes, mask) = self.prepare_cached_prompts(prompts, geometry)?;
        let mut low_res = self.decode_prepared_cached(
            &features,
            coordinates.as_ref(),
            labels.as_ref(),
            boxes.as_ref(),
            mask.as_ref(),
            multimask_output,
        )?;
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
        let mut features = self.image_encoder.forward(image)?;
        features.vision_features = features
            .vision_features
            .broadcast_add(&self.no_mem_embed.reshape((1, 256usize, 1usize, 1usize))?)?;
        if self.device.is_cpu()
            && std::env::var_os("SAM2_CPU_PREDICTOR_PREPROJECT_SKIPS")
                .is_none_or(|value| value != "0")
        {
            features.sam_high_res_features = Some(
                self.mask_decoder
                    .project_high_res_features(&features.backbone_fpn[..2])?,
            );
        }
        Ok(features)
    }

    /// Encodes an image with synchronized Hiera/FPN timings for optimization work.
    pub fn encode_image_profiled(
        &self,
        image: &Tensor,
    ) -> Result<(BackboneOutput, ImageEncoderProfile)> {
        let (mut features, profile) = self.image_encoder.forward_profiled(image)?;
        if self.device.is_cpu()
            && std::env::var_os("SAM2_CPU_PREDICTOR_PREPROJECT_SKIPS")
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

    pub fn load_sam2_1(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            predictor: Sam2ImagePredictor::load_sam2_1_on_device_with_config(
                variant,
                path,
                &Device::Cpu,
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
        let [x0, y0, x1, y1] = crop;
        let crop_width = x1 - x0;
        let crop_height = y1 - y0;
        let mut pixels = Vec::with_capacity(crop_width * crop_height * 3);
        for y in y0..y1 {
            let start = (y * original_width + x0) * 3;
            pixels.extend_from_slice(&image[start..start + crop_width * 3]);
        }
        predictor.set_image(&pixels, crop_width, crop_height)?;
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
        let point_values = points.iter().flatten().copied().collect::<Vec<_>>();
        let point_tensor =
            Tensor::from_vec(point_values, (points.len(), 1, 2), predictor.device())?;
        let labels = Tensor::ones((points.len(), 1), DType::I64, predictor.device())?;
        let initial = predictor.predict_cached(
            &ImagePromptBatch {
                point_coords: Some(point_tensor.clone()),
                point_labels: Some(labels.clone()),
                normalize_coords: true,
                ..Default::default()
            },
            self.config.multimask_output,
            true,
        )?;
        let (masks, ious, point_indices) = if self.config.use_m2m {
            let masks_per_point = initial.low_res.masks.dim(1)?;
            let count = points.len() * masks_per_point;
            let repeated_points = point_tensor
                .broadcast_as((points.len(), masks_per_point, 2))?
                .reshape((count, 1, 2))?;
            let refined = predictor.predict_cached(
                &ImagePromptBatch {
                    point_coords: Some(repeated_points),
                    point_labels: Some(Tensor::ones((count, 1), DType::I64, predictor.device())?),
                    mask_input: Some(initial.low_res.masks.reshape((count, 1, 256, 256))?),
                    normalize_coords: true,
                    ..Default::default()
                },
                false,
                true,
            )?;
            (
                refined.masks,
                refined.low_res.iou_predictions,
                (0..count)
                    .map(|index| index / masks_per_point)
                    .collect::<Vec<_>>(),
            )
        } else {
            let masks_per_point = initial.masks.dim(1)?;
            let count = points.len() * masks_per_point;
            (
                initial
                    .masks
                    .reshape((count, 1, crop[3] - crop[1], crop[2] - crop[0]))?,
                initial.low_res.iou_predictions.reshape(count)?,
                (0..count)
                    .map(|index| index / masks_per_point)
                    .collect::<Vec<_>>(),
            )
        };
        let masks = masks
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .squeeze(1)?
            .to_vec3::<f32>()?;
        let ious = ious
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let mut output = Vec::new();
        for (index, mask) in masks.iter().enumerate() {
            let iou = ious[index];
            if self.config.pred_iou_thresh > 0. && iou <= self.config.pred_iou_thresh {
                continue;
            }
            let stability = stability_score(
                mask,
                self.config.mask_threshold,
                self.config.stability_score_offset,
            );
            if self.config.stability_score_thresh > 0.
                && stability < self.config.stability_score_thresh
            {
                continue;
            }
            let Some((local_rle, local_box, _)) = binary_mask_metadata(
                mask,
                crop[2] - crop[0],
                crop[3] - crop[1],
                self.config.mask_threshold,
            ) else {
                continue;
            };
            if box_near_crop_edge(local_box, crop, original_width, original_height, 20.) {
                continue;
            }
            let rle = uncrop_rle(&local_rle, crop, original_height, original_width)?;
            let global_box = [
                local_box[0] + crop[0] as f32,
                local_box[1] + crop[1] as f32,
                local_box[2] + crop[0] as f32,
                local_box[3] + crop[1] as f32,
            ];
            let point = points[point_indices[index]];
            output.push(AutomaticMaskCandidate {
                rle,
                box_xyxy: global_box,
                crop_box: crop.map(|value| value as f32),
                predicted_iou: iou,
                point: [point[0] + crop[0] as f32, point[1] + crop[1] as f32],
                stability_score: stability,
            });
        }
        Ok(output)
    }
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

fn stability_score(mask: &[Vec<f32>], threshold: f32, offset: f32) -> f32 {
    let mut lower = 0usize;
    let mut upper = 0usize;
    for value in mask.iter().flatten() {
        lower += usize::from(*value > threshold - offset);
        upper += usize::from(*value > threshold + offset);
    }
    if lower == 0 {
        1.
    } else {
        upper as f32 / lower as f32
    }
}

fn binary_mask_metadata(
    mask: &[Vec<f32>],
    width: usize,
    height: usize,
    threshold: f32,
) -> Option<(MaskRle, [f32; 4], usize)> {
    let mut binary = Vec::with_capacity(width * height);
    let mut area = 0usize;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for (y, row) in mask.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            let foreground = *value > threshold;
            binary.push(foreground);
            if foreground {
                area += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if area == 0 {
        return None;
    }
    Some((
        MaskRle::from_row_major_mask(&binary, width, height).expect("validated mask dimensions"),
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
    fn automatic_mask_metadata_is_coco_column_major_and_has_tight_box() {
        let mask = vec![vec![-1., 1., 1.], vec![-1., -1., 1.]];
        let (rle, bbox, area) = binary_mask_metadata(&mask, 3, 2, 0.).unwrap();
        assert_eq!(area, 3);
        assert_eq!(bbox, [1., 0., 2., 1.]);
        assert_eq!(rle.counts, vec![2, 1, 1, 2]);
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
