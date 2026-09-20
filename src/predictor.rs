//! Public static-image SAM2 predictor.

use std::{collections::BTreeMap, path::Path, time::Instant};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::{
    BackboneOutput, ImageEncoder, ImageEncoderProfile, MaskDecoder, MaskDecoderOutput,
    MaskDecoderProfile, PromptEncoder, Sam2Base, Sam2Config, Sam2FrameOutput, Sam2ImageTransforms,
    Sam2Variant, VideoInferenceState,
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

/// Run-length encoding in row-major order, beginning with background pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskRle {
    pub width: usize,
    pub height: usize,
    pub counts: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct AutomaticMask {
    pub segmentation: MaskRle,
    pub bbox: [f32; 4],
    pub area: usize,
    pub predicted_iou: f32,
    pub point: [f32; 2],
    pub stability_score: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct AutomaticMaskConfig {
    pub points_per_side: usize,
    pub pred_iou_thresh: f32,
    pub stability_score_thresh: f32,
    pub stability_score_offset: f32,
    pub mask_threshold: f32,
    pub box_nms_thresh: f32,
    pub crop_n_layers: usize,
    pub crop_overlap_ratio: f32,
    pub crop_n_points_downscale_factor: usize,
    pub use_m2m: bool,
}

impl Default for AutomaticMaskConfig {
    fn default() -> Self {
        Self {
            points_per_side: 32,
            pred_iou_thresh: 0.8,
            stability_score_thresh: 0.95,
            stability_score_offset: 1.,
            mask_threshold: 0.,
            box_nms_thresh: 0.7,
            crop_n_layers: 0,
            crop_overlap_ratio: 512. / 1500.,
            crop_n_points_downscale_factor: 1,
            use_m2m: false,
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

impl Sam2VideoPredictor {
    pub fn load_sam2_1(variant: Sam2Variant, path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            model: Sam2Base::load_sam2_1(variant, path)?,
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

impl Sam2ImagePredictor {
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
        if device.is_cpu() {
            crate::cpu_allocator::configure_cpu_allocator_for_inference();
        }
        let path = path.as_ref();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }?;
        let config = Sam2Config::sam2_1(variant);
        Ok(Self {
            device: device.clone(),
            image_encoder: ImageEncoder::load(&config, vb.pp("image_encoder"))?,
            prompt_encoder: PromptEncoder::load(&config, vb.pp("sam_prompt_encoder"))?,
            mask_decoder: MaskDecoder::load(vb.pp("sam_mask_decoder"))?,
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
        let transforms = Sam2ImageTransforms::default();
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
        let masks = transforms.restore_masks(&low_res.masks, prepared.geometry)?;
        Ok(ImagePrediction { low_res, masks })
    }
}

impl Sam2AutomaticMaskGenerator {
    pub fn load_sam2_1(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        Ok(Self {
            predictor: Sam2ImagePredictor::load_sam2_1(variant, path)?,
            config,
        })
    }

    pub fn load_sam2_1_hiera_tiny(
        path: impl AsRef<Path>,
        config: AutomaticMaskConfig,
    ) -> Result<Self> {
        Self::load_sam2_1(Sam2Variant::HieraTiny, path, config)
    }

    /// Generates full-image grid proposals. Image encoding occurs once; every
    /// point subsequently runs only the prompt encoder and mask decoder.
    pub fn generate_rgb8(
        &self,
        rgb_hwc: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<AutomaticMask>> {
        assert!(self.config.points_per_side > 0);
        if self.config.crop_n_layers > 0 {
            if rgb_hwc.len() != width * height * 3 {
                candle_core::bail!(
                    "expected {} interleaved RGB bytes, got {}",
                    width * height * 3,
                    rgb_hwc.len()
                )
            }
            let mut all = Vec::new();
            for (crop, layer) in crop_boxes(
                width,
                height,
                self.config.crop_n_layers,
                self.config.crop_overlap_ratio,
            ) {
                let crop_width = crop.2 - crop.0;
                let crop_height = crop.3 - crop.1;
                let mut pixels = Vec::with_capacity(crop_width * crop_height * 3);
                for y in crop.1..crop.3 {
                    let start = (y * width + crop.0) * 3;
                    pixels.extend_from_slice(&rgb_hwc[start..start + crop_width * 3]);
                }
                let mut config = self.config;
                config.crop_n_layers = 0;
                config.points_per_side = (config.points_per_side
                    / config
                        .crop_n_points_downscale_factor
                        .max(1)
                        .pow(layer as u32))
                .max(1);
                let sub = Self {
                    predictor: self.predictor.clone(),
                    config,
                };
                for mut mask in sub.generate_rgb8(&pixels, crop_width, crop_height)? {
                    // Crop-local fragments that meet an interior crop edge are
                    // discarded before global NMS, matching SAM's edge guard.
                    let right = mask.bbox[0] + mask.bbox[2];
                    let bottom = mask.bbox[1] + mask.bbox[3];
                    if (crop.0 > 0 && mask.bbox[0] <= 0.)
                        || (crop.1 > 0 && mask.bbox[1] <= 0.)
                        || (crop.2 < width && right >= crop_width as f32)
                        || (crop.3 < height && bottom >= crop_height as f32)
                    {
                        continue;
                    }
                    mask.segmentation =
                        uncrop_rle(&mask.segmentation, crop.0, crop.1, width, height);
                    mask.bbox[0] += crop.0 as f32;
                    mask.bbox[1] += crop.1 as f32;
                    mask.point[0] += crop.0 as f32;
                    mask.point[1] += crop.1 as f32;
                    all.push(mask);
                }
            }
            return Ok(filter_box_nms(all, self.config.box_nms_thresh));
        }
        let transforms = Sam2ImageTransforms::default();
        let prepared = transforms.prepare_rgb8(rgb_hwc, width, height, &self.predictor.device)?;
        let features = self.predictor.image_encoder.forward(&prepared.tensor)?;
        let mut proposals = Vec::new();
        for gy in 0..self.config.points_per_side {
            for gx in 0..self.config.points_per_side {
                let point = [
                    (gx as f32 + 0.5) * width as f32 / self.config.points_per_side as f32,
                    (gy as f32 + 0.5) * height as f32 / self.config.points_per_side as f32,
                ];
                let (x, y) = prepared.geometry.model_point(point[0], point[1]);
                let output = self.predictor.decode_image_features(
                    &features,
                    &[PointPrompt {
                        x,
                        y,
                        positive: true,
                    }],
                    None,
                    None,
                    true,
                )?;
                let (masks, ious) = if self.config.use_m2m {
                    let mut refined_masks = Vec::new();
                    let mut refined_ious = Vec::new();
                    for index in 0..output.masks.dim(1)? {
                        let refinement = self.predictor.decode_image_features(
                            &features,
                            &[PointPrompt {
                                x,
                                y,
                                positive: true,
                            }],
                            None,
                            Some(&output.masks.narrow(1, index, 1)?),
                            false,
                        )?;
                        refined_masks.push(
                            transforms
                                .restore_masks(&refinement.masks, prepared.geometry)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .to_vec2::<f32>()?,
                        );
                        refined_ious.push(refinement.iou_predictions.to_vec2::<f32>()?[0][0]);
                    }
                    (refined_masks, refined_ious)
                } else {
                    (
                        transforms
                            .restore_masks(&output.masks, prepared.geometry)?
                            .squeeze(0)?
                            .to_vec3::<f32>()?,
                        output.iou_predictions.to_vec2::<f32>()?[0].clone(),
                    )
                };
                for (index, mask) in masks.iter().enumerate() {
                    let predicted_iou = ious[index];
                    if predicted_iou < self.config.pred_iou_thresh {
                        continue;
                    }
                    let stability = stability_score(
                        mask,
                        self.config.mask_threshold,
                        self.config.stability_score_offset,
                    );
                    if stability < self.config.stability_score_thresh {
                        continue;
                    }
                    if let Some((segmentation, bbox, area)) =
                        binary_mask_metadata(mask, width, height, self.config.mask_threshold)
                    {
                        proposals.push(AutomaticMask {
                            segmentation,
                            bbox,
                            area,
                            predicted_iou,
                            point,
                            stability_score: stability,
                        });
                    }
                }
            }
        }
        Ok(filter_box_nms(proposals, self.config.box_nms_thresh))
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
    let mut counts = Vec::new();
    let mut current = false;
    let mut run = 0usize;
    for value in binary {
        if value == current {
            run += 1;
        } else {
            counts.push(run);
            run = 1;
            current = value;
        }
    }
    counts.push(run);
    Some((
        MaskRle {
            width,
            height,
            counts,
        },
        [
            min_x as f32,
            min_y as f32,
            (max_x - min_x + 1) as f32,
            (max_y - min_y + 1) as f32,
        ],
        area,
    ))
}

fn box_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let right = (a[0] + a[2]).min(b[0] + b[2]);
    let bottom = (a[1] + a[3]).min(b[1] + b[3]);
    let width = (right - a[0].max(b[0])).max(0.);
    let height = (bottom - a[1].max(b[1])).max(0.);
    let intersection = width * height;
    intersection / (a[2] * a[3] + b[2] * b[3] - intersection)
}

fn filter_box_nms(mut proposals: Vec<AutomaticMask>, threshold: f32) -> Vec<AutomaticMask> {
    proposals.sort_by(|a, b| b.predicted_iou.total_cmp(&a.predicted_iou));
    let mut selected: Vec<AutomaticMask> = Vec::new();
    'candidate: for proposal in proposals {
        for retained in &selected {
            if box_iou(proposal.bbox, retained.bbox) > threshold {
                continue 'candidate;
            }
        }
        selected.push(proposal);
    }
    selected
}

fn crop_boxes(
    width: usize,
    height: usize,
    layers: usize,
    overlap_ratio: f32,
) -> Vec<((usize, usize, usize, usize), usize)> {
    let mut boxes = vec![((0, 0, width, height), 0)];
    let short = width.min(height) as f32;
    for layer in 0..layers {
        let side = 2usize.pow(layer as u32 + 1);
        let overlap = (overlap_ratio * short * 2. / side as f32) as usize;
        let crop_w = (overlap * (side - 1) + width).div_ceil(side);
        let crop_h = (overlap * (side - 1) + height).div_ceil(side);
        for x in 0..side {
            for y in 0..side {
                let x0 = (crop_w - overlap) * x;
                let y0 = (crop_h - overlap) * y;
                boxes.push((
                    (x0, y0, (x0 + crop_w).min(width), (y0 + crop_h).min(height)),
                    layer + 1,
                ));
            }
        }
    }
    boxes
}

fn uncrop_rle(rle: &MaskRle, x0: usize, y0: usize, width: usize, height: usize) -> MaskRle {
    let mut local = Vec::with_capacity(rle.width * rle.height);
    let mut value = false;
    for &count in &rle.counts {
        local.extend(std::iter::repeat_n(value, count));
        value = !value;
    }
    let mut full = vec![false; width * height];
    for y in 0..rle.height {
        let destination = (y0 + y) * width + x0;
        let source = y * rle.width;
        full[destination..destination + rle.width]
            .copy_from_slice(&local[source..source + rle.width]);
    }
    let mut counts = Vec::new();
    let mut current = false;
    let mut run = 0;
    for value in full {
        if value == current {
            run += 1;
        } else {
            counts.push(run);
            run = 1;
            current = value;
        }
    }
    counts.push(run);
    MaskRle {
        width,
        height,
        counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_mask_metadata_is_row_major_and_has_tight_box() {
        let mask = vec![vec![-1., 1., 1.], vec![-1., -1., 1.]];
        let (rle, bbox, area) = binary_mask_metadata(&mask, 3, 2, 0.).unwrap();
        assert_eq!(area, 3);
        assert_eq!(bbox, [1., 0., 2., 2.]);
        assert_eq!(rle.counts, vec![1, 2, 2, 1]);
    }

    #[test]
    fn nms_iou_uses_xywh_boxes() {
        assert!((box_iou([0., 0., 2., 2.], [1., 1., 2., 2.]) - 1. / 7.).abs() < 1e-6);
    }

    #[test]
    fn crop_boxes_match_the_square_pyramid_and_uncrop_preserves_coordinates() {
        let crops = crop_boxes(8, 6, 1, 0.);
        assert_eq!(crops.len(), 5);
        let local = MaskRle {
            width: 2,
            height: 2,
            counts: vec![1, 1, 2],
        };
        let full = uncrop_rle(&local, 3, 2, 8, 6);
        assert_eq!(full.width, 8);
        assert_eq!(full.height, 6);
        assert_eq!(full.counts.iter().sum::<usize>(), 48);
    }
}
