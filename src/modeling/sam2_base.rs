//! Temporal SAM2 base model for native single-object video/z-slice inference.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d, linear, ops::sigmoid};

use crate::{
    BoxPrompt, ImageEncoder, MaskDecoder, MemoryAttention, MemoryEncoder, MemoryFrame, PointPrompt,
    PromptEncoder, Sam2Config, Sam2Variant, VideoInferenceState,
    ops::{Activation, Mlp},
};

/// The decoded result and the memory written for a video frame.
#[derive(Clone, Debug)]
pub struct Sam2FrameOutput {
    /// The 64x64 visual feature after temporal memory conditioning, useful for
    /// inspecting temporal behavior and downstream custom heads.
    pub conditioned_features: Tensor,
    pub low_res_masks: Tensor,
    pub high_res_masks: Tensor,
    pub iou_predictions: Tensor,
    pub object_score_logits: Tensor,
    pub object_pointer: Tensor,
    pub memory_features: Tensor,
    pub memory_pos_enc: Tensor,
}

/// Checkpoint-backed composition of the SAM2 image, temporal-attention, and
/// memory-encoder components. Frame indices are deliberately generic, which
/// permits either video time indices or microscopy z-slice indices.
#[derive(Clone, Debug)]
pub struct Sam2Base {
    device: Device,
    config: Sam2Config,
    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
    memory_attention: MemoryAttention,
    memory_encoder: MemoryEncoder,
    maskmem_tpos_enc: Tensor,
    no_mem_embed: Tensor,
    no_obj_ptr: Tensor,
    no_obj_embed_spatial: Tensor,
    mask_downsample: Conv2d,
    obj_ptr_proj: Mlp,
    obj_ptr_tpos_proj: candle_nn::Linear,
}

impl Sam2Base {
    pub fn load_sam2_1(variant: Sam2Variant, path: impl AsRef<Path>) -> Result<Self> {
        let device = Device::Cpu;
        let config = Sam2Config::sam2_1(variant);
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], DType::F32, &device) }?;
        Ok(Self {
            device: device.clone(),
            config: config.clone(),
            image_encoder: ImageEncoder::load(&config, vb.pp("image_encoder"))?,
            prompt_encoder: PromptEncoder::load(&config, vb.pp("sam_prompt_encoder"))?,
            mask_decoder: MaskDecoder::load(vb.pp("sam_mask_decoder"))?,
            memory_attention: MemoryAttention::load(vb.pp("memory_attention"))?,
            memory_encoder: MemoryEncoder::load(vb.pp("memory_encoder"))?,
            maskmem_tpos_enc: vb.get((config.num_maskmem, 1, 1, 64), "maskmem_tpos_enc")?,
            no_mem_embed: vb.get((1, 1, 256), "no_mem_embed")?,
            no_obj_ptr: vb.get((1, 256), "no_obj_ptr")?,
            no_obj_embed_spatial: vb.get((1, 64), "no_obj_embed_spatial")?,
            mask_downsample: conv2d(
                1,
                1,
                4,
                Conv2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("mask_downsample"),
            )?,
            obj_ptr_proj: Mlp::load(
                256,
                256,
                256,
                3,
                Activation::Relu,
                false,
                vb.pp("obj_ptr_proj"),
            )?,
            obj_ptr_tpos_proj: linear(256, 64, vb.pp("obj_ptr_tpos_proj"))?,
        })
    }

    pub fn load_sam2_1_hiera_tiny(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_sam2_1(Sam2Variant::HieraTiny, path)
    }

    /// Runs one frame and writes its encoded memory into `state`. Use
    /// `is_conditioning=true` for an interacted/reference frame; later frames
    /// are propagated in the selected direction. Images are normalized NCHW
    /// SAM2 inputs (currently 1x3x1024x1024).
    pub fn track(
        &self,
        image: &Tensor,
        frame_index: usize,
        is_conditioning: bool,
        state: &mut VideoInferenceState,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
        reverse: bool,
    ) -> Result<Sam2FrameOutput> {
        self.track_with_num_frames(
            image,
            frame_index,
            is_conditioning,
            state,
            points,
            bounding_box,
            multimask_output,
            reverse,
            None,
        )
    }

    /// As [`Self::track`], while optionally supplying the full sequence length.
    /// This makes object-pointer temporal positional encoding bit-for-bit match
    /// SAM2's offline-video policy instead of its streaming estimate.
    #[allow(clippy::too_many_arguments)]
    pub fn track_with_num_frames(
        &self,
        image: &Tensor,
        frame_index: usize,
        is_conditioning: bool,
        state: &mut VideoInferenceState,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        multimask_output: bool,
        reverse: bool,
        num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
        let backbone = self.image_encoder.forward(image)?;
        let current = &backbone.vision_features;
        let current_pos = backbone.vision_pos_enc.last().unwrap();
        let (batch, channels, height, width) = current.dims4()?;
        let conditioned = if is_conditioning {
            // SAM2.1's published configs use directly_add_no_mem_embed on
            // initial conditioning frames rather than a dummy transformer key.
            current.broadcast_add(
                &self
                    .no_mem_embed
                    .permute((1, 2, 0))?
                    .reshape((1, 256, 1, 1))?,
            )?
        } else {
            self.condition_with_memory(
                current,
                current_pos,
                frame_index,
                state,
                reverse,
                batch,
                channels,
                height,
                width,
                num_frames,
            )?
        };
        let (point_coords, point_labels, box_tensor) = self.prompt_tensors(points, bounding_box)?;
        let prompts = self.prompt_encoder.forward(
            point_coords.as_ref().zip(point_labels.as_ref()),
            box_tensor.as_ref(),
            None,
        )?;
        let decoded = self.mask_decoder.forward(
            &conditioned,
            &self.prompt_encoder.get_dense_pe()?,
            &prompts.sparse_embeddings,
            &prompts.dense_embeddings,
            multimask_output,
            &backbone.backbone_fpn[..2],
        )?;
        let is_obj_appearing = decoded.object_score_logits.gt(0f32)?;
        let low_res_masks = is_obj_appearing
            .reshape((batch, 1, 1, 1))?
            .broadcast_as(decoded.masks.dims())?
            .where_cond(
                &decoded.masks,
                &Tensor::full(-1024f32, decoded.masks.dims(), &self.device)?,
            )?;
        let high_res_masks = low_res_masks.upsample_bilinear2d(
            self.config.image_size,
            self.config.image_size,
            false,
        )?;
        // The memory encoder receives a scaled sigmoid probability, exactly as
        // specified by the SAM2.1 inference configs.
        let memory_mask = sigmoid(&high_res_masks)?.affine(
            self.config.sigmoid_scale_for_mem_enc,
            self.config.sigmoid_bias_for_mem_enc,
        )?;
        let mut encoded = self
            .memory_encoder
            // SAM2 writes memories from the raw backbone feature, rather than
            // the feature already fused with prior memory attention.
            .forward(current, &memory_mask, true)?;
        let appearance = is_obj_appearing
            .to_dtype(DType::F32)?
            .reshape((batch, 1, 1, 1))?;
        encoded.vision_features = encoded.vision_features.broadcast_add(
            &appearance.affine(-1., 1.)?.broadcast_mul(
                &self
                    .no_obj_embed_spatial
                    .reshape((1, 64, 1, 1))?
                    .broadcast_as(encoded.vision_features.dims())?,
            )?,
        )?;
        let projected_pointer = self
            .obj_ptr_proj
            .forward(&decoded.mask_tokens.squeeze(1)?)?;
        let object_pointer = projected_pointer
            .broadcast_mul(&appearance.reshape((batch, 1))?)?
            .broadcast_add(
                &appearance
                    .affine(-1., 1.)?
                    .reshape((batch, 1))?
                    .broadcast_mul(&self.no_obj_ptr.broadcast_as((batch, 256))?)?,
            )?;
        let frame = MemoryFrame {
            memory_features: encoded.vision_features.clone(),
            memory_pos_enc: encoded.vision_pos_enc.clone(),
            object_pointer: object_pointer.clone(),
            low_res_mask_logits: low_res_masks.clone(),
        };
        if is_conditioning {
            state.insert_conditioning(frame_index, frame);
        } else {
            state.insert_propagated(frame_index, frame);
        }
        Ok(Sam2FrameOutput {
            conditioned_features: conditioned,
            low_res_masks,
            high_res_masks,
            iou_predictions: decoded.iou_predictions,
            object_score_logits: decoded.object_score_logits,
            object_pointer,
            memory_features: encoded.vision_features,
            memory_pos_enc: encoded.vision_pos_enc,
        })
    }

    /// Conditions or propagates a frame from a binary high-resolution mask.
    /// This follows SAM2.1's `use_mask_input_as_output_without_sam` route: the
    /// supplied mask is the output while SAM heads are used only to derive the
    /// object pointer required by subsequent temporal attention.
    #[allow(clippy::too_many_arguments)]
    pub fn track_mask(
        &self,
        image: &Tensor,
        mask: &Tensor,
        frame_index: usize,
        is_conditioning: bool,
        state: &mut VideoInferenceState,
        reverse: bool,
        num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
        let backbone = self.image_encoder.forward(image)?;
        let current = &backbone.vision_features;
        let current_pos = backbone.vision_pos_enc.last().unwrap();
        let (batch, channels, height, width) = current.dims4()?;
        let conditioned = if is_conditioning {
            current.broadcast_add(
                &self
                    .no_mem_embed
                    .permute((1, 2, 0))?
                    .reshape((1, 256, 1, 1))?,
            )?
        } else {
            self.condition_with_memory(
                current,
                current_pos,
                frame_index,
                state,
                reverse,
                batch,
                channels,
                height,
                width,
                num_frames,
            )?
        };
        let (mask_batch, mask_channels, mask_h, mask_w) = mask.dims4()?;
        if mask_batch != batch
            || mask_channels != 1
            || mask_h != self.config.image_size
            || mask_w != self.config.image_size
        {
            candle_core::bail!(
                "mask input must be B×1×{}×{}",
                self.config.image_size,
                self.config.image_size
            )
        }
        let high_res_masks = mask.to_dtype(DType::F32)?.affine(20., -10.)?;
        let low_res_masks = high_res_masks.upsample_bilinear2d(256, 256, false)?;
        let mask_prompt = self.mask_downsample.forward(&mask.to_dtype(DType::F32)?)?;
        let prompts = self
            .prompt_encoder
            .forward(None, None, Some(&mask_prompt))?;
        let decoded = self.mask_decoder.forward(
            &conditioned,
            &self.prompt_encoder.get_dense_pe()?,
            &prompts.sparse_embeddings,
            &prompts.dense_embeddings,
            false,
            &backbone.backbone_fpn[..2],
        )?;
        let appears = mask.max_all()?.to_scalar::<f32>()? > 0.;
        let mut object_pointer = self
            .obj_ptr_proj
            .forward(&decoded.mask_tokens.squeeze(1)?)?;
        if !appears {
            object_pointer = self.no_obj_ptr.broadcast_as((batch, 256))?;
        }
        let memory_mask = sigmoid(&high_res_masks)?.affine(
            self.config.sigmoid_scale_for_mem_enc,
            self.config.sigmoid_bias_for_mem_enc,
        )?;
        let mut encoded = self.memory_encoder.forward(current, &memory_mask, true)?;
        if !appears {
            encoded.vision_features = encoded.vision_features.broadcast_add(
                &self
                    .no_obj_embed_spatial
                    .reshape((1, 64, 1, 1))?
                    .broadcast_as(encoded.vision_features.dims())?,
            )?;
        }
        let frame = MemoryFrame {
            memory_features: encoded.vision_features.clone(),
            memory_pos_enc: encoded.vision_pos_enc.clone(),
            object_pointer: object_pointer.clone(),
            low_res_mask_logits: low_res_masks.clone(),
        };
        if is_conditioning {
            state.insert_conditioning(frame_index, frame);
        } else {
            state.insert_propagated(frame_index, frame);
        }
        Ok(Sam2FrameOutput {
            conditioned_features: conditioned,
            low_res_masks,
            high_res_masks,
            iou_predictions: Tensor::ones((batch, 1), DType::F32, &self.device)?,
            object_score_logits: Tensor::full(
                if appears { 10f32 } else { -10f32 },
                (batch, 1),
                &self.device,
            )?,
            object_pointer,
            memory_features: encoded.vision_features,
            memory_pos_enc: encoded.vision_pos_enc,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn condition_with_memory(
        &self,
        current: &Tensor,
        current_pos: &Tensor,
        frame_index: usize,
        state: &VideoInferenceState,
        reverse: bool,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        num_frames: Option<usize>,
    ) -> Result<Tensor> {
        let selected = state.closest_conditioning(frame_index, usize::MAX);
        let mut memories = Vec::new();
        let mut positions = Vec::new();
        let conditioning_tpos = self
            .maskmem_tpos_enc
            .narrow(0, self.config.num_maskmem - 1, 1)?
            .squeeze(0)?;
        for (_, frame) in selected {
            Self::push_spatial_memory(
                &mut memories,
                &mut positions,
                frame,
                Some(&conditioning_tpos),
            )?;
        }
        // The most recent frame always participates; older slots follow the
        // published `num_maskmem` temporal ordering.
        for t_pos in 1..self.config.num_maskmem {
            let relative = self.config.num_maskmem - t_pos;
            let index = if reverse {
                frame_index.checked_add(relative)
            } else {
                frame_index.checked_sub(relative)
            };
            if let Some(index) = index.and_then(|i| state.propagated().get(&i)) {
                let temporal = self
                    .maskmem_tpos_enc
                    .narrow(0, self.config.num_maskmem - t_pos - 1, 1)?
                    .squeeze(0)?;
                Self::push_spatial_memory(&mut memories, &mut positions, index, Some(&temporal))?;
            }
        }
        if memories.is_empty() {
            // Published SAM2 tracking requires a conditioning frame before
            // propagation. Its no-memory tokens are 256-D and are only valid
            // in the initial-frame path, which uses `directly_add_no_mem_embed`.
            candle_core::bail!(
                "cannot propagate frame {frame_index} before adding a conditioning frame"
            )
        }
        let mut obj_ptrs = Vec::new();
        let mut obj_positions = Vec::new();
        // In the reference this denominator is `min(num_frames,
        // max_obj_ptrs_in_encoder) - 1`. Without a predeclared clip length,
        // the populated state plus this frame is the exact value for streamed
        // inference and avoids assigning a needlessly tiny temporal phase.
        let pointer_tpos_max = num_frames
            .unwrap_or(state.conditioning().len() + state.propagated().len() + 1)
            .min(16)
            .saturating_sub(1)
            .max(1) as f32;
        for (index, frame) in state
            .temporal_candidates(frame_index, usize::MAX, 15, reverse)
            .into_iter()
        {
            let delta = if reverse {
                index as f32 - frame_index as f32
            } else {
                frame_index as f32 - index as f32
            };
            obj_ptrs.push(frame.object_pointer.unsqueeze(0)?);
            obj_positions.push(self.object_pointer_position(delta, pointer_tpos_max, batch)?);
        }
        let num_obj_ptr_tokens = obj_ptrs.len() * 4;
        for (pointer, position) in obj_ptrs.iter().zip(obj_positions.iter()) {
            // Memory uses 64 channels, so each 256-D pointer becomes four
            // suffix tokens. RoPE explicitly excludes this suffix.
            memories.push(
                pointer
                    .reshape((1, batch, 4, 64))?
                    .permute((0, 2, 1, 3))?
                    .flatten(0, 1)?,
            );
            positions.push(position.repeat((4, 1, 1))?);
        }
        let memory_refs: Vec<_> = memories.iter().collect();
        let position_refs: Vec<_> = positions.iter().collect();
        let memory = Tensor::cat(&memory_refs, 0)?;
        let memory_pos = Tensor::cat(&position_refs, 0)?;
        let curr = current
            .reshape((batch, channels, height * width))?
            .permute((2, 0, 1))?;
        let curr_pos = current_pos
            .reshape((batch, channels, height * width))?
            .permute((2, 0, 1))?;
        self.memory_attention
            .forward(&curr, &memory, &curr_pos, &memory_pos, num_obj_ptr_tokens)?
            .permute((1, 2, 0))?
            .reshape((batch, channels, height, width))
    }

    fn push_spatial_memory(
        memories: &mut Vec<Tensor>,
        positions: &mut Vec<Tensor>,
        frame: &MemoryFrame,
        temporal: Option<&Tensor>,
    ) -> Result<()> {
        let (batch, channels, height, width) = frame.memory_features.dims4()?;
        memories.push(
            frame
                .memory_features
                .reshape((batch, channels, height * width))?
                .permute((2, 0, 1))?,
        );
        let pos = frame
            .memory_pos_enc
            .reshape((batch, channels, height * width))?
            .permute((2, 0, 1))?;
        positions.push(if let Some(temporal) = temporal {
            pos.broadcast_add(temporal)?
        } else {
            pos
        });
        Ok(())
    }

    fn object_pointer_position(&self, delta: f32, max_delta: f32, batch: usize) -> Result<Tensor> {
        let frequencies = Tensor::from_vec(
            (0..128)
                .map(|i| 10_000f32.powf((2 * (i / 2)) as f32 / 128.))
                .collect::<Vec<_>>(),
            128,
            &self.device,
        )?;
        let values = Tensor::from_vec(vec![delta / max_delta; 128], (1, 128), &self.device)?
            .broadcast_div(&frequencies)?;
        let enc = Tensor::cat(&[&values.sin()?, &values.cos()?], 1)?;
        self.obj_ptr_tpos_proj
            .forward(&enc)?
            .unsqueeze(1)?
            .broadcast_as((1, batch, 64))
    }

    fn prompt_tensors(
        &self,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let point_coords = (!points.is_empty())
            .then(|| {
                Tensor::from_vec(
                    points.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                    (1, points.len(), 2),
                    &self.device,
                )
            })
            .transpose()?;
        let point_labels = (!points.is_empty())
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
        let boxes = bounding_box
            .map(|b| Tensor::from_vec(vec![b.x0, b.y0, b.x1, b.y1], (1, 4), &self.device))
            .transpose()?;
        Ok((point_coords, point_labels, boxes))
    }
}
