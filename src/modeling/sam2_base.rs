//! Temporal SAM2 base model for native single-object video/z-slice inference.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, conv2d, linear, ops::sigmoid};

use crate::{
    BackboneOutput, BoxPrompt, ImageEncoder, MaskDecoder, MaskDecoderConfig, MemoryAttention,
    MemoryEncoder, MemoryFrame, PointPrompt, PromptEncoder, Sam2Config, Sam2Variant,
    VideoInferenceState,
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
        Self::load_sam2_1_with_decoder_config(variant, path, MaskDecoderConfig::raw())
    }

    pub fn load_sam2_1_with_decoder_config(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        decoder_config: MaskDecoderConfig,
    ) -> Result<Self> {
        let device = Device::Cpu;
        Self::load_sam2_1_on_device_with_decoder_config(variant, path, &device, decoder_config)
    }

    pub fn load_sam2_1_on_device_with_decoder_config(
        variant: Sam2Variant,
        path: impl AsRef<Path>,
        device: &Device,
        decoder_config: MaskDecoderConfig,
    ) -> Result<Self> {
        if device.is_cpu() {
            crate::cpu_allocator::configure_cpu_allocator_for_inference();
        }
        let config = Sam2Config::sam2_1(variant);
        crate::checkpoint::validate_sam2_1(path.as_ref(), variant)?;
        let vb = crate::checkpoint::load_var_builder(path, DType::F32, device)?;
        Ok(Self {
            device: device.clone(),
            config: config.clone(),
            image_encoder: ImageEncoder::load(&config, vb.pp("image_encoder"))?,
            prompt_encoder: PromptEncoder::load(&config, vb.pp("sam_prompt_encoder"))?,
            mask_decoder: MaskDecoder::load_with_config(vb.pp("sam_mask_decoder"), decoder_config)?,
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

    pub fn device(&self) -> &Device {
        &self.device
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
        self.track_with_mask_prompt(
            image,
            frame_index,
            is_conditioning,
            state,
            points,
            bounding_box,
            None,
            false,
            multimask_output,
            reverse,
            num_frames,
        )
    }

    /// Tracking variant used for correction clicks, where the previous
    /// low-resolution mask logits are also supplied to the prompt encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn track_with_mask_prompt(
        &self,
        image: &Tensor,
        frame_index: usize,
        is_conditioning: bool,
        state: &mut VideoInferenceState,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        mask_input: Option<&Tensor>,
        binarize_mask_for_memory: bool,
        multimask_output: bool,
        reverse: bool,
        num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
        let backbone = self.image_encoder.forward(image)?;
        self.track_backbone_with_mask_prompt(
            &backbone,
            frame_index,
            is_conditioning,
            state,
            points,
            bounding_box,
            mask_input,
            binarize_mask_for_memory,
            multimask_output,
            reverse,
            num_frames,
        )
    }

    pub fn encode_frame(&self, image: &Tensor) -> Result<BackboneOutput> {
        self.image_encoder.forward(image)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn track_backbone_with_mask_prompt(
        &self,
        backbone: &BackboneOutput,
        frame_index: usize,
        is_conditioning: bool,
        state: &mut VideoInferenceState,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
        mask_input: Option<&Tensor>,
        binarize_mask_for_memory: bool,
        multimask_output: bool,
        reverse: bool,
        num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
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
            mask_input,
        )?;
        let decoded = self.mask_decoder.forward(
            &conditioned,
            &self.prompt_encoder.get_dense_pe()?,
            &prompts.sparse_embeddings,
            &prompts.dense_embeddings,
            multimask_output,
            &backbone.backbone_fpn[..2],
        )?;
        let (decoded_masks, decoded_token) = if multimask_output {
            let best = decoded.iou_predictions.argmax_keepdim(1)?;
            let (_, _, mask_height, mask_width) = decoded.masks.dims4()?;
            let mask_indices = best
                .reshape((batch, 1, 1, 1))?
                .broadcast_as((batch, 1, mask_height, mask_width))?
                .contiguous()?;
            let token_indices = best
                .reshape((batch, 1, 1))?
                .broadcast_as((batch, 1, 256))?
                .contiguous()?;
            (
                decoded.masks.contiguous()?.gather(&mask_indices, 1)?,
                decoded
                    .mask_tokens
                    .contiguous()?
                    .gather(&token_indices, 1)?,
            )
        } else {
            (decoded.masks.clone(), decoded.mask_tokens.clone())
        };
        let is_obj_appearing = decoded.object_score_logits.gt(0f32)?;
        let low_res_masks = is_obj_appearing
            .reshape((batch, 1, 1, 1))?
            .broadcast_as(decoded_masks.dims())?
            .where_cond(
                &decoded_masks,
                &Tensor::full(-1024f32, decoded_masks.dims(), &self.device)?,
            )?;
        let high_res_masks = low_res_masks.upsample_bilinear2d(
            self.config.image_size,
            self.config.image_size,
            false,
        )?;
        // The memory encoder receives a scaled sigmoid probability, exactly as
        // specified by the SAM2.1 inference configs.
        let memory_mask = if binarize_mask_for_memory {
            high_res_masks.gt(0f64)?.to_dtype(DType::F32)?.affine(
                self.config.sigmoid_scale_for_mem_enc,
                self.config.sigmoid_bias_for_mem_enc,
            )?
        } else {
            sigmoid(&high_res_masks)?.affine(
                self.config.sigmoid_scale_for_mem_enc,
                self.config.sigmoid_bias_for_mem_enc,
            )?
        };
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
        let projected_pointer = self.obj_ptr_proj.forward(&decoded_token.squeeze(1)?)?;
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
        self.track_mask_from_backbone(
            &backbone,
            mask,
            frame_index,
            is_conditioning,
            state,
            reverse,
            num_frames,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn track_mask_from_backbone(
        &self,
        backbone: &BackboneOutput,
        mask: &Tensor,
        frame_index: usize,
        is_conditioning: bool,
        state: &mut VideoInferenceState,
        _reverse: bool,
        _num_frames: Option<usize>,
    ) -> Result<Sam2FrameOutput> {
        let current = &backbone.vision_features;
        let batch = current.dim(0)?;
        // `use_mask_input_as_output_without_sam` bypasses temporal memory and
        // the initial-frame no-memory embedding. The raw backbone feature is
        // used only to derive the object pointer and spatial memory.
        let conditioned = current.clone();
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
        let low_res_masks = self.antialiased_bilinear_resize(&high_res_masks, 256, 256)?;
        let mask_prompt = self.mask_downsample.forward(&mask.to_dtype(DType::F32)?)?;
        let (point_coords, point_labels, _) = self.prompt_tensors(&[], None)?;
        let prompts = self.prompt_encoder.forward(
            point_coords.as_ref().zip(point_labels.as_ref()),
            None,
            Some(&mask_prompt),
        )?;
        let decoded = self.mask_decoder.forward(
            &conditioned,
            &self.prompt_encoder.get_dense_pe()?,
            &prompts.sparse_embeddings,
            &prompts.dense_embeddings,
            false,
            &backbone.backbone_fpn[..2],
        )?;
        let appears = mask.max_all()?.to_scalar::<f32>()? > 0.;
        let projected_pointer = self
            .obj_ptr_proj
            .forward(&decoded.mask_tokens.squeeze(1)?)?;
        let decoder_appearance = decoded
            .object_score_logits
            .gt(0f64)?
            .to_dtype(DType::F32)?
            .reshape((batch, 1))?;
        let decoder_pointer = projected_pointer
            .broadcast_mul(&decoder_appearance)?
            .broadcast_add(
                &decoder_appearance
                    .affine(-1., 1.)?
                    .broadcast_mul(&self.no_obj_ptr.broadcast_as((batch, 256))?)?,
            )?;
        let object_pointer = if appears {
            decoder_pointer
        } else {
            self.no_obj_ptr.broadcast_as((batch, 256))?
        };
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

    /// Rebuilds temporal memory from finalized low-resolution mask logits.
    /// Interactive video uses this during preflight after hole filling and
    /// optional multi-object constraints have been applied.
    pub fn reencode_memory(
        &self,
        image: &Tensor,
        low_res_masks: &Tensor,
        object_score_logits: &Tensor,
        object_pointer: &Tensor,
        binarize_mask_from_points: bool,
    ) -> Result<MemoryFrame> {
        let backbone = self.image_encoder.forward(image)?;
        self.reencode_memory_from_backbone(
            &backbone,
            low_res_masks,
            object_score_logits,
            object_pointer,
            binarize_mask_from_points,
        )
    }

    pub fn reencode_memory_from_backbone(
        &self,
        backbone: &BackboneOutput,
        low_res_masks: &Tensor,
        object_score_logits: &Tensor,
        object_pointer: &Tensor,
        binarize_mask_from_points: bool,
    ) -> Result<MemoryFrame> {
        let high_res_masks = low_res_masks.to_dtype(DType::F32)?.upsample_bilinear2d(
            self.config.image_size,
            self.config.image_size,
            false,
        )?;
        self.reencode_memory_from_high_res(
            backbone,
            &high_res_masks,
            low_res_masks,
            object_score_logits,
            object_pointer,
            binarize_mask_from_points,
        )
    }

    pub fn reencode_memory_from_high_res(
        &self,
        backbone: &BackboneOutput,
        high_res_masks: &Tensor,
        low_res_masks: &Tensor,
        object_score_logits: &Tensor,
        object_pointer: &Tensor,
        binarize_mask_from_points: bool,
    ) -> Result<MemoryFrame> {
        let memory_mask = if binarize_mask_from_points {
            high_res_masks.gt(0f64)?.to_dtype(DType::F32)?.affine(
                self.config.sigmoid_scale_for_mem_enc,
                self.config.sigmoid_bias_for_mem_enc,
            )?
        } else {
            sigmoid(&high_res_masks)?.affine(
                self.config.sigmoid_scale_for_mem_enc,
                self.config.sigmoid_bias_for_mem_enc,
            )?
        };
        let mut encoded =
            self.memory_encoder
                .forward(&backbone.vision_features, &memory_mask, true)?;
        let batch = low_res_masks.dim(0)?;
        let appearance = object_score_logits
            .gt(0f64)?
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
        Ok(MemoryFrame {
            memory_features: encoded.vision_features,
            memory_pos_enc: encoded.vision_pos_enc,
            object_pointer: object_pointer.clone(),
            low_res_mask_logits: low_res_masks.clone(),
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
        for (_, frame) in &selected {
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
        // SAM2.1's eval configuration restricts object pointers (but not
        // spatial mask memories) to frames already visited in the tracking
        // direction. Conditioning pointers come first.
        let mut pointer_frames = selected
            .into_iter()
            .filter(|(index, _)| {
                if reverse {
                    *index >= frame_index
                } else {
                    *index <= frame_index
                }
            })
            .collect::<Vec<_>>();
        let max_obj_ptrs = num_frames.unwrap_or(16).min(16);
        for difference in 1..max_obj_ptrs {
            let index = if reverse {
                frame_index.checked_add(difference)
            } else {
                frame_index.checked_sub(difference)
            };
            let Some(index) = index else { break };
            if num_frames.is_some_and(|frames| index >= frames) {
                break;
            }
            if let Some(frame) = state.propagated().get(&index) {
                pointer_frames.push((index, frame));
            }
        }
        for (index, frame) in pointer_frames {
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

    pub fn antialiased_bilinear_resize(
        &self,
        input: &Tensor,
        output_height: usize,
        output_width: usize,
    ) -> Result<Tensor> {
        let (batch, channels, input_height, input_width) = input.dims4()?;
        fn weights(input: usize, output: usize, device: &Device) -> Result<Tensor> {
            let scale = output as f32 / input as f32;
            let filter_scale = scale.min(1.);
            let support = 1. / filter_scale;
            let mut values = vec![0f32; output * input];
            for out in 0..output {
                let center = (out as f32 + 0.5) / scale - 0.5;
                let start = (center - support).ceil() as isize;
                let end = (center + support).floor() as isize;
                let mut sum = 0f32;
                for index in start..=end {
                    if index >= 0 && (index as usize) < input {
                        let value = (1. - ((index as f32 - center) * filter_scale).abs()).max(0.);
                        values[out * input + index as usize] = value;
                        sum += value;
                    }
                }
                for index in 0..input {
                    values[out * input + index] /= sum;
                }
            }
            Tensor::from_vec(values, (output, input), device)
        }
        let height_weights = weights(input_height, output_height, input.device())?;
        let width_weights = weights(input_width, output_width, input.device())?;
        let height = input
            .permute((0, 1, 3, 2))?
            .reshape((batch * channels * input_width, input_height))?
            .matmul(&height_weights.t()?)?
            .reshape((batch, channels, input_width, output_height))?
            .permute((0, 1, 3, 2))?;
        height
            .reshape((batch * channels * output_height, input_width))?
            .matmul(&width_weights.t()?)?
            .reshape((batch, channels, output_height, output_width))
    }

    fn prompt_tensors(
        &self,
        points: &[PointPrompt],
        bounding_box: Option<BoxPrompt>,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        // The video model does not pass `None` when a frame has no clicks.
        // It supplies one label -1 point, after which PromptEncoder adds its
        // usual padding point. Those two sparse no-point tokens are observable
        // by the mask decoder and cannot be replaced by an empty sequence.
        let use_no_point = points.is_empty() && bounding_box.is_none();
        let point_coords = (!points.is_empty() || use_no_point)
            .then(|| {
                if use_no_point {
                    return Tensor::zeros((1, 1, 2), DType::F32, &self.device);
                }
                Tensor::from_vec(
                    points.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                    (1, points.len(), 2),
                    &self.device,
                )
            })
            .transpose()?;
        let point_labels = (!points.is_empty() || use_no_point)
            .then(|| {
                if use_no_point {
                    return Tensor::full(-1i64, (1, 1), &self.device);
                }
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
