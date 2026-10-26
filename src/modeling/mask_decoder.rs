//! Translation of `sam2/modeling/sam/mask_decoder.py` for static inference.

use std::time::Instant;

use candle_core::{DType, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder,
    conv_transpose2d, conv2d,
};

use crate::{
    TwoWayTransformer, TwoWayTransformerProfile,
    ops::{Activation, LayerNorm2d, Mlp},
};

static PRINTED_DECODER_PROFILE_LAYOUT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn sam2_conv_transpose2x2_nchw_f32(
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        residual: *const f32,
        output: *mut f32,
        output_elements: u64,
        input_channels: u32,
        output_channels: u32,
        input_height: u32,
        input_width: u32,
        residual_batch_stride: u64,
        gelu: bool,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
struct CudaConvTranspose2x2Residual {
    bias: Tensor,
    gelu: bool,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp3 for CudaConvTranspose2x2Residual {
    fn name(&self) -> &'static str {
        "sam2-conv-transpose2x2-residual-f32"
    }

    fn cpu_fwd(
        &self,
        _input: &candle_core::CpuStorage,
        _input_layout: &candle_core::Layout,
        _weight: &candle_core::CpuStorage,
        _weight_layout: &candle_core::Layout,
        _residual: &candle_core::CpuStorage,
        _residual_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("2x2 transposed convolution is only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &candle_core::Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &candle_core::Layout,
        residual: &candle_core::CudaStorage,
        residual_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        if input.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || self.bias.dtype() != DType::F32
        {
            candle_core::bail!("2x2 transposed convolution requires f32")
        }
        let (batch, input_channels, input_height, input_width) = input_layout.shape().dims4()?;
        let (weight_channels, output_channels, kernel_h, kernel_w) =
            weight_layout.shape().dims4()?;
        let output_shape = candle_core::Shape::from_dims(&[
            batch,
            output_channels,
            input_height * 2,
            input_width * 2,
        ]);
        let (residual_batch, residual_channels, residual_height, residual_width) =
            residual_layout.shape().dims4()?;
        if input_channels != weight_channels
            || kernel_h != 2
            || kernel_w != 2
            || !matches!(residual_batch, 1) && residual_batch != batch
            || residual_channels != output_channels
            || residual_height != input_height * 2
            || residual_width != input_width * 2
            || self.bias.dims1()? != output_channels
        {
            candle_core::bail!("invalid 2x2 transposed-convolution shape")
        }
        let Some((input_start, _)) = input_layout.contiguous_offsets() else {
            candle_core::bail!("2x2 transposed-convolution input must be contiguous")
        };
        let Some((weight_start, _)) = weight_layout.contiguous_offsets() else {
            candle_core::bail!("2x2 transposed-convolution weight must be contiguous")
        };
        let Some((residual_start, _)) = residual_layout.contiguous_offsets() else {
            candle_core::bail!("2x2 transposed-convolution residual must be contiguous")
        };
        let (bias_storage, bias_layout) = self.bias.storage_and_layout();
        let candle_core::Storage::Cuda(bias_storage) = &*bias_storage else {
            candle_core::bail!("2x2 transposed-convolution bias must be on cuda")
        };
        let Some((bias_start, _)) = bias_layout.contiguous_offsets() else {
            candle_core::bail!("2x2 transposed-convolution bias must be contiguous")
        };
        let output_elements = output_shape.elem_count();
        let dev = input.device();
        let stream = dev.cuda_stream();
        let input = input.as_cuda_slice::<f32>()?.slice(input_start..);
        let weight = weight.as_cuda_slice::<f32>()?.slice(weight_start..);
        let residual = residual.as_cuda_slice::<f32>()?.slice(residual_start..);
        let bias = bias_storage.as_cuda_slice::<f32>()?.slice(bias_start..);
        // SAFETY: the external kernel writes every element of the validated output shape.
        let mut output = unsafe { dev.alloc::<f32>(output_elements)? };
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (weight_ptr, weight_guard) = weight.device_ptr(&stream);
        let (bias_ptr, bias_guard) = bias.device_ptr(&stream);
        let (residual_ptr, residual_guard) = residual.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_conv_transpose2x2_nchw_f32(
                input_ptr as *const f32,
                weight_ptr as *const f32,
                bias_ptr as *const f32,
                residual_ptr as *const f32,
                output_ptr as *mut f32,
                output_elements as u64,
                input_channels as u32,
                output_channels as u32,
                input_height as u32,
                input_width as u32,
                if residual_batch == 1 {
                    0
                } else {
                    (output_channels * input_height * 2 * input_width * 2) as u64
                },
                self.gelu,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("2x2 transposed-convolution CUDA launch failed with status {status}")
        }
        drop((
            input_guard,
            weight_guard,
            bias_guard,
            residual_guard,
            output_guard,
        ));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            output_shape,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct MaskDecoderOutput {
    pub masks: Tensor,
    pub iou_predictions: Tensor,
    pub mask_tokens: Tensor,
    pub object_score_logits: Tensor,
    pub upscaled_embedding: Tensor,
    pub hyper_in: Tensor,
}

/// Inference-time mask selection options normally supplied by the Python
/// builder's `apply_postprocessing` overrides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskDecoderConfig {
    pub dynamic_multimask_via_stability: bool,
    pub dynamic_multimask_stability_delta: f64,
    pub dynamic_multimask_stability_threshold: f64,
}

impl MaskDecoderConfig {
    /// Deterministic token-zero selection used by low-level parity and
    /// component probes that disable Python builder postprocessing.
    pub const fn raw() -> Self {
        Self {
            dynamic_multimask_via_stability: false,
            dynamic_multimask_stability_delta: 0.05,
            dynamic_multimask_stability_threshold: 0.98,
        }
    }
}

impl Default for MaskDecoderConfig {
    /// Matches `build_sam2(..., apply_postprocessing=True)`.
    fn default() -> Self {
        Self {
            dynamic_multimask_via_stability: true,
            ..Self::raw()
        }
    }
}

/// Stable component boundaries for locating CPU decoder regressions.  This is
/// intentionally opt-in: normal inference does not read a clock.
#[derive(Clone, Debug, Default)]
pub struct MaskDecoderProfile {
    pub prompt_encoder_ms: f64,
    pub dense_pe_ms: f64,
    pub token_setup_ms: f64,
    pub transformer_ms: f64,
    pub transformer: TwoWayTransformerProfile,
    pub upconv_0_ms: f64,
    pub skip_1_ms: f64,
    pub upconv_core_0_ms: f64,
    pub upnorm_gelu_ms: f64,
    pub upconv_3_ms: f64,
    pub skip_0_ms: f64,
    pub upconv_core_3_ms: f64,
    pub final_gelu_ms: f64,
    pub hypernetworks_ms: f64,
    pub mask_projection_ms: f64,
    pub heads_ms: f64,
}

#[derive(Clone, Debug)]
pub struct MaskDecoder {
    transformer: TwoWayTransformer,
    iou_token: Tensor,
    mask_tokens: Tensor,
    obj_score_token: Tensor,
    upconv_0: ConvTranspose2d,
    upnorm_1: LayerNorm2d,
    upconv_3: ConvTranspose2d,
    conv_s0: Conv2d,
    conv_s1: Conv2d,
    hypernetworks: [Mlp; 4],
    iou_head: Mlp,
    object_score_head: Mlp,
    dynamic_multimask_via_stability: bool,
    dynamic_multimask_stability_delta: f64,
    dynamic_multimask_stability_thresh: f64,
}

impl MaskDecoder {
    fn upconv_residual(
        conv: &ConvTranspose2d,
        input: &Tensor,
        residual: &Tensor,
        gelu: bool,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if input.device().is_cuda()
            && input.dtype() == DType::F32
            && input.is_contiguous()
            && residual.is_contiguous()
            // The final 64 -> 32 upsampling stage is faster as one direct
            // NCHW kernel. cuDNN remains faster for the 256 -> 64 stage.
            && gelu
            && std::env::var("SAM2_CUDA_FINAL_TRANSPOSE2X2").as_deref() != Ok("0")
            && let Some(bias) = conv.bias()
        {
            return input.apply_op3_no_bwd(
                conv.weight(),
                residual,
                &CudaConvTranspose2x2Residual {
                    bias: bias.clone(),
                    gelu,
                },
            );
        }
        let output = conv.forward_add_residual_cpu(input, residual)?;
        if gelu { output.gelu_erf() } else { Ok(output) }
    }

    fn position_for_batch(image_pe: &Tensor, batch: usize) -> Result<Tensor> {
        let (_, _, height, width) = image_pe.dims4()?;
        // Every prompt in an automatic-mask batch belongs to the same encoded
        // image, so its dense positional encoding is identical. Keep the
        // batch-one CPU tensor and let the transformer's additions broadcast
        // it. Expanding first caused the later NCHW->token preparation to
        // materialize another ~1 GiB tensor at batch 64 on a 64x64 grid.
        if image_pe.device().is_cpu()
            && image_pe.dim(0)? == 1
            && std::env::var_os("SAM2_CPU_DECODER_BROADCAST_IMAGE_PE")
                .is_none_or(|value| value != "0")
        {
            return Ok(image_pe.clone());
        }
        if image_pe.dims() == [batch, 256usize, height, width] {
            Ok(image_pe.clone())
        } else {
            image_pe.broadcast_as((batch, 256usize, height, width))
        }
    }

    pub fn load(vb: VarBuilder) -> Result<Self> {
        Self::load_with_config(vb, MaskDecoderConfig::raw())
    }

    pub fn load_with_config(vb: VarBuilder, config: MaskDecoderConfig) -> Result<Self> {
        Ok(Self {
            transformer: TwoWayTransformer::load(vb.pp("transformer"))?,
            iou_token: vb.get((1, 256), "iou_token.weight")?,
            mask_tokens: vb.get((4, 256), "mask_tokens.weight")?,
            obj_score_token: vb.get((1, 256), "obj_score_token.weight")?,
            upconv_0: conv_transpose2d(
                256,
                64,
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("output_upscaling.0"),
            )?,
            upnorm_1: LayerNorm2d::load(64, 1e-6, vb.pp("output_upscaling.1"))?,
            upconv_3: conv_transpose2d(
                64,
                32,
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("output_upscaling.3"),
            )?,
            conv_s0: conv2d(256, 32, 1, Conv2dConfig::default(), vb.pp("conv_s0"))?,
            conv_s1: conv2d(256, 64, 1, Conv2dConfig::default(), vb.pp("conv_s1"))?,
            hypernetworks: [
                Mlp::load(
                    256,
                    256,
                    32,
                    3,
                    Activation::Relu,
                    false,
                    vb.pp("output_hypernetworks_mlps.0"),
                )?,
                Mlp::load(
                    256,
                    256,
                    32,
                    3,
                    Activation::Relu,
                    false,
                    vb.pp("output_hypernetworks_mlps.1"),
                )?,
                Mlp::load(
                    256,
                    256,
                    32,
                    3,
                    Activation::Relu,
                    false,
                    vb.pp("output_hypernetworks_mlps.2"),
                )?,
                Mlp::load(
                    256,
                    256,
                    32,
                    3,
                    Activation::Relu,
                    false,
                    vb.pp("output_hypernetworks_mlps.3"),
                )?,
            ],
            iou_head: Mlp::load(
                256,
                256,
                4,
                3,
                Activation::Relu,
                true,
                vb.pp("iou_prediction_head"),
            )?,
            object_score_head: Mlp::load(
                256,
                256,
                1,
                3,
                Activation::Relu,
                false,
                vb.pp("pred_obj_score_head"),
            )?,
            dynamic_multimask_via_stability: config.dynamic_multimask_via_stability,
            dynamic_multimask_stability_delta: config.dynamic_multimask_stability_delta,
            dynamic_multimask_stability_thresh: config.dynamic_multimask_stability_threshold,
        })
    }

    /// Computes the IoU between masks thresholded at `delta` and `-delta`.
    /// Empty unions have stability one, matching the reference implementation.
    pub fn stability_scores(mask_logits: &Tensor, delta: f64) -> Result<Tensor> {
        let rank = mask_logits.rank();
        if rank < 2 {
            candle_core::bail!("mask logits must have at least two spatial dimensions")
        }
        let height = mask_logits.dim(rank - 2)?;
        let width = mask_logits.dim(rank - 1)?;
        let flattened = mask_logits.flatten_from(rank - 2)?;
        debug_assert_eq!(flattened.dim(rank - 2)?, height * width);
        let area_i = flattened.gt(delta)?.to_dtype(DType::F32)?.sum(rank - 2)?;
        let area_u = flattened.gt(-delta)?.to_dtype(DType::F32)?.sum(rank - 2)?;
        let nonempty = area_u.gt(0f64)?;
        let ratio = area_i.broadcast_div(&area_u.maximum(1f64)?)?;
        nonempty.where_cond(&ratio, &ratio.ones_like()?)
    }

    /// Selects token zero when stable and otherwise the highest-IoU mask from
    /// tokens one through three.
    pub fn dynamic_multimask_via_stability(
        all_mask_logits: &Tensor,
        all_iou_scores: &Tensor,
        stability_delta: f64,
        stability_threshold: f64,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, tokens, height, width) = all_mask_logits.dims4()?;
        if tokens < 2 || all_iou_scores.dims() != [batch, tokens] {
            candle_core::bail!(
                "dynamic multimask expects matching BxTxHxW logits and BxT IoU scores"
            )
        }
        let multimask_logits = all_mask_logits.narrow(1, 1, tokens - 1)?;
        let multimask_iou_scores = all_iou_scores.narrow(1, 1, tokens - 1)?;
        let best_indices = multimask_iou_scores.argmax_keepdim(1)?;
        let best_iou_scores = multimask_iou_scores
            .contiguous()?
            .gather(&best_indices, 1)?;
        let mask_indices = best_indices
            .reshape((batch, 1, 1, 1))?
            .broadcast_as((batch, 1, height, width))?
            .contiguous()?;
        let best_mask_logits = multimask_logits.contiguous()?.gather(&mask_indices, 1)?;

        let singlemask_logits = all_mask_logits.narrow(1, 0, 1)?;
        let singlemask_iou_scores = all_iou_scores.narrow(1, 0, 1)?;
        let stable =
            Self::stability_scores(&singlemask_logits, stability_delta)?.ge(stability_threshold)?;
        let mask_stable = stable
            .unsqueeze(2)?
            .unsqueeze(3)?
            .broadcast_as(singlemask_logits.shape())?;
        let mask_logits = mask_stable.where_cond(&singlemask_logits, &best_mask_logits)?;
        let iou_scores = stable.where_cond(&singlemask_iou_scores, &best_iou_scores)?;
        Ok((mask_logits, iou_scores))
    }

    fn select_masks(
        &self,
        masks: &Tensor,
        iou_predictions: &Tensor,
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        if multimask_output {
            Ok((masks.narrow(1, 1, 3)?, iou_predictions.narrow(1, 1, 3)?))
        } else if self.dynamic_multimask_via_stability {
            Self::dynamic_multimask_via_stability(
                masks,
                iou_predictions,
                self.dynamic_multimask_stability_delta,
                self.dynamic_multimask_stability_thresh,
            )
        } else {
            Ok((masks.narrow(1, 0, 1)?, iou_predictions.narrow(1, 0, 1)?))
        }
    }

    /// Static-image decoder. `high_res_features` is [256x256, 128x128], the
    /// two retained high-resolution FPN maps in reference order.
    pub fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
    ) -> Result<MaskDecoderOutput> {
        self.forward_impl(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            high_res_features,
            false,
        )
    }

    /// Upstream decoder equations using only operations with Candle backward implementations.
    /// Frozen image and prompt embeddings may be detached by the caller. Dynamic multimask
    /// selection is an inference policy, so training returns token zero for single-mask output.
    pub fn forward_differentiable(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
    ) -> Result<MaskDecoderOutput> {
        if high_res_features.len() != 2 {
            candle_core::bail!("SAM2 mask decoder needs two high-resolution features")
        }
        let batch = sparse_prompt_embeddings.dim(0)?;
        let output_tokens = Tensor::cat(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .unsqueeze(0)?
        .broadcast_as((batch, 6usize, 256usize))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;
        let position = Self::position_for_batch(image_pe, batch)?;
        let (hs, src) = self.transformer.forward_differentiable(
            image_embeddings,
            dense_prompt_embeddings,
            &position,
            &tokens,
        )?;
        let iou_token_out = hs.narrow(1, 1, 1)?.squeeze(1)?;
        let all_mask_tokens = hs.narrow(1, 2, 4)?;
        let object_score_logits = self
            .object_score_head
            .forward(&hs.narrow(1, 0, 1)?.squeeze(1)?)?;
        let (_, channels, height, width) = image_embeddings.dims4()?;
        let src = src
            .permute((0, 2, 1))?
            .contiguous()?
            .reshape((batch, channels, height, width))?;
        let skip_1 = self.conv_s1.forward(&high_res_features[1])?;
        let upscaled = self.upconv_0.forward(&src)?.broadcast_add(&skip_1)?;
        let upscaled = self
            .upnorm_1
            .forward_differentiable(&upscaled)?
            .gelu_erf()?;
        let skip_0 = self.conv_s0.forward(&high_res_features[0])?;
        let upscaled = self
            .upconv_3
            .forward(&upscaled)?
            .broadcast_add(&skip_0)?
            .gelu_erf()?;
        let mut hyper = Vec::with_capacity(4);
        for (index, network) in self.hypernetworks.iter().enumerate() {
            hyper.push(network.forward(&all_mask_tokens.narrow(1, index, 1)?.squeeze(1)?)?);
        }
        let hyper = Tensor::stack(&hyper.iter().collect::<Vec<_>>(), 1)?;
        let (_, up_channels, up_height, up_width) = upscaled.dims4()?;
        let all_masks = hyper
            .matmul(
                &upscaled
                    .contiguous()?
                    .reshape((batch, up_channels, up_height * up_width))?,
            )?
            .reshape((batch, 4usize, up_height, up_width))?;
        let all_iou_predictions = self.iou_head.forward(&iou_token_out)?;
        let (masks, iou_predictions, mask_tokens) = if multimask_output {
            (
                all_masks.narrow(1, 1, 3)?,
                all_iou_predictions.narrow(1, 1, 3)?,
                all_mask_tokens.narrow(1, 1, 3)?,
            )
        } else {
            (
                all_masks.narrow(1, 0, 1)?,
                all_iou_predictions.narrow(1, 0, 1)?,
                all_mask_tokens.narrow(1, 0, 1)?,
            )
        };
        Ok(MaskDecoderOutput {
            masks,
            iou_predictions,
            mask_tokens,
            object_score_logits,
            upscaled_embedding: upscaled,
            hyper_in: hyper,
        })
    }

    /// Project the two high-resolution skips once per encoded image, as in
    /// PyTorch's `forward_image`, for reuse across prompt decodes.
    pub fn project_high_res_features(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        assert_eq!(features.len(), 2);
        Ok(vec![
            self.conv_s0.forward_fused_bias_cpu(&features[0])?,
            self.conv_s1.forward_fused_bias_cpu(&features[1])?,
        ])
    }

    /// Decode using skip features already projected by [`Self::project_high_res_features`].
    pub fn forward_preprojected(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
    ) -> Result<MaskDecoderOutput> {
        self.forward_impl(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            high_res_features,
            true,
        )
    }

    fn forward_impl(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
        preprojected: bool,
    ) -> Result<MaskDecoderOutput> {
        assert_eq!(high_res_features.len(), 2);
        let batch = sparse_prompt_embeddings.dim(0)?;
        let profiling = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
        let profile_layout = profiling
            && !PRINTED_DECODER_PROFILE_LAYOUT.swap(true, std::sync::atomic::Ordering::Relaxed);
        if profiling {
            image_embeddings.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
        macro_rules! profile_stage {
            ($name:expr) => {
                if profiling {
                    image_embeddings.device().synchronize()?;
                    eprintln!(
                        "SAM2_MASK_DECODER_STAGE stage={} batch={} ms={:.6}",
                        $name,
                        batch,
                        profile_start.elapsed().as_secs_f64() * 1000.
                    );
                    profile_start = Instant::now();
                }
            };
        }
        let output_tokens = Tensor::cat(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .unsqueeze(0)?
        .broadcast_as((batch, 6usize, 256usize))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;
        let position = Self::position_for_batch(image_pe, batch)?;
        profile_stage!("token_setup");
        let (hs, src) = self.transformer.forward(
            image_embeddings,
            dense_prompt_embeddings,
            &position,
            &tokens,
        )?;
        profile_stage!("transformer");
        let iou_token_out = hs.narrow(1, 1, 1)?.squeeze(1)?;
        let mask_tokens = hs.narrow(1, 2, 4)?;
        let object_score_logits = self
            .object_score_head
            .forward(&hs.narrow(1, 0, 1)?.squeeze(1)?)?;
        let (_, c, h, w) = image_embeddings.dims4()?;
        let src = if src.device().is_cpu()
            && src.is_contiguous()
            && std::env::var_os("SAM2_CPU_DECODER_ZERO_COPY_UPCONV")
                .is_none_or(|value| value != "0")
        {
            // The transformer returns contiguous Bx(HW)xC tokens. View that
            // storage as BHWC and permute it to a zero-copy NCHW view. The
            // linked Candle transposed-convolution path consumes this
            // pixel-major layout directly, avoiding a 4 MiB reorder per
            // prompt decode.
            src.reshape((batch, h, w, c))?.permute((0, 3, 1, 2))?
        } else {
            src.permute((0, 2, 1))?.reshape((batch, c, h, w))?
        };
        if profile_layout {
            eprintln!(
                "SAM2_AMG_LAYOUT_RS stage=upconv_0_input shape={:?} stride={:?}",
                src.dims(),
                src.stride()
            );
        }
        profile_stage!("token_heads_and_layout");
        let skip = if preprojected {
            high_res_features[1].clone()
        } else {
            self.conv_s1.forward_fused_bias_cpu(&high_res_features[1])?
        };
        let x = Self::upconv_residual(&self.upconv_0, &src, &skip, false)?;
        if profile_layout {
            eprintln!(
                "SAM2_AMG_LAYOUT_RS stage=upconv_0_output shape={:?} stride={:?}",
                x.dims(),
                x.stride()
            );
        }
        profile_stage!("upconv_0");
        let x = self.upnorm_1.forward_gelu(&x)?;
        if profile_layout {
            eprintln!(
                "SAM2_AMG_LAYOUT_RS stage=upconv_3_input shape={:?} stride={:?}",
                x.dims(),
                x.stride()
            );
        }
        profile_stage!("upnorm_gelu");
        let skip_start = profiling.then(Instant::now);
        let skip = if preprojected {
            high_res_features[0].clone()
        } else {
            self.conv_s0.forward_fused_bias_cpu(&high_res_features[0])?
        };
        if let Some(start) = skip_start {
            image_embeddings.device().synchronize()?;
            eprintln!(
                "SAM2_MASK_DECODER_SUBSTAGE stage=skip_0 batch={} ms={:.6}",
                batch,
                start.elapsed().as_secs_f64() * 1000.
            );
        }
        let x = if profiling && x.device().is_cpu() {
            let core_start = Instant::now();
            let x = Self::upconv_residual(&self.upconv_3, &x, &skip, false)?;
            image_embeddings.device().synchronize()?;
            eprintln!(
                "SAM2_MASK_DECODER_SUBSTAGE stage=upconv_core_3 batch={} ms={:.6}",
                batch,
                core_start.elapsed().as_secs_f64() * 1000.
            );
            let gelu_start = Instant::now();
            let x = x.gelu_erf()?;
            image_embeddings.device().synchronize()?;
            eprintln!(
                "SAM2_MASK_DECODER_SUBSTAGE stage=final_gelu batch={} ms={:.6}",
                batch,
                gelu_start.elapsed().as_secs_f64() * 1000.
            );
            x
        } else {
            Self::upconv_residual(&self.upconv_3, &x, &skip, true)?
        };
        if profile_layout {
            eprintln!(
                "SAM2_AMG_LAYOUT_RS stage=upconv_3_output shape={:?} stride={:?}",
                x.dims(),
                x.stride()
            );
        }
        profile_stage!("upconv_3_gelu");
        let mut hyper = Vec::with_capacity(4);
        for (index, network) in self.hypernetworks.iter().enumerate() {
            hyper.push(network.forward(&mask_tokens.narrow(1, index, 1)?.squeeze(1)?)?);
        }
        let hyper_refs: Vec<&Tensor> = hyper.iter().collect();
        let hyper = Tensor::stack(&hyper_refs, 1)?;
        profile_stage!("hypernetworks");
        let (_, channels, up_h, up_w) = x.dims4()?;
        let masks = hyper
            .matmul(&x.contiguous()?.reshape((batch, channels, up_h * up_w))?)?
            .reshape((batch, 4usize, up_h, up_w))?;
        profile_stage!("mask_projection");
        let iou_predictions = self.iou_head.forward(&iou_token_out)?;
        let (masks, iou_predictions) =
            self.select_masks(&masks, &iou_predictions, multimask_output)?;
        if profiling {
            image_embeddings.device().synchronize()?;
            eprintln!(
                "SAM2_MASK_DECODER_STAGE stage=iou_and_selection batch={} ms={:.6}",
                batch,
                profile_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(MaskDecoderOutput {
            masks,
            iou_predictions,
            mask_tokens: if multimask_output {
                mask_tokens.narrow(1, 1, 3)?
            } else {
                mask_tokens.narrow(1, 0, 1)?
            },
            object_score_logits,
            upscaled_embedding: x,
            hyper_in: hyper,
        })
    }

    /// As [`Self::forward`], additionally timing the decoder's major compute
    /// regions.  Kept separate from the hot path so production inference has
    /// no profiling overhead.
    pub fn forward_profiled(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
    ) -> Result<(MaskDecoderOutput, MaskDecoderProfile)> {
        self.forward_profiled_impl(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            high_res_features,
            false,
        )
    }

    /// Profile a decode with image-time skip projections already prepared.
    pub fn forward_profiled_preprojected(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
    ) -> Result<(MaskDecoderOutput, MaskDecoderProfile)> {
        self.forward_profiled_impl(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            multimask_output,
            high_res_features,
            true,
        )
    }

    fn forward_profiled_impl(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        high_res_features: &[Tensor],
        preprojected: bool,
    ) -> Result<(MaskDecoderOutput, MaskDecoderProfile)> {
        assert_eq!(high_res_features.len(), 2);
        let mut profile = MaskDecoderProfile::default();
        let start = Instant::now();
        let batch = sparse_prompt_embeddings.dim(0)?;
        let output_tokens = Tensor::cat(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .unsqueeze(0)?
        .broadcast_as((batch, 6usize, 256usize))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;
        let position = Self::position_for_batch(image_pe, batch)?;
        profile.token_setup_ms = start.elapsed().as_secs_f64() * 1000.;

        let start = Instant::now();
        let ((hs, src), transformer_profile) = self.transformer.forward_profiled(
            image_embeddings,
            dense_prompt_embeddings,
            &position,
            &tokens,
        )?;
        profile.transformer_ms = start.elapsed().as_secs_f64() * 1000.;
        profile.transformer = transformer_profile;
        let iou_token_out = hs.narrow(1, 1, 1)?.squeeze(1)?;
        let mask_tokens = hs.narrow(1, 2, 4)?;
        let (_, c, h, w) = image_embeddings.dims4()?;
        let src = if src.device().is_cpu()
            && src.is_contiguous()
            && std::env::var_os("SAM2_CPU_DECODER_ZERO_COPY_UPCONV")
                .is_none_or(|value| value != "0")
        {
            src.reshape((batch, h, w, c))?.permute((0, 3, 1, 2))?
        } else {
            src.permute((0, 2, 1))?.reshape((batch, c, h, w))?
        };

        let start = Instant::now();
        let skip = if preprojected {
            high_res_features[1].clone()
        } else {
            self.conv_s1.forward_fused_bias_cpu(&high_res_features[1])?
        };
        profile.skip_1_ms = start.elapsed().as_secs_f64() * 1000.;
        let core_start = Instant::now();
        let x = Self::upconv_residual(&self.upconv_0, &src, &skip, false)?;
        profile.upconv_core_0_ms = core_start.elapsed().as_secs_f64() * 1000.;
        profile.upconv_0_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let x = self.upnorm_1.forward_gelu(&x)?;
        profile.upnorm_gelu_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let skip = if preprojected {
            high_res_features[0].clone()
        } else {
            self.conv_s0.forward_fused_bias_cpu(&high_res_features[0])?
        };
        profile.skip_0_ms = start.elapsed().as_secs_f64() * 1000.;
        let core_start = Instant::now();
        let x = Self::upconv_residual(&self.upconv_3, &x, &skip, false)?;
        profile.upconv_core_3_ms = core_start.elapsed().as_secs_f64() * 1000.;
        let gelu_start = Instant::now();
        let x = x.gelu_erf()?;
        profile.final_gelu_ms = gelu_start.elapsed().as_secs_f64() * 1000.;
        profile.upconv_3_ms = start.elapsed().as_secs_f64() * 1000.;

        let start = Instant::now();
        let mut hyper = Vec::with_capacity(4);
        for (index, network) in self.hypernetworks.iter().enumerate() {
            hyper.push(network.forward(&mask_tokens.narrow(1, index, 1)?.squeeze(1)?)?);
        }
        let hyper_refs: Vec<&Tensor> = hyper.iter().collect();
        let hyper = Tensor::stack(&hyper_refs, 1)?;
        profile.hypernetworks_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let (_, channels, up_h, up_w) = x.dims4()?;
        let masks = hyper
            .matmul(&x.contiguous()?.reshape((batch, channels, up_h * up_w))?)?
            .reshape((batch, 4usize, up_h, up_w))?;
        profile.mask_projection_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let object_score_logits = self
            .object_score_head
            .forward(&hs.narrow(1, 0, 1)?.squeeze(1)?)?;
        let iou_predictions = self.iou_head.forward(&iou_token_out)?;
        let (masks, iou_predictions) =
            self.select_masks(&masks, &iou_predictions, multimask_output)?;
        profile.heads_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok((
            MaskDecoderOutput {
                masks,
                iou_predictions,
                mask_tokens: if multimask_output {
                    mask_tokens.narrow(1, 1, 3)?
                } else {
                    mask_tokens.narrow(1, 0, 1)?
                },
                object_score_logits,
                upscaled_embedding: x,
                hyper_in: hyper,
            },
            profile,
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn stability_scores_treat_empty_masks_as_stable() {
        let logits = Tensor::from_vec(
            vec![1f32, 1., 1., 1., -1., -1., -1., -1.],
            (2, 1, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            MaskDecoder::stability_scores(&logits, 0.05)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.], vec![1.]]
        );
    }

    #[test]
    fn unstable_single_mask_falls_back_to_highest_iou_multimask() {
        let mut values = Vec::new();
        values.extend([1f32; 4]);
        values.extend([11f32; 4]);
        values.extend([12f32; 4]);
        values.extend([13f32; 4]);
        values.extend([0f32, 0., 1., -1.]);
        values.extend([21f32; 4]);
        values.extend([22f32; 4]);
        values.extend([23f32; 4]);
        let logits = Tensor::from_vec(values, (2, 4, 2, 2), &Device::Cpu).unwrap();
        let ious = Tensor::from_vec(
            vec![0.7f32, 0.8, 0.6, 0.5, 0.1, 0.2, 0.9, 0.5],
            (2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let (selected_logits, selected_ious) =
            MaskDecoder::dynamic_multimask_via_stability(&logits, &ious, 0.05, 0.98).unwrap();
        assert_eq!(
            selected_logits
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 1., 1., 1., 22., 22., 22., 22.]
        );
        assert_eq!(
            selected_ious.to_vec2::<f32>().unwrap(),
            vec![vec![0.7], vec![0.9]]
        );
    }
}
