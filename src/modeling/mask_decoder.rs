//! Translation of `sam2/modeling/sam/mask_decoder.py` for static inference.

use std::time::Instant;

use candle_core::{DType, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder, conv_transpose2d,
    conv2d,
};

use crate::{
    TwoWayTransformer, TwoWayTransformerProfile,
    ops::{Activation, LayerNorm2d, Mlp},
};

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
    fn position_for_batch(image_pe: &Tensor, batch: usize) -> Result<Tensor> {
        let (_, _, height, width) = image_pe.dims4()?;
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
        let output_tokens = Tensor::cat(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .unsqueeze(0)?
        .broadcast_as((batch, 6usize, 256usize))?;
        let tokens = Tensor::cat(&[&output_tokens, sparse_prompt_embeddings], 1)?;
        let src = image_embeddings.broadcast_add(dense_prompt_embeddings)?;
        let position = Self::position_for_batch(image_pe, batch)?;
        let (hs, src) = self.transformer.forward(&src, &position, &tokens)?;
        let iou_token_out = hs.narrow(1, 1, 1)?.squeeze(1)?;
        let mask_tokens = hs.narrow(1, 2, 4)?;
        let object_score_logits = self
            .object_score_head
            .forward(&hs.narrow(1, 0, 1)?.squeeze(1)?)?;
        let (_, c, h, w) = image_embeddings.dims4()?;
        let src = src.permute((0, 2, 1))?.reshape((batch, c, h, w))?;
        let skip = if preprojected {
            high_res_features[1].clone()
        } else {
            self.conv_s1.forward_fused_bias_cpu(&high_res_features[1])?
        };
        let x = self.upconv_0.forward_add_residual_cpu(&src, &skip)?;
        let x = self.upnorm_1.forward(&x)?.gelu_erf()?;
        let skip = if preprojected {
            high_res_features[0].clone()
        } else {
            self.conv_s0.forward_fused_bias_cpu(&high_res_features[0])?
        };
        let x = self
            .upconv_3
            .forward_add_residual_cpu(&x, &skip)?
            .gelu_erf()?;
        let mut hyper = Vec::with_capacity(4);
        for (index, network) in self.hypernetworks.iter().enumerate() {
            hyper.push(network.forward(&mask_tokens.narrow(1, index, 1)?.squeeze(1)?)?);
        }
        let hyper_refs: Vec<&Tensor> = hyper.iter().collect();
        let hyper = Tensor::stack(&hyper_refs, 1)?;
        let (_, channels, up_h, up_w) = x.dims4()?;
        let masks = hyper
            .matmul(&x.contiguous()?.reshape((batch, channels, up_h * up_w))?)?
            .reshape((batch, 4usize, up_h, up_w))?;
        let iou_predictions = self.iou_head.forward(&iou_token_out)?;
        let (masks, iou_predictions) =
            self.select_masks(&masks, &iou_predictions, multimask_output)?;
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
        let src = image_embeddings.broadcast_add(dense_prompt_embeddings)?;
        let position = Self::position_for_batch(image_pe, batch)?;
        profile.token_setup_ms = start.elapsed().as_secs_f64() * 1000.;

        let start = Instant::now();
        let ((hs, src), transformer_profile) = self
            .transformer
            .forward_profiled(&src, &position, &tokens)?;
        profile.transformer_ms = start.elapsed().as_secs_f64() * 1000.;
        profile.transformer = transformer_profile;
        let iou_token_out = hs.narrow(1, 1, 1)?.squeeze(1)?;
        let mask_tokens = hs.narrow(1, 2, 4)?;
        let (_, c, h, w) = image_embeddings.dims4()?;
        let src = src.permute((0, 2, 1))?.reshape((batch, c, h, w))?;

        let start = Instant::now();
        let skip = if preprojected {
            high_res_features[1].clone()
        } else {
            self.conv_s1.forward_fused_bias_cpu(&high_res_features[1])?
        };
        profile.skip_1_ms = start.elapsed().as_secs_f64() * 1000.;
        let core_start = Instant::now();
        let x = self.upconv_0.forward_add_residual_cpu(&src, &skip)?;
        profile.upconv_core_0_ms = core_start.elapsed().as_secs_f64() * 1000.;
        profile.upconv_0_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let x = self.upnorm_1.forward(&x)?.gelu_erf()?;
        profile.upnorm_gelu_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let skip = if preprojected {
            high_res_features[0].clone()
        } else {
            self.conv_s0.forward_fused_bias_cpu(&high_res_features[0])?
        };
        profile.skip_0_ms = start.elapsed().as_secs_f64() * 1000.;
        let core_start = Instant::now();
        let x = self.upconv_3.forward_add_residual_cpu(&x, &skip)?;
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
