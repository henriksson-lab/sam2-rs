//! Translation of `sam2/modeling/sam/mask_decoder.py` for static inference.

use candle_core::{Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder,
    conv_transpose2d, conv2d,
};

use crate::{
    TwoWayTransformer,
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
}

impl MaskDecoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
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
        })
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
        let position =
            image_pe.broadcast_as((batch, 256usize, image_pe.dim(2)?, image_pe.dim(3)?))?;
        let (hs, src) = self.transformer.forward(&src, &position, &tokens)?;
        let iou_token_out = hs.narrow(1, 1, 1)?.squeeze(1)?;
        let mask_tokens = hs.narrow(1, 2, 4)?;
        let object_score_logits = self
            .object_score_head
            .forward(&hs.narrow(1, 0, 1)?.squeeze(1)?)?;
        let (b, c, h, w) = image_embeddings.dims4()?;
        let src = src.permute((0, 2, 1))?.reshape((b, c, h, w))?;
        let x = self
            .upconv_0
            .forward(&src)?
            .broadcast_add(&self.conv_s1.forward(&high_res_features[1])?)?;
        let x = self.upnorm_1.forward(&x)?.gelu_erf()?;
        let x = self
            .upconv_3
            .forward(&x)?
            .broadcast_add(&self.conv_s0.forward(&high_res_features[0])?)?
            .gelu_erf()?;
        let mut hyper = Vec::with_capacity(4);
        for (index, network) in self.hypernetworks.iter().enumerate() {
            hyper.push(network.forward(&mask_tokens.narrow(1, index, 1)?.squeeze(1)?)?);
        }
        let hyper_refs: Vec<&Tensor> = hyper.iter().collect();
        let hyper = Tensor::stack(&hyper_refs, 1)?;
        let (_, channels, up_h, up_w) = x.dims4()?;
        let masks = hyper
            .matmul(&x.contiguous()?.reshape((b, channels, up_h * up_w))?)?
            .reshape((b, 4usize, up_h, up_w))?;
        let iou_predictions = self.iou_head.forward(&iou_token_out)?;
        let (masks, iou_predictions) = if multimask_output {
            (masks.narrow(1, 1, 3)?, iou_predictions.narrow(1, 1, 3)?)
        } else {
            (masks.narrow(1, 0, 1)?, iou_predictions.narrow(1, 0, 1)?)
        };
        Ok(MaskDecoderOutput {
            masks,
            iou_predictions,
            mask_tokens: mask_tokens.narrow(1, 0, 1)?,
            object_score_logits,
            upscaled_embedding: x,
            hyper_in: hyper,
        })
    }
}
