//! Translation of `sam2/modeling/memory_encoder.py`.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d, linear, ops::sigmoid};

use crate::ops::{LayerNorm2d, PositionEmbeddingSine};

#[derive(Clone, Debug)]
struct MaskDownsampler {
    conv_0: Conv2d,
    norm_1: LayerNorm2d,
    conv_3: Conv2d,
    norm_4: LayerNorm2d,
    conv_6: Conv2d,
    norm_7: LayerNorm2d,
    conv_9: Conv2d,
    norm_10: LayerNorm2d,
    conv_12: Conv2d,
}

impl MaskDownsampler {
    fn load(vb: VarBuilder) -> Result<Self> {
        let config = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv_0: conv2d(1, 4, 3, config, vb.pp("encoder.0"))?,
            norm_1: LayerNorm2d::load(4, 1e-6, vb.pp("encoder.1"))?,
            conv_3: conv2d(4, 16, 3, config, vb.pp("encoder.3"))?,
            norm_4: LayerNorm2d::load(16, 1e-6, vb.pp("encoder.4"))?,
            conv_6: conv2d(16, 64, 3, config, vb.pp("encoder.6"))?,
            norm_7: LayerNorm2d::load(64, 1e-6, vb.pp("encoder.7"))?,
            conv_9: conv2d(64, 256, 3, config, vb.pp("encoder.9"))?,
            norm_10: LayerNorm2d::load(256, 1e-6, vb.pp("encoder.10"))?,
            conv_12: conv2d(256, 256, 1, Conv2dConfig::default(), vb.pp("encoder.12"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm_1.forward(&self.conv_0.forward(x)?)?.gelu_erf()?;
        let x = self.norm_4.forward(&self.conv_3.forward(&x)?)?.gelu_erf()?;
        let x = self.norm_7.forward(&self.conv_6.forward(&x)?)?.gelu_erf()?;
        let x = self
            .norm_10
            .forward(&self.conv_9.forward(&x)?)?
            .gelu_erf()?;
        self.conv_12.forward(&x)
    }
}

#[derive(Clone, Debug)]
struct CxBlock {
    dwconv: Conv2d,
    norm: LayerNorm2d,
    pwconv1: candle_nn::Linear,
    pwconv2: candle_nn::Linear,
    gamma: Tensor,
}
impl CxBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dwconv: conv2d(
                256,
                256,
                7,
                Conv2dConfig {
                    padding: 3,
                    groups: 256,
                    ..Default::default()
                },
                vb.pp("dwconv"),
            )?,
            norm: LayerNorm2d::load(256, 1e-6, vb.pp("norm"))?,
            pwconv1: linear(256, 1024, vb.pp("pwconv1"))?,
            pwconv2: linear(1024, 256, vb.pp("pwconv2"))?,
            gamma: vb.get(256, "gamma")?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x.clone();
        let x = self
            .norm
            .forward(&self.dwconv.forward(x)?)?
            .permute((0, 2, 3, 1))?;
        let x = self
            .pwconv2
            .forward(&self.pwconv1.forward(&x)?.gelu_erf()?)?;
        input.broadcast_add(&x.broadcast_mul(&self.gamma)?.permute((0, 3, 1, 2))?)
    }
}

#[derive(Clone, Debug)]
pub struct MemoryEncoderOutput {
    pub vision_features: Tensor,
    pub vision_pos_enc: Tensor,
}

#[derive(Clone, Debug)]
pub struct MemoryEncoder {
    mask_downsampler: MaskDownsampler,
    pix_feat_proj: Conv2d,
    fuser: [CxBlock; 2],
    position_encoding: PositionEmbeddingSine,
    out_proj: Conv2d,
}
impl MemoryEncoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            mask_downsampler: MaskDownsampler::load(vb.pp("mask_downsampler"))?,
            pix_feat_proj: conv2d(256, 256, 1, Conv2dConfig::default(), vb.pp("pix_feat_proj"))?,
            fuser: [
                CxBlock::load(vb.pp("fuser.layers.0"))?,
                CxBlock::load(vb.pp("fuser.layers.1"))?,
            ],
            position_encoding: PositionEmbeddingSine::new(64, 10_000., true, None),
            out_proj: conv2d(256, 64, 1, Conv2dConfig::default(), vb.pp("out_proj"))?,
        })
    }
    pub fn forward(
        &self,
        pix_feat: &Tensor,
        masks: &Tensor,
        skip_mask_sigmoid: bool,
    ) -> Result<MemoryEncoderOutput> {
        let masks = if skip_mask_sigmoid {
            masks.clone()
        } else {
            sigmoid(masks)?
        };
        let masks = self.mask_downsampler.forward(&masks)?;
        let mut x = self
            .pix_feat_proj
            .forward(pix_feat)?
            .broadcast_add(&masks)?;
        for layer in &self.fuser {
            x = layer.forward(&x)?;
        }
        let features = self.out_proj.forward(&x)?;
        let pos = self
            .position_encoding
            .forward(&features)?
            .to_dtype(features.dtype())?;
        Ok(MemoryEncoderOutput {
            vision_features: features,
            vision_pos_enc: pos,
        })
    }
}
