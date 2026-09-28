//! Translation of `sam2/modeling/memory_encoder.py`.

use std::time::Instant;

use candle_core::{DType, Result, Tensor};
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
    pwconv1_cpu: Conv2d,
    pwconv2_cpu: Conv2d,
    gamma: Tensor,
    gamma_nchw: Tensor,
}
impl CxBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let pwconv1 = linear(256, 1024, vb.pp("pwconv1"))?;
        let pwconv2 = linear(1024, 256, vb.pp("pwconv2"))?;
        // A pointwise Linear over NHWC and a 1x1 convolution over NCHW are
        // mathematically identical. Keep shallow reshaped handles to the same
        // checkpoint storage so CPU inference can stay NCHW and feed one GEMM
        // without materializing the 4 MiB activation in NHWC order.
        let pwconv1_cpu = Conv2d::new(
            pwconv1.weight().reshape((1024, 256, 1, 1))?,
            pwconv1.bias().cloned(),
            Conv2dConfig::default(),
        );
        let pwconv2_cpu = Conv2d::new(
            pwconv2.weight().reshape((256, 1024, 1, 1))?,
            pwconv2.bias().cloned(),
            Conv2dConfig::default(),
        );
        let gamma = vb.get(256, "gamma")?;
        let gamma_nchw = gamma.reshape((1, 256, 1, 1))?;
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
            pwconv1,
            pwconv2,
            pwconv1_cpu,
            pwconv2_cpu,
            gamma,
            gamma_nchw,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let profiling = std::env::var_os("SAM2_PROFILE_VIDEO_STAGES").is_some();
        if profiling {
            x.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
        macro_rules! profile_stage {
            ($name:expr) => {
                if profiling {
                    x.device().synchronize()?;
                    eprintln!(
                        "SAM2_MEMORY_FUSER_STAGE stage={} ms={:.6}",
                        $name,
                        profile_start.elapsed().as_secs_f64() * 1000.
                    );
                    profile_start = Instant::now();
                }
            };
        }
        let input = x.clone();
        let x = self.dwconv.forward(x)?;
        profile_stage!("depthwise_conv");
        let x = self.norm.forward(&x)?;
        profile_stage!("layer_norm_2d");
        if x.device().is_cpu()
            && x.dtype() == DType::F32
            && std::env::var_os("SAM2_CPU_MEMORY_FUSER_NCHW_POINTWISE")
                .is_none_or(|value| value != "0")
        {
            let x = self.pwconv1_cpu.forward_fused_bias_cpu(&x)?;
            profile_stage!("pointwise_1");
            let x = x.gelu_erf()?;
            profile_stage!("gelu");
            let x = self.pwconv2_cpu.forward_fused_bias_cpu(&x)?;
            profile_stage!("pointwise_2");
            let x = x.broadcast_mul(&self.gamma_nchw)?;
            let output = input.broadcast_add(&x)?;
            if profiling {
                x.device().synchronize()?;
                eprintln!(
                    "SAM2_MEMORY_FUSER_STAGE stage=scale_and_residual ms={:.6}",
                    profile_start.elapsed().as_secs_f64() * 1000.
                );
            }
            return Ok(output);
        }
        let x = x.permute((0, 2, 3, 1))?;
        let x = self.pwconv1.forward(&x)?;
        profile_stage!("pointwise_1");
        let x = x.gelu_erf()?;
        profile_stage!("gelu");
        let x = self.pwconv2.forward(&x)?;
        profile_stage!("pointwise_2");
        let x = x.broadcast_mul(&self.gamma)?.permute((0, 3, 1, 2))?;
        let output = input.broadcast_add(&x)?;
        if profiling {
            x.device().synchronize()?;
            eprintln!(
                "SAM2_MEMORY_FUSER_STAGE stage=scale_and_residual ms={:.6}",
                profile_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(output)
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
    cached_position_encoding: Tensor,
    out_proj: Conv2d,
}
impl MemoryEncoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let position_encoding = PositionEmbeddingSine::new(64, 10_000., true, None);
        // SAM2's memory encoder always emits 64x64 features for its fixed
        // 1024px model input. The sine encoding is immutable and identical on
        // every frame, so retain one shallow-cloned device tensor instead of
        // rebuilding its coordinate grids and trigonometric intermediates.
        let cached_position_encoding = position_encoding.pe(1, 64, 64, vb.device())?;
        Ok(Self {
            mask_downsampler: MaskDownsampler::load(vb.pp("mask_downsampler"))?,
            pix_feat_proj: conv2d(256, 256, 1, Conv2dConfig::default(), vb.pp("pix_feat_proj"))?,
            fuser: [
                CxBlock::load(vb.pp("fuser.layers.0"))?,
                CxBlock::load(vb.pp("fuser.layers.1"))?,
            ],
            position_encoding,
            cached_position_encoding,
            out_proj: conv2d(256, 64, 1, Conv2dConfig::default(), vb.pp("out_proj"))?,
        })
    }
    pub fn forward(
        &self,
        pix_feat: &Tensor,
        masks: &Tensor,
        skip_mask_sigmoid: bool,
    ) -> Result<MemoryEncoderOutput> {
        let profiling = std::env::var_os("SAM2_PROFILE_VIDEO_STAGES").is_some();
        if profiling {
            pix_feat.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
        macro_rules! profile_stage {
            ($name:expr) => {
                if profiling {
                    pix_feat.device().synchronize()?;
                    eprintln!(
                        "SAM2_MEMORY_ENCODER_STAGE stage={} ms={:.6}",
                        $name,
                        profile_start.elapsed().as_secs_f64() * 1000.
                    );
                    profile_start = Instant::now();
                }
            };
        }
        let masks = if skip_mask_sigmoid {
            masks.clone()
        } else {
            sigmoid(masks)?
        };
        let masks = self.mask_downsampler.forward(&masks)?;
        profile_stage!("mask_downsampler");
        let mut x = self
            .pix_feat_proj
            .forward(pix_feat)?
            .broadcast_add(&masks)?;
        profile_stage!("pixel_projection");
        for (index, layer) in self.fuser.iter().enumerate() {
            x = layer.forward(&x)?;
            profile_stage!(if index == 0 { "fuser_0" } else { "fuser_1" });
        }
        let features = self.out_proj.forward(&x)?;
        profile_stage!("output_projection");
        let batch = features.dim(0)?;
        let pos = if batch == 1 && features.dims()[2..] == [64, 64] {
            self.cached_position_encoding.clone()
        } else {
            self.position_encoding.forward(&features)?
        }
        .to_dtype(features.dtype())?;
        if profiling {
            pix_feat.device().synchronize()?;
            eprintln!(
                "SAM2_MEMORY_ENCODER_STAGE stage=position_encoding ms={:.6}",
                profile_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(MemoryEncoderOutput {
            vision_features: features,
            vision_pos_enc: pos,
        })
    }
}
