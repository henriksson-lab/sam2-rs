//! Translation of `FpnNeck` and `ImageEncoder` in `image_encoder.py`.

use std::time::Instant;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, conv2d};

use crate::{Hiera, config::Sam2Config, ops::PositionEmbeddingSine};

#[derive(Clone, Debug)]
pub struct FpnNeck {
    position_encoding: PositionEmbeddingSine,
    convs: Vec<Conv2d>,
    fpn_top_down_levels: Vec<usize>,
}

/// Opt-in FPN timing boundaries for CPU backend attribution.
#[derive(Clone, Debug, Default)]
pub struct FpnProfile {
    pub lateral_ms: Vec<f64>,
    pub top_down_ms: Vec<f64>,
    pub position_ms: Vec<f64>,
}

impl FpnNeck {
    pub fn load(config: &Sam2Config, vb: VarBuilder) -> Result<Self> {
        let mut convs = Vec::with_capacity(4);
        for (index, channels) in config.backbone_channel_list.iter().enumerate() {
            convs.push(conv2d(
                *channels,
                config.hidden_dim,
                1,
                Conv2dConfig::default(),
                vb.pp(format!("convs.{index}.conv")),
            )?);
        }
        Ok(Self {
            position_encoding: PositionEmbeddingSine::new(config.hidden_dim, 10_000., true, None),
            convs,
            fpn_top_down_levels: config.fpn_top_down_levels.clone(),
        })
    }

    /// Input is Hiera's fine-to-coarse stage list. Output keeps that ordering.
    pub fn forward(&self, xs: &[Tensor]) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        assert_eq!(xs.len(), self.convs.len());
        let mut out: Vec<Option<Tensor>> = vec![None; self.convs.len()];
        let mut pos: Vec<Option<Tensor>> = vec![None; self.convs.len()];
        let mut previous: Option<Tensor> = None;
        for index in (0..self.convs.len()).rev() {
            let lateral =
                self.convs[self.convs.len() - 1 - index].forward_fused_bias_cpu(&xs[index])?;
            let current = if self.fpn_top_down_levels.contains(&index) {
                if let Some(previous) = previous {
                    lateral.broadcast_add(
                        &previous
                            .to_dtype(DType::F32)?
                            .upsample_nearest2d(lateral.dim(2)?, lateral.dim(3)?)?
                            .to_dtype(lateral.dtype())?,
                    )?
                } else {
                    lateral
                }
            } else {
                lateral
            };
            let position = self
                .position_encoding
                .forward(&current)?
                .to_dtype(current.dtype())?;
            previous = Some(current.clone());
            out[index] = Some(current);
            pos[index] = Some(position);
        }
        Ok((
            out.into_iter().map(Option::unwrap).collect(),
            pos.into_iter().map(Option::unwrap).collect(),
        ))
    }

    pub fn forward_profiled(
        &self,
        xs: &[Tensor],
    ) -> Result<((Vec<Tensor>, Vec<Tensor>), FpnProfile)> {
        assert_eq!(xs.len(), self.convs.len());
        let mut out: Vec<Option<Tensor>> = vec![None; self.convs.len()];
        let mut pos: Vec<Option<Tensor>> = vec![None; self.convs.len()];
        let mut previous: Option<Tensor> = None;
        let mut profile = FpnProfile {
            lateral_ms: vec![0.; self.convs.len()],
            top_down_ms: vec![0.; self.convs.len()],
            position_ms: vec![0.; self.convs.len()],
        };
        for index in (0..self.convs.len()).rev() {
            let start = Instant::now();
            let lateral =
                self.convs[self.convs.len() - 1 - index].forward_fused_bias_cpu(&xs[index])?;
            profile.lateral_ms[index] = start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let current = if self.fpn_top_down_levels.contains(&index) {
                if let Some(previous) = previous {
                    lateral.broadcast_add(
                        &previous
                            .to_dtype(DType::F32)?
                            .upsample_nearest2d(lateral.dim(2)?, lateral.dim(3)?)?
                            .to_dtype(lateral.dtype())?,
                    )?
                } else {
                    lateral
                }
            } else {
                lateral
            };
            profile.top_down_ms[index] = start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let position = self
                .position_encoding
                .forward(&current)?
                .to_dtype(current.dtype())?;
            profile.position_ms[index] = start.elapsed().as_secs_f64() * 1000.;
            previous = Some(current.clone());
            out[index] = Some(current);
            pos[index] = Some(position);
        }
        Ok((
            (
                out.into_iter().map(Option::unwrap).collect(),
                pos.into_iter().map(Option::unwrap).collect(),
            ),
            profile,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ImageEncoder {
    trunk: Hiera,
    neck: FpnNeck,
    scalp: usize,
}

#[derive(Clone, Debug)]
pub struct BackboneOutput {
    pub vision_features: Tensor,
    pub vision_pos_enc: Vec<Tensor>,
    pub backbone_fpn: Vec<Tensor>,
    /// Mask-decoder skip projections, when prepared by an image predictor.
    /// Standalone image encoders leave these absent.
    pub sam_high_res_features: Option<Vec<Tensor>>,
}

/// Synchronized breakdown for one full image-encoder invocation.
#[derive(Clone, Debug)]
pub struct ImageEncoderProfile {
    pub hiera_setup_ms: f64,
    pub hiera_block_ms: Vec<f64>,
    pub hiera_norm1_ms: Vec<f64>,
    pub hiera_shortcut_ms: Vec<f64>,
    pub hiera_shortcut_pool_ms: Vec<f64>,
    pub hiera_partition_ms: Vec<f64>,
    pub hiera_attention_ms: Vec<f64>,
    pub hiera_attention_qkv_ms: Vec<f64>,
    pub hiera_attention_qkv_views_ms: Vec<f64>,
    pub hiera_attention_query_pool_ms: Vec<f64>,
    pub hiera_attention_query_reshape_ms: Vec<f64>,
    pub hiera_attention_query_pool_core_ms: Vec<f64>,
    pub hiera_attention_kernel_ms: Vec<f64>,
    pub hiera_attention_projection_ms: Vec<f64>,
    pub hiera_attention_cleanup_ms: Vec<f64>,
    pub hiera_attention_cleanup_attended_ms: Vec<f64>,
    pub hiera_attention_cleanup_qkv_ms: Vec<f64>,
    pub hiera_unpartition_ms: Vec<f64>,
    pub hiera_residual_add_ms: Vec<f64>,
    pub hiera_norm2_ms: Vec<f64>,
    pub hiera_mlp_ms: Vec<f64>,
    pub hiera_mlp_expansion_ms: Vec<f64>,
    pub hiera_mlp_activation_ms: Vec<f64>,
    pub hiera_mlp_projection_ms: Vec<f64>,
    pub fpn_ms: f64,
    pub fpn_lateral_ms: Vec<f64>,
    pub fpn_top_down_ms: Vec<f64>,
    pub fpn_position_ms: Vec<f64>,
}

impl ImageEncoder {
    pub fn load(config: &Sam2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            trunk: Hiera::load(&config.hierarchy, vb.pp("trunk"))?,
            neck: FpnNeck::load(config, vb.pp("neck"))?,
            scalp: 1,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<BackboneOutput> {
        let (mut features, mut positions) = self.neck.forward(&self.trunk.forward(image)?)?;
        if self.scalp > 0 {
            let retained = features.len() - self.scalp;
            features.truncate(retained);
            positions.truncate(retained);
        }
        Ok(BackboneOutput {
            vision_features: features.last().unwrap().clone(),
            vision_pos_enc: positions,
            backbone_fpn: features,
            sam_high_res_features: None,
        })
    }

    /// Profile Hiera and FPN without changing the inference math.
    pub fn forward_profiled(
        &self,
        image: &Tensor,
    ) -> Result<(BackboneOutput, ImageEncoderProfile)> {
        let (trunk, hiera) = self.trunk.forward_profiled(image)?;
        let start = Instant::now();
        let ((mut features, mut positions), fpn) = self.neck.forward_profiled(&trunk)?;
        image.device().synchronize()?;
        let fpn_ms = start.elapsed().as_secs_f64() * 1000.;
        if self.scalp > 0 {
            let retained = features.len() - self.scalp;
            features.truncate(retained);
            positions.truncate(retained);
        }
        Ok((
            BackboneOutput {
                vision_features: features.last().unwrap().clone(),
                vision_pos_enc: positions,
                backbone_fpn: features,
                sam_high_res_features: None,
            },
            ImageEncoderProfile {
                hiera_setup_ms: hiera.setup_ms,
                hiera_block_ms: hiera.block_ms,
                hiera_norm1_ms: hiera.norm1_ms,
                hiera_shortcut_ms: hiera.shortcut_ms,
                hiera_shortcut_pool_ms: hiera.shortcut_pool_ms,
                hiera_partition_ms: hiera.partition_ms,
                hiera_attention_ms: hiera.attention_ms,
                hiera_attention_qkv_ms: hiera.attention_qkv_ms,
                hiera_attention_qkv_views_ms: hiera.attention_qkv_views_ms,
                hiera_attention_query_pool_ms: hiera.attention_query_pool_ms,
                hiera_attention_query_reshape_ms: hiera.attention_query_reshape_ms,
                hiera_attention_query_pool_core_ms: hiera.attention_query_pool_core_ms,
                hiera_attention_kernel_ms: hiera.attention_kernel_ms,
                hiera_attention_projection_ms: hiera.attention_projection_ms,
                hiera_attention_cleanup_ms: hiera.attention_cleanup_ms,
                hiera_attention_cleanup_attended_ms: hiera.attention_cleanup_attended_ms,
                hiera_attention_cleanup_qkv_ms: hiera.attention_cleanup_qkv_ms,
                hiera_unpartition_ms: hiera.unpartition_ms,
                hiera_residual_add_ms: hiera.residual_add_ms,
                hiera_norm2_ms: hiera.norm2_ms,
                hiera_mlp_ms: hiera.mlp_ms,
                hiera_mlp_expansion_ms: hiera.mlp_expansion_ms,
                hiera_mlp_activation_ms: hiera.mlp_activation_ms,
                hiera_mlp_projection_ms: hiera.mlp_projection_ms,
                fpn_ms,
                fpn_lateral_ms: fpn.lateral_ms,
                fpn_top_down_ms: fpn.top_down_ms,
                fpn_position_ms: fpn.position_ms,
            },
        ))
    }
}
