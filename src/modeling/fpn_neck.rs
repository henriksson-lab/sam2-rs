//! Translation of `FpnNeck` and `ImageEncoder` in `image_encoder.py`.

use std::time::Instant;

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};

use crate::{Hiera, config::Sam2Config, ops::PositionEmbeddingSine};

#[derive(Clone, Debug)]
pub struct FpnNeck {
    position_encoding: PositionEmbeddingSine,
    convs: Vec<Conv2d>,
    fpn_top_down_levels: Vec<usize>,
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
            let lateral = self.convs[self.convs.len() - 1 - index].forward(&xs[index])?;
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
}

/// Synchronized breakdown for one full image-encoder invocation.
#[derive(Clone, Debug)]
pub struct ImageEncoderProfile {
    pub hiera_setup_ms: f64,
    pub hiera_block_ms: Vec<f64>,
    pub fpn_ms: f64,
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
        })
    }

    /// Profile Hiera and FPN without changing the inference math.
    pub fn forward_profiled(
        &self,
        image: &Tensor,
    ) -> Result<(BackboneOutput, ImageEncoderProfile)> {
        let (trunk, hiera) = self.trunk.forward_profiled(image)?;
        let start = Instant::now();
        let (mut features, mut positions) = self.neck.forward(&trunk)?;
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
            },
            ImageEncoderProfile {
                hiera_setup_ms: hiera.setup_ms,
                hiera_block_ms: hiera.block_ms,
                fpn_ms,
            },
        ))
    }
}
