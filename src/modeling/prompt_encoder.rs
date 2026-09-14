//! Translation of `sam2/modeling/sam/prompt_encoder.py`.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};

use crate::{
    config::Sam2Config,
    ops::{LayerNorm2d, PositionEmbeddingRandom},
};

#[derive(Clone, Debug)]
pub struct PromptEncoderOutput {
    pub sparse_embeddings: Tensor,
    pub dense_embeddings: Tensor,
}

#[derive(Clone, Debug)]
pub struct PromptEncoder {
    embed_dim: usize,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    pe_layer: PositionEmbeddingRandom,
    point_embeddings: [Tensor; 4],
    not_a_point_embed: Tensor,
    mask_conv_0: Conv2d,
    mask_norm_1: LayerNorm2d,
    mask_conv_3: Conv2d,
    mask_norm_4: LayerNorm2d,
    mask_conv_6: Conv2d,
    no_mask_embed: Tensor,
}

impl PromptEncoder {
    pub fn load(config: &Sam2Config, vb: VarBuilder) -> Result<Self> {
        let mask_channels = 16;
        Ok(Self {
            embed_dim: config.hidden_dim,
            image_embedding_size: (
                config.image_size / config.backbone_stride,
                config.image_size / config.backbone_stride,
            ),
            input_image_size: (config.image_size, config.image_size),
            pe_layer: PositionEmbeddingRandom::load(config.hidden_dim / 2, vb.pp("pe_layer"))?,
            point_embeddings: [
                vb.pp("point_embeddings.0")
                    .get((1, config.hidden_dim), "weight")?,
                vb.pp("point_embeddings.1")
                    .get((1, config.hidden_dim), "weight")?,
                vb.pp("point_embeddings.2")
                    .get((1, config.hidden_dim), "weight")?,
                vb.pp("point_embeddings.3")
                    .get((1, config.hidden_dim), "weight")?,
            ],
            not_a_point_embed: vb.get((1, config.hidden_dim), "not_a_point_embed.weight")?,
            mask_conv_0: conv2d(
                1,
                mask_channels / 4,
                2,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("mask_downscaling.0"),
            )?,
            mask_norm_1: LayerNorm2d::load(mask_channels / 4, 1e-6, vb.pp("mask_downscaling.1"))?,
            mask_conv_3: conv2d(
                mask_channels / 4,
                mask_channels,
                2,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("mask_downscaling.3"),
            )?,
            mask_norm_4: LayerNorm2d::load(mask_channels, 1e-6, vb.pp("mask_downscaling.4"))?,
            mask_conv_6: conv2d(
                mask_channels,
                config.hidden_dim,
                1,
                Conv2dConfig::default(),
                vb.pp("mask_downscaling.6"),
            )?,
            no_mask_embed: vb.get((1, config.hidden_dim), "no_mask_embed.weight")?,
        })
    }

    pub fn get_dense_pe(&self) -> Result<Tensor> {
        self.pe_layer
            .forward(self.image_embedding_size.0, self.image_embedding_size.1)?
            .unsqueeze(0)
    }

    /// `points` has BxNx2 pixels and `labels` BxN values: -1 padding, 0
    /// negative, 1 positive, 2 top-left box corner, and 3 bottom-right.
    pub fn forward(
        &self,
        points: Option<(&Tensor, &Tensor)>,
        boxes: Option<&Tensor>,
        masks: Option<&Tensor>,
    ) -> Result<PromptEncoderOutput> {
        let batch = match (points, boxes, masks) {
            (Some((coordinates, _)), _, _) => coordinates.dim(0)?,
            (_, Some(boxes), _) => boxes.dim(0)?,
            (_, _, Some(masks)) => masks.dim(0)?,
            _ => 1,
        };
        let mut sparse = Tensor::zeros(
            (batch, 0usize, self.embed_dim),
            DType::F32,
            self.no_mask_embed.device(),
        )?;
        if let Some((coordinates, labels)) = points {
            let embedded = self.embed_points(coordinates, labels, boxes.is_none())?;
            sparse = Tensor::cat(&[&sparse, &embedded], 1)?;
        }
        if let Some(boxes) = boxes {
            sparse = Tensor::cat(&[&sparse, &self.embed_boxes(boxes)?], 1)?;
        }
        let dense_embeddings = match masks {
            Some(mask) => self.embed_masks(mask)?,
            None => self
                .no_mask_embed
                .reshape((1, self.embed_dim, 1, 1))?
                .broadcast_as((
                    batch,
                    self.embed_dim,
                    self.image_embedding_size.0,
                    self.image_embedding_size.1,
                ))?,
        };
        Ok(PromptEncoderOutput {
            sparse_embeddings: sparse,
            dense_embeddings,
        })
    }

    fn embed_points(&self, points: &Tensor, labels: &Tensor, pad: bool) -> Result<Tensor> {
        let mut points = points.affine(1., 0.5)?;
        let mut labels = labels.clone();
        if pad {
            let batch = points.dim(0)?;
            let padding_points =
                Tensor::zeros((batch, 1usize, 2usize), points.dtype(), points.device())?;
            let padding_labels =
                Tensor::full(-1i64, (batch, 1usize), labels.device())?.to_dtype(labels.dtype())?;
            points = Tensor::cat(&[&points, &padding_points], 1)?;
            labels = Tensor::cat(&[&labels, &padding_labels], 1)?;
        }
        let mut embedding = self.pe_layer.forward_with_coords(
            &points,
            self.input_image_size.0,
            self.input_image_size.1,
        )?;
        let shape = embedding.dims().to_vec();
        for (label, replacement) in [
            (-1i64, &self.not_a_point_embed),
            (0, &self.point_embeddings[0]),
            (1, &self.point_embeddings[1]),
            (2, &self.point_embeddings[2]),
            (3, &self.point_embeddings[3]),
        ] {
            let condition = labels
                .eq(label)?
                .unsqueeze(2)?
                .broadcast_as(shape.clone())?;
            let value = if label == -1 {
                replacement.broadcast_as(shape.clone())?
            } else {
                embedding.broadcast_add(&replacement.broadcast_as(shape.clone())?)?
            };
            embedding = condition.where_cond(&value, &embedding)?;
        }
        Ok(embedding)
    }

    fn embed_boxes(&self, boxes: &Tensor) -> Result<Tensor> {
        let batch = boxes.dim(0)?;
        let coords = boxes.affine(1., 0.5)?.reshape((batch, 2usize, 2usize))?;
        let embedding = self.pe_layer.forward_with_coords(
            &coords,
            self.input_image_size.0,
            self.input_image_size.1,
        )?;
        let first = embedding
            .narrow(1, 0, 1)?
            .broadcast_add(&self.point_embeddings[2])?;
        let second = embedding
            .narrow(1, 1, 1)?
            .broadcast_add(&self.point_embeddings[3])?;
        Tensor::cat(&[&first, &second], 1)
    }

    fn embed_masks(&self, masks: &Tensor) -> Result<Tensor> {
        let x = self.mask_conv_0.forward(masks)?;
        let x = self.mask_norm_1.forward(&x)?.gelu_erf()?;
        let x = self.mask_conv_3.forward(&x)?;
        let x = self.mask_norm_4.forward(&x)?.gelu_erf()?;
        self.mask_conv_6.forward(&x)
    }
}
