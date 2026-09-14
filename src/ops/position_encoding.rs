//! Translation of the sine and random prompt position encoders.

use std::f64::consts::PI;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone, Debug)]
pub struct PositionEmbeddingSine {
    num_pos_feats: usize,
    temperature: f64,
    normalize: bool,
    scale: f64,
}

impl PositionEmbeddingSine {
    /// `num_pos_feats` has the same meaning as upstream's constructor argument;
    /// the reference internally halves it for each spatial axis.
    pub fn new(
        num_pos_feats: usize,
        temperature: f64,
        normalize: bool,
        scale: Option<f64>,
    ) -> Self {
        assert_eq!(num_pos_feats % 2, 0);
        assert!(normalize || scale.is_none());
        Self {
            num_pos_feats: num_pos_feats / 2,
            temperature,
            normalize,
            scale: scale.unwrap_or(2.0 * PI),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _, height, width) = x.dims4()?;
        self.pe(batch, height, width, x.device())
    }

    pub fn pe(&self, batch: usize, height: usize, width: usize, device: &Device) -> Result<Tensor> {
        let y = Tensor::arange(1f32, height as f32 + 1., device)?
            .reshape((1, height, 1))?
            .broadcast_as((batch, height, width))?;
        let x = Tensor::arange(1f32, width as f32 + 1., device)?
            .reshape((1, 1, width))?
            .broadcast_as((batch, height, width))?;
        let (x, y) = if self.normalize {
            let y_last = y.narrow(1, height - 1, 1)?;
            let x_last = x.narrow(2, width - 1, 1)?;
            (
                x.broadcast_div(&x_last.affine(1., 1e-6)?)?
                    .affine(self.scale, 0.)?,
                y.broadcast_div(&y_last.affine(1., 1e-6)?)?
                    .affine(self.scale, 0.)?,
            )
        } else {
            (x, y)
        };
        let dim_t = Tensor::from_vec(
            (0..self.num_pos_feats)
                .map(|index| {
                    self.temperature
                        .powf(2. * (index / 2) as f64 / self.num_pos_feats as f64)
                        as f32
                })
                .collect::<Vec<_>>(),
            self.num_pos_feats,
            device,
        )?;
        let pos_x = interleaved_sin_cos(&x.unsqueeze(3)?.broadcast_div(&dim_t)?)?;
        let pos_y = interleaved_sin_cos(&y.unsqueeze(3)?.broadcast_div(&dim_t)?)?;
        Tensor::cat(&[&pos_y, &pos_x], 3)?.permute((0, 3, 1, 2))
    }
}

#[derive(Clone, Debug)]
pub struct PositionEmbeddingRandom {
    positional_encoding_gaussian_matrix: Tensor,
}

impl PositionEmbeddingRandom {
    pub fn load(num_pos_feats: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            positional_encoding_gaussian_matrix: vb
                .get((2, num_pos_feats), "positional_encoding_gaussian_matrix")?,
        })
    }

    fn pe_encoding(&self, coords: &Tensor) -> Result<Tensor> {
        let coords = coords.affine(2., -1.)?;
        let mut output_shape = coords.dims().to_vec();
        let prefix_size: usize = output_shape[..output_shape.len() - 1].iter().product();
        let output_features = self.positional_encoding_gaussian_matrix.dim(1)?;
        let final_dim = output_shape.len() - 1;
        output_shape[final_dim] = output_features;
        let coords = coords
            .reshape((prefix_size, 2usize))?
            .matmul(&self.positional_encoding_gaussian_matrix)?
            .reshape(output_shape)?
            .affine(2. * PI, 0.)?;
        Tensor::cat(&[&coords.sin()?, &coords.cos()?], coords.rank() - 1)
    }

    /// Produces CxHxW dense prompt positional embeddings.
    pub fn forward(&self, height: usize, width: usize) -> Result<Tensor> {
        let device = self.positional_encoding_gaussian_matrix.device();
        let y = Tensor::arange(0.5f32, height as f32, device)?
            .reshape((height, 1))?
            .broadcast_as((height, width))?
            .affine(1. / height as f64, 0.)?;
        let x = Tensor::arange(0.5f32, width as f32, device)?
            .reshape((1, width))?
            .broadcast_as((height, width))?
            .affine(1. / width as f64, 0.)?;
        let coords = Tensor::stack(&[&x, &y], 2)?;
        self.pe_encoding(&coords)?.permute((2, 0, 1))
    }

    /// Produces BxNxC sparse prompt positional embeddings from pixel coordinates.
    pub fn forward_with_coords(
        &self,
        coords: &Tensor,
        image_height: usize,
        image_width: usize,
    ) -> Result<Tensor> {
        let x = coords
            .narrow(2, 0, 1)?
            .affine(1. / image_width as f64, 0.)?;
        let y = coords
            .narrow(2, 1, 1)?
            .affine(1. / image_height as f64, 0.)?;
        self.pe_encoding(&Tensor::cat(&[&x, &y], 2)?.to_dtype(DType::F32)?)
    }
}

fn interleaved_sin_cos(values: &Tensor) -> Result<Tensor> {
    let features = values.dim(values.rank() - 1)?;
    let even = Tensor::from_vec(
        (0..features)
            .step_by(2)
            .map(|index| index as u32)
            .collect::<Vec<_>>(),
        features / 2,
        values.device(),
    )?;
    let odd = Tensor::from_vec(
        (1..features)
            .step_by(2)
            .map(|index| index as u32)
            .collect::<Vec<_>>(),
        features / 2,
        values.device(),
    )?;
    Tensor::stack(
        &[
            &values.index_select(&even, values.rank() - 1)?.sin()?,
            &values.index_select(&odd, values.rank() - 1)?.cos()?,
        ],
        values.rank(),
    )?
    .flatten(values.rank() - 1, values.rank())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Device;
    use candle_nn::VarBuilder;

    use super::*;

    #[test]
    fn random_dense_encoding_matches_python_reference() {
        let device = Device::Cpu;
        let matrix = Tensor::from_vec(vec![0.25f32, -0.5, 0.75, 0.125], (2, 2), &device).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("positional_encoding_gaussian_matrix".to_owned(), matrix);
        let encoder = PositionEmbeddingRandom::load(
            2,
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )
        .unwrap();
        let actual = encoder
            .forward(2, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Generated by the pinned Python reference's PositionEmbeddingRandom.
        let expected = [
            0.25881884,
            -0.70710677,
            -0.9659259,
            0.9659259,
            0.70710677,
            -0.25881884,
            0.9914449,
            -0.38268346,
            -0.60876143,
            0.60876143,
            0.38268346,
            -0.9914449,
            -0.9659259,
            -0.70710677,
            0.25881895,
            0.25881895,
            -0.70710677,
            -0.9659259,
            -0.1305261,
            0.9238795,
            -0.7933533,
            -0.7933533,
            0.9238795,
            -0.1305261,
        ];
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
        }
    }
}
