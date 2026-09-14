//! Translation of `LayerNorm2d` in `sam2_utils.py`.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone, Debug)]
pub struct LayerNorm2d {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm2d {
    pub fn load(num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(num_channels, "weight")?,
            bias: vb.get(num_channels, "bias")?,
            eps,
        })
    }

    /// Normalizes NCHW data across its channel dimension, preserving PyTorch's
    /// mean/subtract/square/mean/sqrt operation order.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, channels, _, _) = x.dims4()?;
        let mean = x.mean_keepdim(1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(1)?;
        let std = variance.affine(1., self.eps)?.sqrt()?;
        let normalized = centered.broadcast_div(&std)?;
        let weight = self.weight.reshape((1, channels, 1, 1))?;
        let bias = self.bias.reshape((1, channels, 1, 1))?;
        normalized.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}
