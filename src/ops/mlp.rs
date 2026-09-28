//! Translation of `MLP` and activation selection in `sam2_utils.py`.

use std::time::Instant;

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, ops::sigmoid};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Gelu,
}

impl Activation {
    pub fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Relu => x.relu(),
            // PyTorch's default is the exact (not tanh approximation) GELU.
            Self::Gelu => x.gelu_erf(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
    sigmoid_output: bool,
}

/// Synchronized phase timings for the two-layer Hiera MLP path.
#[derive(Clone, Debug)]
pub struct MlpProfile {
    /// Includes exact GELU when the fused CPU path is selected.
    pub expansion_ms: f64,
    /// Zero when exact GELU is fused into the expansion output.
    pub activation_ms: f64,
    pub projection_ms: f64,
}

impl Mlp {
    pub fn load(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        activation: Activation,
        sigmoid_output: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            let in_dim = if index == 0 { input_dim } else { hidden_dim };
            let out_dim = if index + 1 == num_layers {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(linear(in_dim, out_dim, vb.pp(format!("layers.{index}")))?);
        }
        Ok(Self {
            layers,
            activation,
            sigmoid_output,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if index + 1 != self.layers.len() {
                x = self.activation.apply(&x)?;
            }
        }
        if self.sigmoid_output {
            sigmoid(&x)
        } else {
            Ok(x)
        }
    }

    /// Use Candle's fused CPU bias destination for decoder Linear layers,
    /// retaining the ordinary implementation as a measured fallback.
    pub fn forward_decoder_fused_bias(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var_os("SAM2_CPU_DECODER_MLP_FUSED_BIAS").is_some_and(|value| value == "0") {
            return self.forward(x);
        }
        let mut x = x.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward_fused_bias_cpu(&x)?;
            if index + 1 != self.layers.len() {
                x = self.activation.apply(&x)?;
            }
        }
        if self.sigmoid_output {
            sigmoid(&x)
        } else {
            Ok(x)
        }
    }

    /// Applies the MLP and adds a same-shaped residual.
    ///
    /// This exposes the final projection/residual data flow to Candle, which
    /// can fuse it for a narrow CPU-f32 path while preserving the usual Tensor
    /// composition as the fallback.
    pub fn forward_add_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if self.sigmoid_output {
            return residual.broadcast_add(&self.forward(x)?);
        }
        let mut x = x.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            if index + 1 == self.layers.len() {
                return layer.forward_add_residual(&x, residual);
            }
            // This method is used by Hiera's residual MLP blocks. On a
            // matching CPU-f32 layout Candle fuses the expansion bias; all
            // other layouts retain the usual linear implementation.
            x = if index == 0 && self.activation == Activation::Gelu {
                layer.forward_fused_bias_gelu_erf_cpu(&x)?
            } else if index == 0 {
                self.activation.apply(&layer.forward_fused_bias_cpu(&x)?)?
            } else {
                self.activation.apply(&layer.forward(&x)?)?
            };
        }
        unreachable!("MLP has at least one layer")
    }

    /// Profile the Hiera two-layer residual MLP without changing its math.
    /// This deliberately has a narrow contract: the benchmarked model has an
    /// expansion, one activation, and a residual final projection.
    pub fn forward_add_residual_profiled(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<(Tensor, MlpProfile)> {
        if self.sigmoid_output || self.layers.len() != 2 {
            let start = Instant::now();
            let output = self.forward_add_residual(x, residual)?;
            output.device().synchronize()?;
            return Ok((
                output,
                MlpProfile {
                    expansion_ms: 0.,
                    activation_ms: 0.,
                    projection_ms: start.elapsed().as_secs_f64() * 1000.,
                },
            ));
        }
        // The specialized implementation fuses exact GELU only on CPU. On
        // CUDA it deliberately falls back to fused-bias Linear followed by a
        // separate GELU kernel, so keep those phases separate in diagnostics.
        let fuse_gelu = x.device().is_cpu()
            && self.activation == Activation::Gelu
            && std::env::var_os("CANDLE_CPU_LINEAR_GELU_FUSION").is_none_or(|value| value != "0");
        let start = Instant::now();
        let expanded = if fuse_gelu {
            self.layers[0].forward_fused_bias_gelu_erf_cpu(x)?
        } else {
            self.layers[0].forward_fused_bias_cpu(x)?
        };
        expanded.device().synchronize()?;
        let expansion_ms = start.elapsed().as_secs_f64() * 1000.;
        let (activated, activation_ms) = if fuse_gelu {
            (expanded, 0.)
        } else {
            let start = Instant::now();
            let activated = self.activation.apply(&expanded)?;
            activated.device().synchronize()?;
            (activated, start.elapsed().as_secs_f64() * 1000.)
        };
        let start = Instant::now();
        let output = self.layers[1].forward_add_residual(&activated, residual)?;
        output.device().synchronize()?;
        Ok((
            output,
            MlpProfile {
                expansion_ms,
                activation_ms,
                projection_ms: start.elapsed().as_secs_f64() * 1000.,
            },
        ))
    }
}
