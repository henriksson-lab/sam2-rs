//! Translation of `MLP` and activation selection in `sam2_utils.py`.

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
}
