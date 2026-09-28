//! Four-layer temporal attention, translated from `memory_attention.py`.

use std::time::Instant;

use crate::RopeAttention;
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder, linear};

#[derive(Clone, Debug)]
struct Layer {
    self_attn: RopeAttention,
    cross: RopeAttention,
    l1: candle_nn::Linear,
    l2: candle_nn::Linear,
    n1: LayerNorm,
    n2: LayerNorm,
    n3: LayerNorm,
}
impl Layer {
    fn load(index: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp(format!("layers.{index}"));
        Ok(Self {
            self_attn: RopeAttention::load(256, 1, 1, None, false, vb.pp("self_attn"))?,
            cross: RopeAttention::load(256, 1, 1, Some(64), true, vb.pp("cross_attn_image"))?,
            l1: linear(256, 2048, vb.pp("linear1"))?,
            l2: linear(2048, 256, vb.pp("linear2"))?,
            n1: candle_nn::layer_norm(256, 1e-5, vb.pp("norm1"))?,
            n2: candle_nn::layer_norm(256, 1e-5, vb.pp("norm2"))?,
            n3: candle_nn::layer_norm(256, 1e-5, vb.pp("norm3"))?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        memory: &Tensor,
        _qp: &Tensor,
        mp: &Tensor,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor> {
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
                        "SAM2_MEMORY_ATTENTION_LAYER_STAGE stage={} ms={:.6}",
                        $name,
                        profile_start.elapsed().as_secs_f64() * 1000.
                    );
                    profile_start = Instant::now();
                }
            };
        }
        let n = self.n1.forward(x)?;
        profile_stage!("norm_1");
        // SAM2.1 configures `pos_enc_at_attn=false`: only the 0.1 input
        // positional injection in `MemoryAttention::forward` is used here.
        let mut x = x.broadcast_add(&self.self_attn.forward(&n, &n, &n, 0)?)?;
        profile_stage!("self_attention_and_residual");
        let n = self.n2.forward(&x)?;
        profile_stage!("norm_2");
        let memory_with_pos = memory.broadcast_add(mp)?;
        profile_stage!("memory_position_add");
        x = x.broadcast_add(&self.cross.forward(
            &n,
            &memory_with_pos,
            memory,
            num_obj_ptr_tokens,
        )?)?;
        profile_stage!("cross_attention_and_residual");
        let n = self.n3.forward(&x)?;
        profile_stage!("norm_3");
        if x.device().is_cpu()
            && std::env::var_os("SAM2_CPU_MEMORY_MLP_FUSION").is_some_and(|value| value == "1")
        {
            let hidden = self.l1.forward_fused_bias_relu_cpu(&n)?;
            profile_stage!("mlp_1_relu");
            let output = self.l2.forward_add_residual(&hidden, &x)?;
            if profiling {
                output.device().synchronize()?;
                eprintln!(
                    "SAM2_MEMORY_ATTENTION_LAYER_STAGE stage=mlp_2_and_residual ms={:.6}",
                    profile_start.elapsed().as_secs_f64() * 1000.
                );
            }
            return Ok(output);
        }
        let hidden = self.l1.forward(&n)?;
        profile_stage!("mlp_1");
        let hidden = hidden.relu()?;
        profile_stage!("relu");
        let hidden = self.l2.forward(&hidden)?;
        profile_stage!("mlp_2");
        let output = x.broadcast_add(&hidden)?;
        if profiling {
            output.device().synchronize()?;
            eprintln!(
                "SAM2_MEMORY_ATTENTION_LAYER_STAGE stage=mlp_residual ms={:.6}",
                profile_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(output)
    }
}
#[derive(Clone, Debug)]
pub struct MemoryAttention {
    layers: [Layer; 4],
    norm: LayerNorm,
}
impl MemoryAttention {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layers: [
                Layer::load(0, vb.clone())?,
                Layer::load(1, vb.clone())?,
                Layer::load(2, vb.clone())?,
                Layer::load(3, vb.clone())?,
            ],
            norm: candle_nn::layer_norm(256, 1e-5, vb.pp("norm"))?,
        })
    }
    /// Inputs and output are sequence-first `(tokens,batch,channels)`.
    pub fn forward(
        &self,
        curr: &Tensor,
        memory: &Tensor,
        curr_pos: &Tensor,
        memory_pos: &Tensor,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor> {
        let profiling = std::env::var_os("SAM2_PROFILE_VIDEO_STAGES").is_some();
        if profiling {
            curr.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
        let mut x = curr
            .permute((1, 0, 2))?
            .broadcast_add(&curr_pos.permute((1, 0, 2))?.affine(0.1, 0.)?)?;
        let m = memory.permute((1, 0, 2))?;
        let qp = curr_pos.permute((1, 0, 2))?;
        let mp = memory_pos.permute((1, 0, 2))?;
        if profiling {
            curr.device().synchronize()?;
            eprintln!(
                "SAM2_MEMORY_ATTENTION_STAGE stage=setup ms={:.6}",
                profile_start.elapsed().as_secs_f64() * 1000.
            );
            profile_start = Instant::now();
        }
        for (index, l) in self.layers.iter().enumerate() {
            x = l.forward(&x, &m, &qp, &mp, num_obj_ptr_tokens)?;
            if profiling {
                curr.device().synchronize()?;
                eprintln!(
                    "SAM2_MEMORY_ATTENTION_STAGE stage=layer_{} ms={:.6}",
                    index,
                    profile_start.elapsed().as_secs_f64() * 1000.
                );
                profile_start = Instant::now();
            }
        }
        let output = self.norm.forward(&x)?.permute((1, 0, 2))?;
        if profiling {
            curr.device().synchronize()?;
            eprintln!(
                "SAM2_MEMORY_ATTENTION_STAGE stage=final_norm ms={:.6}",
                profile_start.elapsed().as_secs_f64() * 1000.
            );
        }
        Ok(output)
    }
}
