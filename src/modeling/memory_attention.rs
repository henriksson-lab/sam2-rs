//! Four-layer temporal attention, translated from `memory_attention.py`.

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
        let n = self.n1.forward(x)?;
        // SAM2.1 configures `pos_enc_at_attn=false`: only the 0.1 input
        // positional injection in `MemoryAttention::forward` is used here.
        let mut x = x.broadcast_add(&self.self_attn.forward(&n, &n, &n, 0)?)?;
        let n = self.n2.forward(&x)?;
        x = x.broadcast_add(&self.cross.forward(
            &n,
            &memory.broadcast_add(mp)?,
            memory,
            num_obj_ptr_tokens,
        )?)?;
        let n = self.n3.forward(&x)?;
        x.broadcast_add(&self.l2.forward(&self.l1.forward(&n)?.relu()?)?)
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
        let mut x = curr
            .permute((1, 0, 2))?
            .broadcast_add(&curr_pos.permute((1, 0, 2))?.affine(0.1, 0.)?)?;
        let m = memory.permute((1, 0, 2))?;
        let qp = curr_pos.permute((1, 0, 2))?;
        let mp = memory_pos.permute((1, 0, 2))?;
        for l in &self.layers {
            x = l.forward(&x, &m, &qp, &mp, num_obj_ptr_tokens)?;
        }
        self.norm.forward(&x)?.permute((1, 0, 2))
    }
}
