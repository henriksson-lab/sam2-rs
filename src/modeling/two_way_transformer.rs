//! Translation of `sam2/modeling/sam/transformer.py` (non-RoPE decoder path).

use candle_core::{Result, Tensor, bail};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, linear, ops::softmax};

use crate::ops::{Activation, Mlp, apply_rotary, axial_frequencies};

#[derive(Clone, Debug)]
pub struct Attention {
    internal_dim: usize,
    num_heads: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl Attention {
    pub fn load(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        kv_in_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let internal_dim = embedding_dim / downsample_rate;
        assert_eq!(internal_dim % num_heads, 0);
        let kv_dim = kv_in_dim.unwrap_or(embedding_dim);
        Ok(Self {
            internal_dim,
            num_heads,
            q_proj: linear(embedding_dim, internal_dim, vb.pp("q_proj"))?,
            k_proj: linear(kv_dim, internal_dim, vb.pp("k_proj"))?,
            v_proj: linear(kv_dim, internal_dim, vb.pp("v_proj"))?,
            out_proj: linear(internal_dim, embedding_dim, vb.pp("out_proj"))?,
        })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.separate_heads(&self.q_proj.forward(q)?)?;
        let k = self.separate_heads(&self.k_proj.forward(k)?)?;
        let v = self.separate_heads(&self.v_proj.forward(v)?)?;
        let dim = self.internal_dim / self.num_heads;
        let attention = softmax(
            &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(1. / (dim as f64).sqrt(), 0.)?,
            3,
        )?;
        self.out_proj
            .forward(&self.recombine_heads(&attention.matmul(&v)?)?)
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = x.dims3()?;
        x.reshape((
            batch,
            tokens,
            self.num_heads,
            self.internal_dim / self.num_heads,
        ))?
        .permute((0, 2, 1, 3))?
        .contiguous()
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _, tokens, _) = x.dims4()?;
        x.permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, tokens, self.internal_dim))
    }
}

/// SAM2's `RoPEAttention`.  It shares the checkpoint layout of [`Attention`]
/// but rotates projected query/key heads before scaled dot-product attention.
#[derive(Clone, Debug)]
pub struct RopeAttention {
    internal_dim: usize,
    num_heads: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    rope_theta: f64,
    rope_k_repeat: bool,
}

impl RopeAttention {
    pub fn load(
        embedding_dim: usize,
        num_heads: usize,
        downsample_rate: usize,
        kv_in_dim: Option<usize>,
        rope_k_repeat: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let internal_dim = embedding_dim / downsample_rate;
        assert_eq!(internal_dim % num_heads, 0);
        let kv_dim = kv_in_dim.unwrap_or(embedding_dim);
        Ok(Self {
            internal_dim,
            num_heads,
            q_proj: linear(embedding_dim, internal_dim, vb.pp("q_proj"))?,
            k_proj: linear(kv_dim, internal_dim, vb.pp("k_proj"))?,
            v_proj: linear(kv_dim, internal_dim, vb.pp("v_proj"))?,
            out_proj: linear(internal_dim, embedding_dim, vb.pp("out_proj"))?,
            rope_theta: 10_000.,
            rope_k_repeat,
        })
    }

    /// `q`, `k`, and `v` are batch-first. The final `num_k_exclude_rope`
    /// tokens are object-pointer keys and deliberately retain no spatial RoPE.
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_k_exclude_rope: usize,
    ) -> Result<Tensor> {
        let q = self.separate_heads(&self.q_proj.forward(q)?)?;
        let mut k = self.separate_heads(&self.k_proj.forward(k)?)?;
        let v = self.separate_heads(&self.v_proj.forward(v)?)?;
        let q_tokens = q.dim(2)?;
        let side = (q_tokens as f64).sqrt() as usize;
        if side * side != q_tokens {
            bail!("SAM2 RoPE queries must form a square grid, got {q_tokens} tokens")
        }
        let k_tokens = k.dim(2)?;
        if num_k_exclude_rope > k_tokens {
            bail!("cannot exclude {num_k_exclude_rope} RoPE keys from {k_tokens} keys")
        }
        let rope_keys = k_tokens - num_k_exclude_rope;
        if rope_keys % q_tokens != 0 {
            bail!("SAM2 RoPE key count {rope_keys} is not a whole number of {q_tokens}-token grids")
        }
        let (cos, sin) = axial_frequencies(
            self.internal_dim / self.num_heads,
            side,
            side,
            self.rope_theta,
            q.device(),
        )?;
        let q = apply_rotary(&q, &cos, &sin)?;
        let repeats = rope_keys / q_tokens;
        if repeats != 1 && !self.rope_k_repeat {
            bail!("RoPE cross-attention received repeated memory keys without rope_k_repeat")
        }
        let key_cos = cos.repeat((repeats, 1))?;
        let key_sin = sin.repeat((repeats, 1))?;
        let rotated = apply_rotary(&k.narrow(2, 0, rope_keys)?, &key_cos, &key_sin)?;
        k = if num_k_exclude_rope == 0 {
            rotated
        } else {
            Tensor::cat(&[&rotated, &k.narrow(2, rope_keys, num_k_exclude_rope)?], 2)?
        };
        let dim = self.internal_dim / self.num_heads;
        let attention = softmax(
            &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(1. / (dim as f64).sqrt(), 0.)?,
            3,
        )?;
        self.out_proj
            .forward(&self.recombine_heads(&attention.matmul(&v)?)?)
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = x.dims3()?;
        x.reshape((
            batch,
            tokens,
            self.num_heads,
            self.internal_dim / self.num_heads,
        ))?
        .permute((0, 2, 1, 3))?
        .contiguous()
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, _, tokens, _) = x.dims4()?;
        x.permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, tokens, self.internal_dim))
    }
}

#[derive(Clone, Debug)]
struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: LayerNorm,
    cross_attn_token_to_image: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    norm3: LayerNorm,
    norm4: LayerNorm,
    cross_attn_image_to_token: Attention,
    skip_first_layer_pe: bool,
}

impl TwoWayAttentionBlock {
    fn load(index: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp(format!("layers.{index}"));
        Ok(Self {
            self_attn: Attention::load(256, 8, 1, None, vb.pp("self_attn"))?,
            norm1: candle_nn::layer_norm(256, 1e-5, vb.pp("norm1"))?,
            cross_attn_token_to_image: Attention::load(
                256,
                8,
                2,
                None,
                vb.pp("cross_attn_token_to_image"),
            )?,
            norm2: candle_nn::layer_norm(256, 1e-5, vb.pp("norm2"))?,
            mlp: Mlp::load(256, 2048, 256, 2, Activation::Relu, false, vb.pp("mlp"))?,
            norm3: candle_nn::layer_norm(256, 1e-5, vb.pp("norm3"))?,
            norm4: candle_nn::layer_norm(256, 1e-5, vb.pp("norm4"))?,
            cross_attn_image_to_token: Attention::load(
                256,
                8,
                2,
                None,
                vb.pp("cross_attn_image_to_token"),
            )?,
            skip_first_layer_pe: index == 0,
        })
    }

    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = queries.broadcast_add(query_pe)?;
            queries.broadcast_add(&self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = self.norm1.forward(&queries)?;
        let q = queries.broadcast_add(query_pe)?;
        let k = keys.broadcast_add(key_pe)?;
        queries = self.norm2.forward(
            &queries.broadcast_add(&self.cross_attn_token_to_image.forward(&q, &k, keys)?)?,
        )?;
        queries = self
            .norm3
            .forward(&queries.broadcast_add(&self.mlp.forward(&queries)?)?)?;
        let q = queries.broadcast_add(query_pe)?;
        let k = keys.broadcast_add(key_pe)?;
        let keys = self.norm4.forward(
            &keys.broadcast_add(&self.cross_attn_image_to_token.forward(&k, &q, &queries)?)?,
        )?;
        Ok((queries, keys))
    }
}

#[derive(Clone, Debug)]
pub struct TwoWayTransformer {
    layers: [TwoWayAttentionBlock; 2],
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNorm,
}

impl TwoWayTransformer {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layers: [
                TwoWayAttentionBlock::load(0, vb.clone())?,
                TwoWayAttentionBlock::load(1, vb.clone())?,
            ],
            final_attn_token_to_image: Attention::load(
                256,
                8,
                2,
                None,
                vb.pp("final_attn_token_to_image"),
            )?,
            norm_final_attn: candle_nn::layer_norm(256, 1e-5, vb.pp("norm_final_attn"))?,
        })
    }

    /// Inputs follow the reference: NCHW image embedding/PE and BxNxC points.
    pub fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, channels, height, width) = image_embedding.dims4()?;
        let mut keys = image_embedding
            .reshape((batch, channels, height * width))?
            .permute((0, 2, 1))?;
        let image_pe = image_pe
            .reshape((image_pe.dim(0)?, channels, height * width))?
            .permute((0, 2, 1))?;
        let mut queries = point_embedding.clone();
        for layer in &self.layers {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, &image_pe)?;
        }
        let q = queries.broadcast_add(point_embedding)?;
        let k = keys.broadcast_add(&image_pe)?;
        queries = self.norm_final_attn.forward(
            &queries.broadcast_add(&self.final_attn_token_to_image.forward(&q, &k, &keys)?)?,
        )?;
        Ok((queries, keys))
    }
}
