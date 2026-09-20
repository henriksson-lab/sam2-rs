//! Translation of `sam2/modeling/sam/transformer.py` (non-RoPE decoder path).

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use candle_core::{DType, Result, Storage, Tensor, TensorId, bail};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, linear, ops::softmax_last_dim};
use rayon::prelude::*;

use crate::ops::{Activation, Mlp, apply_rotary, axial_frequencies, layer_norm_add_residual};

fn image_tokens_nhwc(x: &Tensor, contiguous: bool) -> Result<Tensor> {
    if contiguous
        && std::env::var_os("SAM2_CPU_DECODER_DIRECT_PACK").is_none_or(|value| value != "0")
        && let Some(packed) = image_tokens_nhwc_cpu_f32(x)?
    {
        return Ok(packed);
    }
    let (batch, channels, height, width) = x.dims4()?;
    let tokens = x
        .reshape((batch, channels, height * width))?
        .permute((0, 2, 1))?;
    if contiguous {
        tokens.contiguous()
    } else {
        Ok(tokens)
    }
}

fn image_tokens_nhwc_cpu_f32(x: &Tensor) -> Result<Option<Tensor>> {
    const TILE_PIXELS: usize = 32;
    if x.dtype() != DType::F32 {
        return Ok(None);
    }
    let (batch, channels, height, width) = x.dims4()?;
    let pixels = height * width;
    if batch == 0 || channels == 0 || pixels == 0 || pixels % TILE_PIXELS != 0 {
        return Ok(None);
    }
    let (storage, layout) = x.storage_and_layout();
    let (Storage::Cpu(storage), Some((start, end))) = (&*storage, layout.contiguous_offsets())
    else {
        return Ok(None);
    };
    let input = &storage.as_slice::<f32>()?[start..end];
    let mut output = vec![0f32; batch * pixels * channels];
    output
        .par_chunks_mut(TILE_PIXELS * channels)
        .enumerate()
        .for_each(|(tile_index, destination)| {
            let first_pixel = tile_index * TILE_PIXELS;
            let batch_index = first_pixel / pixels;
            let pixel_index = first_pixel % pixels;
            for channel in 0..channels {
                let source = &input[(batch_index * channels + channel) * pixels + pixel_index..]
                    [..TILE_PIXELS];
                for (local_pixel, value) in source.iter().enumerate() {
                    destination[local_pixel * channels + channel] = *value;
                }
            }
        });
    Ok(Some(Tensor::from_vec(
        output,
        (batch, pixels, channels),
        x.device(),
    )?))
}

fn cached_image_pe_tokens(
    cache: &Mutex<Option<(TensorId, Tensor)>>,
    image_pe: &Tensor,
) -> Result<Tensor> {
    let id = image_pe.id();
    {
        let entry = cache
            .lock()
            .map_err(|_| candle_core::Error::Msg("decoder position cache mutex poisoned".into()))?;
        if let Some((cached_id, tokens)) = &*entry {
            if *cached_id == id {
                return Ok(tokens.clone());
            }
        }
    }
    let tokens = image_tokens_nhwc(image_pe, true)?;
    let mut entry = cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("decoder position cache mutex poisoned".into()))?;
    *entry = Some((id, tokens.clone()));
    Ok(tokens)
}

#[derive(Clone, Debug)]
pub struct Attention {
    internal_dim: usize,
    num_heads: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

/// Opt-in timing boundaries for the two-way decoder transformer.
#[derive(Clone, Debug, Default)]
pub struct TwoWayTransformerProfile {
    pub setup_ms: f64,
    pub self_attention_ms: [f64; 2],
    pub token_to_image_attention_ms: [f64; 2],
    pub mlp_ms: [f64; 2],
    pub image_to_token_attention_ms: [f64; 2],
    pub final_attention_ms: f64,
    /// Cross-attentions in execution order: token-to-image (two layers),
    /// image-to-token (two layers), then final token-to-image.
    pub cross_qkv_ms: [f64; 5],
    pub cross_kernel_ms: [f64; 5],
    pub cross_output_projection_ms: [f64; 5],
    pub cross_total_ms: [f64; 5],
    pub cross_pre_ms: [f64; 5],
    pub cross_post_ms: [f64; 5],
}

#[derive(Clone, Copy, Debug, Default)]
struct AttentionPhaseProfile {
    qkv_ms: f64,
    kernel_ms: f64,
    output_projection_ms: f64,
    total_ms: f64,
    pre_ms: f64,
    post_ms: f64,
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
        self.forward_impl(q, k, v, None)
    }

    fn forward_profiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, AttentionPhaseProfile)> {
        let mut profile = AttentionPhaseProfile::default();
        let start = Instant::now();
        let output = self.forward_impl(q, k, v, Some(&mut profile))?;
        profile.total_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok((output, profile))
    }

    fn forward_impl(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mut profile: Option<&mut AttentionPhaseProfile>,
    ) -> Result<Tensor> {
        let q_tokens = q.dim(1)?;
        let k_tokens = k.dim(1)?;
        // The decoder uses unmasked f32 B=1 attention throughout.  Keep Q/K/V
        // in their native NHSD projection layout and let Candle's CPU flash
        // kernel write that same layout.  The former B,H,S,D matmul sequence
        // materialized K^T, the score matrix and two head permutations.  This
        // especially matters for image-to-token attention (4096 queries),
        // where keeping the large projected activations in one layout avoids
        // extra head-layout materializations.
        // Tiny token queries against 4096 image keys also use Candle's
        // asymmetric GEMM path. Its bounded score buffer avoids the repeated
        // head permutations and K transpose of the generic route.
        if q.device().is_cpu()
            && q.dtype() == candle_core::DType::F32
            && q.dim(0)? == 1
            && (q_tokens >= 512
                || ((2..=16).contains(&q_tokens)
                    && (512..=8192).contains(&k_tokens)
                    && std::env::var_os("SAM2_CPU_DECODER_SHORT_FLASH")
                        .is_none_or(|value| value != "0")))
            && std::env::var_os("SAM2_CPU_DECODER_FLASH").is_none_or(|value| value != "0")
        {
            let phase_start = profile.is_some().then(Instant::now);
            let head_dim = self.internal_dim / self.num_heads;
            let q = self.q_proj.forward_fused_bias_cpu(q)?.reshape((
                1,
                q.dim(1)?,
                self.num_heads,
                head_dim,
            ))?;
            let k = self.k_proj.forward_fused_bias_cpu(k)?.reshape((
                1,
                k.dim(1)?,
                self.num_heads,
                head_dim,
            ))?;
            let v = self.v_proj.forward_fused_bias_cpu(v)?.reshape((
                1,
                v.dim(1)?,
                self.num_heads,
                head_dim,
            ))?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                profile.qkv_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            let phase_start = profile.is_some().then(Instant::now);
            let attended = candle_nn::attention::flash_attn_nhsd_f32(
                &q,
                &k,
                &v,
                1. / (head_dim as f32).sqrt(),
            )?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                profile.kernel_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            let phase_start = profile.is_some().then(Instant::now);
            let output = self
                .out_proj
                .forward_fused_bias_cpu(&attended.flatten_from(2)?)?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                profile.output_projection_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            return Ok(output);
        }
        let phase_start = profile.is_some().then(Instant::now);
        let q = self.separate_heads(&self.q_proj.forward_fused_bias_cpu(q)?)?;
        let k = self.separate_heads(&self.k_proj.forward_fused_bias_cpu(k)?)?;
        let v = self.separate_heads(&self.v_proj.forward_fused_bias_cpu(v)?)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
            profile.qkv_ms = start.elapsed().as_secs_f64() * 1000.;
        }
        let phase_start = profile.is_some().then(Instant::now);
        let dim = self.internal_dim / self.num_heads;
        let attention = softmax_last_dim(
            &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(1. / (dim as f64).sqrt(), 0.)?,
        )?;
        let attended = self.recombine_heads(&attention.matmul(&v)?)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
            profile.kernel_ms = start.elapsed().as_secs_f64() * 1000.;
        }
        let phase_start = profile.is_some().then(Instant::now);
        let output = self.out_proj.forward_fused_bias_cpu(&attended)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
            profile.output_projection_ms = start.elapsed().as_secs_f64() * 1000.;
        }
        Ok(output)
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
        let q = self.separate_heads(&self.q_proj.forward_fused_bias_cpu(q)?)?;
        let mut k = self.separate_heads(&self.k_proj.forward_fused_bias_cpu(k)?)?;
        let v = self.separate_heads(&self.v_proj.forward_fused_bias_cpu(v)?)?;
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
        let attention = softmax_last_dim(
            &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                .affine(1. / (dim as f64).sqrt(), 0.)?,
        )?;
        self.out_proj
            .forward_fused_bias_cpu(&self.recombine_heads(&attention.matmul(&v)?)?)
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
        let image_with_pe = keys.broadcast_add(key_pe)?;
        queries = self.norm2.forward(
            &queries.broadcast_add(&self.cross_attn_token_to_image.forward(
                &q,
                &image_with_pe,
                keys,
            )?)?,
        )?;
        queries = self
            .norm3
            .forward(&queries.broadcast_add(&self.mlp.forward_decoder_fused_bias(&queries)?)?)?;
        let q = queries.broadcast_add(query_pe)?;
        let k = if std::env::var_os("SAM2_CPU_DECODER_REUSE_KEY_PE")
            .is_some_and(|value| value == "0")
        {
            keys.broadcast_add(key_pe)?
        } else {
            image_with_pe
        };
        let keys = layer_norm_add_residual(
            &self.norm4,
            &keys,
            &self.cross_attn_image_to_token.forward(&k, &q, &queries)?,
        )?;
        Ok((queries, keys))
    }

    fn forward_profiled(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> Result<(
        (Tensor, Tensor),
        (f64, f64, f64, f64),
        AttentionPhaseProfile,
        AttentionPhaseProfile,
    )> {
        let start = Instant::now();
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = queries.broadcast_add(query_pe)?;
            queries.broadcast_add(&self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = self.norm1.forward(&queries)?;
        let self_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let q = queries.broadcast_add(query_pe)?;
        let image_with_pe = keys.broadcast_add(key_pe)?;
        let pre_ms = start.elapsed().as_secs_f64() * 1000.;
        let (token_to_image, mut token_to_image_profile) = self
            .cross_attn_token_to_image
            .forward_profiled(&q, &image_with_pe, keys)?;
        token_to_image_profile.pre_ms = pre_ms;
        let post_start = Instant::now();
        queries = self
            .norm2
            .forward(&queries.broadcast_add(&token_to_image)?)?;
        token_to_image_profile.post_ms = post_start.elapsed().as_secs_f64() * 1000.;
        let token_to_image_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        queries = self
            .norm3
            .forward(&queries.broadcast_add(&self.mlp.forward_decoder_fused_bias(&queries)?)?)?;
        let mlp_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let q = queries.broadcast_add(query_pe)?;
        let k = if std::env::var_os("SAM2_CPU_DECODER_REUSE_KEY_PE")
            .is_some_and(|value| value == "0")
        {
            keys.broadcast_add(key_pe)?
        } else {
            image_with_pe
        };
        let pre_ms = start.elapsed().as_secs_f64() * 1000.;
        let (image_to_token, mut image_to_token_profile) = self
            .cross_attn_image_to_token
            .forward_profiled(&k, &q, &queries)?;
        image_to_token_profile.pre_ms = pre_ms;
        let post_start = Instant::now();
        let keys = layer_norm_add_residual(&self.norm4, &keys, &image_to_token)?;
        image_to_token_profile.post_ms = post_start.elapsed().as_secs_f64() * 1000.;
        Ok((
            (queries, keys),
            (
                self_ms,
                token_to_image_ms,
                mlp_ms,
                start.elapsed().as_secs_f64() * 1000.,
            ),
            token_to_image_profile,
            image_to_token_profile,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct TwoWayTransformer {
    layers: [TwoWayAttentionBlock; 2],
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNorm,
    image_pe_token_cache: Arc<Mutex<Option<(TensorId, Tensor)>>>,
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
            image_pe_token_cache: Arc::new(Mutex::new(None)),
        })
    }

    fn image_pe_tokens(&self, image_pe: &Tensor) -> Result<Tensor> {
        let pack = std::env::var_os("SAM2_CPU_DECODER_PACK_IMAGE_PE")
            .is_none_or(|value| value != "0")
            && image_pe.device().is_cpu();
        if !pack {
            return image_tokens_nhwc(image_pe, false);
        }
        let use_cache = !image_pe.is_variable()
            && std::env::var_os("SAM2_CPU_DECODER_CACHE_PACKED_PE")
                .is_none_or(|value| value != "0");
        if !use_cache {
            return image_tokens_nhwc(image_pe, true);
        }
        cached_image_pe_tokens(&self.image_pe_token_cache, image_pe)
    }

    /// Inputs follow the reference: NCHW image embedding/PE and BxNxC points.
    pub fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Both tensors are reused throughout the decoder. Prepare one
        // contiguous token layout each, without per-worker input copies.
        let mut keys = image_tokens_nhwc(image_embedding, true)?;
        // The same positional encoding is added to image tokens five times.
        // One contiguous preparation avoids five slow strided additions and
        // is reused across the two transformer layers and final attention.
        let image_pe = self.image_pe_tokens(image_pe)?;
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

    pub fn forward_profiled(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<((Tensor, Tensor), TwoWayTransformerProfile)> {
        let setup_start = Instant::now();
        let mut keys = image_tokens_nhwc(image_embedding, true)?;
        let image_pe = self.image_pe_tokens(image_pe)?;
        let mut queries = point_embedding.clone();
        let mut profile = TwoWayTransformerProfile::default();
        profile.setup_ms = setup_start.elapsed().as_secs_f64() * 1000.;
        for (index, layer) in self.layers.iter().enumerate() {
            let (
                (next_queries, next_keys),
                (self_ms, token_to_image_ms, mlp_ms, image_to_token_ms),
                token_to_image_profile,
                image_to_token_profile,
            ) = layer.forward_profiled(&queries, &keys, point_embedding, &image_pe)?;
            queries = next_queries;
            keys = next_keys;
            profile.self_attention_ms[index] = self_ms;
            profile.token_to_image_attention_ms[index] = token_to_image_ms;
            profile.mlp_ms[index] = mlp_ms;
            profile.image_to_token_attention_ms[index] = image_to_token_ms;
            for (slot, phase) in [
                (index, token_to_image_profile),
                (index + 2, image_to_token_profile),
            ] {
                profile.cross_qkv_ms[slot] = phase.qkv_ms;
                profile.cross_kernel_ms[slot] = phase.kernel_ms;
                profile.cross_output_projection_ms[slot] = phase.output_projection_ms;
                profile.cross_total_ms[slot] = phase.total_ms;
                profile.cross_pre_ms[slot] = phase.pre_ms;
                profile.cross_post_ms[slot] = phase.post_ms;
            }
        }
        let start = Instant::now();
        let q = queries.broadcast_add(point_embedding)?;
        let k = keys.broadcast_add(&image_pe)?;
        let pre_ms = start.elapsed().as_secs_f64() * 1000.;
        let (final_attention, mut final_profile) = self
            .final_attn_token_to_image
            .forward_profiled(&q, &k, &keys)?;
        final_profile.pre_ms = pre_ms;
        let post_start = Instant::now();
        queries = self
            .norm_final_attn
            .forward(&queries.broadcast_add(&final_attention)?)?;
        final_profile.post_ms = post_start.elapsed().as_secs_f64() * 1000.;
        profile.final_attention_ms = start.elapsed().as_secs_f64() * 1000.;
        profile.cross_qkv_ms[4] = final_profile.qkv_ms;
        profile.cross_kernel_ms[4] = final_profile.kernel_ms;
        profile.cross_output_projection_ms[4] = final_profile.output_projection_ms;
        profile.cross_total_ms[4] = final_profile.total_ms;
        profile.cross_pre_ms[4] = final_profile.pre_ms;
        profile.cross_post_ms[4] = final_profile.post_ms;
        Ok(((queries, keys), profile))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use candle_core::{Device, Result, Tensor};

    use super::{cached_image_pe_tokens, image_tokens_nhwc_cpu_f32};

    #[test]
    fn direct_image_token_pack_matches_generic_transpose() -> Result<()> {
        for (batch, channels, height, width) in [(2, 3, 8, 8), (1, 5, 4, 16)] {
            let x = Tensor::from_vec(
                (0..batch * channels * height * width)
                    .map(|index| index as f32 * 0.125 - 3.)
                    .collect(),
                (batch, channels, height, width),
                &Device::Cpu,
            )?;
            let expected = x
                .reshape((batch, channels, height * width))?
                .permute((0, 2, 1))?
                .contiguous()?;
            let actual = image_tokens_nhwc_cpu_f32(&x)?.expect("eligible CPU f32 input");
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn packed_position_cache_respects_tensor_identity() -> Result<()> {
        let cache = Mutex::new(None);
        let first = Tensor::from_vec(vec![1f32; 2 * 8 * 8], (1, 2, 8, 8), &Device::Cpu)?;
        let second = Tensor::from_vec(vec![2f32; 2 * 8 * 8], (1, 2, 8, 8), &Device::Cpu)?;
        let first_tokens = cached_image_pe_tokens(&cache, &first)?;
        assert_eq!(
            first_tokens.id(),
            cached_image_pe_tokens(&cache, &first)?.id()
        );
        let second_tokens = cached_image_pe_tokens(&cache, &second)?;
        assert_ne!(first_tokens.id(), second_tokens.id());
        assert_eq!(
            second_tokens.flatten_all()?.to_vec1::<f32>()?,
            vec![2f32; 128]
        );
        Ok(())
    }
}
