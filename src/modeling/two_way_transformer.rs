//! Translation of `sam2/modeling/sam/transformer.py` (non-RoPE decoder path).

use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use candle_core::{D, DType, Result, Storage, Tensor, TensorId, bail};
use candle_nn::{
    LayerNorm, Linear, Module, VarBuilder, linear,
    ops::{softmax, softmax_last_dim},
};
use rayon::prelude::*;

use crate::ops::{
    Activation, Mlp, apply_rotary, apply_rotary_repeated, axial_frequencies,
    layer_norm_add_residual,
};

fn layer_norm_differentiable(norm: &LayerNorm, input: &Tensor) -> Result<Tensor> {
    let hidden = input.dim(D::Minus1)?;
    let mean = (input.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let centered = input.broadcast_sub(&mean)?;
    let variance = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let normalized = centered.broadcast_div(&(variance + norm.eps())?.sqrt()?)?;
    let scaled = normalized.broadcast_mul(norm.weight())?;
    match norm.bias() {
        Some(bias) => scaled.broadcast_add(bias),
        None => Ok(scaled),
    }
}

fn profile_a2_transformer_token(stage: &str, queries: &Tensor) -> Result<()> {
    let values = queries
        .narrow(0, 18, 1)?
        // Token 4 is internal mask token 2: object, IoU, then mask 0..3.
        .narrow(1, 4, 1)?
        .flatten_all()?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec1::<f32>()?;
    eprintln!("SAM2_A2_TRANSFORMER_RS stage={stage} values={values:?}");
    Ok(())
}

fn profile_decoder_parity_vector(
    stage: &str,
    values: &Tensor,
    batch: usize,
    token: usize,
) -> Result<()> {
    let values = values
        .narrow(0, batch, 1)?
        .narrow(1, token, 1)?
        .flatten_all()?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec1::<f32>()?;
    eprintln!("SAM2_DECODER_PARITY_RS stage={stage} values={values:?}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn profile_a2_final_attention_vector(stage: &str, value: &Tensor) -> Result<()> {
    let values = value
        .narrow(0, 18, 1)?
        // Token 4 is internal mask token 2: object, IoU, then mask 0..3.
        .narrow(1, 4, 1)?
        .flatten_all()?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec1::<f32>()?;
    eprintln!("SAM2_A2_FINAL_ATTENTION_RS stage={stage} values={values:?}");
    Ok(())
}

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

fn image_tokens_add_nhwc_cpu_f32(image: &Tensor, dense: &Tensor) -> Result<Option<Tensor>> {
    const TILE_PIXELS: usize = 32;
    if image.dtype() != DType::F32
        || dense.dtype() != DType::F32
        || !image.device().is_cpu()
        || image.dims() != dense.dims()
    {
        return Ok(None);
    }
    let (batch, channels, height, width) = image.dims4()?;
    let pixels = height * width;
    if batch != 1 || channels == 0 || pixels == 0 || pixels % TILE_PIXELS != 0 {
        return Ok(None);
    }
    let (image_storage, image_layout) = image.storage_and_layout();
    let (dense_storage, dense_layout) = dense.storage_and_layout();
    let (Storage::Cpu(image_storage), Storage::Cpu(dense_storage)) =
        (&*image_storage, &*dense_storage)
    else {
        return Ok(None);
    };
    let Some((image_start, image_end)) = image_layout.contiguous_offsets() else {
        return Ok(None);
    };
    let dense_stride = dense_layout.stride();
    if dense_stride.len() != 4 || dense_stride[2] != 0 || dense_stride[3] != 0 {
        return Ok(None);
    }
    let image_data = &image_storage.as_slice::<f32>()?[image_start..image_end];
    let dense = dense_storage.as_slice::<f32>()?;
    let dense_start = dense_layout.start_offset();
    let mut output = vec![0f32; batch * pixels * channels];
    output
        .par_chunks_mut(TILE_PIXELS * channels)
        .enumerate()
        .for_each(|(tile_index, destination)| {
            let pixel_index = tile_index * TILE_PIXELS;
            for channel in 0..channels {
                let source = &image_data[channel * pixels + pixel_index..][..TILE_PIXELS];
                let bias = dense[dense_start + channel * dense_stride[1]];
                for (local_pixel, value) in source.iter().enumerate() {
                    destination[local_pixel * channels + channel] = *value + bias;
                }
            }
        });
    Ok(Some(Tensor::from_vec(
        output,
        (batch, pixels, channels),
        image.device(),
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

#[derive(Debug, Default)]
struct PreparedProjectionInner {
    weight: OnceLock<Tensor>,
    initialization: Mutex<()>,
}

#[derive(Clone, Debug, Default)]
struct PreparedProjection(Arc<PreparedProjectionInner>);

impl PreparedProjection {
    fn get_or_prepare(&self, projection: &Linear) -> Result<Tensor> {
        if let Some(weight) = self.0.weight.get() {
            return Ok(weight.clone());
        }
        // Serialize only the one-time materialization. Tensor clones below
        // are shallow handles, so concurrent callers never make a private
        // transposed copy per worker.
        let _guard = self.0.initialization.lock().map_err(|_| {
            candle_core::Error::Msg("decoder projection cache mutex poisoned".into())
        })?;
        if let Some(weight) = self.0.weight.get() {
            return Ok(weight.clone());
        }
        let weight = projection
            .prepare_transposed_weight_cpu_f32()?
            .ok_or_else(|| candle_core::Error::Msg("decoder projection is not CPU f32".into()))?;
        let _ = self.0.weight.set(weight.clone());
        Ok(weight)
    }
}

#[derive(Clone, Debug)]
pub struct Attention {
    internal_dim: usize,
    num_heads: usize,
    decoder_blocked: bool,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_proj_weight_t: Option<PreparedProjection>,
    k_proj_weight_t: Option<PreparedProjection>,
    v_proj_weight_t: Option<PreparedProjection>,
    out_proj_weight_t: Option<PreparedProjection>,
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
        let q_proj = linear(embedding_dim, internal_dim, vb.pp("q_proj"))?;
        let k_proj = linear(kv_dim, internal_dim, vb.pp("k_proj"))?;
        let v_proj = linear(kv_dim, internal_dim, vb.pp("v_proj"))?;
        let out_proj = linear(internal_dim, embedding_dim, vb.pp("out_proj"))?;
        let prepare_weights = vb.device().is_cpu()
            && std::env::var_os("SAM2_CPU_DECODER_PRETRANSPOSE_WEIGHTS")
                .is_none_or(|value| value != "0");
        let prepare = || prepare_weights.then(PreparedProjection::default);
        Ok(Self {
            internal_dim,
            num_heads,
            decoder_blocked: false,
            q_proj_weight_t: prepare(),
            k_proj_weight_t: prepare(),
            v_proj_weight_t: prepare(),
            out_proj_weight_t: prepare(),
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    fn with_decoder_blocked(mut self, decoder_blocked: bool) -> Self {
        self.decoder_blocked = decoder_blocked;
        self
    }

    fn project(
        &self,
        projection: &Linear,
        prepared: Option<&PreparedProjection>,
        x: &Tensor,
    ) -> Result<Tensor> {
        // Reusing the transposed immutable weights wins for the interactive
        // B=1 and low-batch B=2/4 routes. AMG's B=64 matrices amortize the
        // ordinary GEMM packing and are slightly faster in their established
        // orientation, so keep that route unchanged.
        if x.dim(0)? <= 4
            && let Some(prepared) = prepared
        {
            let weight = prepared.get_or_prepare(projection)?;
            projection.forward_fused_bias_cpu_pretransposed(x, &weight)
        } else {
            projection.forward_fused_bias_cpu(x)
        }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if std::env::var_os("SAM2_PROFILE_IMAGE_STAGES").is_some() {
            let (output, profile) = self.forward_profiled(q, k, v)?;
            q.device().synchronize()?;
            eprintln!(
                "SAM2_ATTENTION_STAGE q_tokens={} k_tokens={} qkv_ms={:.6} kernel_ms={:.6} out_ms={:.6} total_ms={:.6}",
                q.dim(1)?,
                k.dim(1)?,
                profile.qkv_ms,
                profile.kernel_ms,
                profile.output_projection_ms,
                profile.total_ms,
            );
            return Ok(output);
        }
        self.forward_impl(q, k, v, None)
    }

    /// Scaled dot-product attention built only from operations that implement backward.
    pub fn forward_differentiable(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self
            .separate_heads(&self.q_proj.forward(q)?)?
            .contiguous()?;
        let k = self
            .separate_heads(&self.k_proj.forward(k)?)?
            .contiguous()?;
        let v = self
            .separate_heads(&self.v_proj.forward(v)?)?
            .contiguous()?;
        let dim = self.internal_dim / self.num_heads;
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1. / (dim as f64).sqrt(), 0.)?;
        let attended = softmax(&scores, D::Minus1)?.matmul(&v)?;
        self.out_proj.forward(&self.recombine_heads(&attended)?)
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
        let batch = q.dim(0)?;
        let batched_asymmetric_flash = batch >= 2
            && std::env::var_os("SAM2_CPU_AMG_BATCHED_FLASH").is_none_or(|value| value != "0")
            && ((q_tokens <= 16 && k_tokens >= 512) || (q_tokens >= 512 && k_tokens <= 16));
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
            && ((batch == 1
                && (q_tokens >= 512
                    || ((2..=16).contains(&q_tokens)
                        && (512..=8192).contains(&k_tokens)
                        && std::env::var_os("SAM2_CPU_DECODER_SHORT_FLASH")
                            .is_none_or(|value| value != "0"))))
                || batched_asymmetric_flash)
            && std::env::var_os("SAM2_CPU_DECODER_FLASH").is_none_or(|value| value != "0")
        {
            let phase_start = profile.is_some().then(Instant::now);
            let head_dim = self.internal_dim / self.num_heads;
            let parity_call = if std::env::var_os("SAM2_PROFILE_DECODER_PARITY").is_some()
                && batch == 1
                && (2..=16).contains(&q_tokens)
                && k_tokens >= 512
            {
                static CALLS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                (call < 3).then_some(call)
            } else {
                None
            };
            let q_projected = self.project(&self.q_proj, self.q_proj_weight_t.as_ref(), q)?;
            let k_projected = self.project(&self.k_proj, self.k_proj_weight_t.as_ref(), k)?;
            let v_projected = self.project(&self.v_proj, self.v_proj_weight_t.as_ref(), v)?;
            if let Some(call) = parity_call {
                profile_decoder_parity_vector(
                    &format!("attention_{call}_q_projection"),
                    &q_projected,
                    0,
                    2,
                )?;
                profile_decoder_parity_vector(
                    &format!("attention_{call}_k_projection"),
                    &k_projected,
                    0,
                    0,
                )?;
                profile_decoder_parity_vector(
                    &format!("attention_{call}_v_projection"),
                    &v_projected,
                    0,
                    0,
                )?;
            }
            let q = q_projected.reshape((batch, q_tokens, self.num_heads, head_dim))?;
            let k = k_projected.reshape((batch, k_tokens, self.num_heads, head_dim))?;
            let v = v_projected.reshape((batch, v.dim(1)?, self.num_heads, head_dim))?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                profile.qkv_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            let phase_start = profile.is_some().then(Instant::now);
            let attended = if batch == 1
                && self.decoder_blocked
                && std::env::var_os("SAM2_CPU_DECODER_LAYER1_BLOCKED")
                    .is_none_or(|value| value != "0")
            {
                candle_nn::attention::flash_attn_nhsd_f32_decoder_blocked(
                    &q,
                    &k,
                    &v,
                    1. / (head_dim as f32).sqrt(),
                )?
            } else if batch == 1 {
                candle_nn::attention::flash_attn_nhsd_f32(
                    &q,
                    &k,
                    &v,
                    1. / (head_dim as f32).sqrt(),
                )?
            } else {
                candle_nn::attention::flash_attn_nhsd_f32_batched(
                    &q,
                    &k,
                    &v,
                    1. / (head_dim as f32).sqrt(),
                )?
            };
            let attended = attended.flatten_from(2)?;
            if let Some(call) = parity_call {
                profile_decoder_parity_vector(
                    &format!("attention_{call}_attended"),
                    &attended,
                    0,
                    2,
                )?;
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                profile.kernel_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            let phase_start = profile.is_some().then(Instant::now);
            let output =
                self.project(&self.out_proj, self.out_proj_weight_t.as_ref(), &attended)?;
            if let Some(call) = parity_call {
                profile_decoder_parity_vector(
                    &format!("attention_{call}_output_projection"),
                    &output,
                    0,
                    2,
                )?;
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                profile.output_projection_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            return Ok(output);
        }
        #[cfg(feature = "cuda")]
        if q.device().is_cuda()
            && (q_tokens >= 512 || k_tokens >= 512)
            && std::env::var_os("SAM2_CUDA_DECODER_DIRECT_QKV_LAYOUT")
                .is_none_or(|value| value != "0")
        {
            // cuBLASLt produces BxSx(HxD) contiguously, and the shared CUDA
            // attention kernel accepts independent sequence/head strides.
            // Preserve that native BxSxHxD view instead of materializing
            // three BxHxSxD copies only to permute them back before launch.
            let phase_start = profile.is_some().then(Instant::now);
            let head_dim = self.internal_dim / self.num_heads;
            let q = self.q_proj.forward_fused_bias_cpu(q)?.reshape((
                q.dim(0)?,
                q_tokens,
                self.num_heads,
                head_dim,
            ))?;
            let k = self.k_proj.forward_fused_bias_cpu(k)?.reshape((
                k.dim(0)?,
                k_tokens,
                self.num_heads,
                head_dim,
            ))?;
            let v = self.v_proj.forward_fused_bias_cpu(v)?.reshape((
                v.dim(0)?,
                v.dim(1)?,
                self.num_heads,
                head_dim,
            ))?;
            let profile_a2_final = if std::env::var_os("SAM2_PROFILE_A2_BOUNDARY").is_some()
                && batch == 64
                && q_tokens == 8
                && k_tokens == 4096
            {
                // Each decoder invocation has two block token-to-image calls
                // followed by this final token-to-image call. AMG's disputed
                // point is in invocation 2, hence long-attention call 8.
                static CALLS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 48 == 8
            } else {
                false
            };
            if profile_a2_final {
                profile_a2_final_attention_vector("q_projection", &q)?;
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                q.device().synchronize()?;
                profile.qkv_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            let phase_start = profile.is_some().then(Instant::now);
            let attended = super::hiera::math_attention(&q, &k, &v, 1. / (head_dim as f64).sqrt())?
                .reshape((q.dim(0)?, q_tokens, self.internal_dim))?;
            if profile_a2_final {
                profile_a2_final_attention_vector("attended", &attended)?;
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                q.device().synchronize()?;
                profile.kernel_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            let phase_start = profile.is_some().then(Instant::now);
            let output = self.out_proj.forward_fused_bias_cpu(&attended)?;
            if profile_a2_final {
                profile_a2_final_attention_vector("output_projection", &output)?;
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
                q.device().synchronize()?;
                profile.output_projection_ms = start.elapsed().as_secs_f64() * 1000.;
            }
            return Ok(output);
        }
        let phase_start = profile.is_some().then(Instant::now);
        let q = self
            .separate_heads(&self.project(&self.q_proj, self.q_proj_weight_t.as_ref(), q)?)?
            .contiguous()?;
        let k = self
            .separate_heads(&self.project(&self.k_proj, self.k_proj_weight_t.as_ref(), k)?)?
            .contiguous()?;
        let v = self
            .separate_heads(&self.project(&self.v_proj, self.v_proj_weight_t.as_ref(), v)?)?
            .contiguous()?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
            q.device().synchronize()?;
            profile.qkv_ms = start.elapsed().as_secs_f64() * 1000.;
        }
        let phase_start = profile.is_some().then(Instant::now);
        let dim = self.internal_dim / self.num_heads;
        #[cfg(feature = "cuda")]
        let attended = if q.device().is_cuda()
            && (q_tokens >= 512 || k_tokens >= 512)
            && std::env::var_os("SAM2_CUDA_DECODER_MEM_EFFICIENT_ATTENTION")
                .is_none_or(|value| value != "0")
        {
            // PyTorch routes these long asymmetric decoder shapes through
            // scaled_dot_product_attention. The shared CUTLASS kernel accepts
            // BxQxHxD stride views and writes BxQxHxD directly, avoiding the
            // materialized score tensor and the output head permutation.
            super::hiera::math_attention(
                &q.permute((0, 2, 1, 3))?,
                &k.permute((0, 2, 1, 3))?,
                &v.permute((0, 2, 1, 3))?,
                1. / (dim as f64).sqrt(),
            )?
            .reshape((q.dim(0)?, q_tokens, self.internal_dim))?
        } else {
            let attention = softmax_last_dim(
                &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                    .affine(1. / (dim as f64).sqrt(), 0.)?,
            )?;
            self.recombine_heads(&attention.contiguous()?.matmul(&v)?)?
        };
        #[cfg(not(feature = "cuda"))]
        let attended = {
            let attention = softmax_last_dim(
                &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                    .affine(1. / (dim as f64).sqrt(), 0.)?,
            )?;
            self.recombine_heads(&attention.contiguous()?.matmul(&v)?)?
        };
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
            q.device().synchronize()?;
            profile.kernel_ms = start.elapsed().as_secs_f64() * 1000.;
        }
        let phase_start = profile.is_some().then(Instant::now);
        let output = self.project(&self.out_proj, self.out_proj_weight_t.as_ref(), &attended)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), phase_start) {
            q.device().synchronize()?;
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
    rope_cos: Tensor,
    rope_sin: Tensor,
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
        // The reference registers the default 64x64 axial RoPE table when the
        // module is constructed and reuses it for every video frame. SAM2's
        // memory attention always receives this fixed feature-grid shape.
        let (rope_cos, rope_sin) =
            axial_frequencies(internal_dim / num_heads, 64, 64, 10_000., vb.device())?;
        Ok(Self {
            internal_dim,
            num_heads,
            q_proj: linear(embedding_dim, internal_dim, vb.pp("q_proj"))?,
            k_proj: linear(kv_dim, internal_dim, vb.pp("k_proj"))?,
            v_proj: linear(kv_dim, internal_dim, vb.pp("v_proj"))?,
            out_proj: linear(internal_dim, embedding_dim, vb.pp("out_proj"))?,
            rope_theta: 10_000.,
            rope_k_repeat,
            rope_cos,
            rope_sin,
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
        let profiling = std::env::var_os("SAM2_PROFILE_VIDEO_STAGES").is_some();
        if profiling {
            q.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
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
        if profiling {
            q.device().synchronize()?;
            eprintln!(
                "SAM2_ROPE_ATTENTION_STAGE kind={} q_tokens={} k_tokens={} stage=qkv ms={:.6}",
                if self.rope_k_repeat { "cross" } else { "self" },
                q_tokens,
                k_tokens,
                profile_start.elapsed().as_secs_f64() * 1000.
            );
            profile_start = Instant::now();
        }
        let rope_keys = k_tokens - num_k_exclude_rope;
        if rope_keys % q_tokens != 0 {
            bail!("SAM2 RoPE key count {rope_keys} is not a whole number of {q_tokens}-token grids")
        }
        let (cos, sin) = if side == 64 {
            (self.rope_cos.clone(), self.rope_sin.clone())
        } else {
            axial_frequencies(
                self.internal_dim / self.num_heads,
                side,
                side,
                self.rope_theta,
                q.device(),
            )?
        };
        let q = apply_rotary(&q, &cos, &sin)?.contiguous()?;
        let repeats = rope_keys / q_tokens;
        if repeats != 1 && !self.rope_k_repeat {
            bail!("RoPE cross-attention received repeated memory keys without rope_k_repeat")
        }
        k = if q.device().is_cpu()
            && std::env::var_os("SAM2_CPU_REPEAT_AWARE_ROPE").is_none_or(|value| value != "0")
        {
            apply_rotary_repeated(&k, &cos, &sin, num_k_exclude_rope)?
        } else {
            let key_cos = cos.repeat((repeats, 1))?;
            let key_sin = sin.repeat((repeats, 1))?;
            let rotated = apply_rotary(&k.narrow(2, 0, rope_keys)?, &key_cos, &key_sin)?;
            if num_k_exclude_rope == 0 {
                rotated
            } else {
                Tensor::cat(&[&rotated, &k.narrow(2, rope_keys, num_k_exclude_rope)?], 2)?
            }
            .contiguous()?
        };
        if profiling {
            q.device().synchronize()?;
            eprintln!(
                "SAM2_ROPE_ATTENTION_STAGE kind={} q_tokens={} k_tokens={} stage=rope ms={:.6}",
                if self.rope_k_repeat { "cross" } else { "self" },
                q_tokens,
                k_tokens,
                profile_start.elapsed().as_secs_f64() * 1000.
            );
            profile_start = Instant::now();
        }
        let dim = self.internal_dim / self.num_heads;
        #[cfg(feature = "cuda")]
        let attended = if q.device().is_cuda()
            && std::env::var_os("SAM2_CUDA_MEM_EFFICIENT_ATTENTION")
                .is_none_or(|value| value != "0")
        {
            // The shared CUTLASS kernel consumes BxQxHxD and supports unequal
            // query/key sequence lengths. RoPE stores BxHxQxD, so these are
            // stride-only views; only the compact BxQxHxD output is allocated.
            super::hiera::math_attention(
                &q.permute((0, 2, 1, 3))?,
                &k.permute((0, 2, 1, 3))?,
                &v.permute((0, 2, 1, 3))?,
                1. / (dim as f64).sqrt(),
            )?
            .permute((0, 2, 1, 3))?
        } else {
            softmax_last_dim(
                &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                    .affine(1. / (dim as f64).sqrt(), 0.)?,
            )?
            .matmul(&v.contiguous()?)?
        };
        #[cfg(not(feature = "cuda"))]
        let attended = {
            #[cfg(feature = "cpu-flash-attention")]
            if std::env::var_os("SAM2_CPU_MEMORY_FLASH").is_none_or(|value| value != "0") {
                // RoPE produces BxHxSxD tensors. The CPU flash kernel accepts
                // their BxSxHxD stride views and emits compact BxSxHxD output,
                // avoiding the O(Q*K) score and softmax allocations used by
                // the generic path. Return a stride-only BxHxSxD view so the
                // existing head recombination remains shared with CUDA.
                candle_nn::attention::flash_attn_nhsd_f32(
                    &q.permute((0, 2, 1, 3))?,
                    &k.permute((0, 2, 1, 3))?,
                    &v.permute((0, 2, 1, 3))?,
                    1. / (dim as f32).sqrt(),
                )?
                .permute((0, 2, 1, 3))?
            } else {
                softmax_last_dim(
                    &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                        .affine(1. / (dim as f64).sqrt(), 0.)?,
                )?
                .matmul(&v.contiguous()?)?
            }
            #[cfg(not(feature = "cpu-flash-attention"))]
            {
                softmax_last_dim(
                    &q.matmul(&k.transpose(2, 3)?.contiguous()?)?
                        .affine(1. / (dim as f64).sqrt(), 0.)?,
                )?
                .matmul(&v.contiguous()?)?
            }
        };
        if profiling {
            q.device().synchronize()?;
            eprintln!(
                "SAM2_ROPE_ATTENTION_STAGE kind={} q_tokens={} k_tokens={} stage=kernel ms={:.6}",
                if self.rope_k_repeat { "cross" } else { "self" },
                q_tokens,
                k_tokens,
                profile_start.elapsed().as_secs_f64() * 1000.
            );
            profile_start = Instant::now();
        }
        let output = self
            .out_proj
            .forward_fused_bias_cpu(&self.recombine_heads(&attended)?)?;
        if profiling {
            q.device().synchronize()?;
            eprintln!(
                "SAM2_ROPE_ATTENTION_STAGE kind={} q_tokens={} k_tokens={} stage=output ms={:.6}",
                if self.rope_k_repeat { "cross" } else { "self" },
                q_tokens,
                k_tokens,
                profile_start.elapsed().as_secs_f64() * 1000.
            );
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

#[derive(Clone, Debug)]
struct TwoWayAttentionBlock {
    index: usize,
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
            index,
            self_attn: Attention::load(256, 8, 1, None, vb.pp("self_attn"))?,
            norm1: candle_nn::layer_norm(256, 1e-5, vb.pp("norm1"))?,
            cross_attn_token_to_image: Attention::load(
                256,
                8,
                2,
                None,
                vb.pp("cross_attn_token_to_image"),
            )?
            .with_decoder_blocked(index == 1),
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
        let profiling = std::env::var_os("SAM2_PROFILE_AMG_BLOCK_STAGES").is_some();
        let profile_decoder_parity =
            std::env::var_os("SAM2_PROFILE_DECODER_PARITY").is_some() && queries.dim(0)? == 1;
        if profiling {
            queries.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
        macro_rules! profile_stage {
            ($name:expr) => {
                if profiling {
                    queries.device().synchronize()?;
                    eprintln!(
                        "SAM2_TRANSFORMER_BLOCK_STAGE stage={} batch={} tokens={} ms={:.6}",
                        $name,
                        queries.dim(0)?,
                        queries.dim(1)?,
                        profile_start.elapsed().as_secs_f64() * 1000.
                    );
                    profile_start = Instant::now();
                }
            };
        }
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = queries.broadcast_add(query_pe)?;
            queries.broadcast_add(&self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = self.norm1.forward(&queries)?;
        if profile_decoder_parity {
            profile_decoder_parity_vector(
                &format!("layer_{}_norm1_query", self.index),
                &queries,
                0,
                2,
            )?;
        }
        profile_stage!("self_attention_norm");
        let q = queries.broadcast_add(query_pe)?;
        let image_with_pe = keys.broadcast_add(key_pe)?;
        queries = self.norm2.forward(
            &queries.broadcast_add(&self.cross_attn_token_to_image.forward(
                &q,
                &image_with_pe,
                keys,
            )?)?,
        )?;
        if profile_decoder_parity {
            profile_decoder_parity_vector(
                &format!("layer_{}_norm2_query", self.index),
                &queries,
                0,
                2,
            )?;
        }
        profile_stage!("token_to_image_norm");
        queries = self
            .norm3
            .forward(&queries.broadcast_add(&self.mlp.forward_decoder_fused_bias(&queries)?)?)?;
        if profile_decoder_parity {
            profile_decoder_parity_vector(
                &format!("layer_{}_norm3_query", self.index),
                &queries,
                0,
                2,
            )?;
        }
        profile_stage!("mlp_norm");
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
        if profile_decoder_parity {
            profile_decoder_parity_vector(&format!("layer_{}_norm4_key", self.index), &keys, 0, 0)?;
        }
        profile_stage!("image_to_token_norm");
        let _ = profile_start;
        Ok((queries, keys))
    }

    fn forward_differentiable(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        query_pe: &Tensor,
        key_pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn
                .forward_differentiable(queries, queries, queries)?
        } else {
            let q = queries.broadcast_add(query_pe)?;
            queries.broadcast_add(&self.self_attn.forward_differentiable(&q, &q, queries)?)?
        };
        queries = layer_norm_differentiable(&self.norm1, &queries)?;
        let q = queries.broadcast_add(query_pe)?;
        let image_with_pe = keys.broadcast_add(key_pe)?;
        queries = layer_norm_differentiable(
            &self.norm2,
            &queries.broadcast_add(&self.cross_attn_token_to_image.forward_differentiable(
                &q,
                &image_with_pe,
                keys,
            )?)?,
        )?;
        queries = layer_norm_differentiable(
            &self.norm3,
            &queries.broadcast_add(&self.mlp.forward(&queries)?)?,
        )?;
        let q = queries.broadcast_add(query_pe)?;
        let keys = layer_norm_differentiable(
            &self.norm4,
            &keys.broadcast_add(&self.cross_attn_image_to_token.forward_differentiable(
                &image_with_pe,
                &q,
                &queries,
            )?)?,
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
    image_token_cache: Arc<Mutex<Option<(TensorId, TensorId, Tensor)>>>,
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
            image_token_cache: Arc::new(Mutex::new(None)),
        })
    }

    fn image_tokens_with_dense(&self, image: &Tensor, dense: &Tensor) -> Result<Tensor> {
        let cache_enabled = image.device().is_cpu()
            && std::env::var_os("SAM2_CPU_DECODER_CACHE_IMAGE_TOKENS")
                .is_none_or(|value| value != "0");
        if !cache_enabled {
            return image_tokens_nhwc(&image.broadcast_add(dense)?, true);
        }
        let image_id = image.id();
        let dense_id = dense.id();
        let mut cache = self.image_token_cache.lock().map_err(|_| {
            candle_core::Error::Msg("decoder image-token cache mutex poisoned".into())
        })?;
        if let Some((cached_image, cached_dense, tokens)) = &*cache
            && *cached_image == image_id
            && *cached_dense == dense_id
        {
            return Ok(tokens.clone());
        }
        // Keep the lock across the one-time preparation. Concurrent callers
        // share the resulting Tensor instead of each allocating and packing a
        // private 4 MiB copy on their first prompt.
        let Some(tokens) = image_tokens_add_nhwc_cpu_f32(image, dense)? else {
            drop(cache);
            return image_tokens_nhwc(&image.broadcast_add(dense)?, true);
        };
        *cache = Some((image_id, dense_id, tokens.clone()));
        Ok(tokens)
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
        dense_prompt_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let profiling = std::env::var_os("SAM2_PROFILE_AMG_STAGES").is_some();
        let profile_a2_boundary = if std::env::var_os("SAM2_PROFILE_A2_BOUNDARY").is_some()
            && point_embedding.dim(0)? == 64
        {
            static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 16 == 2
        } else {
            false
        };
        let profile_decoder_parity = if std::env::var_os("SAM2_PROFILE_DECODER_PARITY").is_some()
            && point_embedding.dim(0)? == 1
        {
            static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0
        } else {
            false
        };
        if profiling {
            image_embedding.device().synchronize()?;
        }
        let mut profile_start = Instant::now();
        macro_rules! profile_stage {
            ($name:expr) => {
                if profiling {
                    image_embedding.device().synchronize()?;
                    eprintln!(
                        "SAM2_TRANSFORMER_STAGE stage={} batch={} tokens={} ms={:.6}",
                        $name,
                        point_embedding.dim(0)?,
                        point_embedding.dim(1)?,
                        profile_start.elapsed().as_secs_f64() * 1000.
                    );
                    profile_start = Instant::now();
                }
            };
        }
        // Both tensors are reused throughout the decoder. Prepare one
        // contiguous token layout each, without per-worker input copies.
        let mut keys = self.image_tokens_with_dense(image_embedding, dense_prompt_embedding)?;
        // The same positional encoding is added to image tokens five times.
        // One contiguous preparation avoids five slow strided additions and
        // is reused across the two transformer layers and final attention.
        let image_pe = self.image_pe_tokens(image_pe)?;
        let mut queries = point_embedding.clone();
        if profile_decoder_parity {
            // Non-multimask image inference selects internal mask token 0,
            // which follows the object and IoU tokens at transformer index 2.
            profile_decoder_parity_vector("input_query", &queries, 0, 2)?;
            profile_decoder_parity_vector("input_key", &keys, 0, 0)?;
        }
        if profile_a2_boundary {
            profile_a2_transformer_token("input", &queries)?;
        }
        profile_stage!("setup");
        for (index, layer) in self.layers.iter().enumerate() {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, &image_pe)?;
            if profile_decoder_parity {
                profile_decoder_parity_vector(
                    if index == 0 {
                        "layer_0_query"
                    } else {
                        "layer_1_query"
                    },
                    &queries,
                    0,
                    2,
                )?;
                profile_decoder_parity_vector(
                    if index == 0 {
                        "layer_0_key"
                    } else {
                        "layer_1_key"
                    },
                    &keys,
                    0,
                    0,
                )?;
            }
            if profile_a2_boundary {
                profile_a2_transformer_token(
                    if index == 0 { "layer_0" } else { "layer_1" },
                    &queries,
                )?;
            }
            profile_stage!(if index == 0 { "layer_0" } else { "layer_1" });
        }
        let q = queries.broadcast_add(point_embedding)?;
        let k = keys.broadcast_add(&image_pe)?;
        let final_attention = self.final_attn_token_to_image.forward(&q, &k, &keys)?;
        if profile_decoder_parity {
            profile_decoder_parity_vector("final_attention", &final_attention, 0, 2)?;
        }
        queries = self
            .norm_final_attn
            .forward(&queries.broadcast_add(&final_attention)?)?;
        if profile_a2_boundary {
            profile_a2_transformer_token("final", &queries)?;
        }
        if profile_decoder_parity {
            profile_decoder_parity_vector("final_query", &queries, 0, 2)?;
            profile_decoder_parity_vector("final_key", &keys, 0, 0)?;
        }
        profile_stage!("final_attention");
        let _ = profile_start;
        Ok((queries, keys))
    }

    /// Reference two-way transformer equations without inference caches or no-backward kernels.
    pub fn forward_differentiable(
        &self,
        image_embedding: &Tensor,
        dense_prompt_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut keys = image_tokens_nhwc(
            &image_embedding.broadcast_add(dense_prompt_embedding)?,
            false,
        )?
        .contiguous()?;
        let image_pe = image_tokens_nhwc(image_pe, false)?.contiguous()?;
        let mut queries = point_embedding.clone();
        for layer in &self.layers {
            (queries, keys) =
                layer.forward_differentiable(&queries, &keys, point_embedding, &image_pe)?;
        }
        let q = queries.broadcast_add(point_embedding)?;
        let k = keys.broadcast_add(&image_pe)?;
        let final_attention = self
            .final_attn_token_to_image
            .forward_differentiable(&q, &k, &keys)?;
        queries = layer_norm_differentiable(
            &self.norm_final_attn,
            &queries.broadcast_add(&final_attention)?,
        )?;
        Ok((queries, keys))
    }

    pub fn forward_profiled(
        &self,
        image_embedding: &Tensor,
        dense_prompt_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> Result<((Tensor, Tensor), TwoWayTransformerProfile)> {
        let setup_start = Instant::now();
        let mut keys = self.image_tokens_with_dense(image_embedding, dense_prompt_embedding)?;
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

    use super::{
        PreparedProjection, cached_image_pe_tokens, image_tokens_add_nhwc_cpu_f32,
        image_tokens_nhwc_cpu_f32,
    };

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
    fn direct_image_token_add_pack_matches_generic_composition() -> Result<()> {
        let image = Tensor::from_vec(
            (0..2 * 8 * 8)
                .map(|index| index as f32 * 0.0625 - 2.)
                .collect(),
            (1, 2, 8, 8),
            &Device::Cpu,
        )?;
        let dense =
            Tensor::new(&[[[[0.25f32]], [[-0.5f32]]]], &Device::Cpu)?.broadcast_as((1, 2, 8, 8))?;
        let expected = image
            .broadcast_add(&dense)?
            .reshape((1, 2, 64))?
            .permute((0, 2, 1))?
            .contiguous()?;
        let actual =
            image_tokens_add_nhwc_cpu_f32(&image, &dense)?.expect("eligible CPU f32 inputs");
        assert_eq!(expected.to_vec3::<f32>()?, actual.to_vec3::<f32>()?);
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

    #[test]
    fn prepared_projection_is_materialized_once_across_workers() -> Result<()> {
        let weight = Tensor::from_vec(
            (0..48 * 32).map(|index| index as f32 * 0.001).collect(),
            (48, 32),
            &Device::Cpu,
        )?;
        let projection = candle_nn::Linear::new(weight, None);
        let prepared = PreparedProjection::default();
        let ids = std::thread::scope(|scope| {
            (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        prepared
                            .get_or_prepare(&projection)
                            .map(|weight| weight.id())
                    })
                })
                .map(|worker| worker.join().expect("projection worker panicked"))
                .collect::<Result<Vec<_>>>()
        })?;
        assert!(ids.iter().all(|id| *id == ids[0]));
        Ok(())
    }
}
