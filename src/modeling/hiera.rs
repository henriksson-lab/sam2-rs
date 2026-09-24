//! Translation of `sam2/modeling/backbones/hieradet.py`.

use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use candle_core::{DType, Result, Storage, Tensor};
#[cfg(feature = "cpu-flash-attention")]
use candle_nn::attention::{flash_attn_nhsd_f32, flash_attn_nhsd_f32_batched};
use candle_nn::{LayerNorm, Module, VarBuilder, linear, ops::softmax_last_dim};
use rayon::prelude::*;

use crate::{
    config::HieraConfig,
    ops::{
        Activation, Mlp, PatchEmbed, WindowPartition, window_partition,
        window_unpartition_add_residual,
    },
};

#[derive(Clone, Debug)]
struct MultiScaleAttention {
    dim_out: usize,
    num_heads: usize,
    q_pool: bool,
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
}

#[derive(Clone, Debug)]
struct MultiScaleAttentionProfile {
    qkv_ms: f64,
    qkv_views_ms: f64,
    query_pool_ms: f64,
    query_reshape_ms: f64,
    query_pool_core_ms: f64,
    kernel_ms: f64,
    projection_ms: f64,
    cleanup_ms: f64,
    cleanup_attended_ms: f64,
    cleanup_qkv_ms: f64,
}

fn math_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda()
        && k.dims() == v.dims()
        && q.dim(0)? == k.dim(0)?
        && q.dim(2)? == k.dim(2)?
        && q.dim(3)? == k.dim(3)?
        && q.dim(3)? <= 128
        && std::env::var_os("SAM2_CUDA_MEM_EFFICIENT_ATTENTION").is_none_or(|value| value != "0")
    {
        return q.apply_op3_no_bwd(
            k,
            v,
            &CudaMemEffAttention {
                scale: scale as f32,
            },
        );
    }
    let q = q.permute((0, 2, 1, 3))?.contiguous()?.affine(scale, 0.)?;
    let k = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v = v.permute((0, 2, 1, 3))?.contiguous()?;
    let k_transposed = k.transpose(2, 3)?.contiguous()?;

    let query_len = q.dim(2)?;
    // Bound the otherwise 268 MiB f32 score tensor in SAM2 Tiny's global
    // blocks. Four reverse-ordered full-model pairs favored 1024 rows, while
    // producing byte-identical masks. Set the value to 0 for the unchunked
    // fallback or use another power-of-two size for shape-level tuning.
    let query_chunk = match std::env::var("SAM2_CUDA_ATTENTION_QUERY_CHUNK") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&value| value >= 128 && value.is_power_of_two()),
        Err(_) => Some(1024),
    };
    let attended = if q.device().is_cuda() && query_chunk.is_some_and(|chunk| chunk < query_len) {
        let chunk = query_chunk.unwrap();
        let mut outputs = Vec::with_capacity(query_len.div_ceil(chunk));
        for start in (0..query_len).step_by(chunk) {
            let len = (query_len - start).min(chunk);
            let scores = q.narrow(2, start, len)?.matmul(&k_transposed)?;
            outputs.push(softmax_last_dim(&scores)?.matmul(&v)?);
        }
        Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?
    } else {
        softmax_last_dim(&q.matmul(&k_transposed)?)?.matmul(&v)?
    };
    attended.permute((0, 2, 1, 3))
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaMemEffAttention {
    scale: f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn sam2_mem_eff_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        batch: i32,
        q_sequence: i32,
        kv_sequence: i32,
        heads: i32,
        head_dim: i32,
        q_stride_b: i64,
        q_stride_m: i32,
        q_stride_h: i32,
        k_stride_b: i64,
        k_stride_m: i32,
        k_stride_h: i32,
        v_stride_b: i64,
        v_stride_m: i32,
        v_stride_h: i32,
        scale: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    fn sam2_max_pool2d_nhwc_f32(
        input: *const f32,
        output: *mut f32,
        output_elements: u64,
        height: u32,
        width: u32,
        channels: u32,
        stride_b: u64,
        stride_y: u64,
        stride_x: u64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp3 for CudaMemEffAttention {
    fn name(&self) -> &'static str {
        "sam2-memory-efficient-attention-f32"
    }

    fn cpu_fwd(
        &self,
        _q: &candle_core::CpuStorage,
        _q_l: &candle_core::Layout,
        _k: &candle_core::CpuStorage,
        _k_l: &candle_core::Layout,
        _v: &candle_core::CpuStorage,
        _v_l: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("memory-efficient attention is only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &candle_core::Layout,
        k: &candle_core::CudaStorage,
        k_l: &candle_core::Layout,
        v: &candle_core::CudaStorage,
        v_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
            candle_core::bail!("memory-efficient attention only supports f32")
        }
        let (batch, q_sequence, heads, head_dim) = q_l.shape().dims4()?;
        let (k_batch, kv_sequence, k_heads, k_head_dim) = k_l.shape().dims4()?;
        if v_l.shape() != k_l.shape()
            || batch != k_batch
            || heads != k_heads
            || head_dim != k_head_dim
            || head_dim > 128
        {
            candle_core::bail!("unsupported memory-efficient attention shape")
        }
        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        if q_stride[3] != 1 || k_stride[3] != 1 || v_stride[3] != 1 {
            candle_core::bail!("attention head dimensions must be contiguous")
        }
        let dev = q.device();
        let stream = dev.cuda_stream();
        let q = q.as_cuda_slice::<f32>()?.slice(q_l.start_offset()..);
        let k = k.as_cuda_slice::<f32>()?.slice(k_l.start_offset()..);
        let v = v.as_cuda_slice::<f32>()?.slice(v_l.start_offset()..);
        // SAFETY: The external kernel writes every element in the validated output shape.
        let mut output = unsafe { dev.alloc::<f32>(q_l.shape().elem_count())? };
        let (q_ptr, q_guard) = q.device_ptr(&stream);
        let (k_ptr, k_guard) = k.device_ptr(&stream);
        let (v_ptr, v_guard) = v.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_mem_eff_f32(
                q_ptr as *const f32,
                k_ptr as *const f32,
                v_ptr as *const f32,
                output_ptr as *mut f32,
                batch as i32,
                q_sequence as i32,
                kv_sequence as i32,
                heads as i32,
                head_dim as i32,
                q_stride[0] as i64,
                q_stride[1] as i32,
                q_stride[2] as i32,
                k_stride[0] as i64,
                k_stride[1] as i32,
                k_stride[2] as i32,
                v_stride[0] as i64,
                v_stride[1] as i32,
                v_stride[2] as i32,
                self.scale,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("memory-efficient attention CUDA launch failed with status {status}")
        }
        drop((q_guard, k_guard, v_guard, output_guard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            q_l.shape().clone(),
        ))
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaNhwcMaxPool {
    height: usize,
    width: usize,
    channels: usize,
    stride_y: usize,
    stride_x: usize,
    output_shape: candle_core::Shape,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for CudaNhwcMaxPool {
    fn name(&self) -> &'static str {
        "sam2-max-pool2d-nhwc-f32"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("NHWC max pool is only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        if storage.dtype() != DType::F32 || layout.stride()[3] != 1 {
            candle_core::bail!("NHWC max pool requires f32 with contiguous channels")
        }
        let dev = storage.device();
        let stream = dev.cuda_stream();
        let input = storage
            .as_cuda_slice::<f32>()?
            .slice(layout.start_offset()..);
        let output_elements = self.output_shape.elem_count();
        // SAFETY: The kernel writes each element in the validated output shape once.
        let mut output = unsafe { dev.alloc::<f32>(output_elements)? };
        let (input_ptr, input_guard) = input.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_max_pool2d_nhwc_f32(
                input_ptr as *const f32,
                output_ptr as *mut f32,
                output_elements as u64,
                self.height as u32,
                self.width as u32,
                self.channels as u32,
                layout.stride()[0] as u64,
                self.stride_y as u64,
                self.stride_x as u64,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("NHWC max-pool CUDA launch failed with status {status}")
        }
        drop((input_guard, output_guard));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            self.output_shape.clone(),
        ))
    }
}

impl MultiScaleAttention {
    fn load(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        q_pool: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            dim_out,
            num_heads,
            q_pool,
            qkv: linear(dim, dim_out * 3, vb.pp("qkv"))?,
            proj: linear(dim_out, dim_out, vb.pp("proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, mut height, mut width, _) = x.dims4()?;
        let qkv = self.qkv.forward_fused_bias_cpu(x)?.reshape((
            batch,
            height * width,
            3usize,
            self.num_heads,
            (),
        ))?;
        let mut q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        if self.q_pool {
            if let Some(pooled) = pool_nhsd_query_cpu_f32(&q, height, width)? {
                q = pooled;
                height /= 2;
                width /= 2;
            } else {
                q = pool_nhwc(&q.reshape((batch, height, width, ()))?)?;
                height = q.dim(1)?;
                width = q.dim(2)?;
                q = q.reshape((batch, height * width, self.num_heads, ()))?;
            }
        }
        let head_dim = self.dim_out / self.num_heads;
        #[cfg(feature = "cpu-flash-attention")]
        if q.device().is_cpu() && q.dim(1)? >= 512 {
            let attended = flash_attn_nhsd_f32(&q, &k, &v, 1. / (head_dim as f32).sqrt())?;
            return self.proj.forward_fused_bias_cpu(&attended.reshape((
                batch,
                height,
                width,
                self.dim_out,
            ))?);
        }
        #[cfg(feature = "cpu-flash-attention")]
        if q.device().is_cpu() && batch > 1 {
            // Window partitioning makes each window a batch entry. This f32
            // NHSD kernel indexes those windows directly, avoiding both the
            // generic materialized attention tensors and varlen pack/unpack.
            let attended = flash_attn_nhsd_f32_batched(&q, &k, &v, 1. / (head_dim as f32).sqrt())?;
            return self.proj.forward_fused_bias_cpu(&attended.reshape((
                batch,
                height,
                width,
                self.dim_out,
            ))?);
        }
        self.proj.forward_fused_bias_cpu(
            &math_attention(&q, &k, &v, 1. / (head_dim as f64).sqrt())?.reshape((
                batch,
                height,
                width,
                self.dim_out,
            ))?,
        )
    }

    /// Synchronized attention subphase timings for comparison with PyTorch's
    /// qkv/proj module hooks. This is benchmark-only and keeps normal forward
    /// free of timers.
    fn forward_profiled(&self, x: &Tensor) -> Result<(Tensor, MultiScaleAttentionProfile)> {
        let (batch, mut height, mut width, _) = x.dims4()?;
        let start = Instant::now();
        let qkv = self.qkv.forward_fused_bias_cpu(x)?.reshape((
            batch,
            height * width,
            3usize,
            self.num_heads,
            (),
        ))?;
        qkv.device().synchronize()?;
        let qkv_ms = start.elapsed().as_secs_f64() * 1000.;
        let views_start = Instant::now();
        let mut q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        let qkv_views_ms = views_start.elapsed().as_secs_f64() * 1000.;
        let pool_start = Instant::now();
        let mut query_reshape_ms = 0.;
        let mut query_pool_core_ms = 0.;
        if self.q_pool {
            if let Some(pooled) = pool_nhsd_query_cpu_f32(&q, height, width)? {
                q = pooled;
                q.device().synchronize()?;
                query_pool_core_ms = pool_start.elapsed().as_secs_f64() * 1000.;
                height /= 2;
                width /= 2;
            } else {
                let query_image = q.reshape((batch, height, width, ()))?;
                query_image.device().synchronize()?;
                query_reshape_ms = pool_start.elapsed().as_secs_f64() * 1000.;
                let core_start = Instant::now();
                q = pool_nhwc(&query_image)?;
                q.device().synchronize()?;
                query_pool_core_ms = core_start.elapsed().as_secs_f64() * 1000.;
                height = q.dim(1)?;
                width = q.dim(2)?;
                q = q.reshape((batch, height * width, self.num_heads, ()))?;
            }
        }
        q.device().synchronize()?;
        let query_pool_ms = pool_start.elapsed().as_secs_f64() * 1000.;
        let head_dim = self.dim_out / self.num_heads;
        let start = Instant::now();
        #[cfg(feature = "cpu-flash-attention")]
        let attended =
            if q.device().is_cpu() && q.dim(1)? >= 512 {
                flash_attn_nhsd_f32(&q, &k, &v, 1. / (head_dim as f32).sqrt())?.reshape((
                    batch,
                    height,
                    width,
                    self.dim_out,
                ))?
            } else if q.device().is_cpu() && batch > 1 {
                flash_attn_nhsd_f32_batched(&q, &k, &v, 1. / (head_dim as f32).sqrt())?
                    .reshape((batch, height, width, self.dim_out))?
            } else {
                math_attention(&q, &k, &v, 1. / (head_dim as f64).sqrt())?.reshape((
                    batch,
                    height,
                    width,
                    self.dim_out,
                ))?
            };
        #[cfg(not(feature = "cpu-flash-attention"))]
        let attended = math_attention(&q, &k, &v, 1. / (head_dim as f64).sqrt())?.reshape((
            batch,
            height,
            width,
            self.dim_out,
        ))?;
        attended.device().synchronize()?;
        let kernel_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let output = self.proj.forward_fused_bias_cpu(&attended)?;
        output.device().synchronize()?;
        let projection_ms = start.elapsed().as_secs_f64() * 1000.;
        let cleanup_start = Instant::now();
        drop(attended);
        let cleanup_attended_ms = cleanup_start.elapsed().as_secs_f64() * 1000.;
        let qkv_cleanup_start = Instant::now();
        drop(q);
        drop(k);
        drop(v);
        drop(qkv);
        let cleanup_qkv_ms = qkv_cleanup_start.elapsed().as_secs_f64() * 1000.;
        let cleanup_ms = cleanup_start.elapsed().as_secs_f64() * 1000.;
        Ok((
            output,
            MultiScaleAttentionProfile {
                qkv_ms,
                qkv_views_ms,
                query_pool_ms,
                query_reshape_ms,
                query_pool_core_ms,
                kernel_ms,
                projection_ms,
                cleanup_ms,
                cleanup_attended_ms,
                cleanup_qkv_ms,
            },
        ))
    }
}

#[derive(Clone, Debug)]
struct MultiScaleBlock {
    dim: usize,
    dim_out: usize,
    norm1: LayerNorm,
    attn: MultiScaleAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    proj: Option<candle_nn::Linear>,
    q_pool: bool,
    window_size: usize,
}

#[derive(Clone, Debug)]
struct MultiScaleBlockProfile {
    norm1_ms: f64,
    shortcut_ms: f64,
    shortcut_pool_ms: f64,
    partition_ms: f64,
    attention_ms: f64,
    attention_qkv_ms: f64,
    attention_qkv_views_ms: f64,
    attention_query_pool_ms: f64,
    attention_query_reshape_ms: f64,
    attention_query_pool_core_ms: f64,
    attention_kernel_ms: f64,
    attention_projection_ms: f64,
    attention_cleanup_ms: f64,
    attention_cleanup_attended_ms: f64,
    attention_cleanup_qkv_ms: f64,
    unpartition_ms: f64,
    residual_add_ms: f64,
    norm2_ms: f64,
    mlp_ms: f64,
    mlp_expansion_ms: f64,
    mlp_activation_ms: f64,
    mlp_projection_ms: f64,
}

impl MultiScaleBlock {
    fn load(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        q_pool: bool,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            dim,
            dim_out,
            norm1: candle_nn::layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: MultiScaleAttention::load(dim, dim_out, num_heads, q_pool, vb.pp("attn"))?,
            norm2: candle_nn::layer_norm(dim_out, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::load(
                dim_out,
                dim_out * 4,
                dim_out,
                2,
                Activation::Gelu,
                false,
                vb.pp("mlp"),
            )?,
            proj: (dim != dim_out)
                .then(|| linear(dim, dim_out, vb.pp("proj")))
                .transpose()?,
            q_pool,
            window_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut shortcut = x.clone();
        let x = self.norm1.forward(x)?;
        if self.dim != self.dim_out {
            shortcut = pool_nhwc(&self.proj.as_ref().unwrap().forward_fused_bias_cpu(&x)?)?;
        }
        let (x, partition) = if self.window_size > 0 {
            let (windows, partition) = window_partition(&x, self.window_size)?;
            (windows, Some(partition))
        } else {
            (x, None)
        };
        let x = self.attn.forward(&x)?;
        let x = if self.window_size > 0 {
            let (window_size, partition) = if self.q_pool {
                let window_size = self.window_size / 2;
                let h = shortcut.dim(1)?;
                let w = shortcut.dim(2)?;
                let partition = WindowPartition {
                    padded_height: h + (window_size - h % window_size) % window_size,
                    padded_width: w + (window_size - w % window_size) % window_size,
                };
                (window_size, partition)
            } else {
                (self.window_size, partition.unwrap())
            };
            window_unpartition_add_residual(&x, window_size, partition, &shortcut)?
        } else {
            shortcut.broadcast_add(&x)?
        };
        self.mlp.forward_add_residual(&self.norm2.forward(&x)?, &x)
    }

    /// Time the same attention and MLP module boundaries as upstream PyTorch.
    fn forward_profiled(&self, x: &Tensor) -> Result<(Tensor, MultiScaleBlockProfile)> {
        let mut shortcut = x.clone();
        let phase_start = Instant::now();
        let x = self.norm1.forward(x)?;
        x.device().synchronize()?;
        let norm1_ms = phase_start.elapsed().as_secs_f64() * 1000.;
        let phase_start = Instant::now();
        let mut shortcut_pool_ms = 0.;
        if self.dim != self.dim_out {
            let projected = self.proj.as_ref().unwrap().forward_fused_bias_cpu(&x)?;
            let pool_start = Instant::now();
            shortcut = pool_nhwc(&projected)?;
            shortcut.device().synchronize()?;
            shortcut_pool_ms = pool_start.elapsed().as_secs_f64() * 1000.;
        }
        shortcut.device().synchronize()?;
        let shortcut_ms = phase_start.elapsed().as_secs_f64() * 1000.;
        let phase_start = Instant::now();
        let (x, partition) = if self.window_size > 0 {
            let (windows, partition) = window_partition(&x, self.window_size)?;
            (windows, Some(partition))
        } else {
            (x, None)
        };
        x.device().synchronize()?;
        let partition_ms = phase_start.elapsed().as_secs_f64() * 1000.;
        let attention_start = Instant::now();
        let (x, attention_profile) = self.attn.forward_profiled(&x)?;
        x.device().synchronize()?;
        let attention_ms = attention_start.elapsed().as_secs_f64() * 1000.;
        let phase_start = Instant::now();
        let x = if self.window_size > 0 {
            let (window_size, partition) = if self.q_pool {
                let window_size = self.window_size / 2;
                let h = shortcut.dim(1)?;
                let w = shortcut.dim(2)?;
                let partition = WindowPartition {
                    padded_height: h + (window_size - h % window_size) % window_size,
                    padded_width: w + (window_size - w % window_size) % window_size,
                };
                (window_size, partition)
            } else {
                (self.window_size, partition.unwrap())
            };
            window_unpartition_add_residual(&x, window_size, partition, &shortcut)?
        } else {
            x
        };
        x.device().synchronize()?;
        let unpartition_ms = phase_start.elapsed().as_secs_f64() * 1000.;
        let phase_start = Instant::now();
        let x = if self.window_size > 0 {
            x
        } else {
            shortcut.broadcast_add(&x)?
        };
        x.device().synchronize()?;
        let residual_add_ms = phase_start.elapsed().as_secs_f64() * 1000.;
        let phase_start = Instant::now();
        let normalized = self.norm2.forward(&x)?;
        normalized.device().synchronize()?;
        let norm2_ms = phase_start.elapsed().as_secs_f64() * 1000.;
        let mlp_start = Instant::now();
        let (x, mlp_profile) = self.mlp.forward_add_residual_profiled(&normalized, &x)?;
        x.device().synchronize()?;
        Ok((
            x,
            MultiScaleBlockProfile {
                norm1_ms,
                shortcut_ms,
                shortcut_pool_ms,
                partition_ms,
                attention_ms,
                attention_qkv_ms: attention_profile.qkv_ms,
                attention_qkv_views_ms: attention_profile.qkv_views_ms,
                attention_query_pool_ms: attention_profile.query_pool_ms,
                attention_query_reshape_ms: attention_profile.query_reshape_ms,
                attention_query_pool_core_ms: attention_profile.query_pool_core_ms,
                attention_kernel_ms: attention_profile.kernel_ms,
                attention_projection_ms: attention_profile.projection_ms,
                attention_cleanup_ms: attention_profile.cleanup_ms,
                attention_cleanup_attended_ms: attention_profile.cleanup_attended_ms,
                attention_cleanup_qkv_ms: attention_profile.cleanup_qkv_ms,
                unpartition_ms,
                residual_add_ms,
                norm2_ms,
                mlp_ms: mlp_start.elapsed().as_secs_f64() * 1000.,
                mlp_expansion_ms: mlp_profile.expansion_ms,
                mlp_activation_ms: mlp_profile.activation_ms,
                mlp_projection_ms: mlp_profile.projection_ms,
            },
        ))
    }
}

/// Hiera image trunk. Its `forward` returns stage features in fine-to-coarse
/// order, exactly as the reference loop appends them.
#[derive(Clone, Debug)]
pub struct Hiera {
    patch_embed: PatchEmbed,
    pos_embed: Tensor,
    pos_embed_window: Tensor,
    blocks: Vec<MultiScaleBlock>,
    stage_ends: [usize; 4],
    // Optional inference-only cache. It stores at most one shape so enabling
    // it cannot turn a variable-shape model into an unbounded tensor cache.
    position_cache: Arc<Mutex<Option<((usize, usize), Tensor)>>>,
}

/// Synchronized component timings for one Hiera image encoding.
#[derive(Clone, Debug)]
pub struct HieraProfile {
    pub setup_ms: f64,
    pub block_ms: Vec<f64>,
    pub norm1_ms: Vec<f64>,
    pub shortcut_ms: Vec<f64>,
    pub shortcut_pool_ms: Vec<f64>,
    pub partition_ms: Vec<f64>,
    pub attention_ms: Vec<f64>,
    pub attention_qkv_ms: Vec<f64>,
    pub attention_qkv_views_ms: Vec<f64>,
    pub attention_query_pool_ms: Vec<f64>,
    pub attention_query_reshape_ms: Vec<f64>,
    pub attention_query_pool_core_ms: Vec<f64>,
    pub attention_kernel_ms: Vec<f64>,
    pub attention_projection_ms: Vec<f64>,
    pub attention_cleanup_ms: Vec<f64>,
    pub attention_cleanup_attended_ms: Vec<f64>,
    pub attention_cleanup_qkv_ms: Vec<f64>,
    pub unpartition_ms: Vec<f64>,
    pub residual_add_ms: Vec<f64>,
    pub norm2_ms: Vec<f64>,
    pub mlp_ms: Vec<f64>,
    pub mlp_expansion_ms: Vec<f64>,
    pub mlp_activation_ms: Vec<f64>,
    pub mlp_projection_ms: Vec<f64>,
}

impl Hiera {
    pub fn load(config: &HieraConfig, vb: VarBuilder) -> Result<Self> {
        let depth: usize = config.stages.iter().sum();
        let stage_ends = [
            config.stages[0] - 1,
            config.stages[0] + config.stages[1] - 1,
            config.stages[0] + config.stages[1] + config.stages[2] - 1,
            depth - 1,
        ];
        let q_pool_blocks = [stage_ends[0] + 1, stage_ends[1] + 1, stage_ends[2] + 1];
        let mut blocks = Vec::with_capacity(depth);
        let mut dim = config.embed_dim;
        let mut heads = config.num_heads;
        let mut stage = 0;
        for index in 0..depth {
            let mut dim_out = dim;
            let mut window_size = config.window_spec[stage];
            if config.global_att_blocks.contains(&index) {
                window_size = 0;
            }
            if index > 0 && stage_ends[..3].contains(&(index - 1)) {
                dim_out *= 2;
                heads *= 2;
                stage += 1;
            }
            blocks.push(MultiScaleBlock::load(
                dim,
                dim_out,
                heads,
                q_pool_blocks.contains(&index),
                window_size,
                vb.pp(format!("blocks.{index}")),
            )?);
            dim = dim_out;
        }
        Ok(Self {
            patch_embed: PatchEmbed::load(3, config.embed_dim, 7, 4, 3, vb.pp("patch_embed"))?,
            pos_embed: vb.get(
                (
                    1,
                    config.embed_dim,
                    config.window_pos_embed_bkg_spatial_size[0],
                    config.window_pos_embed_bkg_spatial_size[1],
                ),
                "pos_embed",
            )?,
            pos_embed_window: vb.get(
                (
                    1,
                    config.embed_dim,
                    config.window_spec[0],
                    config.window_spec[0],
                ),
                "pos_embed_window",
            )?,
            blocks,
            stage_ends,
            position_cache: Arc::new(Mutex::new(None)),
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Vec<Tensor>> {
        let mut x = self.patch_embed.forward(image)?;
        x = x.broadcast_add(&self.position_embedding(x.dim(1)?, x.dim(2)?)?)?;
        let mut output = Vec::with_capacity(4);
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if self.stage_ends.contains(&index) {
                output.push(x.permute((0, 3, 1, 2))?);
            }
        }
        Ok(output)
    }

    /// Runs Hiera while recording the patch/position setup and every block.
    /// Synchronization makes these timings valid for either CPU or CUDA.
    pub fn forward_profiled(&self, image: &Tensor) -> Result<(Vec<Tensor>, HieraProfile)> {
        let start = Instant::now();
        let mut x = self.patch_embed.forward(image)?;
        x = x.broadcast_add(&self.position_embedding(x.dim(1)?, x.dim(2)?)?)?;
        x.device().synchronize()?;
        let setup_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut output = Vec::with_capacity(4);
        let mut block_ms = Vec::with_capacity(self.blocks.len());
        let mut norm1_ms = Vec::with_capacity(self.blocks.len());
        let mut shortcut_ms = Vec::with_capacity(self.blocks.len());
        let mut shortcut_pool_ms = Vec::with_capacity(self.blocks.len());
        let mut partition_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_qkv_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_qkv_views_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_query_pool_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_query_reshape_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_query_pool_core_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_kernel_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_projection_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_cleanup_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_cleanup_attended_ms = Vec::with_capacity(self.blocks.len());
        let mut attention_cleanup_qkv_ms = Vec::with_capacity(self.blocks.len());
        let mut unpartition_ms = Vec::with_capacity(self.blocks.len());
        let mut residual_add_ms = Vec::with_capacity(self.blocks.len());
        let mut norm2_ms = Vec::with_capacity(self.blocks.len());
        let mut mlp_ms = Vec::with_capacity(self.blocks.len());
        let mut mlp_expansion_ms = Vec::with_capacity(self.blocks.len());
        let mut mlp_activation_ms = Vec::with_capacity(self.blocks.len());
        let mut mlp_projection_ms = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let start = Instant::now();
            let (next, profile) = block.forward_profiled(&x)?;
            x = next;
            x.device().synchronize()?;
            block_ms.push(start.elapsed().as_secs_f64() * 1000.);
            norm1_ms.push(profile.norm1_ms);
            shortcut_ms.push(profile.shortcut_ms);
            shortcut_pool_ms.push(profile.shortcut_pool_ms);
            partition_ms.push(profile.partition_ms);
            attention_ms.push(profile.attention_ms);
            attention_qkv_ms.push(profile.attention_qkv_ms);
            attention_qkv_views_ms.push(profile.attention_qkv_views_ms);
            attention_query_pool_ms.push(profile.attention_query_pool_ms);
            attention_query_reshape_ms.push(profile.attention_query_reshape_ms);
            attention_query_pool_core_ms.push(profile.attention_query_pool_core_ms);
            attention_kernel_ms.push(profile.attention_kernel_ms);
            attention_projection_ms.push(profile.attention_projection_ms);
            attention_cleanup_ms.push(profile.attention_cleanup_ms);
            attention_cleanup_attended_ms.push(profile.attention_cleanup_attended_ms);
            attention_cleanup_qkv_ms.push(profile.attention_cleanup_qkv_ms);
            unpartition_ms.push(profile.unpartition_ms);
            residual_add_ms.push(profile.residual_add_ms);
            norm2_ms.push(profile.norm2_ms);
            mlp_ms.push(profile.mlp_ms);
            mlp_expansion_ms.push(profile.mlp_expansion_ms);
            mlp_activation_ms.push(profile.mlp_activation_ms);
            mlp_projection_ms.push(profile.mlp_projection_ms);
            if self.stage_ends.contains(&index) {
                output.push(x.permute((0, 3, 1, 2))?);
            }
        }
        Ok((
            output,
            HieraProfile {
                setup_ms,
                block_ms,
                norm1_ms,
                shortcut_ms,
                shortcut_pool_ms,
                partition_ms,
                attention_ms,
                attention_qkv_ms,
                attention_qkv_views_ms,
                attention_query_pool_ms,
                attention_query_reshape_ms,
                attention_query_pool_core_ms,
                attention_kernel_ms,
                attention_projection_ms,
                attention_cleanup_ms,
                attention_cleanup_attended_ms,
                attention_cleanup_qkv_ms,
                unpartition_ms,
                residual_add_ms,
                norm2_ms,
                mlp_ms,
                mlp_expansion_ms,
                mlp_activation_ms,
                mlp_projection_ms,
            },
        ))
    }

    fn position_embedding(&self, height: usize, width: usize) -> Result<Tensor> {
        let use_cache =
            std::env::var_os("SAM2_STATIC_POSITION_CACHE").is_some_and(|value| value == "1");
        if use_cache {
            let cache = self.position_cache.lock().map_err(|_| {
                candle_core::Error::Msg("Hiera position cache mutex poisoned".into())
            })?;
            if let Some(((cached_height, cached_width), position)) = &*cache {
                if (*cached_height, *cached_width) == (height, width) {
                    return Ok(position.clone());
                }
            }
        }
        let position = self.position_embedding_uncached(height, width)?;
        if use_cache {
            let mut cache = self.position_cache.lock().map_err(|_| {
                candle_core::Error::Msg("Hiera position cache mutex poisoned".into())
            })?;
            // The first observed shape establishes the bounded static cache.
            if cache.is_none() {
                *cache = Some(((height, width), position.clone()));
            }
        }
        Ok(position)
    }

    fn position_embedding_uncached(&self, height: usize, width: usize) -> Result<Tensor> {
        // PyTorch F.interpolate(..., mode="bicubic", align_corners=False).
        let pos = bicubic_resize(&self.pos_embed, height, width)?;
        let repeats_h = height / self.pos_embed_window.dim(2)?;
        let repeats_w = width / self.pos_embed_window.dim(3)?;
        let tiled = self.pos_embed_window.repeat((1, 1, repeats_h, repeats_w))?;
        pos.broadcast_add(&tiled)?.permute((0, 2, 3, 1))
    }
}

fn pool_nhwc(x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && x.dtype() == DType::F32 {
        let (batch, height, width, channels) = x.dims4()?;
        if height > 0 && width > 0 && height % 2 == 0 && width % 2 == 0 && x.stride()[3] == 1 {
            return x.apply_op1_no_bwd(&CudaNhwcMaxPool {
                height,
                width,
                channels,
                stride_y: x.stride()[1],
                stride_x: x.stride()[2],
                output_shape: candle_core::Shape::from_dims(&[
                    batch,
                    height / 2,
                    width / 2,
                    channels,
                ]),
            });
        }
    }
    if let Some(pooled) = pool_nhwc_cpu_f32(x)? {
        return Ok(pooled);
    }
    pool_nhwc_generic(x)
}

fn pool_nhwc_generic(x: &Tensor) -> Result<Tensor> {
    x.permute((0, 3, 1, 2))?
        .max_pool2d_with_stride(2, 2)?
        .permute((0, 2, 3, 1))
}

/// Pool Q directly from the interleaved QKV projection view. Reshaping that
/// view into NHWC first materializes its strided pixels; the pool needs only
/// four source pixel reads per output and can write final NHSD storage once.
fn pool_nhsd_query_cpu_f32(q: &Tensor, height: usize, width: usize) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda()
        && q.dtype() == DType::F32
        && height > 0
        && width > 0
        && height % 2 == 0
        && width % 2 == 0
        && q.stride()[3] == 1
    {
        let (batch, pixels, heads, head_dim) = q.dims4()?;
        if pixels == height * width && q.stride()[2] == head_dim {
            let pooled = q.apply_op1_no_bwd(&CudaNhwcMaxPool {
                height,
                width,
                channels: heads * head_dim,
                stride_y: width * q.stride()[1],
                stride_x: q.stride()[1],
                output_shape: candle_core::Shape::from_dims(&[
                    batch,
                    height / 2 * (width / 2),
                    heads,
                    head_dim,
                ]),
            })?;
            return Ok(Some(pooled));
        }
    }
    if !q.device().is_cpu()
        || q.dtype() != DType::F32
        || height == 0
        || width == 0
        || height % 2 != 0
        || width % 2 != 0
        || std::env::var_os("SAM2_CPU_HIERA_DIRECT_QUERY_POOL").is_some_and(|value| value == "0")
    {
        return Ok(None);
    }
    let (batch, pixels, heads, head_dim) = q.dims4()?;
    if batch == 0 || heads == 0 || head_dim == 0 || pixels != height * width {
        return Ok(None);
    }
    let stride = q.stride();
    if stride[3] != 1 || stride[2] != head_dim {
        return Ok(None);
    }
    let (storage, layout) = q.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        return Ok(None);
    };
    let input = storage.as_slice::<f32>()?;
    let start = layout.start_offset();
    let channels = heads * head_dim;
    let Some(last_input) = (batch - 1)
        .checked_mul(stride[0])
        .and_then(|offset| {
            (pixels - 1)
                .checked_mul(stride[1])
                .and_then(|pixel| offset.checked_add(pixel))
        })
        .and_then(|offset| offset.checked_add(channels - 1))
        .and_then(|offset| offset.checked_add(start))
    else {
        return Ok(None);
    };
    if last_input >= input.len() {
        return Ok(None);
    }
    let out_height = height / 2;
    let out_width = width / 2;
    let out_pixels = out_height * out_width;
    let pixels_per_task = std::env::var("SAM2_CPU_HIERA_POOL_PIXELS_PER_TASK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| (1..=256).contains(&value))
        .unwrap_or(32);
    let mut output = vec![0f32; batch * out_pixels * channels];
    #[cfg(target_arch = "x86_64")]
    let avx512 = std::is_x86_feature_detected!("avx512f");
    output
        .par_chunks_mut(channels * pixels_per_task)
        .enumerate()
        .for_each(|(chunk_index, destinations)| {
            for (local_pixel, destination) in destinations.chunks_exact_mut(channels).enumerate() {
                let output_pixel = chunk_index * pixels_per_task + local_pixel;
                let batch_index = output_pixel / out_pixels;
                let pixel = output_pixel % out_pixels;
                let input_pixel = (pixel / out_width * 2) * width + (pixel % out_width * 2);
                let top_left = start + batch_index * stride[0] + input_pixel * stride[1];
                let top_right = top_left + stride[1];
                let bottom_left = top_left + width * stride[1];
                let bottom_right = bottom_left + stride[1];
                #[cfg(target_arch = "x86_64")]
                if avx512 {
                    // SAFETY: the contiguous channel layout and last-input
                    // bound above cover every load; workers own disjoint
                    // output slices, and AVX-512F is checked at runtime.
                    unsafe {
                        pool_nhwc_channels_avx512(
                            input.as_ptr().add(top_left),
                            input.as_ptr().add(top_right),
                            input.as_ptr().add(bottom_left),
                            input.as_ptr().add(bottom_right),
                            destination.as_mut_ptr(),
                            channels,
                        );
                    }
                    continue;
                }
                for channel in 0..channels {
                    destination[channel] = input[top_left + channel]
                        .max(input[top_right + channel])
                        .max(input[bottom_left + channel])
                        .max(input[bottom_right + channel]);
                }
            }
        });
    Ok(Some(Tensor::from_vec(
        output,
        (batch, out_pixels, heads, head_dim),
        q.device(),
    )?))
}

/// Fixed Hiera query pooling, keeping the model's NHWC layout contiguous.
///
/// The generic composition produces NCHW physical storage and returns an NHWC
/// strided view. This f32 CPU path reads any last-dimension-contiguous NHWC
/// layout and writes the final `(B,H/2,W/2,C)` storage directly.
fn pool_nhwc_cpu_f32(x: &Tensor) -> Result<Option<Tensor>> {
    if !x.device().is_cpu() || x.dtype() != DType::F32 {
        return Ok(None);
    }
    let (batch, height, width, channels) = x.dims4()?;
    if height == 0 || width == 0 || height % 2 != 0 || width % 2 != 0 || x.stride()[3] != 1 {
        return Ok(None);
    }
    let (storage, layout) = x.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        return Ok(None);
    };
    let data = storage.as_slice::<f32>()?;
    let stride = x.stride();
    let start = layout.start_offset();
    let out_height = height / 2;
    let out_width = width / 2;
    let mut output = vec![0f32; batch * out_height * out_width * channels];
    // A pixel contains only a few hundred floats. Amortize Rayon scheduling
    // across adjacent pixels while sharing the input storage read-only and
    // assigning disjoint ranges of the one output buffer to workers.
    let pixels_per_task = std::env::var("SAM2_CPU_HIERA_POOL_PIXELS_PER_TASK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| (1..=256).contains(&value))
        .unwrap_or(32);
    #[cfg(target_arch = "x86_64")]
    let avx512 = std::is_x86_feature_detected!("avx512f");
    output
        .par_chunks_mut(channels * pixels_per_task)
        .enumerate()
        .for_each(|(chunk_index, destinations)| {
            for (local_pixel, destination) in destinations.chunks_exact_mut(channels).enumerate() {
                let output_pixel = chunk_index * pixels_per_task + local_pixel;
                let batch_index = output_pixel / (out_height * out_width);
                let pixel = output_pixel % (out_height * out_width);
                let out_y = pixel / out_width;
                let out_x = pixel % out_width;
                let input_y = out_y * 2;
                let input_x = out_x * 2;
                let top_left =
                    start + batch_index * stride[0] + input_y * stride[1] + input_x * stride[2];
                #[cfg(target_arch = "x86_64")]
                if avx512 {
                    // SAFETY: AVX-512F is checked above; each 16-f32 load
                    // stays within a pixel and output tiles are disjoint.
                    unsafe {
                        pool_nhwc_channels_avx512(
                            data.as_ptr().add(top_left),
                            data.as_ptr().add(top_left + stride[2]),
                            data.as_ptr().add(top_left + stride[1]),
                            data.as_ptr().add(top_left + stride[1] + stride[2]),
                            destination.as_mut_ptr(),
                            channels,
                        );
                    }
                    continue;
                }
                for channel in 0..channels {
                    destination[channel] = data[top_left + channel]
                        .max(data[top_left + stride[2] + channel])
                        .max(data[top_left + stride[1] + channel])
                        .max(data[top_left + stride[1] + stride[2] + channel]);
                }
            }
        });
    Ok(Some(Tensor::from_vec(
        output,
        (batch, out_height, out_width, channels),
        x.device(),
    )?))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pool_nhwc_channels_avx512(
    top_left: *const f32,
    top_right: *const f32,
    bottom_left: *const f32,
    bottom_right: *const f32,
    output: *mut f32,
    channels: usize,
) {
    use std::arch::x86_64::*;
    // SAFETY: caller validates CPU support and all source/destination ranges.
    unsafe {
        let vector_channels = channels / 16 * 16;
        let mut channel = 0;
        while channel < vector_channels {
            let top = _mm512_max_ps(
                _mm512_loadu_ps(top_left.add(channel)),
                _mm512_loadu_ps(top_right.add(channel)),
            );
            let bottom = _mm512_max_ps(
                _mm512_loadu_ps(bottom_left.add(channel)),
                _mm512_loadu_ps(bottom_right.add(channel)),
            );
            _mm512_storeu_ps(output.add(channel), _mm512_max_ps(top, bottom));
            channel += 16;
        }
        while channel < channels {
            *output.add(channel) = (*top_left.add(channel))
                .max(*top_right.add(channel))
                .max(*bottom_left.add(channel))
                .max(*bottom_right.add(channel));
            channel += 1;
        }
    }
}

fn cubic_weight(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1. {
        // PyTorch bicubic interpolation uses Keys' cubic convolution with
        // a=-0.75 (not Catmull-Rom's a=-0.5).
        (1.25 * x - 2.25) * x * x + 1.
    } else if x < 2. {
        ((-0.75 * x + 3.75) * x - 6.) * x + 3.
    } else {
        0.
    }
}

fn bicubic_resize(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    if let Some(output) = bicubic_resize_cpu_f32(x, out_h, out_w)? {
        return Ok(output);
    }
    bicubic_resize_generic(x, out_h, out_w)
}

fn bicubic_resize_generic(x: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (_, _, in_h, in_w) = x.dims4()?;
    let (indices_h, weights_h) = cubic_indices(in_h, out_h, x)?;
    let (indices_w, weights_w) = cubic_indices(in_w, out_w, x)?;
    let mut rows = Tensor::zeros((x.dim(0)?, x.dim(1)?, out_h, in_w), x.dtype(), x.device())?;
    for n in 0..4 {
        rows = rows.broadcast_add(
            &x.index_select(&indices_h[n], 2)?
                .broadcast_mul(&weights_h[n].reshape((1, 1, out_h, 1))?)?,
        )?;
    }
    let mut output = Tensor::zeros((x.dim(0)?, x.dim(1)?, out_h, out_w), x.dtype(), x.device())?;
    for n in 0..4 {
        output = output.broadcast_add(
            &rows
                .index_select(&indices_w[n], 3)?
                .broadcast_mul(&weights_w[n].reshape((1, 1, 1, out_w))?)?,
        )?;
    }
    Ok(output)
}

/// Fused CPU equivalent of PyTorch bicubic interpolate with `align_corners=false`.
///
/// The generic implementation above materializes four vertical terms and four
/// horizontal terms. For contiguous f32 CPU position embeddings, compute the
/// same separable tap order directly into the sole final output allocation.
fn bicubic_resize_cpu_f32(x: &Tensor, out_h: usize, out_w: usize) -> Result<Option<Tensor>> {
    if x.dtype() != DType::F32 {
        return Ok(None);
    }
    let (storage, layout) = x.storage_and_layout();
    let Some((start, end)) = layout.contiguous_offsets() else {
        return Ok(None);
    };
    let input = match &*storage {
        Storage::Cpu(storage) => &storage.as_slice::<f32>()?[start..end],
        _ => return Ok(None),
    };
    let (batch, channels, in_h, in_w) = x.dims4()?;
    let (indices_h, weights_h) = cubic_coefficients(in_h, out_h);
    let (indices_w, weights_w) = cubic_coefficients(in_w, out_w);
    let mut output = vec![0f32; batch * channels * out_h * out_w];
    output.par_chunks_mut(out_w).enumerate().for_each_init(
        || vec![0f32; in_w],
        |vertical, (row, output_row)| {
            let channel_row = row / out_h;
            let out_y = row % out_h;
            let input_channel = channel_row * in_h * in_w;
            // Vertical taps depend on output Y and source X, but not output
            // X. Reuse this small row buffer across all horizontal outputs;
            // it removes the repeated four-tap vertical interpolation without
            // introducing an output-sized separable-interpolation temporary.
            for (source_x, vertical) in vertical.iter_mut().enumerate() {
                let mut value = 0f32;
                for tap_h in 0..4 {
                    value += input[input_channel + indices_h[tap_h][out_y] * in_w + source_x]
                        * weights_h[tap_h][out_y];
                }
                *vertical = value;
            }
            for (out_x, output) in output_row.iter_mut().enumerate() {
                let mut value = 0f32;
                // Match the generic implementation's accumulation order:
                // accumulate vertical rows first, then the horizontal taps.
                for tap_w in 0..4 {
                    value += vertical[indices_w[tap_w][out_x]] * weights_w[tap_w][out_x];
                }
                *output = value;
            }
        },
    );
    Ok(Some(Tensor::from_vec(
        output,
        (batch, channels, out_h, out_w),
        x.device(),
    )?))
}

fn cubic_indices(input: usize, output: usize, x: &Tensor) -> Result<([Tensor; 4], [Tensor; 4])> {
    let (indices, weights) = cubic_coefficients(input, output);
    Ok((
        std::array::from_fn(|n| {
            Tensor::from_vec(
                indices[n]
                    .iter()
                    .map(|&index| index as u32)
                    .collect::<Vec<_>>(),
                output,
                x.device(),
            )
            .unwrap()
        }),
        std::array::from_fn(|n| Tensor::from_vec(weights[n].clone(), output, x.device()).unwrap()),
    ))
}

fn cubic_coefficients(input: usize, output: usize) -> ([Vec<usize>; 4], [Vec<f32>; 4]) {
    let mut indices: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::with_capacity(output));
    let mut weights: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(output));
    for out in 0..output {
        let source = (out as f64 + 0.5) * input as f64 / output as f64 - 0.5;
        let base = source.floor() as isize;
        for n in 0..4 {
            let index = (base + n as isize - 1).clamp(0, input as isize - 1) as usize;
            indices[n].push(index);
            weights[n].push(cubic_weight(source - (base + n as isize - 1) as f64) as f32);
        }
    }
    (indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fused_bicubic_matches_generic_tensor_pipeline() -> Result<()> {
        let input = Tensor::from_vec(
            (0..(2 * 3 * 4))
                .map(|value| value as f32 / 7.)
                .collect::<Vec<_>>(),
            (1, 2, 3, 4),
            &Device::Cpu,
        )?;
        let fused = bicubic_resize_cpu_f32(&input, 7, 9)?.unwrap();
        let generic = bicubic_resize_generic(&input, 7, 9)?;
        let fused = fused.flatten_all()?.to_vec1::<f32>()?;
        let generic = generic.flatten_all()?.to_vec1::<f32>()?;
        let max_error = fused
            .iter()
            .zip(&generic)
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(max_error <= 1e-6, "max error {max_error}");
        Ok(())
    }

    #[test]
    fn fused_nhwc_pool_matches_generic_layout_pipeline() -> Result<()> {
        for (batch, height, width, channels) in [(1, 8, 6, 5), (2, 66, 66, 33)] {
            let input = Tensor::randn(0f32, 1f32, (batch, height, width, channels), &Device::Cpu)?;
            let expected = pool_nhwc_generic(&input)?;
            let actual = pool_nhwc(&input)?;
            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            let max_error = expected
                .iter()
                .zip(actual.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0f32, f32::max);
            assert!(max_error <= 1e-6, "max error {max_error}");
            assert_eq!(actual.len(), batch * height / 2 * width / 2 * channels);
        }
        Ok(())
    }

    #[test]
    fn direct_interleaved_query_pool_matches_materialized_pipeline() -> Result<()> {
        for (batch, height, width, heads, head_dim) in [(2, 8, 6, 4, 8), (1, 66, 66, 3, 11)] {
            let qkv = Tensor::randn(
                0f32,
                1f32,
                (batch, height * width, 3, heads, head_dim),
                &Device::Cpu,
            )?;
            let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
            let materialized = q.reshape((batch, height, width, heads * head_dim))?;
            let expected = pool_nhwc_generic(&materialized)?.reshape((
                batch,
                height / 2 * width / 2,
                heads,
                head_dim,
            ))?;
            let actual = pool_nhsd_query_cpu_f32(&q, height, width)?.unwrap();
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }
}
