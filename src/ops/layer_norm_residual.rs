//! CPU inference fusion for a residual add followed by token LayerNorm.

use candle_core::{DType, Result, Storage, Tensor};
use candle_nn::{LayerNorm, Module};
use rayon::prelude::*;

/// Preserve the usual tensor operations for unsupported devices and layouts.
/// The specialized case writes each 256-channel token once, with only a
/// worker-local stack row, instead of allocating an image-sized sum tensor.
pub fn layer_norm_add_residual(norm: &LayerNorm, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
    // The pinned four-process cached-prompt suite establishes the fused path
    // as the CPU default. Keep the ordinary composition as an attribution and
    // portability fallback; unsupported devices/layouts also use it below.
    if std::env::var_os("SAM2_CPU_DECODER_FUSED_RESIDUAL_NORM").is_some_and(|value| value == "0") {
        return norm.forward(&x.broadcast_add(residual)?);
    }
    if let Some(output) = layer_norm_add_residual_cpu(norm, x, residual)? {
        return Ok(output);
    }
    #[cfg(feature = "cuda")]
    if std::env::var_os("SAM2_CUDA_DECODER_FUSED_RESIDUAL_NORM").is_none_or(|value| value != "0")
        && let Some(output) = layer_norm_add_residual_cuda(norm, x, residual)?
    {
        return Ok(output);
    }
    norm.forward(&x.broadcast_add(residual)?)
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn sam2_layer_norm_residual_f32(
        x: *const f32,
        residual: *const f32,
        weight: *const f32,
        bias: *const f32,
        output: *mut f32,
        rows: i32,
        columns: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaLayerNormResidual<'a> {
    norm: &'a LayerNorm,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp2 for CudaLayerNormResidual<'_> {
    fn name(&self) -> &'static str {
        "sam2-layer-norm-residual-f32"
    }

    fn cpu_fwd(
        &self,
        _x: &candle_core::CpuStorage,
        _x_layout: &candle_core::Layout,
        _residual: &candle_core::CpuStorage,
        _residual_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("fused residual LayerNorm is only implemented for CUDA")
    }

    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        x_layout: &candle_core::Layout,
        residual: &candle_core::CudaStorage,
        residual_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
        use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};

        let Some(bias_tensor) = self.norm.bias() else {
            candle_core::bail!("fused residual LayerNorm requires bias")
        };
        let (weight_storage, weight_layout) = self.norm.weight().storage_and_layout();
        let (bias_storage, bias_layout) = bias_tensor.storage_and_layout();
        let (Storage::Cuda(weight), Storage::Cuda(bias)) = (&*weight_storage, &*bias_storage)
        else {
            candle_core::bail!("fused residual LayerNorm parameters must be on CUDA")
        };
        let columns = *x_layout.shape().dims().last().ok_or_else(|| {
            candle_core::Error::Msg("LayerNorm requires a non-scalar input".into())
        })?;
        let elements = x_layout.shape().elem_count();
        if x.dtype() != DType::F32
            || residual.dtype() != DType::F32
            || weight.dtype() != DType::F32
            || bias.dtype() != DType::F32
            || x_layout.shape() != residual_layout.shape()
            || columns != 256
            || self.norm.weight().elem_count() != columns
            || bias_tensor.elem_count() != columns
        {
            candle_core::bail!("unsupported fused residual LayerNorm input")
        }
        let Some((x_start, _)) = x_layout.contiguous_offsets() else {
            candle_core::bail!("fused residual LayerNorm requires contiguous input")
        };
        let Some((residual_start, _)) = residual_layout.contiguous_offsets() else {
            candle_core::bail!("fused residual LayerNorm requires contiguous residual")
        };
        let Some((weight_start, _)) = weight_layout.contiguous_offsets() else {
            candle_core::bail!("fused residual LayerNorm requires contiguous weight")
        };
        let Some((bias_start, _)) = bias_layout.contiguous_offsets() else {
            candle_core::bail!("fused residual LayerNorm requires contiguous bias")
        };
        let device = x.device();
        let stream = device.cuda_stream();
        let x = x.as_cuda_slice::<f32>()?.slice(x_start..);
        let residual = residual.as_cuda_slice::<f32>()?.slice(residual_start..);
        let weight = weight.as_cuda_slice::<f32>()?.slice(weight_start..);
        let bias = bias.as_cuda_slice::<f32>()?.slice(bias_start..);
        // SAFETY: The kernel writes every element of the validated input shape.
        let mut output = unsafe { device.alloc::<f32>(elements)? };
        let (x_ptr, x_guard) = x.device_ptr(&stream);
        let (residual_ptr, residual_guard) = residual.device_ptr(&stream);
        let (weight_ptr, weight_guard) = weight.device_ptr(&stream);
        let (bias_ptr, bias_guard) = bias.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            sam2_layer_norm_residual_f32(
                x_ptr as *const f32,
                residual_ptr as *const f32,
                weight_ptr as *const f32,
                bias_ptr as *const f32,
                output_ptr as *mut f32,
                (elements / columns) as i32,
                columns as i32,
                self.norm.eps() as f32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if status != 0 {
            candle_core::bail!("fused residual LayerNorm CUDA launch failed with status {status}")
        }
        drop((
            x_guard,
            residual_guard,
            weight_guard,
            bias_guard,
            output_guard,
        ));
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            x_layout.shape().clone(),
        ))
    }
}

#[cfg(feature = "cuda")]
fn layer_norm_add_residual_cuda(
    norm: &LayerNorm,
    x: &Tensor,
    residual: &Tensor,
) -> Result<Option<Tensor>> {
    let Some(bias) = norm.bias() else {
        return Ok(None);
    };
    if !x.device().is_cuda()
        || x.dtype() != DType::F32
        || residual.dtype() != DType::F32
        || norm.weight().dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !norm.remove_mean()
        || x.dims() != residual.dims()
        || x.dims().last() != Some(&256)
        || x.elem_count() < 4096
        || x.layout().contiguous_offsets().is_none()
        || residual.layout().contiguous_offsets().is_none()
    {
        return Ok(None);
    }
    Ok(Some(x.apply_op2_no_bwd(
        residual,
        &CudaLayerNormResidual { norm },
    )?))
}

fn layer_norm_add_residual_cpu(
    norm: &LayerNorm,
    x: &Tensor,
    residual: &Tensor,
) -> Result<Option<Tensor>> {
    let Some(bias) = norm.bias() else {
        return Ok(None);
    };
    if !x.device().is_cpu()
        || x.dtype() != DType::F32
        || residual.dtype() != DType::F32
        || norm.weight().dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !norm.remove_mean()
        || x.dims() != residual.dims()
        || x.dims().last() != Some(&256)
        || x.elem_count() < 4096
    {
        return Ok(None);
    }
    let (Some((x_start, x_end)), Some((residual_start, residual_end))) = (
        x.layout().contiguous_offsets(),
        residual.layout().contiguous_offsets(),
    ) else {
        return Ok(None);
    };
    let (Some((weight_start, weight_end)), Some((bias_start, bias_end))) = (
        norm.weight().layout().contiguous_offsets(),
        bias.layout().contiguous_offsets(),
    ) else {
        return Ok(None);
    };
    let (x_storage, _) = x.storage_and_layout();
    let (residual_storage, _) = residual.storage_and_layout();
    let (weight_storage, _) = norm.weight().storage_and_layout();
    let (bias_storage, _) = bias.storage_and_layout();
    let (
        Storage::Cpu(x_storage),
        Storage::Cpu(residual_storage),
        Storage::Cpu(weight_storage),
        Storage::Cpu(bias_storage),
    ) = (
        &*x_storage,
        &*residual_storage,
        &*weight_storage,
        &*bias_storage,
    )
    else {
        return Ok(None);
    };
    let x_data = &x_storage.as_slice::<f32>()?[x_start..x_end];
    let residual_data = &residual_storage.as_slice::<f32>()?[residual_start..residual_end];
    let weight = &weight_storage.as_slice::<f32>()?[weight_start..weight_end];
    let bias = &bias_storage.as_slice::<f32>()?[bias_start..bias_end];
    if weight.len() != 256 || bias.len() != 256 || x_data.len() != residual_data.len() {
        return Ok(None);
    }
    let mut output = vec![0f32; x_data.len()];
    let eps = norm.eps() as f32;
    // Give each Rayon task a run of rows instead of scheduling one tiny task
    // per token. All three chunks are views of the original storage; only the
    // final output is allocated, and each worker writes disjoint rows.
    const ROWS_PER_TASK: usize = 32;
    output
        .par_chunks_mut(256 * ROWS_PER_TASK)
        .zip(x_data.par_chunks(256 * ROWS_PER_TASK))
        .zip(residual_data.par_chunks(256 * ROWS_PER_TASK))
        .for_each(|((dst_rows, x_rows), residual_rows)| {
            for ((dst, x_row), residual_row) in dst_rows
                .chunks_exact_mut(256)
                .zip(x_rows.chunks_exact(256))
                .zip(residual_rows.chunks_exact(256))
            {
                let mut row = [0f32; 256];
                let mut sum = 0f32;
                let mut sum2 = 0f32;
                for index in 0..256 {
                    let value = x_row[index] + residual_row[index];
                    row[index] = value;
                    sum += value;
                    sum2 += value * value;
                }
                let mean = sum / 256f32;
                let variance = sum2 / 256f32 - mean * mean;
                let inv_std = (variance + eps).sqrt().recip();
                for index in 0..256 {
                    dst[index] = (row[index] - mean) * inv_std * weight[index] + bias[index];
                }
            }
        });
    Ok(Some(Tensor::from_vec(output, x.shape(), x.device())?))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};
    use candle_nn::{LayerNorm, Module};

    use super::layer_norm_add_residual_cpu;

    #[test]
    fn fused_residual_norm_matches_composed_cpu_path() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..3 * 16 * 256)
                .map(|index| (index as f32 * 0.002).sin())
                .collect(),
            (3, 16, 256),
            &device,
        )?;
        let residual = Tensor::from_vec(
            (0..3 * 16 * 256)
                .map(|index| (index as f32 * 0.007).cos())
                .collect(),
            (3, 16, 256),
            &device,
        )?;
        let weight = Tensor::from_vec(
            (0..256).map(|index| 0.5f32 + index as f32 / 512.).collect(),
            256,
            &device,
        )?;
        let bias = Tensor::from_vec(
            (0..256).map(|index| index as f32 / 1024. - 0.125).collect(),
            256,
            &device,
        )?;
        let norm = LayerNorm::new(weight, bias, 1e-5);
        let expected = norm.forward(&x.broadcast_add(&residual)?)?;
        let actual = layer_norm_add_residual_cpu(&norm, &x, &residual)?.unwrap();
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
