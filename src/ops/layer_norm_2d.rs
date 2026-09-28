//! Translation of `LayerNorm2d` in `sam2_utils.py`.

use candle_core::{DType, Result, Storage, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*;

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
        if let Some(output) = self.forward_cpu_f32(x, false)? {
            return Ok(output);
        }
        if let Some(output) = self.forward_cuda_channels_last(x)? {
            return Ok(output);
        }
        self.forward_generic(x)
    }

    /// Normalize and apply exact GELU while the CPU output buffer is already
    /// owned. This preserves channels-last storage and avoids a second full
    /// tensor allocation for the mask upscaler.
    pub fn forward_gelu(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var_os("SAM2_CPU_LAYERNORM_GELU_FUSED").is_none_or(|value| value != "0")
            && let Some(output) = self.forward_cpu_f32(x, true)?
        {
            return Ok(output);
        }
        self.forward(x)?.gelu_erf()
    }

    /// CUDA's fused LayerNorm reduces the contiguous last dimension. Pack one
    /// channels-last copy, normalize it in one kernel, then expose the result
    /// as a stride-only NCHW view. The following cuDNN convolution accepts that
    /// layout directly, avoiding the generic path's seven image-sized tensors.
    fn forward_cuda_channels_last(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if !x.device().is_cuda()
            || std::env::var_os("SAM2_CUDA_LAYER_NORM_2D_LAST_DIM")
                .is_some_and(|value| value == "0")
        {
            return Ok(None);
        }
        let (_, channels, _, _) = x.dims4()?;
        if self.weight.dims1()? != channels || self.bias.dims1()? != channels {
            return Ok(None);
        }
        let channels_last = x.permute((0, 2, 3, 1))?.contiguous()?;
        let normalized =
            candle_nn::ops::layer_norm(&channels_last, &self.weight, &self.bias, self.eps as f32)?;
        Ok(Some(normalized.permute((0, 3, 1, 2))?))
    }

    fn forward_generic(&self, x: &Tensor) -> Result<Tensor> {
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

    /// One-pass CPU implementation of the NCHW channel normalization.  The
    /// generic expression creates seven full image-sized intermediates; here
    /// every output pixel is independent and needs just two channel reductions.
    fn forward_cpu_f32(&self, x: &Tensor, gelu: bool) -> Result<Option<Tensor>> {
        if !x.device().is_cpu() || x.dtype() != DType::F32 {
            return Ok(None);
        }
        let (batch, channels, height, width) = x.dims4()?;
        let Some((x_start, x_end)) = x.layout().contiguous_offsets() else {
            return Ok(None);
        };
        let Some((weight_start, weight_end)) = self.weight.layout().contiguous_offsets() else {
            return Ok(None);
        };
        let Some((bias_start, bias_end)) = self.bias.layout().contiguous_offsets() else {
            return Ok(None);
        };
        let (x_storage, _) = x.storage_and_layout();
        let (weight_storage, _) = self.weight.storage_and_layout();
        let (bias_storage, _) = self.bias.storage_and_layout();
        let (Storage::Cpu(x_storage), Storage::Cpu(weight_storage), Storage::Cpu(bias_storage)) =
            (&*x_storage, &*weight_storage, &*bias_storage)
        else {
            return Ok(None);
        };
        let input = &x_storage.as_slice::<f32>()?[x_start..x_end];
        let weights = &weight_storage.as_slice::<f32>()?[weight_start..weight_end];
        let biases = &bias_storage.as_slice::<f32>()?[bias_start..bias_end];
        let pixels = height * width;
        if input.len() != batch * channels * pixels
            || weights.len() != channels
            || biases.len() != channels
        {
            return Ok(None);
        }
        let mut output = vec![0f32; input.len()];
        // NCHW makes a pixel's channel values strided.  Partition by
        // `(batch, pixel)` anyway: each Rayon range partition writes
        // distinct pixels across every channel, avoiding an image-sized
        // NHWC transpose/copy.
        let output_address = output.as_mut_ptr() as usize;
        (0..batch * pixels).into_par_iter().for_each(|batch_pixel| {
            let batch_index = batch_pixel / pixels;
            let pixel = batch_pixel % pixels;
            let base = batch_index * channels * pixels;
            let mut mean = 0f32;
            for input_channel in 0..channels {
                mean += input[base + input_channel * pixels + pixel];
            }
            mean /= channels as f32;
            let mut variance = 0f32;
            for input_channel in 0..channels {
                let centered = input[base + input_channel * pixels + pixel] - mean;
                variance += centered * centered;
            }
            let inverse_std = 1. / (variance / channels as f32 + self.eps as f32).sqrt();
            for channel in 0..channels {
                // Each `batch_pixel` is unique, so no two jobs write the same
                // NCHW element. `output` stays allocated until the parallel
                // iterator has joined.
                unsafe {
                    *((output_address as *mut f32).add(base + channel * pixels + pixel)) =
                        (input[base + channel * pixels + pixel] - mean)
                            * inverse_std
                            * weights[channel]
                            + biases[channel];
                }
            }
        });
        if gelu {
            candle_core::op::gelu_erf_f32_inplace(&mut output);
        }
        Ok(Some(Tensor::from_vec(
            output,
            (batch, channels, height, width),
            x.device(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn fused_cpu_path_matches_tensor_expression() {
        let device = Device::Cpu;
        let norm = LayerNorm2d {
            weight: Tensor::from_vec(vec![1.0f32, 0.5, 1.5], 3, &device).unwrap(),
            bias: Tensor::from_vec(vec![0.1f32, -0.2, 0.3], 3, &device).unwrap(),
            eps: 1e-6,
        };
        let input = Tensor::from_vec(
            (0..2 * 3 * 5 * 7)
                .map(|index| index as f32 * 0.03125 - 3.0)
                .collect::<Vec<_>>(),
            (2, 3, 5, 7),
            &device,
        )
        .unwrap();
        let fused = norm.forward_cpu_f32(&input, false).unwrap().unwrap();
        let generic = norm.forward_generic(&input).unwrap();
        let max_error = fused
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(generic.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(max_error <= 1e-6, "max error {max_error}");

        let fused_gelu = norm.forward_gelu(&input).unwrap();
        let expected_gelu = generic.gelu_erf().unwrap();
        let max_error = fused_gelu
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(
                expected_gelu
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(max_error <= 1e-6, "GELU max error {max_error}");
    }

    #[test]
    fn fused_cpu_path_matches_multi_batch_non_multiple_pixels() {
        let device = Device::Cpu;
        let channels = 8;
        let norm = LayerNorm2d {
            weight: Tensor::from_vec(vec![1f32; channels], channels, &device).unwrap(),
            bias: Tensor::from_vec(vec![0f32; channels], channels, &device).unwrap(),
            eps: 1e-6,
        };
        let input = Tensor::from_vec(
            (0..2 * channels * 10 * 14)
                .map(|index| (index as f32 * 0.03125).sin())
                .collect::<Vec<_>>(),
            (2, channels, 10, 14),
            &device,
        )
        .unwrap();
        let fused = norm.forward_cpu_f32(&input, false).unwrap().unwrap();
        let generic = norm.forward_generic(&input).unwrap();
        let max_error = fused
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(generic.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(max_error <= 1e-6, "max error {max_error}");
    }
}
