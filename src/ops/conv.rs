//! Translation of `sam2/modeling/backbones/utils.py`.

use candle_core::{DType, Result, Storage, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPartition {
    pub padded_height: usize,
    pub padded_width: usize,
}

/// Partitions NHWC tokens into non-overlapping square windows, zero-padding the
/// bottom and right edges exactly as `torch.nn.functional.pad` does upstream.
pub fn window_partition(x: &Tensor, window_size: usize) -> Result<(Tensor, WindowPartition)> {
    if std::env::var_os("SAM2_CPU_DIRECT_WINDOWS").is_none_or(|value| value != "0") {
        if let Some(result) = window_partition_cpu_f32(x, window_size)? {
            return Ok(result);
        }
    }
    window_partition_generic(x, window_size)
}

fn window_partition_generic(x: &Tensor, window_size: usize) -> Result<(Tensor, WindowPartition)> {
    let (batch, height, width, channels) = x.dims4()?;
    let pad_height = (window_size - height % window_size) % window_size;
    let pad_width = (window_size - width % window_size) % window_size;
    let x = x
        .pad_with_zeros(1, 0, pad_height)?
        .pad_with_zeros(2, 0, pad_width)?;
    let padded_height = height + pad_height;
    let padded_width = width + pad_width;
    let windows = x
        .reshape((
            batch,
            padded_height / window_size,
            window_size,
            padded_width / window_size,
            window_size,
            channels,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape(((), window_size, window_size, channels))?;
    Ok((
        windows,
        WindowPartition {
            padded_height,
            padded_width,
        },
    ))
}

/// Reverses [`window_partition`] and crops any padding from the original image.
pub fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    partition: WindowPartition,
    original_height: usize,
    original_width: usize,
) -> Result<Tensor> {
    if std::env::var_os("SAM2_CPU_DIRECT_WINDOWS").is_none_or(|value| value != "0") {
        if let Some(result) = window_unpartition_cpu_f32(
            windows,
            window_size,
            partition,
            original_height,
            original_width,
            None,
        )? {
            return Ok(result);
        }
    }
    window_unpartition_generic(
        windows,
        window_size,
        partition,
        original_height,
        original_width,
    )
}

/// Unpartitions windows and adds an equal-shaped residual in the output pass.
pub fn window_unpartition_add_residual(
    windows: &Tensor,
    window_size: usize,
    partition: WindowPartition,
    residual: &Tensor,
) -> Result<Tensor> {
    let (_, original_height, original_width, _) = residual.dims4()?;
    if std::env::var_os("SAM2_CPU_DIRECT_WINDOWS").is_none_or(|value| value != "0")
        && std::env::var_os("SAM2_CPU_FUSED_WINDOW_RESIDUAL").is_none_or(|value| value != "0")
    {
        if let Some(result) = window_unpartition_cpu_f32(
            windows,
            window_size,
            partition,
            original_height,
            original_width,
            Some(residual),
        )? {
            return Ok(result);
        }
    }
    residual.broadcast_add(&window_unpartition(
        windows,
        window_size,
        partition,
        original_height,
        original_width,
    )?)
}

fn window_unpartition_generic(
    windows: &Tensor,
    window_size: usize,
    partition: WindowPartition,
    original_height: usize,
    original_width: usize,
) -> Result<Tensor> {
    let (window_batch, _, _, channels) = windows.dims4()?;
    let windows_per_image =
        partition.padded_height * partition.padded_width / window_size / window_size;
    let batch = window_batch / windows_per_image;
    let x = windows
        .reshape((
            batch,
            partition.padded_height / window_size,
            partition.padded_width / window_size,
            window_size,
            window_size,
            channels,
        ))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((
            batch,
            partition.padded_height,
            partition.padded_width,
            channels,
        ))?;
    if partition.padded_height > original_height || partition.padded_width > original_width {
        x.narrow(1, 0, original_height)?
            .narrow(2, 0, original_width)
    } else {
        Ok(x)
    }
}

fn window_partition_cpu_f32(
    x: &Tensor,
    window_size: usize,
) -> Result<Option<(Tensor, WindowPartition)>> {
    if window_size == 0 || x.dtype() != DType::F32 {
        return Ok(None);
    }
    let (batch, height, width, channels) = x.dims4()?;
    if batch == 0 || height == 0 || width == 0 || channels == 0 {
        return Ok(None);
    }
    let (storage, layout) = x.storage_and_layout();
    let (Storage::Cpu(storage), Some((start, end))) = (&*storage, layout.contiguous_offsets())
    else {
        return Ok(None);
    };
    let input = &storage.as_slice::<f32>()?[start..end];
    let padded_height = height.div_ceil(window_size) * window_size;
    let padded_width = width.div_ceil(window_size) * window_size;
    let windows_y = padded_height / window_size;
    let windows_x = padded_width / window_size;
    let windows_per_image = windows_y * windows_x;
    let window_elements = window_size * window_size * channels;
    let mut output = vec![0f32; batch * windows_per_image * window_elements];
    output
        .par_chunks_mut(window_elements)
        .enumerate()
        .for_each(|(window_index, destination)| {
            let batch_index = window_index / windows_per_image;
            let within_batch = window_index % windows_per_image;
            let window_y = within_batch / windows_x;
            let window_x = within_batch % windows_x;
            let source_x = window_x * window_size;
            let valid_width = width.saturating_sub(source_x).min(window_size);
            for local_y in 0..window_size {
                let source_y = window_y * window_size + local_y;
                if source_y < height && valid_width > 0 {
                    let source_offset =
                        ((batch_index * height + source_y) * width + source_x) * channels;
                    let destination_offset = local_y * window_size * channels;
                    destination[destination_offset..destination_offset + valid_width * channels]
                        .copy_from_slice(
                            &input[source_offset..source_offset + valid_width * channels],
                        );
                }
            }
        });
    Ok(Some((
        Tensor::from_vec(
            output,
            (
                batch * windows_per_image,
                window_size,
                window_size,
                channels,
            ),
            x.device(),
        )?,
        WindowPartition {
            padded_height,
            padded_width,
        },
    )))
}

fn window_unpartition_cpu_f32(
    windows: &Tensor,
    window_size: usize,
    partition: WindowPartition,
    original_height: usize,
    original_width: usize,
    residual: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    if window_size == 0 || windows.dtype() != DType::F32 {
        return Ok(None);
    }
    let (window_batch, window_height, window_width, channels) = windows.dims4()?;
    if window_batch == 0
        || window_height != window_size
        || window_width != window_size
        || channels == 0
        || original_height == 0
        || original_width == 0
        || original_height > partition.padded_height
        || original_width > partition.padded_width
        || partition.padded_height % window_size != 0
        || partition.padded_width % window_size != 0
    {
        return Ok(None);
    }
    let windows_y = partition.padded_height / window_size;
    let windows_x = partition.padded_width / window_size;
    let windows_per_image = windows_y * windows_x;
    if windows_per_image == 0 || window_batch % windows_per_image != 0 {
        return Ok(None);
    }
    let (storage, layout) = windows.storage_and_layout();
    let (Storage::Cpu(storage), Some((start, end))) = (&*storage, layout.contiguous_offsets())
    else {
        return Ok(None);
    };
    let source = &storage.as_slice::<f32>()?[start..end];
    let batch = window_batch / windows_per_image;
    if let Some(residual) = residual {
        if residual.dtype() != DType::F32
            || residual.dims() != [batch, original_height, original_width, channels]
        {
            return Ok(None);
        }
    }
    let residual_storage = residual.map(Tensor::storage_and_layout);
    let residual_values = if let Some((storage, layout)) = residual_storage.as_ref() {
        let (Storage::Cpu(storage), Some((start, end))) = (&**storage, layout.contiguous_offsets())
        else {
            return Ok(None);
        };
        Some(&storage.as_slice::<f32>()?[start..end])
    } else {
        None
    };
    let mut output = vec![0f32; batch * original_height * original_width * channels];
    output
        .par_chunks_mut(original_width * channels)
        .enumerate()
        .for_each(|(row_index, destination)| {
            let batch_index = row_index / original_height;
            let output_y = row_index % original_height;
            let window_y = output_y / window_size;
            let local_y = output_y % window_size;
            for window_x in 0..windows_x {
                let output_x = window_x * window_size;
                let valid_width = original_width.saturating_sub(output_x).min(window_size);
                if valid_width == 0 {
                    break;
                }
                let window_index =
                    (batch_index * windows_per_image + window_y * windows_x + window_x)
                        * window_size
                        * window_size;
                let source_offset = (window_index + local_y * window_size) * channels;
                let destination_offset = output_x * channels;
                let count = valid_width * channels;
                if let Some(residual_values) = residual_values {
                    let residual_offset =
                        row_index * original_width * channels + destination_offset;
                    for index in 0..count {
                        destination[destination_offset + index] = residual_values
                            [residual_offset + index]
                            + source[source_offset + index];
                    }
                } else {
                    destination[destination_offset..destination_offset + count]
                        .copy_from_slice(&source[source_offset..source_offset + count]);
                }
            }
        });
    Ok(Some(Tensor::from_vec(
        output,
        (batch, original_height, original_width, channels),
        windows.device(),
    )?))
}

#[derive(Clone, Debug)]
pub struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    pub fn load(
        in_channels: usize,
        embed_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj = conv2d(
            in_channels,
            embed_dim,
            kernel_size,
            Conv2dConfig {
                stride,
                padding,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        Ok(Self { proj })
    }

    /// Input is NCHW; output is the Hiera trunk's NHWC layout.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)?.permute((0, 2, 3, 1))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn partitions_and_restores_non_multiple_extent() {
        let x = Tensor::arange(0f32, 2. * 3. * 5. * 2., &Device::Cpu)
            .unwrap()
            .reshape((2, 3, 5, 2))
            .unwrap();
        let (windows, partition) = window_partition(&x, 4).unwrap();
        assert_eq!(windows.dims(), &[4, 4, 4, 2]);
        assert_eq!(
            partition,
            WindowPartition {
                padded_height: 4,
                padded_width: 8
            }
        );
        let restored = window_unpartition(&windows, 4, partition, 3, 5).unwrap();
        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn direct_cpu_windows_match_generic_for_padded_and_exact_tiles() -> Result<()> {
        for (batch, height, width, channels, window_size) in [(2, 3, 5, 2, 4), (1, 8, 8, 3, 4)] {
            let x = Tensor::from_vec(
                (0..batch * height * width * channels)
                    .map(|index| index as f32 * 0.125 - 2.)
                    .collect(),
                (batch, height, width, channels),
                &Device::Cpu,
            )?;
            let (expected, expected_partition) = window_partition_generic(&x, window_size)?;
            let (actual, actual_partition) =
                window_partition_cpu_f32(&x, window_size)?.expect("eligible CPU f32 input");
            assert_eq!(actual_partition, expected_partition);
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
            let actual_windows = actual.clone();
            let expected = window_unpartition_generic(
                &expected,
                window_size,
                expected_partition,
                height,
                width,
            )?;
            let actual = window_unpartition_cpu_f32(
                &actual,
                window_size,
                actual_partition,
                height,
                width,
                None,
            )?
            .expect("eligible CPU f32 windows");
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
            let residual = x.affine(0.25, 1.0)?;
            let expected_with_residual = residual.broadcast_add(&expected)?;
            let fused = window_unpartition_cpu_f32(
                &actual_windows,
                window_size,
                actual_partition,
                height,
                width,
                Some(&residual),
            )?
            .expect("eligible CPU f32 residual");
            assert_eq!(
                fused.flatten_all()?.to_vec1::<f32>()?,
                expected_with_residual.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }
}
