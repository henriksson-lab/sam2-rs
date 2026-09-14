//! Translation of `sam2/modeling/backbones/utils.py`.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, conv2d};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPartition {
    pub padded_height: usize,
    pub padded_width: usize,
}

/// Partitions NHWC tokens into non-overlapping square windows, zero-padding the
/// bottom and right edges exactly as `torch.nn.functional.pad` does upstream.
pub fn window_partition(x: &Tensor, window_size: usize) -> Result<(Tensor, WindowPartition)> {
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
}
