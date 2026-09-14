//! SAM2 image/coordinate preprocessing and mask restoration utilities.

use candle_core::{DType, Device, Result, Tensor, bail};

/// Metadata retained to transform prompts and masks between original pixels and
/// SAM2's fixed square input coordinate system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageGeometry {
    pub original_width: usize,
    pub original_height: usize,
    pub resolution: usize,
}

impl ImageGeometry {
    pub fn scale_x(self) -> f32 {
        self.resolution as f32 / self.original_width as f32
    }
    pub fn scale_y(self) -> f32 {
        self.resolution as f32 / self.original_height as f32
    }
    pub fn model_point(self, x: f32, y: f32) -> (f32, f32) {
        (x * self.scale_x(), y * self.scale_y())
    }
}

/// RGB image converted to normalized, square NCHW SAM2 input.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub tensor: Tensor,
    pub geometry: ImageGeometry,
}

/// Equivalent to upstream `SAM2Transforms` for RGB byte buffers. SAM2 resizes
/// independently to 1024x1024 (it does not letterbox the image predictor).
#[derive(Clone, Copy, Debug)]
pub struct Sam2ImageTransforms {
    resolution: usize,
}

impl Default for Sam2ImageTransforms {
    fn default() -> Self {
        Self { resolution: 1024 }
    }
}

impl Sam2ImageTransforms {
    pub const fn new(resolution: usize) -> Self {
        Self { resolution }
    }

    pub fn prepare_rgb8(
        &self,
        rgb_hwc: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<PreparedImage> {
        if width == 0 || height == 0 {
            bail!("SAM2 image dimensions must be non-zero")
        }
        let expected = width * height * 3;
        if rgb_hwc.len() != expected {
            bail!(
                "expected {expected} interleaved RGB bytes, got {}",
                rgb_hwc.len()
            )
        }
        let mut chw = vec![0f32; expected];
        for y in 0..height {
            for x in 0..width {
                let source = (y * width + x) * 3;
                let destination = y * width + x;
                for channel in 0..3 {
                    chw[channel * width * height + destination] =
                        rgb_hwc[source + channel] as f32 / 255.;
                }
            }
        }
        let tensor = Tensor::from_vec(chw, (1, 3, height, width), device)?.upsample_bilinear2d(
            self.resolution,
            self.resolution,
            false,
        )?;
        let mean = Tensor::from_vec(vec![0.485f32, 0.456, 0.406], (1, 3, 1, 1), device)?;
        let std = Tensor::from_vec(vec![0.229f32, 0.224, 0.225], (1, 3, 1, 1), device)?;
        Ok(PreparedImage {
            tensor: tensor.broadcast_sub(&mean)?.broadcast_div(&std)?,
            geometry: ImageGeometry {
                original_width: width,
                original_height: height,
                resolution: self.resolution,
            },
        })
    }

    /// Restores N×M×h×w logits to the original image size. This deliberately
    /// keeps logits rather than thresholding, matching Python's `return_logits`.
    pub fn restore_masks(&self, low_res_masks: &Tensor, geometry: ImageGeometry) -> Result<Tensor> {
        low_res_masks.to_dtype(DType::F32)?.upsample_bilinear2d(
            geometry.original_height,
            geometry.original_width,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn rgb_layout_normalization_and_coordinate_scaling_are_explicit() {
        let image = Sam2ImageTransforms::new(2)
            .prepare_rgb8(&[255, 0, 0, 0, 255, 0], 2, 1, &Device::Cpu)
            .unwrap();
        assert_eq!(image.tensor.dims(), &[1, 3, 2, 2]);
        assert_eq!(image.geometry.model_point(1., 0.5), (1., 1.));
        let values = image
            .tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((values[0] - (1. - 0.485) / 0.229).abs() < 1e-5);
    }
}
