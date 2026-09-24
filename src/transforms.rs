//! SAM2 image/coordinate preprocessing and mask restoration utilities.

use candle_core::{DType, Device, Result, Tensor, bail};

use crate::ops::connected_components_8;

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
    mask_threshold: f64,
    max_hole_area: usize,
    max_sprinkle_area: usize,
}

impl Default for Sam2ImageTransforms {
    fn default() -> Self {
        Self {
            resolution: 1024,
            mask_threshold: 0.,
            max_hole_area: 0,
            max_sprinkle_area: 0,
        }
    }
}

impl Sam2ImageTransforms {
    pub const fn new(resolution: usize) -> Self {
        Self {
            resolution,
            mask_threshold: 0.,
            max_hole_area: 0,
            max_sprinkle_area: 0,
        }
    }

    pub const fn with_postprocessing(
        resolution: usize,
        mask_threshold: f64,
        max_hole_area: usize,
        max_sprinkle_area: usize,
    ) -> Self {
        Self {
            resolution,
            mask_threshold,
            max_hole_area,
            max_sprinkle_area,
        }
    }

    pub const fn mask_threshold(&self) -> f64 {
        self.mask_threshold
    }

    pub const fn max_hole_area(&self) -> usize {
        self.max_hole_area
    }

    pub const fn max_sprinkle_area(&self) -> usize {
        self.max_sprinkle_area
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

    /// Batched equivalent of upstream `forward_batch`. Every image is resized
    /// independently and the retained geometries map prompts back to its
    /// original dimensions.
    pub fn forward_batch(
        &self,
        images: &[(&[u8], usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Vec<ImageGeometry>)> {
        if images.is_empty() {
            bail!("image batch must not be empty")
        }
        let prepared = images
            .iter()
            .map(|&(pixels, width, height)| self.prepare_rgb8(pixels, width, height, device))
            .collect::<Result<Vec<_>>>()?;
        let tensors = prepared
            .iter()
            .map(|image| &image.tensor)
            .collect::<Vec<_>>();
        let tensor = Tensor::cat(&tensors, 0)?;
        let geometries = prepared.into_iter().map(|image| image.geometry).collect();
        Ok((tensor, geometries))
    }

    pub fn transform_coords(
        &self,
        coordinates: &Tensor,
        normalize: bool,
        original_height_width: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        if coordinates.dim(coordinates.rank().saturating_sub(1))? != 2 {
            bail!("coordinates must have length two in the last dimension")
        }
        let coordinates = coordinates.to_dtype(DType::F32)?;
        let scale = if normalize {
            let (height, width) = original_height_width.ok_or_else(|| {
                candle_core::Error::Msg("normalized coordinates need orig_hw".into())
            })?;
            Tensor::new(
                &[
                    self.resolution as f32 / width as f32,
                    self.resolution as f32 / height as f32,
                ],
                coordinates.device(),
            )?
        } else {
            Tensor::new(
                &[self.resolution as f32, self.resolution as f32],
                coordinates.device(),
            )?
        };
        coordinates.broadcast_mul(&scale)
    }

    pub fn transform_boxes(
        &self,
        boxes: &Tensor,
        normalize: bool,
        original_height_width: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        if boxes.dim(boxes.rank().saturating_sub(1))? != 4 {
            bail!("boxes must have length four in the last dimension")
        }
        let mut shape = boxes.dims().to_vec();
        shape.pop();
        shape.extend([2, 2]);
        self.transform_coords(
            &boxes.reshape(candle_core::Shape::from_dims(&shape))?,
            normalize,
            original_height_width,
        )
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

    /// Applies reference hole/sprinkle cleanup at the native mask resolution,
    /// then restores logits to `(original_height, original_width)`.
    pub fn postprocess_masks(
        &self,
        masks: &Tensor,
        original_height: usize,
        original_width: usize,
    ) -> Result<Tensor> {
        let masks = masks.to_dtype(DType::F32)?;
        let (batch, channels, height, width) = masks.dims4()?;
        let flat = masks.reshape((batch * channels, 1, height, width))?;
        let mut output = masks.clone();
        if self.max_hole_area > 0 {
            let (labels, areas) = connected_components_8(&flat.le(self.mask_threshold)?)?;
            let region = labels
                .gt(0f64)?
                .to_dtype(DType::U32)?
                .mul(&areas.le(self.max_hole_area as f64)?.to_dtype(DType::U32)?)?
                .ne(0f64)?
                .reshape(masks.shape())?;
            output = region.where_cond(
                &output.ones_like()?.affine(0., self.mask_threshold + 10.)?,
                &output,
            )?;
        }
        if self.max_sprinkle_area > 0 {
            // Reference intentionally computes both component maps from the
            // original masks, not from the hole-filled intermediate output.
            let (labels, areas) = connected_components_8(&flat.gt(self.mask_threshold)?)?;
            let region = labels
                .gt(0f64)?
                .to_dtype(DType::U32)?
                .mul(
                    &areas
                        .le(self.max_sprinkle_area as f64)?
                        .to_dtype(DType::U32)?,
                )?
                .ne(0f64)?
                .reshape(masks.shape())?;
            output = region.where_cond(
                &output.ones_like()?.affine(0., self.mask_threshold - 10.)?,
                &output,
            )?;
        }
        output.upsample_bilinear2d(original_height, original_width, false)
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
