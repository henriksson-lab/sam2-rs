//! SAM2 image/coordinate preprocessing and mask restoration utilities.

use candle_core::{DType, Device, Result, Tensor, bail};
use rayon::prelude::*;

fn sparse_resize_weights(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = output as f32 / input as f32;
    let filter_scale = scale.min(1.);
    let support = 1. / filter_scale;
    (0..output)
        .map(|out| {
            let center = (out as f32 + 0.5) / scale - 0.5;
            let start = (center - support).ceil() as isize;
            let end = (center + support).floor() as isize;
            let mut weights = (start..=end)
                .filter(|&index| index >= 0 && (index as usize) < input)
                .filter_map(|index| {
                    let value = (1. - ((index as f32 - center) * filter_scale).abs()).max(0.);
                    (value > 0.).then_some((index as usize, value))
                })
                .collect::<Vec<_>>();
            let sum = weights.iter().map(|(_, value)| value).sum::<f32>();
            for (_, value) in &mut weights {
                *value /= sum;
            }
            weights
        })
        .collect()
}

fn resize_normalize_rgb8(
    rgb_hwc: &[u8],
    input_width: usize,
    input_height: usize,
    output: usize,
) -> Vec<f32> {
    let mut resized = vec![0f32; 3 * output * output];
    resize_normalize_rgb8_into(rgb_hwc, input_width, input_height, output, &mut resized);
    resized
}

fn resize_normalize_rgb8_into(
    rgb_hwc: &[u8],
    input_width: usize,
    input_height: usize,
    output: usize,
    resized: &mut [f32],
) {
    assert_eq!(resized.len(), 3 * output * output);
    let height_weights = sparse_resize_weights(input_height, output);
    let width_weights = sparse_resize_weights(input_width, output);
    let mut height = vec![0f32; 3 * output * input_width];
    height
        .par_chunks_mut(input_width)
        .enumerate()
        .for_each(|(channel_row, row)| {
            let channel = channel_row / output;
            let out_y = channel_row % output;
            for (x, value) in row.iter_mut().enumerate() {
                *value = height_weights[out_y]
                    .iter()
                    .map(|&(y, weight)| {
                        rgb_hwc[(y * input_width + x) * 3 + channel] as f32 / 255. * weight
                    })
                    .sum();
            }
        });
    let means = [0.485f32, 0.456, 0.406];
    let stds = [0.229f32, 0.224, 0.225];
    resized
        .par_chunks_mut(output)
        .enumerate()
        .for_each(|(channel_row, row)| {
            let channel = channel_row / output;
            let out_y = channel_row % output;
            let input_row = &height[(channel * output + out_y) * input_width
                ..(channel * output + out_y + 1) * input_width];
            for (out_x, value) in row.iter_mut().enumerate() {
                let value_before_normalization = width_weights[out_x]
                    .iter()
                    .map(|&(x, weight)| input_row[x] * weight)
                    .sum::<f32>();
                *value = (value_before_normalization - means[channel]) / stds[channel];
            }
        });
}

/// Separable triangle-filter resize matching TorchVision's antialiased tensor
/// bilinear path. Upscaling naturally reduces to ordinary bilinear sampling.
pub(crate) fn antialiased_bilinear_resize(
    input: &Tensor,
    output_height: usize,
    output_width: usize,
) -> Result<Tensor> {
    let (batch, channels, input_height, input_width) = input.dims4()?;
    fn weights(input: usize, output: usize, device: &Device) -> Result<Tensor> {
        let scale = output as f32 / input as f32;
        let filter_scale = scale.min(1.);
        let support = 1. / filter_scale;
        let mut values = vec![0f32; output * input];
        for out in 0..output {
            let center = (out as f32 + 0.5) / scale - 0.5;
            let start = (center - support).ceil() as isize;
            let end = (center + support).floor() as isize;
            let mut sum = 0f32;
            for index in start..=end {
                if index >= 0 && (index as usize) < input {
                    let value = (1. - ((index as f32 - center) * filter_scale).abs()).max(0.);
                    values[out * input + index as usize] = value;
                    sum += value;
                }
            }
            for index in 0..input {
                values[out * input + index] /= sum;
            }
        }
        Tensor::from_vec(values, (output, input), device)
    }
    let height_weights = weights(input_height, output_height, input.device())?;
    let width_weights = weights(input_width, output_width, input.device())?;
    let height = input
        .permute((0, 1, 3, 2))?
        .reshape((batch * channels * input_width, input_height))?
        .matmul(&height_weights.t()?)?
        .reshape((batch, channels, input_width, output_height))?
        .permute((0, 1, 3, 2))?;
    height
        .reshape((batch * channels * output_height, input_width))?
        .matmul(&width_weights.t()?)?
        .reshape((batch, channels, output_height, output_width))
}

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
        let chw = resize_normalize_rgb8(rgb_hwc, width, height, self.resolution);
        Ok(PreparedImage {
            tensor: Tensor::from_vec(chw, (1, 3, self.resolution, self.resolution), device)?,
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
        let frame_elements = 3 * self.resolution * self.resolution;
        let mut geometries = Vec::with_capacity(images.len());
        for &(pixels, width, height) in images {
            if width == 0 || height == 0 {
                bail!("SAM2 image dimensions must be non-zero")
            }
            let expected = width * height * 3;
            if pixels.len() != expected {
                bail!(
                    "expected {expected} interleaved RGB bytes, got {}",
                    pixels.len()
                )
            }
            geometries.push(ImageGeometry {
                original_width: width,
                original_height: height,
                resolution: self.resolution,
            });
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            use candle_core::{CudaStorage, Storage, op::BackpropOp};
            // Allocate final device storage once, then reuse one normalized
            // host-frame buffer and copy directly into each destination
            // range. This avoids per-frame device allocations without
            // retaining a second full-video host batch.
            let mut device_values = unsafe { cuda.alloc::<f32>(images.len() * frame_elements)? };
            let mut frame = vec![0f32; frame_elements];
            for (index, &(pixels, width, height)) in images.iter().enumerate() {
                resize_normalize_rgb8_into(pixels, width, height, self.resolution, &mut frame);
                let start = index * frame_elements;
                cuda.memcpy_htod(
                    frame.as_slice(),
                    &mut device_values.slice_mut(start..start + frame_elements),
                )?;
            }
            let storage = CudaStorage::wrap_cuda_slice(device_values, cuda.clone());
            let tensor = Tensor::from_storage(
                Storage::Cuda(storage),
                (images.len(), 3, self.resolution, self.resolution),
                BackpropOp::none(),
                false,
            );
            return Ok((tensor, geometries));
        }
        let mut values = vec![0f32; images.len() * frame_elements];
        for (frame, &(pixels, width, height)) in
            values.chunks_exact_mut(frame_elements).zip(images.iter())
        {
            resize_normalize_rgb8_into(pixels, width, height, self.resolution, frame);
        }
        // Match upstream forward_batch: one contiguous batch and one device
        // storage. Callers can retain per-image narrow views without copying
        // or allocating a separate storage buffer for every frame.
        let tensor = Tensor::from_vec(
            values,
            (images.len(), 3, self.resolution, self.resolution),
            device,
        )?;
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

    pub(crate) fn has_mask_cleanup(&self) -> bool {
        self.max_hole_area > 0 || self.max_sprinkle_area > 0
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

    #[test]
    fn forward_batch_matches_individual_preparation_without_frame_copies() {
        let transforms = Sam2ImageTransforms::new(2);
        let red_green = [255, 0, 0, 0, 255, 0];
        let blue_white = [0, 0, 255, 255, 255, 255];
        let images = [
            (red_green.as_slice(), 2usize, 1usize),
            (blue_white.as_slice(), 2usize, 1usize),
        ];
        let (batch, geometries) = transforms.forward_batch(&images, &Device::Cpu).unwrap();
        assert_eq!(batch.dims(), &[2, 3, 2, 2]);
        assert_eq!(geometries.len(), 2);
        for (index, &(pixels, width, height)) in images.iter().enumerate() {
            let expected = transforms
                .prepare_rgb8(pixels, width, height, &Device::Cpu)
                .unwrap();
            let actual = batch.narrow(0, index, 1).unwrap();
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected
                    .tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );
        }
    }
}
