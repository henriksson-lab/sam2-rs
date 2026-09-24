//! Mask postprocessing leaves shared by image, AMG, and video prediction.

use std::collections::VecDeque;

use candle_core::{DType, Result, Shape, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmallRegionMode {
    Holes,
    Islands,
}

fn components_8(mask: &[bool], width: usize, height: usize) -> (Vec<u32>, Vec<usize>) {
    debug_assert_eq!(mask.len(), width * height);
    let mut labels = vec![0u32; mask.len()];
    let mut sizes = vec![0usize];
    let mut queue = VecDeque::new();
    let mut next_label = 0u32;
    for seed in 0..mask.len() {
        if !mask[seed] || labels[seed] != 0 {
            continue;
        }
        next_label += 1;
        labels[seed] = next_label;
        queue.push_back(seed);
        let mut size = 0usize;
        while let Some(index) = queue.pop_front() {
            size += 1;
            let x = index % width;
            let y = index / width;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let neighbor = ny as usize * width + nx as usize;
                    if mask[neighbor] && labels[neighbor] == 0 {
                        labels[neighbor] = next_label;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        sizes.push(size);
    }
    (labels, sizes)
}

/// Reference-compatible 8-connected component labels and per-pixel areas for
/// binary `N×1×H×W` masks.
pub fn connected_components_8(mask: &Tensor) -> Result<(Tensor, Tensor)> {
    let (batch, channels, height, width) = mask.dims4()?;
    if channels != 1 {
        candle_core::bail!("connected components expects N×1×H×W masks")
    }
    let binary = mask
        .ne(0f64)?
        .to_dtype(DType::U8)?
        .reshape((batch, height, width))?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec3::<u8>()?;
    let mut all_labels = Vec::with_capacity(batch * height * width);
    let mut all_areas = Vec::with_capacity(batch * height * width);
    for sample in binary {
        let flat: Vec<bool> = sample.iter().flatten().map(|&value| value != 0).collect();
        let (labels, sizes) = components_8(&flat, width, height);
        all_areas.extend(labels.iter().map(|&label| sizes[label as usize] as u32));
        all_labels.extend(labels);
    }
    let shape = (batch, 1, height, width);
    Ok((
        Tensor::from_vec(all_labels, shape, mask.device())?,
        Tensor::from_vec(all_areas, shape, mask.device())?,
    ))
}

/// Fills background components of area at most `max_area` with score `0.1`.
pub fn fill_holes_in_mask_scores(mask: &Tensor, max_area: usize) -> Result<Tensor> {
    if max_area == 0 {
        candle_core::bail!("max_area must be positive")
    }
    let (batch, channels, height, width) = mask.dims4()?;
    if channels != 1 || mask.dtype() != DType::F32 {
        candle_core::bail!("hole filling expects f32 N×1×H×W mask scores")
    }
    let values = mask
        .reshape((batch, height, width))?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec3::<f32>()?;
    let mut output = Vec::with_capacity(batch * height * width);
    for sample in values {
        let mut flat: Vec<f32> = sample.iter().flatten().copied().collect();
        let background: Vec<bool> = flat.iter().map(|&value| value <= 0.).collect();
        let (labels, sizes) = components_8(&background, width, height);
        for (value, &label) in flat.iter_mut().zip(&labels) {
            if label > 0 && sizes[label as usize] <= max_area {
                *value = 0.1;
            }
        }
        output.extend(flat);
    }
    Tensor::from_vec(output, (batch, 1, height, width), mask.device())
}

fn boxes_impl(masks: &Tensor, empty_is_zero: bool) -> Result<Tensor> {
    if masks.rank() < 2 {
        candle_core::bail!("mask-to-box expects at least H×W dimensions")
    }
    let dims = masks.dims();
    let height = dims[dims.len() - 2];
    let width = dims[dims.len() - 1];
    let outer = dims[..dims.len() - 2].iter().product::<usize>();
    let binary = masks
        .ne(0f64)?
        .to_dtype(DType::U8)?
        .reshape((outer, height, width))?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec3::<u8>()?;
    let mut boxes = Vec::with_capacity(outer * 4);
    for sample in binary {
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x: Option<usize> = None;
        let mut max_y: Option<usize> = None;
        for (y, row) in sample.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                if value != 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = Some(max_x.map_or(x, |current| current.max(x)));
                    max_y = Some(max_y.map_or(y, |current| current.max(y)));
                }
            }
        }
        if let (Some(max_x), Some(max_y)) = (max_x, max_y) {
            boxes.extend([min_x as i64, min_y as i64, max_x as i64, max_y as i64]);
        } else if empty_is_zero {
            boxes.extend([0i64; 4]);
        } else {
            boxes.extend([width as i64, height as i64, -1, -1]);
        }
    }
    let mut output_dims = dims[..dims.len() - 2].to_vec();
    output_dims.push(4);
    Tensor::from_vec(boxes, Shape::from_dims(&output_dims), masks.device())
}

/// `utils.misc.mask_to_box`, including its empty-mask sentinel.
pub fn mask_to_box(masks: &Tensor) -> Result<Tensor> {
    boxes_impl(masks, false)
}

/// `utils.amg.batched_mask_to_box`, returning zero boxes for empty masks.
pub fn batched_mask_to_box(masks: &Tensor) -> Result<Tensor> {
    boxes_impl(masks, true)
}

/// Appends new point prompts along the prompt dimension.
pub fn concat_points(
    old: Option<(&Tensor, &Tensor)>,
    new_points: &Tensor,
    new_labels: &Tensor,
) -> Result<(Tensor, Tensor)> {
    match old {
        Some((old_points, old_labels)) => Ok((
            Tensor::cat(&[old_points, new_points], 1)?,
            Tensor::cat(&[old_labels, new_labels], 1)?,
        )),
        None => Ok((new_points.clone(), new_labels.clone())),
    }
}

/// Keeps only the highest-scoring object at each pixel and clamps every other
/// object's score to at most -10, matching SAM2 video consolidation.
pub fn apply_non_overlapping_constraints(predicted_masks: &Tensor) -> Result<Tensor> {
    let batch = predicted_masks.dim(0)?;
    if batch == 1 {
        return Ok(predicted_masks.clone());
    }
    let maximum_object = predicted_masks.argmax_keepdim(0)?;
    let mut index_shape = vec![1usize; predicted_masks.rank()];
    index_shape[0] = batch;
    let object_indices = Tensor::arange(0u32, batch as u32, predicted_masks.device())?
        .reshape(Shape::from_dims(&index_shape))?
        .broadcast_as(predicted_masks.shape())?;
    let keep = maximum_object
        .broadcast_as(predicted_masks.shape())?
        .eq(&object_indices)?;
    keep.where_cond(predicted_masks, &predicted_masks.minimum(-10f64)?)
}

pub fn restore_video_masks(
    masks: &Tensor,
    original_height: usize,
    original_width: usize,
    non_overlapping: bool,
) -> Result<Tensor> {
    let restored = if masks.dim(masks.rank() - 2)? == original_height
        && masks.dim(masks.rank() - 1)? == original_width
    {
        masks.clone()
    } else {
        masks
            .to_dtype(DType::F32)?
            .upsample_bilinear2d(original_height, original_width, false)?
    };
    if non_overlapping {
        apply_non_overlapping_constraints(&restored)
    } else {
        Ok(restored)
    }
}

/// Removes 8-connected holes or islands smaller than `area_threshold`.
pub fn remove_small_regions(
    mask: &[bool],
    width: usize,
    height: usize,
    area_threshold: usize,
    mode: SmallRegionMode,
) -> Result<(Vec<bool>, bool)> {
    if mask.len() != width * height {
        candle_core::bail!("mask length does not match width × height")
    }
    let holes = mode == SmallRegionMode::Holes;
    let working: Vec<bool> = mask.iter().map(|&value| holes ^ value).collect();
    let (labels, sizes) = components_8(&working, width, height);
    let small: Vec<usize> = sizes
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(label, &size)| (size < area_threshold).then_some(label))
        .collect();
    if small.is_empty() {
        return Ok((mask.to_vec(), false));
    }
    let mut keep = vec![false; sizes.len()];
    if holes {
        keep[0] = true;
        for label in small {
            keep[label] = true;
        }
    } else {
        for label in 1..sizes.len() {
            keep[label] = !small.contains(&label);
        }
        if !keep.iter().any(|&value| value) {
            let mut largest = 0usize;
            for label in 1..sizes.len() {
                if largest == 0 || sizes[label] > sizes[largest] {
                    largest = label;
                }
            }
            if largest != 0 {
                keep[largest] = true;
            }
        }
    }
    Ok((
        labels.iter().map(|&label| keep[label as usize]).collect(),
        true,
    ))
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn connected_components_use_eight_connectivity_and_report_area_per_pixel() {
        let mask = Tensor::from_vec(
            vec![1u8, 0, 0, 1, 0, 0, 0, 1, 1],
            (1, 1, 3, 3),
            &Device::Cpu,
        )
        .unwrap();
        let (labels, areas) = connected_components_8(&mask).unwrap();
        assert_eq!(
            labels.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![1, 0, 0, 1, 0, 0, 0, 1, 1]
        );
        assert_eq!(
            areas.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![4, 0, 0, 4, 0, 0, 0, 4, 4]
        );
    }

    #[test]
    fn fills_only_background_components_within_the_area_limit() {
        let mask = Tensor::from_vec(
            vec![1f32, 1., 1., 1., -1., 1., 1., 1., 1.],
            (1, 1, 3, 3),
            &Device::Cpu,
        )
        .unwrap();
        let output = fill_holes_in_mask_scores(&mask, 1).unwrap();
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 1., 1., 1., 0.1, 1., 1., 1., 1.]
        );
    }

    #[test]
    fn mask_box_variants_match_reference_empty_behavior() {
        let masks =
            Tensor::from_vec(vec![0u8, 1, 1, 0, 0, 0, 0, 0], (2, 2, 2), &Device::Cpu).unwrap();
        assert_eq!(
            batched_mask_to_box(&masks)
                .unwrap()
                .to_vec2::<i64>()
                .unwrap(),
            vec![vec![0, 0, 1, 1], vec![0, 0, 0, 0]]
        );
        assert_eq!(
            mask_to_box(&masks).unwrap().to_vec2::<i64>().unwrap()[1],
            vec![2, 2, -1, -1]
        );
    }

    #[test]
    fn small_island_removal_keeps_the_largest_if_every_component_is_small() {
        let mask = vec![
            true, false, false, false, false, false, true, true, false, false, false, false,
        ];
        let (output, changed) =
            remove_small_regions(&mask, 4, 3, 3, SmallRegionMode::Islands).unwrap();
        assert!(changed);
        assert_eq!(
            output,
            vec![
                false, false, false, false, false, false, true, true, false, false, false, false
            ]
        );
    }

    #[test]
    fn concat_points_appends_on_the_prompt_axis() {
        let old_points = Tensor::from_vec(vec![1f32, 2.], (1, 1, 2), &Device::Cpu).unwrap();
        let old_labels = Tensor::from_vec(vec![1i64], (1, 1), &Device::Cpu).unwrap();
        let new_points = Tensor::from_vec(vec![3f32, 4.], (1, 1, 2), &Device::Cpu).unwrap();
        let new_labels = Tensor::from_vec(vec![0i64], (1, 1), &Device::Cpu).unwrap();
        let (points, labels) =
            concat_points(Some((&old_points, &old_labels)), &new_points, &new_labels).unwrap();
        assert_eq!(points.dims(), &[1, 2, 2]);
        assert_eq!(labels.to_vec2::<i64>().unwrap(), vec![vec![1, 0]]);
    }

    #[test]
    fn non_overlap_keeps_pixel_winner_and_clamps_other_objects() {
        let masks = Tensor::from_vec(
            vec![2f32, -20., 1., 3., -2., 4., 0., 2.],
            (2, 1, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let output = apply_non_overlapping_constraints(&masks)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(output, vec![2., -20., 1., 3., -10., 4., -10., -10.]);
    }
}
