//! Pure automatic-mask-generator geometry and RLE utilities.

use std::ops::Range;

use candle_core::{DType, Result, Shape, Tensor};

use super::{SmallRegionMode, remove_small_regions};

/// Uncompressed COCO-style RLE. Runs are column-major (Fortran order) and
/// always begin with the background run, matching `utils.amg`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskRle {
    pub width: usize,
    pub height: usize,
    pub counts: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmallRegionMask {
    pub original_index: usize,
    pub rle: MaskRle,
    pub box_xyxy: [usize; 4],
    pub changed: bool,
}

impl MaskRle {
    pub fn from_row_major_mask(mask: &[bool], width: usize, height: usize) -> Result<Self> {
        if mask.len() != width * height {
            candle_core::bail!("mask length does not match width × height")
        }
        let mut counts = Vec::new();
        let mut current = false;
        let mut run = 0usize;
        for x in 0..width {
            for y in 0..height {
                let value = mask[y * width + x];
                if value == current {
                    run += 1;
                } else {
                    counts.push(run);
                    run = 1;
                    current = value;
                }
            }
        }
        counts.push(run);
        Ok(Self {
            width,
            height,
            counts,
        })
    }

    pub fn to_row_major_mask(&self) -> Result<Vec<bool>> {
        if self.counts.iter().sum::<usize>() != self.width * self.height {
            candle_core::bail!("RLE counts do not cover its declared size")
        }
        let mut fortran = Vec::with_capacity(self.width * self.height);
        let mut value = false;
        for &count in &self.counts {
            fortran.extend(std::iter::repeat_n(value, count));
            value = !value;
        }
        let mut row_major = vec![false; fortran.len()];
        for x in 0..self.width {
            for y in 0..self.height {
                row_major[y * self.width + x] = fortran[x * self.height + y];
            }
        }
        Ok(row_major)
    }

    pub fn area(&self) -> usize {
        self.counts.iter().skip(1).step_by(2).sum()
    }

    /// Compresses the run lengths using the COCO API's signed, delta-coded
    /// LEB128 variant. The returned string is the `counts` field used by
    /// `coco_rle` output mode.
    pub fn to_coco_counts(&self) -> Result<String> {
        let mut output = Vec::new();
        for index in 0..self.counts.len() {
            let mut value = i64::try_from(self.counts[index])
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if index > 2 {
                value -= i64::try_from(self.counts[index - 2])
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            }
            loop {
                let mut byte = (value & 0x1f) as u8;
                value >>= 5;
                let more = if byte & 0x10 != 0 {
                    value != -1
                } else {
                    value != 0
                };
                if more {
                    byte |= 0x20;
                }
                output.push(byte + 48);
                if !more {
                    break;
                }
            }
        }
        String::from_utf8(output).map_err(|error| candle_core::Error::Msg(error.to_string()))
    }
}

pub fn mask_to_rles(masks: &Tensor) -> Result<Vec<MaskRle>> {
    let (batch, height, width) = masks.dims3()?;
    let values = masks
        .ne(0f64)?
        .to_dtype(DType::U8)?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec3::<u8>()?;
    values
        .into_iter()
        .map(|mask| {
            let flat = mask
                .iter()
                .flatten()
                .map(|&value| value != 0)
                .collect::<Vec<_>>();
            MaskRle::from_row_major_mask(&flat, width, height)
        })
        .collect::<Result<Vec<_>>>()
        .map(|rles| {
            debug_assert_eq!(rles.len(), batch);
            rles
        })
}

pub fn build_point_grid(points_per_side: usize) -> Result<Vec<[f32; 2]>> {
    if points_per_side == 0 {
        candle_core::bail!("points_per_side must be positive")
    }
    let offset = 1. / (2 * points_per_side) as f64;
    let step = if points_per_side == 1 {
        0.
    } else {
        (1. - 2. * offset) / (points_per_side - 1) as f64
    };
    let mut points = Vec::with_capacity(points_per_side * points_per_side);
    for y in 0..points_per_side {
        for x in 0..points_per_side {
            points.push([
                (offset + x as f64 * step) as f32,
                (offset + y as f64 * step) as f32,
            ]);
        }
    }
    Ok(points)
}

pub fn build_all_layer_point_grids(
    points_per_side: usize,
    layers: usize,
    scale_per_layer: usize,
) -> Result<Vec<Vec<[f32; 2]>>> {
    if scale_per_layer == 0 {
        candle_core::bail!("scale_per_layer must be positive")
    }
    (0..=layers)
        .map(|layer| {
            let count = points_per_side / scale_per_layer.pow(layer as u32);
            build_point_grid(count)
        })
        .collect()
}

/// Returns crop boxes in `[x0, y0, x1, y1]` order and their layer indices.
pub fn generate_crop_boxes(
    height: usize,
    width: usize,
    layers: usize,
    overlap_ratio: f64,
) -> (Vec<[usize; 4]>, Vec<usize>) {
    let mut boxes = vec![[0, 0, width, height]];
    let mut layer_indices = vec![0];
    let short_side = height.min(width) as f64;
    for layer in 0..layers {
        let crops_per_side = 2usize.pow(layer as u32 + 1);
        let overlap = (overlap_ratio * short_side * (2. / crops_per_side as f64)) as usize;
        let crop_width = ((overlap * (crops_per_side - 1) + width) as f64 / crops_per_side as f64)
            .ceil() as usize;
        let crop_height = ((overlap * (crops_per_side - 1) + height) as f64 / crops_per_side as f64)
            .ceil() as usize;
        for x in 0..crops_per_side {
            for y in 0..crops_per_side {
                let x0 = (crop_width - overlap) * x;
                let y0 = (crop_height - overlap) * y;
                boxes.push([
                    x0,
                    y0,
                    (x0 + crop_width).min(width),
                    (y0 + crop_height).min(height),
                ]);
                layer_indices.push(layer + 1);
            }
        }
    }
    (boxes, layer_indices)
}

pub fn batch_ranges(length: usize, batch_size: usize) -> Result<Vec<Range<usize>>> {
    if batch_size == 0 {
        candle_core::bail!("batch_size must be positive")
    }
    Ok((0..length)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(length))
        .collect())
}

pub fn is_box_near_crop_edge(
    boxes: &Tensor,
    crop_box: [f32; 4],
    original_box: [f32; 4],
    tolerance: f32,
) -> Result<Tensor> {
    let rows = boxes
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?;
    let width = rows.dim(rows.rank() - 1)?;
    if width != 4 || rows.rank() != 2 {
        candle_core::bail!("edge test expects Nx4 boxes")
    }
    let rows = rows.to_vec2::<f32>()?;
    let mut output = Vec::with_capacity(rows.len());
    for local in rows {
        let global = [
            local[0] + crop_box[0],
            local[1] + crop_box[1],
            local[2] + crop_box[0],
            local[3] + crop_box[1],
        ];
        output.push(u8::from((0..4).any(|index| {
            (global[index] - crop_box[index]).abs() <= tolerance
                && (global[index] - original_box[index]).abs() > tolerance
        })));
    }
    Tensor::from_vec(output, boxes.dim(0)?, boxes.device())
}

pub fn box_xyxy_to_xywh(box_xyxy: [f32; 4]) -> [f32; 4] {
    [
        box_xyxy[0],
        box_xyxy[1],
        box_xyxy[2] - box_xyxy[0],
        box_xyxy[3] - box_xyxy[1],
    ]
}

pub fn calculate_stability_score(
    masks: &Tensor,
    mask_threshold: f64,
    threshold_offset: f64,
) -> Result<Tensor> {
    if masks.rank() < 2 {
        candle_core::bail!("stability score expects at least two spatial dimensions")
    }
    let spatial = masks.rank() - 1;
    let intersections = masks
        .gt(mask_threshold + threshold_offset)?
        .to_dtype(DType::F32)?
        .sum(spatial)?
        .sum(spatial - 1)?;
    let unions = masks
        .gt(mask_threshold - threshold_offset)?
        .to_dtype(DType::F32)?
        .sum(spatial)?
        .sum(spatial - 1)?;
    intersections.div(&unions)
}

pub fn uncrop_boxes_xyxy(boxes: &Tensor, crop_box: [usize; 4]) -> Result<Tensor> {
    let offset = Tensor::new(
        &[
            crop_box[0] as f32,
            crop_box[1] as f32,
            crop_box[0] as f32,
            crop_box[1] as f32,
        ],
        boxes.device(),
    )?;
    boxes.broadcast_add(&offset)
}

pub fn uncrop_points(points: &Tensor, crop_box: [usize; 4]) -> Result<Tensor> {
    let offset = Tensor::new(&[crop_box[0] as f32, crop_box[1] as f32], points.device())?;
    points.broadcast_add(&offset)
}

pub fn uncrop_masks(
    masks: &Tensor,
    crop_box: [usize; 4],
    original_height: usize,
    original_width: usize,
) -> Result<Tensor> {
    if masks.rank() < 2 {
        candle_core::bail!("uncrop masks expects at least two spatial dimensions")
    }
    let [x0, y0, x1, y1] = crop_box;
    if x1 < x0 || y1 < y0 || x1 > original_width || y1 > original_height {
        candle_core::bail!("invalid crop box")
    }
    let height_dim = masks.rank() - 2;
    let width_dim = masks.rank() - 1;
    if masks.dims()[height_dim] != y1 - y0 || masks.dims()[width_dim] != x1 - x0 {
        candle_core::bail!("mask spatial dimensions do not match crop box")
    }
    if crop_box == [0, 0, original_width, original_height] {
        return Ok(masks.clone());
    }
    masks
        .pad_with_zeros(width_dim, x0, original_width - x1)?
        .pad_with_zeros(height_dim, y0, original_height - y1)
}

pub fn uncrop_rle(
    rle: &MaskRle,
    crop_box: [usize; 4],
    original_height: usize,
    original_width: usize,
) -> Result<MaskRle> {
    let [x0, y0, x1, y1] = crop_box;
    if rle.width != x1 - x0 || rle.height != y1 - y0 {
        candle_core::bail!("RLE dimensions do not match crop box")
    }
    if crop_box == [0, 0, original_width, original_height] {
        return Ok(rle.clone());
    }
    let local = rle.to_row_major_mask()?;
    let mut full = vec![false; original_width * original_height];
    for y in 0..rle.height {
        let source = y * rle.width;
        let destination = (y0 + y) * original_width + x0;
        full[destination..destination + rle.width]
            .copy_from_slice(&local[source..source + rle.width]);
    }
    MaskRle::from_row_major_mask(&full, original_width, original_height)
}

pub fn rles_to_counts_tensor(rles: &[MaskRle]) -> Result<(Tensor, Tensor)> {
    let maximum = rles.iter().map(|rle| rle.counts.len()).max().unwrap_or(0);
    let mut counts = vec![-1i64; rles.len() * maximum];
    let mut lengths = Vec::with_capacity(rles.len());
    for (row, rle) in rles.iter().enumerate() {
        lengths.push(rle.counts.len() as i64);
        for (column, &count) in rle.counts.iter().enumerate() {
            counts[row * maximum + column] = count as i64;
        }
    }
    Ok((
        Tensor::from_vec(
            counts,
            Shape::from((rles.len(), maximum)),
            &candle_core::Device::Cpu,
        )?,
        Tensor::from_vec(lengths, rles.len(), &candle_core::Device::Cpu)?,
    ))
}

/// Typed equivalent of AMG `postprocess_small_regions`. Returned indices map
/// retained masks back to any caller-owned metadata.
pub fn postprocess_small_regions(
    rles: &[MaskRle],
    minimum_area: usize,
    nms_threshold: f32,
) -> Result<Vec<SmallRegionMask>> {
    let mut processed = Vec::with_capacity(rles.len());
    for (original_index, rle) in rles.iter().enumerate() {
        let mask = rle.to_row_major_mask()?;
        let (mask, holes_changed) = remove_small_regions(
            &mask,
            rle.width,
            rle.height,
            minimum_area,
            SmallRegionMode::Holes,
        )?;
        let (mask, islands_changed) = remove_small_regions(
            &mask,
            rle.width,
            rle.height,
            minimum_area,
            SmallRegionMode::Islands,
        )?;
        let changed = holes_changed || islands_changed;
        let box_xyxy = bool_mask_box(&mask, rle.width, rle.height);
        processed.push(SmallRegionMask {
            original_index,
            rle: if changed {
                MaskRle::from_row_major_mask(&mask, rle.width, rle.height)?
            } else {
                rle.clone()
            },
            box_xyxy,
            changed,
        });
    }
    processed.sort_by_key(|mask| mask.changed);
    let mut retained: Vec<SmallRegionMask> = Vec::new();
    'candidate: for candidate in processed {
        for selected in &retained {
            if box_iou_usize(candidate.box_xyxy, selected.box_xyxy) > nms_threshold {
                continue 'candidate;
            }
        }
        retained.push(candidate);
    }
    Ok(retained)
}

fn bool_mask_box(mask: &[bool], width: usize, height: usize) -> [usize; 4] {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if min_x == width {
        [0; 4]
    } else {
        [min_x, min_y, max_x, max_y]
    }
}

fn box_iou_usize(a: [usize; 4], b: [usize; 4]) -> f32 {
    let intersection_width = a[2].min(b[2]).saturating_sub(a[0].max(b[0])) as f32;
    let intersection_height = a[3].min(b[3]).saturating_sub(a[1].max(b[1])) as f32;
    let intersection = intersection_width * intersection_height;
    let area_a = (a[2].saturating_sub(a[0]) * a[3].saturating_sub(a[1])) as f32;
    let area_b = (b[2].saturating_sub(b[0]) * b[3].saturating_sub(b[1])) as f32;
    intersection / (area_a + area_b - intersection)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_is_column_major_and_round_trips() {
        let mask = vec![false, true, true, false, false, true];
        let rle = MaskRle::from_row_major_mask(&mask, 3, 2).unwrap();
        assert_eq!(rle.counts, vec![2, 1, 1, 2]);
        assert_eq!(rle.area(), 3);
        assert_eq!(rle.to_row_major_mask().unwrap(), mask);
    }

    #[test]
    fn crop_generation_preserves_reference_order() {
        let (boxes, layers) = generate_crop_boxes(6, 8, 1, 0.);
        assert_eq!(boxes.len(), 5);
        assert_eq!(boxes[1], [0, 0, 4, 3]);
        assert_eq!(boxes[2], [0, 3, 4, 6]);
        assert_eq!(layers, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn region_postprocessing_prefers_an_unchanged_duplicate() {
        let unchanged = MaskRle::from_row_major_mask(
            &[true, true, true, true, true, true, true, true, true],
            3,
            3,
        )
        .unwrap();
        let changed = MaskRle::from_row_major_mask(
            &[true, true, true, true, false, true, true, true, true],
            3,
            3,
        )
        .unwrap();
        let output = postprocess_small_regions(&[changed, unchanged.clone()], 2, 0.5).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].original_index, 1);
        assert_eq!(output[0].rle, unchanged);
    }
}
