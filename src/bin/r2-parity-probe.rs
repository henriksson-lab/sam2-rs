//! Compares batch-R2 transform and AMG leaves with the pinned Python fixture.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use sam2_rs::{
    Sam2ImageTransforms,
    ops::{
        batch_ranges, box_xyxy_to_xywh, build_all_layer_point_grids, calculate_stability_score,
        generate_crop_boxes, is_box_near_crop_edge, mask_to_rles, uncrop_boxes_xyxy, uncrop_masks,
        uncrop_points,
    },
};

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture is missing {name}")))
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, name: &str) -> Result<()> {
    let actual = actual.to_dtype(DType::F32)?;
    let expected = expected.to_dtype(DType::F32)?;
    if actual.dims() != expected.dims() {
        candle_core::bail!(
            "{name} shape mismatch: {:?} != {:?}",
            actual.dims(),
            expected.dims()
        )
    }
    let error = actual
        .sub(&expected)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    if error > tolerance {
        candle_core::bail!("{name} maximum error {error} exceeds {tolerance}")
    }
    println!("{name}: max_abs_error={error}");
    Ok(())
}

fn rgb(tensor: Tensor) -> Result<(Vec<u8>, usize, usize)> {
    let (height, width, channels) = tensor.dims3()?;
    if channels != 3 {
        candle_core::bail!("fixture image is not HWC RGB")
    }
    Ok((tensor.flatten_all()?.to_vec1::<u8>()?, width, height))
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2_r2_parity.safetensors".into());
    let mut tensors = candle_core::safetensors::load(path, &Device::Cpu)?;

    let landscape = rgb(take(&mut tensors, "image_landscape")?)?;
    let portrait = rgb(take(&mut tensors, "image_portrait")?)?;
    let transform = Sam2ImageTransforms::new(6);
    let images = [
        (landscape.0.as_slice(), landscape.1, landscape.2),
        (portrait.0.as_slice(), portrait.1, portrait.2),
    ];
    let (batch, _) = transform.forward_batch(&images, &Device::Cpu)?;
    assert_close(
        &batch,
        &take(&mut tensors, "image_batch")?,
        1e-5,
        "image_batch",
    )?;

    let coordinates = take(&mut tensors, "coordinates")?;
    assert_close(
        &transform.transform_coords(&coordinates, true, Some((5, 5)))?,
        &take(&mut tensors, "coordinates_transformed")?,
        1e-6,
        "coordinates_transformed",
    )?;
    let boxes = take(&mut tensors, "boxes")?;
    assert_close(
        &transform.transform_boxes(&boxes, true, Some((5, 5)))?,
        &take(&mut tensors, "boxes_transformed")?,
        1e-6,
        "boxes_transformed",
    )?;
    let postprocess = Sam2ImageTransforms::with_postprocessing(6, 0., 1, 1);
    assert_close(
        &postprocess.postprocess_masks(&take(&mut tensors, "postprocess_input")?, 7, 9)?,
        &take(&mut tensors, "postprocess_output")?,
        1e-6,
        "postprocess_output",
    )?;

    let grids = build_all_layer_point_grids(8, 2, 2)?;
    let lengths = grids
        .iter()
        .map(|grid| grid.len() as i64)
        .collect::<Vec<_>>();
    let points = grids.into_iter().flatten().flatten().collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(
            points,
            (lengths.iter().sum::<i64>() as usize, 2),
            &Device::Cpu,
        )?,
        &take(&mut tensors, "point_grids")?,
        1e-7,
        "point_grids",
    )?;
    assert_close(
        &Tensor::from_vec(lengths, 3, &Device::Cpu)?,
        &take(&mut tensors, "point_grid_lengths")?,
        0.,
        "point_grid_lengths",
    )?;

    let (crop_boxes, crop_layers) = generate_crop_boxes(7, 11, 2, 0.2);
    let flat_boxes = crop_boxes.into_iter().flatten().map(|v| v as i64).collect();
    assert_close(
        &Tensor::from_vec(flat_boxes, (crop_layers.len(), 4), &Device::Cpu)?,
        &take(&mut tensors, "crop_boxes")?,
        0.,
        "crop_boxes",
    )?;
    assert_close(
        &Tensor::from_vec(
            crop_layers.into_iter().map(|v| v as i64).collect(),
            21,
            &Device::Cpu,
        )?,
        &take(&mut tensors, "crop_layers")?,
        0.,
        "crop_layers",
    )?;
    let ranges = batch_ranges(8, 3)?;
    let ranges = ranges
        .into_iter()
        .flat_map(|range| [range.start as i64, range.end as i64])
        .collect();
    assert_close(
        &Tensor::from_vec(ranges, (3, 2), &Device::Cpu)?,
        &take(&mut tensors, "batch_ranges")?,
        0.,
        "batch_ranges",
    )?;

    let edge_boxes = take(&mut tensors, "edge_boxes")?;
    assert_close(
        &is_box_near_crop_edge(&edge_boxes, [30., 0., 90., 80.], [0., 0., 120., 80.], 20.)?,
        &take(&mut tensors, "near_crop_edge")?,
        0.,
        "near_crop_edge",
    )?;

    let rles = mask_to_rles(&take(&mut tensors, "binary_masks")?)?;
    let expected_counts = take(&mut tensors, "rle_counts")?.to_vec2::<i64>()?;
    let expected_lengths = take(&mut tensors, "rle_lengths")?.to_vec1::<i64>()?;
    let expected_areas = take(&mut tensors, "rle_areas")?.to_vec1::<i64>()?;
    let expected_coco = take(&mut tensors, "coco_counts")?.to_vec2::<u8>()?;
    let expected_coco_lengths = take(&mut tensors, "coco_lengths")?.to_vec1::<i64>()?;
    for (row, rle) in rles.iter().enumerate() {
        let expected = expected_counts[row][..expected_lengths[row] as usize]
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        if rle.counts != expected {
            candle_core::bail!("rle_counts mismatch in row {row}")
        }
        if rle.area() != expected_areas[row] as usize {
            candle_core::bail!("RLE area mismatch in row {row}")
        }
        if rle.to_coco_counts()?.as_bytes()
            != &expected_coco[row][..expected_coco_lengths[row] as usize]
        {
            candle_core::bail!("compressed COCO RLE mismatch in row {row}")
        }
    }
    println!("rle_counts_and_areas: exact");
    let decoded = rles
        .iter()
        .map(|rle| rle.to_row_major_mask())
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .map(u8::from)
        .collect();
    assert_close(
        &Tensor::from_vec(decoded, (2, 2, 3), &Device::Cpu)?,
        &take(&mut tensors, "rle_decoded")?,
        0.,
        "rle_round_trip",
    )?;
    let xyxy = take(&mut tensors, "xyxy_box")?.to_vec1::<f32>()?;
    let xywh = box_xyxy_to_xywh([xyxy[0], xyxy[1], xyxy[2], xyxy[3]]);
    assert_close(
        &Tensor::new(&xywh, &Device::Cpu)?,
        &take(&mut tensors, "xywh_box")?,
        0.,
        "xywh_box",
    )?;

    let stability_masks = take(&mut tensors, "stability_masks")?;
    assert_close(
        &calculate_stability_score(&stability_masks, 0., 1.)?,
        &take(&mut tensors, "stability_scores")?,
        1e-6,
        "stability_scores",
    )?;
    let crop = [2, 1, 5, 3];
    assert_close(
        &uncrop_boxes_xyxy(&take(&mut tensors, "local_boxes")?, crop)?,
        &take(&mut tensors, "uncropped_boxes")?,
        0.,
        "uncropped_boxes",
    )?;
    assert_close(
        &uncrop_points(&take(&mut tensors, "local_points")?, crop)?,
        &take(&mut tensors, "uncropped_points")?,
        0.,
        "uncropped_points",
    )?;
    assert_close(
        &uncrop_masks(&take(&mut tensors, "local_masks")?, crop, 5, 7)?,
        &take(&mut tensors, "uncropped_masks")?,
        0.,
        "uncropped_masks",
    )?;
    Ok(())
}
