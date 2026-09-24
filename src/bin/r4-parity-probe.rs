//! Compares automatic mask generation against the pinned real-image fixture.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use sam2_rs::{
    AutomaticMaskConfig, AutomaticMaskOutputMode, AutomaticMaskSegmentation,
    Sam2AutomaticMaskGenerator, Sam2Variant,
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

fn compare(
    prefix: &str,
    masks: &[sam2_rs::AutomaticMask],
    tensors: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let expected_count = take(tensors, &format!("{prefix}_count"))?.to_vec1::<i64>()?[0];
    if masks.len() != expected_count as usize {
        candle_core::bail!(
            "{prefix} count mismatch: {} != {expected_count}",
            masks.len()
        )
    }
    let rows = masks.len();
    let boxes = masks.iter().flat_map(|mask| mask.bbox).collect::<Vec<_>>();
    let points = masks.iter().flat_map(|mask| mask.point).collect::<Vec<_>>();
    let crops = masks
        .iter()
        .flat_map(|mask| mask.crop_box)
        .collect::<Vec<_>>();
    let ious = masks
        .iter()
        .map(|mask| mask.predicted_iou)
        .collect::<Vec<_>>();
    let stability = masks
        .iter()
        .map(|mask| mask.stability_score)
        .collect::<Vec<_>>();
    let areas = masks
        .iter()
        .map(|mask| mask.area as i64)
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(boxes, (rows, 4), &Device::Cpu)?,
        &take(tensors, &format!("{prefix}_boxes"))?,
        0.,
        &format!("{prefix}_boxes"),
    )?;
    assert_close(
        &Tensor::from_vec(points, (rows, 2), &Device::Cpu)?,
        &take(tensors, &format!("{prefix}_points"))?,
        1e-6,
        &format!("{prefix}_points"),
    )?;
    assert_close(
        &Tensor::from_vec(crops, (rows, 4), &Device::Cpu)?,
        &take(tensors, &format!("{prefix}_crops"))?,
        0.,
        &format!("{prefix}_crops"),
    )?;
    assert_close(
        &Tensor::from_vec(ious, rows, &Device::Cpu)?,
        &take(tensors, &format!("{prefix}_ious"))?,
        5e-5,
        &format!("{prefix}_ious"),
    )?;
    assert_close(
        &Tensor::from_vec(stability, rows, &Device::Cpu)?,
        &take(tensors, &format!("{prefix}_stability"))?,
        1e-6,
        &format!("{prefix}_stability"),
    )?;
    assert_close(
        &Tensor::from_vec(areas, rows, &Device::Cpu)?,
        &take(tensors, &format!("{prefix}_areas"))?,
        0.,
        &format!("{prefix}_areas"),
    )?;
    let expected_counts = take(tensors, &format!("{prefix}_rle_counts"))?.to_vec2::<i64>()?;
    let expected_lengths = take(tensors, &format!("{prefix}_rle_lengths"))?.to_vec1::<i64>()?;
    for (row, mask) in masks.iter().enumerate() {
        let AutomaticMaskSegmentation::UncompressedRle(rle) = &mask.segmentation else {
            candle_core::bail!("{prefix} did not return uncompressed RLE")
        };
        let expected = expected_counts[row][..expected_lengths[row] as usize]
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        if rle.counts != expected {
            candle_core::bail!("{prefix} RLE mismatch at row {row}")
        }
    }
    println!("{prefix}_rles: exact");
    Ok(())
}

fn main() -> Result<()> {
    let fixture = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2_r4_parity.safetensors".into());
    let checkpoint = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2.1_hiera_tiny.safetensors".into());
    let mut tensors = candle_core::safetensors::load(fixture, &Device::Cpu)?;
    let image = take(&mut tensors, "image")?;
    let (height, width, channels) = image.dims3()?;
    if channels != 3 {
        candle_core::bail!("fixture image is not RGB")
    }
    let pixels = image.flatten_all()?.to_vec1::<u8>()?;
    let config = AutomaticMaskConfig {
        points_per_side: Some(2),
        points_per_batch: 3,
        pred_iou_thresh: -1.,
        stability_score_thresh: -1.,
        output_mode: AutomaticMaskOutputMode::UncompressedRle,
        ..Default::default()
    };
    let generator =
        Sam2AutomaticMaskGenerator::load_sam2_1(Sam2Variant::HieraTiny, &checkpoint, config)?;
    compare(
        "base",
        &generator.generate_rgb8(&pixels, width, height)?,
        &mut tensors,
    )?;

    for (prefix, config) in [
        (
            "explicit",
            AutomaticMaskConfig {
                points_per_side: None,
                point_grids: Some(vec![vec![[0.3, 0.4], [0.7, 0.6]]]),
                points_per_batch: 1,
                pred_iou_thresh: -1.,
                stability_score_thresh: -1.,
                multimask_output: false,
                output_mode: AutomaticMaskOutputMode::UncompressedRle,
                ..Default::default()
            },
        ),
        (
            "m2m",
            AutomaticMaskConfig {
                points_per_side: Some(1),
                points_per_batch: 1,
                pred_iou_thresh: -1.,
                stability_score_thresh: -1.,
                use_m2m: true,
                multimask_output: true,
                output_mode: AutomaticMaskOutputMode::UncompressedRle,
                ..Default::default()
            },
        ),
        (
            "cleanup",
            AutomaticMaskConfig {
                points_per_side: Some(1),
                points_per_batch: 1,
                pred_iou_thresh: -1.,
                stability_score_thresh: -1.,
                min_mask_region_area: 20,
                multimask_output: false,
                output_mode: AutomaticMaskOutputMode::UncompressedRle,
                ..Default::default()
            },
        ),
        (
            "crops",
            AutomaticMaskConfig {
                points_per_side: Some(1),
                points_per_batch: 2,
                pred_iou_thresh: -1.,
                stability_score_thresh: -1.,
                crop_n_layers: 1,
                crop_nms_thresh: 0.3,
                multimask_output: false,
                output_mode: AutomaticMaskOutputMode::UncompressedRle,
                ..Default::default()
            },
        ),
    ] {
        let generator =
            Sam2AutomaticMaskGenerator::load_sam2_1(Sam2Variant::HieraTiny, &checkpoint, config)?;
        compare(
            prefix,
            &generator.generate_rgb8(&pixels, width, height)?,
            &mut tensors,
        )?;
    }
    let default_generator = Sam2AutomaticMaskGenerator::load_sam2_1(
        Sam2Variant::HieraTiny,
        &checkpoint,
        AutomaticMaskConfig {
            output_mode: AutomaticMaskOutputMode::UncompressedRle,
            ..Default::default()
        },
    )?;
    compare(
        "default",
        &default_generator.generate_rgb8(&pixels, width, height)?,
        &mut tensors,
    )?;
    Ok(())
}
