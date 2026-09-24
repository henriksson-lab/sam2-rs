//! Compares batch-R1 mask leaves with the pinned Python fixture.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use sam2_rs::{
    MaskDecoder,
    ops::{
        SmallRegionMode, batched_mask_to_box, concat_points, connected_components_8,
        fill_holes_in_mask_scores, mask_to_box, remove_small_regions,
    },
};

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture is missing {name}")))
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, name: &str) -> Result<()> {
    let error = actual
        .to_dtype(DType::F32)?
        .sub(&expected.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    if error > tolerance {
        candle_core::bail!("{name} maximum error {error} exceeds {tolerance}")
    }
    println!("{name}: max_abs_error={error}");
    Ok(())
}

fn bools(tensor: &Tensor) -> Result<Vec<bool>> {
    Ok(tensor
        .flatten_all()?
        .to_vec1::<u8>()?
        .into_iter()
        .map(|value| value != 0)
        .collect())
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2_r1_parity.safetensors".into());
    let mut tensors = candle_core::safetensors::load(path, &Device::Cpu)?;

    let logits = take(&mut tensors, "dynamic_logits")?;
    let ious = take(&mut tensors, "dynamic_ious")?;
    let stability = MaskDecoder::stability_scores(&logits.narrow(1, 0, 1)?, 0.05)?;
    let (selected_logits, selected_ious) =
        MaskDecoder::dynamic_multimask_via_stability(&logits, &ious, 0.05, 0.98)?;
    assert_close(
        &stability,
        &take(&mut tensors, "dynamic_stability")?,
        1e-6,
        "dynamic_stability",
    )?;
    assert_close(
        &selected_logits,
        &take(&mut tensors, "dynamic_selected_logits")?,
        0.,
        "dynamic_selected_logits",
    )?;
    assert_close(
        &selected_ious,
        &take(&mut tensors, "dynamic_selected_ious")?,
        0.,
        "dynamic_selected_ious",
    )?;

    let component_mask = take(&mut tensors, "component_mask")?;
    let (labels, areas) = connected_components_8(&component_mask)?;
    assert_close(
        &labels,
        &take(&mut tensors, "component_labels")?,
        0.,
        "component_labels",
    )?;
    assert_close(
        &areas,
        &take(&mut tensors, "component_areas")?,
        0.,
        "component_areas",
    )?;

    let filled = fill_holes_in_mask_scores(&take(&mut tensors, "hole_scores")?, 1)?;
    assert_close(
        &filled,
        &take(&mut tensors, "hole_scores_filled")?,
        0.,
        "hole_scores_filled",
    )?;

    let boxes_mask = take(&mut tensors, "boxes_mask")?;
    assert_close(
        &batched_mask_to_box(&boxes_mask)?,
        &take(&mut tensors, "boxes_amg")?,
        0.,
        "boxes_amg",
    )?;
    assert_close(
        &mask_to_box(&boxes_mask.unsqueeze(1)?)?,
        &take(&mut tensors, "boxes_misc")?,
        0.,
        "boxes_misc",
    )?;

    for (prefix, width, height, threshold, mode) in [
        ("island", 4, 3, 3, SmallRegionMode::Islands),
        ("hole", 3, 3, 2, SmallRegionMode::Holes),
    ] {
        let input = take(&mut tensors, &format!("{prefix}_mask"))?;
        let (filtered, changed) =
            remove_small_regions(&bools(&input)?, width, height, threshold, mode)?;
        let actual = Tensor::from_vec(
            filtered.into_iter().map(u8::from).collect::<Vec<_>>(),
            (height, width),
            &Device::Cpu,
        )?;
        assert_close(
            &actual,
            &take(&mut tensors, &format!("{prefix}_filtered"))?,
            0.,
            &format!("{prefix}_filtered"),
        )?;
        let expected_changed =
            take(&mut tensors, &format!("{prefix}_changed"))?.to_vec1::<u8>()?[0] != 0;
        if changed != expected_changed {
            candle_core::bail!("{prefix}_changed mismatch")
        }
    }

    let old_points = take(&mut tensors, "old_points")?;
    let old_labels = take(&mut tensors, "old_labels")?;
    let new_points = take(&mut tensors, "new_points")?;
    let new_labels = take(&mut tensors, "new_labels")?;
    let (points, labels) =
        concat_points(Some((&old_points, &old_labels)), &new_points, &new_labels)?;
    assert_close(
        &points,
        &take(&mut tensors, "concatenated_points")?,
        0.,
        "concatenated_points",
    )?;
    assert_close(
        &labels,
        &take(&mut tensors, "concatenated_labels")?,
        0.,
        "concatenated_labels",
    )?;
    Ok(())
}
