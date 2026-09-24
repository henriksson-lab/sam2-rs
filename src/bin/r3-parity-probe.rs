//! Compares cached/batched image prediction against the pinned Python fixture.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use sam2_rs::{ImagePromptBatch, Sam2ImagePredictor, Sam2Variant};

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
    let fixture_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2_r3_parity.safetensors".into());
    let checkpoint_path = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2.1_hiera_tiny.safetensors".into());
    let mut tensors = candle_core::safetensors::load(fixture_path, &Device::Cpu)?;
    let image0 = rgb(take(&mut tensors, "image0")?)?;
    let image1 = rgb(take(&mut tensors, "image1")?)?;

    let mut predictor = Sam2ImagePredictor::load_sam2_1(Sam2Variant::HieraTiny, checkpoint_path)?;
    let images = [
        (image0.0.as_slice(), image0.1, image0.2),
        (image1.0.as_slice(), image1.1, image1.2),
    ];
    predictor.set_image_batch(&images)?;
    if !predictor.is_image_set() || !predictor.is_batch() {
        candle_core::bail!("predictor did not retain batched image state")
    }
    let embedding_id = predictor.get_image_embedding()?.id();
    assert_close(
        predictor.get_image_embedding()?,
        &take(&mut tensors, "embedding")?,
        2e-5,
        "embedding",
    )?;

    let prompts = vec![
        ImagePromptBatch {
            point_coords: Some(take(&mut tensors, "coords0")?),
            point_labels: Some(take(&mut tensors, "labels0")?),
            boxes: Some(take(&mut tensors, "boxes0")?),
            mask_input: Some(take(&mut tensors, "masks0")?),
            normalize_coords: true,
        },
        ImagePromptBatch {
            normalize_coords: true,
            ..Default::default()
        },
    ];

    for (multimask, prefix) in [(true, "multi"), (false, "single")] {
        let outputs = predictor.predict_batch(&prompts, multimask, true)?;
        if predictor.get_image_embedding()?.id() != embedding_id {
            candle_core::bail!("cached embedding changed during repeated prediction")
        }
        for (index, output) in outputs.iter().enumerate() {
            assert_close(
                &output.masks.squeeze(0)?,
                &take(&mut tensors, &format!("{prefix}_masks_{index}"))?,
                5e-4,
                &format!("{prefix}_masks_{index}"),
            )?;
            assert_close(
                &output.low_res.iou_predictions.squeeze(0)?,
                &take(&mut tensors, &format!("{prefix}_ious_{index}"))?,
                5e-5,
                &format!("{prefix}_ious_{index}"),
            )?;
            assert_close(
                &output.low_res.masks.squeeze(0)?,
                &take(&mut tensors, &format!("{prefix}_low_{index}"))?,
                5e-4,
                &format!("{prefix}_low_{index}"),
            )?;
            assert_close(
                &output.low_res.object_score_logits,
                &take(&mut tensors, &format!("{prefix}_objects_{index}"))?,
                5e-5,
                &format!("{prefix}_objects_{index}"),
            )?;
        }
    }

    predictor.set_image(image0.0.as_slice(), image0.1, image0.2)?;
    let single_embedding_id = predictor.get_image_embedding()?.id();
    assert_close(
        predictor.get_image_embedding()?,
        &take(&mut tensors, "single_embedding")?,
        2e-5,
        "single_embedding",
    )?;
    let cached = predictor.predict_cached(
        &ImagePromptBatch {
            point_coords: Some(Tensor::new(&[[4f32, 3.]], &Device::Cpu)?),
            point_labels: Some(Tensor::new(&[1i64], &Device::Cpu)?),
            normalize_coords: true,
            ..Default::default()
        },
        true,
        false,
    )?;
    assert_close(
        &cached.masks.squeeze(0)?,
        &take(&mut tensors, "cached_binary_masks")?,
        0.,
        "cached_binary_masks",
    )?;
    assert_close(
        &cached.low_res.iou_predictions.squeeze(0)?,
        &take(&mut tensors, "cached_ious")?,
        5e-5,
        "cached_ious",
    )?;
    assert_close(
        &cached.low_res.masks.squeeze(0)?,
        &take(&mut tensors, "cached_low")?,
        5e-4,
        "cached_low",
    )?;
    if predictor.get_image_embedding()?.id() != single_embedding_id {
        candle_core::bail!("single cached embedding changed during prediction")
    }

    predictor.reset_predictor();
    if predictor.is_image_set() || predictor.get_image_embedding().is_ok() {
        candle_core::bail!("reset_predictor retained cached image state")
    }
    println!("cache_reuse_and_reset: exact");
    Ok(())
}
