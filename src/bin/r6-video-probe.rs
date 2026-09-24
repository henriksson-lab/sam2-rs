//! Initial interactive-video model parity probe for translation batch R6.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use sam2_rs::{PointPrompt, Sam2InteractiveVideoPredictor, Sam2Variant, VideoPredictorConfig};

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture is missing {name}")))
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, name: &str) -> Result<()> {
    if actual.dims() != expected.dims() {
        candle_core::bail!(
            "{name} shape mismatch: {:?} != {:?}",
            actual.dims(),
            expected.dims()
        )
    }
    let expected = expected.to_device(actual.device())?;
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

fn main() -> Result<()> {
    let fixture = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2_r6_video.safetensors".into());
    let checkpoint = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2.1_hiera_tiny.safetensors".into());
    let mut tensors = candle_core::safetensors::load(fixture, &Device::Cpu)?;
    let mut pixels = Vec::new();
    for index in 0..3 {
        let image = take(&mut tensors, &format!("image_{index}"))?;
        pixels.push(image.flatten_all()?.to_vec1::<u8>()?);
    }
    let images = pixels
        .iter()
        .map(|pixels| (pixels.as_slice(), 48usize, 32usize))
        .collect::<Vec<_>>();
    let device = match std::env::var("SAM2_R6_DEVICE").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let memory_tolerance = if device.is_cuda() { 2e-2 } else { 8e-3 };
    let predictor = Sam2InteractiveVideoPredictor::load_sam2_1_on_device(
        Sam2Variant::HieraTiny,
        checkpoint,
        &device,
        VideoPredictorConfig::default(),
    )?;
    let mut session = predictor.init_state(&images)?;
    let add1 = predictor.add_new_points(
        &mut session,
        0,
        101,
        &[PointPrompt {
            x: 16.,
            y: 16.,
            positive: true,
        }],
        true,
    )?;
    assert_close(
        &Tensor::from_vec(
            add1.object_ids.iter().map(|&id| id as i64).collect(),
            1,
            &Device::Cpu,
        )?,
        &take(&mut tensors, "add1_ids")?,
        0.,
        "add1_ids",
    )?;
    assert_close(
        &add1.masks,
        &take(&mut tensors, "add1_masks")?,
        5e-4,
        "add1_masks",
    )?;
    assert_close(
        &session.output(101, 0).unwrap().low_res_masks,
        tensors.get("low_101_0").unwrap(),
        2.,
        "add1_low_vs_committed_reference",
    )?;
    let add2 = predictor.add_new_points(
        &mut session,
        0,
        202,
        &[PointPrompt {
            x: 34.,
            y: 16.,
            positive: true,
        }],
        true,
    )?;
    assert_close(
        &Tensor::from_vec(
            add2.object_ids.iter().map(|&id| id as i64).collect(),
            2,
            &Device::Cpu,
        )?,
        &take(&mut tensors, "add2_ids")?,
        0.,
        "add2_ids",
    )?;
    assert_close(
        &add2.masks,
        &take(&mut tensors, "add2_masks")?,
        5e-4,
        "add2_masks",
    )?;

    let propagated = predictor.propagate_in_video(&mut session, None, None, false)?;
    let frames = propagated
        .iter()
        .map(|output| output.frame_index as i64)
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(frames.clone(), frames.len(), &Device::Cpu)?,
        &take(&mut tensors, "forward_frames")?,
        0.,
        "forward_frames",
    )?;
    for output in propagated {
        let frame = output.frame_index;
        for object_id in output.object_ids {
            let model_output = session.output(object_id, frame).unwrap();
            if frame == 1 {
                assert_close(
                    &model_output.conditioned_features,
                    &take(&mut tensors, &format!("conditioned_{object_id}_1"))?,
                    1e-3,
                    &format!("conditioned_{object_id}_1"),
                )?;
            }
            assert_close(
                &model_output.low_res_masks,
                &take(&mut tensors, &format!("low_{object_id}_{frame}"))?,
                1e-3,
                &format!("low_{object_id}_{frame}"),
            )?;
            assert_close(
                &model_output.object_pointer,
                &take(&mut tensors, &format!("ptr_{object_id}_{frame}"))?,
                1e-4,
                &format!("ptr_{object_id}_{frame}"),
            )?;
            assert_close(
                &model_output.memory_features,
                &take(&mut tensors, &format!("memory_{object_id}_{frame}"))?,
                memory_tolerance,
                &format!("memory_{object_id}_{frame}"),
            )?;
        }
        assert_close(
            &output.masks,
            &take(&mut tensors, &format!("forward_masks_{frame}"))?,
            5e-4,
            &format!("forward_masks_{frame}"),
        )?;
    }

    let correction = predictor.add_new_points(
        &mut session,
        1,
        101,
        &[PointPrompt {
            x: 22.,
            y: 12.,
            positive: true,
        }],
        true,
    )?;
    assert_close(
        &Tensor::from_vec(
            correction.object_ids.iter().map(|&id| id as i64).collect(),
            correction.object_ids.len(),
            &Device::Cpu,
        )?,
        &take(&mut tensors, "correction_ids")?,
        0.,
        "correction_ids",
    )?;
    assert_close(
        &correction.masks,
        &take(&mut tensors, "correction_masks")?,
        1e-3,
        "correction_masks",
    )?;
    let corrected = session.output(101, 1).unwrap();
    assert_close(
        &corrected.low_res_masks,
        &take(&mut tensors, "correction_low")?,
        2e-3,
        "correction_low",
    )?;
    assert_close(
        &corrected.object_pointer,
        &take(&mut tensors, "correction_ptr")?,
        1e-4,
        "correction_ptr",
    )?;

    let reverse = predictor.propagate_in_video(&mut session, Some(1), Some(1), true)?;
    let reverse_frames = reverse
        .iter()
        .map(|output| output.frame_index as i64)
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(reverse_frames.clone(), reverse_frames.len(), &Device::Cpu)?,
        &take(&mut tensors, "reverse_frames")?,
        0.,
        "reverse_frames",
    )?;
    for output in reverse {
        let frame = output.frame_index;
        assert_close(
            &Tensor::from_vec(
                output.object_ids.iter().map(|&id| id as i64).collect(),
                output.object_ids.len(),
                &Device::Cpu,
            )?,
            &take(&mut tensors, &format!("reverse_ids_{frame}"))?,
            0.,
            &format!("reverse_ids_{frame}"),
        )?;
        for object_id in &output.object_ids {
            let model_output = session.output(*object_id, frame).unwrap();
            assert_close(
                &model_output.low_res_masks,
                &take(&mut tensors, &format!("reverse_low_{object_id}_{frame}"))?,
                2e-3,
                &format!("reverse_low_{object_id}_{frame}"),
            )?;
            assert_close(
                &model_output.object_pointer,
                &take(&mut tensors, &format!("reverse_ptr_{object_id}_{frame}"))?,
                2e-4,
                &format!("reverse_ptr_{object_id}_{frame}"),
            )?;
            assert_close(
                &model_output.memory_features,
                &take(&mut tensors, &format!("reverse_memory_{object_id}_{frame}"))?,
                memory_tolerance,
                &format!("reverse_memory_{object_id}_{frame}"),
            )?;
        }
        assert_close(
            &output.masks,
            &take(&mut tensors, &format!("reverse_masks_{frame}"))?,
            1e-3,
            &format!("reverse_masks_{frame}"),
        )?;
    }

    let cleared = predictor.clear_all_prompts_in_frame(&mut session, 1, 101)?;
    assert_close(
        &Tensor::from_vec(
            cleared.object_ids.iter().map(|&id| id as i64).collect(),
            cleared.object_ids.len(),
            &Device::Cpu,
        )?,
        &take(&mut tensors, "clear_ids")?,
        0.,
        "clear_ids",
    )?;
    assert_close(
        &cleared.masks,
        &take(&mut tensors, "clear_masks")?,
        1e-3,
        "clear_masks",
    )?;

    let removed = predictor.remove_object_with_outputs(&mut session, 202)?;
    assert_close(
        &Tensor::from_vec(
            removed.object_ids.iter().map(|&id| id as i64).collect(),
            removed.object_ids.len(),
            &Device::Cpu,
        )?,
        &take(&mut tensors, "remove_ids")?,
        0.,
        "remove_ids",
    )?;
    let removed_frames = removed
        .updated_frames
        .iter()
        .map(|output| output.frame_index as i64)
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(removed_frames.clone(), removed_frames.len(), &Device::Cpu)?,
        &take(&mut tensors, "remove_frames")?,
        0.,
        "remove_frames",
    )?;
    for output in removed.updated_frames {
        assert_close(
            &output.masks,
            &take(
                &mut tensors,
                &format!("remove_masks_{}", output.frame_index),
            )?,
            1e-3,
            &format!("remove_masks_{}", output.frame_index),
        )?;
    }

    predictor.reset_state(&mut session);
    assert_close(
        &Tensor::from_vec(
            vec![
                session.state().object_ids().len() as i64,
                session.state().object_count() as i64,
            ],
            2,
            &Device::Cpu,
        )?,
        &take(&mut tensors, "reset_counts")?,
        0.,
        "reset_counts",
    )?;

    let input_mask = take(&mut tensors, "mask_input")?;
    let mask_output = predictor.add_new_mask(&mut session, 2, 303, &input_mask)?;
    assert_close(
        &Tensor::from_vec(
            mask_output.object_ids.iter().map(|&id| id as i64).collect(),
            mask_output.object_ids.len(),
            &Device::Cpu,
        )?,
        &take(&mut tensors, "mask_ids")?,
        0.,
        "mask_ids",
    )?;
    assert_close(
        &mask_output.masks,
        &take(&mut tensors, "mask_masks")?,
        1e-3,
        "mask_masks",
    )?;
    let model_mask_output = session.output(303, 2).unwrap();
    assert_close(
        &model_mask_output.low_res_masks,
        &take(&mut tensors, "mask_low")?,
        1e-3,
        "mask_low",
    )?;
    assert_close(
        &model_mask_output.object_pointer,
        &take(&mut tensors, "mask_ptr")?,
        1e-4,
        "mask_ptr",
    )?;
    Ok(())
}
