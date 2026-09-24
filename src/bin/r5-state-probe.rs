//! Compares typed video-state primitives with the pinned Python state rules.

use std::{collections::HashMap, path::PathBuf};

use candle_core::{DType, Device, Result, Tensor};
use sam2_rs::{
    BackboneOutput, MemoryFrame, VideoInferenceState, ops::apply_non_overlapping_constraints,
};

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture is missing {name}")))
}

fn assert_close(actual: &Tensor, expected: &Tensor, name: &str) -> Result<()> {
    if actual.dims() != expected.dims() {
        candle_core::bail!("{name} shape mismatch")
    }
    let error = actual
        .to_dtype(DType::F32)?
        .sub(&expected.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    if error != 0. {
        candle_core::bail!("{name} maximum error {error}")
    }
    println!("{name}: exact");
    Ok(())
}

fn memory(value: f32) -> Result<MemoryFrame> {
    let tensor = Tensor::new(&[[[[value]]]], &Device::Cpu)?;
    Ok(MemoryFrame {
        memory_features: tensor.clone(),
        memory_pos_enc: tensor.clone(),
        object_pointer: Tensor::new(&[[value]], &Device::Cpu)?,
        low_res_mask_logits: tensor,
    })
}

fn feature(value: f32) -> Result<BackboneOutput> {
    let tensor = Tensor::new(&[[[[value]]]], &Device::Cpu)?;
    Ok(BackboneOutput {
        vision_features: tensor.clone(),
        vision_pos_enc: vec![tensor.clone()],
        backbone_fpn: vec![tensor],
        sam_high_res_features: None,
    })
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "checkpoints/sam2_r5_state.safetensors".into());
    let mut tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
    let mut state = VideoInferenceState::with_video(640, 360, 5)?;
    for id in [101, 7, 55, 7] {
        state.object_index(id);
    }
    state.add_points(
        101,
        0,
        &Tensor::new(&[[[10f32, 20.], [30., 40.]]], &Device::Cpu)?,
        &Tensor::new(&[[2i64, 3]], &Device::Cpu)?,
        true,
    )?;
    state.add_points(
        101,
        0,
        &Tensor::new(&[[[50f32, 60.]]], &Device::Cpu)?,
        &Tensor::new(&[[1i64]], &Device::Cpu)?,
        false,
    )?;
    state.mark_tracked(101, 0, false);
    state.mark_tracked(101, 4, true);
    {
        let object = state.object_mut(101);
        object.conditioning_outputs.insert(0, memory(100.)?);
        for (frame, value) in [(0, 200.), (1, 201.), (3, 203.), (4, 204.)] {
            object
                .non_conditioning_outputs
                .insert(frame, memory(value)?);
        }
        object
            .temporary_non_conditioning_outputs
            .insert(2, memory(302.)?);
    }
    state.commit_temporary_frame(2, false);
    state.add_points(
        55,
        2,
        &Tensor::new(&[[[1f32, 2.]]], &Device::Cpu)?,
        &Tensor::new(&[[1i64]], &Device::Cpu)?,
        true,
    )?;
    state.add_mask(55, 2, Tensor::new(&[[[[1f32]]]], &Device::Cpu)?);
    for (frame, value) in [(1, 401.), (2, 402.), (3, 403.), (4, 404.)] {
        state
            .object_mut(55)
            .non_conditioning_outputs
            .insert(frame, memory(value)?);
    }
    state.clear_non_conditioning_around(2, 1);
    state.remove_object(7);
    state.cache_feature(0, feature(0.)?);
    state.cache_feature(1, feature(1.)?);

    assert_close(
        &Tensor::from_vec(
            state.object_ids().iter().map(|&id| id as i64).collect(),
            state.object_count(),
            &Device::Cpu,
        )?,
        &take(&mut tensors, "object_ids")?,
        "object_ids",
    )?;
    let indices = (0..state.object_count())
        .map(|index| state.object_index(state.object_id_for_index(index).unwrap()) as i64)
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(indices, state.object_count(), &Device::Cpu)?,
        &take(&mut tensors, "object_indices")?,
        "object_indices",
    )?;
    let object101 = state.object(101).unwrap();
    let points = object101.point_inputs.get(&0).unwrap();
    assert_close(
        &points.coordinates,
        &take(&mut tensors, "point_coords")?,
        "point_coords",
    )?;
    assert_close(
        &points.labels,
        &take(&mut tensors, "point_labels")?,
        "point_labels",
    )?;
    for (name, frames) in [
        (
            "obj101_non_cond_frames",
            object101
                .non_conditioning_outputs
                .keys()
                .copied()
                .collect::<Vec<_>>(),
        ),
        (
            "obj55_non_cond_frames",
            state
                .object(55)
                .unwrap()
                .non_conditioning_outputs
                .keys()
                .copied()
                .collect::<Vec<_>>(),
        ),
        (
            "tracked_frames",
            object101.tracked_frames.keys().copied().collect::<Vec<_>>(),
        ),
    ] {
        let values = frames
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>();
        assert_close(
            &Tensor::from_vec(values.clone(), values.len(), &Device::Cpu)?,
            &take(&mut tensors, name)?,
            name,
        )?;
    }
    let reverse = object101
        .tracked_frames
        .values()
        .map(|frame| u8::from(frame.reverse))
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(reverse.clone(), reverse.len(), &Device::Cpu)?,
        &take(&mut tensors, "tracked_reverse")?,
        "tracked_reverse",
    )?;
    assert_close(
        state.object(55).unwrap().mask_inputs.get(&2).unwrap(),
        &take(&mut tensors, "mask55")?,
        "mask55",
    )?;
    let cache = state
        .cached_feature_indices()
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    assert_close(
        &Tensor::from_vec(cache.clone(), cache.len(), &Device::Cpu)?,
        &take(&mut tensors, "cache_indices")?,
        "cache_indices",
    )?;
    assert_close(
        &apply_non_overlapping_constraints(&take(&mut tensors, "non_overlap_input")?)?,
        &take(&mut tensors, "non_overlap_output")?,
        "non_overlap_output",
    )?;
    Ok(())
}
