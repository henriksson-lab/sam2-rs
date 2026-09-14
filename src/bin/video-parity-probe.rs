//! Compare the native two-frame tracker with the pinned Python fixture.

use std::{env, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{PointPrompt, Sam2Base, VideoInferenceState};

fn max_abs(actual: &Tensor, expected: &Tensor) -> f32 {
    actual
        .sub(expected)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(checkpoint) = args.next() else {
        eprintln!("usage: video-parity-probe <checkpoint.safetensors> <reference.safetensors>");
        return ExitCode::from(2);
    };
    let Some(reference) = args.next() else {
        eprintln!("usage: video-parity-probe <checkpoint.safetensors> <reference.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let result = (|| -> candle_core::Result<()> {
        let model = Sam2Base::load_sam2_1_hiera_tiny(checkpoint)?;
        let refs =
            unsafe { VarBuilder::from_mmaped_safetensors(&[reference], DType::F32, &device) }?;
        let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &device)?;
        let mut state = VideoInferenceState::new();
        let first = model.track(
            &image,
            0,
            true,
            &mut state,
            &[PointPrompt {
                x: 512.,
                y: 512.,
                positive: true,
            }],
            None,
            false,
            false,
        )?;
        let second = model.track(&image, 1, false, &mut state, &[], None, false, false)?;
        println!(
            "frame0 masks max_abs={:.8}",
            max_abs(
                &first.low_res_masks,
                &refs.get((1, 1, 256, 256), "frame0_low_res_masks")?
            )
        );
        println!(
            "frame0 high-res max_abs={:.8}",
            max_abs(
                &first.high_res_masks,
                &refs.get((1, 1, 1024, 1024), "frame0_high_res_masks")?,
            )
        );
        println!(
            "frame0 memory max_abs={:.8}",
            max_abs(
                &first.memory_features,
                &refs.get((1, 64, 64, 64), "frame0_memory_features")?
            )
        );
        println!(
            "frame0 memory-pos max_abs={:.8}",
            max_abs(
                &first.memory_pos_enc,
                &refs.get((1, 64, 64, 64), "frame0_memory_pos")?
            )
        );
        println!(
            "frame0 pointer max_abs={:.8}",
            max_abs(
                &first.object_pointer,
                &refs.get((1, 256), "frame0_object_pointer")?
            )
        );
        println!(
            "frame1 masks max_abs={:.8}",
            max_abs(
                &second.low_res_masks,
                &refs.get((1, 1, 256, 256), "frame1_low_res_masks")?
            )
        );
        println!(
            "frame1 conditioned max_abs={:.8}",
            max_abs(
                &second.conditioned_features,
                &refs.get((1, 256, 64, 64), "frame1_conditioned")?,
            )
        );
        println!(
            "frame1 memory max_abs={:.8}",
            max_abs(
                &second.memory_features,
                &refs.get((1, 64, 64, 64), "frame1_memory_features")?
            )
        );
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
