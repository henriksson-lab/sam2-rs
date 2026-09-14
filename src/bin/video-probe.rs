//! Runs two native SAM2 temporal steps using the typed inference state.

use std::{env, path::PathBuf, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use sam2_rs::{PointPrompt, Sam2Base, VideoInferenceState};

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: video-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let result = (|| -> candle_core::Result<()> {
        let model = Sam2Base::load_sam2_1_hiera_tiny(path)?;
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
        println!("frame 0 logits: {:?}", first.low_res_masks.dims());
        println!("frame 1 logits: {:?}", second.low_res_masks.dims());
        println!(
            "conditioning={}, propagated={}",
            state.conditioning().len(),
            state.propagated().len()
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
