//! Smoke test SAM2's direct binary-mask conditioning route.

use std::{env, path::PathBuf, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use sam2_rs::{Sam2Base, VideoInferenceState};

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: mask-video-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let result = (|| -> candle_core::Result<()> {
        let model = Sam2Base::load_sam2_1_hiera_tiny(path)?;
        let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &device)?;
        let mut mask_values = vec![0f32; 1024 * 1024];
        mask_values[512 * 1024 + 512] = 1.;
        let mask = Tensor::from_vec(mask_values, (1, 1, 1024, 1024), &device)?;
        let mut state = VideoInferenceState::new();
        let output = model.track_mask(&image, &mask, 0, true, &mut state, false, Some(1))?;
        println!(
            "mask-conditioned logits: {:?}; conditioning={}",
            output.low_res_masks.dims(),
            state.conditioning().len()
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
