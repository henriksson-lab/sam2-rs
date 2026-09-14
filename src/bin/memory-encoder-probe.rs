//! Executes checkpoint-loaded memory encoding for one 64x64 feature/mask pair.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::MemoryEncoder;
use std::{env, process::ExitCode};
fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: memory-encoder-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let vb = match unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) } {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let encoder = match MemoryEncoder::load(vb.pp("memory_encoder")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let features = Tensor::zeros((1, 256, 64, 64), DType::F32, &device).unwrap();
    let masks = Tensor::zeros((1, 1, 1024, 1024), DType::F32, &device).unwrap();
    match encoder.forward(&features, &masks, false) {
        Ok(output) => {
            println!(
                "features: {:?}; pos: {:?}",
                output.vision_features.dims(),
                output.vision_pos_enc.dims()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
