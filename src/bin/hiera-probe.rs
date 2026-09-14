//! Loads a converted checkpoint and executes the SAM2.1 Hiera-Tiny trunk.

use std::{env, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{ImageEncoder, Sam2Config};

fn main() -> ExitCode {
    let Some(checkpoint) = env::args().nth(1) else {
        eprintln!("usage: hiera-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let builder =
        match unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device) } {
            Ok(builder) => builder,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(1);
            }
        };
    let model = match ImageEncoder::load(&Sam2Config::hiera_tiny(), builder.pp("image_encoder")) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };
    let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &device).unwrap();
    match model.forward(&image) {
        Ok(output) => {
            println!("vision features: {:?}", output.vision_features.dims());
            for (index, feature) in output.backbone_fpn.iter().enumerate() {
                println!("FPN stage {index}: {:?}", feature.dims());
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
