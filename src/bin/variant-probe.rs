//! Runtime smoke test for each published SAM2.1 Hiera checkpoint variant.

use std::{env, path::PathBuf, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use sam2_rs::{PointPrompt, Sam2ImagePredictor, Sam2Variant};

fn variant(name: &str) -> Option<Sam2Variant> {
    match name {
        "tiny" => Some(Sam2Variant::HieraTiny),
        "small" => Some(Sam2Variant::HieraSmall),
        "base-plus" => Some(Sam2Variant::HieraBasePlus),
        "large" => Some(Sam2Variant::HieraLarge),
        _ => None,
    }
}

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(name) = args.next().and_then(|value| value.into_string().ok()) else {
        eprintln!("usage: variant-probe <tiny|small|base-plus|large> <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let Some(path) = args.next().map(PathBuf::from) else {
        eprintln!("usage: variant-probe <tiny|small|base-plus|large> <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let Some(variant) = variant(&name) else {
        eprintln!("unknown SAM2.1 Hiera variant: {name}");
        return ExitCode::from(2);
    };
    let result = (|| -> candle_core::Result<()> {
        let predictor = Sam2ImagePredictor::load_sam2_1(variant, path)?;
        let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &Device::Cpu)?;
        let output = predictor.predict(
            &image,
            &[PointPrompt {
                x: 512.,
                y: 512.,
                positive: true,
            }],
            None,
            false,
        )?;
        println!("{name}: {:?}", output.masks.dims());
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
