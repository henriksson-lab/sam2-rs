//! Checks Tiny prompt weights and point/box encoding against their checkpoint paths.

use std::{env, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{PromptEncoder, Sam2Config};

fn main() -> ExitCode {
    let Some(checkpoint) = env::args().nth(1) else {
        eprintln!("usage: prompt-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let builder =
        match unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device) } {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(1);
            }
        };
    let encoder =
        match PromptEncoder::load(&Sam2Config::hiera_tiny(), builder.pp("sam_prompt_encoder")) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(1);
            }
        };
    let points = Tensor::from_vec(vec![128f32, 256., 700., 800.], (1, 2, 2), &device).unwrap();
    let labels = Tensor::from_vec(vec![1i64, 0], (1, 2), &device).unwrap();
    let boxes = Tensor::from_vec(vec![100f32, 150., 900., 800.], (1, 4), &device).unwrap();
    match encoder.forward(Some((&points, &labels)), Some(&boxes), None) {
        Ok(output) => {
            println!(
                "sparse: {:?}; dense: {:?}; dense PE: {:?}",
                output.sparse_embeddings.dims(),
                output.dense_embeddings.dims(),
                encoder.get_dense_pe().unwrap().dims()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
