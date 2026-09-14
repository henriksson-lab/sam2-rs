//! Executes the real two-way transformer with checkpoint-loaded image and prompts.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{ImageEncoder, PromptEncoder, Sam2Config, TwoWayTransformer};
use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let Some(checkpoint) = env::args().nth(1) else {
        eprintln!("usage: transformer-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let vb =
        match unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device) } {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(1);
            }
        };
    let config = Sam2Config::hiera_tiny();
    let image_encoder = ImageEncoder::load(&config, vb.pp("image_encoder")).unwrap();
    let prompt_encoder = PromptEncoder::load(&config, vb.pp("sam_prompt_encoder")).unwrap();
    let transformer = TwoWayTransformer::load(vb.pp("sam_mask_decoder.transformer")).unwrap();
    let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &device).unwrap();
    let points = Tensor::from_vec(vec![512f32, 512.], (1, 1, 2), &device).unwrap();
    let labels = Tensor::from_vec(vec![1i64], (1, 1), &device).unwrap();
    let image_output = image_encoder.forward(&image).unwrap();
    let prompt_output = prompt_encoder
        .forward(Some((&points, &labels)), None, None)
        .unwrap();
    let src = image_output
        .vision_features
        .broadcast_add(&prompt_output.dense_embeddings)
        .unwrap();
    match transformer.forward(
        &src,
        &prompt_encoder.get_dense_pe().unwrap(),
        &prompt_output.sparse_embeddings,
    ) {
        Ok((queries, keys)) => {
            println!("queries: {:?}; keys: {:?}", queries.dims(), keys.dims());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
