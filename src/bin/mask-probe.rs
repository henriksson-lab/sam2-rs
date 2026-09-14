//! Executes a real SAM2.1 Tiny static mask-decoder pass.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{ImageEncoder, MaskDecoder, PromptEncoder, Sam2Config};
use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let Some(checkpoint) = env::args().nth(1) else {
        eprintln!("usage: mask-probe <checkpoint.safetensors>");
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
    let decoder = MaskDecoder::load(vb.pp("sam_mask_decoder")).unwrap();
    let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &device).unwrap();
    let points = Tensor::from_vec(vec![512f32, 512.], (1, 1, 2), &device).unwrap();
    let labels = Tensor::from_vec(vec![1i64], (1, 1), &device).unwrap();
    let features = image_encoder.forward(&image).unwrap();
    let prompts = prompt_encoder
        .forward(Some((&points, &labels)), None, None)
        .unwrap();
    match decoder.forward(
        &features.vision_features,
        &prompt_encoder.get_dense_pe().unwrap(),
        &prompts.sparse_embeddings,
        &prompts.dense_embeddings,
        false,
        &features.backbone_fpn[..2],
    ) {
        Ok(output) => {
            println!(
                "masks: {:?}; IoU: {:?}; object scores: {:?}",
                output.masks.dims(),
                output.iou_predictions.dims(),
                output.object_score_logits.dims()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
