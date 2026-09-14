//! Compares deterministic Rust outputs with tensors saved by the pinned Python reference.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{ImageEncoder, MaskDecoder, PromptEncoder, Sam2Config, Sam2Variant};
use std::{env, process::ExitCode};

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
        eprintln!("usage: parity-probe <checkpoint.safetensors> <python-reference.safetensors>");
        return ExitCode::from(2);
    };
    let Some(reference) = args.next() else {
        eprintln!("usage: parity-probe <checkpoint.safetensors> <python-reference.safetensors>");
        return ExitCode::from(2);
    };
    let variant = match args.next().as_deref().unwrap_or("tiny") {
        "tiny" => Sam2Variant::HieraTiny,
        "small" => Sam2Variant::HieraSmall,
        "base-plus" => Sam2Variant::HieraBasePlus,
        "large" => Sam2Variant::HieraLarge,
        value => {
            eprintln!("unknown variant: {value}");
            return ExitCode::from(2);
        }
    };
    let device = Device::Cpu;
    let model_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device) }.unwrap();
    let reference_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[reference], DType::F32, &device) }.unwrap();
    let config = Sam2Config::sam2_1(variant);
    let image_encoder = ImageEncoder::load(&config, model_vb.pp("image_encoder")).unwrap();
    let prompt_encoder = PromptEncoder::load(&config, model_vb.pp("sam_prompt_encoder")).unwrap();
    let decoder = MaskDecoder::load(model_vb.pp("sam_mask_decoder")).unwrap();
    let image = Tensor::zeros((1, 3, 1024, 1024), DType::F32, &device).unwrap();
    let points = Tensor::from_vec(vec![512f32, 512.], (1, 1, 2), &device).unwrap();
    let labels = Tensor::from_vec(vec![1i64], (1, 1), &device).unwrap();
    let features = image_encoder.forward(&image).unwrap();
    let prompts = prompt_encoder
        .forward(Some((&points, &labels)), None, None)
        .unwrap();
    let masks = decoder
        .forward(
            &features.vision_features,
            &prompt_encoder.get_dense_pe().unwrap(),
            &prompts.sparse_embeddings,
            &prompts.dense_embeddings,
            false,
            &features.backbone_fpn[..2],
        )
        .unwrap();
    let ref_features = reference_vb
        .get((1, 256, 64, 64), "vision_features")
        .unwrap();
    let ref_sparse = reference_vb.get((1, 2, 256), "sparse_embeddings").unwrap();
    let ref_masks = reference_vb.get((1, 1, 256, 256), "mask_logits").unwrap();
    let ref_mask_tokens = reference_vb.get((1, 1, 256), "mask_tokens").unwrap();
    let ref_upscaled = reference_vb
        .get((1, 32, 256, 256), "upscaled_embedding")
        .unwrap();
    let ref_hyper = reference_vb.get((1, 4, 32), "hyper_in").unwrap();
    println!(
        "vision_features max_abs={:.8}",
        max_abs(&features.vision_features, &ref_features)
    );
    println!(
        "sparse_embeddings max_abs={:.8}",
        max_abs(&prompts.sparse_embeddings, &ref_sparse)
    );
    println!(
        "mask_logits max_abs={:.8}",
        max_abs(&masks.masks, &ref_masks)
    );
    println!(
        "mask_tokens max_abs={:.8}",
        max_abs(&masks.mask_tokens, &ref_mask_tokens)
    );
    println!(
        "upscaled_embedding max_abs={:.8}",
        max_abs(&masks.upscaled_embedding, &ref_upscaled)
    );
    println!(
        "hyper_in max_abs={:.8}",
        max_abs(&masks.hyper_in, &ref_hyper)
    );
    println!(
        "mask_logits reference max_abs={:.8}",
        ref_masks
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    );
    println!(
        "mask_logits Rust max_abs={:.8}",
        masks
            .masks
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    );
    println!(
        "upscaled reference max_abs={:.8}",
        ref_upscaled
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    );
    let reference_dot = ref_hyper
        .matmul(
            &ref_upscaled
                .contiguous()
                .unwrap()
                .reshape((1, 32, 256 * 256))
                .unwrap(),
        )
        .unwrap()
        .reshape((1, 4, 256, 256))
        .unwrap()
        .narrow(1, 0, 1)
        .unwrap();
    println!(
        "Candle dot(reference operands) max_abs={:.8}",
        max_abs(&reference_dot, &ref_masks)
    );
    ExitCode::SUCCESS
}
