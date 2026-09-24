//! Compare published PyTorch archives with converted safetensors manifests.

use std::path::PathBuf;

use sam2_rs::{CheckpointIndex, CheckpointManifest, Sam2Base, Sam2Variant};

fn variant(name: &str) -> candle_core::Result<Sam2Variant> {
    match name {
        "tiny" => Ok(Sam2Variant::HieraTiny),
        "small" => Ok(Sam2Variant::HieraSmall),
        "base-plus" => Ok(Sam2Variant::HieraBasePlus),
        "large" => Ok(Sam2Variant::HieraLarge),
        _ => candle_core::bail!("unknown variant {name}"),
    }
}

fn main() -> candle_core::Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.is_empty() || arguments.len() % 3 != 0 {
        candle_core::bail!(
            "usage: r7-checkpoint-probe <variant> <checkpoint.pt> <checkpoint.safetensors> [...]"
        )
    }
    for group in arguments.chunks_exact(3) {
        let variant = variant(&group[0])?;
        let pytorch_path = PathBuf::from(&group[1]);
        let safetensors_path = PathBuf::from(&group[2]);
        let pytorch = CheckpointIndex::from_path(&pytorch_path)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let safetensors = CheckpointIndex::from_path(&safetensors_path)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let manifest = CheckpointManifest::sam2_1(variant);
        pytorch
            .validate(&manifest)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        safetensors
            .validate(&manifest)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        if pytorch != safetensors {
            candle_core::bail!("{} archive indexes differ", group[0])
        }
        drop(Sam2Base::load_sam2_1(variant, &pytorch_path)?);
        println!(
            "{}: {} names, shapes, and dtypes exact; .pt model loaded",
            group[0],
            pytorch.len()
        );
    }
    Ok(())
}
