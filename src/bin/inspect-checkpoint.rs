//! Inspect a converted SAM2 checkpoint before model construction.

use std::{env, path::PathBuf, process::ExitCode};

use sam2_rs::{CheckpointIndex, CheckpointManifest};

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: inspect-checkpoint <checkpoint.safetensors> [--sam2.1-hiera-tiny]");
        return ExitCode::from(2);
    };
    let validate_tiny = env::args()
        .skip(2)
        .any(|argument| argument == "--sam2.1-hiera-tiny");
    match CheckpointIndex::from_path(&path) {
        Ok(index) => {
            println!("{} tensors", index.len());
            if validate_tiny {
                match index.validate(&CheckpointManifest::sam2_1_hiera_tiny()) {
                    Ok(()) => println!("SAM2.1 Hiera-Tiny manifest: valid"),
                    Err(error) => {
                        eprintln!("SAM2.1 Hiera-Tiny manifest: {error}");
                        return ExitCode::from(1);
                    }
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
