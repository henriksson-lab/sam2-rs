use cudaforge::{KernelBuilder, Result};
use std::path::PathBuf;

const CUTLASS_COMMIT: &str = "e05f953a5b3d38adc240df2ff928e0421c2abba3";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=cuda/mem_eff_attention.cu");
    println!("cargo::rerun-if-changed=cuda/mem_eff_attention");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let output = out_dir.join("libsam2_mem_eff_attention.a");
    KernelBuilder::new()
        .source_files(["cuda/mem_eff_attention.cu"])
        .include_path(manifest_dir)
        .with_cutlass(Some(CUTLASS_COMMIT))
        .out_dir(&out_dir)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .build_lib(output)?;

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::rustc-link-lib=static=sam2_mem_eff_attention");
    println!("cargo::rustc-link-lib=dylib=cudart");
    println!("cargo::rustc-link-lib=dylib=stdc++");
    Ok(())
}
