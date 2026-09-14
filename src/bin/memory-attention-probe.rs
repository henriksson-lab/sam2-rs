//! Smoke-test the checkpoint-backed temporal transformer on its real token shapes.

use std::{env, path::PathBuf, process::ExitCode};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::MemoryAttention;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: memory-attention-probe <checkpoint.safetensors>");
        return ExitCode::from(2);
    };
    let device = Device::Cpu;
    let result = (|| -> candle_core::Result<()> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }?;
        let attention = MemoryAttention::load(vb.pp("memory_attention"))?;
        let curr = Tensor::zeros((4096, 1, 256), DType::F32, &device)?;
        let memory = Tensor::zeros((4096, 1, 64), DType::F32, &device)?;
        let curr_pos = Tensor::zeros((4096, 1, 256), DType::F32, &device)?;
        let memory_pos = Tensor::zeros((4096, 1, 64), DType::F32, &device)?;
        let output = attention.forward(&curr, &memory, &curr_pos, &memory_pos, 0)?;
        println!("temporal attention output shape: {:?}", output.dims());
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
