//! Exact-shape CPU microbenchmark for SAM2 linear projections.

use std::{env, time::Instant};

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> candle_core::Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "fused".to_owned());
    if mode != "fused" && mode != "standard" {
        candle_core::bail!("usage: linear-probe [fused|standard] [rows input_dim output_dim]")
    }
    let dimensions = if args.len() == 4 {
        [
            args[1].parse::<usize>()?,
            args[2].parse::<usize>()?,
            args[3].parse::<usize>()?,
        ]
    } else if args.len() == 1 || args.is_empty() {
        [4096, 256, 128]
    } else {
        candle_core::bail!("usage: linear-probe [fused|standard] [rows input_dim output_dim]")
    };
    let [rows, input_dim, output_dim] = dimensions;
    if rows == 0 || input_dim == 0 || output_dim == 0 {
        candle_core::bail!("all dimensions must be positive")
    }
    let device = Device::Cpu;
    // The default is the decoder's K/V and image-query projection.
    let input = Tensor::randn(0f32, 1f32, (rows, input_dim), &device)?;
    let weight = Tensor::randn(0f32, 1f32, (output_dim, input_dim), &device)?;
    let bias = Tensor::randn(0f32, 1f32, output_dim, &device)?;
    let linear = Linear::new(weight, Some(bias));
    for _ in 0..3 {
        let output = if mode == "fused" {
            linear.forward_fused_bias_cpu(&input)?
        } else {
            linear.forward(&input)?
        };
        output.sum_all()?.to_scalar::<f32>()?;
    }
    let mut samples = Vec::with_capacity(11);
    let mut checksum = 0.;
    for _ in 0..11 {
        let start = Instant::now();
        let output = if mode == "fused" {
            linear.forward_fused_bias_cpu(&input)?
        } else {
            linear.forward(&input)?
        };
        checksum = output.sum_all()?.to_scalar::<f32>()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
    }
    println!(
        "{{\"operation\":\"linear_{rows}x{input_dim}x{output_dim}\",\"mode\":\"{mode}\",\"median_ms\":{:.6},\"checksum\":{checksum:.8}}}",
        median(samples)
    );
    Ok(())
}
