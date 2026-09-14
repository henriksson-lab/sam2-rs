//! Reusable steady-state static inference benchmark over a shared fixture.

use std::{env, fs, process::ExitCode, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use sam2_rs::{PointPrompt, Sam2ImagePredictor, Sam2Variant};

fn hwm_kb() -> usize {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
            })
        })
        .unwrap_or(0)
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> ExitCode {
    let mut args: Vec<_> = env::args().skip(1).collect();
    let device_name = args
        .iter()
        .position(|arg| arg == "--device")
        .map(|index| {
            if index + 1 >= args.len() {
                String::new()
            } else {
                let name = args.remove(index + 1);
                args.remove(index);
                name
            }
        })
        .unwrap_or_else(|| "cpu".to_owned());
    let profile_stages = args
        .iter()
        .position(|arg| arg == "--profile-stages")
        .map(|index| args.remove(index))
        .is_some();
    let profile_encoder = args
        .iter()
        .position(|arg| arg == "--profile-encoder")
        .map(|index| args.remove(index))
        .is_some();
    if args.len() < 2 {
        eprintln!(
            "usage: bench-static <checkpoint.safetensors> <fixture.safetensors> [iterations] [output.f32] [--device cpu|cuda] [--profile-stages] [--profile-encoder]"
        );
        return ExitCode::from(2);
    }
    let iterations = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(3usize);
    let result = (|| -> candle_core::Result<()> {
        let device = match device_name.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0)?,
            _ => candle_core::bail!("unsupported device {device_name:?}; expected cpu or cuda"),
        };
        let predictor =
            Sam2ImagePredictor::load_sam2_1_on_device(Sam2Variant::HieraTiny, &args[0], &device)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&args[1]], DType::F32, &device) }?;
        let image = vb.get((1, 3, 1024, 1024), "image")?;
        let point = PointPrompt {
            x: 512.,
            y: 512.,
            positive: true,
        };
        let _ = predictor.predict(&image, &[point], None, false)?;
        device.synchronize()?;
        let mut samples = Vec::with_capacity(iterations);
        let mut last: Option<Tensor> = None;
        let mut stage_profile = None;
        let mut encoder_profile = None;
        for iteration in 0..iterations {
            let start = Instant::now();
            if (profile_stages || profile_encoder) && iteration == 0 {
                let (features, profile) = if profile_encoder {
                    let (features, profile) = predictor.encode_image_profiled(&image)?;
                    (features, Some(profile))
                } else {
                    (predictor.encode_image(&image)?, None)
                };
                device.synchronize()?;
                let encoder_ms = start.elapsed().as_secs_f64() * 1000.;
                let decoder_start = Instant::now();
                last = Some(
                    predictor
                        .decode_image_features(&features, &[point], None, None, false)?
                        .masks,
                );
                device.synchronize()?;
                if profile_stages {
                    stage_profile =
                        Some((encoder_ms, decoder_start.elapsed().as_secs_f64() * 1000.));
                }
                encoder_profile = profile;
            } else {
                last = Some(predictor.predict(&image, &[point], None, false)?.masks);
                device.synchronize()?;
            }
            samples.push(start.elapsed().as_secs_f64() * 1000.);
        }
        let output = last.unwrap();
        if let Some(path) = args.get(3) {
            let values = output.flatten_all()?.to_vec1::<f32>()?;
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            fs::write(path, bytes)?;
        }
        let checksum = output.sum_all()?.to_scalar::<f32>()?;
        let stage_fields = stage_profile.map_or_else(String::new, |(encoder_ms, decoder_ms)| {
            format!("\",\"encoder_ms\":{encoder_ms:.6},\"decoder_ms\":{decoder_ms:.6}")
        });
        let encoder_fields = encoder_profile.map_or_else(String::new, |profile| {
            format!(
                ",\"hiera_setup_ms\":{:.6},\"hiera_block_ms\":{:?},\"fpn_ms\":{:.6}",
                profile.hiera_setup_ms, profile.hiera_block_ms, profile.fpn_ms
            )
        });
        println!(
            "{{\"implementation\":\"rust-candle\",\"device\":\"{device_name}\",\"iterations\":{iterations},\"median_ms\":{:.6},\"rss_hwm_kb\":{},\"mask_checksum\":{checksum:.8}{stage_fields}{encoder_fields}}}",
            median(samples),
            hwm_kb()
        );
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
