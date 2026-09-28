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
    let profile_decoder = args
        .iter()
        .position(|arg| arg == "--profile-decoder")
        .map(|index| args.remove(index))
        .is_some();
    let decoder_only = args
        .iter()
        .position(|arg| arg == "--decoder-only")
        .map(|index| args.remove(index))
        .is_some();
    let variant_name = args
        .iter()
        .position(|arg| arg == "--variant")
        .map(|index| {
            if index + 1 >= args.len() {
                String::new()
            } else {
                let name = args.remove(index + 1);
                args.remove(index);
                name
            }
        })
        .unwrap_or_else(|| "tiny".to_owned());
    if args.len() < 2 {
        eprintln!(
            "usage: bench-static <checkpoint.safetensors> <fixture.safetensors> [iterations] [output.f32] [--device cpu|cuda] [--variant tiny|small|base-plus|large] [--profile-stages] [--profile-encoder] [--profile-decoder] [--decoder-only]"
        );
        return ExitCode::from(2);
    }
    let iterations = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(3usize);
    let result = (|| -> candle_core::Result<()> {
        if decoder_only && profile_encoder {
            candle_core::bail!("--decoder-only cannot be combined with --profile-encoder")
        }
        let device = match device_name.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0)?,
            _ => candle_core::bail!("unsupported device {device_name:?}; expected cpu or cuda"),
        };
        #[cfg(feature = "cuda")]
        if device_name == "cuda"
            && !decoder_only
            && !profile_stages
            && !profile_encoder
            && !profile_decoder
            && std::env::var_os("SAM2_CUDA_GRAPH").is_some_and(|value| value == "1")
        {
            let Device::Cuda(cuda) = &device else {
                unreachable!()
            };
            // SAFETY: Static graph benchmarking uses only Candle's one device
            // stream and synchronizes every replay. Cross-stream lifetime
            // events therefore provide no ordering that this path needs.
            unsafe { cuda.disable_event_tracking() };
        }
        let variant = match variant_name.as_str() {
            "tiny" => Sam2Variant::HieraTiny,
            "small" => Sam2Variant::HieraSmall,
            "base-plus" | "base_plus" => Sam2Variant::HieraBasePlus,
            "large" => Sam2Variant::HieraLarge,
            _ => candle_core::bail!("unsupported variant {variant_name:?}"),
        };
        let predictor = Sam2ImagePredictor::load_sam2_1_on_device(variant, &args[0], &device)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&args[1]], DType::F32, &device) }?;
        let image = vb.get((1, 3, 1024, 1024), "image")?;
        let point = PointPrompt {
            x: 512.,
            y: 512.,
            positive: true,
        };
        let _ = predictor.predict(&image, &[point], None, false)?;
        device.synchronize()?;
        let decoder_features = if decoder_only {
            Some(predictor.encode_image(&image)?)
        } else {
            None
        };
        device.synchronize()?;
        #[cfg(feature = "cuda")]
        let graph_state = if device_name == "cuda"
            && !decoder_only
            && !profile_stages
            && !profile_encoder
            && !profile_decoder
            && std::env::var_os("SAM2_CUDA_GRAPH").is_some_and(|value| value == "1")
        {
            use candle_core::cuda_backend::cudarc::driver::sys::{
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED,
            };
            let Device::Cuda(cuda) = &device else {
                unreachable!()
            };
            let upload_cache = cuda.enable_cuda_graph_htod_cache();
            let _ = predictor.predict(&image, &[point], None, false)?;
            device.synchronize()?;
            let stream = cuda.cuda_stream();
            stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|error| {
                    candle_core::Error::Msg(format!("CUDA graph capture failed: {error}"))
                })?;
            let captured = predictor.predict(&image, &[point], None, false)?.masks;
            let graph = stream
                .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|error| {
                    candle_core::Error::Msg(format!("CUDA graph instantiation failed: {error}"))
                })?
                .ok_or_else(|| {
                    candle_core::Error::Msg("CUDA graph capture produced no graph".into())
                })?;
            device.synchronize()?;
            Some((graph, captured, upload_cache))
        } else {
            None
        };
        let mut samples = Vec::with_capacity(iterations);
        let mut last: Option<Tensor> = None;
        let mut stage_profile = None;
        let mut encoder_profile = None;
        let mut decoder_profile = None;
        for iteration in 0..iterations {
            let start = Instant::now();
            #[cfg(feature = "cuda")]
            if let Some((graph, captured, _upload_cache)) = &graph_state {
                graph.launch().map_err(|error| {
                    candle_core::Error::Msg(format!("CUDA graph launch failed: {error}"))
                })?;
                device.synchronize()?;
                samples.push(start.elapsed().as_secs_f64() * 1000.);
                last = Some(captured.clone());
                continue;
            }
            if (profile_stages || profile_encoder || profile_decoder) && iteration == 0 {
                let (features, profile) = if profile_encoder {
                    let (features, profile) = predictor.encode_image_profiled(&image)?;
                    (features, Some(profile))
                } else if let Some(features) = &decoder_features {
                    (features.clone(), None)
                } else {
                    (predictor.encode_image(&image)?, None)
                };
                device.synchronize()?;
                let encoder_ms = start.elapsed().as_secs_f64() * 1000.;
                let decoder_start = Instant::now();
                if profile_decoder {
                    let (output, profile) = predictor.decode_image_features_profiled(
                        &features,
                        &[point],
                        None,
                        None,
                        false,
                    )?;
                    decoder_profile = Some(profile);
                    last = Some(output.masks);
                } else {
                    last = Some(
                        predictor
                            .decode_image_features(&features, &[point], None, None, false)?
                            .masks,
                    );
                }
                device.synchronize()?;
                if profile_stages {
                    stage_profile =
                        Some((encoder_ms, decoder_start.elapsed().as_secs_f64() * 1000.));
                }
                encoder_profile = profile;
            } else if let Some(features) = &decoder_features {
                last = Some(
                    predictor
                        .decode_image_features(features, &[point], None, None, false)?
                        .masks,
                );
                device.synchronize()?;
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
            format!(",\"encoder_ms\":{encoder_ms:.6},\"decoder_ms\":{decoder_ms:.6}")
        });
        let encoder_fields = encoder_profile.map_or_else(String::new, |profile| {
            format!(
                ",\"hiera_setup_ms\":{:.6},\"hiera_block_ms\":{:?},\"hiera_norm1_ms\":{:?},\"hiera_shortcut_ms\":{:?},\"hiera_shortcut_pool_ms\":{:?},\"hiera_partition_ms\":{:?},\"hiera_attention_ms\":{:?},\"hiera_attention_qkv_ms\":{:?},\"hiera_attention_qkv_views_ms\":{:?},\"hiera_attention_query_pool_ms\":{:?},\"hiera_attention_query_reshape_ms\":{:?},\"hiera_attention_query_pool_core_ms\":{:?},\"hiera_attention_kernel_ms\":{:?},\"hiera_attention_projection_ms\":{:?},\"hiera_attention_cleanup_ms\":{:?},\"hiera_attention_cleanup_attended_ms\":{:?},\"hiera_attention_cleanup_qkv_ms\":{:?},\"hiera_unpartition_ms\":{:?},\"hiera_residual_add_ms\":{:?},\"hiera_norm2_ms\":{:?},\"hiera_mlp_ms\":{:?},\"hiera_mlp_expansion_ms\":{:?},\"hiera_mlp_activation_ms\":{:?},\"hiera_mlp_projection_ms\":{:?},\"fpn_ms\":{:.6},\"fpn_lateral_ms\":{:?},\"fpn_top_down_ms\":{:?},\"fpn_position_ms\":{:?}",
                profile.hiera_setup_ms, profile.hiera_block_ms, profile.hiera_norm1_ms, profile.hiera_shortcut_ms, profile.hiera_shortcut_pool_ms, profile.hiera_partition_ms, profile.hiera_attention_ms, profile.hiera_attention_qkv_ms, profile.hiera_attention_qkv_views_ms, profile.hiera_attention_query_pool_ms, profile.hiera_attention_query_reshape_ms, profile.hiera_attention_query_pool_core_ms, profile.hiera_attention_kernel_ms, profile.hiera_attention_projection_ms, profile.hiera_attention_cleanup_ms, profile.hiera_attention_cleanup_attended_ms, profile.hiera_attention_cleanup_qkv_ms, profile.hiera_unpartition_ms, profile.hiera_residual_add_ms, profile.hiera_norm2_ms, profile.hiera_mlp_ms, profile.hiera_mlp_expansion_ms, profile.hiera_mlp_activation_ms, profile.hiera_mlp_projection_ms, profile.fpn_ms, profile.fpn_lateral_ms, profile.fpn_top_down_ms, profile.fpn_position_ms
            )
        });
        let decoder_fields = decoder_profile.map_or_else(String::new, |profile| format!(
            ",\"decoder_prompt_encoder_ms\":{:.6},\"decoder_dense_pe_ms\":{:.6},\"decoder_token_setup_ms\":{:.6},\"decoder_transformer_ms\":{:.6},\"decoder_transformer_setup_ms\":{:.6},\"decoder_transformer_self_attention_ms\":{:?},\"decoder_transformer_token_to_image_ms\":{:?},\"decoder_transformer_mlp_ms\":{:?},\"decoder_transformer_image_to_token_ms\":{:?},\"decoder_transformer_final_attention_ms\":{:.6},\"decoder_cross_qkv_ms\":{:?},\"decoder_cross_kernel_ms\":{:?},\"decoder_cross_output_projection_ms\":{:?},\"decoder_cross_total_ms\":{:?},\"decoder_cross_pre_ms\":{:?},\"decoder_cross_post_ms\":{:?},\"decoder_upconv_0_ms\":{:.6},\"decoder_skip_1_ms\":{:.6},\"decoder_upconv_core_0_ms\":{:.6},\"decoder_upnorm_gelu_ms\":{:.6},\"decoder_upconv_3_ms\":{:.6},\"decoder_skip_0_ms\":{:.6},\"decoder_upconv_core_3_ms\":{:.6},\"decoder_final_gelu_ms\":{:.6},\"decoder_hypernetworks_ms\":{:.6},\"decoder_mask_projection_ms\":{:.6},\"decoder_heads_ms\":{:.6}",
            profile.prompt_encoder_ms, profile.dense_pe_ms, profile.token_setup_ms, profile.transformer_ms, profile.transformer.setup_ms, profile.transformer.self_attention_ms, profile.transformer.token_to_image_attention_ms, profile.transformer.mlp_ms, profile.transformer.image_to_token_attention_ms, profile.transformer.final_attention_ms, profile.transformer.cross_qkv_ms, profile.transformer.cross_kernel_ms, profile.transformer.cross_output_projection_ms, profile.transformer.cross_total_ms, profile.transformer.cross_pre_ms, profile.transformer.cross_post_ms, profile.upconv_0_ms, profile.skip_1_ms, profile.upconv_core_0_ms, profile.upnorm_gelu_ms, profile.upconv_3_ms, profile.skip_0_ms, profile.upconv_core_3_ms, profile.final_gelu_ms, profile.hypernetworks_ms, profile.mask_projection_ms, profile.heads_ms
        ));
        println!(
            "{{\"implementation\":\"rust-candle\",\"device\":\"{device_name}\",\"scope\":\"{}\",\"iterations\":{iterations},\"median_ms\":{:.6},\"rss_hwm_kb\":{},\"mask_checksum\":{checksum:.8}{stage_fields}{encoder_fields}{decoder_fields}}}",
            if decoder_only { "decoder" } else { "full" },
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
