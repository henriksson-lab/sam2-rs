//! Independently runnable CPU/GPU parity-suite cells.
//!
//! This binary deliberately has no TOML/JSON dependency. The Python
//! orchestrator expands `benchmarks/cpu-suite.toml` and invokes one process per
//! variant/fixture cell; this process emits one stable JSON Lines record.

use std::{
    collections::HashMap,
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use candle_core::{Device, Tensor};
use sam2_rs::{
    AutomaticMaskConfig, AutomaticMaskOutputMode, AutomaticMaskSegmentation, BoxPrompt,
    ImagePromptBatch, PointPrompt, Sam2AutomaticMaskGenerator, Sam2ImagePredictor,
    Sam2InteractiveVideoPredictor, Sam2Variant, VideoPredictorConfig,
};

#[derive(Debug)]
struct Args {
    cell: String,
    operation: String,
    variant: String,
    checkpoint: PathBuf,
    fixture: PathBuf,
    iterations: usize,
    warmups: usize,
    output: Option<PathBuf>,
    video_lowres_output: Option<PathBuf>,
    device: String,
    mode: String,
    sessions: usize,
    point_counts: Vec<usize>,
    batch_fixtures: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut values = env::args().skip(1);
    let mut map = HashMap::new();
    while let Some(key) = values.next() {
        if !key.starts_with("--") {
            return Err(format!("unexpected positional argument {key:?}"));
        }
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        map.insert(key, value);
    }
    let required = |key: &str| {
        map.get(key)
            .cloned()
            .ok_or_else(|| format!("missing {key}"))
    };
    Ok(Args {
        cell: required("--cell")?,
        operation: required("--operation")?,
        variant: required("--variant")?,
        checkpoint: required("--checkpoint")?.into(),
        fixture: required("--fixture")?.into(),
        iterations: map.get("--iterations").map_or(Ok(3), |value| {
            value.parse().map_err(|_| "invalid --iterations")
        })?,
        warmups: map.get("--warmups").map_or(Ok(1), |value| {
            value.parse().map_err(|_| "invalid --warmups")
        })?,
        output: map.get("--output").map(PathBuf::from),
        video_lowres_output: map.get("--video-lowres-output").map(PathBuf::from),
        device: map.get("--device").cloned().unwrap_or_else(|| "cpu".into()),
        mode: map.get("--mode").cloned().unwrap_or_else(|| "eager".into()),
        sessions: map.get("--sessions").map_or(Ok(10), |value| {
            value.parse().map_err(|_| "invalid --sessions")
        })?,
        point_counts: map
            .get("--point-counts")
            .map_or(Ok(vec![0, 1, 8, 32]), |value| {
                value
                    .split(',')
                    .map(|count| count.parse().map_err(|_| "invalid --point-counts"))
                    .collect()
            })?,
        batch_fixtures: map
            .get("--batch-fixtures")
            .map(|value| value.split(',').map(PathBuf::from).collect())
            .unwrap_or_default(),
    })
}

fn write_v3_lowres_outputs(
    path: &Path,
    session: &sam2_rs::InteractiveVideoSession,
    frame_count: usize,
    object_ids: &[u32],
) -> candle_core::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for frame_index in 0..frame_count {
        for &object_id in object_ids {
            let output = session.output(object_id, frame_index).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "missing V3 low-resolution output for object {object_id}, frame {frame_index}"
                ))
            })?;
            for value in output.low_res_masks.flatten_all()?.to_vec1::<f32>()? {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn run_image_batch(
    args: &Args,
    device: &Device,
    predictor: &Sam2ImagePredictor,
    load_ms: f64,
    rss_after_load_kb: usize,
) -> candle_core::Result<()> {
    if args.batch_fixtures.is_empty() {
        candle_core::bail!("image_batch requires --batch-fixtures")
    }
    let fixture_start = Instant::now();
    let mut images = Vec::with_capacity(args.batch_fixtures.len());
    let mut points = Vec::with_capacity(args.batch_fixtures.len());
    for path in &args.batch_fixtures {
        let mut tensors = candle_core::safetensors::load(path, device)?;
        images.push(
            tensors
                .remove("image")
                .ok_or_else(|| candle_core::Error::Msg(format!("{path:?} has no image tensor")))?,
        );
        points.push(
            tensors
                .remove("point")
                .ok_or_else(|| candle_core::Error::Msg(format!("{path:?} has no point tensor")))?,
        );
    }
    let image_refs = images.iter().collect::<Vec<_>>();
    let point_refs = points.iter().collect::<Vec<_>>();
    let images = Tensor::cat(&image_refs, 0)?;
    let points = Tensor::cat(&point_refs, 0)?;
    let labels = Tensor::ones(
        points.dims()[..points.rank() - 1].to_vec(),
        candle_core::DType::I64,
        device,
    )?;
    let fixture_decode_ms = fixture_start.elapsed().as_secs_f64() * 1000.;
    let run_once = || -> candle_core::Result<(Tensor, f64, f64)> {
        let encode_start = Instant::now();
        let features = predictor.encode_image(&images)?;
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.;
        let decode_start = Instant::now();
        let masks = predictor
            .decode_image_features_batch(
                &features,
                &ImagePromptBatch {
                    point_coords: Some(points.clone()),
                    point_labels: Some(labels.clone()),
                    ..Default::default()
                },
                false,
            )?
            .masks;
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.;
        Ok((masks, encode_ms, decode_ms))
    };
    for _ in 0..args.warmups {
        let _ = run_once()?;
        device.synchronize()?;
    }
    reset_gpu_memory_stats(device)?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut encode_samples = Vec::with_capacity(args.iterations);
    let mut decode_samples = Vec::with_capacity(args.iterations);
    let mut output = None;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let (masks, encode_ms, decode_ms) = run_once()?;
        output = Some(masks);
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
        encode_samples.push(encode_ms);
        decode_samples.push(decode_ms);
    }
    let output = output.unwrap();
    let parity_values = output.flatten_all()?.to_vec1::<f32>()?;
    if let Some(path) = &args.output {
        let bytes = parity_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(path, bytes)?;
    }
    let median_ms = median(samples.clone());
    let p95_ms = quantile(samples.clone(), 0.95);
    let checksum = parity_values.iter().map(|&value| value as f64).sum::<f64>();
    let encode_ms = encode_samples.iter().sum::<f64>() / encode_samples.len() as f64;
    let decode_ms = decode_samples.iter().sum::<f64>() / decode_samples.len() as f64;
    let gpu_memory = gpu_memory_json(device)?;
    println!(
        "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"mode\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":{},\"warmups\":{},\"samples_ms\":{},\"median_ms\":{median_ms:.6},\"p95_ms\":{p95_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6},\"fixture_decode\":{fixture_decode_ms:.6},\"encode\":{encode_ms:.6},\"decode\":{decode_ms:.6},\"capture\":null}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"after_encode\":null,\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"output\":{{\"checksum\":{checksum:.8},\"elements\":{}}}}}",
        args.device,
        args.mode,
        args.cell,
        args.operation,
        args.variant,
        args.iterations,
        args.warmups,
        json_array(&samples),
        current_rss_kb(),
        hwm_kb(),
        parity_values.len(),
    );
    Ok(())
}

fn variant(name: &str) -> Result<Sam2Variant, String> {
    match name {
        "tiny" => Ok(Sam2Variant::HieraTiny),
        "small" => Ok(Sam2Variant::HieraSmall),
        "base-plus" | "base_plus" => Ok(Sam2Variant::HieraBasePlus),
        "large" => Ok(Sam2Variant::HieraLarge),
        _ => Err(format!("unsupported variant {name:?}")),
    }
}

fn status_kb(field: &str) -> usize {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix(field)
                    .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
            })
        })
        .unwrap_or(0)
}

fn current_rss_kb() -> usize {
    status_kb("VmRSS:")
}

fn hwm_kb() -> usize {
    status_kb("VmHWM:")
}

fn quantile(mut values: Vec<f64>, fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * fraction).ceil() as usize;
    values[index]
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.
    } else {
        values[middle]
    }
}

fn tensor_points(coords: &Tensor, labels: &Tensor) -> candle_core::Result<Vec<PointPrompt>> {
    let coords = coords.flatten_all()?.to_vec1::<f32>()?;
    let labels = labels.flatten_all()?.to_vec1::<i64>()?;
    if coords.len() != labels.len() * 2 {
        candle_core::bail!("point-coordinate and label counts differ")
    }
    Ok(labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| PointPrompt {
            x: coords[index * 2],
            y: coords[index * 2 + 1],
            positive: label != 0,
        })
        .collect())
}

fn point_only(tensors: &HashMap<String, Tensor>) -> candle_core::Result<Vec<PointPrompt>> {
    tensor_points(
        tensors
            .get("point")
            .ok_or_else(|| candle_core::Error::Msg("missing point".into()))?,
        &Tensor::new(&[[1i64]], &Device::Cpu)?,
    )
}

fn rich_prompts(
    tensors: &HashMap<String, Tensor>,
) -> candle_core::Result<(Vec<PointPrompt>, BoxPrompt)> {
    let points = tensor_points(
        tensors
            .get("rich_points")
            .ok_or_else(|| candle_core::Error::Msg("missing rich_points".into()))?,
        tensors
            .get("rich_labels")
            .ok_or_else(|| candle_core::Error::Msg("missing rich_labels".into()))?,
    )?;
    let bounds = tensors
        .get("box")
        .ok_or_else(|| candle_core::Error::Msg("missing box".into()))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((
        points,
        BoxPrompt {
            x0: bounds[0],
            y0: bounds[1],
            x1: bounds[2],
            y1: bounds[3],
        },
    ))
}

fn json_array(values: &[f64]) -> String {
    let body = values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn reset_gpu_memory_stats(device: &Device) -> candle_core::Result<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = device {
        cuda.reset_memory_pool_peak_stats()?;
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(())
}

fn gpu_memory_json(device: &Device) -> candle_core::Result<String> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda) = device {
        let stats = cuda.memory_pool_stats()?;
        return Ok(format!(
            "{{\"allocated\":{},\"reserved\":{},\"peak_allocated\":{},\"peak_reserved\":{}}}",
            stats.used_current, stats.reserved_current, stats.used_high, stats.reserved_high
        ));
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(
        "{\"allocated\":null,\"reserved\":null,\"peak_allocated\":null,\"peak_reserved\":null}"
            .into(),
    )
}

fn encode_automatic_masks(masks: &[sam2_rs::AutomaticMask]) -> candle_core::Result<Vec<f32>> {
    let mut values = Vec::new();
    values.push(masks.len() as f32);
    for mask in masks {
        values.extend(mask.bbox);
        values.push(mask.area as f32);
        values.push(mask.predicted_iou);
        values.extend(mask.point);
        values.push(mask.stability_score);
        values.extend(mask.crop_box);
        let AutomaticMaskSegmentation::UncompressedRle(rle) = &mask.segmentation else {
            candle_core::bail!("AMG benchmark requires uncompressed RLE output")
        };
        values.push(rle.counts.len() as f32);
        values.extend(rle.counts.iter().map(|&count| count as f32));
    }
    Ok(values)
}

fn run_amg(args: &Args, device: &Device) -> candle_core::Result<()> {
    let config = AutomaticMaskConfig {
        points_per_side: Some(if args.operation == "amg_practical" {
            8
        } else {
            32
        }),
        use_m2m: args.operation == "amg_practical",
        output_mode: AutomaticMaskOutputMode::UncompressedRle,
        ..Default::default()
    };
    let load_start = Instant::now();
    let generator = Sam2AutomaticMaskGenerator::load_sam2_1_on_device(
        variant(&args.variant).map_err(candle_core::Error::Msg)?,
        &args.checkpoint,
        device,
        config,
    )?;
    device.synchronize()?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
    let rss_after_load_kb = current_rss_kb();

    let fixture_start = Instant::now();
    let tensors = candle_core::safetensors::load(&args.fixture, &Device::Cpu)?;
    let rgb = tensors
        .get("rgb_u8")
        .ok_or_else(|| candle_core::Error::Msg("fixture has no rgb_u8 tensor".into()))?;
    let (channels, height, width) = rgb.dims3()?;
    if channels != 3 {
        candle_core::bail!("rgb_u8 must be CHW with three channels")
    }
    // Fixtures store CHW for tensor consumers; the public AMG API accepts
    // interleaved HWC bytes.
    let rgb_hwc = rgb
        .permute((1, 2, 0))?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let fixture_decode_ms = fixture_start.elapsed().as_secs_f64() * 1000.;

    for _ in 0..args.warmups {
        let _ = generator.generate_rgb8(&rgb_hwc, width, height)?;
        device.synchronize()?;
    }
    reset_gpu_memory_stats(device)?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut last = Vec::new();
    for _ in 0..args.iterations {
        let start = Instant::now();
        last = generator.generate_rgb8(&rgb_hwc, width, height)?;
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let parity_values = encode_automatic_masks(&last)?;
    if let Some(path) = &args.output {
        let bytes = parity_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(path, bytes)?;
    }
    let checksum = parity_values.iter().map(|&value| value as f64).sum::<f64>();
    let median_ms = median(samples.clone());
    let p95_ms = quantile(samples.clone(), 0.95);
    let gpu_memory = gpu_memory_json(device)?;
    println!(
        "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":{},\"warmups\":{},\"samples_ms\":{},\"median_ms\":{median_ms:.6},\"p95_ms\":{p95_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6},\"fixture_decode\":{fixture_decode_ms:.6}}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"output\":{{\"checksum\":{checksum:.8},\"elements\":{},\"masks\":{}}}}}",
        args.device,
        args.cell,
        args.operation,
        args.variant,
        args.iterations,
        args.warmups,
        json_array(&samples),
        current_rss_kb(),
        hwm_kb(),
        parity_values.len(),
        last.len(),
    );
    Ok(())
}

fn run_video(args: &Args, device: &Device) -> candle_core::Result<()> {
    let load_start = Instant::now();
    let predictor = Sam2InteractiveVideoPredictor::load_sam2_1_on_device(
        variant(&args.variant).map_err(candle_core::Error::Msg)?,
        &args.checkpoint,
        device,
        VideoPredictorConfig::default(),
    )?;
    device.synchronize()?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
    let rss_after_load_kb = current_rss_kb();
    let gpu_after_load = gpu_memory_json(device)?;

    // The fixture is decoded RGB by construction. Loading it on CPU keeps the
    // same boundary as Python: session initialization owns normalization and
    // host-to-device transfer, while JPEG/FFmpeg are separate suite cells.
    let fixture_start = Instant::now();
    let tensors = candle_core::safetensors::load(&args.fixture, &Device::Cpu)?;
    let mut names = tensors
        .keys()
        .filter(|name| name.starts_with("frame_"))
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    if names.is_empty() {
        candle_core::bail!("video fixture has no frame tensors")
    }
    let mut owned_frames = Vec::with_capacity(names.len());
    let mut width = 0;
    let mut height = 0;
    for name in names {
        let frame = &tensors[&name];
        let (frame_height, frame_width, channels) = frame.dims3()?;
        if channels != 3
            || (!owned_frames.is_empty() && (width, height) != (frame_width, frame_height))
        {
            candle_core::bail!("video fixture has invalid or inconsistent RGB frames")
        }
        width = frame_width;
        height = frame_height;
        owned_frames.push(frame.flatten_all()?.to_vec1::<u8>()?);
    }
    let fixture_decode_ms = fixture_start.elapsed().as_secs_f64() * 1000.;
    let inputs = owned_frames
        .iter()
        .map(|frame| (frame.as_slice(), width, height))
        .collect::<Vec<_>>();

    let mut samples = Vec::with_capacity(args.iterations);
    let mut session_init_ms = 0.;
    let mut checksum = 0f32;
    let mut elements = 0usize;
    let mut rss_after_init_kb = 0usize;
    let mut rss_after_reset_kb = None;
    let mut parity_values = Vec::new();
    let mut video_profile = None;
    let mut gpu_after_init = "null".to_string();
    let mut gpu_after_interaction = "null".to_string();
    let mut gpu_after_propagation = "null".to_string();
    let mut gpu_after_reset = "null".to_string();
    let mut gpu_after_drop = "null".to_string();
    for iteration in 0..(args.warmups + args.iterations) {
        if iteration == args.warmups {
            device.synchronize()?;
            reset_gpu_memory_stats(device)?;
        }
        let init_start = Instant::now();
        let mut session = predictor.init_state(&inputs)?;
        device.synchronize()?;
        let current_init_ms = init_start.elapsed().as_secs_f64() * 1000.;
        rss_after_init_kb = current_rss_kb();
        if iteration >= args.warmups {
            gpu_after_init = gpu_memory_json(device)?;
        }
        let measured_start = Instant::now();
        let outputs = match args.operation.as_str() {
            "video_init" => Vec::new(),
            "video_forward_one" | "video_long_lifetime" | "video_forward_profile" => {
                predictor.add_new_points(
                    &mut session,
                    0,
                    1,
                    &[PointPrompt {
                        x: 210.,
                        y: 350.,
                        positive: true,
                    }],
                    true,
                )?;
                if iteration >= args.warmups {
                    gpu_after_interaction = gpu_memory_json(device)?;
                }
                if args.operation == "video_forward_profile" {
                    let (outputs, profile) = predictor.propagate_in_video_profiled(
                        &mut session,
                        Some(0),
                        None,
                        false,
                    )?;
                    video_profile = Some(profile);
                    outputs
                } else {
                    predictor.propagate_in_video(&mut session, Some(0), None, false)?
                }
            }
            "video_edit_three" => {
                for (object_id, x, y) in [(1, 210., 350.), (2, 430., 310.), (3, 600., 345.)] {
                    predictor.add_new_points(
                        &mut session,
                        0,
                        object_id,
                        &[PointPrompt {
                            x,
                            y,
                            positive: true,
                        }],
                        true,
                    )?;
                }
                if iteration >= args.warmups {
                    gpu_after_interaction = gpu_memory_json(device)?;
                }
                let mut outputs =
                    predictor.propagate_in_video(&mut session, Some(0), None, false)?;
                if iteration + 1 == args.warmups + args.iterations
                    && let Some(path) = &args.video_lowres_output
                {
                    write_v3_lowres_outputs(path, &session, inputs.len(), &[1, 2, 3])?;
                }
                let middle = inputs.len() / 2;
                outputs.push(predictor.add_new_points(
                    &mut session,
                    middle,
                    1,
                    &[PointPrompt {
                        x: 250.,
                        y: 220.,
                        positive: true,
                    }],
                    true,
                )?);
                outputs.extend(predictor.propagate_in_video(
                    &mut session,
                    Some(middle),
                    Some(middle + 1),
                    true,
                )?);
                let _ = predictor.remove_object_with_outputs(&mut session, 3)?;
                outputs
            }
            name => candle_core::bail!("unsupported video operation {name:?}"),
        };
        device.synchronize()?;
        if iteration >= args.warmups {
            gpu_after_propagation = gpu_memory_json(device)?;
        }
        let elapsed = if args.operation == "video_init" {
            current_init_ms
        } else {
            measured_start.elapsed().as_secs_f64() * 1000.
        };
        if iteration >= args.warmups {
            samples.push(elapsed);
            session_init_ms = current_init_ms;
            checksum = 0.;
            elements = 0;
            parity_values.clear();
            let selected = [0usize, outputs.len() / 2, outputs.len().saturating_sub(1)];
            for (index, output) in outputs.into_iter().enumerate() {
                checksum += output.masks.sum_all()?.to_scalar::<f32>()?;
                elements += output.masks.elem_count();
                if iteration + 1 == args.warmups + args.iterations && selected.contains(&index) {
                    parity_values.extend(output.masks.flatten_all()?.to_vec1::<f32>()?);
                }
            }
        }
        if args.operation == "video_long_lifetime" {
            predictor.reset_state(&mut session);
            device.synchronize()?;
            if iteration >= args.warmups {
                gpu_after_reset = gpu_memory_json(device)?;
            }
            drop(session);
            device.synchronize()?;
            if iteration >= args.warmups {
                gpu_after_drop = gpu_memory_json(device)?;
            }
            rss_after_reset_kb = Some(current_rss_kb());
        }
    }
    let median_ms = median(samples.clone());
    let p95_ms = quantile(samples.clone(), 0.95);
    let after_reset = rss_after_reset_kb.map_or("null".into(), |value| value.to_string());
    let gpu_memory = gpu_memory_json(device)?;
    let profile_fields = video_profile.map_or_else(String::new, |profile| {
        let sum = |values: &[f64]| values.iter().sum::<f64>();
        format!(
            ",\"video_preflight_ms\":{:.6},\"video_image_encoder_ms\":{:.6},\"video_temporal_and_heads_ms\":{:.6},\"video_restore_masks_ms\":{:.6}",
            profile.preflight_ms,
            sum(&profile.image_encoder_ms),
            sum(&profile.temporal_and_heads_ms),
            sum(&profile.restore_masks_ms),
        )
    });
    if let Some(path) = &args.output {
        let bytes = parity_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(path, bytes)?;
    }
    println!(
        "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":{},\"warmups\":{},\"samples_ms\":{},\"median_ms\":{median_ms:.6},\"p95_ms\":{p95_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6},\"fixture_decode\":{fixture_decode_ms:.6},\"session_init\":{session_init_ms:.6}{profile_fields}}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"after_init\":{rss_after_init_kb},\"after_reset\":{after_reset},\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"gpu_memory_phases_bytes\":{{\"after_load\":{gpu_after_load},\"after_init\":{gpu_after_init},\"after_interaction\":{gpu_after_interaction},\"after_propagation\":{gpu_after_propagation},\"after_reset\":{gpu_after_reset},\"after_drop\":{gpu_after_drop}}},\"output\":{{\"checksum\":{checksum:.8},\"elements\":{elements}}}}}",
        args.device,
        args.cell,
        args.operation,
        args.variant,
        args.iterations,
        args.warmups,
        json_array(&samples),
        current_rss_kb(),
        hwm_kb(),
    );
    Ok(())
}

#[cfg(feature = "video-decoding")]
fn run_encoded_video(args: &Args, device: &Device) -> candle_core::Result<()> {
    let load_start = Instant::now();
    let predictor = Sam2InteractiveVideoPredictor::load_sam2_1_on_device(
        variant(&args.variant).map_err(candle_core::Error::Msg)?,
        &args.checkpoint,
        device,
        VideoPredictorConfig::default(),
    )?;
    device.synchronize()?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
    let rss_after_load_kb = current_rss_kb();
    let gpu_after_load = gpu_memory_json(device)?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut rss_after_init_kb = 0;
    let mut rss_after_drop_kb = 0;
    let mut gpu_after_init = "null".to_string();
    let mut gpu_after_drop = "null".to_string();
    for iteration in 0..(args.warmups + args.iterations) {
        if iteration == args.warmups {
            device.synchronize()?;
            reset_gpu_memory_stats(device)?;
        }
        let start = Instant::now();
        let session = if args.fixture.is_dir() {
            predictor.init_state_from_jpeg_directory(&args.fixture)?
        } else {
            predictor.init_state_from_video(&args.fixture)?
        };
        device.synchronize()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        if iteration >= args.warmups {
            samples.push(elapsed);
            rss_after_init_kb = current_rss_kb();
            gpu_after_init = gpu_memory_json(device)?;
        }
        drop(session);
        device.synchronize()?;
        if iteration >= args.warmups {
            rss_after_drop_kb = current_rss_kb();
            gpu_after_drop = gpu_memory_json(device)?;
        }
    }
    if let Some(path) = &args.output {
        fs::write(path, [])?;
    }
    let median_ms = median(samples.clone());
    let p95_ms = quantile(samples.clone(), 0.95);
    let gpu_memory = gpu_memory_json(device)?;
    println!(
        "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":{},\"warmups\":{},\"samples_ms\":{},\"median_ms\":{median_ms:.6},\"p95_ms\":{p95_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6},\"fixture_decode\":null}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"after_init\":{rss_after_init_kb},\"after_drop\":{rss_after_drop_kb},\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"gpu_memory_phases_bytes\":{{\"after_load\":{gpu_after_load},\"after_init\":{gpu_after_init},\"after_drop\":{gpu_after_drop}}},\"output\":{{\"checksum\":0.0,\"elements\":0}}}}",
        args.device,
        args.cell,
        args.operation,
        args.variant,
        args.iterations,
        args.warmups,
        json_array(&samples),
        current_rss_kb(),
        hwm_kb(),
    );
    Ok(())
}

#[cfg(not(feature = "video-decoding"))]
fn run_encoded_video(_args: &Args, _device: &Device) -> candle_core::Result<()> {
    candle_core::bail!("encoded-video benchmarking requires --features video-decoding")
}

fn run_repeated_lifetime(args: &Args, device: &Device) -> candle_core::Result<()> {
    let fixture_start = Instant::now();
    let tensors = candle_core::safetensors::load(&args.fixture, &Device::Cpu)?;
    let fixture_decode_ms = fixture_start.elapsed().as_secs_f64() * 1000.;
    let is_video = tensors.keys().any(|name| name.starts_with("frame_"));
    let mut samples = Vec::with_capacity(args.iterations);
    let mut rss_plateau_kb = Vec::with_capacity(args.sessions);
    let mut parity_values = Vec::new();
    let load_start = Instant::now();
    let (load_ms, rss_after_load_kb) = if is_video {
        let predictor = Sam2InteractiveVideoPredictor::load_sam2_1_on_device(
            variant(&args.variant).map_err(candle_core::Error::Msg)?,
            &args.checkpoint,
            device,
            VideoPredictorConfig::default(),
        )?;
        device.synchronize()?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
        let rss_after_load = current_rss_kb();
        let mut names = tensors
            .keys()
            .filter(|name| name.starts_with("frame_"))
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        let mut frames = Vec::with_capacity(names.len());
        let mut width = 0;
        let mut height = 0;
        for name in names {
            let frame = &tensors[&name];
            let (frame_height, frame_width, channels) = frame.dims3()?;
            if channels != 3 {
                candle_core::bail!("video lifetime fixture must contain RGB frames")
            }
            width = frame_width;
            height = frame_height;
            frames.push(frame.flatten_all()?.to_vec1::<u8>()?);
        }
        let inputs = frames
            .iter()
            .map(|frame| (frame.as_slice(), width, height))
            .collect::<Vec<_>>();
        let profile_lifetime = std::env::var_os("SAM2_PROFILE_VIDEO_STAGES").is_some();
        for iteration in 0..(args.warmups + args.iterations) {
            if iteration == args.warmups {
                reset_gpu_memory_stats(device)?;
            }
            let start = Instant::now();
            for session_index in 0..args.sessions {
                let init_start = Instant::now();
                let mut session = predictor.init_state(&inputs)?;
                if profile_lifetime {
                    device.synchronize()?;
                    eprintln!(
                        "SAM2_VIDEO_LIFETIME_RS stage=init ms={:.6}",
                        init_start.elapsed().as_secs_f64() * 1000.
                    );
                }
                let prompt_start = Instant::now();
                let output = predictor.add_new_points(
                    &mut session,
                    0,
                    1,
                    &[PointPrompt {
                        x: 210.,
                        y: 350.,
                        positive: true,
                    }],
                    true,
                )?;
                device.synchronize()?;
                if profile_lifetime {
                    eprintln!(
                        "SAM2_VIDEO_LIFETIME_RS stage=prompt ms={:.6}",
                        prompt_start.elapsed().as_secs_f64() * 1000.
                    );
                }
                if iteration + 1 == args.warmups + args.iterations
                    && session_index + 1 == args.sessions
                {
                    parity_values = output.masks.flatten_all()?.to_vec1::<f32>()?;
                }
                let reset_start = Instant::now();
                predictor.reset_state(&mut session);
                drop(session);
                device.synchronize()?;
                if profile_lifetime {
                    eprintln!(
                        "SAM2_VIDEO_LIFETIME_RS stage=reset_drop ms={:.6}",
                        reset_start.elapsed().as_secs_f64() * 1000.
                    );
                }
                if iteration >= args.warmups {
                    rss_plateau_kb.push(current_rss_kb());
                }
            }
            if iteration >= args.warmups {
                samples.push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        (load_ms, rss_after_load)
    } else {
        let mut predictor = Sam2ImagePredictor::load_sam2_1_on_device(
            variant(&args.variant).map_err(candle_core::Error::Msg)?,
            &args.checkpoint,
            device,
        )?;
        device.synchronize()?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
        let rss_after_load = current_rss_kb();
        let rgb = tensors
            .get("rgb_u8")
            .ok_or_else(|| candle_core::Error::Msg("image fixture has no rgb_u8".into()))?;
        let (_, height, width) = rgb.dims3()?;
        let rgb = rgb
            .permute((1, 2, 0))?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let point_coords = Tensor::new(&[[[210f32, 350.]]], device)?;
        let point_labels = Tensor::new(&[[1i64]], device)?;
        let prompts = ImagePromptBatch {
            point_coords: Some(point_coords),
            point_labels: Some(point_labels),
            normalize_coords: true,
            ..Default::default()
        };
        for iteration in 0..(args.warmups + args.iterations) {
            if iteration == args.warmups {
                reset_gpu_memory_stats(device)?;
            }
            let start = Instant::now();
            for session_index in 0..args.sessions {
                predictor.set_image(&rgb, width, height)?;
                let output = predictor.predict_cached(&prompts, false, true)?;
                device.synchronize()?;
                if iteration + 1 == args.warmups + args.iterations
                    && session_index + 1 == args.sessions
                {
                    parity_values = output.masks.flatten_all()?.to_vec1::<f32>()?;
                }
                predictor.reset_predictor();
                device.synchronize()?;
                if iteration >= args.warmups {
                    rss_plateau_kb.push(current_rss_kb());
                }
            }
            if iteration >= args.warmups {
                samples.push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        (load_ms, rss_after_load)
    };
    if let Some(path) = &args.output {
        fs::write(
            path,
            parity_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    let median_ms = median(samples.clone());
    let p95_ms = quantile(samples.clone(), 0.95);
    let plateau = rss_plateau_kb
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let checksum = parity_values.iter().map(|&value| value as f64).sum::<f64>();
    let gpu_memory = gpu_memory_json(device)?;
    println!(
        "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":{},\"warmups\":{},\"sessions\":{},\"samples_ms\":{},\"median_ms\":{median_ms:.6},\"p95_ms\":{p95_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6},\"fixture_decode\":{fixture_decode_ms:.6}}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"after_each_session\":[{plateau}],\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"output\":{{\"checksum\":{checksum:.8},\"elements\":{}}}}}",
        args.device,
        args.cell,
        args.operation,
        args.variant,
        args.iterations,
        args.warmups,
        args.sessions,
        json_array(&samples),
        current_rss_kb(),
        hwm_kb(),
        parity_values.len(),
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) if args.iterations > 0 => args,
        Ok(_) => {
            eprintln!("--iterations must be positive");
            return ExitCode::from(2);
        }
        Err(error) => {
            eprintln!(
                "{error}\nusage: bench-cpu-suite --cell I1 --operation first_point --variant tiny --checkpoint FILE --fixture FILE [--device cpu|cuda] [--iterations N] [--warmups N] [--output FILE]"
            );
            return ExitCode::from(2);
        }
    };
    let result = (|| -> candle_core::Result<()> {
        let device = match args.device.as_str() {
            "cpu" => Device::Cpu,
            #[cfg(feature = "cuda")]
            "cuda" => Device::new_cuda(0)?,
            #[cfg(not(feature = "cuda"))]
            "cuda" => candle_core::bail!("CUDA support requires --features cuda"),
            name => candle_core::bail!("unsupported device {name:?}"),
        };
        if args.mode == "graph" {
            if !matches!(
                args.operation.as_str(),
                "cached_prompt_sequence" | "cached_batch_and_prior_mask"
            ) {
                candle_core::bail!("graph mode currently supports I3 and I4 image operations")
            }
            #[cfg(feature = "cuda")]
            if let Device::Cuda(cuda) = &device {
                // SAFETY: graph mode uses Candle's single device stream and
                // synchronizes every replay, so cross-stream lifetime events
                // provide no ordering required by this benchmark.
                unsafe { cuda.disable_event_tracking() };
            } else {
                candle_core::bail!("graph mode requires CUDA")
            }
            #[cfg(not(feature = "cuda"))]
            candle_core::bail!("graph mode requires the cuda feature")
        } else if args.mode != "eager" {
            candle_core::bail!("unsupported execution mode {:?}", args.mode)
        }
        if args.operation == "encoded_video_init" {
            return run_encoded_video(&args, &device);
        }
        if args.operation == "repeated_lifetime" {
            return run_repeated_lifetime(&args, &device);
        }
        if args.operation.starts_with("video_") {
            return run_video(&args, &device);
        }
        if args.operation.starts_with("amg_") {
            return run_amg(&args, &device);
        }
        let load_start = Instant::now();
        let predictor = Sam2ImagePredictor::load_sam2_1_on_device(
            variant(&args.variant).map_err(candle_core::Error::Msg)?,
            &args.checkpoint,
            &device,
        )?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
        let rss_after_load_kb = current_rss_kb();
        if args.operation == "model_load" {
            if let Some(path) = &args.output {
                fs::write(path, [])?;
            }
            let gpu_memory = gpu_memory_json(&device)?;
            println!(
                "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":1,\"warmups\":0,\"samples_ms\":[{load_ms:.6}],\"median_ms\":{load_ms:.6},\"p95_ms\":{load_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6}}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"output\":{{\"checksum\":0.0,\"elements\":0}}}}",
                args.device,
                args.cell,
                args.operation,
                args.variant,
                current_rss_kb(),
                hwm_kb(),
            );
            drop(predictor);
            return Ok(());
        }
        if args.operation == "image_batch" {
            return run_image_batch(&args, &device, &predictor, load_ms, rss_after_load_kb);
        }
        let mut tensors = candle_core::safetensors::load(&args.fixture, &device)?;
        let image = tensors
            .remove("image")
            .ok_or_else(|| candle_core::Error::Msg("fixture has no image tensor".into()))?;
        let point = point_only(&tensors)?;
        let (rich, bounds) = rich_prompts(&tensors)?;

        let mut encode_ms = None;
        let cached = match args.operation.as_str() {
            "cached_prompt_sequence"
            | "cached_batch_and_prior_mask"
            | "first_and_cached"
            | "prompt_diversity" => {
                // This suite reports warmed inference; L1 owns construction and
                // first-ready costs. Prime lazy CUDA/cuDNN/backend setup before
                // the one image encode included in first_and_cached.
                let warm_features = predictor.encode_image(&image)?;
                device.synchronize()?;
                drop(warm_features);
                let start = Instant::now();
                let features = predictor.encode_image(&image)?;
                device.synchronize()?;
                encode_ms = Some(start.elapsed().as_secs_f64() * 1000.);
                Some(features)
            }
            "first_point" | "first_rich" => None,
            name => candle_core::bail!("operation {name:?} is not implemented by the image runner"),
        };
        let rss_after_encode_kb = current_rss_kb();

        let batch_prompts = if args.operation == "cached_batch_and_prior_mask" {
            let offsets = Tensor::new(
                &[[[0f32, 0.]], [[32., 0.]], [[0., 32.]], [[-32., 0.]]],
                &device,
            )?;
            let points = tensors["point"].broadcast_as((4, 1, 2))?.add(&offsets)?;
            let correction = tensors["correction_points"]
                .narrow(1, 0, 1)?
                .broadcast_as((4, 1, 2))?
                .add(&offsets)?;
            let labels = Tensor::ones((4, 1), candle_core::DType::I64, &device)?;
            Some((points, correction, labels))
        } else {
            None
        };

        let diversity_points = if args.operation == "prompt_diversity" {
            let base = point[0];
            Some(
                args.point_counts
                    .iter()
                    .map(|&count| {
                        (0..count)
                            .map(|index| {
                                if index == 0 {
                                    base
                                } else {
                                    let offset = index - 1;
                                    PointPrompt {
                                        x: base.x + (offset % 8) as f32 * 8. - 28.,
                                        y: base.y + (offset / 8) as f32 * 8. - 12.,
                                        positive: true,
                                    }
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let run_once = |index: usize| -> candle_core::Result<Tensor> {
            if args.operation == "prompt_diversity" {
                let features = cached.as_ref().unwrap();
                let mut outputs = Vec::new();
                for points in diversity_points.as_ref().unwrap() {
                    outputs.push(
                        predictor
                            .decode_image_features(features, points, None, None, false)?
                            .masks
                            .flatten_all()?,
                    );
                }
                outputs.push(
                    predictor
                        .decode_image_features(features, &[], Some(bounds), None, false)?
                        .masks
                        .flatten_all()?,
                );
                let initial =
                    predictor.decode_image_features(features, &point, None, None, false)?;
                outputs.push(
                    predictor
                        .decode_image_features(
                            features,
                            &rich[..1],
                            None,
                            Some(&initial.masks),
                            false,
                        )?
                        .masks
                        .flatten_all()?,
                );
                outputs.push(
                    predictor
                        .decode_image_features(features, &point, None, None, true)?
                        .masks
                        .flatten_all()?,
                );
                return Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0);
            }
            let output = match args.operation.as_str() {
                "first_point" => predictor.predict(&image, &point, None, false)?,
                "first_rich" => predictor.predict(&image, &rich, Some(bounds), false)?,
                "cached_prompt_sequence" | "first_and_cached" => {
                    // Rotate between realistic edits without rebuilding image
                    // features. This catches prompt-count/layout specialization.
                    if index % 3 == 0 {
                        predictor.decode_image_features(
                            cached.as_ref().unwrap(),
                            &point,
                            None,
                            None,
                            false,
                        )?
                    } else if index % 3 == 1 {
                        predictor.decode_image_features(
                            cached.as_ref().unwrap(),
                            &rich,
                            Some(bounds),
                            None,
                            false,
                        )?
                    } else {
                        predictor.decode_image_features(
                            cached.as_ref().unwrap(),
                            &rich[..1],
                            Some(bounds),
                            None,
                            true,
                        )?
                    }
                }
                "cached_batch_and_prior_mask" => {
                    let (points, correction, labels) = batch_prompts.as_ref().unwrap();
                    let initial = predictor.decode_image_features_batch(
                        cached.as_ref().unwrap(),
                        &ImagePromptBatch {
                            point_coords: Some(points.clone()),
                            point_labels: Some(labels.clone()),
                            ..Default::default()
                        },
                        false,
                    )?;
                    predictor.decode_image_features_batch(
                        cached.as_ref().unwrap(),
                        &ImagePromptBatch {
                            point_coords: Some(correction.clone()),
                            point_labels: Some(labels.clone()),
                            mask_input: Some(initial.masks),
                            ..Default::default()
                        },
                        false,
                    )?
                }
                _ => unreachable!(),
            };
            Ok(output.masks)
        };

        for index in 0..args.warmups {
            let _ = run_once(index)?;
            device.synchronize()?;
        }
        #[cfg(feature = "cuda")]
        let mut capture_ms: Option<f64> = None;
        #[cfg(not(feature = "cuda"))]
        let capture_ms: Option<f64> = None;
        #[cfg(feature = "cuda")]
        let graph_state = if args.mode == "graph" {
            use candle_core::cuda_backend::cudarc::driver::sys::{
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED,
            };
            let Device::Cuda(cuda) = &device else {
                unreachable!()
            };
            let upload_cache = cuda.enable_cuda_graph_htod_cache();
            let capture_count = if args.operation == "cached_prompt_sequence" {
                3
            } else {
                1
            };
            // Populate Candle's graph-safe HtoD cache for every static route.
            // This warmup is startup work and is excluded from capture/replay.
            for index in 0..capture_count {
                let _ = run_once(index)?;
                device.synchronize()?;
            }
            reset_gpu_memory_stats(&device)?;
            let capture_start = Instant::now();
            let mut captures = Vec::with_capacity(capture_count);
            for index in 0..capture_count {
                device.synchronize()?;
                let stream = cuda.cuda_stream();
                stream
                    .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
                    .map_err(|error| {
                        candle_core::Error::Msg(format!("CUDA graph capture failed: {error}"))
                    })?;
                let output = run_once(index)?;
                let graph = stream
                    .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                    .map_err(|error| {
                        candle_core::Error::Msg(format!("CUDA graph instantiation failed: {error}"))
                    })?
                    .ok_or_else(|| {
                        candle_core::Error::Msg("CUDA graph capture produced no graph".into())
                    })?;
                captures.push((graph, output));
            }
            device.synchronize()?;
            capture_ms = Some(capture_start.elapsed().as_secs_f64() * 1000.);
            Some((captures, upload_cache))
        } else {
            None
        };
        if args.mode == "eager" {
            reset_gpu_memory_stats(&device)?;
        }
        let mut samples = Vec::with_capacity(args.iterations);
        let mut last = None;
        for index in 0..args.iterations {
            let start = Instant::now();
            #[cfg(feature = "cuda")]
            if let Some((captures, _upload_cache)) = &graph_state {
                let (graph, output) = &captures[index % captures.len()];
                graph.launch().map_err(|error| {
                    candle_core::Error::Msg(format!("CUDA graph launch failed: {error}"))
                })?;
                last = Some(output.clone());
            } else {
                last = Some(run_once(index)?);
            }
            #[cfg(not(feature = "cuda"))]
            {
                last = Some(run_once(index)?);
            }
            device.synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1000.);
        }
        let output = last.unwrap();
        if let Some(path) = &args.output {
            let values = output.flatten_all()?.to_vec1::<f32>()?;
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            fs::write(path, bytes)?;
        }
        let checksum = output.sum_all()?.to_scalar::<f32>()?;
        let median_ms = median(samples.clone());
        let p95_ms = quantile(samples.clone(), 0.95);
        let encode = encode_ms.map_or("null".to_owned(), |value| format!("{value:.6}"));
        let capture = capture_ms.map_or("null".to_owned(), |value| format!("{value:.6}"));
        let gpu_memory = gpu_memory_json(&device)?;
        println!(
            "{{\"schema_version\":1,\"implementation\":\"rust-candle\",\"device\":\"{}\",\"mode\":\"{}\",\"cell\":\"{}\",\"operation\":\"{}\",\"variant\":\"{}\",\"iterations\":{},\"warmups\":{},\"samples_ms\":{},\"median_ms\":{median_ms:.6},\"p95_ms\":{p95_ms:.6},\"phases_ms\":{{\"load\":{load_ms:.6},\"encode\":{encode},\"capture\":{capture}}},\"rss_kb\":{{\"after_load\":{rss_after_load_kb},\"after_encode\":{rss_after_encode_kb},\"final\":{},\"hwm\":{}}},\"gpu_memory_bytes\":{gpu_memory},\"output\":{{\"checksum\":{checksum:.8},\"elements\":{}}}}}",
            args.device,
            args.mode,
            args.cell,
            args.operation,
            args.variant,
            args.iterations,
            args.warmups,
            json_array(&samples),
            current_rss_kb(),
            hwm_kb(),
            output.elem_count(),
        );
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
