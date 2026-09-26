# sam2-rs

Native Candle-backed SAM2.1 inference with static, video/z-slice,
automatic-mask, and direct PyTorch/safetensors checkpoint-loading APIs.

Translation coverage and the bottom-up remaining-feature checklist are tracked
in [TRANSLATION.md](TRANSLATION.md). The original architecture handover remains
in [handover.md](handover.md).

## Inference APIs

Published `.pt` checkpoints load directly; the same APIs also accept converted
`.safetensors` files. Cached image prediction reuses the image encoder:

```rust,no_run
use sam2_rs::{ImagePromptBatch, Sam2ImagePredictor, Sam2Variant};

let mut predictor = Sam2ImagePredictor::load_sam2_1(
    Sam2Variant::HieraTiny,
    "checkpoints/sam2.1_hiera_tiny.pt",
)?;
predictor.set_image(&rgb_bytes, width, height)?;
let prediction = predictor.predict_cached(
    &ImagePromptBatch::default(),
    false,
    false,
)?;
```

Automatic masks and editable multi-object video use typed configurations:

```rust,no_run
use sam2_rs::{
    AutomaticMaskConfig, PointPrompt, Sam2AutomaticMaskGenerator,
    Sam2InteractiveVideoPredictor, Sam2Variant, VideoPredictorConfig,
};

let generator = Sam2AutomaticMaskGenerator::load_sam2_1(
    Sam2Variant::HieraTiny,
    "checkpoints/sam2.1_hiera_tiny.pt",
    AutomaticMaskConfig::default(),
)?;
let masks = generator.generate_rgb8(&rgb_bytes, width, height)?;

let video = Sam2InteractiveVideoPredictor::load_sam2_1(
    Sam2Variant::HieraTiny,
    "checkpoints/sam2.1_hiera_tiny.pt",
    VideoPredictorConfig::default(),
)?;
let mut session = video.init_state(&frames)?;
video.add_new_points(
    &mut session,
    0,
    101,
    &[PointPrompt { x: 120.0, y: 80.0, positive: true }],
    true,
)?;
let tracked = video.propagate_in_video(&mut session, None, None, false)?;
```

Here `rgb_bytes` is interleaved RGB8 and `frames` is a slice of
`(&[u8], width, height)` tuples. See [PARITY.md](PARITY.md) for reproducible
Python/Rust conformance commands.

## Optional downloading and media decoding

The default build remains network- and codec-free. Enable only the integrations
an application needs:

```toml
sam2-rs = { version = "0.1", features = ["huggingface", "video-decoding"] }
```

`huggingface` adds cached checkpoint retrieval and `from_pretrained`
constructors. It honors the standard `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, and
`HF_ENDPOINT` environment variables:

```rust,no_run
use sam2_rs::{Sam2ImagePredictor, Sam2Variant};

let mut predictor = Sam2ImagePredictor::from_pretrained(Sam2Variant::HieraTiny)?;
predictor.set_image_path("input.jpg")?;
```

Use `HuggingFaceCheckpoint::for_variant(variant)` when a pinned revision,
custom cache directory, offline-only lookup, or forced refresh is needed.

`image-decoding` accepts JPEG/PNG images and upstream-style directories whose
JPEG filenames have numeric stems. `video-decoding` includes image decoding and
adds encoded-video support through the system FFmpeg libraries:

```rust,no_run
use sam2_rs::{Sam2InteractiveVideoPredictor, Sam2Variant, VideoPredictorConfig};

let predictor = Sam2InteractiveVideoPredictor::from_pretrained(
    Sam2Variant::HieraTiny,
    VideoPredictorConfig::default(),
)?;
let jpeg_session = predictor.init_state_from_jpeg_directory("frames")?;
let mp4_session = predictor.init_state_from_video("clip.mp4")?;
# let _ = (jpeg_session, mp4_session);
```

Building `video-decoding` requires FFmpeg development packages discoverable by
`pkg-config` (on Debian/Ubuntu: `libavcodec-dev`, `libavformat-dev`,
`libavutil-dev`, and `libswscale-dev`). `decode_video_prefix` is available for
bounded previews without retaining an entire video in memory.

## Reproducible performance baseline

Run `python3 tools/run_benchmarks.py` after following [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md). The generated table compares separate fresh CPU processes over an identical frozen real-image fixture; ratios below `1.0×` favor Rust. Use `--threads N` to cap Rayon and PyTorch to the same CPU-worker count; this is required when comparing thread-count tuning.

For CUDA, run `python3 tools/run_gpu_benchmarks.py` on a CUDA-capable host. It writes a separate table and synchronizes device work around each timing sample; its RSS column is host `VmHWM`, not VRAM.

The evidence and parity-gated optimization sequence are in [PERFORMANCE_INVESTIGATION.md](PERFORMANCE_INVESTIGATION.md).

<!-- BENCHMARK_RESULTS_START -->
_Generated on Linux 6.8.0-58-generic, CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz; fixture SHA-256: `24271176ecdc`. Peak RSS is Linux `VmHWM`._

| Fixture / operation | Rust median ms | Python median ms | Rust/Python latency | Rust peak RSS MiB | Python peak RSS MiB | Rust/Python RSS | Max abs mask-logit error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bedroom/00089.jpg, Tiny static point prompt (3 measured iterations) | 12606.34 | 726.56 | 17.351× | 1713.5 | 1434.2 | 1.195× | 4.50611e-05 |
<!-- BENCHMARK_RESULTS_END -->

### CPU stage-attribution diagnostic

This one-iteration diagnostic is separate from the median baseline above. It
uses the same warmed-up fixture and includes prompt encoding with the decoder.

| Stage | Rust ms | Python ms | Rust/Python |
| --- | ---: | ---: | ---: |
| Image encoder (Hiera + FPN) | 10813.10 | 738.39 | 14.644× |
| Prompt encoder + mask decoder | 484.93 | 20.92 | 23.181× |

Reproduce with `target/release/bench-static checkpoints/sam2.1_hiera_tiny.safetensors checkpoints/bench_bedroom.safetensors 1 --profile-stages` and the equivalent `tools/bench_static_python.py` command.

### Local Candle-fork CPU diagnostic

The direct local Candle fork includes its CPU Conv2D fast paths. On the same
warmed-up one-iteration diagnostic, it preserved the full mask-logit maximum
error (`4.50611e-05`) and improved the default-Candle result as follows.

| Metric | Default Candle | Local Candle fork | Fork/default |
| --- | ---: | ---: | ---: |
| Total static inference | 10939.36 ms | 9758.35 ms | 0.892× |
| Image encoder | 10398.43 ms | 9349.32 ms | 0.899× |
| Prompt encoder + mask decoder | 529.60 ms | 396.46 ms | 0.749× |
| FPN | 776.81 ms | 576.28 ms | 0.742× |

### SIMD CPU flash-attention diagnostic

The local Candle commit `480dbc1b` adds a float32 AVX2/FMA online-softmax path
for long attention. It is opt-in with `--features cpu-flash-attention`; the
default remains available for comparison.

| Metric | Fork default | SIMD flash feature | Feature/default |
| --- | ---: | ---: | ---: |
| Total static inference | 9758.35 ms | 7007.11 ms | 0.718× |
| Image encoder | 9349.32 ms | 6569.83 ms | 0.703× |
| Peak RSS | 1688.9 MiB | 661.4 MiB | 0.392× |
| Max abs mask-logit error vs PyTorch | 4.50611e-05 | 5.00679e-05 | — |

The feature is still `10.75×` the one-iteration PyTorch CPU diagnostic, but
its three global-attention blocks fall from about 1.5 s each to 0.53–0.55 s.
Reproduce the full ratio table with `python3 tools/run_benchmarks.py --rust-features cpu-flash-attention`.

## Reproducible experimental performance baseline

<!-- BENCHMARK_EXPERIMENTAL_RESULTS_START -->
_Generated on Linux 6.8.0-58-generic, CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz; fixture SHA-256: `24271176ecdc`. CPU workers: `20`, pinned to CPUs `0-19`. Peak RSS is Linux `VmHWM`._

| Fixture / operation | Rust mean ms | Python mean ms | Rust/Python latency | Rust mean peak RSS MiB | Python mean peak RSS MiB | Rust/Python RSS | Max abs mask-logit error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bedroom/00089.jpg, Tiny static point prompt on cpu; Rust `cpu-flash-attention,cpu-glibc-vector-math` (8 independent runs × 5 measured iterations) | 580.06 | 652.42 | 0.889× | 629.6 | 1411.8 | 0.446× | 2.86102e-05 |
<!-- BENCHMARK_EXPERIMENTAL_RESULTS_END -->

This current-code row combines two four-run batches on the shared host. Their
latency ratios were `0.891×` and `0.887×`; all eight individual Rust/PyTorch
pairs favored Rust. The arithmetic mean is reproducible with the runner, but
contention makes small cross-date changes inconclusive. CPU speed parity is
reached for this fixture, machine, and 20-worker configuration, not established
for every variant, prompt, or CPU. Use paired A/B measurements in
[PERFORMANCE_INVESTIGATION.md](PERFORMANCE_INVESTIGATION.md) before attributing
an optimization.
Exact within-block decoder key/position reuse and image-time projection of
high-resolution skips are included in this row. Their repeated-decode gains
are measured, but paired full-model A/B runs have not established a clean
gain under the shared host's varying load.
The direct Hiera query-pool path is included in this row. Its paired
full-model A/B averaged `0.967×` of the prior Rust path with byte-identical
masks. An earlier attempted cross-runtime batch had a `6.58` s PyTorch
contention outlier and was excluded; see
[PERFORMANCE_INVESTIGATION.md](PERFORMANCE_INVESTIGATION.md).
On Linux/glibc, loading a CPU model also sets glibc's mmap and trim thresholds
to 256 MiB once per process. This keeps large temporary attention buffers on
the heap and trades some peak RSS for speed. It has no effect on CUDA model
loading. Set `SAM2_CPU_KEEP_LARGE_ALLOCATIONS=0` to opt out; existing
`MALLOC_MMAP_THRESHOLD_` or `MALLOC_TRIM_THRESHOLD_` settings also take
precedence. This tuning is process-global, so embedding applications should
consider its effect on other allocations.
Run `python3 tools/run_benchmarks.py --iterations 5 --repeats 4 --threads 20 --cpu-list 0-19 --rust-features cpu-flash-attention,cpu-glibc-vector-math --no-write`
twice to collect the same number of independent runs.

An earlier run before the tiny-query decoder change, with eight pinned CPU
workers, measured 1190.83 ms Rust versus
828.61 ms PyTorch (`1.437×`) across four independent runs of five iterations.
Mean peak RSS was 588.2 MiB versus 1582.1 MiB (`0.372×`), with maximum
mask-logit error `2.90871e-05`. Reproduce with
`python3 tools/run_benchmarks.py --iterations 5 --repeats 4 --threads 8 --cpu-list 0-7 --rust-features cpu-flash-attention,cpu-glibc-vector-math --no-write`.

## Reproducible GPU performance baseline

<!-- BENCHMARK_GPU_RESULTS_START -->
_Generated on Linux 6.8.0-58-generic, CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz. GPU: Quadro RTX 5000, 570.133.20. Fixture SHA-256: `24271176ecdc`. CPU workers: `implementation default`, alternating process order. Peak RSS is Linux `VmHWM`._

| Fixture / operation | Rust mean ms | Python mean ms | Rust/Python latency | Rust mean peak RSS MiB | Python mean peak RSS MiB | Rust/Python RSS | Max abs mask-logit error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bedroom/00089.jpg, Tiny static point prompt on cuda; Rust `cuda,cuda-graph-both` (4 independent runs × 10 measured iterations) | 67.97 | 67.16 | 1.012× | 470.8 | 1192.6 | 0.395× | 1.43051e-05 |
<!-- BENCHMARK_GPU_RESULTS_END -->

GPU timings synchronize device work around every sample. Peak RSS is host
`VmHWM`, not device VRAM. The row uses CUDA graph capture and replay for both
implementations because its image, checkpoint, shape, and prompt are static.
Reproduce it with
`python3 tools/run_gpu_benchmarks.py --cuda-graph --alternate-order --repeats 4 --iterations 10`.
The retained eager path measured 75.90 ms Rust versus 67.20 ms PyTorch
(`1.129×`) in an independent four-run batch. Graph mode removes Candle's
remaining repeated allocation/event/launch bookkeeping; it does not change
the model output.
