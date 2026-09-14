# sam2-rs

Native Candle-backed SAM2.1 inference with static, video/z-slice, automatic-mask, and checkpoint-conversion APIs.

## Reproducible performance baseline

Run `python3 tools/run_benchmarks.py` after following [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md). The generated table compares separate fresh CPU processes over an identical frozen real-image fixture; ratios below `1.0×` favor Rust.

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
