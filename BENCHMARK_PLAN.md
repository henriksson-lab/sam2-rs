# Apples-to-apples benchmark and parity plan

## Scope

The baseline is the published SAM2.1 Hiera-Tiny checkpoint on the vendored
SAM2 bedroom video frame `notebooks/videos/bedroom/00089.jpg`, with one
positive point at the frame centre. The fixture generator decodes that real
image once and saves the exact normalized `1x3x1024x1024` tensor and prompt.
Both implementations consume that fixture, so image decoding, JPEG libraries,
resizing policy, and Python/Rust preprocessing differences are out of scope for
the inference comparison.

## Protocol

1. Convert the published `.pt` once with `tools/convert_checkpoint.py`.
2. Create `checkpoints/bench_bedroom.safetensors` with
   `tools/prepare_benchmark_fixture.py`.
3. Build Rust in release mode; benchmark Python and Rust in separate fresh
   processes. Model loading and one warm-up inference are excluded from latency;
   each process reports peak resident set including checkpoint/model load.
4. Run the same number of measured static prompt inferences (default 3 on CPU).
   Report median latency, process high-water RSS, and a deterministic output
   checksum. Save one output from each process and compute maximum absolute
   mask-logit error.
5. `tools/run_benchmarks.py` calculates Rust/Python ratios and replaces the
   generated table bounded by markers in `README.md`, including CPU/OS and a
   short SHA-256 identity of the frozen fixture.

For a noisy shared host, add `--alternate-order` to run Python first in odd
pairs. The runner records each pair's order in its JSON output and identifies
the ordering in any generated table, so process-order drift is visible rather
than silently biasing one implementation.

## CUDA companion suite

Run `python3 tools/run_gpu_benchmarks.py` on a host with a functional NVIDIA
driver, CUDA toolkit compatible with Candle, and CUDA-enabled PyTorch. It uses
the same converted/published checkpoints, frozen fixture, prompt, warm-up,
output comparison, iteration count, and fresh-process `VmHWM` protocol as the
CPU suite. GPU work is synchronized around each timed inference. Results are
written to a separate GPU table in `README.md`, so they cannot overwrite the
CPU baseline. `VmHWM` remains host RSS; it is deliberately not labelled as
device VRAM.

## Fairness constraints

- Same checkpoint architecture, CPU device, float32, fixture, prompt, warm-up,
  and measured operation (`image encoder + prompt encoder + mask decoder`).
- No model-load time in latency; no image I/O/preprocessing in latency.
- RSS is each process's Linux `VmHWM`, so it includes model/checkpoint mapping
  and allocator high water. It must not be interpreted as per-inference memory.
- Latency ratios below 1.0 favor Rust; RSS ratios below 1.0 use less Rust memory.

## Reuse

The runners accept alternate checkpoint and fixture paths. Future optimization
work should rerun the same command and commit only regenerated README rows,
keeping host, fixture checksum, and iteration count visible.

For optimization attribution, pass `--profile-stages` to the Rust or Python
static runner. It splits one measured iteration into image encoding and
prompt-plus-mask decoding; it is a diagnostic rather than a replacement for
the multi-sample total-latency baseline.

The experimental long-CPU-attention implementation is benchmarked with
`python3 tools/run_benchmarks.py --rust-features cpu-flash-attention`. Keep its
ratios separate from the default until its parity and multi-sample result are
reproduced on the target host.
