# CPU optimization execution plan

## Objective

Close the measured CPU latency gap between `sam2-rs` and the pinned PyTorch
SAM2.1 Hiera-Tiny reference without weakening output parity, changing the
public variable-shape API, or hiding work in a benchmark-only path.

The authoritative end-to-end benchmark is `BENCHMARK_PLAN.md`: the frozen
bedroom fixture, float32 checkpoint, static point prompt, separate warmed-up
processes, identical worker cap, median latency, `VmHWM`, and maximum mask
logit error. A latency ratio below `1.0x` favors Rust.

## Guardrails

- Keep the existing generic Candle and SAM2 paths as fallbacks. Optimized CPU
  paths must be explicitly shape/dtype/layout gated.
- Test every new kernel against its generic equivalent and compare final mask
  logits against the Python fixture. Preserve the established order of
  floating-point operations where practical; document any changed tolerance.
- Avoid an unbounded attention matrix and avoid unconditional materialization
  of Q/K/V. Parallel work owns scratch buffers; do not combine Rayon outer
  parallelism with multithreaded BLAS.
- Treat latency, RSS, and parity as one acceptance decision. Do not retain a
  speedup that materially regresses either of the other two without a recorded
  reason and an explicit opt-in.
- Benchmark on a quiet host. If it is contended, retain only diagnostic results
  and do not regenerate the README comparison table.

## Baselines and attribution

Before and after each accepted change:

1. Run the focused unit/regression tests for the changed crate.
2. Capture one `--profile-stages` diagnostic to identify encoder, FPN, and
   global-attention effects.
3. Run the fair end-to-end runner with the selected identical CPU worker cap:

   ```bash
   python3 tools/run_benchmarks.py --threads 20 --rust-features cpu-flash-attention
   ```

4. Record median latency ratio, RSS ratio, maximum mask-logit error, fixture
   digest, worker count, and host state in `PERFORMANCE_INVESTIGATION.md`.
5. Update the generated README table only for a stable multi-sample result.

The current retained reference is the AVX2 row-wise online-softmax attention
path with direct NHSD output and bounded K/V packing. The three Hiera global
blocks (`B=1, H=4, S=4096, D=96`) are still the primary CPU hotspot.

## Phase 1 — blocked CPU f32 attention

### Design

Add an experimental Candle prefill path for dense f32, dropout-free long
attention. It should follow PyTorch CPU FlashAttention's algorithm:

- Partition `S >= 768` attention into query tiles of 256 rows and key/value
  tiles of 512 rows. Use smaller query tiles only after profiling short sizes.
- Parallelize `(batch, head, query-tile)` work items with Rayon. Allocate and
  reuse bounded worker-local scratch: score tile, row maxima, row sums, and
  output accumulator. Never allocate `S × S` scores.
- For each KV tile, calculate `Q × Kᵀ`, apply scale and tiled online softmax,
  rescale the accumulated output once per tile, then calculate `P × V`.
- Return contiguous NHSD directly, retaining the current no-output-copy Hiera
  integration.

### Layout and BLAS experiments

Implement and compare these variants independently:

1. A zero-copy variant that accepts the interleaved SAM Q/K/V views when the
   last dimension is contiguous, passing leading strides to the tile GEMM.
2. The current bounded K/V packing variant, enabled only when it demonstrably
   wins for the relevant stride/shape.
3. A pure in-tree tiled implementation for correctness and a serial SGEMM tile
   backend only if it can be linked without nested thread pools. Do not revive
   the rejected globally-threaded OpenBLAS path.

Acceptance requires exact shape/layout gating, focused numerical tests,
end-to-end final-logit parity no worse than the current envelope, and a stable
end-to-end win over the retained row-wise kernel. Keep the winner behind the
existing `cpu-flash-attention` feature until it has repeated results.

## Phase 2 — remove SAM2 layout and memory passes

### Fixed NHWC query pool

Replace Hiera's `NHWC -> NCHW -> max_pool2d -> NHWC` composition with a
CPU-f32 2×2, stride-2, no-padding max-pool that writes contiguous NHWC output
directly. Gate it to the exact Hiera operation and retain the generic fallback.
Test odd/even valid dimensions and compare every output element with Candle's
generic pool. Measure both pool time and downstream QKV/attention effects,
since output contiguity is part of the intended gain.

### Fused bias for CPU dense paths

In the Candle fork, add a contiguous f32 CPU `Linear` fast path that seeds the
final output with the broadcast bias and invokes GEMM with accumulation
(`beta=1` / read-destination). This must eliminate the separate output-sized
`broadcast_add` allocation and pass. First validate linear; then apply the
same approach to the 1×1 convolution fast path used by FPN and decoder
projections. Keep generic paths unchanged for all other devices/layouts.

Profile the 7×7 patch convolution only after these changes. Consider a
oneDNN/Candle integration only if it remains a leading hotspot and isolated
benchmarks show that conversion/prepacking costs are amortized.

## Phase 3 — secondary, parity-gated refinements

1. Add a vector exponentiation path inside the blocked attention tile only
   after Phase 1. Compare its tile and final-logit error against scalar `exp`;
   retain it only with a documented acceptable error bound.
2. Evaluate inference-only caches for fixed static-image position embeddings
   and retained FPN sine encodings. Make caching opt-in or shape-keyed, bound
   its memory, and report the RSS tradeoff. Do not cache the FPN level that
   `scalp` discards.
3. Revisit AVX-512 only as per-shape runtime dispatch. The prior whole-kernel
   AVX-512 trial regressed on this Skylake-SP host, so AVX2 remains default
   until a reproducible end-to-end result proves otherwise.

## Completion criteria

This plan is complete when each phase has either produced and retained a
parity-tested benchmark win or has a documented rejected result; the best
combination has stable end-to-end CPU measurements published in `README.md`;
and the remaining Rust/PyTorch ratio and top stage hotspots are recorded with
enough evidence to decide whether further work belongs in `sam2-rs`, Candle,
or the benchmark environment.

## Phase 4 — persistent static RHS packing (Candle/GEMM extension)

The remaining CPU gap is GEMM throughput. Candle's fused f32 Linear route
already avoids the output-sized bias-add allocation, but the `gemm` backend
packs its immutable RHS weight for every invocation. Existing threshold A/Bs
proved that disabling or globally retuning packing is slower; the improvement
must reuse the backend's own packed representation.

Implement this only as a clean upstream-shaped API in `gemm`/`gemm-common`:

1. Expose an opaque, architecture-specific packed-RHS value constructed from
   a contiguous `(K, N)` source and its exact GEMM dispatch configuration.
   It must retain `KC`, `NC`, `NR`, dtype, conjugation/layout metadata and the
   packed bytes; callers must never infer this layout.
2. Add a GEMM entry point consuming that opaque RHS while preserving ordinary
   LHS packing, thread partitioning, alpha/beta and tail microkernels.
3. In Candle, make it opt-in for immutable contiguous CPU f32 Linear weights.
   Cache by storage identity plus tensor layout/dimensions, use bounded
   ownership, and keep the existing call for dynamic/view weights and all
   other devices/dtypes.
4. Test the primitive against ordinary GEMM for row/column tails, beta zero
   and one, serial and Rayon modes. Then test Candle Linear exactly against
   the composed bias path and run SAM2 real-fixture parity.
5. Benchmark the exact decoder 4096×256→128 projection, Hiera QKV/MLP
   shapes, RSS, and the pinned full model. Retain only a repeatable full-model
   win that remains inside the established mask-logit error envelope.

Do not vendor or reinterpret `gemm-common`'s private packed layout: its
packing is tiled by private architecture dispatch parameters and any guessed
format would be unsound across kernels or versions.

## Sources of the design

- PyTorch CPU blocked FlashAttention:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp>
- PyTorch SDPA dispatch:
  <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp>
- Upstream SAM2 Hiera layout and SDPA call:
  <https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/backbones/hieradet.py>
