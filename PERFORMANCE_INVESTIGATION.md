# Closing the Rust/PyTorch performance gap

## Evidence so far

The reproducible CPU baseline in `README.md` measures the same frozen real
bedroom frame, prompt, model architecture, and float32 static inference path.
It reports Rust at `18.270x` Python latency and `1.151x` host peak RSS. A
one-iteration stage probe after warm-up measured the following:

| Stage | Rust | Python | Rust/Python |
| --- | ---: | ---: | ---: |
| Hiera image encoder + FPN | 10.813 s | 0.738 s | 14.644x |
| Prompt encoder + mask decoder | 0.485 s | 0.021 s | 23.181x |

The process consumed about four CPU cores; this is a genuine compute/runtime
gap, not a one-core scheduling mistake. The encoder accounts for 95.6% of
Rust's profiled inference, so it is the first application-level target.

The encoder trace makes the hotspot unambiguous:

| Component | Rust time |
| --- | ---: |
| Hiera patch embedding + positional setup | 427 ms |
| Hiera blocks 0–4, 6, 8, 10–11 combined | 4.31 s |
| Hiera global-attention block 5 | 1.64 s |
| Hiera global-attention block 7 | 1.62 s |
| Hiera global-attention block 9 | 1.63 s |
| FPN | 777 ms |

Tiny config declares blocks 5, 7, and 9 as global attention. Each computes a
4096-token attention matrix; upstream routes that operation through PyTorch
scaled-dot-product attention. The direct Candle translation materializes the
score matrix and uses generic CPU matrix multiplication. Thus the remaining
large CPU gap cannot be closed materially by image preprocessing or prompt
API changes.

No GPU number is recorded yet: this host has neither `/dev/nvidia*` nor a
working NVIDIA driver. The CUDA runner fails explicitly instead of falling
back to CPU.

## What must be measured next

Run the CUDA companion suite on a CUDA host:

```bash
python3 tools/run_gpu_benchmarks.py --iterations 10
```

It builds `bench-static` with the `cuda` feature, which deliberately includes
cuDNN. Compare its independent README row with PyTorch CUDA only after both
processes complete. The runner synchronizes before/after every timed call,
so asynchronous launch time cannot masquerade as inference time. Its RSS is
host `VmHWM`; measure VRAM separately with a device profiler before making
memory claims.

## Ranked optimization experiments

1. The application-level attribution is complete: retain the explicit Hiera
   layout and numerical order, but avoid speculative SAM2 rewrites. The three
   global-attention blocks are the dominant target.
2. Before modifying Candle, test a separately-labelled optimized CPU backend
   (the optional `mkl` feature) against the same fixture and parity result.
   This changes dispatch, not SAM2 math.
3. Within Hiera, inspect the repeated NHWC/NCHW permutations, `.contiguous()`
   calls around Q/K/V, and `window_partition`/`window_unpartition`. These can
   materialize large temporary tensors at every block. Remove a materialized
   layout conversion only when an intermediate parity fixture proves it is
   unchanged.
4. Only after stages 1–3, consider graph-level buffer reuse or fusion of
   elementwise operations (normalization, residual add, GELU). Each change
   needs the same frozen-fixture parity check and regenerated README ratios.
5. Establish the CUDA/cuDNN baseline later, once a compatible GPU host is
   available. That result decides whether any remaining bottleneck belongs in
   Candle rather than in SAM2's layout and allocation behavior.

The hand-written bicubic positional-embedding resize is intentionally *not*
the first target: it runs once per image and is tiny relative to 1024-pixel
Hiera blocks. This is a hypothesis document, not a claim that any proposed
change already improves performance.

## Rejected experiment

Candle 0.10.2's `cpu_flash_attention::run_flash_attn_cpu` was tried only for
the three 4096-token global blocks. Its warm-up plus one measured inference
did not complete within the baseline's 30-second execution window, whereas
the direct attention path did. It is therefore not enabled. A future Candle
investigation may improve that kernel, but SAM2 must not carry it as a CPU
optimization until it wins this exact benchmark and retains mask parity.

## Local Candle fork

`sam2-rs` directly depends on `/home/mahogny/github/claude/candle` while CPU
work is active. Its `sam2-cpu-perf` branch starts from
`integration/stardist-clearmap-ops`, which already contains the independently
developed CPU Conv2D improvements: batch-1 1x1 fast path, direct-stride tiled
im2col, and 3x3 same-padding specialization. The separate CPU PR branches
were inspected but not re-applied because their changes are already present
under equivalent commits (`7c861ac6` and `d9d14408`).

Against the prior default-Candle one-iteration diagnostic, the linked fork
reduced total static latency from 10.939 s to 9.758 s (`0.892x`) with unchanged
maximum mask-logit error (`4.50611e-05`). This validates the direct dependency
and makes CPU matrix multiplication for global attention the next target.

The optional fork `mkl` feature now builds after fixing its unavailable static
`hgemm_` reference, but it is not enabled by default: its one-iteration
diagnostic was 10.642 s (`1.091x` slower than the fork default's 9.758 s) and
raised peak RSS from 1,689 MiB to 1,888 MiB. Its global blocks were somewhat
faster, but the surrounding FPN/setup costs regressed. Retain it only as a
separate backend experiment, not as the SAM2 CPU configuration.

The system OpenBLAS CBLAS SGEMM path was also tested with its thread pool
capped to Candle's configured thread count. It still regressed to 10.816 s
(`1.108x` the fork default) and 1,776 MiB peak RSS, so it is not retained.

## Accepted SIMD flash-attention path

Candle commit `480dbc1b` specializes unmasked float32 CPU prefill attention
with AVX2/FMA dot-product and value-accumulation loops. `sam2-rs` enables it
only with the `cpu-flash-attention` feature and only for long Hiera attention
sequences (at least 512 tokens). The one-iteration diagnostic reduced total
latency from 9.758 s to 7.007 s (`0.718x`), encoder latency from 9.349 s to
6.570 s (`0.703x`), and peak RSS from 1,689 MiB to 661 MiB (`0.392x`). Full
mask-logit parity remained tight at `5.00679e-05` maximum absolute error.

The subsequent NHSD-output specialization writes this same attention result
directly into the contiguous `(B, S, H, D)` order consumed by the projection.
The prior `(B, H, S, D)` result required a permutation followed by Candle's
materializing `reshape`; the new path neither copies Q/K/V nor creates an
intermediate output buffer. Its Candle regression test checks numerical
equivalence and contiguity.

## Fair CPU worker cap

The benchmark runner accepts `--threads N`. It applies `RAYON_NUM_THREADS=N`
to Rust and caps PyTorch plus its common CPU math backends to the same `N`.
On this 20-physical-core host, a one-iteration diagnostic favored 20 workers
(6.999 s, 641 MiB) over 16 (7.591 s, 635 MiB) and 32 (7.377 s, 662 MiB).
The fair two-sample 20-worker run measured 6.896 s Rust, 0.905 s PyTorch,
0.470x Rust/PyTorch RSS, and `4.98295e-05` maximum mask-logit error. This cap
is intentionally a benchmark/deployment setting rather than a hard-coded
library policy.

## Rejected AVX-512 flash trial

An exact, runtime-gated AVX-512F/FMA version of the flash dot-product, value
accumulation, and rescale primitives was tested on the Xeon Gold 6138. It took
7.199 s at the 20-worker cap versus the retained AVX2 diagnostic's 6.999 s.
The likely frequency reduction on this Skylake-SP processor outweighed the
wider vectors, so the AVX-512 code was removed. The portable AVX2/FMA path
remains the selected implementation.

## Accepted fused CPU bicubic position resize

Hiera's position embedding uses bicubic interpolation at image-encoder setup.
The generic Candle composition created four vertical `index_select` terms and
four horizontal terms, repeatedly allocating full-size tensors. The retained
CPU-f32 specialization reads the contiguous embedding directly and writes only
the final output allocation, parallelized by output row. It preserves the
generic implementation's vertical-then-horizontal four-tap accumulation order;
the dedicated regression test compares both paths within `1e-6`.

At the fair 20-worker setting this reduced the two-sample Rust median from
6.896 s to 6.313 s (`0.915x`) while holding peak RSS at approximately 641 MiB
and retaining `4.98295e-05` maximum mask-logit error against PyTorch. A stage
diagnostic attributes the expected portion to Hiera setup (about 281 ms to
75 ms); FPN remains a separate roughly 580 ms target.

## Accepted long-attention K/V locality packing

Hiera's Q/K/V projection stores the three tensors interleaved. The CPU flash
kernel now packs only strided K and V views once per long prefill call into
head-major contiguous rows; that bounded setup copy replaces repeated strided
K/V reads for every query. The three global-attention blocks improved to about
0.51--0.58 s in the stage diagnostic, and direct raw-logit comparison retained
the established `5.00679e-05` maximum error. Full multi-sample ratios remain
pending a less-contended benchmark host.

## Experimental blocked f32 CPU attention

PyTorch's CPU FlashAttention uses tiled `QKᵀ` and `P×V` GEMMs with online
softmax. Candle now has the corresponding experimental path for the direct
NHSD f32, unmasked, long-prefill API: 256-query by 512-key tiles, bounded
per-task score/probability/output scratch, zero-copy leading-stride Q/K/V
access where possible, and outer Rayon work over `(head, query tile)`. Inner
GEMMs use `Parallelism::None`, preventing a Rayon-by-BLAS thread explosion.

`CANDLE_CPU_FLASH_BLOCKED=0` selects the retained row-wise kernel for
process-level A/B attribution; the blocked kernel is the default for eligible
NHSD calls. The focused 768-token reference regression passes with a maximum
error below `2e-4`; final real-fixture Python parity was `2.71797e-05` maximum
absolute mask-logit error.

On one immediately adjacent 20-worker, one-iteration diagnostic (host was not
quiet), the retained row kernel measured 5.483 s total / 5.187 s encoder; the
first blocked result measured 5.278 s / 4.985 s. The head-and-query-tile
scheduling revision subsequently measured 5.201 s / 4.905 s with 701 MiB
peak RSS. This is encouraging but is intentionally not a published README
result: a stable multi-sample A/B run remains required.

## Accepted fused NHWC Hiera query pool

Hiera query pooling previously composed NHWC-to-NCHW permutation, generic
2×2 stride-2 max-pool, and NCHW-to-NHWC permutation. A CPU-f32 specialization
now reads last-dimension-contiguous NHWC layouts directly and writes contiguous
NHWC output, with the generic route retained for other devices, dtypes, and
layouts. Its dedicated regression test is elementwise identical to the generic
pipeline. Its end-to-end effect must be attributed in the next stable blocked
attention benchmark. The first combined 20-worker diagnostic measured 4.883 s
total / 4.632 s encoder, 659 MiB peak RSS, and retained `2.71797e-05` maximum
mask-logit error against the already-produced Python reference. This is a
single noisy-host observation, not yet a README baseline.

## Rejected default fused CPU linear bias

Candle now contains a tested f32 contiguous `Linear` experiment that seeds the
final output with bias and uses GEMM destination accumulation, eliminating the
generic matmul-plus-broadcast-add tensor. Its focused test and Candle-NN's
library suite pass, but its first integrated 20-worker diagnostic regressed to
7.892 s total / 7.587 s encoder (versus the preceding approximately 4.9--5.2
s blocked-attention diagnostics). It is therefore opt-in only through
`CANDLE_CPU_LINEAR_BIAS=1`; the established matmul plus bias-add path remains
default. Revisit only with isolated shape-level profiling that explains the
regression.

## Fair blocked-attention benchmark gate

The reusable runner now supports `--no-write`, allowing a fair benchmark to
verify latency, RSS, and output parity without changing README. A three-sample
20-worker run measured Rust 5,113.42 ms versus PyTorch 1,022.08 ms (`5.003x`),
Rust 659.1 MiB versus PyTorch 1,609.3 MiB (`0.410x`), and
`2.71797e-05` maximum mask-logit error. A second identical run used to
regenerate the README experimental row measured 4,820.47 ms versus 795.94 ms
(`6.056x`) with `0.413x` RSS and identical parity. Rust is consistently faster
than its retained pre-blocked diagnostic and meets the parity/RSS gates, but
the 20% ratio spread confirms this shared host remains too variable for a
strong speed-parity claim. Keep the generated row explicitly experimental and
use a quieter host before accepting it as a final baseline.

## Accepted zero-copy f32 K/V layout policy

The blocked f32 kernel now preserves SAM2's interleaved Q/K/V views by default
when their final dimension is contiguous, passing their leading strides to
tile GEMM directly. `CANDLE_CPU_FLASH_PACK_KV=1` retains the prior bounded
head-major K/V pack as a controlled experiment rather than an unconditional
copy. An adjacent one-iteration 20-worker A/B measured zero-copy at 4.716 s
total / 4.455 s encoder and packed at 4.849 s / 4.560 s, with essentially
identical RSS (650 MiB) and identical final Python parity (`2.71797e-05`).
The host is variable, but the zero-copy default also satisfies the model's
layout contract and avoids unnecessary activation copies.

## Accepted in-place blocked-attention probability tile

The blocked kernel previously retained both a score tile and a separate
probability tile per `(head, query tile)` task. It now overwrites each score
tile in place after its row maximum is read, then uses that same storage for
the `P×V` GEMM. The 768-token reference test and full-fixture parity both
remain within the established `2.71797e-05` Python maximum error. A one-run
20-worker diagnostic measured 5.063 s total / 4.767 s encoder and 642 MiB
RSS; host variability prevents a reliable latency comparison, but the removed
allocation is unconditional and lowers the bounded scratch requirement.

## Retained 256-query blocked-attention tile

`CANDLE_CPU_FLASH_Q_BLOCK` now permits bounded power-of-two scheduler A/B
experiments from 32 through 512 rows. On one adjacent triple, 128/256/512
measured 4.946/4.996/4.837 s, apparently favoring 512. The immediate repeat
measured 4.828 s at 256 and 5.023 s at 512, reversing that conclusion. All
variants had identical mask checksums. This is further evidence of host noise,
not a basis for changing the PyTorch-derived 256-row default; retain the
switch only for a quieter machine's shape-level profiling.

## Bounded opt-in Hiera position cache

`SAM2_STATIC_POSITION_CACHE=1` caches one Hiera positional embedding shape
after its first computation. It is bounded to a single tensor (about 24 MiB at
the 256² tiny shape), shape-safe for variable-size calls, and returns a cloned
tensor handle so cached versus uncached final logits were bit-identical. The
first warm-fixture diagnostic measured 4.804 s / 688 MiB with the cache and
the immediate control measured 4.714 s / 690 MiB without it. It therefore
remains an opt-in deployment knob, not a retained default speed optimization.

## Rejected Hiera tanh-GELU approximation

The default exact GELU (`erf`) is a visible profile cost, so the Candle tanh
approximation was tested only in Hiera MLPs. It regressed the real-fixture
diagnostic to 5.921 s total / 5.630 s encoder and increased final maximum
Python mask-logit error from `2.71797e-05` to `1.55302e-02`. The experiment
was removed; exact GELU remains required for this translation.

## Rejected MKL backend retest

The profile's GEMM dominance justified retesting Candle's existing optional
MKL backend after the blocked-attention changes. At the same 20-worker cap it
measured 5.908 s total / 5.551 s encoder and 793 MiB RSS, materially worse
than the current default diagnostics. Its mask checksum remained close, but
the latency and memory regression reject it as a CPU deployment backend.

## Accepted non-overlapping ConvTranspose2D GEMM path

Candle's generic CPU transposed convolution copied NCHW input to NHWC then
performed a separate channel dot product for each output channel and kernel
offset. SAM2's mask decoder uses only the non-overlapping f32 case (`kernel=2,
stride=2,padding=0,dilation=1`), so the new Candle specialization packs each
kernel offset as `(Cin,Cout)`, performs one GEMM per offset, and scatters into
the disjoint output checkerboard. All other configurations retain the generic
kernel; `CANDLE_CONVTRANSPOSE_GEMM=0` is available for process-level A/B.

The direct core regression and full SAM2 suite pass. Immediate real-fixture
A/B measured decoder time 272.0 ms generic versus 225.4 ms specialized
(`0.829x`), with special-versus-generic mask difference `1.43051e-06` and
special-versus-Python difference `2.71797e-05`. End-to-end timing is noisy;
the specialized process's temporary GEMM tiles also raised that run's `VmHWM`,
so this is retained as a decoder-speed optimization rather than an RSS win.

## Current apples-to-apples CPU baseline and remaining gap

The reusable three-iteration runner was rerun after all retained changes
(blocked attention, NHWC pooling, and the decoder specialization): Rust
median latency was 4.981 s versus 0.751 s for the matching PyTorch fixture
(`6.633x` Rust/Python), while peak RSS was 647.2 MiB versus 1,583.6 MiB
(`0.409x`). Maximum mask-logit error remained `2.71797e-05`.

`perf` attributes about 41% of the Rust CPU samples to Candle's generic f32
GEMM microkernel, well ahead of attention-specific code. The next CPU-speed
work should therefore target a measured GEMM improvement in Candle (or a
production-quality accelerated backend that wins this fixture), rather than
add approximate math or more tensor caches. The tested MKL backend, static
position cache, and approximate GELU all failed that acceptance test and are
not defaults.

## Accepted Candle `gemm` x86-v4 runtime kernels

Candle used `gemm` 0.19 without its `x86-v4` feature, so its CPU dispatch
table omitted AVX-512 kernels even on this AVX-512-capable Xeon Gold 6138.
Enabling that dependency feature retains runtime dispatch (older CPUs continue
to select the prior kernels) and does not alter any SAM2 operation or numeric
order. This is distinct from the rejected hand-written AVX-512 attention
experiment: it applies to the generic GEMM hot path.

The focused Candle convolution suite (14 tests) and the complete SAM2
`cpu-flash-attention` suite (14 tests) pass. Two independent three-sample
fair runs measured Rust medians of 4.116 s and 4.229 s; the latter matched
PyTorch's 0.773 s median for a published `5.473x` latency ratio, `0.423x`
RSS ratio, and unchanged `2.71797e-05` mask-logit error. A one-iteration
stage profile measured 4.137 s in the encoder and 219.7 ms in the decoder.
A matched 16-worker rerun regressed Rust to 4.906 s while PyTorch was 0.775 s
(`6.333x`), so the documented 20-worker cap remains the selected deployment
setting on this host.

## Reconfirmed rejection of fused CPU linear bias

The opt-in `CANDLE_CPU_LINEAR_BIAS=1` experiment was repeated after enabling
the faster x86-v4 GEMM dispatch. Its one-pass real-fixture diagnostic was
4.923 s total / 4.665 s encoder / 247.7 ms decoder, versus the current
4.369 s total control profile. It also changed the checksum from
`-57954.73046875` to `-57954.75390625`. The implementation therefore remains
off by default; avoiding the follow-up broadcast allocation does not recover
the cost of its less favorable GEMM scheduling and bias initialization.

## Rejected batched short-window CPU flash attention

Encoder attribution showed Hiera blocks 0 and 1 at 833 ms and 598 ms in one
profile. Those blocks contain many batched 8x8 windows, so Candle's existing
B>1 flash route was tried as an opt-in replacement for the materialized
attention expression. Its varlen packing and output-layout conversion lost to
the generic path: the real-fixture encoder diagnostic rose from 4.428 s to
5.095 s and peak RSS reached 753.4 MiB. The trial checksum
(`-57954.70703125`) also differed from the retained path, so the dispatch was
removed rather than retained as a switch. A useful future short-window kernel
must operate directly on the existing batched NHSD storage without packing.

## Accepted direct batched-NHSD short-window attention

The follow-up retains the existing f32 online-softmax/SIMD row algorithm but
extends its NHSD prefill indexing across independent batch entries. Hiera
window partitioning already represents each 8x8 window as one such entry, so
the new Candle entry point reads Q/K/V through their batch strides and writes
contiguous NHSD output directly. It neither packs varlen sequences nor creates
the generic QK/softmax/PV intermediates. The long, B=1 blocked-GEMM attention
path is unchanged.

The new Candle unit test compares a three-batch, two-head case with the tensor
reference, and the CPU-flash suite now passes 10 tests; the SAM2 suite passes
14. The first direct diagnostic reduced total time to 3.709 s with the same
checksum, and a three-sample fair run measured 3.820 s versus PyTorch's
0.848 s (`4.506x`) at `2.98023e-05` maximum final-logit error. After promoting
the exact CPU/batch shape gate, the final default three-sample run measured
3.808 s versus 0.717 s (`5.315x`) with 800.3 MiB versus 1,642.9 MiB peak RSS
(`0.487x`) and the same error. The PyTorch median varied across runs on the
shared host; the retained conclusion is the reproducible Rust reduction from
the preceding 4.23 s AVX-512-GEMM baseline to about 3.81 s.

## Rejected AVX-512 short-window row kernel

The direct window kernel's AVX2 dot/AXPY/scale loops were given an explicit
AVX-512F/FMA trial, isolated to batched short windows so the retained long
attention path could not be affected. On this Skylake-SP host it regressed the
one-pass encoder diagnostic to 4.892 s total, with block 0 rising to 1.103 s,
and changed the checksum to `-57954.69921875`. The code was removed; AVX2/FMA
remains the selected vector width for this online-softmax row kernel, while
Candle's generic GEMM continues to use its separately beneficial x86-v4
runtime dispatch.

## Rejected bounded K/V packing for short windows

The direct batched kernel was also tested with a one-time `(batch,head,S,D)`
K/V pack, intended to replace its repeated interleaved-QKV reads. It preserved
the final checksum exactly, but the real-fixture encoder diagnostic was
3.984 s, slower than the retained roughly 3.8 s path. At 8x8 window size the
source rows have sufficient locality, so the bounded copy is not repaid. The
experiment was removed; direct batch-stride reads remain the default and the
older K/V pack switch remains restricted to B=1 long attention.

## Retained 64-row short-window task granularity

The direct kernel was compared with 64 versus 128 output rows as the minimum
Rayon task. Immediate one-pass controls measured 3.738 s at 64 rows and
3.792 s at 128 rows with identical checksum. A 64-row task is exactly one
8x8 window for one head, preserving K/V locality without reducing the number
of independent tasks too aggressively; it remains a fixed implementation
choice rather than an exposed deployment knob.

## Accepted final-MLP projection plus residual fusion

Each Hiera block normally materializes its final MLP linear result and then
allocates a separate residual `broadcast_add`. A new Candle
`Linear::forward_add_residual` path is gated to contiguous CPU f32 tensors
with matching output/residual shape. It seeds the only output allocation with
`residual + bias` and lets GEMM accumulate the final projection into it; every
other layout/device follows the ordinary composition. Hiera uses this method
only for its final MLP residual, so ordinary `Linear::forward` behavior is
unchanged.

The new Candle numerical regression compares this result against the composed
operation. On an immediate real-fixture A/B, the fused path measured 3.671 s
versus 3.791 s control (`0.968x`) and reduced peak RSS from 776.6 MiB to
728.4 MiB. A three-sample run before promotion measured 3.778 s / 777.9 MiB
with `2.95639e-05` maximum Python logit error. The post-promotion fair run was
host-contended at 4.194 s Rust and 0.934 s PyTorch (`4.489x`) with the same
error and 752.5 MiB RSS; it is recorded for parity, not used to replace the
stable README table. The fusion is retained because the immediate A/B and
lower-memory result agree, while its changed accumulation order remains inside
the established final-logit error envelope.

## Accepted Hiera MLP expansion bias fusion

The first MLP projection (the 4x expansion before exact GELU) also incurred a
separate broadcast bias allocation. The existing Candle contiguous-f32 linear
bias kernel is therefore exposed only through `Mlp::forward_add_residual`, the
Hiera residual-block flow already using the final-projection fusion; ordinary
Mlp and Linear forwards remain untouched. Immediate diagnostics measured
3.624 s fused versus 3.728 s control (`0.972x`), with the largest early block
falling from 669.7 ms to 591.3 ms. The three-sample fair run measured 4.021 s
Rust versus 0.910 s PyTorch (`4.420x`), 778.4 MiB versus 1,511.0 MiB RSS, and
`2.88486e-05` maximum Python logit error. This is within the retained parity
envelope, so the narrow shape-gated path is now the default for Hiera's MLP
residual blocks.

## Rejected blocked-attention vector-exp follow-up

The scalar exponential helper accounts for about 6.7% of the current profile,
far below GEMM. Candle has no exact portable f32 vector-exp primitive to reuse;
adding an approximation would both duplicate a numerical kernel and expand the
attention error budget. Given the GEMM-dominated profile and the already
rejected approximate-GELU result, this plan rejects an approximate vector-exp
path. Revisit only if Candle gains a parity-tested exact/vector math backend
and a tiled benchmark shows an end-to-end win.

## Cross-runtime Hiera attribution: MLP GEMM is the primary remaining gap

End-to-end samples on the shared host vary substantially, so the static
benchmark now has `--profile-encoder` in both implementations.  It times the
same Hiera block, attention, MLP, and FPN boundaries on one warmed real
1024x1024 fixture.  The profiling instrumentation is intentionally excluded
from normal timing runs.

| Encoder component | Candle CPU (ms) | PyTorch CPU (ms) | Candle / PyTorch |
| --- | ---: | ---: | ---: |
| Hiera MLPs, 12 blocks total | 2206.2 | 199.4 | 11.06x |
| Hiera attention, 12 blocks total | 919.4 | 256.8 | 3.58x |
| FPN neck | 158.5 | 47.2 | 3.35x |

The MLPs alone contributed about 2.007 s of the approximately 3.1 s
single-pass encoder difference.  The follow-up phase profile showed that this
was not primarily a GEMM issue: Candle spent about 1.92 s in exact GELU,
while its expansion and projection GEMMs were only about 0.14 s and 0.15 s.
The reference spent approximately 0.07 s in the corresponding activations.

The cause was Candle's default `GeluErf::f32_vec`: one serial loop calling
scalar `libm::erff` for every element.  The linked fork now splits only large
(at least 16K elements) contiguous exact-GELU tensors into 16K Rayon chunks,
writes directly into Candle's already-allocated output, and retains the same
scalar calculation for every value.  The first profiled pass reduced total
time from 3.764 s to 1.878 s and reduced Hiera GELU from about 1.92 s to
0.17 s, with the Rust checksum unchanged.

An unprofiled, three-measurement same-worker run then measured 1.947 s for
Rust versus 0.712 s for PyTorch (`2.735x`, previously `5.315x` in the retained
experimental table), with 748.5 MiB versus 1,463.2 MiB peak RSS and
`2.88486e-05` maximum final-logit error.  The remaining profile is dominated
by attention and FPN; only after measuring their sub-phases should another
matrix backend be considered.

The corresponding reusable commands are:

```bash
target/release/bench-static checkpoints/sam2.1_hiera_tiny.safetensors checkpoints/bench_bedroom.safetensors 1 --profile-encoder
python3 tools/bench_static_python.py checkpoints/sam2.1_hiera_tiny.pt checkpoints/bench_bedroom.safetensors --iterations 1 --threads 20 --profile-encoder
```

These are attribution runs rather than README benchmarks: hooks/timers alter
the execution schedule, and one pass is used to expose where time is spent,
not to establish a stable throughput ratio.

## Later CPU backend experiments (retained and rejected)

The linked Candle fork now has AVX-512 runtime paths for Hiera's contiguous
NHWC 2x2 max-pool and CPU-flash attention dot/AXPY operations. Both preserve
the scalar fallback and the final-logit envelope; the pool path has a direct
generic-equivalence test. The fused bicubic position resize also now computes
the four vertical taps once per `(channel, output-row)` in a worker-local
`in_width` buffer, then applies horizontal taps in the original order. It
does not introduce an output-sized intermediate.

On the shared 1024x1024 fixture, the latest pinned five-sample comparison
measured 1.198 s Rust and 0.715 s Python (1.675x), 0.432x RSS, and
`2.76566e-05` maximum final-logit error. Wall time remains host-variable, so
the runner supports `--cpu-list` and `--native` for controlled comparisons.

The following did **not** provide a repeatable win and must remain rejected
or opt-in attribution only:

- MKL and OpenBLAS replacements for fused linear projections: nested/small
  GEMM scheduling regressed end-to-end latency.
- Disabling GEMM RHS packing, including a temporary raised threshold: a
  no-repack run was 1.621 s, so the in-tree GEMM packing is beneficial.
- Static-position cache: 1.260 s cached versus 1.235 s uncached in a pinned
  seven-sample comparison.
- jemalloc preload: reduced RSS but regressed latency (1.949 s vs 1.249 s).
- Decoder CPU flash attention and broad vector-exp paths: no stable win.

## Decoder attribution and retained low-copy CPU paths

The static benchmark now exposes matching decoder boundaries in Rust and the
reference Python runner (`--profile-decoder`). On the pinned Tiny fixture,
PyTorch's decoder is roughly 18--21 ms while Candle's remains about 198 ms,
so the decoder is a real independent contributor rather than encoder timing
noise. The largest Candle regions are the two-way cross-attentions and the two
2x2 stride-2 transposed convolutions.

`LayerNorm2d` previously built seven image-sized tensor intermediates for the
channel mean, centered values, variance, standard deviation and affine result.
The CPU f32 route now performs the two reductions and affine transform directly
per `(batch, pixel)` and writes the NCHW destination once. Its small, documented
unsafe write is sound because Rayon partitions unique pixels and every task
writes all channels only for its own pixel; it avoids an NHWC transpose. The
direct equivalence test is within `1e-6`, and the real final logit error remains
`2.74181e-05`. This reduced the decoder's normalization-plus-GELU phase from
about 28 ms to about 2--3 ms.

The Candle fork's specialized non-overlapping stride-2/kernel-2 transpose
convolution originally issued four narrow GEMMs, one per checkerboard offset.
It now packs the four offsets as columns of one `4*Cout` GEMM and scatters that
single result. A pinned seven-iteration A/B with the generic route disabled by
`CANDLE_CONVTRANSPOSE_GEMM=0` measured decoder time of 198.1 ms for the fused
route versus 214.3 ms generic, with the same final checksum. It is retained.

An attempted `-C target-cpu=native` build is rejected: it regressed a
one-pass run to about 4.55 s and changed the checksum, so portable dispatch
remains the parity configuration. A shape-exact decoder-linear probe found the
ordinary Linear path faster than forced fused bias for one projection, but the
extra output allocation showed no repeatable full-decoder benefit. The
lower-copy fused-bias decoder path remains the default pending a GEMM backend
improvement.

Further CPU-flash attribution retained the current no-copy defaults: disabling
the blocked long-attention kernel regressed the shared fixture from about
1.20 s to 2.28 s, while one-time K/V packing regressed it to 1.26 s and the
short-window two-pass path to 1.28 s. A 512-query blocked tile was also slower
than the default 256. These remain opt-in diagnostic switches only.

Likewise, `CANDLE_CPU_FORCE_FUSED_BIAS=0` is an attribution switch, not a
deployment setting. Disabling the contiguous CPU f32 bias-fused linear route
raised a pinned five-iteration run to 2.164 s and RSS to 762 MiB. The fused
route therefore both lowers copying and is essential to current throughput;
future GEMM work must accelerate it rather than replace it with the generic
matmul-plus-bias composition.

The linked `gemm` version does repack fused-linear static weights according to
its RHS packing threshold, but does not expose a reusable packed-RHS object.
Both safe attribution extremes were worse on the pinned fixture:
`CANDLE_GEMM_RHS_PACKING_THRESHOLD=1` measured 1.400 s and `=64` measured
1.355 s, versus the roughly 1.2--1.25 s default configuration; checksums were
unchanged. Keep the backend default heuristic. A future persistent-weight
packing optimization requires a proper `gemm`/Candle API extension rather than
vendoring an undocumented packed layout.

GEMM's Rayon crossover was also made attributable through
`CANDLE_GEMM_THREADING_THRESHOLD`. On the pinned fixture, 4M regressed to
1.244 s; 16M (1.215 s) and 64M (1.208 s) were not better than the paired
default (about 1.192 s). The default `gemm` threading threshold is retained.

PyTorch 2.7's local CPU build uses MKL 2024.2, oneDNN 3.7, OpenMP, and AVX-512.
The reusable Python `--disable-mkldnn` attribution switch measured 668.3 ms
without oneDNN versus 612.0 ms normally on the pinned five-iteration fixture.
Thus oneDNN accounts for a measurable but minority part of the reference
advantage; the larger remaining gap is MKL-grade dense GEMM throughput.

A fused-linear MKL FFI experiment was rejected and removed. The existing
optional Candle MKL backend does not cover the lower-copy fused Linear path;
when routed explicitly, its unthreaded call completed at about 3.77 s, and the
static MKL package did not provide the existing `mkl_set_num_threads_local`
symbol used by the per-call scheduler. The portable in-tree GEMM path remains
the only safe default.

oneDNN 2.2.4 was installed and tested only as an optional fused-linear SGEMM
route, with the same bias-seeded destination and no activation copies. It was
rejected and removed: 20 OpenMP workers measured about 1.83 s with increased
RSS, while one worker measured 2.93 s, versus roughly 1.23 s for portable
Candle. Candle-only optimization remains the intended path.

The generic tiled CPU Conv2d implementation now has bounded
`CANDLE_CPU_CONV2D_TILE_SIZE` attribution. Larger image tiles did not improve
the Hiera patch embedding or the full model: 1024 measured 1.549 s and 2048
1.588 s, both with the normal checksum, versus the default 512-tile path near
1.2 s. The 512 default remains retained.

The broad decoder-flash experiment was rejected because it also routed the
5--7 token query cases, where setup and Rayon scheduling outweighed the small
score matrix. A narrower retained path routes only CPU f32 B=1 decoder
attention with at least 512 query tokens (the 4096-query image-to-token
operation) to Candle's existing NHSD CPU flash kernel. It keeps projected
Q/K/V in NHSD layout and avoids the K transpose, score-matrix materialization,
and head-layout copies. Against the generic decoder on the shared fixture its
mask output differs by at most `2.98023e-06`. In immediate matched stage
profiles, decoder time was 172.2 ms versus 199.4 ms generic; end-to-end timing
remains host-variable, so `SAM2_CPU_DECODER_FLASH=0` is retained as an
attribution switch rather than a parity claim.

The optimized stride-2/kernel-2 CPU transpose convolution also had a serial
checkerboard scatter after its GEMM. Its NCHW output planes are independent,
so Candle now scatters output channels in parallel without allocating another
buffer or creating overlapping writes. The attribution switch
`CANDLE_CONVTRANSPOSE_PAR_SCATTER=0` restored serial scatter: an immediate
decoder profile measured the two upconvs at 19.1 ms and 25.7 ms serial, versus
13.7 ms and 15.6 ms with parallel scatter. The default is parallel; the
switch remains for repeatable shape-specific attribution.

The same transpose-convolution fast path now consumes already-contiguous NCHW
input directly as a strided GEMM matrix instead of repacking it into a
pixel-major temporary. This applies to the decoder upscaler outputs while
preserving the conversion for non-contiguous transformer views. With
`CANDLE_CONVTRANSPOSE_NCHW_DIRECT=0`, immediate attribution measured the two
upconvs at 12.2 ms and 29.1 ms; the direct path measured 6.9 ms and 13.0 ms
with the identical final checksum. The direct no-copy path is the default.

FPN receives Hiera feature maps as regularly strided logical NCHW views. The
Candle fused CPU 1×1 convolution now accepts this layout directly as a
strided `(channel, pixel)` GEMM input rather than falling back to generic
convolution or materializing a transpose. On immediate attribution,
`CANDLE_CPU_CONV1X1_STRIDED=0` measured 87.3 ms for the largest lateral layer
and 133.4 ms for the full FPN, versus 41.7 ms and 60.0 ms on the direct path.
The two final masks differ by at most `3.09944e-06`; the direct route is
default-on and the switch remains for reproducible attribution.

A current `perf record -g` sample on the default Tiny CPU fixture attributes
about 31% of cycles directly to `gemm-f32`'s AVX-512 4x4x6 microkernel and
about 10% to `memmove` (largely GEMM packing/copy traffic). This shifts the
remaining performance priority from SAM2 layout work to the planned reusable
packed-RHS support in `gemm`/Candle. It also explains why toggling isolated
attention or allocator paths has not closed the remaining PyTorch gap: those
paths are no longer the largest fraction of CPU time.

Symbolized sampling also identified `gemm-common` LHS packing at roughly 3.7%
of cycles. Candle now exposes `CANDLE_GEMM_LHS_PACKING_THRESHOLD` for
attribution, applying the value to both serial and Rayon thresholds. Raising
it to 256 (effectively avoiding LHS packing for SAM2's normal widths) regressed
the pinned five-iteration benchmark to 1.086 s versus 1.036 s at the default.
LHS packing therefore remains enabled: its copy cost is repaid by the current
microkernel. This reinforces that a reusable RHS representation must be added
inside GEMM rather than disabled through a global threshold.

An opt-in Rayon worker-affinity experiment, which pinned workers only within
the existing `taskset` CPU mask, was also rejected. A five-iteration immediate
comparison measured 1.019 s pinned versus 0.965 s unpinned. CPU migration
counts therefore do not justify overriding the scheduler on this host.

The Python runner now has `--profile-ops`, which profiles one extra inference
with the installed PyTorch CPU profiler after all timing samples. It does not
modify PyTorch or contaminate comparable timings. On the shared Tiny fixture,
PyTorch reported 19 `aten::_scaled_dot_product_flash_attention_for_cpu` calls
taking 166.6 ms total and 101 `aten::addmm` calls taking 243.0 ms total.
This is sufficient to compare the reference's operator mix directly: the
remaining CPU work belongs in Candle's flash-attention and GEMM paths, not in
a patched PyTorch reference.

## Accepted CPU-flash query tile and RHS-packing crossover

The CPU-flash blocked prefill default is now a 128-query tile (still with a
512-key tile).  Two adjacent five-iteration runs measured 1.012 s at 128
versus 1.020 s at 256; paired fifteen-iteration runs measured 1.135 s versus
1.153 s.  This is a small but repeatable scheduling/locality improvement, and
`CANDLE_CPU_FLASH_Q_BLOCK` remains available for shape-specific attribution.
The dedicated Candle CPU-flash numerical suite passes all ten cases after the
change.

More importantly, current real-model A/B testing found that the in-tree GEMM
RHS-packing crossover of 128 was too eager for SAM2's medium-width projection
matrices.  With all other settings fixed, two reverse-ordered 15-iteration
pairs measured 1.004 s and 1.032 s with a crossover of 256, versus 1.097 s
and 1.085 s at 128.  Checksum was identical in every run.  The vendored GEMM
default is therefore 256 on non-AArch64 targets; AArch64 retains its required
threshold of 2, and the public tuning setter still permits A/B experiments.
This removes avoidable input-side packing traffic while retaining packing for
larger matrices.  It supersedes the earlier, pre-layout-optimization RHS
attribution note above.

An AVX-512 GEMM attribution switch, `CANDLE_GEMM_DISABLE_AVX512=1`, was added
to the vendored GEMM runtime dispatcher.  It chooses the existing AVX2/FMA
kernel before the first GEMM, without changing allocations or numerical
order.  This is deliberately **not** a default: one adjacent pair favored it
(1.074 s versus 1.164 s), but the reverse pair did not (1.167 s versus 1.159
s).  The loaded shared host cannot establish an AVX-512 frequency conclusion;
normal runtime dispatch remains enabled until the experiment can be repeated
on an isolated machine.

## PyTorch CPU-flash source comparison

A temporary sparse checkout of the current PyTorch source confirms that its
CPU FlashAttention uses 256-query by 512-key tiles for sequences of at least
768 tokens (64×512 from 192 tokens, otherwise 32×512).  More importantly, it
fuses vector score scaling/max, exponentiation, and sum reduction within each
softmax row.  Candle's blocked path already fuses score scaling into GEMM and
uses 512-key tiles, but its direct partial SIMD substitutions were not enough:
an AVX2 max/shift/sum experiment measured 1.043 s versus 1.034 s scalar over
15 iterations, and a glibc AVX-512 fused shift/exp/sum experiment measured
1.082 s versus 1.048 s for Candle's established shift + glibc exp + scalar
sum.  Both were removed; the latter also changed reduction order.

Testing PyTorch's 256-query policy directly was inconclusive on this shared
host (1.086 s at 256 versus 1.093 s at Candle's current 128 over 15
iterations).  The 128 default remains because earlier paired testing favored
it slightly.  Candle does retain per-Rayon-worker reusable blocked-attention
scratch and an attributed `CANDLE_CPU_FLASH_KV_BLOCK` setting; neither copies
activations.  The next attention gain requires a complete, shape-specialized
vector softmax implementation, not another isolated loop substitution.

## Accepted 1024-key CPU-flash tile

The blocked prefill default now uses a 128-query by 1024-key tile. Four
reverse-ordered, pinned five-iteration samples averaged 1.002 s at 1024 keys
versus 1.024 s at 512 keys. The resulting full four-repeat Rust/PyTorch run
measured 1.014 s Rust, `2.81334e-05` maximum final-logit error, and 713.5 MiB
Rust peak RSS (0.437× PyTorch's 1632.3 MiB). The larger bounded worker-local
score tile raises RSS by about 65 MiB, but adds no activation copy and is
retained because the active objective is CPU throughput. Set
`CANDLE_CPU_FLASH_KV_BLOCK=512` where the smaller bounded workspace is the
more important deployment constraint.

With the 1024-key tile fixed, reducing query rows from 128 to 64 lowered RSS
by about 20 MiB but averaged 1.040 s versus 1.014 s (2.6% slower), so 128
remains the default. An attempted GEMM RHS-packing comparison labeled 512
measured 1.173 s versus 1.137 s at 256, but the vendored setter then clamped
both settings to 256. Those numbers measured host variation, not a packing
policy difference. The corrected setter and fresh comparison are described
below.

## Accepted in-place Hiera expansion GELU

The linked Candle fork now exposes a CPU f32 linear+exact-GELU path that
applies the existing scalar or glibc vector GELU kernel directly to the owned
GEMM output. Hiera's first MLP layer uses it, avoiding a second activation
allocation and a full tensor read/write. `CANDLE_CPU_LINEAR_GELU_FUSION=0`
retains the separate-op path for attribution. A dedicated 32K-element Candle
test matches the separate exact GELU within `1e-6`; the real fixture produced
an identical mask checksum and `2.81334e-05` maximum PyTorch logit error.

Four reverse-ordered five-iteration runs with eight pinned workers averaged
1.206 s fused versus 1.228 s separate (1.8% faster). On the same eight-worker
fixture, four independent Rust/PyTorch runs averaged 1.195 s versus 0.828 s
(`1.443×`) and 590.3 versus 1541.4 MiB peak RSS (`0.383×`). The 20-worker
host was too variable for an updated speed ratio: PyTorch medians ranged from
0.56 to 5.27 s during one set of otherwise identical runs, so that batch was
not used to update the 20-worker README row.

## Accepted batched short-window attention GEMMs

Hiera's batched short windows now use two serial GEMMs per `(batch, head)`:
`QKᵀ` into one reusable worker-local score matrix, exact softmax in place,
then `P×V` directly into the final interleaved NHSD output. This removes the
per-score dot/AXPY loop without copying Q, K, V, or the output. The previous
row kernel remains available with `CANDLE_CPU_FLASH_SHORT_GEMM=0`.

Four reverse-ordered five-iteration runs on eight pinned workers averaged
1.099 s for the new path versus 1.208 s for the row kernel (9.0% faster).
The 64-token batched Candle reference test passes, and the full real-fixture
maximum PyTorch mask-logit error is `2.90871e-05`. Four independent
apples-to-apples 20-worker runs averaged 0.928 s Rust versus 0.615 s PyTorch
(`1.510×`), with 596.8 versus 1599.6 MiB peak RSS (`0.373×`). Absolute Rust
time improved from the preceding 1.014 s baseline; the ratio did not improve
because PyTorch also ran faster in this shared-host batch. One eight-worker
Rust sample was contended, so its four-run mean improved only slightly
(1.191 s versus the previous 1.195 s).

## Accepted tiny-query decoder attention through asymmetric GEMM

The decoder's 5--7 token queries against 4096 image keys now use the same
NHSD GEMM attention kernel, bounded to 2--16 query tokens and 512--8192 keys.
The score buffer is below 1 MiB at the fixture shape; projected Q/K/V and
the final output stay in their native layout. This avoids K transposition and
the repeated head-layout materializations in the generic decoder path.
`SAM2_CPU_DECODER_SHORT_FLASH=0` isolates this change while leaving the
previous long-query decoder flash route enabled.

The asymmetric Candle reference test passes. Four reverse-ordered five-pass
full-model samples on eight pinned workers averaged 1.121 s enabled versus
1.162 s disabled (3.6% faster). A matched decoder profile showed about
119 ms versus the preceding 136 ms. Four independent 20-worker Rust/PyTorch
runs averaged 0.948 s versus 0.682 s (`1.390×`), with 600.2 versus
1584.1 MiB peak RSS (`0.379×`) and `2.93255e-05` maximum final-logit error.
The enabled path raised peak RSS in three of four paired eight-worker runs,
so the speed gain has a memory trade-off despite its small explicit score
buffer.

A 2048-key blocked-attention tile was rejected after four reverse-ordered
five-pass eight-worker samples: it averaged 1.368 s versus 1.208 s for the
retained 1024-key tile (13.3% slower). Its larger score workspace did not
repay the reduced softmax/GEMM setup frequency.

Routing the decoder's reverse asymmetry (4096 image queries against 5--7
token keys) through the full-score GEMM kernel was also rejected. Four
reverse-ordered five-pass samples averaged 1.087 s enabled versus 1.092 s
for the retained blocked attention route, a 0.4% difference within host
variation. The candidate changed the reduction-order checksum without a
reliable full-model gain, so its dispatch and test were removed.

## Corrected GEMM RHS-packing attribution

The vendored GEMM's `set_rhs_packing_threshold` previously capped every
requested value at 256. It now allows explicit experimental settings through
4096 while retaining the 256 default. This fixes the benchmark control; it
does not change normal inference. A four-pair 256-versus-512 real-fixture run
after the fix was inconclusive because the shared host's five-iteration
medians ranged from 1.44 to 3.01 seconds, and the paired ordering reversed.
The default remains 256 until a lower-contention run can establish whether
less repeated packing repays the GEMM microkernel slowdown.

## Copy-free fused-linear GEMM orientation probe

The fused CPU linear kernel can now compute the same row-major result as
`(X Wᵀ)ᵀ = W Xᵀ`, selected with `CANDLE_CPU_LINEAR_TRANSPOSE_GEMM=1` in the
linked Candle fork. This only swaps GEMM operands/strides; it allocates no
transposed tensor. It makes the output contiguous along GEMM's `m` dimension
and can reduce LHS packing when the flattened input has many more rows than
the weight has output channels. The default orientation is unchanged.

On the real Hiera Tiny fixture with 20 pinned workers, two reverse-ordered
sets of four independent five-iteration pairs gave baseline medians (ms)
`[982.745, 993.539, 1048.220, 3461.858]` and
`[984.208, 943.007, 973.400, 952.357]`, against transposed medians
`[927.704, 949.713, 983.896, 3199.431]` and
`[995.207, 965.291, 917.576, 934.787]`. The first set favored the
transposed call in all four pairs, but the second split two/two. The host
also had severe contention in one pair, so this does not yet establish a
reliable end-to-end gain. At eight pinned workers, the first three pairs
were essentially tied (normal `1193.164, 1156.070, 1150.289` ms;
transposed `1199.338, 1143.386, 1163.798` ms), and the last pair was
contention-affected. A 262,144-byte real-fixture mask dump was byte-identical
between orientations. Keep the option as an attribution probe; do not claim
an accepted speedup or change the README ratio on these measurements.

## Decoder cross-attention input-layout attribution and retained fix

The matching `--profile-decoder` runners now split each of the five decoder
cross-attentions into Q/K/V projection, attention kernel, output projection,
and module-total times. Rust additionally reports pre-module positional-encoding
adds and post-module residual/normalization times. The five slots are the two
token-to-image attentions, two image-to-token attentions, then final attention.
This distinguishes work inside attention from its enclosing decoder block;
the older broad block labels alone overstated the GEMM/kernel contribution.

The image positional encoding arrives NCHW. The decoder flattens and permutes
it to `(B, HW, C)`, producing a non-contiguous view that was added to the same
image-token matrix five times. Each strided `broadcast_add` was expensive.
The decoder now makes **one** contiguous `(B, HW, C)` encoding per call and
reuses it for all five additions. This is a bounded 4 MiB preparation on the
Tiny fixture, not five materializations or any thread-local duplication.
`SAM2_CPU_DECODER_PACK_IMAGE_PE=0` retains the old view route for A/B testing;
non-CPU devices keep the original route.

Four reverse-ordered 20-worker, real-fixture profiled pairs put the sum of
Rust's five pre-module phases at `51.1` ms with the view and `5.3` ms with
the contiguous preparation, on average. The enclosing two-way transformer
averaged `101.9` versus `60.5` ms. Checksums were identical. In the new
apples-to-apples decoder-only benchmark (one precomputed image embedding,
five measured decoder calls per process), four independent runs averaged
`208.9` ms view versus `109.4` ms prepared versus `19.5` ms PyTorch. One
view run was heavily contended at `381.9` ms; its other three runs were
`138.8`, `157.6`, and `157.3` ms, and every paired prepared run was faster.
The complete 262,144-byte Rust mask dump was byte-identical with the switch
off/on. The stage win is retained as the CPU default even though the loaded
host obscured its roughly 40 ms contribution in full-model A/B pairs.

The subsequent standard four-run/five-iteration full-model comparison measured
`983.79` ms Rust versus `734.37` ms PyTorch (`1.340×`), peak RSS `606.6`
versus `1431.7` MiB (`0.424×`), and maximum mask-logit error
`2.93255e-05`. That is the README experimental row, not an isolated causal
estimate relative to the older `1.390×` run. Decoder-only measurements are
reproducible with `--decoder-only` on both `bench-static` and
`tools/bench_static_python.py`; the full benchmark runner now records the
CPU pinning list in its generated table metadata.

## Decoder upscaler fusion and phase attribution

The second 2x2/stride-2 transposed convolution remained a visible decoder
hotspot. A safe channel-grouped scatter experiment was rejected and removed:
four five-iteration decoder-only runs averaged `109.71` ms for the existing
scatter, `110.26` ms with groups of two, and `115.01` ms with groups of four.
Reducing repeated tile reads did not repay the lower parallelism and changed
write order.

The linked Candle fork now has a gated `CANDLE_CONVTRANSPOSE_PROFILE=1` timer
for input preparation, weight packing, GEMM, and scatter. On a representative
20-worker decoder run, its core 64x64 and 128x128 upscalers took about `2.6`
and `3.5` ms, while the corresponding broad SAM2 decoder stages took `6.1`
and `11.5` ms. Those broad stages also include the high-resolution skip
projection/addition, and the second includes GELU; they are not pure
transposed-convolution times.

For eligible CPU f32 2x2/stride-2 convolutions, Candle now supports writing
the bias while scattering GEMM output, then writing the equal-shaped skip
residual in the same pass. `sam2-rs` uses this no-extra-output path for both
mask-upscaler layers. It does not allocate or copy a second upscaled tensor;
the standard composed route remains available with
`CANDLE_CONVTRANSPOSE_FUSED_RESIDUAL=0`. Direct Candle tests cover a strided
input and multiple batches. The full 262,144-byte real mask was byte-identical
with the switch off/on.

Four reverse-ordered five-iteration decoder-only pairs measured `106.22` ms
with the composed residual add and `104.69` ms fused, on average. The broad
first-upscaler phase averaged `6.63` versus `5.10` ms, and the second
`12.35` versus `9.51` ms. Three of four decoder pairs favored fusion; the
stage gains were more consistent than the small whole-decoder difference.
The later standard four-run full-model sample measured `940.69` ms Rust and
`654.90` ms PyTorch (`1.436×`), RSS `611.5` and `1429.8` MiB (`0.428×`),
and unchanged `2.93255e-05` maximum logit error. A repeat contained one
3.65-second Rust outlier while other Rust samples were near one second, and
an eight-worker repeat likewise had a 4.08-second Rust outlier. These host
spikes make cross-run ratio changes unsuitable for causal attribution.

## Hiera subphase attribution and direct window copies

Four paired, five-iteration runs pinned to 20 workers narrowed the CPU gap to
the encoder: Rust averaged `835.34` ms versus PyTorch `565.72` ms for the
encoder, and `92.47` versus `18.95` ms for the decoder. Within Hiera, Rust's
attention kernel itself was faster (`151.34` versus `174.88` ms), but its QKV
and attention-output projections took `119.37` and `34.15` ms versus PyTorch's
`68.77` and `18.29` ms. The MLP took `231.63` versus `172.44` ms, and the FPN
`51.78` versus `33.27` ms. These are measured stage sums, not isolated-kernel
benchmarks; host contention and surrounding layout costs still matter.

Further Rust-only Hiera timers identified approximately `25.94` ms in window
partition, `13.45` ms in unpartition, `18.01` and `16.20` ms in the two norms,
`19.39` ms in shortcut, and `25.62` ms in residual adds. These sum across the
12 Hiera blocks. The partition route previously padded and permuted through
multiple tensor operations. An eligible contiguous CPU-f32 path now directly
copies each window to one preallocated output, in disjoint Rayon chunks; the
reverse path copies each cropped output row once. Neither path duplicates the
input for parallel workers, and both retain the generic fallback. Set
`SAM2_CPU_DIRECT_WINDOWS=0` to compare against that fallback.

Four paired runs at eight pinned workers gave partition times of
`22.0, 22.4, 21.9, 22.1` ms on the generic route and
`9.7, 9.9, 9.7, 10.1` ms on the direct route. Unpartition was
`11.7, 11.3, 10.8, 11.5` versus `7.6, 7.8, 7.4, 7.7` ms. The full real-mask
output was byte-identical with the switch off/on. The full-model timings in
these runs were too noisy to attribute a model-wide win to this change alone.
Targeted unit tests compare padded and exact-tile outputs with the generic
implementation.

With direct CPU windows enabled, the standard four-run/five-iteration
20-worker full-model comparison averaged `829.91` ms Rust and `600.44` ms
PyTorch (`1.382×`), with mean peak RSS `590.5` versus `1442.5` MiB (`0.409×`)
and maximum mask-logit error `2.93255e-05`. Rust run medians ranged
`817–838` ms; PyTorch ranged `531–668` ms on this shared host. The speed
ratio remains above parity, and the cross-run change from the prior README
row is not by itself evidence of a causal whole-model speedup.

## Fuse Hiera unpartition with the shortcut residual

Nine of the 12 Hiera blocks unpartition windowed attention output and
immediately add the equal-shaped shortcut. The contiguous CPU-f32 path now
writes the sum directly into the unpartitioned output, avoiding the separate
output allocation and full-tensor add. Rayon owns disjoint output rows and
reads the shared windows and shortcut without worker-local copies. Unsupported
shapes, dtypes, devices, and layouts retain the composed route;
`SAM2_CPU_FUSED_WINDOW_RESIDUAL=0` disables only this fusion, leaving direct
windows available. `SAM2_CPU_DIRECT_WINDOWS=0` disables both direct paths.

Padded and exact-tile unit cases match the composed tensor route, and the
full real-mask output was byte-identical with fusion off/on. In four
reverse-ordered five-iteration pairs at eight pinned workers, the fused
window-unpartition-plus-add stage was about `9` ms in three clean samples;
the corresponding unfused unpartition alone ranged `23–50` ms, with
additional residual-add time. One fused sample also suffered an isolated
`50` ms outlier. Severe host contention (one full-model run at `4.09` s)
precludes attributing the whole-model result of those pairs to the fusion.

A separate reverse-ordered four-pair full-model A/B at 20 pinned workers
averaged `891.19` ms with fusion off and `847.28` ms with fusion on
(`0.951×`), with essentially unchanged mean peak RSS (`590.2` versus
`590.4` MiB). Three of four pairs favored fusion. The subsequent standard
four-run Rust/PyTorch sample contained a `2.28` s Rust outlier and averaged
`1.766×`; it is not used to replace the more stable README row. Maximum
mask-logit error versus PyTorch remained `2.93255e-05`.

The repeated standard four-run comparison with the fusion enabled was more
stable: `882.99` ms Rust versus `618.39` ms PyTorch (`1.428×`), mean peak RSS
`586.1` versus `1421.7` MiB (`0.412×`), and the same `2.93255e-05` maximum
mask-logit error. This is the current README row; the CPU parity gap remains.

## Single-write output seeding for Candle fused Linear

The linked Candle fused CPU-f32 Linear path previously allocated a zero-filled
output and then overwrote every element with the bias or residual-plus-bias
before GEMM. It now initializes the allocated spare capacity once, in disjoint
Rayon row chunks of `MaybeUninit<f32>`, and exposes the old zero-fill route
through `CANDLE_CPU_LINEAR_SINGLE_WRITE_SEED=0`. The only new unsafe operation
sets the vector length after every element has been written; no worker copies
the input, weight, bias, or residual. The six linked Candle linear tests pass,
including bias, residual, and exact-GELU paths. The complete real-image masks
are byte-identical with seeding off/on.

At the Hiera `4096×384×1536` MLP-expansion shape, four reverse-ordered
microbenchmark pairs averaged `18.49` ms for zero-fill-plus-seed and `16.57`
ms for single-write seeding. The larger `65536×96×384` shape improved only
slightly (`54.15` versus `53.41` ms), and a QKV-sized shape was too noisy to
attribute a gain. The more relevant unprofiled full-model A/B, four
reverse-ordered five-iteration pairs at 20 pinned workers, favored
single-write seeding in all four pairs: `853.95` versus `752.00` ms on
average (`0.881×`). The earlier profiled A/B contained multi-second
contention spikes and was not used for this decision.

Two consecutive standard four-run Rust/PyTorch batches with single-write
seeding enabled averaged, across all eight independent runs, `832.46` ms
Rust and `682.85` ms PyTorch (`1.219×`); mean peak RSS was `592.3` versus
`1438.8` MiB (`0.412×`), and maximum mask-logit error remained
`2.93255e-05`. The batch ratios were `1.194×` and `1.244×`, respectively;
the shared host's run-to-run spread remains considerable. The combined
eight-run result replaces the README row, but the lower-copy CPU path is
accepted chiefly on its paired A/B evidence, not on a cross-date ratio.

Fresh four-pair full-model phase profiling after the Linear change moves the
next priority toward the decoder. The encoder averaged `729.61` ms Rust and
`679.39` ms PyTorch; the decoder `100.74` versus `26.37` ms. Hiera attention
kernel time was `153.14` versus `208.77` ms, while QKV took `89.01` versus
`75.39` ms. Hiera MLP as a whole was `190.35` versus `195.87` ms, and FPN
`72.86` versus `39.08` ms. These synchronized profiled calls are diagnostic,
not a replacement for the unprofiled README ratio.

Within the decoder, the two transformer MLPs took `7.19` ms Rust versus
`0.41` ms PyTorch in those profiles. Routing them through the existing
fused-residual MLP API was tested but rejected: four reverse-ordered
decoder-only pairs averaged about `108` ms with the composed path versus
`109` ms fused, and the full mask bytes differed. The experiment was removed.
The decoder's 2×2 upscalers and cross-attention remain the larger actionable
stage gaps.

A decoder-wide few-row Linear experiment also failed the model-level gate.
For the second MLP projection (`5×2048×256`), a parallel row-wise dot-product
kernel reduced the isolated linear probe from roughly `3` to `2` ms. Four
decoder-only pairs instead averaged `95.6` ms on the existing GEMM route and
`98.8` ms with the direct kernel. Its full-mask bytes differed as well. The
kernel was removed; faster isolated tiny GEMM is insufficient unless it wins
in the surrounding decoder schedule.

The decoder upscaler was split into skip 1×1 projections and transpose-conv
cores. Four 20-worker decoder-only profiles averaged `2.59` and `5.45` ms
for the two high-resolution skips, `3.67` and `3.76` ms for the two upconv
cores, `2.28` ms for the intermediate norm/GELU, and `0.66` ms for final
GELU. The remaining broad decoder-transformer phase was `51.03` ms, of
which about `13.26` ms was one-time image-token layout preparation.

For eligible contiguous NCHW CPU-f32 image embeddings and position encoding,
the decoder now transposes directly into its reused contiguous token layout.
Disjoint 32-pixel output tiles are written in parallel; workers share the
original input storage and create no private copies. Other layouts and devices
retain the generic reshape/permute/contiguous path, and
`SAM2_CPU_DECODER_DIRECT_PACK=0` selects it for attribution. A unit test
compares multi-batch and multi-channel tile outputs elementwise against the
generic path; the complete real-image mask is byte-identical off/on.
Four reverse-ordered five-iteration decoder-only pairs at 20 pinned workers
measured setup `12.97` ms generic versus `10.09` ms direct, and whole-decoder
`103.60` versus `95.76` ms (`0.924×`). All four pairs favored direct packing.

Two consecutive standard four-run Rust/PyTorch batches with direct decoder
packing enabled averaged `783.72` versus `631.20` ms over all eight fresh
processes (`1.242×`), mean peak RSS `571.8` versus `1425.4` MiB (`0.401×`),
and unchanged `2.93255e-05` maximum mask-logit error. Their separate ratios
were `1.312×` and `1.172×`, illustrating shared-host variability. The paired
decoder A/B and byte-identical masks, rather than the cross-date README ratio,
are the acceptance evidence for the direct transpose.

## Channel-parallel single-write 1×1-convolution seeding

The linked Candle fused CPU-f32 1×1 convolution path had the same redundant
output initialization as Linear, but worse parallelism: for B=1 it seeded all
output-channel planes serially after zero-filling the allocation. The largest
FPN lateral writes a 256-channel, 256×256 output. It now initializes disjoint
channel planes in parallel through `MaybeUninit<f32>` and sets the vector
length only after every element is written. GEMM still accumulates into the
bias-seeded output, and all workers share the original input, weights, and
bias without private copies. `CANDLE_CPU_CONV1X1_SINGLE_WRITE_SEED=0` retains
the old seeding route for attribution. Both linked Candle fused 1×1-conv tests
pass, including strided NCHW input, and the complete real-image mask is
byte-identical off/on.

Four reverse-ordered five-iteration full-model pairs at 20 pinned workers
measured the largest FPN lateral at `45.04` ms old versus `8.98` ms new, the
whole FPN at `65.84` versus `23.44` ms, and the full Rust model at `762.34`
versus `707.64` ms (`0.928×`). All four pairs favored the channel-parallel
single-write path. This is a direct model-level win; the upcoming Rust/PyTorch
ratio remains subject to shared-host variability.

Two subsequent standard four-run Rust/PyTorch batches with this path enabled
averaged, across eight fresh processes, `696.05` ms Rust versus `601.50` ms
PyTorch (`1.157×`), mean peak RSS `580.0` versus `1429.5` MiB (`0.406×`),
and unchanged `2.93255e-05` maximum mask-logit error. The individual batch
ratios were `1.200×` and `1.118×`. The README now reports their combined
average; CPU speed parity remains unproven.

Fresh four-pair synchronized stage profiling after this change measured
encoder `590.71` ms Rust versus `565.49` ms PyTorch (`1.045×`) and decoder
`85.72` versus `18.75` ms (`4.573×`). Hiera blocks were `505.46` versus
`482.34` ms, Hiera MLP `166.62` versus `166.11` ms, and FPN `21.92` versus
`35.66` ms. The decoder now dominates the measured gap: its two-way
transformer setup took about `11.01` ms; the five matched cross-attention
module calls totaled about `15.03` versus `6.40` ms. Decoder upscaler broad
stages were `4.88` and `7.88` ms versus PyTorch's `1.73` and `1.83` ms.
The narrower Python MLP hooks exclude the residual and normalization included
in Rust's `7.54` ms MLP phase, so that particular pair is not an
apples-to-apples operator comparison. These stage timings include profiling
overhead and are diagnostic.

Raising GEMM's process-wide threading crossover to 4M was rechecked on four
reverse-ordered decoder-only pairs. Excluding one heavily contended 249 ms
trial, the decoder totals remained near the default; the MLP subphase
improved only slightly and the cross-attention did not consistently win.
The default crossover remains unchanged. Subsequent work should target the
decoder's large projection/layout and upscaler costs in context, rather than
globally changing GEMM's thread policy.

## Cache the fixed CPU dense prompt positional encoding

The decoder-only profile left roughly `19` ms outside the mask-decoder's
instrumented components. `PromptEncoder::get_dense_pe()` was regenerating
the same 64×64 random Fourier encoding from fixed checkpoint weights on
every decode. For CPU models, the encoder now computes it once at load and
returns a shallow Tensor clone thereafter. The cache is bounded to one
`1×256×64×64` f32 tensor (4 MiB); GPU and
`SAM2_CPU_CACHE_DENSE_PE=0` retain the original per-call path. This changes
no prompt-dependent result, and the complete real-image mask is
byte-identical with the cache off/on.

Four reverse-ordered five-iteration decoder-only pairs at 20 pinned workers
averaged `93.96` ms without the cache and `75.25` ms with it (`0.801×`).
The time outside mask-decoder components dropped from `19.38` to `7.80` ms.
All four decoder-only pairs favored the cache. A full-model A/B was much
noisier: its last cache-off process took `1.35` s, while earlier processes
in either mode were near `0.67–0.72` s. That sample does not isolate the
model-wide saving; the decoder-only evidence does.

The first standard four-run Rust/PyTorch comparison with the cache enabled
averaged `709.23` versus `627.91` ms (`1.130×`), mean peak RSS `581.0` versus
`1415.6` MiB (`0.410×`), and unchanged `2.93255e-05` maximum mask-logit
error. A second four-run batch contained a `2.75` s Rust outlier and is not
used for the README headline; its other Rust medians were `647–777` ms.
The cache's bounded 4 MiB cost is small relative to observed process RSS,
but the host variability prevents attributing the cross-date ratio change
to the cache alone.

## Decoder thread and projection attribution after the cache

A four-run decoder-only worker sweep initially suggested eight or twelve
workers might be faster than 20, but it also changed the CPU affinity mask
and had multi-second contention outliers. A controlled reverse-ordered
four-pair test kept affinity at CPUs `0–19` while varying only the worker
count: 20 workers averaged `73.28` ms versus `89.59` ms at eight. No local
decoder thread-pool policy was added. A steady-state delayed `perf record`
sample attributed about 22% of decoder CPU cycles to the vendored GEMM's
AVX-512 microkernel, 5.6% to Rayon epoch pinning, 5.4% to the transpose-conv
kernel, and 4.4% to strided-copy work. Sampling identifies priorities, not
latency savings.

Prompt-level profiling now separates prompt construction and dense-PE lookup
from the mask decoder. Four decoder-only runs averaged `0.11` ms for the
point-prompt encoder and effectively zero for cached dense-PE retrieval;
the mask decoder's broad transformer phase was `41.40` ms, with `10.07` ms
in initial image-token setup. About `7.2` ms remains outside the presently
instrumented subphases, so the prompt encoder is not the next useful target.

Routing only the decoder's large Q/K/V and output projections through
Candle's ordinary Linear path was also rejected. Four reverse-ordered
decoder-only pairs averaged `73.43` ms with the retained fused-bias path
versus `83.58` ms generic. Matched QKV and output-projection timings both
regressed, although final masks were byte-identical. The experimental route
was removed.

A separate four-pair trial of Candle's transposed-GEMM linear orientation
(`CANDLE_CPU_LINEAR_TRANSPOSE_GEMM=1`) did not show a consistent win at the
Hiera projection boundaries, so the default remains the existing orientation.

## Reuse the decoder's packed image positional encoding

The cached dense positional encoding was still transposed and packed from
`1×256×64×64` NCHW to `1×4096×256` tokens on every decode. The ordinary
`broadcast_as` call in `MaskDecoder` also created a new Tensor identity even
when its batch shape already matched, defeating a cache keyed by tensor
identity. The decoder now retains the original identity in that case and the
two-way transformer holds one packed CPU-f32 positional tensor per instance.
The cache is bounded to one 4 MiB tensor, uses shallow Tensor clones, and
does not create a copy per worker. It is disabled for variable tensors, GPU,
or with `SAM2_CPU_DECODER_CACHE_PACKED_PE=0`.

Four reverse-ordered decoder-only pairs, each with five measured iterations,
20 workers pinned to CPUs `0–19`, averaged `85.15` ms with this cache disabled
and `66.92` ms enabled (`0.786×`). Every pair favored the cache. The
transformer's image-token setup fell from `10.59` to `0.70` ms on average;
the remaining improvement includes fewer layout/allocator costs. The full
mask files were byte-identical, and a unit test covers same-identity reuse
and replacement on a different positional tensor. Mean decoder-process peak
RSS was `525.6` MiB off versus `552.0` MiB on; only 4 MiB is the deliberate
cache payload, so do not attribute the entire observed difference to it.

The contemporaneous four-run full-model Rust/PyTorch batch averaged `709.67`
versus `691.06` ms (`1.027×`) with unchanged `2.93255e-05` maximum mask-logit
error and mean peak RSS `551.7` versus `1451.2` MiB (`0.380×`). It is **not**
a stable new headline: PyTorch medians were `593, 932, 588, 651` ms, and a
separate full-model cache A/B had `1.88` and `3.63` s cache-off outliers.
The shared host reported a load average near 30 during this work. Keep the
previous README row until a less-contended four-run comparison can establish
an end-to-end ratio. The controlled decoder-only result does establish this
optimization's local effect without claiming full-model parity.

## Experimental fused residual LayerNorm

The image-to-token path adds a 4096×256 residual before LayerNorm twice per
decode. A narrowly gated CPU-f32 implementation now computes the sum and
normalization in one pass using a 256-float stack row per worker; it does not
copy the shared input per thread. The fallback uses the original Candle
`broadcast_add` and `LayerNorm`. An exact-value unit test and byte-identical
real-fixture masks cover the specialized path.

Four reverse-ordered decoder-only pairs averaged `65.00` ms for the composed
path and `60.14` ms for fusion (`0.925×`), and all four pairs favored fusion.
However, two reverse-ordered four-pair full-model batches averaged about
`669.9` ms with fusion off versus `684.3` ms on (`1.022×`), with highly
variable individual encoder times. Stage profiling confirmed that the fused
image-to-token postprocessing itself is faster, but did not establish a
stable full-model win. The experiment is therefore **off by default** and
can be retested with `SAM2_CPU_DECODER_FUSED_RESIDUAL_NORM=1` on a quiet host;
the README ratio does not include it.

## Candle token LayerNorm allocation and SIMD trials

The linked Candle `LayerNorm` CPU path was tested with two reversible
backend experiments. A single-write destination avoided the initial zero
fill while retaining each row's reduction and affine order. Eight
reverse-ordered, five-iteration full-model pairs gave contradictory batches:
`673.1` ms old versus `696.4` ms single-write in the first and `635.4`
versus `625.2` ms in the second. The combined `654.3` versus `660.8` ms
does not establish a model-level gain.

An AVX2 affine pass kept the scalar mean/variance reduction order and changed
only the independent final per-channel arithmetic. Four reverse-ordered
full-model pairs averaged about `702.3` ms on the original Candle route
versus `731.0` ms with AVX2. The 20-worker real-fixture masks were
byte-identical for both experiments, and Candle's CPU LayerNorm tests passed,
but neither route met the speed gate. Both experimental edits were removed
from the Candle fork; its existing LayerNorm path is unchanged.

Two fresh four-run Rust/PyTorch comparisons on the current default code
averaged `664.58` versus `558.91` ms (`1.189×`) and `717.24` versus
`670.82` ms (`1.069×`), with the same `2.93255e-05` maximum mask-logit
error. The four-run mean is the intended reporting unit, but the second
batch included sharply increased host contention, including a `756` ms
PyTorch median, and a subsequent deep Rust profile ran above `1.8` s.
These observations confirm that CPU parity is not established; they do not
justify replacing the less-contended README headline with a noisy batch.

## Parallel short-row decoder Linear, without per-worker input copies

Four paired decoder profiles on the real Tiny fixture put Rust's two
transformer MLP modules near `5` ms together versus about `0.35` ms in
PyTorch. The exact second projection is only `8×2048 → 256`, but Candle's
ordinary GEMM route took roughly `3–4` ms in the isolated reusable
`linear-probe`. The preceding `8×256 → 2048` expansion was already around
`0.5` ms, so optimizing both as one undifferentiated MLP obscured the target.

The linked Candle fork now has a shape-gated CPU-f32 Linear route for
`2–16` rows, at least `1024` input features, and at least `128` outputs.
For bias-only inference with at least four Rayon workers, Rayon partitions
*output columns* into disjoint 16-channel tiles (32 by default above 512
outputs). Each tile invokes a serial GEMM on the same read-only input and
weight storage, accumulating into
its own columns of one final output. No worker gets a private input copy and
there is no nested GEMM thread pool. Nonmatching shapes, devices, dtypes,
residual/GELU fusions, and `CANDLE_CPU_LINEAR_SHORTROW_SPLIT=0` retain the
prior path. `CANDLE_CPU_LINEAR_SHORTROW_CHUNK` permits attribution of tile
size. A Candle unit test compares both full and tail output tiles against
ordinary matmul.

On four isolated runs, the 8×2048→256 projection improved from about
`3.39` ms to `0.13` ms with 16-channel tiles. The SAM2 decoder now routes
its two transformer MLPs through Candle's fused-bias Linear (revert with
`SAM2_CPU_DECODER_MLP_FUSED_BIAS=0`); this and the shape-gated split are
enabled by default. Four reverse-ordered decoder-only pairs averaged
`66.13` ms with the split disabled versus `53.29` ms enabled (`0.806×`),
all four favoring the new route. The paired transformer MLP phases fell
from roughly `5` ms to about `1` ms.

The first four-pair full-model A/B averaged `679.80` ms with both decoder
switches off versus `656.18` ms on (`0.965×`); three of four pairs favored
on, and mean peak RSS was about `587` versus `581` MiB. In a reverse-order
repeat, the first three pairs all favored on by roughly `35` ms on average;
the last pair hit major host contention (`2.69` s off, `1.05` s on) and is
not useful as a latency estimate. The output is not byte-identical to the
prior GEMM order: the maximum on/off mask difference is `3.34e-06`, while
the maximum error versus the same PyTorch mask improved slightly from
`2.93255e-05` to `2.86102e-05`. This is a parity-gated, model-level win,
but it does **not** by itself establish Rust/PyTorch speed parity.

The first standard four-run Rust/PyTorch comparison after enabling this path
averaged `649.97` versus `552.78` ms (`1.176×`), and the second averaged
`656.35` versus `594.13` ms (`1.105×`). Across all eight fresh processes,
the then-current README row was `653.16` versus `573.46` ms (`1.139×`), mean
peak RSS `574.1` versus `1456.9` MiB (`0.394×`), and maximum mask error
`2.86102e-05`. PyTorch medians ranged `520–663` ms and Rust `620–676` ms;
the shared host's variation prevents treating this cross-date ratio as a
causal estimate of the new kernel. The paired full-model A/B above is the
acceptance evidence. CPU speed parity remains open.

Four fresh matched decoder-only profiles after enabling the short-row route
averaged about `52.24` ms Rust versus `18.60` ms PyTorch. Rust's two
transformer MLPs now total about `0.87` ms versus PyTorch's `0.34` ms, so
that was a real but no longer dominant gap. The two image-to-token attention
modules total about `9.51` versus `3.23` ms, and the two token-to-image
modules about `5.47` versus `2.32` ms. These are matching module hooks and
make cross-attention the next decoder target. Rust's two broad upscaler
stages total roughly `11.5` ms, while Python's hooked transposed-convolution
modules total roughly `3.9` ms; those boundaries are *not* identical because
the Rust stages also include high-resolution skip projections. Isolate those
skip costs before attributing the whole difference to Candle's transpose
convolution.

## Reuse within each two-way attention block

The fixed image positional encoding is already packed once per decoder.
Within each two-way transformer layer, the image keys themselves do not
change between token-to-image and image-to-token attention, yet the same
`keys + image_pe` tensor was being materialized twice. Both attentions now
share the first result; `SAM2_CPU_DECODER_REUSE_KEY_PE=0` restores the two
adds. No thread-local copy is involved. Four reverse-ordered decoder-only
pairs averaged `61.47` ms with reuse off versus `53.80` ms on; all four
pairs favored reuse. The two image-to-token pre-module additions fell from
about `0.9` ms each to tensor reuse overhead, and final mask files were
byte-identical.

Full-model A/B remains host-limited. One batch ended with a `3.16` s
cache-off outlier; the reverse-order batch averaged `710.35` ms on versus
`684.20` ms off, but its first-iteration encoder stages differed by about
`24` ms in the same direction, while decoder stages averaged `44.59` ms on
versus `45.80` ms off. This does not show a clean full-model gain. The exact
reuse remains enabled because it removes two proven redundant 4 MiB additions
per decode without changing masks. The later README ratio includes this reuse,
but should not be interpreted as its isolated model-wide speedup.

Two related attribution checks were not promoted. Re-enabling the fused
residual-add/LayerNorm after the new MLP path cut its large postprocessing
subphases but gave mixed whole-decoder results (`54.73` ms off versus
`52.76` ms on; only two of four pairs favored on), so it stays opt-in.
Disabling Candle's long blocked flash-attention route for the 4096-query,
short-key decoder attention averaged `50.06` ms blocked versus `49.59` ms
row-wise decoder time, with no consistent pair win and a larger mask error
versus PyTorch (`4.58e-05` instead of `2.86e-05`). The established blocked
route remains the default.

## Match image-time skip projection placement

The Rust decoder profile initially attributed roughly 4–7 ms to the two
high-resolution 1×1 skip projections on every prompt decode. Inspection of
the pinned PyTorch source showed that `SAM2Base.forward_image` applies
`conv_s0` and `conv_s1` once when it encodes the image; its decoder receives
the projected skips. Python decoder hooks confirmed that neither projection
runs during a cached-image decode. The old decoder-only module timings were
therefore not apples-to-apples: Rust repeated image work inside every decode.

`Sam2ImagePredictor.encode_image` now stores separately projected skips in
its `BackboneOutput`, and the predictor's decode path consumes them without
projecting again. Standalone `ImageEncoder` and `MaskDecoder` entry points
retain their original raw-feature behavior; GPU is unchanged. Set
`SAM2_CPU_PREDICTOR_PREPROJECT_SKIPS=0` to restore the old CPU predictor path.
This moves work rather than removing it from a one-prompt full-image
inference, and retains the raw FPN maps for public callers rather than
silently changing their meaning. There are no per-worker copies.

Four reverse-ordered decoder-only A/B pairs (20 pinned workers, five measured
iterations per fresh process) averaged `49.04` ms with projection per decode
versus `45.05` ms with image-time projection; all four pairs favored the new
path and the final mask files were byte-identical. In four fresh matched
Rust/PyTorch cached-image decoder runs after this change, Rust averaged
`47.48` ms versus PyTorch `17.37` ms (`2.73×`); CPU speed parity remains
open. A four-pair full-model A/B averaged `634.26` ms off versus `653.64` ms
on, but the measured encoder stages alone were about `14` ms slower on amid
host variation. The decoder stage saved about `3.6` ms while that work moved
to encoding, so these runs do not establish a full-model gain.

Two subsequent standard four-run Rust/PyTorch batches with the key reuse and
image-time skip projections both enabled averaged `684.20` versus `608.84` ms
(`1.124×`) and `693.56` versus `687.77` ms (`1.008×`). Across eight independent
processes, the README row is `688.88` versus `648.30` ms (`1.063×`), peak RSS
`586.4` versus `1404.1` MiB (`0.418×`), and maximum mask-logit error
`2.86102e-05`. The batch spread is substantial and the full-model ratio is
still above one; do not claim CPU speed parity from the near-one second batch.

## Pool interleaved Hiera queries without materializing NHWC

A later encoder profile exposed about `22` ms inside Hiera's three query-pool
boundaries. Merely batching adjacent output pixels into Rayon tasks did **not**
explain the apparent full-model improvement: direct timers found only about
`3` ms in the max-pool kernel, versus roughly `18–20` ms in the preceding
`NHSD → NHWC` reshape of the interleaved QKV projection view. The reshape
materialized Q before it could be pooled. The initial four-pair tile-size
experiment favored 32 pixels/task (`648.28` versus `633.47` ms full model),
but subsequent pool-core timings stayed near `3` ms in both modes; that
apparent model win cannot safely be attributed to task batching alone.

`pool_nhsd_query_cpu_f32` now reads Q directly from its strided QKV view and
writes the pooled NHSD tensor once. It is restricted to even spatial sizes,
contiguous head channels, and CPU f32; all other cases use the existing
reshape/pool path. `SAM2_CPU_HIERA_DIRECT_QUERY_POOL=0` restores the old route,
and `SAM2_CPU_HIERA_POOL_PIXELS_PER_TASK=1` restores one-pixel Rayon tasks for
attribution. An interleaved, multi-batch, channel-tail unit test and the real
fixture both produced byte-identical results with the direct path.

Four reverse-ordered full-model A/B pairs measured query pooling at `21.79`
ms old versus `3.99` ms direct and full latency at `691.65` versus `666.95`
ms. A second reverse-order four-pair batch averaged `659.66` versus `639.32`
ms. Across eight pairs, direct pooling averaged `653.13` versus `675.65` ms
(`0.967×`), with six of eight pairs favoring it; two unfavorable pairs were
affected by host variation. This is a measured model-level optimization, not
proof of cross-runtime parity. A subsequent standard Rust/PyTorch batch was
invalid for headline use because one PyTorch process took `6.58` s under
contention while its peers were near `0.55–0.67` s.

Two later, usable standard four-run Rust/PyTorch batches with direct query
pooling averaged `662.30` versus `671.74` ms (`0.986×`) and `668.38` versus
`622.45` ms (`1.074×`). Across eight independent runs the current README row
is `665.34` versus `647.09` ms (`1.028×`), peak RSS `586.5` versus `1398.4`
MiB (`0.419×`), and maximum mask error `2.86102e-05`. The batch spread means
CPU speed parity is still not established. A further Rust attention profile
showed creating the Q/K/V strided Tensor views takes only about `0.05` ms
across all Hiera blocks; that is not the next useful target. Some broad
attention time remains outside QKV, direct query pooling, flash kernel, and
output projection, but its present attribution is too indirect to assign to
one operator.

## Reject broad Linear bias-seeding task batching

The linked Candle fused f32 Linear path seeds one output row per Rayon task
before GEMM. A temporary fork experiment batched 32 adjacent rows per task
for projections with at least 256 rows, sharing input and weights and writing
one output buffer. All seven focused Candle Linear tests passed and the real
fixture mask was byte-identical. The paired full-model test did not support
retaining it: its first pair was contaminated by multi-second host contention,
and the other three pairs had mixed results with no stable QKV/MLP or latency
gain. The experiment was removed from the fork, leaving the previous seeding
route intact. The remaining Hiera QKV gap calls for GEMM throughput work,
possibly the packed-RHS API sketched in `OPT.md`, rather than this scheduling
change.

## Decoder scheduling checks and Hiera cleanup attribution

The NCHW decoder LayerNorm2d CPU path was tested with 64 adjacent pixels per
explicit Rayon task, sharing the input and final output buffer. Four
reverse-ordered decoder-only pairs averaged `48.39` ms with the existing
indexed Rayon range versus `48.64` ms batched; the final masks were
byte-identical. The indexed range is already partitioned by Rayon, so the
manual batching was removed. A larger multi-batch equivalence test remains.

The optional fused residual-add/LayerNorm was also retested in four
reverse-ordered full-model pairs after direct Hiera query pooling. Whole-model
pairs split two-to-two, while first-iteration decoder stages often favored
fusion; final masks were byte-identical. It remains opt-in with
`SAM2_CPU_DECODER_FUSED_RESIDUAL_NORM=1`, not a claimed model-level win.

New Hiera attention profiling accounts for the release of its temporary QKV
and attended tensors after projection. In the least-contended of four early
diagnostics it took about `21.8` ms over 12 blocks; other samples reached
`27–46` ms while host load rose sharply. This may explain much of the prior
time outside QKV, pooling, flash attention, and output projection, but the
shared-host samples are not yet strong enough to assign a causal speed gap to
allocation or deallocation. The next measurement should split the two
temporary releases and check whether a bounded reuse mechanism can remove
work without pushing RSS toward PyTorch's level.

## Hiera QKV release and glibc allocator tuning

Splitting the release profile showed that the attention-output tensor was
negligible (`~0.02` ms across 12 blocks), while releasing the QKV tensor cost
`19.8–28.7` ms in four clean diagnostic processes. The expensive blocks were
the early large projections. The allocator, not Q/K/V view construction, was
therefore the next useful target.

On Linux/glibc, setting both `MALLOC_MMAP_THRESHOLD_` and
`MALLOC_TRIM_THRESHOLD_` to 256 MiB removed almost all measured QKV release
time (`~0.03` ms per encoder), without changing the mask bytes. Two
reverse-ordered, four-pair full-model A/B batches then favored the tuned
allocator in all eight pairs: mean Rust median fell from `693.00` to `606.50`
ms (`0.875×`). Mean peak RSS rose from `577.0` to `607.2` MiB (`1.052×`),
remaining well below PyTorch's RSS on this fixture. This experiment did not
copy input memory per worker.

The same settings are now applied before normal CPU model loading through
glibc `mallopt`, once per process. CUDA loading is unchanged. The process-wide
setting is opt-out via `SAM2_CPU_KEEP_LARGE_ALLOCATIONS=0`; pre-existing
`MALLOC_MMAP_THRESHOLD_` or `MALLOC_TRIM_THRESHOLD_` environment settings
override it. A direct profiled check of the integrated path measured QKV
cleanup at `~0.026` ms versus `~20.6` ms with the opt-out, and produced
byte-identical masks. There is an RSS trade-off, so this should not be
described as a pure allocator improvement for every workload.

Two standard, independent four-run Rust/PyTorch benchmark batches with this
integrated default averaged `608.86` versus `683.71` ms (`0.891×`) and
`551.26` versus `621.14` ms (`0.887×`). Across all eight independent runs,
the README row is `580.06` versus `652.42` ms (`0.889×`), peak RSS `629.6`
versus `1411.8` MiB (`0.446×`), and maximum mask-logit error
`2.86102e-05`. Every individual paired run favored Rust. This establishes
CPU speed parity for the frozen Tiny static-image fixture at 20 workers on
this machine; it does not prove parity for the other variants, video path,
different prompts, or other machines.

## Eight-worker follow-up after allocator tuning

The older eight-worker README row predates the allocator change, so two new
four-run batches were collected on pinned CPUs 0–7. They averaged `869.69`
versus `890.84` ms (`0.976×`) and `950.13` versus `941.71` ms (`1.009×`)
for Rust/PyTorch. Only two of the eight individual pairs favored Rust;
several processes in both runtimes had large contention outliers. A new
`--alternate-order` option in `tools/run_benchmarks.py` reverses process order
in odd pairs and records that order per run. Its first four-run diagnostic
averaged `821.06` versus `866.47` ms (`0.948×`), but one PyTorch process took
`1104.84` ms versus its cleaner `730–762` ms peers. None of these batches
supports an eight-worker speed-parity claim or a replacement README row.

One lower-contention matched stage attribution reported the Rust long Hiera
attention kernel at `~272` ms versus PyTorch at `~199` ms; Rust's Hiera MLP
and FPN were competitive or faster in the same probes. Because the profiles
were separate processes and time only their first measured iteration, this
is a target for focused A/B work, not a causal model-wide delta.

A four-pair eight-worker test of 128 versus 256 attention query rows produced
byte-identical final masks but unusable latency data: individual Rust medians
spanned `779–1730` ms and host load reached `19.25` on a 20-core machine.
No tile-size default was changed. Repeat only on a quieter host or use a
more isolated attention-level benchmark before considering a thread-count-
dependent tile policy.

## CUDA baseline on Quadro RTX 5000

The separate GPU runner was executed with Candle CUDA/cuDNN and PyTorch
2.7.0+cu128/cuDNN 9.7.1 on a Quadro RTX 5000 (driver 570.133.20). Both paths
used the frozen Tiny checkpoint fixture, one warm-up, ten synchronized
measured iterations per fresh process, and alternating process order.

The first four-run validation averaged `161.12` ms Rust versus `67.32` ms
PyTorch (`2.393×`), host peak RSS `457.6` versus `1188.1` MiB (`0.385×`),
and maximum mask-logit error `1.26362e-05`. The independent README batch
averaged `162.11` versus `67.65` ms (`2.396×`), host peak RSS `457.5` versus
`1188.3` MiB (`0.385×`), with the same maximum error. Each implementation's
four medians were tightly grouped, so this GPU latency gap is reproducible.
The RSS metric is Linux process `VmHWM`; it does not measure peak VRAM.

The CUDA output remained within the established mask-logit parity envelope,
and Rust used substantially less host RSS, but this initial Rust GPU path was
about 2.4 times slower than the original PyTorch implementation.

### CUDA translation and bottleneck isolation

Matching the implementation more closely to PyTorch removed nearly all of
that gap. The retained changes are:

- use a last-dimension CUDA softmax and apply the attention scale to Q before
  the GEMM;
- fuse CUDA linear bias with cuBLASLt in the linked Candle fork;
- translate PyTorch's SM75 f32 memory-efficient attention specialization to a
  self-contained CUTLASS kernel, including pooled-query attention;
- pool Hiera's NHWC activations and interleaved Q view directly on CUDA rather
  than permuting/copying through NCHW.

The first two changes reduced Rust from `162.11` to about `99.43` ms. The
CUTLASS attention translation reduced it to `79.73` ms, and direct NHWC
pooling produced the retained eager four-run result: `75.90` ms Rust versus
`67.20` ms PyTorch (`1.129×`), host peak RSS `459.1` versus `1188.3` MiB
(`0.386×`), and maximum mask-logit error `1.43051e-05`.

Nsight Systems then showed approximate GPU-compute parity. Across the traced
workload, total Rust and PyTorch kernel time was essentially equal, while the
translated Rust attention kernels matched or beat their PyTorch counterparts.
The remaining eager wall-time difference correlates with Candle creating and
destroying roughly 1,670 CUDA events and issuing roughly 835 stream-ordered
allocations per inference. PyTorch's caching allocator avoids most of that
host-side work. This isolates a later Candle allocator/event-lifetime project;
additional SAM2 math changes are unlikely to recover the residual by
themselves.

For the benchmark's actual static-input use case, both harnesses now support
the same CUDA graph capture/replay mode. The alternating-order, four-process,
ten-iteration graph-vs-graph comparison measured `67.97` ms Rust versus
`67.16` ms PyTorch (`1.012×`), host peak RSS `470.8` versus `1192.6` MiB
(`0.395×`), and the same `1.43051e-05` maximum error. This reaches practical
speed parity without hiding work asymmetrically. The Rust graph path also
disables Candle's cross-stream lifetime events before model construction;
this is safe here because the benchmark uses one stream and synchronizes every
replay, and it is not applied to general eager execution. Reproduce with:

```console
python3 tools/run_gpu_benchmarks.py --cuda-graph --alternate-order --repeats 4 --iterations 10
```

CUDA graph mode is deliberately limited to the fixed image/shape/prompt
benchmark. Variable inputs remain on the eager path; improving Candle's
allocator is the next route to eager parity.
