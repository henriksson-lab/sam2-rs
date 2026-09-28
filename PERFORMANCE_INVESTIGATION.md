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

## Realistic CPU/GPU suite, round 1

The generalized suite initially found a false output-parity failure. The Rust
high-level predictor uses upstream's default dynamic multimask stability
fallback, but the old Python microbenchmark built the model with
`apply_postprocessing=False`. Center clicks happened not to expose the
difference. Both benchmark drivers now use upstream defaults; the low-level
component fixture remains explicitly raw. A real-bedroom CPU intermediate
probe then measured `1.076e-5` feature error and `8.774e-5` mask-logit error,
while the corrected GPU `I1` smoke measured `2.861e-5` final error.

Change: Cache projected high-resolution skip features on CUDA during image
encoding, matching `SAM2ImagePredictor.set_image`.

Hypothesis and PyTorch/Candle evidence: PyTorch projects both high-resolution
FPN levels in `forward_image` and retains them for all subsequent clicks.
Candle did this only on CPU, rerunning both convolutions on every CUDA decode.

Affected suite cells: GPU `I1-I4`, `S1`, video edit paths that share image
decoding, and AMG prompt batches.

Focused A/B (all pairs): one-process diagnostic `I3/bedroom`, before `7.058`
vs `5.837` ms (`1.209x`), after `6.520` vs `5.946` ms (`1.096x`). The after run
used nine measured prompt changes following three warmups. This is sufficient
to retain the translation fix, but not to promote a four-pair headline.

GPU and CPU P0 aggregates before/after: incomplete; the full matrix is still
being implemented.

Worst per-device P0 regression: GPU `I1` changed from a pre-fix `1.121x`
diagnostic to `1.117x`; no observed regression. GPU `I2` was `1.087x`.

Peak/current RSS and GPU allocated/reserved bytes before/after: after-change
external peak VRAM was 690 MiB Rust versus 984 MiB PyTorch (`0.701x`) in `I3`;
host peak RSS ratio was `0.392x`. Candle allocated/reserved counters are not yet
exposed, so process VRAM is the current apples-to-apples measure.

Parity result: retained (`7.343e-5` maximum and `2.077e-5` mean absolute mask
error for the mixed cached prompt sequence).

Decision: retain.

Next workload family: full/rich GPU interaction regression smoke completed;
rotate next to video initialization/propagation infrastructure.

## Realistic GPU video and AMG, round 2

Change: Cache the immutable 64×64 memory-encoder sine position tensor on its
target device rather than constructing it for every propagated frame.

Hypothesis and PyTorch/Candle evidence: the reference reuses its positional
encoding while Rust rebuilt identical trigonometric tensors. On `V2`, Rust
fell from `15729.6` to `15241.4 ms` (`2.705x` to `2.626x` relative to
PyTorch), while process VRAM fell by about 32 MiB. Thresholded selected-frame
masks remained exact. Decision: retain.

Change: Use the already validated CUTLASS memory-efficient CUDA attention in
video RoPE memory attention instead of materializing the Q×K score tensor.

Hypothesis and PyTorch/Candle evidence: PyTorch selects a memory-efficient
attention backend for the same long-key shapes. The `V2` diagnostic measured
`14916.95` versus `5792.83 ms` (`2.575x`); the speed win was modest, but Rust
process VRAM fell from roughly `2.05x` to `1.30x` PyTorch. Selected-frame
threshold masks remained exact, with expected accumulated float-logit drift.
Decision: retain for the large memory reduction. Next family: AMG.

Change: Match upstream's M2M refinement batch bound in AMG and drop unused
initial restored masks before refinement.

Hypothesis and PyTorch/Candle evidence: a 64-point multimask call produces 192
candidates. PyTorch refines those in three batches of 64; Rust accidentally
refined all 192 at once. Correct batching reduced A1 Rust process VRAM from
`10354` to `3442 MiB`, changing its ratio from `2.087x` to `0.694x`, with
unchanged output parity. Latency stayed near 2.1 seconds. Decision: retain.

Change: Calculate AMG IoU/stability filtering on the GPU per bounded batch and
download only survivors before host RLE construction.

Hypothesis and PyTorch/Candle evidence: Rust downloaded all 192 full-resolution
float masks; PyTorch filters them on-device first. Rust A1 latency fell from
about `2100` to `1413 ms` in the focused diagnostic, and host HWM fell from
about `1.15 GiB` to `0.59 GiB`. The four-pair retained baseline averaged
`1463.38` versus `646.84 ms` (`2.262x`), host RSS `0.373x`, and process VRAM
`0.694x`. Complete mask metadata and RLE parity passed at `4.768e-7` maximum
float error. The remaining synchronized attribution points to the decoder:
Rust/PyTorch were `340/239 ms` for the initial batch and `911/457 ms` for
three refinement batches before this filtering change. Decision: retain;
latency still fails. Next workload family: eager image interaction.

Change: Diagnostic-only disabling of Candle cross-stream storage lifetime
events for the suite's owned single CUDA stream.

Hypothesis and PyTorch/Candle evidence: prior Nsight traces counted many CUDA
events, and this benchmark never shares tensors across streams. Four fresh
bedroom pairs measured `77.58` versus `73.37 ms` (`1.057x`), statistically
indistinguishable from the ordinary row's `78.04` versus `74.05 ms` (`1.054x`).
Parity and memory were unchanged. Decision: reject and remove the suite flag;
event tracking alone is not the remaining eager bottleneck. Investigate
allocation reuse/launch bookkeeping instead.

Change: Set a bounded 256 MiB release threshold on CUDA's stream-ordered
memory pool at SAM2 model-loading boundaries; add current/high used and
reserved pool metrics to Candle.

Hypothesis and PyTorch/Candle evidence: cudarc uses `cuMemAllocAsync`, but the
default pool release threshold can discard reusable blocks at synchronization.
The suite synchronizes each sample, while PyTorch retains cached blocks. Four
I1 bedroom pairs improved Rust from `78.04` to `74.97 ms` (about 3.9%), moving
the ratio from `1.054x` to `1.018x`; external peak VRAM remained `690 MiB`
versus PyTorch's `984 MiB`. A subsequent diagnostic measured `74.58` versus
`73.86 ms` and reported Rust pool bytes of 144 MB live, 268 MB reserved,
497 MB peak-used, and 570 MB peak-reserved, versus PyTorch's 327/879/629/879
MB. The policy is opt-out with `SAM2_CUDA_POOL_RETAIN_MIB=0` and configurable
for constrained devices.

The same default reduced a warmed `V2` diagnostic from the prior `14916.95`
to `12508.25 ms`; PyTorch measured `5905.97 ms`, so the cell remains a large
`2.118x` latency failure. External process VRAM was `2194` versus `1998 MiB`
(`1.098x`), narrowly inside the per-cell cap, and host RSS was `0.260x`.
AMG was neutral (`1451.52` versus `646.14 ms`, `2.246x`) and retained its
strong memory ratios. Output parity was unchanged. Decision: retain; rerun the
full image matrix before promoting the optimized I1 aggregate.

Change: Route only the mask decoder's long asymmetric CUDA attention through
the translated CUTLASS memory-efficient kernel.

Hypothesis and PyTorch/Candle evidence: `I4` exposes the batched cached-prompt
and prior-mask path used during interactive correction. PyTorch dispatches its
long asymmetric decoder attention through scaled-dot-product attention, while
Candle materialized the full score tensor and both head-layout copies. The
retained path applies only when either query or key length is at least 512;
the decoder's tiny self-attention stays on the existing implementation.

Affected suite cells: GPU `I1-I4`, AMG decoder batches, and video correction
decodes. Four paired `I4/bedroom` processes with two warmups and twenty measured
iterations averaged `39.8283 ms` Rust versus `20.6766 ms` PyTorch (`1.926x`).
Four otherwise identical processes with
`SAM2_CUDA_MEM_EFFICIENT_ATTENTION=0` averaged `40.5588` versus `21.5132 ms`.
The retained Rust path is therefore about 1.8% faster than the Rust control and
reduces external peak VRAM from 754 to 722 MiB. Its process-VRAM ratio is
`0.731x`, host-RSS ratio `0.349x`, peak pool allocated ratio `0.919x`, and peak
reserved ratio `0.684x`. Maximum mask-logit error improved from `6.294e-5` to
`2.956e-5` and remains within the image parity envelope.

Decision: retain. The `I4` latency cell still fails substantially, so this is
not a parity claim. After two image-path experiments (allocator retention and
asymmetric attention), rotate to the video family and attribute the `V2`
memory-encoder fusers.

## Realistic GPU video, round 3: native depthwise convolution and RoPE lifetime

Change: Execute inference-only grouped CUDA convolution as one native cuDNN
operation in Candle instead of splitting it into one convolution per group.

Hypothesis and PyTorch/Candle evidence: synchronized matched profiling found
that each Rust memory-encoder ConvNeXt fuser took about `48.3 ms`, versus
roughly `1.2 ms` for the complete PyTorch block. Its 7x7, 256-group depthwise
convolution alone averaged `46.0 ms`. Candle decomposed `groups=256` into 256
separate convolution launches and concatenated the results even though cuDNN
supports a native group count. The retained Candle path sets that group count
for CUDA tensors that do not track gradients; training and non-cuDNN backends
keep the existing decomposition. The depthwise stage fell to `0.49 ms` and a
complete Rust fuser to about `3.15 ms`. End-to-end parity was unchanged.

Change: Precompute each memory-attention module's fixed 64x64 axial RoPE table
at load time, matching the reference module's registered `freqs_cis` lifetime.

Hypothesis and PyTorch/Candle evidence: Rust rebuilt the same coordinate grids,
sines, and cosines in every self- and cross-attention call. Across a warmed
32-frame profile this happened 496 times. Caching reduced Rust self-attention
from `37.38` to `4.79 ms/call`, matching PyTorch's `4.84 ms/call`, and reduced
cross-attention from `65.70` to `33.80 ms/call` versus PyTorch's `28.83 ms`.
The synchronized Rust pass fell from `16.77` to `9.02 s`.

Affected suite cells: GPU `V2-V4` and video edit portions of `V3`. A subsequent
ordinary warmed `V2` diagnostic measured `6358.89 ms` Rust versus `6124.09 ms`
PyTorch (`1.038x`), compared with `12508.25` versus `5905.97 ms` (`2.118x`)
before these changes. Host RSS was `0.256x`; peak allocated device bytes
`0.986x`; allocator peak-reserved and external process VRAM were both `1.002x`.
Selected-frame logit error remained `0.07999` maximum and `0.00558` mean, with
zero threshold-mask mismatches.

Decision: retain both changes. This focused result satisfies the individual
latency and memory caps but needs the required four paired processes before it
becomes the scorecard baseline. Rotate away from video after these two retained
experiments.

## Realistic GPU AMG and cached decoder, round 4

Change: Dispatch CUDA transposed convolution through cuDNN backward-data in
Candle instead of its naive elementwise kernel.

Hypothesis and PyTorch/Candle evidence: matched `A1` decoder profiling found
the two Rust upscaler convolutions far behind PyTorch. Candle had no cuDNN
transposed-convolution route even when built with cuDNN. The first Rust upconv
fell from `83.2` to `20.0 ms/call`; the second upconv plus GELU fell from
`69.2` to `36.1 ms/call`. The implementation accepts strided inputs, makes a
contiguous weight only when required, and retains the existing CUDA kernel in
non-cuDNN builds.

Change: Use Candle's fused last-dimension CUDA layer norm for SAM2
`LayerNorm2d`, with one channels-last packing copy and a stride-only NCHW output
view for the following cuDNN convolution.

Hypothesis and PyTorch/Candle evidence: the generic expression allocated seven
full NCHW intermediates for an activation as large as 268 MB in the 64-prompt
AMG batch. The fused route reduced Rust normalization plus GELU from `45.9` to
`15.4 ms/call`, matching PyTorch's approximately `15.2 ms/call`. It is
independently disableable with `SAM2_CUDA_LAYER_NORM_2D_LAST_DIM=0`.

Affected suite cells: all GPU image/cached decodes, `A1-A2`, and video prompt
edits. An ordinary warmed `A1/groceries` diagnostic improved from `1391.85` to
`918.90 ms`; PyTorch measured `639.37 ms`, so the ratio improved from `2.177x`
to `1.437x`. The prior validated four-pair baseline was `2.262x`. External
VRAM remained favorable at `0.758x`, peak allocated bytes at `0.922x`, host RSS
at `0.382x`, and complete mask/RLE parity passed with `1.192e-7` maximum error.
`I4/bedroom` improved from `39.83` to `26.26 ms` Rust against `20.21 ms`
PyTorch, changing its focused ratio from `1.926x` to `1.299x`; its device and
host memory gates pass.

Decision: retain both changes. AMG and `I4` still fail their latency caps, so
rotate after these two experiments. The remaining synchronized `A1` decoder
gap is primarily transformer time (`109.1` versus `71.5 ms/call`) and the two
upconvs (`55.4` versus about `26.7 ms/call`).

## GPU cached-prompt promotion and CPU diagnostic

Four paired `I3` processes on each realistic fixture now establish a passing
GPU baseline. Bedroom measured `4.8879` versus `5.1183 ms` (`0.955x`) and
groceries `4.7674` versus `5.0571 ms` (`0.943x`), for weighted latency
`0.949x`. Weighted external VRAM was `0.701x`, peak allocated bytes `0.808x`,
peak reserved bytes `0.496x`, and host RSS `0.360x`. Maximum image-logit error
was `2.575e-5`.

The corresponding 20-worker CPU diagnostic remains a large failure at
`2.329x` weighted latency despite `0.357x` host RSS. Bedroom's rich-click/box
output also exceeded the mean-error gate (`3.645e-5` versus `3e-5`) while its
maximum error and thresholded mask passed; point-only (`1.401e-5`) and the
multimask/box case (`2.305e-5`) passed. Removing glibc vector exp or CPU flash
attention worsened error, and removing flash attention increased Rust RSS from
roughly 0.36x to 0.75x PyTorch. Both features remain enabled. The benchmark
runner now records an explicit `--rust-features` override for reproducible
feature A/Bs. The rich-prompt drift and CPU decoder latency remain open.

A first four-pair promotion with parity failures retained as diagnostics was
unintentionally unpinned and measured `2.063x`; it is superseded. After the
orchestrator began applying the suite's explicit 20-worker `0–19` affinity to
both implementations, the authoritative four-pair run measured `57.34` versus
`21.05 ms` on bedroom (`2.724x`) and `55.89` versus `21.46 ms` on groceries
(`2.604x`), for a weighted `2.664x`. Host RSS remains strong at `0.351x`.
Bedroom repeats the same `3.645e-5` rich-prompt mean error with no threshold-
mask mismatch; groceries passes at `1.878e-5`. The retained failed-baseline run
is `.benchmarks/runs/20260926T232523Z`. The override does not relax the parity
gate; it allowed complete timing/RSS evidence to be recorded.

A later pinned refresh re-evaluated the existing fused residual-add/LayerNorm
path on the actual `I3` prompt sequence. Four fusion-on processes averaged
`49.23 ms` bedroom and `48.00 ms` groceries, while the adjacent four-process
fusion-off control averaged `52.08` and `53.06 ms`. Both runs passed the new
repeat-spread noise gate; parity errors and threshold decisions were unchanged.
The resulting Rust/PyTorch ratio improves from `2.677x` control to `2.465x`
fused, still far from parity. Fusion is now the CPU default, with
`SAM2_CPU_DECODER_FUSED_RESIDUAL_NORM=0` retaining the ordinary composition
for attribution and unsupported layouts automatically falling back.

Detailed decoder-only profiling explains the bounded gain and the remaining
gap. The Rust transformer takes about `24.3 ms`; five cross-attention modules
account for roughly `12.7 ms` (`6.8 ms` projections, `4.3 ms` attention
kernels, and `1.6 ms` output projections). The two image-to-token residual
normalizations were about `3.3 ms` before fusion, and three necessary
key-position additions about `3.1 ms`. Current matched broad I3 profiling is
still `30–33 ms` Rust versus `10.6–12.4 ms` PyTorch for the transformer, so
future CPU work should target SDPA/projection throughput rather than another
LayerNorm scheduling variant.

The current decoder/backend changes were then promoted through two additional
four-pair GPU image rows. `I1` across all six image fixtures improved from
`1.056x` to `1.021x` weighted latency; all individual rows pass, weighted
external VRAM is `0.725x`, peak allocated bytes `0.791x`, and host RSS
`0.347x`. `I2` across bedroom, groceries, and portrait measured `1.031x`
weighted latency with all individual rows between `1.012x` and `1.046x`;
weighted external VRAM is `0.713x`, peak allocated bytes `0.790x`, and host RSS
`0.364x`. Both weighted latency gates remain narrowly open, while their memory
and output-parity gates pass.

## GPU video promotion and retained-state lifetime

The native grouped-convolution and RoPE-cache changes were promoted through
four paired runs. `V1` initialization is substantially faster than PyTorch at
`0.424x` latency and uses `0.677x` peak process VRAM. `V2` 32-frame forward
propagation is near parity at `1.033x`, with `0.986x` process VRAM and peak
allocated device memory. The retained runs are
`.benchmarks/runs/20260926T211109Z` and
`.benchmarks/runs/20260926T211157Z`.

Investigation of `V3` found that `InteractiveVideoSession` retained complete
`Sam2FrameOutput` values for every object and frame in addition to the actual
temporal-memory bank. That duplicated conditioned 64x64 features and retained
1024x1024 output masks that later edits never consume. The session now keeps a
compact output containing only low-resolution mask logits, object score, and
object pointer. Tensor handles are cloned shallowly; no extra data copy is
introduced. The temporal-memory bank remains the single owner of encoded
memory, and an internal parity probe reads it there.

On the promoted four-pair `V3` run this reduces warmed Rust peak process VRAM
from `2834 MiB` to `2034 MiB`, changing the process-VRAM ratio from `1.212x` to
`0.870x`; peak allocated is `0.885x`, peak reserved `0.862x`, and host RSS
`0.271x`. Latency is `1.034x` (`23075.18` versus `22307.29 ms`). The stable
third click was moved from a pillow/background edge to the pillow interior so
the workflow measures implementation drift rather than an ambiguous prompt.
The only larger drift is the accumulated corrected-frame logit (`0.108863`
maximum, `0.005405` mean); thresholded masks remain exact. A V3-only `0.12`
maximum envelope records this without weakening the ordinary-video `0.10`
gate. The retained run is `.benchmarks/runs/20260926T212849Z`.

## Realistic antialiasing and high-resolution AMG

The cars fixture exposed a translation omission hidden by smaller images:
TorchVision antialiases tensor-image downscaling by default, while Rust used
plain bilinear input sampling. The retained separable triangle-filter resize
runs over only nonzero taps on the host, normalizes in the same pass, and then
performs one device transfer. It restores practical-AMG mask/RLE parity on
cars; a correct dense-matrix prototype was rejected as far too slow.

Matched profiling then attributed roughly two seconds to transferring and
serially scanning F32 high-resolution masks. Quality filtering already
finishes on device, so the retained path thresholds there, transfers one byte
per pixel, scans masks in parallel, and skips uncropping for a full-image crop.
Cars fell from `3.64 s` to `1.69 s`, while host RSS fell from about `1.00x`
PyTorch to `0.60x`. Using F32 arithmetic in Candle's F32 CUDA bilinear kernel
reduced it further to about `1.36 s`; a hybrid arithmetic experiment changed
RLE structure and was rejected.

The promoted four-pair `A1` matrix is `1.409x` weighted latency versus the
original `2.262x`; groceries is `1.129x` and cars `1.759x`. Weighted process
VRAM is `0.571x`, peak allocated bytes `0.639x`, and host RSS `0.454x`. The
retained run is `.benchmarks/runs/20260926T220012Z`. Cars profiling still shows
the repeated decoder as the main gap: refinement is about `742 ms` versus
`471 ms`, including transformer calls around `107 ms` versus roughly `72 ms`
and a slower final transposed-convolution stage.

The corrected preprocessing refresh moves the six-fixture `I1` weighted ratio
to `1.063x`, with every row below `1.10x` and stronger output agreement. The
retained run is `.benchmarks/runs/20260926T220158Z`. A first default-grid `A2`
diagnostic has matching 52-mask structure under the faster kernel but one
threshold-boundary pixel in one RLE remains different, so it is not promoted.

## Long-video temporal storage

`V4` showed that Rust retained every 64x64 temporal feature in F32, while the
reference stores CUDA memory in BF16 and promotes only frames selected for the
current attention call. The translated state now does the same and also shares
the invariant memory-position tensor across frames. On the 128-frame focused
run, latency improves from `1.031x` to `1.016x`, peak allocated memory falls by
about 59 MiB to `1.051x`, and selected-frame maximum error improves from
`0.01860` to `0.00826`.

The promoted four-pair run (`.benchmarks/runs/20260926T231616Z`) confirms
`1.013x` latency, `1.062x` external VRAM, `1.051x` peak allocated, `1.065x`
peak reserved, and `0.187x` host RSS. Maximum selected-frame error is
`0.00660`. Every individual cap passes, but the above-one device-memory ratios
remain work for the global weighted device-memory gate.

## GPU model scaling and exact-GELU experiments

The generalized suite now implements explicit model-load (`L1`) and
Small/Base-Plus/Large scaling (`S1`/`M1`) rows. `first_and_cached` previously
used only cached-decode latency as its headline, which hid encoder scaling.
Its score is now encoder plus cached decode, with both components retained in
the JSON summary. The old `S1` `0.914x` cached-only result is superseded by the
corrected `1.275x` first-interaction result. The four-pair M1 weighted result
is `1.206x`; Base-Plus is `1.190x` and Large is `1.222x`. Cached decoding is
already faster for all three variants, and all device/host memory ratios pass,
so the remaining scaling issue is specifically Hiera image encoding.

The reusable synchronized static profiler now accepts all four model variants.
On a warmed Base-Plus pass, it measured about `183 ms` Rust versus `153 ms`
PyTorch. The largest repeated gap was the two-layer Hiera MLP, particularly
exact GELU between its projections; Rust attention was often comparable or
faster, including the global-attention blocks.

Two backend experiments were rejected. Giving Candle's exact F32 CUDA GELU
four values per thread regressed the profiled pass from about `183` to
`202 ms`; the existing kernel was restored. Fusing bias and GELU through the
cuBLASLt GELU epilogue reduced one profiled Base-Plus pass to about `172 ms`,
but it is not numerically the same exact GELU: full-model maximum error rose to
`0.0368` for Base-Plus and `0.862` for Large, and Large latency regressed.
That experiment was also removed. Future work should retain exact-GELU parity
and investigate PyTorch's pointwise kernel/launch strategy or a parity-tested
exact Candle specialization rather than using the approximate cuBLASLt
epilogue.

A later audit found that the scored `first_and_cached` path timed the first-ever
encoder invocation in every fresh process. That includes one-time CUDA/cuDNN
setup and violates the plan's warmed-inference boundary; `L1` already measures
startup. Both drivers now prime the identical encoder route before the timed
encode. Four paired processes reduce Small from the superseded `1.275x` to
`1.038x` (`.benchmarks/runs/20260927T013657Z`). Base-Plus is `0.989x` and
Large is `1.042x`, for `1.015x` weighted M1 latency
(`.benchmarks/runs/20260927T013537Z`). All individual speed, memory, and parity
caps pass.

PyTorch's contiguous pointwise launcher uses aligned vectorized transactions,
so Candle's exact F32 GELU was also retested with a true aligned `float4`
load/store, one vector per thread, a 256-thread block, and a grid sized to
vectors rather than scalar elements. Four paired Base-Plus processes averaged
`136.293 ms` with the candidate and `136.026 ms` in the adjacent original-
kernel control. Output and RSS were unchanged. The corrected vector launch is
neutral on this GPU and was removed; synchronized nested profiles remain
attribution tools rather than end-to-end timing evidence.

Candle's generic CUDA unary launcher uses 1024-thread blocks. A 256-thread
exact-GELU launch was tested because Base-Plus profiling identified GELU as a
repeated Hiera MLP gap. Four paired Base-Plus processes improved Rust encoding
from `323.90` to `306.50 ms` (5.37%) and the cell from `1.229x` to `1.192x`,
with identical parity and memory. However, four paired Tiny bedroom processes
regressed Rust from `73.61` to `75.35 ms` (2.36%), changing a passing `0.997x`
cell into `1.017x`. A 512-thread test did not preserve the Base-Plus win.
Traces showed Tiny's dominant GELU has 6,291,456 elements while Base-Plus's has
7,340,032, but no general crossover was established. The launch change was
removed rather than adding a model-specific threshold. Candidate and control
runs are `.benchmarks/runs/20260926T233058Z`, `20260926T233135Z`,
`20260926T233218Z`, and `20260926T233256Z`.

The next rotation returned to `A1/cars`. A fresh synchronized profile measured
`390.96 ms` Rust versus `333.31 ms` PyTorch for image encoding, `341.57`
versus `268.49 ms` for the initial decoder, and `792.45` versus `478.11 ms`
for three refinement decodes. Rust's four transformer calls totaled about
`466.5 ms` versus `282.8 ms`; its final upconvolution/GELU stages totaled
about `146.6 ms` versus `68.9 ms` for PyTorch's matching transposed-convolution
and activation hooks. The separate Rust filter/transfer/RLE phase was
`425.65 ms`.

Allocating cuDNN convolution workspace without zero-filling was tested because
the workspace is opaque scratch. A full four-pair A1 matrix measured `1.414x`
weighted latency versus the retained `1.409x`; groceries regressed from
`1.129x` to `1.144x`, while cars changed from `1.759x` to `1.748x`. This is
noise rather than a repeatable aggregate improvement, so zeroed workspace was
restored. The next upscaler experiment should instead reduce output-sized
memory traffic by fusing bias, high-resolution residual addition, and exact
GELU after the non-overlapping transposed convolution.

A follow-up custom CUDA post-pass fused bias, the high-resolution residual,
and exact GELU after the final transposed convolution. Remapping the launch to
`(batch, channel, spatial)` removed per-element integer division, but the
fused pass still took about `53.1 ms` versus about `36.7 ms` for the retained
composed Candle operations and increased peak allocated device memory. Both
versions were removed. Any future attempt must fuse into the transposed-
convolution write or otherwise eliminate an output allocation and memory pass;
another standalone post-kernel is not justified by this evidence.

## GPU batched cached-prompt attribution

The refreshed four-pair `I4` matrix is `1.263x` weighted latency: bedroom is
`1.287x` and cars `1.240x`. Memory remains favorable (`0.765x` process VRAM,
`0.919x` peak allocated, `0.724x` peak reserved, and `0.357x` host RSS), and
output parity passes. The retained run is `.benchmarks/runs/20260926T225915Z`.

The suite now has `--profile-image-stages`, which emits matched synchronized
Rust and PyTorch prompt-encoder and mask-decoder boundaries. On warmed batch-4
decodes, Rust's transformer was about `7.25 ms` versus `6.58 ms`, first upconv
`1.28` versus `0.87 ms`, and final upconv plus exact GELU `2.10` versus about
`1.22 ms`. The correction prompt's mask encoder was `2.53` versus `1.76 ms`;
its final 1x1 projection alone was about `1.10` versus `0.20 ms`.

A CUDA GEMM translation for that 1x1 projection cached the transposed weight
but needed NCHW-to-NHWC input and output materializations. A focused matched
run regressed Rust from `26.74` to `27.66 ms` (about 3.4%); parity and memory
were unchanged. The experiment was removed. A future projection improvement
must consume or produce the existing NCHW layout directly rather than paying
two full tensor copies.

Deeper batch-64 projection telemetry is archived in
`.benchmarks/runs/20260927T014025Z`. On typical warmed calls, Rust token-to-
image QKV takes about `12.4 ms`, versus roughly `4.6 ms` summed across the
matching PyTorch projections. Image-to-token QKV is about `5.3 ms` versus
`2.4 ms`. Attention kernels and output projections are materially closer, so
the batch-scaling gap is concentrated in the large F32 linear projections.
The retained profiler now emits these matched Q/K/V/output boundaries.

Candle's fused-bias CUDA linear already collapses all leading dimensions into
one 2D cuBLASLt GEMM, so an explicit SAM-side reshape would not change the
operation. Disabling that path for ordinary cuBLAS plus a separate bias kernel
was neutral/slightly slower on A2. A second candidate initialized the final
output with bias and accumulated classic cuBLAS directly into it, avoiding
both the extra output tensor and cuBLASLt heuristic selection. It improved a
single A2 diagnostic by about 1.6%, but failed the four-pair A1 acceptance
test: weighted latency was `1.442x` (`.benchmarks/runs/20260927T014528Z`)
versus `1.435x` for its adjacent cuBLASLt control
(`.benchmarks/runs/20260927T014656Z`). The candidate was removed.

The earlier post-upconv fusion was retested with a bounded grid-stride launch,
eliminating its pathological number of blocks on large prompt batches. It
still regressed focused `I4/bedroom` from `26.35` to `27.68 ms` and increased
peak allocated device memory from `0.919x` to `0.975x` because it retained a
separate convolution output and allocated another full output tensor. It was
removed again. This closes the standalone-post-pass direction: fusion must
write the convolution destination itself to reduce traffic and lifetime.

## Generalized CUDA graph image benchmarks

The paired suite now supports CUDA graph capture/replay for `I3` and `I4` in
both Rust and PyTorch. `I3` captures three graphs for its point, rich-prompt,
and multimask routes; `I4` captures the entire initial batch plus dependent
prior-mask correction. Graph-specific warmup populates Candle's capture-safe
HtoD cache and is excluded from capture/replay timing. Capture time and graph-
persistent allocated/reserved memory remain explicit in every record.

Four-pair `I3` graph replay is `1.051x` weighted, with bedroom `1.066x` and
groceries `1.035x`; all parity and memory gates pass. Rust captures in about
`15.2 ms` versus PyTorch's `24.8 ms`. The retained run is
`.benchmarks/runs/20260926T234506Z`.

Four-pair `I4` graph replay improves the eager `1.263x` result to `1.102x`:
bedroom is `1.097x` and cars `1.108x`. Weighted peak allocated and reserved
ratios fall to `0.562x` and `0.476x`; parity passes. Rust capture averages
about `12.5 ms` versus PyTorch's `22.4 ms`. The retained failed-baseline run is
`.benchmarks/runs/20260926T234354Z`. Graph replay therefore closes much of the
batched-decoder gap but does not yet pass, and eager remains independently
release-blocking.

## Default-grid AMG batch-64 attribution

The unpromoted `A2/groceries` default 32x32-grid workload processes sixteen
batches of 64 prompts. A fresh ordinary diagnostic measured `2993.44 ms` Rust
versus `2342.89 ms` PyTorch (`1.278x`). It reproduced the known parity defect:
all 52 masks, ordering, boxes, and RLE stream lengths match, but mask 45 has
one additional foreground pixel (area 2004 versus 2003), splitting one RLE
run. Score differences are otherwise below `8e-7`.

Matched synchronized profiling attributes the GPU gap to the repeated decoder,
not image encoding or RLE construction. A typical Rust batch-64 decoder spends
about `145-148 ms` in the two-way transformer and about `69-72 ms` across its
upscaling stages. Aggregated PyTorch hooks imply roughly half the transformer
time and about 60% of the upscaler time. Profiling synchronizes many nested
boundaries and therefore its `2.220x` end-to-end ratio is diagnostic only; the
ordinary `1.278x` result is the valid timing comparison.

Three bounded experiments were rejected. Reordering Candle's F32 bilinear
expression to match PyTorch's CUDA source produced byte-identical output, so
compiler contraction erased the source-level distinction and the one-pixel
RLE difference remained. Disabling Candle's memory-efficient CUTLASS decoder
attention regressed latency to `3139.47 ms` and increased process VRAM. Using
it only for short-query/long-image attention still regressed latency to
`3081.29 ms`, showing that the retained kernel is beneficial for both
cross-attention directions. An Nsight Systems attempt was also non-evidence:
runtime injection caused CURAND initialization to fail before any kernel
launch, so no kernel timings were collected.

## GPU upscaler layout translation experiment

Matched layout telemetry now records shapes and strides at every PyTorch
upscaler module and the corresponding Rust transposed-convolution boundaries.
On `A1/groceries` batch 64, PyTorch keeps the complete upscaler channels-last:
its logical `NCHW` input stride is `[1048576, 1, 16384, 256]`, the first output
is `[1048576, 1, 8192, 64]`, and the final output is
`[2097152, 1, 8192, 32]`. The retained Rust route uses ordinary contiguous
NCHW storage throughout. The diagnostic run is
`.benchmarks/runs/20260927T001753Z`.

A full translation experiment added channels-last cuDNN backward-data output,
one-time NHWC packing of the two immutable transposed-convolution filters,
channels-last bias/residual additions, and layout-preserving LayerNorm/GELU.
The filter packs total about 288 KiB and were created once during model load,
not copied per prompt. A Candle backend guard initially re-contiguous the
packed filter; correcting that guard restored numerical agreement. The final
candidate had the same cached-decoder parity as its adjacent control
(`2.575e-5` maximum and `3.826e-6` mean error).

The candidate is rejected on this Quadro RTX 5000. `I3/groceries` regressed
from `5.4617 ms` Rust in the adjacent NCHW control to `8.5565 ms` channels-last
(`+56.7%`), while PyTorch measured about `5.06 ms` in both runs. Peak allocated
device memory also rose from `379.0 MB` to `391.6 MB`. Synchronized attribution
showed that channels-last cut LayerNorm plus GELU from about `15` to `8 ms`,
but cuDNN transposed convolutions became substantially slower and the final
logical-NCHW materialization before mask projection rose from about `1.8` to
`37 ms`. Candidate and control runs are
`.benchmarks/runs/20260927T003312Z` and
`.benchmarks/runs/20260927T003331Z`. All channels-last execution changes were
removed; the reusable layout telemetry remains.

Caching cuDNN's selected transposed-convolution algorithm per thread was also
rejected: adjacent `A1/groceries` runs measured `760.55` and `762.06 ms`, only
a `0.2%` difference. Future upscaler work should not retry legacy cuDNN NHWC or
algorithm caching. A plausible next direction is a native non-overlapping
2x2/stride-2 kernel that writes the fused bias/residual/activation destination
directly, or a mask projection that consumes channels-last strides without a
full materialization; either must first beat the retained NCHW route in a
focused A/B.

## GPU mask-projection cuDNN algorithm experiment

The correction-mask encoder's final `16 -> 256` NCHW 1x1 convolution remains
a visible I4 hotspot (`~1.04 ms` Rust versus `~0.20 ms` PyTorch at batch 4).
Candle's cuDNN heuristic selects `IMPLICIT_PRECOMP_GEMM`. Forcing
`IMPLICIT_GEMM` reduced that isolated warmed call to about `0.49 ms`; explicit
`GEMM` was flat and `DIRECT` was unsupported.

The end-to-end acceptance result did not justify a shape override. Across four
paired eager I4 processes, implicit GEMM changed Rust bedroom from `26.2184`
to `26.1407 ms` (`-0.30%`) and cars from `25.6071` to `25.5987 ms` (`-0.03%`).
Memory and output parity were unchanged. At AMG batch 64 it did not improve the
projection at all: warmed calls averaged `9.67 ms` versus the retained
heuristic's `9.56 ms`. Candidate and adjacent control I4 runs are
`.benchmarks/runs/20260927T004133Z` and
`.benchmarks/runs/20260927T004242Z`; the batch-64 diagnostic is
`.benchmarks/runs/20260927T004356Z`. The override and temporary backend logging
were removed. A future improvement needs a generally faster NCHW 1x1
implementation rather than a cuDNN algorithm special case.

## GPU in-place exact-GELU experiment

To preserve exact GELU while avoiding the approximate cuBLASLt GELU epilogue,
Candle was tested with its existing exact `ugelu_erf_f32` kernel launched
in-place over the cuBLASLt bias-epilogue destination before the tensor became
observable. The GEMM, bias addition, and exact-GELU operation order stayed the
same, and Base-Plus output errors were byte-for-byte unchanged.

The four-process result rejected the change. Base-Plus encoding regressed from
`302.6641` to `308.9826 ms` (`+2.09%`); the complete first-and-cached cell moved
from `307.2462` to `314.4189 ms`. Candle's measured peak allocated bytes were
identical (`546,821,192`), while reserved/external process readings moved in
opposite directions and therefore did not establish a memory win. Candidate
and adjacent control runs are `.benchmarks/runs/20260927T004829Z` and
`.benchmarks/runs/20260927T004907Z`. The path was removed. Further exact-GELU
work needs a faster kernel or true GEMM epilogue, not merely in-place reuse.

## GPU video lifetime phase attribution

Phase-level CUDA allocator snapshots now cover model load, video-state
initialization, prompt interaction, propagation, reset, and state drop in both
implementations. The paired V2 validation is archived at
`.benchmarks/runs/20260927T005426Z`; it measured Rust at `1.0150x` PyTorch
latency while using `0.8197x` PyTorch peak allocated bytes.

The 128-frame V4 lifetime run at `.benchmarks/runs/20260927T005516Z` localized
the remaining device-memory excess. Rust was effectively at speed parity
(`1.0007x`) and remained below PyTorch through state initialization
(`2,157,581,576` versus `2,224,907,776` peak allocated bytes). The ordering
reversed only during propagation: Rust peaked at `2,834,726,660` bytes versus
PyTorch's `2,696,824,832` (`1.0511x`), despite lower live allocation after
propagation (`2,380,928,776` versus `2,397,696,512`). After dropping the state,
Rust retained `194,647,304` allocated bytes versus PyTorch's `314,378,752`.
Therefore the next V4 memory experiment should target transient propagation
workspace or tensor lifetime, not persistent video-state storage or the model
weights. The orchestrator now aggregates these per-phase byte counts and
ratios into each applicable summary row so future changes can be compared
without manually parsing raw process records.

## GPU short-query attention tiling

Matched synchronized V2 profiling at
`.benchmarks/runs/20260927T010043Z` showed that ordinary video image encoding
is already approximately at parity (`2.20 s` Rust versus `2.16 s` PyTorch per
32-frame pass). The remaining synchronized gap was concentrated in temporal
cross-attention. Additional opt-in RoPE phase telemetry at
`.benchmarks/runs/20260927T010305Z` attributed about 97% of Rust's cross-
attention time to the attention kernel rather than projections or output
layout. Ordinary V2 and V4 remain much closer than synchronized profiling
(`1.015x` and `1.001x` respectively), so this evidence redirects future video
work without treating profiler overhead as headline latency.

The subsequent A1/cars profile at `.benchmarks/runs/20260927T010411Z`
confirmed the higher-weight AMG failure remains the repeated decoder. The
translated CUTLASS attention route used a 64-query tile even for decoder
token-to-image calls with only eight queries. A shape-derived 32-query tile—the
smallest allowed by the upstream kernel—was therefore added only for
`q_sequence <= 32` and head dimensions at most 64; long-query image and
temporal attention keep their existing configuration.

Four paired full-A1 candidate and adjacent control runs are
`.benchmarks/runs/20260927T010649Z` and
`.benchmarks/runs/20260927T010847Z`. Rust groceries improved from `778.2400`
to `759.8853 ms` (`-2.36%`) and cars from `1371.3907` to `1341.5533 ms`
(`-2.18%`). The weighted Rust/PyTorch ratio improved from `1.4643x` to
`1.4378x`; repeat spreads were low, output parity was unchanged, and all
measured host/device memory values were unchanged. The four-pair I4 regression
at `.benchmarks/runs/20260927T011052Z` was neutral-to-slightly-positive versus
the nearby original-tile control: Rust measured `26.1358 ms` bedroom and
`25.5662 ms` cars, with unchanged parity and memory. Decision: retain.

A second compile-time specialization declared that short-query kernel unable
to support dropout or attention bias, matching this inference call site but
not changing its inputs or outputs. Four groceries pairs averaged `761.7581`
ms versus the retained kernel's `759.8853 ms` (`+0.25%`), with no memory or
parity change. The flags were removed. This closes the two-experiment AMG
attention round; the next optimization round rotates to model scaling or CPU
cached decoding.

## CPU cached-decoder projection attribution

The required CPU rotation refreshed I3/bedroom with synchronized matched
module boundaries at `.benchmarks/runs/20260927T011603Z`. Rust's repeated
two-way transformer remained the dominant gap at roughly `25-34 ms` per
decode versus `8-9 ms` in PyTorch; prompt construction and dense positional-
encoding lookup are negligible. The reusable decoder profile measured a
`19.76 ms` Rust transformer, including about `6.1 ms` in five Q/K/V projection
groups, `3.8 ms` in their attention kernels, `2.6 ms` in positional additions,
and `1.4 ms` in output projections.

Python profiling now records Q, K, V, and output projection modules separately
for every self-, token-to-image, image-to-token, and final attention. The
paired diagnostic at `.benchmarks/runs/20260927T011917Z` confirms that no one
projection is an anomalous translation error: PyTorch is consistently faster
across the large-row image projections, while small token projections are a
minor cost. This directs future work toward reusable immutable-weight packing
or the portable GEMM backend, rather than adding another SAM2-specific tensor
copy.

Candle's optional MKL feature was tested as a separately labelled attribution
bound, not as a proposed default. On I3/bedroom it regressed Rust from the
current roughly `49-53 ms` range to `84.97 ms`, raised RSS, and slightly
worsened numerical agreement (`.benchmarks/runs/20260927T011752Z`). This
confirms the existing MKL integration does not accelerate the lower-copy fused
Linear route and should remain disabled.

A final bounded experiment exposed Candle's already implemented copy-free
transposed GEMM orientation through a per-Linear API and applied it only to
decoder attention projections. The focused decoder profile improved from
`34.84` to `33.59 ms`, but the required four-pair real I3 workflow did not
establish a stable cross-fixture win. Candidate
`.benchmarks/runs/20260927T012137Z` averaged `48.99/48.36 ms` Rust on
bedroom/groceries; adjacent control
`.benchmarks/runs/20260927T012254Z` averaged `48.81/49.21 ms`. The fixtures
split and the combined Rust change was only `-0.68%`, within process variation;
parity and RSS were unchanged. The SAM2 dispatch and new Candle API were
removed. The detailed Python projection hooks are retained as diagnostic
infrastructure. Next workload family: GPU model scaling.

## GPU direct NCHW correction-mask projection

The prompt encoder's final correction-mask projection is a fixed F32 NCHW
`16→256` 1x1 convolution. The earlier GEMM translation lost because it copied
NCHW to NHWC and back. A retained CUDA kernel now assigns one output element
per thread, loops over the 16 input channels, reads and writes the existing
contiguous NCHW layout directly, and allocates only its required destination.
The decision is shape-derived from the 1x1 convolution itself and has an
opt-out control (`SAM2_CUDA_MASK_CONV1X1=0`).

Four paired I4 processes (`.benchmarks/runs/20260927T015504Z`) measure `1.184x`
weighted latency versus `1.191x` in the adjacent cuDNN control
(`.benchmarks/runs/20260927T015618Z`). Rust improves by about 1% on both image
fixtures with identical memory and passing parity. More importantly, the same
kernel improves four-pair A1 (`.benchmarks/runs/20260927T015732Z`) from the
adjacent `1.435x` control to `1.420x`: Rust groceries falls from `762.52` to
`755.60 ms`, and cars from `1333.89` to `1320.85 ms`. Memory is unchanged and
cars maximum output error improves from `3.737e-5` to `1.788e-7`. Decision:
retain as the default CUDA correction-mask projection.

## GPU fused final 2x2 upsampler

The decoder's two transposed convolutions use 2x2 kernels with stride 2, so
each output has exactly one contributing input position and there is no output
overlap. A direct contiguous-NCHW CUDA kernel was tested for each stage. It
also consumes a batch-one high-resolution residual by zero-stride broadcast,
adds bias, and can apply the exact GELU in the same launch; no expansion or
layout copy is required.

Using the kernel for both stages regressed the full four-pair A1 workload to
`1.500x` (`.benchmarks/runs/20260927T020509Z`). Isolated diagnostics showed
that the first `256→64` stage was the loss: first-only groceries was `1.290x`,
whereas final `64→32` stage-only groceries was `1.126x`. The retained dispatch
therefore leaves the first stage on cuDNN and uses the fused direct kernel only
for the final stage. `SAM2_CUDA_FINAL_TRANSPOSE2X2=0` is the diagnostic opt-out.

The four-pair A1 candidate (`.benchmarks/runs/20260927T020651Z`) improves
weighted latency from `1.420x` to `1.389x`. Rust groceries falls from about
`755.60` to `724.68 ms`; cars falls from `1320.85` to `1317.04 ms`. Weighted
external VRAM improves from `0.571x` to `0.484x`, and peak allocated memory
from `0.639x` to `0.598x`, because the fused path avoids large cuDNN workspace
and intermediate allocations. Maximum output error is `1.788e-7`.

The win generalizes beyond AMG. Four-pair eager I4
(`.benchmarks/runs/20260927T021218Z`) improves from `1.184x` to `1.163x`, while
four-pair I3 (`.benchmarks/runs/20260927T021035Z`) remains faster than PyTorch
at `0.973x`. CUDA-graph I4 (`.benchmarks/runs/20260927T021447Z`) improves from
`1.102x` to `1.066x`; capture time, parity, and memory gates pass. Decision:
retain the final-stage kernel across batch sizes and reject the first-stage
route.

## CPU decoder packing and external-BLAS bounds

After the GPU upsampler round, work rotated to the failing CPU I3 decoder.
On a quiet pinned bedroom diagnostic, the retained Rust GEMM measured
`50.57 ms` (`.benchmarks/runs/20260927T021950Z`). Raising the RHS-packing
threshold to 4096, which suppresses repeated weight packing for the relevant
large-row projections, regressed Rust to `51.54 ms`
(`.benchmarks/runs/20260927T022006Z`). Ordinary transposed weights therefore
cannot substitute for the backend's packed microkernel format. A persistent
packed-RHS API could recover packing cost, but profiles cap that opportunity
well below the full `2.6x` decoder gap, so it is deferred rather than driving
a large Candle API change prematurely.

A temporary, opt-in OpenBLAS implementation of Candle's fused F32 Linear was
also measured as a backend upper-bound experiment. With 20 OpenBLAS threads,
I3/bedroom regressed to `90.35 ms`; with one OpenBLAS thread it was
`85.75 ms` (`.benchmarks/runs/20260927T022111Z` and
`.benchmarks/runs/20260927T022134Z`). Both increased RSS and exceeded the CPU
mean-error envelope. The feature, FFI, and link dependency were removed.
Decision: retain the pure-Rust shape-specific GEMM path and next target shared
decoder projection/layout overhead rather than a generic external BLAS.

## GPU AMG packed mask transfer

Matched A1/cars profiling (`.benchmarks/runs/20260927T022352Z`) added Python
boundaries around stability, box extraction, crop-edge filtering, uncropping,
RLE, and NMS. Per measured iteration these total about `97 ms`; Rust's earlier
combined transfer/filter/RLE boundary was about `300 ms`. A split Rust profile
(`.benchmarks/runs/20260927T022751Z`) localized `231 ms` to thresholding and
transferring the selected full-resolution byte masks, versus `58 ms` for the
parallel CPU metadata/RLE scan. The cars case moved roughly 142 MB per batch.
Upstream instead finds transitions on-device and transfers compact run data.

The retained CUDA path now thresholds 32 pixels per output word and transfers
packed `u32` bits. Batch-one residual or output expansion is not involved, and
the CPU metadata scanner consumes packed bits directly without a full-size
unpack allocation. `SAM2_CUDA_PACK_AMG_MASKS=0` keeps the byte-transfer control.
The focused profile (`.benchmarks/runs/20260927T023046Z`) reduces measured
threshold/transfer from `231` to `25 ms`, total transfer+RLE from `289` to
`95 ms`, and peak host RSS from roughly 980 to 700 MB, with identical RLEs.

The four-pair promotion (`.benchmarks/runs/20260927T023114Z`) improves A1
weighted latency from `1.389x` to `1.265x`. Cars Rust latency falls from
`1317.04` to `1087.38 ms` (`-17.4%`); groceries is neutral at `723.31 ms`.
Weighted host RSS improves from `0.452x` to `0.385x`; external VRAM remains
`0.484x`, peak allocated device memory remains `0.598x`, repeat spreads are
low, and strict mask/metadata/RLE parity passes. Decision: retain.

A follow-up removed the CPU scanner's full `Vec<bool>` expansion and second
mask traversal. It now constructs COCO column-major runs and the tight box in
one pass over packed words. Focused metadata/RLE time falls from `69` to
`40 ms`, and profiled host HWM falls by another roughly 120 MB
(`.benchmarks/runs/20260927T023430Z`). The final four-pair result
(`.benchmarks/runs/20260927T023457Z`) is `1.246x` weighted: groceries is
`1.124x`, cars is `1.381x`, and cars Rust time is `1066.38 ms`. Weighted host
RSS is now `0.345x`; device-memory ratios and exact output parity are
unchanged. Decision: retain the single-pass scanner.

The same round tested a batched cuBLASLt K/V projection using a zero input
batch stride to remove one launch per attention. A1/groceries candidate
`.benchmarks/runs/20260927T022630Z` measured `719.13 ms`, but the adjacent
disabled control `.benchmarks/runs/20260927T022649Z` was faster at `716.94 ms`.
Memory and parity were identical, so the prototype was removed.

## GPU default-AMG decoder and CUDA LayerNorm attribution

The first complete four-pair `A2/groceries` baseline is
`.benchmarks/runs/20260927T024156Z`. Rust averages `2889.54 ms` versus
PyTorch `2336.83 ms` (`1.237x`). Host RSS is `0.335x`, external peak VRAM is
`0.684x`, peak allocated bytes are `0.894x`, and peak reserved bytes are
`0.673x`; all memory gates pass. Strict output parity fails at exactly one
threshold-boundary pixel: mask 45 has identical order, point, box, and RLE
length, but Rust area is 2004 versus Python 2003. The disputed restored logit
is `+7.23e-6` versus `-1.75e-5`. Matched low-resolution values differ before
bilinear restoration by roughly `2e-5` to `4e-5`, while both interpolation
implementations contribute the same rounding offset. The residual mismatch is
therefore decoder arithmetic drift, not a resize or threshold translation
error.

Synchronized attribution (`.benchmarks/runs/20260927T024936Z` and
`.benchmarks/runs/20260927T025107Z`) identifies repeated two-way decoding as
the latency gap. Packed threshold transfer and single-pass RLE are already
faster than the corresponding Python postprocessing. Within Rust's decoder,
the 256-wide large-row residual/normalization stages dominate; the
256→2048→256 token MLP contributes only about 20 ms across the full image.

Two generic Candle CUDA LayerNorm launch experiments were rejected. Packing
eight independent rows into one block produced `CUDA_ERROR_INVALID_VALUE` on
the target Quadro RTX 5000 and was removed before timing. A safe alternative
used one block per row with 256 threads instead of one 32-thread warp. Its
four-pair run, `.benchmarks/runs/20260927T025609Z`, regressed Rust to
`2913.37 ms` and the ratio to `1.248x`, versus `2889.54 ms` and `1.237x` in
the retained baseline. The Candle kernel was restored exactly. Decision:
reject both launch-only changes; a future round must reduce or fuse the large
residual/normalization data flow rather than merely widen its reduction.
Next workload family: another failing GPU P0 scenario.

## First unified GPU P0 eager scorecard

The first complete same-revision four-pair GPU P0 eager run is
`.benchmarks/runs/20260927T025949Z`. Weighted latency is `0.9981x`, weighted
external peak VRAM `0.764x`, peak allocated device bytes `0.792x`, peak
reserved bytes `0.691x`, and host RSS `0.313x`. Aggregate speed and memory
therefore pass, but individual latency caps still fail for `V1` (`1.110x`),
`I4` (`1.162x`), `A1` (`1.266x`; cars `1.395x`), and `A2` (`1.269x`). `A2`
also retains the known one-pixel strict-parity failure. The largest weighted
excess scores are A1 `2.127`, I4 `1.294`, A2 `1.076`, and V1 `0.552`; these
rather than the favorable aggregate determine the next optimization order.

The run exposed a scorecard-only noise bug for scaling cells. `S1` was scored
as encoder plus cached decode but its repeat spread was calculated from the
cached subphase alone. That 4–6 ms subphase produced a misleading `1.347x`
Rust spread. The actual scored per-process totals are `93.70`, `84.92`,
`86.37`, and `84.91 ms`, a passing `1.104x` spread; Python is `1.007x`.
`run_parity_suite.py` now uses the scored totals for both latency and noise and
reports cached-only process samples separately. Decision: retain the harness
fix. The next GPU optimization round rotates away from AMG to `V1`, the only
failing video cell and the next distinct workload family.

## Contiguous streamed video initialization

V1 attribution found a translation-level difference in `forward_batch`.
PyTorch builds one contiguous frame batch and performs one device allocation;
Rust normalized, allocated, and uploaded every frame independently. A first
candidate built one full normalized host batch and uploaded it once. Four GPU
pairs improved V1 from `382.38` to `343.66 ms` (`1.059x`), but V4 host HWM
rose from `0.187x` to `0.511x` of PyTorch because the 128-frame staging batch
coexisted with decoded inputs. That version was refined rather than retained.

The retained implementation allocates final contiguous device storage once,
normalizes into one reused 12 MiB host-frame buffer, copies each frame directly
to its final offset, and stores per-frame narrow views over the shared device
storage. No full-video duplicate or per-frame device allocation is retained.
The four-pair V1 run `.benchmarks/runs/20260927T033655Z` averages `320.54 ms`
Rust versus `333.67 ms` PyTorch (`0.961x`), a `16.2%` Rust reduction from the
unified baseline. Host RSS returns to `0.273x`; external VRAM (`0.640x`), peak
allocated (`0.623x`), and peak reserved (`0.607x`) are unchanged. The repeat
spread passes at `1.238x`.

Real propagation on the exact retained path
(`.benchmarks/runs/20260927T033743Z`) is `0.996x` with unchanged video parity,
`0.270x` host RSS, and unchanged device-memory phase values. The four-pair CPU
V1 regression (`.benchmarks/runs/20260927T033848Z`) also passes at `1.046x`
latency and `0.441x` RSS. Replacing the unified scorecard's V1 row projects
GPU P0 latency from `0.9981x` to `0.9909x`; a complete second suite is still
required before promotion. Decision: retain. Next workload family: the
highest-weighted remaining failure, A1, after the required CPU rotation.

## Skip unused initial AMG restoration

Upstream's M2M AMG flow requests restored full-resolution masks from its first
decoder pass, stores them, and immediately deletes them because refinement
uses only the 256×256 logits. Rust originally translated that behavior
literally. The cached decoder is now factored into a raw low-resolution path;
M2M clamps those logits exactly as the public predictor does, but never creates
the unused restored tensor. Non-M2M A2 and all public prediction APIs retain
their original restoration order and semantics.

The first draft moved only the logits out of `MaskDecoderOutput`, accidentally
keeping its roughly 512 MiB upscaled embedding alive across refinement. It
improved speed but raised peak memory and was not retained. The final code
explicitly drops the embedding, hypernetwork input, score tensors, and token
outputs before refinement. Four paired A1 runs at
`.benchmarks/runs/20260927T034710Z` improve weighted latency from the unified
`1.266x` to `1.215x`. Groceries is `725.91` versus `644.17 ms` (`1.127x`);
cars is `1010.98` versus `771.79 ms` (`1.310x`), down from `1077.17 ms` in the
unified run. Weighted external VRAM improves from about `0.484x` to `0.475x`,
peak allocated memory from `0.598x` to `0.586x`, and host RSS remains
`0.345x`. All RLEs, metadata, ordering, and float parity remain exact.

A one-pair 20-worker CPU A1/groceries regression check
(`.benchmarks/runs/20260927T034851Z`) also preserves parity and records the
current cross-device gap: `28232.59` versus `10136.56 ms` (`2.785x`) and
`0.889x` RSS. This is diagnostic rather than a four-pair CPU promotion, but it
confirms the optimized shared control flow works on CPU and identifies a major
future CPU workload. Decision: retain. With the retained V1 and A1 changes,
the projected GPU P0 aggregate is `0.9876x`; individual A1, A2, and I4 speed
gates still fail.

## Early M2M IoU filtering and zero-copy decoder Q/K/V

The next A1 profile showed that the three 64-candidate refinement batches kept
only 46, 44, and 53 masks after the predicted-IoU gate. Rust nevertheless
restored all 192 native 256×256 masks to the crop size before applying that
score. The retained M2M-only path now transfers the small IoU vector first,
selects the 143 survivors on-device, restores only those survivors, and then
computes the unchanged full-resolution stability score. Non-M2M A2 and the
public predictor retain their existing order. Four paired A1 runs at
`.benchmarks/runs/20260927T035413Z` improve weighted latency from `1.215x` to
`1.151x`; all masks, scores, metadata, ordering, and RLEs remain exact.

The remaining synchronized cars profile exposed a more general translation
copy in two-way decoder attention. cuBLASLt already writes projected Q/K/V as
contiguous `B×S×(H·D)`. Rust reshaped and materialized those tensors as
contiguous `B×H×S×D`, then immediately permuted them back to `B×S×H×D` views
for the strided memory-efficient kernel. The retained CUDA path preserves the
projection's native sequence-major storage and passes its strides directly to
the same attention kernel. `SAM2_CUDA_DECODER_DIRECT_QKV_LAYOUT=0` restores
the old materializing route for attribution.

The four-pair A1 promotion `.benchmarks/runs/20260927T035757Z` measures
`1.083x` weighted latency, down from `1.151x`: groceries is `1.035x` and cars
is `1.134x`, with exact output parity. The complete same-binary scorecard below
measures A1 at `1.092x` (`1.044x` groceries, `1.140x` cars). The same shared
change reduces A2 from the first unified `1.269x` to `1.179x` and I4 from
`1.162x` to `1.069x`; I4 now passes both fixture caps. Decision: retain both
translation-level changes.

## Second unified GPU P0 eager scorecard

The complete four-pair post-optimization run is
`.benchmarks/runs/20260927T040220Z`. Weighted latency is `0.9548x`, external
peak VRAM `0.749x`, peak allocated device bytes `0.786x`, peak reserved bytes
`0.680x`, and host RSS `0.309x`. Rust is therefore about 4.5% faster overall
than the pinned PyTorch reference and uses less memory by every aggregate
measure. Strict output parity passes everywhere except the already-localized
single A2 threshold-boundary pixel.

Only A1/cars (`1.140x`) and A2 (`1.179x`) remain above the `1.10x` individual
latency cap. V1's mean passes comfortably at `0.925x`, but its `1.261x` Rust
process spread narrowly misses the `1.25x` noise gate. An immediate independent
four-pair rerun (`.benchmarks/runs/20260927T042849Z`) again favors Rust at
`0.967x` but has `1.294x` spread; the earlier accepted V1 promotion was
`0.961x` with `1.238x` spread. The scorecard reports this variance rather than
relaxing the gate. The next GPU round returns to A2 decoder arithmetic and
large-batch throughput; CPU P0 and the planned P1 portability rows remain
incomplete.

## CUDA residual LayerNorm and filter-before-restore AMG

The optimized A2 profile still showed a full `B×4096×256` residual tensor
written immediately before each image-to-token LayerNorm. The existing CPU
path already fused this pair. A retained SAM-specific CUDA kernel now performs
the same f32 add and Candle-compatible 32-thread reduction in one pass, behind
`SAM2_CUDA_DECODER_FUSED_RESIDUAL_NORM=0` for attribution. Four A2 pairs at
`.benchmarks/runs/20260927T043502Z` improved from `1.179x` to `1.149x`; A1
also improved to `1.074x` in `.benchmarks/runs/20260927T043627Z`. Tracked peak
allocation and output decisions were unchanged. The CUDA pool reserved one
additional 256 MiB block in batch-64 AMG, but remained far below PyTorch and
all memory caps passed.

The larger translation waste was full-resolution restoration before quality
filtering. Default A2 restored 192 masks per point batch although the IoU gate
kept only 6--82. Applying the same score-only selection already proven in M2M
before restoration reduced four-pair A2 latency to `0.985x` at
`.benchmarks/runs/20260927T044009Z`, with identical survivor ordering and the
same pre-existing boundary pixel.

M2M still restored every IoU survivor before calculating stability. On cars,
only 42 of 143 such masks survive stability, and each discarded float mask is
1500×2250. A retained CUDA kernel evaluates Candle's bilinear interpolation
formula directly from the native 256×256 logits and reduces only the two
integer threshold counts. Rust then restores only stability survivors. The
route is disabled when hole/sprinkle cleanup is configured and has an
attribution switch, `SAM2_CUDA_FUSED_AMG_STABILITY=0`. It does not approximate
stability: emitted scores, candidate decisions, metadata, and RLEs pass the
strict A1 comparison.

Four paired A1 promotion runs at `.benchmarks/runs/20260927T044508Z` measure
`0.968x` weighted latency, with groceries `0.982x` and cars `0.953x`. Host RSS
is `0.345x`, external VRAM `0.391x`, peak allocated bytes `0.569x`, and peak
reserved bytes `0.380x`. The complete fused path also improves A2 to `0.937x`
in `.benchmarks/runs/20260927T044637Z`. Decision: retain all three changes.

## GPU P0 speed parity publication run

The final exact-binary four-pair scorecard is
`.benchmarks/runs/20260927T045104Z`. Weighted latency is `0.9360x`, host RSS
`0.309x`, external VRAM `0.745x`, peak allocated device bytes `0.786x`, and
peak reserved bytes `0.677x`. Every cell and fixture meets the individual
`1.10x` latency cap; the slowest is I4/cars at `1.074x`. A1 is `0.978x`
geometrically across its fixtures and A2 is `0.948x`. Thus GPU P0 speed and
memory parity are established on the realistic matrix.

The remaining GPU qualifications are measurement/correctness rather than
speed. V1 averages `0.987x` but has `1.308x` Rust process spread, above the
`1.25x` noise gate; multiple independent four-pair runs agree on a favorable
mean, including one accepted `0.961x` run with `1.238x` spread. A2 retains one
known threshold-boundary pixel caused by decoder arithmetic drift. CPU P0 and
P1 portability work remain open, so this is not a whole-plan completion claim.

## Current CPU I1/I2 breadth and feature attribution

After closing GPU P0 speed parity, work rotated to the incomplete CPU matrix.
Pinned 20-worker I1 across all six frozen shapes
(`.benchmarks/runs/20260927T051931Z`) measures `0.943x` weighted latency and
`0.360x` RSS. Every fixture mean passes, and the old portrait outlier does not
recur, but groceries has `1.466x` PyTorch spread and fails the noise gate.

Pinned I2 (`.benchmarks/runs/20260927T052356Z`) measures `0.928x` latency and
`0.350x` RSS. Bedroom and portrait are noisy; bedroom's mean logit error is
`3.645e-5` against the `3e-5` envelope, with zero threshold mismatches. The
diagnostic archive override was used only to collect the complete matrix.

Two one-pair controls reject the hypothesis that the optimized CPU features
cause this small drift. Without CPU flash attention, bedroom regresses to
`2.355x` latency, `0.730x` RSS, and `3.948e-5` mean error. With flash but
without glibc vector math, it is `1.262x`, `0.340x` RSS, and `4.729e-5` mean
error. Both default features improve numerical agreement as well as speed, so
they remain enabled. Next workload family: CPU I4 breadth, followed by the
known I3 batch-one decoder/GEMM bottleneck.

## CPU I4 boundary profile and rejected batched-attention route

The first pinned four-pair CPU I4 matrix is
`.benchmarks/runs/20260927T052804Z`: weighted latency is `3.624x` and RSS is
`0.427x`. Bedroom is `3.766x` and cars is `3.488x`; repeat spreads and output
parity pass. This supersedes I3 as the largest measured CPU image-path speed
failure.

Matched synchronized boundaries in `.benchmarks/runs/20260927T053112Z`
cover 44 decoder invocations. Rust spends `7366 ms` in the transformer versus
`1915 ms` in PyTorch, and roughly `2178 ms` in the two transposed-convolution,
normalization, and GELU upscaler stages versus `402 ms`. Prompt encoding is
small. Within transformer attention, B=4 bypasses the existing B=1 direct-NHSD
CPU flash route and uses materialized generic attention.

Change: allow B=4 direct NHSD attention and extend Candle's asymmetric
short-query/4096-key GEMM dispatch to independent `(batch, head)` tasks.
Hypothesis and evidence: the implementation already accepted batched NHSD
views but its fast asymmetric gate required B=1. Focused kernels fell from
roughly `18-21 ms` to `1.4-2.9 ms`.
Affected cells: CPU I4; likely other large prompt batches.
Focused A/B: four pairs at `.benchmarks/runs/20260927T054213Z` improved
weighted latency from `3.624x` to `2.481x`.
Memory: RSS worsened from `0.427x` to `0.733x`.
Parity: cars mean/max error worsened from `2.67e-5`/`1.81e-4` to
`3.97e-5`/`2.29e-4`, exceeding both fixed gates. A bounded two-pass AVX route
was numerically worse (`7.61e-5` mean, `4.35e-4` max). The original online
route and layout-copy eliminations did not produce a valid end-to-end win.
Decision: reject all batched-attention and neutral copy/fusion changes; retain
the parity-preserving implementation. Do not relax the envelope.
Next workload family: rotate to CPU video/AMG breadth before returning to a
numerically equivalent packed-RHS or upscaler primitive.

## CPU V2 native depthwise convolution and bounded temporal attention

The first ordinary current V2 pair (`.benchmarks/runs/20260927T055750Z`) was
`3.468x` PyTorch latency and `1.026x` RSS. Synchronized Rust stages measured
about 2.3 seconds per memory encoder call; each of its two 256-channel 7x7
depthwise convolutions took roughly 1.0--1.1 seconds. Candle's generic grouped
path created 256 independent one-channel convolution tensors and concatenated
their results. The linked fork now dispatches CPU-f32 `groups == Cin == Cout`
to one direct, parallel depthwise kernel. It allocates only the final output,
uses existing strides, contains no unsafe code, and has a manual-reference
multi-batch test. Individual convolutions fell to roughly 15--23 ms and the
memory encoder to about 0.2 seconds per frame. The ordinary pair
`.benchmarks/runs/20260927T061844Z` improved V2 to `3.154x` with unchanged
parity; attention still kept HWM near 3 GB.

RoPE temporal attention was then routed through Candle's bounded online
softmax kernel using stride-only BSHD views. It removes the O(Q*K) score and
softmax tensors while emitting compact BSHD output. The profile fell from
141.0 to 77.0 seconds, temporal/head work from 123.0 to 58.4 seconds, and HWM
from 2.96 to 1.60 GB. The ordinary pair
`.benchmarks/runs/20260927T062728Z` measured `1.645x` latency and `0.576x` RSS;
mean/max video error remained `0.004044`/`0.06434`.

The remaining RoPE expression still built index tensors and multiple complete
intermediates for every Q and K. A CPU-f32 specialization now applies each
even/odd pair in one Rayon pass and one output allocation. Self-attention RoPE
fell from roughly 25--45 to 1.6--2.5 ms; full-memory cross RoPE fell from
105--125 to 9--15 ms. The profile reached 62.3 seconds. One ordinary pair at
`.benchmarks/runs/20260927T063729Z` measured `0.982x` latency and `0.569x` RSS
with identical error to the flash-only path. PyTorch was unusually slow in
that pair, so the change is retained on strong stage evidence, but the V2
headline awaits four quiet pairs.

## CPU I4 follow-up rejections

A proposed batched transposed-convolution schedule packed all batch rows into
one GEMM. Although a synchronized batch-4 upscaler substage fell from about
46--50 to 30--34 ms, four alternating 20-iteration Rust processes averaged
`717.05` ms for the candidate versus `678.12` ms for the existing per-batch
route, a 5.7% regression. It was removed.

Selective batched flash attention also failed promotion. Token-to-image alone
worsened cars parity to `5.95e-5` mean and `3.74e-4` maximum error. Image-to-
token alone stayed within the envelope, but its four-pair matrix
(`.benchmarks/runs/20260927T065738Z`) was `3.814x`, worse than the retained
`3.624x` baseline, with no stable Rust gain. Both selective branches were
removed. I4 therefore still requires parity-equivalent projection/GEMM work.

## CPU A1 batch-64 attention and upscaler scheduling

The matched A1/groceries profile
`.benchmarks/runs/20260927T070137Z` measured `2.893x` and showed that the image
encoder was already tied (0.73 versus 0.75 seconds). Four batch-64 decoder
calls dominated: Rust transformer time was 17.5 seconds versus PyTorch's 4.66,
and Rust layout/upscaling was about 6.4 seconds versus 1.6. The B>=32-only
batched attention route reduced an ordinary Rust pass from 28.16 to 24.05
seconds (`.benchmarks/runs/20260927T070314Z`) without affecting I3/I4 or
changing strict AMG output decisions.

For the 2x2 transposed convolutions, running 64 full-Rayon GEMMs serially paid
the scheduler barrier 64 times. The linked Candle path now gives independent
batches to Rayon and uses `Parallelism::None` inside each GEMM. Each worker
reuses one required 4--8 MiB scatter tile; inputs and weights remain shared,
so there is no model/input copy per thread. The focused combined pair
`.benchmarks/runs/20260927T070717Z` reduced Rust to 20.98 seconds, used
`0.934x` PyTorch RSS, and preserved strict parity.

The four-pair/two-fixture promotion at
`.benchmarks/runs/20260927T070842Z` is `2.209x` weighted latency and `0.666x`
weighted RSS, but shared-host contention makes both rows noisy (process
spreads `1.396--1.958x`). The mean improvement, focused pair, and synchronized
stages agree, so both changes are retained behind
`SAM2_CPU_AMG_BATCHED_FLASH=0` and `CANDLE_CONVTRANSPOSE_PAR_BATCH=0` opt-outs.
The noisy matrix is not a parity headline and must be repeated quietly.

## GPU A2 threshold-boundary operand attribution

The reusable `--profile-a2-boundary` parity-suite diagnostic now archives the
matched Rust/PyTorch decoder operands for the one disputed A2 pixel. It implies
synchronized AMG/image-stage profiling, is restricted to `--cell A2`, and its
timing is therefore diagnostic rather than headline evidence.

The matched run at `.benchmarks/runs/20260927T073321Z` decomposes the four
native `256x256` logits contributing to the restored boundary pixel. The
32-value hypernetwork operand differs by at most `3.48e-5` (mean `1.33e-5`),
while the corresponding upscaled feature values differ by at most `6.13e-7`
(mean `1.65e-7`). Reconstructing the four dot products in f64 reproduces the
observed direction and magnitude. The final mask-projection GEMM and bilinear
restoration are therefore not the source; most drift enters through the mask
token/hypernetwork side.

The selected 256-value mask token differs by at most `1.14e-5`. Feeding the
captured Rust and Python tokens through the same PyTorch CUDA hypernetwork
explains essentially all of the `3.48e-5` hypernetwork difference, so a
separate MLP translation defect is not indicated. A matched stage-boundary run
at `.benchmarks/runs/20260927T073705Z` localizes the amplification further:
the transformer input matches within `7.24e-9`, layer 0 within `1.20e-6`, layer
1 within `2.06e-6`, and the final token-to-image attention plus LayerNorm
within `1.15e-5`. Future A2 correctness work should therefore start at that
final attention boundary rather than changing thresholding, interpolation,
upscaling, or RLE construction.

One bounded hypothesis was rejected. Disabling the retained CUDA fused
residual LayerNorm in `.benchmarks/runs/20260927T073523Z` produced bit-identical
Rust mask-token and hypernetwork vectors and the same output mismatch, while
making the synchronized diagnostic slower. The fusion remains enabled. No
epsilon or scenario-specific mask threshold was introduced: that would hide
the arithmetic difference and risk changing other near-boundary masks rather
than improving the translation.

The follow-up diagnostic at
`.benchmarks/runs/a2-final-attention-attribution` captures the selected mask
token at the final attention's Q projection, attention output, and output
projection on both implementations. Rust/PyTorch maximum differences are
`2.03e-6`, `6.23e-6`, and `3.05e-6`, respectively (mean differences
`3.71e-7`, `1.10e-6`, and `8.10e-7`). This confirms that final attention
amplifies the already-present layer-1 input drift, but does not reveal a
missing projection or layout operation.

Three correctness controls were rejected. The materialized QK/softmax/PV path
at `.benchmarks/runs/a2-generic-control` was `1.848x` PyTorch and produced
more output drift, so it is neither the reference arithmetic nor a viable
fallback. PyTorch's SM75 f32 registry contains a 64x64 CUTLASS tile; using
that tile instead of the retained 32-query specialization at
`.benchmarks/runs/a2-pytorch-tile-attribution` produced bit-identical captured
values and did not change output parity. Finally, disabling Candle's cuBLASLt
linear-bias fusion at `.benchmarks/runs/a2-linear-control` increased the Q and
attention differences and was substantially slower. The translated short-Q
CUTLASS tile and fused linear path therefore remain defaults. The remaining
boundary pixel is cumulative f32 backend-order sensitivity across the two
decoder layers; strict output thresholds remain unchanged.

## GPU benchmark-duration correction and superseding scorecard

V1 originally timed one roughly 0.35-second initialization per process. A new
four-pair run at `.benchmarks/runs/20260927T073910Z` exposed the consequence:
the favorable mean was meaningless for promotion because Rust/PyTorch process
spreads reached `2.14x`/`5.21x`. This also violated the plan's requirement for
at least two seconds of measured work per short cell.

The suite schema and orchestrator now support `gpu_iterations` independently
of CPU iterations. V1 uses eight independent initialization samples per
process; other short GPU P0 cells also receive enough device-specific samples
to accumulate roughly two seconds without inflating their CPU runs. The first
implementation accidentally looked for `cuda_iterations` while the schema
uses `gpu_iterations`, so `.benchmarks/runs/20260927T074124Z` and the adjacent
full run are superseded and must not be cited as duration-qualified evidence.
The runner now maps CUDA to the `gpu_` prefix and asserts the child-reported
warmup and iteration counts.

The corrected V1 run at `.benchmarks/runs/20260927T081323Z` has eight samples
per implementation in every process and measures `341.32 ms` Rust versus
`320.81 ms` PyTorch (`1.06393x`). Host RSS is `0.269x`, external VRAM is
`0.762x`, peak allocated device memory is `0.623x`, peak reserved memory is
`0.742x`, strict parity passes, and both process spreads pass the noise gate.

The authoritative full duration-qualified GPU P0 eager scorecard is
`.benchmarks/runs/20260927T081450Z`: weighted latency `0.94053x`, host RSS
`0.30903x`, external VRAM `0.75185x`, peak allocated memory `0.78563x`, and
peak reserved memory `0.68378x`. The aggregate and all memory gates pass, but
I1 (`1.11458x`) and I2 (`1.11812x`) fail the individual speed cap;
I3/bedroom is noisy; and A2 retains the known one-pixel strict-parity failure.
GPU speed parity is therefore not established. Matched stage profiling places
the I1/I2 gap in the Hiera image encoder, especially the first four blocks, so
that is the next primary optimization target.

## GPU I1/I2 early-Hiera attribution

The expanded matched diagnostic at `.benchmarks/runs/20260927T085549Z`
separates normalization, shortcut/pooling, partitioning, QKV, attention,
projection, cleanup, unpartitioning, and both MLP projections for every Hiera
block. On the warmed pass, Rust's first four blocks total `38.27 ms` versus
`26.81 ms` for PyTorch. Pooling is not the main gap. Block 0 is dominated by
the large-NHWC QKV and MLP work; blocks 1--3 add smaller linear, attention,
and tensor-lifetime overheads. Later global-attention blocks are close enough
that optimizing them first would not address the failing I1/I2 cells.

One broad Candle hypothesis was tested and rejected. Hiera's final MLP linear
uses `Linear::forward_add_residual`; routing its CUDA fallback through the
existing cuBLASLt fused-bias path avoids one bias allocation/kernel before the
residual add. The candidate pair at `.benchmarks/runs/20260927T085939Z` looked
promising at `74.89 ms`, but the two-process candidate run
`.benchmarks/runs/20260927T090036Z` averaged `79.46 ms` with `1.086x` Rust
spread. The same binary with the route disabled at
`.benchmarks/runs/20260927T090127Z` averaged `75.46 ms` with `1.013x` spread,
used less peak-reserved memory, and retained the prior numerical result. The
candidate was removed. Next GPU image round should target block-0 large-NHWC
linear algorithm/allocation behavior or window-partition lifetime with a
lower-level isolated benchmark, not this residual route.

Further attribution showed that warm cuBLASLt host setup is not the missing
time: after initialization, each fused linear call spends roughly `10--36 us`
on the host and the complete encoder total is below one millisecond. Correcting
the diagnostic to time CUDA's exact GELU separately showed block 0's measured
MLP kernels at about `2.75 ms` versus a `6.26 ms` module boundary, implicating
large temporary allocation/lifetime work rather than heuristic search.

Two allocation-removal experiments were rejected. cuBLASLt's bias+GELU
epilogue changed the final I1 logits by `0.0580` maximum and `0.00643` mean,
so its approximate GELU is not a valid translation of PyTorch's exact GELU.
Launching Candle's exact-erf GELU in place on the newly owned linear output
preserved the existing output exactly and reduced peak allocated device memory
from about `497 MB` to `471 MB`, but regressed the adjacent paired median from
`74.64 ms` (`.benchmarks/runs/20260927T090829Z`) to `78.97 ms`
(`.benchmarks/runs/20260927T090755Z`). Both candidates were removed.

Per the rotation rule, work moved to the noisy cached-prompt row. The
duration-qualified four-pair I3/bedroom repeat
`.benchmarks/runs/20260927T091000Z` is clean: `3.998 ms` Rust versus `4.742 ms`
PyTorch (`0.84319x`), process spreads `1.008x`/`1.054x`, strict parity, host
RSS `0.360x`, peak allocated memory `0.794x`, and peak reserved memory
`0.573x`. Replacing the noisy bedroom row in the complete scorecard makes I3
`0.83776x` and projects GPU P0 eager latency at `0.94992x`. I1/I2 remain the
only speed-cap failures.

The scheduled cross-device round established a current, quiet CPU
I3/bedroom diagnostic at `.benchmarks/runs/20260927T091322Z`. Across four
20-worker processes Rust averages `47.92 ms` per cached prompt versus
`21.05 ms` for PyTorch (`2.2768x`); spreads are `1.139x`/`1.136x` and RSS is
`0.322x`. It is not promotable because mean logit error is `3.645e-5` against
the fixed `3e-5` gate, although maximum error (`8.20e-5`) passes and there are
zero thresholded-mask mismatches. A matched boundary diagnostic at
`.benchmarks/runs/20260927T091411Z` puts the warmed Rust transformer at
`24.31 ms` versus `12.95 ms` PyTorch, with another roughly `9.69 ms` in Rust
token-head/layout work before the upscaler. This confirms a broad decoder gap
rather than a CPU RSS problem; no parity threshold was relaxed.

## GPU bounded positional-cache promotion

The image encoder previously rebuilt the same bicubic positional resize and
tiled window embedding for every normalized 1024x1024 input. The existing
cache is shape-checked and bounded to one immutable tensor; returning it clones
only a tensor handle and does not copy its roughly 24 MiB storage per call or
per thread. Earlier CPU measurements rejected making it a CPU default, so the
retained policy enables it by default only on CUDA, keeps CPU opt-in, and
supports `SAM2_STATIC_POSITION_CACHE=0` as a CUDA attribution control.

The duration-qualified four-pair I1 promotion
`.benchmarks/runs/20260927T092250Z` improves the six-fixture geometric mean
from `1.11259x` (`.091545Z`) to `1.05626x`; the worst fixture is cars at
`1.06487x`, all spreads are below `1.021x`, and parity is unchanged. Host RSS
is `0.347x`, external VRAM `0.741x`, peak allocated memory `0.831x`, and peak
reserved memory `0.696x`. The corresponding I2 promotion
`.benchmarks/runs/20260927T092731Z` is `1.05747x`, worst fixture `1.06523x`,
host RSS `0.364x`, and peak allocated memory `0.830x`, again with unchanged
parity. A classic-cuBLAS control was rejected at `1.227x` and materially higher
device memory (`.092047Z`).

Replacing I1, I2, and the clean I3 row in the complete duration-qualified
matrix projects GPU P0 eager latency at `0.94027x`, host RSS `0.30901x`,
external VRAM `0.75253x`, peak allocated memory `0.79292x`, and peak reserved
memory `0.68455x`. The worst latency row is I4/cars at `1.07702x`, and no row
is noisy. Decision: retain. Next GPU requirement: complete same-binary
confirmation and the A2 final-attention correctness mismatch.

The complete current-binary one-pair P0 regression sweep
`.benchmarks/runs/20260927T093212Z` confirms the cache is safe across image,
cached prompt, video, AMG, load, and Small-model families: weighted latency is
`0.94834x`, host RSS `0.30862x`, external VRAM `0.76222x`, peak allocated
memory `0.81140x`, and peak reserved memory `0.68985x`. Output parity is
unchanged, including the already-known A2 boundary pixel. The only apparent
speed-cap failure was its single V1 sample at `1.183x`; the immediate
duration-qualified four-pair repeat `.benchmarks/runs/20260927T094022Z` is
clean at `1.05902x`, with Rust/PyTorch spreads `1.097x`/`1.087x`. Replacing
that outlier gives a current-binary smoke score of `0.94288x`, with I4/cars
worst at `1.07729x`. This validates retention; the required full four-pair
same-binary graduation run remains outstanding.

## Complete current-binary GPU confirmation

The clean release scorecard `.benchmarks/runs/20260927T095000Z` supersedes the
compatible-row projection above. It contains 96 alternating-order process
pairs: four pairs for every implemented GPU P0 row across Tiny image/cached
decode, video, AMG, load, Small, Base-Plus, and Large scenarios. Weighted
Rust/PyTorch latency is `0.94099x`; host RSS is `0.31174x`, external peak VRAM
is `0.76589x`, peak allocated device memory is `0.81540x`, and peak reserved
memory is `0.68771x`. All rows pass the repeat-spread gate.

The worst latency row is I4/cars at `1.07961x`. The worst peak-allocation row
is V4/bedroom128 at `1.06047x`; it remains below the `1.10x` cell cap, while
V4 latency is `0.99760x`. Per-family geometric latency ratios are I1 `1.05389x`,
I2 `1.06113x`, I3 `0.87378x`, I4 `1.06012x`, V1 `1.01592x`, V2 `1.00077x`,
V3 `1.01435x`, V4 `0.99760x`, A1 `1.00335x`, A2 `0.95316x`, L1 `0.14613x`,
S1 `0.95217x`, and M1 `1.00497x`. Output parity is unchanged: A2's one
threshold-boundary pixel is the only strict failure. Decision: promote this
run as the current GPU P0 scorecard. A second independent complete run is
still required for graduation.

## Rejected CPU decoder-head fused-bias route

After the complete GPU confirmation, the rotation rule returned to CPU I3.
The four hypernetwork MLPs plus object-score and IoU heads were routed through
Candle's retained CPU fused-bias Linear destination, guarded by an attribution
switch. Four-pair I3/bedroom candidate and control runs are archived at
`.benchmarks/runs/cpu-i3-head-fused-candidate2` and
`.benchmarks/runs/cpu-i3-head-fused-control`. Rust medians were `48.96 ms` and
`48.90 ms`, respectively, so the candidate was neutral; it also left the
known `3.645e-5` mean-logit error unchanged. Decision: reject and remove. The
CPU decoder transformer and its large layout/attention regions remain the
evidence-backed target, rather than the small output heads.

## CPU interactive-video deferred memory encoding

Change: match upstream's interactive-video memory-encoder lifetime. A click or
mask input now produces a temporary mask/pointer output; temporal memory is
encoded once during propagation preflight rather than once during the click
and again in preflight.

Hypothesis and PyTorch/Candle evidence: upstream calls single-frame inference
with `run_mem_encoder=False` from `add_new_points_or_box`, explicitly deferring
memory work until temporary outputs are consolidated. Matched R1 diagnostics
at `.benchmarks/runs/r1-cpu-lifetime-profile` measured initialization at
`832.43/834.96 ms` Rust/PyTorch and reset/drop at `50.90/41.88 ms`, but prompt
handling at `255.30/22.90 ms`. Rust's prompt profile included roughly 170 ms of
memory encoding that upstream did not execute there.

Affected suite cells: R1 video interactions directly; V2-V4 and V3 preflight
retain the required memory encoding and avoid its former duplicate.

Focused A/B: the earlier unprofiled R1 diagnostic was `1.034x`; the corrected
one-pair run `.benchmarks/runs/r1-cpu-deferred-memory-smoke` is `0.944x`
(`8440.97/8943.57 ms`). A full V2 propagation check at
`.benchmarks/runs/v2-cpu-deferred-memory-smoke` keeps Rust work at `55060.24
ms`, close to the prior `55298.17 ms`, confirming that required propagation
work was moved rather than skipped.

Peak/current RSS: R1 HWM is `0.501x` PyTorch; session-growth parity remains
bounded. V2 HWM is `0.589x`.

Parity result: R1 max/mean error remains `0.000702/0.000102` with zero
threshold mismatches; V2 remains `0.06434/0.004044`, matching the established
video envelope.

Decision: retain. This corrects the translation, removes redundant work and
allocations, and does not alter propagation output. Next workload family:
complete the four-pair R1 promotion, then return to the failing cached image
decoder or CPU AMG according to weighted impact.

The four-pair follow-up
`.benchmarks/runs/20260927T112000Z-r1-cpu-deferred-memory` measures `0.960x`
weighted latency and `0.410x` RSS. Rust repeat spreads pass (`1.208x` image,
`1.070x` video), but PyTorch spreads fail (`1.346x`, `1.286x`), so the run is
retained as corroborating diagnostic evidence rather than promoted. Video R1
is now `1.024x`, versus `1.462x` in the superseded pre-fix four-pair run.

## CPU decoder zero-copy transformer-to-upscaler layout

Change: preserve the transformer's contiguous `B×(H·W)×C` storage as a BHWC
view and expose it to the first upscaler as a strided NCHW view. The linked
Candle 2×2/stride-2 fused transposed convolution now detects this exact
pixel-major stride pattern and passes its storage directly to GEMM. It does
not allocate or copy the former 4 MiB NCHW staging tensor per decoder call.
The route is CPU-only and has `SAM2_CPU_DECODER_ZERO_COPY_UPCONV=0` as an A/B
control.

Hypothesis and PyTorch/Candle evidence: PyTorch's decoder likewise transposes
the contiguous token tensor and views it as BCHW; its convolution backend can
consume that layout. Candle's `reshape` materialized every non-contiguous
input even though its fused GEMM immediately repacked it back to pixel-major.
The existing Candle residual-transposed-convolution test constructs this same
stride pattern and passes against the composed implementation after the
change.

Affected suite cells: I3/I4, image portions of V3, and A1/A2 decoder batches.

Focused A/B: four-pair I3 control
`.benchmarks/runs/i3-zero-copy-control` averages `49.42 ms` Rust; candidate
`.benchmarks/runs/i3-zero-copy-candidate` averages `37.27 ms`, a `24.6%`
reduction. The ratio improves from `2.371x` to `1.669x`; both Rust spreads pass.
I4 candidate Rust spreads pass on both fixtures and retains fixed parity,
although cars' PyTorch spread is noisy. A1/groceries
`.benchmarks/runs/a1-zero-copy-smoke` is `19.57 s` Rust versus the prior
focused `20.98 s`, and strict AMG output parity passes.

Peak/current RSS: I3 remains about `0.319x` PyTorch; I4 is `0.409x` weighted.
The A1 smoke is `0.872x`, still above the P0 weighted RSS target when treated
alone but below PyTorch.

Parity result: I3 candidate and control are identical (`8.202e-5` max,
`3.645e-5` mean); I4 stays within its fixed fixture envelopes; A1 strict mask,
metadata, ordering, and RLE parity passes.

Decision: retain in sam2-rs and the linked Candle fork. It is a reusable layout
capability, removes memory traffic rather than adding a cache or worker copy,
and improves multiple workload families. Next workload family: implement the
missing CPU P1 prompt-diversity/batch coverage or return to the transformer
projection gap after rotation.

## CPU P1 prompt-diversity baseline

The newly implemented P1 cell covers empty, 1/8/32-point, box-only,
prior-mask, and multimask decoder routes against identical cached image
features. Its concatenated parity artifact contains every returned logit mask.
The four-pair run `.benchmarks/runs/20260927T114500Z-p1-prompt-diversity` is
`1.938x` latency and `0.315x` RSS, with max/mean error
`1.469e-4`/`2.002e-5`. Rust spread passes at `1.054x`; PyTorch's `1.270x`
spread narrowly fails the `1.25x` noise gate. This is a usable hotspot
baseline but not a promoted P1 score. It confirms the residual decoder deficit
is broad rather than a one-point-only artifact.

## CPU B1 batched-image dispatch

Change: implement the mixed-image B1 runners, correct Hiera's batched-NHSD
dispatch, and generalize Candle's bounded blocked long-attention kernel across
the leading batch stride. The input tensors are concatenated once before
timing; Q/K/V are read through existing strides and are not repacked or copied.

Hypothesis and PyTorch/Candle evidence: the single-image-optimized Rust branch
tested sequence length before batch size, so a true image batch first reached
the batch-one entry point and failed. After correcting that, matched B1 stage
profiling showed global-attention kernel times of `1105`, `945`, and `1082 ms`:
batch greater than one bypassed the blocked GEMM route and used row-wise online
softmax. PyTorch's entire matched encoder was only `1777 ms`. The generalized
kernel assigns unique `(batch, head, query tile)` output regions to Rayon and
retains its bounded worker-local tiles, avoiding nested parallelism and all
Q/K/V copies.

Affected suite cells: B1 directly; ordinary batch-one image/video paths retain
their existing entry point.

Focused result: batch 2 improves from `4188` to `1581 ms` (`-62%`) and its
encoder from `4007` to `1407 ms`. Batch 4 improves from `12248` to `2882 ms`
(`-76.5%`). A new Candle batch-2/S=768 regression test compares both batch
entries with the tensor-expression reference and passes.

Four-pair result: the clean batch-2 repeat
`.benchmarks/runs/20260927T125300Z-b1-batch2-repeat` is `0.844x` latency and
`0.514x` RSS. Batch 4 in
`.benchmarks/runs/20260927T125000Z-b1-blocked-batched` is clean at `0.758x`
latency and `0.607x` RSS. The batch-4 encoder is `2326` versus `3345 ms`; its
decoder remains `245` versus `83 ms`, identifying the next B1-local gap without
preventing end-to-end parity.

Parity result: both four-pair rows pass. Batch 2 max/mean error is
`5.722e-5`/`1.875e-5`; batch 4 is `9.155e-5`/`1.780e-5`.

Decision: retain and promote B1. Both speed/RSS/parity rows pass independently,
and batch-one dispatch is unchanged. Next workload family: rotate away from B1
to a failing P0 video/AMG/decoder row.

## CPU worker scaling and tiny Linear output seeding

Change: implement W1 as a meta-cell over the real I1/I3/V2/A1 runners at 8
and 20 pinned workers. Add explicit summary gate evaluation and a
`--source-cell` selector. In linked Candle, seed fused-Linear outputs below
16,384 f32 elements sequentially; larger outputs retain the existing Rayon
path. `CANDLE_CPU_LINEAR_PARALLEL_SEED_MIN=0` restores the old behavior.

Hypothesis and PyTorch/Candle evidence: W1 shows that cached decoding widens
from `1.303x` at 8 workers to `1.817x` at 20 even though RSS stays near `0.30x`.
Candle's GEMM already executes the decoder's small projection shapes on one
thread, but output bias seeding still entered Rayon. Matched stage profiles
show the candidate reducing 8-token self-attention QKV from `0.753` to
`0.449 ms` and its output projection from `0.192` to `0.138 ms`; 9/10-token
self-attention QKV falls by roughly `0.22--0.31 ms`. No input, weight, or
per-worker scratch is copied.

Affected suite cells: all CPU image decoders, most directly I3/P1 and prompt
portions of video/AMG. Large encoder and AMG matrices stay on parallel seeding.

Focused A/B: four alternating 100-iteration Rust processes averaged `36.25
ms` candidate. The comparable non-outlier control processes averaged `36.86
ms`; one earlier control was `53.87 ms` during host contention. An alternating
full I1 A/B averaged `657.23 ms` candidate versus `674.59 ms` control
(`-2.6%`), with identical output checksums. Separate four-process cross-runtime
runs were too noisy for promotion, including two 800--925 ms PyTorch decoder
outliers.

Peak/current RSS: the change adds no tensor or scratch allocation. Process HWM
varied with allocator retention in the I1 A/B and showed no systematic rise.

Parity result: all A/B artifacts are byte-identical within Rust. Existing
PyTorch comparison retains the same I3 error (`8.202e-5` maximum,
`3.645e-5` mean); this change neither fixes nor worsens that pre-existing mean
gate failure.

Decision: retain as a small reusable scheduling improvement, backed primarily
by the matched projection stages and full I1 A/B. It does not establish I3
speed parity. Next workload family: video/AMG attribution or the remaining
large long-projection/attention work after rotation.

## CPU V2 memory-fuser layout and depthwise traversal

Change: keep the ConvNeXt memory fuser in NCHW for CPU f32 pointwise layers,
using shallow 1x1-convolution views of the existing Linear weights, and make
Candle's stride-1 depthwise kernel accumulate contiguous spatial rows per
kernel tap. `SAM2_CPU_MEMORY_FUSER_NCHW_POINTWISE=0` and
`CANDLE_CPU_DEPTHWISE_SPATIAL_ROWS=0` independently restore the controls.

Hypothesis and PyTorch/Candle evidence: a current matched profile at
`.benchmarks/runs/20260927T152000Z-v2-current-stage-profile` measures memory
encoding at `5.824 s` Rust versus `0.835 s` PyTorch. Image encoding favors
Rust (`19.224` versus `21.398 s`) and memory attention is close (`33.005`
versus `32.350 s`), making the memory fuser the dominant actionable gap.
PyTorch's source documents NHWC Linear and NCHW 1x1 convolution as equivalent.
Candle instead sent the non-contiguous NHWC view through broadcasted matmul.
The new CPU route shares checkpoint storage and avoids a 4 MiB activation
materialization. The depthwise traversal writes the final output directly and
uses no im2col or per-worker scratch.

Affected suite cells: V2-V4 and video portions of R1 directly; memory encoding
in interactive video edits also benefits. Non-CPU execution retains the
translated NHWC Linear path.

Focused A/B (all pairs): synchronized aggregate `pointwise_1` falls from
`1.969` to `0.208 s`; aggregate depthwise convolution then falls from `1.490`
to `0.174 s`. The two fusers fall from about `144` to `28.4 ms/frame`. The
current profiled Rust V2 total falls from `60.455` to `56.621 s`. A later
four-by-four alternating direct Rust A/B was contaminated by a host-load
transition: candidate times were `56.824/65.442/87.099/66.490 s` and control
times were `57.768/57.841/71.490/72.163 s`; candidate spread `1.533x` fails
the noise gate, so its unfavorable `1.064x` mean ratio is not promotion
evidence.

CPU P0 aggregates before/after: the full matrix remains incomplete. The quiet
four-pair V2 baseline
`.benchmarks/runs/20260927T150000Z-v2-cpu-confirmation` is `1.079x` latency
and `0.583x` RSS. The candidate
`.benchmarks/runs/20260927T155000Z-v2-memory-fuser-candidate` improves Rust
`62.712→59.041 s` (`-5.9%`), but PyTorch concurrently shifts
`58.100→51.973 s`, producing a clean but failing `1.136x` ratio.

Worst CPU P0 regression: none attributable from this focused evidence; the
candidate changes only the memory-fuser CPU route. A future affected-subset
run must cover V3/V4/R1 before graduation.

Peak/current RSS before/after: candidate V2 peak RSS is `0.586x` PyTorch and
passes both RSS gates. No persistent buffer, packed activation, or per-worker
model copy was added.

Parity result: candidate max/mean error is `0.064376`/`0.004047`, with zero
threshold-mask mismatches. All 15 Candle convolution tests pass.

Decision: retain as CPU kernel/layout wins, but do not promote V2 speed
parity; its current four-pair ratio is `1.136x`. The stage evidence is strong,
while the direct A/B is explicitly noisy. Next workload family: rotate to A1
or the cached decoder, then return for the required V3/V4/R1 regression set.

## Current CPU I3 refresh and transformer attribution

Change: no kernel change in this round; refresh the complete bedroom/groceries
I3 baseline on the current binary and collect matched decoder-module timings
after rotating away from V2.

Hypothesis and PyTorch/Candle evidence: older I3 evidence mixed code revisions
and single-fixture runs. The clean four-pair refresh
`.benchmarks/runs/20260927T160000Z-i3-current-refresh` is `1.715x` weighted
latency and `0.317x` RSS. Bedroom is `1.753x`; groceries is `1.677x`. All
runtime spreads pass. Bedroom retains its `3.645e-5` mean-logit error; groceries
passes its full output envelope.

Affected suite cells: I3 directly; the same transformer is used by I4, P1,
AMG decoder batches, and video prompt corrections.

Focused A/B (all pairs): not applicable; this is a baseline/attribution round.
The matched profile `.benchmarks/runs/20260927T160300Z-i3-current-stage-profile`
measures Rust's transformer at `529.315 ms` over 22 calls versus PyTorch
`256.935 ms`. Rust layers 0/1 average `9.435`/`9.660 ms` per call versus
PyTorch `4.925`/`4.973 ms`; final attention is `79.618` versus `29.977 ms`
in aggregate. Rust attention aggregates are `160.367 ms` token-to-image,
`136.982 ms` image-to-token, and `56.240 ms` self-attention, with projection
time dominating the token-to-image path. Rust's decoder-stage sum is about
`801.7 ms` versus PyTorch decoder total `454.6 ms`, so the transformer explains
roughly 78% of the remaining decoder deficit.

CPU P0 aggregates before/after: unchanged; the full matrix remains incomplete.

Worst CPU P0 regression: none; no implementation changed in this round.

Peak/current RSS before/after: current I3 weighted RSS is `0.317x` PyTorch and
passes both gates.

Parity result: groceries passes; bedroom maximum error passes at `8.202e-5`,
but mean error `3.645e-5` exceeds the `3e-5` limit. Threshold masks remain
exact.

Decision: use this as the authoritative current I3 baseline. Next workload
family after the required rotation is A1; the next I3 implementation round
should target reusable transformer projection/layout work and separately
trace the bedroom mean-logit divergence.

## CPU A1 fused bilinear stability counts

Change: add a CPU f32 custom operation that computes both AMG stability
threshold counts directly from low-resolution masks while evaluating the
same bilinear coordinates as Candle. Restore only masks that survive the
stability gate. `SAM2_CPU_FUSED_AMG_STABILITY=0` restores the composed route.

Hypothesis and PyTorch/Candle evidence: the current cars profile showed an
unattributed `~12 s` gap after decoder timing. Deeper matched boundaries at
`.benchmarks/runs/20260927T162500Z-a1-cars-quality-profile` attributed
`4.432 s` to restoring all IoU candidates and `7.090 s` to allocating and
reducing threshold tensors. The existing CUDA translation already fused this
operation. The new CPU implementation reads the existing contiguous input,
precomputes only the small interpolation-index vectors, and returns two
`u32` values per mask. It creates no full-resolution candidate tensor and no
per-worker input copy.

Affected suite cells: A1 and A2 directly. Any future automatic-mask workflow
using stability filtering benefits; ordinary image and video prediction are
unchanged.

Focused A/B (all pairs): the matched candidate profile at
`.benchmarks/runs/20260927T170000Z-a1-cpu-fused-stability-profile` totals
`269.9 ms` across the three fused count calls, replacing `11.522 s` of
restore-plus-stability work. Cars end-to-end Rust time falls from `33.328 s`
to `21.657 s`. The clean four-pair candidate at
`.benchmarks/runs/20260927T171000Z-a1-cpu-fused-stability-candidate` measures
Rust `19.750 s` on groceries and `21.858 s` on cars, versus the pre-change
four-pair means of `22.309 s` and `34.395 s` (`-11.5%` and `-36.4%`).

CPU P0 aggregates before/after: the full matrix remains incomplete. Within
A1, weighted latency improves `2.242x→1.712x`; groceries improves
`1.943x→1.801x` and cars `2.587x→1.628x`. Both remain above the `1.10x`
individual cap.

Worst CPU P0 regression: none outside A1 because dispatch is confined to the
automatic-mask quality filter. Both A1 fixtures improve. A2 remains to be run.

Peak/current RSS before/after: weighted peak-RSS ratio is effectively flat at
`0.6090x→0.6097x`; groceries is `0.860x` and cars `0.432x`. The candidate
avoids the large temporary restored-candidate tensor but process HWM is set by
earlier decoder batches.

Parity result: a unit test exactly matches fused counts to the composed Candle
operations. All four-pair outputs pass: groceries max/mean error is
`4.768e-7`/`1.421e-9`, cars is `2.050e-5`/`2.913e-9`, and process spreads are
clean.

Decision: retain. This is a large, allocation-free A1 win but not speed
parity. Next workload family: rotate back to the shared I3 transformer
projection/layout gap before another AMG-specific experiment.

## Rejected CPU decoder paired image projections

Change: concatenate each transformer's token-to-image K and image-to-token Q
weights once at model load, because both linears consume the same 4096-token
`keys + key_pe` activation. One 256→256 GEMM produced stride views for the two
attention calls in place of two 256→128 GEMMs. The experiment added about
0.5 MiB of model-owned CPU weight storage, no per-worker copy, and no extra
activation bytes.

Hypothesis and PyTorch/Candle evidence: matched I3 profiling identifies the
large-row attention projections as a leading shared gap. Combining them could
reuse input packing and reduce launch/scheduling overhead without a generic
backend change. PyTorch keeps separate `nn.Linear` modules, so this was tested
as a Candle-specific data-flow experiment rather than assumed equivalent in
performance.

Affected suite cells: I3 directly, plus I4/A1/A2 and image-decoder portions of
video workflows if retained.

Focused A/B (all pairs): adjacent four-pair, two-fixture runs are
`.benchmarks/runs/20260927T174000Z-i3-paired-projection-candidate` and
`.benchmarks/runs/20260927T174200Z-i3-paired-projection-control`. Candidate
Rust means are `44.741/44.446 ms` for bedroom/groceries; control means are
`38.226/37.234 ms`. Both sides pass the process-spread gate. The candidate is
about 18% slower geometrically despite removing one Linear invocation.

CPU P0 aggregates before/after: candidate weighted I3 ratio is `2.204x`
versus adjacent control `1.840x`; neither passes. Host variation makes neither
run a replacement for the authoritative `1.715x` baseline, but the adjacent
Rust A/B is decisive for this experiment.

Worst CPU P0 regression: both I3 fixtures regress by far more than the 3%
limit. The wider GEMM's changed tiling/packing behavior outweighs saved input
traversal.

Peak/current RSS before/after: candidate/control weighted RSS ratios are
`0.322x`/`0.317x`; the small persistent combined weights provide no memory
benefit.

Parity result: candidate and control produce identical I3 comparison errors;
groceries passes, while bedroom retains its pre-existing mean-error failure.

Decision: reject and remove completely. Do not fuse these projections by
widening the GEMM. The next projection experiment must improve reusable
immutable-weight packing or the existing individual GEMMs without duplicating
model weights.

## Rejected compact score stride for long-query/short-key attention

Change: size Candle's blocked CPU-flash score scratch and row stride to the
actual key count when it is smaller than the configured 1024-key block. For
SAM2's 4096-query, 8--10-key image-to-token attention this reduced worker-local
score scratch from about 512 KiB to 4--5 KiB without copying Q/K/V. A control
switch restored the wide stride for adjacent A/B measurement.

Hypothesis and PyTorch/Candle evidence: the current matched I3 profile measures
the image-to-token attention kernel at about `1.5 ms/call` Rust versus roughly
`0.6 ms/call` PyTorch. The original blocked route used a 1024-float score-row
stride even though only 8--10 elements were live, creating avoidable cache
distance and scratch traffic.

Affected suite cells: I3/I4, A1/A2, and image decoder portions of video edits.

Focused A/B (all pairs): the matched profile candidate/control at
`.benchmarks/runs/20260927T175000Z-i3-compact-score-profile` and
`.benchmarks/runs/20260927T175100Z-i3-wide-score-control-profile` reduced the
two targeted kernel calls from `3.433` to `2.708 ms` (`-21%`). However, the
required adjacent four-pair whole-I3 runs reversed that result. Wide-stride
control `.benchmarks/runs/20260927T180200Z-i3-wide-score-control` measured
Rust `36.171/34.677 ms` on bedroom/groceries. Compact-stride candidate
`.benchmarks/runs/20260927T180000Z-i3-compact-score-candidate` measured
`39.454 ms` on clean bedroom; groceries included a `52.835 ms` outlier and its
three ordinary processes still averaged about `36.75 ms`.

CPU P0 aggregates before/after: the clean control is `1.682x` weighted I3.
The candidate summary is `1.963x` and noisy; even removing its outlier leaves
both Rust fixtures slower than control. Neither establishes parity.

Worst CPU P0 regression: bedroom regresses about 9%; groceries' ordinary
processes regress about 6%. This exceeds the 3% acceptance limit.

Peak/current RSS before/after: process RSS remains about `0.31--0.32x`
PyTorch. The smaller transient worker scratch does not move process HWM.

Parity result: the dedicated long-query/short-key Candle test exactly matched
the tensor-expression reference within `2e-4`; I3 output errors were unchanged.

Decision: reject and remove. The compact stride improves the isolated kernel
but apparently selects an inferior tiny-GEMM/cache interaction at workflow
scale. This closes the two-experiment I3 round; rotate to a different failing
CPU family before returning with a materially different transformer design.

## CPU V2 post-fuser confirmation and re-ranking

Change: no implementation change. After rotating away from the closed I3
round, rerun four complete current-binary V2 process pairs and re-rank the
post-fuser matched stages before selecting another video kernel.

Hypothesis and PyTorch/Candle evidence: the retained NCHW pointwise and spatial
depthwise routes removed the previously dominant memory-encoder gap, but their
first promotion ratio was distorted by a simultaneous PyTorch timing shift.
The post-fix matched profile now measures Rust/PyTorch broad stages as image
encoder `19.174/24.136 s`, memory attention `32.988/38.032 s`, decoder
`1.308/4.297 s`, and memory encoder `2.259/2.795 s`. No remaining measured
stage supports a targeted Rust optimization from that sample.

Affected suite cells: V2 directly; the retained kernels also affect V3/V4 and
video R1.

Focused A/B (all pairs): not applicable; this is a current-combination
confirmation. `.benchmarks/runs/20260927T181000Z-v2-current-confirmation`
measures Rust processes at `54.926/51.552/51.214/48.467 s` (mean `51.540 s`,
spread `1.133x`) and PyTorch at `55.921/48.154/40.838/41.735 s` (mean
`46.662 s`, spread `1.369x`).

CPU P0 aggregates before/after: V2's diagnostic ratio improves from the prior
clean candidate's `1.136x` to `1.105x`, just above the `1.10x` individual cap.
The full P0 matrix remains incomplete, and the Python noise prevents promotion.

Worst CPU P0 regression: none; no code changed in this round.

Peak/current RSS before/after: latest peak-RSS ratio is `0.678x` and passes.
Absolute variation from the prior `0.586x` ratio follows different process
peaks, not a new allocation path.

Parity result: passes unchanged at `0.064376` maximum and `0.004047` mean
error, with zero threshold-mask mismatches.

Decision: keep the retained fuser changes and classify V2 as near-parity but
unconfirmed. Do not optimize against the noisy aggregate when matched stages
show no stable Rust loser. Repeat on a quieter host after rotating to a stable
failing CPU family.

## CPU A1 batch-one positional encoding

Change: keep the dense image positional encoding at batch one on CPU, cache
its contiguous 4 MiB token layout once, and use existing broadcast additions
for prompt batches. `SAM2_CPU_DECODER_BROADCAST_IMAGE_PE=0` restores the
materialized batch expansion. The prompt-dependent image activation remains a
single ordinary batch tensor.

Hypothesis and PyTorch/Candle evidence: the post-stability cars profile
`.benchmarks/runs/20260927T183000Z-a1-post-fused-deep-profile` attributes
`4.025 s` to transformer setup across four batch-64 decoder calls. The
translation first broadcast a single 256×64×64 PE to batch 64, then packed
that expanded NCHW view into a second NHWC token tensor. Every subsequent PE
use is an elementwise addition and Candle already broadcasts a batch-one
operand. Avoiding the expansion removes about 1 GiB of copied PE per decoder
call and adds no cache beyond the existing batch-one token cache.

Affected suite cells: A1/A2 directly. I3/I4 have batch-one PE already and are
unchanged; prompt-batched video/image workflows can benefit when they share one
encoded image.

Focused A/B (all pairs): matched cars profiling at
`.benchmarks/runs/20260927T184000Z-a1-broadcast-pe-profile` reduces aggregate
transformer setup `4.282→0.130 s` and end-to-end Rust `21.657→15.785 s` versus
the immediately preceding retained profile. The clean four-pair candidate
`.benchmarks/runs/20260927T185000Z-a1-broadcast-pe-candidate` measures Rust
`14.022 s` groceries and `17.015 s` cars, versus the prior retained
`19.750/21.858 s` (`-29.0%` and `-22.2%`).

CPU P0 aggregates before/after: within A1, weighted latency improves
`1.712x→1.333x`; groceries improves `1.801x→1.383x`, cars
`1.628x→1.284x`. Both still exceed the `1.10x` individual cap.

Worst CPU P0 regression: none in the affected A1 rows. Batch-one image paths
retain their existing representation and behavior.

Peak/current RSS before/after: weighted peak-RSS ratio improves
`0.610x→0.563x`; groceries is `0.788x`, cars `0.402x`. Absolute Rust HWM falls
from about 3.44/3.90 GiB to 3.15/3.63 GiB on groceries/cars.

Parity result: all eight paired outputs pass with unchanged strict errors:
groceries max/mean `4.768e-7`/`1.421e-9`, cars
`2.050e-5`/`2.913e-9`. All process spreads pass.

Decision: retain. This is a translation and lifetime correction: identical PE
values remain shared until the elementwise consumer instead of being copied
per prompt. A1 is substantially closer but not at parity; rotate workload
families before targeting its remaining transformer/upscaler compute.

## CPU I4 low-batch asymmetric flash dispatch

Change: use the existing CPU batched NHSD flash path for asymmetric decoder
attention at any batch `>=2`, rather than reserving it for AMG batch `>=32`.
Extend Candle's short-query/long-key GEMM dispatch to independent batch
entries. Each `(batch, head)` task owns bounded score scratch; Q/K/V are read
through their existing strides and are not packed or copied.

Hypothesis and PyTorch/Candle evidence: the clean I4 baseline
`.benchmarks/runs/20260927T190000Z-i4-current-after-pe` is `2.778x` weighted
with passing noise/parity/RSS gates. The matched profile
`.benchmarks/runs/20260927T191000Z-i4-current-stage-profile` measures two Rust
transformer calls at `294.4 ms` versus `100.5 ms` PyTorch. Batch 4 missed the
translated batch-32 flash gate and used generic materialized attention.

Affected suite cells: I4 directly; B1 batch 2/4 and any prompt-batched image or
video decoder also enter the generalized dispatch. I3 batch one is unchanged;
A1 batch 64 already used the same algorithms.

Focused A/B (all pairs): matched I4 profiling at
`.benchmarks/runs/20260927T192000Z-i4-batched-flash-profile` reduces transformer
time `294.4→142.9 ms`, token-to-image attention `142.6→32.9 ms`,
image-to-token `80.9→39.6 ms`, and Rust total `476.2→332.7 ms`. Four-pair I4
candidate `.benchmarks/runs/20260927T193000Z-i4-batched-flash-candidate`
improves Rust bedroom `440.82→293.22 ms` and cars `454.26→276.00 ms` versus
the clean baseline (`-33.5%`/`-39.2%`).

CPU P0 aggregates before/after: within I4, weighted latency improves
`2.778x→1.765x`. Both rows still exceed `1.10x`; bedroom's Python spread
`1.348x` makes the new ratio diagnostic, although both Rust distributions are
clean.

Worst CPU P0 regression: none in I4. The affected P1 B1 regression suite
`.benchmarks/runs/20260927T194000Z-b1-after-lowbatch-flash` passes every gate:
batch 2 is `0.790x`, batch 4 `0.762x`, and combined latency is `0.776x` versus
approximately `0.800x` before.

Peak/current RSS before/after: I4 weighted RSS remains effectively flat at
`0.405x→0.409x`; B1 combined RSS is `0.525x`. Scratch remains bounded and no
persistent storage is added.

Parity result: the new Candle batch-4 asymmetric test matches the composed
reference within `2e-4`; all 14 CPU-flash tests pass. I4 and B1 suite outputs
remain inside their fixed envelopes.

Decision: retain in sam2-rs and linked Candle. The change corrects a shape-
based dispatch omission and generalizes across I4/B1, but I4 still needs
further transformer/upscaler work after rotation.

## CPU V3 upstream-compatible BF16 memory lifetime

Change: make the high-level interactive video predictor store temporal
mask-memory features as BF16 on CPU as well as CUDA, matching upstream
`SAM2VideoPredictor._run_memory_encoder` and `_run_single_frame_inference`.
Selected memories are promoted to the active compute dtype only while
assembling attention. The public low-level `Sam2Base` API remains in its
native dtype. Add `--profile-v3-frames`, which streams per-frame/per-object
256x256 logits to bounded diagnostic artifacts without retaining a full video
copy.

Hypothesis and upstream evidence: the failed current V3 pair
`.benchmarks/runs/20260927T200000Z-v3-current-cpu` measured passing speed/RSS
but failed parity at `0.268817` maximum, `0.012666` mean error, and two
threshold mismatches. Upstream unconditionally converts stored
`maskmem_features` to `torch.bfloat16`; Rust guarded the same conversion with
`device.is_cuda()`. The small R6 probe showed the omitted quantization at the
first memory boundary and recurrent amplification on frame 1.

Affected suite cells: V2-V4 and the video half of R1. Static image, cached
decoder, AMG, batch-image, and the low-level model API are unchanged.

Focused A/B: frame diagnostics at
`.benchmarks/runs/20260927T203000Z-v3-frame-diagnostic` reduce selected V3
maximum/mean error `0.268817/0.012666→0.004349/0.000220` and threshold
mismatches `2→0`. Frame-25/object-2 low-resolution error falls from
`0.272742/0.110398` to `0.003154/0.001258`; maximum error across all 32-frame
low-resolution outputs is `0.007757`. The R6 frame-1 error falls about sevenfold
from `0.001904` to `0.000269`.

Rejected alternatives: disabling low-batch decoder flash is byte-identical.
Disabling temporal CPU flash worsens low-resolution error
`0.438149→0.472395`, increases one-workflow time from about `184 s` to
`580 s`, and raises RSS. Both flash paths remain enabled.

CPU P0 aggregates before/after: no complete P0 aggregate exists yet. Within
V3, the four-pair candidate
`.benchmarks/runs/20260927T210000Z-v3-bf16-memory-candidate` is `0.951x`
latency and `0.607x` peak RSS with passing output parity. Rust spread is
`1.217x`; PyTorch spread is `1.473x`, so the result is noisy and cannot be
promoted.

Worst CPU P0 regression: none. The direct V3 candidate is `184.238 s`, close
to the pre-fix `180.072 s` pair, while the four-pair Rust mean is `193.144 s`;
host variance prevents a sub-3% V3 claim until a quiet confirmation. The
affected clean V2 regression
`.benchmarks/runs/20260927T220000Z-v2-after-bf16-memory` measures Rust at
`51.253 s`, `0.6%` faster than the prior `51.540 s` diagnostic, with `0.583x`
RSS. Its clean `1.213x` ratio exposes a remaining speed gap because PyTorch is
stable and faster in this run. Memory representation is half-sized and peak
RSS remains well below PyTorch.

Decision: retain. This is an upstream translation/lifetime correction with a
large deterministic parity improvement, bounded conversion cost, no extra
copy per worker, and favorable diagnostic speed/RSS. Repeat V3 on a quiet host
before promoting its latency row; V2 is regressed cleanly, while V4 still
requires confirmation because it shares the temporal-memory representation.

## CPU A2 current baseline and shared upscaler attribution

Change: no kernel change in this entry; establish the missing clean A2
baseline and matched module attribution before another implementation round.

Hypothesis and PyTorch/Candle evidence: the clean four-pair run
`.benchmarks/runs/20260927T231000Z-a2-current-four-pair` is `47.076 s` Rust
versus `41.580 s` PyTorch (`1.132x`) and `0.856x` peak RSS. Both process
spreads pass. Numeric parity is `2.980e-6` maximum and `4.329e-9` mean with
zero observed threshold mismatches.

The synchronized diagnostic
`.benchmarks/runs/20260927T233000Z-a2-stage-profile` shows a tied image encoder
(`742` versus `725 ms`) and similar total transformer time (about `21--22 s`
Rust versus `20.880 s` PyTorch) across 16 batch-64 decoders. PyTorch's five
upscaling modules total about `6.95 s`; Rust's matching stages total about
`16.2 s`. The largest shared discrepancy is the second `64→32`
2x2/stride-2 transposed convolution at roughly `9.3 s` Rust versus `0.889 s`
PyTorch. PyTorch keeps channels-last strides throughout the upscaler; Rust's
first transposed convolution emits contiguous NCHW and makes the next GEMM
consume channel-strided input. AMG filtering/postprocessing is too small to
explain the end-to-end gap.

Affected suite cells: A1/A2 directly and every image/video path sharing the
mask decoder. Decision: use this as the baseline for one bounded, reusable
channels-last upscaler experiment; retain only with focused and cross-family
evidence. Next workload family after that experiment: video or cached image
decoder, per the rotation rule.

## Accepted batch-one upscaler residual fusion and owned-buffer GELU

Change: allow Candle's fused CPU 2x2/stride-2 transposed convolution to consume
either an equal-batch residual or a contiguous batch-one residual. The kernel
uses residual batch stride zero for the latter, so AMG's projected image skip
is read in place for every prompt without expanding or copying it. Also apply
the exact GELU directly to `LayerNorm2d`'s already-owned CPU output buffer,
removing one full output allocation while preserving operation order.

Hypothesis and PyTorch/Candle evidence: PyTorch broadcasts the one-image skip
across a batch of 64 prompts while its convolution backend keeps the combined
operation efficient. Candle's prior fused path required residual dimensions
to equal the convolution batch, rejected the batch-one view, and fell back to
separate convolution and broadcast-add tensors. The matched A2 profile had
attributed roughly `16.2 s` to the upscaler versus `6.95 s` in PyTorch.

Affected suite cells: A1/A2 directly; the owned-buffer normalization/GELU path
also applies to every mask-decoder invocation without changing its values.

Focused A/B (all pairs): the prior clean A2 four-pair baseline
`.benchmarks/runs/20260927T231000Z-a2-current-four-pair` averaged `47.076 s`
Rust and `41.580 s` PyTorch (`1.132x`). The retained candidate
`.benchmarks/runs/20260927T260000Z-a2-broadcast-residual-candidate` averages
`36.144 s` Rust and `38.972 s` PyTorch (`0.927x`), a `23.2%` Rust reduction.
Both distributions are clean (`1.059x`/`1.067x` spreads).

Cross-family evidence: A1
`.benchmarks/runs/20260927T263000Z-a1-after-broadcast-residual` improves from
`1.333x` to `1.049x` weighted latency. Groceries/cars are `1.035x`/`1.064x`,
both inside the per-cell cap with clean spreads. Weighted RSS is `0.566x`.
The cached-decoder diagnostic
`.benchmarks/runs/20260927T270100Z-i3-after-upscaler-fusion` is noisy on the
Python groceries row and therefore is not promoted. A same-binary four-pair
Rust A/B isolates the owned-buffer GELU at `36.38 ms` fused versus `37.59 ms`
composed (`-3.2%`), with byte-identical output; the fusion itself does not
regress batch-one decoding.

Peak/current RSS before/after: A2 is effectively flat at `0.856x→0.854x`;
A1 is `0.563x→0.566x`. No residual expansion, persistent cache, or worker
copy is introduced.

Parity result: A2 remains `2.980e-6` maximum / `4.329e-9` mean with zero
observed threshold mismatches. Both A1 fixtures pass their configured numeric
and structural parity gates. Direct Candle testing covers the actual
batch-one-residual/multi-batch-output case; fused exact GELU matches the former
tensor expression within `1e-6`.

Rejected experiment: retaining channels-last storage between the two
upconvolutions was correct and produced favorable synchronized stages, but a
four-pair same-binary Rust A/B averaged `36.79 s` channels-last versus
`36.34 s` for the simpler NCHW control. The channels-last candidate was also
noisier (`1.269x` spread versus `1.105x`), so all of that additional path was
removed.

Decision: retain the batch-one residual fusion and owned-buffer exact GELU;
reject channels-last. A2 now passes speed, while A1 misses only its strict
weighted `1.00x` latency target. Next workload family: cached image decoder,
then video, rather than further A2 tuning.

## Opt-in CPU temporal-memory MLP fusion

Change: add a Candle CPU linear+ReLU owned-output primitive, and use it with
the existing fused final-linear/residual primitive for temporal memory MLPs.
No persistent cache or worker-local tensor copy is introduced. Non-CPU routes
are unchanged. The complete SAM2 path is opt-in with
`SAM2_CPU_MEMORY_MLP_FUSION=1` until clean end-to-end promotion evidence is
available.

Hypothesis and PyTorch/Candle evidence: the current matched V2 profile
`.benchmarks/runs/20260927T273000Z-v2-current-matched-profile` measures tied
image encoders (`18.163/17.945 s`) but memory attention at `29.104 s` Rust
versus `20.911 s` PyTorch. Rust's four-layer MLP phases total `3.844 s`
(`1.794 s` first linear, `0.692 s` ReLU, `1.230 s` second linear, and
`0.128 s` residual), versus roughly `1.48 s` for the corresponding PyTorch
linears.

Affected suite cells: V2-V4 and video R1 temporal attention. Static image and
AMG paths do not invoke this MLP.

Focused evidence: the matched candidate profile
`.benchmarks/runs/20260927T280000Z-v2-memory-mlp-fusion-profile` reduces the
two combined MLP phases to `1.957 s` (`-49.1%`) and full temporal/head time
`33.290→31.889 s`. A same-binary direct candidate/control pair also reduces
the affected temporal/head phase `31.654→30.739 s`; total time ties because
the candidate image encoder was `0.878 s` slower in that pair.

Peak/current RSS: the matched candidate remains below PyTorch at `0.696x`.
The direct pair's HWM differed by about 35 MiB, but allocator final RSS varied
far more and the loaded host prevents attributing that difference.

Parity result: candidate/PyTorch error is `0.002441` maximum and `0.000213`
mean with zero threshold mismatches. Candidate/control is `0.001909` maximum
and `0.000137` mean with zero threshold mismatches. The small float change is
expected from bias/residual accumulation order and stays well inside the
video envelope. Candle's fused ReLU test matches the composed primitive within
`1e-6` on a deterministic tensor.

Promotion run: `.benchmarks/runs/20260927T283000Z-v2-memory-mlp-fusion-candidate`
reports `0.854x`, but both distributions fail the noise gate: Rust spread is
`1.270x`, Python `1.388x`, and system load was `21.6--26.2`. The noisy Rust
mean `53.190 s` also does not beat the clean default `51.253 s` baseline.

Decision: keep opt-in, not default. The repeatable synchronized substage win
justifies preserving the reusable primitive, while the absence of a clean
end-to-end win blocks promotion. Repeat on a quiet host before enabling it;
rotate next to scorecard/CI coverage or a different stable failing family.
## Current CPU V1 validation refresh

Change: no model-kernel change. Refresh the 32-frame video-session
initialization baseline after adding the CPU-only runner schema, timing-phase,
RSS, and summary-statistic validation.

Hypothesis and PyTorch/Candle evidence: V1 had favorable older evidence, but a
current same-binary run was needed before it could contribute to the active
scorecard. Both implementations time decoded-frame tensor preparation plus the
first cached image feature at the same `session_init` boundary; fixture decode
and model load remain separately reported phases.

Affected suite cells: `V1/bedroom32` directly; no code path changed.

Focused A/B (all pairs): four alternating fresh-process pairs at
`.benchmarks/runs/20260927T-v1-current-validator-baseline`, 20 workers pinned
to CPUs 0–19. Rust process medians are `794.88`, `743.71`, `693.00`, and
`776.16 ms`; PyTorch medians are `789.43`, `918.18`, `789.59`, and
`776.11 ms`.

CPU P0 aggregates before/after: the full aggregate remains incomplete. This
V1 row is `751.94/818.33 ms`, or `0.9189x`, and passes its `1.10x` cell cap.

Worst CPU P0 regression: none; this was a baseline-only run. Rust/PyTorch
repeat spreads are `1.147x`/`1.183x`, below the `1.25x` noise limit.

Peak/current RSS before/after: peak RSS is `0.4352x` PyTorch
(`1,099,911/2,527,547 KiB`), passing both P0 memory gates.

Parity result: exact for the canonical empty-session output (`0` maximum and
mean error, zero threshold differences).

Decision: promote this as the current passing V1 row. Next workload family:
refresh another incomplete P0 row rather than optimizing the already-passing
initializer.

## Current CPU S1 validation refresh

Change: no model-kernel change. Refresh the Small-model first-interaction and
cached-decode row under the current CPU runner validation.

Hypothesis and PyTorch/Candle evidence: S1 tests whether the optimized Tiny
paths generalize without introducing a variant-specific threshold. The scored
boundary is image encode plus cached decode; the runner also preserves each
component so an encoder-heavy total cannot hide decoder attribution.

Affected suite cells: `S1/bedroom` directly; its cached component informs the
shared I3/I4 decoder work.

Focused A/B (all pairs): four alternating current-binary pairs at
`.benchmarks/runs/20260927T-s1-current-validator-baseline`. Scored Rust totals
are `714.80`, `750.88`, `837.31`, and `776.81 ms`; PyTorch totals are
`679.14`, `713.81`, `800.11`, and `749.89 ms`.

CPU P0 aggregates before/after: full P0 remains incomplete. S1 is
`769.95/735.74 ms` (`1.0465x`) and passes the `1.10x` cell cap. Its cached-only
means are `38.32/21.49 ms` (`1.783x`), while encode means are
`731.63/714.25 ms` (`1.024x`).

Worst CPU P0 regression: none; this was a baseline-only run. Rust/PyTorch
spreads are `1.171x`/`1.178x`, both clean.

Peak/current RSS before/after: peak RSS is `542,632/1,604,628 KiB`
(`0.3382x`), passing memory gates.

Parity result: passes at `7.725e-5` maximum error, `1.898e-5` mean error, and
zero threshold mismatches.

Decision: promote S1 as a current passing row, while retaining the cached-only
ratio as cross-variant evidence for the shared decoder bottleneck. Next
workload family: video propagation/current lifetime coverage.

## Current-validator V2 repeat and MLP-fusion disposition

Change: no model-kernel change. Repeat the default one-object, 32-frame CPU
propagation path with four alternating pairs and the new runner validator.

Hypothesis and PyTorch/Candle evidence: the prior clean V2 row was `1.213x`,
while a later opt-in memory-MLP fusion run appeared faster only under severe
host noise. A fresh default run can determine whether that candidate merits
another promotion attempt.

Affected suite cells: `V2/bedroom32`; the memory-attention MLP is also used by
V3/V4.

Focused A/B (all pairs): default Rust medians at
`.benchmarks/runs/20260927T-v2-current-validator-baseline` are `49.85`,
`48.71`, `50.74`, and `49.67 s` (`1.042x` spread). PyTorch medians are
`41.63`, `42.37`, `59.79`, and `44.71 s` (`1.436x` spread). The raw means are
`49.74/47.12 s` (`1.0555x`), but fail the noise gate. The three ordinary
PyTorch medians average `42.90 s`, implying approximately `1.16x`.

CPU P0 aggregates before/after: no aggregate update; this row is noisy and
does not supersede the clean `1.213x` baseline.

Worst CPU P0 regression: none; no default code changed. The earlier opt-in
fusion candidate averaged `53.19 s` Rust under its noisy run, so it does not
beat this default Rust mean and remains disabled.

Peak/current RSS before/after: default V2 peak RSS is `0.5571x` PyTorch and
passes memory gates.

Parity result: passes unchanged at `0.002213` maximum error, `9.204e-5` mean
error, and zero threshold mismatches.

Decision: retain the default path and keep
`SAM2_CPU_MEMORY_MLP_FUSION=1` diagnostic-only. Do not promote the noisy raw
speed ratio. Next workload family: complete missing scorecard coverage before
another V2 kernel experiment.
## Retained batch-one decoder image-token cache

Change: cache the stable batch-one CPU token representation of
`image_embedding + no_mask_embed` across repeated prompt decodes. The first
call fuses the NCHW addition and NCHW-to-token pack into one 4 MiB allocation;
later calls return shallow Tensor handles. PromptEncoder retains one stable
broadcast view of its no-mask embedding, making the cache identity reliable.
`SAM2_CPU_DECODER_CACHE_IMAGE_TOKENS=0` restores the composed route.

Hypothesis and PyTorch/Candle evidence: the translated decoder rebuilt the
same 1x256x64x64 addition and token-layout copy for every cached prompt even
though neither operand changes for no-mask prompts. PyTorch keeps the image
predictor state and benefits from mature tensor/allocator handling around this
boundary. This cache is bounded to one derived tensor and keyed by both input
identities. The mutex remains held during first preparation so simultaneous
callers never allocate one 4 MiB copy per worker. Batched and mask-input routes
fall back without replacing the cache.

Affected suite cells: I3 directly; batch-one portions of S1/P1 and interactive
image use. I4/AMG batches do not dispatch this shape gate.

Focused A/B (all pairs): four reverse-ordered Rust-only I3/bedroom pairs with
20 workers pinned to CPUs 0–19. Enabled medians are `95.40`, `99.29`, `93.41`,
and `96.99 ms`; disabled medians are `97.01`, `99.66`, `105.23`, and
`107.86 ms`. Means are `96.28/102.44 ms` (`0.9398x`), every pair favors the
cache, and spreads are `1.063x`/`1.112x`. The invariant I4 batch-four A/B is
`396.02/393.17 ms` (`1.007x`) with both spreads passing; the cache correctly
does not dispatch there.

CPU P0 aggregates before/after: unchanged pending a new paired PyTorch I3
matrix. The focused Rust evidence predicts about a 6% I3 improvement, not
speed parity by itself.

Worst CPU P0 regression: no dispatched regression observed. I4, which uses
the unchanged fallback, differs by 0.7% under host variation.

Peak/current RSS before/after: I3 process HWM is approximately 530–533 MiB on
both routes. The retained tensor is 4 MiB; removing per-decode temporaries does
not materially increase measured peak RSS.

Parity result: all eight I3 output artifacts have SHA-256
`4248f1dc8af2a85c9638480b6eb52f6ed5affc749578f944ad967761e6fa931a`.
The direct fused add/pack unit test matches the original tensor composition
exactly; all 43 CPU-feature library tests pass.

Decision: retain. This is a reusable translation/state-lifetime improvement,
not a benchmark-specific kernel, and it avoids per-worker copies. Next
workload family: refresh a quiet-host I3 matrix when possible, while rotating
implementation work away from this now-closed cache experiment.

## Current CPU baseline noise and duration-floor correction

Change: no model change. Refresh L1 and the six-fixture I1 matrix under the
validated runner, then raise I1/I2 from three to five measured samples because
the fastest three-sample rows did not always exceed the plan's two-second
duration floor. The fixture manifest records the new spec digest.

Hypothesis and PyTorch/Candle evidence: repeated P0 promotion failures could
reflect real runtime variability or contention hidden by the aggregate host
load gate. Per-core `mpstat` shows this 40-logical-CPU machine is one 20-core
socket with SMT siblings; external work intermittently occupies several of
the same physical cores, so no alternative 20-core affinity mask provides
isolation.

Affected suite cells: I1/I2 sampling protocol; I1 and L1 current evidence.

Focused A/B (all pairs): I1's six rows are archived at
`.benchmarks/runs/20260927T-i1-current-validator-baseline`; weighted latency is
`0.9483x` and RSS `0.3508x`, with all cell/parity gates passing. PyTorch
groceries/cars/portrait spreads are `1.521x`/`1.310x`/`1.296x`. The subsequent
five-sample groceries run stabilizes PyTorch at `1.154x` but catches one Rust
process at `900.72 ms`, making Rust spread `1.539x`. L1 at
`.benchmarks/runs/20260927T-l1-current-validator-baseline` is `0.1375x`
latency and `0.2758x` RSS, but PyTorch spread is `1.444x`.

CPU P0 aggregates before/after: no promotion; both matrices fail the explicit
noise gate. Five samples make future I1/I2 rows protocol-compliant and more
resistant to brief within-process interruptions.

Worst CPU P0 regression: none; only benchmark iteration count changed.

Peak/current RSS before/after: unchanged in meaning; I1 weighted RSS is
`0.3508x`, L1 is `0.2758x`.

Parity result: every I1 row passes; L1 emits the expected empty artifact.

Decision: retain the five-sample protocol and keep these results diagnostic.
Do not loosen the spread gate or promote outlier-filtered ratios. A quiet or
exclusive host is still required for final graduation runs.

## Rejected CPU temporal positioned-memory reuse

Change: hoist `memory + memory_position` out of the four temporal-attention
layers and reuse one materialized tensor per frame. The experiment was CPU-only
and could be disabled with `SAM2_CPU_REUSE_POSITIONED_MEMORY=0`; other backends
kept the original route.

Hypothesis and PyTorch/Candle evidence: the operands are immutable across all
four layers, so the direct translation appeared to perform four equivalent
adds and allocate four memory-sized destinations. Hoisting would remove three
of those operations without copying the shared tensor per worker.

Affected suite cells: V2 directly, with the same memory-attention path used by
V3 and V4.

Focused A/B (all pairs): two reverse-ordered, 20-worker Rust-only V2 pairs are
archived at `.benchmarks/runs/20260927T-v2-positioned-memory-ab`. Enabled runs
are `51.738` and `53.800 s`; disabled runs are `52.241` and `51.738 s`.
Enabled/disabled means are `52.769/51.989 s`, or `1.0150x`. The first pair
favored reuse by 1.0%, while the reverse pair disfavored it by 4.0%, so there
is no repeatable local speed win.

CPU P0 aggregates before/after: not updated; this was an isolated Rust A/B and
the candidate was slower on average.

Worst CPU P0 regression: the focused V2 mean regressed 1.5%; no other cells
were run because the candidate failed its local acceptance gate.

Peak/current RSS before/after: enabled/disabled mean peak RSS is
`1,589,194/1,583,662 KiB` (`1.0035x`), effectively neutral and not an RSS win.

Parity result: all four full output artifacts have identical SHA-256
`6ded85132d763c3f92b9937d9945f08e34a218bdbc326bc5fce74c356a853207`;
all 43 CPU-feature library tests passed before measurement.

Decision: reject and remove. Candle's allocator reuse makes the apparent
allocation cleanup immaterial here, while extending the tensor's live range
does not improve RSS. Next workload family: cached image decoding, using
matched attribution rather than another temporal-attention micro-change.

## Accepted low-batch prepared decoder projection weights

Change: add an explicit linked-Candle API that prepares a Linear weight once
in contiguous `(input, output)` order and consumes that representation in the
existing fused-bias destination. The two-way decoder lazily prepares only its
attention projections, only when batch is 1--4, and reuses shallow Tensor
handles thereafter. `SAM2_CPU_DECODER_PRETRANSPOSE_WEIGHTS=0` restores the
original route. A synchronized one-time initializer prevents concurrent
callers from producing a private copy per worker. Batch-64 AMG never prepares
or dispatches the alternate orientation.

Hypothesis and PyTorch/Candle evidence: profiling consistently placed the
remaining I3 gap in repeated immutable-weight projections. Candle's GEMM
transposes the row-major destination internally, then repacks the original
`(output,input)` weight as its effective LHS on every call. A persistent
contiguous transpose makes that effective LHS unit-stride and removes this
packing. Current matched profiling confirms causality across 22 decodes:
self-attention projection time falls `46.07→17.35 ms`, token-to-image
`121.30→91.21 ms`, and image-to-token `69.78→52.56 ms`; attention kernels are
essentially unchanged.

Affected suite cells: I3/I4 directly, plus low-batch image, mixed-batch, and
video-edit decoders. A1/A2 batch-64 decoder arithmetic is explicitly unchanged.

Focused A/B (all pairs): four reverse-ordered I3/bedroom pairs at
`.benchmarks/runs/20260927T-i3-pretransposed-weight-ab` average
`30.766/34.127 ms` prepared/control (`0.9015x`), with all four pairs favoring
the candidate. Four I4/bedroom pairs average `240.164/245.730 ms` (`0.9773x`);
four independent I4/cars pairs average `239.141/250.781 ms` (`0.9536x`). Three
of four I4 pairs favor preparation on each fixture. The final batch-64 gate
was added after an ungated two-pair A1 diagnostic averaged `10.461/10.222 s`
(`1.023x`); the final route does not prepare or use these weights there. A
batch-2 B1 diagnostic has noisy encoder totals, but its isolated decoder phase
averages `52.74/53.95 ms` (`0.978x`).

CPU P0 aggregates before/after: the full matrix remains incomplete. The clean
four-pair/two-fixture I3 matrix at
`.benchmarks/runs/20260927T-i3-pretransposed-weight-candidate` improves weighted
latency `1.715x→1.543x`; bedroom/groceries are `1.498x`/`1.590x`. The I4
matrix at `.benchmarks/runs/20260927T-i4-pretransposed-weight-candidate`
improves `1.765x→1.600x`; bedroom is clean at `1.588x`, while cars is
diagnostic at `1.613x` because PyTorch spread is `1.254x` against the `1.25x`
limit. Both cells remain above their latency caps.

Worst CPU P0 regression: none observed on a dispatched P0 route. I3 and both
I4 Rust A/Bs improve; final A1/A2 dispatch is identical to control. A complete
P0 regression suite is still required after the next retained change.

Peak/current RSS before/after: the eager I3 A/B measured
`526,670/514,398 KiB` mean HWM (`1.024x`) while remaining about `0.32x`
PyTorch RSS. Final preparation is lazy and bounded to approximately 4.5 MiB
of decoder weights; ineligible AMG calls allocate none. The full candidate
matrices measure `0.3215x` I3 and `0.4043x` I4 RSS.

Parity result: all eight I3 A/B artifacts are byte-identical with SHA-256
`4248f1dc8af2a85c9638480b6eb52f6ed5affc749578f944ad967761e6fa931a`;
I4 A/B checksums are identical on both fixtures. The PyTorch comparison is
unchanged: I3/groceries and both I4 fixtures pass, while I3/bedroom retains
its pre-existing `3.645e-5` mean-logit failure. Candle's short/long-row unit
test passes, all 44 sam2-rs library tests pass, and a concurrency test proves
four workers receive the same prepared Tensor identity. The stale
`transformer-probe` call was also updated to the current zero-copy four-input
API, restoring the all-target test build.

Decision: retain. This is a bounded reusable Candle/translation improvement
with repeatable real-data wins and no per-worker copy. Next workload family:
return to video propagation or complete V4 lifetime coverage before another
decoder experiment.

## Accepted repeat-aware temporal key RoPE

Change: add a contiguous CPU-f32 RoPE operation that indexes one spatial
frequency table modulo each repeated memory grid and copies the unrotated
object-pointer tail into the same output allocation. Temporal cross-attention
uses it by default on CPU; `SAM2_CPU_REPEAT_AWARE_ROPE=0` restores the original
frequency `repeat`, generic rotation, and tail concatenation composition. The
query route and non-CPU fallback remain unchanged.

Hypothesis and PyTorch/Candle evidence: SAM2 temporal cross-attention rotates
up to seven repeated 64x64 memory grids on every layer and frame. The direct
translation allocated repeated cosine and sine tensors, rotated a narrow key
view, then copied both rotated keys and the pointer tail through concatenation.
The fused route reads immutable inputs and writes the one required final key
tensor, without per-worker input copies.

Affected suite cells: V2 directly, with the same temporal attention route used
by V3 and V4. Image and AMG cells do not dispatch it.

Focused A/B (all pairs): four reverse-ordered Rust-only V2/bedroom32 pairs at
`.benchmarks/runs/20260927T-v2-repeat-aware-rope-ab` record candidate times
`51.461`, `52.607`, `48.396`, and `60.255 s`, and control times `54.460`,
`53.893`, `52.048`, and `54.139 s`. Candidate/control means are
`53.180/53.635 s` (`0.9915x`); medians are `52.034/54.016 s` (`0.9633x`).
Three of four matched pairs favor the candidate, but its `1.245x` spread is
close to the `1.25x` limit because the final run was interrupted by host load.

CPU P0 aggregates before/after: not updated without a paired PyTorch run. The
clean promoted V2 ratio remains `1.213x`. A subsequent four-pair
Rust/PyTorch candidate at
`.benchmarks/runs/20260927T-v2-repeat-aware-rope-candidate` measures
`57.651/61.835 s` (`0.9323x`) and `0.5792x` RSS. It passes latency, RSS, and
parity gates, but PyTorch spread is `1.276x`, so the noise gate correctly
prevents promotion.

Worst CPU P0 regression: none established. The full-workflow mean improves
0.9%, and the isolated affected stage is unambiguously faster.

Peak/current RSS before/after: mean HWM is `1,616,382/1,582,512 KiB`
candidate/control (`1.0214x`), within observed process variation and still
well below PyTorch. The new operation eliminates repeat/cat temporaries but the
large attention working set determines process HWM.

Parity result: all eight A/B artifacts and both profiled artifacts are
byte-identical with SHA-256
`6ded85132d763c3f92b9937d9945f08e34a218bdbc326bc5fce74c356a853207`.
The focused multi-head/tail unit test matches the original composition, and
all 45 CPU-feature library tests pass.

Matched attribution: across the 124 cross-attention calls in one 32-frame
propagation, cross-key RoPE falls from `1544.33` to `554.90 ms` (`0.3593x`,
64.1% faster). Self-RoPE is unchanged at `256.59` versus `247.57 ms`; the
small difference is run noise because that code path is identical.

Decision: retain. It is a shape-general translation improvement with exact
outputs and one necessary output allocation. Refresh the paired V2 headline
on a quieter host, then rotate to another workload family rather than
continuing to tune this now-subsecond stage.

## Accepted layer-specific blocked decoder attention

Change: expose an explicit linked-Candle
`flash_attn_nhsd_f32_decoder_blocked` API and select it only for the second
two-way-transformer token-to-image attention at batch one. The default is
locally reversible with `SAM2_CPU_DECODER_LAYER1_BLOCKED=0`; Candle's ordinary
dispatch remains unchanged. The blocked kernel uses 32-query/512-key tiles,
bounded online softmax state, and no full score-matrix allocation.

Hypothesis and PyTorch/Candle evidence: matched decoder-vector tracing showed
projection differences below `2.6e-6`, followed by the first material
amplification in layer 1 token-to-image attention (`3.34e-5` at its attended
value and `4.64e-5` after the final residual normalization). PyTorch 2.9's CPU
flash implementation uses a block-online reduction with a 512-key tile for
short queries, whereas Candle's asymmetric path formed the full 4096-key
score row. Applying the blocked route to all three attentions fixed parity but
was 12--14% slower, so only the causative layer is selected.

Affected suite cells: I2 and I3 directly; batch-one image/video decoder paths
share the correction. I4/B1 batches and A1/A2's batch-64 decoder do not enter
this dispatch.

Focused A/B and paired evidence: the first I2/bedroom pair is `669.72/705.59
ms` Rust/PyTorch (`0.949x`) and passes parity. Two reverse-ordered ten-sample
Rust A/B pairs average `726.43 ms` candidate versus `718.46 ms` control
(`1.011x`), within the 3% allowance despite shared-host variance. The complete
four-pair I2 candidate at
`.benchmarks/runs/20260927T-i2-layer1-blocked-candidate` is `0.8657x` weighted
latency and `0.3491x` RSS, with every cell passing speed/RSS/parity. It is not
promoted because all three rows fail the repeat-spread gate under host
contention.

Parity result: I2 bedroom/groceries/portrait maximum errors are
`4.959e-5`/`5.341e-5`/`4.292e-5`; mean errors are at most `1.900e-5`, all below
the ordinary image envelope. The one-pair I3 regression at
`.benchmarks/runs/20260927T-i3-layer1-blocked-smoke` likewise makes both
fixtures pass and improves the raw weighted ratio from the clean older
`1.543x` row to `1.360x`. All 15 linked-Candle CPU-flash tests, including the
new explicit decoder API test, pass.

The independent prompt-diversity smoke at
`.benchmarks/runs/20260927T-p1-layer1-blocked-smoke` covers empty/1/8/32-point,
box, prior-mask, and multimask routes. It passes at `1.278e-4` maximum and
`1.914e-5` mean error with `0.3339x` RSS; latency improves from the older noisy
`1.938x` matrix to `1.836x` in this one pair, but remains a clear speed
bottleneck rather than a passing claim.

Decision: retain as the default narrowly scoped translation. It repairs
observable batch-one parity with only a bounded A/B cost, while the prior
all-layer and row-wise fallbacks remain rejected. A quiet four-pair I2/I3
promotion is still required.

## Current-validator M1 restoration and Large vision-stem parity

Change: fix suite result validation to distinguish stable TOML variant keys
(`base_plus`) from executable/runtime spellings (`base-plus`), and cover the
alias in the model-free validation smoke. In linked Candle, route only a large
RGB 7x7/stride-4 vision stem (`3 -> >=144` channels) through full im2col GEMM;
smaller stems and every 3x3 convolution retain the tiled default. Attribution
switches remain available as `CANDLE_CPU_CONV2D_FULL_KERNEL` and
`CANDLE_CPU_CONV2D_LARGE_VISION_FULL=0`.

Hypothesis and attribution: the stricter current validator first made M1
runnable and exposed a stable Large maximum-logit error of `1.14e-3`, above
its local `5e-4` envelope. Disabling decoder blocking, decoder flash, prepared
weights, Hiera window fusions, 1x1 convolution fusion, or native depthwise did
not move the outlier. `CANDLE_CPU_CONV2D_IMPL=full` reduced it to `3.25e-4`;
kernel-specific tests then proved that only the 7x7 patch embedding mattered
(`3.21e-4`), while forcing 3x3 full left the failure unchanged. A tiled GEMM
orientation experiment produced byte-identical failing output and was slower,
so it was removed.

Focused performance: two reverse-ordered Large Rust A/B pairs measure image
encode means of `3113.0 ms` for the parity-preserving full 7x7 stem versus
`2854.2 ms` tiled, a 9.1% local cost. This is confined to Large: Tiny/Small
use 96 output channels and Base-Plus uses 112, so their established paths are
unchanged. Large's paired end-to-end margin absorbs the cost.

Current paired diagnostic: `.benchmarks/runs/20260927T-m1-current-smoke`
measures Base-Plus `1251.93/1224.40 ms` (`1.0225x`) and Large
`3137.91/3129.79 ms` (`1.0026x`), or `1.0125x` weighted. Weighted RSS is
`0.5145x`; Large RSS is `0.6789x`. Base-Plus maximum/mean error is
`6.104e-5`/`5.028e-6`; Large is `3.214e-4`/`7.251e-6`, with exact threshold
decisions. Every P1 gate passes provisionally.

The configured four-pair follow-up at
`.benchmarks/runs/20260927T-m1-current-four-pair` measures Base-Plus
`1240.04/1384.33 ms` (`0.8958x`) and Large `3011.78/3679.05 ms` (`0.8186x`),
or `0.8563x` weighted, with `0.5122x` weighted RSS. Rust is stable at
`1.135x`/`1.076x` spread and both parity envelopes pass unchanged. PyTorch has
large process outliers (`1.423x`/`1.543x` spread), so this favorable raw ratio
fails the noise gate and is not promoted.

Verification: `python3 tools/run_parity_suite.py --validate-only` passes all
18 cells and now exercises the aliased variant name. The dedicated Candle
large-vision-stem scalar-reference test passes, as do all 15 CPU-flash tests.

Decision: retain. This restores the declared Large parity contract without
weakening thresholds and without regressing the important Tiny/Small routes.
The required four-pair shape is now exercised, but M1 still needs a quieter
four-pair repetition before promotion.

## Rejected fused short-key attention/output projection

Change tested, then removed: for decoder image-to-token attention, reassociate
`softmax(QK') V W` as `softmax(QK') (V W)` and compute the short-side value
projection once before blocked attention. The experimental linked-Candle
kernel used query-block-owned output rows, bounded worker-local score storage,
and no intermediate head-output tensor or reduction copy. The public API,
dispatch, implementation, and test were removed after the full-workflow A/B.

Hypothesis and attribution: the corrected matched I3 profile at
`.benchmarks/runs/20260927T-i3-blocked-stage-profile` measures Rust/PyTorch at
`38.289/21.194 ms` per cached decode. Ordinary cross-attention remains the
largest attributable gap: the long-query/short-key image-to-token calls take
roughly `3.5--4.0 ms` in Rust versus `1.7--1.9 ms` in PyTorch, and the reverse
token-to-image calls take roughly `2.3--3.2 ms` versus `1.25--1.3 ms`.
Reassociation reduced the two isolated image-to-token boundaries to about
`2.1--2.8 ms`, including removal of the following output projection.

Parity and end-to-end evidence: the composition unit test matched the ordinary
route within `2e-4`. The paired smoke at
`.benchmarks/runs/20260927T-i3-fused-short-key-smoke` passed both fixtures at
`5.15e-5` maximum and `2.089e-5` mean error, with `0.300x` RSS, but remained
`1.551x` PyTorch latency. More importantly, four reverse-ordered adjacent Rust
I3 A/B pairs measured candidate medians of `32.420`, `33.335`, `30.841`, and
`30.254 ms`, versus control medians of `31.551`, `28.874`, `29.583`, and
`29.737 ms`. Candidate/control is `1.0593x`, and every pair favors control.

Decision: reject and remove completely. Projecting each head onto the much
wider output dimension makes the short-side algebra cheaper at the isolated
boundary but loses 5.9% in the synchronized user workflow. This experiment
does not change the retained decoder path or its parity contract. Rotate away
from I3 before another decoder experiment.

## Motion-diverse V5 fixture and deterministic fixture serialization

Change: add 32 stride-four frames from the upstream SAM2 juggling gallery MP4
to V5, so encoded-input coverage is no longer bedroom-only. The fixture
builder now applies configured MP4 start/stride selection through FFmpeg and
canonicalizes safetensors JSON headers while streaming tensor payloads
unchanged. This fixes process-random metadata ordering that previously changed
fixture SHA-256 values even when every tensor byte was identical.

Scope: V5 includes decoder-library work and remains separate from core
inference. FFmpeg/Decord codec optimization is explicitly out of scope. A
sam2-rs change would only be justified if attribution found avoidable wrapper
work; V1--V4 continue to use identical predecoded inputs for model comparisons.

Evidence: two independent full fixture generations now reproduce byte-exact
image and video tensors; representative stable SHA-256 values are
`1baf2d8ce73df1d2925cf2cca5eb065eca5295cd456d2c23ac2741e017723be2`
for bedroom and
`73711dce4a75aa924076fb3f033286e8f3d082ed1887c5cc0fe3d6518e928afa`
for bedroom32. The generated juggling MP4 has exactly 32 decoded frames at
1280x720 spanning 5.208 seconds. Suite validation still passes all 18 cells
and 16 timing contracts.

The initial one-pair juggling smoke was `1.161x`, but four short pairs reversed
to `0.876x` with both implementations noisy. Raising each process to ten
iterations gives `.benchmarks/runs/20260927T-v5-juggle-duration-qualified`:
Rust/PyTorch is `1124.03/1402.63 ms` (`0.8014x`), RSS is `0.4530x`, and Rust
spread passes at `1.107x`. PyTorch spread narrowly fails at `1.273x`, so the
favorable ratio remains diagnostic.

Decision: retain the broader fixture and deterministic builder; do not tune
video decoder dependencies. Repeat V5 only as part of scorecard promotion,
then return to a core inference family.

## Current I4 attribution and rejected prepared upscaler weights

Change tested, then removed: expose a linked-Candle representation that packs
each immutable 2x2/stride-2 transposed-convolution weight once at model load,
then shares the approximately 288 KiB total read-only preparation across all
decoder calls. No worker copied the prepared weights. The ordinary path was
available as `CANDLE_CONVTRANSPOSE_PREPARED_WEIGHT=0` during the experiment.

Attribution: `.benchmarks/runs/20260928T-i4-current-stage-profile` measures a
current I4/bedroom pair at `242.31/183.87 ms` (`1.3178x`) and `0.4127x` RSS.
Across 44 decoder calls, Rust/PyTorch transformer time is
`3018.98/2230.98 ms`. Rust upconv-0 is `444.47 ms` versus `114.85 ms`; Rust's
combined second upconvolution/GELU is `728.22 ms` versus PyTorch's
`186.54 ms`. Since each timed I4 sample performs two decodes, transformer and
upscaler differences account for essentially the full end-to-end gap.

The focused kernel profile initially attributed about 2.8 ms/decode to a
"weights" interval, but inspection showed that interval also includes final
output allocation/zeroing; it was not valid evidence for packing alone. The
prepared and ordinary kernels produce byte-identical output, and the focused
composition test passed.

Whole-workflow A/B: four reverse-ordered I4/bedroom pairs record candidate
medians `377.774`, `382.000`, `399.441`, and `398.564 ms`, versus controls
`381.520`, `388.991`, `376.245`, and `371.778 ms`. Candidate/control averages
are `389.44/379.63 ms` (`1.0258x`); only two pairs favor the candidate.

Decision: reject and remove completely. The saved repack does not survive the
real decoder workflow and does not justify persistent state. The remaining
upscaler opportunity is GEMM/scatter scheduling or a direct primitive, not an
immutable-weight cache. Rotate before another I4 experiment.
