# CPU-only performance, RSS, and output-parity plan

**Active scope:** CPU only. GPU work is paused and is not part of this plan's
execution or completion criteria.

## Objective

Reach or beat the pinned PyTorch SAM2.1 implementation in CPU latency and host
RSS on the important ways SAM2 is actually used, while preserving the existing
output-parity envelope. The result must generalize across real images, prompt
shapes, repeated interaction, video propagation, worker counts, and more than
one model size. A single fast Tiny/one-point fixture is evidence for one cell,
not completion of this plan.

This plan supersedes `OPT.md` as the execution checklist. `OPT.md` and
`PERFORMANCE_INVESTIGATION.md` remain the detailed history of accepted and
rejected operator experiments. The filename is retained for continuity with
the active goal.

### Scope lock (2026-09-27)

From this revision onward, execute this plan on CPU only. Do not spend an
optimization round, benchmark run, profiling run, or graduation gate on the
GPU path unless the user explicitly reopens GPU work in a later request.
Existing GPU measurements remain archival material in the companion results
and investigation documents; they are not active evidence for this plan and
must not be refreshed as part of the remaining work.

Shared code should remain device-neutral where practical, and a CPU change
must not knowingly break compilation or correctness on another backend. That
maintenance constraint does not require GPU performance measurement. All
unqualified references below to speed, RSS, workers, baselines, scorecards,
regressions, and parity mean CPU execution and host RSS.

### Performance priority

1. **CPU end-to-end speed is the primary headline and optimization target.**
   Rank work by the weighted P0 CPU score and per-cell caps.
2. **CPU host RSS and lifetime behavior are co-equal constraints.** Track both
   process peak RSS and the live/reset plateau for repeated workflows.
3. **Output parity constrains every speed result.** A faster result is not
   accepted if it misses the scenario-specific float or exact-output gate.
4. **Generalization prevents benchmark overfitting.** Exercise 8 and 20
   workers, multiple workload families, realistic data, and representative
   model sizes before promotion.

GPU results do not accept or reject a CPU optimization here. Historical GPU
sections in the companion results and investigation documents must not be
counted toward this plan's gates. GPU compilation and correctness checks are
also outside the graduation checklist; avoiding a knowingly device-specific
regression remains an implementation-review concern, not a benchmark lane.

## Current evidence, not the completion claim

| Workload | Current Rust/PyTorch latency | Current Rust/PyTorch host RSS | Status |
| --- | ---: | ---: | --- |
| Tiny static image, one point, 20 workers, optimized CPU features | `0.889x` | `0.446x` | faster on this machine and fixture |
| Full first-interaction matrix (`I1`, six fixtures, four pairs) | `0.948x` | `0.351x` | speed/RSS/parity pass, but groceries/cars/portrait are noisy from process outliers; I1/I2 now use five samples so every process clears the two-second duration floor |
| Rich first interaction (`I2`, three fixtures, 20 workers) | `0.866x` latest noisy diagnostic | `0.349x` | layer-1 blocked decoder attention makes all fixtures pass parity and every speed/RSS cell pass, but shared-host spreads fail; a quiet four-pair promotion remains pending |
| Tiny static image, one point, 8 workers | approximately `0.98-1.01x`, noisy | substantially below `1.0x` | parity not established |
| Tiny cached prompt decode (`I3`, bedroom/groceries, 20 workers) | `1.543x` clean; `1.360x` latest one-pair diagnostic | `0.322x` clean; `0.323x` diagnostic | layer-1 blocked attention fixes both fixtures' parity and improves the latest sample, but speed gates still fail and a four-pair refresh is pending |
| Cached batched prompts/prior mask (`I4`, bedroom/cars, 20 workers) | `1.600x` clean matrix; `1.318x` latest bedroom profile | `0.404x` clean; `0.413x` profile | parity/RSS pass and current attribution splits the remaining gap between transformer and transposed convolution; immutable upscaler-weight preparation was rejected at `1.026x` Rust A/B |
| Video initialization (`V1/bedroom32`, 20 workers) | `0.919x` | `0.435x` | current four-pair row passes latency, RSS, noise, and exact-output gates |
| Video forward propagation (`V2/bedroom32`, 20 workers) | `1.213x` clean; `0.932x` latest noisy diagnostic | `0.583x` clean; `0.579x` diagnostic | clean post-BF16-memory baseline still fails latency; repeat-aware key-RoPE saves 64% in its isolated stage and the paired candidate passes speed/RSS/parity, but Python spread `1.276x` prevents promotion |
| Three-object video edit (`V3/bedroom32`, 20 workers) | `0.951x` latest four-pair diagnostic | `0.607x` | matching upstream CPU BF16 memory storage fixes prior parity failure; speed/RSS/parity pass, but PyTorch spread `1.473x` makes promotion noisy |
| Long video lifetime (`V4/bedroom128`, 20 workers) | `0.994x` one-pair diagnostic | `0.769x` peak; `0.573x` after reset | latency/RSS/parity pass provisionally; 128-frame route and reset metric are now exercised, but configured four-pair warmed baseline remains pending |
| Repeated image/video sessions (`R1`, four pairs) | `0.960x` weighted; video `1.024x` | `0.410x` weighted | speed/RSS pass, but PyTorch spreads make both rows noisy; repeat before promotion |
| Practical automatic masks (`A1`, groceries/cars, four pairs each) | `1.049x` weighted; `1.035x`/`1.064x` | `0.566x` weighted | batch-one residual fusion improves prior `1.333x`; both rows now pass the cell cap with clean parity/noise/RSS, while the weighted speed target remains open |
| Automatic masks, upstream default grid (`A2`, groceries, four pairs) | `0.927x` | `0.854x` | batch-one residual fusion brings clean speed parity with matching outputs; per-cell RSS passes but standalone weighted RSS remains above `0.80x` |
| Model load/first-ready (`L1`, Tiny, four pairs) | `0.138x` | `0.276x` | current Rust spread is `1.071x`, but PyTorch spread `1.444x` keeps the row noisy |
| Small-model first interaction plus cached decode (`S1`, four pairs) | `1.047x` | `0.338x` | scored cell passes latency/RSS/noise/parity; cached-only subphase remains `1.783x` and supports the shared decoder diagnosis |
| Prompt diversity (`P1`) | `1.938x` older noisy matrix; `1.836x` latest one-pair diagnostic | `0.315x` older; `0.334x` diagnostic | layer-specific attention preserves parity across empty/1/8/32-point, box, prior-mask, and multimask routes, but speed remains far from the P1 cap |
| Encoded JPEG/MP4 input (`V5`) | `0.591x` prior mixed row; `0.801x` motion-clip diagnostic | `0.480x` prior; `0.453x` motion clip | upstream juggling footage adds real motion/content breadth; ten-iteration Rust is stable and faster, but PyTorch spread `1.273x` keeps the new row diagnostic; codec-library tuning is out of scope |
| Mixed image batch (`B1`, batch 2/4) | `0.776x` combined | `0.525x` combined | both four-pair rows pass after low-batch decoder flash; batch 2 `0.790x`, batch 4 `0.762x` |
| Worker scaling (`W1`) | image subset `1.150x`; A1 smoke `1.288x`; V2 smoke `0.899x` | image `0.315x`; A1 `0.834x`; V2 `0.582x` | all four families runnable at 8/20 workers; image row is four-pair but noisy, A1/V2 are one-pair diagnostics |
| Larger models (`M1`, Base-Plus/Large) | `1.012x` conservative one-pair; `0.856x` latest noisy four-pair | `0.512x` latest weighted | targeted Large vision-stem dispatch restores parity; the configured four-pair run passes raw speed/RSS/parity with stable Rust, but PyTorch spreads of `1.423x`/`1.543x` block promotion |

Prior GPU evidence remains archived in `CPU_PARITY_RESULTS.md` and
`PERFORMANCE_INVESTIGATION.md`; it is not an active requirement here.

The old default-CPU row in `README.md` predates most retained optimizations and
must be labelled historical or regenerated. The current favorable CPU result
uses `cpu-flash-attention,cpu-glibc-vector-math`, 20 pinned workers, and the
integrated allocator policy. It does not establish video, AMG, other variants,
or other worker-count parity.

## Definition of parity

Ratios are Rust divided by PyTorch; lower is better.

### Required CPU P0 gate

- Weighted geometric-mean latency ratio across all P0 cells: `<= 1.00x`.
- Every individual P0 latency cell: `<= 1.10x`.
- Weighted geometric-mean peak-RSS ratio: `<= 0.80x`.
- Every individual P0 peak-RSS cell: `<= 1.00x`.
- No retained change may regress any already-passing P0 cell by more than 3%
  unless the total suite improves by at least 5% and the exception is recorded.
- Float output error must stay within the current scenario-specific parity
  envelope; integer metadata, ordering, frame indices, RLEs, and thresholded
  masks remain exact where the existing parity fixtures require exactness.

### P1 portability gate

- Weighted latency ratio across P1 cells: `<= 1.05x`.
- No P1 cell above `1.15x`.
- No P1 RSS cell above `1.00x`.
- Results must be repeated on two worker counts. The initial target machine
  uses 8 and 20 pinned workers; add a second CPU architecture when available.

CPU P0 is the sole headline latency and RSS score. Parity is declared only when
two independent CPU suite runs on a quiet host pass the same gates. A result
with load or steal-time outliers is diagnostic and does not update the
headline.

## Apples-to-apples protocol

Every benchmark cell must enforce all of the following:

- Identical published checkpoint, float32 model, decoded RGB bytes or frozen
  normalized tensor, prompt coordinates, output selection, and postprocessing.
- Identical logical worker cap and CPU affinity. Avoid nested Rayon/BLAS thread
  pools; record `RAYON_NUM_THREADS`, PyTorch thread counts, affinity, governor,
  CPU model, kernel, allocator settings, and relevant Cargo features.
- Separate fresh Rust and Python processes, with alternating launch order.
- Exclude model construction from steady-state inference timing, but measure it
  in its own startup cell. Include checkpoint/model memory in process `VmHWM`.
- Warm up the exact measured route. Synchronization is irrelevant on CPU, but
  lazy initialization must finish before samples begin.
- At least four paired processes. Use enough iterations for a cell to exceed
  two seconds of measured work without making one suite impractical.
- Report each pair, arithmetic mean of per-process medians, p50/p95 where a
  sequence has many operations, peak `VmHWM`, output error, and fixture digest.
- Run core inference with predecoded/frozen inputs and user-visible end-to-end
  loading as separate cells. Never charge JPEG/FFmpeg work to only one side.
- Use the same `.pt` or equivalent byte-identical weights. Compare `.pt` and
  safetensors loading separately rather than mixing formats in one cell.
RSS is Linux process peak RSS, not tensor-only memory. For long workflows also
sample current RSS after load, encode, interaction, propagation, reset, and
drop. This distinguishes a necessary live set from allocator retention.

## Frozen realistic data set

Record SHA-256, dimensions, frame selection, and licensing/source in a fixture
manifest. Keep a small checked-in manifest; large images, videos, checkpoints,
and generated tensors remain outside Git.

### Image set

- `bedroom/00089.jpg`: ordinary video frame and continuity with current work.
- `notebooks/images/truck.jpg`: landscape image with a large dominant object.
- `notebooks/images/groceries.jpg`: cluttered multi-object scene.
- `notebooks/images/cars.jpg`: high-detail scene and larger source image.
- One portrait crop and one odd-sized image derived deterministically from the
  above, to cover aspect ratio and resize boundaries without new data.

For each image, freeze realistic positive/negative clicks, a box, a correction
click, and a prior mask from the Python reference. Prompts must target actual
foreground/background locations rather than always using the image center.

### Video set

- Bedroom JPEG directory: deterministic 32-frame and 128-frame subsets.
- `bedroom.mp4`: encoded-input end-to-end loading and the same decoded frame
  range used by both implementations.
- One gallery MP4 with faster motion and multiple objects.

Source-image pixel count alone is not useful inference breadth because both
implementations resize images to SAM2's fixed encoder resolution. Prefer
different content, aspect ratios, prompts, model sizes, batches, and temporal
length/motion. Report JPEG/video decode separately so oversized encoded inputs
do not disguise fixed-shape model performance as an I/O result.

Freeze interactions for one and three objects: initial click/box, forward
propagation, a middle-frame correction, reverse propagation, object removal,
and reset. Store reference output summaries and selected full logits rather
than every large intermediate tensor.

## Workload matrix

Weights are used only for the aggregate score. Per-cell caps still apply.

### CPU P0: release-blocking scenarios

| ID | Scenario | Variants/data | Primary metric | Weight |
| --- | --- | --- | --- | ---: |
| `I1` | First image interaction: encode plus one positive click | Tiny; all six image shapes | end-to-end inference latency | 14 |
| `I2` | Rich first interaction: positive/negative clicks plus box | Tiny; bedroom, groceries, portrait | end-to-end inference latency | 8 |
| `I3` | Cached single prompt decode | Tiny; bedroom and groceries | median of 20 prompt changes | 10 |
| `I4` | Cached batched prompts and prior-mask correction | Tiny; bedroom and cars | prompts/second and p95 | 8 |
| `V1` | Video session initialization from predecoded RGB | Tiny; 32 bedroom frames | latency and peak/current RSS | 7 |
| `V2` | One-object forward propagation | Tiny; 32 bedroom frames | ms/frame and p95 | 13 |
| `V3` | Three-object edit workflow, including correction/reverse/remove | Tiny; 32 frames | total and per-operation latency | 12 |
| `V4` | Long propagation and memory lifetime | Tiny; 128 frames | ms/frame, peak RSS, RSS after reset | 8 |
| `A1` | Automatic masks, practical interactive configuration | Tiny; groceries and cars | total latency, masks/second | 8 |
| `A2` | Automatic masks, upstream default grid | Tiny; one cluttered image | total latency and peak RSS | 5 |
| `L1` | Model load and first-ready state | Tiny `.pt` and safetensors | load latency and peak/current RSS | 3 |
| `S1` | Representative model scaling | Small; first interaction and cached decode | latency and RSS | 4 |

### P1: generalization and portability

| ID | Scenario | Coverage |
| --- | --- | --- |
| `W1` | Worker scaling | P0 `I1`, `I3`, `V2`, and `A1` at 8 and 20 workers |
| `M1` | Larger models | Base-Plus first interaction/cached decode; Large load plus first interaction |
| `B1` | Image batch | batches 2 and 4, mixed aspect ratios |
| `V5` | Encoded video input | JPEG directory, bedroom MP4, and stride-sampled upstream juggling MP4 decode plus session initialization, reported separately from inference |
| `P1` | Prompt diversity | empty prompt, 1/8/32 points, box, mask input, multimask on/off |
| `R1` | Repeated lifetime | create/use/drop or reset ten image/video sessions; report RSS plateau |

### P2: useful but non-blocking

- Different allocator/environment configurations.
- Additional x86-64 and ARM64 machines.
- Very long videos, batch sizes above four, and unusual image dimensions.
- Hugging Face download time; cache lookup and model loading remain relevant,
  but internet throughput is not an inference-performance metric.

## Benchmark infrastructure

- [x] Add `benchmarks/cpu-suite.toml` describing fixtures, prompts, variants,
  operations, weights, warmups, iterations, and parity thresholds.
- [x] Add a deterministic fixture builder that emits a manifest with source
  SHA-256 and generated-tensor SHA-256. Reuse the current bedroom fixture.
- [x] Add one Rust suite binary that loads a model once per requested cell and
  emits JSON Lines with phase timings, current RSS, `VmHWM`, and checksums.
- [x] Add a Python driver with the same cell schema and timing boundaries.
- [x] Add an orchestrator that pins CPUs, alternates process order, checks host
  load, runs paired repetitions, verifies outputs, and computes CPU ratios.
- [x] Make every implemented cell independently runnable for profiling and regression
  bisection; do not require the full suite during kernel development.
- [x] Add `CPU_PARITY_RESULTS.md` generated from raw JSON, with a compact P0/P1
  scorecard and links to archived run metadata.
- [x] Keep only a small stable headline table in `README.md`; label the old
  17.351x default result historical rather than presenting it as current.
- [x] Add a CI-sized smoke subset covering schema, timing-boundary consistency,
  and parity. Full performance runs remain pinned-host jobs, not ordinary CI.

  Run `python3 tools/run_parity_suite.py --validate-only`. It needs no model or
  large fixture: it validates the full CPU P0/P1 matrix and fixture-manifest
  digest, exercises all 16 emitted timing contracts (including rejection of a
  one-sided phase change), and compares synthetic float artifacts through the
  production parity function. Every real paired run now applies the same
  result-schema, median/p95, RSS, timing-phase, and output-metadata checks
  before scoring. A live two-iteration I3 run at
  `.benchmarks/runs/20260927T-ci-smoke-even-median-diagnostic` verifies the
  current Rust and Python runners pass those checks; it also reproduces the
  separately tracked I3/bedroom mean-logit parity failure.

## Execution sequence

CPU speed and RSS are the only active optimization lanes. Rank work by weighted
impact, rotate across workload families, and retain changes only when focused
and suite-level CPU evidence supports them.

### Phase 0: establish the scorecard

- [x] Implement the infrastructure above; later optimization phases changed
  model kernels only after the independent runners and frozen fixtures existed.
- [ ] Generate all Python reference fixtures first.
- [ ] Run the complete CPU P0 matrix twice at 20 workers and the P1 worker
  subset at 8 workers. Mark noisy cells rather than silently dropping them.
- [ ] Record time and RSS by broad stage: load, decode/prepare, encoder, prompt
  encoder, decoder, memory encoder, memory attention, postprocessing, and drop.
- [ ] Publish independent CPU speed and RSS scores at 20 and 8 workers.
- [ ] Rank gaps by `weight * max(latency_ratio - 1, 0)` and separately by RSS.

Exit criterion: every CPU P0 cell has reproducible parity evidence and
an assigned owner layer (`sam2-rs`, Candle, dependency, or harness).

### Phase 1: close CPU cross-scenario overhead

Prioritize costs appearing in at least two P0 scenarios:

- [ ] Audit tensor lifetime and allocation counts in image decoder, video frame
  loop, memory encoder/attention, and AMG batches.
- [ ] Reuse bounded scratch and immutable derived tensors by shape/device; no
  unbounded cache and no copy per worker.
- [ ] Remove repeated layout materializations and host copies at API boundaries.
- [ ] Measure reset/drop behavior and allocator retention, not just `VmHWM`.
- [ ] Move a primitive into Candle only when it benefits multiple SAM2 paths or
  is a generally missing tensor/backend operation.

Exit criterion: aggregate P0 improves at least 5%, no P0 regression exceeds
3%, and RSS gates still pass.

### Phase 2: interactive image decoder parity

Current evidence makes cached decoding a high-priority gap, but work is capped
by the suite rotation rule below.

- [ ] Attribute matched module boundaries for cross-attention, MLP, upscaler,
  skip projections, normalization, tensor setup, allocation, and teardown.
- [ ] Compare PyTorch CPU dispatch/layout choices for the dominant shapes.
- [ ] Evaluate persistent packed immutable RHS weights through a clean Candle
  API; include cache bytes in RSS and invalidate by storage/layout identity.
- [ ] Optimize short-query/asymmetric attention and transposed convolution only
  when focused wins survive `I3`, `I4`, `V3`, and full P0 regression runs.
- [ ] Test prompt counts 1/8/32 and batch sizes 1/2/4 to avoid one-shape tuning.

Exit criterion: `I3` and `I4` are each `<=1.10x`, with no video-edit regression.

### Phase 3: CPU video and AMG parity

- [ ] Profile `V2-V4` for repeated image encoding, memory selection, memory
  attention, consolidation, mask restoration, and state-container copying.
- [ ] Ensure frame/object state retains shallow tensor handles where safe and
  does not clone full masks or features per object unnecessarily.
- [ ] Bound memory by the reference temporal policy; verify RSS after removal
  and reset actually returns to the expected allocator plateau.
- [ ] Profile `A1/A2` separately for backbone reuse, prompt batching, mask
  filtering, RLE conversion, crop copies, and NMS.
- [ ] Optimize shared decoder/batching primitives before specialized AMG code.
- [ ] Keep image/video decoding out of core inference timings, then optimize the
  separately reported end-to-end I/O cells only if they are material.

Exit criterion: `V1-V4` and `A1-A2` meet individual latency/RSS caps.

### Phase 4: variants and worker portability

- [x] Run Small, Base-Plus, and Large cells before changing any size threshold.
- [ ] Replace Tiny-specific thresholds with shape-derived decisions backed by
  isolated benchmarks and end-to-end cells.
- [ ] Investigate the 8-worker long-attention gap without changing the 20-worker
  winner blindly. Prefer runtime work partitioning keyed by shape and workers.
- [ ] Test for nested parallelism and oversubscription in GEMM, Rayon, image
  preprocessing, and postprocessing.
- [ ] Repeat P0/P1 on a second CPU architecture when available.

Exit criterion: all P1 gates pass without a separate benchmark-only code path.

### Phase 5: graduation

- [ ] Run two complete quiet-host suites from clean release builds.
- [ ] Verify all existing Python/Rust parity fixtures and unit tests.
- [ ] Make stable optimized paths the documented/default CPU configuration, or
  explain any remaining opt-in feature and publish both results.
- [ ] Update `README.md` with the weighted score and key scenario rows, including
  workload limits and machine details.
- [ ] Record remaining gaps and rejected experiments in
  `PERFORMANCE_INVESTIGATION.md`.

## Rotation rule: do not get stuck on one item

Optimization proceeds in rounds, not an unlimited sequence against one kernel:

1. Select the highest weighted failing CPU P0 cell, breaking close ties by
   choosing a different workload family from the preceding round.
2. Allow at most two implementation experiments or five working days on its
   current leading hotspot, whichever comes first.
3. Retain only changes with a repeatable local win, valid parity, and no suite
   regression. Record and remove rejected experiments promptly.
4. Then rotate to the highest failing cell from a different workload family:
   image, cached decoder, video, AMG, model scaling, or memory lifetime.
5. After every two retained changes, rerun the affected CPU P0 subset; run the
   complete CPU P0 suite after four retained changes.
6. If a hotspot remains after two rounds, escalate one layer: inspect PyTorch,
   isolate the Candle primitive, or redesign the benchmark attribution. Do not
   keep micro-tuning the same loop without new evidence.

A change is especially valuable when it improves two or more workload families.
The plan's objective is the scorecard, not the fastest individual cell.

## Change acceptance template

Append one entry per experiment to `PERFORMANCE_INVESTIGATION.md`:

```text
Change:
Hypothesis and PyTorch/Candle evidence:
Affected suite cells:
Focused A/B (all pairs):
CPU P0 aggregates before/after:
Worst CPU P0 regression:
Peak/current RSS before/after:
Parity result:
Decision: retain / reject / keep opt-in
Next workload family:
```

## Completion checklist

- [x] Active execution and graduation scope is CPU-only; historical GPU results
  remain archived and require no reruns or updates.
- [ ] CPU results are the README performance headline and include static image,
  cached-prompt, video, AMG, startup, and lifetime workloads on realistic data.
- [ ] README reports CPU speed and RSS independently at both required worker
  counts; no combined or single-fixture score obscures a failing CPU cell.
- [ ] CPU P0 weighted latency `<=1.00x`; every CPU P0 cell `<=1.10x`.
- [ ] CPU P0 weighted RSS `<=0.80x`; every CPU P0 cell `<=1.00x`.
- [ ] P1 weighted latency `<=1.05x`; every P1 cell `<=1.15x`.
- [ ] Both 8-worker and 20-worker suites pass their stated gates.
- [ ] Tiny, Small, and representative Base-Plus/Large coverage is recorded.
- [ ] Static image, cached prompt, video, AMG, startup, and lifetime cells pass.
- [ ] Two independent quiet-host runs agree.
- [ ] Output parity and exact metadata checks pass.
- [ ] README headline is current, reproducible, and does not overclaim.
