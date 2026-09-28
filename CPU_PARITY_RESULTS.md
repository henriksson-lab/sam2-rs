# CPU performance-parity results and historical GPU archive

This is the stable scorecard generated from archived JSON Lines runs described
by [`CPU_PARITY_PLAN.md`](CPU_PARITY_PLAN.md). CPU latency, host RSS, and output
parity are the active scope. The GPU results below are retained as historical
evidence only and do not contribute to current completion gates. Ratios are
Rust divided by pinned PyTorch; lower is better.

## Current scorecard

| Device / tier | Weighted latency | Weighted memory | Worst latency | Output parity | Status |
| --- | ---: | ---: | ---: | --- | --- |
| CPU P0 | pending | pending | matrix incomplete | latest I2/I3 diagnostics pass; promoted V1/V2 parity passes | baseline incomplete; I3, V2, and A1 still fail active speed gates |
| P1 portability | pending | pending | pending | latest M1 diagnostic passes | baseline incomplete |
| GPU P0 eager (historical) | `0.941x` | `0.815x` peak allocated; `0.688x` peak reserved | `I4/cars` `1.080x` | all rows except one `A2` boundary pixel | archived; outside active scope |
| GPU graph (historical) | `1.057x` on implemented I3/I4 | approximately `0.76x` process VRAM | `I4/cars` `1.066x` | passing | archived partial baseline |

The model-free CPU infrastructure smoke is
`python3 tools/run_parity_suite.py --validate-only`. It validates all 18
P0/P1 cell definitions, fixture-manifest identity, 16 concrete runner timing
contracts, even-sample median/p95 semantics, RSS/output record fields, and the
production float-artifact parity calculation. Real paired runs apply the same
record and timing-boundary checks before any ratio is scored. The live
two-iteration runner check at
`.benchmarks/runs/20260927T-ci-smoke-even-median-diagnostic` completes through
the new validator; as expected, it retains the known I3/bedroom mean error
(`3.645e-5` versus the `3e-5` limit), so it is infrastructure evidence rather
than a passing performance row.

The current four-pair CPU `V1/bedroom32` baseline at
`.benchmarks/runs/20260927T-v1-current-validator-baseline` passes every active
cell gate. Rust averages `751.94 ms` versus PyTorch `818.33 ms` (`0.9189x`),
with `0.4352x` peak RSS and exact empty-session output parity. Rust/PyTorch
process spreads are `1.147x`/`1.183x`, both below the `1.25x` noise cap. This
run uses the current binary, 20 workers pinned to CPUs 0–19, four alternating
fresh-process pairs, and the new schema/timing validator.

The matching current CPU `S1/bedroom` Small-model baseline is
`.benchmarks/runs/20260927T-s1-current-validator-baseline`. The scored first
interaction (one warmed image encode plus cached decode) averages `769.95 ms`
Rust versus `735.74 ms` PyTorch (`1.0465x`) with `0.3382x` peak RSS. Both
process spreads are below `1.18x`; maximum/mean logit errors are
`7.725e-5`/`1.898e-5`, so cell latency, memory, noise, and parity gates pass.
The separately retained cached-only component is `38.32/21.49 ms`
(`1.783x`), evidence that the decoder gap generalizes beyond Tiny even though
the S1 scored total stays below the `1.10x` cap.

A current-validator V2 repeat is archived at
`.benchmarks/runs/20260927T-v2-current-validator-baseline`, but is diagnostic
because one PyTorch process rose to `59.79 s`, producing a `1.436x` Python
spread. Rust is stable at `49.74 s` mean and `1.042x` spread; the noisy raw
ratio is `1.0555x`, RSS is `0.5571x`, and parity passes at `0.002213` maximum,
`9.204e-5` mean error with exact threshold decisions. The three non-outlier
Python medians average `42.90 s`, corresponding to roughly `1.16x`, so this
does not supersede the clean `1.213x` V2 row or establish speed parity. It also
does not rehabilitate the opt-in memory-MLP fusion: that candidate's earlier
noisy Rust mean (`53.19 s`) is slower than this stable default mean.

The full current-validator I1 matrix at
`.benchmarks/runs/20260927T-i1-current-validator-baseline` is favorable but
not promotable because three PyTorch fixture rows are noisy. Across all six
fixtures it is `0.9483x` weighted latency and `0.3508x` RSS; every speed,
memory, and parity cell passes. Bedroom, truck, and odd have clean spreads;
groceries, cars, and portrait have PyTorch spreads of `1.521x`, `1.310x`, and
`1.296x`. Rust is within the `1.25x` limit on all six. The suite now uses five
I1/I2 samples per process because three samples could total less than the
protocol's two-second measured-duration floor. The first five-sample groceries
repeat remains diagnostic: its PyTorch side stabilizes, but one sustained Rust
process outlier makes Rust spread `1.539x`. This confirms shared-host process
contention rather than a one-runtime correctness issue.

The current L1 repeat at
`.benchmarks/runs/20260927T-l1-current-validator-baseline` is likewise noisy
only on PyTorch: Rust/PyTorch load means are `146.38/1064.40 ms` (`0.1375x`),
RSS is `0.2758x`, and spreads are `1.071x`/`1.444x`. It does not replace the
scorecard row until a quiet-host repeat passes the noise gate.

The current-code I2 layer-specific blocked-attention matrix is archived at
`.benchmarks/runs/20260927T-i2-layer1-blocked-candidate`. All three fixtures
pass output parity: maximum errors are `4.959e-5`, `5.341e-5`, and `4.292e-5`,
with means at or below `1.900e-5`. The raw weighted result is `0.8657x`
latency and `0.3491x` RSS, and every individual speed/RSS gate passes. It is
diagnostic because host contention makes all rows noisy (Rust bedroom spread
`2.234x`; PyTorch spreads up to `3.813x`). Two reverse-ordered, ten-sample
Rust A/B pairs average `726.43 ms` for blocking only layer 1 versus
`718.46 ms` for the prior path (`1.011x`), within the 3% regression allowance;
the parity benefit is deterministic. The earlier all-three-attention blocked
experiment was rejected because it was 12--14% slower.

The same retained default fixes the prior I3 parity failure in the one-pair
current-code diagnostic at
`.benchmarks/runs/20260927T-i3-layer1-blocked-smoke`. Bedroom/groceries are
`1.410x`/`1.312x` (`1.360x` weighted), `0.3231x` weighted RSS, and both pass at
`1.900e-5`/`1.815e-5` mean error. This improves the clean older `1.543x` row,
but remains a one-pair diagnostic and does not satisfy the speed gate.
The prompt-diversity regression at
`.benchmarks/runs/20260927T-p1-layer1-blocked-smoke` also passes all combined
empty/1/8/32-point, box, prior-mask, and multimask outputs at `1.278e-4`
maximum and `1.914e-5` mean error. Its `1.836x` latency and `0.3339x` RSS
confirm generalization but leave the P1 speed gate open.

The first current-validator M1 run exposed and fixed the `base_plus` TOML-key
versus `base-plus` runtime-name mismatch. The post-fix, one-pair diagnostic at
`.benchmarks/runs/20260927T-m1-current-smoke` is `1.0125x` weighted latency and
`0.5145x` RSS. Base-Plus is `1.0225x`; Large is `1.0026x`. Both pass parity,
including Large at `3.214e-4` maximum and `7.251e-6` mean error with exact
threshold decisions. This is provisional until a four-pair run passes noise.

The configured four-pair follow-up at
`.benchmarks/runs/20260927T-m1-current-four-pair` is `0.8563x` weighted
latency and `0.5122x` RSS. Base-Plus and Large are `0.8958x` and `0.8186x`;
their Rust spreads are a controlled `1.135x` and `1.076x`, and parity is
unchanged. PyTorch spreads are `1.423x` and `1.543x`, however, so the run is
diagnostic rather than promoted. The less favorable one-pair values above
remain the conservative estimate until a quiet four-pair repetition passes
the noise gate.

V5 now includes 32 frames sampled every fourth frame from the upstream SAM2
juggling gallery clip, in addition to the bedroom JPEG directory and MP4. The
ten-iteration four-pair diagnostic at
`.benchmarks/runs/20260927T-v5-juggle-duration-qualified` measures
`1124.03/1402.63 ms` (`0.8014x`) and `0.4530x` RSS. Rust spread is `1.107x`;
PyTorch spread is just above the noise cap at `1.273x`, so the row is not
promoted. This is an encoded-input/I/O result and does not replace the
predecoded V1--V4 inference rows. Codec-library optimization remains outside
the CPU parity scope.

The current I4/bedroom matched profile at
`.benchmarks/runs/20260928T-i4-current-stage-profile` measures
`242.31/183.87 ms` (`1.3178x`) and `0.4127x` RSS with passing parity. Across
44 decoder calls, Rust/PyTorch transformer time is `3018.98/2230.98 ms`.
Rust's full transposed-convolution/normalization/activation upscaler totals
`1398.76 ms`; the comparable PyTorch stages total `472.27 ms`. These
components explain essentially all
of the two-decode per-sample gap. An explicit prepared upscaler-weight
experiment was removed after four adjacent Rust A/B pairs regressed
`389.44/379.63 ms` (`1.0258x`).

The retained batch-one decoder image-token cache removes a repeated translation
artifact from I3/S1/P1-style prompt updates. PromptEncoder now returns one
stable shallow no-mask broadcast view; TwoWayTransformer performs the first
NCHW add and token pack directly into one 4 MiB tensor, then reuses it by image
and dense-embedding identity. Initialization holds a mutex so concurrent first
callers do not create a copy per worker. `SAM2_CPU_DECODER_CACHE_IMAGE_TOKENS=0`
restores the composed per-decode route. Four reverse-ordered Rust I3/bedroom
pairs average `96.28 ms` enabled versus `102.44 ms` disabled (`0.9398x`), with
all four pairs favoring the cache and spreads `1.063x`/`1.112x`. All eight
canonical artifacts have identical SHA-256. I4's batch-four route does not
dispatch the cache and is neutral within noise (`396.02/393.17 ms`, `1.007x`).

The subsequent linked-Candle prepared-weight optimization supersedes the I3
headline. Candle exposes an explicit reusable contiguous `(input, output)`
weight representation, and the decoder lazily creates it only for batch 1--4.
One synchronized initializer prevents concurrent callers from making private
copies; batch-64 AMG retains the prior GEMM route and never allocates the
prepared tensors. Four reverse-ordered Rust I3/bedroom pairs average
`30.766/34.127 ms` enabled/control (`0.9015x`), with every pair favoring the
candidate and byte-identical artifacts. Matched profiling attributes the win
to projection work: aggregate attention projection time across 22 decodes
falls from `237.1` to `161.1 ms`, while attention-kernel time is essentially
unchanged.

The complete two-fixture four-pair matrix at
`.benchmarks/runs/20260927T-i3-pretransposed-weight-candidate` is clean at
`1.5430x` weighted latency and `0.3215x` RSS, improving the prior `1.715x`.
Bedroom/groceries are `1.4977x`/`1.5897x`; both runtime spreads pass. Output
behavior is unchanged: groceries passes and bedroom retains the pre-existing
`3.645e-5` mean-error failure with exact threshold masks.

The corresponding I4 matrix at
`.benchmarks/runs/20260927T-i4-pretransposed-weight-candidate` improves the
prior `1.765x` to `1.6001x`, with `0.4043x` RSS and passing output parity.
Bedroom is clean at `1.5877x`; cars is `1.6126x`, but PyTorch spread is
`1.2545x`, narrowly above the `1.25x` noise limit, so the weighted I4 row is
diagnostic rather than a quiet-host promotion. Separate four-pair Rust A/Bs
show `0.977x` on bedroom and `0.954x` on cars. I3 and I4 therefore remain
above their latency gates despite the retained improvement.

The authoritative complete current-binary, duration-qualified four-pair GPU
P0 eager scorecard is `.benchmarks/runs/20260927T095000Z`. Its 96
alternating-order process pairs have weighted Rust/PyTorch latency `0.94099x`,
external peak VRAM `0.76589x`, peak allocated device memory `0.81540x`, peak
reserved memory `0.68771x`, and host RSS `0.31174x`. Rust is about 5.9%
faster overall and uses materially less host/device memory. The worst latency
row is I4/cars at `1.07961x`; the worst peak-allocation row is V4 at
`1.06047x`. All speed, memory, and noise gates pass with no noisy rows. A2
retains its single threshold-boundary pixel mismatch. No gate or workload was
relaxed.

The complete current-binary one-pair P0 smoke at
`.benchmarks/runs/20260927T093212Z` passes aggregate speed and every memory
gate at `0.94834x` latency, `0.30862x` host RSS, `0.76222x` external VRAM,
`0.81140x` peak allocated memory, and `0.68985x` peak reserved memory. Its
single V1 process was an outlier at `1.183x`; the immediate duration-qualified
four-pair V1 confirmation `.benchmarks/runs/20260927T094022Z` passes at
`1.05902x` with `1.097x`/`1.087x` spreads. Substituting that confirmed row
gives `0.94288x`, with I4/cars worst at `1.07729x`. A2 remains the only strict
parity failure.

| GPU P0 cell | Current latency ratio | Individual status |
| --- | ---: | --- |
| `I1` | `1.054x` | passes; worst fixture `1.060x` |
| `I2` | `1.061x` | passes; worst fixture `1.062x` |
| `I3` | `0.874x` | faster; all rows pass noise gate |
| `I4` | `1.060x` | passes; worst fixture `1.080x` |
| `V1` | `1.016x` | passes |
| `V2` | `1.001x` | passes |
| `V3` | `1.014x` | passes |
| `V4` | `0.998x` | faster; peak allocation `1.060x` passes |
| `A1` | `1.003x` | passes; strict parity passes |
| `A2` | `0.953x` | faster; strict parity still fails at one pixel |
| `L1` | `0.146x` | faster |
| `S1` | `0.952x` | faster |
| `M1` | `1.005x` | passes; Large worst at `1.030x` |

The superseded first unified four-pair GPU P0 eager scorecard is
`.benchmarks/runs/20260927T025949Z`. Unlike an aggregate assembled from runs
on different code revisions, every row uses the same retained binary and
alternating fresh-process protocol. Weighted Rust/PyTorch latency is
`0.9981x`, external peak VRAM is `0.764x`, peak allocated device memory is
`0.792x`, peak reserved memory is `0.691x`, and host RSS is `0.313x`. Thus the
aggregate speed and every memory gate pass. This is only the first of the two
required independent suites and is not a completion claim: `V1` narrowly
misses its individual cap at `1.110x`; `I4`, `A1`, and `A2` remain larger
latency failures; and `A2` retains its one-pixel strict-parity failure.

| GPU P0 cell | Unified latency ratio | Individual status |
| --- | ---: | --- |
| `I1` | `1.018x` | passes individual cap |
| `I2` | `1.020x` | passes individual cap |
| `I3` | `0.988x` | passes |
| `I4` | `1.162x` | fails |
| `V1` | `1.110x` | fails narrowly |
| `V2` | `1.021x` | passes |
| `V3` | `1.025x` | passes |
| `V4` | `1.017x` | passes; peak allocated memory `1.051x` passes |
| `A1` | `1.266x` | fails; groceries `1.149x`, cars `1.395x` |
| `A2` | `1.269x` | fails speed and strict parity |
| `L1` | `0.146x` | passes |
| `S1` | `1.018x` | passes |
| `M1` | `1.013x` | passes; Base-Plus `0.983x`, Large `1.044x` |

The archived summary initially marked `S1` noisy because it computed spread
from only the roughly 5 ms cached subphase while scoring encoder plus cached
decode. Recomputing the same four process totals gives `1.104x` Rust and
`1.007x` Python spread, both below the `1.25x` cap. The runner now applies the
noise gate to the same per-process total used by the latency score and retains
the cached-only samples as separate diagnostics.

After that first unified baseline, video initialization was aligned more closely to
upstream `forward_batch`: Rust now allocates one contiguous frame tensor,
streams each normalized frame into its final device offset through one reused
host-frame buffer, and retains narrow tensor views that share the storage.
The four-pair GPU V1 result (`.benchmarks/runs/20260927T033655Z`) improves from
`382.38 ms` (`1.110x`) to `320.54 ms` Rust versus `333.67 ms` PyTorch
(`0.961x`). Host RSS remains `0.273x`; all device-memory ratios are unchanged.
The propagated V2 check (`.benchmarks/runs/20260927T033743Z`) passes output
parity at `0.996x` latency and `0.270x` RSS. Replacing only V1 in the unified
score projects GPU P0 latency to approximately `0.9909x`; this projection is
not promoted until the next complete same-binary suite. The matching four-pair
CPU V1 check (`.benchmarks/runs/20260927T033848Z`) passes its individual gates
at `1.046x` latency and `0.441x` RSS.

The subsequent low-resolution-only M2M change avoids restoring initial AMG
candidates that are immediately discarded before refinement. Its final
four-pair A1 run (`.benchmarks/runs/20260927T034710Z`) improves weighted
latency from the unified `1.266x` to `1.215x`: groceries is `1.127x` and cars
is `1.310x`. Weighted external VRAM is `0.475x`, peak allocated memory
`0.586x`, and host RSS `0.345x`; strict metadata/RLE parity remains exact.
Together with V1 this projects the GPU P0 aggregate to approximately
`0.9876x`; the later complete suite supersedes that projection. A CPU A1/groceries smoke run
passes exact parity but exposes a separate large CPU gap at `2.785x` latency
and `0.889x` RSS (`.benchmarks/runs/20260927T034851Z`).

The first generalized GPU `I1/bedroom` smoke run on 2026-09-26 was a harness
diagnostic, not a publishable baseline. It exposed that the old Python
benchmark disabled upstream's dynamic multimask stability fallback while the
translated high-level Rust predictor correctly enabled it. Realistic inputs
therefore selected different mask tokens. The benchmark boundary is now
aligned to upstream defaults. A corrected one-process warm diagnostic measured
`83.00 ms` Rust versus `74.03 ms` PyTorch (`1.121x`), `0.377x` host RSS, and
`2.86e-5` maximum mask-logit error. The matching CPU diagnostic was `0.980x`
latency and `0.370x` host RSS. They are deliberately not promoted to the
scorecard aggregate until the four-pair protocol passes.

The first four-pair realistic AMG baseline (`A1/groceries`, 8×8 grid with
mask-to-mask refinement) was recorded on 2026-09-26. After correcting Rust's
refinement batching and moving quality filtering before the device-to-host
transfer, Rust averaged `1463.38 ms` versus PyTorch `646.84 ms` (`2.262x`).
Peak process VRAM was `3442 MiB` versus `4962 MiB` (`0.694x`), and host peak
RSS was `0.373x`. All four complete canonical output streams—including mask
count/order, metadata, and every uncompressed RLE run—matched with maximum
float error `4.768e-7`. Memory gates pass for this cell; its latency gate does
not. This row is included as validated partial coverage but cannot populate
the weighted GPU P0 score until the remaining matrix is implemented.

The four-pair `I1` baseline covers all six frozen image shapes. Its weighted
GPU latency ratio is `1.056x`; individual rows range from `1.039x` to `1.078x`
and therefore pass the per-cell `1.10x` cap, but the aggregate does not yet
meet the `1.00x` gate. Weighted process VRAM is `0.725x` and host RSS is
`0.339x`. Maximum logit error across the matrix is `7.439e-5`. The retained
run is `.benchmarks/runs/20260926T200220Z` and contains 48 compared output
artifacts with hashes. A four-pair attempt to remove cross-stream lifetime
events measured `1.057x` on bedroom versus the ordinary row's `1.054x`, so
that experiment was rejected rather than presented as an optimization.

The current four-pair GPU `I4` baseline
(`.benchmarks/runs/20260927T021218Z`) covers a batch of four cached prompt sets
followed by four prior-mask corrections on bedroom and cars. Direct F32 NCHW
kernels avoid cuDNN overhead for the `16→256` correction-mask projection and
fuse the final non-overlapping `64→32` 2x2 transposed convolution, high-
resolution residual, bias, and exact GELU without layout copies. Bedroom
measures `23.35` versus `20.13 ms` (`1.160x`) and cars `23.51` versus
`20.17 ms` (`1.166x`), for a weighted `1.163x`. Weighted process VRAM is
`0.781x`, peak allocated device bytes `0.879x`, peak reserved bytes `0.743x`,
and host RSS `0.356x`; output parity passes. This supersedes the original
`1.926x` bedroom baseline but still fails the eager latency cap.

A promoted four-pair `V2/bedroom32` run after native cuDNN grouped convolution
and fixed-lifetime RoPE caching measured `6234.48 ms` Rust versus `6037.86 ms`
PyTorch (`1.033x`). Before these changes, the comparable warmed diagnostic was
`12508.25` versus `5905.97 ms` (`2.118x`). Host RSS is `0.262x`, peak allocated
device memory `0.986x`, peak reserved memory `0.985x`, and external peak VRAM
`0.986x`. Selected-frame output parity remains within the video envelope and
all thresholded masks are exact. The retained run is
`.benchmarks/runs/20260926T211157Z`.

The current four-pair GPU `I3` baseline passes both realistic fixtures after
the fused final upsampler (`.benchmarks/runs/20260927T021035Z`). Bedroom is
`0.976x` and groceries `0.969x`, yielding `0.973x` weighted latency. Weighted
external VRAM is `0.766x`, peak allocated device bytes `0.794x`, peak reserved
bytes `0.573x`, and host RSS `0.360x`. Maximum logit error is `2.575e-5`.

The generalized runner now captures and replays CUDA graphs for cached image
workloads while reporting capture time and persistent graph memory separately.
The four-pair graph `I3` run (`.benchmarks/runs/20260926T234506Z`) is `1.051x`
weighted: bedroom is `1.066x` and groceries `1.035x`, so both individual caps
pass. Weighted process VRAM is `0.701x`, peak allocated `0.702x`, peak reserved
`0.458x`, and host RSS `0.367x`; Rust capture averages about `15.2 ms` versus
PyTorch's `24.8 ms`.

The current four-pair graph `I4` run
(`.benchmarks/runs/20260927T021447Z`) confirms the fused kernel is capture-safe.
Bedroom is `1.065x`, cars `1.066x`, and the weighted ratio is `1.066x`, so both
individual latency caps now pass. Weighted process VRAM is `0.840x`, peak
allocated `0.562x`, peak reserved `0.590x`, and host RSS `0.362x`; Rust capture
averages about `12.0 ms` versus PyTorch's `21.9 ms`. Across implemented I3/I4
graph rows, weighted latency is approximately `1.057x`. These graph results
are additional evidence and do not replace the failing eager `I4` row.

Native cuDNN transposed convolution and fused channels-last CUDA `LayerNorm2d`
reduced the focused `A1/groceries` result to `918.90` versus `639.37 ms`
(`1.437x`), from the validated `2.262x` baseline. `I4/bedroom` likewise fell
from `1.926x` to `1.299x`. Both remain latency failures and require four-pair
promotion after further optimization; all measured memory and output-parity
gates pass.

The current four-pair 20-worker CPU `I3` failed baseline
(`.benchmarks/runs/20260927T015058Z`) pins both implementations to CPUs
`0–19`, uses the now-default fused residual/LayerNorm decoder path, and primes
the untimed encoder symmetrically before cached decoding. It is `2.616x`
weighted latency and `0.291x` RSS. Bedroom measures `2.587x` and groceries
`2.644x`. Bedroom's rich
positive/negative-click plus box output has `3.645e-5` mean error, slightly
above the `3e-5` gate, although maximum error passes and the thresholded mask
is exact; groceries passes. The runner's parity-failure override was used only
to collect all four pairs, so this remains a recorded latency and output-parity
failure rather than a passing CPU headline. The prior `2.465x` run and its
adjacent `2.677x` fusion-off control remain useful variation and attribution
evidence, not the current scorecard row.

Fresh pinned 20-worker CPU breadth runs cover I1 and I2 on the current shared
code. I1 (`.benchmarks/runs/20260927T051931Z`) is `0.943x` weighted latency
and `0.360x` RSS; all six fixture means pass, but groceries has a `1.466x`
PyTorch process spread and therefore fails the noise gate. The previously
problematic portrait fixture is stable in this run. I2
(`.benchmarks/runs/20260927T052356Z`) is `0.928x` latency and `0.350x` RSS,
but bedroom and portrait are noisy and bedroom mean logit error is `3.645e-5`
versus the `3e-5` gate, despite exact threshold masks. These are favorable
speed/memory results, not promoted passing rows.

One-pair feature attribution rules out the optional accelerators as the
parity-drift source. Removing CPU flash attention worsens I2/bedroom to
`2.355x`, `0.730x` RSS, and `3.948e-5` mean error
(`.benchmarks/runs/20260927T052618Z`). Keeping flash but removing glibc vector
math gives `1.262x`, `0.340x` RSS, and `4.729e-5` mean error
(`.benchmarks/runs/20260927T052714Z`). The default pair of features is both
faster and numerically closer and remains retained.

Pinned four-pair CPU I4 (`.benchmarks/runs/20260927T052804Z`) is the largest
current image-path failure: `3.624x` weighted latency and `0.427x` RSS.
Bedroom is `3.766x` (`546.89` versus `145.21 ms`) and cars is `3.488x`
(`524.35` versus `150.33 ms`). Both rows are stable and stay within the float
envelope. A synchronized boundary run at
`.benchmarks/runs/20260927T053112Z` attributes most of the excess to the
two-way transformer (`7366` versus `1915 ms` over 44 decodes) and mask
upscaler (`2178` versus `402 ms`), not prompt encoding.

A bounded batched-attention experiment exposed a real implementation gap but
was rejected. Routing B=4 short-query/4096-key decoder attention through
Candle's direct NHSD GEMM path reduced I4 from `3.624x` to `2.481x` across
four pairs (`.benchmarks/runs/20260927T054213Z`). However, cars mean logit
error rose from `2.67e-5` to `3.97e-5` and maximum error from `1.81e-4` to
`2.29e-4`, while RSS rose from `0.427x` to `0.733x`. A bounded two-pass variant
also failed parity. Both were removed. Copy-elision and fused normalization
experiments were neutral and were removed rather than accumulating complexity.
The retained path remains the pre-experiment parity-preserving implementation.

The first pinned four-pair CPU `I1` matrix
(`.benchmarks/runs/20260926T234919Z`) is diagnostic rather than promoted. Its
raw weighted ratios are `0.883x` latency and `0.361x` RSS, and all six fixture
rows pass their individual caps. However, one of four PyTorch portrait
processes measured `2571.99 ms` while the other three measured `600.06` to
`632.85 ms`; that single 4x outlier changes the portrait mean to `1106.34 ms`
and materially overstates the Rust advantage. Output parity passes. The row
will be repeated on a quiet host and does not update the CPU headline until an
independent stable run confirms it.

The refreshed four-pair GPU `I1` matrix (`.benchmarks/runs/20260926T210413Z`)
improves weighted latency from `1.056x` to `1.021x`. All six fixture rows pass
the `1.10x` individual cap, while the weighted `1.00x` gate remains narrowly
missed. Weighted external VRAM is `0.725x`, peak allocated bytes `0.791x`, peak
reserved bytes `0.677x`, and host RSS `0.347x`; maximum error is `2.575e-5`.

The new four-pair GPU `I2` baseline (`.benchmarks/runs/20260926T210728Z`)
measures `1.031x` weighted latency across bedroom, groceries, and portrait.
Individual ratios are `1.012x`, `1.034x`, and `1.046x`, so each passes while
the weighted gate narrowly fails. Weighted external VRAM is `0.713x`, peak
allocated bytes `0.790x`, peak reserved bytes `0.663x`, and host RSS `0.364x`.
Maximum error is `2.575e-5`.

After matching TorchVision's antialiased input resize, compacting AMG masks to
one byte per pixel before host transfer, parallelizing canonical RLE metadata,
and using F32 arithmetic for Candle's F32 CUDA bilinear kernel, the promoted
four-pair `A1` matrix (`.benchmarks/runs/20260926T220012Z`) improves weighted
latency from the original `2.262x` to `1.409x`. Groceries is `1.129x`; the
larger 2250x1500 cars image is `1.759x` and remains the important failure.
Weighted external VRAM is `0.571x`, peak allocated device memory `0.639x`, and
host RSS `0.454x`.

The refreshed antialiased six-fixture `I1` matrix
(`.benchmarks/runs/20260926T220158Z`) measures `1.063x` weighted latency; every
row remains below the `1.10x` individual cap. External VRAM is `0.703x`, peak
allocated memory `0.791x`, and host RSS `0.348x`. Maximum logit error is
`2.575e-5`.

The first complete four-pair `A2` default-32x32 baseline
(`.benchmarks/runs/20260927T024156Z`) measures `2889.54 ms` Rust versus
`2336.83 ms` PyTorch (`1.237x`). Host RSS is `0.335x`, external peak VRAM is
`0.684x`, peak allocated bytes are `0.894x`, and peak reserved bytes are
`0.673x`, so all memory gates pass. All 52 masks and stream lengths match, but
one mask differs by one threshold-boundary pixel (area 2004 versus 2003).
The sign difference is already present in decoder logits before bilinear
restoration. This is a promoted performance baseline but remains a recorded
strict-output-parity failure.

The current four-pair GPU `V1` initialization row measures `320.54` versus
`333.67 ms` (`0.961x`). External VRAM is `0.640x`, peak allocated memory is
`0.623x`, and host RSS is `0.273x`. It supersedes both the unified baseline's
`1.110x` row and an earlier isolated `0.424x` row whose much shorter Rust
boundary was not representative of the complete initialization path.

The promoted four-pair GPU `V3` three-object edit workflow
(`.benchmarks/runs/20260926T212849Z`) measures `23075.18` versus `22307.29 ms`
(`1.034x`). Compacting retained per-frame output state to the three tensors
needed by later edits reduced warmed Rust peak process VRAM from `2834 MiB` to
`2034 MiB`; the final ratios are `0.870x` external VRAM, `0.885x` peak
allocated, `0.862x` peak reserved, and `0.271x` host RSS. The corrected-frame
logit drift is deterministic at `0.108863` maximum and `0.005405` mean with
zero threshold-mask mismatches. It is checked by a V3-only `0.12` maximum
envelope so the stricter `0.10` ordinary-video limit remains unchanged.

The promoted four-pair `V4/bedroom128` lifetime baseline
(`.benchmarks/runs/20260926T231616Z`) stores CUDA temporal features as BF16 and
reuses the invariant memory-position tensor, matching upstream state lifetime.
It measures `26559.53` versus `26222.18 ms` (`1.013x`), host RSS `0.187x`,
external VRAM `1.062x`, peak allocated memory `1.051x`, and peak reserved
memory `1.065x`. Maximum selected-frame error is `0.00660`. Latency and every
individual memory cap pass, although the above-one memory ratios still weigh
against the required aggregate GPU memory gate.

The promoted four-pair GPU `L1` load-only row
(`.benchmarks/runs/20260926T222127Z`) measures `150.85` ms Rust versus
`1020.28` ms PyTorch (`0.148x`). External peak VRAM is `0.570x`, peak
allocated bytes `0.429x`, peak reserved bytes `0.418x`, and host RSS `0.345x`.
The implementations load their native checkpoint formats (safetensors and
`.pt`, respectively), so this measures each public implementation's real
first-ready construction path rather than forcing a shared parser.

The model-scaling harness originally charged the first-ever encoder invocation
to `S1`/`M1`, contrary to the suite's warmed-inference protocol and while `L1`
already owns construction/first-ready cost. After both implementations prime
the exact encoder route symmetrically, Small `S1`
(`.benchmarks/runs/20260927T013657Z`) is `1.038x`, down from the superseded
`1.275x`; all latency, memory, and parity cell caps pass. The corrected
Base-Plus/Large `M1` run (`.benchmarks/runs/20260927T013537Z`) is `1.015x`
weighted, down from `1.206x`: Base-Plus is faster at `0.989x`, and Large is
`1.042x`. Weighted external VRAM is `0.824x`, peak allocated bytes `0.887x`,
peak reserved bytes `0.623x`, and host RSS `0.402x`. Base-Plus maximum logit
error remains `4.387e-5`; Large remains inside its local `5e-4` envelope at
`3.595e-4`, with low mean error and exact threshold decisions. The aggregate
latency gates remain narrowly above `1.00x`, but the previous apparent large
encoder deficit was mostly a measurement-boundary error rather than steady-
state model scaling.

The current same-revision four-pair A1 rows are part of
`.benchmarks/runs/20260927T045104Z`. In addition to the earlier kernels and
packed-mask path, both AMG modes apply the predicted-IoU gate to native
256×256 logits and restore only survivors, decoder attention preserves
cuBLASLt's native sequence-major Q/K/V layout, large residual LayerNorm is
fused, and CUDA computes exact bilinear stability counts before restoring only
stability survivors. A1 is `0.978x` geometrically across fixtures: groceries
is `0.999x` and cars is `0.958x`; strict float, metadata, ordering, and RLE
parity pass. Its dedicated promotion run is
`.benchmarks/runs/20260927T044508Z` (`0.968x`). A2 is `0.948x` in the unified
run and `0.937x` in its dedicated promotion
`.benchmarks/runs/20260927T044637Z`; it retains only the previously localized
single threshold-boundary pixel mismatch. GPU AMG speed parity is established.
The matched operand diagnostic at `.benchmarks/runs/20260927T073705Z` now
localizes that pixel to decoder arithmetic: transformer inputs match within
`7.24e-9`, layer 0 within `1.20e-6`, layer 1 within `2.06e-6`, and the final
token-to-image attention plus LayerNorm within `1.15e-5`. The hypernetwork and
upscaler are downstream amplifiers rather than independent translation defects;
mask projection, interpolation, threshold packing, and RLE construction are
excluded. Disabling fused CUDA residual LayerNorm left the vectors bit-identical,
so it was rejected as a correctness fix.

The preceding four-pair A1 result (`.benchmarks/runs/20260927T034710Z`) combines
the shape-derived 32-query CUTLASS tile, the two direct NCHW kernels, and
on-device bit-packing of thresholded masks before host RLE generation. It also
skips the unused initial full-resolution restoration for mask-to-mask
refinement and explicitly drops every other initial decoder tensor before the
refinement batches. It is `1.215x` weighted latency: groceries is `1.127x`
and cars is `1.310x`; cars Rust latency is `1010.98 ms`. Weighted external
VRAM is `0.475x`, peak allocated device memory `0.586x`, and host RSS is
`0.345x`; strict output parity passes. It is superseded by the current unified
run above.

## CPU V2 temporal-attention and memory progress

The first current 20-worker V2 pair
(`.benchmarks/runs/20260927T055750Z`) measured `3.468x` latency and `1.026x`
RSS. A matched Rust profile attributed about 65 seconds to Candle's generic
grouped-convolution implementation splitting each 256-channel 7x7 depthwise
convolution into 256 one-channel tensors. The linked Candle fork now has a
native CPU-f32 depthwise path that writes only the final output. The ordinary
pair at `.benchmarks/runs/20260927T061844Z` improved to `3.154x`; output parity
was unchanged.

The larger remaining allocation was temporal attention's materialized
4096-by-up-to-28736 score matrix. Routing CPU RoPE attention through Candle's
bounded tiled flash kernel reduced the instrumented profile from 141.0 to
77.0 seconds and HWM from 2.96 to 1.60 GB. Replacing the generic RoPE graph
(two gathers, four products, two arithmetic tensors, and a stack) with one
parallel pairwise CPU-f32 pass reduced the profile again to 62.3 seconds.
The ordinary pair at `.benchmarks/runs/20260927T063729Z` is `0.982x` latency,
`0.569x` RSS, and stays within the video envelope at `0.004044` mean and
`0.06434` maximum error. PyTorch was slower than its nearby samples in that
pair, so this is a passing diagnostic, not a promoted headline; V2 still
requires the quiet four-pair repeat mandated by the plan.

A later repeat-aware CPU key-RoPE path removes per-layer frequency-table
repeats and writes rotated memory grids plus the unchanged object-pointer tail
into one destination. Four reverse-ordered Rust-only V2 pairs at
`.benchmarks/runs/20260927T-v2-repeat-aware-rope-ab` measure candidate times
of `51.461`, `52.607`, `48.396`, and `60.255 s`, versus control times of
`54.460`, `53.893`, `52.048`, and `54.139 s`. The noisy candidate outlier
leaves means at `53.180/53.635 s` (`0.9915x`), while medians are
`52.034/54.016 s` (`0.9633x`) and three of four matched pairs favor the new
path. Matched stage profiles at
`.benchmarks/runs/20260927T-v2-repeat-aware-rope-profile` show the causal local
effect: 124 cross-key RoPE calls fall from `1544.33` to `554.90 ms`
(`0.3593x`, 64.1% faster), while self-RoPE is effectively unchanged. All ten
full output artifacts have SHA-256
`6ded85132d763c3f92b9937d9945f08e34a218bdbc326bc5fce74c356a853207`.
The path is retained, but the clean `1.213x` Rust/PyTorch headline remains in
force until a new paired promotion run passes the noise gates. The subsequent
configured four-pair run at
`.benchmarks/runs/20260927T-v2-repeat-aware-rope-candidate` averages
`57.651 s` Rust versus `61.835 s` PyTorch (`0.9323x`) with `0.5792x` peak RSS
and unchanged passing parity. Rust spread passes at `1.082x`, but PyTorch's
`54.390--69.425 s` range is `1.276x`, narrowly failing the `1.25x` noise gate.
It therefore confirms that the candidate is viable end to end but remains a
diagnostic rather than replacing the clean headline.

The first quiet four-pair confirmation on the current pre-fuser binary is
`.benchmarks/runs/20260927T150000Z-v2-cpu-confirmation`: `1.079x` latency,
`0.583x` RSS, valid parity, and `1.104x`/`1.187x` Rust/PyTorch spreads. It
passes the individual `1.10x` cap but not the weighted `1.00x` target.

Matched current profiling then isolated memory encoding at `5.824 s` Rust
versus `0.835 s` PyTorch, despite Rust's image encoder being faster (`19.224`
versus `21.398 s`) and memory attention being close (`33.005` versus
`32.350 s`). A retained NCHW pointwise route reduces Rust's aggregate first
pointwise projection from `1.969` to `0.208 s` without an activation reorder.
A linked-Candle spatial-row depthwise traversal reduces aggregate depthwise
convolution from `1.490` to `0.174 s` without scratch or packing.

The four-pair candidate
`.benchmarks/runs/20260927T155000Z-v2-memory-fuser-candidate` reduces the clean
Rust mean from `62.712` to `59.041 s` (`-5.9%`) and passes parity, RSS, and
noise, but PyTorch shifted from `58.100` to `51.973 s`; the resulting `1.136x`
ratio fails both latency gates. A subsequent alternating Rust-only A/B was
invalidated by host contention (`1.533x` candidate spread). Therefore the
kernel changes are retained as measured Rust-side wins, while V2 speed parity
remains open and no passing headline is claimed.

The first CPU V4/bedroom128 lifetime diagnostic is archived at
`.benchmarks/runs/20260927T-v4-cpu-smoke`. This one-pair, zero-warmup feasibility
run measures `232.683 s` Rust versus `234.090 s` PyTorch (`0.9940x`) with
`0.7694x` peak RSS. After reset and session drop, current RSS is
`1,617,808 KiB` Rust versus `2,821,896 KiB` PyTorch (`0.5733x`). Selected
first/middle/final frame logits pass at `0.009653` maximum and `0.000303` mean
error with zero threshold mismatches. This establishes that the 128-frame
route, output contract, and reset measurement work on CPU, but it is not a
promoted baseline: V4 still requires four alternating pairs with its configured
warmup and a second independent suite run.

## CPU A1 large-batch progress

A matched CPU A1/groceries profile at
`.benchmarks/runs/20260927T070137Z` measured `2.893x` latency and localized the
gap to four batch-64 decoder calls: Rust transformer time was 17.5 seconds
versus 4.66 seconds for PyTorch, with another roughly 6.4 versus 1.6 seconds
in layout/upscaling work. A batch-at-least-32 attention route reduces the
transformer without changing ordinary I3/I4 decoding; a linked-Candle
schedule assigns independent transposed-convolution batches to Rayon workers,
uses single-thread GEMMs inside each task, and reuses one 4--8 MiB tile per
worker without copying the model input.

The focused ordinary groceries pair
(`.benchmarks/runs/20260927T070717Z`) improved Rust from 28.16 to 20.98 seconds,
kept strict AMG parity, and used `0.934x` PyTorch RSS. The subsequent full
four-pair/two-fixture run (`.benchmarks/runs/20260927T070842Z`) is `2.209x`
weighted latency and `0.666x` weighted RSS, but both rows fail the noise gate:
Rust spreads are `1.396x`/`1.455x` and PyTorch spreads are
`1.509x`/`1.958x`. The changes are retained with opt-out switches because the
synchronized stages and focused pair agree, but the noisy aggregate is
provisional and does not establish A1 parity.

## CPU repeated-session translation correction

Matched stage profiling of `R1/bedroom32` showed that session initialization
was already at parity: Rust averaged `832.43 ms` versus PyTorch `834.96 ms`.
Reset/drop was also close at `50.90` versus `41.88 ms`. The first interaction,
however, was `255.30 ms` in Rust versus `22.90 ms` in PyTorch because Rust ran
the temporal memory encoder during `add_new_points` and then ran it again in
propagation preflight. Upstream deliberately defers that encoder until
preflight so all object masks can first be consolidated.

The Rust interactive path now follows that lifetime: click and mask results retain the
low-resolution mask, object score, and pointer, and preflight performs the one
required memory encoding. The unprofiled one-pair check at
`.benchmarks/runs/r1-cpu-deferred-memory-smoke` improved `R1/bedroom32` to
`0.944x` latency (`8440.97` versus `8943.57 ms`) and `0.501x` RSS with unchanged
parity. A full `V2` propagation check at
`.benchmarks/runs/v2-cpu-deferred-memory-smoke` proves the encoder still runs:
Rust time is `55060.24 ms`, essentially unchanged from the prior `55298.17 ms`
optimized sample, while output error is unchanged. Its one-pair ratio is
`1.150x` because PyTorch varied materially between samples, so V2 and R1 still
need their required multi-pair confirmation.

The subsequent four-pair R1 run
`.benchmarks/runs/20260927T112000Z-r1-cpu-deferred-memory` is `0.960x` weighted
latency and `0.410x` weighted RSS. Image sessions are `0.900x`; video sessions
are `1.024x`. Rust spreads pass at `1.208x` and `1.070x`, while PyTorch spreads
of `1.346x` and `1.286x` make both rows noisy. This confirms the direction and
removes the earlier stable Rust-side video cost, but is diagnostic rather than
promoted evidence until a quiet repeat passes the process-spread gate.

## CPU decoder zero-copy upscaler input

The two-way transformer already returns a contiguous `B×(H·W)×C` image-token
tensor. The previous Rust translation permuted it to `B×C×(H·W)` and then
reshaped it, which forced a complete NCHW materialization before every first
transposed convolution. The retained CPU route instead views the same storage
as BHWC and permutes that to a zero-copy NCHW view. The linked Candle fused
2×2/stride-2 transposed convolution recognizes the resulting pixel-major
strides and feeds the existing GEMM directly, without an input staging copy.

Four-pair I3 candidate/control runs are archived at
`.benchmarks/runs/i3-zero-copy-candidate` and
`.benchmarks/runs/i3-zero-copy-control`. Rust improves from `49.42` to
`37.27 ms` (`-24.6%`), and the Rust/PyTorch ratio improves from `2.371x` to
`1.669x`. Both spreads pass, RSS is effectively unchanged, and candidate and
control have identical output error (`8.202e-5` maximum, `3.645e-5` mean).

The full four-pair I4 candidate keeps both fixture outputs within their fixed
envelopes and has clean Rust spreads; cars' PyTorch process is noisy, so that
run is regression evidence rather than a new headline. A one-pair practical
AMG cross-family smoke improves Rust groceries from the prior focused `20.98`
seconds to `19.57` seconds, uses `0.872x` PyTorch RSS, and preserves strict AMG
parity. The Candle pixel-major convolution unit test also passes against the
composed reference. Decision: retain; I3 and A1 remain above their speed gates.

The current-binary four-pair/two-fixture refresh is
`.benchmarks/runs/20260927T160000Z-i3-current-refresh`. Bedroom is `1.753x`
latency and `0.313x` RSS; groceries is `1.677x` and `0.322x`. The weighted
result is `1.715x` latency and `0.317x` RSS, with all four Rust and PyTorch
process spreads inside the noise gate. Groceries passes output parity;
bedroom retains the known `3.645e-5` mean error against the `3e-5` limit.

Matched profiling at
`.benchmarks/runs/20260927T160300Z-i3-current-stage-profile` attributes about
`529.3 ms` of Rust's 22 decoder calls to the transformer versus `256.9 ms` in
PyTorch. The two Rust transformer layers average `9.44`/`9.66 ms` per call,
while PyTorch averages `4.93`/`4.97 ms`; final attention is `79.6` versus
`30.0 ms` in aggregate. The upscaler is a smaller remaining gap. The next I3
round therefore targets reusable projection/layout work inside the
transformer, while the parity discrepancy remains an independent gate.

The post-upscaler-change diagnostic at
`.benchmarks/runs/20260927T270100Z-i3-after-upscaler-fusion` reproduces the
same bedroom error exactly and keeps groceries inside its envelope. Its
weighted latency is `1.815x`, but groceries' Python spread is `1.303x`, so it
does not replace the clean row above. A same-binary, four-pair Rust A/B of the
only shared behavior—the owned-buffer LayerNorm/GELU path—averages `36.38 ms`
fused versus `37.59 ms` composed (`-3.2%`) with byte-identical output. The
fusion is therefore retained; the cross-run ratio movement is host/reference
variation rather than evidence of a code regression.

## CPU P1 prompt-diversity baseline

The P1 harness now runs empty prompts, 1/8/32 deterministic points, a box-only
prompt, prior-mask correction, and multimask output against one cached bedroom
image. Both implementations time the whole sequence, while the parity artifact
concatenates all nine returned masks so no prompt shape can be hidden by an
aggregate checksum. Fifteen iterations put measured work above two seconds on
both implementations.

The four-pair baseline at
`.benchmarks/runs/20260927T114500Z-p1-prompt-diversity` measures `357.53 ms`
Rust versus `184.46 ms` PyTorch (`1.938x`) and `0.315x` RSS. Rust spread is
clean at `1.054x`; PyTorch spread is narrowly noisy at `1.270x`, so the row is
diagnostic rather than promoted. Max/mean error is `1.469e-4`/`2.002e-5`,
inside the fixed image envelope. The result confirms that the remaining CPU
decoder gap generalizes across prompt count, box, mask, and multimask routes.

## CPU P1 mixed-image batching

The `B1` harness now constructs heterogeneous normalized-image batches from
the frozen fixtures: batch 2 uses bedroom and portrait, and batch 4 uses all
four configured images. Both runners perform one true batched encoder call and
one matching batched one-point decoder call; fixture loading and model loading
remain outside measured inference.

The first apples-to-apples runs exposed two Rust dispatch gaps. Hiera initially
selected a batch-one flash entry point, then after that correctness fix its
three global-attention blocks still fell back to scalar row-wise attention.
Their kernels alone took `1105`, `945`, and `1082 ms` at batch 2. Linked
Candle's existing bounded GEMM/online-softmax kernel now indexes the leading
batch stride directly and schedules `(batch, head, query tile)` jobs. It uses
one shared output plus bounded per-worker score/accumulator tiles and does not
copy Q/K/V.

The retained route reduces batch-2 Rust inference from `4.188` to `1.581 s`
and batch 4 from `12.248` to `2.882 s` in the focused samples. The required
four-pair batch-2 repeat at
`.benchmarks/runs/20260927T125300Z-b1-batch2-repeat` passes all gates at
`0.844x` latency, `0.514x` RSS, and max/mean error
`5.722e-5`/`1.875e-5`. Batch 4 in
`.benchmarks/runs/20260927T125000Z-b1-blocked-batched` is also clean at
`0.758x` latency, `0.607x` RSS, and `9.155e-5`/`1.780e-5` error. Combining the
two independent rows gives approximately `0.800x` latency and `0.558x` RSS.
The decoder remains slower at batch 4 (`244.5` versus `82.5 ms`), but the
larger encoder win makes both complete scenarios faster than PyTorch.

After generalizing decoder asymmetric flash dispatch to batch 2 and above,
the four-pair regression run
`.benchmarks/runs/20260927T194000Z-b1-after-lowbatch-flash` passes every P1
gate. Batch 2 is `0.790x` latency and `0.455x` RSS; batch 4 is `0.762x` and
`0.606x`. Combined results are `0.776x` latency and `0.525x` RSS. Rust/Python
spreads and output parity pass on both rows. Batch-2 decoder time improves to
`59.5 ms`; batch 4 is `117.9 ms`. This supersedes the earlier B1 headline.

## CPU I4 low-batch asymmetric attention

The clean current-binary baseline at
`.benchmarks/runs/20260927T190000Z-i4-current-after-pe` exposed a stable
low-batch decoder gap: bedroom was `2.737x`, cars `2.819x`, and weighted
latency was `2.778x`. Both Rust and Python spreads passed, as did strict output
parity and `0.405x` weighted RSS.

I4 uses prompt batch 4. Rust previously reserved its batched NHSD decoder
attention route for AMG batches of at least 32, so I4 materialized the generic
attention expression. The retained dispatch now uses batched flash for any
batch of at least two when one sequence is short (`<=16`) and the other is
long (`>=512`). Candle's short-query GEMM route now accepts those independent
batch entries directly; each `(batch, head)` task owns bounded score scratch
and Q/K/V remain in their existing storage.

Matched profiling at
`.benchmarks/runs/20260927T192000Z-i4-batched-flash-profile` reduces two-decoder
transformer time from `294.4` to `142.9 ms`. Aggregate token-to-image attention
falls `142.6→32.9 ms`, image-to-token `80.9→39.6 ms`, and end-to-end Rust
`476.2→332.7 ms`, with passing parity.

The four-pair/two-fixture candidate is
`.benchmarks/runs/20260927T193000Z-i4-batched-flash-candidate`:

| Fixture | Rust mean | PyTorch mean | Latency ratio | RSS ratio |
| --- | ---: | ---: | ---: | ---: |
| bedroom | `293.22 ms` | `165.85 ms` | `1.768x` | `0.406x` |
| cars | `276.00 ms` | `156.58 ms` | `1.763x` | `0.412x` |
| weighted | — | — | `1.765x` | `0.409x` |

Rust spreads are clean and both parity envelopes pass. Bedroom's PyTorch
spread is `1.348x`, so the candidate is retained evidence rather than a final
promoted I4 row. Rust nevertheless improves `33.5%` on bedroom and `39.2%` on
cars versus the adjacent clean baseline. I4 remains above its CPU speed caps.

## CPU P1 worker-scaling coverage

`W1` now expands the existing I1, I3, V2, and A1 operations at independently
pinned 8- and 20-worker settings. Each child process receives matching Rayon,
OpenMP, MKL, OpenBLAS, PyTorch, and affinity limits; rows preserve their source
cell and worker count. `--source-cell` allows the expensive families to be run
and retried independently. Summaries now evaluate, rather than merely repeat,
the applicable latency, RSS, parity, and noise gates.

The four-pair bedroom image subset at
`.benchmarks/runs/20260927T120000Z-w1-image-workers` measures I1 at `1.007x`
with 8 workers and a noisy `0.734x` with 20. I3 is a clean `1.303x` at 8 and
`1.817x` at 20; its known `3.645e-5` mean-logit error fails the `3e-5` gate.
Weighted image-subset latency is `1.150x` and RSS is `0.315x`, but the noisy
20-worker I1 row prevents promotion.

One-pair family diagnostics add the remaining coverage. A1/groceries is
`1.251x` at 8 workers and `1.327x` at 20, with `0.816x`/`0.853x` RSS and exact
AMG parity (`.benchmarks/runs/w1-a1-workers-smoke`). V2 is `1.339x` at 8 and
`0.604x` at 20 with `0.566x`/`0.599x` RSS and valid video parity
(`.benchmarks/runs/w1-v2-workers-smoke`). PyTorch V2 slowed from 78.8 to 108.4
seconds at 20 workers while Rust improved from 105.5 to 65.5 seconds, so those
single pairs establish coverage and a scaling lead to investigate, not a
promoted score.

## CPU A1 fused restoration and stability counting

The practical AMG path previously restored every IoU-surviving 256×256 mask
to the full source resolution, allocated those logits, and then reduced them
to the two threshold counts used by the stability score. On the 1500×2250
cars fixture, a matched profile measured `4.432 s` restoring these candidates
and `7.090 s` calculating stability across three refinement batches. Most
candidates were then rejected.

The retained CPU custom operation now evaluates Candle's exact
`upsample_bilinear2d(..., false)` coordinates in place and emits only two
`u32` counters per mask. It shares the input storage, parallelizes independent
masks, and never allocates or copies the discarded full-resolution logits.
Only final stability survivors are restored. The composed implementation can
be selected for A/B with `SAM2_CPU_FUSED_AMG_STABILITY=0`.

The exact-equivalence unit test compares the fused counters with Candle's
materialized bilinear output and passes. In the matched cars profile at
`.benchmarks/runs/20260927T170000Z-a1-cpu-fused-stability-profile`, the three
fused count calls total `269.9 ms`, replacing `11.522 s` of restore-plus-score
work. End-to-end Rust time falls from `33.33 s` in the matched pre-change
profile to `21.66 s`, with identical parity statistics.

The clean four-pair, two-fixture promotion run is
`.benchmarks/runs/20260927T171000Z-a1-cpu-fused-stability-candidate`:

| Fixture | Rust mean | PyTorch mean | Latency ratio | RSS ratio |
| --- | ---: | ---: | ---: | ---: |
| groceries | `19.750 s` | `10.969 s` | `1.801x` | `0.860x` |
| cars | `21.858 s` | `13.426 s` | `1.628x` | `0.432x` |
| weighted | — | — | `1.712x` | `0.610x` |

All process spreads pass the `1.25x` noise gate. Strict AMG parity remains
unchanged: groceries max/mean error is `4.768e-7`/`1.421e-9`, and cars is
`2.050e-5`/`2.913e-9`. Against the preceding clean four-pair baseline, Rust
improves `11.5%` on groceries and `36.4%` on cars; weighted latency ratio
improves from `2.242x` to `1.712x`, while weighted RSS is effectively flat
(`0.6090x` to `0.6097x`). A1 still fails the CPU latency gates, so this is a
retained improvement rather than a parity claim.

The next retained translation fix removes another batch-sized materialization.
AMG expands a single image's positional encoding to one entry per prompt, but
every expanded value is identical and all downstream uses are additions that
already support broadcasting. Rust previously expanded the NCHW PE to batch
64 and then packed it again to NHWC tokens, materializing roughly 1 GiB per
decoder call. The CPU route now keeps PE at batch one, caches its 4 MiB token
layout once, and broadcasts directly during the additions. The prompt-varying
image activation is still materialized exactly once.

In the matched profile
`.benchmarks/runs/20260927T184000Z-a1-broadcast-pe-profile`, aggregate
transformer setup falls from `4.282 s` to `0.130 s` across four decoder calls;
cars falls from `21.657 s` to `15.785 s`, and strict parity is identical. The
four-pair promotion at
`.benchmarks/runs/20260927T185000Z-a1-broadcast-pe-candidate` records:

| Fixture | Rust mean | PyTorch mean | Latency ratio | RSS ratio |
| --- | ---: | ---: | ---: | ---: |
| groceries | `14.022 s` | `10.135 s` | `1.383x` | `0.788x` |
| cars | `17.015 s` | `13.249 s` | `1.284x` | `0.402x` |
| weighted | — | — | `1.333x` | `0.563x` |

All process spreads and parity checks pass. Relative to the immediately prior
four-pair candidate, Rust improves `29.0%` on groceries and `22.2%` on cars;
the weighted latency ratio improves `1.712x→1.333x` and weighted peak RSS
improves `0.610x→0.563x`. A1 remains above its latency caps, but its former
setup/materialization gap is closed.

## CPU V2 post-fuser confirmation

The four-pair current-binary confirmation at
`.benchmarks/runs/20260927T181000Z-v2-current-confirmation` measures Rust at
`51.540 s` versus PyTorch at `46.662 s`, or `1.105x`, with `0.678x` peak RSS.
Rust spread is clean at `1.133x`, output parity passes at `0.064376` maximum
and `0.004047` mean error with zero threshold mismatches, and RSS passes both
gates. PyTorch spans `40.838--55.921 s` (`1.369x`), so the row is noisy and
cannot replace the clean headline.

This nevertheless strengthens the retained memory-fuser evidence: Rust falls
from the clean pre-change `62.712 s` mean to `51.540 s`. In the matched
post-fix profile, Rust is faster in every broad measured stage—image encoder
`19.174 vs 24.136 s`, memory attention `32.988 vs 38.032 s`, mask decoder
`1.308 vs 4.297 s`, and memory encoder `2.259 vs 2.795 s`. The profile's
PyTorch process is itself slow, so those comparisons are attribution rather
than a speed headline. V2 is best classified as near parity with unstable
reference timing; it needs another quiet confirmation, not an unsupported
kernel change.

The clean four-pair post-BF16-memory baseline at
`.benchmarks/runs/20260927T220000Z-v2-after-bf16-memory` supersedes that noisy
diagnostic. Rust averages `51.253 s` versus `42.238 s` PyTorch (`1.213x`) with
`0.583x` peak RSS. Rust/PyTorch spreads pass at `1.112x`/`1.172x`. Matching
upstream's CPU BF16 temporal-memory storage reduces maximum/mean error from
`0.064376`/`0.004047` to `0.002213`/`9.204e-5`, with zero threshold
mismatches. Rust itself is `0.6%` faster than the prior diagnostic mean, so
the newly established gap comes from a stable, faster PyTorch distribution
rather than a regression from the translation correction. V2 remains a
latency target despite clean parity and RSS.

The current matched profile at
`.benchmarks/runs/20260927T273000Z-v2-current-matched-profile` confirms that
image encoding is tied (`18.163 s` Rust versus `17.945 s` PyTorch), while
memory attention is the main current gap (`29.104 s` versus `20.911 s`). A
CPU memory-MLP experiment fuses ReLU into the first owned linear destination
and the residual into the second. In
`.benchmarks/runs/20260927T280000Z-v2-memory-mlp-fusion-profile`, its matched
MLP stages fall from `3.844 s` to `1.957 s`, and Rust temporal/head time falls
from `33.290 s` to `31.889 s`. Parity passes at `0.002441` maximum and
`0.000213` mean error with zero threshold mismatches.

The four-pair candidate
`.benchmarks/runs/20260927T283000Z-v2-memory-mlp-fusion-candidate` is not
promotable: Rust/Python spreads are `1.270x`/`1.388x` on a host with load
average above 20. Its apparent `0.854x` ratio is therefore not a headline,
and the noisy Rust mean (`53.190 s`) does not improve the clean default mean
(`51.253 s`). The implementation remains available only with
`SAM2_CPU_MEMORY_MLP_FUSION=1` pending a quiet same-binary confirmation; the
default V2 row remains `1.213x`.

## CPU V3 BF16 temporal-memory translation correction

The first current CPU V3 pair at
`.benchmarks/runs/20260927T200000Z-v3-current-cpu` exposed a translation
failure before a performance baseline could be promoted. Rust measured
`180.072 s` versus `187.439 s` PyTorch (`0.961x`) with `0.655x` peak RSS, but
selected logits failed at `0.268817` maximum, `0.012666` mean error, and two
threshold mismatches. The first and final selected outputs were close; the
middle forward output contained the drift.

The reusable `--profile-v3-frames` diagnostic added to the suite streams the
initial-forward 256x256 logits one mask at a time, avoiding a full-video
in-memory copy. The paired run at
`.benchmarks/runs/20260927T203000Z-v3-frame-diagnostic` localized the failure
to recurrent object-2 state: frame 25 reached `0.272742` maximum and
`0.110398` mean low-resolution error, while frame 0 had no threshold
difference. Disabling low-batch decoder flash produced byte-identical output.
Disabling temporal CPU flash made the workflow `3.2x` slower (`580.468 s`)
and worsened maximum low-resolution error to `0.472395`, so neither flash path
caused the drift.

Upstream `SAM2VideoPredictor` stores mask-memory features as BF16 on every
device, including CPU, and promotes only selected memories for attention.
The Rust interactive predictor had translated that lifetime policy only for
CUDA and retained F32 memories on CPU. Applying BF16 at the high-level video
state boundary, while leaving the low-level `Sam2Base` API in its native
dtype, reduces selected-output maximum/mean error to
`0.004349`/`0.000220`, removes both threshold mismatches, and reduces
frame-25/object-2 error to `0.003154` maximum and `0.001258` mean. Across all
32 frames, low-resolution maximum error is `0.007757`.

The four-pair candidate at
`.benchmarks/runs/20260927T210000Z-v3-bf16-memory-candidate` averages
`193.144 s` Rust versus `203.177 s` PyTorch (`0.951x`) and `0.607x` peak RSS.
Rust spread passes at `1.217x`, and all four pairs repeat the same passing
parity statistics with zero restored-mask threshold mismatches. PyTorch spans
`169.365--249.541 s` (`1.473x`), so this is retained translation evidence but
not a clean promoted latency row. A quiet-host four-pair confirmation remains
required.

## CPU L1 current load baseline

The current four-pair load/first-ready run at
`.benchmarks/runs/20260927T223000Z-l1-current-cpu` measures `142.823 ms` Rust
versus `1083.337 ms` PyTorch (`0.132x`) with `0.276x` peak RSS and exact
structural parity. Rust spread passes at `1.149x`; PyTorch spans
`943.029--1326.650 ms` (`1.407x`), so the row is favorable but noisy and is
not promoted. As specified by the plan, this compares each public
implementation's native checkpoint path—safetensors for Rust and `.pt` for
PyTorch—rather than forcing an artificial shared parser.

## CPU A2 current baseline and matched attribution

The clean four-pair upstream-default AMG run at
`.benchmarks/runs/20260927T231000Z-a2-current-four-pair` measures `47.076 s`
Rust versus `41.580 s` PyTorch (`1.132x`) with `0.856x` peak RSS. Rust and
PyTorch spreads pass at `1.139x` and `1.163x`. Maximum/mean numeric error is
`2.980e-6`/`4.329e-9`, and the observed threshold-mask mismatch count is zero.
The configured A2 fixture deliberately does not yet require exact threshold
masks, so the observation does not silently tighten that gate. A2 misses its
`1.10x` cell-latency cap and the standalone `0.80x` weighted RSS target, while
passing the `1.00x` per-cell RSS cap.

The matched diagnostic at
`.benchmarks/runs/20260927T233000Z-a2-stage-profile` keeps the image encoder
close (`742` versus `725 ms`) and attributes the 16 decoder calls at matched
boundaries. Rust transformer time is roughly `21--22 s`, close to PyTorch's
`20.880 s`. PyTorch's five upscaling modules total about `6.95 s`; Rust's
corresponding stages total about `16.2 s`. In particular, the second
`64→32`, 2x2/stride-2 transposed convolution is roughly `9.3 s` in Rust versus
`0.889 s` in PyTorch. Filtering, stability, RLE, and NMS are comparatively
small. PyTorch preserves channels-last strides through both upconvolutions
and normalization, whereas Rust currently materializes NCHW output after the
first upconvolution. The next shared decoder experiment therefore targets
that layout boundary rather than A2-specific filtering.

The retained batch-one residual/GELU correction supersedes that latency row.
In `.benchmarks/runs/20260927T260000Z-a2-broadcast-residual-candidate`, Rust
averages `36.144 s` versus `38.972 s` PyTorch (`0.927x`) across four pairs.
Spreads are `1.059x`/`1.067x`, peak RSS is `0.854x`, and parity remains
`2.980e-6` maximum / `4.329e-9` mean with zero observed threshold mismatches.
A2 therefore passes its speed and per-cell RSS caps; the standalone weighted
RSS target of `0.80x` remains open.

The affected A1 regression at
`.benchmarks/runs/20260927T263000Z-a1-after-broadcast-residual` improves from
`1.333x` to `1.049x` weighted latency. Groceries is `1.035x`, cars `1.064x`,
and both pass their `1.10x` cell caps with clean spreads and parity. Weighted
RSS is `0.566x`. A1's strict weighted latency target remains open even though
neither constituent row now exceeds its per-cell cap.

## Reproduction

```console
python3 tools/prepare_cpu_suite_fixtures.py
python3 tools/run_parity_suite.py --cell I1 --device cpu --threads 20 --repeats 4
python3 tools/run_parity_suite.py --cell B1 --device cpu --threads 20 --repeats 4
python3 tools/run_parity_suite.py --cell W1 --device cpu --repeats 4
python3 tools/run_parity_suite.py --cell V3 --device cpu --threads 20 --repeats 1 --warmups 0 --profile-v3-frames --allow-parity-failure
```

Raw local runs are written beneath `.benchmarks/runs/` and ignored by Git.
Archive accepted run directories separately before regenerating this file.
