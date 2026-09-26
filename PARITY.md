# SAM2.1 Hiera-Tiny parity record

## Optional checkpoint and media integrations

The completed convenience-I/O boundary is feature-gated so the default model
build has no network or codec dependency. Its focused checks are:

```bash
cargo test --lib --features image-decoding media::tests
cargo test --lib --features video-decoding media::tests
cargo test --lib --features huggingface huggingface::tests
```

Recorded 2026-09-24: numeric JPEG ordering and RGB dimensions passed; mixed
frame dimensions were rejected; and FFmpeg decoded two tightly packed RGB8
frames from upstream's `05_default_juggle.mp4`, retaining its 1280×720 original
geometry. The Hugging Face mapping test covers all four published SAM2.1 Hiera
repository/file pairs without requiring network access. Actual downloads use
the Hub client's content-addressed cache and can be pinned to a revision or
restricted to cached files. The ignored network test was also run explicitly:
it downloaded the Tiny repository's `README.md` into a fresh temporary cache,
then resolved the identical path with network access disabled.

This record covers the first native Candle static-image path. It uses upstream
commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, the published
`sam2.1_hiera_tiny.pt` checkpoint, a zero `1x3x1024x1024` RGB-normalized test
input, and one positive point at `(512, 512)`.

The reference fixture deliberately calls `build_sam2(...,
apply_postprocessing=False)`: the normal Python builder enables optional
dynamic multimask fallback, whereas this initial Rust predictor records the
deterministic single-mask-token path.

```bash
python3 tools/convert_checkpoint.py checkpoints/sam2.1_hiera_tiny.pt checkpoints/sam2.1_hiera_tiny.safetensors
python3 tools/generate_parity_reference.py
cargo run --release --bin parity-probe -- \
  checkpoints/sam2.1_hiera_tiny.safetensors \
  checkpoints/sam2.1_hiera_tiny_parity.safetensors
```

CPU float32 result recorded 2026-09-14:

| Tensor | Max absolute error |
| --- | ---: |
| `vision_features` | `5.86e-6` |
| sparse prompt embeddings | `1.20e-7` |
| final low-resolution mask logits | `1.311e-5` |

The expected mask-logit magnitude is `2.57461`, so this is a relative maximum
error of approximately `5.1e-6`.

SAM2.1 Hiera-Small was independently checked with the same zero-image/point
fixture through the generic variant loader. Its maximum errors were `1.276e-5`
for vision features and `1.454e-5` for final low-resolution mask logits:

```bash
python3 tools/generate_parity_reference.py small
cargo run --release --bin parity-probe -- \
  checkpoints/sam2.1_hiera_small.safetensors \
  checkpoints/sam2.1_hiera_small_parity.safetensors small
```

SAM2.1 Hiera-Base-Plus was also loaded from the published checkpoint after
validating its distinct 14x14 positional background. Its final low-resolution
mask-logit maximum error is `4.816e-5` (vision features `7.087e-5`).

SAM2.1 Hiera-Large was loaded using its distinct `[8, 4, 16, 8]` Hiera window
schedule. The zero-image fixture records `2.906e-3` vision-feature and
`7.027e-4` final-mask-logit maximum errors on CPU float32; this deeper 48-block
backbone accumulates more floating-point reduction error than the other three
variants while retaining the same output shape and checkpoint key alignment.

## Two-frame temporal record

The native `Sam2Base` / `Sam2VideoPredictor` path has a separate deterministic
two-frame fixture: frame 0 is a conditioning click at `(512, 512)` and frame 1
is propagated from its spatial memory and object-pointer suffix. It verifies
the actual fused 64x64 temporal feature in addition to memory output (the
public frame-1 mask in this zero-image fixture is intentionally no-object
suppressed by both implementations).

```bash
python3 tools/generate_video_parity_reference.py
cargo run --release --bin video-parity-probe -- \
  checkpoints/sam2.1_hiera_tiny.safetensors \
  checkpoints/sam2.1_hiera_tiny_video_parity.safetensors
```

CPU float32 result recorded 2026-09-14:

| Tensor | Max absolute error |
| --- | ---: |
| frame-0 low-resolution mask logits | `1.347e-5` |
| frame-0 memory features | `2.313e-5` |
| frame-0 object pointer | `1.61e-6` |
| frame-1 memory-conditioned visual features | `1.419e-5` |
| frame-1 memory features | `2.122e-5` |

## R1 decoder and mask-postprocessing leaves

The first continuation batch is checked against deterministic tensors produced
by the pinned Python implementation. The fixture covers stable and unstable
dynamic single-mask selection, 8-connected labels and per-pixel component
areas, score-hole filling, both upstream mask-to-box variants (including their
different empty-mask sentinels), hole/island removal, and point concatenation.

```bash
python3 tools/generate_r1_parity_reference.py
cargo run --release --bin r1-parity-probe
```

CPU result recorded 2026-09-24: every integer/bool output matched exactly and
every reported floating-point maximum absolute error was `0`. The builder-level
typed decoder defaults now match upstream postprocessing (`delta = 0.05`,
`threshold = 0.98`), while the low-level `MaskDecoder::load` remains the raw
no-postprocessing constructor used by the original component parity probes.

## R2 transforms and AMG utilities

The second continuation batch uses constant RGB images with odd landscape and
portrait dimensions, batched coordinates and boxes, simultaneous hole and
sprinkle cleanup, two crop layers, interior/exterior crop-edge cases, and
column-major uncompressed/COCO-compressed RLEs.

```bash
python3 tools/generate_r2_parity_reference.py
cargo run --release --bin r2-parity-probe
```

CPU result recorded 2026-09-24: all integer metadata, crop geometry, edge
classification, RLE counts/areas/compressed bytes, decoded masks, and uncropped
outputs matched exactly. The largest floating-point error was `9.54e-7` after
bilinear mask restoration (`4.77e-7` for coordinate transformation); normalized
image batches and all other reported values matched exactly.

## R3 cached and batched image prediction

The third continuation fixture uses the published Hiera-Tiny checkpoint and
the upstream `SAM2ImagePredictor`. It covers a cached single image and a
two-image batch with opposite aspect ratios; repeated prompt batches; mixed
positive/negative points, boxes, and prior masks; an empty prompt; single- and
multimask selection; logits and thresholded output; image embeddings; object
scores; cache reuse; and reset.

```bash
python3 tools/generate_r3_parity_reference.py
cargo run --release --bin r3-parity-probe
```

CPU result recorded 2026-09-24: thresholded masks, cache reuse, and reset
matched exactly. Maximum absolute errors were `1.25e-5` for cached embeddings,
`5.06e-5` for low-resolution logits, `2.51e-5` for restored logits, `3.79e-6`
for IoU predictions, and `1.34e-5` for object-score logits. This fixture also
identified and corrected the missing image-predictor `no_mem_embed` addition
and the decoder's repeated-image batch reshape.

## R4 full automatic mask generation

The fourth continuation fixture uses a resized real photograph and checks a
permissive batched grid, explicit point grids, batched mask-to-mask refinement,
small-region cleanup, one crop layer with an independent crop-NMS threshold,
and the complete default 32×32 grid. Every retained proposal is compared in
reference order for its point, crop, XYWH box, area, IoU, stability, and full
uncompressed mask RLE. Separate unit coverage checks binary, uncompressed-RLE,
and compressed COCO-RLE output modes.

```bash
python3 tools/generate_r4_parity_reference.py
cargo run --release --bin r4-parity-probe
```

CPU result recorded 2026-09-24: proposal counts, ordering, boxes, points,
crops, areas, stability scores, and every RLE matched exactly in all six cases.
The largest IoU error was `8.17e-6` on the full default grid. This parity probe
is not a performance comparison: Python reuses one loaded model, while the
Rust probe deliberately reconstructs generators with distinct typed configs.

## R5 video-state primitives

The fifth continuation fixture serializes the same state-editing sequence in
Python and Rust: stable object insertion, index compaction after removal,
append/replace point prompts, point-to-mask replacement, temporary and
consolidated outputs, forward/reverse tracking metadata, bounded image-feature
eviction, surrounding non-conditioning-memory removal, and multi-object
non-overlap constraints.

```bash
python3 tools/generate_r5_state_reference.py
cargo run --release --bin r5-state-probe
```

CPU result recorded 2026-09-24: every object ID/index, prompt tensor, retained
frame index, direction flag, mask input, cache index, and constrained mask
score matched exactly.

## R6 interactive video prediction

The sixth continuation fixture runs the published Hiera-Tiny checkpoint on a
deterministic three-frame, two-object video. It covers initial clicks, preflight
memory encoding, full forward propagation, a correction click on an already
tracked frame, bounded reverse propagation, prompt clearing, object removal and
its affected-frame output, and session reset.

```bash
python3 tools/generate_r6_video_reference.py
cargo run --release --bin r6-video-probe
```

CPU result recorded 2026-09-24: object IDs, frame orders, affected frame indices,
and reset state matched exactly. The largest restored-mask error was `4.60e-4`,
the largest low-resolution-logit error was `8.89e-4`, and the largest object
pointer error was `5.77e-5`. Stored memory features matched within `7.83e-3`:
the Python predictor deliberately stores these tensors as BF16, while Candle's
homogeneous CPU attention path retains those same BF16-quantized values in an
F32 tensor. The fixture also compares the frame-1 memory-conditioned features
directly (`3.68e-5` maximum error), isolating temporal attention from decoding.
The same complete fixture passes on the Quadro RTX 5000 CUDA path. Its largest
restored-mask error was `4.65e-4`, low-resolution-logit error `8.25e-4`, and
pointer error `5.94e-5`; BF16-quantized stored memory differed by at most
`1.5625e-2`. Direct binary mask input matched exactly for restored masks,
256×256 logits, and its no-object-gated pointer on both CPU and CUDA.

## R7 checkpoint loading and manifests

Published PyTorch archives now load directly through Candle's `model`
state-dict reader; safetensors conversion is optional. Checked-in manifests
cover every key and shape for all four SAM2.1 variants and are validated before
model construction.

```bash
cargo run --release --bin r7-checkpoint-probe -- \
  tiny checkpoints/sam2.1_hiera_tiny.pt checkpoints/sam2.1_hiera_tiny.safetensors \
  small checkpoints/sam2.1_hiera_small.pt checkpoints/sam2.1_hiera_small.safetensors \
  base-plus checkpoints/sam2.1_hiera_base_plus.pt checkpoints/sam2.1_hiera_base_plus.safetensors \
  large checkpoints/sam2.1_hiera_large.pt checkpoints/sam2.1_hiera_large.safetensors
```

CPU result recorded 2026-09-24: PyTorch and safetensors indexes matched exactly
for Tiny (471 tensors), Small (519), Base-Plus (615), and Large (903). Running
each `.pt` archive through full `Sam2Base` construction also succeeded. Running
the R3 predictor fixture directly from `sam2.1_hiera_tiny.pt` reproduced the
same errors recorded for the safetensors loader, including exact thresholded
masks and cache behavior. Unit coverage also verifies that missing,
wrong-shaped, and unexpected keys are reported together and that a truncated
PyTorch archive fails while indexing, before model inference starts.
