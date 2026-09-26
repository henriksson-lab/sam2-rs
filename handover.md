# SAM2 Rust translation handover

> Historical design note: this file records the original translation method
> and batch structure. Its starting-state description is no longer current.
> See [TRANSLATION.md](TRANSLATION.md) for implemented coverage, the exact
> remaining function map, parity gates, and the live completion checklist.

## Scope and starting state

Build a proper native Rust translation of Meta’s **Segment Anything Model 2** (SAM2), using Candle as the first tensor backend. Preserve the reference model’s image and video/z-stack APIs, published checkpoint compatibility, and numerical behavior before optimizing or redesigning it.

`sam2-rs` is currently an empty Git repository containing one untracked checkout, `segment-anything/`. That checkout is **SAM v1**, identifiable by `build_sam.py`, `segment_anything/modeling/image_encoder.py`, and the absence of Hiera and video-memory modules. It is not the source to translate as SAM2.

Before writing model code, add a pinned checkout of [Meta’s SAM2 reference](https://github.com/facebookresearch/segment-anything-2) and record the exact commit in this repository. Keep the SAM v1 checkout separate: it is useful to compare older prompt/mask-decoder conventions, but must not silently become the implementation target.

Start with SAM2.1 Hiera-Tiny. It is the smallest published checkpoint while exercising the complete architecture: Hiera image encoder, FPN neck, prompt encoder, mask decoder, object pointers, memory encoder and memory attention. Expand to small, base-plus and large only after Tiny loads and runs through the same code paths.

## Rules for the translation

- Translate Python functions one-for-one when possible. Preserve helper boundaries, control flow, tensor layouts, constants and operation order until numerical parity has been checked.
- Use Rust snake case for functions and methods. Use idiomatic Rust PascalCase for type names, but preserve original checkpoint key paths in the loader.
- Do not fuse “obvious” operations or replace reference algorithms with convenience APIs during the initial port. Those changes belong in a later, benchmarked optimization pass.
- Translate bottom-up in batches. A batch may include a whole small source file when it is internally cohesive. Independent leaves within a batch should be assigned to separate agents/workers; integration happens only after each leaf compiles.
- Save comprehensive numeric/model tests until the final parity phase, as requested. Still run `cargo fmt`, `cargo check`, and focused shape/manifest checks after every batch. A broken checkpoint mapping discovered late invalidates all later work.
- Keep image-only SAM2 usable before porting video memory. The z-slice use case can then build on `Sam2VideoPredictor`, but SAM2 video propagation is not native volumetric 3D segmentation.

## Backend decision

Use **Candle** for the first backend. The existing Cellpose-RS code is the best local reference for Candle integration, not for SAM2 architecture:

- `cellpose-rs/src/network.rs`: model config, `VarBuilder` construction, checkpoint key/shape manifest, raw checkpoint inspection, `.pth`/safetensors loading, dtype/device policy, encoder timing and batched-forward probes.
- `cellpose-rs/src/transforms.rs`: image conversion, normalization, resize and deterministic tiling conventions.
- `cellpose-rs/src/core.rs`: device ownership and tiled execution patterns.
- Blockflow’s Cellpose adapter: concurrency ownership and decomposition-aware inference concerns.

Cellpose’s executable CP-SAM uses Candle’s `ImageEncoderViT`, but it is a SAM-v1-derived encoder with Cellpose-specific weights and geometry. It does **not** implement SAM2 Hiera, FPN or video memory. Reuse its loading/API ideas when stuck; do not reuse it as the SAM2 model definition.

The model API must not know whether a checkpoint came from PyTorch `.pt` or a one-time safetensors conversion. Put that choice behind `checkpoint.rs`; validate a raw key-to-shape index before constructing modules.

## Proposed Rust layout

```text
src/
  lib.rs
  error.rs
  config.rs
  checkpoint.rs
  transforms.rs
  ops/
    mod.rs
    attention.rs
    conv.rs
    layer_norm_2d.rs
    mlp.rs
    position_encoding.rs
    rope.rs
  modeling/
    mod.rs
    hiera.rs
    fpn_neck.rs
    prompt_encoder.rs
    two_way_transformer.rs
    mask_decoder.rs
    sam2_base.rs
    memory_encoder.rs
    memory_attention.rs
  predictor.rs
  automatic_mask_generator.rs
  bin/sam2.rs                 # optional, last
tests/
  fixtures/                   # fixed images, prompts and reference tensors
  ...                         # add parity suite in the final phase
```

Core public types should be explicit rather than Python dictionaries:

```text
Sam2Config
CheckpointIndex
Hiera
FpnNeck
PromptEncoder
TwoWayTransformer
MaskDecoder
Sam2Base
Sam2ImagePredictor
MemoryEncoder
MemoryAttention
VideoInferenceState
BackboneOutput
SamHeadOutput
```

## Python-to-Rust naming map

Keep field-to-checkpoint mapping separate from Rust spelling. For example, a Rust `hiera` field may load `image_encoder.trunk.blocks.0.attn.qkv.weight` unchanged.

| Python reference name | Rust name |
|---|---|
| `SAM2Base` | `Sam2Base` |
| `SAM2ImagePredictor` | `Sam2ImagePredictor` |
| `SAM2VideoPredictor` | `Sam2VideoPredictor` |
| `MLP` | `Mlp` |
| `FpnNeck` | `FpnNeck` |
| `RoPEAttention` | `RopeAttention` |
| `CXBlock` | `CxBlock` |
| `MaskDownSampler` | `MaskDownsampler` (or `MaskDownSampler` if mirroring paths is clearer) |
| `forward_image` | `forward_image` |
| `prepare_backbone_features` | `prepare_backbone_features` |
| `prepare_memory_conditioned_features` | `prepare_memory_conditioned_features` |
| `encode_new_memory` | `encode_new_memory` |
| `track_step` | `track_step` |
| `window_partition` | `window_partition` |
| `window_unpartition` | `window_unpartition` |
| `predict_masks` | `predict_masks` |
| `_get_stability_scores` | `get_stability_scores` |
| `_dynamic_multimask_via_stability` | `dynamic_multimask_via_stability` |
| `compute_axial_cis` | `compute_axial_cis` |
| `apply_rotary_enc` | `apply_rotary_enc` |

Do not mechanically translate `nn.ModuleList`, `copy.deepcopy`, or mutable state dictionaries. Represent them with typed `Vec<T>`, cloneable parameter containers where needed, and typed state structs. Retain a direct `forward` method per reference module.

## Bottom-up translation batches

The order below follows the reference call graph loosely. It intentionally permits whole-file and parallel work where dependencies allow. The code-complexity comparator can generate a finer leaf ordering:

```bash
ccc-rs order vendor/segment-anything-2/sam2 --recurse -l python -o translation-order.csv
```

Use the CSV as a checklist, not as a constraint. Mark functions translated as work lands, merge the previous CSV on re-runs, and translate mutually recursive groups together.

### Batch 0 — skeleton, configuration and checkpoint boundary

One worker establishes the crate, feature flags, errors, config structs and module declarations. A second worker implements checkpoint indexing and manifest validation.

- Define variants/configuration for `hiera_tiny`, `hiera_small`, `hiera_base_plus`, and `hiera_large` from upstream YAML/configuration.
- Implement checkpoint prefix normalization, raw tensor key/shape inspection and a required-key manifest.
- Add a loader abstraction which can later support direct `.pt` and converted safetensors input.
- Do not implement a model forward here.

This batch must compile before model translation begins.

### Batch 1 — primitive operations (parallel)

Translate these independent leaves from:

- `sam2/modeling/sam2_utils.py`: `DropPath`, `MLP` → `Mlp`, `LayerNorm2d`, `get_activation_fn`, `get_1d_sine_pe`.
- `sam2/modeling/position_encoding.py`: `PositionEmbeddingSine`, `PositionEmbeddingRandom`, `compute_axial_cis`, `apply_rotary_enc`.
- `sam2/modeling/backbones/utils.py`: `PatchEmbed`, `window_partition`, `window_unpartition`.

Defer training/error-click selection helpers such as `sample_*`, `get_next_point`, and `select_closest_cond_frames` until a public video predictor requires them.

### Batch 2 — Hiera and common attention (parallel)

Translate the two independent attention systems in parallel:

- `sam2/modeling/backbones/hieradet.py`: `do_pool`, `MultiScaleAttention`, `MultiScaleBlock`, `Hiera`.
- `sam2/modeling/sam/transformer.py`: `Attention`, `RoPEAttention` → `RopeAttention`, `TwoWayAttentionBlock`, `TwoWayTransformer`.

Hiera is the SAM2-specific trunk. Preserve its internal NHWC layout and only convert around Candle operations that require another layout. Preserve window partitioning, query pooling, positional embeddings and residual operation order exactly.

### Batch 3 — image features, prompts and memory-feature encoding (parallel)

Translate in parallel after Batch 2 primitives are available:

- `sam2/modeling/backbones/image_encoder.py`: `FpnNeck`, `ImageEncoder`.
- `sam2/modeling/sam/prompt_encoder.py`: `PromptEncoder`.
- `sam2/modeling/memory_encoder.py`: `MaskDownSampler`, `CXBlock`, `Fuser`, `MemoryEncoder`.

At the end of this batch, implement a Hiera + FPN image-feature forward path. It is the first meaningful checkpoint-loading milestone, even though it is not yet a user-facing predictor.

### Batch 4 — decoding and temporal attention (parallel)

Translate:

- `sam2/modeling/sam/mask_decoder.py`: `MaskDecoder`, `predict_masks`, `get_stability_scores`, `dynamic_multimask_via_stability`.
- `sam2/modeling/memory_attention.py`: `MemoryAttentionLayer`, `MemoryAttention`.

Keep decoder and memory attention separate until their stable tensor types are agreed. This avoids making the video state shape implicit.

### Batch 5 — model integration, image path first

Translate `sam2/modeling/sam2_base.py` in this order:

1. `build_sam_heads` (from `_build_sam_heads`)
2. `forward_image`
3. `prepare_backbone_features`
4. `forward_sam_heads` (from `_forward_sam_heads`)
5. image-only `Sam2Base` construction and inference

Do not start temporal integration until static-image prompts yield a coherent mask-decoder output structure.

### Batch 6 — memory and video/z-slice integration

Finish the remaining `Sam2Base` flow as one coordinated batch:

- `prepare_memory_conditioned_features`
- `encode_new_memory`
- `track_step`
- typed `VideoInferenceState`/memory-bank ownership and reset behavior.

This is the correct point to decide how z-slices map onto SAM2 frames. Keep its stateful propagation API distinct from Blockflow’s block decomposition; neighbouring z-block boundaries need explicit memory/reset semantics.

### Batch 7 — public APIs and utilities

Translate last:

- `sam2/sam2_image_predictor.py` → `Sam2ImagePredictor`.
- `sam2/sam2_video_predictor.py` → `Sam2VideoPredictor`.
- `sam2/automatic_mask_generator.py` → `Sam2AutomaticMaskGenerator`.
- `sam2/utils/transforms.py`, `sam2/utils/amg.py`, then `sam2/utils/misc.py` as required.
- `build_sam.py` becomes typed Rust configuration/builders, not a Hydra reimplementation.
- CLI and image I/O come last, after the library API is stable.

## Deferred test plan

Comprehensive tests are intentionally deferred until the model path is complete. Do not defer the data needed for them: save fixed Python reference inputs and intermediate outputs as each batch becomes available.

Final test phase:

1. Compare raw checkpoint manifest keys and shapes for every Hiera variant.
2. Compare preprocessing tensors, Hiera stage features and FPN outputs to Python.
3. Compare prompt embeddings, mask logits, IoU/object scores and resized masks for points, boxes and mask prompts.
4. Compare one-frame video behavior before multi-frame memory behavior.
5. Compare memory tensors/masks over a fixed multi-frame sequence and verify deterministic reset/replay.
6. Test CUDA against CPU with tolerances stated by operation/dtype; then benchmark before any fusion or tiling optimization.

## Completion definition for the first milestone

The first milestone is not an automatic mask generator. It is a Candle-backed SAM2.1 Hiera-Tiny static-image predictor that loads a real checkpoint, accepts positive/negative points and boxes, and has a recorded Python parity result for encoder features and final mask logits. Only then add video/z-slice memory propagation and larger checkpoints.
