# SAM2 translation continuation plan

This is the live checklist for completing the native Rust translation of the
pinned Meta SAM2 inference package. The original design and bottom-up method
remain in [handover.md](handover.md); this document replaces its obsolete
starting-state and proposed-work sections with the current status and remaining
work.

## Reference and completion scope

- Upstream: `vendor/segment-anything-2` at commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Primary target: published SAM2.1 Hiera Tiny, Small, Base-Plus, and Large
  inference behavior.
- Included in completion: model inference, image prediction, interactive video
  prediction, multi-object state, automatic mask generation, preprocessing and
  postprocessing, and published-checkpoint loading.
- Not part of inference completion: training/data pipelines, notebooks, demo
  servers, benchmark scripts from upstream, or the `torch.compile`-specific
  `SAM2VideoPredictorVOS` subclass.
- `sam2_video_predictor_legacy.py` is a superseded duplicate implementation.
  Translate a behavior from it only if the pinned modern predictor lacks that
  behavior; do not maintain two Rust predictors.
- Hydra and Python object construction are represented by typed Rust configs
  and builders. Exact Hydra machinery is not a translation target.
- Optional Hugging Face downloading and JPEG/video decoding are included as
  feature-gated integration conveniences. They do not add dependencies to the
  default native-inference build.

Completion means every in-scope row below is implemented or explicitly shown
to be already covered, every batch has a Python/Rust parity fixture, and no
public reference behavior is silently omitted.

Explicitly audited exclusions:

- `sam2_utils.sample_box_points`, `sample_random_points_from_errors`,
  `sample_one_point_from_error_center`, and `get_next_point` generate training
  clicks and are outside inference completion.
- `DropPath` training behavior is outside inference completion; published eval
  inference uses the identity behavior already preserved by the Rust model.
- `utils.misc.get_sdpa_settings` selects PyTorch attention backends and has no
  Candle API equivalent.
- Python's threaded `AsyncVideoFrameLoader` is not reproduced: the Rust decoder
  owns tightly packed frames directly and callers can move loading to their own
  worker. Synchronous JPEG-directory and encoded-video loading are included.
- Python's dynamic `MaskData` container is not itself a public semantic
  requirement; its filtering, concatenation, and metadata behavior must be
  preserved in typed Rust AMG structures.
- `from_pretrained` is represented by the optional Hugging Face integration in
  R7 rather than Python class methods.

## Translation method

Continue using the method that produced the existing model:

1. Translate bottom-up: pure helpers before state containers, state containers
   before predictors, and predictors before convenience I/O.
2. Preserve reference control flow, constants, tensor layouts, and operation
   order until the batch passes parity.
3. Keep checkpoint key names exact even when Rust field names differ.
4. Prefer typed Rust structures over Python dictionaries, but retain one clear
   mapping from each upstream function to its Rust equivalent.
5. Preserve the existing lower-level APIs while adding reference-compatible
   stateful APIs; avoid forcing callers to use mutable cached state when the
   current explicit encode/decode interface is useful.
6. For every batch, generate the Python reference fixture first, implement the
   Rust path second, then run formatting, compilation, focused tests, and the
   parity probe.
7. Do not optimize a newly translated path until its reference fixture passes.
   Existing optimized model kernels remain in place unless they prevent parity.

Checklist legend:

- `[x]` implemented and covered by existing evidence.
- `[-]` partially implemented; remaining behavior is named explicitly.
- `[ ]` not yet translated.
- `[n/a]` deliberately outside the completion scope above.

## Current completed foundation

### Model and checkpoint architecture

- [x] Typed SAM2.1 configuration for all four published Hiera variants.
- [x] Hiera, FPN image encoder, prompt encoder, two-way transformer, mask
  decoder, memory encoder, memory attention, and the temporal model path.
- [x] Checkpoint-key-preserving safetensors loading.
- [x] Positive and negative point prompts, boxes, dense mask prompts, and
  explicit single/multimask selection.
- [x] Object score, object pointer, memory features, temporal positions, and
  forward/reverse temporal memory selection.
- [x] RGB8 preparation and restoration to original image dimensions.
- [x] Tiny static and two-frame temporal fixtures recorded in
  [PARITY.md](PARITY.md); static checks also exist for Small, Base-Plus, and
  Large.
- [x] Complete key/shape manifests validate every parameter for all four
  published variants before construction.
- [x] Published PyTorch `.pt` checkpoints load their `model` state directly;
  safetensors conversion is optional.

### Existing public inference coverage

- [x] Stateless image encode/decode and one-shot RGB8 prediction.
- [x] Single-object conditioning, mask conditioning, propagation, and reset.
- [x] Basic object-keyed multi-object propagation.
- [x] Grid automatic-mask generation, crops, RLE metadata, NMS, and optional
  mask-to-mask refinement.
- [x] Stateful image caching, prompt batching, full AMG options, correction
  clicks, temporary video outputs, output consolidation, and prompt editing.

## Remaining bottom-up batches

### R1 — decoder and mask postprocessing leaves

These are independent of predictor state and must land first.

| Upstream function or behavior | Rust destination | Status |
| --- | --- | --- |
| `MaskDecoder._get_stability_scores` | `modeling/mask_decoder.rs` | [x] |
| `MaskDecoder._dynamic_multimask_via_stability` | `modeling/mask_decoder.rs` | [x] |
| builder postprocessing defaults: delta `0.05`, threshold `0.98` | typed decoder config | [x] |
| `utils.misc.get_connected_components` | `ops/mask.rs` | [x] |
| `utils.misc.fill_holes_in_mask_scores` | `ops/mask.rs` | [x] |
| `utils.misc.mask_to_box` | `ops/mask.rs` | [x] |
| `utils.misc.concat_points` | `ops/mask.rs` | [x] |
| `utils.amg.remove_small_regions` | `ops/mask.rs` | [x] |
| `utils.amg.batched_mask_to_box` | `ops/mask.rs` | [x] |

Acceptance fixture:

- Construct stable and unstable synthetic logits around the exact thresholds.
- Compare selected token, logits, IoU score, connected components, hole fill,
  sprinkle removal, and boxes against Python.
- Require exact integer metadata and float error no worse than `1e-6` for pure
  postprocessing operations.

Completed by `tools/generate_r1_parity_reference.py` and
`src/bin/r1-parity-probe.rs`; see [PARITY.md](PARITY.md). The component helper
currently uses an exact CPU implementation and copies CUDA masks to and from
the host. This preserves behavior and is deliberately left unoptimized until
the predictor paths that consume it are translated and benchmarked.

### R2 — transforms and AMG utility completeness

Translate the reusable leaves before changing the generator.

| Upstream function or behavior | Rust destination | Status |
| --- | --- | --- |
| `SAM2Transforms.forward_batch` | `transforms.rs` | [x] |
| batched `transform_coords` | `transforms.rs` | [x] |
| batched `transform_boxes` | `transforms.rs` | [x] |
| `postprocess_masks` hole/sprinkle behavior | `transforms.rs` | [x] |
| `build_point_grid` | `ops/amg.rs` | [x] |
| `build_all_layer_point_grids` | `ops/amg.rs` | [x] |
| `generate_crop_boxes` | `ops/amg.rs` | [x] |
| `batch_iterator` | `ops/amg.rs::batch_ranges` | [x] |
| `is_box_near_crop_edge` | `ops/amg.rs` | [x] |
| `box_xyxy_to_xywh` | AMG utility | [x] current metadata is XYWH |
| `mask_to_rle_pytorch` / `rle_to_mask` / `area_from_rle` | `MaskRle` methods | [x] |
| `calculate_stability_score` | AMG utility | [x] scalar equivalent exists |
| `uncrop_boxes_xyxy` / `uncrop_points` / `uncrop_masks` | `ops/amg.rs` | [x] |
| `coco_encode_rle` | `MaskRle::to_coco_counts` | [x] |

Acceptance fixture:

- Use odd-sized landscape and portrait RGB images, batched coordinates and
  boxes, every crop layer through two, and masks touching interior/exterior
  crop edges.
- Compare transformed coordinates, restored logits, crop boxes, RLE round
  trips, areas, and uncropped metadata.

Completed by `tools/generate_r2_parity_reference.py` and
`src/bin/r2-parity-probe.rs`; see [PARITY.md](PARITY.md). This batch corrected
the earlier simplified generator's row-major RLE to the upstream COCO
column-major representation.

### R3 — reference-compatible image predictor state and batching

Preserve the current explicit `encode_image` / `decode_image_features` API.
Add reference-compatible cached-image behavior alongside it.

| Upstream method | Rust destination | Status |
| --- | --- | --- |
| `set_image` | cached single-image state in `Sam2ImagePredictor` | [x] |
| `set_image_batch` | cached batch state | [x] |
| `predict_batch` | batched prompt decode and restored outputs | [x] |
| `predict` | `predict_cached` alongside retained stateless `predict` | [x] |
| `_prep_prompts` | typed `ImagePromptBatch` preparation | [x] |
| `_predict` including repeated-image semantics | batched decoder path | [x] |
| `get_image_embedding` | cached-state accessor | [x] |
| `reset_predictor` | cached-state reset | [x] |
| `device` | public device accessor | [x] |
| `mask_threshold`, `max_hole_area`, `max_sprinkle_area` | `ImagePredictorConfig` | [x] |

Acceptance fixture:

- One image and a two-image batch.
- Per-image mixtures of positive/negative points, boxes, prior masks, empty
  prompts, single-mask output, and multimask output.
- Compare low-resolution logits, restored masks, IoU scores, object scores,
  and embeddings. Repeating prediction from cached features must not rerun the
  image encoder.

Completed by `tools/generate_r3_parity_reference.py` and
`src/bin/r3-parity-probe.rs`; see [PARITY.md](PARITY.md). Rust retains the prior
stateless `predict` API and names the stateful equivalent `predict_cached`
because Rust does not support Python-style method overloading.

### R4 — full automatic mask generator

Implement this only after R1–R3 so it uses the same batching and
postprocessing as the reference.

| Upstream option or method | Rust destination | Status |
| --- | --- | --- |
| `points_per_batch` and `_process_batch` | batched prompt decode | [x] |
| explicit `point_grids` | `AutomaticMaskConfig` | [x] |
| `crop_nms_thresh` separate from `box_nms_thresh` | `AutomaticMaskConfig` | [x] |
| `min_mask_region_area` / `postprocess_small_regions` | predictor transforms and `ops/amg.rs` | [x] |
| `output_mode`: binary, uncompressed RLE, COCO RLE | typed output enum | [x] |
| configurable `multimask_output` | `AutomaticMaskConfig` | [x] |
| `_generate_masks` / `_process_crop` metadata ordering | generator internals | [x] |
| `refine_with_m2m` batching | generator internals | [x] |

Acceptance fixture:

- Freeze one small real image so the full default grid remains affordable.
- Compare proposal count and every retained proposal's point, crop, box, area,
  predicted IoU, stability, and decoded binary mask.
- Run default, crops, explicit point grids, small-region cleanup, and M2M as
  separate fixtures so a failure is attributable.

Completed by `tools/generate_r4_parity_reference.py` and
`src/bin/r4-parity-probe.rs`; see [PARITY.md](PARITY.md). The full default grid
is intentionally retained in the fixture even though it is much slower than
the focused cases.

### R5 — video state primitives

Do not add high-level editing methods until their state representation is
complete. Extend `VideoInferenceState` with typed equivalents of the reference
dictionaries rather than reproducing string-keyed maps.

| Upstream state or helper | Rust destination | Status |
| --- | --- | --- |
| stable object-ID/index mapping | multi-object state | [x] |
| per-object point and mask prompt history | video state | [x] |
| temporary conditioning/non-conditioning outputs | video state | [x] |
| consolidated conditioning/non-conditioning outputs | video state | [x] |
| cached image features with bounded eviction | video state | [x] |
| original video dimensions and frame count | video state | [x] |
| frames already tracked and direction metadata | video state | [x] |
| cached constant mask-memory positional encodings | video state | [x] |
| `_apply_non_overlapping_constraints` | `ops/mask.rs` | [x] |
| `_clear_non_cond_mem_around_input` | video state | [x] |
| `_get_orig_video_res_output` | `ops/mask.rs::restore_video_masks` | [x] |

Acceptance fixture:

- Unit-test insertion, replacement, eviction, forward/reverse ordering,
  object-index stability, reset, and removal without running the model.
- Compare a serialized Python state summary after the same editing sequence.

Completed by `tools/generate_r5_state_reference.py` and
`src/bin/r5-state-probe.rs`; see [PARITY.md](PARITY.md).

### R6 — interactive video predictor behavior

Build these methods on R5 and preserve the existing simple streaming API as a
convenience wrapper.

| Upstream method or behavior | Rust destination | Status |
| --- | --- | --- |
| `init_state` from a prepared frame sequence | video predictor/session builder | [x] |
| `add_new_points_or_box` with append/replace semantics | video predictor | [x] |
| `add_new_points` alias | video predictor | [x] |
| `add_new_mask` | video predictor | [x] integrated with edit state |
| `_consolidate_temp_output_across_obj` | multi-object predictor | [x] |
| `propagate_in_video_preflight` | video predictor | [x] |
| `propagate_in_video` range, start, max-frame and reverse behavior | video predictor | [x] |
| `clear_all_prompts_in_frame` | video predictor | [x] |
| `reset_state` / `_reset_tracking_results` | video predictor | [x] |
| `_get_image_feature` cache behavior | video predictor | [x] |
| `_run_single_frame_inference` | video predictor | [x] |
| `_run_memory_encoder` | video predictor | [x] core memory encoding exists |
| `_get_maskmem_pos_enc` cache behavior | video predictor | [x] |
| `remove_object` and affected-frame outputs | multi-object predictor | [x] |
| postprocessing defaults: mask binarization and hole fill | typed video config | [x] |

Acceptance fixture:

- A deterministic three-frame, two-object sequence.
- Add initial prompts, propagate, add a correction click to an already tracked
  frame, clear prompts, remove one object, then propagate in reverse.
- Compare object ordering, affected frame indices, temporary/consolidated
  outputs, low-resolution logits, restored masks, memory tensors, and object
pointers at every step. A post-reset binary-mask interaction additionally
checks the direct-mask logits and its two-stage no-object pointer gating.

The end-to-end fixture is implemented by
`tools/generate_r6_video_reference.py` and `src/bin/r6-video-probe.rs`. Forward
and reverse propagation, correction, clear, removal, and reset pass on CPU; see
[PARITY.md](PARITY.md). The fixture caught four control-flow details that are
now preserved: zero-click frames carry explicit no-point sparse tokens,
SAM2.1 uses multimask selection during ordinary tracking and corrections, and
reverse eval attention excludes conditioning object pointers that lie in the
unvisited direction; direct mask inputs retain both the decoder-presence and
input-mask-presence pointer gates.

### R7 — checkpoint and builder completion

This batch is last because it should construct the already-complete APIs rather
than influence model design.

| Upstream behavior | Rust destination | Status |
| --- | --- | --- |
| load `.pt` state under the `model` key | checkpoint loader using Candle PTH support | [x] |
| complete required-key/shape manifest for all four variants | `checkpoint.rs` | [x] |
| typed image/video postprocessing builder defaults | builder/config API | [x] |
| actionable missing/unexpected key report | checkpoint loader | [x] |
| optional Hugging Face model-ID download | `huggingface.rs`, cached `from_pretrained` constructors | [x] |
| optional JPEG-directory/video decoding | `media.rs`, predictor path/session helpers | [x] |

Acceptance fixture:

- Load each published `.pt` directly and its converted safetensors equivalent.
- Compare every tensor name and shape, then compare final inference outputs
  byte-for-byte where the loaded f32 weights are identical.
- A malformed, wrong-variant, and truncated checkpoint must fail before model
  inference with the precise offending key.

Optional-integration acceptance:

- All four variants map to their published Hugging Face repository and `.pt`
  filename; revision, cache-directory, offline-only, and force-download options
  are typed and passed to the Hub client.
- JPEG directories are sorted by integer filename stem and reject empty,
  malformed, duplicate-index, or mixed-dimension frame sets.
- FFmpeg decoding is exercised on the vendored SAM2 MP4 and returns tightly
  packed RGB8 frames with the original dimensions. Predictor helpers accept
  decoded images, JPEG directories, and encoded videos without another owned
  RGB copy at the API boundary.

## Per-batch verification gate

No batch is complete until all applicable items pass:

```text
cargo fmt --all -- --check
cargo check
cargo test --lib
cargo check --features cuda
Python reference fixture generation
Rust fixture comparison
```

For model tensors, retain the existing f32 parity envelope unless a fixture
establishes a tighter operation-specific tolerance. For masks and metadata,
compare thresholded masks, object IDs, frame indices, boxes, RLEs, and counts
exactly. Record accepted results in [PARITY.md](PARITY.md).

## Final completion audit

- [x] Every in-scope modern image-predictor method has a mapped Rust API.
- [x] Every in-scope modern video-predictor method has a mapped Rust API.
- [x] Every automatic-mask-generator option is implemented or explicitly
  represented by a documented optional integration.
- [x] Published `.pt` and safetensors checkpoints load through the same public
  builder for all four SAM2.1 variants.
- [x] Static single/batch image, AMG, single-object video, and interactive
  multi-object video fixtures pass on CPU.
- [x] The same model paths compile on CUDA; representative image and video
  fixtures pass within the recorded CUDA tolerance.
- [x] README examples cover image caching, automatic masks, and editable video
  propagation.
- [x] `handover.md` marks its starting state as historical and points here as
  the live status.
