#!/usr/bin/env python3
"""PyTorch half of independently runnable CPU/GPU parity-suite cells."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/segment-anything-2"))
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
import sam2.automatic_mask_generator as amg_module  # noqa: E402
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.utils.transforms import SAM2Transforms  # noqa: E402


def status_kb(field: str) -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, int((len(values) - 1) * fraction + 0.999999))
    return values[index]


parser = argparse.ArgumentParser()
parser.add_argument("--cell", required=True)
parser.add_argument("--operation", required=True)
parser.add_argument("--variant", required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--warmups", type=int, default=1)
parser.add_argument("--sessions", type=int, default=10)
parser.add_argument("--point-counts", default="0,1,8,32")
parser.add_argument("--batch-fixtures", default="")
parser.add_argument("--threads", type=int)
parser.add_argument("--output", type=Path)
parser.add_argument("--video-lowres-output", type=Path)
args = parser.parse_args()
args.point_counts = [int(value) for value in args.point_counts.split(",")]
args.batch_fixtures = [Path(value) for value in args.batch_fixtures.split(",") if value]
if args.iterations < 1 or args.warmups < 0 or args.sessions < 1:
    parser.error("iterations must be positive and warmups nonnegative")
if args.threads:
    torch.set_num_threads(args.threads)
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but PyTorch cannot access it")
if args.mode == "graph" and (
    args.device != "cuda"
    or args.operation not in ("cached_prompt_sequence", "cached_batch_and_prior_mask")
):
    parser.error("graph mode currently supports CUDA I3 and I4 image operations")


def synchronize() -> None:
    if args.device == "cuda":
        torch.cuda.synchronize()


def write_v3_lowres_outputs(path: Path, state) -> None:
    """Stream initial-forward 256x256 logits in frame-major, object-major order."""
    with path.open("wb") as handle:
        for frame_idx in range(state["num_frames"]):
            for obj_idx in range(len(state["obj_ids"])):
                outputs = state["output_dict_per_obj"][obj_idx]
                current = outputs["cond_frame_outputs"].get(frame_idx)
                if current is None:
                    current = outputs["non_cond_frame_outputs"].get(frame_idx)
                if current is None:
                    raise RuntimeError(
                        f"missing V3 low-resolution output for object {obj_idx}, frame {frame_idx}"
                    )
                values = (
                    current["pred_masks"]
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype("<f4", copy=False)
                )
                values.tofile(handle)


def gpu_memory_snapshot() -> dict[str, int | None]:
    if args.device != "cuda":
        return {
            "allocated": None,
            "reserved": None,
            "peak_allocated": None,
            "peak_reserved": None,
        }
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def make_video_state(predictor, raw_frames):
    profile = bool(os.environ.get("SAM2_PROFILE_VIDEO_STAGES"))
    if profile:
        synchronize()
    transform_start = time.perf_counter()
    transform = SAM2Transforms(1024, 0.0)
    images = transform.forward_batch(raw_frames).to(args.device)
    if profile:
        synchronize()
        print(
            f"SAM2_VIDEO_INIT_PY stage=transform frames={len(raw_frames)} "
            f"ms={(time.perf_counter() - transform_start) * 1000.0:.6f}",
            file=sys.stderr,
        )
    state = {
        "images": images,
        "num_frames": len(raw_frames),
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": raw_frames[0].shape[0],
        "video_width": raw_frames[0].shape[1],
        "device": torch.device(args.device),
        "storage_device": torch.device(args.device),
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    feature_start = time.perf_counter()
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    if profile:
        synchronize()
        print(
            f"SAM2_VIDEO_INIT_PY stage=frame_zero_feature frames={len(raw_frames)} "
            f"ms={(time.perf_counter() - feature_start) * 1000.0:.6f}",
            file=sys.stderr,
        )
    return state


def encode_automatic_masks(masks) -> np.ndarray:
    """Canonical float stream matching the Rust runner, including full RLEs."""
    values = [float(len(masks))]
    for mask in masks:
        values.extend(float(value) for value in mask["bbox"])
        values.append(float(mask["area"]))
        values.append(float(mask["predicted_iou"]))
        values.extend(float(value) for value in mask["point_coords"][0])
        values.append(float(mask["stability_score"]))
        values.extend(float(value) for value in mask["crop_box"])
        counts = mask["segmentation"]["counts"]
        values.append(float(len(counts)))
        values.extend(float(value) for value in counts)
    return np.asarray(values, dtype="<f4")


def run_amg() -> None:
    load_start = time.perf_counter()
    model = build_sam2(
        str(args.config), str(args.checkpoint), device=args.device,
        apply_postprocessing=True,
    )
    kwargs = {
        "points_per_side": 8 if args.operation == "amg_practical" else 32,
        "use_m2m": args.operation == "amg_practical",
        "output_mode": "uncompressed_rle",
    }
    generator = SAM2AutomaticMaskGenerator(model, **kwargs)
    profile_values = {"image_encoder": [], "initial_decoder": [], "refine_decoder": []}
    profile_layouts = {}
    profile_handles = []
    if os.environ.get("SAM2_PROFILE_AMG_STAGES"):
        def profile_function(name, function):
            profile_values[name] = []

            def timed(*positional, **keyword):
                synchronize()
                start = time.perf_counter()
                try:
                    return function(*positional, **keyword)
                finally:
                    synchronize()
                    profile_values[name].append((time.perf_counter() - start) * 1000.0)

            return timed

        # These functions form the GPU-to-host/postprocessing side of AMG and
        # are module globals in the upstream implementation. Keep their normal
        # composition intact while adding matched synchronized boundaries.
        for function_name in (
            "calculate_stability_score",
            "batched_mask_to_box",
            "is_box_near_crop_edge",
            "uncrop_masks",
            "mask_to_rle_pytorch",
            "batched_nms",
        ):
            setattr(
                amg_module,
                function_name,
                profile_function(
                    f"postprocess_{function_name}",
                    getattr(amg_module, function_name),
                ),
            )

        def hook_module(name, module):
            starts = []
            profile_values[name] = []

            def before(_module, _inputs):
                synchronize()
                if name.startswith("mask_decoder_upscaling_") and name not in profile_layouts:
                    tensor = _inputs[0]
                    profile_layouts[name] = {
                        "input_shape": list(tensor.shape),
                        "input_stride": list(tensor.stride()),
                        "input_channels_last": tensor.is_contiguous(memory_format=torch.channels_last),
                    }
                starts.append(time.perf_counter())

            def after(_module, _inputs, _output):
                synchronize()
                if name in profile_layouts and "output_shape" not in profile_layouts[name]:
                    profile_layouts[name].update({
                        "output_shape": list(_output.shape),
                        "output_stride": list(_output.stride()),
                        "output_channels_last": _output.is_contiguous(memory_format=torch.channels_last),
                    })
                profile_values[name].append((time.perf_counter() - starts.pop()) * 1000.0)

            profile_handles.append(module.register_forward_pre_hook(before))
            profile_handles.append(module.register_forward_hook(after))

        decoder = model.sam_mask_decoder
        hook_module("mask_decoder_transformer", decoder.transformer)
        for layer_index, layer in enumerate(decoder.transformer.layers):
            for attention_name in (
                "self_attn",
                "cross_attn_token_to_image",
                "cross_attn_image_to_token",
            ):
                attention = getattr(layer, attention_name)
                for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    hook_module(
                        f"transformer_{layer_index}_{attention_name}_{projection_name}",
                        getattr(attention, projection_name),
                    )
        for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            hook_module(
                f"transformer_final_{projection_name}",
                getattr(decoder.transformer.final_attn_token_to_image, projection_name),
            )
        for index, module in enumerate(decoder.output_upscaling):
            hook_module(f"mask_decoder_upscaling_{index}", module)
        for index, module in enumerate(decoder.output_hypernetworks_mlps):
            hook_module(f"mask_decoder_hypernetwork_{index}", module)
        hook_module("mask_decoder_iou_head", decoder.iou_prediction_head)
        if decoder.pred_obj_score_head is not None:
            hook_module("mask_decoder_object_score_head", decoder.pred_obj_score_head)

        # A2 has one known threshold-boundary pixel. Capture both operands of
        # its final 32-wide mask projection so a matched Rust/PyTorch run can
        # attribute the first meaningful drift without changing inference.
        # Output mask 1 after multimask selection is internal mask token 2.
        a2_boundary_values = {}
        if os.environ.get("SAM2_PROFILE_A2_BOUNDARY"):
            boundary_calls = {"upscaled": 0, "hyper_token2": 0}
            transformer_calls = {"input": 0, "layer_0": 0, "layer_1": 0, "final": 0}
            final_attention_calls = {"q_projection": 0, "attended": 0, "output_projection": 0}

            def capture_final_attention_value(name, from_input=False):
                def capture(_module, inputs, output=None):
                    call = final_attention_calls[name]
                    final_attention_calls[name] += 1
                    if call == 2:
                        value = inputs[0] if from_input else output
                        a2_boundary_values[f"final_attention_{name}"] = (
                            value[18, 4].detach().float().cpu().flatten().tolist()
                        )

                return capture

            final_attention = decoder.transformer.final_attn_token_to_image
            profile_handles.append(
                final_attention.q_proj.register_forward_hook(
                    capture_final_attention_value("q_projection")
                )
            )
            profile_handles.append(
                final_attention.out_proj.register_forward_pre_hook(
                    capture_final_attention_value("attended", from_input=True)
                )
            )
            profile_handles.append(
                final_attention.out_proj.register_forward_hook(
                    capture_final_attention_value("output_projection")
                )
            )

            def capture_transformer_input(_module, inputs):
                call = transformer_calls["input"]
                transformer_calls["input"] += 1
                if call == 2:
                    a2_boundary_values["transformer_input"] = (
                        inputs[2][18, 4].detach().float().cpu().flatten().tolist()
                    )

            def capture_transformer_stage(name):
                def capture(_module, _inputs, output):
                    call = transformer_calls[name]
                    transformer_calls[name] += 1
                    if call == 2:
                        a2_boundary_values[name] = (
                            output[0][18, 4].detach().float().cpu().flatten().tolist()
                        )

                return capture

            profile_handles.append(
                decoder.transformer.register_forward_pre_hook(capture_transformer_input)
            )
            for layer_index, layer in enumerate(decoder.transformer.layers):
                profile_handles.append(
                    layer.register_forward_hook(
                        capture_transformer_stage(f"layer_{layer_index}")
                    )
                )
            profile_handles.append(
                decoder.transformer.register_forward_hook(
                    capture_transformer_stage("final")
                )
            )

            def capture_upscaled(_module, _inputs, output):
                call = boundary_calls["upscaled"]
                boundary_calls["upscaled"] += 1
                if call == 2:
                    a2_boundary_values["upscaled"] = (
                        output[18, :, 37:39, 158:160]
                        .detach().float().cpu().flatten().tolist()
                    )

            def capture_hyper_token2(_module, inputs, output):
                call = boundary_calls["hyper_token2"]
                boundary_calls["hyper_token2"] += 1
                if call == 2:
                    a2_boundary_values["mask_token2"] = (
                        inputs[0][18].detach().float().cpu().flatten().tolist()
                    )
                    a2_boundary_values["hyper_token2"] = (
                        output[18].detach().float().cpu().flatten().tolist()
                    )

            profile_handles.append(
                decoder.output_upscaling[4].register_forward_hook(capture_upscaled)
            )
            profile_handles.append(
                decoder.output_hypernetworks_mlps[2].register_forward_hook(
                    capture_hyper_token2
                )
            )

        original_set_image = generator.predictor.set_image
        original_predict = generator.predictor._predict
        a2_predict_calls = [0]

        def profiled_set_image(*positional, **keyword):
            synchronize()
            start = time.perf_counter()
            result = original_set_image(*positional, **keyword)
            synchronize()
            profile_values["image_encoder"].append((time.perf_counter() - start) * 1000.0)
            return result

        def profiled_predict(*positional, **keyword):
            stage = "refine_decoder" if keyword.get("mask_input") is not None else "initial_decoder"
            synchronize()
            start = time.perf_counter()
            result = original_predict(*positional, **keyword)
            synchronize()
            profile_values[stage].append((time.perf_counter() - start) * 1000.0)
            if os.environ.get("SAM2_PROFILE_A2_BOUNDARY") and stage == "initial_decoder":
                batch_index = a2_predict_calls[0] % 16
                a2_predict_calls[0] += 1
                if batch_index == 2:
                    masks, ious, low_res = result
                    print(
                        "SAM2_A2_BOUNDARY_PY "
                        f"batch={batch_index} "
                        f"logits={masks[18, :, 79, 495].detach().cpu().tolist()} "
                        f"ious={ious[18].detach().cpu().tolist()} "
                        "low_res_y37_38_x158_159="
                        f"{low_res[18, 1, 37:39, 158:160].detach().cpu().flatten().tolist()} "
                        f"mask_token2={a2_boundary_values.get('mask_token2')} "
                        f"hyper_token2={a2_boundary_values.get('hyper_token2')} "
                        f"upscaled_y37_38_x158_159={a2_boundary_values.get('upscaled')}",
                        file=sys.stderr,
                    )
                    print(
                        "SAM2_A2_TRANSFORMER_PY "
                        f"input={a2_boundary_values.get('transformer_input')} "
                        f"layer_0={a2_boundary_values.get('layer_0')} "
                        f"layer_1={a2_boundary_values.get('layer_1')} "
                        f"final={a2_boundary_values.get('final')}",
                        file=sys.stderr,
                    )
                    print(
                        "SAM2_A2_FINAL_ATTENTION_PY "
                        f"q_projection={a2_boundary_values.get('final_attention_q_projection')} "
                        f"attended={a2_boundary_values.get('final_attention_attended')} "
                        f"output_projection={a2_boundary_values.get('final_attention_output_projection')}",
                        file=sys.stderr,
                    )
            return result

        generator.predictor.set_image = profiled_set_image
        generator.predictor._predict = profiled_predict
    synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    rss_after_load_kb = status_kb("VmRSS:")

    fixture_start = time.perf_counter()
    tensors = load_file(str(args.fixture))
    rgb_hwc = tensors["rgb_u8"].permute(1, 2, 0).contiguous().numpy()
    fixture_decode_ms = (time.perf_counter() - fixture_start) * 1000.0
    with torch.inference_mode():
        for _ in range(args.warmups):
            generator.generate(rgb_hwc)
            synchronize()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        samples_ms = []
        masks = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            masks = generator.generate(rgb_hwc)
            synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    parity_values = encode_automatic_masks(masks)
    if args.output:
        args.output.write_bytes(parity_values.tobytes())
    gpu_memory = {"allocated": None, "reserved": None, "peak_allocated": None, "peak_reserved": None}
    if args.device == "cuda":
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
    for handle in profile_handles:
        handle.remove()
    for name, values in profile_values.items():
        if values:
            print(
                f"SAM2_AMG_STAGE_PY stage={name} sum_ms={sum(values):.6f} "
                f"calls={len(values)}",
                file=sys.stderr,
            )
    for name, layout in profile_layouts.items():
        print(
            f"SAM2_AMG_LAYOUT_PY stage={name} json={json.dumps(layout, separators=(',', ':'))}",
            file=sys.stderr,
        )
    print(json.dumps({
        "schema_version": 1, "implementation": "python-pytorch", "device": args.device,
        "cell": args.cell, "operation": args.operation, "variant": args.variant,
        "iterations": args.iterations, "warmups": args.warmups, "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms), "p95_ms": quantile(samples_ms, 0.95),
        "phases_ms": {"load": load_ms, "fixture_decode": fixture_decode_ms},
        "rss_kb": {"after_load": rss_after_load_kb, "final": status_kb("VmRSS:"),
                   "hwm": status_kb("VmHWM:")},
        "gpu_memory_bytes": gpu_memory,
        "output": {"checksum": float(parity_values.astype(np.float64).sum()),
                   "elements": int(parity_values.size), "masks": len(masks)},
    }, separators=(",", ":")))


def run_video() -> None:
    load_start = time.perf_counter()
    predictor = build_sam2_video_predictor(
        str(args.config), str(args.checkpoint), device=args.device, apply_postprocessing=True
    )
    profile_handles = []
    profile_values = {}
    if os.environ.get("SAM2_PROFILE_VIDEO_STAGES"):
        def hook_module(name, module):
            starts = []
            profile_values[name] = []

            def before(_module, _inputs):
                synchronize()
                starts.append(time.perf_counter())

            def after(_module, _inputs, _output):
                synchronize()
                profile_values[name].append((time.perf_counter() - starts.pop()) * 1000.0)

            profile_handles.append(module.register_forward_pre_hook(before))
            profile_handles.append(module.register_forward_hook(after))

        hook_module("image_encoder", predictor.image_encoder)
        hook_module("memory_attention", predictor.memory_attention)
        for index, layer in enumerate(predictor.memory_attention.layers):
            hook_module(f"memory_attention_layer_{index}", layer)
            hook_module(f"memory_layer_{index}_norm_1", layer.norm1)
            hook_module(f"memory_layer_{index}_self_attention", layer.self_attn)
            hook_module(f"memory_layer_{index}_norm_2", layer.norm2)
            hook_module(f"memory_layer_{index}_cross_attention", layer.cross_attn_image)
            hook_module(f"memory_layer_{index}_norm_3", layer.norm3)
            hook_module(f"memory_layer_{index}_mlp_1", layer.linear1)
            hook_module(f"memory_layer_{index}_mlp_2", layer.linear2)
        hook_module("mask_decoder", predictor.sam_mask_decoder)
        hook_module("mask_decoder_transformer", predictor.sam_mask_decoder.transformer)
        for index, module in enumerate(predictor.sam_mask_decoder.output_upscaling):
            hook_module(f"mask_decoder_upscaling_{index}", module)
        for index, module in enumerate(predictor.sam_mask_decoder.output_hypernetworks_mlps):
            hook_module(f"mask_decoder_hypernetwork_{index}", module)
        hook_module("mask_decoder_iou_head", predictor.sam_mask_decoder.iou_prediction_head)
        if predictor.sam_mask_decoder.pred_obj_score_head is not None:
            hook_module(
                "mask_decoder_object_score_head",
                predictor.sam_mask_decoder.pred_obj_score_head,
            )
        hook_module("memory_encoder", predictor.memory_encoder)
        hook_module("memory_mask_downsampler", predictor.memory_encoder.mask_downsampler)
        hook_module("memory_pixel_projection", predictor.memory_encoder.pix_feat_proj)
        hook_module("memory_fuser_0", predictor.memory_encoder.fuser.layers[0])
        hook_module("memory_fuser_1", predictor.memory_encoder.fuser.layers[1])
        hook_module("memory_output_projection", predictor.memory_encoder.out_proj)
    synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    rss_after_load_kb = status_kb("VmRSS:")
    gpu_after_load = gpu_memory_snapshot()
    fixture_start = time.perf_counter()
    tensors = load_file(str(args.fixture))
    names = sorted(name for name in tensors if name.startswith("frame_"))
    if not names:
        raise ValueError("video fixture has no frame tensors")
    raw_frames = [tensor.numpy() for name in names for tensor in [tensors[name]]]
    fixture_decode_ms = (time.perf_counter() - fixture_start) * 1000.0
    samples_ms = []
    output_checksum = 0.0
    output_elements = 0
    session_init_ms = 0.0
    rss_after_init_kb = 0
    rss_after_reset_kb = None
    parity_tensors = []
    gpu_memory_phases = {
        "after_load": gpu_after_load,
        "after_init": None,
        "after_interaction": None,
        "after_propagation": None,
        "after_reset": None,
        "after_drop": None,
    }
    with torch.inference_mode():
        for iteration in range(args.warmups + args.iterations):
            if iteration == args.warmups and args.device == "cuda":
                synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            state = make_video_state(predictor, raw_frames)
            synchronize()
            current_init_ms = (time.perf_counter() - start) * 1000.0
            rss_after_init_kb = status_kb("VmRSS:")
            if iteration >= args.warmups:
                gpu_memory_phases["after_init"] = gpu_memory_snapshot()
            measured_start = time.perf_counter()
            outputs = []
            if args.operation in ("video_forward_one", "video_long_lifetime"):
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    points=np.array([[210.0, 350.0]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )
                if iteration >= args.warmups:
                    gpu_memory_phases["after_interaction"] = gpu_memory_snapshot()
                outputs = list(predictor.propagate_in_video(state, start_frame_idx=0))
            elif args.operation == "video_edit_three":
                for object_id, x, y in ((1, 210.0, 350.0), (2, 430.0, 310.0), (3, 600.0, 345.0)):
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=object_id,
                        points=np.array([[x, y]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                if iteration >= args.warmups:
                    gpu_memory_phases["after_interaction"] = gpu_memory_snapshot()
                outputs = list(predictor.propagate_in_video(state, start_frame_idx=0))
                if (
                    iteration + 1 == args.warmups + args.iterations
                    and args.video_lowres_output is not None
                ):
                    write_v3_lowres_outputs(args.video_lowres_output, state)
                middle = len(raw_frames) // 2
                outputs.append(predictor.add_new_points_or_box(
                    state, frame_idx=middle, obj_id=1,
                    points=np.array([[250.0, 220.0]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32), clear_old_points=True,
                ))
                outputs.extend(predictor.propagate_in_video(
                    state, start_frame_idx=middle, max_frame_num_to_track=middle + 1, reverse=True
                ))
                predictor.remove_object(state, 3)
            elif args.operation != "video_init":
                raise ValueError(f"unsupported video operation {args.operation!r}")
            synchronize()
            if iteration >= args.warmups:
                gpu_memory_phases["after_propagation"] = gpu_memory_snapshot()
            elapsed = current_init_ms if args.operation == "video_init" else (time.perf_counter() - measured_start) * 1000.0
            if iteration >= args.warmups:
                samples_ms.append(elapsed)
                session_init_ms = current_init_ms
                output_checksum = 0.0
                output_elements = 0
                parity_tensors = []
                selected = {0, len(outputs) // 2, max(0, len(outputs) - 1)}
                for index, item in enumerate(outputs):
                    masks = item[2]
                    output_checksum += float(masks.sum())
                    output_elements += masks.numel()
                    if iteration + 1 == args.warmups + args.iterations and index in selected:
                        parity_tensors.append(masks.detach().cpu().reshape(-1))
            if args.operation == "video_long_lifetime":
                outputs = []
                predictor.reset_state(state)
                synchronize()
                if iteration >= args.warmups:
                    gpu_memory_phases["after_reset"] = gpu_memory_snapshot()
                del state
                synchronize()
                if iteration >= args.warmups:
                    gpu_memory_phases["after_drop"] = gpu_memory_snapshot()
                rss_after_reset_kb = status_kb("VmRSS:")
    gpu_memory = gpu_memory_snapshot()
    if args.output:
        if parity_tensors:
            values = torch.cat(parity_tensors).contiguous().numpy().astype("<f4")
            args.output.write_bytes(values.tobytes())
        else:
            args.output.write_bytes(b"")
    record = {
        "schema_version": 1, "implementation": "python-pytorch", "device": args.device,
        "cell": args.cell, "operation": args.operation, "variant": args.variant,
        "iterations": args.iterations, "warmups": args.warmups, "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms), "p95_ms": quantile(samples_ms, 0.95),
        "phases_ms": {"load": load_ms, "fixture_decode": fixture_decode_ms, "session_init": session_init_ms},
        "rss_kb": {"after_load": rss_after_load_kb, "after_init": rss_after_init_kb,
                   "after_reset": rss_after_reset_kb, "final": status_kb("VmRSS:"), "hwm": status_kb("VmHWM:")},
        "gpu_memory_bytes": gpu_memory,
        "gpu_memory_phases_bytes": gpu_memory_phases,
        "output": {"checksum": output_checksum, "elements": output_elements},
    }
    for handle in profile_handles:
        handle.remove()
    for name, values in profile_values.items():
        print(
            f"SAM2_VIDEO_STAGE_PY stage={name} sum_ms={sum(values):.6f} "
            f"calls={len(values)}",
            file=sys.stderr,
        )
    print(json.dumps(record, separators=(",", ":")))


def run_encoded_video() -> None:
    load_start = time.perf_counter()
    predictor = build_sam2_video_predictor(
        str(args.config), str(args.checkpoint), device=args.device, apply_postprocessing=True
    )
    synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    rss_after_load_kb = status_kb("VmRSS:")
    gpu_after_load = gpu_memory_snapshot()
    samples_ms = []
    rss_after_init_kb = 0
    rss_after_drop_kb = 0
    gpu_after_init = None
    gpu_after_drop = None
    with torch.inference_mode():
        for iteration in range(args.warmups + args.iterations):
            if iteration == args.warmups and args.device == "cuda":
                synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            state = predictor.init_state(
                video_path=str(args.fixture),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if iteration >= args.warmups:
                samples_ms.append(elapsed)
                rss_after_init_kb = status_kb("VmRSS:")
                gpu_after_init = gpu_memory_snapshot()
            del state
            synchronize()
            if iteration >= args.warmups:
                rss_after_drop_kb = status_kb("VmRSS:")
                gpu_after_drop = gpu_memory_snapshot()
    if args.output:
        args.output.write_bytes(b"")
    print(json.dumps({
        "schema_version": 1, "implementation": "python-pytorch", "device": args.device,
        "cell": args.cell, "operation": args.operation, "variant": args.variant,
        "iterations": args.iterations, "warmups": args.warmups, "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms), "p95_ms": quantile(samples_ms, 0.95),
        "phases_ms": {"load": load_ms, "fixture_decode": None},
        "rss_kb": {"after_load": rss_after_load_kb, "after_init": rss_after_init_kb,
                   "after_drop": rss_after_drop_kb, "final": status_kb("VmRSS:"),
                   "hwm": status_kb("VmHWM:")},
        "gpu_memory_bytes": gpu_memory_snapshot(),
        "gpu_memory_phases_bytes": {"after_load": gpu_after_load,
                                    "after_init": gpu_after_init, "after_drop": gpu_after_drop},
        "output": {"checksum": 0.0, "elements": 0},
    }, separators=(",", ":")))


def run_repeated_lifetime() -> None:
    fixture_start = time.perf_counter()
    tensors = load_file(str(args.fixture))
    fixture_decode_ms = (time.perf_counter() - fixture_start) * 1000.0
    names = sorted(name for name in tensors if name.startswith("frame_"))
    is_video = bool(names)
    load_start = time.perf_counter()
    if is_video:
        predictor = build_sam2_video_predictor(
            str(args.config), str(args.checkpoint), device=args.device,
            apply_postprocessing=True,
        )
        raw_frames = [tensors[name].numpy() for name in names]
    else:
        model = build_sam2(
            str(args.config), str(args.checkpoint), device=args.device,
            apply_postprocessing=True,
        )
        predictor = SAM2ImagePredictor(model)
        rgb_hwc = tensors["rgb_u8"].permute(1, 2, 0).contiguous().numpy()
    synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    rss_after_load_kb = status_kb("VmRSS:")
    samples_ms = []
    rss_plateau_kb = []
    parity = None
    profile_lifetime = bool(os.environ.get("SAM2_PROFILE_VIDEO_STAGES"))
    with torch.inference_mode():
        for iteration in range(args.warmups + args.iterations):
            if iteration == args.warmups and args.device == "cuda":
                synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for session_index in range(args.sessions):
                if is_video:
                    init_start = time.perf_counter()
                    state = make_video_state(predictor, raw_frames)
                    if profile_lifetime:
                        synchronize()
                        print(
                            "SAM2_VIDEO_LIFETIME_PY stage=init "
                            f"ms={(time.perf_counter() - init_start) * 1000.0:.6f}",
                            file=sys.stderr,
                        )
                    prompt_start = time.perf_counter()
                    _, _, masks = predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        points=np.array([[210.0, 350.0]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                    if profile_lifetime:
                        synchronize()
                        print(
                            "SAM2_VIDEO_LIFETIME_PY stage=prompt "
                            f"ms={(time.perf_counter() - prompt_start) * 1000.0:.6f}",
                            file=sys.stderr,
                        )
                    if iteration + 1 == args.warmups + args.iterations and session_index + 1 == args.sessions:
                        parity = masks.detach().cpu().reshape(-1).numpy().astype("<f4")
                    reset_start = time.perf_counter()
                    predictor.reset_state(state)
                    del state
                    if profile_lifetime:
                        synchronize()
                        print(
                            "SAM2_VIDEO_LIFETIME_PY stage=reset_drop "
                            f"ms={(time.perf_counter() - reset_start) * 1000.0:.6f}",
                            file=sys.stderr,
                        )
                else:
                    predictor.set_image(rgb_hwc)
                    masks, _, _ = predictor.predict(
                        point_coords=np.array([[210.0, 350.0]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=False,
                        return_logits=True,
                    )
                    if iteration + 1 == args.warmups + args.iterations and session_index + 1 == args.sessions:
                        parity = np.asarray(masks, dtype="<f4").reshape(-1)
                    predictor.reset_predictor()
                synchronize()
                if iteration >= args.warmups:
                    rss_plateau_kb.append(status_kb("VmRSS:"))
            if iteration >= args.warmups:
                samples_ms.append((time.perf_counter() - start) * 1000.0)
    if parity is None:
        parity = np.empty(0, dtype="<f4")
    if args.output:
        args.output.write_bytes(parity.tobytes())
    print(json.dumps({
        "schema_version": 1, "implementation": "python-pytorch", "device": args.device,
        "cell": args.cell, "operation": args.operation, "variant": args.variant,
        "iterations": args.iterations, "warmups": args.warmups, "sessions": args.sessions,
        "samples_ms": samples_ms, "median_ms": statistics.median(samples_ms),
        "p95_ms": quantile(samples_ms, 0.95),
        "phases_ms": {"load": load_ms, "fixture_decode": fixture_decode_ms},
        "rss_kb": {"after_load": rss_after_load_kb, "after_each_session": rss_plateau_kb,
                   "final": status_kb("VmRSS:"), "hwm": status_kb("VmHWM:")},
        "gpu_memory_bytes": gpu_memory_snapshot(),
        "output": {"checksum": float(parity.astype(np.float64).sum()),
                   "elements": int(parity.size)},
    }, separators=(",", ":")))


if args.operation == "repeated_lifetime":
    run_repeated_lifetime()
    raise SystemExit
if args.operation == "encoded_video_init":
    run_encoded_video()
    raise SystemExit
if args.operation.startswith("video_"):
    run_video()
    raise SystemExit
if args.operation.startswith("amg_"):
    run_amg()
    raise SystemExit


load_start = time.perf_counter()
model = build_sam2(
    str(args.config),
    str(args.checkpoint),
    device=args.device,
    apply_postprocessing=True,
)
synchronize()
load_ms = (time.perf_counter() - load_start) * 1000.0
rss_after_load_kb = status_kb("VmRSS:")
if args.operation == "model_load":
    if args.output:
        args.output.write_bytes(b"")
    gpu_memory = {"allocated": None, "reserved": None, "peak_allocated": None, "peak_reserved": None}
    if args.device == "cuda":
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
    print(json.dumps({
        "schema_version": 1, "implementation": "python-pytorch", "device": args.device,
        "cell": args.cell, "operation": args.operation, "variant": args.variant,
        "iterations": 1, "warmups": 0, "samples_ms": [load_ms],
        "median_ms": load_ms, "p95_ms": load_ms, "phases_ms": {"load": load_ms},
        "rss_kb": {"after_load": rss_after_load_kb, "final": status_kb("VmRSS:"),
                   "hwm": status_kb("VmHWM:")},
        "gpu_memory_bytes": gpu_memory,
        "output": {"checksum": 0.0, "elements": 0},
    }, separators=(",", ":")))
    raise SystemExit

if args.operation == "image_batch":
    if not args.batch_fixtures:
        parser.error("image_batch requires --batch-fixtures")
    fixture_start = time.perf_counter()
    fixtures = [load_file(str(path)) for path in args.batch_fixtures]
    image = torch.cat([fixture["image"] for fixture in fixtures]).to(args.device)
    point = torch.cat([fixture["point"] for fixture in fixtures]).to(args.device)
    point_label = torch.ones(point.shape[:-1], dtype=torch.int64, device=args.device)
    fixture_decode_ms = (time.perf_counter() - fixture_start) * 1000.0

    def run_image_batch_once():
        encode_start = time.perf_counter()
        image_output = model.forward_image(image)
        if model.directly_add_no_mem_embed:
            image_output["vision_features"] = (
                image_output["vision_features"]
                + model.no_mem_embed.reshape(1, -1, 1, 1)
            )
        encode_ms = (time.perf_counter() - encode_start) * 1000.0
        decode_start = time.perf_counter()
        sparse, dense = model.sam_prompt_encoder((point, point_label), None, None)
        masks, _, _, _ = model.sam_mask_decoder(
            image_output["vision_features"],
            model.sam_prompt_encoder.get_dense_pe(),
            sparse,
            dense,
            False,
            False,
            image_output["backbone_fpn"][:2],
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        return masks, encode_ms, decode_ms

    with torch.inference_mode():
        for _ in range(args.warmups):
            output, _, _ = run_image_batch_once()
            synchronize()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        samples_ms = []
        encode_samples_ms = []
        decode_samples_ms = []
        for _ in range(args.iterations):
            synchronize()
            start = time.perf_counter()
            output, encode_ms, decode_ms = run_image_batch_once()
            synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000.0)
            encode_samples_ms.append(encode_ms)
            decode_samples_ms.append(decode_ms)
    if args.output:
        args.output.write_bytes(output.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    print(json.dumps({
        "schema_version": 1, "implementation": "python-pytorch", "device": args.device,
        "mode": args.mode, "cell": args.cell, "operation": args.operation,
        "variant": args.variant, "iterations": args.iterations, "warmups": args.warmups,
        "samples_ms": samples_ms, "median_ms": statistics.median(samples_ms),
        "p95_ms": quantile(samples_ms, 0.95),
        "phases_ms": {"load": load_ms, "fixture_decode": fixture_decode_ms,
                      "encode": statistics.fmean(encode_samples_ms),
                      "decode": statistics.fmean(decode_samples_ms), "capture": None},
        "rss_kb": {"after_load": rss_after_load_kb, "after_encode": None,
                   "final": status_kb("VmRSS:"), "hwm": status_kb("VmHWM:")},
        "gpu_memory_bytes": gpu_memory_snapshot(),
        "output": {"checksum": float(output.sum()), "elements": output.numel()},
    }, separators=(",", ":")))
    raise SystemExit

profile_handles = []
profile_values = {}
if os.environ.get("SAM2_PROFILE_DECODER_PARITY"):
    decoder = model.sam_mask_decoder
    parity_layer_calls = [0 for _ in decoder.transformer.layers]
    parity_transformer_calls = [0]
    parity_boundary_calls = {}

    def parity_values(tensor, token):
        return tensor[0, token].detach().float().cpu().flatten().tolist()

    def capture_parity_layer_input(index):
        def capture(_module, inputs, kwargs):
            if parity_layer_calls[index] == 0 and index == 0:
                queries = kwargs.get("queries", inputs[0] if inputs else None)
                keys = kwargs.get("keys", inputs[1] if len(inputs) > 1 else None)
                print(
                    "SAM2_DECODER_PARITY_PY "
                    f"stage=input_query values={parity_values(queries, 2)}",
                    file=sys.stderr,
                )
                print(
                    "SAM2_DECODER_PARITY_PY "
                    f"stage=input_key values={parity_values(keys, 0)}",
                    file=sys.stderr,
                )

        return capture

    def capture_parity_layer_output(index):
        def capture(_module, _inputs, output):
            if parity_layer_calls[index] == 0:
                print(
                    "SAM2_DECODER_PARITY_PY "
                    f"stage=layer_{index}_query values={parity_values(output[0], 2)}",
                    file=sys.stderr,
                )
                print(
                    "SAM2_DECODER_PARITY_PY "
                    f"stage=layer_{index}_key values={parity_values(output[1], 0)}",
                    file=sys.stderr,
                )
            parity_layer_calls[index] += 1

        return capture

    def capture_parity_transformer_output(_module, _inputs, output):
        if parity_transformer_calls[0] == 0:
            print(
                "SAM2_DECODER_PARITY_PY "
                f"stage=final_query values={parity_values(output[0], 2)}",
                file=sys.stderr,
            )
            print(
                "SAM2_DECODER_PARITY_PY "
                f"stage=final_key values={parity_values(output[1], 0)}",
                file=sys.stderr,
            )
        parity_transformer_calls[0] += 1

    def capture_parity_boundary(stage, token):
        parity_boundary_calls[stage] = 0

        def capture(_module, _inputs, output):
            if parity_boundary_calls[stage] == 0:
                print(
                    "SAM2_DECODER_PARITY_PY "
                    f"stage={stage} values={parity_values(output, token)}",
                    file=sys.stderr,
                )
            parity_boundary_calls[stage] += 1

        return capture

    def capture_parity_boundary_input(stage, token):
        parity_boundary_calls[stage] = 0

        def capture(_module, inputs):
            if parity_boundary_calls[stage] == 0:
                print(
                    "SAM2_DECODER_PARITY_PY "
                    f"stage={stage} values={parity_values(inputs[0], token)}",
                    file=sys.stderr,
                )
            parity_boundary_calls[stage] += 1

        return capture

    for index, layer in enumerate(decoder.transformer.layers):
        profile_handles.append(
            layer.register_forward_pre_hook(
                capture_parity_layer_input(index), with_kwargs=True
            )
        )
        profile_handles.append(
            layer.register_forward_hook(capture_parity_layer_output(index))
        )
        for norm_name, token in (
            ("norm1", 2),
            ("norm2", 2),
            ("norm3", 2),
            ("norm4", 0),
        ):
            stage = f"layer_{index}_{norm_name}_{'key' if norm_name == 'norm4' else 'query'}"
            profile_handles.append(
                getattr(layer, norm_name).register_forward_hook(
                    capture_parity_boundary(stage, token)
                )
            )
    profile_handles.append(
        decoder.transformer.final_attn_token_to_image.register_forward_hook(
            capture_parity_boundary("final_attention", 2)
        )
    )
    parity_attentions = [
        decoder.transformer.layers[0].cross_attn_token_to_image,
        decoder.transformer.layers[1].cross_attn_token_to_image,
        decoder.transformer.final_attn_token_to_image,
    ]
    for index, attention in enumerate(parity_attentions):
        for projection_name, token in (("q_proj", 2), ("k_proj", 0), ("v_proj", 0)):
            profile_handles.append(
                getattr(attention, projection_name).register_forward_hook(
                    capture_parity_boundary(
                        f"attention_{index}_{projection_name.replace('proj', 'projection')}",
                        token,
                    )
                )
            )
        profile_handles.append(
            attention.out_proj.register_forward_pre_hook(
                capture_parity_boundary_input(f"attention_{index}_attended", 2)
            )
        )
        profile_handles.append(
            attention.out_proj.register_forward_hook(
                capture_parity_boundary(f"attention_{index}_output_projection", 2)
            )
        )
    profile_handles.append(
        decoder.transformer.register_forward_hook(capture_parity_transformer_output)
    )

if os.environ.get("SAM2_PROFILE_IMAGE_STAGES"):
    def hook_image_module(name, module):
        starts = []
        profile_values[name] = []

        def before(_module, _inputs):
            synchronize()
            starts.append(time.perf_counter())

        def after(_module, _inputs, _output):
            synchronize()
            profile_values[name].append((time.perf_counter() - starts.pop()) * 1000.0)

        profile_handles.append(module.register_forward_pre_hook(before))
        profile_handles.append(module.register_forward_hook(after))

    image_encoder = model.image_encoder
    hook_image_module("image_encoder", image_encoder)
    hook_image_module("image_encoder_trunk", image_encoder.trunk)
    for index, block in enumerate(image_encoder.trunk.blocks):
        prefix = f"image_encoder_trunk_block_{index}"
        hook_image_module(prefix, block)
        hook_image_module(f"{prefix}_norm1", block.norm1)
        hook_image_module(f"{prefix}_attention", block.attn)
        hook_image_module(f"{prefix}_attention_qkv", block.attn.qkv)
        hook_image_module(f"{prefix}_attention_projection", block.attn.proj)
        hook_image_module(f"{prefix}_norm2", block.norm2)
        hook_image_module(f"{prefix}_mlp", block.mlp)
        for layer_index, layer in enumerate(block.mlp.layers):
            hook_image_module(f"{prefix}_mlp_layer_{layer_index}", layer)
        if hasattr(block, "proj"):
            hook_image_module(f"{prefix}_shortcut_projection", block.proj)
        if block.pool is not None:
            # Stage-transition blocks invoke the shared pool once for the
            # shortcut and once inside attention for Q. Preserve both samples.
            hook_image_module(f"{prefix}_pool", block.pool)
    hook_image_module("image_encoder_neck", image_encoder.neck)
    for index, module in enumerate(image_encoder.neck.convs):
        hook_image_module(f"image_encoder_fpn_lateral_{index}", module)
    hook_image_module("image_encoder_position", image_encoder.neck.position_encoding)

    decoder = model.sam_mask_decoder
    hook_image_module("image_encoder_skip_0", decoder.conv_s0)
    hook_image_module("image_encoder_skip_1", decoder.conv_s1)
    hook_image_module("prompt_encoder", model.sam_prompt_encoder)
    for index, module in enumerate(model.sam_prompt_encoder.mask_downscaling):
        hook_image_module(f"mask_downscaling_{index}", module)
    hook_image_module("decoder_total", decoder)
    hook_image_module("transformer", decoder.transformer)
    for index, layer in enumerate(decoder.transformer.layers):
        hook_image_module(f"transformer_layer_{index}", layer)
        for attention_name, attention in (
            ("self_attention", layer.self_attn),
            ("token_to_image", layer.cross_attn_token_to_image),
            ("image_to_token", layer.cross_attn_image_to_token),
        ):
            prefix = f"transformer_layer_{index}_{attention_name}"
            hook_image_module(prefix, attention)
            hook_image_module(f"{prefix}_q_proj", attention.q_proj)
            hook_image_module(f"{prefix}_k_proj", attention.k_proj)
            hook_image_module(f"{prefix}_v_proj", attention.v_proj)
            hook_image_module(f"{prefix}_out_proj", attention.out_proj)
    final_attention = decoder.transformer.final_attn_token_to_image
    hook_image_module("transformer_final_attention", final_attention)
    for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        hook_image_module(
            f"transformer_final_attention_{projection_name}",
            getattr(final_attention, projection_name),
        )
    hook_image_module("upconv_0", decoder.output_upscaling[0])
    hook_image_module("upnorm", decoder.output_upscaling[1])
    hook_image_module("first_gelu", decoder.output_upscaling[2])
    hook_image_module("upconv_3", decoder.output_upscaling[3])
    hook_image_module("final_gelu", decoder.output_upscaling[4])
    for index, module in enumerate(decoder.output_hypernetworks_mlps):
        hook_image_module(f"hypernetwork_{index}", module)
    hook_image_module("iou_head", decoder.iou_prediction_head)
    if decoder.pred_obj_score_head is not None:
        hook_image_module("object_score_head", decoder.pred_obj_score_head)
fixture = {name: value.to(args.device) for name, value in load_file(str(args.fixture)).items()}
image = fixture["image"]
point = fixture["point"]
point_label = torch.ones(point.shape[:-1], dtype=torch.int64, device=args.device)
rich_points = fixture["rich_points"]
rich_labels = fixture["rich_labels"]
correction_points = fixture["correction_points"]
box = fixture["box"]


def forward_image():
    image_output = model.forward_image(image)
    # Match SAM2ImagePredictor.set_image: static-image inference adds the
    # learned no-memory embedding after the image encoder. Calling the base
    # model's forward_image directly omits this required predictor step.
    if model.directly_add_no_mem_embed:
        image_output["vision_features"] = image_output["vision_features"] + model.no_mem_embed.reshape(1, -1, 1, 1)
    return image_output


def decode(image_output, points, labels, bounds, multimask, mask_input=None):
    point_inputs = None if points is None else (points, labels)
    sparse, dense = model.sam_prompt_encoder(point_inputs, bounds, mask_input)
    masks, _, _, _ = model.sam_mask_decoder(
        image_output["vision_features"],
        model.sam_prompt_encoder.get_dense_pe(),
        sparse,
        dense,
        multimask,
        points is not None and points.shape[0] > 1,
        image_output["backbone_fpn"][:2],
    )
    return masks


with torch.inference_mode():
    encode_ms = None
    cached = None
    if args.operation in ("cached_prompt_sequence", "cached_batch_and_prior_mask", "first_and_cached", "prompt_diversity"):
        # Match the suite protocol: L1 measures construction/first-ready work,
        # while inference cells start after lazy CUDA/cuDNN setup is complete.
        warm_cached = forward_image()
        synchronize()
        del warm_cached
        start = time.perf_counter()
        cached = forward_image()
        synchronize()
        encode_ms = (time.perf_counter() - start) * 1000.0
    elif args.operation not in ("first_point", "first_rich"):
        raise ValueError(f"operation {args.operation!r} is not implemented by the image runner")
    rss_after_encode_kb = status_kb("VmRSS:")
    offsets = torch.tensor(
        [[[0.0, 0.0]], [[32.0, 0.0]], [[0.0, 32.0]], [[-32.0, 0.0]]],
        device=args.device,
    )
    batch_points = point.expand(4, -1, -1) + offsets
    batch_corrections = correction_points[:, :1].expand(4, -1, -1) + offsets
    batch_labels = torch.ones((4, 1), dtype=torch.int64, device=args.device)
    diversity_points = []
    base_point = point[0, 0]
    for count in args.point_counts:
        values = []
        for index in range(count):
            if index == 0:
                values.append(base_point)
            else:
                offset = index - 1
                values.append(base_point + torch.tensor(
                    [(offset % 8) * 8.0 - 28.0, (offset // 8) * 8.0 - 12.0],
                    device=args.device,
                ))
        diversity_points.append(
            torch.stack(values).unsqueeze(0) if values else None
        )

    def run_once(index: int):
        if args.operation == "first_point":
            return decode(forward_image(), point, point_label, None, False)
        if args.operation == "first_rich":
            return decode(forward_image(), rich_points, rich_labels, box, False)
        if args.operation == "cached_batch_and_prior_mask":
            initial = decode(cached, batch_points, batch_labels, None, False)
            return decode(
                cached,
                batch_corrections,
                batch_labels,
                None,
                False,
                mask_input=initial,
            )
        if args.operation == "prompt_diversity":
            outputs = []
            for points in diversity_points:
                labels = None if points is None else torch.ones(
                    points.shape[:-1], dtype=torch.int64, device=args.device
                )
                outputs.append(decode(cached, points, labels, None, False))
            outputs.append(decode(cached, None, None, box, False))
            initial = decode(cached, point, point_label, None, False)
            outputs.append(decode(
                cached, rich_points[:, :1], rich_labels[:, :1], None, False,
                mask_input=initial,
            ))
            outputs.append(decode(cached, point, point_label, None, True))
            return torch.cat([value.reshape(-1) for value in outputs])
        if index % 3 == 0:
            return decode(cached, point, point_label, None, False)
        if index % 3 == 1:
            return decode(cached, rich_points, rich_labels, box, False)
        return decode(cached, rich_points[:, :1], rich_labels[:, :1], box, True)

    for index in range(args.warmups):
        output = run_once(index)
        synchronize()
    capture_ms = None
    graphs = []
    graph_outputs = []
    if args.mode == "graph":
        capture_count = 3 if args.operation == "cached_prompt_sequence" else 1
        # Warm every static route before capture. This is graph startup work,
        # not replay latency or capture time.
        for index in range(capture_count):
            output = run_once(index)
            synchronize()
        torch.cuda.reset_peak_memory_stats()
        capture_start = time.perf_counter()
        for index in range(capture_count):
            synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_output = run_once(index)
            graphs.append(graph)
            graph_outputs.append(captured_output)
        synchronize()
        capture_ms = (time.perf_counter() - capture_start) * 1000.0
    elif args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples_ms = []
    output = None
    for index in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        if args.mode == "graph":
            graph_index = index % len(graphs)
            graphs[graph_index].replay()
            output = graph_outputs[graph_index]
        else:
            output = run_once(index)
        synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

if args.output:
    args.output.write_bytes(output.detach().cpu().contiguous().numpy().astype("<f4").tobytes())

gpu_memory = {
    "allocated": None,
    "reserved": None,
    "peak_allocated": None,
    "peak_reserved": None,
}
if args.device == "cuda":
    gpu_memory = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }

for handle in profile_handles:
    handle.remove()
for name, values in profile_values.items():
    if values:
        recent = values[-2:]
        print(
            f"SAM2_IMAGE_STAGE_PY stage={name} sum_ms={sum(values):.6f} "
            f"calls={len(values)} recent_ms={','.join(f'{value:.6f}' for value in recent)}",
            file=sys.stderr,
        )

record = {
    "schema_version": 1,
    "implementation": "python-pytorch",
    "device": args.device,
    "mode": args.mode,
    "cell": args.cell,
    "operation": args.operation,
    "variant": args.variant,
    "iterations": args.iterations,
    "warmups": args.warmups,
    "samples_ms": samples_ms,
    "median_ms": statistics.median(samples_ms),
    "p95_ms": quantile(samples_ms, 0.95),
    "phases_ms": {"load": load_ms, "encode": encode_ms, "capture": capture_ms},
    "rss_kb": {
        "after_load": rss_after_load_kb,
        "after_encode": rss_after_encode_kb,
        "final": status_kb("VmRSS:"),
        "hwm": status_kb("VmHWM:"),
    },
    "gpu_memory_bytes": gpu_memory,
    "output": {"checksum": float(output.sum()), "elements": output.numel()},
}
print(json.dumps(record, separators=(",", ":")))
