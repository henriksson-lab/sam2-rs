#!/usr/bin/env python3
"""Generate the interactive three-frame/two-object R6 reference fixture."""

from collections import OrderedDict
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
from sam2.utils.transforms import SAM2Transforms  # noqa: E402
import sam2.utils.misc as sam2_misc  # noqa: E402


def connected_components(mask):
    labels_out, areas_out = [], []
    for sample in mask[:, 0].cpu().numpy().astype(np.uint8):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(sample, 8)
        areas = np.where(labels > 0, stats[labels, cv2.CC_STAT_AREA], 0)
        labels_out.append(torch.from_numpy(labels.astype(np.int64)))
        areas_out.append(torch.from_numpy(areas.astype(np.int64)))
    return torch.stack(labels_out)[:, None], torch.stack(areas_out)[:, None]


sam2_misc.get_connected_components = connected_components


def make_state(predictor, raw_frames):
    transform = SAM2Transforms(1024, 0.0)
    images = transform.forward_batch(raw_frames)
    state = {
        "images": images,
        "num_frames": len(raw_frames),
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": raw_frames[0].shape[0],
        "video_width": raw_frames[0].shape[1],
        "device": torch.device("cpu"),
        "storage_device": torch.device("cpu"),
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
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    return state


def main():
    checkpoint = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device="cpu",
        apply_postprocessing=True,
    )
    colors = [[30, 90, 180], [45, 105, 165], [60, 120, 150]]
    raw_frames = []
    for color in colors:
        image = np.empty((32, 48, 3), dtype=np.uint8)
        image[:] = color
        raw_frames.append(image)
    state = make_state(predictor, raw_frames)
    tensors = {
        f"image_{index}": torch.from_numpy(image.copy())
        for index, image in enumerate(raw_frames)
    }

    _, ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=101,
        points=np.array([[16.0, 16.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    tensors["add1_ids"] = torch.tensor(ids, dtype=torch.int64)
    tensors["add1_masks"] = masks
    _, ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=202,
        points=np.array([[34.0, 16.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    tensors["add2_ids"] = torch.tensor(ids, dtype=torch.int64)
    tensors["add2_masks"] = masks

    predictor.propagate_in_video_preflight(state)
    _, _, vision_feats, vision_pos, feat_sizes = predictor._get_image_feature(
        state, frame_idx=1, batch_size=1
    )
    for obj_index, object_id in enumerate(state["obj_ids"]):
        tensors[f"conditioned_{object_id}_1"] = predictor._prepare_memory_conditioned_features(
            1,
            False,
            [vision_feats[-1]],
            [vision_pos[-1]],
            [feat_sizes[-1]],
            state["output_dict_per_obj"][obj_index],
            3,
            False,
        )
    propagated = list(predictor.propagate_in_video(state))
    tensors["forward_frames"] = torch.tensor(
        [frame for frame, _, _ in propagated], dtype=torch.int64
    )
    for frame, ids, masks in propagated:
        tensors[f"forward_ids_{frame}"] = torch.tensor(ids, dtype=torch.int64)
        tensors[f"forward_masks_{frame}"] = masks
        for obj_index, object_id in enumerate(ids):
            out = state["output_dict_per_obj"][obj_index]
            item = out["cond_frame_outputs"].get(frame)
            if item is None:
                item = out["non_cond_frame_outputs"][frame]
            tensors[f"low_{object_id}_{frame}"] = item["pred_masks"]
            tensors[f"ptr_{object_id}_{frame}"] = item["obj_ptr"]
            tensors[f"memory_{object_id}_{frame}"] = item["maskmem_features"]

    _, ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=1,
        obj_id=101,
        points=np.array([[22.0, 12.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        clear_old_points=True,
    )
    tensors["correction_ids"] = torch.tensor(ids, dtype=torch.int64)
    tensors["correction_masks"] = masks
    corrected_outputs = state["temp_output_dict_per_obj"][0]
    corrected = corrected_outputs["cond_frame_outputs"].get(1)
    if corrected is None:
        corrected = corrected_outputs["non_cond_frame_outputs"][1]
    tensors["correction_low"] = corrected["pred_masks"]
    tensors["correction_ptr"] = corrected["obj_ptr"]

    reverse = list(
        predictor.propagate_in_video(
            state, start_frame_idx=1, max_frame_num_to_track=1, reverse=True
        )
    )
    tensors["reverse_frames"] = torch.tensor(
        [frame for frame, _, _ in reverse], dtype=torch.int64
    )
    for frame, ids, masks in reverse:
        tensors[f"reverse_ids_{frame}"] = torch.tensor(ids, dtype=torch.int64)
        tensors[f"reverse_masks_{frame}"] = masks
        for obj_index, object_id in enumerate(ids):
            out = state["output_dict_per_obj"][obj_index]
            item = out["cond_frame_outputs"].get(frame)
            if item is None:
                item = out["non_cond_frame_outputs"][frame]
            tensors[f"reverse_low_{object_id}_{frame}"] = item["pred_masks"]
            tensors[f"reverse_ptr_{object_id}_{frame}"] = item["obj_ptr"]
            tensors[f"reverse_memory_{object_id}_{frame}"] = item["maskmem_features"]

    _, ids, masks = predictor.clear_all_prompts_in_frame(state, 1, 101)
    tensors["clear_ids"] = torch.tensor(ids, dtype=torch.int64)
    tensors["clear_masks"] = masks

    ids, updated = predictor.remove_object(state, 202)
    tensors["remove_ids"] = torch.tensor(ids, dtype=torch.int64)
    tensors["remove_frames"] = torch.tensor(
        [frame for frame, _ in updated], dtype=torch.int64
    )
    for frame, masks in updated:
        tensors[f"remove_masks_{frame}"] = masks

    predictor.reset_state(state)
    tensors["reset_counts"] = torch.tensor(
        [len(state["obj_ids"]), len(state["output_dict_per_obj"])], dtype=torch.int64
    )

    input_mask = np.zeros((32, 48), dtype=bool)
    input_mask[8:24, 12:36] = True
    _, ids, masks = predictor.add_new_mask(state, 2, 303, input_mask)
    tensors["mask_input"] = torch.from_numpy(input_mask.astype(np.uint8))
    tensors["mask_ids"] = torch.tensor(ids, dtype=torch.int64)
    tensors["mask_masks"] = masks
    mask_output = state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][2]
    tensors["mask_low"] = mask_output["pred_masks"]
    tensors["mask_ptr"] = mask_output["obj_ptr"]

    destination = ROOT / "checkpoints" / "sam2_r6_video.safetensors"
    save_file(
        {name: value.detach().cpu().clone().contiguous() for name, value in tensors.items()},
        str(destination),
        metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"},
    )
    print(destination)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
