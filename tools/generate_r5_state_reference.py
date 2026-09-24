#!/usr/bin/env python3
"""Serialize a deterministic SAM2 video-state editing sequence for R5."""

from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def object_index(state, object_id):
    if object_id not in state["obj_id_to_idx"]:
        index = len(state["obj_id_to_idx"])
        state["obj_id_to_idx"][object_id] = index
        state["objects"].append(
            {
                "points": {},
                "masks": {},
                "cond": {},
                "non_cond": {},
                "temp_cond": {},
                "temp_non_cond": {},
                "tracked": {},
            }
        )
    return state["obj_id_to_idx"][object_id]


def remove_object(state, object_id):
    index = state["obj_id_to_idx"].pop(object_id)
    state["objects"].pop(index)
    state["obj_id_to_idx"] = OrderedDict(
        (key, value - int(value > index))
        for key, value in state["obj_id_to_idx"].items()
    )


def main():
    state = {"obj_id_to_idx": OrderedDict(), "objects": []}
    for object_id in [101, 7, 55, 7]:
        object_index(state, object_id)

    obj101 = state["objects"][state["obj_id_to_idx"][101]]
    coords_a = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    labels_a = torch.tensor([[2, 3]], dtype=torch.int64)
    coords_b = torch.tensor([[[50.0, 60.0]]])
    labels_b = torch.tensor([[1]], dtype=torch.int64)
    obj101["points"][0] = (coords_a, labels_a)
    obj101["points"][0] = (
        torch.cat([obj101["points"][0][0], coords_b], dim=1),
        torch.cat([obj101["points"][0][1], labels_b], dim=1),
    )
    obj101["tracked"][0] = False
    obj101["tracked"][4] = True
    obj101["cond"][0] = 100
    obj101["non_cond"].update({0: 200, 1: 201, 3: 203, 4: 204})
    obj101["temp_non_cond"][2] = 302

    obj55 = state["objects"][state["obj_id_to_idx"][55]]
    obj55["points"][2] = (torch.tensor([[[1.0, 2.0]]]), torch.tensor([[1]]))
    obj55["masks"][2] = torch.tensor([[[[1.0]]]])
    obj55["points"].pop(2, None)
    obj55["non_cond"].update({1: 401, 2: 402, 3: 403, 4: 404})

    for obj in state["objects"]:
        for frame in range(1, 4):
            obj["non_cond"].pop(frame, None)

    remove_object(state, 7)

    pred_masks = torch.tensor(
        [[[[2.0, -20.0], [1.0, 3.0]]], [[[-2.0, 4.0], [0.0, 2.0]]]]
    )
    max_indices = pred_masks.argmax(dim=0, keepdim=True)
    object_indices = torch.arange(2)[:, None, None, None]
    constrained = torch.where(
        max_indices == object_indices, pred_masks, pred_masks.clamp(max=-10.0)
    )

    tensors = {
        "object_ids": torch.tensor(list(state["obj_id_to_idx"]), dtype=torch.int64),
        "object_indices": torch.tensor(
            list(state["obj_id_to_idx"].values()), dtype=torch.int64
        ),
        "point_coords": obj101["points"][0][0],
        "point_labels": obj101["points"][0][1],
        "obj101_non_cond_frames": torch.tensor(
            list(obj101["non_cond"]), dtype=torch.int64
        ),
        "obj55_non_cond_frames": torch.tensor(
            list(obj55["non_cond"]), dtype=torch.int64
        ),
        "tracked_frames": torch.tensor(list(obj101["tracked"]), dtype=torch.int64),
        "tracked_reverse": torch.tensor(
            [int(value) for value in obj101["tracked"].values()], dtype=torch.uint8
        ),
        "mask55": obj55["masks"][2],
        "cache_indices": torch.tensor([1], dtype=torch.int64),
        "non_overlap_input": pred_masks,
        "non_overlap_output": constrained,
    }
    destination = ROOT / "checkpoints" / "sam2_r5_state.safetensors"
    save_file(
        tensors,
        str(destination),
        metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"},
    )
    print(destination)


if __name__ == "__main__":
    main()
