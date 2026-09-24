#!/usr/bin/env python3
"""Generate deterministic parity tensors for translation batch R1."""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.modeling.sam.mask_decoder import MaskDecoder  # noqa: E402
from sam2.utils.amg import batched_mask_to_box, remove_small_regions  # noqa: E402
from sam2.utils.misc import concat_points, mask_to_box  # noqa: E402


def dynamic_multimask(logits, ious, delta=0.05, threshold=0.98):
    """Pinned MaskDecoder formulas without constructing checkpoint weights."""
    single = logits[:, :1]
    flattened = single.flatten(-2)
    area_i = (flattened > delta).sum(-1).float()
    area_u = (flattened > -delta).sum(-1).float()
    stability = torch.where(area_u > 0, area_i / area_u, 1.0)
    multi_ious = ious[:, 1:]
    indices = multi_ious.argmax(-1)
    batch = torch.arange(logits.shape[0])
    best_logits = logits[:, 1:][batch, indices].unsqueeze(1)
    best_ious = multi_ious[batch, indices].unsqueeze(1)
    stable = stability >= threshold
    return (
        stability,
        torch.where(stable[..., None, None], single, best_logits),
        torch.where(stable, ious[:, :1], best_ious),
    )


def components_8(masks):
    labels_out = []
    areas_out = []
    for mask in masks[:, 0].cpu().numpy().astype(np.uint8):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        areas = np.where(labels > 0, stats[labels, cv2.CC_STAT_AREA], 0)
        labels_out.append(torch.from_numpy(labels.astype(np.int64)))
        areas_out.append(torch.from_numpy(areas.astype(np.int64)))
    return torch.stack(labels_out)[:, None], torch.stack(areas_out)[:, None]


def fill_holes(scores, max_area):
    labels, areas = components_8(scores <= 0)
    return torch.where((labels > 0) & (areas <= max_area), 0.1, scores)


def main():
    logits = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]], [[11.0] * 2] * 2, [[12.0] * 2] * 2, [[13.0] * 2] * 2],
            [[[0.0, 0.0], [1.0, -1.0]], [[21.0] * 2] * 2, [[22.0] * 2] * 2, [[23.0] * 2] * 2],
        ],
        dtype=torch.float32,
    )
    ious = torch.tensor(
        [[0.7, 0.8, 0.6, 0.5], [0.1, 0.2, 0.9, 0.5]], dtype=torch.float32
    )
    stability, selected_logits, selected_ious = dynamic_multimask(logits, ious)

    component_mask = torch.tensor(
        [[[[1, 0, 0], [1, 0, 0], [0, 1, 1]]]], dtype=torch.uint8
    )
    component_labels, component_areas = components_8(component_mask)
    hole_scores = torch.tensor(
        [[[[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]]],
        dtype=torch.float32,
    )

    boxes_mask = torch.tensor(
        [[[0, 1], [1, 0]], [[0, 0], [0, 0]]], dtype=torch.uint8
    )
    misc_boxes_mask = boxes_mask[:, None]

    island_mask = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool
    )
    islands, islands_changed = remove_small_regions(island_mask, 3, "islands")
    hole_mask = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool
    )
    holes, holes_changed = remove_small_regions(hole_mask, 2, "holes")

    old_points = torch.tensor([[[1.0, 2.0]]])
    old_labels = torch.tensor([[1]], dtype=torch.int64)
    new_points = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
    new_labels = torch.tensor([[0, 1]], dtype=torch.int64)
    concatenated = concat_points(
        {"point_coords": old_points, "point_labels": old_labels},
        new_points,
        new_labels,
    )

    tensors = {
        "dynamic_logits": logits,
        "dynamic_ious": ious,
        "dynamic_stability": stability,
        "dynamic_selected_logits": selected_logits,
        "dynamic_selected_ious": selected_ious,
        "component_mask": component_mask,
        "component_labels": component_labels,
        "component_areas": component_areas,
        "hole_scores": hole_scores,
        "hole_scores_filled": fill_holes(hole_scores, 1),
        "boxes_mask": boxes_mask,
        # Both upstream helpers are defined for boolean masks. Keep the stored
        # input as uint8 because Candle safetensors does not expose a bool dtype.
        "boxes_amg": batched_mask_to_box(boxes_mask.bool()).to(torch.int64),
        "boxes_misc": mask_to_box(misc_boxes_mask.bool()).to(torch.int64),
        "island_mask": torch.from_numpy(island_mask.astype(np.uint8)),
        "island_filtered": torch.from_numpy(islands.astype(np.uint8)),
        "island_changed": torch.tensor([islands_changed], dtype=torch.uint8),
        "hole_mask": torch.from_numpy(hole_mask.astype(np.uint8)),
        "hole_filtered": torch.from_numpy(holes.astype(np.uint8)),
        "hole_changed": torch.tensor([holes_changed], dtype=torch.uint8),
        "old_points": old_points,
        "old_labels": old_labels,
        "new_points": new_points,
        "new_labels": new_labels,
        "concatenated_points": concatenated["point_coords"],
        "concatenated_labels": concatenated["point_labels"],
    }
    destination = ROOT / "checkpoints" / "sam2_r1_parity.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        destination,
        metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"},
    )
    print(destination)


if __name__ == "__main__":
    main()
