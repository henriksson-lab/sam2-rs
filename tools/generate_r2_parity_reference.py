#!/usr/bin/env python3
"""Generate deterministic parity tensors for translation batch R2."""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.utils.amg import (  # noqa: E402
    batch_iterator,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from sam2.utils.transforms import SAM2Transforms  # noqa: E402


def components_8(masks):
    labels_out = []
    areas_out = []
    for mask in masks[:, 0].cpu().numpy().astype(np.uint8):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        areas = np.where(labels > 0, stats[labels, cv2.CC_STAT_AREA], 0)
        labels_out.append(torch.from_numpy(labels.astype(np.int64)))
        areas_out.append(torch.from_numpy(areas.astype(np.int64)))
    return torch.stack(labels_out)[:, None], torch.stack(areas_out)[:, None]


def postprocess_masks(masks, threshold, max_hole, max_sprinkle, output_hw):
    masks = masks.float()
    flat = masks.flatten(0, 1).unsqueeze(1)
    output = masks
    if max_hole:
        labels, areas = components_8(flat <= threshold)
        output = torch.where(
            ((labels > 0) & (areas <= max_hole)).reshape_as(masks),
            threshold + 10.0,
            output,
        )
    if max_sprinkle:
        labels, areas = components_8(flat > threshold)
        output = torch.where(
            ((labels > 0) & (areas <= max_sprinkle)).reshape_as(masks),
            threshold - 10.0,
            output,
        )
    return F.interpolate(output, output_hw, mode="bilinear", align_corners=False)


def padded_rle_counts(rles):
    width = max(len(rle["counts"]) for rle in rles)
    counts = torch.full((len(rles), width), -1, dtype=torch.int64)
    lengths = torch.empty(len(rles), dtype=torch.int64)
    for row, rle in enumerate(rles):
        values = torch.tensor(rle["counts"], dtype=torch.int64)
        counts[row, : len(values)] = values
        lengths[row] = len(values)
    return counts, lengths


def coco_counts_bytes(rle):
    """Pinned frontend jscocotools rleToString implementation."""
    output = []
    counts = rle["counts"]
    for index, count in enumerate(counts):
        value = count - (counts[index - 2] if index > 2 else 0)
        while True:
            char = value & 0x1F
            value >>= 5
            more = value != (-1 if char & 0x10 else 0)
            if more:
                char |= 0x20
            output.append(char + 48)
            if not more:
                break
    return output


def main():
    resolution = 6
    transforms = SAM2Transforms(resolution, 0.0)
    image_landscape = np.empty((3, 5, 3), dtype=np.uint8)
    image_landscape[:] = [32, 128, 224]
    image_portrait = np.empty((5, 3, 3), dtype=np.uint8)
    image_portrait[:] = [200, 80, 10]
    image_batch = transforms.forward_batch([image_landscape, image_portrait])

    coordinates = torch.tensor(
        [[[0.0, 0.0], [4.0, 2.0]], [[1.5, 3.5], [2.0, 4.0]]],
        dtype=torch.float32,
    )
    boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0], [0.5, 1.0, 2.5, 4.0]])

    masks = torch.tensor(
        [[
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, 1.0, 1.0],
            ],
        ]],
        dtype=torch.float32,
    )

    grids = build_all_layer_point_grids(8, 2, 2)
    grid_values = torch.from_numpy(np.concatenate(grids).astype(np.float32))
    grid_lengths = torch.tensor([len(grid) for grid in grids], dtype=torch.int64)
    crop_boxes, crop_layers = generate_crop_boxes((7, 11), 2, 0.2)

    edge_boxes = torch.tensor(
        [
            [0.0, 30.0, 20.0, 60.0],
            [20.0, 0.0, 50.0, 80.0],
            [20.0, 20.0, 60.0, 60.0],
        ]
    )
    crop_box = [30, 0, 90, 80]
    original_box = [0, 0, 120, 80]

    binary_masks = torch.tensor(
        [
            [[0, 1, 1], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
        ],
        dtype=torch.bool,
    )
    rles = mask_to_rle_pytorch(binary_masks)
    rle_counts, rle_lengths = padded_rle_counts(rles)
    coco_counts = [coco_counts_bytes(rle) for rle in rles]
    coco_width = max(map(len, coco_counts))
    coco_bytes = torch.zeros((len(rles), coco_width), dtype=torch.uint8)
    coco_lengths = torch.empty(len(rles), dtype=torch.int64)
    for row, values in enumerate(coco_counts):
        coco_bytes[row, : len(values)] = torch.tensor(values, dtype=torch.uint8)
        coco_lengths[row] = len(values)
    rle_decoded = torch.from_numpy(
        np.stack([rle_to_mask(rle).astype(np.uint8) for rle in rles])
    )

    stability_masks = torch.tensor(
        [[[2.0, 0.5], [-0.5, -2.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    local_boxes = torch.tensor([[0.0, 0.0, 2.0, 1.0], [1.0, 1.0, 3.0, 2.0]])
    local_points = torch.tensor([[0.5, 0.5], [2.0, 1.5]])
    local_masks = torch.tensor(
        [[[1, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 1]]], dtype=torch.uint8
    )
    uncrop_box = [2, 1, 5, 3]

    batches = list(batch_iterator(3, np.arange(8)))
    batch_ranges = torch.tensor(
        [[int(batch[0][0]), int(batch[0][-1]) + 1] for batch in batches],
        dtype=torch.int64,
    )

    tensors = {
        "image_landscape": torch.from_numpy(image_landscape),
        "image_portrait": torch.from_numpy(image_portrait),
        "image_batch": image_batch,
        "coordinates": coordinates,
        "coordinates_transformed": transforms.transform_coords(
            coordinates, normalize=True, orig_hw=(5, 5)
        ),
        "boxes": boxes,
        "boxes_transformed": transforms.transform_boxes(
            boxes, normalize=True, orig_hw=(5, 5)
        ),
        "postprocess_input": masks,
        "postprocess_output": postprocess_masks(masks, 0.0, 1, 1, (7, 9)),
        "point_grids": grid_values,
        "point_grid_lengths": grid_lengths,
        "crop_boxes": torch.tensor(crop_boxes, dtype=torch.int64),
        "crop_layers": torch.tensor(crop_layers, dtype=torch.int64),
        "batch_ranges": batch_ranges,
        "edge_boxes": edge_boxes,
        "near_crop_edge": is_box_near_crop_edge(
            edge_boxes, crop_box, original_box
        ).to(torch.uint8),
        "binary_masks": binary_masks.to(torch.uint8),
        "rle_counts": rle_counts,
        "rle_lengths": rle_lengths,
        "rle_areas": torch.tensor(
            [sum(rle["counts"][1::2]) for rle in rles], dtype=torch.int64
        ),
        "coco_counts": coco_bytes,
        "coco_lengths": coco_lengths,
        "rle_decoded": rle_decoded,
        "xyxy_box": torch.tensor([1.0, 2.0, 6.0, 9.0]),
        "xywh_box": box_xyxy_to_xywh(torch.tensor([1.0, 2.0, 6.0, 9.0])),
        "stability_masks": stability_masks,
        "stability_scores": calculate_stability_score(stability_masks, 0.0, 1.0),
        "local_boxes": local_boxes,
        "uncropped_boxes": uncrop_boxes_xyxy(local_boxes, uncrop_box),
        "local_points": local_points,
        "uncropped_points": uncrop_points(local_points, uncrop_box),
        "local_masks": local_masks,
        "uncropped_masks": uncrop_masks(local_masks, uncrop_box, 5, 7),
    }
    destination = ROOT / "checkpoints" / "sam2_r2_parity.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        str(destination),
        metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"},
    )
    print(destination)


if __name__ == "__main__":
    main()
