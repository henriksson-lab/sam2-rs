#!/usr/bin/env python3
"""Generate a real-image automatic-mask-generator parity fixture for R4."""

from pathlib import Path
import sys

import numpy as np
import torch
import cv2
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
import sam2.utils.misc as sam2_misc  # noqa: E402


def connected_components(mask):
    labels_out = []
    areas_out = []
    device = mask.device
    for sample in mask[:, 0].cpu().numpy().astype(np.uint8):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(sample, 8)
        areas = np.where(labels > 0, stats[labels, cv2.CC_STAT_AREA], 0)
        labels_out.append(torch.from_numpy(labels.astype(np.int64)))
        areas_out.append(torch.from_numpy(areas.astype(np.int64)))
    return (
        torch.stack(labels_out)[:, None].to(device),
        torch.stack(areas_out)[:, None].to(device),
    )


sam2_misc.get_connected_components = connected_components


def encode_annotations(prefix, annotations, tensors):
    tensors[f"{prefix}_count"] = torch.tensor([len(annotations)], dtype=torch.int64)
    if not annotations:
        return
    tensors[f"{prefix}_boxes"] = torch.tensor([item["bbox"] for item in annotations])
    tensors[f"{prefix}_points"] = torch.tensor(
        [item["point_coords"][0] for item in annotations]
    )
    tensors[f"{prefix}_ious"] = torch.tensor(
        [item["predicted_iou"] for item in annotations]
    )
    tensors[f"{prefix}_stability"] = torch.tensor(
        [item["stability_score"] for item in annotations]
    )
    tensors[f"{prefix}_crops"] = torch.tensor(
        [item["crop_box"] for item in annotations]
    )
    tensors[f"{prefix}_areas"] = torch.tensor(
        [item["area"] for item in annotations], dtype=torch.int64
    )
    counts = [item["segmentation"]["counts"] for item in annotations]
    maximum = max(map(len, counts))
    padded = torch.full((len(counts), maximum), -1, dtype=torch.int64)
    lengths = torch.empty(len(counts), dtype=torch.int64)
    for row, values in enumerate(counts):
        padded[row, : len(values)] = torch.tensor(values, dtype=torch.int64)
        lengths[row] = len(values)
    tensors[f"{prefix}_rle_counts"] = padded
    tensors[f"{prefix}_rle_lengths"] = lengths


def run(model, image, **kwargs):
    generator = SAM2AutomaticMaskGenerator(
        model,
        pred_iou_thresh=-1.0,
        stability_score_thresh=-1.0,
        output_mode="uncompressed_rle",
        **kwargs,
    )
    return generator.generate(image)


def main():
    checkpoint = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device="cpu",
        apply_postprocessing=True,
    )
    image = np.asarray(
        Image.open(ROOT / "segment-anything" / "notebooks" / "images" / "dog.jpg")
        .convert("RGB")
        .resize((160, 96))
    )
    tensors = {"image": torch.from_numpy(image.copy())}

    encode_annotations(
        "base",
        run(
            model,
            image,
            points_per_side=2,
            points_per_batch=3,
            multimask_output=True,
        ),
        tensors,
    )
    encode_annotations(
        "explicit",
        run(
            model,
            image,
            points_per_side=None,
            point_grids=[np.array([[0.3, 0.4], [0.7, 0.6]], dtype=np.float32)],
            points_per_batch=1,
            multimask_output=False,
        ),
        tensors,
    )
    encode_annotations(
        "m2m",
        run(
            model,
            image,
            points_per_side=1,
            points_per_batch=1,
            use_m2m=True,
            multimask_output=True,
        ),
        tensors,
    )
    encode_annotations(
        "cleanup",
        run(
            model,
            image,
            points_per_side=1,
            points_per_batch=1,
            min_mask_region_area=20,
            multimask_output=False,
        ),
        tensors,
    )
    encode_annotations(
        "crops",
        run(
            model,
            image,
            points_per_side=1,
            points_per_batch=2,
            crop_n_layers=1,
            crop_nms_thresh=0.3,
            multimask_output=False,
        ),
        tensors,
    )
    encode_annotations(
        "default",
        SAM2AutomaticMaskGenerator(
            model,
            output_mode="uncompressed_rle",
        ).generate(image),
        tensors,
    )

    destination = ROOT / "checkpoints" / "sam2_r4_parity.safetensors"
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(destination),
        metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"},
    )
    print(destination)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
