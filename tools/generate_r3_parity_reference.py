#!/usr/bin/env python3
"""Generate cached/batched image-predictor parity tensors for R3."""

from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


def decoder_object_scores(predictor, image_index, coords, labels, boxes, masks, multimask):
    mask, coords, labels, boxes = predictor._prep_prompts(
        coords, labels, boxes, masks, True, img_idx=image_index
    )
    points = (coords, labels) if coords is not None else None
    if boxes is not None:
        box_coords = boxes.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
        box_labels = box_labels.repeat(boxes.size(0), 1)
        if points is None:
            points = (box_coords, box_labels)
        else:
            points = (
                torch.cat([box_coords, points[0]], dim=1),
                torch.cat([box_labels, points[1]], dim=1),
            )
    sparse, dense = predictor.model.sam_prompt_encoder(points, None, mask)
    high_res = [
        level[image_index].unsqueeze(0)
        for level in predictor._features["high_res_feats"]
    ]
    _, _, _, object_scores = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][image_index].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask,
        repeat_image=points is not None and points[0].shape[0] > 1,
        high_res_features=high_res,
    )
    return object_scores


def main():
    checkpoint = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device="cpu",
        apply_postprocessing=True,
    )
    predictor = SAM2ImagePredictor(model)

    image0 = np.empty((7, 11, 3), dtype=np.uint8)
    image0[:] = [24, 96, 208]
    image1 = np.empty((11, 7, 3), dtype=np.uint8)
    image1[:] = [180, 60, 12]
    predictor.set_image_batch([image0, image1])

    coords0 = np.array(
        [[[2.0, 1.0], [8.0, 5.0]], [[5.0, 3.0], [9.0, 2.0]]],
        dtype=np.float32,
    )
    labels0 = np.array([[1, 0], [1, 1]], dtype=np.int32)
    boxes0 = np.array([[1.0, 0.5, 9.0, 6.0], [2.0, 1.0, 10.0, 5.5]], dtype=np.float32)
    masks0 = np.zeros((2, 1, 256, 256), dtype=np.float32)
    masks0[1, 0, 96:160, 80:176] = 2.0

    coords = [coords0, None]
    labels = [labels0, None]
    boxes = [boxes0, None]
    masks = [masks0, None]

    tensors = {
        "image0": torch.from_numpy(image0),
        "image1": torch.from_numpy(image1),
        "embedding": predictor.get_image_embedding(),
        "coords0": torch.from_numpy(coords0),
        "labels0": torch.from_numpy(labels0.astype(np.int64)),
        "boxes0": torch.from_numpy(boxes0),
        "masks0": torch.from_numpy(masks0),
    }
    for multimask, prefix in [(True, "multi"), (False, "single")]:
        output_masks, output_ious, output_low = predictor.predict_batch(
            coords,
            labels,
            boxes,
            masks,
            multimask_output=multimask,
            return_logits=True,
            normalize_coords=True,
        )
        for index in range(2):
            tensors[f"{prefix}_masks_{index}"] = torch.from_numpy(output_masks[index])
            tensors[f"{prefix}_ious_{index}"] = torch.from_numpy(output_ious[index])
            tensors[f"{prefix}_low_{index}"] = torch.from_numpy(output_low[index])
            tensors[f"{prefix}_objects_{index}"] = decoder_object_scores(
                predictor,
                index,
                coords[index],
                labels[index],
                boxes[index],
                masks[index],
                multimask,
            )

    predictor.set_image(image0)
    cached_masks, cached_ious, cached_low = predictor.predict(
        np.array([[4.0, 3.0]], dtype=np.float32),
        np.array([1], dtype=np.int32),
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    )
    tensors["single_embedding"] = predictor.get_image_embedding()
    tensors["cached_binary_masks"] = torch.from_numpy(cached_masks.astype(np.uint8))
    tensors["cached_ious"] = torch.from_numpy(cached_ious)
    tensors["cached_low"] = torch.from_numpy(cached_low)

    destination = ROOT / "checkpoints" / "sam2_r3_parity.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        str(destination),
        metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"},
    )
    print(destination)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
