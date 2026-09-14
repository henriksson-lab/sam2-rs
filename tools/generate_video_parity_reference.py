#!/usr/bin/env python3
"""Save a deterministic two-frame SAM2.1 temporal inference fixture."""

from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.build_sam import build_sam2  # noqa: E402


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
    destination = ROOT / "checkpoints" / "sam2.1_hiera_tiny_video_parity.safetensors"
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device="cpu",
        apply_postprocessing=False,
    )
    # The Rust fixture exercises the deterministic single-mask API on both
    # frames, not Python's optional dynamic/multimask selection policy.
    model.multimask_output_in_sam = False
    model.multimask_output_for_tracking = False
    image = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    points = {"point_coords": torch.tensor([[[512.0, 512.0]]]), "point_labels": torch.tensor([[1]])}
    with torch.inference_mode():
        backbone_out = model.forward_image(image)
        _, feats, pos, sizes = model._prepare_backbone_features(backbone_out)
        outputs = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        first = model.track_step(0, True, feats, pos, sizes, points, None, outputs, 2)
        outputs["cond_frame_outputs"][0] = first
        frame1_conditioned = model._prepare_memory_conditioned_features(
            1, False, [feats[-1]], [pos[-1]], [sizes[-1]], outputs, 2, False
        )
        second = model.track_step(1, False, feats, pos, sizes, None, None, outputs, 2)
    tensors = {
        "frame0_low_res_masks": first["pred_masks"].contiguous(),
        "frame0_high_res_masks": first["pred_masks_high_res"].contiguous(),
        "frame0_memory_features": first["maskmem_features"].contiguous(),
        "frame0_memory_pos": first["maskmem_pos_enc"][-1].contiguous(),
        "frame0_object_pointer": first["obj_ptr"].contiguous(),
        "frame1_low_res_masks": second["pred_masks"].contiguous(),
        "frame1_conditioned": frame1_conditioned.contiguous(),
        "frame1_memory_features": second["maskmem_features"].contiguous(),
    }
    save_file(tensors, str(destination), metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"})
    print(destination)


if __name__ == "__main__":
    main()
