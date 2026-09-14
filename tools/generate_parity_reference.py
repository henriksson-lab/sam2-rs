#!/usr/bin/env python3
"""Save deterministic intermediate tensors from the pinned SAM2 reference."""

from pathlib import Path
import sys
import argparse

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "segment-anything-2"))

from sam2.build_sam import build_sam2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", default="tiny", choices=["tiny", "small", "base_plus", "large"])
    variant = parser.parse_args().variant
    yaml_name = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}[variant]
    checkpoint = ROOT / "checkpoints" / f"sam2.1_hiera_{variant}.pt"
    destination = ROOT / "checkpoints" / f"sam2.1_hiera_{variant}_parity.safetensors"
    # Match the Rust decoder's deterministic single-mask path. The upstream
    # builder otherwise injects an optional dynamic multimask fallback.
    model = build_sam2(f"configs/sam2.1/sam2.1_hiera_{yaml_name}.yaml", str(checkpoint), device="cpu", apply_postprocessing=False)
    image = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
    points = torch.tensor([[[512.0, 512.0]]])
    labels = torch.tensor([[1]], dtype=torch.int64)
    with torch.inference_mode():
        image_output = model.forward_image(image)
        sparse, dense = model.sam_prompt_encoder((points, labels), None, None)
        masks, ious, mask_tokens, obj_scores = model.sam_mask_decoder(
            image_output["vision_features"],
            model.sam_prompt_encoder.get_dense_pe(),
            sparse,
            dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=image_output["backbone_fpn"][:2],
        )
        # Mirror MaskDecoder.predict_masks' high-resolution upscaling branch.
        output_tokens = torch.cat([
            model.sam_mask_decoder.obj_score_token.weight,
            model.sam_mask_decoder.iou_token.weight,
            model.sam_mask_decoder.mask_tokens.weight,
        ]).unsqueeze(0)
        tokens = torch.cat((output_tokens, sparse), dim=1)
        src = image_output["vision_features"] + dense
        hs, src = model.sam_mask_decoder.transformer(src, model.sam_prompt_encoder.get_dense_pe(), tokens)
        b, c, h, w = src.shape[0], src.shape[2], 64, 64
        src = src.transpose(1, 2).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = model.sam_mask_decoder.output_upscaling
        upscaled = act2(dc2(act1(ln1(dc1(src) + image_output["backbone_fpn"][1]))) + image_output["backbone_fpn"][0])
        mask_tokens_for_hyper = hs[:, 2:6, :]
        hyper_in = torch.stack([mlp(mask_tokens_for_hyper[:, i, :]) for i, mlp in enumerate(model.sam_mask_decoder.output_hypernetworks_mlps)], dim=1)
        manual_masks = (hyper_in @ upscaled.view(1, 32, 256 * 256)).view(1, 4, 256, 256)[:, :1]
        print("Python manual mask difference", (manual_masks - masks).abs().max().item())
    tensors = {
        "vision_features": image_output["vision_features"].contiguous(),
        "backbone_fpn_0": image_output["backbone_fpn"][0].contiguous(),
        "backbone_fpn_1": image_output["backbone_fpn"][1].contiguous(),
        "sparse_embeddings": sparse.contiguous(),
        "dense_embeddings": dense.contiguous(),
        "mask_logits": masks.contiguous(),
        "mask_tokens": mask_tokens.contiguous(),
        "upscaled_embedding": upscaled.contiguous(),
        "hyper_in": hyper_in.contiguous(),
        "iou_predictions": ious.contiguous(),
        "object_score_logits": obj_scores.contiguous(),
    }
    save_file(tensors, str(destination), metadata={"upstream_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"})
    print(destination)


if __name__ == "__main__":
    main()
