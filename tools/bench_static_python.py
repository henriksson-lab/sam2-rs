#!/usr/bin/env python3
"""Pinned Python SAM2 static benchmark over the shared safetensors fixture."""
from pathlib import Path
import argparse, json, os, statistics, sys, time
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/segment-anything-2"))
from sam2.build_sam import build_sam2

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("fixture", type=Path)
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--output", type=Path)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--profile-stages", action="store_true")
args = parser.parse_args()

fixture = load_file(str(args.fixture))
if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device cuda requested but PyTorch cannot access a CUDA device")
model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(args.checkpoint), device=args.device, apply_postprocessing=False)
image, points, labels = (fixture["image"].to(args.device), fixture["points"].to(args.device), fixture["labels"].to(args.device))

def infer():
    image_output = model.forward_image(image)
    sparse, dense = model.sam_prompt_encoder((points, labels), None, None)
    masks, _, _, _ = model.sam_mask_decoder(image_output["vision_features"], model.sam_prompt_encoder.get_dense_pe(), sparse, dense, False, False, image_output["backbone_fpn"][:2])
    return masks

with torch.inference_mode():
    infer()
    if args.device == "cuda":
        torch.cuda.synchronize()
    samples = []
    stage_profile = None
    for iteration in range(args.iterations):
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if args.profile_stages and iteration == 0:
            image_output = model.forward_image(image)
            if args.device == "cuda":
                torch.cuda.synchronize()
            encoder_ms = (time.perf_counter() - start) * 1000
            decoder_start = time.perf_counter()
            sparse, dense = model.sam_prompt_encoder((points, labels), None, None)
            masks, _, _, _ = model.sam_mask_decoder(image_output["vision_features"], model.sam_prompt_encoder.get_dense_pe(), sparse, dense, False, False, image_output["backbone_fpn"][:2])
            if args.device == "cuda":
                torch.cuda.synchronize()
            stage_profile = (encoder_ms, (time.perf_counter() - decoder_start) * 1000)
        else:
            masks = infer()
            if args.device == "cuda":
                torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

rss_hwm_kb = 0
for line in Path("/proc/self/status").read_text().splitlines():
    if line.startswith("VmHWM:"):
        rss_hwm_kb = int(line.split()[1])
if args.output:
    masks.detach().cpu().float().numpy().astype("<f4").tofile(args.output)
result = {"implementation": "python-torch", "device": args.device, "iterations": args.iterations, "median_ms": statistics.median(samples), "rss_hwm_kb": rss_hwm_kb, "mask_checksum": float(masks.sum())}
if stage_profile:
    result["encoder_ms"], result["decoder_ms"] = stage_profile
print(json.dumps(result))
