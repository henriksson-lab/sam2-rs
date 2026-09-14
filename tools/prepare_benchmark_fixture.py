#!/usr/bin/env python3
"""Freeze a real SAM2 image and prompt into a shared benchmark fixture."""
from pathlib import Path
import argparse
import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path, default=ROOT / "vendor/segment-anything-2/notebooks/videos/bedroom/00089.jpg")
parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/bench_bedroom.safetensors")
args = parser.parse_args()

image = Image.open(args.image).convert("RGB")
width, height = image.size
image = image.resize((1024, 1024), Image.Resampling.BILINEAR)
x = torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).float() / 255
x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
save_file({
    "image": x.unsqueeze(0).contiguous(),
    "points": torch.tensor([[[512.0, 512.0]]]),
    "labels": torch.tensor([[1]], dtype=torch.int64),
}, str(args.output), metadata={"source": str(args.image), "original_size": f"{width}x{height}"})
print(args.output)
