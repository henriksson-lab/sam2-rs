#!/usr/bin/env python3
"""Convert an official SAM2 PyTorch checkpoint to a flat safetensors state dict.

The Rust implementation intentionally reads only this safe, inspectable format.
The source archive is loaded with ``weights_only=True`` and its ``model`` member
is preserved verbatim: no key renaming or dtype conversion occurs here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="official .pt checkpoint")
    parser.add_argument("destination", type=Path, help="output .safetensors path")
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
        raise ValueError("expected a string-keyed state_dict or a checkpoint with a `model` state_dict")
    tensors = {key: tensor.contiguous() for key, tensor in state_dict.items()}
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise ValueError("state_dict contains a non-tensor value")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.destination, metadata={"source": str(args.source), "format": "sam2-pytorch-state-dict"})
    print(f"wrote {len(tensors)} tensors to {args.destination}")


if __name__ == "__main__":
    main()
