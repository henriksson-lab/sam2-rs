#!/usr/bin/env python3
"""Regenerate the checked-in SAM2.1 checkpoint key/shape manifests."""

from pathlib import Path

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("tiny", "small", "base_plus", "large")


def main():
    destination = ROOT / "manifests"
    destination.mkdir(exist_ok=True)
    for variant in VARIANTS:
        checkpoint = ROOT / "checkpoints" / f"sam2.1_hiera_{variant}.safetensors"
        output = destination / f"sam2.1_hiera_{variant}.manifest"
        with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
            lines = []
            for key in sorted(tensors.keys()):
                tensor = tensors.get_slice(key)
                shape = ",".join(str(value) for value in tensor.get_shape())
                lines.append(f"{key}\t{shape}\t{tensor.get_dtype()}\n")
        output.write_text("".join(lines))
        print(f"{output}: {len(lines)} tensors")


if __name__ == "__main__":
    main()
