#!/usr/bin/env python3
"""Build deterministic shared CPU-suite tensors and a hash manifest.

The checked-in TOML is authoritative. Large generated tensors stay under the
ignored checkpoints directory; the compact JSON manifest is checked in and
lets benchmark archives prove exactly which decoded data they used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tomllib

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "benchmarks/cpu-suite.toml"
DEFAULT_OUTPUT = ROOT / "checkpoints/cpu-suite"
DEFAULT_MANIFEST = ROOT / "benchmarks/fixtures.json"
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_safetensors(path: Path) -> None:
    """Make a safetensors header byte-stable without copying its payload."""
    temporary = path.with_name(f".{path.name}.canonical")
    with path.open("rb") as source:
        encoded_length = source.read(8)
        if len(encoded_length) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        header_length = struct.unpack("<Q", encoded_length)[0]
        encoded_header = source.read(header_length)
        if len(encoded_header) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(encoded_header)
        canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        canonical += b" " * (-len(canonical) % 8)
        with temporary.open("wb") as destination:
            destination.write(struct.pack("<Q", len(canonical)))
            destination.write(canonical)
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(path)


def save_fixture(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    save_file(tensors, str(path), metadata=metadata)
    canonicalize_safetensors(path)


def derived_image(name: str, entry: dict, fixtures: dict) -> tuple[Image.Image, Path]:
    parent = fixtures[entry["derived_from"]]
    source = ROOT / parent["source"]
    image = Image.open(source).convert("RGB")
    transform = entry["transform"]
    kind, value = transform.split(":", 1)
    if kind == "crop" and value.startswith("center:"):
        width, height = map(int, value.removeprefix("center:").split("x"))
        if width > image.width or height > image.height:
            raise ValueError(f"{name}: crop {width}x{height} exceeds {image.size}")
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
    elif kind == "resize":
        width, height = map(int, value.split("x"))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    else:
        raise ValueError(f"unsupported deterministic transform: {transform}")
    return image, source


def load_image(name: str, entry: dict, fixtures: dict) -> tuple[Image.Image, Path]:
    if "source" in entry:
        source = ROOT / entry["source"]
        return Image.open(source).convert("RGB"), source
    return derived_image(name, entry, fixtures)


def scale_xy(points: list[list[float]], width: int, height: int) -> torch.Tensor:
    values = [[[x * 1024.0 / width, y * 1024.0 / height] for x, y in points]]
    return torch.tensor(values, dtype=torch.float32)


def build_one(name: str, entry: dict, fixtures: dict, output_dir: Path) -> dict:
    image, source = load_image(name, entry, fixtures)
    width, height = image.size
    rgb = np.asarray(image, dtype=np.uint8).copy()
    resized = image.resize((1024, 1024), Image.Resampling.BILINEAR)
    normalized = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255.0
    normalized = ((normalized - MEAN) / STD).unsqueeze(0).contiguous()

    positive = entry["positive_points"]
    negative = entry["negative_points"]
    correction = entry["correction_points"]
    rich = [*positive, *negative]
    box = entry["box"]
    tensors = {
        "image": normalized,
        "rgb_u8": torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
        "point": scale_xy(positive[:1], width, height),
        # Compatibility aliases for the original single-cell benchmark.
        "points": scale_xy(positive[:1], width, height),
        "labels": torch.ones((1, 1), dtype=torch.int64),
        "positive_points": scale_xy(positive, width, height),
        "negative_points": scale_xy(negative, width, height),
        "correction_points": scale_xy(correction, width, height),
        "rich_points": scale_xy(rich, width, height),
        "rich_labels": torch.tensor([[1] * len(positive) + [0] * len(negative)], dtype=torch.int64),
        "box": torch.tensor(
            [[box[0] * 1024.0 / width, box[1] * 1024.0 / height,
              box[2] * 1024.0 / width, box[3] * 1024.0 / height]],
            dtype=torch.float32,
        ),
        "original_size": torch.tensor([height, width], dtype=torch.int64),
    }
    output = output_dir / f"{name}.safetensors"
    save_fixture(
        tensors,
        output,
        metadata={
            "fixture": name,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "original_size": f"{width}x{height}",
            "transform": entry.get("transform", "none"),
        },
    )
    return {
        "name": name,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "generated": str(output.relative_to(ROOT)),
        "generated_sha256": sha256(output),
        "width": width,
        "height": height,
        "transform": entry.get("transform"),
        "license": entry["license"],
    }


def video_manifest(name: str, entry: dict, output_dir: Path) -> dict:
    source = ROOT / entry["source"]
    if source.is_dir():
        candidates = sorted(source.glob("*.jpg"), key=lambda path: int(path.stem))
        selected = candidates[entry["start"]::entry["stride"]][:entry["frames"]]
        if len(selected) != entry["frames"]:
            raise ValueError(f"{name}: wanted {entry['frames']} frames, found {len(selected)}")
        digest = hashlib.sha256()
        frames = []
        for path in selected:
            value = sha256(path)
            digest.update(bytes.fromhex(value))
            frames.append({"path": str(path.relative_to(ROOT)), "sha256": value})
        tensors = {}
        dimensions = None
        for index, path in enumerate(selected):
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
            if dimensions is None:
                dimensions = (rgb.shape[1], rgb.shape[0])
            elif dimensions != (rgb.shape[1], rgb.shape[0]):
                raise ValueError(f"{name}: frame dimensions are inconsistent")
            tensors[f"frame_{index:05}"] = torch.from_numpy(rgb).contiguous()
        output = output_dir / f"{name}.safetensors"
        save_fixture(
            tensors,
            output,
            metadata={
                "fixture": name,
                "source": str(source.relative_to(ROOT)),
                "frames": str(len(selected)),
                "size": f"{dimensions[0]}x{dimensions[1]}",
            },
        )
        encoded_dir = output_dir / "encoded" / name
        if encoded_dir.exists():
            shutil.rmtree(encoded_dir)
        encoded_dir.mkdir(parents=True)
        for index, path in enumerate(selected):
            shutil.copyfile(path, encoded_dir / f"{index:05}.jpg")
        return {
            "name": name,
            "source": str(source.relative_to(ROOT)),
            "frames": frames,
            "sequence_sha256": digest.hexdigest(),
            "generated": str(output.relative_to(ROOT)),
            "generated_sha256": sha256(output),
            "encoded": str(encoded_dir.relative_to(ROOT)),
            "width": dimensions[0],
            "height": dimensions[1],
            "license": entry["license"],
        }
    encoded_dir = output_dir / "encoded"
    encoded_dir.mkdir(parents=True, exist_ok=True)
    encoded = encoded_dir / f"{name}.mp4"
    start = entry["start"]
    stride = entry["stride"]
    # Keep an exact, deterministic frame subset in the encoded artifact. Both
    # implementations consume this same file, so decode and session setup stay
    # apples-to-apples while a stride can cover motion across a longer clip.
    select = f"gte(n\\,{start})*not(mod(n-{start}\\,{stride}))"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", f"select={select}", "-vsync", "vfr",
            "-frames:v", str(entry["frames"]), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(encoded),
        ],
        check=True,
    )
    return {
        "name": name,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "encoded": str(encoded.relative_to(ROOT)),
        "encoded_sha256": sha256(encoded),
        "frames": entry["frames"],
        "start": start,
        "stride": stride,
        "license": entry["license"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    spec_bytes = args.spec.read_bytes()
    spec = tomllib.loads(spec_bytes.decode())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [
        build_one(name, entry, spec["fixtures"], args.output_dir)
        for name, entry in spec["fixtures"].items()
    ]
    videos = [video_manifest(name, entry, args.output_dir) for name, entry in spec["videos"].items()]
    manifest = {
        "schema_version": 1,
        "spec": str(args.spec.relative_to(ROOT)),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "images": images,
        "videos": videos,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(args.manifest), "images": len(images), "videos": len(videos)}))


if __name__ == "__main__":
    main()
