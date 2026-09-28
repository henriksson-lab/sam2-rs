#!/usr/bin/env python3
"""Run paired Rust/PyTorch SAM2 parity-suite processes and score them.

Examples:
  tools/run_parity_suite.py --list
  tools/run_parity_suite.py --cell I1 --fixture bedroom --device cuda
  tools/run_parity_suite.py --cell I3 --device cpu --threads 20 --repeats 4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "benchmarks/cpu-suite.toml"
SUPPORTED_OPERATIONS = {
    "first_point",
    "first_rich",
    "cached_prompt_sequence",
    "cached_batch_and_prior_mask",
    "first_and_cached",
    "model_load",
    "video_init",
    "video_forward_one",
    "video_edit_three",
    "video_long_lifetime",
    "encoded_video_init",
    "repeated_lifetime",
    "prompt_diversity",
    "image_batch",
    "worker_scaling",
    "amg_practical",
    "amg_default",
}

# User-visible timing boundaries emitted by both runners. The orchestrator
# validates them before scoring, so a one-sided boundary change cannot silently
# produce a non-apples-to-apples result.
TIMING_PHASES = {
    "first_point": {"load", "encode", "capture"},
    "first_rich": {"load", "encode", "capture"},
    "cached_prompt_sequence": {"load", "encode", "capture"},
    "cached_batch_and_prior_mask": {"load", "encode", "capture"},
    "first_and_cached": {"load", "encode", "capture"},
    "prompt_diversity": {"load", "encode", "capture"},
    "model_load": {"load"},
    "video_init": {"load", "fixture_decode", "session_init"},
    "video_forward_one": {"load", "fixture_decode", "session_init"},
    "video_edit_three": {"load", "fixture_decode", "session_init"},
    "video_long_lifetime": {"load", "fixture_decode", "session_init"},
    "encoded_video_init": {"load", "fixture_decode"},
    "repeated_lifetime": {"load", "fixture_decode"},
    "image_batch": {"load", "fixture_decode", "encode", "decode", "capture"},
    "amg_practical": {"load", "fixture_decode"},
    "amg_default": {"load", "fixture_decode"},
}

REQUIRED_CPU_P0 = {"I1", "I2", "I3", "I4", "V1", "V2", "V3", "V4", "A1", "A2", "L1", "S1"}
REQUIRED_CPU_P1 = {"W1", "M1", "B1", "V5", "P1", "R1"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--cell", action="append", help="cell ID; repeat to select several")
    parser.add_argument("--fixture", action="append", help="limit to named fixture(s)")
    parser.add_argument("--variant", action="append", help="limit to named variant(s)")
    parser.add_argument(
        "--source-cell",
        action="append",
        help="limit W1 to source cell(s), for example I1 or V2",
    )
    parser.add_argument(
        "--batch-size",
        action="append",
        type=int,
        help="limit B1 to one or more configured image batch sizes",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--cuda-pool-retain-mib",
        type=int,
        help="override sam2-rs CUDA pool retention in MiB (0 disables retention)",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--rust-features",
        help="comma-separated Cargo features overriding the suite defaults",
    )
    parser.add_argument("--cpu-list")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--profile-video-stages",
        action="store_true",
        help="diagnostic: use synchronized Rust video stages; headline timing is invalid",
    )
    parser.add_argument(
        "--profile-amg-stages",
        action="store_true",
        help="diagnostic: synchronize and report AMG stages; headline timing is invalid",
    )
    parser.add_argument(
        "--profile-image-stages",
        action="store_true",
        help="diagnostic: synchronize and report matched image-decoder stages; headline timing is invalid",
    )
    parser.add_argument(
        "--profile-a2-boundary",
        action="store_true",
        help=(
            "diagnostic: archive matched A2 boundary-pixel decoder operands; "
            "implies synchronized AMG/image-stage profiling"
        ),
    )
    parser.add_argument(
        "--profile-decoder-parity",
        action="store_true",
        help=(
            "diagnostic: archive matched batch-one decoder query/key vectors "
            "at transformer boundaries; headline timing is invalid"
        ),
    )
    parser.add_argument(
        "--profile-v3-frames",
        action="store_true",
        help="diagnostic: archive initial-forward 256x256 logits for every V3 frame/object",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run model-free CPU suite schema, timing-contract, and parity smoke checks",
    )
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--allow-parity-failure",
        action="store_true",
        help="diagnostic only: archive a pair that violates output thresholds",
    )
    parser.add_argument("--max-load-per-cpu", type=float, default=0.75)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepend(env: dict, name: str, path: Path) -> None:
    current = env.get(name)
    env[name] = f"{path}{os.pathsep}{current}" if current else str(path)


def cuda_build_environment(env: dict) -> tempfile.TemporaryDirectory | None:
    """Expose the cuDNN bundled with PyTorch to Rust's linker and loader."""
    if ARGS.device != "cuda":
        return None
    import torch

    cudnn_lib = Path(torch.__file__).resolve().parent.parent / "nvidia/cudnn/lib"
    if not cudnn_lib.is_dir():
        return None
    prepend(env, "LD_LIBRARY_PATH", cudnn_lib)
    if (cudnn_lib / "libcudnn.so").exists():
        prepend(env, "LIBRARY_PATH", cudnn_lib)
        return None
    versioned = sorted(cudnn_lib.glob("libcudnn.so.*"))
    if not versioned:
        return None
    temporary_link = tempfile.TemporaryDirectory(prefix="sam2-cudnn-link-")
    link_dir = Path(temporary_link.name)
    (link_dir / "libcudnn.so").symlink_to(versioned[-1])
    prepend(env, "LIBRARY_PATH", link_dir)
    return temporary_link


def cpu_model() -> str:
    try:
        return next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (FileNotFoundError, StopIteration):
        return "unknown"


def gpu_info() -> dict | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,temperature.gpu,pstate,clocks.current.sm,power.draw,compute_mode",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None
    fields = [value.strip() for value in output.split(",")]
    names = ("index", "name", "uuid", "driver", "temperature_c", "pstate", "sm_clock_mhz", "power_w", "compute_mode")
    return dict(zip(names, fields))


def gpu_used_mib(pid: int) -> float | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=1,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return float(fields[1])
            except ValueError:
                return None
    return None


def run_process(
    command: list[str], env: dict, device: str, cpu_list: str | None
) -> tuple[dict, str, float | None]:
    if cpu_list:
        command = ["taskset", "--cpu-list", cpu_list, *command]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak_gpu_mib = None
    stop = threading.Event()

    def monitor():
        nonlocal peak_gpu_mib
        while not stop.wait(0.02):
            value = gpu_used_mib(process.pid)
            if value is not None:
                peak_gpu_mib = value if peak_gpu_mib is None else max(peak_gpu_mib, value)

    thread = threading.Thread(target=monitor, daemon=True)
    if device == "cuda":
        thread.start()
    stdout, stderr = process.communicate()
    stop.set()
    if device == "cuda":
        thread.join(timeout=1)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}\n{stderr}")
    lines = [line for line in stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"runner emitted no JSON: {' '.join(command)}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    record = json.loads(lines[-1])
    record["external_gpu_peak_mib"] = peak_gpu_mib
    return record, stderr, peak_gpu_mib


def parity(left: Path, right: Path, segments: int = 1) -> dict:
    rust_bytes = left.read_bytes()
    python_bytes = right.read_bytes()
    if len(rust_bytes) != len(python_bytes) or len(rust_bytes) % 4:
        raise RuntimeError(f"output sizes differ: Rust {len(rust_bytes)}, Python {len(python_bytes)}")
    pairs = [(a[0], b[0]) for a, b in zip(struct.iter_unpack("<f", rust_bytes), struct.iter_unpack("<f", python_bytes))]
    differences = [abs(a - b) for a, b in pairs]
    segment_size = len(differences) // segments if differences and len(differences) % segments == 0 else 0
    return {
        "elements": len(differences),
        "max_abs": max(differences, default=0.0),
        "mean_abs": statistics.fmean(differences) if differences else 0.0,
        "threshold_mismatches": sum((a > 0.0) != (b > 0.0) for a, b in pairs),
        "segment_max_abs": (
            [max(differences[index:index + segment_size]) for index in range(0, len(differences), segment_size)]
            if segment_size else []
        ),
    }


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def validate_spec(spec: dict, spec_bytes: bytes) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("suite schema_version must be 1")
    for section in ("suite", "gates", "parity", "variants", "fixtures", "videos", "cells"):
        if section not in spec:
            raise ValueError(f"suite is missing [{section}]")
    suite = spec["suite"]
    if suite.get("paired_processes", 0) < 1 or suite.get("primary_workers", 0) < 1:
        raise ValueError("suite paired_processes and primary_workers must be positive")
    if not suite.get("portability_workers"):
        raise ValueError("suite portability_workers must not be empty")

    cells = spec["cells"]
    ids = [cell.get("id") for cell in cells]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in ids):
        raise ValueError("every cell needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("suite cell IDs must be unique")
    by_id = {cell["id"]: cell for cell in cells}
    cpu_ids = {
        cell["id"]
        for cell in cells
        if "cpu" in cell.get("devices", ["cpu"])
    }
    missing_p0 = REQUIRED_CPU_P0 - cpu_ids
    missing_p1 = REQUIRED_CPU_P1 - cpu_ids
    if missing_p0 or missing_p1:
        raise ValueError(
            f"CPU matrix is incomplete; missing P0={sorted(missing_p0)}, P1={sorted(missing_p1)}"
        )

    fixture_names = set(spec["fixtures"]) | set(spec["videos"])
    variant_names = set(spec["variants"])
    for cell in cells:
        label = cell["id"]
        if cell.get("tier") not in {"p0", "p1"}:
            raise ValueError(f"{label}: tier must be p0 or p1")
        operation = cell.get("operation")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"{label}: unsupported operation {operation!r}")
        if operation != "worker_scaling" and operation not in TIMING_PHASES:
            raise ValueError(f"{label}: operation {operation!r} has no timing contract")
        if not cell.get("variants") or not set(cell["variants"]) <= variant_names:
            raise ValueError(f"{label}: variants are empty or reference an unknown variant")
        if not cell.get("fixtures"):
            raise ValueError(f"{label}: fixtures must not be empty")
        if operation == "worker_scaling":
            unknown = set(cell["fixtures"]) - set(by_id)
            if unknown:
                raise ValueError(f"{label}: unknown source cells {sorted(unknown)}")
            if set(cell.get("workers", [])) != set(suite["portability_workers"]):
                raise ValueError(f"{label}: workers must match suite.portability_workers")
        else:
            unknown = set(cell["fixtures"]) - fixture_names
            if unknown:
                raise ValueError(f"{label}: unknown fixtures {sorted(unknown)}")
        for key in ("weight", "iterations"):
            if cell.get(key, 0) < 1:
                raise ValueError(f"{label}: {key} must be positive")
        if cell.get("warmups", -1) < 0:
            raise ValueError(f"{label}: warmups must be nonnegative")

    manifest = json.loads((ROOT / "benchmarks/fixtures.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("fixture manifest schema_version must be 1")
    expected_spec_hash = hashlib.sha256(spec_bytes).hexdigest()
    if manifest.get("spec_sha256") != expected_spec_hash:
        raise ValueError("fixture manifest was generated from a different suite specification")
    manifest_images = {entry.get("name") for entry in manifest.get("images", [])}
    manifest_videos = {entry.get("name") for entry in manifest.get("videos", [])}
    if manifest_images != set(spec["fixtures"]):
        raise ValueError("fixture manifest image names do not match the suite")
    if manifest_videos != set(spec["videos"]):
        raise ValueError("fixture manifest video names do not match the suite")
    for group in (manifest.get("images", []), manifest.get("videos", [])):
        for entry in group:
            digest = entry.get("generated_sha256", entry.get("encoded_sha256"))
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"fixture {entry.get('name')!r} has an invalid generated SHA-256")


def validate_result_record(
    record: dict,
    implementation: str,
    cell: dict,
    device: str,
    iterations: int,
    warmups: int,
    runtime_variant: str | None = None,
) -> None:
    expected_implementation = {"rust": "rust-candle", "python": "python-pytorch"}[implementation]
    expected = {
        "schema_version": 1,
        "implementation": expected_implementation,
        "device": device,
        "cell": cell["id"],
        "operation": cell["operation"],
        # Expanded cells retain the stable TOML key (for example
        # `base_plus`), while both executables receive and report the public
        # runtime spelling (`base-plus`). Most variants happen to use the same
        # spelling, so validate the explicit runtime value rather than relying
        # on that coincidence.
        "variant": runtime_variant or cell["variant"],
        "iterations": iterations,
        "warmups": warmups,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"{implementation} {cell['id']} reported {key}={record.get(key)!r}; expected {value!r}"
            )
    samples = record.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != iterations:
        raise RuntimeError(f"{implementation} {cell['id']} emitted the wrong sample count")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in samples):
        raise RuntimeError(f"{implementation} {cell['id']} emitted an invalid timing sample")
    for key, value in (
        ("median_ms", statistics.median(samples)),
        ("p95_ms", nearest_rank(samples, 0.95)),
    ):
        actual = record.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(
                f"{implementation} {cell['id']} reported {key}={actual!r}; expected {value:.9g}"
            )
    phases = record.get("phases_ms")
    required_phases = TIMING_PHASES[cell["operation"]]
    if not isinstance(phases, dict) or not required_phases <= set(phases):
        missing = required_phases - set(phases or {})
        raise RuntimeError(f"{implementation} {cell['id']} is missing timing phases {sorted(missing)}")
    if any(
        value is not None
        and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0)
        for value in phases.values()
    ):
        raise RuntimeError(f"{implementation} {cell['id']} emitted an invalid phase timing")
    rss = record.get("rss_kb")
    if not isinstance(rss, dict) or not all(isinstance(rss.get(key), int) and rss[key] >= 0 for key in ("final", "hwm")):
        raise RuntimeError(f"{implementation} {cell['id']} emitted invalid RSS fields")
    if rss["hwm"] < rss["final"]:
        raise RuntimeError(f"{implementation} {cell['id']} has VmHWM below final VmRSS")
    output = record.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("elements"), int) or output["elements"] < 0:
        raise RuntimeError(f"{implementation} {cell['id']} emitted invalid output metadata")


def validate_timing_pair(rust: dict, python: dict, cell: dict, allow_diagnostics: bool) -> None:
    rust_phases = set(rust["phases_ms"])
    python_phases = set(python["phases_ms"])
    if not allow_diagnostics and rust_phases != python_phases:
        raise RuntimeError(
            f"{cell['id']} timing boundaries differ: Rust={sorted(rust_phases)}, "
            f"Python={sorted(python_phases)}"
        )


def run_validation_smoke(spec: dict) -> None:
    # Exercise the same validator used for real processes, including the
    # even-length median case that previously differed by runner.
    for operation, phases in TIMING_PHASES.items():
        variant_key = "base_plus" if operation == "first_and_cached" else "tiny"
        runtime_variant = spec["variants"][variant_key]["rust"]
        cell = {"id": "SMOKE", "operation": operation, "variant": variant_key}
        records = {}
        for implementation, implementation_name in (
            ("rust", "rust-candle"),
            ("python", "python-pytorch"),
        ):
            record = {
                "schema_version": 1,
                "implementation": implementation_name,
                "device": "cpu",
                "cell": "SMOKE",
                "operation": operation,
                "variant": runtime_variant,
                "iterations": 4,
                "warmups": 1,
                "samples_ms": [4.0, 1.0, 3.0, 2.0],
                "median_ms": 2.5,
                "p95_ms": 4.0,
                "phases_ms": {phase: 0.0 for phase in phases},
                "rss_kb": {"final": 10, "hwm": 12},
                "output": {"checksum": 0.0, "elements": 4},
            }
            validate_result_record(
                record, implementation, cell, "cpu", 4, 1, runtime_variant
            )
            records[implementation] = record
        validate_timing_pair(records["rust"], records["python"], cell, False)

    mismatched = {
        "phases_ms": {"load": 0.0},
    }
    try:
        validate_timing_pair(
            {"phases_ms": {"load": 0.0, "encode": 0.0}},
            mismatched,
            {"id": "SMOKE"},
            False,
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("timing-boundary mismatch smoke was not rejected")

    with tempfile.TemporaryDirectory(prefix="sam2-cpu-suite-smoke-") as directory:
        directory = Path(directory)
        rust_output = directory / "rust.f32"
        python_output = directory / "python.f32"
        rust_output.write_bytes(
            b"".join(struct.pack("<f", value) for value in (-1.0, 0.25, 1.0, 4.0))
        )
        python_output.write_bytes(
            b"".join(struct.pack("<f", value) for value in (-1.0, -0.25, 1.5, 4.0))
        )
        result = parity(rust_output, python_output, segments=2)
        if result != {
            "elements": 4,
            "max_abs": 0.5,
            "mean_abs": 0.25,
            "threshold_mismatches": 1,
            "segment_max_abs": [0.5, 0.5],
        }:
            raise RuntimeError(f"parity smoke produced an unexpected result: {result}")

    print(
        f"CPU suite validation passed: {len(spec['cells'])} cells, "
        f"{len(TIMING_PHASES)} timing contracts, parity smoke passed"
    )


def expand(spec: dict) -> list[dict]:
    selected = []
    cells_by_id = {cell["id"]: cell for cell in spec["cells"]}
    requested_cells = set(ARGS.cell or [])
    requested_fixtures = set(ARGS.fixture or [])
    requested_variants = set(ARGS.variant or [])
    requested_sources = set(ARGS.source_cell or [])
    requested_batch_sizes = set(ARGS.batch_size or [])
    available_worker_sources = set()
    for cell in spec["cells"]:
        if requested_cells and cell["id"] not in requested_cells:
            continue
        if ARGS.device not in cell.get("devices", ["cpu"]):
            continue
        if ARGS.device == "cuda" and ARGS.mode not in cell.get("gpu_modes", ["eager"]):
            continue
        if cell["operation"] == "worker_scaling":
            if ARGS.device != "cpu":
                continue
            workers = [
                count for count in cell["workers"]
                if ARGS.threads is None or count == ARGS.threads
            ]
            for source_id in cell["fixtures"]:
                available_worker_sources.add(source_id)
                if requested_sources and source_id not in requested_sources:
                    continue
                source = cells_by_id.get(source_id)
                if source is None:
                    raise ValueError(f"W1 references unknown source cell {source_id!r}")
                if "cpu" not in source.get("devices", ["cpu"]):
                    raise ValueError(f"W1 source cell {source_id!r} does not support CPU")
                source_fixtures = [
                    name for name in source["fixtures"]
                    if not requested_fixtures or name in requested_fixtures
                ]
                source_variants = [
                    name for name in source["variants"]
                    if name in cell["variants"]
                    and (not requested_variants or name in requested_variants)
                ]
                for worker_count in workers:
                    for fixture in source_fixtures:
                        for variant in source_variants:
                            selected.append({
                                **source,
                                "id": cell["id"],
                                "tier": cell["tier"],
                                "weight": cell["weight"],
                                "source_cell": source_id,
                                "worker_count": worker_count,
                                "result_fixture": f"{source_id}/{fixture}/w{worker_count}",
                                "fixture": fixture,
                                "variant": variant,
                            })
            continue
        fixtures = [name for name in cell["fixtures"] if not requested_fixtures or name in requested_fixtures]
        variants = [name for name in cell["variants"] if not requested_variants or name in requested_variants]
        if cell["operation"] == "image_batch":
            # A B1 row is one heterogeneous batch, not one row per fixture.
            # Spread batch=2 across the fixture pool so it also exercises
            # different source aspect ratios; batch=4 consumes the full pool.
            for batch_size in cell["batches"]:
                if requested_batch_sizes and batch_size not in requested_batch_sizes:
                    continue
                if len(fixtures) < batch_size:
                    continue
                if batch_size == 1:
                    batch_fixtures = fixtures[:1]
                else:
                    indices = [round(index * (len(fixtures) - 1) / (batch_size - 1)) for index in range(batch_size)]
                    batch_fixtures = [fixtures[index] for index in indices]
                for variant in variants:
                    selected.append({
                        **cell,
                        "fixture": f"mixed{batch_size}",
                        "fixture_names": batch_fixtures,
                        "batch_size": batch_size,
                        "variant": variant,
                    })
            continue
        for fixture in fixtures:
            for variant in variants:
                selected.append({**cell, "fixture": fixture, "variant": variant})
    unknown = requested_cells - {cell["id"] for cell in spec["cells"]}
    if unknown:
        raise ValueError(f"unknown cells: {', '.join(sorted(unknown))}")
    unknown_sources = requested_sources - available_worker_sources
    if unknown_sources:
        raise ValueError(f"unknown W1 source cells: {', '.join(sorted(unknown_sources))}")
    configured_batch_sizes = {
        size
        for cell in spec["cells"]
        if cell["operation"] == "image_batch"
        for size in cell["batches"]
    }
    unknown_batch_sizes = requested_batch_sizes - configured_batch_sizes
    if unknown_batch_sizes:
        raise ValueError(f"unknown B1 batch sizes: {', '.join(map(str, sorted(unknown_batch_sizes)))}")
    return selected


def weighted_geomean(rows: list[dict], ratio_key: str) -> float | None:
    grouped = {}
    for row in rows:
        if ratio_key not in row or row[ratio_key] is None:
            continue
        grouped.setdefault(row["cell"], []).append(row[ratio_key])
    if not grouped:
        return None
    weights = {row["cell"]: row["weight"] for row in rows}
    total = sum(weights[cell] for cell in grouped)
    return math.exp(sum(weights[cell] * statistics.fmean(map(math.log, ratios)) for cell, ratios in grouped.items()) / total)


ARGS = parse_args()
SPEC_BYTES = ARGS.spec.read_bytes()
SPEC = tomllib.loads(SPEC_BYTES.decode())
validate_spec(SPEC, SPEC_BYTES)
if ARGS.validate_only:
    if ARGS.device != "cpu":
        raise SystemExit("--validate-only is a CPU-only smoke check")
    run_validation_smoke(SPEC)
    raise SystemExit
CELLS = expand(SPEC)
if ARGS.list:
    for cell in SPEC["cells"]:
        support = "ready" if cell["operation"] in SUPPORTED_OPERATIONS else "planned"
        devices = ",".join(cell.get("devices", ["cpu"]))
        print(f"{cell['id']:>3} {support:>7} {devices:<8} {cell['operation']}")
    raise SystemExit
if ARGS.mode == "graph" and ARGS.device != "cuda":
    raise SystemExit("graph mode requires CUDA")
if ARGS.mode == "graph" and any(
    cell["operation"] not in {"cached_prompt_sequence", "cached_batch_and_prior_mask"}
    for cell in CELLS
):
    raise SystemExit("graph mode currently supports only I3 and I4")
if not CELLS:
    raise SystemExit("selection expanded to no workload rows")
unsupported = sorted({cell["operation"] for cell in CELLS} - SUPPORTED_OPERATIONS)
if unsupported:
    raise SystemExit(f"selected operations are not implemented yet: {', '.join(unsupported)}")
if ARGS.repeats is None:
    ARGS.repeats = SPEC["suite"]["paired_processes"]
if ARGS.repeats < 1:
    raise SystemExit("--repeats must be positive")
if ARGS.threads is None:
    ARGS.threads = SPEC["suite"]["primary_workers"]
worker_counts = sorted({cell.get("worker_count", ARGS.threads) for cell in CELLS})
if ARGS.device == "cpu" and ARGS.cpu_list is not None and len(worker_counts) > 1:
    raise SystemExit("--cpu-list cannot describe a multi-worker W1 run; omit it or select one --threads value")
if ARGS.device == "cpu" and ARGS.cpu_list is None and len(worker_counts) == 1:
    ARGS.cpu_list = SPEC["suite"].get("cpu_affinity", {}).get(str(ARGS.threads))
    if ARGS.cpu_list is None:
        raise SystemExit(
            f"no CPU affinity is specified for {ARGS.threads} workers; "
            "pass --cpu-list or add suite.cpu_affinity"
        )

load_per_cpu = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
if load_per_cpu > ARGS.max_load_per_cpu and not ARGS.allow_busy:
    raise SystemExit(
        f"host is busy: one-minute load / logical CPUs is {load_per_cpu:.3f}, "
        f"above {ARGS.max_load_per_cpu:.3f}; use --allow-busy for diagnostics"
    )
gpu_before = gpu_info() if ARGS.device == "cuda" else None
if ARGS.device == "cuda" and gpu_before is None:
    raise SystemExit("CUDA suite requested but nvidia-smi did not report a GPU")

features = (
    [value for value in ARGS.rust_features.split(",") if value]
    if ARGS.rust_features is not None
    else list(SPEC["suite"]["rust_features"] if ARGS.device == "cpu" else [])
)
if ARGS.device == "cuda":
    if "cuda" not in features:
        features.append("cuda")
if any(cell["operation"] == "encoded_video_init" for cell in CELLS):
    if "video-decoding" not in features:
        features.append("video-decoding")
env = os.environ.copy()
cudnn_link = cuda_build_environment(env)
if not ARGS.no_build:
    build = ["cargo", "build", "--release", "--bin", "bench-cpu-suite"]
    if features:
        build.extend(["--features", ",".join(features)])
    subprocess.run(build, cwd=ROOT, check=True, env=env)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_dir = ARGS.output_dir or ROOT / ".benchmarks" / "runs" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)
outputs_dir = output_dir / "outputs"
outputs_dir.mkdir(exist_ok=True)
diagnostics_dir = output_dir / "diagnostics"
if ARGS.profile_v3_frames:
    if any(cell["id"] != "V3" for cell in CELLS):
        raise SystemExit("--profile-v3-frames requires --cell V3")
    diagnostics_dir.mkdir(exist_ok=True)
if ARGS.profile_video_stages:
    env["SAM2_PROFILE_VIDEO_STAGES"] = "1"
if ARGS.profile_amg_stages:
    env["SAM2_PROFILE_AMG_STAGES"] = "1"
if ARGS.profile_image_stages or ARGS.profile_a2_boundary:
    # Rust's existing synchronized decoder boundaries are shared with AMG.
    # PyTorch uses the image-specific switch added by the suite runner.
    env["SAM2_PROFILE_AMG_STAGES"] = "1"
    env["SAM2_PROFILE_IMAGE_STAGES"] = "1"
if ARGS.profile_a2_boundary:
    if any(cell["id"] != "A2" for cell in CELLS):
        raise SystemExit("--profile-a2-boundary requires --cell A2")
    env["SAM2_PROFILE_A2_BOUNDARY"] = "1"
if ARGS.profile_decoder_parity:
    env["SAM2_PROFILE_DECODER_PARITY"] = "1"
if ARGS.cuda_pool_retain_mib is not None:
    if ARGS.device != "cuda" or ARGS.cuda_pool_retain_mib < 0:
        raise SystemExit("--cuda-pool-retain-mib requires CUDA and a nonnegative value")
    env["SAM2_CUDA_POOL_RETAIN_MIB"] = str(ARGS.cuda_pool_retain_mib)
binary = ROOT / "target/release/bench-cpu-suite"
raw_path = output_dir / "runs.jsonl"
rows = []
with raw_path.open("w") as raw, tempfile.TemporaryDirectory(prefix="sam2-parity-") as temp_name:
    temp = Path(temp_name)
    for expanded_index, cell in enumerate(CELLS):
        cell_threads = cell.get("worker_count", ARGS.threads)
        cell_cpu_list = ARGS.cpu_list or SPEC["suite"].get("cpu_affinity", {}).get(str(cell_threads))
        if ARGS.device == "cpu" and cell_cpu_list is None:
            raise SystemExit(
                f"no CPU affinity is specified for {cell_threads} workers; "
                "pass --cpu-list for a single-worker-count run or add suite.cpu_affinity"
            )
        cell_env = env.copy()
        cell_env.update({
            "RAYON_NUM_THREADS": str(cell_threads),
            "OMP_NUM_THREADS": str(cell_threads),
            "MKL_NUM_THREADS": str(cell_threads),
            "OPENBLAS_NUM_THREADS": str(cell_threads),
        })
        if cell["operation"] == "encoded_video_init":
            encoded_root = ROOT / "checkpoints/cpu-suite/encoded"
            directory = encoded_root / cell["fixture"]
            video = encoded_root / f"{cell['fixture']}.mp4"
            fixture = directory if directory.is_dir() else video
        elif cell["operation"] == "image_batch":
            batch_fixtures = [
                ROOT / "checkpoints/cpu-suite" / f"{name}.safetensors"
                for name in cell["fixture_names"]
            ]
            missing = [path for path in batch_fixtures if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"missing {', '.join(map(str, missing))}; "
                    "run tools/prepare_cpu_suite_fixtures.py"
                )
            fixture = batch_fixtures[0]
        else:
            fixture = ROOT / "checkpoints/cpu-suite" / f"{cell['fixture']}.safetensors"
        if not fixture.exists():
            raise FileNotFoundError(f"missing {fixture}; run tools/prepare_cpu_suite_fixtures.py")
        variant = SPEC["variants"][cell["variant"]]
        device_prefix = "gpu" if ARGS.device == "cuda" else "cpu"
        device_iterations = f"{device_prefix}_iterations"
        device_warmups = f"{device_prefix}_warmups"
        iterations = (
            ARGS.iterations
            if ARGS.iterations is not None
            else cell.get(device_iterations, cell["iterations"])
        )
        warmups = (
            ARGS.warmups
            if ARGS.warmups is not None
            else cell.get(device_warmups, cell["warmups"])
        )
        pairs = []
        for repeat in range(ARGS.repeats):
            rust_output = temp / f"rust-{expanded_index}-{repeat}.f32"
            python_output = temp / f"python-{expanded_index}-{repeat}.f32"
            rust_v3_lowres = temp / f"rust-v3-lowres-{expanded_index}-{repeat}.f32"
            python_v3_lowres = temp / f"python-v3-lowres-{expanded_index}-{repeat}.f32"
            common = [
                "--cell", cell["id"], "--operation", cell["operation"],
                "--variant", variant["rust"], "--fixture", str(fixture),
                "--device", ARGS.device, "--mode", ARGS.mode,
                "--iterations", str(iterations),
                "--warmups", str(warmups),
            ]
            if "sessions" in cell:
                common.extend(["--sessions", str(cell["sessions"])])
            if "point_counts" in cell:
                common.extend(["--point-counts", ",".join(map(str, cell["point_counts"]))])
            if cell["operation"] == "image_batch":
                common.extend(["--batch-fixtures", ",".join(map(str, batch_fixtures))])
            rust_common = common
            if ARGS.profile_video_stages and cell["operation"] == "video_forward_one":
                rust_common = common.copy()
                rust_common[rust_common.index("video_forward_one")] = "video_forward_profile"
            rust_command = [
                str(binary), *rust_common,
                "--checkpoint", str(ROOT / variant["rust_checkpoint"]),
                "--output", str(rust_output),
            ]
            python_command = [
                sys.executable, "tools/bench_parity_python.py", *common,
                "--config", variant["python_config"],
                "--checkpoint", str(ROOT / variant["python_checkpoint"]),
                "--threads", str(cell_threads), "--output", str(python_output),
            ]
            if ARGS.profile_v3_frames:
                rust_command.extend(["--video-lowres-output", str(rust_v3_lowres)])
                python_command.extend(["--video-lowres-output", str(python_v3_lowres)])
            order = ("python", "rust") if repeat % 2 else ("rust", "python")
            results = {}
            diagnostics = {}
            for implementation in order:
                command = rust_command if implementation == "rust" else python_command
                results[implementation], stderr, _ = run_process(
                    command, cell_env, ARGS.device, cell_cpu_list
                )
                validate_result_record(
                    results[implementation], implementation, cell,
                    ARGS.device, iterations, warmups, variant["rust"],
                )
                if (
                    ARGS.profile_video_stages
                    or ARGS.profile_amg_stages
                    or ARGS.profile_image_stages
                    or ARGS.profile_a2_boundary
                    or ARGS.profile_decoder_parity
                ) and stderr:
                    diagnostics[f"{implementation}_stderr"] = stderr.splitlines()
            validate_timing_pair(
                results["rust"],
                results["python"],
                cell,
                ARGS.profile_video_stages
                or ARGS.profile_amg_stages
                or ARGS.profile_image_stages
                or ARGS.profile_a2_boundary
                or ARGS.profile_decoder_parity,
            )
            rust_archive = outputs_dir / f"rust-{expanded_index}-{repeat}.f32"
            python_archive = outputs_dir / f"python-{expanded_index}-{repeat}.f32"
            shutil.copyfile(rust_output, rust_archive)
            shutil.copyfile(python_output, python_archive)
            # Preserve incompatible outputs as diagnostic evidence even when
            # structural parity (for example AMG mask count) prevents a
            # numeric comparison and aborts the run.
            comparison = parity(
                rust_output,
                python_output,
                segments=3 if cell["operation"].startswith("video_") and cell["operation"] != "video_init" else 1,
            )
            pair = {
                "kind": "pair",
                "cell": cell["id"],
                "fixture": cell.get("result_fixture", cell["fixture"]),
                "variant": cell["variant"],
                "source_cell": cell.get("source_cell"),
                "workers": cell_threads,
                "device": ARGS.device,
                "mode": ARGS.mode,
                "repeat": repeat,
                "order": list(order),
                "rust": results["rust"],
                "python": results["python"],
                "parity": comparison,
                "output_artifacts": {
                    "rust": {"path": str(rust_archive.relative_to(output_dir)), "sha256": sha256(rust_archive)},
                    "python": {"path": str(python_archive.relative_to(output_dir)), "sha256": sha256(python_archive)},
                },
            }
            if ARGS.profile_v3_frames:
                rust_lowres_archive = diagnostics_dir / f"rust-v3-lowres-{expanded_index}-{repeat}.f32"
                python_lowres_archive = diagnostics_dir / f"python-v3-lowres-{expanded_index}-{repeat}.f32"
                shutil.copyfile(rust_v3_lowres, rust_lowres_archive)
                shutil.copyfile(python_v3_lowres, python_lowres_archive)
                pair["v3_lowres_artifacts"] = {
                    "layout": "frame,object,1,1,256,256 float32 little-endian",
                    "rust": {
                        "path": str(rust_lowres_archive.relative_to(output_dir)),
                        "sha256": sha256(rust_lowres_archive),
                    },
                    "python": {
                        "path": str(python_lowres_archive.relative_to(output_dir)),
                        "sha256": sha256(python_lowres_archive),
                    },
                }
            if diagnostics:
                pair["diagnostics"] = diagnostics
            is_video = cell["operation"].startswith("video_") and cell["operation"] != "video_init"
            fixture_profile = cell.get("parity_profile_by_fixture", {}).get(cell["fixture"])
            if cell["operation"] == "video_edit_three":
                parity_gates = SPEC["parity"].get(
                    "video_edit", SPEC["parity"].get("video", SPEC["parity"])
                )
            elif fixture_profile:
                parity_gates = SPEC["parity"].get(fixture_profile, SPEC["parity"])
            elif profile := cell.get("parity_profile"):
                parity_gates = SPEC["parity"].get(profile, SPEC["parity"])
            else:
                parity_gates = SPEC["parity"].get("video", SPEC["parity"]) if is_video else SPEC["parity"]
            parity_failed = (
                comparison["max_abs"] > parity_gates["mask_logit_max_abs"]
                or comparison["mean_abs"] > parity_gates["mask_logit_mean_abs"]
                or (parity_gates.get("threshold_masks_exact", False) and comparison["threshold_mismatches"] != 0)
            )
            comparison["passing"] = not parity_failed
            comparison["limits"] = {
                "mask_logit_max_abs": parity_gates["mask_logit_max_abs"],
                "mask_logit_mean_abs": parity_gates["mask_logit_mean_abs"],
                "threshold_masks_exact": parity_gates.get("threshold_masks_exact", False),
            }
            raw.write(json.dumps(pair, separators=(",", ":")) + "\n")
            raw.flush()
            if parity_failed and not ARGS.allow_parity_failure:
                raise RuntimeError(
                    f"{cell['id']}/{cell.get('result_fixture', cell['fixture'])} output parity failed: "
                    f"max {comparison['max_abs']:.8g} > {parity_gates['mask_logit_max_abs']}, "
                    f"mean {comparison['mean_abs']:.8g} > {parity_gates['mask_logit_mean_abs']}, "
                    f"threshold mismatches {comparison['threshold_mismatches']}"
                )
            pairs.append(pair)
        rust_cached_process_medians = [pair["rust"]["median_ms"] for pair in pairs]
        python_cached_process_medians = [pair["python"]["median_ms"] for pair in pairs]
        rust_cached_ms = statistics.fmean(rust_cached_process_medians)
        python_cached_ms = statistics.fmean(python_cached_process_medians)
        def repeat_spread(values: list[float]) -> float:
            minimum = min(values)
            return max(values) / minimum if minimum > 0 else math.inf

        rust_capture_ms = (
            statistics.fmean(pair["rust"]["phases_ms"]["capture"] for pair in pairs)
            if ARGS.mode == "graph"
            else None
        )
        python_capture_ms = (
            statistics.fmean(pair["python"]["phases_ms"]["capture"] for pair in pairs)
            if ARGS.mode == "graph"
            else None
        )
        rust_encode_phase_ms = statistics.fmean(
            pair["rust"]["phases_ms"]["encode"] for pair in pairs
        ) if cell["operation"] in {"first_and_cached", "image_batch"} else None
        python_encode_phase_ms = statistics.fmean(
            pair["python"]["phases_ms"]["encode"] for pair in pairs
        ) if cell["operation"] in {"first_and_cached", "image_batch"} else None
        rust_decode_phase_ms = statistics.fmean(
            pair["rust"]["phases_ms"]["decode"] for pair in pairs
        ) if cell["operation"] == "image_batch" else None
        python_decode_phase_ms = statistics.fmean(
            pair["python"]["phases_ms"]["decode"] for pair in pairs
        ) if cell["operation"] == "image_batch" else None
        # Scaling cells intentionally score the user's first interaction while
        # preserving cached decode as a separately reported component. Noise
        # must be computed over these same per-process totals, not over the
        # much shorter cached subphase alone.
        if cell["operation"] == "first_and_cached":
            rust_process_medians = [
                cached + pair["rust"]["phases_ms"]["encode"]
                for cached, pair in zip(rust_cached_process_medians, pairs)
            ]
            python_process_medians = [
                cached + pair["python"]["phases_ms"]["encode"]
                for cached, pair in zip(python_cached_process_medians, pairs)
            ]
        else:
            rust_process_medians = rust_cached_process_medians
            python_process_medians = python_cached_process_medians
        rust_ms = statistics.fmean(rust_process_medians)
        python_ms = statistics.fmean(python_process_medians)
        rust_repeat_spread = repeat_spread(rust_process_medians)
        python_repeat_spread = repeat_spread(python_process_medians)
        max_repeat_spread = SPEC["suite"]["max_repeat_spread_ratio"]
        rust_rss = statistics.fmean(pair["rust"]["rss_kb"]["hwm"] for pair in pairs)
        python_rss = statistics.fmean(pair["python"]["rss_kb"]["hwm"] for pair in pairs)
        rows.append({
            "cell": cell["id"],
            "fixture": cell.get("result_fixture", cell["fixture"]),
            "variant": cell["variant"],
            "source_cell": cell.get("source_cell"),
            "workers": cell_threads,
            "weight": cell.get("gpu_weight", cell["weight"]) if ARGS.device == "cuda" else cell["weight"],
            "rust_mean_median_ms": rust_ms, "python_mean_median_ms": python_ms,
            "latency_ratio": rust_ms / python_ms,
            "rust_process_medians_ms": rust_process_medians,
            "python_process_medians_ms": python_process_medians,
            "rust_cached_process_medians_ms": (
                rust_cached_process_medians if cell["operation"] == "first_and_cached" else None
            ),
            "python_cached_process_medians_ms": (
                python_cached_process_medians if cell["operation"] == "first_and_cached" else None
            ),
            "rust_repeat_spread_ratio": rust_repeat_spread,
            "python_repeat_spread_ratio": python_repeat_spread,
            "noisy": max(rust_repeat_spread, python_repeat_spread) > max_repeat_spread,
            "rust_mean_encode_ms": rust_encode_phase_ms,
            "python_mean_encode_ms": python_encode_phase_ms,
            "rust_mean_decode_ms": rust_decode_phase_ms,
            "python_mean_decode_ms": python_decode_phase_ms,
            "rust_mean_cached_ms": rust_cached_ms if cell["operation"] == "first_and_cached" else None,
            "python_mean_cached_ms": python_cached_ms if cell["operation"] == "first_and_cached" else None,
            "rust_mean_capture_ms": rust_capture_ms,
            "python_mean_capture_ms": python_capture_ms,
            "rust_mean_hwm_kb": rust_rss, "python_mean_hwm_kb": python_rss,
            "rss_ratio": rust_rss / python_rss,
            "rust_mean_gpu_peak_mib": (
                statistics.fmean(pair["rust"]["external_gpu_peak_mib"] for pair in pairs)
                if ARGS.device == "cuda" and all(pair["rust"]["external_gpu_peak_mib"] is not None for pair in pairs)
                else None
            ),
            "python_mean_gpu_peak_mib": (
                statistics.fmean(pair["python"]["external_gpu_peak_mib"] for pair in pairs)
                if ARGS.device == "cuda" and all(pair["python"]["external_gpu_peak_mib"] is not None for pair in pairs)
                else None
            ),
            "max_abs_error": max(pair["parity"]["max_abs"] for pair in pairs),
            "mean_abs_error": max(pair["parity"]["mean_abs"] for pair in pairs),
            "parity_passing": all(pair["parity"]["passing"] for pair in pairs),
        })
        if cell["operation"] == "repeated_lifetime":
            for implementation in ("rust", "python"):
                plateaus = [
                    pair[implementation]["rss_kb"]["after_each_session"] for pair in pairs
                ]
                rows[-1][f"{implementation}_mean_rss_session_growth_kb"] = statistics.fmean(
                    values[-1] - values[0] for values in plateaus
                )
                rows[-1][f"{implementation}_mean_rss_tail_spread_kb"] = statistics.fmean(
                    max(values[-5:]) - min(values[-5:]) for values in plateaus
                )
        if rows[-1]["rust_mean_gpu_peak_mib"] is not None:
            rows[-1]["gpu_memory_ratio"] = rows[-1]["rust_mean_gpu_peak_mib"] / rows[-1]["python_mean_gpu_peak_mib"]
        if ARGS.device == "cuda":
            for field in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
                rust_values = [pair["rust"]["gpu_memory_bytes"][field] for pair in pairs]
                python_values = [pair["python"]["gpu_memory_bytes"][field] for pair in pairs]
                if all(value is not None for value in rust_values + python_values):
                    rust_mean = statistics.fmean(rust_values)
                    python_mean = statistics.fmean(python_values)
                    rows[-1][f"rust_mean_gpu_{field}_bytes"] = rust_mean
                    rows[-1][f"python_mean_gpu_{field}_bytes"] = python_mean
                    rows[-1][f"gpu_{field}_ratio"] = rust_mean / python_mean
            phase_names = (
                "after_load",
                "after_init",
                "after_interaction",
                "after_propagation",
                "after_reset",
                "after_drop",
            )
            phase_summary = {}
            for phase in phase_names:
                rust_snapshots = [
                    pair["rust"].get("gpu_memory_phases_bytes", {}).get(phase)
                    for pair in pairs
                ]
                python_snapshots = [
                    pair["python"].get("gpu_memory_phases_bytes", {}).get(phase)
                    for pair in pairs
                ]
                if any(snapshot is None for snapshot in rust_snapshots + python_snapshots):
                    continue
                phase_fields = {}
                for field in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
                    rust_values = [snapshot[field] for snapshot in rust_snapshots]
                    python_values = [snapshot[field] for snapshot in python_snapshots]
                    if any(value is None for value in rust_values + python_values):
                        continue
                    rust_mean = statistics.fmean(rust_values)
                    python_mean = statistics.fmean(python_values)
                    phase_fields[field] = {
                        "rust_mean_bytes": rust_mean,
                        "python_mean_bytes": python_mean,
                        "ratio": rust_mean / python_mean if python_mean else None,
                    }
                if phase_fields:
                    phase_summary[phase] = phase_fields
            if phase_summary:
                rows[-1]["gpu_memory_phases"] = phase_summary

metadata = {
    "schema_version": 1,
    "created_utc": timestamp,
    "spec": str(ARGS.spec.relative_to(ROOT)),
    "spec_sha256": hashlib.sha256(SPEC_BYTES).hexdigest(),
    "fixture_manifest_sha256": sha256(ROOT / "benchmarks/fixtures.json"),
    "device": ARGS.device,
    "mode": ARGS.mode,
    "rust_cuda_pool_retain_mib": (
        ARGS.cuda_pool_retain_mib if ARGS.cuda_pool_retain_mib is not None else "library-default"
    ),
    "threads": worker_counts[0] if len(worker_counts) == 1 else worker_counts,
    "cpu_list": (
        ARGS.cpu_list
        if ARGS.cpu_list is not None
        else {str(count): SPEC["suite"].get("cpu_affinity", {}).get(str(count)) for count in worker_counts}
    ),
    "cpu_model": cpu_model(),
    "platform": platform.platform(),
    "load_per_logical_cpu_before": load_per_cpu,
    "load_per_logical_cpu_after": os.getloadavg()[0] / max(1, os.cpu_count() or 1),
    "gpu_before": gpu_before,
    "gpu_after": gpu_info() if ARGS.device == "cuda" else None,
    "rust_features": features,
    "repeats": ARGS.repeats,
}
gate_tier = "p1" if all(cell["tier"] == "p1" for cell in CELLS) else (
    "gpu_p0" if ARGS.device == "cuda" else "p0"
)
gates = SPEC["gates"][gate_tier]
weighted_latency_ratio = weighted_geomean(rows, "latency_ratio")
weighted_rss_ratio = weighted_geomean(rows, "rss_ratio")
noise_passing = not any(row["noisy"] for row in rows)
parity_passing = all(row["parity_passing"] for row in rows)
cell_rss_limit = gates.get("cell_rss_ratio_max", gates.get("cell_host_rss_ratio_max"))
gate_evaluation = {
    "weighted_latency": {
        "value": weighted_latency_ratio,
        "limit": gates["weighted_latency_ratio_max"],
        "passing": weighted_latency_ratio <= gates["weighted_latency_ratio_max"],
    },
    "cell_latency": {
        "limit": gates["cell_latency_ratio_max"],
        "passing": all(row["latency_ratio"] <= gates["cell_latency_ratio_max"] for row in rows),
        "failing_rows": [
            f"{row['cell']}/{row['fixture']}"
            for row in rows
            if row["latency_ratio"] > gates["cell_latency_ratio_max"]
        ],
    },
    "cell_rss": {
        "limit": cell_rss_limit,
        "passing": all(row["rss_ratio"] <= cell_rss_limit for row in rows),
        "failing_rows": [
            f"{row['cell']}/{row['fixture']}"
            for row in rows
            if row["rss_ratio"] > cell_rss_limit
        ],
    },
    "parity": {
        "passing": parity_passing,
        "failing_rows": [
            f"{row['cell']}/{row['fixture']}" for row in rows if not row["parity_passing"]
        ],
    },
    "noise": {
        "passing": noise_passing,
        "failing_rows": [f"{row['cell']}/{row['fixture']}" for row in rows if row["noisy"]],
    },
}
if "weighted_rss_ratio_max" in gates:
    gate_evaluation["weighted_rss"] = {
        "value": weighted_rss_ratio,
        "limit": gates["weighted_rss_ratio_max"],
        "passing": weighted_rss_ratio <= gates["weighted_rss_ratio_max"],
    }
gate_evaluation["passing"] = all(
    check["passing"] for check in gate_evaluation.values() if isinstance(check, dict)
)
summary = {
    "metadata": metadata,
    "rows": rows,
    "weighted_latency_ratio": weighted_latency_ratio,
    "weighted_rss_ratio": weighted_rss_ratio,
    "weighted_gpu_memory_ratio": weighted_geomean(rows, "gpu_memory_ratio"),
    "weighted_gpu_peak_allocated_ratio": weighted_geomean(rows, "gpu_peak_allocated_ratio"),
    "weighted_gpu_peak_reserved_ratio": weighted_geomean(rows, "gpu_peak_reserved_ratio"),
    "noise_gate": {
        "max_repeat_spread_ratio": SPEC["suite"]["max_repeat_spread_ratio"],
        "passing": noise_passing,
        "noisy_rows": [f"{row['cell']}/{row['fixture']}" for row in rows if row["noisy"]],
    },
    "gates": gates,
    "gate_evaluation": gate_evaluation,
    "gate_tier": gate_tier,
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output_dir": str(output_dir), **summary}, separators=(",", ":")))
if cudnn_link is not None:
    cudnn_link.cleanup()
