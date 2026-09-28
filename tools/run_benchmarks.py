#!/usr/bin/env python3
"""Run fair Rust/Python static SAM2 benchmarks and update README ratios."""
from pathlib import Path
import argparse, hashlib, json, os, platform, statistics, struct, subprocess, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument(
    "--repeats",
    type=int,
    default=1,
    help="independent Rust/Python benchmark pairs; reported values are their arithmetic means",
)
parser.add_argument("--rust-checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_tiny.safetensors")
parser.add_argument("--python-checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_tiny.pt")
parser.add_argument("--fixture", type=Path, default=ROOT / "checkpoints/bench_bedroom.safetensors")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument(
    "--variant",
    choices=("tiny", "small", "base-plus", "large"),
    default="tiny",
)
parser.add_argument(
    "--profile-encoder",
    action="store_true",
    help="diagnostic: emit synchronized matched Hiera/FPN boundaries",
)
parser.add_argument(
    "--cuda-graph",
    action="store_true",
    help="capture and replay the identical static CUDA inference in both implementations",
)
parser.add_argument("--rust-features", default="", help="comma-separated Cargo features for the Rust benchmark")
parser.add_argument(
    "--threads",
    type=int,
    help="cap both implementations to this many CPU workers (Rayon and PyTorch)",
)
parser.add_argument(
    "--cpu-list",
    help="optional taskset CPU list applied identically to Rust and Python (for example 0-19)",
)
parser.add_argument(
    "--alternate-order",
    action="store_true",
    help="run Python first in odd pairs to expose process-order/host-drift bias",
)
parser.add_argument(
    "--native",
    action="store_true",
    help="compile Rust with -C target-cpu=native for a host-specific CPU comparison",
)
parser.add_argument(
    "--no-write",
    action="store_true",
    help="print the fair comparison without replacing a README benchmark table",
)
args = parser.parse_args()
if args.cuda_graph and args.device != "cuda":
    parser.error("--cuda-graph requires --device cuda")

binary = ROOT / "target/release/bench-static"
# Let Cargo perform its inexpensive freshness check so runner edits never use
# a stale executable; build time is outside measured inference processes.
build = ["cargo", "build", "--release", "--bin", "bench-static"]
features = [feature for feature in args.rust_features.split(",") if feature]
if args.device == "cuda":
    features.append("cuda")
if features:
    build.extend(["--features", ",".join(features)])
build_env = os.environ.copy()
if args.native:
    build_env["RUSTFLAGS"] = "-C target-cpu=native"
subprocess.run(build, cwd=ROOT, check=True, env=build_env)

def run(command, env=None):
    if args.cpu_list:
        command = ["taskset", "--cpu-list", args.cpu_list, *command]
    result = subprocess.run(command, cwd=ROOT, env=env, check=True, text=True, capture_output=True)
    return json.loads(result.stdout.strip().splitlines()[-1])

if args.repeats < 1:
    parser.error("--repeats must be at least 1")

runs = []
with tempfile.TemporaryDirectory(prefix="sam2-rs-bench-") as temp:
    benchmark_env = build_env.copy()
    if args.cuda_graph:
        benchmark_env["SAM2_CUDA_GRAPH"] = "1"
    if args.threads:
        benchmark_env.update({
            "RAYON_NUM_THREADS": str(args.threads),
            "OMP_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
        })
    for repeat in range(args.repeats):
        rust_out = Path(temp) / f"rust-{repeat}.f32"
        python_out = Path(temp) / f"python-{repeat}.f32"
        rust_command = [str(binary), str(args.rust_checkpoint), str(args.fixture), str(args.iterations), str(rust_out), "--device", args.device, "--variant", args.variant]
        python_command = [sys.executable, "tools/bench_static_python.py", str(args.python_checkpoint), str(args.fixture), "--iterations", str(args.iterations), "--output", str(python_out), "--device", args.device, "--variant", args.variant]
        if args.profile_encoder:
            rust_command.append("--profile-encoder")
            python_command.append("--profile-encoder")
        if args.threads:
            python_command.extend(["--threads", str(args.threads)])
        order = ("python", "rust") if args.alternate_order and repeat % 2 else ("rust", "python")
        results = {}
        for implementation in order:
            command = rust_command if implementation == "rust" else python_command
            results[implementation] = run(command, benchmark_env)
        rust, python = results["rust"], results["python"]
        left, right = rust_out.read_bytes(), python_out.read_bytes()
        if len(left) != len(right) or len(left) % 4:
            raise RuntimeError(f"incompatible output sizes: {len(left)} and {len(right)}")
        parity = max(abs(a[0] - b[0]) for a, b in zip(struct.iter_unpack("<f", left), struct.iter_unpack("<f", right)))
        runs.append({"order": "/".join(order), "rust": rust, "python": python, "max_abs_mask_error": parity})

rust = {
    "median_ms": statistics.fmean(run["rust"]["median_ms"] for run in runs),
    "rss_hwm_kb": statistics.fmean(run["rust"]["rss_hwm_kb"] for run in runs),
}
python = {
    "median_ms": statistics.fmean(run["python"]["median_ms"] for run in runs),
    "rss_hwm_kb": statistics.fmean(run["python"]["rss_hwm_kb"] for run in runs),
}
parity = max(run["max_abs_mask_error"] for run in runs)

fixture_digest = hashlib.sha256(args.fixture.read_bytes()).hexdigest()[:12]
cpu_model = "unknown"
try:
    cpu_model = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
except (FileNotFoundError, StopIteration):
    pass

if args.device == "cuda":
    marker = "BENCHMARK_GPU_RESULTS"
    heading = "## Reproducible GPU performance baseline"
elif args.rust_features:
    marker = "BENCHMARK_EXPERIMENTAL_RESULTS"
    heading = "## Reproducible experimental performance baseline"
else:
    marker = "BENCHMARK_RESULTS"
    heading = "## Reproducible performance baseline"
rust_config = args.rust_features or ("cuda" if args.device == "cuda" else "default")
if args.cuda_graph:
    rust_config += ",cuda-graph-both"
if args.native:
    rust_config += ",target-cpu=native"
cpu_pin_note = f", pinned to CPUs `{args.cpu_list}`" if args.cpu_list else ""
order_note = ", alternating process order" if args.alternate_order else ""
gpu_note = ""
if args.device == "cuda":
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()[0].strip()
        gpu_note = f" GPU: {gpu}."
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        gpu_note = " GPU identity unavailable."
table = f"""<!-- {marker}_START -->
_Generated on {platform.system()} {platform.release()}, CPU: {cpu_model}.{gpu_note} Fixture SHA-256: `{fixture_digest}`. CPU workers: `{args.threads or 'implementation default'}`{cpu_pin_note}{order_note}. Peak RSS is Linux `VmHWM`._

| Fixture / operation | Rust mean ms | Python mean ms | Rust/Python latency | Rust mean peak RSS MiB | Python mean peak RSS MiB | Rust/Python RSS | Max abs mask-logit error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bedroom/00089.jpg, Tiny static point prompt on {args.device}; Rust `{rust_config}` ({args.repeats} independent runs × {args.iterations} measured iterations) | {rust['median_ms']:.2f} | {python['median_ms']:.2f} | {rust['median_ms']/python['median_ms']:.3f}× | {rust['rss_hwm_kb']/1024:.1f} | {python['rss_hwm_kb']/1024:.1f} | {rust['rss_hwm_kb']/python['rss_hwm_kb']:.3f}× | {parity:.6g} |
<!-- {marker}_END -->"""

if not args.no_write:
    readme = ROOT / "README.md"
    text = readme.read_text() if readme.exists() else "# sam2-rs\n\n"
    start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
    if start in text and end in text:
        text = text[:text.index(start)] + table + text[text.index(end) + len(end):]
    else:
        text += f"\n{heading}\n\n" + table + "\n"
    readme.write_text(text)
print(json.dumps({"runs": runs, "rust": rust, "python": python, "latency_ratio": rust['median_ms']/python['median_ms'], "rss_ratio": rust['rss_hwm_kb']/python['rss_hwm_kb'], "max_abs_mask_error": parity}))
