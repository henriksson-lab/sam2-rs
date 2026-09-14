#!/usr/bin/env python3
"""Run fair Rust/Python static SAM2 benchmarks and update README ratios."""
from pathlib import Path
import argparse, hashlib, json, platform, struct, subprocess, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=3)
parser.add_argument("--rust-checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_tiny.safetensors")
parser.add_argument("--python-checkpoint", type=Path, default=ROOT / "checkpoints/sam2.1_hiera_tiny.pt")
parser.add_argument("--fixture", type=Path, default=ROOT / "checkpoints/bench_bedroom.safetensors")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()

binary = ROOT / "target/release/bench-static"
# Let Cargo perform its inexpensive freshness check so runner edits never use
# a stale executable; build time is outside measured inference processes.
build = ["cargo", "build", "--release", "--bin", "bench-static"]
if args.device == "cuda":
    build.extend(["--features", "cuda"])
subprocess.run(build, cwd=ROOT, check=True)

def run(command):
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    return json.loads(result.stdout.strip().splitlines()[-1])

with tempfile.TemporaryDirectory(prefix="sam2-rs-bench-") as temp:
    rust_out = Path(temp) / "rust.f32"
    python_out = Path(temp) / "python.f32"
    rust = run([str(binary), str(args.rust_checkpoint), str(args.fixture), str(args.iterations), str(rust_out), "--device", args.device])
    python = run([sys.executable, "tools/bench_static_python.py", str(args.python_checkpoint), str(args.fixture), "--iterations", str(args.iterations), "--output", str(python_out), "--device", args.device])
    left, right = rust_out.read_bytes(), python_out.read_bytes()
    if len(left) != len(right) or len(left) % 4:
        raise RuntimeError(f"incompatible output sizes: {len(left)} and {len(right)}")
    parity = max(abs(a[0] - b[0]) for a, b in zip(struct.iter_unpack("<f", left), struct.iter_unpack("<f", right)))

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

marker = "BENCHMARK_RESULTS" if args.device == "cpu" else "BENCHMARK_GPU_RESULTS"
heading = "## Reproducible GPU performance baseline" if args.device == "cuda" else "## Reproducible performance baseline"
table = f"""<!-- {marker}_START -->
_Generated on {platform.system()} {platform.release()}, CPU: {cpu_model}; fixture SHA-256: `{fixture_digest}`. Peak RSS is Linux `VmHWM`._

| Fixture / operation | Rust median ms | Python median ms | Rust/Python latency | Rust peak RSS MiB | Python peak RSS MiB | Rust/Python RSS | Max abs mask-logit error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bedroom/00089.jpg, Tiny static point prompt on {args.device} ({args.iterations} measured iterations) | {rust['median_ms']:.2f} | {python['median_ms']:.2f} | {rust['median_ms']/python['median_ms']:.3f}× | {rust['rss_hwm_kb']/1024:.1f} | {python['rss_hwm_kb']/1024:.1f} | {rust['rss_hwm_kb']/python['rss_hwm_kb']:.3f}× | {parity:.6g} |
<!-- {marker}_END -->"""

readme = ROOT / "README.md"
text = readme.read_text() if readme.exists() else "# sam2-rs\n\n"
start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
if start in text and end in text:
    text = text[:text.index(start)] + table + text[text.index(end) + len(end):]
else:
    text += f"\n{heading}\n\n" + table + "\n"
readme.write_text(text)
print(json.dumps({"rust": rust, "python": python, "latency_ratio": rust['median_ms']/python['median_ms'], "rss_ratio": rust['rss_hwm_kb']/python['rss_hwm_kb'], "max_abs_mask_error": parity}))
