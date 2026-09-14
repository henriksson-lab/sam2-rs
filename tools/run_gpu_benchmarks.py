#!/usr/bin/env python3
"""Run the separate CUDA SAM2 benchmark suite over the frozen real fixture."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(
    [sys.executable, "tools/run_benchmarks.py", "--device", "cuda", *sys.argv[1:]],
    cwd=ROOT,
    check=True,
)
