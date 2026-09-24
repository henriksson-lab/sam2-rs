#!/usr/bin/env python3
"""Run the separate CUDA SAM2 benchmark suite over the frozen real fixture."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def prepend(env, name, path):
    current = env.get(name)
    env[name] = f"{path}{os.pathsep}{current}" if current else str(path)


env = os.environ.copy()
temporary_link = None
try:
    import torch

    cudnn_lib = Path(torch.__file__).resolve().parent.parent / "nvidia/cudnn/lib"
    if cudnn_lib.is_dir():
        prepend(env, "LD_LIBRARY_PATH", cudnn_lib)
        unversioned = cudnn_lib / "libcudnn.so"
        if unversioned.exists():
            prepend(env, "LIBRARY_PATH", cudnn_lib)
        else:
            versioned = sorted(cudnn_lib.glob("libcudnn.so.*"))
            if versioned:
                temporary_link = tempfile.TemporaryDirectory(prefix="sam2-cudnn-link-")
                link_dir = Path(temporary_link.name)
                (link_dir / "libcudnn.so").symlink_to(versioned[-1])
                prepend(env, "LIBRARY_PATH", link_dir)
except ImportError:
    # The underlying benchmark emits the actionable error if PyTorch is absent.
    pass

try:
    subprocess.run(
        [sys.executable, "tools/run_benchmarks.py", "--device", "cuda", *sys.argv[1:]],
        cwd=ROOT,
        check=True,
        env=env,
    )
finally:
    if temporary_link is not None:
        temporary_link.cleanup()
