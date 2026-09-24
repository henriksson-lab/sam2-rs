# Memory-efficient attention sources

The headers in this directory are the inference-only forward-kernel subset of
PyTorch's CUDA memory-efficient attention implementation, taken from PyTorch
commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`. Local changes replace ATen and
c10 dependencies with `compat.h`; SAM2-specific f32/SM75 instantiation and the
C ABI launcher live in `../mem_eff_attention.cu`.

The sources retain their copyright headers and are distributed under the
PyTorch BSD-style terms in `LICENSE.pytorch`. CUTLASS is fetched at the same
commit used by that PyTorch revision and remains under NVIDIA's own license.
