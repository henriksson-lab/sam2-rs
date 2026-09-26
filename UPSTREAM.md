# Pinned upstream references

The SAM2 translation target is a checkout of Meta's
[`segment-anything-2`](https://github.com/facebookresearch/segment-anything-2)
repository under `vendor/segment-anything-2`.

| Item | Value |
| --- | --- |
| Commit | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Date | 2024-12-15 |
| Subject | `remove .pin_memory() in obj_pos of SAM2Base to resolve an error in MPS (#495)` |
| Initial target config | `sam2/configs/sam2.1/sam2.1_hiera_t.yaml` |
| Initial target checkpoint | `sam2.1_hiera_tiny.pt` (`facebook/sam2.1-hiera-tiny`) |

`segment-anything/` is the pre-existing SAM v1 checkout. It is intentionally
kept separate and must not be used as the SAM2 implementation source.
