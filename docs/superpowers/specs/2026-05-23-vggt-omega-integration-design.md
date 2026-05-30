# VGGT-Omega Integration Design

**Date:** 2026-05-23  
**Status:** Draft  
**Branch:** refactor/cu121  

---

## Overview

Add `VGGTOmegaCreator` as a third feedforward backend alongside `VGGTXCreator` and
`MapAnythingCreator`. VGGT-Omega is a newer Meta model with the same depth+pose output
contract but a different API surface, preprocessing strategy, and checkpoint delivery
mechanism.

---

## Architecture

`VGGTOmegaCreator` extends `BaseFeedforwardCreator` directly — no changes to the
template-method base, no shared intermediate class.

```
BaseFeedforwardCreator
├── VGGTXCreator          (existing, unchanged)
├── MapAnythingCreator    (existing, unchanged)
└── VGGTOmegaCreator      (new)   ← feedforward/vggt_omega.py
```

`BundleAdjustment` and `LoopClosure` wrappers work transparently — they
duck-type `BaseFeedforwardCreator`, so no wrapper changes needed.

---

## Installation

### Submodule

```
third_party/vggt-omega   → git@github.com:facebookresearch/vggt-omega.git
```

Added as a git submodule, pinned to a specific commit, consistent with how
`third_party/VGGT-X` is managed.

### `setup/feedforward.sh` block

```bash
# vggt-omega
# --no-deps: skips numpy<2 metadata constraint (env has numpy 2.4.6; same bypass as VGGT-X)
pip install --no-deps -e third_party/vggt-omega
```

Placed after the existing vggt-x and mapanything blocks. The `numpy<2` constraint in
vggt-omega's `pyproject.toml` is metadata-only — VGGT-X declares the same constraint and
runs fine with numpy 2.4.6. Installing with `--no-deps` bypasses the metadata check
without downgrading numpy.

### Dependencies brought in

| Package | Already in env? |
|---|---|
| numpy<2 | yes (2.4.6, bypass via --no-deps) |
| Pillow | yes |
| einops | yes |
| safetensors | yes |
| opencv-python | yes |

No torch/torchvision pins in vggt-omega's deps — no version conflict risk.

---

## Checkpoint Loading

VGGT-Omega does not support `from_pretrained`. Checkpoints are downloaded from
the gated HuggingFace repo `facebook/VGGT-Omega`.

**Two available checkpoints:**

| Name | Resolution | Text alignment |
|---|---|---|
| `vggt_omega_1b_512.pt` | 512 | No |
| `vggt_omega_1b_256_text.pt` | 256 | Yes |

**Scope of this integration:** standard 512-res checkpoint only.

### `VGGTOmegaCreator` params

```python
model_path: str | None = None       # local .pt path; None → auto-download
model_repo: str = "facebook/VGGT-Omega"
model_filename: str = "vggt_omega_1b_512.pt"
image_resolution: int = 512
```

`_load_model` logic:
1. If `model_path` is given, load from disk.
2. Otherwise, call `hf_hub_download(repo_id=self.model_repo, filename=self.model_filename)`.
   User must be logged in with `huggingface-cli login`. Checkpoint cached to HF cache dir.
3. `VGGTOmega().to(device).eval()` then `load_state_dict(torch.load(path, map_location="cpu"))`.

---

## Image Preprocessing (`_preprocess`)

### Difference from VGGT-X

VGGT-X pads images to square; VGGT-Omega center-crops extreme aspect ratios (AR outside
`[0.5, 2.0]`) then resizes. The two transforms produce different `original_coords` values.

### Strategy

Call `load_and_preprocess_images(paths, image_resolution=self.image_resolution)` for the
actual tensor. Before calling it, compute the crop info ourselves (replicating Omega's crop
rule) so we can populate `original_coords` in the base-class format:

```
original_coords[i] = [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
```

where `cr_x, cr_y` is the bottom-right corner of the crop in original image pixels
(matching the `FeedforwardResult` field layout used by `_rescale_reconstruction_to_original_dimensions`).

**Crop rule** (matches `load_fn._crop_to_supported_aspect_ratio`):
- AR = height / width
- AR > 2.0 → center-crop height: `crop_h = width * 2.0`, `tl_y = (height - crop_h) // 2`
  → `cr_x = orig_w`, `cr_y = tl_y + crop_h`
- AR < 0.5 → center-crop width: `crop_w = height / 0.5`, `tl_x = (width - crop_w) // 2`
  → `cr_x = tl_x + crop_w`, `cr_y = orig_h`
- Otherwise → no crop: `tl_x=0, tl_y=0, cr_x=orig_w, cr_y=orig_h`

This `original_coords` is compatible with `_rescale_reconstruction_to_original_dimensions`
in `base.py` (same field layout as VGGTXCreator).

The `image_preproc` param from `VGGTXCreator` is **not** present on `VGGTOmegaCreator`;
Omega always uses its balanced-resize mode.

---

## Forward Pass (`_forward`)

```python
def _forward(self, model, views, **kwargs):
    with torch.inference_mode():
        return model(views)
```

`views` is the stacked tensor `[N, 3, H, W]` returned by `_preprocess`.
`raw_outputs` is the predictions dict with keys:
`pose_enc`, `depth`, `depth_conf`, `images`.

---

## Postprocessing (`_postprocess`)

```python
from vggt_omega.utils.pose_enc import encoding_to_camera

extrinsics, intrinsics = encoding_to_camera(
    raw["pose_enc"], raw["images"].shape[-2:]
)
```

`encoding_to_camera` returns:
- `extrinsics`: `[N, 3, 4]` or `[1, N, 3, 4]` — camera-from-world, OpenCV convention
- `intrinsics`: `[N, 3, 3]` — 3×3 camera matrix with separate `fx, fy`

Shape handling: squeeze leading batch dim if present (model may output `[1, N, ...]`).

**Camera model:** `PINHOLE` (not `SIMPLE_PINHOLE`) because Omega predicts separate `fx, fy`.

**Depth unprojection:** same pattern as VGGTXCreator — confidence-threshold filtering
then world-point unprojection using `_raw_to_world_points` from `base.py`.

`FeedforwardResult` fields populated: `pts3d`, `colors`, `extrinsics`, `intrinsics`,
`depth`, `conf`, `images`, `world_points`, `pixel_indices`,
`model_width`, `model_height`, `image_paths`.

---

## Loop Closure Support (`_verify_loop_candidate`)

Re-run model on a 2-frame pair and return `(accepted: bool, poses: np.ndarray | None)`.

```python
def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio):
    # stack 2-frame tensor, run forward, decode poses, compute match ratio
    # return (True, poses_2x4x4) or (False, None)
```

Contract identical to `VGGTXCreator._verify_loop_candidate`.

---

## Files Changed

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | New — `VGGTOmegaCreator` |
| `collab_splats/pointcloud/feedforward/__init__.py` | Export `VGGTOmegaCreator` |
| `setup/feedforward.sh` | Add vggt-omega install block |
| `.gitmodules` | Add `third_party/vggt-omega` |
| `tests/pointcloud/test_vggt_omega.py` | New — unit tests |

---

## Testing

**Unit tests** (no GPU required — mock model and checkpoint):
- `_preprocess` produces correct `original_coords` for images with normal, wide, and tall
  aspect ratios
- `_postprocess` converts model output dict to `FeedforwardResult` with correct shapes
- `_load_model` with explicit `model_path` loads without HF download

**Smoke test** (GPU, gated checkpoint required):
- Run `VGGTOmegaCreator` on a small image dir
- Assert `PointcloudResult` has non-empty `points` and valid `cameras`

---

## Open Questions / Non-Scope

- Text-aligned checkpoint (`enable_alignment=True`, 256-res): out of scope for now.
- `use_global_alignment`: VGGT-Omega does not expose this; not implemented.
- World-point output: Omega does not output an explicit point map (`world_points` in VGGT-X
  sense) — we derive it from depth unprojection, same as VGGT-X's `_postprocess`.
- Numpy 2.x runtime compatibility: assumed OK based on VGGT-X precedent; flag if runtime
  errors appear on numpy 2.x-specific APIs.
