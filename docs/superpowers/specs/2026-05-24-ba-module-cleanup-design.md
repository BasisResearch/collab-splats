# Bundle Adjustment Module Cleanup — Design Spec

**Date:** 2026-05-24
**Branch:** refactor/cu121
**File scope:** `collab_splats/pointcloud/bundle_adjustment.py`, `tests/pointcloud/test_bundle_adjustment.py`

---

## Problem

`bundle_adjustment.py` violates multiple coding principles accumulated over iterative development:

1. **Lazy imports scattered through functions** — contradicts "hard imports at top" style
2. **`_run_bundle_adjustment` is 220 lines** with 7 labeled steps using `########` dividers inside a function body — wrong scope for dividers, signals decomposition needed
3. **`ReprojNonBatched` inner class** — `nn.Module` subclass redefined on every call; not testable in isolation
4. **`rotate_quat` inner function** — 3-line wrapper for `pp.SE3(...).Act(pts)`, redefined on every call
5. **Two same-name `reproject_simple_pinhole` closures** under `if/else` inside a function — same name, different signatures, result of lazy-import workaround
6. **`import torch.nn.functional as F` inside a conditional block** — import buried in `if conf_tensor is not None ...`
7. **`image_size` dead parameter** — declared in `_run_bundle_adjustment`, never read; 3 test call sites pass it
8. **`_run_bundle_adjustment` called exactly once** from `refine()`, with params that mirror `self.config` — redundant standalone function
9. **Stale test mock** — `_make_bae_mock()` wires `bae.utils.ba.rotate_quat` but production defines it locally; never reached

---

## Decision: Drop Lazy Imports

CLAUDE.md primary rule: "Hard imports — no stub backends; let missing deps raise `ImportError` at import time."

`bae`, `pypose`, and `vggt` are required for any BA operation. If absent, the module correctly raises `ImportError` at import time. Tests that run without `bae` already use `patch.dict(sys.modules, ...)` before exec'ing the module — that approach works with top-level imports.

---

## Target Module Structure

Public API first, helpers below. Python resolves function-body names at call time (not class-definition time), so helpers defined after the class are fine.

```
collab_splats/pointcloud/bundle_adjustment.py

  [imports — all hard, top of file]

  #### Public API ####################################

  BundleAdjustmentConfig            dataclass — configuration
  BundleAdjustment                  class — public entry point
    refine(result)                  public method
    _optimize(...)                  private method; absorbs _run_bundle_adjustment

  #### Helpers #######################################

  _get_default_solver()             solver auto-selection (unchanged)
  _extract_tracks_vggsfm()          VGGSfM track prediction
  _reproject_per_camera()           @map_transform reprojection, per-camera focal
  _reproject_shared()               @map_transform reprojection, shared focal
  _BAModel(nn.Module)               reprojection residual module for LM
```

---

## Hard Imports (top of file)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp
import numpy as np
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.projection import project_3D_points_np

if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult
```

---

## Component Designs

### `BundleAdjustmentConfig` — unchanged

No changes. Keeps all existing fields.

### `BundleAdjustment.refine(result)` — minor simplification

Removes `image_size` from `_optimize` call (dead param eliminated). Otherwise unchanged logic.

```python
def refine(self, result: "FeedforwardResult") -> "FeedforwardResult":
    """Refine camera poses; return updated FeedforwardResult with new extrinsics/intrinsics.

    pts3d, colors, and pixel_indices are unchanged — call creator.reproject(result)
    after to re-extract pts3d from refined poses.
    """
    cfg = self.config
    extrinsics_3x4 = result.extrinsics[:, :3, :]

    # Extract 2D tracks across all frames via VGGSfM
    tracks, vis_scores, pts3d_tracks = _extract_tracks_vggsfm(
        result.images, result.conf, result.world_points,
        max_query_pts=cfg.max_query_pts,
        query_frame_num=cfg.query_frame_num,
        device=cfg.device,
    )

    # Refine poses and intrinsics via LM bundle adjustment
    _, refined_extrinsics, refined_intrinsics = self._optimize(
        pts3d_tracks, extrinsics_3x4, result.intrinsics, tracks, vis_scores,
    )

    # Pad (N, 3, 4) extrinsics back to (N, 4, 4) for FeedforwardResult convention
    n = refined_extrinsics.shape[0]
    bottom_row = np.tile([[0, 0, 0, 1]], (n, 1, 1)).astype(np.float32)
    refined_extrinsics_4x4 = np.concatenate([refined_extrinsics, bottom_row], axis=1)
    return replace(result, extrinsics=refined_extrinsics_4x4, intrinsics=refined_intrinsics)
```

### `BundleAdjustment._optimize(pts3d, extrinsics, intrinsics, tracks, vis_scores)`

Absorbs all logic from `_run_bundle_adjustment`. Reads `self.config` directly — no redundant param list. Named sections with inline block comments (no `########` dividers). Readable variable names throughout.

Key sections:
1. **Filter observations** — remove high-reproj-error, degenerate, and under-constrained entries
2. **Build index arrays** — unique active keyframes and landmarks, flat observation list
3. **Build camera and point tensors** — SE3 poses, per-camera or shared focal, 3D point tensor
4. **Instantiate `_BAModel`** — pass `shared_camera` flag; model selects reproject function internally
5. **Run LM optimiser** — `TrustRegion` strategy, `StopOnPlateau` scheduler
6. **Extract results** — SE3 → (3,4) extrinsics, focal lengths back to intrinsics matrix

```python
def _optimize(
    self,
    pts3d: np.ndarray,           # (P, 3)    initial 3D keypoint positions
    extrinsics: np.ndarray,      # (N, 3, 4) world-to-camera poses
    intrinsics: np.ndarray,      # (N, 3, 3) camera intrinsics
    tracks: np.ndarray,          # (N, P, 2) 2D pixel observations from VGGSfM
    vis_scores: np.ndarray,      # (N, P)    visibility scores in [0, 1]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine 3D points and camera poses with Levenberg-Marquardt BA.

    Returns:
        refined_pts3d:        (P, 3)    float64
        refined_extrinsics:   (N, 3, 4) float32
        refined_intrinsics:   (N, 3, 3) float32
    """
    cfg = self.config
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ...
```

### `_reproject_per_camera` and `_reproject_shared` — module-level

Replace the two same-name conditional closures. With hard imports, `@map_transform` is available at module load time — no factory, no injection, no closures.

> **TODO (verify during implementation):** `@map_transform` is bae's vmap-based decorator for efficient sparse Jacobian computation in the LM solver. If bae's LM can compute Jacobians via standard autograd on a regular `nn.Module.forward()` without this decorator, the two functions can be eliminated entirely and the projection math (3 lines) can be inlined directly in `_BAModel.forward()`. Two separate function signatures exist because bae's vmap cannot handle `None` as a batched argument — verify this constraint is still present in the installed bae version before deciding.

```python
@map_transform
def _reproject_per_camera(pts, cam_params, principal_point):
    """Project pts using per-camera focal length stored in cam_params[..., 7:]."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * cam_params[..., 7:] + principal_point


@map_transform
def _reproject_shared(pts, cam_params, principal_point, focal):
    """Project pts using a single shared focal length focal."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * focal + principal_point
```

`rotate_quat` is gone — `pp.SE3(...).Act(pts)` is used directly. No wrapper needed.

### `_BAModel(nn.Module)` — proper module-level class

Replaces `ReprojNonBatched`. Stores `shared_camera` flag; calls the appropriate module-level reproject function in `forward()`. No injection, no closures.

```python
class _BAModel(nn.Module):
    """Reprojection residual module for pypose Levenberg-Marquardt optimisation."""

    def __init__(
        self,
        cam_params: torch.Tensor,          # (K, 7) SE3 or (K, 8) SE3+focal
        pts_3d: torch.Tensor,              # (L, 3) 3D landmark positions
        shared_focal: "torch.Tensor | None",  # (1, 1) if shared_camera, else None
        shared_camera: bool,
    ):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(cam_params))
        self.pts = nn.Parameter(TrackingTensor(pts_3d))
        self.pose.trim_SE3_grad = True
        self.shared_intr = (
            nn.Parameter(TrackingTensor(shared_focal)) if shared_focal is not None else None
        )
        self.shared_camera = shared_camera

    def forward(self, points_2d, camera_indices, point_indices, principal_points):
        """Return reprojection residuals (predicted_2d - observed_2d) for all observations."""
        pts = self.pts[point_indices]
        cam = self.pose[camera_indices]
        ctr = principal_points[camera_indices]

        if self.shared_camera:
            # Broadcast shared focal to match (M,) observation count
            focal = self.shared_intr[torch.zeros_like(camera_indices)]
            pts_proj = _reproject_shared(pts, cam, ctr, focal)
        else:
            pts_proj = _reproject_per_camera(pts, cam, ctr)

        return pts_proj - points_2d
```

### `_extract_tracks_vggsfm` — import fix only

Move `import torch.nn.functional as F` from inside the conditional block to the hard import block at the top of the file. No logic changes.

---

## Variable Naming Conventions (throughout)

| Old name | New name | Reason |
|---|---|---|
| `vis_mask` | `vis` (within filter block) | shorter, same meaning |
| `pts3d_kp` | `pts3d_tracks` | VGGSfM keypoints, not feedforward pts3d |
| `observations_t` | `obs_2d` | clearer tensor meaning |
| `camera_indices_t` | `cam_idx` | standard BA shorthand |
| `point_indices_t` | `pt_idx` | standard BA shorthand |
| `shared_intr` | `shared_focal` | more descriptive |
| `points_3d_tensor` | `pts3d_tensor` | consistent with codebase pts3d usage |
| `pred_tracks` | `tracks` | already converted before use |
| `ReprojNonBatched` | `_BAModel` | descriptive, private |

---

## Coding Principles Checklist

- [ ] One-line summary docstring on every public function and class
- [ ] Inline block comments on each logical section (not line-by-line)
- [ ] Blank lines between logical blocks within functions
- [ ] `########` dividers at file level only (between Config / Public API / Helpers sections)
- [ ] No function longer than ~80 lines
- [ ] Typed params with shape comments on `_optimize` and `_BAModel.__init__`

---

## Test Changes

1. Remove `image_size=(H, W)` from 3 call sites in test file
2. Rename direct `_run_bundle_adjustment(...)` calls → `BundleAdjustment(config=...)._optimize(...)`
3. Remove `bae.utils.ba.rotate_quat` from `_make_bae_mock()` — stale, never reached
4. Tests that directly `import BundleAdjustment` without mocking now require `bae`/`pypose`/`vggt` — apply `@pytest.mark.skipif(not _pypose_available(), ...)` pattern already in use

---

## Line Count Estimate

| Section | Before | After |
|---|---|---|
| `_extract_tracks_vggsfm` | ~75 lines | ~65 lines |
| `_run_bundle_adjustment` (standalone) | ~220 lines | removed |
| `_reproject_*` functions (module-level) | 0 | ~20 lines |
| `_BAModel` | ~25 lines (inner) | ~30 lines (module-level) |
| `BundleAdjustment` class | ~45 lines | ~110 lines (absorbs `_optimize`) |
| **Total** | **~438 lines** | **~290 lines (~34% reduction)** |

---

## Out of Scope

- `BundleAdjustmentConfig` field changes
- `_extract_tracks_vggsfm` logic changes (import fix only)
- `wrappers.py` or any caller changes
- Batched reprojection (current indexed-select over M observations is already GPU-vectorized)
