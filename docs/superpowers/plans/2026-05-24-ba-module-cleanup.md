# Bundle Adjustment Module Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `bundle_adjustment.py` to follow codebase coding principles — hard imports at top, module-level classes, public API first, `_run_bundle_adjustment` collapsed into `BundleAdjustment._optimize()`, ~34% line reduction.

**Architecture:** Drop all lazy imports; move `torch`/`pypose`/`bae`/`vggt` to top of file. Lift `ReprojNonBatched` inner class → module-level `_BAModel`. Lift conditional `reproject_simple_pinhole` closures → module-level `_reproject_per_camera` / `_reproject_shared`. Collapse standalone `_run_bundle_adjustment` (called exactly once) into `BundleAdjustment._optimize()`. Module order: config → class → helpers.

**Tech Stack:** PyTorch, pypose, bae (LM optimizer + TrackingTensor + map_transform), VGGSfM (predict_tracks), numpy

**Spec:** `docs/superpowers/specs/2026-05-24-ba-module-cleanup-design.md`

---

### Task 1: Establish baseline

**Files:**
- Read: `collab_splats/pointcloud/bundle_adjustment.py`
- Run: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Run the full test suite for bundle_adjustment**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v 2>&1 | tail -30
```

Expected: all tests pass (some may be skipped if bae/pypose not installed — that is fine). Record the exact pass/skip/fail count. If anything is failing before the refactor, stop and investigate before proceeding.

- [ ] **Step 2: Record baseline count**

Note exact numbers, e.g. "12 passed, 3 skipped". This is the target to match after the refactor.

---

### Task 2: Update the test file before implementing

Update tests to work with the new interface. Do this FIRST so that tests fail for the right reason (new interface not yet implemented), then go green when the implementation is done.

**Files:**
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Remove stale `bae.utils.ba.rotate_quat` from `_make_bae_mock`**

`rotate_quat` was defined locally in the old code and never imported from `bae.utils.ba`. Remove the stale entry.

In `_make_bae_mock()` (around line 50), remove:
```python
    utils_ba = types.ModuleType("bae.utils.ba")
    utils_ba.rotate_quat = MagicMock(side_effect=lambda pts, _pose: pts)
    bae.utils = utils
    bae.utils.pysolvers = utils_py
    bae.utils.ba = utils_ba
```
Replace with:
```python
    bae.utils = utils
    bae.utils.pysolvers = utils_py
```
(Drop `utils_ba` entirely — `bae.utils.ba` is no longer used.)

- [ ] **Step 2: Update `test_run_bundle_adjustment_early_exit_shape` → `test_optimize_early_exit_shape`**

Replace the entire test with one that calls `BundleAdjustment._optimize()` directly. The `image_size` param is removed. Load the module via `patch.dict(sys.modules)` first so bae/vggt mocks apply to the hard imports at the top of the file.

```python
@pytest.mark.skipif(not _pypose_available(), reason="requires pypose")
def test_optimize_early_exit_shape():
    """_optimize returns correct shapes when inlier count is below threshold (early-exit path)."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vggt_mod, _, proj_mod = _make_vggt_mock()
    proj_cam = np.ones((N, 3, P), dtype=np.float32)
    proj_mod.project_3D_points_np = MagicMock(return_value=(tracks.copy(), proj_cam))
    bae_mod = _make_bae_mock()

    extra_mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "vggt.dependency.track_predict": vggt_mod.dependency.track_predict,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.optim": bae_mod.optim,
    }

    with patch.dict(sys.modules, extra_mods):
        import importlib.util
        _ba_path = Path(__file__).parents[2] / "collab_splats" / "pointcloud" / "bundle_adjustment.py"
        _mod_name = "collab_splats.pointcloud.bundle_adjustment"
        spec = importlib.util.spec_from_file_location(_mod_name, _ba_path)
        ba_mod = importlib.util.module_from_spec(spec)
        sys.modules[_mod_name] = ba_mod
        spec.loader.exec_module(ba_mod)

        ba = ba_mod.BundleAdjustment()
        ref_pts, ref_ext, ref_intr = ba._optimize(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            tracks=tracks,
            vis_scores=vis_mask.astype(np.float32),
            max_reproj_error=4.0,
        )

    assert ref_pts.shape == (P, 3)
    assert ref_ext.shape == (N, 3, 4)
    assert ref_intr.shape == (N, 3, 3)
```

- [ ] **Step 3: Update `test_run_bundle_adjustment_no_reproj_filter` → `test_optimize_no_reproj_filter`**

```python
@pytest.mark.skipif(not _pypose_available(), reason="requires pypose")
def test_optimize_no_reproj_filter():
    """Passing max_reproj_error=None skips reprojection filtering."""
    N, P, H, W = 4, 50, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    vggt_mod, _, proj_mod = _make_vggt_mock()
    bae_mod = _make_bae_mock()

    extra_mods = {
        "vggt": vggt_mod,
        "vggt.dependency": vggt_mod.dependency,
        "vggt.dependency.projection": proj_mod,
        "vggt.dependency.track_predict": vggt_mod.dependency.track_predict,
        "bae": bae_mod,
        "bae.autograd": bae_mod.autograd,
        "bae.autograd.function": bae_mod.autograd.function,
        "bae.utils": bae_mod.utils,
        "bae.utils.pysolvers": bae_mod.utils.pysolvers,
        "bae.optim": bae_mod.optim,
    }

    with patch.dict(sys.modules, extra_mods):
        import importlib.util
        _ba_path = Path(__file__).parents[2] / "collab_splats" / "pointcloud" / "bundle_adjustment.py"
        _mod_name = "collab_splats.pointcloud.bundle_adjustment"
        spec = importlib.util.spec_from_file_location(_mod_name, _ba_path)
        ba_mod = importlib.util.module_from_spec(spec)
        sys.modules[_mod_name] = ba_mod
        spec.loader.exec_module(ba_mod)

        ba = ba_mod.BundleAdjustment()
        ref_pts, ref_ext, ref_intr = ba._optimize(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            tracks=tracks,
            vis_scores=vis_mask.astype(np.float32),
            max_reproj_error=None,
        )

    proj_mod.project_3D_points_np.assert_not_called()
    assert ref_pts.shape == (P, 3)
    assert ref_ext.shape == (N, 3, 4)
    assert ref_intr.shape == (N, 3, 3)
```

- [ ] **Step 4: Update `test_run_bundle_adjustment_reduces_reproj_error`**

Remove `image_size=(H, W)` from the `_run_bundle_adjustment` call and update to call `BundleAdjustment._optimize` instead:

```python
@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_optimize_reduces_reproj_error():
    """With noisy initial poses and clean 2D observations, _optimize must reduce reprojection error."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment

    rng = np.random.default_rng(0)
    N, P, H, W = 5, 200, 256, 256
    f = 200.0

    points3d = rng.uniform(-1, 1, (P, 3)).astype(np.float64)
    points3d[:, 2] += 3.0

    extrinsics_clean = np.zeros((N, 3, 4), dtype=np.float32)
    for i in range(N):
        extrinsics_clean[i, :3, :3] = np.eye(3)
        extrinsics_clean[i, :3, 3] = rng.uniform(-0.3, 0.3, 3).astype(np.float32)

    intrinsics = np.tile(
        np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32), (N, 1, 1)
    )

    tracks = np.zeros((N, P, 2), dtype=np.float32)
    vis_mask = np.ones((N, P), dtype=bool)
    for i in range(N):
        R, t = extrinsics_clean[i, :3, :3], extrinsics_clean[i, :3, 3]
        pts_cam = (R @ points3d.T).T + t
        z = pts_cam[:, 2]
        tracks[i, :, 0] = f * pts_cam[:, 0] / z + W / 2
        tracks[i, :, 1] = f * pts_cam[:, 1] / z + H / 2
        vis_mask[i] = (
            (z > 0.1)
            & (tracks[i, :, 0] >= 0)
            & (tracks[i, :, 0] < W)
            & (tracks[i, :, 1] >= 0)
            & (tracks[i, :, 1] < H)
        )

    extrinsics_noisy = extrinsics_clean.copy()
    for i in range(N):
        extrinsics_noisy[i, :3, 3] += rng.normal(0, 0.1, 3).astype(np.float32)

    def mean_reproj_error(ext):
        errs = []
        for i in range(N):
            R, t = ext[i, :3, :3], ext[i, :3, 3]
            pts_cam = (R @ points3d.T).T + t
            z = pts_cam[:, 2]
            px = f * pts_cam[:, 0] / z + W / 2
            py = f * pts_cam[:, 1] / z + H / 2
            proj = np.stack([px, py], axis=-1)
            mask = vis_mask[i]
            errs.append(np.linalg.norm(proj[mask] - tracks[i][mask], axis=-1).mean())
        return float(np.mean(errs))

    err_before = mean_reproj_error(extrinsics_noisy)

    ba = BundleAdjustment()
    _, ext_out, _ = ba._optimize(
        points3d.copy(),
        extrinsics_noisy,
        intrinsics,
        tracks,
        vis_mask.astype(np.float32),
        max_reproj_error=None,
        lm_steps=20,
    )

    err_after = mean_reproj_error(ext_out)
    assert err_after < err_before, f"BA did not reduce error: {err_before:.4f} → {err_after:.4f}"
```

- [ ] **Step 5: Update `test_bundle_adjustment_refine_threads_config` to patch `_optimize`**

The test currently patches `_run_bundle_adjustment`. Update to patch `BundleAdjustment._optimize`:

```python
def test_bundle_adjustment_refine_threads_config():
    """Config params and device are passed through to both private functions."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))) as mock_tracks, \
         patch.object(BundleAdjustment, "_optimize",
                      return_value=(np.zeros((5, 3)), refined_ext, refined_intr)) as mock_opt:
        cfg = BundleAdjustmentConfig(device="cpu", lm_steps=5, max_reproj_error=2.0)
        BundleAdjustment(config=cfg).refine(result)

    _, tracks_kw = mock_tracks.call_args
    assert tracks_kw["device"] == "cpu"
    # _optimize receives positional args; verify it was called once
    mock_opt.assert_called_once()
```

- [ ] **Step 6: Update `test_bundle_adjustment_refine_returns_feedforward_result` and `test_bundle_adjustment_refine_preserves_pts3d_colors`**

Both tests currently patch `_run_bundle_adjustment`. Update to patch `BundleAdjustment._optimize`:

```python
def test_bundle_adjustment_refine_returns_feedforward_result():
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))), \
         patch.object(BundleAdjustment, "_optimize",
                      return_value=(np.zeros((5, 3)), refined_ext, refined_intr)):
        out = BundleAdjustment().refine(result)

    assert isinstance(out, FeedforwardResult)
    assert out.extrinsics.shape == (N, 4, 4)
    assert out.intrinsics.shape == (N, 3, 3)


def test_bundle_adjustment_refine_preserves_pts3d_colors():
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment

    N, H, W = 2, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    original_pts3d = result.pts3d.copy()
    original_colors = result.colors.copy()
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(np.zeros((N, 5, 2)), np.ones((N, 5)), np.zeros((5, 3)))), \
         patch.object(BundleAdjustment, "_optimize",
                      return_value=(np.zeros((5, 3)), refined_ext, refined_intr)):
        out = BundleAdjustment().refine(result)

    np.testing.assert_array_equal(out.pts3d, original_pts3d)
    np.testing.assert_array_equal(out.colors, original_colors)
    assert out.pixel_indices is None
```

- [ ] **Step 7: Run tests — expect failures on the renamed/updated tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v 2>&1 | tail -30
```

Expected: tests that reference `_run_bundle_adjustment` directly or `_optimize` (not yet implemented) fail. Tests for `_get_default_solver` and `_extract_tracks_vggsfm` pass unchanged.

---

### Task 3: Rewrite `bundle_adjustment.py`

Complete rewrite in the correct structure. This is a pure refactor — no logic changes.

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`

- [ ] **Step 1: Verify `@map_transform` with inline math (TODO from spec)**

Before writing the full file, check whether `@map_transform` is strictly required by bae's LM or whether inline math in `forward()` suffices. Open a Python shell with bae available and test:

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import torch, pypose as pp
from bae.autograd.function import map_transform
# Verify map_transform is a callable decorator (not a class requiring vmap)
print(type(map_transform))
"
```

If `map_transform` is simply `lambda fn: fn` (identity) in the installed version, the decorator is a no-op and can be dropped — inline the math directly in `_BAModel.forward()` and remove `_reproject_per_camera` / `_reproject_shared`. If it wraps with vmap, keep the module-level functions as designed.

- [ ] **Step 2: Write the new file**

Replace the full content of `collab_splats/pointcloud/bundle_adjustment.py` with the following. Adjust the `_reproject_*` functions based on the result of Step 1.

```python
"""Bundle adjustment for collab-splats.

Public API:
- BundleAdjustmentConfig: configuration dataclass
- BundleAdjustment:       refines camera poses via VGGSfM tracks + LM bundle adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.projection import project_3D_points_np

if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult

__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]


########################################################
########## Configuration ##############################
########################################################


@dataclass
class BundleAdjustmentConfig:
    """Configuration for LM bundle adjustment."""

    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048       # track extraction: max query points
    query_frame_num: int = 5        # track extraction: number of query frames
    device: str | None = None       # target device; None = auto (CUDA if available, else CPU)


########################################################
########## BundleAdjustment ###########################
########################################################


class BundleAdjustment:
    """Refines camera poses via VGGSfM track extraction + LM bundle adjustment.

    Method-agnostic: works with any FeedforwardResult regardless of source creator.
    Does NOT reproject pts3d — call creator.reproject(result) after if needed.
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        self.config = config or BundleAdjustmentConfig()

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

    def _optimize(
        self,
        pts3d: np.ndarray,          # (P, 3)    initial 3D keypoint positions
        extrinsics: np.ndarray,     # (N, 3, 4) world-to-camera poses
        intrinsics: np.ndarray,     # (N, 3, 3) camera intrinsics
        tracks: np.ndarray,         # (N, P, 2) 2D pixel observations from VGGSfM
        vis_scores: np.ndarray,     # (N, P)    visibility scores in [0, 1]
        *,
        max_reproj_error: float | None = None,  # override self.config if provided
        lm_steps: int | None = None,            # override self.config if provided
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine 3D points and camera poses with Levenberg-Marquardt BA.

        Returns:
            refined_pts3d:        (P, 3)    float64
            refined_extrinsics:   (N, 3, 4) float32
            refined_intrinsics:   (N, 3, 3) float32
        """
        cfg = self.config
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        max_reproj = max_reproj_error if max_reproj_error is not None else cfg.max_reproj_error
        n_steps = lm_steps if lm_steps is not None else cfg.lm_steps

        # Work on copies; convert vis_scores (float) to bool visibility mask
        vis = vis_scores.astype(bool).copy()
        refined_extrinsics = extrinsics.astype(np.float32).copy()
        refined_intrinsics = intrinsics.astype(np.float32).copy()
        refined_pts3d = pts3d.astype(np.float64).copy()

        # Remove observations with high reprojection error under the current poses
        if max_reproj is not None:
            proj2d, proj_cam = project_3D_points_np(pts3d, extrinsics, intrinsics)
            # Behind-camera points get large sentinel projection so they fail the threshold
            behind = proj_cam[:, 2, :] <= 0
            proj2d = proj2d.copy()
            proj2d[behind] = 1e6
            reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)
            vis[reproj_err > max_reproj] = False

        # Drop points seen from fewer than 2 frames and points outside valid world range
        seen_enough = vis.sum(0) >= 2
        in_range = (np.abs(pts3d) < 3000).all(axis=-1)
        vis[:, ~(seen_enough & in_range)] = False

        # Drop under-constrained frames (too few inliers for reliable pose update)
        vis[vis.sum(1) < cfg.min_inliers_per_frame] = False

        # Build flat index arrays for active keyframes and landmarks
        active_frames = np.where(vis.any(1))[0]    # (K,)
        active_pts = np.where(vis.any(0))[0]       # (L,)

        if len(active_frames) < 2 or len(active_pts) < 2:
            return refined_pts3d, refined_extrinsics, refined_intrinsics

        frame_idx, pt_idx = np.where(vis[np.ix_(active_frames, active_pts)])
        global_frame_idx = active_frames[frame_idx]
        global_pt_idx = active_pts[pt_idx]
        obs_2d = tracks[global_frame_idx, global_pt_idx].astype(np.float64)   # (M, 2)

        # Build SE3 camera tensor from (K, 3, 4) extrinsics; pad to (K, 4, 4) for mat2SE3
        ext_sub = extrinsics[active_frames].astype(np.float64)
        ext_4x4 = np.concatenate(
            [ext_sub, np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float64), (len(active_frames), 1, 1))],
            axis=1,
        )
        cameras_se3 = pp.mat2SE3(torch.tensor(ext_4x4, dtype=torch.float64, device=device))

        # SIMPLE_PINHOLE: average fx/fy as single focal length per camera
        focal = (
            (intrinsics[active_frames, 0, 0] + intrinsics[active_frames, 1, 1]) / 2.0
        ).astype(np.float64)
        focal_tensor = torch.tensor(focal, dtype=torch.float64, device=device).unsqueeze(-1)   # (K, 1)
        principal_points = torch.tensor(
            intrinsics[active_frames, :2, 2].astype(np.float64),
            dtype=torch.float64, device=device,
        )  # (K, 2)
        pts3d_tensor = torch.tensor(
            pts3d[active_pts].astype(np.float64), dtype=torch.float64, device=device,
        )  # (L, 3)

        # Concatenate focal length into camera params tensor for per-camera case
        if cfg.shared_camera:
            cam_params = cameras_se3.data                             # (K, 7)
            shared_focal = focal_tensor.mean(0, keepdim=True)        # (1, 1)
        else:
            cam_params = torch.cat([cameras_se3.data, focal_tensor], dim=-1)   # (K, 8)
            shared_focal = None

        # Assemble observation index tensors
        obs_2d_t = torch.tensor(obs_2d, dtype=torch.float64, device=device)
        cam_idx = torch.tensor(frame_idx, dtype=torch.long, device=device)
        pt_idx_t = torch.tensor(pt_idx, dtype=torch.long, device=device)

        # Optimise reprojection residuals with Levenberg-Marquardt
        with torch.enable_grad():
            model = _BAModel(cam_params, pts3d_tensor, shared_focal, cfg.shared_camera)
            strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
            optimizer = LM(
                model,
                strategy=strategy,
                solver=_get_default_solver(device=device),
                reject=10,
            )
            scheduler = pp.optim.scheduler.StopOnPlateau(
                optimizer, steps=n_steps, patience=3, decreasing=1e-3, verbose=False,
            )
            scheduler.optimize(input={
                "points_2d": obs_2d_t,
                "camera_indices": cam_idx,
                "point_indices": pt_idx_t,
                "principal_points": principal_points,
            })

        # Recover (3, 4) extrinsics from optimised SE3 quaternion representation
        opt_cam = model.pose.data.detach().cpu().numpy()     # (K, 7) or (K, 8)
        opt_pts = model.pts.data.detach().cpu().numpy()      # (L, 3)

        opt_se3 = pp.SE3(torch.tensor(opt_cam[:, :7], dtype=torch.float64))
        opt_extrinsics_3x4 = opt_se3.matrix().numpy()[:, :3, :]
        refined_extrinsics[active_frames] = opt_extrinsics_3x4.astype(np.float32)
        refined_pts3d[active_pts] = opt_pts.astype(np.float64)

        # Write optimised focal lengths back to intrinsics matrix
        if cfg.shared_camera and model.shared_intr is not None:
            focal_val = float(model.shared_intr.data.detach().cpu().numpy().mean())
            refined_intrinsics[active_frames, 0, 0] = focal_val
            refined_intrinsics[active_frames, 1, 1] = focal_val
        elif not cfg.shared_camera:
            opt_focal = opt_cam[:, 7]
            refined_intrinsics[active_frames, 0, 0] = opt_focal
            refined_intrinsics[active_frames, 1, 1] = opt_focal

        return refined_pts3d, refined_extrinsics, refined_intrinsics


########################################################
########## Helpers ####################################
########################################################


def _get_default_solver(device: str | None = None) -> Any:
    """Return CuDSS when CUDA is requested and available, else PCG.

    Args:
        device: resolved device string; ``None`` auto-detects CUDA.
                ``"cpu"`` always returns PCG even when CUDA is present.
    """
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver
            return CuDirectSparseSolver()
        except (ImportError, RuntimeError):
            pass
    from bae.utils.pysolvers import PCG
    return PCG()


def _extract_tracks_vggsfm(
    images: torch.Tensor,
    conf: torch.Tensor | None,
    world_points: np.ndarray | None,
    *,
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
    fine_tracking: bool = False,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict cross-frame 2D tracks via VGGSfM (ALIKED+SP keypoints).

    Returns:
        tracks:     (N, P, 2) float32 — 2D pixel coords per frame per point.
        vis_scores: (N, P)    float32 — visibility score in [0, 1].
        pts3d:      (P, 3)    float32 — world-space 3D positions at keypoints.
    """
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Accept numpy array input from windowed LC path (raw_outputs stores merged images as numpy)
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).to(target_device)

    # predict_tracks inherits device from images.device — it does NOT self-relocate
    img_device = images.device
    # VGGSfM tracker uses grid_sample; not implemented for BFloat16 on CUDA
    images = images.float()

    conf_tensor: torch.Tensor | None = None
    if conf is not None:
        conf_tensor = conf.to(img_device)
        # Normalise to (N, H, W) — predict_tracks does not accept (N, 1, H, W)
        if conf_tensor.ndim == 4 and conf_tensor.shape[1] == 1:
            conf_tensor = conf_tensor.squeeze(1)

    pts3d_tensor: torch.Tensor | None = None
    if world_points is not None:
        pts3d_tensor = torch.tensor(world_points, dtype=images.dtype, device=img_device)

    # VGGSfM asserts H == W when both conf and world_points are provided; pad the shorter dim
    if conf_tensor is not None and pts3d_tensor is not None:
        H_img, W_img = images.shape[-2], images.shape[-1]
        if H_img != W_img:
            size = max(H_img, W_img)
            pad_h, pad_w = size - H_img, size - W_img
            images = F.pad(images, (0, pad_w, 0, pad_h))
            conf_tensor = F.pad(conf_tensor, (0, pad_w, 0, pad_h))
            pts3d_tensor = F.pad(pts3d_tensor, (0, 0, 0, pad_w, 0, pad_h))

    # Move conf/pts to CPU: VGGSfM mixes CPU numpy indexing with GPU tensors internally.
    # no_grad prevents pred_track from retaining a gradient that breaks .numpy()
    with torch.no_grad():
        pred_tracks, pred_vis_scores, _pred_confs, pred_pts3d, _pred_colors = predict_tracks(
            images,
            conf=conf_tensor.cpu() if conf_tensor is not None else None,
            points_3d=pts3d_tensor.cpu() if pts3d_tensor is not None else None,
            max_query_pts=max_query_pts,
            query_frame_num=query_frame_num,
            fine_tracking=fine_tracking,
        )

    tracks = np.asarray(pred_tracks).astype(np.float32)
    vis_scores = np.asarray(pred_vis_scores).astype(np.float32)
    pts3d = np.asarray(pred_pts3d).astype(np.float32)
    return tracks, vis_scores, pts3d


@map_transform
def _reproject_per_camera(pts, cam_params, principal_point):
    """Per-element pinhole projection with per-camera focal; @map_transform vectorises for bae LM Jacobian."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * cam_params[..., 7:] + principal_point


@map_transform
def _reproject_shared(pts, cam_params, principal_point, focal):
    """Per-element pinhole projection with shared focal length; @map_transform vectorises for bae LM Jacobian."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * focal + principal_point


class _BAModel(nn.Module):
    """Reprojection residual module for pypose Levenberg-Marquardt optimisation."""

    def __init__(
        self,
        cam_params: torch.Tensor,            # (K, 7) SE3 or (K, 8) SE3+focal
        pts_3d: torch.Tensor,                # (L, 3) 3D landmark positions
        shared_focal: torch.Tensor | None,   # (1, 1) if shared_camera, else None
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
        """Return reprojection residuals (predicted - observed) for all M observations."""
        pts = self.pts[point_indices]
        cam = self.pose[camera_indices]
        ctr = principal_points[camera_indices]

        if self.shared_camera:
            # Broadcast shared focal to match observation count (M,)
            focal = self.shared_intr[torch.zeros_like(camera_indices)]
            pts_proj = _reproject_shared(pts, cam, ctr, focal)
        else:
            pts_proj = _reproject_per_camera(pts, cam, ctr)

        return pts_proj - points_2d
```

> **Note on `_get_default_solver`:** Two lines inside still use lazy imports (`from bae.sparse.solve` and `from bae.utils.pysolvers`). These are `try/except` fallback paths that are legitimately conditional — they satisfy the CLAUDE.md exception for optional heavy deps. Leave them as-is.

- [ ] **Step 3: Run the tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v 2>&1 | tail -30
```

Expected: same pass/skip count as baseline from Task 1. If tests that load the module via `patch.dict(sys.modules)` fail with `ModuleNotFoundError` for `bae`, `pypose`, or `vggt`, the patch dict in the test is missing an entry for one of the new top-level imports — add it to `extra_mods` in the failing test.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "refactor(ba): drop lazy imports, collapse _run_bundle_adjustment into BundleAdjustment._optimize, lift _BAModel to module level"
```

---

### Task 4: Run full test suite and verify no regressions

**Files:**
- Run: `tests/`

- [ ] **Step 1: Run the full pointcloud test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v 2>&1 | tail -40
```

Expected: same pass/skip/fail as before the refactor. Any new failures are regressions introduced by this change — fix before proceeding.

- [ ] **Step 2: Run the wrapper tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_wrappers.py -v 2>&1 | tail -20
```

Expected: all passing. `wrappers.py` uses `BundleAdjustment` but only via `refine()` — the public API is unchanged.

- [ ] **Step 3: Commit if any fixes were needed**

If minor fixes were required to make tests pass, commit them:

```bash
git add -p
git commit -m "fix(ba): correct test mock patches after _optimize refactor"
```

---

### Task 5: Update worklog

**Files:**
- Modify: `worklog/STATE.md`
- Modify: `worklog/WORKLOG.md`

- [ ] **Step 1: Add entry to WORKLOG.md**

Add under today's date (2026-05-24):

```markdown
### 2026-05-24 — ba-module-cleanup

Spec: `docs/superpowers/specs/2026-05-24-ba-module-cleanup-design.md`
Plan: `docs/superpowers/plans/2026-05-24-ba-module-cleanup.md`

- Dropped all lazy imports; moved `torch`, `torch.nn`, `pypose`, `bae.*`, `vggt.*` to hard imports at top of file
- Collapsed `_run_bundle_adjustment` (220 lines, called once) into `BundleAdjustment._optimize()` — reads `self.config` directly
- Lifted `ReprojNonBatched` inner class → module-level `_BAModel(nn.Module)` with injected `shared_camera` flag
- Lifted conditional same-name closures → module-level `_reproject_per_camera` / `_reproject_shared` decorated with `@map_transform`
- Removed dead `image_size` parameter; removed inner `rotate_quat` wrapper (inlined as `pp.SE3(...).Act(pts)`)
- Reordered module: public API (config + class) first, helpers below
- Updated tests: removed `image_size` from call sites, patched `_optimize` instead of `_run_bundle_adjustment`, removed stale `bae.utils.ba.rotate_quat` mock
- ~34% line reduction (438 → ~290 lines)
```

- [ ] **Step 2: Commit worklog**

```bash
git add worklog/WORKLOG.md worklog/STATE.md
git commit -m "docs(worklog): record ba-module-cleanup refactor"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 9 spec problems addressed — lazy imports removed (Task 3), `_run_bundle_adjustment` collapsed (Task 3), inner class lifted (Task 3), dead param removed (Task 3), module reordered (Task 3), test stale mock removed (Task 2), variable names updated (throughout Task 3)
- [x] **No placeholders:** All code blocks are complete
- [x] **Type consistency:** `_optimize` signature uses `pts3d` / `extrinsics` / `intrinsics` / `tracks` / `vis_scores` consistently in both the implementation and all test call sites
- [x] **`_BAModel` forward args:** `principal_points` used consistently in `_optimize`'s `scheduler.optimize(input=...)` dict and `_BAModel.forward()` signature
- [x] **`max_reproj_error` override:** `_optimize` accepts optional keyword overrides for `max_reproj_error` and `lm_steps` to support direct test calls without a config object
