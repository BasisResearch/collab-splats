# Generalise Multiview Depth Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `compute_multiview_depth_confidence` function to `base.py` and wire it into all three feedforward creators so the geometric depth-consistency filter is available model-agnostically.

**Architecture:** Implement the algorithm natively in torch (no mapanything dependency); expose it as a standalone function that takes raw numpy arrays. Add `use_multiview_confidence` / `mv_conf_threshold` fields to each creator; VGGTXCreator/VGGTOmega pass an `extra_mask` to `unproject_and_filter_points`; MapAnythingCreator is fully refactored to drop the upstream `use_multiview_confidence` kwarg and call the shared function instead.

**Tech Stack:** `torch`, `torch.nn.functional.grid_sample`, numpy, pytest

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward/base.py` | Add `compute_multiview_depth_confidence` function |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Add `extra_mask` param to `unproject_and_filter_points`; add `use_multiview_confidence`, `mv_conf_threshold` fields to `VGGTXCreator`; wire in `_postprocess` |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Same two fields + wiring in `_postprocess` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Drop upstream `use_multiview_confidence` kwarg; add `mv_conf_abs_thresh`, `mv_conf_threshold` fields; call shared fn in `_postprocess` |
| `collab_splats/pointcloud/feedforward/__init__.py` | Re-export `compute_multiview_depth_confidence` |
| `tests/pointcloud/test_mv_conf.py` | New: unit tests for shared function |
| `tests/pointcloud/test_vggtx_creator.py` | Add: `use_multiview_confidence` creator integration test |
| `tests/pointcloud/test_vggt_omega_creator.py` | Add: same |
| `tests/pointcloud/test_mapanything_creator.py` | Add: refactor regression test |

---

### Task 1: Add `compute_multiview_depth_confidence` to `base.py`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`
- Test: `tests/pointcloud/test_mv_conf.py` (create)

- [ ] **Step 1: Create failing tests**

Create `tests/pointcloud/test_mv_conf.py`:

```python
"""Unit tests for compute_multiview_depth_confidence in base.py."""
import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence


def _make_intrinsics(H: int, W: int) -> np.ndarray:
    """Simple pinhole K with focal = W, principal at image centre."""
    return np.array(
        [[float(W), 0.0, W / 2.0], [0.0, float(H), H / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_compute_mv_conf_identical_cameras():
    """Two co-located cameras, same depth → mv_conf = 1.0 for all valid pixels."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])                                              # (2, 3, 3)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)                  # (2, 4, 4) w2c = I

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    )

    assert mv_conf.shape == (N, H, W)
    assert mv_conf.dtype == np.float32
    assert np.all(mv_conf > 0.9), f"Expected all >0.9; min={mv_conf.min():.4f}"


def test_compute_mv_conf_depth_disagreement():
    """Same pose, very different depths → mv_conf = 0.0 everywhere."""
    N, H, W = 2, 4, 4
    depth = np.zeros((N, H, W), dtype=np.float32)
    depth[0] = 1.0    # frame 0: 1 m
    depth[1] = 100.0  # frame 1: 100 m — way outside 5 % tolerance

    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    )

    assert np.all(mv_conf == 0.0), f"Expected all 0.0; max={mv_conf.max():.4f}"


def test_compute_mv_conf_depth_masks_source():
    """depth_masks=False on frame 0 → frame 0 source pixels get mv_conf = 0."""
    N, H, W = 2, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    depth_masks = np.ones((N, H, W), dtype=bool)
    depth_masks[0] = False  # mask out all source pixels in frame 0

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics,
        depth_masks=depth_masks, abs_thresh=0.0, rel_thresh=0.1, device="cpu",
    )

    # Frame 0 fully masked → no valid source pixels → 0
    assert np.all(mv_conf[0] == 0.0), f"Frame 0 should be 0; got {mv_conf[0]}"
    # Frame 1 unmasked → projects into frame 0 target (depth=5) → inliers
    assert np.any(mv_conf[1] > 0.0), "Frame 1 should have some inliers"


def test_compute_mv_conf_output_shape():
    """Output shape matches (N, H, W) regardless of N."""
    for N in (1, 3, 5):
        H, W = 6, 6
        depth = np.ones((N, H, W), dtype=np.float32) * 3.0
        K = _make_intrinsics(H, W)
        intrinsics = np.stack([K] * N)
        extrinsics = np.stack([np.eye(4, dtype=np.float32)] * N)
        out = compute_multiview_depth_confidence(
            depth, intrinsics, extrinsics, device="cpu"
        )
        assert out.shape == (N, H, W), f"N={N}: expected {(N,H,W)}, got {out.shape}"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'compute_multiview_depth_confidence'`

- [ ] **Step 3: Implement `compute_multiview_depth_confidence` in `base.py`**

Add the following import at the top of `base.py` (after existing imports):

```python
import torch.nn.functional as F
from typing import Optional
```

Then add the function after the existing geometry helpers section (before `BaseFeedforwardCreator`):

```python
def compute_multiview_depth_confidence(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_masks: "Optional[np.ndarray]" = None,
    abs_thresh: float = 0.0,
    rel_thresh: float = 0.05,
    device: str = "cuda",
) -> np.ndarray:
    """Geometric cross-view depth consistency confidence per pixel.

    For each source pixel, projects it into all other frames and checks whether
    the reprojected and sampled depths agree within abs_thresh + rel_thresh * depth.
    Returns per-pixel inlier ratio across overlapping views, in [0, 1].

    Args:
        depth:       (N, H, W) float32 Z-depth per frame.
        intrinsics:  (N, 3, 3) float32 pinhole intrinsics in pixel units.
        extrinsics:  (N, 4, 4) float32 world-to-cam transforms.
        depth_masks: (N, H, W) bool — source pixels to include; None = all valid depth.
        abs_thresh:  Absolute depth tolerance (depth units). 0.0 for non-metric depth.
        rel_thresh:  Relative depth tolerance as fraction of expected depth.
        device:      Torch device for computation.
    """
    dev = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    N, H, W = depth.shape

    depth_t = torch.from_numpy(depth.astype(np.float32)).to(dev)       # (N, H, W)
    K = torch.from_numpy(intrinsics.astype(np.float32)).to(dev)        # (N, 3, 3)
    E = torch.from_numpy(extrinsics.astype(np.float32)).to(dev)        # (N, 4, 4) w2c
    cam2world = torch.linalg.inv(E)                                      # (N, 4, 4) c2w

    # Pixel grid: grid[r, c] = [col=x, row=y, 1] homogeneous pixel coordinate
    rows, cols = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=dev),
        torch.arange(W, dtype=torch.float32, device=dev),
        indexing="ij",
    )  # (H, W) each
    pixel_h = torch.stack(
        [cols, rows, torch.ones(H, W, device=dev)], dim=-1
    ).reshape(-1, 3)  # (H*W, 3) [x, y, 1]

    inlier_sum = torch.zeros(N, H, W, dtype=torch.float32, device=dev)
    valid_sum  = torch.zeros(N, H, W, dtype=torch.float32, device=dev)

    for i in range(N):
        # Unproject source pixels to world space
        K_i_inv = torch.linalg.inv(K[i])                              # (3, 3)
        cam_rays = (K_i_inv @ pixel_h.T).T                            # (H*W, 3) unit rays
        src_d = depth_t[i].reshape(-1, 1)                             # (H*W, 1)
        src_valid = (src_d > 0).squeeze(-1)                            # (H*W,)
        if depth_masks is not None:
            dm_i = torch.from_numpy(depth_masks[i]).bool().to(dev).reshape(-1)
            src_valid = src_valid & dm_i

        # 3-D world-space points for all source pixels
        pts_cam_i = cam_rays * src_d                                   # (H*W, 3)
        pts_cam_h = torch.cat(
            [pts_cam_i, torch.ones(H * W, 1, device=dev)], dim=-1
        )                                                               # (H*W, 4)
        pts_world = (cam2world[i] @ pts_cam_h.T).T[:, :3]             # (H*W, 3)

        for j in range(N):
            if i == j:
                continue

            # Project world points into camera j
            pts_world_h = torch.cat(
                [pts_world, torch.ones(H * W, 1, device=dev)], dim=-1
            )
            pts_cam_j = (E[j] @ pts_world_h.T).T[:, :3]               # (H*W, 3)

            expected_d = pts_cam_j[:, 2]                               # (H*W,) Z in cam-j
            in_front = expected_d > 0

            # Project to pixel coords in frame j
            proj_j = (K[j] @ pts_cam_j.T).T                           # (H*W, 3)
            z_j = proj_j[:, 2:3].clamp(min=1e-6)
            px_j = proj_j[:, :2] / z_j                                # (H*W, 2)

            # Normalise to [-1, 1] (align_corners=True convention)
            px_norm = torch.stack(
                [px_j[:, 0] / (W - 1) * 2 - 1,
                 px_j[:, 1] / (H - 1) * 2 - 1],
                dim=-1,
            )                                                           # (H*W, 2)
            in_bounds = (
                (px_norm[:, 0] >= -1) & (px_norm[:, 0] <= 1)
                & (px_norm[:, 1] >= -1) & (px_norm[:, 1] <= 1)
            )
            valid_ij = src_valid & in_front & in_bounds               # (H*W,)

            # Sample target depth at projected pixel locations.
            # grid shape (1, H, W, 2): grid[0,r,c,:] = norm coords where source pixel (r,c) projects
            grid = px_norm.reshape(1, H, W, 2)
            sampled_d = F.grid_sample(
                depth_t[j].unsqueeze(0).unsqueeze(0),                 # (1, 1, H, W)
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze()                                                 # (H, W)
            sampled_d_flat = sampled_d.reshape(-1)                     # (H*W,)

            # Inlier: depths agree within abs + rel tolerance
            tol = abs_thresh + rel_thresh * expected_d.abs()
            inlier = (
                (torch.abs(expected_d - sampled_d_flat) < tol)
                & valid_ij
                & (sampled_d_flat > 0)
            )

            inlier_sum[i] += inlier.reshape(H, W).float()
            valid_sum[i]  += valid_ij.reshape(H, W).float()

    # Inlier ratio; pixels with no overlapping views → 0
    mv_conf = torch.where(
        valid_sum > 0,
        inlier_sum / valid_sum.clamp(min=1.0),
        torch.zeros_like(inlier_sum),
    )
    return mv_conf.cpu().numpy().astype(np.float32)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_mv_conf.py
git commit -m "feat(feedforward): add native compute_multiview_depth_confidence to base.py"
```

---

### Task 2: Add `extra_mask` to `unproject_and_filter_points` in `vggtx.py`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:81`
- Test: `tests/pointcloud/test_vggtx_creator.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pointcloud/test_vggtx_creator.py`:

```python
def test_unproject_and_filter_points_extra_mask():
    """extra_mask=False pixels are excluded from the output."""
    import numpy as np
    from collab_splats.pointcloud.feedforward.vggtx import unproject_and_filter_points

    N, H, W = 2, 4, 4
    depth = np.ones((N, H, W, 1), dtype=np.float32)
    depth_conf = np.ones((N, H, W), dtype=np.float32)
    images = np.zeros((N, 3, H, W), dtype=np.float32)
    extrinsic = np.stack([np.eye(4)[:3, :]] * N).astype(np.float32)
    intrinsic = np.stack([np.eye(3)] * N).astype(np.float32)

    # Without extra_mask: all pixels survive (conf_threshold=0.0)
    pts_all, _, _ = unproject_and_filter_points(
        depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.0
    )

    # extra_mask zeros out frame 0 completely
    extra_mask = np.ones((N, H, W), dtype=bool)
    extra_mask[0] = False
    pts_masked, _, _ = unproject_and_filter_points(
        depth, depth_conf, images, extrinsic, intrinsic, conf_threshold=0.0,
        extra_mask=extra_mask,
    )

    assert len(pts_masked) < len(pts_all), (
        f"Expected fewer points with extra_mask; got {len(pts_masked)} vs {len(pts_all)}"
    )
    # Frame 0 masked → only frame 1's H*W points survive
    assert len(pts_masked) == H * W, f"Expected {H*W}, got {len(pts_masked)}"
```

- [ ] **Step 2: Run — verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py::test_unproject_and_filter_points_extra_mask -v
```

Expected: `TypeError: unproject_and_filter_points() got an unexpected keyword argument 'extra_mask'`

- [ ] **Step 3: Add `extra_mask` parameter**

In `collab_splats/pointcloud/feedforward/vggtx.py`, modify `unproject_and_filter_points` at line 81. Change the signature and add the AND logic after `conf_mask = depth_conf >= threshold_val`:

```python
def unproject_and_filter_points(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    images: Any,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    conf_threshold: float = 50.0,
    max_points: int = 500_000,
    extra_mask: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject depth to world-space points and filter by confidence.

    Args:
        depth:          (N, H, W, 1) float32 depth maps.
        depth_conf:     (N, H, W) float32 confidence maps.
        images:         (N, 3, H, W) tensor or array of preprocessed images.
        extrinsic:      (N, 3, 4) or (N, 4, 4) camera extrinsics.
        intrinsic:      (N, 3, 3) camera intrinsics.
        conf_threshold: Percentile cutoff (>1.0) or raw threshold (≤1.0).
        max_points:     Maximum number of output points; excess are randomly subsampled.
        extra_mask:     (N, H, W) bool — additional mask ANDed with conf_mask. None = no-op.

    Returns:
        pts3d:         (P, 3) float32 world-space points.
        colors:        (P, 3) uint8 RGB.
        pixel_indices: (P, 3) int32 — [frame_id, row, col].
    """
```

Then, after the line `conf_mask = depth_conf >= threshold_val`, add:

```python
    if extra_mask is not None:
        conf_mask = conf_mask & extra_mask
```

The rest of the function is unchanged.

- [ ] **Step 4: Run test — verify it passes**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py::test_unproject_and_filter_points_extra_mask -v
```

Expected: PASS.

- [ ] **Step 5: Run all existing tests — verify no regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_vggt_omega_creator.py -v 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_vggtx_creator.py
git commit -m "feat(feedforward): add extra_mask param to unproject_and_filter_points"
```

---

### Task 3: Wire `use_multiview_confidence` into `VGGTXCreator`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py` (~line 141+ for class, ~line 288 for `_postprocess`)
- Test: `tests/pointcloud/test_vggtx_creator.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pointcloud/test_vggtx_creator.py`:

```python
def test_vggtx_use_multiview_confidence_calls_compute_fn(tmp_path):
    """VGGTXCreator with use_multiview_confidence=True passes extra_mask to unproject."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    N, H, W = 2, 4, 4
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": np.zeros((N, 3, H, W), dtype=np.float32),
        "extrinsic": np.stack([np.eye(4)[:3, :]] * N).astype(np.float32),
        "intrinsics": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
        "intrinsics_downsampled": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
    }

    creator = VGGTXCreator(use_multiview_confidence=True, mv_conf_threshold=0.0)
    creator.image_paths = [tmp_path / f"{i:06d}.jpg" for i in range(N)]
    creator.original_coords = np.zeros((N, 6), dtype=np.float32)
    creator.views = None

    # compute_multiview_depth_confidence returns all-ones → mv_mask all True → no points dropped
    mv_conf_ones = np.ones((N, H, W), dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.compute_multiview_depth_confidence",
        return_value=mv_conf_ones,
    ) as mock_mv:
        result = creator._postprocess(raw_outputs)

    mock_mv.assert_called_once()
    # First positional arg is depth (N, H, W) after squeezing
    called_depth = mock_mv.call_args[0][0]
    assert called_depth.shape == (N, H, W), f"Expected ({N},{H},{W}), got {called_depth.shape}"
    assert result is not None
    assert len(result.points) > 0
```

- [ ] **Step 2: Run — verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py::test_vggtx_use_multiview_confidence_calls_compute_fn -v
```

Expected: `TypeError: VGGTXCreator.__init__() got an unexpected keyword argument 'use_multiview_confidence'`

- [ ] **Step 3: Add fields to `VGGTXCreator` dataclass**

In `vggtx.py`, find the `VGGTXCreator` class (line ~141). Add these two fields after the existing `conf_threshold` and `max_points` fields:

```python
    use_multiview_confidence: bool = False
    mv_conf_threshold: float = 0.0
```

Also add `compute_multiview_depth_confidence` to the import from `.base`:

```python
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _raw_to_world_points,
    compute_multiview_depth_confidence,
)
```

- [ ] **Step 4: Wire into `VGGTXCreator._postprocess`**

In `_postprocess` (line ~288 in `vggtx.py`), after the optional `run_global_alignment` block and before the `unproject_and_filter_points` call, insert:

```python
        # Optionally compute geometric cross-view depth consistency mask
        mv_mask = None
        if self.use_multiview_confidence:
            depth_np = raw_outputs["depth"]
            if depth_np.ndim == 4:
                depth_np = depth_np.squeeze(-1)   # (N, H, W)
            extr_4x4 = extrinsics_to_homogeneous(extrinsic)
            mv_conf = compute_multiview_depth_confidence(
                depth_np,
                intrinsic,
                extr_4x4,
                abs_thresh=0.0,
                rel_thresh=0.05,
            )
            mv_mask = mv_conf > self.mv_conf_threshold
```

Then change the `unproject_and_filter_points` call to pass `extra_mask=mv_mask`:

```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
        )
```

- [ ] **Step 5: Run test — verify it passes**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py::test_vggtx_use_multiview_confidence_calls_compute_fn -v
```

Expected: PASS.

- [ ] **Step 6: Run full vggtx test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_vggtx_creator.py
git commit -m "feat(feedforward): wire use_multiview_confidence into VGGTXCreator"
```

---

### Task 4: Wire `use_multiview_confidence` into `VGGTOmegaCreator`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py` (line ~92 for class, ~227 for `_postprocess`)
- Test: `tests/pointcloud/test_vggt_omega_creator.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pointcloud/test_vggt_omega_creator.py`:

```python
def test_omega_use_multiview_confidence_calls_compute_fn(tmp_path):
    """VGGTOmegaCreator with use_multiview_confidence=True passes extra_mask to unproject."""
    import numpy as np
    from unittest.mock import patch
    from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator

    N, H, W = 2, 4, 4
    raw_outputs = {
        "depth": np.ones((N, H, W, 1), dtype=np.float32),
        "depth_conf": np.ones((N, H, W), dtype=np.float32),
        "images": np.zeros((N, 3, H, W), dtype=np.float32),
        "extrinsic": np.stack([np.eye(4)[:3, :]] * N).astype(np.float32),
        "intrinsics": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
        "intrinsics_downsampled": np.eye(3, dtype=np.float32)[np.newaxis].repeat(N, axis=0),
    }

    creator = VGGTOmegaCreator(use_multiview_confidence=True, mv_conf_threshold=0.0)
    creator.image_paths = [tmp_path / f"{i:06d}.jpg" for i in range(N)]
    creator.original_coords = np.zeros((N, 6), dtype=np.float32)
    creator.views = None

    mv_conf_ones = np.ones((N, H, W), dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.vggt_omega.compute_multiview_depth_confidence",
        return_value=mv_conf_ones,
    ) as mock_mv:
        result = creator._postprocess(raw_outputs)

    mock_mv.assert_called_once()
    called_depth = mock_mv.call_args[0][0]
    assert called_depth.shape == (N, H, W)
    assert result is not None
```

- [ ] **Step 2: Run — verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py::test_omega_use_multiview_confidence_calls_compute_fn -v
```

Expected: `TypeError: VGGTOmegaCreator.__init__() got an unexpected keyword argument 'use_multiview_confidence'`

- [ ] **Step 3: Add fields to `VGGTOmegaCreator` and update imports**

In `vggt_omega.py`, update the import from `.base` to include `compute_multiview_depth_confidence`:

```python
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _raw_to_world_points,
    compute_multiview_depth_confidence,
)
```

Then in the `VGGTOmegaCreator` dataclass (line ~138), add after `conf_threshold`:

```python
    use_multiview_confidence: bool = False
    mv_conf_threshold: float = 0.0
```

- [ ] **Step 4: Wire into `VGGTOmegaCreator._postprocess`**

In `_postprocess` (line ~227), after the `extrinsic = raw_outputs["extrinsic"]` / `intrinsic = raw_outputs["intrinsics"]` lines and before the `unproject_and_filter_points` call, insert:

```python
        # Optionally compute geometric cross-view depth consistency mask
        mv_mask = None
        if self.use_multiview_confidence:
            depth_np = raw_outputs["depth"]
            if depth_np.ndim == 4:
                depth_np = depth_np.squeeze(-1)   # (N, H, W)
            extr_4x4 = extrinsics_to_homogeneous(extrinsic)
            mv_conf = compute_multiview_depth_confidence(
                depth_np,
                intrinsic,
                extr_4x4,
                abs_thresh=0.0,
                rel_thresh=0.05,
            )
            mv_mask = mv_conf > self.mv_conf_threshold
```

Change the `unproject_and_filter_points` call to:

```python
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
        )
```

- [ ] **Step 5: Run test + full suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_vggt_omega_creator.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggt_omega.py tests/pointcloud/test_vggt_omega_creator.py
git commit -m "feat(feedforward): wire use_multiview_confidence into VGGTOmegaCreator"
```

---

### Task 5: Refactor `MapAnythingCreator._postprocess` to use shared function

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Test: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pointcloud/test_mapanything_creator.py`:

```python
def test_mapanything_postprocess_calls_shared_mv_conf(monkeypatch):
    """After refactor, use_multiview_confidence calls compute_multiview_depth_confidence,
    not the upstream postprocess_model_outputs_for_inference with use_multiview_confidence=True."""
    import numpy as np
    from unittest.mock import patch, MagicMock, call
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    # Track whether postprocess_model_outputs_for_inference is called with use_multiview_confidence=True
    upstream_calls = []

    def fake_postprocess(raw_outputs, processed_views, **kwargs):
        upstream_calls.append(kwargs.get("use_multiview_confidence", False))
        # Return minimal fake preds — one per raw_output entry
        N = len(raw_outputs)
        H, W = 4, 4
        preds = []
        for _ in range(N):
            preds.append({
                "mask": [np.ones((1, H, W, 1), dtype=np.float32)],
                "depth_z": [np.ones((1, H, W, 1), dtype=np.float32) * 2.0],
                "pts3d": [np.zeros((1, H, W, 3), dtype=np.float32)],
                "img_no_norm": [np.zeros((1, H, W, 3), dtype=np.float32)],
                "camera_poses": [np.eye(4, dtype=np.float32)[np.newaxis]],
                "intrinsics": [np.eye(3, dtype=np.float32)[np.newaxis]],
            })
        return preds

    mv_conf_return = np.ones((2, 4, 4), dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        side_effect=fake_postprocess,
    ), patch(
        "collab_splats.pointcloud.feedforward.mapanything.compute_multiview_depth_confidence",
        return_value=mv_conf_return,
    ) as mock_mv:
        creator = MapAnythingCreator(use_multiview_confidence=True)
        # _processed_views must match raw_outputs length
        H, W = 4, 4
        creator._processed_views = [
            {"img": np.zeros((1, 3, H, W), dtype=np.float32)} for _ in range(2)
        ]
        creator.image_paths = []
        creator.original_coords = np.zeros((2, 6), dtype=np.float32)

        raw_outputs = [{"dummy": i} for i in range(2)]
        result = creator._postprocess(raw_outputs)

    # upstream must NOT be called with use_multiview_confidence=True
    assert all(not v for v in upstream_calls), (
        f"postprocess_model_outputs_for_inference was called with use_multiview_confidence=True: {upstream_calls}"
    )
    # shared fn must be called
    mock_mv.assert_called_once()
```

- [ ] **Step 2: Run — verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_postprocess_calls_shared_mv_conf -v
```

Expected: test fails because `postprocess_model_outputs_for_inference` is still called with `use_multiview_confidence=True`.

- [ ] **Step 3: Add `compute_multiview_depth_confidence` to mapanything.py imports**

In `mapanything.py`, find the import from `.base` and add `compute_multiview_depth_confidence`:

```python
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    compute_multiview_depth_confidence,
)
```

- [ ] **Step 4: Add `mv_conf_abs_thresh` and `mv_conf_threshold` fields to `MapAnythingCreator`**

In the `MapAnythingCreator` dataclass, after `use_multiview_confidence: bool = True`, add:

```python
    mv_conf_abs_thresh: float = 0.02   # metric depth (metres) — calibrated for MapAnything
    mv_conf_threshold: float = 0.0     # keep any pixel with ≥1 inlier view
```

Also update the docstring for the `mv_conf_threshold` field description if there's existing documentation about the parameters.

- [ ] **Step 5: Refactor `_postprocess` in `MapAnythingCreator`**

Change the `postprocess_model_outputs_for_inference` call from:

```python
        processed = postprocess_model_outputs_for_inference(
            raw_outputs,
            self._processed_views,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=not self.use_multiview_confidence,
            use_multiview_confidence=self.use_multiview_confidence,
            confidence_percentile=self.confidence_percentile,
        )
```

to:

```python
        # Always apply the learned confidence mask via percentile; mv_conf applied separately below.
        processed = postprocess_model_outputs_for_inference(
            raw_outputs,
            self._processed_views,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            confidence_percentile=self.confidence_percentile,
        )
```

Then, in the per-frame loop, remove the block that reads `pred["conf"]`:

```python
        # REMOVE this block entirely:
        # if self.use_multiview_confidence and "conf" in pred:
        #     mv = pred["conf"][0].cpu().numpy().astype(np.float32)
        #     valid = valid & (mv > 0)
```

So the per-frame loop becomes:

```python
        for pred in processed:
            m = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)       # (H, W)
            dz = pred["depth_z"][0].squeeze(-1).cpu().numpy()                # (H, W)
            valid = m & (dz > 0)
            masks.append(valid)
            depth_list.append(dz)
            pts3d_grid.append(pred["pts3d"][0].cpu().numpy())
            colors_grid.append(
                (pred["img_no_norm"][0].cpu().numpy() * 255).astype(np.uint8)
            )
            images_list.append(pred["img_no_norm"][0].cpu().permute(2, 0, 1))
            if pred.get("conf") is not None:
                c = pred["conf"][0]
                conf_list.append(c[0] if c.ndim == 3 else c)
            cam2world = pred["camera_poses"][0].cpu().numpy()
            extrinsics_list.append(invert_poses(cam2world)[:3, :4])
            intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())
```

Then, after `combined_mask = np.stack(masks)` and before the `randomly_limit_trues` block, add:

```python
        # Apply shared geometric mv_conf filter (replaces upstream use_multiview_confidence path)
        if self.use_multiview_confidence:
            stacked_depth = np.stack(depth_list)                              # (N, H, W)
            stacked_intr  = np.stack(intrinsics_list)                         # (N, 3, 3)
            stacked_extr  = extrinsics_to_homogeneous(
                np.stack(extrinsics_list)
            )                                                                  # (N, 4, 4) w2c
            mv_conf = compute_multiview_depth_confidence(
                stacked_depth,
                stacked_intr,
                stacked_extr,
                depth_masks=combined_mask,
                abs_thresh=self.mv_conf_abs_thresh,
                rel_thresh=0.02,
            )
            combined_mask = combined_mask & (mv_conf > self.mv_conf_threshold)
```

- [ ] **Step 6: Run test — verify it passes**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_postprocess_calls_shared_mv_conf -v
```

Expected: PASS.

- [ ] **Step 7: Run full mapanything test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py tests/pointcloud/test_mapanything.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_mapanything_creator.py
git commit -m "refactor(feedforward): MapAnythingCreator uses shared compute_multiview_depth_confidence"
```

---

### Task 6: Export from `__init__.py` and full test run

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`
- Test: run full suite

- [ ] **Step 1: Add export**

In `collab_splats/pointcloud/feedforward/__init__.py`, add `compute_multiview_depth_confidence` to the public types section and to `__all__`:

```python
from .base import (
    FeedforwardResult,
    BaseFeedforwardCreator,
    build_pycolmap_reconstruction,
    compute_multiview_depth_confidence,
)
```

And in `__all__`:

```python
__all__ = [
    "FeedforwardResult",
    "BaseFeedforwardCreator",
    "build_pycolmap_reconstruction",
    "compute_multiview_depth_confidence",
    "VGGTXCreator",
    "MapAnythingCreator",
    "VGGTOmegaCreator",
    "VGGTSPARKCreator",
    "unproject_and_filter_points",
]
```

- [ ] **Step 2: Verify import works**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.pointcloud.feedforward import compute_multiview_depth_confidence; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass. If any fail, investigate before proceeding.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/feedforward/__init__.py
git commit -m "feat(feedforward): export compute_multiview_depth_confidence from feedforward __init__"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 4 spec goals covered — native function in base.py (Task 1); shared function for all creators (Tasks 3-5); MapAnything refactored (Task 5); scale-invariant defaults with abs_thresh=0.0 for VGGT, abs_thresh=0.02 for MapAnything (Tasks 3, 4, 5).
- [x] **No placeholders:** All steps contain actual code.
- [x] **Type consistency:** `compute_multiview_depth_confidence` signature is identical across all import sites; `extra_mask` parameter name consistent in all call sites; `mv_conf_threshold` field name consistent across all three creators.
- [x] **Watch-outs addressed:** extrinsics inversion handled inside `compute_multiview_depth_confidence` (using `torch.linalg.inv`); depth squeezed to (N,H,W) before passing; `extrinsics_to_homogeneous` called before passing (N,3,4) → (N,4,4) in Tasks 3-5.
