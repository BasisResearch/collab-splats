# Feedforward Tutorial Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb` with a clean tutorial notebook that runs both VGGT-X and MapAnythingCreator on C0043 keyframes, compares results with stats and a camera overlay viewer, and adds `pointcloud_to_polydata` to the package.

**Architecture:** Two-part change: (1) add `pointcloud_to_polydata(**point_data)` utility to `collab_splats/utils/visualization.py`, (2) rewrite the notebook to use it alongside the existing `visualize_splat` + `create_camera_frustum_pyvista`. The notebook follows the keyframe extraction tutorial by calling `score_all_frames` in a single cell, then running each creator step-by-step.

**Tech Stack:** Python 3.10 (`/opt/conda/envs/nerfstudio/bin/python`), PyVista, Open3D, NumPy, PyTorch, Jupyter notebook JSON.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/utils/visualization.py` | Modify | Add `pointcloud_to_polydata` |
| `tests/test_visualization.py` | Modify | Test `pointcloud_to_polydata` |
| `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb` | Rewrite | Tutorial notebook |

`docs/source/tutorials/index.rst` — already lists `pointcloud/feedforward_exploration`, no change needed.

---

## Task 1: Add `pointcloud_to_polydata` to visualization.py

**Files:**
- Modify: `collab_splats/utils/visualization.py`
- Modify: `tests/test_visualization.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_visualization.py`:

```python
def test_pointcloud_to_polydata_attaches_rgb():
    from collab_splats.utils.visualization import pointcloud_to_polydata
    pts3d = np.random.rand(100, 3).astype(np.float32)
    colors = (np.random.rand(100, 3) * 255).astype(np.uint8)
    cloud = pointcloud_to_polydata(pts3d, RGB=colors)
    assert cloud.n_points == 100
    assert "RGB" in cloud.array_names


def test_pointcloud_to_polydata_multiple_scalars():
    from collab_splats.utils.visualization import pointcloud_to_polydata
    pts3d = np.random.rand(50, 3).astype(np.float32)
    colors = (np.random.rand(50, 3) * 255).astype(np.uint8)
    scores = np.random.rand(50).astype(np.float32)
    cloud = pointcloud_to_polydata(pts3d, RGB=colors, similarity=scores)
    assert "RGB" in cloud.array_names
    assert "similarity" in cloud.array_names


def test_pointcloud_to_polydata_no_scalars():
    from collab_splats.utils.visualization import pointcloud_to_polydata
    pts3d = np.zeros((10, 3), dtype=np.float32)
    cloud = pointcloud_to_polydata(pts3d)
    assert cloud.n_points == 10
    assert cloud.array_names == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_visualization.py::test_pointcloud_to_polydata_attaches_rgb -v
```

Expected: `FAILED` with `ImportError` or `AttributeError: module has no attribute 'pointcloud_to_polydata'`

- [ ] **Step 3: Implement `pointcloud_to_polydata`**

Add after the `create_camera_frustum_pyvista` function (around line 377) in `collab_splats/utils/visualization.py`:

```python
def pointcloud_to_polydata(pts3d: np.ndarray, **point_data) -> "pv.PolyData":
    """Convert pts3d + named scalar arrays to a PyVista PolyData.

    Args:
        pts3d: (P, 3) float32 world-space XYZ
        **point_data: named scalar arrays to attach as PyVista point arrays.
            e.g. RGB=colors, features=feat_arr, similarity=scores
    """
    cloud = pv.PolyData(pts3d)
    for k, v in point_data.items():
        cloud[k] = v
    return cloud
```

- [ ] **Step 4: Run all three tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_visualization.py -v
```

Expected: All tests PASS (including existing `test_pca_to_rgb_*`, `test_compute_masked_image_*`, `test_overlay_masks_*`).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/visualization.py tests/test_visualization.py
git commit -m "feat(viz): add pointcloud_to_polydata utility"
```

---

## Task 2: Rewrite feedforward_exploration.ipynb

**Files:**
- Rewrite: `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb`

The notebook must be valid Jupyter JSON (nbformat 4). Build it cell-by-cell below.

- [ ] **Step 1: Write the notebook**

Write `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb` as a fresh nbformat 4 notebook with the following cells in order. Use the Write tool with the full JSON. Cell metadata: `{"trusted": true}` for code cells, `{}` for markdown cells. No outputs (empty `"outputs": []`, `"execution_count": null`).

**Cell 0 — markdown:**
```
# Feedforward Pointcloud Reconstruction

This tutorial runs **VGGT-X** and **MapAnything** on a set of keyframes extracted
from a real video, then compares the resulting pointclouds side by side.

**Prerequisite:** [Keyframe Extraction](../preprocessing/keyframe_extraction.ipynb)
— this notebook uses the same video and the same `score_all_frames` API to select frames.
```

**Cell 1 — code:**
```python
%load_ext autoreload
%autoreload 2
```

**Cell 2 — code:**
```python
import time
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
from pathlib import Path

import torch

from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import pointcloud_to_polydata, visualize_splat, create_camera_frustum_pyvista
from collab_splats.utils.frame_sampling import score_all_frames

pv.set_jupyter_backend("trame")
```

**Cell 3 — code:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
FRAME_DIR = Path("/tmp/feedforward_tutorial/frames")
FRAME_DIR.mkdir(parents=True, exist_ok=True)
```

**Cell 4 — markdown:**
```
## §1 Keyframe Extraction

Select frames from the video using optical-flow scoring (same as the keyframe extraction
tutorial). `score_all_frames` returns a per-frame dict with a `selected` boolean;
we extract only those frames to disk.
```

**Cell 5 — code:**
```python
t0 = time.time()
frame_scores = score_all_frames(VIDEO_PATH)
selected_indices = [s["frame_idx"] for s in frame_scores if s["selected"]]
print(f"Selected {len(selected_indices)} of {len(frame_scores)} frames")

cap = cv2.VideoCapture(VIDEO_PATH)
for idx in selected_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(FRAME_DIR / f"frame_{idx:05d}.jpg"), frame)
cap.release()

image_dir = FRAME_DIR
print(f"Frames extracted to {image_dir}  ({time.time() - t0:.1f}s)")
```

**Cell 6 — markdown:**
```
## §2 VGGT-X Reconstruction

VGGT-X jointly predicts camera poses and per-frame depth maps in a single forward pass.
Depth maps are unprojected to a 3D pointcloud; `conf_threshold` discards low-confidence
depth predictions.

Run each pipeline step explicitly so you can inspect intermediate state.
```

**Cell 7 — code:**
```python
t_vggt_start = time.time()

creator_vggt = VGGTXCreator()
creator_vggt.load_model(device=device)
creator_vggt.setup_inference(image_dir)
creator_vggt.run_inference()
creator_vggt.postprocess()

result_vggt = creator_vggt.outputs
t_vggt = time.time() - t_vggt_start

print(f"pts3d:      {result_vggt.pts3d.shape}  dtype={result_vggt.pts3d.dtype}")
print(f"extrinsics: {result_vggt.extrinsics.shape}")
print(f"time:       {t_vggt:.1f}s")
```

**Cell 8 — code:**
```python
# Clean via Open3D statistical outlier removal + voxel downsampling
_pcd = o3d.geometry.PointCloud()
_pcd.points = o3d.utility.Vector3dVector(result_vggt.pts3d)
_pcd.colors = o3d.utility.Vector3dVector(result_vggt.colors.astype(np.float64) / 255.0)
_cleaned, _ = clean_pointcloud(_pcd)

pts3d_vggt = np.asarray(_cleaned.points, dtype=np.float32)
colors_vggt = (np.asarray(_cleaned.colors) * 255).astype(np.uint8)

conf_vggt_mean = result_vggt.conf.mean().item() if result_vggt.conf is not None else float("nan")
conf_vggt_std  = result_vggt.conf.std().item()  if result_vggt.conf is not None else float("nan")

print(f"Points: {len(result_vggt.pts3d):,} raw → {len(pts3d_vggt):,} filtered")
print(f"Conf:   mean={conf_vggt_mean:.3f}  std={conf_vggt_std:.3f}")
```

**Cell 9 — markdown:**
```
## §3 VGGT-X Viewer

`pointcloud_to_polydata` converts the cleaned numpy arrays to a PyVista PolyData.
`visualize_splat` renders the cloud + camera frustums.
Extrinsics are world-to-camera (w2c); `np.linalg.inv` converts to camera-to-world (c2w)
for the frustum renderer.
```

**Cell 10 — code:**
```python
cloud_vggt = pointcloud_to_polydata(pts3d_vggt, RGB=colors_vggt)
c2w_vggt = [np.linalg.inv(ext) for ext in result_vggt.extrinsics]
pl = visualize_splat(cloud_vggt, aligned_cameras=c2w_vggt)
pl.show()
```

**Cell 11 — markdown:**
```
## §4 MapAnything Reconstruction

MapAnything predicts dense depth and camera poses via cross-view consistency.
Uses `confidence_percentile` to mask unreliable pixels before unprojection.
```

**Cell 12 — code:**
```python
t_map_start = time.time()

creator_map = MapAnythingCreator()
creator_map.load_model(device=device)
creator_map.setup_inference(image_dir)
creator_map.run_inference()
creator_map.postprocess()

result_map = creator_map.outputs
t_map = time.time() - t_map_start

print(f"pts3d:      {result_map.pts3d.shape}  dtype={result_map.pts3d.dtype}")
print(f"extrinsics: {result_map.extrinsics.shape}")
print(f"time:       {t_map:.1f}s")
```

**Cell 13 — code:**
```python
_pcd = o3d.geometry.PointCloud()
_pcd.points = o3d.utility.Vector3dVector(result_map.pts3d)
_pcd.colors = o3d.utility.Vector3dVector(result_map.colors.astype(np.float64) / 255.0)
_cleaned, _ = clean_pointcloud(_pcd)

pts3d_map = np.asarray(_cleaned.points, dtype=np.float32)
colors_map = (np.asarray(_cleaned.colors) * 255).astype(np.uint8)

conf_map_mean = result_map.conf.mean().item() if result_map.conf is not None else float("nan")
conf_map_std  = result_map.conf.std().item()  if result_map.conf is not None else float("nan")

print(f"Points: {len(result_map.pts3d):,} raw → {len(pts3d_map):,} filtered")
print(f"Conf:   mean={conf_map_mean:.3f}  std={conf_map_std:.3f}")
```

**Cell 14 — markdown:**
```
## §5 MapAnything Viewer
```

**Cell 15 — code:**
```python
cloud_map = pointcloud_to_polydata(pts3d_map, RGB=colors_map)
c2w_map = [np.linalg.inv(ext) for ext in result_map.extrinsics]
pl = visualize_splat(cloud_map, aligned_cameras=c2w_map)
pl.show()
```

**Cell 16 — markdown:**
```
## §6 Comparison

### Stats
```

**Cell 17 — code:**
```python
col = 14
print(f"{'Model':<{col}} {'Pts raw':>10} {'Pts filt':>10} {'Conf mean':>10} {'Conf std':>10} {'Time (s)':>10}")
print("-" * (col + 52))
print(f"{'VGGT-X':<{col}} {len(result_vggt.pts3d):>10,} {len(pts3d_vggt):>10,} {conf_vggt_mean:>10.3f} {conf_vggt_std:>10.3f} {t_vggt:>10.1f}")
print(f"{'MapAnything':<{col}} {len(result_map.pts3d):>10,} {len(pts3d_map):>10,} {conf_map_mean:>10.3f} {conf_map_std:>10.3f} {t_map:>10.1f}")
```

**Cell 18 — markdown:**
```
### Camera Trajectory Overlay

Both sets of camera frustums in a single scene — **blue** = VGGT-X, **orange** = MapAnything.
Overlap indicates pose agreement; divergence reveals where the two models disagree on camera
placement.
```

**Cell 19 — code:**
```python
pl = pv.Plotter()
for ext in result_vggt.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="cornflowerblue", line_width=2)
for ext in result_map.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="darkorange", line_width=2)
pl.add_axes()
pl.show()
```

**Cell 20 — markdown:**
```
## §7 When To Use Each

| | VGGT-X | MapAnything |
|---|---|---|
| **Strength** | Dense, high-detail clouds | Fast inference, lighter model |
| **Best for** | Scene reconstruction, BA input | Quick pose estimates, previews |
| **After this** | Add `BundleAdjustment` wrapper for track-refined poses | Same — BA improves both |
| **Memory** | Higher (VGGT-1B) | Lower |

Both models write disk-compatible COLMAP output via `build_colmap()` if you need downstream
tools. Feed either result into `BundleAdjustmentWrapper` (see the bundle adjustment tutorial)
for refined poses and denser tracks.
```

- [ ] **Step 2: Validate the notebook parses as JSON**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json, sys
nb = json.load(open('docs/source/tutorials/pointcloud/feedforward_exploration.ipynb'))
print(f'nbformat: {nb[\"nbformat\"]}.{nb[\"nbformat_minor\"]}')
print(f'cells: {len(nb[\"cells\"])}')
for i, c in enumerate(nb['cells']):
    print(f'  [{i}] {c[\"cell_type\"]}')
"
```

Expected: prints 21 cells (0-20) with correct types.

- [ ] **Step 3: Verify imports are resolvable (dry-run)**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import pointcloud_to_polydata, visualize_splat, create_camera_frustum_pyvista
from collab_splats.utils.frame_sampling import score_all_frames
print('all imports OK')
"
```

Expected: `all imports OK`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/pointcloud/feedforward_exploration.ipynb
git commit -m "docs(tutorials): rewrite feedforward_exploration notebook with model comparison"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - ✅ `pointcloud_to_polydata(**point_data)` — Task 1
  - ✅ Keyframe extraction light coupling (`score_all_frames` + extract) — Cell 5
  - ✅ VGGT-X step-by-step + timing — Cells 7-8
  - ✅ VGGT-X PyVista viewer — Cell 10
  - ✅ MapAnything step-by-step + timing — Cells 12-13
  - ✅ MapAnything PyVista viewer — Cell 15
  - ✅ Stats table (pts raw/filtered, conf mean/std, time) — Cell 17
  - ✅ Camera overlay viewer (blue/orange) — Cell 19
  - ✅ When to use each markdown — Cell 20
  - ✅ `index.rst` already correct — no task needed
- [x] **No placeholders** — all code is complete
- [x] **Type consistency** — `result_vggt.outputs` → `FeedforwardResult`; `result.conf` accessed as `torch.Tensor | None`; `result.extrinsics` is `(N, 4, 4)` w2c throughout
