# Feedforward Viz Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix camera frustum geometry bugs (wrong direction, missing edges, wrong API) and modernize `feedforward_methods.ipynb` with kwargs, a `_clean` helper, and corrected section numbering.

**Architecture:** Two files. Fix `create_camera_frustum_pyvista` in `visualization.py` first (TDD), then update the notebook to use the corrected function and new kwargs. The frustum fix is the root cause of the visual bugs; the notebook changes are mechanical rewrites.

**Tech Stack:** PyVista, NumPy, Jupyter notebook JSON (nbformat), pytest

---

## File Map

| Action | File |
|--------|------|
| Modify | `collab_splats/utils/visualization.py` |
| Create | `tests/utils/test_visualization.py` |
| Modify | `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` |

---

### Task 1: Fix `create_camera_frustum_pyvista`

**Files:**
- Modify: `collab_splats/utils/visualization.py:335-402`
- Create: `tests/utils/test_visualization.py`

**Context:** The function currently takes a pose in OpenGL c2w convention (camera looks down -Z), so near/far planes are at negative Z. Our feedforward pipeline uses OpenCV w2c convention. Fix: accept w2c, invert internally, use positive Z for near/far planes, close the near/far rectangles.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_visualization.py`:

```python
import numpy as np
import pytest
from collab_splats.utils.visualization import create_camera_frustum_pyvista


def test_frustum_apex_at_origin_for_identity_w2c():
    """Identity w2c → camera at world origin → apex at [0, 0, 0]."""
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    assert np.allclose(frustum.points[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_frustum_apex_at_camera_position():
    """Camera translated to [1, 2, 3] → apex at [1, 2, 3] in world space."""
    # w2c with camera at world [1, 2, 3]: R=I, t = -R @ p = [-1, -2, -3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 3] = [-1.0, -2.0, -3.0]
    frustum = create_camera_frustum_pyvista(w2c)
    assert np.allclose(frustum.points[0], [1.0, 2.0, 3.0], atol=1e-5)


def test_frustum_near_plane_at_positive_z_for_identity():
    """OpenCV convention: identity w2c → camera looks +Z → near/far at +Z."""
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    # Vertices 1-4: near plane; 5-8: far plane
    assert np.all(frustum.points[1:5, 2] > 0), "near plane z must be positive"
    assert np.all(frustum.points[5:9, 2] > 0), "far plane z must be positive"


def test_frustum_rectangles_closed():
    """Near and far plane rectangles must be closed (48 total line array entries)."""
    # Closed rects: near=[5,1,2,3,4,1] + far=[5,5,6,7,8,5] = 12 entries total
    # Unclosed rects: near=[4,1,2,3,4] + far=[4,5,6,7,8] = 10 entries total
    # Other lines (apex→near×4, apex→far×4, near→far×4) = 36 entries
    # Total closed = 48, unclosed = 46
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    assert len(frustum.lines) == 48, (
        f"Expected 48 line array entries (closed rects), got {len(frustum.lines)}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_visualization.py -v
```

Expected: 4 FAIL (function exists but has wrong behaviour).

- [ ] **Step 3: Fix `create_camera_frustum_pyvista` in `visualization.py`**

Replace the entire function (lines 335–402) with:

```python
def create_camera_frustum_pyvista(pose, scale=0.02, aspect_ratio=1.33, fov=60):
    """Create a camera frustum wireframe in world space.

    Args:
        pose: (4, 4) float32 world-to-camera matrix (OpenCV convention).
            Matches FeedforwardResult.extrinsics[i] directly — no inversion needed.
        scale: Controls overall frustum size (near = scale*0.1, far = scale*5).
        aspect_ratio: Width / height of the image plane.
        fov: Vertical field of view in degrees.
    """
    fov_rad = np.radians(fov)
    near = scale * 0.1
    far = scale * 5.0

    near_height = 2 * near * np.tan(fov_rad / 2)
    near_width = near_height * aspect_ratio
    far_height = 2 * far * np.tan(fov_rad / 2)
    far_width = far_height * aspect_ratio

    # Frustum in camera space: apex at origin, camera looks in +Z (OpenCV)
    vertices = np.array(
        [
            # Apex (camera centre)
            [0, 0, 0],
            # Near plane corners (+Z)
            [-near_width / 2, -near_height / 2, near],
            [ near_width / 2, -near_height / 2, near],
            [ near_width / 2,  near_height / 2, near],
            [-near_width / 2,  near_height / 2, near],
            # Far plane corners (+Z)
            [-far_width / 2, -far_height / 2, far],
            [ far_width / 2, -far_height / 2, far],
            [ far_width / 2,  far_height / 2, far],
            [-far_width / 2,  far_height / 2, far],
        ],
        dtype=np.float64,
    )

    lines = []
    # Apex → near corners
    for i in range(1, 5):
        lines.extend([2, 0, i])
    # Apex → far corners
    for i in range(5, 9):
        lines.extend([2, 0, i])
    # Near plane rectangle (closed)
    lines.extend([5, 1, 2, 3, 4, 1])
    # Far plane rectangle (closed)
    lines.extend([5, 5, 6, 7, 8, 5])
    # Near → far edges
    for i in range(4):
        lines.extend([2, i + 1, i + 5])

    frustum = pv.PolyData(vertices, lines=lines)

    # Transform camera-space vertices to world space via c2w = inv(w2c)
    c2w = np.linalg.inv(pose)
    pts_h = np.column_stack([frustum.points, np.ones(len(frustum.points))])
    frustum.points = (c2w @ pts_h.T).T[:, :3]

    return frustum
```

Also update the `visualize_splat` docstring line 269 from:
```
        aligned_cameras: List of 4x4 world-to-camera pose matrices.
```
to:
```
        aligned_cameras: List of (4,4) world-to-camera matrices (OpenCV convention,
            e.g. FeedforwardResult.extrinsics). Passed directly to
            create_camera_frustum_pyvista, which inverts internally.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_visualization.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/utils/visualization.py tests/utils/test_visualization.py
git commit -m "fix(visualization): create_camera_frustum_pyvista takes w2c OpenCV, +Z frustum, closed rects"
```

---

### Task 2: Update notebook imports and add `_clean` helper

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` (cell-2, cell-3)

**Context:** The imports cell needs `CAMERA_KWARGS, PCD_KWARGS, VIZ_KWARGS`. The config cell needs a `_clean(result)` helper that condenses the 12-line post-processing block repeated 3 times.

- [ ] **Step 1: Update imports cell (cell-2)**

Use `NotebookEdit` to replace cell-2 source with:

```python
import os
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import open3d as o3d
import pyvista as pv
import torch
%matplotlib inline

# pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")
pv.set_jupyter_backend("trame")

from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import (
    CAMERA_KWARGS,
    PCD_KWARGS,
    VIZ_KWARGS,
    create_camera_frustum_pyvista,
    pointcloud_to_polydata,
    visualize_splat,
)
```

- [ ] **Step 2: Update config cell (cell-3) — append `_clean` helper**

Use `NotebookEdit` to replace cell-3 source with:

```python
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────

assert IMAGES.exists() and any(IMAGES.glob("*.jpg")), (
    f"No images found in {IMAGES}. Run 01_preprocessing/keyframe_extraction first."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}  |  images: {len(list(IMAGES.glob('*.jpg')))}")


def _clean(result):
    """Filter outliers, downsample; return (pts3d, colors, conf_mean, conf_std)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result.points)
    pcd.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64) / 255.0)
    cleaned, _ = clean_pointcloud(pcd)
    pts3d = np.asarray(cleaned.points, dtype=np.float32)
    colors = (np.asarray(cleaned.colors) * 255).astype(np.uint8)
    conf = result.confidence
    conf_mean = conf.cpu().float().mean().item() if conf is not None else float("nan")
    conf_std  = conf.cpu().float().std().item()  if conf is not None else float("nan")
    print(f"Points: {len(result.points):,} raw → {len(pts3d):,} filtered")
    print(f"Conf:   mean={conf_mean:.3f}  std={conf_std:.3f}")
    return pts3d, colors, conf_mean, conf_std
```

- [ ] **Step 3: Verify JSON is valid**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
src2 = ''.join(nb['cells'][2]['source'])
assert 'CAMERA_KWARGS' in src2, 'CAMERA_KWARGS missing from imports'
assert 'PCD_KWARGS' in src2, 'PCD_KWARGS missing from imports'
src3 = ''.join(nb['cells'][3]['source'])
assert 'def _clean' in src3, '_clean helper missing'
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): add CAMERA/PCD/VIZ_KWARGS imports + _clean helper"
```

---

### Task 3: Condense §2, §5, §10 post-processing cells

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` (cells f4758d25, cell-9, cell-omega-postproc)

- [ ] **Step 1: Replace §2 post-proc cell (id: f4758d25)**

Use `NotebookEdit` with `cell_id="f4758d25"`:

```python
pts3d_vggt, colors_vggt, conf_vggt_mean, conf_vggt_std = _clean(result_vggt)
```

- [ ] **Step 2: Replace §5 post-proc cell (id: cell-9)**

Use `NotebookEdit` with `cell_id="cell-9"`:

```python
pts3d_ma, colors_ma, conf_ma_mean, conf_ma_std = _clean(result_ma)
```

- [ ] **Step 3: Replace §10 post-proc cell (id: cell-omega-postproc)**

Use `NotebookEdit` with `cell_id="cell-omega-postproc"`:

```python
pts3d_omega, colors_omega, conf_omega_mean, conf_omega_std = _clean(result_omega)
```

- [ ] **Step 4: Verify**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
for cid, var in [('f4758d25','pts3d_vggt'), ('cell-9','pts3d_ma'), ('cell-omega-postproc','pts3d_omega')]:
    cell = next(c for c in nb['cells'] if c.get('id') == cid)
    src = ''.join(cell['source'])
    assert var in src, f'{var} missing from {cid}'
    assert len(src.strip().splitlines()) == 1, f'{cid} should be 1 line'
print('OK — all 3 post-proc cells condensed')
"
```

Expected: `OK — all 3 post-proc cells condensed`

- [ ] **Step 5: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): condense §2/§5/§10 post-proc to _clean() one-liners"
```

---

### Task 4: Update §3, §6, §11 viewer cells with kwargs

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` (cells cell-6, 69212a84, cell-omega-viewer)

**Context:** Replace the bare `visualize_splat(cloud, aligned_cameras=c2w)` calls with full kwargs. Pass `result.extrinsics` directly (w2c) — no more `np.linalg.inv`. Use viridis colormap by omitting `"color"` from `CAMERA_KWARGS`.

- [ ] **Step 1: Replace §3 VGGT-X viewer cell (id: cell-6)**

Use `NotebookEdit` with `cell_id="cell-6"`:

```python
cloud_vggt = pointcloud_to_polydata(pts3d_vggt, RGB=colors_vggt)
visualize_splat(
    cloud_vggt,
    aligned_cameras=list(result_vggt.extrinsics),
    mesh_kwargs={**PCD_KWARGS, "rgb": True},
    camera_kwargs={k: v for k, v in CAMERA_KWARGS.items() if k != "color"},
    viz_kwargs=VIZ_KWARGS,
).show()
```

- [ ] **Step 2: Replace §6 MapAnything viewer cell (id: 69212a84)**

Use `NotebookEdit` with `cell_id="69212a84"`:

```python
cloud_ma = pointcloud_to_polydata(pts3d_ma, RGB=colors_ma)
visualize_splat(
    cloud_ma,
    aligned_cameras=list(result_ma.extrinsics),
    mesh_kwargs={**PCD_KWARGS, "rgb": True},
    camera_kwargs={k: v for k, v in CAMERA_KWARGS.items() if k != "color"},
    viz_kwargs=VIZ_KWARGS,
).show()
```

- [ ] **Step 3: Replace §11 VGGT-Omega viewer cell (id: cell-omega-viewer)**

Use `NotebookEdit` with `cell_id="cell-omega-viewer"`:

```python
cloud_omega = pointcloud_to_polydata(pts3d_omega, RGB=colors_omega)
visualize_splat(
    cloud_omega,
    aligned_cameras=list(result_omega.extrinsics),
    mesh_kwargs={**PCD_KWARGS, "rgb": True},
    camera_kwargs={k: v for k, v in CAMERA_KWARGS.items() if k != "color"},
    viz_kwargs=VIZ_KWARGS,
).show()
```

- [ ] **Step 4: Verify**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
for cid in ('cell-6', '69212a84', 'cell-omega-viewer'):
    cell = next(c for c in nb['cells'] if c.get('id') == cid)
    src = ''.join(cell['source'])
    assert 'PCD_KWARGS' in src, f'PCD_KWARGS missing in {cid}'
    assert 'VIZ_KWARGS' in src, f'VIZ_KWARGS missing in {cid}'
    assert 'CAMERA_KWARGS' in src, f'CAMERA_KWARGS missing in {cid}'
    assert 'np.linalg.inv' not in src, f'np.linalg.inv should be gone in {cid}'
print('OK — all 3 viewer cells updated')
"
```

Expected: `OK — all 3 viewer cells updated`

- [ ] **Step 5: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): update §3/§6/§11 viewers to use PCD/VIZ/CAMERA_KWARGS"
```

---

### Task 5: Update §13 camera overlay cell

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` (cell-12)

**Context:** Remove `np.linalg.inv` calls (pass `extrinsics` directly). Apply `VIZ_KWARGS` to the plotter camera position. Keep per-model solid colors since this is a 3-model comparison overlay.

- [ ] **Step 1: Replace §13 overlay cell (id: cell-12)**

Use `NotebookEdit` with `cell_id="cell-12"`:

```python
pl = pv.Plotter()
for ext in result_vggt.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(ext, scale=0.05), color="cornflowerblue", line_width=2)
for ext in result_ma.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(ext, scale=0.05), color="darkorange", line_width=2)
for ext in result_omega.extrinsics:
    pl.add_mesh(create_camera_frustum_pyvista(ext, scale=0.05), color="mediumseagreen", line_width=2)
pl.camera_position = [
    VIZ_KWARGS["position"], VIZ_KWARGS["focal_point"], VIZ_KWARGS["view_up"]
]
pl.camera.azimuth = VIZ_KWARGS["azimuth"]
pl.camera.elevation = VIZ_KWARGS["elevation"]
pl.camera.Zoom(VIZ_KWARGS["zoom"])
pl.add_axes()
pl.show()
```

- [ ] **Step 2: Verify**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
cell = next(c for c in nb['cells'] if c.get('id') == 'cell-12')
src = ''.join(cell['source'])
assert 'np.linalg.inv' not in src, 'inv should be gone'
assert 'VIZ_KWARGS' in src, 'VIZ_KWARGS missing'
assert 'mediumseagreen' in src, 'omega color missing'
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): update §13 overlay to use extrinsics directly + VIZ_KWARGS"
```

---

### Task 6: Renumber sections §9-§13 → §7-§11

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` (5 markdown cells)

**Context:** Sections jump from §6 to §9 because §7-§8 were consumed during a previous restructure. Fix to sequential §7-§11. Cell IDs are unchanged.

Renaming map:
- `cell-md-s9` (§9 Omega Reconstruction) → §7
- `cell-md-s10` (§10 Omega Post-processing) → §8
- `cell-md-s11` (§11 Omega Viewer) → §9
- `cell-md-s7` (§12 Side-by-side) → §10
- `cell-md-s8` (§13 Camera Overlay) → §11

- [ ] **Step 1: Renumber `cell-md-s9` to §7**

Use `NotebookEdit` with `cell_id="cell-md-s9"`, `cell_type="markdown"`:

```
## §7 — VGGT-Omega Reconstruction

Loads a cached VGGT-Omega reconstruction from zarr if available, skipping GPU inference (~3–5 min on GPU). VGGT-Omega is a third feedforward method that uses a unified vision transformer for depth and pose estimation. Runs with default settings (`VGGTOmegaCreator()`).
```

- [ ] **Step 2: Renumber `cell-md-s10` to §8**

Use `NotebookEdit` with `cell_id="cell-md-s10"`, `cell_type="markdown"`:

```
## §8 — VGGT-Omega Post-processing and Visualisation

Applies the same outlier removal and voxel downsampling pipeline as §2/§5. Confidence metrics are collected for the three-way comparison table.
```

- [ ] **Step 3: Renumber `cell-md-s11` to §9**

Use `NotebookEdit` with `cell_id="cell-md-s11"`, `cell_type="markdown"`:

```
## §9 — VGGT-Omega Pointcloud Viewer

Renders the filtered VGGT-Omega pointcloud with camera frustums. Camera frustums show predicted camera poses; compare pose spread vs. VGGT-X (§3) and MapAnything (§6) above.
```

- [ ] **Step 4: Renumber `cell-md-s7` to §10**

Use `NotebookEdit` with `cell_id="cell-md-s7"`, `cell_type="markdown"`:

```
## §10 — Side-by-side Comparison

Tabulates raw vs. filtered point counts and confidence statistics for all three models. Higher confidence mean with lower std indicates more reliable depth predictions.
```

- [ ] **Step 5: Renumber `cell-md-s8` to §11**

Use `NotebookEdit` with `cell_id="cell-md-s8"`, `cell_type="markdown"`:

```
## §11 — Camera Pose Overlay

Overlays camera frustums from all three models in a single scene — blue for VGGT-X, orange for MapAnything, green for VGGT-Omega. Alignment across the three sets indicates consistent global pose estimation.
```

- [ ] **Step 6: Verify section sequence**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json, re
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
sections = []
for c in nb['cells']:
    if c['cell_type'] == 'markdown':
        m = re.search(r'^## (§\d+)', ''.join(c['source']))
        if m:
            sections.append(int(m.group(1)[1:]))
print('Section numbers:', sections)
assert sections == list(range(1, len(sections)+1)), 'Sections not sequential!'
print('OK — sequential section numbering confirmed')
"
```

Expected:
```
Section numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
OK — sequential section numbering confirmed
```

- [ ] **Step 7: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): renumber sections to sequential §1-§11"
```
