# Pointcloud Feedforward Exploration Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `docs/pointcloud/feedforward_exploration.ipynb` — an interactive notebook for running VGGTXCreator and MapAnythingCreator, cleaning the output pointcloud, and visualizing it with PyVista.

**Architecture:** Single notebook with a dataset config cell at top (`DATASET = "bicycle"` or `"c0043"`), a notebook-local PyVista visualization helper, then two method sections mirroring the pattern of `docs/semantics/feature_extraction.ipynb`. No library code is modified.

**Tech Stack:** PyVista (trame backend), Open3D, `collab_splats.pointcloud.feedforward` (VGGTXCreator/MapAnythingCreator), `collab_splats.pointcloud.utils.clean_pointcloud`, `collab_splats.utils.visualization.create_camera_frustum_pyvista`, `collab_splats.utils.frame_sampling.sample_frames_fps`

---

### Task 1: Write design spec

**Files:**
- Create: `worklog/history/specs/2026-04-22-pointcloud-feedforward-notebook-design.md`

- [ ] **Step 1: Verify file exists**

```bash
ls worklog/history/specs/2026-04-22-pointcloud-feedforward-notebook-design.md
```

Expected: file listed.

- [ ] **Step 2: Commit**

```bash
git add worklog/history/specs/2026-04-22-pointcloud-feedforward-notebook-design.md
git commit -m "docs(pointcloud): add feedforward exploration notebook design spec"
```

---

### Task 2: Create `docs/pointcloud/feedforward_exploration.ipynb`

**Files:**
- Create: `docs/pointcloud/feedforward_exploration.ipynb`

The notebook is a JSON `.ipynb` file. Write it with the following cell sequence.

- [ ] **Step 1: Verify import chain works**

```bash
cd /workspace/collab-splats && python -c "
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import create_camera_frustum_pyvista
from collab_splats.utils.frame_sampling import sample_frames_fps
print('imports ok')
"
```

Expected: `imports ok`

- [ ] **Step 2: Write the notebook file**

Write `docs/pointcloud/feedforward_exploration.ipynb` as a valid `.ipynb` JSON with the cells below (see Cell Content section).

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/feedforward_exploration.ipynb
git commit -m "feat(pointcloud): add feedforward exploration notebook"
```

---

## Cell Content

### Cell 0 — markdown title

```markdown
# Pointcloud Feedforward Exploration

Runs feedforward pointcloud reconstruction via **VGGTXCreator** and **MapAnythingCreator**,
cleans the output with `clean_pointcloud`, and renders an interactive PyVista viewer.

Set `DATASET` in §1 to switch between bicycle images and C0043 video frames.
```

### Cell 1 — autoreload

```python
%load_ext autoreload
%autoreload 2
```

### Cell 2 — imports

```python
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
from pathlib import Path

import torch

from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.utils.visualization import create_camera_frustum_pyvista
from collab_splats.utils.frame_sampling import sample_frames_fps

pv.set_jupyter_backend("trame")
```

### Cell 3 — device

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
```

### Cell 4 — visualization helper

```python
def visualize_feedforward_result(
    pts3d,
    colors,
    extrinsics,
    intrinsics,
    point_size=2.0,
    camera_scale=0.05,
    show_cameras=True,
):
    """Render pointcloud + camera frustums with PyVista.

    Args:
        pts3d:      (P, 3) float32 world-space XYZ.
        colors:     (P, 3) uint8 RGB.
        extrinsics: (N, 3, 4) world-to-camera [R|t].
        intrinsics: (N, 3, 3) camera K matrices.
    """
    pl = pv.Plotter()

    cloud = pv.PolyData(pts3d)
    cloud["RGB"] = colors
    pl.add_mesh(cloud, scalars="RGB", rgb=True, point_size=point_size, render_points_as_spheres=False)

    if show_cameras:
        for ext, K in zip(extrinsics, intrinsics):
            R, t = ext[:3, :3], ext[:3, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t

            fx, fy = K[0, 0], K[1, 1]
            aspect_ratio = fx / fy if fy > 0 else 1.33

            frustum = create_camera_frustum_pyvista(c2w, scale=camera_scale, aspect_ratio=aspect_ratio)
            pl.add_mesh(frustum, color="red", line_width=1)

    pl.add_axes()
    return pl
```

### Cell 5 — markdown §1

```markdown
## §1 Data Configuration

Set `DATASET = "bicycle"` to use pre-extracted bicycle images, or `"c0043"` to extract
frames from the C0043 bird-field video at 1 FPS.
```

### Cell 6 — dataset config

```python
DATASET = "bicycle"  # "bicycle" or "c0043"

BICYCLE_IMAGE_DIR = Path("/workspace/bicycle/images_4")
C0043_VIDEO_PATH = "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
C0043_FRAME_DIR = Path("/tmp/feedforward_exploration/c0043_frames")
```

### Cell 7 — dataset routing

```python
if DATASET == "bicycle":
    image_dir = BICYCLE_IMAGE_DIR
    n_images = len([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"bicycle: {n_images} images from {image_dir}")

elif DATASET == "c0043":
    C0043_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    frames = sample_frames_fps(C0043_VIDEO_PATH, fps=1.0)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(C0043_FRAME_DIR / f"frame_{i:04d}.jpg"), frame)
    image_dir = C0043_FRAME_DIR
    print(f"c0043: extracted {len(frames)} frames → {image_dir}")

else:
    raise ValueError(f"Unknown DATASET: {DATASET!r}. Use 'bicycle' or 'c0043'.")

print(f"image_dir = {image_dir}")
```

### Cell 8 — markdown §2

```markdown
## §2 VGGTXCreator

VGGT-X jointly predicts camera poses and per-frame depth maps in a single forward pass.
Depth maps are unprojected to a 3D point cloud. Uses `conf_threshold` to discard
low-confidence depth predictions.
```

### Cell 9 — VGGT inference

```python
creator_vggt = VGGTXCreator()
creator_vggt.load_model()
creator_vggt.setup_inference(image_dir)
creator_vggt.run_inference()
creator_vggt.postprocess()
```

### Cell 10 — VGGT inspect

```python
result_vggt = creator_vggt.outputs
print(f"pts3d:      {result_vggt.pts3d.shape}  dtype={result_vggt.pts3d.dtype}")
print(f"colors:     {result_vggt.colors.shape}  dtype={result_vggt.colors.dtype}")
print(f"extrinsics: {result_vggt.extrinsics.shape}")
print(f"intrinsics: {result_vggt.intrinsics.shape}")
print(f"images:     {len(result_vggt.image_paths)}")
```

### Cell 11 — VGGT clean

```python
_pcd = o3d.geometry.PointCloud()
_pcd.points = o3d.utility.Vector3dVector(result_vggt.pts3d)
_pcd.colors = o3d.utility.Vector3dVector(result_vggt.colors.astype(np.float64) / 255.0)
_cleaned, _ = clean_pointcloud(_pcd)

pts3d_vggt = np.asarray(_cleaned.points, dtype=np.float32)
colors_vggt = (np.asarray(_cleaned.colors) * 255).astype(np.uint8)
print(f"Points: {len(result_vggt.pts3d):,} → {len(pts3d_vggt):,} after cleaning")
```

### Cell 12 — VGGT visualize

```python
pl = visualize_feedforward_result(
    pts3d_vggt, colors_vggt,
    result_vggt.extrinsics, result_vggt.intrinsics,
)
pl.show()
```

### Cell 13 — markdown §3

```markdown
## §3 MapAnythingCreator

MapAnything jointly predicts dense depth and camera poses via cross-view consistency.
Uses `confidence_percentile` to mask unreliable pixels before unprojection.
```

### Cell 14 — MapAnything inference

```python
creator_map = MapAnythingCreator()
creator_map.load_model()
creator_map.setup_inference(image_dir)
creator_map.run_inference()
creator_map.postprocess()
```

### Cell 15 — MapAnything inspect

```python
result_map = creator_map.outputs
print(f"pts3d:      {result_map.pts3d.shape}  dtype={result_map.pts3d.dtype}")
print(f"colors:     {result_map.colors.shape}  dtype={result_map.colors.dtype}")
print(f"extrinsics: {result_map.extrinsics.shape}")
print(f"intrinsics: {result_map.intrinsics.shape}")
print(f"images:     {len(result_map.image_paths)}")
```

### Cell 16 — MapAnything clean

```python
_pcd = o3d.geometry.PointCloud()
_pcd.points = o3d.utility.Vector3dVector(result_map.pts3d)
_pcd.colors = o3d.utility.Vector3dVector(result_map.colors.astype(np.float64) / 255.0)
_cleaned, _ = clean_pointcloud(_pcd)

pts3d_map = np.asarray(_cleaned.points, dtype=np.float32)
colors_map = (np.asarray(_cleaned.colors) * 255).astype(np.uint8)
print(f"Points: {len(result_map.pts3d):,} → {len(pts3d_map):,} after cleaning")
```

### Cell 17 — MapAnything visualize

```python
pl = visualize_feedforward_result(
    pts3d_map, colors_map,
    result_map.extrinsics, result_map.intrinsics,
)
pl.show()
```
