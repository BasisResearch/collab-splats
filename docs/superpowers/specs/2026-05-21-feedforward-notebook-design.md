# Feedforward Tutorial Notebook Design

**Date:** 2026-05-21
**Status:** Approved
**Branch:** refactor/core-modules

## Goal

Replace `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb` with a clean,
tutorial-style notebook that:
- Builds on the keyframe extraction tutorial (same video, same API)
- Runs both VGGTXCreator and MapAnythingCreator
- Compares results with stats, timing, and a camera overlay viewer

## Package Change

**`collab_splats/utils/visualization.py`** — add `pointcloud_to_polydata`:

```python
def pointcloud_to_polydata(pts3d: np.ndarray, **point_data) -> pv.PolyData:
    """Convert pts3d + named scalar arrays to a PyVista PolyData.

    Args:
        pts3d: (P, 3) float32 world-space XYZ
        **point_data: named scalar arrays to attach (e.g. RGB=colors, features=feat_arr)
    """
    cloud = pv.PolyData(pts3d)
    for k, v in point_data.items():
        cloud[k] = v
    return cloud
```

Export via `__all__`. Composes with existing `visualize_splat()` — no changes to that function.

## Notebook Structure

**Path:** `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb`
**Replaces:** existing notebook of the same name

### Title + Intro
Markdown: what feedforward reconstruction is, pointer to keyframe extraction notebook as prerequisite.

### Imports
```python
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
from collab_splats.utils.visualization import pointcloud_to_polydata, visualize_splat
from collab_splats.utils.frame_sampling import score_all_frames
```

### §0 Setup
- Device detection (`cuda` / `cpu`)
- Video path + output dirs (`/tmp/feedforward_tutorial/`)

### §1 Keyframe Extraction
Single code cell: `score_all_frames(video_path)` + threshold → extract selected frames to
tmp dir. Mirrors the API from the keyframe extraction notebook without re-explaining it.
Brief markdown note: "Using optical-flow keyframes from the prior tutorial."

### §2 VGGT-X Reconstruction
- Instantiate `VGGTXCreator` (default `model_name`)
- Timed `reconstruct(image_dir, output_dir)`
- Print: N raw points, N after `clean_pointcloud`, mean/std confidence, wall-clock time

### §3 VGGT-X Viewer
```python
cloud = pointcloud_to_polydata(result_vggt.pts3d, RGB=result_vggt.colors)
c2w = [np.linalg.inv(ext) for ext in result_vggt.extrinsics]
pl = visualize_splat(cloud, aligned_cameras=c2w)
pl.show()
```

### §4 MapAnything Reconstruction
Same pattern as §2 using `MapAnythingCreator`.

### §5 MapAnything Viewer
Same pattern as §3 using `result_map`.

### §6 Comparison
Two parts:

**Stats table** (printed via pandas or plain print):
| Model | Points (raw) | Points (filtered) | Conf mean | Conf std | Time (s) |
|---|---|---|---|---|---|
| VGGT-X | ... | ... | ... | ... | ... |
| MapAnything | ... | ... | ... | ... | ... |

**Camera overlay viewer** — single PyVista scene, no pointcloud, just frustums from both
models in different colors (VGGT-X blue, MapAnything orange). Shows pose agreement/disagreement.

```python
pl = pv.Plotter()
for ext in result_vggt.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
    pl.add_mesh(frustum, color="blue")
for ext in result_map.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
    pl.add_mesh(frustum, color="orange")
pl.show()
```

### §7 When To Use Each
Markdown commentary covering:
- **VGGT-X**: higher point density, slower, better for dense reconstructions
- **MapAnything**: faster inference, lighter model, good for quick scene understanding
- Guidance on when to run BA after each

## Visualization Strategy

`visualize_splat()` (existing) handles rendering. `pointcloud_to_polydata()` (new) bridges
raw arrays to PyVista. No inline helper functions in the notebook.

Extrinsics inversion (`w2c → c2w`) stays inline as a one-liner — visible in the notebook
because the pedagogical value of seeing the transform is worth the one line.

## Files Changed

| File | Change |
|---|---|
| `collab_splats/utils/visualization.py` | Add `pointcloud_to_polydata` |
| `docs/source/tutorials/pointcloud/feedforward_exploration.ipynb` | Full rewrite |
| `docs/source/tutorials/index.rst` | Verify pointcloud section exists |
