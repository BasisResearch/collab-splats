# Design: Feedforward Notebook Viz Fixes

**Date:** 2026-05-26  
**Status:** approved

## Summary

Fix three visual bugs in `feedforward_methods.ipynb` (wrong camera direction, missing rectangle sides, no kwargs) and reduce repeated post-processing boilerplate. Root cause: `create_camera_frustum_pyvista` assumed OpenGL c2w convention; feedforward pipeline uses OpenCV w2c. Fix the utility to take w2c (our canonical type) and invert internally.

## Scope

Two files:
- `collab_splats/utils/visualization.py` — fix frustum geometry + update API
- `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` — use kwargs + add helper + clean up cells

No other notebooks or library code touched. BA notebook gets the frustum fix for free (it already passes w2c correctly).

---

## Component 1: `visualization.py`

### `create_camera_frustum_pyvista(pose, scale, aspect_ratio, fov)`

**Current:** takes c2w (OpenGL convention), frustum points in -Z, near/far rectangles unclosed.

**Changes:**
1. Accept `pose` as **w2c (world-to-camera, OpenCV)** — matches `FeedforwardResult.extrinsics`
2. Invert internally: `c2w = invert_poses(pose[np.newaxis])[0]` then apply c2w to vertices
3. Change near/far plane z-values from `-near`/`-far` to `+near`/`+far` (OpenCV: camera looks in +Z)
4. Close near rectangle: `[4, 1, 2, 3, 4]` → `[5, 1, 2, 3, 4, 1]`
5. Close far rectangle: `[4, 5, 6, 7, 8]` → `[5, 5, 6, 7, 8, 5]`

Update docstring: "pose: (4,4) float32 world-to-camera matrix (OpenCV convention, e.g. FeedforwardResult.extrinsics[i])."

### `visualize_splat(..., aligned_cameras, ...)`

Update docstring: `aligned_cameras` takes a list of **w2c** (4×4) matrices. No code change to the loop — it passes each pose directly to `create_camera_frustum_pyvista`, which now handles inversion.

---

## Component 2: `feedforward_methods.ipynb`

### Imports cell (cell-2)

Add to import line:
```python
from collab_splats.utils.visualization import (
    CAMERA_KWARGS, PCD_KWARGS, VIZ_KWARGS,
    pointcloud_to_polydata, visualize_splat, create_camera_frustum_pyvista,
)
```

### Setup cell (cell-3) — add `_clean` helper

Append to the existing config cell:
```python
def _clean(result):
    """Filter outliers, downsample, return (pts3d, colors, conf_mean, conf_std)."""
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

### §2 post-proc cell (f4758d25) — replace entire source

```python
pts3d_vggt, colors_vggt, conf_vggt_mean, conf_vggt_std = _clean(result_vggt)
```

### §5 post-proc cell (cell-9) — replace entire source

```python
pts3d_ma, colors_ma, conf_ma_mean, conf_ma_std = _clean(result_ma)
```

### §10 post-proc cell (cell-omega-postproc) — replace entire source

```python
pts3d_omega, colors_omega, conf_omega_mean, conf_omega_std = _clean(result_omega)
```

### §3 viewer cell (cell-6) — replace entire source

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

(No `color` in camera_kwargs → viridis colormap applied per camera.)

### §6 viewer cell (69212a84) — replace entire source

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

### §11 viewer cell (cell-omega-viewer) — replace entire source

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

### §13 overlay cell (cell-12) — replace entire source

Uses w2c directly (no `np.linalg.inv`). Applies VIZ_KWARGS for camera position. Keeps per-model colors since this is a comparison overlay.

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

---

## Convention contract (post-fix)

| Function | Input | Notes |
|---|---|---|
| `create_camera_frustum_pyvista(pose)` | w2c (OpenCV) | inverts internally |
| `visualize_splat(..., aligned_cameras)` | list of w2c (OpenCV) | passes to above |
| `FeedforwardResult.extrinsics` | w2c (OpenCV) | canonical — use directly |

## Component 3: Section renumbering in notebook

Current sections jump from §6 to §9 (§7–§8 were consumed by the comparison/overlay before Omega was added). Fix to sequential:

| Current | → | New |
|---------|---|-----|
| §9 — VGGT-Omega Reconstruction | → | §7 |
| §10 — VGGT-Omega Post-processing | → | §8 |
| §11 — VGGT-Omega Pointcloud Viewer | → | §9 |
| §12 — Side-by-side Comparison | → | §10 |
| §13 — Camera Pose Overlay | → | §11 |

Affects markdown cell headings and cross-references (e.g. "vs. VGGT-X (§3) and MapAnything (§6)" in §9 viewer stays correct). Cell IDs unchanged.

---

## Out of scope

- BA notebook fix (gets frustum correction for free; `aligned_cameras` contract already matches)
- Nerfstudio notebook / `PointcloudResult` (c2w OpenGL — different pipeline, separate fix if needed)
