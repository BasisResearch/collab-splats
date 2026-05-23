# Feedforward → Direct Mesh Design

**Date:** 2026-05-14
**Branch:** refactor/core-modules

## Problem

Current mesh path: feedforward → nerfstudio trains splat → renders depth maps → TSDF.
Goal: mesh directly from MapAnything pointcloud, bypassing nerfstudio entirely.
Both paths must remain available.

## Approach

TSDF from `world_points`. `PointcloudResult.world_points` (N, H, W, 3) contains per-pixel
world-space 3D points from MapAnything inference. Project into each camera frame via
extrinsics to recover Z depth, load RGB from `image_paths`, invert w2c extrinsics → c2w.
Feed into existing `Open3DTSDFFusion.create()` unchanged.

No OpenGL-to-OpenCV conversion needed — MapAnything extrinsics are already OpenCV convention.

## Components

### 1. `collab_splats/mesh/adapter.py`

New file. Single public function:

```python
def pointcloud_result_to_tsdf_inputs(
    result: PointcloudResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

**Returns:** `(depths, rgbs, c2w, intrinsics)`

| Output | Shape | Notes |
|--------|-------|-------|
| `depths` | (N, H, W) float32 | Z depth in metres, clipped ≥ 0 |
| `rgbs` | (N, H, W, 3) float32 [0,1] | loaded from `result.image_paths`, resized to model (H, W) |
| `c2w` | (N, 4, 4) float32 | `np.linalg.inv(result.extrinsics)` |
| `intrinsics` | (N, 3, 3) float32 | passed through from `result.intrinsics` |

**Depth extraction per frame `i`:**
```python
R = result.extrinsics[i, :3, :3]   # world-to-cam rotation
t = result.extrinsics[i, :3, 3]    # world-to-cam translation
cam_pts = result.world_points[i] @ R.T + t  # (H, W, 3)
depth = cam_pts[..., 2].clip(0)             # (H, W)
```

**Precondition:** raises `ValueError` if `result.world_points is None`.

**RGB resize:** PIL `Image.open(...).resize((W, H), BILINEAR)` — model resolution, not original.

**Intrinsics:** already at model resolution from MapAnything; passed through unchanged.

Export from `collab_splats/mesh/__init__.py`.

### 2. `docs/pointcloud/feedforward_mesh.ipynb`

6-cell notebook. Style matches `bundle_adjustment.ipynb` (small, runnable, API-docs focus).
Dataset: 7scenes chess subset at `data/7scenes/chess/` (small, ≤50 frames, runs <90s).

| Cell | Purpose |
|------|---------|
| 1 | Imports + config (`image_dir`, `output_dir`, `mesh_dir`) |
| 2 | `MapAnythingCreator(...).reconstruct(image_dir, output_dir)` → `result` |
| 3 | Inspect shapes: `world_points`, `pts3d`, `extrinsics` |
| 4 | `pointcloud_result_to_tsdf_inputs(result)` + imshow one depth frame |
| 5 | `Open3DTSDFFusion(output_dir=mesh_dir).create(depths, rgbs, c2w, intrinsics)` |
| 6 | Load + display mesh |

## What does NOT change

- `Open3DTSDFFusion` — no modifications
- `BaseMeshCreator` interface — no modifications
- `PointcloudResult` dataclass — no modifications
- Existing nerfstudio splat → mesh path in `Splatter.mesh()` — untouched

## Out of scope

- BA-refined pose variant (deferred)
- `GaussiansPoisson` / `DepthNormalPoisson` implementation (both still `NotImplementedError`)
- Poisson-from-pts3d path
- VGGTX variant (world_points populated differently via `_raw_to_world_points` with subsample; adapter works but resolution differs — note in notebook)
