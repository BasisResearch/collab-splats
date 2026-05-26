# Design: Mesh Method Comparison Notebook

**Date:** 2026-05-26
**Status:** approved

## Problem

Feedforward (VGGT-Omega) → TSDF meshing produces degenerate outdoor meshes: empty regions, disconnected blobs, distorted surfaces. Root cause: `sdf_trunc=0.08m` << per-frame outdoor depth disagreement (~0.3–2m), so SDF cancels across frames; `depth_trunc=10m` clips most outdoor geometry. Need empirical comparison of meshing approaches on a real cached result before committing to a new mesher implementation.

## Goal

A disposable comparison notebook that loads the existing C0043 omega zarr (no GPU), tests multiple denoising + meshing combinations, and visualizes each via `visualize_splat` with camera frustums. Outcome determines which method gets promoted to `ImageGridMesher` in the library.

## What changes

### 1. `collab_splats/utils/visualization.py` — add `o3d_mesh_to_polydata`

One new function appended to the file. Converts an in-memory `o3d.geometry.TriangleMesh` → `pv.PolyData` with RGB vertex scalars compatible with `visualize_splat`.

```python
def o3d_mesh_to_polydata(mesh: "o3d.geometry.TriangleMesh") -> pv.PolyData:
    """Convert Open3D TriangleMesh to PyVista PolyData with RGB vertex scalars."""
    import open3d as o3d
    verts  = np.asarray(mesh.vertices, dtype=np.float32)
    faces  = np.asarray(mesh.triangles, dtype=np.int32)
    pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=np.int32), faces]).ravel()
    pd = pv.PolyData(verts, pv_faces)
    if mesh.has_vertex_colors():
        pd["RGB"] = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    return pd
```

### 2. `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb` — new notebook

Exploratory (stage/ = not in tutorial index). Cells:

| § | Purpose |
|---|---|
| 0 Setup | imports, `CACHE_DIR = docs/source/.cache/birds_c0043`, `OUTPUT_DIR = /tmp/mesh_compare`, `OMEGA_ZARR` |
| 1 Load | `FeedforwardResult.load_zarr(OMEGA_ZARR)` — prints shapes, point count |
| 2 Inspect | depth range, conf distribution histogram, world-space bbox, scene extent — informs parameter choices |
| 3 Denoising | Three variants printed with point counts: (A) raw result.points, (B) tier-2: conf+depth_edge+depth_trunc mask on world_points, (C) tier-3: stat outlier removal on result.points with aligned pixel_indices |
| 4 TSDF tuned | `Open3DTSDFFusion(voxel_size=0.1, sdf_trunc=0.5, depth_trunc=200, clean_repair=False)` — outdoor-scaled params |
| 5 Image-grid (world_points) | Inline `_image_grid_from_world_points(result, conf_percentile, depth_trunc, edge_threshold)` — full N×H×W grid, mask drives quad validity |
| 6 Image-grid (pixel_indices) | Inline `_image_grid_from_pixel_indices(result)` — vertices = result.points, quad validity from pixel_to_point lookup |
| 7 Visualize each | `o3d_mesh_to_polydata` → `visualize_splat(pd, aligned_cameras=result.extrinsics, mesh_kwargs=MESH_KWARGS, camera_kwargs=CAMERA_KWARGS, viz_kwargs=VIZ_KWARGS)` per method |
| 8 Summary | Table: method × (vertex count, face count, coverage, visual quality note) |

## Inline mesh helpers (notebook-only, not promoted yet)

### `_image_grid_from_world_points`

```python
def _image_grid_from_world_points(result, conf_percentile=30.0,
                                   depth_trunc=200.0, edge_threshold=0.01):
    """Image-grid mesh from full world_points grid. Quad mask = conf + edge + depth."""
    import open3d as o3d
    from collab_splats.mesh.utils import find_depth_edges

    N, H, W, _ = result.world_points.shape
    depth = result.depth  # (N, H, W)
    conf  = result.confidence.numpy()  # (N, H, W)

    conf = conf.copy()
    for i in range(N):
        conf[i][find_depth_edges(depth[i], threshold=edge_threshold)] = 0.0
    conf[depth > depth_trunc] = 0.0
    valid_conf = conf[conf > 0]
    threshold = np.percentile(valid_conf, conf_percentile) if len(valid_conf) else 0.0
    mask = conf > threshold  # (N, H, W) bool

    # Build vertex set from valid pixels across all frames
    all_verts, all_colors, all_faces = [], [], []
    offset = 0
    for i in range(N):
        m = mask[i]  # (H, W)
        # vertex index map: -1 = invalid
        idx = np.full((H, W), -1, dtype=np.int64)
        valid_pixels = np.argwhere(m)  # (P, 2)
        idx[valid_pixels[:, 0], valid_pixels[:, 1]] = np.arange(len(valid_pixels)) + offset

        all_verts.append(result.world_points[i][m])  # (P, 3)
        # Load RGB from image_paths at model resolution
        from PIL import Image as PILImage
        img = PILImage.open(result.image_paths[i]).convert("RGB").resize(
            (W, H), PILImage.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)
        all_colors.append(rgb[m])  # (P, 3)

        # Quads: all 4 corners must be valid
        quad_mask = (
            (idx[:-1, :-1] >= 0) & (idx[1:, :-1] >= 0) &
            (idx[1:,  1:] >= 0) & (idx[:-1, 1:] >= 0)
        )
        v00 = idx[:-1, :-1][quad_mask]
        v10 = idx[1:,  :-1][quad_mask]
        v11 = idx[1:,   1:][quad_mask]
        v01 = idx[:-1,  1:][quad_mask]
        tris = np.concatenate([
            np.stack([v00, v10, v11], 1),
            np.stack([v00, v11, v01], 1),
        ])
        all_faces.append(tris)
        offset += len(valid_pixels)

    verts  = np.concatenate(all_verts).astype(np.float32)
    colors = np.concatenate(all_colors)
    faces  = np.concatenate(all_faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(verts)
    mesh.triangles     = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return mesh
```

### `_image_grid_from_pixel_indices`

```python
def _image_grid_from_pixel_indices(result):
    """Image-grid mesh where vertices == result.points. Topology from pixel_indices."""
    import open3d as o3d
    from collab_splats.mesh.utils import find_depth_edges

    if result.pixel_indices is None:
        raise ValueError("pixel_indices not populated — load via load_zarr()")

    N = len(result.image_paths)
    H, W = result.model_height, result.model_width

    # pixel → point-index lookup; -1 = filtered out
    pixel_to_point = np.full((N, H, W), -1, dtype=np.int64)
    fi, ri, ci = result.pixel_indices[:, 0], result.pixel_indices[:, 1], result.pixel_indices[:, 2]
    pixel_to_point[fi, ri, ci] = np.arange(len(result.points))

    depth = result.depth  # (N, H, W)
    all_faces = []
    for i in range(N):
        idx = pixel_to_point[i]
        valid = idx >= 0
        edge_mask = find_depth_edges(depth[i]) if depth is not None else np.zeros((H, W), bool)
        quad_mask = (
            valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:] &
            ~(edge_mask[:-1, :-1] | edge_mask[1:, :-1] | edge_mask[1:, 1:] | edge_mask[:-1, 1:])
        )
        v00 = idx[:-1, :-1][quad_mask]
        v10 = idx[1:,  :-1][quad_mask]
        v11 = idx[1:,   1:][quad_mask]
        v01 = idx[:-1,  1:][quad_mask]
        tris = np.concatenate([
            np.stack([v00, v10, v11], 1),
            np.stack([v00, v11, v01], 1),
        ])
        all_faces.append(tris)

    faces = np.concatenate(all_faces)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(result.points.astype(np.float64))
    mesh.triangles     = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        result.colors.astype(np.float64) / 255.0)
    return mesh
```

## What does NOT change

- `Open3DTSDFFusion` — tested as-is with new outdoor params, not modified
- `collab_splats/mesh/__init__.py` — no new registered meshers yet
- Any other library code — all mesh logic stays notebook-inline until a winner is clear

## Out of scope

- Promoting winner to `ImageGridMesher` class (next spec after comparison)
- Poisson mesher implementation
- Final `feedforward_mesh.ipynb` rewrite (follows after winner confirmed)
- BA hookup / arbitrary creator interface (follows after winner confirmed)

## Success criteria

Notebook runs top-to-bottom without error on omega zarr. At least one method produces a visually coherent mesh on C0043 (connected surfaces, no giant floating blobs, cameras visible alongside mesh). Visual comparison informs which method to promote.
