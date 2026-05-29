# Design: VGGT-Omega in feedforward_methods.ipynb

**Date:** 2026-05-26  
**Status:** approved

## Summary

Add VGGT-Omega as a third feedforward method in `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`, following the same sequential per-method structure already used for VGGT-X (§1–3) and MapAnything (§4–6). Extend the comparison sections to cover all three models.

## Scope

Notebook-only change. No library code is modified. `VGGTOmegaCreator` is already implemented and exported (`collab_splats/pointcloud/feedforward/vggt_omega.py`, added in `03ddc88`).

## New cells (inserted before current §7/§8)

### §9 — VGGT-Omega Reconstruction (markdown + code)

Markdown: explain VGGT-Omega is the third feedforward method; zarr-cache pattern identical to §1/§4; defaults only (`VGGTOmegaCreator()`).

Code pattern:
```python
_omega_cache = CACHE_DIR / "omega" / "reconstruction.zarr"
_omega_cache.parent.mkdir(parents=True, exist_ok=True)

if _omega_cache.exists():
    result_omega = FeedforwardResult.load_zarr(_omega_cache)
    print(f"Loaded VGGT-Omega result from cache  ({result_omega.points.shape[0]:,} pts)")
else:
    result_omega = VGGTOmegaCreator().run(IMAGES, device)
    result_omega.save_zarr(_omega_cache)
    print(f"VGGT-Omega done  →  saved to {_omega_cache}")
```

### §10 — VGGT-Omega Post-processing (markdown + code)

Markdown: same outlier removal + voxel downsample pipeline as §2/§5; metrics collected for comparison table.

Code: identical pattern to §5, variables named `pts3d_omega`, `colors_omega`, `conf_omega_mean`, `conf_omega_std`.

### §11 — VGGT-Omega Pointcloud Viewer (markdown + code)

Markdown: renders filtered VGGT-Omega pointcloud with green frustums (VGGT-X=blue, MapAnything=orange, Omega=green).

Code: identical to §6 pattern using `pts3d_omega`, `colors_omega`, `result_omega`.

## Updated sections (renumbered)

### §12 — Side-by-side Comparison (was §7)

Extend stats table with Omega row. Same columns: `Model / Pts raw / Pts filt / Conf mean / Conf std`.

```python
print(f"{'VGGT-Omega':<{col}} {len(result_omega.points):>10,} {len(pts3d_omega):>10,} {conf_omega_mean:>10.3f} {conf_omega_std:>10.3f}")
```

Header markdown updated to "three models".

### §13 — Camera Pose Overlay (was §8)

Add third frustum loop in green after VGGT-X (blue) and MapAnything (orange):

```python
for ext in result_omega.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="mediumseagreen", line_width=2)
```

Header markdown updated to "blue for VGGT-X, orange for MapAnything, green for VGGT-Omega".

## Import change

Cell 2 gains `VGGTOmegaCreator` in the import:

```python
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
```

## Notebook title / intro cell

Cell 0 prose updated to mention VGGT-Omega alongside VGGT-X and MapAnything as a third method.

## Out of scope

- No `enable_text_alignment` demo
- No refactor of existing §1–§8 cells
- No library code changes
