# Notebook Viz Polish: Colormap Dispatch + tqdm Auto

**Date:** 2026-05-23  
**Scope:** `collab_splats/utils/visualization.py`, 7 library files, notebooks

---

## Problem

Two independent notebook polish issues:

1. **Colormap dispatch:** RGB pointclouds (N×3 uint8) render with viridis in notebooks. Root cause: `visualize_splat` defaults `mesh_kwargs={}`. PyVista sees the "RGB" point array but without `rgb=True`, treats it as a 3-component scalar and applies the default viridis colormap. The RGB data is present — dispatch logic is missing.

2. **tqdm repetitive output:** Library code uses `from tqdm import tqdm` (plain). In Jupyter, plain tqdm appends a new line per update instead of updating in place, producing walls of `42%|████▏ | 5/12` lines in cell output.

---

## Design

### 1. Colormap Dispatch — `visualization.py`

Add private helper `_resolve_mesh_kwargs(mesh, mesh_kwargs)`:

```python
def _resolve_mesh_kwargs(mesh: pv.PolyData, mesh_kwargs: dict) -> dict:
    if mesh_kwargs:
        return mesh_kwargs
    if (
        isinstance(mesh, pv.PolyData)
        and "RGB" in mesh.point_data
        and mesh.point_data["RGB"].ndim == 2
        and mesh.point_data["RGB"].shape[1] == 3
    ):
        return PCD_KWARGS
    return {}
```

**Rule:**
- Caller passes explicit `mesh_kwargs` → use as-is (no override)
- Mesh has "RGB" point array of shape (N, 3) → use `PCD_KWARGS` (`scalars="RGB", rgb=True, render_points_as_spheres=True, ...`)
- Otherwise → `{}` (PyVista default)

**Only `visualize_splat` changes** — insert one call before `plotter.add_mesh`:

```python
mesh_kwargs = _resolve_mesh_kwargs(mesh, mesh_kwargs)
plotter.add_mesh(mesh, **mesh_kwargs)
```

This is backward-compatible: existing callers passing `mesh_kwargs=PCD_KWARGS` explicitly are unaffected.

**Notebooks:** No changes needed. `feedforward_methods.ipynb` calling `visualize_splat(cloud_vggt)` will auto-resolve to RGB mode.

**localization.ipynb** passes `pointcloud_to_polydata(pts3d)` with no colors → auto-detect correctly skips rgb mode → grey cloud. Acceptable for pose visualization.

**1D scalars (similarity, features):** Notebooks calling `pl.add_mesh(cloud, scalars="semantic", cmap="plasma")` pass explicit `mesh_kwargs` → helper returns them unchanged → correct viridis/plasma behavior preserved.

---

### 2. tqdm Auto — Library Files

Change `from tqdm import tqdm/trange` → `from tqdm.auto import tqdm/trange` in:

| File | Current import |
|------|---------------|
| `collab_splats/utils/frame_sampling.py` | `from tqdm import tqdm` |
| `collab_splats/semantics/compression.py` | `from tqdm import tqdm` |
| `collab_splats/pointcloud/wrappers.py` | `from tqdm import tqdm` (inline, inside method) |
| `collab_splats/pointcloud/utils.py` | `from tqdm import trange` |
| `collab_splats/mesh/utils.py` | `from tqdm import tqdm, trange` |
| `collab_splats/mesh/tsdf.py` | `from tqdm import tqdm` |
| `collab_splats/nerfstudio/datamanagers/features.py` | `from tqdm import trange` |

`tqdm.auto` detects environment: Jupyter → inline ipywidgets progress bar (updates in place); terminal/tmux → standard text bar. No behavior change for CLI eval runs.

**Notebooks:** `semantic_lifting.ipynb` already uses `tqdm.auto`. No other source notebooks import tqdm directly in cells — the repetitive output came from library calls.

---

## Non-Goals

- No changes to `pointcloud_to_polydata` signature or behavior
- No changes to `localization.ipynb` (grey cloud is fine)
- No changes to semantic notebooks (their explicit cmap args are correct)
- No tqdm changes in test files

---

## Files Changed

```
collab_splats/utils/visualization.py        # _resolve_mesh_kwargs + visualize_splat
collab_splats/utils/frame_sampling.py       # tqdm.auto
collab_splats/semantics/compression.py      # tqdm.auto
collab_splats/pointcloud/wrappers.py        # tqdm.auto
collab_splats/pointcloud/utils.py           # tqdm.auto
collab_splats/mesh/utils.py                 # tqdm.auto
collab_splats/mesh/tsdf.py                  # tqdm.auto
collab_splats/nerfstudio/datamanagers/features.py  # tqdm.auto
```

Total: 8 files, no new files, no notebook changes.
