# Notebook Viz Polish: Colormap Dispatch + tqdm Auto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix RGB pointclouds rendering as viridis in notebooks, and replace `tqdm.tqdm` with `tqdm.auto` in library code so Jupyter shows inline progress widgets instead of repetitive text lines.

**Architecture:** Two independent changes. (1) Add `_resolve_mesh_kwargs` private helper to `visualization.py` that inspects the mesh for an "RGB" point array and injects `rgb=True` when the caller passed no explicit kwargs. (2) Mechanical import swap across 7 library files — `from tqdm import tqdm/trange` → `from tqdm.auto import tqdm/trange`.

**Tech Stack:** PyVista (pointcloud rendering), tqdm.auto (environment-aware progress bars), pytest

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/utils/visualization.py` | Add `_resolve_mesh_kwargs`; call it in `visualize_splat` |
| `tests/test_visualization.py` | Add tests for `_resolve_mesh_kwargs` dispatch |
| `collab_splats/utils/frame_sampling.py:6` | `from tqdm import tqdm` → `from tqdm.auto import tqdm` |
| `collab_splats/semantics/compression.py:22` | `from tqdm import tqdm` → `from tqdm.auto import tqdm` |
| `collab_splats/pointcloud/wrappers.py:239` | `from tqdm import tqdm` → `from tqdm.auto import tqdm` (inline inside method) |
| `collab_splats/pointcloud/utils.py:19` | `from tqdm import trange` → `from tqdm.auto import trange` |
| `collab_splats/mesh/utils.py:15` | `from tqdm import tqdm, trange` → `from tqdm.auto import tqdm, trange` |
| `collab_splats/mesh/tsdf.py:10` | `from tqdm import tqdm` → `from tqdm.auto import tqdm` |
| `collab_splats/nerfstudio/datamanagers/features.py:17` | `from tqdm import trange` → `from tqdm.auto import trange` |

---

## Task 1: Colormap dispatch — `_resolve_mesh_kwargs` + `visualize_splat`

**Files:**
- Modify: `collab_splats/utils/visualization.py:240-264`
- Modify: `tests/test_visualization.py`

- [ ] **Step 1: Write failing tests for `_resolve_mesh_kwargs`**

First, add `import pyvista as pv` to the imports at the top of `tests/test_visualization.py` (alongside the existing `import numpy as np` and `import torch`).

Then add to the bottom of `tests/test_visualization.py`:

```python
def make_rgb_cloud(N=100):
    pts = np.random.rand(N, 3).astype(np.float32)
    colors = (np.random.rand(N, 3) * 255).astype(np.uint8)
    cloud = pv.PolyData(pts)
    cloud["RGB"] = colors
    return cloud


def make_bare_cloud(N=100):
    pts = np.random.rand(N, 3).astype(np.float32)
    return pv.PolyData(pts)


def test_resolve_mesh_kwargs_rgb_returns_pcd_kwargs():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs, PCD_KWARGS
    cloud = make_rgb_cloud()
    result = _resolve_mesh_kwargs(cloud, {})
    assert result == PCD_KWARGS


def test_resolve_mesh_kwargs_explicit_passthrough():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs
    cloud = make_rgb_cloud()
    explicit = {"scalars": "RGB", "rgb": True, "point_size": 3.0}
    result = _resolve_mesh_kwargs(cloud, explicit)
    assert result is explicit  # exact same dict object, no copy


def test_resolve_mesh_kwargs_bare_cloud_returns_empty():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs
    cloud = make_bare_cloud()
    result = _resolve_mesh_kwargs(cloud, {})
    assert result == {}


def test_resolve_mesh_kwargs_non_polydata_returns_empty():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs
    # A mesh loaded from file would be pv.PolyData but with faces; simulate with PolyData without "RGB"
    mesh = pv.Sphere()
    result = _resolve_mesh_kwargs(mesh, {})
    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_visualization.py::test_resolve_mesh_kwargs_rgb_returns_pcd_kwargs tests/test_visualization.py::test_resolve_mesh_kwargs_explicit_passthrough tests/test_visualization.py::test_resolve_mesh_kwargs_bare_cloud_returns_empty tests/test_visualization.py::test_resolve_mesh_kwargs_non_polydata_returns_empty -v
```

Expected: `ImportError` or `FAILED` — `_resolve_mesh_kwargs` does not exist yet.

- [ ] **Step 3: Add `_resolve_mesh_kwargs` to `visualization.py`**

In `collab_splats/utils/visualization.py`, insert the helper immediately before the `# ── 3D Visualization` section divider (before line 240). Add it right after `overlay_masks` ends (after line 237):

```python
def _resolve_mesh_kwargs(mesh: pv.PolyData, mesh_kwargs: dict) -> dict:
    """Return mesh_kwargs, auto-detecting RGB pointcloud mode when kwargs are empty."""
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

- [ ] **Step 4: Update `visualize_splat` to call the helper**

In `visualize_splat`, replace the `plotter.add_mesh(mesh, **mesh_kwargs)` line (line 264) with:

```python
    mesh_kwargs = _resolve_mesh_kwargs(mesh, mesh_kwargs)
    plotter.add_mesh(mesh, **mesh_kwargs)
```

The insertion point is after the `if isinstance(mesh, str): mesh = pv.read(mesh)` block.

- [ ] **Step 5: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_visualization.py -v
```

Expected: all tests PASS including the 4 new ones and all pre-existing tests.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/utils/visualization.py tests/test_visualization.py
git commit -m "fix(viz): auto-detect RGB pointcloud in visualize_splat, skip viridis"
```

---

## Task 2: tqdm.auto — mechanical import swap across library files

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py:6`
- Modify: `collab_splats/semantics/compression.py:22`
- Modify: `collab_splats/pointcloud/wrappers.py:239`
- Modify: `collab_splats/pointcloud/utils.py:19`
- Modify: `collab_splats/mesh/utils.py:15`
- Modify: `collab_splats/mesh/tsdf.py:10`
- Modify: `collab_splats/nerfstudio/datamanagers/features.py:17`

- [ ] **Step 1: Run existing tests as baseline**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration -q 2>&1 | tail -20
```

Record pass/fail count — must match after the changes.

- [ ] **Step 2: Apply import swaps**

Make each of the following single-line edits:

**`collab_splats/utils/frame_sampling.py` line 6:**
```python
# Before:
from tqdm import tqdm
# After:
from tqdm.auto import tqdm
```

**`collab_splats/semantics/compression.py` line 22:**
```python
# Before:
from tqdm import tqdm
# After:
from tqdm.auto import tqdm
```

**`collab_splats/pointcloud/wrappers.py` line 239** (inside method body — keep it inline):
```python
# Before:
        from tqdm import tqdm
# After:
        from tqdm.auto import tqdm
```

**`collab_splats/pointcloud/utils.py` line 19:**
```python
# Before:
from tqdm import trange
# After:
from tqdm.auto import trange
```

**`collab_splats/mesh/utils.py` line 15:**
```python
# Before:
from tqdm import tqdm, trange
# After:
from tqdm.auto import tqdm, trange
```

**`collab_splats/mesh/tsdf.py` line 10:**
```python
# Before:
from tqdm import tqdm
# After:
from tqdm.auto import tqdm
```

**`collab_splats/nerfstudio/datamanagers/features.py` line 17:**
```python
# Before:
from tqdm import trange
# After:
from tqdm.auto import trange
```

- [ ] **Step 3: Verify no test regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration -q 2>&1 | tail -20
```

Expected: same pass/fail count as Step 1 baseline.

- [ ] **Step 4: Commit**

```bash
git add \
  collab_splats/utils/frame_sampling.py \
  collab_splats/semantics/compression.py \
  collab_splats/pointcloud/wrappers.py \
  collab_splats/pointcloud/utils.py \
  collab_splats/mesh/utils.py \
  collab_splats/mesh/tsdf.py \
  collab_splats/nerfstudio/datamanagers/features.py
git commit -m "fix(tqdm): use tqdm.auto across library for Jupyter inline progress bars"
```
