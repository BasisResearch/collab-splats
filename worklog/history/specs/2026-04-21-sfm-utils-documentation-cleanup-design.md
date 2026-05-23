# SFM & Pointcloud Utils — Documentation & Cleanup Design

**Date**: 2026-04-21  
**Branch**: `refactor/core-modules`  
**Scope**: `collab_splats/pointcloud/{sfm.py, utils.py, base.py, __init__.py}` + tests + shim

## Context

The pointcloud utils module has grown organically with duplicate downsampling implementations, inconsistent naming, missing docstrings, and inline TODO comments requesting cleanup. The SFM creators lack documentation explaining pipeline steps and configuration options. A broken re-export shim (`clean_pcd`) also needs fixing.

**Goal**: Document all functions with Google-style docstrings, decompose `clean_pointcloud` into composable pieces, unify naming, eliminate redundancy, and fix the broken shim.

---

## Design Decisions

### 1. Composable Filter Architecture

`clean_pointcloud` currently inlines adaptive voxel downsample logic and calls `remove_far_points` directly. Refactor into three independent functions that `clean_pointcloud` delegates to as a convenience wrapper.

### 2. Two Filter Functions

- **`filter_density(pcd, radius, percentile)`** — KDTree neighbor count, removes sparse regions. Open3D.
- **`filter_distance(pcd, method="radial"|"bbox", ...)`** — unified dispatcher:
  - `method="radial"`: sphere from centroid/origin. Params: `max_distance`, `n_points`, `reference`
  - `method="bbox"`: percentile axis-aligned bounding box. Params: `percentile_range`, `max_extent`

### 3. Single `voxel_downsample` Using Open3D

Drop pure-numpy `voxel_downsample_point_cloud`. Extract Open3D adaptive logic from `clean_pointcloud` into standalone `voxel_downsample(pcd, voxel_size, radius, adaptive)`.

### 4. Renames (Clean Break)

| Current | New | Notes |
|---------|-----|-------|
| `remove_far_points` | `filter_distance(method="radial")` | Absorbed |
| `density_filter` | `filter_density` | Renamed |
| `filter_points_by_spatial_extent` | `filter_distance(method="bbox")` | Absorbed |
| `voxel_downsample_point_cloud` | `voxel_downsample` | Open3D, replaces numpy |
| `clean_pointcloud` | `clean_pointcloud` | Kept as wrapper |
| `colmap_reconstruction_to_result` | `colmap_reconstruction_to_result` | Unchanged |

### 5. `clean_pointcloud` Wrapper Interface

```python
def clean_pointcloud(
    pcd: "o3d.geometry.PointCloud",
    downsample: bool = True,
    outlier_removal: bool = True,
    distance_removal: bool = True,
    downsample_kwargs: dict | None = None,
    density_kwargs: dict | None = None,
    distance_kwargs: dict | None = None,
) -> tuple["o3d.geometry.PointCloud", np.ndarray]:
```

Delegates to `voxel_downsample`, `pcd.remove_statistical_outlier`, `filter_distance` internally. Each `*_kwargs` dict forwarded to corresponding function.

---

## Function Signatures

### `filter_distance`

```python
def filter_distance(
    pcd: "o3d.geometry.PointCloud",
    method: str = "radial",
    *,
    max_distance: float | None = None,
    n_points: int | None = None,
    reference: str = "centroid",
    percentile_range: tuple[float, float] = (1.0, 99.0),
    max_extent: float | None = None,
    return_mask: bool = False,
) -> "o3d.geometry.PointCloud | tuple[o3d.geometry.PointCloud, np.ndarray]":
```

- `method="radial"`: requires `max_distance` or `n_points`
- `method="bbox"`: uses `percentile_range`, optional `max_extent`
- All Open3D PointCloud in/out (unified interface)

### `filter_density`

```python
def filter_density(
    pcd: "o3d.geometry.PointCloud",
    radius: float = 0.03,
    percentile: float = 10.0,
) -> "o3d.geometry.PointCloud":
```

### `voxel_downsample`

```python
def voxel_downsample(
    pcd: "o3d.geometry.PointCloud",
    voxel_size: float = 0.015,
    radius: float = 0.05,
    adaptive: bool = True,
) -> tuple["o3d.geometry.PointCloud", np.ndarray]:
```

Returns `(downsampled_pcd, index_mapping)` for provenance tracking.

---

## Implementation Steps

### Step 1: Add new composable functions (additive)

Add `filter_distance`, `filter_density`, `voxel_downsample` to `utils.py`. No removals yet. All tests still pass.

### Step 2: Rewrite `clean_pointcloud` as thin wrapper

Replace inlined logic with delegation to new functions via `*_kwargs`. Default behavior unchanged: `clean_pointcloud(pcd)` produces same result.

### Step 3: Update `__init__.py` exports

Export new function names. Remove old names from `__all__`.

### Step 4: Fix shim + update feedforward caller

**`collab_splats/utils/pointcloud.py`**: Fix broken `clean_pcd` import. Add backward-compat aliases mapping old names → new.

**`collab_splats/pointcloud/feedforward.py` line 350**: Replace `voxel_downsample_point_cloud(pts3d, colors)` with Open3D conversion + `voxel_downsample` call.

### Step 5: Remove old functions

Delete from `utils.py`:
- `remove_far_points`
- `density_filter`
- `filter_points_by_spatial_extent`
- `voxel_downsample_point_cloud`
- All TODO comments (lines 76, 153, 211, 245, 303)

### Step 6: Documentation + tests

**sfm.py docstrings**:
- `ColmapCreator`: Document 3-step pycolmap pipeline (extract → match → map), camera_model options, single_camera usage
- `HlocCreator`: Document 4-step hloc pipeline (retrieval → features → matching → reconstruction), lazy import rationale, config string meanings

**utils.py inline comments**:
- Coordinate transform math in `colmap_reconstruction_to_result` (row-swap explanation)
- Adaptive voxel sizing formula and magic numbers in `voxel_downsample`
- KDTree density estimation logic in `filter_density`

**Test updates**:
- `tests/pointcloud/test_base.py`: Update imports to new names
- `tests/pointcloud/test_pointcloud_utils.py`: Rewrite for Open3D interface + new function names
- Add tests for `filter_distance` (radial + bbox modes), `filter_density`, `voxel_downsample`

---

## Known Issues to Fix

1. **Broken shim**: `collab_splats/utils/pointcloud.py` imports `clean_pcd` which doesn't exist in `collab_splats.pointcloud.utils`. Currently fails at import time.
2. **Feedforward numpy↔Open3D**: `feedforward.py:350` calls numpy-interface `voxel_downsample_point_cloud`. New `voxel_downsample` takes Open3D pcd — need inline conversion in feedforward `_postprocess`.

---

## Critical Files

| File | Action |
|------|--------|
| `collab_splats/pointcloud/utils.py` | Add 3 functions, rewrite wrapper, remove 4 old, add docstrings + comments |
| `collab_splats/pointcloud/sfm.py` | Add Google-style docstrings for both creators |
| `collab_splats/pointcloud/__init__.py` | Update exports |
| `collab_splats/utils/pointcloud.py` | Fix broken shim, add backward-compat aliases |
| `collab_splats/pointcloud/feedforward.py` | Update voxel_downsample call at line 350 |
| `tests/pointcloud/test_base.py` | Update imports |
| `tests/pointcloud/test_pointcloud_utils.py` | Rewrite for new interfaces |

---

## Verification

1. `python -m pytest tests/pointcloud/ -v` — all existing + new tests pass
2. `python -c "from collab_splats.pointcloud import filter_distance, filter_density, voxel_downsample, clean_pointcloud"` — imports work
3. `python -c "from collab_splats.utils.pointcloud import clean_pcd, remove_far_points, density_filter"` — shim works
4. `python -c "from collab_splats.pointcloud.sfm import ColmapCreator, HlocCreator; help(ColmapCreator); help(HlocCreator)"` — docstrings render
5. Grep for removed function names across codebase — no stale references
