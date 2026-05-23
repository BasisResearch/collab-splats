# Feedforward Base Class Refactor — Architecture Cleanup

**Date:** 2026-04-21
**Branch:** `refactor/core-modules`
**Scope:** `collab_splats/pointcloud/base.py` architecture + ripple effects

## Context

`base.py` has accumulated inline annotation TODOs during the feedforward refactor: inconsistent field naming between `PointcloudResult` and `FeedforwardResult`, a utility function (`_colmap_recon_to_result`) living in the wrong module, lazy imports that should be top-level, and globals placed after class definitions. This spec resolves the architectural questions before we move to documentation/naming cleanup of `utils.py` and `feedforward.py`.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CoordinateFrame enum | Keep, add docstring | Tags which convention PointcloudResult uses. Used in 5 test files. |
| PointcloudResult field names | `extrinsics` / `intrinsics` | Match FeedforwardResult and standard CV terminology |
| `_write_transforms` nerfstudio dep | Keep as-is, move import to top | nerfstudio is a hard project dependency (registered entry points) |
| `_WORLD_TRANSFORM` placement | Move to `utils.py` | Keep with its only consumer `colmap_reconstruction_to_result` |
| `_colmap_recon_to_result` | Rename to `colmap_reconstruction_to_result`, move to `utils.py` | Pure conversion function used by all creator families (SfM + feedforward) |
| Imports | All top-level, no lazy imports in base.py | Consistent with "globals at top" organization |

## Architecture After Refactor

### base.py (~55 lines)

```
imports (including nerfstudio colmap_to_json at top)
CoordinateFrame enum (with docstring)
PointcloudResult dataclass (extrinsics/intrinsics fields, documented)
BasePointcloudCreator ABC (_write_transforms kept here)
```

### utils.py (gains ~60 lines)

```
existing imports + pycolmap, CoordinateFrame, PointcloudResult
_WORLD_TRANSFORM constant
colmap_reconstruction_to_result() function  ← renamed, now public
... existing utility functions ...
```

### Import direction (no cycles)

```
base.py ← utils.py (utils imports types from base)
base.py ← sfm.py
base.py ← feedforward.py
utils.py ← sfm.py (colmap_reconstruction_to_result)
utils.py ← feedforward.py (colmap_reconstruction_to_result, voxel_downsample_point_cloud)
```

## File Changes

| File | Change |
|------|--------|
| `pointcloud/utils.py` | Add `_WORLD_TRANSFORM` + `colmap_reconstruction_to_result()` (renamed, public). Add `pycolmap` import + base types. |
| `pointcloud/base.py` | Remove `_WORLD_TRANSFORM` + `_colmap_recon_to_result`. Rename PointcloudResult fields (`camera_poses` → `extrinsics`, `camera_intrinsics` → `intrinsics`). Move nerfstudio import to top. Add CoordinateFrame docstring. Remove annotation comments. |
| `pointcloud/__init__.py` | Update export source for `colmap_reconstruction_to_result` (from utils, not base). |
| `pointcloud/sfm.py` | Update import path + function name. |
| `pointcloud/feedforward.py` | Update import path + function name. |
| `tests/pointcloud/test_base.py` | Update import, field names, function name. |
| `tests/pointcloud/test_sfm_creator.py` | Update `result.camera_poses` → `result.extrinsics`. |

### Files verified safe (no changes needed)

- `wrapper/splatter.py` — `camera_poses` is a local variable from transforms.json, not PointcloudResult
- `_mapanything.py` — `pred["camera_poses"]` is a dict key on raw model output
- `_vggt.py` — no PointcloudResult field access
- `test_mapanything_creator.py`, `test_vggtx_creator.py` — no PointcloudResult field access

## Execution Order

1. **Step 1:** Add `colmap_reconstruction_to_result` + `_WORLD_TRANSFORM` to `utils.py` (create target first)
2. **Steps 2-5:** Update `base.py`, `__init__.py`, `sfm.py`, `feedforward.py` (imports + names)
3. **Steps 6-7:** Update test files (field names + imports)
4. Single commit covering all files

## Verification

1. `python -m pytest tests/pointcloud/ -v` — all tests pass
2. `python -c "from collab_splats.pointcloud import colmap_reconstruction_to_result"` — import works
3. `python -c "from collab_splats.pointcloud import PointcloudResult; r = PointcloudResult.__dataclass_fields__; assert 'extrinsics' in r and 'intrinsics' in r"` — field rename verified
4. `grep -r "camera_poses\|camera_intrinsics" collab_splats/pointcloud/` — no stale references in pointcloud module
5. `grep -r "_colmap_recon_to_result" collab_splats/` — no stale references in source
