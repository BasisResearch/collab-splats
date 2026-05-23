# Utils Cleanup — Design Spec

**Date:** 2026-04-28  
**Branch:** refactor/core-modules  
**Scope:** Option B + grouping move

---

## Goal

Shrink `collab_splats/utils/` by removing dead shims, migrating nerfstudio-owned code to `nerfstudio/utils/`, and parking unimplemented Gaussian grouping in `stage/`.

---

## Files Deleted from `utils/`

| File | Reason |
|------|--------|
| `utils/pointcloud.py` | Dead shim — 0 callers |
| `utils/segmentation.py` | Shim — 1 caller migrated then deleted |
| `utils/features.py` | Real code dispersed; then deleted |
| `utils/utils.py` | All 4 functions dispersed; then deleted |
| `utils/grouping.py` | Not implemented/used — moved to `stage/` |

---

## Code Movements

### `utils/utils.py` → dispersed

`depth_double_to_normal` and `convert_to_colmap_camera` are in `camera_utils.py`, not `utils.py` — `utils/__init__.py` re-exports them. Since `camera_utils.py` stays, those imports require **no changes** in nerfstudio models.

| Symbol | New home | Notes |
|--------|----------|-------|
| `get_device` | `semantics/utils.py` (append) | Used by `semantics/features.py` |
| `project_gaussians` | travels with `grouping.py` → `stage/` | Only caller is grouping |
| `calculate_accuracy` | `stage/metrics.py` (new) | Eval utility, no production callers |
| `calculate_completeness` | `stage/metrics.py` (new) | Eval utility, no production callers |
| `mean_angular_error` | `stage/metrics.py` (new) | Eval utility, no production callers |

### `utils/features.py` → dispersed

| Symbol | New home | Notes |
|--------|----------|-------|
| `TwoLayerMLP` | inline into `nerfstudio/models/rade_features.py` | Single caller; no abstraction needed |
| `BaseFeatureExtractor` re-export | removed | Callers updated to `semantics.features` directly |

### `utils/grouping.py`

Move to `stage/grouping.py` as-is. Not wired into any production path.

---

## Caller Updates

| File | Old import | New import |
|------|-----------|-----------|
| `nerfstudio/models/rade_features.py` | `utils.features.TwoLayerMLP` | local definition |
| `nerfstudio/models/rade_features.py` | `utils.features.BaseFeatureExtractor` | `semantics.features.BaseFeatureExtractor` |
| `nerfstudio/models/rade_gs.py` | `collab_splats.utils.{convert_to_colmap_camera,depth_double_to_normal}` | **unchanged** — already via `camera_utils.py` re-export |
| `nerfstudio/datamanagers/features.py` | `utils.segmentation.{Segmentation,aggregate_masked_features}` | `semantics.segmentation` |
| `semantics/features.py` | `collab_splats.utils.get_device` | `collab_splats.semantics.utils.get_device` |
| `stage/feedforward.py` | `utils.features.{BaseFeatureExtractor,...}` | `semantics.features` |

---

## `utils/` End State

```
collab_splats/utils/
    __init__.py       # remove stale re-exports of deleted symbols
    image.py          # keep — general I/O used across semantics + nerfstudio
    frame_sampling.py # keep — canonical per WORKLOG
    camera_utils.py   # keep — deferred to future cleanup (C-scope)
    visualization.py  # keep — deferred to future cleanup (C-scope)
```

---

## New Files Created

| File | Contents |
|------|----------|
| `stage/metrics.py` | `calculate_accuracy`, `calculate_completeness`, `mean_angular_error` (eval utilities) |

No new files needed in `nerfstudio/utils/` — `depth_double_to_normal` and `convert_to_colmap_camera` stay in `camera_utils.py`.

---

## Out of Scope

- `utils/camera_utils.py` — stays in utils (C-scope: move to `nerfstudio/utils/` later)
- `utils/visualization.py` — stays in utils (C-scope: move to `semantics/` later)
- Loop closure, mesh, pointcloud modules — untouched

---

## Testing

No behavioral changes. After each file deletion, run:

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats"
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q
```
