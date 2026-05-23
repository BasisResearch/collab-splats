# clean_pointcloud: API Refactor + Logging

**Date:** 2026-04-22
**Branch:** refactor/core-modules
**File:** `collab_splats/pointcloud/utils.py`

## Context

`clean_pointcloud` has a flag+kwargs pattern (`downsample=True` + `downsample_kwargs={}`) that is redundant and creates a footgun: a caller can pass `downsample=False, downsample_kwargs={...}` and silently have kwargs ignored. The function also has no logging, making it opaque in production pipelines. The current `max_distance=1.0` default is far too small for large-scale MapAnything scenes. This refactor fixes all three issues.

## API Changes

### Remove boolean flags

Remove `downsample`, `outlier_removal`, `distance_removal`. Replace with `None`-as-skip on the kwargs params.

### New signature

```python
_DEFAULT_DOWNSAMPLE_KWARGS = {"voxel_size": 0.015, "adaptive": True}
_DEFAULT_OUTLIER_KWARGS    = {"nb_neighbors": 20, "std_ratio": 2.0}
_DEFAULT_DISTANCE_KWARGS   = {"method": "radial", "max_distance": 50.0}

def clean_pointcloud(
    pcd,
    downsample_kwargs: Optional[dict] = _DEFAULT_DOWNSAMPLE_KWARGS,
    outlier_kwargs:    Optional[dict] = _DEFAULT_OUTLIER_KWARGS,
    distance_kwargs:   Optional[dict] = _DEFAULT_DISTANCE_KWARGS,
) -> tuple[Any, np.ndarray]:
```

### Merge semantics

Each active step merges user kwargs over module defaults: `{**_DEFAULT_X, **user_kwargs}`.

- Caller overrides only what they need: `distance_kwargs={"max_distance": 100.0}` keeps `method="radial"`
- `None` = skip step entirely
- `{}` = run with full defaults

### Remove `density_kwargs`

Currently unused/reserved. Drop it. Add `outlier_kwargs` in its place to expose previously hardcoded outlier params.

### Breaking changes

- Callers passing boolean flags (`downsample=False`, etc.) will break — update call sites
- `density_kwargs` removed — any callers can drop it (it did nothing)
- `distance_kwargs` callers unaffected (same name, same semantics)

## Logging

Use `logging.getLogger(__name__)` at module level (consistent with `collab_splats/semantics/segmentation.py`).

Log at `DEBUG` level before/after each step and a summary at the end:

```python
logger.debug("clean_pointcloud: start %d points", n_start)
logger.debug("downsample: %d → %d (%d removed)", n_before, n_after, n_before - n_after)
logger.debug("outlier_removal: %d → %d (%d removed)", n_before, n_after, n_before - n_after)
logger.debug("distance_removal: %d → %d (%d removed)", n_before, n_after, n_before - n_after)
logger.debug("clean_pointcloud: done %d → %d (%d total removed)", n_start, n_final, n_start - n_final)
```

Skipped steps produce no log line.

## Default Value Rationale

| Param | Old | New | Reason |
|---|---|---|---|
| `max_distance` | 1.0 m | 50.0 m | MapAnything outdoor scenes span tens of meters; 1.0 m removed almost everything |
| `nb_neighbors` | 20 (hardcoded) | 20 (default dict) | No change, now overridable |
| `std_ratio` | 2.0 (hardcoded) | 2.0 (default dict) | No change, now overridable |

## Files to Modify

- `collab_splats/pointcloud/utils.py` — main change
- `collab_splats/utils/pointcloud.py` — re-export shim, verify `clean_pcd` alias still works
- `tests/pointcloud/test_pointcloud_utils.py` — update tests using boolean flags

## Verification

1. Run `tests/pointcloud/test_pointcloud_utils.py` — all tests pass
2. Call `clean_pointcloud(pcd)` — all 3 steps run, 4 debug log lines emitted
3. Call `clean_pointcloud(pcd, downsample_kwargs=None)` — downsample skipped, no downsample log line
4. Call `clean_pointcloud(pcd, distance_kwargs={"max_distance": 100.0})` — merges to `{"method": "radial", "max_distance": 100.0}`
5. Confirm `clean_pcd` alias in `collab_splats/utils/pointcloud.py` still importable
