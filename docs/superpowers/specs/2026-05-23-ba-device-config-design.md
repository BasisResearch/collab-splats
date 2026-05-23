# BA Device Config & Standalone Refiner Design

**Date:** 2026-05-23
**Scope:** `collab_splats/pointcloud/bundle_adjustment.py`, `collab_splats/pointcloud/wrappers.py`, `tests/pointcloud/test_bundle_adjustment.py`

## Problem

Device placement for bundle adjustment is inconsistent and non-configurable:

1. `extract_tracks_vggsfm` gained a `device` param this session (LC-path bug fix), but `_apply_ba` never passes it — always auto-detects.
2. `run_bundle_adjustment` hardcodes `device = "cuda" if torch.cuda.is_available() else "cpu"` internally — not overridable.
3. `_get_default_solver` ignores device — could pick CuDSS even when `device="cpu"` is requested.
4. `BundleAdjustment` only works as a creator wrapper — no standalone refine path for COLMAP or other non-feedforward inputs.

## Design

### 1. `BundleAdjustmentConfig` — add `device` field

```python
device: "str | None" = None  # None = auto (CUDA if available, else CPU)
```

Single control point. `None` preserves current auto-detect behavior — fully backward compatible.

### 2. `_get_default_solver(device=None)` — device-aware solver selection

```python
def _get_default_solver(device: str | None = None) -> Any:
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver
            return CuDirectSparseSolver()
        except (ImportError, RuntimeError):
            pass
    from bae.utils.pysolvers import PCG
    return PCG()
```

If `device="cpu"`, skips CuDSS entirely (GPU-only solver). Returns bare solver — no wrapper class.

### 3. `run_bundle_adjustment` — accept and use `device`

Add `device: str | None = None` param. Resolve at top of function, replace hardcoded detection. Pass resolved device to `_get_default_solver(device=resolved)`.

### 4. `BundleAdjustment` — optional `base`, new `refine()` method

**Constructor:** `base` becomes optional:
```python
def __init__(
    self,
    base: BaseFeedforwardCreator | None = None,
    *,
    config: BundleAdjustmentConfig = BundleAdjustmentConfig(),
): ...
```

`run_inference` guards with `raise ValueError` if `self.base is None`.

**New `refine()` method:**
```python
def refine(
    self,
    images: np.ndarray | Tensor,
    extrinsics: np.ndarray,        # (N, 3, 4) or (N, 4, 4)
    intrinsics: np.ndarray,        # (N, 3, 3)
    image_size: tuple[int, int],   # (H, W)
    *,
    conf: Tensor | None = None,
    world_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:  # (refined_extrinsics, refined_intrinsics)
```

Calls `extract_tracks_vggsfm` then `run_bundle_adjustment` directly — same logic as `_apply_ba` but no `FeedforwardResult` dependency. `_apply_ba` refactors to delegate to `refine()`.

### 5. `_apply_ba` — delegate to `refine()`, thread `device`

`_apply_ba` extracts `images`, `extrinsics`, `intrinsics`, `conf`, `world_points` from `FeedforwardResult`, then delegates to `refine()`. It handles the feedforward-specific concerns (dedup, pixel reprojection, writing refined poses back to result) that `refine()` doesn't know about.

`refine()` internally:
1. Pops `device`, `max_query_pts`, `query_frame_num` from config
2. Calls `extract_tracks_vggsfm(..., device=device)`
3. Calls `run_bundle_adjustment(..., device=device)`
4. Returns `(refined_extrinsics, refined_intrinsics)`

### 6. Test fix

`test_get_default_solver_prefers_cudss_when_available`: drop `._inner` assertion, replace with `assert solver is mock_cudss_instance` (matches PCG test pattern — bare return, no wrapper).

## Affected Files

| File | Change |
|---|---|
| `collab_splats/pointcloud/bundle_adjustment.py` | `BundleAdjustmentConfig.device`, `_get_default_solver(device)`, `run_bundle_adjustment(device)` |
| `collab_splats/pointcloud/wrappers.py` | `BundleAdjustment.__init__(base=None)`, `BundleAdjustment.refine()`, `_apply_ba` threads device |
| `tests/pointcloud/test_bundle_adjustment.py` | Fix `._inner` assertion |

## Non-Goals

- `fine_tracking` config field — separate concern
- Multi-GPU round-robin — not needed
- Wrapper class for solver — explicit decided-against
