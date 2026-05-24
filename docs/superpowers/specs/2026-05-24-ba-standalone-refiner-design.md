# BundleAdjustment Standalone Refiner Design

**Date:** 2026-05-24
**Scope:** `collab_splats/pointcloud/bundle_adjustment.py`, `collab_splats/pointcloud/wrappers.py`, `tests/pointcloud/test_bundle_adjustment.py`, `tests/pointcloud/test_wrappers.py`

## Problem

`BundleAdjustment` currently inherits `BasePointcloudCreator` — a leaky abstraction. BA refines poses; it doesn't create point clouds. This forces it to carry `reconstruct`, `build_colmap`, and `self.base` (a creator) as coupling points. It cannot be used without a feedforward creator, even though its core operation (track extraction + LM optimization) is method-agnostic.

Additionally, device placement is not configurable: each function auto-detects independently, and the LC windowed path silently ran the VGGSfM tracker on CPU (bug fixed this session for `extract_tracks_vggsfm`, but not yet threaded through via config).

## Goals

1. `BundleAdjustment` is a **standalone refiner** — not a `BasePointcloudCreator` subclass.
2. Works with any source of poses: VGGTX, MapAnything, COLMAP, manual construction.
3. Primary interface: `ba.run(ff_result) -> FeedforwardResult` — takes and returns `FeedforwardResult`.
4. Device is configurable via `BundleAdjustmentConfig.device`.
5. Migration path for existing call sites.

## Design

### `BundleAdjustmentConfig` — add `device`

```python
@dataclass
class BundleAdjustmentConfig:
    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048
    query_frame_num: int = 5
    device: "str | None" = None  # None = auto (CUDA if available, else CPU)
```

### `_get_default_solver(device=None)` — device-aware

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

`device="cpu"` skips CuDSS. Returns bare solver (no wrapper class).

### `run_bundle_adjustment` — accept `device`

Add `device: str | None = None`. Resolve at top; replace hardcoded detection. Pass to `_get_default_solver(device=resolved)`.

### `BundleAdjustment` — plain class, not `BasePointcloudCreator`

```python
class BundleAdjustment:
    """Refines camera poses and 3D points via VGGSfM track extraction + LM BA.

    Method-agnostic: works with any FeedforwardResult regardless of source.
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        self.config = config or BundleAdjustmentConfig()

    def run(self, result: FeedforwardResult) -> FeedforwardResult:
        """Refine poses in-place; return updated FeedforwardResult."""
        ...
```

No `self.base`. No `reconstruct`. No `build_colmap`. No `load_model`.

### `run()` implementation

Extracts from `FeedforwardResult`: `images`, `extrinsics`, `intrinsics`, `conf`, `world_points`, `model_height`, `model_width`. Calls `extract_tracks_vggsfm` + `run_bundle_adjustment` with `device` from config. Returns `dataclasses.replace(result, extrinsics=refined_ext_4x4, intrinsics=refined_intr)`.

Also handles pixel reprojection when `result.pixel_indices` is set (existing `_reproject_ba` logic).

### Creator pipeline integration

Creators (`VGGTXCreator`, `MapAnythingCreator`) gain an optional `bundle_adjustment: BundleAdjustmentConfig | None = None` field. When set, `_postprocess` calls `BundleAdjustment(self.bundle_adjustment).run(result)` before returning. This keeps BA opt-in per-creator without a wrapper class.

Alternatively (simpler): callers compose explicitly:

```python
creator = VGGTXCreator(...)
ba = BundleAdjustment(config=BundleAdjustmentConfig(device="cuda"))

ff_result = creator.run(image_dir)
refined   = ba.run(ff_result)
colmap    = creator.build_colmap(output_dir, result=refined)
```

### Migration — existing `BundleAdjustment(creator, config)` usage

Current pattern: `BundleAdjustment(creator, config=cfg).reconstruct(image_dir, output_dir)`

Migration: remove `BundleAdjustment` wrapper entirely from these call sites. Callers use the creator directly and call `ba.run()` after postprocess. The old instance `refine(result, output_dir)` is removed; `reconstruct` is removed.

Existing `wrappers.py` `BundleAdjustment` class is **replaced** by the new plain class.

### `LoopClosure` integration

`LoopClosure` currently wraps a base creator and optionally wraps `BundleAdjustment`. After this refactor, `LoopClosure` holds an optional `BundleAdjustment` instance and calls `ba.run(result)` at the appropriate stage of the LC pipeline. No inheritance change needed for `LoopClosure`.

## Non-Goals

- `fine_tracking` config field — separate concern
- Multi-GPU round-robin
- Solver wrapper class

## Affected Files

| File | Change |
|---|---|
| `collab_splats/pointcloud/bundle_adjustment.py` | `BundleAdjustmentConfig.device`, `_get_default_solver(device)`, `run_bundle_adjustment(device)`, `BundleAdjustment` plain class with `run()` |
| `collab_splats/pointcloud/wrappers.py` | Remove `BundleAdjustment` class (replaced); update `LoopClosure` to hold `BundleAdjustment` instance |
| `tests/pointcloud/test_bundle_adjustment.py` | Fix `._inner` test, add `run()` tests |
| `tests/pointcloud/test_wrappers.py` | Update for new composition pattern |
| `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` | Update usage to new API |

## Open Questions

- Does `creator.build_colmap(output_dir, result=refined)` exist? If not, needs adding or the colmap step stays on the creator's `reconstruct`.
- `LoopClosure` currently stores merged `raw_outputs` on `self.base` — verify the BA call site in LC can access `FeedforwardResult` fields after this refactor.
