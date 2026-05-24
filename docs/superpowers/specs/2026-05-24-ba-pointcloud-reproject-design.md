# BA Pointcloud Reprojection Fix — Design

**Date:** 2026-05-24
**Branch:** refactor/cu121

## Problem

`BundleAdjustment.refine(result)` updates `extrinsics` and `intrinsics` only.
`pts3d` is explicitly not updated (by design — the refiner is pose-only).
Three bugs result from the `ba-module-cleanup` refactor that turned `BundleAdjustment`
from a wrapper class into a standalone refiner:

1. **Wrong import in `eval_gt.py`:** imports `BundleAdjustment` from `wrappers` — not there.
2. **API mismatch in `eval_gt.py`:** calls `BundleAdjustment(base_creator, config=cfg)` (old
   wrapper constructor); new class takes `(config=None)` only.
3. **pts3d never reprojected after BA:** old wrapper called `reproject_pixels()` internally;
   new standalone class documents "call `creator.reproject(result)` after" but no caller does,
   and `BaseFeedforwardCreator.reproject()` requires a loaded creator in memory.

## Goals

- Fix the import and API breakage in `eval_gt.py`.
- Provide a creator-free reprojection path so callers don't need to hold a live creator.
- Follow existing codebase patterns (active-record `FeedforwardResult`, top-level imports,
  block comments, one-line docstrings).

## Non-Goals

- No new wrapper class for `BundleAdjustment`.
- No changes to `BundleAdjustment.refine()` itself.
- No changes to `wrappers.py`.

## Design

### 1. `FeedforwardResult.reproject()` — new instance method

Location: `collab_splats/pointcloud/feedforward/base.py`

`FeedforwardResult` is already an active record (`save`, `load`, `save_zarr`, `load_zarr`).
All inputs needed for reprojection (`depth`, `pixel_indices`, `extrinsics`, `intrinsics`)
are fields — the method belongs on the object.

Add `reproject_pixels` to the existing top-level import:

```python
from ..utils import colmap_reconstruction_to_result, cross_frame_attention_ratio, reproject_pixels
```

Add method to `FeedforwardResult`:

```python
def reproject(self) -> "FeedforwardResult":
    """Re-project pts3d under current extrinsics using stored source pixels and depth."""
    if self.depth is None or self.pixel_indices is None:
        raise ValueError(
            "reproject() requires depth and pixel_indices; load via load_zarr() "
            "or ensure the creator's _postprocess populated both fields."
        )
    # Reproject stored source pixels under new extrinsics — deterministic,
    # point set stays index-aligned with colors and features.
    pts3d = reproject_pixels(
        self.depth, self.pixel_indices,
        self.extrinsics[:, :3, :], self.intrinsics,
    )
    return replace(self, pts3d=pts3d)
```

Colors are unchanged — source pixels don't move, only world positions shift.

`reproject_pixels` is a pure function (no side effects, no creator state). No circular
dependency: `feedforward/base.py` already imports from `..utils`; `utils.py` does not
import from `feedforward/`.

Fluent API: `ba.refine(result).reproject()`

### 2. `eval_gt.py` — fix import + refactor `_make_creator` / `_run_condition`

**Fix import (line 43):**

```python
# Before (broken):
from collab_splats.pointcloud.wrappers import BundleAdjustment, LoopClosure

# After:
from collab_splats.pointcloud import BundleAdjustment, BundleAdjustmentConfig
from collab_splats.pointcloud.wrappers import LoopClosure
```

**Refactor `_make_creator`** to return `(creator, ba_config | None)`:

```python
def _make_creator(condition, submap_size=None):
    base = get_creator("vggtx")()
    if condition == "lc":
        return LoopClosure(base), None
    m = re.fullmatch(r"ba_track-density-(\d+)", condition)
    if m:
        n = int(m.group(1))
        cfg = BundleAdjustmentConfig(max_query_pts=n, query_frame_num=max(5, n // 512))
        if submap_size is not None:
            windowed = LoopClosure(base, config=LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0))
            return windowed, cfg
        return base, cfg
    if submap_size is not None:
        windowed = LoopClosure(base, config=LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0))
        if condition == "ba":
            return windowed, BundleAdjustmentConfig()
        return windowed, None  # baseline
    if condition == "ba":
        return base, BundleAdjustmentConfig()
    return base, None  # baseline
```

**Refactor `_run_condition`** to use standalone BA flow:

```python
def _run_condition(name, image_dir, output_dir, submap_size=None):
    creator, ba_cfg = _make_creator(name, submap_size=submap_size)
    if ba_cfg is None:
        creator.reconstruct(image_dir, output_dir)
    else:
        # Run inference, refine poses with BA, reproject pts3d, then write COLMAP output
        creator.load_model()
        creator.setup_inference(image_dir)
        creator.run_inference()
        creator.postprocess()
        ba = BundleAdjustment(ba_cfg)
        creator.outputs = ba.refine(creator.outputs).reproject()
        creator.build_colmap(output_dir)
    if creator.outputs is None:
        raise RuntimeError(f"Condition '{name}' produced no outputs")
    return creator.outputs.extrinsics
```

### 3. `BaseFeedforwardCreator.reproject()` — docstring update

Add note: "Prefer `result.reproject()` when `depth` and `pixel_indices` are populated — no creator state needed, deterministic point set."

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward/base.py` | Add `reproject_pixels` import; add `FeedforwardResult.reproject()` |
| `evals/eval_gt.py` | Fix import; refactor `_make_creator` → returns tuple; refactor `_run_condition` |
| `collab_splats/pointcloud/feedforward/base.py` | Update `BaseFeedforwardCreator.reproject()` docstring |

## Testing

- Existing `tests/pointcloud/test_bundle_adjustment.py` — run to confirm no regressions.
- Add `tests/pointcloud/test_feedforward_reproject.py`:
  - `test_reproject_updates_pts3d` — mock `reproject_pixels`, confirm pts3d replaced.
  - `test_reproject_raises_without_depth` — `depth=None` raises `ValueError`.
  - `test_reproject_raises_without_pixel_indices` — `pixel_indices=None` raises `ValueError`.
  - `test_reproject_fluent_chain` — `ba.refine(result).reproject()` returns correct shape.
- Smoke test `eval_gt.py --conditions ba` on chess seq-01 subset to confirm no crash.
