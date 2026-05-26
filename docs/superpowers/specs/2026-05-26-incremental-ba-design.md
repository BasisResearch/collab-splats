# Incremental Bundle Adjustment — Design Spec

**Date:** 2026-05-26
**Branch:** refactor/cu121
**Status:** approved

---

## Problem

Current `BundleAdjustment.refine()` runs a single LM solve on all N frames simultaneously, initialized from raw feedforward poses. Large systems are poorly conditioned and susceptible to bad local minima. The last measured all-at-once BA result on 7-Scenes chess seq-01 (50 frames) was ATE 0.0577m vs baseline 0.0553m — BA made alignment worse. `bundle_adjustment.py` has since been modified; current performance is unknown and must be re-evaluated before implementation begins.

**Hypothesis:** Incrementally integrating frames into the optimization — growing the registered set from `add_size` to N, re-running global BA each step with warm initialization from the previous step — gives each LM solve a better starting point and better conditioning, producing a lower final minimum than one-shot all-at-once BA.

---

## Scope

- Incremental BA via `add_size` config field on existing `BundleAdjustment`
- Track zarr cache (shared across all BA/LC pipelines, no redundant VGGSfM inference)
- Intrinsics scaling: compute once, reuse across all k-steps
- Per-step loss recording
- New `incremental_ba` eval condition in `eval_gt.py` + `eval_suite.sh`

Out of scope: composability with LC per-submap (separate spec if needed).

---

## Architecture

### Config change

```python
@dataclass
class BundleAdjustmentConfig:
    # ... existing fields unchanged ...
    add_size: int = 0                      # 0 or >= N → all-at-once (current behavior); 1..N-1 → incremental
    tracks_cache_dir: Path | None = None   # zarr cache dir; None = no caching (always extract)
```

### Dispatch in `refine()`

```python
def refine(self, result: FeedforwardResult) -> FeedforwardResult:
    N = len(result.images)
    tracks, vis_scores, pts3d_tracks = self._load_or_extract_tracks(result)
    intrinsics_model, sx, sy, tl_x, tl_y = _scale_intrinsics_to_model(
        result.intrinsics, result.images, result.original_coords
    )
    add_size = self.config.add_size
    if add_size == 0 or add_size >= N:
        refined_extrinsics, refined_intrinsics_model = self._refine_allonce(
            result, tracks, vis_scores, pts3d_tracks, intrinsics_model
        )
    else:
        refined_extrinsics, refined_intrinsics_model = self._refine_incremental(
            result, tracks, vis_scores, pts3d_tracks, intrinsics_model, add_size
        )
    # rescale + build updated FeedforwardResult (shared, unchanged)
    ...
```

`_refine_allonce()` = current `refine()` body, no behavior change.

### `_refine_incremental()`

```
init:
  refined_extrinsics  ← feedforward extrinsics  (N, 3, 4)
  refined_intrinsics  ← scaled intrinsics        (N, 3, 3)
  refined_pts3d       ← None

for k in [add_size, 2*add_size, ..., N]:
    k = min(k, N)
    # pts3d_tracks[:k] re-averaged each step — landmark set L_k grows with k,
    # so refined_pts3d from step k-1 cannot be directly reused (different L).
    # Warm initialization is applied to extrinsics/intrinsics only (the main benefit).
    refined_pts3d_k, refined_ext_k, refined_intr_k = self._optimize(
        pts3d_tracks[:k],
        refined_extrinsics[:k],   ← warm from previous step
        refined_intrinsics[:k],   ← warm from previous step
        tracks[:k],
        vis_scores[:k],
    )
    refined_extrinsics[:k]  = refined_ext_k
    refined_intrinsics[:k]  = refined_intr_k
    self._last_loss_history.append(step_losses)
```

### `_optimize()` return change

Currently: `(_, extrinsics, intrinsics)` — `_` discarded.
Change to: `(pts3d, extrinsics, intrinsics)` — expose refined landmark positions for diagnostics. pts3d is NOT threaded between incremental steps because landmark set size L_k grows with k (different shape per step); threading would require a P-space remapping. Warm initialization is applied to extrinsics and intrinsics only.

### `_last_loss_history` type change

`list[float]` → `list[list[float]]` — one inner list per k-step, each containing per-LM-iteration losses. Enables convergence diagnostics across the full incremental sequence.

---

## Track Zarr Cache

VGGSfM track extraction is expensive. Cache to disk so any pipeline (BA, LC, incremental BA) reuses extracted tracks without re-running inference.

**Location:** `{output_dir}/tracks.zarr/`

**Arrays stored:**
- `tracks`: `(N, P, 2)` float32
- `vis_scores`: `(N, P)` float32
- `pts3d_tracks`: `(N, P, 3)` float32

**Metadata group** (cache key):
- `image_paths`: sorted list of image path strings
- `max_query_pts`: int
- `query_frame_num`: int
- `fine_tracking`: bool

**Invalidation:** on load, recompute hash of metadata fields. If mismatch → log warning, delete zarr, re-extract and re-save.

**Interface:** new private method `_load_or_extract_tracks(result) -> (tracks, vis_scores, pts3d_tracks)`. Uses `config.tracks_cache_dir` for cache path. Falls back to always-extract if `tracks_cache_dir` is `None`.

---

## Intrinsics Scaling Cleanup

`_scale_intrinsics_to_model()` computes `(sx, sy, tl_x, tl_y)` — called once before the k-loop, factors reused at each step. Previously this computation was implicitly repeated per call; now explicit and shared.

The scaling remains necessary: VGGSfM tracks are in model-resolution pixel space; `result.intrinsics` is stored at original-image resolution. These must match for reprojection error filtering.

---

## Error Handling

| Condition | Behavior |
|---|---|
| `add_size == 0` or `add_size >= N` | Fall back to all-at-once, log debug |
| NaN poses or loss increase at step k | Log warning, keep step k−1 refined state, continue |
| Track cache missing | Extract and save |
| Track cache hash mismatch | Log warning, delete, re-extract, re-save |

---

## Eval Integration

### `eval_gt.py`
Add `incremental_ba` condition using `BundleAdjustment(BundleAdjustmentConfig(add_size=<cli_arg>))`. CLI: `--incremental_ba_add_size` (default 5).

### `eval_suite.sh`
Add `incremental_ba` to the condition sweep.

### Pre-implementation gate
**Before writing any code:** run `eval_gt.py --conditions baseline ba` on chess seq-01 to establish current BA performance with the modified `bundle_adjustment.py`. This is the baseline the incremental BA implementation must beat.

If incremental BA also underperforms baseline after implementation, the problem is likely track quality or LM configuration — not BA architecture — and needs separate investigation.

---

## Testing

All tests flat functions in `tests/pointcloud/test_bundle_adjustment.py`.

- `test_incremental_ba_add_size_n_matches_allonce` — `add_size=N` produces equivalent result to `add_size=0`
- `test_incremental_ba_warm_start_updates_all_registered_frames` — after step k, `extrinsics[:k]` differ from feedforward init
- `test_incremental_ba_loss_history_length` — `_last_loss_history` has one inner list per k-step
- `test_tracks_cache_save_load` — save zarr, reload, verify arrays equal
- `test_tracks_cache_invalidates_on_config_change` — change `query_frame_num`, verify cache miss + re-extract

---

## Reprojection

The incremental loop is agnostic to reprojection. `creator.reproject(result)` is called once after the full incremental sequence completes — identical to current all-at-once BA behavior.

**Terminology:** `pts3d_tracks` = VGGSfM sparse 3D estimates used internally by the BA solver (shape `(N, P, 3)`). `result.points` = `FeedforwardResult` dense point cloud field (populated by `reproject()`). These are distinct; the incremental loop only touches `pts3d_tracks`.

---

## File Changes Summary

| File | Change |
|---|---|
| `collab_splats/pointcloud/bundle_adjustment.py` | `add_size` config field; `_load_or_extract_tracks()`; `_refine_incremental()`; `_optimize()` exposes pts3d return; `_last_loss_history` type change; `_refine_allonce()` extracted |
| `evals/eval_gt.py` | `incremental_ba` condition + `--incremental_ba_add_size` CLI arg |
| `evals/eval_suite.sh` | `incremental_ba` condition added to sweep |
| `tests/pointcloud/test_bundle_adjustment.py` | 5 new tests above |
