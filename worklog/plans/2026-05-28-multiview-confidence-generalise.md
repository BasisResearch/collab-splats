# Handoff: Generalise Multiview Depth Consistency to All Models

**Date:** 2026-05-28  
**Status:** Ready for implementation  
**Depends on:** decision 013, fix in `collab_splats/pointcloud/feedforward/mapanything.py`

---

## What was done before this

### Decision 013 (2026-05-28)
Evaluated whether MapAnything's `compute_multiview_depth_confidence` (geometric
cross-view depth consistency) adds value for VGGT-X and VGGTOmega.

Result: **No benefit for VGGT-X/Omega.** Their learned `depth_conf` head is already
trained end-to-end and saturated (mv_conf_mean ≈ 0.95). The geometric filter is
redundant. Decision: do not add it.

### MapAnything bug fix (same session)
`MapAnythingCreator(use_multiview_confidence=True)` was producing 0 points.

**Root cause:** `mv_conf = k / N_views` (quantized inlier ratio). With N≈9 overlapping
views, 65%+ of pixels land at conf=1.0. `torch.quantile(conf, 0.35)` = 1.0. The
upstream `conf > 1.0` (strict) = False for every pixel. Percentile-based thresholding
is designed for smooth learned-confidence distributions; it is broken on quantized
inlier ratios.

**Fix applied:** `apply_confidence_mask=False` when `use_multiview_confidence=True`,
then `mv_conf > 0` applied directly in `_postprocess`. Keeps any pixel with geometric
support from ≥1 other view. See `mapanything.py:_postprocess` docstring for full
explanation.

---

## What you are building

A **model-agnostic** multiview depth consistency postprocessing step that can run on
the `FeedforwardResult` of any feedforward creator (VGGT-X, VGGTOmega, MapAnything,
future models). The goal is not to replace learned confidence but to be an optional
additional filter or quality signal — particularly for models that do NOT have a
learned confidence head.

---

## Key technical facts

### `compute_multiview_depth_confidence` API

```python
from mapanything.utils.multiview_confidence import compute_multiview_depth_confidence

mv_conf_list = compute_multiview_depth_confidence(
    depth_z:     List[Tensor(B, H, W, 1)],   # Z-depth per view (metric scale)
    intrinsics:  List[Tensor(B, 3, 3)],        # pinhole intrinsics per view
    camera_poses: List[Tensor(B, 4, 4)],       # cam2world per view
    depth_assoc_abs_thresh: float = 0.02,      # metres — tight for metric depth
    depth_assoc_rel_thresh: float = 0.02,      # relative fraction
    depth_masks: Optional[List[Tensor(B, H, W)]] = None,  # valid-pixel filter
) -> List[Tensor(B, H, W)]   # confidence in [0, 1]
```

Algorithm: for each source pixel, project it into all overlapping target views via
depth-image warping. A pixel is an inlier if `|expected_depth − sampled_depth| <
abs_thresh + rel_thresh * expected_depth`. Confidence = inlier_ratio across views.

### Signal characteristics by model (chess seq-01, 50 frames)

| Model      | mv_conf_mean | Distribution | Learned conf? |
|------------|-------------|--------------|---------------|
| VGGT-X     | 0.954       | Bimodal, concentrated at 1.0 | Yes — saturated |
| VGGTOmega  | 0.938       | Bimodal, concentrated at 1.0 | Yes — saturated |
| MapAnything | ~0.96 internal | Concentrated at 1.0 | No (mv_conf IS the signal) |

All three models produce depths that are already highly geometrically consistent. For
VGGT-X and VGGTOmega, adding mv_conf on top of learned conf removes < 7% additional
points (mostly the same pixels).

### The percentile mechanism is broken for this signal

`torch.quantile(mv_conf, p)` collapses to 1.0 when ≥(100-p)% of pixels are at 1.0.
The strict `conf > 1.0` then excludes everyone. **Do not use percentile thresholding
on mv_conf.** Use a fixed absolute threshold instead. `mv_conf > 0` (any geometric
support) is the correct semantic for a "has this depth been verified?" filter.

### `FeedforwardResult` already has the fields you need

```python
result.depth        # (N, H, W) float32 — Z-depth per frame
result.intrinsics   # (N, 3, 3) float32 — pinhole intrinsics
result.extrinsics   # (N, 4, 4) float32 — world-to-camera (invert to cam2world)
result.world_points # (N, H, W, 3) float32 — 3D points in world frame
result.confidence   # (N, H, W) float32 — learned confidence (None for MapAnything)
```

`result.extrinsics` is world-to-camera. Use `invert_poses(extrinsics)` (from
`collab_splats.utils.geometry`) to get cam2world for `camera_poses` argument.

### Memory / performance

50 frames, 392×518 = 10.2M pixels total. `compute_multiview_depth_confidence` is
O(N²) in view pairs and O(H×W) per pair. Uses `frustum_chunk_size=50` to batch the
frustum intersection matrix. On A40 (44GB): 50 frames ≈ 2–3 minutes. For 500 frames,
run in submaps (≤50 frames each) and aggregate.

---

## Suggested implementation

### Where to put it

`collab_splats/pointcloud/feedforward/base.py` — add a standalone function:

```python
def compute_mv_conf_from_result(
    result: FeedforwardResult,
    depth_masks: Optional[np.ndarray] = None,  # (N, H, W) bool
    abs_thresh: float = 0.02,
    rel_thresh: float = 0.02,
    device: str = "cuda",
) -> np.ndarray:  # (N, H, W) float32 confidence in [0, 1]
```

This decouples the consistency computation from any specific creator and works on
any `FeedforwardResult`. Callers can then threshold however they want.

### Filtering: use `mv_conf > threshold` not percentile

Recommended defaults:
- `mv_conf > 0`: keep any pixel with ≥1 inlier view. Permissive, handles degenerate
  scenes (few overlapping views) gracefully.
- `mv_conf >= 0.5`: majority-vote filter. More conservative, removes pixels where
  fewer than half of overlapping views agree.

Avoid percentile — see above.

### Integration with creators

For VGGT-X/VGGTOmega: expose `use_multiview_confidence: bool = False` in their
constructors. When True, AND the learned `depth_conf` with mv_conf > 0 in the mask
building loop (decision 013 says pure mv_conf adds nothing; learned AND mv is a
legitimate quality gate if you want belt-and-suspenders).

For MapAnything: already fixed. `use_multiview_confidence=True` (default) applies
`mv_conf > 0` on top of `non_ambiguous_mask + edge_mask`.

### Watch out for

1. **result.depth has zeros at masked pixels** — `valid_src_depth = depth_z > 0` in
   `compute_multiview_depth_confidence` handles this correctly (skips zero-depth
   source pixels). But zeros in the TARGET depth map mean sampled_depth=0 →
   outlier. Consider passing a `depth_masks` boolean array to exclude zero-depth
   pixels from BOTH source and target comparisons.

2. **Intrinsics convention** — `result.intrinsics` is in pixel units (fx, fy, cx, cy
   in image pixel coordinates). This is correct for `compute_multiview_depth_confidence`.

3. **cam2world vs world2cam** — `result.extrinsics` is world-to-camera. The function
   expects cam2world. Always invert before passing.

4. **Submap boundary effects** — if you apply this per submap, frames at submap
   boundaries have fewer overlapping views → lower mv_conf. Consider running across
   the full sequence or padding submaps by ±N frames.

5. **MapAnything depth scale** — confirmed metric (no scale mismatch). The 0.02m
   abs_thresh is appropriate.

---

## Files to read

- `worklog/decisions/013-multiview-confidence-not-generalised.md` — original decision
- `evals/eval_multiview_conf.py` — existing eval + `--diagnose` flags
- `collab_splats/pointcloud/feedforward/mapanything.py:_postprocess` — reference
  implementation of the `mv_conf > 0` pattern
- `collab_splats/pointcloud/feedforward/base.py:FeedforwardResult` — output dataclass
- `collab_splats/utils/geometry.py:invert_poses` — pose inversion utility
