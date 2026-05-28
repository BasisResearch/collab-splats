# Design: Generalise Multiview Depth Confidence to All Feedforward Models

**Date:** 2026-05-28  
**Status:** Approved  
**Branch:** refactor/cu121

---

## Context

MapAnything's `compute_multiview_depth_confidence` (geometric cross-view depth
consistency filter) was previously MapAnything-only, entangled with
`postprocess_model_outputs_for_inference`. Decision 013 established that the filter
adds no quality benefit for VGGT-X/Omega (their learned `depth_conf` head is already
~95% geometrically consistent). However, making the filter available to all models
establishes postprocessing equivalence — callers can apply the same geometric filter
regardless of backend — and removes the mapanything library as a hidden dependency
inside `base.py`.

**Background investigation:** `worklog/decisions/013-multiview-confidence-not-generalised.md`  
**Bug fix context:** `worklog/plans/2026-05-28-multiview-confidence-generalise.md`

---

## Goals

1. Native implementation of `compute_multiview_depth_confidence` in `base.py` — no
   mapanything import required.
2. All three creators (`VGGTXCreator`, `VGGTOmegaCreator`, `MapAnythingCreator`) expose
   `use_multiview_confidence: bool` with consistent semantics.
3. MapAnythingCreator refactored to use the shared function (drop upstream
   `use_multiview_confidence` kwarg from `postprocess_model_outputs_for_inference`).
4. Scale-invariant defaults so the filter works on non-metric VGGT depths.

---

## Algorithm

For each source pixel (u, v) in frame i with depth `d_src`:
1. Unproject to world space via `K_i^{-1}` and cam2world pose.
2. Project into every target frame j → `expected_depth`.
3. Sample frame j's depth map at the projected location → `sampled_depth`.
4. Inlier if: `|expected_depth − sampled_depth| < abs_thresh + rel_thresh × expected_depth`
5. Confidence = inlier_ratio = (number of inlier target frames) / (number of overlapping frames).

Output: `mv_conf ∈ [0, 1]` per pixel. Pixels with no overlapping views → `mv_conf = 0`.

---

## Threshold Design

`abs_thresh` is a scale-dependent floor in depth units. `rel_thresh` is a scale-invariant
fractional tolerance.

| Model        | Depth scale | abs_thresh | rel_thresh |
|-------------|-------------|------------|------------|
| MapAnything | Metric (m)  | 0.02       | 0.02       |
| VGGT-X      | Non-metric  | 0.0        | 0.05       |
| VGGTOmega   | Non-metric  | 0.0        | 0.05       |

With `abs_thresh=0.0`, the filter is fully scale-invariant: `|Δd| / d_expected < rel_thresh`.
MapAnything keeps `abs_thresh=0.02` (calibrated for metres) as an explicit per-creator field.

**Why not percentile:** mv_conf is a quantized inlier ratio (k/N for integer k, N). Most
pixels reach conf=1.0 when views agree closely, so `torch.quantile` collapses to 1.0 for
any percentile where >0% of pixels are 1.0, and the strict `conf > 1.0` check excludes
every pixel. Use `mv_conf > mv_conf_threshold` directly (default `0.0`).

---

## Shared Function

**Location:** `collab_splats/pointcloud/feedforward/base.py`

```python
def compute_multiview_depth_confidence(
    depth: np.ndarray,                      # (N, H, W) float32
    intrinsics: np.ndarray,                 # (N, 3, 3) float32, pixel-units pinhole
    extrinsics: np.ndarray,                 # (N, 4, 4) float32, world-to-cam
    depth_masks: np.ndarray | None = None,  # (N, H, W) bool — exclude zero/invalid pixels
    abs_thresh: float = 0.0,
    rel_thresh: float = 0.05,
    device: str = "cuda",
) -> np.ndarray:                            # (N, H, W) float32 in [0, 1]
    """Geometric cross-view depth consistency confidence per pixel.

    For each source pixel, projects it into all other views and checks whether
    the expected and sampled depths agree within abs_thresh + rel_thresh * expected.
    Returns per-pixel inlier ratio across overlapping views.
    """
```

**Implementation notes:**
- Pure torch, runs on `device`. Accepts and returns numpy; conversion is internal.
- Inverts `extrinsics` (world→cam) to cam2world before projection.
- Uses `torch.nn.functional.grid_sample` for differentiable depth sampling.
- Source pixels with `depth <= 0` or `depth_masks == False` are skipped.
- No dependency on `mapanything` package.

---

## Creator Changes

### `VGGTXCreator` and `VGGTOmegaCreator`

Add two fields:
```python
use_multiview_confidence: bool = False
mv_conf_threshold: float = 0.0
```

Inside `_postprocess`, after building `stacked_depth`, `stacked_intrinsics`,
`stacked_extrinsics` but before `unproject_and_filter_points`:

```python
if self.use_multiview_confidence:
    mv_conf = compute_multiview_depth_confidence(
        stacked_depth, stacked_intrinsics, stacked_extrinsics,
        depth_masks=combined_mask,
        abs_thresh=0.0,
        rel_thresh=0.05,
    )
    combined_mask = combined_mask & (mv_conf > self.mv_conf_threshold)
```

### `MapAnythingCreator` (full refactor)

1. `postprocess_model_outputs_for_inference` called with `use_multiview_confidence=False`
   always (drop the parameter). `apply_confidence_mask` follows `confidence_percentile`
   as before.
2. `pred["conf"]` (upstream mv_conf field) no longer read.
3. After building per-frame masks from `pred["mask"]` and `depth_z > 0`, if
   `self.use_multiview_confidence`:
   ```python
   mv_conf = compute_multiview_depth_confidence(
       stacked_depth, stacked_intrinsics, stacked_extrinsics,
       depth_masks=combined_mask,
       abs_thresh=self.mv_conf_abs_thresh,   # default 0.02 — metric
       rel_thresh=0.02,
   )
   combined_mask = combined_mask & (mv_conf > self.mv_conf_threshold)
   ```
4. New fields on `MapAnythingCreator`:
   ```python
   mv_conf_abs_thresh: float = 0.02   # metric depth default
   mv_conf_threshold: float = 0.0
   ```
   `use_multiview_confidence: bool = True` stays unchanged.

---

## Watch-outs

1. **Extrinsics convention:** `result.extrinsics` is world-to-cam. Always invert before
   passing as `camera_poses`. The shared function handles this internally.
2. **Submap boundary effects:** frames at submap boundaries have fewer overlapping views →
   lower mv_conf. If applying per-submap, consider ±N frame padding.
3. **MapAnything depth scale:** confirmed metric (metres). `abs_thresh=0.02` appropriate.
4. **VGGT depth scale:** non-metric (scene-normalized). `abs_thresh=0.0` required for
   scale invariance.
5. **Zero-depth pixels:** pass `depth_masks` to exclude zero-depth source pixels from
   both source and target comparisons.
6. **Memory:** runs on GPU; for large sequences (>200 frames at 512×512) reduce via
   `minibatch_size` or process per submap.

---

## Tests

- **Unit — `compute_multiview_depth_confidence`:** synthetic 2-frame scene with known
  overlap and planar depth. Assert `mv_conf > 0` at overlapping pixels, `mv_conf == 0`
  at non-overlapping border pixels.
- **VGGTXCreator / VGGTOmegaCreator:** mock `compute_multiview_depth_confidence` → assert
  it is called with correct arrays when `use_multiview_confidence=True`; assert mask is
  AND-combined with mv_conf output.
- **MapAnythingCreator:** assert `postprocess_model_outputs_for_inference` is called with
  `use_multiview_confidence=False` after refactor. Assert `pred["conf"]` no longer read.
  Assert `compute_multiview_depth_confidence` called when `use_multiview_confidence=True`.

---

## Future Work

MapAnything's full `_postprocess` pipeline (confidence percentile, edge masking, non-
ambiguous mask via `postprocess_model_outputs_for_inference`) remains a separate code
path from VGGT-X/Omega's `unproject_and_filter_points` + learned `depth_conf` path.
Unifying these into a single shared postprocessing pipeline — so all models use the same
masking, filtering, and unprojection logic — is out of scope here but is the natural
next step toward full postprocessing equivalence.

---

## Files Modified

| File | Change |
|------|--------|
| `collab_splats/pointcloud/feedforward/base.py` | Add `compute_multiview_depth_confidence` |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Add `use_multiview_confidence`, `mv_conf_threshold` fields; wire in `_postprocess` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Refactor `_postprocess`: drop upstream mv_conf, call shared function |
| `collab_splats/pointcloud/feedforward/__init__.py` | Re-export `compute_multiview_depth_confidence` |
| `tests/pointcloud/feedforward/test_mv_conf.py` | New: unit tests for shared function + creator integration |
