# Localization 2D→3D via per-frame depth lookup (pose_from_cluster parity)

**Date:** 2026-07-22
**Status:** Design — awaiting review
**Scope:** `collab_splats/localization/` (localizer, extractors, viz) + two load-site wiring changes.

## Problem

The tutorial query image fails to localize against the tutorial scene. XFeat fails
outright; **every** matcher (Disk, XFeat, LoMa, LoMa-G) fails to recover a pose. The
matchers are not the cause — the shared 2D→3D correspondence stage starves them.

## Root cause

`CameraLocalizer` recovers a 3D point per keypoint with `_build_frame_assignments`
(`localizer.py:80`), which does a **lossy reverse-projection**:

1. Project the **global, flattened** point cloud (`FeedforwardResult.points`, `(P,3)`)
   into each reference frame.
2. A keypoint keeps a 3D point only if a projected point lands within `radius=8.0` px,
   one-to-one (competing keypoints discarded, `localizer.py:150`).

The global cloud is merged and **subsampled** (the LC-OOM `max_points` cap). Projected
density is sparse, so most keypoints get **no** 3D point:

```
100 matches → ~15 keypoints within 8px of a projected point → <4 survive PnP → pose=None
```

This is not how standard image→3D localization works. hloc/InLoc `pose_from_cluster`
(`localize_inloc.py`) never reverse-projects a global cloud — it samples the reference
frame's **dense XYZ map at the matched pixel** (`interpolate_scan`, `grid_sample`), so
match yield ≈ correspondence yield.

## Goal

Recover pose robustly for the tutorial query (and in general) by mirroring
`pose_from_cluster`: per matched pair `(query_px, ref_px)`, look up 3D by sampling the
reference frame's **dense per-frame `world_points`** at `ref_px`. Feedforward already
produces and persists `world_points (N,H,W,3)` in world space — the direct analog of
InLoc's XYZ scan, and already world-space so no pose multiply is needed.

Secondary asks folded in:
- Add an **XFeat\*** (semi-dense) matcher option alongside sparse XFeat, configured per
  the accelerated_features `xfeat_matching` notebook.
- **Decouple** `LocalizationResult` from the reusable helpers (`plot_correspondences`
  and any viz): package functions take plain arrays / core objects, not our dataclass.

## Non-goals

- Retrieval gating (localize stays exhaustive over all reference frames).
- Changing the PnP backend (`pycolmap.estimate_and_refine_absolute_pose` stays).
- Perf tuning of exhaustive XFeat\* matching (acceptable for tutorial-scale scenes).

## Design

### 1. Correspondence model — carry pixels, look up depth

The unifier is the **3D-lookup stage**, not the match output format. Two changes:

**Match contract → pixel pairs.** `BaseLocalExtractor.match()` returns matched
**pixel coordinates**, not indices into cached keypoints:

```python
@dataclass
class MatchResult:
    query_px: np.ndarray   # (K, 2) float32 xy in query image
    ref_px:   np.ndarray   # (K, 2) float32 xy in reference image
    scores:   np.ndarray | None = None  # (K,) optional match confidence, for ordering
```

- Sparse extractors (Disk, XFeat, LoMa, LoMa-G) fill `query_px = q.keypoints[idx_q]`,
  `ref_px = db.keypoints[idx_db]` — they keep their existing index-based internals; only
  the *return* changes. Keypoints are already pixel coords (`LocalFeatures.keypoints`),
  so this is a lookup, not a rework.
- XFeat\* fills refined subpixel coords directly (`match_xfeat_star` returns
  `(x1,y1,x2,y2)` per pair; no stable index exists, which is why the pixel contract is
  the right common shape).

**3D lookup replaces `_build_frame_assignments`.** New helper, the `interpolate_scan`
analog:

```python
def sample_world_points(world_points: np.ndarray,  # (H, W, 3) world-space
                        px: np.ndarray              # (K, 2) xy
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-sample per-pixel world points at px. Returns (pts3d (K,3), valid (K,)).
    Invalid where the sample is NaN or zero-depth (unmapped pixels)."""
```

`localize()` then, per reference frame `f` with matches `MatchResult`:
1. `pts3d, valid = sample_world_points(world_points[f], match.ref_px)`
2. keep valid rows → `(query_px → pts3d)` correspondences, tagged with frame index `f`.
3. Concatenate across frames → pycolmap RANSAC PnP (unchanged, `max_error=50`).

**Deleted:** `_build_frame_assignments`, the `radius` constructor param, `self._assignments`,
and the index/dedup bookkeeping at `localizer.py:884-896`. RANSAC absorbs redundant
correspondences; optional light dedup by `scores` per world-point cell may be kept if a
regression shows duplicate inflation (decide during implementation, default: none).

### 2. Plumbing per-frame world_points

- `CameraLocalizer.__init__` / `from_feedforward` gain `world_points: np.ndarray (N,H,W,3)`.
  `from_feedforward` sources it from `result.world_points`.
- Load sites set `load_world_points=True`:
  - `wrapper/reconstructor.py:376` (`_build_localization_db`)
  - `webapp/routers/localize.py` ff load
  - `dashboard/pipeline.py` ff load (if it reads zarr for the localizer)
- The zarr already stores `world_points` (`pointcloud/feedforward/base.py:158`); no
  persistence change.
- **Memory:** the localizer holds `(N,H,W,3)` float32 in RAM (same array feedforward
  already loads). Fine at tutorial scale. Note for a future retrieval-gating follow-up:
  lazy per-frame zarr reads for matched frames only. Not in this scope.

### 3. Extractors — XFeat sparse + XFeat\* semi-dense

Two **separately registered** extractors (registry is name-based; contracts differ):

- `xfeat` — existing sparse path (`detectAndCompute` + `match_lighterglue`), unchanged
  except `match()` now returns `MatchResult`. Configurable `top_k`, `min_cossim` per the
  notebook.
- `xfeat-star` — new. `extract()` caches per-frame **dense** features
  (`detectAndComputeDense` → keypoints, descriptors, `scales`); `match()` runs the star
  match + `refine_matches` and returns refined `MatchResult`. Configurable `top_k` per
  the notebook.

`LocalFeatures` gains an optional `scales: torch.Tensor | None = None` field for the
dense path. The zarr feature cache (CSR layout) stores whatever `LocalFeatures` carries;
add `scales` to the persisted arrays when present.

> Decision (recommend): two extractors, not one `mode=` flag — keeps each `match()`
> single-purpose and the registry names explicit (`--extractor xfeat-star`).

### 4. Decouple LocalizationResult from helpers

`LocalizationResult` stays as the localizer's typed output, but **reusable functions
must not require it**:

- `plot_correspondences` signature → plain arrays:
  ```python
  plot_correspondences(query_image, ref_image,
                       query_px, ref_px,          # (M,2) each
                       inlier_mask=None,          # (M,) bool | None
                       max_pairs=200, warp_corners=False, show=True)
  ```
- Add a thin adapter so result-specific slicing lives outside the plotting fn:
  ```python
  def correspondences_for_ref(loc, ref_idx):
      """Return (query_px, ref_px, inlier_mask) for one reference frame from a
      LocalizationResult-shaped object. Accepts anything exposing pts2d / pts2d_ref
      / ref_frame_indices / inlier_mask (duck-typed, not isinstance-gated)."""
  ```
  Callers: `dashboard/localize.py:750` and nb07 become
  `plot_correspondences(q_img, ref_img, *correspondences_for_ref(loc, ri))`.
- Audit `viz.py` for any other `loc.`-typed access; convert to array params. Any other
  helper that today takes `LocalizationResult` gets the same treatment (array/core-object
  in, our dataclass only at the localizer boundary).

## Data flow (after)

```
query image ─ extract ─▶ query LocalFeatures
                              │
       for each ref frame f:  ▼
   match(query, db[f]) ─▶ MatchResult(query_px, ref_px)
                              │
   sample_world_points(world_points[f], ref_px) ─▶ (pts3d, valid)
                              │
   accumulate (query_px[valid] → pts3d[valid], frame=f)
                              ▼
   pycolmap.estimate_and_refine_absolute_pose ─▶ pose (4×4 world-to-camera)
```

## Error handling

- `sample_world_points`: NaN / zero-depth → dropped via `valid` mask (mirrors
  `interpolate_scan`'s NaN handling).
- `< 4` valid correspondences total → `pose=None` (existing gate, unchanged).
- Empty match → `MatchResult` with zero-length arrays; frame contributes nothing.
- XFeat\* returning zero refined matches → same zero-length path.

## Testing

- `sample_world_points`: known `(H,W,3)` grid, integer + subpixel `px`, NaN cells →
  correct 3D + valid mask.
- `match()` returns pixel-aligned `MatchResult` for each extractor (query_px/ref_px map
  back to detected keypoints for sparse).
- **Regression (the bug):** on the tutorial scene + query, correspondence count and
  inliers jump from the failing baseline to a passing localization; assert `pose is not
  None` and `n_inliers ≥ threshold`.
- `xfeat-star` extractor: extract caches dense feats (incl. scales); match yields
  refined pixel pairs.
- `plot_correspondences` array-based signature + `correspondences_for_ref` adapter;
  update `test_viz_correspondences.py`, `test_viz_distribution.py`, and the
  `dashboard/test_localize_page.py` monkeypatch.
- Decoupling test: `plot_correspondences` callable with plain numpy arrays and **no**
  `LocalizationResult` import.

## Implementation principles

- **Reuse:** PnP path, RANSAC config, zarr feature cache, `seed_intrinsics`, extractor
  registry all stay. New code is `sample_world_points` (small) + `MatchResult` + the
  `xfeat-star` extractor.
- **Retire:** delete `_build_frame_assignments`, `radius`, `self._assignments`, and the
  index-dedup block — obsoleted by depth lookup. No shim, no dead branch.
- **Flexible:** helpers take arrays/core objects; `LocalizationResult` appears only at the
  localizer boundary.
- **No overengineering:** no retrieval, no lazy zarr sampling, no perf work this pass —
  logged as follow-ups.

## Affected files

- `localization/localizer.py` — `sample_world_points`, correspondence loop, ctor +
  `from_feedforward` world_points param, delete assignment code.
- `localization/extractors.py` — `MatchResult`, `match()` returns pixels, `xfeat-star`
  extractor, `LocalFeatures.scales`.
- `localization/viz.py` — array-based `plot_correspondences` + `correspondences_for_ref`.
- `wrapper/reconstructor.py`, `webapp/routers/localize.py`, `dashboard/pipeline.py` —
  `load_world_points=True` + pass `world_points`; update `plot_correspondences` call.
- Tests + nb07 call sites as above.

## Open questions

1. Keep optional score-based dedup of duplicate world-point hits, or rely on RANSAC?
   Default: rely on RANSAC; revisit only if a regression shows inflation.
2. XFeat\* cache footprint (dense feats per frame) vs sparse — acceptable at tutorial
   scale; flag if a larger scene test balloons the zarr.
