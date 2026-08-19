# BA track-quality parity — design

**Date:** 2026-08-19
**Status:** approved (brainstormed in-session)
**Follows:** `2026-08-18-ba-pipeline-wiring-design.md` (§Validation measured BA hurting sub-cm baselines)

## Problem

The 2026-08-19 chess/seq-01 sweep showed BA *worsens* ATE on every sub-cm baseline
(vggtx/omega/loger +28–72%) and helps only the weakest (mapanything −10%). A line-by-line
audit against upstream `zitongzhan/vggt demo_colmap.py` (the reference bae usage) found the
LM optimizer core matches exactly — the divergence is entirely in **track quality and
observation filtering**. BA faithfully converges onto noisier observations than upstream
ever feeds it.

## Audit findings (ranked)

1. **No visibility threshold.** `_optimize` does `vis = vis_scores.astype(bool)` — any
   score > 0 passes. Upstream gates `pred_vis_scores > 0.2` before BA. Occluded/drifted
   tracks with reproj error < 4 px pull poses systematically.
2. **`fine_tracking` forced off.** `_extract_tracks_vggsfm` hardcodes `fine_tracking=False`.
   Upstream (and the installed `predict_tracks` default) always runs the fine refinement
   stage. Coarse-only tracks carry ~1–2 px error vs ~0.3–0.5 px fine; at f≈300 px, Z≈2 m,
   1 px correlated error ≈ 6–7 mm pose error — the size of the measured regression.
3. **Track density.** Ours 2048 query pts / 5 query frames; upstream demo 4096 / 8.
4. **Per-frame focal is model noise, not signal.** Measured fx spread across frames of one
   video: vggtx 1.2%, omega **11%**, loger 0%. All scenes are one physical camera.
   `shared_camera=False` (current default) lets BA absorb per-frame pose error into
   per-frame focal. `shared_camera=True` is physically correct and upstream-supported.
5. **Filter-order gap.** We check the ≥2-observation landmark rule *before* dropping
   under-inlier frames; upstream drops frames first, then counts. Our order can retain
   single-observation landmarks (depth-unconstrained along the ray).

Out of scope (deliberate): 1024-square track extraction from original frames (chess source
is 640×480 — upstream would *upsample*; revisit for high-res scenes), focal freezing
(departs from upstream; confounds the parity test), windowed BA (`images` gap, known).

## Design

All changes in `collab_splats/geometry/bundle_adjustment.py`. Config surface stays
`BundleAdjustmentConfig` dataclass defaults — pipeline (`reconstructor.py:927`) and eval
(`eval.py` `ba` condition) both construct it bare, so defaults flow everywhere. No YAML
changes; `pointcloud.bundle_adjustment` remains one boolean.

### Config changes (`BundleAdjustmentConfig`)

| field | old | new |
|---|---|---|
| `vis_thresh` | — (implicit ~0) | `0.2` (new field) |
| `fine_tracking` | — (hardcoded False) | `True` (new field) |
| `max_query_pts` | 2048 | 4096 |
| `query_frame_num` | 5 | 8 |
| `shared_camera` | False | True |

### Code changes

- `_optimize`: `vis = vis_scores > cfg.vis_thresh` replaces `astype(bool)`. `vis_thresh=0.0`
  reproduces old behaviour for any score > 0.
- `_load_or_extract_tracks` threads `cfg.fine_tracking` into `_extract_tracks_vggsfm`.
- Filter order in `_optimize` reordered to upstream: (a) reproj + vis mask →
  (b) drop frames under `min_inliers_per_frame` → (c) drop landmarks with <2 surviving
  observations or out of range. Single-observation landmarks after (b) become impossible.
- **Track cache key** (`_compute_tracks_cache_key`) gains `fine_tracking`. Without it, a
  stale coarse-track cache would silently defeat the fix. `vis_thresh` and `shared_camera`
  are post-extraction knobs — deliberately *not* in the key, so ablations reuse the cache.

### Validation (chess/seq-01, 100 frames, single-pass — same protocol as the 2026-08-19 sweep)

Runs, serial in tmux:

1. `vggtx` (worst regression) — baseline vs ba, full new defaults.
2. `mapanything` (the one BA helped) — regression guard in the other direction.
3. `vggtx` ablation — new defaults but `shared_camera=False`, isolating the shared-camera
   contribution (track cache makes this cheap; extraction dominates runtime).

Decision gate:

- vggtx ba ATE ≤ baseline → land; update spec §Validation table + CLAUDE.md verdict.
- Improved but still > baseline → follow-on: 1024/original-res track extraction (option B).
- No improvement → follow-on: focal freeze / pose-only BA (option C).
- mapanything must not regress below its measured BA gain (ATE 0.0120); if it does, ablate
  which new knob costs it.

### Testing

- `tests/geometry/test_bundle_adjustment.py` (25 green) adapts: new-field defaults, vis
  threshold actually filters (score 0.1 observation dropped at 0.2, kept at 0.0),
  filter-order property (no landmark with <2 observations survives after frame drop),
  cache-key invalidation on `fine_tracking` change, cache-hit on `vis_thresh` change.
- No new test files; extend the existing module.

## Implementation principles

Reuse existing seams — every change lands inside functions that already exist; no new
modules, no wrappers. Delete nothing except the hardcoded `fine_tracking=False` and the
`astype(bool)` line. Eval protocol and scripts unchanged.
