# Eval Parity Handoff — 2026-05-28

## Status: canonical eval table complete, two blockers remain

---

## What was done this session

### 1. Cross-frame attention analysis (Q&A, no code changes)

Three questions investigated:

**Q: Why do cross_frame_attention_ratio values differ across models?**
Algorithm is identical (`get_similarity` port). Differences come from:
- Trained weight distributions — threshold 0.85 calibrated on VGGT-1B only
- Different attention block semantics hooked (VGGT-X: `global_blocks`, MapAnything: `info_sharing`, Omega: `inter_frame_blocks` — purpose-built cross-frame blocks → max_self is inherently low → ratio inflates above 1.0, hence Omega mtq=1.328 at layer=16)
- Layer selection: optimal layers VGGT-X=20, Omega=16, MapAnything=4; wrong layer shifts entire distribution
- Token count differences (different patch grid per model)

**To properly understand distributions**: need mean/std/p25/p50/p75/p90 for correct AND incorrect loop pairs per model. We only have `mean` right now — not enough to set per-model thresholds confidently.

**Q: Norm ratio vs. pairwise distance — are we the same as VGGT-SLAM?**
Two separate things:
- `cross_frame_attention_ratio`: YES, identical to VGGT-SPARK `get_similarity()` now (mean_top_quarter fix applied earlier)
- `estimate_scale_pairwise`: we CALL VGGT-SLAM's function (norm ratio), which is biased when intra-submap camera travel D is not << scene depth d

The norm ratio bias:
```
norm_ratio = (D + d) / (D + s·d)
# D = camera travel within prev submap, d = depth from overlap camera
# For D << d (short submaps): ratio ≈ 1/s  ✓
# For D >> d (long submaps): ratio ≈ 1     ✗ collapses to 1 regardless of s
```
VGGT-SLAM works because submap_size=16 → small D. Our 50-frame submaps (past evals) → D ≈ d → severe bias. At canonical submap_size=16 the bias is smaller but still present.

Pairwise distance fix (from spec, not yet implemented):
```python
def _estimate_scale_pairwise_dist(X, Y):
    idx = rng.choice(X.shape[0], (n, 2), replace=True)
    i, j = idx[:,0], idx[:,1]
    x_dists = np.linalg.norm(X[i] - X[j], axis=1)
    y_dists = np.linalg.norm(Y[i] - Y[j], axis=1)
    valid = x_dists > 1e-6
    return float(np.median(y_dists[valid] / x_dists[valid]))
```
Translation cancels in subtraction → origin-invariant → exact for pure scale differences.

**Why VGGT-SLAM was apples-to-oranges**: prior VGGT-SLAM LC run covered 30 sparse keyframes, ours covered 200 dense frames. Fixed this session.

---

### 2. Code fixes made

**`eval_gt.py`**: Added `_config` key to `metrics.json` output. Every run now records `backbone`, `max_frames`, `submap_size`, `dataset`, `seq_dir`. Previously unrecoverable from output alone.

**`evals/runners/run_vggt_slam_lc.py`**: Fixed sort bug. `utils.sort_images_by_number` regex fails on `frame-XXXXXX.color.png` (double extension breaks lookahead) — all files returned `float('inf')` → unsorted → arbitrary frame selection. Previous run accidentally processed frames 800-999. Fixed to `sorted(all_images)[:max_frames]` (lexicographic = correct ascending order).

---

### 3. Canonical eval table established

**Parameters**: `submap_size=16`, `max_frames=200`, `frames 0-199`, `chess/seq-01`

| Model | Condition | ATE RMSE | Source | submap_size confirmed |
|---|---|---|---|---|
| VGGT-X | baseline | 0.1801m | `chess_seq01_vggtx_sub16/` | ✓ dir name |
| VGGT-X | lc | 0.2664m | `run-20260528-035543/` | ✓ session log |
| Omega | baseline | 0.1672m | `chess_seq01_sub16/` | ✓ dir name |
| Omega | lc | 0.4959m | `run-20260528-035337/` | ✓ session log |
| MapAnything | baseline | 0.1103m | `run-20260528-040615/` | single-pass (windowed broken) |
| MapAnything | lc | ❌ | — | `_lc_collate_outputs` bug |
| VGGT-SLAM (SPARK) | no-LC | 0.2967m | `vggt_slam_dense_nolc.tum` | 212 frames, ~16 |
| VGGT-SLAM (SPARK) | lc | 0.3452m | `vggt_slam_lc.tum` | ✓ 212 poses, frames 0-199 |

**Key finding**: LC hurts ALL models in current state:
- VGGT-X: +0.086m (1.48× worse)
- Omega: +0.329m (2.97× worse — catastrophic)
- VGGT-SLAM: +0.048m (1.16× worse)

---

## Two blockers remaining

### Blocker 1: MapAnything windowed (`_lc_collate_outputs` + `_forward` bug)

Error: `KeyError: 'camera_poses'` at `mapanything.py:229` in `_lc_collate_outputs`.
Also: `TypeError: list indices must be integers or slices, not str` in `_forward`.

Both bugs are described in spec `2026-05-28-lc-diagnostics-and-model-parity-design.md` (Architecture sections "MapAnything `_forward` windowing fix" and "MapAnything `_lc_collate_outputs` fix"). Implementation not yet done. Fix is self-contained (~2 hours per spec).

Until fixed: MapAnything eval only has single-pass baseline (0.1103m). No LC result. Not comparable to windowed runs of other models.

### Blocker 2: Pairwise distance scale estimator not implemented

The norm-ratio bias is understood and the fix is specified. Not yet in code. The scale drift (previously observed cumulative product 0.363) is not addressed in current results. All LC results above may still suffer from scale estimation error.

Spec location: `docs/superpowers/specs/2026-05-28-lc-diagnostics-and-model-parity-design.md`, section "Scale estimation fix".

---

## Remaining open questions from analysis

1. **Per-model attention ratio distributions**: need mean/std/p25/p50/p75/p90 for correct vs incorrect pairs per model to set thresholds properly. Parity harness results only have mean.

2. **VGGT-SLAM pose convention**: ATE computed via c2w→w2c inversion. Should cross-check against evo_ape output to confirm convention is right.

3. **Omega verify gate**: threshold 0.85 is effectively disabled (Omega mtq=1.328 >> 0.85). Spec says raise to 0.99. Not yet implemented.

4. **MapAnything windowed LC**: once `_lc_collate_outputs` + `_forward` fixed, need full re-run.

---

## Next steps (priority order)

1. Implement MapAnything `_forward` + `_lc_collate_outputs` fix (spec §MapAnything) → re-run eval
2. Implement pairwise distance scale estimator → re-run all LC evals
3. Raise Omega `default_verify_match_ratio` to 0.99 + add `default_max_jump_ratio=0.3` → re-run Omega LC
4. Collect per-model attention ratio distributions (correct vs incorrect pairs) → calibrate per-model thresholds
5. After all fixes: re-run full canonical table, compare against VGGT-SLAM baseline

---

## File locations

- Canonical results: `evals/results/chess_seq01_sub16/`, `evals/results/chess_seq01_vggtx_sub16/`, `evals/results/7scenes/seq-01/run-20260528-035337/`, `evals/results/7scenes/seq-01/run-20260528-035543/`
- VGGT-SLAM baselines: `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` (212 lines, frames 0-199), `vggt_slam_dense_nolc.tum` (212 lines)
- Active spec: `docs/superpowers/specs/2026-05-28-lc-diagnostics-and-model-parity-design.md`
- Eval runner: `evals/eval_gt.py` (now records `_config` in metrics.json)
- Fixed runner: `evals/runners/run_vggt_slam_lc.py` (sort bug fixed)
