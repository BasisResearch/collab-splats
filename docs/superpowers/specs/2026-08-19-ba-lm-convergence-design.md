# BA LM convergence, dropped-frame gauge, eval track cache — design

**Date:** 2026-08-19
**Status:** draft — awaiting user review (rev 2)
**Follows:** `2026-08-19-ba-track-quality-parity-design.md` (§Measured results)

## Problem

Three defects, found while diagnosing why BA improves poses less than expected.

### 1. Eval and production run different solvers

`_optimize` branches on `cfg.capture_loss_history`:

- **`True`** → plain `for i in range(n_steps)` loop, all 40 LM steps.
- **`False`** → `StopOnPlateau`, which aborts after 1–2 steps (below).

`reconstructor.py:927` builds `BundleAdjustmentConfig(tracks_cache_dir=..., capture_loss_history=True)`.
`evals/scripts/eval.py` builds `BundleAdjustmentConfig()` bare.

So **the production pipeline runs 40 LM steps and every measured number came from a 1–2 step
solve.** A flag whose stated purpose is "record per-step loss for visualisation" silently
selects the optimizer's stopping rule. Confirmed from the run logs:

```
$ for f in *.log; do echo "$f: $(grep -c 'LM step' $f)"; done
mapanything.log: 2
vggtx.log: 1
vggtx_percam.log: 1
```

Production BA has never been measured against ground truth.

### 2. `StopOnPlateau` aborts on the first damping retry

`pypose/optim/scheduler.py`, `StopOnPlateau.step()` has three stop conditions:

| lines | condition | fires on our data? |
|---|---|---|
| ~139 | `steps` exhausted | after 40 |
| 143-149 | `(optimizer.last - optimizer.loss) < decreasing` for `patience` steps | **never** — absolute difference against `1e-3`, our loss is ~1e5–1e6 |
| 153-155 | `optimizer.reject_count > 0` | **immediately** |

`bae/optim/optimizer.py:32` resets `self.reject_count = 0` per `step()` and line ~51 increments
it whenever a trial step raises the loss inside the trust-region damping loop — routine LM
behaviour. So the first outer step needing any damping retry ends the optimization.

Neutralize that one condition and `StopOnPlateau` degenerates to `for i in range(40)`, because
its plateau test cannot fire at our loss scale. It contributes nothing here.

### 3. Dropped frames are left in the pre-BA gauge

`_optimize` seeds `refined_extrinsics = extrinsics.copy()` (`:294`) and writes back only
`refined_extrinsics[active_frames] = ...` (`:444`). A frame dropped by `min_inliers_per_frame`
keeps its **original pose verbatim**, and `refine()` returns one full-N array (`:268`) that
COLMAP, `transforms.json`, and the zarr poses all consume as uniformly refined.

This is worse than a missed correction. `_BAModel` (`:662-676`) makes every pose, every point,
and the shared focal a free `nn.Parameter` — **no frame is frozen and scale is unconstrained**,
so the solve has 7 DoF of gauge freedom held only by LM damping. A dropped frame is therefore
left behind in the *old gauge* while all others move to the solved one: its error is the global
gauge drift plus the missed local correction, not just the latter. Nothing marks it, and ATE's
trajectory-wide alignment absorbs the drift so the frame reads as a single noisy outlier.

Same path leaves its intrinsics stale: `:448-455` writes the shared focal only to
`active_frames`, so a dropped frame keeps its per-frame focal in a shared-camera reconstruction.

Latent, not live — measured `LM optimize: 100/100 frames` on every run to date, so the gate has
never fired on our data. Fixing it is cheap; leaving a silent wrong-gauge pose in the output is
not acceptable.

### 4. Eval never uses the track cache

`tracks_cache_dir` defaults to `None` (= always extract) and eval builds the config bare, so
every eval BA condition re-extracts tracks (~11 min of a ~14 min run). Ablations differing only
in post-extraction knobs pay full price.

## Design

Four changes. All in `collab_splats/geometry/bundle_adjustment.py` except the eval wiring and
one helper move.

### 1. One LM loop — keep the one that already exists

Delete the `StopOnPlateau` branch. Always take the existing plain loop, which is already written
and already what production runs:

```python
# Manual LM loop. pypose's StopOnPlateau is unusable here: it aborts the whole
# optimization as soon as bae's LM needs a single trust-region damping retry
# (reject_count > 0), which routinely fires on step 1, and its plateau test is an
# absolute difference against 1e-3 so it never fires at our loss scale (~1e5-1e6).
loss_hist: list[float] = []
for i in range(n_steps):
    step_loss = optimizer.step(input=input_dict)
    loss_hist.append(float(step_loss))
    logger.info("LM step %d/%d: loss=%.6e", i + 1, n_steps, float(step_loss))
self._last_loss_history.append(loss_hist)
```

**No new code and no new config fields** — this is the `capture_loss_history=True` body,
unconditional. `capture_loss_history` becomes dead and is **deleted**; its one production reader
(`reconstructor.py:927`) drops the kwarg, and `tests/geometry/test_bundle_adjustment.py`
(~670-700, ~920-942) is rewritten to assert unconditional capture.

Rejected alternatives: zeroing `optimizer.reject_count` before `scheduler.step()` (one line, but
silently defeats a documented stop condition of a third-party class and lands on identical
behaviour); adding our own `lm_rel_tol`/`lm_patience` early stop (new config surface for an
early exit we have not measured a need for — revisit if the 40-step loss curves plateau).

Damping retries stay inside bae's LM (`reject=10` unchanged). We stop treating "this step needed
a retry" as termination.

**Parity note:** upstream `demo_colmap.py` calls `scheduler.optimize()` and hits the same abort,
so this is a deliberate deviation. It is not a *new* one — our production path has diverged since
`capture_loss_history=True` was wired in. This change makes eval and production agree, and picks
the branch that actually runs the configured 40 steps.

### 2. Dropped frames: carry the gauge, warn

After writeback, transform any inactive frame by the same Sim(3) the active set underwent, so it
stays consistent with the refined reconstruction.

```python
# Frames dropped by the min-inlier gate keep their pre-BA pose, which sits in the
# pre-BA gauge — BA fixes no frame and no scale, so the refined set can drift as a
# whole. Carry dropped frames along by the Sim(3) the active set underwent.
inactive = np.setdiff1d(np.arange(vis.shape[0]), active_frames)
if len(inactive) and len(active_frames) >= 3:
    src_c = _camera_centers(extrinsics[active_frames])
    dst_c = _camera_centers(refined_extrinsics[active_frames])
    s, R_g, t_g = umeyama_sim3(src_c, dst_c)
    # World gauge X' = s R_g X + t_g maps world-to-cam [R|t] to [R R_g^T | s t - R R_g^T t_g]
    R_in = refined_extrinsics[inactive, :, :3]
    t_in = refined_extrinsics[inactive, :, 3]
    R_new = R_in @ R_g.T
    refined_extrinsics[inactive, :, :3] = R_new
    refined_extrinsics[inactive, :, 3] = s * t_in - np.einsum("nij,j->ni", R_new, t_g)
    logger.warning(
        "BA: %d/%d frames dropped by min_inliers_per_frame=%d (indices %s); "
        "carried by the active-set Sim(3) (s=%.6f) but not refined",
        len(inactive), vis.shape[0], cfg.min_inliers_per_frame, inactive.tolist(), s,
    )
elif len(inactive):
    logger.warning(
        "BA: %d/%d frames dropped and left in the pre-BA gauge (<3 active frames, "
        "cannot estimate the gauge transform)", len(inactive), vis.shape[0],
    )
```

Derivation for the pose map, for the reviewer: world points move as `X' = s·R_g·X + t_g` and
camera-frame coordinates scale by `s`, so `x'_cam = s(RX + t) = R'(s·R_g·X + t_g) + t'` gives
`R' = R·R_gᵀ` and `t' = s·t − R·R_gᵀ·t_g`. Check on centres: `C' = −R'ᵀt' = s·R_g·C + t_g`, the
same map as the points.

Shared focal, same block: when `cfg.shared_camera`, write `focal_val` to **all** frames rather
than only `active_frames` (`:448-455`), so a dropped frame does not keep a stale per-frame focal
in a reconstruction that is otherwise single-camera by construction.

`umeyama_sim3` already exists at `collab_splats/geometry/loop_closure/graph.py:499`, but that
module imports `gtsam` at module level and BA must not pull it in. **Move `umeyama_sim3` and its
sibling `umeyama_se3` to `collab_splats/geometry/transforms.py`** (pure numpy, generic, sits
beside `extrinsics_to_homogeneous`/`invert_poses`/`rotation_align_vectors`, which BA already
imports from) and re-import them in `graph.py`. Both move together because splitting the pair
across modules is worse than moving the one we do not use yet. `_camera_centers` is a two-line
local helper (`-Rᵀt` over a batch) unless an equivalent already exists in `transforms.py`.

### 3. Wire the track cache into eval

Add `--tracks_cache_dir` to `evals/scripts/eval.py`, default `None` so behaviour is
byte-identical when unset. When set, `_make_creator` passes it into every
`BundleAdjustmentConfig` it constructs.

**Known limitation, documented not fixed:** the cache is single-slot — one `tracks.zarr` plus a
validating key. Conditions whose *extraction* config differs (`fine_tracking`, hence `ba` vs
`ba_coarse`) evict each other, so interleaving them thrashes. Conditions differing only in
post-extraction knobs (`ba` vs `ba_percam`) share a key and hit. Sweep scripts should group by
extraction config, or use a per-condition cache dir.

### 4. `ba_coarse` eval condition

`BundleAdjustmentConfig(fine_tracking=False)`, mirroring the existing `ba_percam` sibling in both
`_make_creator` branches, `_FIXED_CONDITIONS`, and `_COLORS`.

Rationale: `fine_tracking=True` was flipped as one of five simultaneous changes and never
attributed on its own, and it dominates runtime. Whether coarse tracks suffice is empirical — BA
is least-squares, so zero-mean track noise averages down over ~34k points × 100 frames, but
coarse tracking's error is quantization-like at downsampled feature resolution and therefore
partly correlated, which does not average out. Measure it.

## Considered and deferred

**Lowering `max_reproj_error`.** Rejected on three grounds. (a) It does not make tracks more
accurate — the tracks are whatever VGGSfM predicted; the threshold only decides which
*observations* are admitted. (b) Upstream's `--max_reproj_error` default is **8.0** and ours is
`4.0`; lowering moves further from parity, not toward it. (c) The filter runs once, against the
*initial* poses, so tightening preferentially keeps observations already agreeing with the
starting geometry — BA is handed less evidence that the poses should move, which is the wrong
direction when the complaint is that poses are not correcting. Any optimum measured now would be
measured against a 1-step solver and would move once BA converges.

**Fixing the gauge inside the solve** (freeze `pose[0]` and scale). Would eliminate drift at the
source and make change 2 unnecessary, and is a plausible contributor to the unexplained rotation
regression, since an unconstrained gauge lets the solve rotate the whole cloud. Deferred because
it changes the optimization problem itself and would confound this spec's measurement; worth its
own A/B once BA runs its full 40 steps.

**Dropped frames in `colmap/refine.json` provenance.** Would need a new `FeedforwardResult` field
or a side-channel attribute on `BundleAdjustment`; not worth either while the measured drop count
is zero. The warning log carries the information.

**Iterative BA** (upstream's own `# TODO: add iterative BA`): BA → refilter → BA, tightening the
reprojection threshold across rounds so it is applied against improving poses. The principled home
for the threshold question. Follow-on.

## Validation (chess/seq-01, 100 frames, tmux, serial — human-gated)

With BA running its configured 40 steps, re-measure. Track cache set per backbone.

1. `vggtx` — baseline, ba
2. `mapanything` — baseline, ba
3. `vggt_omega` — baseline, ba (never validated under parity defaults; largest fx spread at 11%,
   so `shared_camera=True` should matter most here)
4. `vggtx` — ba_coarse (fine-tracking attribution; separate cache dir, different extraction key)

Compare against the 1–2 step numbers recorded in the parity spec's §Measured results. Note these
runs are also the first measurement of what the **production** pipeline has been computing.

Success: BA runs to 40 steps (or a genuine loss plateau visible in the logged curve) and ATE
improves relative to the 1–2 step measurement on at least the backbones where BA already helps.
Expected cost: runtime rises — 40 real steps against ~830k observations, partly offset by the
cache on re-runs.

Open question this should inform: the small rotation regression on both backbones (vggtx RPE-rot
0.186°→0.201°) whose cause is unknown — `ba_percam` ruled out the shared-focal flip (percam
rotation was *worse*, 0.2067°). Unconstrained gauge is the current leading hypothesis.

## Testing

Extend `tests/geometry/test_bundle_adjustment.py`:

- loss history is captured unconditionally (`capture_loss_history` removal)
- the loop runs the full `lm_steps` with a fake optimizer whose `reject_count` is non-zero — the
  direct regression test for the abort
- dropped-frame gauge: construct a case where one frame falls under `min_inliers_per_frame`, apply
  a known Sim(3) to the active refined poses, assert the dropped frame's camera centre lands at
  `s·R_g·C + t_g` and that its focal matches the shared value
- fewer than 3 active frames: dropped frames are left untransformed and the second warning fires

Extend `tests/geometry/loop_closure/` coverage only as needed to keep the `umeyama_*` move green
(re-import, no behaviour change).

Extend `tests/evals/test_eval_gt_helpers.py`: `ba_coarse` → `fine_tracking=False`;
`--tracks_cache_dir` reaches the config when set and stays `None` when not.

## Implementation principles

Net branch count goes *down*: two loop branches become one and `capture_loss_history` is deleted.
No new modules and no new config fields — the LM change is "keep the branch that already exists",
and the gauge fix reuses `umeyama_sim3` rather than writing an alignment. The helper move is a
relocation, not a rewrite. The eval flag defaults to `None` so nothing changes for existing
callers, and `ba_coarse` mirrors the `ba_percam` sibling rather than introducing an ablation
framework.
