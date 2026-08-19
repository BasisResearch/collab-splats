# BA LM convergence + eval track cache — design

**Date:** 2026-08-19
**Status:** draft — awaiting user review
**Follows:** `2026-08-19-ba-track-quality-parity-design.md` (§Measured results)

## Problem

The track-quality parity fix landed and improved BA measurably (mapanything ATE −32.7%,
vggtx +72% → +8%). But every one of those numbers came from a solver that **ran 1–2
Levenberg–Marquardt steps out of a configured 40**.

pypose `StopOnPlateau.step()` (`pypose/optim/scheduler.py:153-157`):

```python
if hasattr(self.optimizer, 'reject_count'):
    if self.optimizer.reject_count > 0:
        self._continual = False
```

bae's LM resets `reject_count = 0` at the top of each `step()`
(`bae/optim/optimizer.py:32`) and increments it inside the damping loop whenever a trial
step raises the loss (lines 50-52) — routine trust-region behaviour. So **the first outer
LM step that needs any damping retry terminates the entire optimization.**

Measured on chess/seq-01, 100 frames:

| backbone | inner damping trials | outer LM steps |
|---|---|---|
| vggtx | `Reject Count: 0` → rejected → `1` → accepted | **1** of 40 |
| mapanything | step 1 clean; step 2 needed a retry | **2** of 40 |

`lm_steps=40` is therefore decorative. This is inherited from upstream
`zitongzhan/vggt demo_colmap.py` (same `StopOnPlateau`, same `reject=10`;
`scheduler.optimize()` is the identical loop), so we are at parity — but upstream carries
`# TODO: test with more cases` and never validated BA output quality. Parity with an
unvalidated defect is not a reason to keep it.

Secondary defect in the same scheduler: the plateau test
`(self.optimizer.last - self.optimizer.loss) < self.decreasing` is an **absolute**
difference against `decreasing=1e-3`, despite the docstring saying "relative". At our loss
scale (~1e5–1e6) it can never fire, so it contributes nothing even when reached.

Third problem, unrelated to convergence but blocking iteration speed: `tracks_cache_dir`
defaults to `None` (= always extract) and `evals/scripts/eval.py` constructs
`BundleAdjustmentConfig()` bare, so **every eval BA condition re-extracts tracks** (~11 min
of the ~14 min run). Ablations that differ only in post-extraction knobs pay full price.

## Design

Three changes. All in `collab_splats/geometry/bundle_adjustment.py` except the eval wiring.

### 1. Replace `StopOnPlateau` with an explicit LM loop

`_optimize` currently has two branches — a manual loop under `cfg.capture_loss_history` and
a `StopOnPlateau` loop otherwise. Both are replaced by **one** loop that always records the
loss history and always logs per step. This deletes a branch rather than adding one.

```python
# Manual LM loop. pypose's StopOnPlateau is unusable here: it aborts the whole
# optimization as soon as bae's LM needs a single trust-region damping retry
# (reject_count > 0), which routinely fires on the first step, and its plateau test
# is absolute rather than relative so it never triggers at our loss scale.
prev_loss: float | None = None
plateau = 0
loss_hist: list[float] = []
for i in range(n_steps):
    step_loss = float(optimizer.step(input=input_dict))
    loss_hist.append(step_loss)
    logger.info("LM step %d/%d: loss=%.6e", i + 1, n_steps, step_loss)
    # Stop once relative loss reduction stays below tolerance for lm_patience steps
    if prev_loss is not None:
        rel = (prev_loss - step_loss) / max(abs(prev_loss), 1e-12)
        plateau = plateau + 1 if rel < cfg.lm_rel_tol else 0
        if plateau >= cfg.lm_patience:
            logger.info("LM converged: relative reduction < %.1e for %d steps",
                        cfg.lm_rel_tol, cfg.lm_patience)
            break
    prev_loss = step_loss
self._last_loss_history.append(loss_hist)
```

New `BundleAdjustmentConfig` fields:

| field | value | meaning |
|---|---|---|
| `lm_rel_tol` | `1e-4` | relative loss reduction below which a step counts as plateaued |
| `lm_patience` | `3` | consecutive plateaued steps before stopping |

`capture_loss_history` becomes dead — history is now always captured. **Delete the field**
and its branch (per the repo's reuse/retire principle). It has one production reader,
`collab_splats/wrapper/reconstructor.py:927`, which passes `capture_loss_history=True`
alongside `tracks_cache_dir=self.backend_dir`; that kwarg is simply dropped. Tests in
`tests/geometry/test_bundle_adjustment.py` (lines ~670-700, ~920-942) assert the old
flag-gated behaviour and must be rewritten to assert unconditional capture.

Damping retries stay entirely inside bae's LM (`reject=10` unchanged). We simply stop
treating "this step needed a retry" as a termination condition.

### 2. Wire the track cache into eval

Add `--tracks_cache_dir` to `evals/scripts/eval.py`, default `None` so current behaviour is
byte-identical when unset. When set, `_make_creator` passes it into every
`BundleAdjustmentConfig` it constructs.

**Known limitation, documented not fixed:** the cache is single-slot — one `tracks.zarr`
plus a validating key. Conditions whose extraction config differs (`fine_tracking`, and
hence `ba` vs `ba_coarse`) evict each other, so interleaving them thrashes. Conditions
differing only in post-extraction knobs (`ba` vs `ba_percam`) share a key and hit. Sweep
scripts should group by extraction config, or use a per-condition cache dir.

### 3. `ba_coarse` eval condition

`BundleAdjustmentConfig(fine_tracking=False)`, mirroring the existing `ba_percam` sibling in
both `_make_creator` branches, `_FIXED_CONDITIONS`, and `_COLORS`.

Rationale: `fine_tracking=True` was flipped as one of five simultaneous changes and never
attributed on its own, and it is the dominant runtime cost. Whether coarse tracks suffice is
an empirical question — BA is least-squares, so zero-mean track noise averages down over
~34k points × 100 frames, but coarse tracking's error is quantization-like at downsampled
feature resolution and therefore partly correlated, which does *not* average out. Measure it.

### Considered and deferred: lowering `max_reproj_error`

Rejected for now, on three grounds.

1. **It does not make tracks more accurate.** The tracks are whatever VGGSfM predicted; the
   threshold only decides which *observations* are admitted. Track accuracy is governed by
   `fine_tracking` and the density knobs.
2. **We are already stricter than upstream.** Upstream's `--max_reproj_error` default is
   **8.0**; ours is `4.0`. Lowering further moves further *from* parity, not toward it.
3. **It would bias BA toward its own initialization.** The filter is applied once, against
   the *initial* feedforward poses. Tightening it preferentially keeps observations that
   already agree with the starting geometry — so BA is handed less evidence that the poses
   should move. Given the presenting complaint is "the poses aren't correcting", tightening
   is the wrong direction.

The principled version of this knob is **iterative BA** (upstream's own
`# TODO: add iterative BA`): start loose, re-filter and tighten across rounds as the poses
improve, the way classical COLMAP does. Then the threshold is applied against improving
poses rather than the initial guess.

Sequencing matters too: any reprojection-threshold optimum measured now would be measured
against a 1-step solver and would move once BA actually converges. Revisit as part of
iterative BA, after this spec lands.

## Validation (chess/seq-01, 100 frames, tmux, serial — human-gated)

With BA actually converging, re-measure. Track cache set per backbone.

1. `vggtx` — baseline, ba
2. `mapanything` — baseline, ba
3. `vggt_omega` — baseline, ba (never validated under parity defaults; largest fx spread at
   11%, so `shared_camera=True` should matter most here)
4. `vggtx` — ba_coarse (fine-tracking attribution; separate cache dir, different key)

Compare against the 1-step numbers now recorded in the parity spec's §Measured results.

Success: BA runs >2 LM steps and ATE improves relative to the 1-step measurement on at least
the backbones where BA already helps. Expected cost: runtime *rises* (40 real steps vs 1),
partly offset by the cache on re-runs.

Open question this is expected to answer: the small rotation regression seen on both
backbones (vggtx RPE-rot 0.186°→0.201°) whose cause is still unknown — the `ba_percam`
ablation ruled out the shared-focal flip (percam rotation was *worse*, 0.2067°).

## Testing

Extend `tests/geometry/test_bundle_adjustment.py`:

- new-field defaults (`lm_rel_tol`, `lm_patience`)
- the loop runs to `lm_steps` when loss keeps falling (fake optimizer, monotone losses) —
  the direct regression test for the `reject_count` abort
- the loop stops after `lm_patience` plateaued steps (fake optimizer, flat losses)
- loss history is captured unconditionally (`capture_loss_history` removal)

Extend `tests/evals/test_eval_gt_helpers.py`: `ba_coarse` → `fine_tracking=False`;
`--tracks_cache_dir` reaches the config when set and stays `None` when not.

## Implementation principles

Net branch count goes *down*: two loop branches become one, and `capture_loss_history` is
deleted. No new modules, no wrapper layers. The eval flag defaults to `None` so nothing
changes for existing callers. `ba_coarse` mirrors the `ba_percam` sibling exactly rather
than introducing an ablation framework.
