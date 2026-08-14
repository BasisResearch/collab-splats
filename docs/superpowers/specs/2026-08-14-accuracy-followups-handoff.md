# Handoff — reconstruction accuracy follow-ups

**Date:** 2026-08-14
**Predecessor work:** `2026-08-13-multiview-confidence-parity-design.md`,
`2026-08-13-multiview-confidence-measured-report.md`, `2026-08-11-mesh-tsdf-adapter-convergence-design.md`
**Status:** briefing only. No plan yet, no code written against it.

This document exists so a fresh agent can pick up accuracy work without re-deriving what the
multiview-confidence pass established. It states what is measured, what is assumed, what is
wrong in the current docs, and three candidate workstreams with a falsifiable first experiment
each. It deliberately does **not** pick a winner — the ordering argument is in
[Sequencing](#sequencing), but the first job of whoever reads this is Workstream 1, because it
is cheap and it changes the baseline every other measurement is compared against.

---

## State of the world (verified 2026-08-14)

- `pointcloud.use_multiview_confidence` ships **`false`**. The geometric cross-view filter
  exists, is shared by all four feedforward backbones, is tested, and is calibrated
  (`rel=0.01, K=2` on vggtx/vggt_omega/loger; `rel=0.02/abs=0.02, K=1` on mapanything) — but it
  is off, because on 7-Scenes chess/seq-01 it never beat the learned-confidence percentile at
  comparable retention. **Any accuracy experiment that assumes mv is active is measuring
  something the pipeline does not do.**
- `mv_ratio` / `mv_inlier_count` / `mv_valid_count` are written to `feedforward.zarr` only when
  mv is computed, and are **absent — not zero** otherwise. Existing stores have none and are
  not backfilled.
- The learned confidence and mv are **ANDed** (upstream MapAnything substitutes). The shipped
  behaviour is strictly more conservative than upstream.
- Learned-confidence defaults, per creator dataclass:

  | Backend | Field | Default |
  |---|---|---|
  | `VGGTXCreator` | `conf_threshold` | `35.0` |
  | `VGGTOmegaCreator` | `conf_threshold` | `50.0` |
  | `LoGeRCreator` | `conf_threshold` | `50.0` |
  | `MapAnythingCreator` | `confidence_percentile` | `35.0` |

  These are **percentiles**, and none of them has been swept against ground-truth depth. Step D
  contains exactly one learned-confidence row (`learned_conf_p50`), used as a reference point
  for mv — not as a calibration of the percentile itself.
- Ground-truth depth sweeping is reproducible from the repo:
  `evals/scripts/eval_run_backend.py` (one backend, fresh process) →
  `evals/scripts/eval_multiview_conf.py` (GT-depth sweep, writes JSON).
- Local data: `data/7scenes/chess/seq-01` … `seq-06`. **Only the `chess` scene is on disk.** A
  second *sequence* is free; a second *scene* or dataset requires a download.

## Corrections to existing docs — read before trusting a number

1. **`depth_trunc` is `1.0`, not `2.0`.** `configs/base.yaml` has shipped `depth_trunc: 1.0`
   since `d001edc` (2026-07-20). The mesh convergence spec labels its `depth_trunc=2.0` row
   "shipping baseline" and CLAUDE.md repeats it. The recorded 711,079-vertex post-fix baseline
   was therefore measured at a value **nothing ships**, and the real shipping mesh is strictly
   smaller and currently unmeasured. Fix the label when you measure; do not silently change the
   config.
2. **The K≥8 rows in Step D are upper bounds.** `multiview_mask` clamps per pixel
   (`required = min(min_views, valid_count)`), so on a scene with fewer partners than K the
   filter weakens to "all your partners must agree". The sweep rows were produced before the
   clamp existed.
3. **LoGeR has two confidence knobs with different units.** `LOGER_CONF_THRESHOLD = 0.02` is a
   raw post-sigmoid floor for the intrinsics fit; `conf_threshold = 50.0` is a percentile for
   point filtering. Do not conflate them. More generally, `conf_threshold > 1.0` is read as a
   percentile and `<= 1.0` as a raw value — a "0.5" typo silently becomes a different filter.

---

## Workstream 1 — `depth_trunc` (cheapest, blocks the others)

**Claim.** `depth_trunc` is the dominant limit on mesh extent, costing ~86% of vertices at
`2.0` (711,079 vs 5,060,214 at `20`) on `data/outputs/`, and the shipping value is tighter
still at `1.0`. Depth is **non-metric** for every backbone except MapAnything, so `1.0` is not
"1 metre" — it is an arbitrary cut in model units that varies per scene.

**Why it blocks.** Every downstream accuracy claim about the mesh is measured through this cut.
If the shipping baseline is wrong by a factor of 2 in truncation, the mesh numbers in the
2026-08-11 spec do not describe the shipping pipeline.

**First experiment.** Re-mesh `data/outputs/` at the actual shipping `depth_trunc: 1.0` and at
a scale-aware alternative, and record vertex count, bounding box, and a GT-depth-free proxy for
correctness (surface completeness against the sparse cloud). Then decide whether the knob
should be an absolute value at all, or a percentile of the per-scene depth distribution — which
is the only form that is scale-invariant across non-metric backbones.

**Falsification.** If `1.0` and `20.0` produce meshes whose *useful* extent is the same — i.e.
everything beyond `1.0` is noise, not surface — then `depth_trunc` is a correctly-tuned noise
gate and this workstream ends. Measure that before proposing a change.

**Cost.** Hours. No new code required for the measurement; `Reconstructor.mesh(overwrite=True)`
already accepts the parameter.

**Note.** Every `mesh.ply` on disk — local and under `environments-processed/` — predates the
2026-08-11 intrinsics fix and needs `mesh(overwrite=True)` before it can be compared to
anything.

## Workstream 2 — sweep the learned-confidence percentile

**Claim.** The learned percentile is the stronger of the two filters we have (Step D:
mv never beat it at comparable retention), it is the one knob every backbone shares, and it has
never been calibrated against ground truth. Three of four defaults are round numbers with no
recorded provenance.

**First experiment.** Extend `eval_multiview_conf.py` to sweep `conf_threshold` over a grid
(e.g. 10/20/35/50/65/80) on the same GT-depth harness that produced Step D, per backend,
reporting the same statistics. The harness already computes everything needed — the learned
confidence is in the zarr and `retained_error` already takes an arbitrary keep-mask.

```bash
for B in vggt_omega vggtx mapanything; do
  /opt/venv/reconstruction/bin/python evals/scripts/eval_run_backend.py --backend $B \
      --seq data/7scenes/chess/seq-01 --out evals/results/mv_$B --max-frames 60
  /opt/venv/reconstruction/bin/python evals/scripts/eval_multiview_conf.py \
      --zarr evals/results/mv_$B/feedforward.zarr --seq data/7scenes/chess/seq-01 \
      --max-frames 60 --out evals/results/mv_sweep_$B.json
done
```

**Falsification.** If the error-vs-retention curve is flat across the grid, the percentile is
not a lever and the defaults are fine wherever they happen to sit. A flat *median* is not
evidence of this — see [Measurement discipline](#measurement-discipline).

**Second half of the same job.** Re-run the mv sweep on `chess/seq-02` (free, on disk). Step D
is one sequence of one scene; a lever that reverses sign on a second sequence is not a lever.
This is already recorded as owed in CLAUDE.md.

**Cost.** ~1 day, mostly compute. The harness exists.

## Workstream 3 — consensus depth / triangulation (highest ceiling, new spec)

**Claim.** Filtering can only delete. The multiview loop already computes every ingredient of a
median-of-inlier-back-projections depth estimate and throws it away. Turning the filter into an
estimator is the highest-ceiling change available, and there is no upstream precedent to copy.

**The circularity that makes this hard.** Depth consistency is self-referential: views that
share a bias agree with each other and score high. A consensus built from the same depths
inherits the bias. Triangulated tracks from a model-agnostic matcher are the only mechanism in
reach that breaks it — which is why the parity spec sequenced tracks+retriangulation as a
separate effort rather than folding it in.

**What exists.** `grep -rn "triangulat" collab_splats/` returns **zero hits**. BA's track source
is `vggt.dependency.track_predict`, which is VGGT-specific and not cross-model; BA is recorded
as a no-op on the small baselines that dominate our data; `geometry/global_alignment.py` is
parked and unreferenced. The matchers do exist and are model-agnostic:
`localization/extractors.py` (DISK, XFeat, LoMa, all with LightGlue).

**Also unresolved here.** Round-trip reprojection (i→j→i pixel error) was ruled out of the
parity pass because discrimination decays to zero as baseline → 0, and video-sampled keyframes
are mostly small-baseline. `test_compute_mv_conf_identical_cameras` asserts co-located cameras
score 1.0 — the degenerate case is a *passing test*. Its natural companion is a minimum-baseline
pair gate, which does not exist.

**Recommended entry point.** Do not start by writing an estimator. Start by measuring how much
headroom there is: take the existing mv machinery, compute the consensus depth it already
discards, and score it against GT on chess/seq-01. If consensus depth is not measurably better
than the model's own depth on the pixels where views disagree, the ceiling is lower than
assumed and the tracks work should be sequenced ahead of it.

**Cost.** New spec, new plan. Weeks, not days.

---

## Sequencing

1 → 2 → 3, for two reasons. Workstream 1 corrects the baseline that 2 and 3 will be measured
against, and costs hours. Workstream 2 reuses a harness that already exists and answers
"is our strongest existing filter tuned?" — a question that must be settled before adding a
second filter or an estimator on top of it. Workstream 3 should not begin until its headroom
measurement (above) says the ceiling is real.

## Measurement discipline

These are not general advice; each one cost time during the multiview pass.

- **Median hides a tail filter.** The first mv sweep looked completely inert because it reported
  median relative depth error. p90, p95, and `frac_over_10pct` are what discriminate. Always
  include an `unfiltered` reference row, or a flat table reads as "no effect" when it is
  actually "wrong statistic".
- **Compare at equal retention.** A filter that removes more pixels will always look better on
  a raw error metric. The only honest comparison is error at matched retention, which is why
  Step D reports both.
- **A threshold nobody mutated is presumed inert.** The shipped `rel=0.05, K=1` retained 99.9%
  of pixels and moved the target metric by 0.0004 — it had been in the code, documented, and
  believed, without ever having been shown to do anything. Before trusting any gate, perturb an
  input and confirm the gate reacts.
- **`min_views` is an absolute count, clamped per pixel.** K=1 is exactly the old
  `mv_conf_threshold=0.0`. K larger than a pixel's partner count degrades to "all partners
  agree", never to "delete".
- **Test-fixture traps in this area.** A view with constant depth has a zero-thickness frustum
  AABB, so the pair gate rejects every pair and `judged` comes back `False` — vary depth across
  the grid. Occlusion is asymmetric: to build a "this view disagrees" fixture the outlier must
  be *nearer* (e.g. 0.7×), because a farther view is occluded and therefore never judged.

## Environment traps

- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Base-shell `python` may be 3.13.
- Never run repo-wide `black .` — the venv's black is newer than the repo's formatting. Format
  only the files you touched, with `--fast`.
- `pytest -p no:randomly` (pytest-randomly is installed).
- `docs/superpowers/` is gitignored; commit with `git add -f`.
- Other sessions share this working tree and index. Commit with an explicit pathspec, and
  re-run before believing a suite failure in a file you did not touch.
- Heavy inference/eval belongs in tmux or a background task, not a notebook. Container cgroup
  cap is 46.6 GB; do not run a second heavy process alongside.

## Open questions for the user

1. Is mesh quality or pose/ATE accuracy the target? Workstream 1 is mesh-only; Workstream 3
   affects the cloud and therefore everything downstream. The answer changes the ordering.
2. Is downloading a second 7-Scenes scene (or another dataset) acceptable? Every conclusion
   currently rests on `chess`.
3. Should `use_multiview_confidence` stay off pending Workstream 2, or should the AND-ed pair be
   re-evaluated as a unit once the learned percentile is calibrated? The two filters interact,
   and calibrating one while the other is off is only valid if they stay independent.
