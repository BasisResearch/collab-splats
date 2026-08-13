# Multiview confidence — upstream parity, then all backbones

**Date:** 2026-08-13
**Status:** design — ready for planning
**Supersedes:** `2026-08-12-multiview-confidence-all-models-design.md` (the original handoff; its
task list rests on premises that did not survive verification — see "Corrections to the handoff")
**Amended 2026-08-13** against measurement on all four backbones —
`2026-08-13-multiview-confidence-measured-report.md`. Three claims in this doc were overstated and
are corrected inline: the sampler halo is ≤0.2pp (not the central risk), the occlusion fix cannot
change the mask at any K (persisted-ratio improvement only), and the sparse-cloud point-count
acceptance gate is as dead as the mesh-vertex gate. Measured cost: **0.56 s at N=30**
(0.65 ms/pair), ~11 min at N=1000 — the handoff's frame-window mitigation is unnecessary.

## Goal

One geometric multiview-confidence function, shared by all four feedforward backbones, that
**eliminates points without good cross-view correspondence**. Calibrated, default-on where it
measurably wins, with the confidence arrays persisted to `feedforward.zarr`.

Confidence only. Depth stays whatever the model predicted — improving depth itself is deferred to a
later align + triangulate spec.

## Corrections to the handoff

The 2026-08-12 handoff described this as "calibration + wiring + persistence, not new algorithms."
Verification found four premises that do not hold:

1. **T4 has no pattern to follow.** `reconstructor.py:194` is
   `creator_map[backend](max_points=max_points)` — a single hardcoded kwarg. Only `loop_closure` has
   a knob-dict. There is no generic creator-kwarg path to extend.
2. **T1's mesh gate measures nothing.** `mesh/utils.py:434` fuses `result.depth` raw, with no
   confidence mask. mv confidence only feeds `extra_mask` into `unproject_and_filter_points`, which
   produces `points`/`colors`/`pixel_indices`. **A filter cannot move the mesh.**
   *(Amended 2026-08-13, measured:* the sparse-cloud point count is dead for the same reason.
   `unproject_and_filter_points` randomly caps to `max_points` **after** filtering
   (`pointcloud/utils.py:322-323`); the masked pool is 3.8–4.0M pixels against a 500k cap, so all
   four backbones return exactly 500,000 points with and without mv. mv changes cloud composition,
   never its size. **Acceptance must be measured at pixel-mask level or against GT depth.**)
3. **The Omega intrinsics caution is a non-issue.** `vggt_omega.py:233` sets
   `"intrinsics_downsampled": intrinsic` — a literal alias of the same model-res array. Both call
   sites use the identical object. Worth locking with a test, not worth investigating.
4. **`rel_thresh` is hardcoded at three call sites with two different values**, not one:
   `0.02` (`mapanything.py:449`), `0.05` (`vggtx.py:344`), `0.05` (`vggt_omega.py:255`). T4 must
   promote three literals.

And one finding the handoff did not anticipate, which reorders the whole task:

5. **Our `base.py` reimplementation silently diverged from upstream MapAnything.** Details below.

## The central finding: we are not at parity

`collab_splats/pointcloud/feedforward/base.py:378-492` is a reimplementation of
`mapanything/utils/multiview_confidence.py`, deliberately substituted (`mapanything.py:438`:
"replaces upstream use_multiview_confidence path"). It diverged in three ways:

| | Upstream | Ours (`base.py`) | Consequence |
|---|---|---|---|
| Depth sampling | `mode="nearest"` | `mode="bilinear"` (`:473`) | **We introduced an edge halo upstream never had** |
| Pair selection | Frustum-intersection gate, loops only overlapping views | All N² pairs | Cost only (see below) |
| View with no overlapping views | `confidence = 1.0` (keep) | `confidence = 0.0` (drop) | We delete isolated views entirely |
| Denominator | `inlier + outlier` | `valid_sum` | Equivalent — both count valid projections |
| `abs`/`rel` at the MapAnything call site | 0.02 / 0.02 | 0.02 / 0.02 | Preserved ✓ |

**Why the sampler divergence matters — MEASURED, and far smaller than this section originally
claimed.** The mechanism is real: bilinear interpolation across a depth discontinuity produces a
value that lies on no surface (blending foreground 2.0 m and background 8.0 m yields 4.4 m),
manufacturing both false outliers and false inliers. Meta calibrated `abs=0.02, rel=0.02` against
*nearest*; we run those constants against bilinear.

But the effect size is negligible. Measured halo delta (nearest − bilinear) at depth edges:
**−0.19pp (Omega), +0.02pp (VGGT-X), +0.06pp (SPARK); interior ≤0.02pp everywhere** — at every
`rel_thresh` in {0.02, 0.05, 0.10} and both edge definitions (10% and 30% relative depth jump). On
MapAnything's real shipping config the sampler change moves 36,442 pixels, 1.32% of valid, +0.519pp
retention.

The original estimate conflated "pixels near an edge in the *source* frame" with "pixels whose
*projection* lands within a pixel of an edge in the *target* frame" — the latter is a far thinner
set. Compounding this: the learned confidence filter already removes edges preferentially (edges are
12.2% of valid pixels but **0.2%** of learned-kept pixels on Omega), so most of what bilinear blurs
is gone before mv runs.

**Restoring nearest is still correct** — it is upstream parity, it is where Meta's constants were
calibrated, and it costs nothing. It is simply not the central risk of Step A, and it does not
justify the "latent defect, this task is the trigger" framing this section previously carried. Full
numbers: `2026-08-13-multiview-confidence-measured-report.md`.

**Upstream does not refine depth.** `mapanything/utils/inference.py:401-402` replaces the *learned
confidence* with the mv ratio (`processed_output["conf"] = mv_conf_list[i]`); `depth_z` is never
touched. No consensus depth exists anywhere in the package. Ours intersects rather than substitutes
(`mapanything.py:451`: `combined_mask & (mv_conf > threshold)`), and the docstring at
`mapanything.py:115-146` describes substitution while the comment at `:386` says the percentile is
always applied. **Reconcile against the code during Step A; trust neither.**

## Design

### The shared function

Model-agnostic by construction — it consumes only `(depth, K, E)`, which every backbone emits.
Generality is not the hard part; the unstated contract is:

- **Z-depth**, not ray length, not disparity, not shifted-affine. VGGT family ✓, MapAnything ✓.
  A future backbone emitting disparity produces silent garbage. Currently unasserted.
- **depth, K, E pixel-aligned at one resolution**, OpenCV convention, +Z forward. Currently
  unasserted — the same bug class that caused the mesh intrinsics regression.
- **Scale invariance via `rel_thresh`.** Multiply all depth by *s*: `expected_d`, `sampled_d` and the
  tolerance `abs + rel·expected_d` all scale together, so with `abs=0` the output is invariant. This
  is the property that lets one function serve four models with different depth scales, and it is
  why `abs_thresh` must stay `0.0` for the non-metric VGGT backbones.

Signature:

```python
@dataclass
class MultiviewConfidence:
    ratio: np.ndarray         # (N, H, W) float32 — inlier / valid, upstream-compatible
    inlier_count: np.ndarray  # (N, H, W) int32
    valid_count: np.ndarray   # (N, H, W) int32
    judged: np.ndarray        # (N,) bool — False if the view had no overlapping partners

def compute_multiview_depth_confidence(
    depth, intrinsics, extrinsics,
    depth_masks=None, abs_thresh=0.0, rel_thresh=0.05,
    pair_gate=True, device="cuda",
) -> MultiviewConfidence
```

Mask derivation lives in one helper so all four backbones share it:

```python
# valid_depth = (depth > 0), further restricted by depth_masks when the caller passes one.
# Unjudged views (no overlapping partners) keep their valid pixels, matching upstream.
mask = np.where(judged[:, None, None], inlier_count >= min_views, valid_depth)
```

### `min_views`, not a float threshold or a percentile

Replace `mv_conf_threshold: float` with `min_views: int` — "at least K other views agree."

- **Percentile is a documented dead end.** `mapanything.py:127-137`: mv_conf is a quantized ratio
  k/N with a large atom at 1.0, so `torch.quantile(conf, p)` collapses to 1.0 for any p where >0% of
  pixels sit at 1.0, and strict `conf > threshold` then excludes every pixel including those at
  exactly 1.0. Percentiles need smooth learned-confidence distributions.
- **A count sidesteps the quantization entirely** — it thresholds in the units that are actually
  discrete, rather than fighting atoms in a derived ratio.
- **Stable across sequence length.** "≥2 views agree" means the same at N=10 and N=1000. A ratio does
  not. Our sequences span both.
- **Exact MapAnything parity at K=1.** `mv_conf > 0.0` ⟺ `inlier_count ≥ 1`, since
  `ratio > 0 ⟺ inlier_count > 0`. Bit-identical to today, so the parity gate holds by construction.
- K=2 is the MVS standard (COLMAP fusion, MVSNet).

### Occlusion asymmetry — improves the persisted ratio only, never the mask

**Amended 2026-08-13, measured: this change cannot alter the filter at any K.** Symmetric and
asymmetric produce byte-identical masks on all four backbones at K=1,2,3,4, and differ by exactly
**0 pixels** on MapAnything's real shipping config. The reason is structural and stronger than the
K=1 proof below: the occlusion policy alters only `valid_count` (the denominator), while `min_views`
thresholds `inlier_count` (the numerator). Under count thresholding the fix is **inert**.

Keep it — but justify it as improving the **persisted `ratio`** for downstream consumers (splat
depth supervision, per-frame diagnostics), not as improving the filter. It is not an argument for
Step B carrying filter risk, and `test_positive_mask_invariant_across_occlusion_policy` should
assert invariance at K=1..4, not only K=1.

The mechanism, unchanged:

`base.py:481` tests `|expected − sampled| < tol` symmetrically. But the two directions mean opposite
things:

- `sampled < expected − tol` — something is **in front**. The view is occluded. It is *evidence
  absent*, not evidence against.
- `sampled > expected + tol` — nothing is there. A **free-space violation**: real evidence the source
  depth is wrong.

Counting occlusion as disagreement punishes correct geometry for being occluded. A pixel visible in
2 of 10 views scores 0.2 and dies at any threshold above that. Fix: exclude occluded views from
`valid_count`; keep free-space violations as outliers.

**This is provably mask-neutral for MapAnything.** Occluded views leave the denominator; the
numerator is untouched, because an occluded view was never an inlier. So:

- `inlier_count > 0` ⇒ those inlier views are by definition not occluded ⇒ they survive in the
  reduced `valid_count` ⇒ still kept.
- `inlier_count == 0` ⇒ dropped either way.

`(inlier_count ≥ 1)` is therefore invariant, and MapAnything runs at `min_views = 1`. Protected by
proof, not only by test.

### Pair gating is a cost optimisation, not a quality one

Upstream's frustum gate skips non-overlapping pairs. Our `in_bounds` check (`base.py:463`) already
excludes those projections from both accumulators, so gating them earlier changes results only via
the view-level exemption — it is primarily about not launching N² GPU kernels. Say so plainly rather
than claiming a quality win.

Implement our own lightweight frustum-overlap test in `base.py` (near/far from each view's depth
range, frusta in world space, pairwise intersection). **Do not import
`mapanything.utils.wai.intersection_check`** — the shared function must not depend on one backend's
package, and project memory records a torch-compat monkey-patch (`_patch_mapanything_torch_compat`)
against exactly that function.

Suppressing *near-duplicate* pairs is a different mechanism (a minimum-baseline gate) and is
deferred with the round-trip work below.

### Config surface: one boolean

```yaml
pointcloud:
  backend: vggt_omega
  use_multiview_confidence: true    # the only knob
```

`rel_thresh` and `min_views` stay as calibrated **field defaults on each creator** — real fields so
the sweep can drive them, absent from `base.yaml` so nobody hand-tunes floats. Step D picks them,
Step E bakes them in. `abs_thresh` stays MapAnything-only and undocumented elsewhere: it is a dead
parameter for non-metric depth and must remain `0.0` there.

Read it in `reconstructor.py` explicitly, the way `max_points` is read today. One boolean does not
justify a knob-dict layer.

### Persistence

New `feedforward.zarr` arrays beside `confidence`, chunked by frame, written only when mv was
computed (no key at all when it wasn't — not a zeros array, and no backfill of existing scenes):

```
mv_ratio         (N, H, W) float32
mv_inlier_count  (N, H, W) int32
mv_valid_count   (N, H, W) int32
```

Counts are free — both accumulators already exist in the loop — and they are what the later
triangulate/align spec will threshold on. Zarr is v3 (3.1.5): `compressors=[BloscCodec(...)]`, not
`codecs=`.

## Scope of work

**Step A — restore upstream parity.** Gate: MapAnything on `data/outputs/`, sparse cloud point count.
1. `mode="bilinear"` → `"nearest"` (`base.py:473`).
2. Add the frustum pair gate.
3. View-level exemption: a view with no overlapping partners keeps its valid-depth pixels
   (`judged=False`), matching upstream. Pixel-level behaviour is unchanged — a pixel that projects
   into no valid view is still dropped, which already matches upstream.
4. Reconcile substitute-vs-intersect in `mapanything.py` against the code; fix whichever of the
   docstring (`:115-146`) or the comment (`:386`) is wrong.

**Step B — improve on upstream.** Provably MapAnything-neutral.
5. Occlusion exclusion.
6. Return `MultiviewConfidence`; add the shared mask helper.
7. Convention assertions: depth/K resolution agreement; document the Z-depth contract.
8. Drop the redundant `valid_sum.clamp(min=1.0)` inside a `torch.where` already guarded by
   `valid_sum > 0` (`base.py:489`).

**Step C — wire the three VGGT backbones.**
9. Promote the three hardcoded `rel_thresh` literals to fields; replace `mv_conf_threshold: float`
   with `min_views: int` on all four creators. **MapAnything ships `min_views = 1`** — exactly
   equivalent to its current `mv_conf_threshold = 0.0`, so Step C changes none of its behaviour.
   The VGGT backbones keep `min_views = 1` until Step D says otherwise.
10. Read `use_multiview_confidence` in `reconstructor.py`; add it to `configs/base.yaml`.
11. Persist the three arrays.

**Step D — calibrate.** 7-Scenes, GT depth available via `evals/datasets.py`.
- Metric: **retained-pixel depth error vs GT, against retention fraction.** VGGT depth is non-metric,
  so align per scene by median ratio `s = median(d_gt) / median(d_pred)` over valid GT pixels before
  computing relative error. A win = lower retained-pixel error at comparable retention than the
  current learned-confidence percentile baseline.
- Sweep `rel_thresh` × `min_views` (K ∈ {1, 2, 3, 4}).
- **Start with one shared `rel_thresh`.** It is scale-invariant, so per-backbone optima should
  cluster. A wide spread indicates a convention bug, not a model difference — use it as a sanity
  check, and split per-backbone only if the data forces it.
- Rewrite `evals/scripts/eval_multiview_conf.py`: it imports `mapanything.utils.multiview_confidence`
  rather than the shipping `base.py` function, so **it measures the wrong code**, and its
  `--diagnose`/`--diagnose-h2` flags monkey-patch upstream postprocessing. Replace with a sweep
  harness over the shared function. CLI/tmux only; results under `evals/results/` (gitignored).

**Step E — flip defaults.** Per backbone, only where Step D shows a win. Record the sweep table and
any backbone left off, with the reason.

## Tests

Flat functions, no classes. `tests/pointcloud/test_mv_conf.py` plus per-creator files.

| Test | Guards |
|---|---|
| `test_occluded_view_excluded_not_penalised` — occluder slab in front of cam *j*; correct source depth keeps full confidence | the occlusion fix |
| `test_positive_mask_invariant_across_occlusion_policy` — seeded scenes, `(old > 0) == (new ≥ 1)` | **the MapAnything guarantee, as a property test** |
| `test_min_views_one_matches_ratio_gt_zero` | the K=1 ⟺ `ratio > 0` parity equivalence |
| `test_scale_invariance` — scale depth and translations by 7.3 with `abs_thresh=0` → identical output | the contract that makes one function serve four models; currently untested |
| `test_nearest_sampling_no_edge_halo` — step edge, count rejections within 2 px of the discontinuity | the sampler divergence, quantified |
| `test_counts_consistent_with_ratio` | the persistence contract |
| `test_unjudged_view_keeps_pixels` | the view-level exemption |
| `test_intrinsics_resolution_mismatch_raises` — K implying a different grid than `depth.shape` | **the mesh-regression bug class** |
| `test_omega_mv_and_unprojection_share_intrinsics` — same array object at both call sites | locks the `intrinsics_downsampled` alias |
| `test_<backend>_mv_kwargs_forwarded` ×3 — monkeypatch the shared function, assert kwargs match the dataclass fields | catches a hardcoded literal surviving promotion |
| `test_spark_inherits_mv_path` — **fresh process** via subprocess | the import-cache trap; `_assert_loaded_from_spark` guards it |
| `test_zarr_roundtrip_mv_arrays` — absent keys when mv was not computed, not zeros | persistence |

Explicitly **not** written: a test pinning MapAnything's literal default values. It is a
change-detector that fails on every intentional edit and proves nothing about behaviour; the
invariance property test is the real guarantee.

Any test touching semantics write paths must respect the artifact-naming contract (CLAUDE.md).

## Acceptance criteria

1. `base.py` matches upstream on sampler, pair gating and no-overlap semantics. Gate: MapAnything's
   **pixel-level mask retention** on `data/outputs/`, not point count (see correction 2 above —
   `max_points` saturates). Measured baseline to reproduce: shipping mask keeps **81.646%** of
   2,752,924 valid pixels; after the sampler change, **82.165%** (+0.519pp, 36,442 px differing).
   A larger deviation than that means something else changed.
2. Occlusion exclusion landed, with the mask-invariance property test passing **at K=1..4** (measured
   invariant at all four; the doc's proof covers only K=1).
3. All four backbones share one code path and one `min_views` semantic. **One `rel_thresh` for all
   three VGGT backbones, not per-backbone** — measured retention at `rel_thresh=0.05`, K=1 is
   86.03% / 86.56% / 86.32% (Omega / VGGT-X / SPARK), inside 0.53pp at every K. Sweep table recorded
   in the completion notes.
4. `use_multiview_confidence` is the only mv key in `configs/base.yaml`.
5. `mv_ratio` / `mv_inlier_count` / `mv_valid_count` persisted when computed; absent otherwise.
6. Spark verified to execute the path in a fresh process.
7. Suite green: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly`
   (pytest-randomly is installed; disable it). `docs/known-test-failures.md` entries exempt.

## Out of scope

- **Consensus depth.** The loop computes every ingredient for a median-of-inlier-back-projections
  depth estimate and discards it — turning the filter into an estimator is the highest-ceiling
  follow-up, and no upstream precedent exists. Deferred deliberately: depth gets improved by
  alignment and triangulation, not by this filter.
- **Round-trip reprojection (i→j→i pixel error).** Discrimination decays to zero as baseline → 0, and
  our frames come from video sampling, so consecutive keyframes contribute confident,
  information-free agreement. `test_compute_mv_conf_identical_cameras` currently asserts co-located
  cameras score 1.0 — the degenerate case is a passing test. Known limitation; document, do not fix
  here. Its natural companion is a minimum-baseline pair gate.
- **Tracks + retriangulation.** Separate spec, sequenced second. `grep -rn "triangulat"` over
  `collab_splats/` returns **zero hits** — nothing exists. BA's track source is
  `vggt.dependency.track_predict` (`bundle_adjustment.py:28`), VGGT-specific and not cross-model;
  BA is recorded as a no-op on the small baselines that dominate our data;
  `geometry/global_alignment.py` is parked and unreferenced. The model-agnostic matchers do exist
  (`localization/extractors.py` — DISK, XFeat, LoMa with LightGlue). Value: depth consistency is
  **self-referential** — views sharing a bias agree with each other and score high — so triangulated
  tracks are the only mechanism that breaks the circularity. Build it after poses and filtering are
  settled, not alongside.
- **Wiring `vggt_spark` into `reconstructor.py`'s `creator_map`.** It is absent, so Spark is
  unreachable from config; verification is test-only this pass, by decision.
- The gsplat trainer port and continuous-weight loss supervision.

## Environment traps

- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Base-shell `python` may be 3.13.
- Never run repo-wide `black .` — venv black 26.5.1 is newer than repo formatting. Format only
  touched files.
- Concurrent sessions edit `configs/base.yaml` mid-run. A sudden ~60 unrelated failures means re-run
  before believing it.
- Commit with explicit pathspec (parallel sessions share the index). `docs/superpowers/` needs
  `git add -f`.
- Heavy eval runs in tmux, no parallel heavy side-shells (cgroup cap 46.6 GB).
- Check `git status` before `Write` on a "new" test file — a prior session clobbered an existing
  `tests/mesh/test_tsdf.py` this way.
- Conventional commits, e.g. `fix(pointcloud): restore upstream parity in multiview confidence`.
