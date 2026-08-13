# Multiview confidence — measured effect on all four backbones

**Date:** 2026-08-13
**Status:** measurement report; companion to `2026-08-13-multiview-confidence-parity-design.md`
**Question asked:** how do the proposed changes alter MapAnything's output, and how does
applying multiview consistency alter the other backbones' outputs?

All numbers below are measured, not estimated. Scene: `data/outputs/frames.zarr`, 30 frames,
one A40. Each backbone was re-run from the same frames in a fresh process
(`max_points=500000`, matching `configs/base.yaml`).

## Harness validation

The variant harness reproduces the shipping function exactly before any variant is trusted:

```
harness vs shipping fn: max|diff| = 0.000e+00   masks equal (>0): True
```

`depth_masks` is omitted from variant runs deliberately. It enters only through `src_valid`
(`base.py:436-437`), identically in every variant, so agreement across the full pixel set
implies agreement on any masked subset. MapAnything is the only caller that passes it.

## Part 1 — How the proposed changes alter MapAnything

Shipping config (`mapanything.py:436-455`): `abs_thresh=0.02` m, `rel_thresh=0.02`,
`mv_conf_threshold=0.0`, bilinear sampling, symmetric occlusion. Depth is metric
(min 2.27 m, median 4.92 m, max 73.85 m). 30 frames at 518×294, 2,752,924 valid pixels.

**Shipping mask keeps 81.646% of valid pixels (2,247,648 px).** Against that baseline:

| variant | K≥1 keep | vs shipping | pixels differing |
|---|---|---|---|
| bilinear/symmetric (today) | 81.646% | — | 0 |
| bilinear/asymmetric | 81.646% | +0.000pp | **0** |
| nearest/symmetric (upstream parity) | 82.165% | +0.519pp | 36,442 |
| nearest/asymmetric | 82.165% | +0.519pp | 36,442 |

`min_views` sweep, today's sampler:

| K | keep | vs shipping |
|---|---|---|
| 1 | 81.646% | +0.00% (exact parity) |
| 2 | 66.738% | −18.26% |
| 3 | 56.805% | −30.42% |
| 4 | 50.345% | −38.34% |
| 5 | 43.546% | −46.66% |

**Verdict:** with `min_views=1` retained as MapAnything's default, the total change to its
output is 36,442 pixels — 1.32% of valid pixels, +0.519pp retention — and all of it comes
from the sampler. The occlusion change moves nothing. MapAnything is safe.

## Part 2 — Applying multiview consistency to the other backbones

`rel_thresh=0.05`, `abs_thresh=0.0` (non-metric depth). Retention of valid pixels:

| backbone | res | valid px | K≥1 | K≥2 | K≥3 | K≥4 |
|---|---|---|---|---|---|---|
| VGGT-Omega | 688×384 | 7,925,760 | 86.03% | 72.76% | 61.15% | 53.33% |
| VGGT-X | 518×518 | 8,049,720 | 86.56% | 73.33% | 61.59% | 53.29% |
| VGGT-SPARK | 518×518 | 8,049,720 | 86.32% | 73.13% | 61.58% | 53.26% |

The three land within 0.53pp of each other at every K. One `rel_thresh` serves all three;
per-backbone calibration of `rel_thresh` is not needed, which collapses the handoff's T1
sweep from three sweeps to one. `rel_thresh` sensitivity (VGGT-Omega, K=1): 76.82% at 0.02,
86.03% at 0.05, 90.90% at 0.10.

### mv's marginal effect over the learned-confidence filter

Standalone retention overstates mv's impact, because the creators AND it with a learned
confidence percentile (`conf_threshold=50.0`). The number that matters is how many pixels mv
removes that the learned filter would have kept:

| backbone | rel | K | mv alone | mv ∧ learned | marginal kill | edge share of kill |
|---|---|---|---|---|---|---|
| Omega | 0.05 | 1 | 86.03% | 45.55% | **8.91%** | 0.2% |
| Omega | 0.05 | 2 | 72.76% | 40.54% | 18.92% | 0.2% |
| VGGT-X | 0.05 | 1 | 86.56% | 47.48% | **5.08%** | 2.8% |
| VGGT-X | 0.05 | 2 | 73.33% | 42.84% | 14.35% | 1.5% |

Rejection-set overlap at rel=0.05, K=1: Omega Jaccard 0.175 (learned rejects 3,962,880;
mv rejects 1,106,929; overlap 753,858), VGGT-X Jaccard 0.208. So roughly two thirds of what
mv rejects was already gone, and mv contributes ~353k genuinely new rejections on Omega.

mv's kills are **correlated with, but not reducible to, learned confidence.** Kill rate by
learned-confidence quintile among learned-kept pixels (Omega, K=1, low→high):
15.3% / 16.6% / 8.9% / 3.4% / 0.4%. Monotone — mv agrees with the learned model's ordering —
but it still rejects 0.4% of the most-confident quintile. Those confident-but-geometrically-
inconsistent pixels are mv's unique contribution; nothing else in the pipeline finds them.

### mv is a bad-frame detector, not a uniform trim

Per-frame marginal kill on Omega at K=1: min 0.0%, median 1.7%, **max 62.8%**. Frames 21, 22,
26 dominate. Per-frame breakdown shows a step change around frame 20 — median inlier ratio
runs 0.81–1.00 for frames 0–17 and 0.09–0.54 for frames 21–29.

This is not an overlap artifact. Mean overlapping partners per pixel never drops below 8.88
on Omega (4.98 on VGGT-X); **no frame is partner-starved**. The decisive comparison: frame 17
has 8.88 mean partners and median ratio 1.000, frame 21 has 9.93 partners and median ratio
0.136. Same volume of evidence, 7× difference in agreement. mv is measuring geometry, not
coverage. Correlation between mean partners and K≥1 retention is +0.569 (Omega) / +0.687
(VGGT-X) — real but not the driver.

The practical consequence: the second half of this sequence is geometrically inconsistent
with the first half, and mv localises it. That is a reconstruction defect worth surfacing
independently of filtering.

### Performance

Measured, warm: **0.56 s for N=30** at 688×384, 0.58 s at 518×518 — 0.64–0.66 ms per frame
pair. Extrapolating the O(N²) pair count: ~0.4 min at N=200, ~11 min at N=1000. Not a
blocker at any scale this project runs; the handoff's frame-window mitigation is unnecessary.

## Three corrections to the parity design doc

Measurement contradicts three things in
`2026-08-13-multiview-confidence-parity-design.md`. The spec has been amended.

1. **The bilinear→nearest sampler divergence is not the central risk of Step A.** Measured
   halo delta at depth edges: −0.19pp (Omega), +0.02pp (VGGT-X), +0.06pp (SPARK); interior
   ≤0.02pp everywhere, at every `rel_thresh` in {0.02, 0.05, 0.10} and both edge definitions
   (10% and 30% relative depth jump). My error was conflating "pixels near an edge in the
   source frame" with "pixels whose projection lands within a pixel of an edge in the target
   frame" — the latter is a far thinner set. Reinforcing this: the learned confidence filter
   already removes edges preferentially (edges are 12.2% of valid pixels but 0.2% of
   learned-kept pixels on Omega), so the pixels bilinear blurs are mostly gone before mv runs.

2. **The occlusion-asymmetry fix cannot change the mask at any K, not just K=1.** Measured:
   symmetric and asymmetric produce byte-identical masks on all four backbones at K=1,2,3,4,
   and differ by exactly 0 pixels on MapAnything's real config. The reason is structural —
   the occlusion policy only alters `valid_count` (the denominator), while `min_views`
   thresholds `inlier_count` (the numerator). Under count thresholding the fix is inert. It
   affects only the persisted `ratio`, so it should be justified as improving the persisted
   signal for downstream consumers, not as improving the filter. The spec's proof covered
   only K=1; the empirical result is stronger.

3. **The sparse-cloud point-count acceptance gate is dead, like the mesh-vertex gate.**
   `unproject_and_filter_points` randomly caps to `max_points` *after* confidence filtering
   (`pointcloud/utils.py:322-323`). The masked pool is 3.8–4.0M pixels against a 500k cap, so
   every backbone returned exactly 500,000 points with and without mv. mv changes cloud
   composition, never its size. Acceptance must be measured at pixel-mask level, or against
   GT depth on 7-Scenes.

## What this does not establish

Retention is not accuracy. Nothing here shows the pixels mv removes are *wrong* — only that
they disagree across views, that the disagreement is not an overlap artifact, and that it is
concentrated where a human would expect a reconstruction defect. The GT-depth evaluation on
7-Scenes (Step D of the parity spec) remains the test that decides whether default-on is
justified, and it is unchanged by this report.

## Reproduction

Scripts under the session scratchpad: `mv_variants.py` (variant sweep + halo split),
`mv_validate.py` (harness validation + `rel_thresh` sweep), `mv_marginal.py` (marginal effect
over learned confidence), `mv_interpret.py` (kill-set composition), `mv_perframe.py`
(per-frame overlap vs agreement), `mv_mapany_parity.py` (MapAnything shipping-config parity),
`run_backend.py` (one backend per fresh process — required for SPARK's
`_assert_loaded_from_spark` guard).
