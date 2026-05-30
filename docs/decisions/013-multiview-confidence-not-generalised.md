# 013 — Multiview confidence not generalised to VGGT-X/Omega

**Date:** 2026-05-28
**Status:** Decided — do not implement

## Decision

Do not add `compute_multiview_depth_confidence` (MapAnything's geometric
cross-view depth consistency filter) as a generalised postprocessing feature
for `VGGTXCreator` or `VGGTOmegaCreator`.

## Context

MapAnything uses `compute_multiview_depth_confidence` by default
(`use_multiview_confidence=True`) during `_postprocess` to filter unreliable
depth pixels. The question was whether applying the same geometric filter to
VGGT-X and VGGTOmega would improve point cloud quality (and downstream BA
track quality).

## Empirical findings

Eval run: chess seq-01, 50 frames, p35 percentile threshold.
Script: `evals/eval_multiview_conf.py`
Outputs: `evals/results/mv_conf_eval/`

```
Model          Variant          N_points  mv_conf_mean
VGGT-X         learned_only    6,600,612         0.954
VGGT-X         mv_only         8,692,466         0.954
VGGT-X         intersect       6,113,530         0.954
VGGTOmega      learned_only    8,619,521         0.938
VGGTOmega      mv_only        11,057,377         0.938
VGGTOmega      intersect       8,576,053         0.938
MapAnything    mv_off            500,000           n/a
MapAnything    mv_on                   0           n/a  ← bug (see below)
```

**mv_conf mean ≈ 0.95 for both models** — depth predictions are already ~95%
geometrically consistent cross-view. Signal is nearly saturated; almost nothing
to filter.

**mv_only is more permissive than learned_only** — at the same percentile
threshold, the geometric filter retains 28–32% more points than the learned
`depth_conf` filter. Adding mv_conf would increase density, not improve quality.

**intersect removes only 0.5–7% additional points** — the two signals are
largely redundant. The learned `depth_conf` head already captures the same
uncertainty the geometric filter would target.

## Reason

The learned `depth_conf` head in VGGT-X/Omega is trained end-to-end and
produces strong per-pixel uncertainty estimates. Geometric cross-view
consistency is a coarser proxy that adds no information on top of a model
already producing high-quality, geometrically consistent depth maps.

## Open bug: MapAnything mv_on = 0 points

`MapAnythingCreator(use_multiview_confidence=True)` produces 0 surviving points
on 50-frame chess seq-01. The production default is `use_multiview_confidence=True`.

Suspected cause: the internal depth thresholds in `compute_multiview_depth_confidence`
(`depth_assoc_abs_thresh=0.02`, `depth_assoc_rel_thresh=0.02`, in metres) are
calibrated for metric depth but MapAnything's `depth_z` output may be on a
different scale or in a different convention at this frame count. On the full
sequence the masking may degrade gracefully rather than zeroing out completely —
needs verification.

**Next step:** run MapAnything mv_on on the full chess seq-01 (500 frames) and
compare point counts to mv_off. If still 0, investigate depth scale in
`mapanything.utils.inference.postprocess_model_outputs_for_inference`.
