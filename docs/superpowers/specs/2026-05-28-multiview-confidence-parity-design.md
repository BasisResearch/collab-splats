# Multiview Confidence Diagnostic — Design Spec

**Date:** 2026-05-28  
**Status:** Approved for planning  
**Scope:** Diagnose MapAnything `use_multiview_confidence=True` → 0 surviving points on 50-frame chess seq-01. No production code changes.

---

## Background

Decision 013 (`worklog/decisions/013-multiview-confidence-not-generalised.md`) established:

- VGGT-X and VGGTOmega do **not** benefit from geometric multiview confidence — learned `depth_conf` already saturated (~0.95 mean), filter is redundant.
- MapAnything uses `use_multiview_confidence=True` by default. At 50 frames it produces **0 surviving points**, which is a bug. Root cause unknown.

The eval that produced those findings is `evals/eval_multiview_conf.py`, which already covers VGGT-X/Omega variants. This spec extends that script with a targeted diagnostic for the MapAnything case.

---

## Problem Statement

`MapAnythingCreator(use_multiview_confidence=True)` calls the upstream `postprocess_model_outputs_for_inference` function, which internally calls `compute_multiview_depth_confidence`. At 50 frames on chess seq-01, all confidence values end up ≤ 0, and the p=35 percentile threshold masks every pixel.

Three candidate root causes:

| # | Hypothesis | Signature |
|---|-----------|-----------|
| H1 | `depth_z` not in metric scale — `metric_scaling_factor` not applied before Z extraction, so thresholds (0.02m) are wrong for the actual depth units | External `compute_mv_conf` on MapAnything depth also returns near-zero confidence |
| H2 | `confidence_percentile=35` (our default) too aggressive — upstream default is 10. Bimodal distribution: if 35%+ pixels have zero confidence, the threshold is 0, cutting everything | `MapAnythingCreator(..., confidence_percentile=10)` produces nonzero points |
| H3 | `non_ambiguous_mask` near-empty at 50 frames — passed as `depth_masks` inside the upstream call. When nearly all target depth pixels are masked (zeroed), `|expected_depth − 0| > threshold` → all outliers | Mask density < 20% per frame on 50-frame run |

---

## Design

### Location

Extend `evals/eval_multiview_conf.py`. Add a new `Section 7: MapAnything mv_conf diagnostic`, called from `main()` after the existing summary table. Gated behind `--diagnose` CLI flag (default False) so existing eval runs are unaffected.

### New function

```python
def diagnose_mapanything_mv_conf(
    image_paths: list[Path],
    result_ma: FeedforwardResult,
    result_vggtx: FeedforwardResult,
) -> None:
```

Takes `result_mapanything` (mv_off, already produced in Section 4) and `result_vggtx` (already in memory). `image_paths` needed for H2/H3 reruns.

### H1 block — external reproduce + depth scale comparison

No model reload. Pure numpy/torch.

1. Call existing `compute_mv_conf(result_ma)` helper — same function used for VGGT-X/Omega.
2. Print mv_conf distribution at percentiles: [0, 5, 10, 25, 35, 50, 75, 95, 100].
3. Print surviving point counts: `apply_mask(mv_conf_ma, 35)`, `apply_mask(mv_conf_ma, 10)`, mv_off baseline (~500k).
4. Print depth_z stats side-by-side for MapAnything vs VGGT-X:
   - Computed on `depth[depth > 0]` (valid pixels only)
   - Rows: min, max, mean, p5, p95
   - If MapAnything range is << 0.1 while VGGT-X is 0.3–3.0 → scale mismatch confirmed

**Interpretation:** If external mv_conf on MapAnything depth is also near-zero → bug is in the depth values themselves (H1). If external mv_conf is high (≈0.95) → bug is inside `postprocess_model_outputs_for_inference` mechanics (H2 or H3).

### H2 block — percentile rerun

Gated behind `--diagnose-h2` (slow — loads model again).

Run `MapAnythingCreator(use_multiview_confidence=True, confidence_percentile=10)` on the same image dir. Print surviving point count. Compare to mv_off and p=35 result.

### H3 block — non_ambiguous_mask density

Runs when `--diagnose` is set. Requires a second MapAnything model run (same wall-clock cost as a production run).

Monkey-patch `postprocess_model_outputs_for_inference` to capture `non_ambiguous_mask` values from all processed outputs before masking is applied. Run `MapAnythingCreator(use_multiview_confidence=True)` with the patch active, then restore. Print per-frame mask density (fraction of valid pixels): mean ± std across 50 frames.

```python
from mapanything.utils.inference import postprocess_model_outputs_for_inference as _upstream

captured_masks = []

def _patched(*args, **kwargs):
    out = _upstream(*args, **kwargs)
    for p in out:
        if "non_ambiguous_mask" in p:
            captured_masks.append(p["non_ambiguous_mask"].float().mean().item())
    return out
```

### Output format

Plain text tables matching existing script style:

```
══ MapAnything mv_conf diagnostic (50 frames, chess seq-01) ══════════
H1  external mv_conf:  mean=X.XXX  std=X.XXX
    conf percentile distribution:  p0=X  p5=X  p10=X  p25=X  p35=X  p50=X  p75=X  p95=X  p100=X
    surviving pts @ p35: NNN,NNN  |  @ p10: NNN,NNN  |  mv_off baseline: NNN,NNN

    depth_z stats (valid pixels):
                   MapAnything    VGGT-X
    min             X.XXX          X.XXX
    max             X.XXX          X.XXX
    mean            X.XXX          X.XXX
    p5              X.XXX          X.XXX
    p95             X.XXX          X.XXX

H3  non_ambiguous_mask density:  mean=X.XXX  std=X.XXX  (50 frames)
    → [likely culprit: H1 / H2 / H3 / unclear — print explicit verdict line]
══════════════════════════════════════════════════════════════════════
```

Print a verdict line at the end summarising which hypothesis is implicated based on the findings.

---

## Out of scope

- Fixing the bug (no production code changes)
- Generalising multiview confidence to VGGT-X/Omega (decision 013, closed)
- Running on 500-frame full chess seq-01 (separate follow-up in decision 013)

---

## Success criteria

After running `python evals/eval_multiview_conf.py --diagnose`, we can unambiguously identify which of H1/H2/H3 is the primary cause (or confirm multiple are contributing). Output is sufficient to write a targeted fix in a follow-up session.
