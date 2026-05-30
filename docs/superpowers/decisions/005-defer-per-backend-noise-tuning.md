# ADR 005: Per-Backend Noise Tuning

- **Status:** Deferred
- **Date:** 2026-04-28
- **Deciders:** Tommy
- **Re-evaluate when:** Field ATE on at least 3 distinct scenes shows MapAnything and VGGTX have systematically different residual distributions (e.g., MapAnything intra-noise needs to tighten, VGGTX inter-noise needs to loosen).

## Context

`PoseGraph` exposes 4 noise buckets — `_odom_intra`, `_inter`, `_loop_intra`, `_loop_inter` — with starting sigmas chosen from theoretical priors (VGGT-SLAM tuning + heuristic ratios):

| Bucket | σ_r (rad) | σ_t (m) | Rationale |
|---|---|---|---|
| odom_intra | 0.02 | 0.05 | VGGT/MapAnything intra-window is coherent |
| inter | 0.05 | 0.20 | Submap boundary needs Umeyama room |
| loop_intra | 0.10 | 0.30 | Rare; unconstrained prior |
| loop_inter | 0.15 | 0.50 | LC pair re-run can fail; Huber backstop |

These sigmas are the same across MapAnything and VGGTX backends. In principle each backend has its own residual distribution and could justify backend-specific tuning.

## Decision

**One sigma table for all backends.** Tune by field measurement, not theoretical priors.

## Reasoning

- Without a representative dataset of field runs, per-backend tuning is theatre — the numbers would still be guessed, just guessed twice.
- Huber kernel + translation-jump gate already absorb backend-specific outlier behavior; sigma tuning is a finer correction.
- Pre-launch tuning risks overfitting to development scenes that don't represent field conditions.

## Alternatives considered

- **Tune sigmas now from synthetic ATE optimization.** Rejected: synthetic scenes don't capture VGGT/MapAnything failure modes.
- **Expose sigmas as `LoopClosureConfig` fields, leave defaults shared.** Possible refinement; defer until first field-tuning iteration to know which knobs we actually need.

## Consequences

- Both backends use identical sigma table. If one is systematically more confident, we under-trust it (safe direction — Huber will accept good edges anyway).
- Risk: starting sigmas mis-balance a specific backend → suboptimal LM convergence. Mitigation: log per-edge residual magnitudes after `optimize()` so we have data to retune from.

## Re-evaluation triggers

1. Three or more field scenes captured where ATE breakdown shows MapAnything edges and VGGTX edges contribute residuals at systematically different magnitudes (>2× ratio).
2. User reports "MapAnything LC works but VGGTX doesn't" (or vice versa) in a way that residual analysis confirms.
3. Decision to add a third backend (DUSt3R, MASt3R, etc.) — natural moment to revisit the sigma table.
