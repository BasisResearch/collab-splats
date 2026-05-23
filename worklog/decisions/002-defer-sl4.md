# ADR 002: SL(4) Pose Graph

- **Status:** Deferred
- **Date:** 2026-04-28
- **Deciders:** Tommy
- **Re-evaluate when:** Project moves to Python 3.11+ AND SE(3) pose graph shows quality ceiling on field data (specifically: residual scale drift after Umeyama alignment that ATE measurably tracks).

## Context

MIT-SPARK/VGGT-SLAM uses SL(4) — a 15-DOF projective transformation group — in its pose graph. SL(4) generalizes SE(3) by adding affine + projective DOFs, useful when the visual reconstruction backbone (VGGT) outputs poses that include scale/skew components beyond rigid SE(3).

GTSAM's SL(4) bindings live in the `develop` branch and require Python 3.11+ wheels. Our nerfstudio environment is pinned to Python 3.10 (transitive constraint from gsplat-rade, mobile_sam, etc.) — SL(4) is unavailable.

## Decision

**Use SE(3) (`Pose3` + `BetweenFactorPose3`) from gtsam 4.2 stable.** Loop closure operates on rigid SE(3) constraints only.

## Reasoning

- SE(3) is mathematically valid for rigid camera poses — the residual quality matters for our ATE, not the group choice.
- VGGT/MapAnything outputs are extracted as rigid SE(3) extrinsics in our pipeline; we never see scale-affine outputs.
- Umeyama overlap-region alignment (F11) absorbs cross-submap scale via translation-only correction; if scale-drift becomes the dominant ATE term, we'd see it in field data and trigger this ADR.
- Python 3.11 migration is non-trivial (gsplat-rade pins, mobile_sam wheels) — not justified by current ATE numbers.

## Alternatives considered

- **Migrate env to Python 3.11.** Rejected: gsplat-rade + mobile_sam wheel uncertainty.
- **Vendor SL(4) bindings from gtsam-develop.** Rejected: maintenance debt + binary compatibility concerns.

## Consequences

- LC pose graph is SE(3) only; no projective DOFs corrected.
- Umeyama (`overlap_region_align`) carries the full burden of cross-submap alignment.
- Risk: scale drift across many submaps if VGGT inference produces inconsistent global scale. Mitigation: translation-jump gate (item 15) catches obvious scale violations.

## Re-evaluation triggers

1. Field data shows ATE > 30% of trajectory length on a closed-loop scene where SE(3) loop edges close cleanly but the trajectory still drifts.
2. Project Python upgrade to 3.11+ for unrelated reasons (then SL(4) becomes a 1-day spike).
3. VGGT-SLAM publishes SL(4) port for Python 3.10.
