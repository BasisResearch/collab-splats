# ADR 011: `CoordinateFrame` Enum + `world_transform`

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** pointcloud, coordinate-frames, sfm

## Context
Each pointcloud backend produces points and poses in its own native coordinate convention: COLMAP world frame is z-up but origin-arbitrary; VGGT-X uses a normalized scene-centered frame; MapAnything assumes a metric reconstruction with a derived world frame. Downstream stages (mesh, dashboard render, eval against ground truth) require points in a known frame. Hard-coding conversions per backend at every consumer is brittle and easily inconsistent.

## Decision
A `CoordinateFrame` enum names the source convention (`COLMAP`, `VGGTX`, `MAPANYTHING`, `METRIC_WORLD`, …). Every `PointcloudResult` carries a `frame: CoordinateFrame` field plus a `world_transform: SE3` that maps from `frame` to the canonical metric world frame. Consumers that need world-frame points apply `world_transform`; consumers that need native frame use the points as-is.

## Consequences
**Positive:**
- Self-describing results — a `PointcloudResult` is interpretable without out-of-band knowledge.
- One conversion logic per backend instead of N consumer-side branches.
- Eval against ground truth becomes uniform.

**Negative:**
- Every backend must populate `world_transform` correctly. Mistakes here propagate silently.
- Adding a new frame requires a new enum variant and corresponding conversion.

**Revisit if:** backends start producing multi-frame outputs (e.g. dense scene + sparse landmarks in different frames) — the single-frame field will no longer be enough.

## Alternatives Considered
- **Force all backends to a canonical frame at creation time.** Rejected: backend-internal optimization stages benefit from native frame.
- **String frame identifiers instead of enum.** Rejected: typo risk + no IDE completion.
