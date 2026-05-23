# ADR 004: FrameTracker (Continuous Pose Tracking Inside Submap)

- **Status:** Deferred
- **Date:** 2026-04-28
- **Deciders:** Tommy
- **Re-evaluate when:** VGGT/MapAnything intra-window pose drift becomes the dominant ATE term on field data.

## Context

VGGT-SLAM has a `FrameTracker` component that runs frame-to-frame pose tracking *inside* a submap window, complementing the foundation model's batch inference. This handles fast camera motion or scenes where the model's window inference produces noisy intermediate poses.

Our `_odom_intra_noise` (σ_t=0.05, σ_r=0.02) is tight precisely because we trust VGGT/MapAnything intra-window outputs. Adding a FrameTracker would loosen this trust and introduce a second pose source to fuse.

## Decision

**Trust the foundation model intra-window.** No FrameTracker. `_odom_intra_noise` stays tight; intra-submap edges constrain the LM optimizer.

## Reasoning

- VGGT/MapAnything are trained for batch coherent inference — their intra-window poses are temporally consistent by construction.
- Adding a frame-tracker introduces a fusion problem (which source to trust when they disagree?) without measured evidence the foundation model is failing.
- Most failure modes we worry about (loop closures, cross-submap drift) occur at submap boundaries — that's where `_inter_noise` and Umeyama do work.

## Alternatives considered

- **Optical-flow-based intra-window tracker.** Rejected: increases per-window compute by ~20–30%, no measured ATE benefit.
- **Bundle-adjustment refinement inside submap.** Rejected: VGGT already does this internally; running a second pass duplicates work.

## Consequences

- Reliance on foundation model quality for intra-window poses.
- Risk: fast-motion scenes where VGGT loses intra-window consistency manifest as inflated ATE within a submap. Mitigation: shrink `submap_size` config (smaller windows, less drift accumulation).

## Re-evaluation triggers

1. Field test on a fast-motion scene shows intra-submap pose error > inter-submap error (the inverse of what we expect).
2. ATE breakdown attributes >30% of total error to intra-submap residuals (vs. cross-submap or LC residuals).
3. VGGT-Long or another upstream publishes a clean FrameTracker we can adopt with low integration cost.
