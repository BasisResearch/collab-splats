# ADR 003: GraphMap (Multi-Session Map Storage)

- **Status:** Deferred
- **Date:** 2026-04-28
- **Deciders:** Tommy
- **Re-evaluate when:** A second collaborating robot or session needs to share submap state in real time.

## Context

VGGT-SLAM uses a `GraphMap` data structure to persist submaps + factor graph state across sessions, enabling multi-session SLAM where a second robot can localize against a prior map. Our use-case is currently single-robot, single-session preprocessing for Gaussian Splatting — submap state is in-memory only and discarded after `reconstruct()` returns.

## Decision

**No `GraphMap` abstraction.** Submaps live in-memory inside `_run_loop_closure_inference`; pose graph is rebuilt each run.

## Reasoning

- Single-session: no persistence requirement.
- The `Submap` dataclass already carries everything `GraphMap` would need (poses, image_paths, world_points, retrieval_vectors). Promoting to a persistent store is mechanical when needed.
- Adding GraphMap now adds a serialization format, on-disk layout, and version-skew concerns — premature.

## Alternatives considered

- **In-memory dict keyed by session_id.** Rejected: not yet useful, and would expose a half-baked API.
- **Pickle submaps to disk per run.** Rejected: nerfstudio cache already persists `transforms.json` + COLMAP binaries; LC adds nothing reusable across runs.

## Consequences

- Multi-session support requires future work (persist + reload submaps, merge factor graphs).
- No risk to current single-session correctness.

## Re-evaluation triggers

1. User asks to split a long session across multiple `Splatter.reconstruct()` calls and have the second call build on the first's pose graph.
2. Multi-robot CollabSplat scenario where two operators capture overlapping scenes and want a shared map.
3. Quality benefit observed when re-running LC over a previous session's submaps (e.g., revisit detection improves with a memory of priors).
