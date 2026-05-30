# ADR 010: Pointcloud Creator Registry

**Status:** Accepted
**Date:** 2026-04-16
**Tags:** pointcloud, registry, sfm, feedforward

## Context
Pointcloud reconstruction supports multiple backends with different requirements: COLMAP (binary, classical SfM), hloc (Python wrapper around classical SfM with learned features), VGGT-X (feedforward transformer, GPU), MapAnything (feedforward, optional dep). Each backend has different output paths, transform conventions, and progress reporting. Without a unifying interface, consumers (dashboard, pipeline scripts, eval harness) end up with backend-specific branches.

## Decision
A registry `{"colmap": ColmapCreator, "hloc": HlocCreator, "vggtx": VGGTXCreator, "mapanything": MapAnythingCreator}` keyed by short string. All creators subclass `BasePointcloudCreator` with method `reconstruct(images, output_dir) -> PointcloudResult`. Backends decide their own world coordinate convention internally; the `CoordinateFrame` enum and `world_transform` field on the result expose it (see ADR 011).

## Consequences
**Positive:**
- Switching backends is a one-line config change.
- Eval harness can iterate over the full backend set without per-backend code paths.
- Adding a new backend is a contained change: implement subclass + register.

**Negative:**
- Forces all backends through the same interface — leaky abstractions when a backend produces something the contract doesn't model (e.g. dense depth maps).
- Registry hard-codes the string keys; rename is a multi-call-site change.

**Revisit if:** the contract surface grows to include backend-specific features that pollute the base class.

## Alternatives Considered
- **Per-backend top-level entry points.** Rejected: forces every consumer to know about every backend.
- **Factory function with kwargs.** Rejected: harder to enumerate backends.
- **Plugin system with entry points.** Deferred: registry is enough at current backend count.
