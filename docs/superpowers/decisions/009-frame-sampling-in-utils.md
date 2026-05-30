# ADR 009: `frame_sampling` Lives in `utils/`, Not `semantics/`

**Status:** Accepted
**Date:** 2026-04-20
**Tags:** module-boundaries, utils, semantics

## Context
`frame_sampling` was originally placed in `collab_splats/semantics/` because that was where it was first consumed (optical-flow-based frame selection for semantic feature extraction). Subsequent integrations (dashboard preview, pointcloud preprocessing, mesh generation) all needed the same utility. Importing from `semantics/` for non-semantic consumers created a misleading dependency arrow: pointcloud should not depend on semantics.

## Decision
Canonical home is `collab_splats/utils/frame_sampling.py`. A re-export shim at `collab_splats/semantics/frame_sampling.py` preserves backwards-compat for the original import path. Test moved to `tests/utils/test_frame_sampling.py`.

## Consequences
**Positive:**
- Module dependency graph cleaner: pointcloud / dashboard / mesh import from `utils/`, not `semantics/`.
- `utils/` is the correct conceptual home for general preprocessing.

**Negative:**
- Two import paths for the same symbol during the transition.
- Future readers may be confused by the shim; comment it as transitional.

**Revisit if:** the shim is no longer used by any caller (drop it then).

## Alternatives Considered
- **Leave it in `semantics/`.** Rejected: misleads readers about layering.
- **Move and break compat.** Rejected: cherry-pick risk during the single-branch absorption.
