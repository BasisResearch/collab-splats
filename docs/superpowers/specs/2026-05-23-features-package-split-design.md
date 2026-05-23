# Design: Split `features.py` into `features/` Package

**Date:** 2026-05-23
**Scope:** `collab_splats/semantics/`
**Type:** Pure structural refactor — no behavior change

## Motivation

`features.py` currently holds five classes (`BaseFeatureExtractor`, `BaseQueryableExtractor`,
`MaskCLIPExtractor`, `DINOFeatureExtractor`, `Talk2DinoExtractor`) in a single flat file.
Phase 3 will add more backends (DINOv3, SAM2 panoptic, learned per-Gaussian features).
Converting to a package now — before those backends land — avoids growing the file further
and makes the structure consistent with `segmentation/`, which already splits by backend.

## Target Structure

```
collab_splats/semantics/features/
    __init__.py      # re-exports identical public API
    base.py          # BaseFeatureExtractor, BaseQueryableExtractor,
                     # _DEBIAS_VALIDATED, TORCH_HOME
    maskclip.py      # MaskCLIPExtractor
    dino.py          # DINOFeatureExtractor
    talk2dino.py     # Talk2DinoExtractor
```

Mirrors `segmentation/` exactly: one `base.py` for abstract classes and shared
constants, one file per concrete backend.

## What Changes

- `collab_splats/semantics/features.py` deleted.
- `collab_splats/semantics/features/` created with the five files above.
- Each concrete file imports only from `.base`.
- `features/__init__.py` re-exports the same names that `features.py` exported.

## What Does Not Change

- `collab_splats/semantics/__init__.py` — unchanged; already re-exports from `features`.
- All consumer imports (`from collab_splats.semantics import ...`) — unchanged.
- Test files — unchanged.
- Any in-flight branch that imports from `collab_splats.semantics` — unchanged.

## Shared Constants

`TORCH_HOME` and `_DEBIAS_VALIDATED` both live in `base.py`. They are
base-class infrastructure, not backend-specific.

## Timing

First commit on `refactor/cu121` before any Phase 3 extractor work begins.
Single squash commit: `refactor(semantics): split features.py into features/ package`.
