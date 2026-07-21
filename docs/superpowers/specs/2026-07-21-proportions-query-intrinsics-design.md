# Proportions-Seeded Query Intrinsics (drop feedforward-per-query)

**Date:** 2026-07-21
**Status:** Design — approved
**Scope:** `collab_splats/localization`, `collab_splats/dashboard`, tutorial nb07, tests

## Problem

Query-camera intrinsics for localization are estimated by running a **feedforward
backbone (VGGT-X) on the single query image** (`localization/intrinsics.py:estimate_intrinsics`).

This is wrong in principle and wasteful in practice:

- **Wrong assumption.** The feedforward model exists to define the *scene* from the
  reconstruction batch. Reusing it to predict the intrinsics of a *new, external*
  query image imports an unwanted dependency: the query camera's calibration should
  not ride on top of loading a scene-reconstruction model. FF single-frame intrinsics
  are approximate (few-percent focal error) and unvalidated for arbitrary query cameras.
- **Wasteful.** The K produced is only a *seed*. `CameraLocalizer.localize` already runs
  pycolmap `estimate_and_refine_absolute_pose` with `refine_focal_length=True`, which
  solves the true focal from the actual 2D↔3D correspondences to the known scene. A
  heavy model loads a GPU backbone to produce an initializer that gets refined away.

The intrinsics *requirement* is legitimate and universal — every PnP-based localizer
(COLMAP, hloc) needs a query K. What is anomalous here is the *source*. Feature matchers
(XFeat, LightGlue, DISK, LoMa) produce only 2D↔2D correspondences and never touch K;
intrinsics enter one stage later at pose solve. Standard practice for an unknown query
camera is EXIF or a proportions-based focal heuristic, then refine.

## Approach

Replace the FF seed with the **COLMAP-standard proportions rule** and let existing
pycolmap focal refinement do the rest.

```
f  = 1.2 * max(W, H)
fx = fy = f
cx, cy = W/2, H/2
```

Model-free, metadata-free, ~zero cost. `refine_focal_length=True` (already enabled)
converges the true focal from the query-vs-scene geometry after PnP.

The seed is trivial (3 lines) and belongs inline where the query image already lives —
inside `localize()`. No separate module, no separate estimation step for callers.

### Implementation principles

- **Reuse:** lean on the pycolmap focal refinement already wired into `localize`.
- **Retire:** delete the whole `intrinsics.py` FF-per-query path and its test module;
  no shim, no deprecated re-export.
- **Minimal:** one optional param, one new result field, one inline seed. No EXIF path,
  no geometry focal-solver, no calibration-override plumbing (see Out of scope).

## Changes

### `collab_splats/localization/localizer.py`
- `localize(self, query_image, query_intrinsics=None)` — make `query_intrinsics`
  **optional**. When `None`, build the seed K from `query_image.shape` via the
  proportions rule inline (just before the `pycolmap.Camera` construction, which
  already reads `H, W` from the frame). An explicit K, when passed, is used verbatim
  (calibrated path preserved for future use).
- Add `query_intrinsics: np.ndarray | None = None` to `LocalizationResult`; set it to
  the seed K that `localize` fed to PnP, so provenance travels with the result.
  (Surfacing pycolmap's *refined* focal is a possible later enhancement; the seed is
  sufficient for display now.)

### `collab_splats/localization/__init__.py`
- Remove the `estimate_intrinsics` import and its `__all__` entry.

### `collab_splats/localization/intrinsics.py`
- **Delete** the file.

### `collab_splats/dashboard/pipeline.py`
- **Keep** the `config.calibration_path` branch of `_resolve_query_intrinsics` — a real
  user-supplied YAML calibration is an objective K and stays. Remove only the feedforward
  fallback + its heavy lazy import; when no calibration is configured the resolver returns
  `None` (proportions seed happens inside `localize`).
- Call site (~545): `K = _resolve_query_intrinsics(frame, config, op_log)` may now be
  `None`; pass it straight to `localize(frame, K)`. Read the seed back from
  `result.query_intrinsics` for the dataclass field / DB append when `K is None`.
- `intr_source` (~546): `"calibration file"` when configured, else `"proportions seed"`
  (was `"estimated (experimental)"`).

### `collab_splats/dashboard/localize.py`
- `localize.py:729` display of `out.query_intrinsics[0,0]` keeps working — the value now
  originates from the proportions seed carried on the result. No logic change beyond
  sourcing.

### `docs/source/tutorials/07_localization/localization.ipynb` (nb07)
- Drop `estimate_intrinsics` from the `collab_splats.localization` import.
- §2: remove the `query_K = estimate_intrinsics(query_image)` call; call
  `localizer.localize(query_image)` with no K.
- Update §2 markdown: unknown query intrinsics are seeded from image proportions
  (COLMAP `1.2·max(W,H)` rule) and refined by pycolmap during PnP — no model involved.

### Tests
- **Delete** `tests/localization/test_intrinsics.py`.
- Add proportions-seed unit tests (in `tests/localization/test_localizer.py` or a small
  new `test_intrinsics_seed.py`): asserts `f = 1.2·max(W,H)`, centered principal point,
  square pixels, correct for landscape and portrait shapes; and that an explicit K passed
  to `localize` is used unchanged.
- `tests/dashboard/test_run_localization.py:107` — drop the `_resolve_query_intrinsics`
  monkeypatch; adapt to `localize(frame)` returning a seed K on the result.
- `tests/dashboard/test_localize_page.py` (359, 411) already pass explicit `query_intrinsics`
  — unaffected.

## Out of scope (future work)

- **EXIF-derived K** as a fallback when the query image carries lens metadata.
- **Broader calibration UX** — the `config.calibration_path` YAML override is kept as-is;
  richer per-camera calibration management is future work.
- **Geometry-only focal solver** (P3.5Pf / P4Pf) if refinement-from-seed proves unstable.

## Testing / verification

- `pytest tests/localization tests/dashboard` green.
- Dashboard smoke gate: `python -m collab_splats.dashboard --smoke` prints `SMOKE PASS`.
- nb07 executes end-to-end against the tutorial data with no feedforward load in the
  localization path.
