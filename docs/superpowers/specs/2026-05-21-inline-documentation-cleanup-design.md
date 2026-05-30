---
name: inline-documentation-cleanup
description: Audit and fix inline block comments, section dividers, and print→logging across core collab_splats/ modules
metadata:
  type: project
---

# Inline Documentation & Organization Cleanup

**Date:** 2026-05-21
**Branch:** `refactor/core-modules` (worktree)
**Approach:** File-by-file commits, priority order

## Motivation

CLAUDE.md defines three style rules that several core modules violate:
1. Inline block comments — each logical block needs a short comment
2. `########`-style section dividers in long files
3. `logging` not `print()`

Recent work (semantics-refactor, keyframe-extraction-tutorial) brought `semantics/` and `utils/` into compliance. This spec closes the remaining gaps in `pointcloud/` and `mesh/`.

## Scope

Core `collab_splats/` modules only. `nerfstudio/` excluded (upstream CONSOLE.print convention).

## Files and Changes

### 1. `pointcloud/utils.py` — full cleanup

**Issues:** ~20 `print()` calls, no `########` dividers, missing inline block comments in multi-step functions.

**Changes:**
- Add module-level docstring summarizing what the file provides
- Add `logger = logging.getLogger(__name__)` at top
- Replace all `print()` with `logger.debug()` / `logger.info()` (diagnostic prints → debug; structural info → info)
- Add `########` section dividers grouping:
  - COLMAP utilities (`colmap_reconstruction_to_result`)
  - Confidence / reprojection (`_radial_mask`, `_bbox_mask`, `lift_features`, `reproject_pixels`)
  - Point filtering (`filter_distance`, `filter_density`, `clean_pointcloud`)
  - Voxel downsampling (`voxel_downsample`)
  - Legacy cleaning (`clean_pcd`, `remove_far_points`)
- Add inline block comments to `clean_pcd`, `remove_far_points`, `voxel_downsample` — each has multi-step logic (adaptive sizing, outlier removal, spatial filtering) with no commentary

### 2. `pointcloud/wrappers.py` — divider style + block comments

**Issues:** Uses `# ------------------------------------------------------------------` dash style instead of `########`; sparse inline comments in complex methods.

**Changes:**
- Replace `# ---` dash dividers with `########` style between class sections
- Add inline block comments to `BundleAdjustment._apply_ba`: filter step, track extraction, BA optimization, result update
- Add inline block comments to `LoopClosure._run_lc_loop`: submap iteration, loop detection, alignment, merge

### 3. `pointcloud/bundle_adjustment.py` — divider style

**Issues:** Uses `# ------------------------------------------------------------------` dash style; no class-level section dividers.

**Changes:**
- Add `########` dividers between top-level sections: Config dataclass → Track extraction (`extract_tracks_vggsfm`) → Core BA function (`run_bundle_adjustment`) → Helper math functions
- Replace `# ---` dividers inside `run_bundle_adjustment` with `########` style

### 4. `mesh/utils.py` — two print() calls

**Issues:** Lines 214 and 252 use `print()`.

**Changes:**
- `print(f"Removed {n_removed} components")` → `logger.info(...)`
- `print(f"Skipping hole ...")` → `logger.debug(...)`
- Add `logger = logging.getLogger(__name__)` if not present

## Commit Order

1. `docs(pointcloud): full inline doc cleanup for utils.py`
2. `docs(pointcloud): fix divider style + block comments in wrappers.py`
3. `docs(pointcloud): fix divider style in bundle_adjustment.py`
4. `docs(mesh): print→logging in utils.py`

## Non-Changes

- No logic changes — comments and logging only
- No new tests required (pure documentation / logging refactor)
- `localization.py`, `feedforward/base.py`, `semantics/` already compliant — leave untouched
- `nerfstudio/` out of scope
