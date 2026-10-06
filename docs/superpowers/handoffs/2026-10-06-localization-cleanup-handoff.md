# Handoff: localization cleanup (clean/localization)

**Date:** 2026-10-06
**Status:** done on its branch, gates green. Not merged, not pushed.
**Branch:** `clean/localization` @ `bbf154d3`, worktree `.worktrees/localization`
**Base:** `feat/rgbd-ba-cf` @ `cb5fac64` (merge base). rgbd-ba-cf has since moved to `5a5ee011` (+6 commits).
**Spec / plan:** [spec](../specs/2026-10-05-localization-vismatch-design.md) · [plan](../plans/2026-10-05-localization-vismatch.md) (deviations 1-18 at the top of the plan)
**Merge owner:** the rgbd-ba-cf session merges this branch; this branch does not merge itself.

---

## 1. What landed

- `LocalMatcher` over vismatch batch `extract` / `match` (pin `basis` `aa81830`); xfeat-star refused (`supports_batches` False upstream)
- `geometry/tracks.py`: star-chain matcher tracks for BA, pycolmap fixed keypoint ids
- `CameraLocalizer`: leaner feature DB, `load_index` / `update_index` / `clear_localized_frames`; localized frames persist pose + id only
- `localize()` shrinks the query to the reference long side, returns K and px on the original grid
- `sample_world_points` lives in `geometry/projection.py` (breaks localizer → geometry → bundle_adjustment → tracks → localizer cycle); no `localization` re-export
- `preproc.frames.read_frames_chunked`: one generator, used by reconstructor + dashboard localizer DB build; tracks keeps `batch_iterator` + `read_frames`
- `utils/visualization.py` `render_points` / `plot_reprojection` reuse `project` + `rescale_intrinsics`

## 2. Merging into feat/rgbd-ba-cf

Dry run (`git merge-tree <base> feat/rgbd-ba-cf clean/localization`, nothing written):

- textual conflicts: `collab_splats/geometry/bundle_adjustment.py` (2 hunks), `collab_splats/reconstructor.py` (3 hunks)
- changed on both sides, auto-merge, re-test: `geometry/projection.py`, `preproc/frames.py`, `tests/geometry/test_bundle_adjustment.py`, `tests/reconstructor/test_refine_stage.py`
- after merge run the full suite, the dashboard gate (section 4) and `tests/test_docstring_contract.py tests/test_import_style.py`

## 3. Config change

- `localization.top_k` removed from `configs/base.yaml`; retired, not refused — `CameraLocalizer(top_k=8)` default applies
- recorded in `configs/README.md` (Migration 2026-10-06)

## 4. Dashboard test shim

Most of `tests/dashboard` fails before this branch: `dashboard/pipeline.py` imports `cache_store_path` and friends from `semantics.utils`, gone since ocr-lens. Not caused here; not in `docs/known-test-failures.md`.

- shim: `/tmp/claude-0/-workspace-collab-splats/dc7c2928-acb2-4306-badd-c5bfbc78f028/scratchpad/shim/t7_shim.py` (stubs the six missing names, raises if called)
- run: `PYTHONPATH=$PWD:<shim dir> /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_run_localization.py tests/dashboard/test_localize_page.py -p t7_shim`
- expected: 48 passed; 20 semantics/lift dashboard tests still fail with the shim (pre-existing)

## 5. Gates at bbf154d3

| Run | Result |
|---|---|
| `tests/ --ignore=tests/dashboard` | 3744 passed, 2 skipped, 80 xpassed, 0 failed |
| dashboard localization (shim) | 48 passed |
| docstring + import-style contract | 1213 passed, 80 xpassed |

## 6. Deferred to consistency phase 3

grid_sample-at-pixels near-duplicates, dedup candidates:

- `geometry/projection.py` `sample_world_points` (bilinear, align_corners=True)
- `semantics/lifting.py:156` `_grid_sample_at_pixels`
- `mesh/texture.py:686`
- `geometry/projection.py:206` `depth_residual` (nearest)

## 7. Open items

- pre-existing dashboard failures (section 4) missing from `docs/known-test-failures.md`
- first `localize()` call ~6 s extra, mostly DINO-SALAD warm-up; steady state loma ~0.94 s, xfeat ~1.10 s
- xfeat PnP RANSAC ~1.0 s at low inlier ratio is the xfeat bottleneck
- PnP RANSAC unseeded: inlier counts vary run to run
- xfeat-star out of scope: refinement moves keypoints per pair, conflicts with fixed-id star tracks; would need a vismatch fork PR
