# Geometry release round 3 — design

Status: implemented 2026-09-26 on `clean/geometry-round3` (`df468d6b..e393cacf`); see Deltas from implementation.
Branch: `clean/geometry-round3` at `.worktrees/geometry-round3`, forked from `clean/final` `ac93e96b`.
Follows: geometry-release (`e52117da`), rules in decision 017 and the `release-cleanup` skill.

## Goal

Second review pass over `collab_splats/geometry` on `clean/final`:

- contract-conformant docstrings, comments and spacing across every file
- no nested functions
- each explanation written once, cited elsewhere
- geometry functions take arrays, not pointcloud result types (except the LC wrapper, see Non-goals)
- both geometry JSON reports follow the video QA report contract
- duplicated and dead code removed

## Decisions

| Topic | Decision |
|---|---|
| `BundleAdjustmentConfig` (C4) | keep all 11 fields; reorganize only |
| Incremental BA | stays — `evals/configs/7scenes.yaml` runs `incremental_ba-3` |
| LC wrapper move (C17) | not now — `loop_closure/wrapper.py` stays in geometry |
| `scale_method: none` (E2) | deleted; the `scale_method` field goes with it |
| Extra cleanups E1–E6 | accepted (E5 dropped: it only existed for C17) |

## Report contract (C11 + E1)

Mirror of `preproc/qa.py`:

- provenance block + `params` block + columnar tables `{column: [value per row]}`
- no verdicts, no derived statistics: anything computable from the columns is not stored
- nan written as null
- module docstring lists every column, units and grid
- `compute_*` in geometry takes arrays; the Reconstructor stage owns zarr load, reuse-by-existence and the temp-file write

### reconstruction_quality_report.json

| Old key | New home | Status |
|---|---|---|
| `scene` | `scene` (+ `image_width`) | kept |
| — | `params` {`rel_thresh`} | new |
| `source_frame_indices` | `frames.frame_idx` | column |
| `crop_coverage` | `frames.covered_fraction` | column |
| (computed, never written) | `frames.median_rel_depth_error`, `frames.confidence_median` | new columns |
| `epipolar.frames` | `frames.mean_reproj_error_px`, `frames.mean_reproj_error_frac_width` | columns, null without verify |
| `depth.pair_directions` | `depth_pairs` | columnar |
| `photometric.pairs` | `photometric_pairs` | columnar |
| `epipolar.pairs` | `epipolar_pairs` | columnar, null without verify |
| `depth.residual_histogram` | `depth_residual_histogram` {`counts`, `bin_edges`} | quantiles dropped |
| `frame_separation`, `measurements_available`, `available`/`grid`/`units`/`reason` | — | deleted |
| `correlations` ×2, `confidence_vs_error`, `running_error`, `frame_percentile_ranks`, `pair_directions_under_one_pixel_disparity` | — | deleted (derivable) |
| `notes` | module docstring | moved |

### verification.json (E1)

| Old key | New home | Status |
|---|---|---|
| `pair_stats` rows | `pairs` | columnar |
| `frame_stats` {name: row} | `frames` | columnar, `name` a column |
| `summary` | — | deleted; `eval_verification.py` computes its own via the moved `_distribution` |

`PairStats` stays a dataclass: `pointcloud/feedforward/base.py` builds depth rows with it.

## Change table

Line deltas are estimates from measured function spans, not a dry run.

| # | Round | File(s) | Change | Review point | Est. Δ |
|---|---|---|---|---|---|
| R1 | prose | all `geometry/` | contract docstrings: ~20 functions, ctor params, no quote-line one-liners | 5, 13 | +40 |
| R2 | prose | all | 2-line prose comments → header + bullets; blank line above block comments; comment the walls | 10, 13 | +30 |
| R3 | prose | all | explain once: R^T ×5, intrinsics ratio ×7, overlap ×4, 2-frame carrier ×3, `_lc_assembled` ×4 | 10 | −70 |
| R4 | prose | all | one divider style (`#`×72 + title); `log` → `logger` at top | 13 | 0 |
| R5 | prose | `verification.py` | prose `Args:` → fragments | 5 | −6 |
| C1 | code | `tests/test_docstring_contract.py` | fail on nested def, quote-line private docstring, 2-line prose comment run — **geometry only** (see Risks 3) | 4, 5 | +60 test |
| C2 | code | `bundle_adjustment.py` | hoist `_extract` to module level | 4 | 0 |
| C3 | code | `bundle_adjustment.py` | `_refine_allonce` → `_refine_global` | 3 | 0 |
| C4 | code | `bundle_adjustment.py` | config fields grouped by step (tracks / filter / solve / runtime), trailing comments into the docstring; no field removed | 2 | ~0 |
| C5 | code | `bundle_adjustment.py`, `eval.py` | `world_points` digest in the track-cache key; drop eval's per-backbone dir workaround | 1 | −3 |
| C6 | code | `bundle_adjustment.py`, `reconstructor.py`, `eval.py`, `ba_start_at_gt.py` | `refine` takes arrays, returns `(extrinsics, intrinsics)`; `_check_model_resolution` to the caller | 6 | −15 |
| C7 | code | `transforms.py` | `project_to_so3()` replaces 2 SVD snaps (BA, `decompose_camera`) | 7 | −5 |
| C8 | code | `transforms.py`, `graph.py`, `submap.py` | `decompose_camera` → `transforms.py`; inline import gone | 12 | −3 |
| C9 | code | `transforms.py` | shared Umeyama input guard | 13 | −8 |
| C10 | code | `metrics.py` | reorder like `qa.py`; drop `_scale_intrinsics_to_original` if an existing helper covers it | 8 | 0 to −29 |
| C11 | code | `metrics.py`, `reconstructor.py`, `configs/README.md` | report contract above; old-format file rejected (Risks 4); `read_quantiles`, `_running_error`, correlations, ranks gone; `compute_reconstruction_quality(arrays)`; Reconstructor runs the dense pass and passes `collected` in | 8, 9, 6 | −180 |
| C12 | code | `verification.py` | `clean_for_json` → `transforms.py`; id map built ×3 → one helper | 7, 13 | −8 |
| C13 | code | `graph.py`, `wrapper.py` | delete `check_scale_method` (with E2 nothing is left to check) | 11 | −14 |
| C14 | code | `graph.py` | hoist `_frame_data`; merge duplicated inner-chain loop, 4× K_4x4 build, `_cam_local_points` copy | 4, 13 | −22 |
| C14b | code | `graph.py` | merge the diverged confidence fallback (keep `prior_conf > 0` tier) — own commit, see Risks 1 | 13 | −8 |
| C15 | code | `wrapper.py` | `_camera_centers_from_poses` → `invert_poses`; view-count helper; rename `add_points`; `LoopClosureConfig` into `__all__` | 13 | −15 |
| C16 | code | `submap.py`, `map.py`, `graph.py`, `wrapper.py` | delete dead `Submap.conf_masks`; test-only hooks into tests | 13 | −20 |
| C18 | code | `reconstructor.py`, `eval.py` | hoist function-level BA / metrics / verification imports once no cycle remains | 6 | 0 |
| E1 | code | `verification.py`, `eval_verification.py` | `verification.json` contract above; old-format file rejected (Risks 4) | 8 | −20 |
| E2 | code (**last**, Risks 2) | `graph.py`, `wrapper.py`, `eval.py`, `configs/loop_closure.yaml`, `evals/configs/*.yaml`, `evals/README.md` | delete `scale_method: none`, the `scale_method` field, `SCALE_METHODS`, `--lc_scale_method` | 11 | −30 |
| E3 | code | `loop_closure/__init__.py` | stop exporting test-only names; tests import from submodules | 13 | −10 |
| E4 | code | `evals/scripts/eval.py` | hoist nested `_ba`; one builder for the ×3 no-LC `LoopClosureConfig` | 4 | −10 |
| E6 | code | `reconstructor.py` | one `_write_json_atomic` for the hand-rolled reuse + temp-write, only if planning measures ≥2 real copies | — | −10 |
| C19 | docs | `docs/source/api/geometry.rst`, `docs/parity.md`, CHANGELOG, CLAUDE.md tree | match the code | — | ~0 |
| T1 | tests | `tests/geometry/test_metrics.py`, `tests/wrapper/*` | drop deleted-key tests, add column tests | 8 | −150 |
| T2 | tests | `tests/geometry/loop_closure/` | receive C16 hooks; drop E2 `none` cases; E3 imports | 13 | 0 |

### Net lines (estimate)

| Scope | Now | Δ | After |
|---|---|---|---|
| `collab_splats/geometry/` | 4,146 | −390 to −420 | ~3,740 (≈ −10%) |
| callers + evals | — | ~−10 | — |
| tests (`tests/geometry` + contract) | 7,105 | ~−90 | ~7,015 |

## Round rules

- Round 1 (R1–R5): prose only, one commit
  - proof: AST equal after deleting every docstring statement on both sides
  - plus one sanity mutation showing the proof can fail
- Round 2 (C*, E*): one commit per row, gate after each
- `git commit --only <paths>`; never amend, rebase or reset
- US spelling; contract comment style (header + fragment bullets)

## Gates

- G′ from the worktree, with a printed `collab_splats.__file__` proof line:
  `cd <wt> && PYTHONUTF8=1 PYTHONPATH=<wt> pytest tests/geometry tests/evals tests/pointcloud/feedforward tests/pointcloud/test_pose_extraction.py tests/wrapper tests/test_docstring_contract.py -q -p no:cacheprovider`
  - baseline measured on the untouched worktree first; known entries in `docs/known-test-failures.md`
- LC parity (`lc_parity_t29b.py compare`, all `scale_method` cases) after every loop_closure commit; re-baselined only after E2
- C6: BA poses identical before/after on a fixture scene (CPU-skipped tests must be named in the report)
- C11 / E1: every value that survives into the new report equals the old report's value on `data/outputs`

## Risks

1. C14b is a behavior change
   - the two fallback copies differ: only one has the `prior_conf > 0` tier
   - own commit; LC parity must pass
   - parity fails → keep both copies, document the difference instead
2. E2 removes LC parity coverage
   - `lc_parity_t29b.py` runs every `scale_method × timing` case; the baseline holds `none` cases
   - every loop_closure commit runs full parity with `none` still present
   - E2 lands last, then parity re-baselines on `rotation_only` only
3. C1's new checks would fail outside this round
   - nested defs today: pointcloud 8, preproc 1, semantics 0, geometry 3
   - new checks scoped to `geometry`; the 9 others listed as follow-up in the changelog entry
4. Old-format reports reused silently
   - report stages reuse by existence; old files sit in `data/outputs` and processed buckets
   - `reconstruction_quality_report.json` without `frames`, `verification.json` without `pairs` → raise "stale report; delete and re-run", like `load_video_quality`
5. C5 invalidates every track cache
   - first BA per scene re-extracts on GPU; one-time cost, no correctness risk
6. Downstream consumers
   - `clean/final` is live (another session commits); squash later may conflict
   - tutorial-rework uses the BA API and the report; hand-off note, no notebook edits

## Non-goals

- LC wrapper stays in `geometry/loop_closure/wrapper.py` and keeps importing pointcloud types
  - so the lazy `__getattr__` in both `__init__` files stays
  - point 6 applies to BA and metrics only this round
- no moves of single-consumer `transforms.py` functions (E8)
- no split of `compute_multiview_depth_confidence(collect=)` (E9)
- no removal of the LC live viewer (E7)
- no notebook edits on `clean/final`; `02_pointcloud/bundle_adjustment.ipynb` breaks after C6 → hand-off note for tutorial-rework
- no merge, no push

## Deltas from implementation

Recorded after the fact; the approved tables above are left as written.

| Row | Delta |
|---|---|
| C11 | frame column is `median_abs_rel_depth_error`, not `median_rel_depth_error`: median of `abs()` per frame, so the old name read as signed |
| C11 / E1 | ratio columns kept as an exception to "no derived statistics": `inlier_ratio`, `mean_reproj_error_frac_width`, `track_survival` |
| C7 | landed late (`08a4ae8f`), after the C14b merge and the report commits, not in plan order; covers the BA snap only: `decompose_camera` keeps upstream's plain `U @ Vt` snap (faithful port, det<0 returns the reflection like upstream), restored in `193d60c8` with a det<0 pin test |
| C6 | the resolution guard is public `check_model_resolution`, not a private `_check_model_resolution` |
| C16 | `debug_out` was removed too, beside `conf_masks` and the `_last_*` hooks |
| C15 (plan Task 16) | the `_n_views` helper covers 2 of the 3 view-count sites |
| E4 (plan Task 21) | no `_no_lc_config` helper: only one no-LC build site was left |
| C18 (plan Task 20) | also hoisted the verification and localization imports; every hoisted import adds zero new modules at reconstructor import time (all already loaded via the creators) |
| E2 | a processed scene's old `run_config.yaml` carrying `pointcloud.loop_closure.scale_method` fails on re-run: `LoopClosureConfig(**knobs)` raises `TypeError`, re-raised as `ValueError: Invalid pointcloud.loop_closure knob`; delete the key. `evals/scripts/eval.py` `load_eval_config` silently ignores unknown keys; rejecting them is optional |
| C1 | the prose-pair hanging-indent exemption was removed (0 uses). Outside geometry: semantics 0 (could join the geometry scope), preproc 23, pointcloud 98 |
| C19 | VGGT-SLAM pin `604efe85` is not on GitHub; `graph.py`'s `solver.py:132-143`, `161-162`, `256`, `162-166` match no public commit (nearest `fd3fd218`, fallback at 129-138). Re-pin or annotate: left to the user |
| C14b | the two confidence fallbacks were merged into one `_conf_fallback_mask`; not kept as two documented copies |
| E6 | not done (dropped at planning: one hand-rolled temp+replace, below the ≥2 bar) |

### Net lines (measured)

- `collab_splats/geometry/`: 4,146 → 4,389 (+243), against the −390 to −420 estimate
  - 4,393 (+247) after the final-review fixes (`193d60c8`: plain snap restored in `decompose_camera`)
  - prose round (R1–R5): +805/−298 (+507)
  - code rounds: +693/−957 (−264)
  - the estimate priced the prose round at ~0; contract docstrings and bulleted comments grew it
- `tests/`: +790/−773

