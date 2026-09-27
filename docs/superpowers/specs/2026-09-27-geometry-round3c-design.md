# Geometry round 3c — second user review

Date: 2026-09-27 · branch `clean/geometry-round3c` (forked from `clean/geometry-round3` `27f36d1d`)

Nine review points on round 3b. Decisions below are the user's.

## Decisions

| # | Point | Decision |
|---|---|---|
| BA 1-2 | `_extract_tracks` sat under Configuration; tracks helpers placement | `_compute_tracks_cache_key` + `_extract_tracks` move under Helpers |
| BA 3 | observation filtering generalizable? | no: `_filter_observations` is BA-specific (track-shaped inputs), stays |
| BA 4 | `_extract_tracks` + `_extract_tracks_vggsfm` redundant | one `_extract_tracks(images, conf, world_points, cfg)` |
| BA 5 | `_pinhole` unnecessary | inlined back into both `@map_transform` wrappers; bodies AST-equal to `f9850269^` |
| metrics 1 | docstring should state what it measures | one-line summary + a short-statement table of every measurement |
| metrics 2 | `_EPIPOLAR_COLUMNS` / `_TRACK_COLUMNS` single-use | moot: deleted with verification |
| metrics 3 | `depth_error_in_pixels` | deleted with the `depth_error_px` column; `compute_depth_error` loses `focal_px` |
| verify | overbuilt for its purpose? | option C: remove geometric verification entirely |

## Verification removal (supersedes round 3b row 7)

- deleted: `geometry/verification.py`, `tests/geometry/test_verification.py`, `evals/scripts/eval_verification.py`
- deleted: `Reconstructor._run_verification`, `reconstruction_quality_report.verify` (never shipped to `clean/final`)
- report: no `epipolar_pairs`, no frames track columns, `params` is `{rel_thresh}` only
- `PairStats` is depth-only, lives in `geometry/metrics.py`
- kept: `--stages verify` and `geometric_verification: true` raise; `geometric_verification: false` accepted
- kept: `/*/colmap/database.db` in `PUSH_EXCLUDES`, since older scenes carry it

## Follow-ups (not done here)

- LocalMatcher index machinery (`idx_q`/`idx_db`, `has_stable_indices`, `_probe_index_stability`,
  `_recover_indices`) has no consumer now; vismatch-fork scope
- `pointcloud/sfm/instantsfm.py:433,484` comments still name verification; pointcloud-release in flight
- "number of shared keypoints" is no longer measured; `depth_pairs.n_pixels` (overlap) is the nearest
