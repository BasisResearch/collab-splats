# Geometry round 3b — user review follow-ups

Date: 2026-09-26 · branch `clean/geometry-round3` (tip `7a6ad529` at start)

Seven review points after round 3. Decisions below are the user's.

## Decisions

| # | Point | Decision |
|---|---|---|
| 1 | VGGT-SLAM pin `604efe85` not public | re-pin every citation to public `fd3fd218`; explicit attribution header per ported module |
| 2 | `_reproject_per_camera` / `_reproject_shared` general transforms? | no: `@map_transform` + pypose SE3 layout tie them to BA; dedupe the shared pinhole body, stay in BA |
| 3 | metrics docstring unreadable | module docstrings of metrics + verification follow `preproc/qa.py` (tables + column names + pointer); column meanings move to `docs/source/geometry.rst` |
| 4 | `_scale_intrinsics_to_original` is a general transform | public `transforms.intrinsics_to_original(K, crop_box, model_hw)`; reuse wherever the same mapping is inlined |
| 5 | docstrings grew long | contract requires Args/Returns on PUBLIC defs only; geometry over-applied it to 31/31 private helpers. Strip Args/Returns from geometry private helpers unless a name is not self-explanatory; CLAUDE.md states the rule |
| 6 | `clean_for_json` is not a transform | one `collab_splats/utils/io.py` (`to_json_safe`, `write_json` atomic); delete `clean_for_json`, `preproc/frames._jsonable`, qa's inline nan/atomic write; sweep the package for other copies. Preproc interface may change |
| 7 | verification vs reconstruction report | one report. `verify` stage removed; `reconstruction_quality_report` runs verification behind `verify: bool` (default false, it needs the matcher + feature cache). One `reconstruction_quality_report.json`; verify's frame columns join `frames`, its pairs are `epipolar_pairs`. `verification.json` no longer written; `colmap/verified/` + `database.db` still are |

| 8 | `decompose_camera`'s SVD snap (added after review: "yes to both") | delete it; upstream @ `fd3fd218` returns RQ's R unsnapped. Chess ATE bit-identical |
| 9 | LC scale inputs match upstream (same) | dense `submap.points`/`conf`, `conf_threshold` -> `conf_percentile` (25th pct). Upstream's frame-0-camera read regressed chess ATE 2.4-9.7x, so points move into the overlap camera: a deliberate divergence, `docs/parity.md` |

## Gates

- BA EQ (atol 1e-6), LC parity (`rotation_only`), round-3 gate (`r3_gate.sh`)
- report: old `verification.json` + old report mapped into the new layout, values equal
- contract test green; no notebook edits (hand-off lists breaks)

## Deltas

- 3: column meanings live in `docs/source/api/geometry.rst`, not `docs/source/geometry.rst`
- 5: 33 private helpers stripped (BA 15, rest of geometry 18), not 31
- 9: LC parity baseline moves to the overlap-camera tree (`e77c84b9`); 8+9 change behavior, so LC parity vs the round-3 tree is expected to differ
- 9: chess seq-01, 500 frames, submap 50, lc ATE before -> after: vggtx 0.0327 -> 0.0324, omega 0.0149 -> 0.0152, mapanything 0.0406 -> 0.0366
- 9: overlap > 1 still leaves current-side overlap frames after the first un-moved (as before 3b); default overlap is 1
- streams: `clean/r4-ba`, `clean/r4-report`, `clean/r4-lc` cherry-picked; r4-lc's two scale commits squashed into `fd89f441`
