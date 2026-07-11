# Geometry package + LC cleanup — design

**Date:** 2026-07-11 · **Branch:** `refactor/cu121-uv-migration` · **Status:** approved
**Predecessor:** `docs/superpowers/handoffs/2026-07-10-lc-parity-integration-cleanup-handoff.md` (all 9 targets absorbed here)

## Goals

1. Create top-level `collab_splats/geometry/` package: loop closure + bundle adjustment + geometry transforms, following the `localization/` promotion precedent (hard cut, no shim).
2. Remove deprecated/dead code across the package (~700 LOC target), stale eval scripts, and duplicated test scaffolding — **without reducing documentation content or test coverage**.

## Non-goals

- TALO integration (deferred — see `2026-07-10` LC-vs-TALO report; the geometry package creates the seam for it).
- BA de-VGGT-ing (generic track interface stays future work).
- `mesh/` changes of any kind (uncommitted feedforward-mesh work in progress; candidates listed under Deferred).
- `eval_gt.py` ablation-block split (handoff #7 — works as is; churn not justified).
- Behavior changes to LC/BA numerics. Suite must stay green (1056 pass / 2 skip / 3 xfail baseline).

## 1. Package layout

```
collab_splats/geometry/
  __init__.py            # re-exports: LoopClosure, LoopClosureConfig, BundleAdjustment,
                         #             BundleAdjustmentConfig, transforms symbols
  transforms.py          # ex utils/geometry.py (extrinsics_to_homogeneous, invert_poses,
                         #                       extract_intrinsics, rotation_align_vectors,
                         #                       OPENGL_TO_OPENCV)
  bundle_adjustment.py   # ex pointcloud/bundle_adjustment.py
  loop_closure/
    __init__.py
    closure.py           # gates, umeyama, PGO build, merge
    graph.py             # GTSAM SL(4)/SE(3) PoseGraph
    submap.py
    eval.py              # capture_pose_graph_loss only (see §2.1)
    wrapper.py           # ex pointcloud/wrappers.py — LoopClosure proxy
```

Hard cut: old modules deleted, `pointcloud/__init__.py` re-exports removed. `git mv` to preserve history. Moves preserve internal file structure (no flattening).

**Importers to update:**
- `collab_splats/wrapper/reconstructor.py`
- `collab_splats/pointcloud/` internals (base, feedforward backbones, utils) importing `utils.geometry` → `geometry.transforms`; 16+ sites total incl. `mesh/` (import-line-only change — allowed despite mesh freeze), `nerfstudio/`, `wrapper/splatter.py`
- evals: `eval_gt.py`, `metrics.py`, `check_ate_methods.py`, `_ba_finding_eval.py`, kept runners (§3)
- live notebooks: `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`, `bundle_adjustment.ipynb` — **frozen `feedforward_mesh.ipynb` / `feedforward_methods.ipynb` are NOT touched** (uncommitted user edits); if they import moved paths, defer their update to the mesh session
- docs: `docs/source/api/pointcloud.rst` (+ new `geometry.rst`), CLAUDE.md architecture tree
- tests: move to `tests/geometry/` (§4)

Historical `docs/superpowers/{specs,plans}/*.md` keep old paths (dated documents — not updated).

## 2. Dead / deprecated code removal

### 2.1 Tier A (low risk, zero external refs — verified)

| Item | Location | LOC |
|---|---|---|
| Deprecated `lc_threshold` / `lc_cosine_threshold` fields + `__post_init__` warnings + `lc_threshold_l2` alias kept | `closure.py:186-208` | ~25 |
| GT-phase stub block: `_umeyama_sim3`, `umeyama_align`, `ate_translation`, `rpe`, `auc_at_threshold` (zero callers; kills umeyama duplication) | `loop_closure/eval.py:107-295` | ~178 |
| Unused `normalize_to_sl4` import (canonical stays in graph.py + pkg re-export) | `closure.py:20` | 1 |
| Never-passed `subsample: int = 8` param on `_cam_local_points` | `closure.py:395` | ~3 |
| `filter_density` (shadowed by live `density_filter`), `filter_points_by_spatial_extent`, `extrinsics_to_c2w` | `pointcloud/utils.py:171,596,1030` | ~98 |
| `get_cache_dir`, `collect_image_paths` | `utils/paths.py:32`, `utils/io.py:9` | ~34 |
| `docs/examples/run_tutorials.sh` (orphan, zero refs) | file delete | — |
| `env.yml` (conda-era, superseded by uv; gate: grep Dockerfile/setup.sh/CI first) | file delete | ~15 |

Tests consuming deprecated fields (`tests/pointcloud/test_loop_closure.py`, `test_loop_closure_integration.py`) migrate to `lc_retrieval_threshold` — migrated, not deleted.

### 2.2 Tier B (medium risk — each deletion gated on `grep docs/source/**/*.ipynb` + tests; anything referenced stays)

| Item | Location | LOC |
|---|---|---|
| Unused plot fns: `plot_heatmap`, `plot_context_segmentation`, `query_heatmap`, `overlay_masks`, `o3d_mesh_to_polydata` | `utils/visualization.py:84,103,168,262,465` | ~149 |
| `utils/notebook.py` — whole file (98/125 LOC dead; delete file if notebook grep clean) | `utils/notebook.py` | ~125 |
| Vendored cruft: `ndc2pix_x/y`, `euclidean_to_z_depth`, `project_pix`, `get_colored_points_from_depth`, `get_rays_x_y_1` | `nerfstudio/utils/camera_utils.py:300-487` | ~123 |

### 2.3 Hygiene (handoff items, no deletions)

- `wrapper.py`: module-level logger replaces two inline `logging.getLogger(__name__)` call-sites.
- `feedforward/base.py` `_verify_loop_candidate`: docstring describes per-model resolution chain, not "0.85 matches VGGT-SPARK".
- `vggt_omega.py:~126`: reconcile stale ratio comment with L13/1.55 classvar.
- spark/omega `unproject_depth_map_to_point_map` import: one comment acknowledging shared VGGT-X dependency (no import change).
- `closure.py`: reorder `_cam_local_points` above `_lc_anchor_scale` (read order).
- se3-manifold non-orthonormal Rot3 limitation: documented at `LoopClosureConfig.manifold` field docstring. No behavior change.
- `MESH_PIPELINE_AUDIT.md`: move repo root → `docs/superpowers/audits/2026-05-29-mesh-pipeline-audit.md` (content preserved).

## 3. Evals prune (surgical)

**Delete** (superseded one-off diagnostics; gate: verify nothing kept imports them):
`evals/runners/debug_lc_steps.py`, `parity_trace.py`, `our_solver_dump.py`, `compare_slam_ours.py`, `diagnose_lc_parity.py`, `evals/compare_loop_edges.py`, `evals/eval_vggt_slam_comparison.py`.

**Keep** (durable): `eval_gt.py`, `datasets.py`, `ate_utils.py`, `metrics.py`, `trajectory_io.py`, `download_7scenes.py`, parity driver (`runners/run_lc_parity.py`, `lc_parity_common.py`, `run_vggt_slam_lc.py`), `lc_loop_pr.py`, `visualize_lc_correction.py`, `build_parity_table.py`, `eval_similarity_calibration.py`, `_ba_finding_eval.py`, `check_ate_methods.py`, `reconstruction_quality.py` + all `tests/evals/`.

**Gitignore fix**: add `evals/baselines/lc_parity*/` (heavy trajectory/npz dirs currently untracked-but-exposed; handoff line 35 claimed ignored — false). *Implementation deviation (2026-07-11): narrowed from the originally planned blanket `evals/baselines/` — that directory holds 257 committed reference-result files per `evals/README.md`, and a blanket ignore would silently exclude future reference baselines. Only the heavy `lc_parity*` output dirs are ignored.*

## 4. Tests

- Move LC/BA/wrapper tests → `tests/geometry/` mirroring the code tree; the 5 scattered loop-closure files consolidate under `tests/geometry/loop_closure/` (file merge allowed, zero test deletion).
- Shared conftest fixture for `_FakeQKV` / `_FakeMapAnythingModel` (currently duplicated in `tests/pointcloud/feedforward/test_verify_lc_data.py` and `test_mapanything_creator.py`).
- Merge same-named splits: `tests/test_visualization.py` + `tests/utils/test_visualization.py`; `tests/pointcloud/test_mapanything_creator.py` + `tests/pointcloud/feedforward/test_mapanything_creator.py`.
- **`tests/mesh/` untouched** (frozen).
- Acceptance: full suite green, test count ≥ baseline minus tests of deleted dead functions (each such removal listed in the plan explicitly).

## 5. Commit sequence

1. `style:` black/isort on `mapanything.py`, `vggtx.py`, `closure.py` — no logic
2. `refactor(lc):` Tier A dead code + hygiene (§2.1, §2.3)
3. `refactor(geometry):` package move + import rewrite (`git mv`)
4. `test(geometry):` test moves + conftest consolidation + merges
5. `chore(evals):` prune + `.gitignore` fix
6. `refactor(utils):` Tier B deletions (post-gate) (§2.2)
7. `docs:` path updates (api rst, notebooks, CLAUDE.md, audit move) — paths only, content preserved

Full suite run after commits 2, 3, 4, 6; targeted runs elsewhere.

## 6. Deferred (recorded, not in scope)

- `mesh/utils.py` dead candidates (~142 LOC, zero refs as of 2026-07-11): `pick_indices_at_random`, `find_depth_edges`, `normals2vertex`, `align_geometry_floor`, `mesh_clustering` — revisit after feedforward-mesh work lands.
- `nerfstudio/utils/camera_utils.py::pix2ndc_x/pix2ndc_y` — became dead when `get_rays_x_y_1` was deleted (Task 6); not on the gated candidate list, left for a follow-up.
- TALO-style registration backend; BA generic track interface.
- `eval_gt.py` LC-stats/ablation helper split.

## 7. Implementation deviations (as-built, 2026-07-11, commits ce6391a..38bd972)

- §2.1 eval.py stub block KEPT: `ate_translation`, `rpe`, `umeyama_align`, `auc_at_threshold` etc. have live consumers (`evals/eval_gt.py`, `evals/metrics.py`, tests) — the "zero callers" premise was wrong. `tests/.../test_auc_metric.py` moved to `tests/geometry/loop_closure/` instead of deleted.
- §2.1 extended: `utils/paths.py` and `utils/io.py` deleted whole-file (remainders `get_project_root`, `_IMAGE_EXTENSIONS` also had zero refs).
- §3: 3 of the 7 eval scripts KEPT per the reference gate: `compare_loop_edges.py` (imported by tests), `parity_trace.py` (named in a product warning string), `our_solver_dump.py` (consumed by kept `compare_solver_internals.py`). 4 deleted.
- §3 gitignore narrowed (see note above).
- §2.2: 11 of 14 Tier-B symbols deleted; kept per notebook gate: `overlay_masks`, `feature_viz_row` (and `utils/notebook.py` survives as partial file); `project_pix` deleted after word-boundary re-grep cleared a `reproject_pixels` substring false positive.
