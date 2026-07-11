# Geometry Package + LC Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create top-level `collab_splats/geometry/` package (loop closure + bundle adjustment + transforms) and remove ~700 LOC of dead/deprecated code, per `docs/superpowers/specs/2026-07-11-geometry-package-lc-cleanup-design.md`.

**Architecture:** Seven commits, one per task. `git mv` preserves history. Hard cut on old import paths (no shims). Full suite green after Tasks 2, 3, 4, 6. FROZEN — do not edit: `collab_splats/mesh/*` (except single import lines in Task 3), `tests/mesh/*`, `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`, `feedforward_methods.ipynb` (uncommitted user work).

**Tech Stack:** Python 3.11 at `/opt/venv/reconstruction/bin/python`; pytest; black/isort; git. Test command used throughout: `PY=/opt/venv/reconstruction/bin/python`.

**Baseline:** suite = 1056 passed / 2 skipped / 3 xfailed (plus any drift from the user's in-flight mesh edits — record actual baseline in Task 1 Step 1 and compare against that).

---

### Task 1: Style commit (no logic)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`, `collab_splats/pointcloud/feedforward/vggtx.py`, `collab_splats/pointcloud/loop_closure/closure.py` (formatting only)

- [ ] **Step 1: Record baseline test count**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q --ignore=tests/mesh 2>&1 | tail -3`
Record the pass/skip/xfail counts — this is the comparison baseline for all later tasks. (`--ignore=tests/mesh` because mesh tests are mid-edit by the user; use the same ignore flag consistently in every suite run in this plan.)

- [ ] **Step 2: Format the three dirty files**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/pointcloud/feedforward/mapanything.py collab_splats/pointcloud/feedforward/vggtx.py collab_splats/pointcloud/loop_closure/closure.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/pointcloud/feedforward/mapanything.py collab_splats/pointcloud/feedforward/vggtx.py collab_splats/pointcloud/loop_closure/closure.py
```

- [ ] **Step 3: Verify diff is formatting-only**

Run: `git diff --stat` — only the 3 files. Spot-check `git diff` output: whitespace/quotes/import-order changes only, no logic.

- [ ] **Step 4: Quick test of affected areas**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud -q 2>&1 | tail -3`
Expected: same counts as baseline for that dir (no new failures).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py collab_splats/pointcloud/feedforward/vggtx.py collab_splats/pointcloud/loop_closure/closure.py
git commit -m "style: black/isort on mapanything, vggtx, closure (no logic)"
```

---

### Task 2: Tier A dead code + hygiene

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`, `collab_splats/pointcloud/loop_closure/eval.py`, `collab_splats/pointcloud/loop_closure/__init__.py`, `collab_splats/pointcloud/wrappers.py`, `collab_splats/pointcloud/feedforward/base.py`, `collab_splats/pointcloud/feedforward/vggt_omega.py`, `collab_splats/pointcloud/feedforward/vggt_spark_creator.py`, `collab_splats/pointcloud/utils.py`, `collab_splats/utils/paths.py`, `collab_splats/utils/io.py`
- Modify: `tests/pointcloud/test_loop_closure.py`, `tests/pointcloud/test_loop_closure_integration.py`
- Delete: `docs/examples/run_tutorials.sh`, `env.yml`, `tests/pointcloud/test_auc_metric.py` (tests deleted stubs — see Step 3)

- [ ] **Step 1: Remove deprecated config fields from `closure.py`**

In `LoopClosureConfig` (closure.py ~186-208): delete the two fields `lc_threshold: float | None = None` and `lc_cosine_threshold: float | None = None`, the entire `__post_init__` method (it only services those fields), and their comment lines. KEEP the `lc_threshold_l2` property (live consumer: `wrappers.py` uses `cfg.lc_threshold_l2`). NOTE: `find_loop_closures(..., lc_threshold=...)` at closure.py:244 is a different local parameter — do not touch.

- [ ] **Step 2: Migrate/remove the deprecated-field tests**

In `tests/pointcloud/test_loop_closure.py` (lines ~48-74, 149, 176) and `tests/pointcloud/test_loop_closure_integration.py` (lines ~36-60):
- Tests that merely construct `LoopClosureConfig(lc_threshold=X)` as setup → change kwarg to `lc_retrieval_threshold=X`. For `lc_cosine_threshold=Y` setup → `lc_retrieval_threshold=math.sqrt(2 * (1 - Y))` (add `import math` if missing).
- Tests whose *assertion target* is the DeprecationWarning itself (e.g. `pytest.warns(DeprecationWarning)`) → delete those test functions; they test the removed feature. List every deleted test name in the commit body.

- [ ] **Step 3: Delete GT-phase stub block in `eval.py`**

Delete `loop_closure/eval.py` lines ~107-295: `_umeyama_sim3`, `umeyama_align`, `ate_translation`, `rpe`, `auc_at_threshold` (zero package callers; header says "implement when GT dataset arrives"). Keep `capture_pose_graph_loss` and everything above line ~107. Then:
- Check `loop_closure/__init__.py` — remove any re-export of the deleted names.
- `tests/pointcloud/test_auc_metric.py` tests `auc_at_threshold` → delete the file. Check `tests/pointcloud/test_eval_metrics.py`: delete only test functions exercising the deleted stubs (`ate_translation`, `rpe`, `umeyama_align`); keep tests of `capture_pose_graph_loss`. List deleted test names in commit body.
- Verify no other consumer: `grep -rn "ate_translation\|umeyama_align\|auc_at_threshold\|loop_closure.eval import\|loop_closure import eval" collab_splats evals tests --include="*.py" | grep -v test_auc` — expect only `capture_pose_graph_loss`-related hits.

- [ ] **Step 4: Small closure.py hygiene**

- Remove `normalize_to_sl4` from the `from .graph import ...` line at closure.py:20 (stays exported from `graph.py` + package `__init__`).
- Remove the never-passed `subsample: int = 8` parameter from `_cam_local_points` (~line 395): inline the current default into the function body (verify with `grep -rn "_cam_local_points(" collab_splats evals tests` that no caller passes it).
- Move the `_cam_local_points` function definition above `_lc_anchor_scale` (its caller) for read order. Pure cut/paste, no edits.

- [ ] **Step 5: Dead functions in `pointcloud/utils.py`, `utils/paths.py`, `utils/io.py`**

Delete (each verified zero-reference; re-verify with grep before deleting each):
```bash
grep -rn "filter_density\b" collab_splats evals tests docs/source --include="*.py" --include="*.ipynb"
grep -rn "filter_points_by_spatial_extent\|extrinsics_to_c2w\|get_cache_dir\|collect_image_paths" collab_splats evals tests docs/source --include="*.py" --include="*.ipynb"
```
If grep hit = definition-only → delete: `pointcloud/utils.py::filter_density` (~line 171; the live one is `density_filter` ~457 — do not confuse), `pointcloud/utils.py::filter_points_by_spatial_extent` (~596), `pointcloud/utils.py::extrinsics_to_c2w` (~1030), `utils/paths.py::get_cache_dir` (~32), `utils/io.py::collect_image_paths` (~9). If any grep shows a live consumer, keep that function and note it in the commit body. Also remove any `__init__.py` re-exports of deleted names and delete their tests if any exist (`grep -rln "filter_density\|extrinsics_to_c2w\|get_cache_dir\|collect_image_paths" tests/`).

- [ ] **Step 6: Wrappers logger + docstring/comment fixes**

- `wrappers.py`: add module-level `logger = logging.getLogger(__name__)` after imports; replace the two inline `logging.getLogger(__name__).warning(...)` call-sites (~194, ~287) with `logger.warning(...)`.
- `feedforward/base.py` `_verify_loop_candidate` (~869): rewrite docstring — the `verify_match_ratio` default is always overridden by the wrapper's resolution chain (explicit config → creator's `default_verify_match_ratio` classvar → LoopClosureConfig default). Remove "matches VGGT-SPARK calibration" claim.
- `vggt_omega.py` (~126): fix stale comment claiming inheritance of `default_verify_match_ratio=0.85` — actual calibration is L13/1.55 classvar.
- `vggt_spark_creator.py` and `vggt_omega.py`: at the `from vggt.utils... import unproject_depth_map_to_point_map` line add comment: `# Imported from installed VGGT-X tree (shared dep) — byte-identical to this model's vendored copy today; revisit if trees diverge.`
- `closure.py` `LoopClosureConfig.manifold` field: extend inline comment: `# "se3": scale folds into Pose3 with non-orthonormal Rot3 (pre-existing limitation; "sl4" is the parity default and handles scale correctly).`

- [ ] **Step 7: Orphan file deletions**

```bash
grep -rn "run_tutorials" . --include="*.md" --include="*.sh" --include="*.yml" --include="*.yaml" --include="Makefile" -l | grep -v _build   # expect empty
grep -rn "env\.yml" Dockerfile* setup*.sh Makefile .github/ docs/README.md README.md 2>/dev/null                                            # expect empty
git rm docs/examples/run_tutorials.sh env.yml
```
If either grep hits, keep that file and note in commit body.

- [ ] **Step 8: Full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q --ignore=tests/mesh 2>&1 | tail -3`
Expected: baseline minus the explicitly deleted tests (Step 2 deprecation tests, Step 3 stub tests), zero new failures.

- [ ] **Step 9: Commit**

```bash
git add -A -- collab_splats tests docs/examples env.yml
git commit -m "refactor(lc): remove deprecated config fields, dead eval stubs, unused utils; hygiene fixes

Deleted tests (features removed with them): <list from Steps 2-3>"
```

---

### Task 3: Geometry package move

**Files:**
- Create: `collab_splats/geometry/__init__.py`
- Move: `collab_splats/utils/geometry.py` → `collab_splats/geometry/transforms.py`; `collab_splats/pointcloud/bundle_adjustment.py` → `collab_splats/geometry/bundle_adjustment.py`; `collab_splats/pointcloud/loop_closure/` → `collab_splats/geometry/loop_closure/`; `collab_splats/pointcloud/wrappers.py` → `collab_splats/geometry/loop_closure/wrapper.py`
- Modify: every importer (Step 4 list)

- [ ] **Step 1: `git mv` the modules**

```bash
mkdir -p collab_splats/geometry
git mv collab_splats/utils/geometry.py collab_splats/geometry/transforms.py
git mv collab_splats/pointcloud/loop_closure collab_splats/geometry/loop_closure
git mv collab_splats/pointcloud/bundle_adjustment.py collab_splats/geometry/bundle_adjustment.py
git mv collab_splats/pointcloud/wrappers.py collab_splats/geometry/loop_closure/wrapper.py
find collab_splats -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
```
Check `collab_splats/utils/__init__.py` for a `geometry` re-export; remove it if present.

- [ ] **Step 2: Create `collab_splats/geometry/__init__.py`**

```python
"""Geometry backend: loop closure, bundle adjustment, and SE(3)/pose transforms."""
from .bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
from .loop_closure import LoopClosureConfig, PoseGraph, Submap
from .loop_closure.wrapper import LoopClosure
from .transforms import (
    OPENGL_TO_OPENCV,
    extract_intrinsics,
    extrinsics_to_homogeneous,
    invert_poses,
    rotation_align_vectors,
)

__all__ = [
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "LoopClosure",
    "LoopClosureConfig",
    "PoseGraph",
    "Submap",
    "OPENGL_TO_OPENCV",
    "extract_intrinsics",
    "extrinsics_to_homogeneous",
    "invert_poses",
    "rotation_align_vectors",
]
```
Adjust names to match the actual current exports of `bundle_adjustment.py` / `loop_closure/__init__.py` (check `__all__` in each; keep existing re-export surface, e.g. `LoopMatch`, `find_loop_closures` if `loop_closure/__init__.py` exported them). Add `wrapper` re-export `LoopClosure` to `geometry/loop_closure/__init__.py` as well.

- [ ] **Step 3: Fix relative imports inside moved files**

- `geometry/loop_closure/wrapper.py`: `from .base import PointcloudResult` → `from collab_splats.pointcloud.base import PointcloudResult`; `from .feedforward import FeedforwardResult, _raw_to_world_points` → `from collab_splats.pointcloud.feedforward import ...`; `from .loop_closure import ...` / `from .loop_closure.closure import ...` / `from .loop_closure.submap import ...` → relative same-package imports (`from . import ...` / `from .closure import ...` / `from .submap import ...`); `from collab_splats.utils.geometry import ...` → `from collab_splats.geometry.transforms import ...`. Also the inline `from .loop_closure.closure import dedup_overlap` in `_assemble_precorrection_extrinsics` → `from .closure import dedup_overlap` (move to top of file per import policy).
- `geometry/bundle_adjustment.py`: `from .feedforward.base import FeedforwardResult` (TYPE_CHECKING) → `from collab_splats.pointcloud.feedforward.base import FeedforwardResult`; `utils.geometry` → `geometry.transforms`.
- `geometry/loop_closure/*.py`: intra-package relative imports (`.graph`, `.submap`) unchanged; any `collab_splats.utils.geometry` → `collab_splats.geometry.transforms`.

- [ ] **Step 4: Rewrite external importers**

```bash
grep -rln "utils\.geometry\|utils import geometry\|pointcloud\.loop_closure\|pointcloud\.bundle_adjustment\|pointcloud\.wrappers\|pointcloud import.*\(LoopClosure\|BundleAdjustment\)" collab_splats evals tests docs/source --include="*.py" --include="*.ipynb" --include="*.rst"
```
Apply mapping everywhere the grep hits:
| old | new |
|---|---|
| `collab_splats.utils.geometry` | `collab_splats.geometry.transforms` |
| `collab_splats.pointcloud.loop_closure` | `collab_splats.geometry.loop_closure` |
| `collab_splats.pointcloud.bundle_adjustment` | `collab_splats.geometry.bundle_adjustment` |
| `collab_splats.pointcloud.wrappers` | `collab_splats.geometry.loop_closure.wrapper` |
| `from collab_splats.pointcloud import LoopClosure/BundleAdjustment(Config)/LoopClosureConfig` | `from collab_splats.geometry import ...` |

Known sites: `pointcloud/base.py`, `pointcloud/utils.py`, all `pointcloud/feedforward/*.py`, `mesh/*.py` (**import lines ONLY — no other edits in mesh/, and use the working-tree file as is: user has uncommitted edits there**), `nerfstudio/**`, `wrapper/reconstructor.py`, `wrapper/splatter.py`, `evals/eval_gt.py`, `evals/metrics.py`, `evals/check_ate_methods.py`, `evals/_ba_finding_eval.py`, `evals/runners/run_lc_parity.py`, `evals/lc_parity_common.py`, `evals/run_vggt_slam_lc.py`, `evals/lc_loop_pr.py`, `evals/runners/visualize_lc_correction.py`, `evals/build_parity_table.py`, `evals/eval_similarity_calibration.py`, `evals/reconstruction_quality.py`, all `tests/**` hits. Skip: `docs/superpowers/**` (historical), `docs/_build/**` (generated), the 7 evals files being deleted in Task 5 (still update them if trivial — they must not break collection before Task 5; simplest is to update them too), FROZEN notebooks (`feedforward_mesh.ipynb`, `feedforward_methods.ipynb` — if grep hits them, leave and note for the mesh session). Live notebooks `slam_loop_closure.ipynb` / `bundle_adjustment.ipynb`: update in Task 7, not here (not imported by suite).

- [ ] **Step 5: Break the `pointcloud/__init__` ↔ `geometry` import cycle**

`pointcloud/__init__.py` currently re-exports LC/BA names and `create_pointcloud_creator` constructs `LoopClosure` (~lines 17-19, 60-63). A top-level `from collab_splats.geometry import LoopClosure` there is circular (geometry.wrapper imports pointcloud.feedforward). Fix:
- Delete the LC/BA re-export lines and their `__all__` entries from `pointcloud/__init__.py` (hard cut).
- Inside `create_pointcloud_creator`, use a deferred import:

```python
def create_pointcloud_creator(...):
    ...
    # Deferred import — avoids circular dependency: geometry.loop_closure.wrapper
    # imports pointcloud.feedforward, so geometry cannot be imported at module load.
    from collab_splats.geometry import BundleAdjustment, LoopClosure
    ...
```
(Keep the existing construction logic; only the import location changes.)

- [ ] **Step 6: Import smoke test both directions**

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry, collab_splats.pointcloud; from collab_splats.geometry import LoopClosure, BundleAdjustment; print('geometry-first ok')"
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud, collab_splats.geometry; from collab_splats.pointcloud import create_pointcloud_creator; print('pointcloud-first ok')"
grep -rn "pointcloud\.loop_closure\|pointcloud\.bundle_adjustment\|pointcloud\.wrappers\|utils\.geometry\b" collab_splats evals tests --include="*.py"   # expect empty
```

- [ ] **Step 7: Full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q --ignore=tests/mesh 2>&1 | tail -3`
Expected: identical counts to end of Task 2. Any `ImportError`/`ModuleNotFoundError` → fix the missed importer, re-run.

- [ ] **Step 8: Commit**

```bash
git add -A -- collab_splats evals tests
git commit -m "refactor(geometry): promote loop_closure + bundle_adjustment + transforms to collab_splats/geometry

Hard cut, no shims (localization-promotion precedent). git mv preserves history.
utils/geometry.py -> geometry/transforms.py; pointcloud/wrappers.py -> geometry/loop_closure/wrapper.py.
create_pointcloud_creator uses deferred geometry import to break the package cycle."
```

---

### Task 4: Test tree reorganization

**Files:**
- Create: `tests/geometry/`, `tests/geometry/loop_closure/`, `tests/pointcloud/feedforward/conftest.py`
- Move/modify: LC/BA test files (Step 1 list); `tests/utils/test_geometry.py` → `tests/geometry/test_transforms.py`
- Merge: `tests/test_visualization.py` + `tests/utils/test_visualization.py`; two `test_mapanything_creator.py` files

- [ ] **Step 1: Move geometry tests**

```bash
mkdir -p tests/geometry/loop_closure
git mv tests/utils/test_geometry.py tests/geometry/test_transforms.py
git mv tests/pointcloud/test_bundle_adjustment.py tests/geometry/
git mv tests/pointcloud/test_wrappers.py tests/geometry/loop_closure/test_wrapper.py
for f in test_loop_closure.py test_loop_closure_eval.py test_loop_closure_integration.py test_graph.py test_verify_threshold_resolution.py test_submap_reprojection.py test_eval_metrics.py test_pgo_parity.py test_alignment_dedup.py test_translation_jump.py test_pose_convention.py test_hw_formula.py test_closure_split.py test_feedforward_lc_state.py; do git mv tests/pointcloud/$f tests/geometry/loop_closure/; done
git mv tests/pointcloud/loop_closure/test_loop_ablation.py tests/geometry/loop_closure/
git mv tests/pointcloud/loop_closure/test_loop_edge_chain.py tests/geometry/loop_closure/
rmdir tests/pointcloud/loop_closure 2>/dev/null; true
```
If any file is missing (deleted in Task 2, e.g. `test_auc_metric.py` already gone), skip it. Check for `conftest.py` fixtures the moved tests relied on in `tests/pointcloud/conftest.py` — if any moved test breaks on a missing fixture, copy that fixture into `tests/geometry/conftest.py` (do not duplicate if it can live in `tests/conftest.py`).

- [ ] **Step 2: Shared fake-model fixture**

Create `tests/pointcloud/feedforward/conftest.py`. Move the `_FakeQKV` and `_FakeMapAnythingModel` classes (currently duplicated at `tests/pointcloud/feedforward/test_verify_lc_data.py:20,27` and `tests/pointcloud/test_mapanything_creator.py:141,148` — take the more complete variant if they've drifted; diff them first) into the conftest as plain classes plus fixtures:

```python
"""Shared fakes for feedforward LC tests."""
import pytest

# _FakeQKV / _FakeMapAnythingModel moved verbatim from test_verify_lc_data.py
# (diffed against test_mapanything_creator.py copy — identical / superset taken).


@pytest.fixture
def fake_mapanything_model():
    return _FakeMapAnythingModel()
```
Update both test files to use the fixture / import from conftest; delete the duplicated class bodies.

- [ ] **Step 3: Merge same-named test files**

- `tests/utils/test_visualization.py` (41L) → append its test functions into `tests/test_visualization.py` (143L) unless a same-named test already exists (then keep the canonical one and drop the duplicate — they test the same symbol); `git rm tests/utils/test_visualization.py`. Result: one file, union of unique tests.
- `tests/pointcloud/feedforward/test_mapanything_creator.py` (193L) → merge into `tests/pointcloud/test_mapanything_creator.py` (644L) the same way, then `git mv tests/pointcloud/test_mapanything_creator.py tests/pointcloud/feedforward/test_mapanything_creator.py` (canonical home = feedforward subdir, mirrors code tree).
- Verify: `/opt/venv/reconstruction/bin/python -m pytest tests/test_visualization.py tests/pointcloud/feedforward/test_mapanything_creator.py -q` — total test count for these = sum of unique tests pre-merge (count before with `--collect-only -q`).

- [ ] **Step 4: Full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q --ignore=tests/mesh 2>&1 | tail -3`
Expected: same totals as end of Task 3 (minus exact-duplicate tests dropped in Step 3 — list them in the commit body if any).

- [ ] **Step 5: Commit**

```bash
git add -A -- tests
git commit -m "test(geometry): move LC/BA tests to tests/geometry, consolidate fake-model stubs, merge same-named files"
```

---

### Task 5: Evals prune + gitignore fix

**Files:**
- Delete: `evals/runners/debug_lc_steps.py`, `evals/runners/parity_trace.py`, `evals/runners/our_solver_dump.py`, `evals/runners/compare_slam_ours.py`, `evals/runners/diagnose_lc_parity.py`, `evals/compare_loop_edges.py`, `evals/eval_vggt_slam_comparison.py`
- Modify: `.gitignore`

- [ ] **Step 1: Verify nothing kept imports the deletions**

```bash
grep -rn "debug_lc_steps\|parity_trace\|our_solver_dump\|compare_slam_ours\|diagnose_lc_parity\|compare_loop_edges\|eval_vggt_slam_comparison" collab_splats evals tests docs Makefile .github --include="*.py" --include="*.md" --include="*.yml" --include="*.sh" | grep -v docs/superpowers | grep -v docs/_build
```
Expected: hits only within the 7 files themselves. Any external hit → keep that file, note in commit body.

- [ ] **Step 2: Delete**

```bash
git rm evals/runners/debug_lc_steps.py evals/runners/parity_trace.py evals/runners/our_solver_dump.py evals/runners/compare_slam_ours.py evals/runners/diagnose_lc_parity.py evals/compare_loop_edges.py evals/eval_vggt_slam_comparison.py
```

- [ ] **Step 3: Gitignore fix**

Add to `.gitignore` next to the existing `evals/results/` entry:
```
evals/baselines/
```
Verify: `git status --short evals/ | grep '??'` → the `lc_parity*` dirs disappear from untracked.

- [ ] **Step 4: Evals tests still pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals -q 2>&1 | tail -3`
Expected: all pass (no kept test imports a deleted script — Step 1 guarantees).

- [ ] **Step 5: Commit**

```bash
git add -A -- evals .gitignore
git commit -m "chore(evals): remove superseded LC-parity one-off diagnostics; gitignore evals/baselines"
```

---

### Task 6: Tier B gated deletions

**Files:**
- Modify/Delete: `collab_splats/utils/visualization.py`, `collab_splats/utils/notebook.py`, `collab_splats/nerfstudio/utils/camera_utils.py`

- [ ] **Step 1: Run the notebook + repo gate per symbol**

```bash
for s in plot_heatmap plot_context_segmentation query_heatmap overlay_masks o3d_mesh_to_polydata clean_and_extract_result add_camera_frustums feature_viz_row ndc2pix_x ndc2pix_y euclidean_to_z_depth project_pix get_colored_points_from_depth get_rays_x_y_1; do
  echo "== $s"; grep -rln "$s" collab_splats evals tests webapp docs/source --include="*.py" --include="*.ipynb" | grep -v _build
done
```
For each symbol: definition-file-only hits → DELETE. Any other hit (tutorial notebook, webapp, test) → KEEP, record in commit body.

- [ ] **Step 2: Apply deletions**

- `utils/visualization.py`: delete the gated-clean functions among `plot_heatmap`, `plot_context_segmentation`, `query_heatmap`, `overlay_masks`, `o3d_mesh_to_polydata` (+ their now-unused imports; run `black`/`isort` on the file after).
- `utils/notebook.py`: if all three functions gate clean → `git rm collab_splats/utils/notebook.py` and remove any `utils/__init__.py` re-export; otherwise delete only the clean ones.
- `nerfstudio/utils/camera_utils.py`: delete gated-clean functions among `ndc2pix_x`, `ndc2pix_y`, `euclidean_to_z_depth`, `project_pix`, `get_colored_points_from_depth`, `get_rays_x_y_1` (+ unused imports).
- Delete tests that exist solely for deleted functions (gate grep in Step 1 reveals them; list names in commit body).

- [ ] **Step 3: Full suite + import smoke**

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.utils.visualization, collab_splats.nerfstudio.utils.camera_utils; print('ok')"
/opt/venv/reconstruction/bin/python -m pytest tests/ -q --ignore=tests/mesh 2>&1 | tail -3
```
Expected: green, totals = end of Task 4 minus explicitly listed deleted tests.

- [ ] **Step 4: Commit**

```bash
git add -A -- collab_splats tests
git commit -m "refactor(utils): remove unused visualization/notebook/camera_utils functions (notebook-grep gated)

Kept (gate hit): <list or 'none'>. Deleted tests: <list or 'none'>"
```

---

### Task 7: Docs path updates + audit move

**Files:**
- Modify: `docs/source/api/pointcloud.rst`, `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`, `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`, `CLAUDE.md`
- Create: `docs/source/api/geometry.rst`
- Move: `MESH_PIPELINE_AUDIT.md` → `docs/superpowers/audits/2026-05-29-mesh-pipeline-audit.md`

- [ ] **Step 1: API rst**

Remove `loop_closure` / `bundle_adjustment` / `wrappers` automodule entries from `docs/source/api/pointcloud.rst`. Create `docs/source/api/geometry.rst` mirroring the existing rst style (open `pointcloud.rst` and copy its structure) with automodule entries for `collab_splats.geometry.transforms`, `collab_splats.geometry.bundle_adjustment`, `collab_splats.geometry.loop_closure.closure`, `.graph`, `.submap`, `.wrapper`. Add `geometry` to the api toctree (find it: `grep -rn "pointcloud" docs/source/api/index.rst docs/source/index.rst 2>/dev/null`).

- [ ] **Step 2: Live notebooks — import paths only**

In `slam_loop_closure.ipynb` and `bundle_adjustment.ipynb` update import cells per the Task 3 Step 4 mapping (edit the JSON `source` arrays; content/markdown untouched). Do NOT touch `feedforward_mesh.ipynb` / `feedforward_methods.ipynb` (frozen). If a frozen notebook imports moved paths, add one line to the commit body: "frozen mesh notebooks still reference old paths — update when mesh work lands".

- [ ] **Step 3: CLAUDE.md architecture tree**

Update the tree in `/workspace/collab-splats/CLAUDE.md`: remove `bundle_adjustment.py`, `wrappers.py`, `loop_closure/` lines from `pointcloud/`; add:
```
  geometry/                # pose/geometry backend: loop closure + bundle adjustment
    transforms.py          # extrinsics_to_homogeneous, invert_poses, OPENGL_TO_OPENCV (ex utils/geometry.py)
    bundle_adjustment.py   # Levenberg-Marquardt BA
    loop_closure/          # submap pose graph (SL4/SE3), DINO-SALAD retrieval gate, LoopClosure wrapper
```
Remove the `utils/geometry` mention if present.

- [ ] **Step 4: Audit doc move**

```bash
mkdir -p docs/superpowers/audits
git mv MESH_PIPELINE_AUDIT.md docs/superpowers/audits/2026-05-29-mesh-pipeline-audit.md
```

- [ ] **Step 5: Docs build smoke (optional, skip if sphinx env absent)**

Run: `ls docs/Makefile 2>/dev/null && grep -rn "loop_closure\|bundle_adjustment" docs/source/api/*.rst`
Expected: only `geometry.rst` hits. (Full sphinx build is slow; grep check suffices.)

- [ ] **Step 6: Commit**

```bash
git add -f docs/source docs/superpowers/audits CLAUDE.md
git rm --cached MESH_PIPELINE_AUDIT.md 2>/dev/null; true
git commit -m "docs: geometry package path updates; move mesh audit under docs/superpowers/audits

Paths only — no content reduction."
```

---

## Final verification (after Task 7)

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q --ignore=tests/mesh 2>&1 | tail -3
grep -rn "pointcloud\.loop_closure\|pointcloud\.bundle_adjustment\|pointcloud\.wrappers\|utils\.geometry\b" collab_splats evals tests docs/source --include="*.py" --include="*.ipynb" | grep -v _build   # expect empty
git log --oneline -8   # 7 commits in spec order + spec commit
```
Report: final pass/skip/xfail vs baseline, every deleted test name, every Tier B symbol kept due to gate hit, net LOC delta (`git diff --stat <baseline-sha>..HEAD | tail -1`).
