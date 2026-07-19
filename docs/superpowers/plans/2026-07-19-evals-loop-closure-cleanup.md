# Evals Suite + Loop-Closure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce LOC and consolidate duplicated logic in `evals/` and `collab_splats/geometry/loop_closure/` while keeping `pytest tests/` green throughout — dead-code removal takes its covering tests with it; everything else reproduces existing behavior exactly.

**Architecture:** Seven independently-landable sections (see spec `docs/superpowers/specs/2026-07-19-evals-loop-closure-cleanup-design.md`): (1) delete dead eval files/dirs, (2) extract genuine library code out of `evals/`, (3) minor table/viz dedup, (4) reorg `evals/runners/`→`evals/scripts/` + consolidate downloaders, (5) remove LC dead branches, (6) split `closure.py` into `matching.py`/`graph.py`/`merge.py`, (7) rename the `LoopClosure` per-window loop to mirror VGGT-SLAM. Land in that order; each section is a separable commit so a regression bisects cleanly.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, numpy, gtsam, evo, scipy. Format with `black . && isort .` before each commit.

**Reference the spec constantly.** This plan encodes the spec's "Correction (verified this session)" notes as hard requirements. Where this plan and the spec's prose disagree, the spec's *corrections* win (they are the latest verified state).

---

## Pre-flight: authoritative scope decisions

Three spec ambiguities resolved before any task runs — these are not open questions, they are decisions baked into the tasks below:

1. **§3 shared `sweep_driver.py` is OUT OF SCOPE.** §4's verified correction supersedes §3's prose: no `sweep_driver.py` exists, and the three runners' CLI-flag generalization is *already committed* (git log: `generalize run_disparity_sweep.py`, `generalize run_cross_model_benchmark.py`, `expose build_benchmark_table.py --reference_backbone`). The remaining §3 work is only the two *minor* items (table-render helper, `visualize_lc_correction` reuse) — Tasks in Section 3 below cover just those, marked optional.
2. **§1 `baselines/lc_parity*/` flatten is local-disk-only.** `.gitignore:16` (`evals/baselines/lc_parity*/`) ignores all four dirs, so there are no tracked files to move — flattening has zero committed effect and no test impact. Marked non-blocking; skip unless doing local housekeeping.
3. **`scale_method="none"` STAYS** (§5 correction). Only `normalize_to_sl4` and `manifold="se3"` are dead. Do not touch the `"none"` branch in `_lc_anchor_scale` / `run_pose_graph_optimization`.

---

## File Structure

**Deleted:** `evals/notebooks/` (empty), `evals/envs/` (`vggt_long.yml`), `evals/ate_utils.py`, `evals/data/download_parity_scenes.sh`, `evals/download_7scenes.sh`, `evals/download_co3dv2.sh`, `evals/download_tum.sh`, `evals/download_waymo.sh`, `evals/download_kitti.sh`, `tests/evals/test_ate_utils_tum.py`, `collab_splats/geometry/loop_closure/closure.py` (dissolved).

**Created:** `collab_splats/geometry/loop_closure/diagnostics.py` (from `reconstruction_quality.py`), `collab_splats/geometry/loop_closure/matching.py`, `collab_splats/geometry/loop_closure/merge.py`, `evals/data/download.py`, `tests/geometry/loop_closure/test_diagnostics.py`.

**Renamed:** `evals/runners/` → `evals/scripts/` (+ all `eval_*.py` entry points folded in); `evals/runners/extract_waymo.py` → `evals/data/extract_waymo.py`.

**Modified in place:** `evals/trajectory_io.py`, `evals/datasets.py`, `evals/metrics.py` (unchanged API), `collab_splats/geometry/loop_closure/graph.py` (gains split functions), `collab_splats/geometry/loop_closure/wrapper.py` (gains `LoopClosureConfig`, `run_predictions`/`add_points`), `collab_splats/geometry/loop_closure/__init__.py`, `collab_splats/geometry/__init__.py`, `collab_splats/geometry/loop_closure/eval.py`, plus the enumerated test files and `CLAUDE.md`.

---

## Section 0: Baseline

### Task 0: Green baseline

**Files:** none (verification only)

- [ ] **Step 1: Record the pre-change suite result**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -20`
Expected: the known-good pass/xfail counts (cross-check against `docs/known-test-failures.md`). Note the exact numbers; every section below must return to this baseline (minus tests deleted alongside dead code).

- [ ] **Step 2: Confirm working tree is committed or stashed**

Run: `git status --short`
Expected: only the pre-existing modifications from `git status` at session start. Do not start on a dirty tree you can't attribute.

---

## Section 1: Evals dead files/dirs

### Task 1.1: Delete empty/unreferenced dirs

**Files:**
- Delete: `evals/notebooks/` (empty)
- Delete: `evals/envs/vggt_long.yml` then `evals/envs/`

- [ ] **Step 1: Confirm nothing references them**

Run: `grep -rn "envs/vggt_long\|evals/notebooks\|vggt_long.yml" evals/ tests/ docs/ setup/ *.sh *.md 2>/dev/null`
Expected: no hits (README's `envs/` mention is the layout table only — updated in Section 4).

- [ ] **Step 2: Delete**

```bash
rmdir evals/notebooks
git rm evals/envs/vggt_long.yml && rmdir evals/envs 2>/dev/null || true
```

- [ ] **Step 3: Suite still green**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/ -q`
Expected: same as baseline (these dirs have no tests).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(evals): delete empty notebooks/ and unreferenced envs/"
```

### Task 1.2 (non-blocking, local-disk only): note baselines flatten

`evals/baselines/{lc_parity,lc_parity_d5,lc_parity_d5_postfix,lc_parity_matrix}/` are all matched by `.gitignore:16` (`evals/baselines/lc_parity*/`). There are **no tracked files** to flatten — skip in the committed pass. If a developer wants local tidiness, `mv` the three variant dirs' contents under `lc_parity/<config>/` on disk; nothing in git or tests changes. **No task, no commit.**

---

## Section 2: Evals library extraction → `collab_splats`

### Task 2.1: Delete `trajectory_io._invert_se3`, use `geometry.transforms.invert_poses`

**Files:**
- Modify: `evals/trajectory_io.py` (delete `_invert_se3` at lines 29-37; add import; swap 4 call sites)
- Test: `tests/evals/test_trajectory_io.py` (add a case pinning the swap)

`invert_poses` (`collab_splats/geometry/transforms.py:39`) is a strict superset: same `R^T, -R^T@t`, arbitrary leading batch dims. `evals/` already imports from `collab_splats` (e.g. `metrics.py`), so this is not a new dependency direction. Keep `_check_poses`.

- [ ] **Step 1: Add a test that trajectory_io round-trips via invert_poses**

In `tests/evals/test_trajectory_io.py`, add:

```python
def test_write_read_tum_roundtrip_matches_invert_poses():
    # A non-symmetric rotation exposes R vs R^T bugs the identity case hides.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    import trajectory_io
    from collab_splats.geometry.transforms import invert_poses

    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation as R
    w2c = np.tile(np.eye(4), (3, 1, 1))
    w2c[:, :3, :3] = R.random(3, random_state=1).as_matrix()
    w2c[:, :3, 3] = rng.normal(size=(3, 3))

    tmp = Path(trajectory_io.__file__).parent  # not used for write; use tmp_path below
    # write then read is identity on w2c
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.tum")
        trajectory_io.write_tum(p, w2c)
        back, _ = trajectory_io.read_tum(p)
    np.testing.assert_allclose(back, w2c, atol=1e-6)
    # and the c2w it serialized equals invert_poses(w2c)
    np.testing.assert_allclose(invert_poses(w2c), invert_poses(w2c), atol=1e-9)
```

- [ ] **Step 2: Run it against the current code to confirm it passes pre-change (guards the refactor)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_trajectory_io.py -q`
Expected: PASS (current `_invert_se3` already round-trips).

- [ ] **Step 3: Replace `_invert_se3` with `invert_poses`**

In `evals/trajectory_io.py`: delete the `_invert_se3` function (lines 29-37). Add near the top imports:

```python
from collab_splats.geometry.transforms import invert_poses
```

Replace the four `_invert_se3(...)` call sites (`write_tum`, `read_tum`, `kitti_3x4_flat_to_w2c`, `w2c_to_kitti_3x4_flat`) with `invert_poses(...)`. Signatures match (both take/return `(N,4,4)`).

- [ ] **Step 4: Run the suite for trajectory_io + everything importing it**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_trajectory_io.py tests/evals/test_metrics.py tests/evals/test_datasets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black evals/trajectory_io.py && isort evals/trajectory_io.py
git add evals/trajectory_io.py tests/evals/test_trajectory_io.py
git commit -m "refactor(evals): drop trajectory_io._invert_se3 for geometry.invert_poses"
```

### Task 2.2: Move `reconstruction_quality.py` → `loop_closure/diagnostics.py`

**Files:**
- Create: `collab_splats/geometry/loop_closure/diagnostics.py` (verbatim body of `evals/reconstruction_quality.py`)
- Delete: `evals/reconstruction_quality.py`
- Modify: `evals/eval_gt.py:405` inline import
- Move: `tests/evals/test_reconstruction_quality.py` → `tests/geometry/loop_closure/test_diagnostics.py`

Justification (spec §2): its aggregator reads private `_lc_*` attributes off a post-run `LoopClosure.base` (set in `wrapper.py:381-398`) — it's a diagnostics extension of `geometry/loop_closure`, cohesion-driven, not eval-comparison code. Public metrics (`loop_match_residual`, `submap_boundary_gap`, `pointcloud_chamfer`) + helpers (`_cam_positions`, `_symmetric_chamfer`, `_global_frame`) move as-is.

- [ ] **Step 1: Create the new module**

```bash
git mv evals/reconstruction_quality.py collab_splats/geometry/loop_closure/diagnostics.py
```
(Content is import-clean — only `numpy` + lazy `scipy.spatial.cKDTree` inside `_symmetric_chamfer`; no `evals` imports — so a plain move suffices.)

- [ ] **Step 2: Update `eval_gt.py`'s inline import**

In `evals/eval_gt.py` line ~405, change:

```python
from reconstruction_quality import compute_alignment_metrics
```
to:
```python
from collab_splats.geometry.loop_closure.diagnostics import compute_alignment_metrics
```
Leave it inline (it is on the optional/heavy LC-metrics path).

- [ ] **Step 3: Move the test and repoint its import**

```bash
git mv tests/evals/test_reconstruction_quality.py tests/geometry/loop_closure/test_diagnostics.py
```
In `tests/geometry/loop_closure/test_diagnostics.py`, delete the `sys.path.insert(..., "evals")` + `from reconstruction_quality import ...` lines and replace with:
```python
from collab_splats.geometry.loop_closure.diagnostics import (
    compute_alignment_metrics,
    loop_match_residual,
    submap_boundary_gap,
    pointcloud_chamfer,
)
```
(Import only the names the test actually uses — check the file.)

- [ ] **Step 4: Run the moved test**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/geometry/loop_closure/diagnostics.py tests/geometry/loop_closure/test_diagnostics.py
isort collab_splats/geometry/loop_closure/diagnostics.py tests/geometry/loop_closure/test_diagnostics.py
git add -A
git commit -m "refactor(geometry): move reconstruction_quality → loop_closure/diagnostics"
```

### Task 2.3: Dissolve `ate_utils.py`

**Files:**
- Modify: `evals/datasets.py` (gain `_load_gt_as_tum_trajectory`)
- Modify: `evals/runners/run_vggt_slam_lc.py` (rewrite the ATE call path; kill the `importlib` workaround)
- Delete: `evals/ate_utils.py`
- Modify: `tests/evals/test_datasets.py` (absorb `test_ate_utils_tum.py`'s cases)
- Delete: `tests/evals/test_ate_utils_tum.py`

`ate_utils.py` bundles (a) `_load_gt_as_tum_trajectory` — eval-specific GT loading, belongs with `datasets.py`'s other loaders; (b) `compute_ate_rmse` — duplicates `metrics.compute_ate` via a second evo entry point, differing only in accepting a raw `seq_dir`. Its one caller writes the loaded GT through `write_tum` then calls `metrics.compute_ate`, collapsing to one evo path.

> **Note:** this task edits `run_vggt_slam_lc.py` in its current `evals/runners/` location. Section 4 later moves the whole dir to `evals/scripts/`. Doing 2.3 first keeps sections independently bisectable; the Section-4 move is a pure `git mv` afterward.

- [ ] **Step 1: Move `_load_gt_as_tum_trajectory` into `datasets.py`**

Cut the full `_load_gt_as_tum_trajectory` function (and its `from evo.core.trajectory import PoseTrajectory3D` — keep it inline in the function, since `datasets.py` should not hard-import evo at module load) from `evals/ate_utils.py` into `evals/datasets.py`. It already uses `re`, `Path`, `np`, `scipy.spatial.transform.Rotation` — add `import re` if absent and keep `Rotation` imported inside the function (matches `datasets.py`'s existing `_read_tum_groundtruth` lazy-import style).

- [ ] **Step 2: Rewrite the caller in `run_vggt_slam_lc.py`**

Remove the `importlib.util.spec_from_file_location("ate_utils", ...)` block (lines ~51-56) and the `compute_ate_rmse = _ate_mod.compute_ate_rmse` binding. At the ATE call site (line ~179), replace `compute_ate_rmse(out_tum, seq_dir, selected_frames_path=kf_path)` with a two-step: load GT via the relocated `datasets._load_gt_as_tum_trajectory` (through the same `sys.path`/shadow-safe import the file already uses for `lc_parity_common`), write it to a temp TUM via `trajectory_io.write_tum`, and call `metrics.compute_ate(pred=out_tum, gt=<temp>, align="sim3")`.

Concretely (adapt to the file's existing import mechanics):

```python
# GT trajectory for the selected frames → temp TUM → evo APE via metrics.compute_ate.
import tempfile
from datasets import _load_gt_as_tum_trajectory
import trajectory_io
import metrics as _metrics

def _ate_rmse(out_tum, seq_dir, kf_path):
    selected = [Path(p) for p in kf_path.read_text().splitlines() if p.strip()]
    traj_ref = _load_gt_as_tum_trajectory(seq_dir, selected_frames=selected)
    with tempfile.NamedTemporaryFile("w", suffix=".tum", delete=False) as f:
        gt_tum = Path(f.name)
    # traj_ref is a PoseTrajectory3D (cam-to-world); write_tum expects world-to-cam.
    w2c = invert_poses(np.stack(traj_ref.poses_se3).astype(np.float64))
    trajectory_io.write_tum(gt_tum, w2c, timestamps=traj_ref.timestamps)
    return _metrics.compute_ate(out_tum, gt_tum, align="sim3")["rmse"]
```

> **Verify the convention:** `PoseTrajectory3D.poses_se3` are cam-to-world `(4,4)`; `write_tum` inverts world-to-cam→cam-to-world internally, so pass `invert_poses(...)` as above. If a simpler evo writer (`evo.tools.file_interface.write_tum_trajectory_file`) is preferred, it takes the `PoseTrajectory3D` directly and avoids the manual inversion — use whichever the reviewer finds clearer, but **assert numerical equality against the old `compute_ate_rmse` on one sequence before deleting `ate_utils.py`** (Step 4).

- [ ] **Step 3: Absorb the ate_utils tests into `test_datasets.py`**

Read `tests/evals/test_ate_utils_tum.py`; its cases exercise `_load_gt_as_tum_trajectory` (TUM nearest-neighbour association, 7-Scenes frame-index timestamps). Move those test bodies into `tests/evals/test_datasets.py` (which already has the `sys.path.insert(..., "evals")` pattern per-test), repointing `from ate_utils import _load_gt_as_tum_trajectory` → `from datasets import _load_gt_as_tum_trajectory`. Then:
```bash
git rm tests/evals/test_ate_utils_tum.py
```

- [ ] **Step 4: Numerical-equivalence guard, then delete `ate_utils.py`**

Before deleting, run the old and new ATE paths on one fixture sequence (or a synthetic GT+pred pair) and assert `abs(old - new) < 1e-6`. Once equal:
```bash
git rm evals/ate_utils.py
```

- [ ] **Step 5: Run the affected tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_datasets.py tests/evals/test_run_vggt_slam_lc.py tests/evals/test_metrics.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
black evals/datasets.py evals/runners/run_vggt_slam_lc.py tests/evals/test_datasets.py
isort evals/datasets.py evals/runners/run_vggt_slam_lc.py tests/evals/test_datasets.py
git add -A
git commit -m "refactor(evals): dissolve ate_utils into datasets loader + metrics.compute_ate"
```

### Task 2.4: Section-2 checkpoint

- [ ] **Step 1: Full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -5`
Expected: baseline counts minus the deleted `test_ate_utils_tum.py` cases (now living in `test_datasets.py`) and `test_reconstruction_quality.py` (now `test_diagnostics.py`) — net pass count unchanged.

---

## Section 3: Minor table/viz dedup (optional)

> **Scope, per Pre-flight decision 1:** the three-runner sweep-driver dedup is OUT OF SCOPE; runner CLI-generalization is already committed. Only two small items remain. Both are optional polish — land them only if the suite stays green with no behavior change.

### Task 3.1 (optional): `visualize_lc_correction.py` reuse `umeyama_sim3`

**Files:** `evals/runners/visualize_lc_correction.py`

It already imports `umeyama_sim3` (line 23) but keeps hand-rolled `sim3_align`/`apply_sim3` (lines ~56-65). If `sim3_align` reimplements the Umeyama estimate rather than delegating, replace its estimation body with a call to the imported `umeyama_sim3(source, target)` (returns `(s, R, t)`), keeping `apply_sim3` as the thin `s*R@x+t` applier.

- [ ] **Step 1:** Confirm current behavior via `tests/evals/test_visualize_lc_correction.py`, run it, note output.
- [ ] **Step 2:** Rewire `sim3_align` to delegate to `umeyama_sim3`.
- [ ] **Step 3:** Re-run `tests/evals/test_visualize_lc_correction.py`; assert identical results.
- [ ] **Step 4:** Commit `refactor(evals): visualize_lc_correction reuses umeyama_sim3`.

> **Section-6 dependency:** `umeyama_sim3` moves `closure.py`→`graph.py` in Section 6. If Section 3 lands first, the import stays `from ...loop_closure.closure import umeyama_sim3`; Section 6's Task 6.4 will repoint it. If ordering swaps, point it at `.graph` here. Either way the enumerated-importer list in Section 6 includes this file.

### Task 3.2 (optional): table-builder shared render helper

**Files:** `evals/runners/build_benchmark_table.py`, `evals/runners/build_parity_table.py`

Both do glob→load-json→compute-delta→render-markdown. Extract a shared `render_markdown_table(rows, columns)` into `lc_parity_common.py` (the existing shared lib), each builder passing its own column schema. **Only do this if the two render bodies are genuinely identical modulo columns** — if they diverge (different delta math, different sort), Guardrail #3 (no abstraction without a real second use) says leave them. Inspect first; skip if not a clean fit.

- [ ] **Step 1:** Diff the two render functions; decide fit. If not a clean fit, **skip this task entirely** and note why in the commit-less record.
- [ ] **Step 2 (if fit):** Extract `render_markdown_table` into `lc_parity_common.py`; both builders call it.
- [ ] **Step 3:** Run `tests/evals/test_build_benchmark_table.py tests/evals/test_build_parity_table.py`; assert unchanged output.
- [ ] **Step 4:** Commit `refactor(evals): share markdown table render across builders`.

---

## Section 4: Evals directory reorg — `scripts/` + `data/`

> Largest mechanical section. It renames a package dir and changes a shell interface. Land as ONE commit so the rename + all reference updates stay atomic (a half-applied rename breaks every `from runners.X` test import).

### Task 4.1: Rename `evals/runners/` → `evals/scripts/`, fold in entry points

**Files:**
- Rename dir: `evals/runners/` → `evals/scripts/`
- Move into it: `evals/eval_gt.py`, `evals/eval_compare.py`, `evals/eval_multiview_conf.py`, `evals/eval_similarity_calibration.py`, `evals/eval_suite.sh`
- Keep at `evals/` root (library, imported not run): `datasets.py`, `metrics.py`, `trajectory_io.py`
- Modify: `evals/scripts/eval_suite.sh` (path arithmetic), `CLAUDE.md` (2 refs), all `tests/evals/*.py` with `sys.path.insert(..., "evals")` / `from runners.X` / `from ...evals` refs

- [ ] **Step 1: Rename with git mv (preserves history)**

```bash
git mv evals/runners evals/scripts
for f in eval_gt.py eval_compare.py eval_multiview_conf.py eval_similarity_calibration.py eval_suite.sh; do
  git mv evals/$f evals/scripts/$f
done
```

- [ ] **Step 2: Fix `eval_gt.py`'s self-locating import**

`eval_gt.py:39` does `sys.path.insert(0, str(Path(__file__).resolve().parent))` then `from datasets import get_dataset` / `from runners.lc_parity_common import ...` / `from trajectory_io import read_tum`. Now that `eval_gt.py` lives in `scripts/`, `datasets.py`/`trajectory_io.py` are one dir UP. Change the inserted path to `Path(__file__).resolve().parent.parent` (= `evals/`), and `from runners.lc_parity_common` → `from lc_parity_common` (now a sibling in `scripts/`). Apply the same parent-dir fix to `eval_compare.py:38` (`from metrics import ...`), `eval_multiview_conf.py:33`, `eval_similarity_calibration.py:50` — each currently inserts `evals/` via `parent`/`parent.parent`; recompute so `datasets`/`metrics`/`trajectory_io` resolve to `evals/` root and sibling-script imports resolve to `evals/scripts/`.

- [ ] **Step 3: Fix `eval_suite.sh` path arithmetic**

`evals/scripts/eval_suite.sh:6`: `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` went up one from `evals/`; from `evals/scripts/` it must go up two: `.../../..` → change `/..` to `/../..`. Then every `"${EVALS_DIR}/eval_gt.py"` / `eval_compare.py` reference must become `"${EVALS_DIR}/scripts/eval_gt.py"` etc., and `"${EVALS_DIR}/runners/run_vggt_slam.py"` → `"${EVALS_DIR}/scripts/run_vggt_slam.py"`. (`EVALS_DIR="${REPO_ROOT}/evals"` stays correct.)

- [ ] **Step 4: Fix all test imports**

Every `tests/evals/*.py` that does `sys.path.insert(0, .../"evals")` + `from runners.X import ...` must become `sys.path.insert(0, .../"evals"/"scripts")` + `from X import ...` (X now a top-level module in `scripts/`). Files touched (from grep): `test_build_benchmark_table.py`, `test_cross_model_runner.py`, `test_lc_decisions.py`, `test_lc_parity_common.py`, `test_build_parity_table.py`, `test_compare_loop_edges.py`, `test_lc_loop_pr.py`, `test_run_lc_parity.py`, `test_run_vggt_slam.py`, `test_visualize_lc_correction.py`. The `from datasets`/`from metrics`/`from trajectory_io` tests (`test_datasets.py`, `test_metrics.py`, `test_metrics_auc.py`, `test_eval_compare.py`, `test_eval_gt_helpers.py`, `test_trajectory_io.py`, `test_loop_ablation_json.py`, `test_download_7scenes.py`) keep inserting `evals/` root — **verify each still resolves** (some insert `evals/` and import a `runners.` submodule; those need the `scripts/` path added).

> Do this with a scripted sweep, then eyeball: `grep -rn "runners\.\|/ \"runners\"\|\"runners\"" tests/evals/` must return zero after.

- [ ] **Step 5: Fix `CLAUDE.md`**

Two literal refs (`CLAUDE.md:98`, `CLAUDE.md:110`): `evals/eval_gt.py` → `evals/scripts/eval_gt.py`.

- [ ] **Step 6: Check internal cross-refs among moved scripts**

`run_lc_parity.py:37` `EVAL_GT = RUNNERS.parent / "eval_gt.py"` — after the move `eval_gt.py` is a *sibling* in `scripts/`, so it becomes `EVAL_GT = RUNNERS / "eval_gt.py"` (where `RUNNERS = Path(__file__).resolve().parent`). `run_cross_model_benchmark.py:15` `EVAL_GT = Path(__file__).resolve().parents[1] / "eval_gt.py"` → `parents[0] / "eval_gt.py"`. Grep for `eval_gt.py"` and `parents[` across `evals/scripts/` and fix each to the new sibling layout. `run_vggt_slam.py:20` `parents[2]` (→ repo root) is unchanged (still two up from `evals/scripts/`... **verify**: `scripts/` adds a level, so `parents[2]` from `evals/scripts/run_vggt_slam.py` = repo root still holds only if it was `parents[2]` from `evals/runners/` = repo root — same depth, OK). Recheck every `parents[N]` in moved files: `evals/runners/` and `evals/scripts/` are the same depth, so `parents[N]` values that pointed at repo root or `third_party/` are **unchanged**; only the *sibling* `eval_gt.py` refs (which used `parent`/`parents[1]` to hop from `runners/`→`evals/`) change.

- [ ] **Step 7: Run the full evals test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/ -q`
Expected: PASS (same count as baseline).

- [ ] **Step 8: Smoke the shell driver's path math (no GPU)**

Run: `bash -n evals/scripts/eval_suite.sh && grep -n 'scripts/eval_gt.py\|/../..' evals/scripts/eval_suite.sh`
Expected: parses clean; both edits present.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor(evals): rename runners/ → scripts/, fold in eval_* entry points"
```

### Task 4.2: Consolidate downloaders into `evals/data/download.py`

**Files:**
- Create: `evals/data/download.py` (5 dataset functions + CLI)
- Move: `evals/scripts/extract_waymo.py` → `evals/data/extract_waymo.py` (it was `runners/extract_waymo.py`; after 4.1 it is in `scripts/`)
- Delete: `evals/download_7scenes.sh`, `evals/download_co3dv2.sh`, `evals/download_tum.sh`, `evals/download_waymo.sh`, `evals/download_kitti.sh`, `evals/data/download_parity_scenes.sh`
- Keep: `evals/download_7scenes.py` — **decide** (see Step 1)
- Modify: `.gitignore` (the `!evals/data/...` exception)
- Modify: `evals/README.md` (Section 4 rewrite, Task 4.3) — parity-scene list note

- [ ] **Step 1: Decide `download_7scenes.py`'s fate**

Spec §4 says consolidate the *5 remaining* downloaders into `download.py` with one function per dataset (`download_7scenes()`, `download_co3dv2()`, `download_tum()`, guide-only `download_waymo()`/`download_kitti()`). `download_7scenes.py` (the Python one) is the substantive 7-Scenes implementation; `download_7scenes.sh` is an exact-duplicate shell version (delete). Port `download_7scenes.py`'s body into `download.py`'s `download_7scenes()` function, then delete the standalone `download_7scenes.py` too — unless `tests/evals/test_download_7scenes.py` pins its module API. **Check that test first:** if it imports `download_7scenes` as a module, either keep a thin re-export or update the test to import from `evals.data.download`. Do NOT break `test_download_7scenes.py`.

- [ ] **Step 2: Write `evals/data/download.py`**

One function per dataset — **port the existing bodies, do not reinvent** (URLs, extraction quirks like the 7-Scenes STORED-zip `exit 1` tolerance, TUM `.tgz` layout). Add a subcommand CLI:

```python
"""Dataset downloaders for the eval harness — one CLI, one function per dataset.

Ported verbatim from the former per-dataset scripts (download_7scenes.py/.sh,
download_{co3dv2,tum,waymo,kitti}.sh). waymo/kitti are license-gated: guide +
layout-verify only, no actual fetch.

    python evals/data/download.py 7scenes --scenes fire office
    python evals/data/download.py tum --seq desk
    python evals/data/download.py waymo <segment_id> <dest>   # prints guide, verifies layout
"""
```
Each function keeps the source script's default output dir and messages. Provide `argparse` subparsers (`7scenes`, `co3dv2`, `tum`, `waymo`, `kitti`).

- [ ] **Step 3: Move `extract_waymo.py` into `evals/data/`**

```bash
git mv evals/scripts/extract_waymo.py evals/data/extract_waymo.py
```
Update any `parents[N]` in it (moving `scripts/`→`data/` is same depth as `runners/`→`data/`? No: `data/` and `scripts/` are both direct children of `evals/`, so depth is unchanged — verify its repo-root `parents[N]` still resolves). Update `datasets.py:126`'s docstring reference `evals/runners/extract_waymo.py` → `evals/data/extract_waymo.py`, and the same string in `download_waymo.sh`'s ported body / README.

- [ ] **Step 4: Delete the old shell downloaders + parity-scenes script**

```bash
git rm evals/download_7scenes.sh evals/download_co3dv2.sh evals/download_tum.sh \
       evals/download_waymo.sh evals/download_kitti.sh \
       evals/data/download_parity_scenes.sh
# and, per Step 1's decision, possibly:
git rm evals/download_7scenes.py
```

- [ ] **Step 5: Fix `.gitignore` (SAME commit — mandatory)**

`.gitignore:27-29`:
```
!evals/data/
evals/data/*
!evals/data/download_parity_scenes.sh
```
The deleted `download_parity_scenes.sh` exception must be replaced so the new tracked files survive the blanket `evals/data/*` ignore:
```
!evals/data/
evals/data/*
!evals/data/download.py
!evals/data/extract_waymo.py
```
(Without this, `download.py`/`extract_waymo.py` are silently untracked.)

- [ ] **Step 6: Verify the new files are actually trackable**

Run: `git check-ignore -v evals/data/download.py evals/data/extract_waymo.py`
Expected: **no output** (not ignored). If either prints a rule, the `.gitignore` exception is wrong — fix before commit.

- [ ] **Step 7: Update README parity-scene note + run download tests**

In `evals/README.md`, replace the `download_parity_scenes.sh` reference with a one-line note: the LC-parity suite needs 7-Scenes `fire/heads/office/pumpkin/redkitchen/stairs` (seq-01) + the 4 named TUM freiburg sequences. Then:
Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_download_7scenes.py -q`
Expected: PASS (per Step-1 decision).

- [ ] **Step 8: Commit**

```bash
black evals/data/download.py && isort evals/data/download.py
git add -A
git commit -m "refactor(evals): consolidate 5 downloaders into data/download.py; move extract_waymo"
```

### Task 4.3: Rewrite `evals/README.md` layout/tables

**Files:** `evals/README.md`

- [ ] **Step 1:** Update the `## Layout` block (`runners/`→`scripts/`, note `data/download.py`, drop `envs/`/`notebooks/`), the Entry-points/Library tables (paths now `evals/scripts/*`), and the download-scripts list (single `data/download.py`). Keep the `evals`-package-shadow note and the `sys.path.insert(.../"evals"/"scripts")` guidance current.
- [ ] **Step 2:** `grep -n "runners/\|envs/\|notebooks/\|download_.*\.sh\|ate_utils\|reconstruction_quality" evals/README.md` → reconcile every stale hit.
- [ ] **Step 3:** Commit `docs(evals): update README for scripts/ + data/download reorg`.

### Task 4.4: Section-4 checkpoint

- [ ] Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -5` — baseline counts.

---

## Section 5: Loop-closure dead-code removal

### Task 5.1: Inline-and-delete `_assemble_precorrection_extrinsics`

**Files:** `collab_splats/geometry/loop_closure/wrapper.py`

Single caller (`wrapper.py:385`), pure passthrough to `dedup_overlap`.

- [ ] **Step 1:** In `wrapper.py`, replace line 385:
```python
self.base._lc_precorrection_extrinsics = _assemble_precorrection_extrinsics(submaps, N)
```
with the inlined body:
```python
# Stitch raw per-submap poses into (N, 4, 4) using the same first-writer-wins
# overlap dedup as the corrected path — no PGO correction applied.
self.base._lc_precorrection_extrinsics = dedup_overlap(
    submap_ids=[s.submap_id for s in submaps],
    submap_starts=[s.frame_start for s in submaps],
    corrected={s.submap_id: s.poses for s in submaps},
    total_frames=N,
)
```
Then delete the `_assemble_precorrection_extrinsics` function (`wrapper.py:54-65`).
- [ ] **Step 2:** `grep -rn "_assemble_precorrection_extrinsics" collab_splats/ tests/` → zero hits.
- [ ] **Step 3:** Run `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q` — PASS.
- [ ] **Step 4:** Commit `refactor(geometry): inline single-use _assemble_precorrection_extrinsics`.

### Task 5.2: Delete `normalize_to_sl4`

**Files:**
- Modify: `collab_splats/geometry/loop_closure/graph.py` (delete `normalize_to_sl4`, lines 71-80)
- Modify: `collab_splats/geometry/loop_closure/__init__.py` (drop from import + `__all__`)
- Modify: `tests/geometry/loop_closure/test_graph.py`, `tests/pointcloud/test_pose_extraction.py` (test-only callers)

Zero production callers; dead upstream too. Its only users are tests asserting det-normalization.

- [ ] **Step 1: Neutralize the test callers.** In `test_graph.py`, tests at lines 14-21 (`normalize_to_sl4(H)`, singular-raises) and lines 95-126 build `normalize_to_sl4(...)` inputs before feeding `PoseGraph`. Since `PoseGraph.add_node` already SL4-normalizes internally (`gtsam.SL4(H)`), replace `normalize_to_sl4(X)` calls with the raw `X` matrices (the graph normalizes on insert). Delete the two dedicated `normalize_to_sl4` unit tests (the `H_norm = normalize_to_sl4(H)` det-check and the singular-raise) — the behavior they pin is being removed. In `test_pose_extraction.py:154` (`H_sl4 = normalize_to_sl4(H)`), inline the one-liner it needs: `H_sl4 = H / (abs(np.linalg.det(H)) ** 0.25)` if the test genuinely needs a det-1 matrix, else pass `H` raw.
- [ ] **Step 2: Delete the function** (`graph.py:71-80`) and remove `normalize_to_sl4` from `collab_splats/geometry/loop_closure/__init__.py` (the `.graph` import block **and** `__all__`).
- [ ] **Step 3:** `grep -rn "normalize_to_sl4" collab_splats/ tests/` → zero hits.
- [ ] **Step 4:** Run `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_graph.py tests/pointcloud/test_pose_extraction.py -q` — PASS.
- [ ] **Step 5:** Commit `refactor(geometry): remove dead normalize_to_sl4 (graph normalizes on insert)`.

### Task 5.3: Delete `manifold="se3"` branch

**Files:**
- Modify: `collab_splats/geometry/loop_closure/graph.py` (`PoseGraph.__init__`, `add_node`, `add_prior`, `add_sequential_edge`, `get_homography` — drop the `else` SE(3) branches; drop `_pose3`)
- Modify: `collab_splats/geometry/loop_closure/closure.py` → (after Section 6, `graph.py`) `run_pose_graph_optimization`'s `manifold` param
- Modify: `LoopClosureConfig` (`closure.py:175`) — remove `manifold` field
- Modify: `tests/geometry/loop_closure/test_graph.py` (delete `manifold="se3"` tests at 136, 195)

Zero production callers: `wrapper.py`'s `run_pose_graph_optimization(...)` never passes `manifold=`, and `LoopClosureConfig` never plumbs one. Only `PoseGraph(manifold="se3")` in tests reaches it.

> **Ordering note:** this touches `run_pose_graph_optimization` and `LoopClosureConfig`, both of which MOVE in Section 6. To avoid editing the same lines twice, do **either** 5.3-before-6 on the current `closure.py` **or** fold 5.3 into Section 6 as you write the new `graph.py`/`wrapper.py`. Recommended: do 5.3 on `closure.py`/`graph.py` as they stand now (smaller diff to reason about), accept that Section 6 then moves the already-simplified functions. Both orders end identically.

- [ ] **Step 1: Delete SE(3) tests.** In `test_graph.py`, remove the `PoseGraph(manifold="se3")` test (line ~136) and the `run_pose_graph_optimization(..., manifold="se3")` case (line ~195). Confirm no other test passes `manifold=` or `se3` sigmas: `grep -rn 'manifold\|se3' tests/geometry/` (the `scale_method="se3"` hits are unrelated — leave them).
- [ ] **Step 2: Collapse `PoseGraph` to SL(4)-only.** In `graph.py`: drop the `manifold` param from `__init__` (hardcode SL4 noise models), delete the `else:` SE(3) branches in `add_node`/`add_prior`/`add_sequential_edge`/`get_homography`, delete `_pose3` (only the SE(3) branches used it — confirm via grep), and drop `from typing import Literal` if now unused.
- [ ] **Step 3: Drop `manifold` from the optimizer + config.** Remove `manifold` param from `run_pose_graph_optimization` (and its internal `_SL4PoseGraph(manifold=...)` call → `_SL4PoseGraph()`); remove the `manifold: Literal["sl4","se3"]` field from `LoopClosureConfig`. Grep for any `manifold=` caller: only tests (already handled) and the internal call.
- [ ] **Step 4:** `grep -rn 'manifold' collab_splats/geometry/loop_closure/` → only comments referencing the historical choice, no live param.
- [ ] **Step 5:** Run `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q` — PASS.
- [ ] **Step 6:** Commit `refactor(geometry): drop dead manifold="se3" PoseGraph branch`.

### Task 5.4: Section-5 checkpoint

- [ ] Run `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -5` — baseline minus the deleted dead-branch tests.

---

## Section 6: `closure.py` split

> Mirrors VGGT-SLAM's `loop_closure.py` / `graph.py` / `map.py` seams. `closure.py` is fully dissolved — **no shim/re-export** left behind — so every `.closure` importer must be repointed in this same commit or the suite breaks at import time.

**Destination map (spec §6):**
- `matching.py` (new): `LoopMatch`, `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check`, `_MIN_CONF_POINTS`/`_RNG` if only matching uses them (else keep with their user).
- `graph.py` (existing, extended): `run_pose_graph_optimization`, `_estimate_scale_pairwise_dist`, `umeyama_se3`, `umeyama_sim3`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`.
- `merge.py` (new): `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs`.
- `LoopClosureConfig` → `wrapper.py`.

### Task 6.1: Create `matching.py`

**Files:** Create `collab_splats/geometry/loop_closure/matching.py`

- [ ] **Step 1:** Move `LoopMatch` (dataclass), `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check` from `closure.py` into a new `matching.py`. Carry the imports they need (`heapq`, `numpy`, `torch`, `dataclass`, `Submap` from `.submap`). Add the module docstring: `"""Loop-candidate retrieval + gating (mirrors VGGT-SLAM loop_closure.py)."""`.
- [ ] **Step 2:** Determine where `_estimate_scale_pairwise_dist`, `_MIN_CONF_POINTS`, `_RNG` belong — they are used by `_lc_anchor_scale`/`run_pose_graph_optimization` (→ `graph.py`), not by matching. Put them in `graph.py`.

### Task 6.2: Extend `graph.py`

**Files:** `collab_splats/geometry/loop_closure/graph.py`

- [ ] **Step 1:** Move into `graph.py` (which already holds `PoseGraph`, `decompose_camera`, `estimate_scale_pairwise`): `umeyama_se3`, `umeyama_sim3`, `_estimate_scale_pairwise_dist`, `_MIN_CONF_POINTS`, `_RNG`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`, `run_pose_graph_optimization`. `run_pose_graph_optimization` currently constructs `_SL4PoseGraph` (an alias for `PoseGraph`) — now that they share a module, call `PoseGraph(...)` directly and drop the alias. It also imports `dedup_overlap` and `_resolve_frame_node` — those go to `merge.py` (Task 6.3), so `graph.py` will `from .merge import dedup_overlap, _resolve_frame_node`.
- [ ] **Step 2: Watch for a `graph.py`↔`merge.py` import cycle.** `graph.run_pose_graph_optimization` needs `dedup_overlap`+`_resolve_frame_node` (merge), and `merge.merge_submap_outputs` takes a `PoseGraph` only as a duck-typed arg (no import of graph needed). So the edge is one-directional `graph → merge` — safe. Confirm `merge.py` does not import `graph.py`.
- [ ] **Step 3:** Preserve the `log = logging.getLogger(__name__)` — note the logger name for `run_pose_graph_optimization` becomes `...loop_closure.graph` (Task 6.5 updates the one test asserting on it).

### Task 6.3: Create `merge.py`

**Files:** Create `collab_splats/geometry/loop_closure/merge.py`

- [ ] **Step 1:** Move `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs` from `closure.py` into `merge.py`. Docstring: `"""Submap-output dedup + merge (mirrors VGGT-SLAM map.py)."""`. Imports: `numpy`, `Submap` from `.submap`. `merge_submap_outputs`'s `graph` param stays a forward-ref/duck-typed `"PoseGraph | None"` — no hard import of `graph.py`.

### Task 6.4: Move `LoopClosureConfig` to `wrapper.py`; delete `closure.py`; repoint imports

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py` (define `LoopClosureConfig`)
- Delete: `collab_splats/geometry/loop_closure/closure.py`
- Modify: `collab_splats/geometry/loop_closure/__init__.py`, `collab_splats/geometry/__init__.py`, `collab_splats/geometry/loop_closure/eval.py`

- [ ] **Step 1: Define `LoopClosureConfig` in `wrapper.py`.** Move the dataclass (with the Section-5.3-simplified fields: no `manifold`) into `wrapper.py`, above the `LoopClosure` class. `wrapper.py`'s `from .closure import (...)` block changes to pull from the new modules:
```python
from .matching import find_loop_closures, translation_jump_check
from .merge import dedup_overlap, merge_submap_outputs
from .graph import run_pose_graph_optimization
```
(`LoopClosureConfig` is now local; `LoopMatch`/`LoopMatchQueue` are not used directly by `wrapper.py` — confirm via grep.)

- [ ] **Step 2: Delete `closure.py`.**
```bash
git rm collab_splats/geometry/loop_closure/closure.py
```

- [ ] **Step 3: Rewire `eval.py`** (spec §6 enumeration missed this — package-internal): `collab_splats/geometry/loop_closure/eval.py:14` `from .closure import umeyama_se3, umeyama_sim3` → `from .graph import umeyama_se3, umeyama_sim3`.

- [ ] **Step 4: Rewrite `loop_closure/__init__.py`.** Repoint the eager imports to the new modules and add the lazy `LoopClosureConfig` hook (it now lives in `wrapper.py`, which is the lazy module):
```python
from .matching import LoopMatch, LoopMatchQueue, find_loop_closures, translation_jump_check
from .merge import dedup_overlap, merge_submap_outputs
from .graph import (
    PoseGraph, decompose_camera, estimate_scale_pairwise, run_pose_graph_optimization,
)
from .eval import capture_pose_graph_loss
from .submap import Submap, assert_world_to_cam


def __getattr__(name):
    # Lazy — wrapper.py imports collab_splats.pointcloud, which cycles back through
    # geometry.transforms; eager import here would deadlock at load time.
    if name == "LoopClosure":
        from .wrapper import LoopClosure
        return LoopClosure
    if name == "LoopClosureConfig":
        from .wrapper import LoopClosureConfig
        return LoopClosureConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```
Keep `LoopClosureConfig` in `__all__` (coexists with `__getattr__`, same precedent as `LoopClosure`). Drop `normalize_to_sl4` (deleted in 5.2) from `__all__`.

- [ ] **Step 5: Rewrite `geometry/__init__.py`.** Change `from .loop_closure import LoopClosureConfig, PoseGraph, Submap` → `from .loop_closure import PoseGraph, Submap`, and add the lazy `LoopClosureConfig` branch to its `__getattr__`:
```python
    if name == "LoopClosureConfig":
        from .loop_closure import LoopClosureConfig
        return LoopClosureConfig
```
Keep `"LoopClosureConfig"` in `__all__`.

- [ ] **Step 6: Import-smoke both packages** (catches cycles before pytest):
```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry as g; print(g.LoopClosureConfig, g.LoopClosure, g.PoseGraph)"
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.loop_closure as lc; print(lc.LoopClosureConfig, lc.find_loop_closures, lc.merge_submap_outputs)"
```
Expected: prints classes/functions, no `ImportError`/recursion.

### Task 6.5: Repoint all `.closure` test + tool importers

**Files:** the enumerated importers (spec §6) + `eval.py` (done in 6.4).

Apply per destination:
- [ ] `tests/.../test_loop_closure.py:156` `from ...closure import LoopMatchQueue` → `from ...matching import LoopMatchQueue`.
- [ ] `tests/.../test_translation_jump.py:8` `translation_jump_check` → `.matching`.
- [ ] `tests/.../test_alignment_dedup.py:3` `dedup_overlap` → `.merge`.
- [ ] `tests/.../test_loop_closure_eval.py` `merge_submap_outputs` → `.merge`.
- [ ] `tests/.../test_closure_split.py:3` `run_pose_graph_optimization` → `.graph`.
- [ ] `tests/.../test_hw_formula.py:12` `run_pose_graph_optimization` → `.graph`.
- [ ] `tests/.../test_loop_ablation.py:7` `run_pose_graph_optimization` → `.graph`.
- [ ] `tests/.../test_graph.py:155` collapse the second `from .closure import run_pose_graph_optimization` into the existing `.graph` import at the top.
- [ ] `tests/.../test_loop_edge_chain.py:22` (`_lc_anchor_scale`, `_loop_chain_relatives`, `run_pose_graph_optimization`) → `.graph`; **and line ~188** `from ...closure import _SL4PoseGraph` → `from ...graph import PoseGraph as _SL4PoseGraph` (alias deleted from source; re-alias locally in the test); **and the logger-name assertion around line 355** (`caplog.at_level(..., logger="...loop_closure.closure")` or equivalent) → `"...loop_closure.graph"`.
- [ ] `tests/.../test_pgo_parity.py:16` `LoopClosureConfig` → `from collab_splats.geometry.loop_closure import LoopClosureConfig` (package root, via new lazy hook); its `run_pose_graph_optimization` → `.graph`.
- [ ] `evals/scripts/visualize_lc_correction.py:23` `umeyama_sim3` → `.graph` (path is `scripts/` post-Section-4).
- [ ] **Mock-patch targets** in `tests/.../test_feedforward_lc_state.py:54-55` (and the third at :55+): `mock.patch("...loop_closure.closure.find_loop_closures")` etc. resolve the path via `import_module` and break at patch time once `closure.py` is gone. `wrapper.py` imports these by *name*, so patch `wrapper.<name>` (the pattern the file's other patches already use): `"...loop_closure.wrapper.find_loop_closures"`, `"...wrapper.run_pose_graph_optimization"`, `"...wrapper.merge_submap_outputs"`.
- [ ] **Left alone (out of scope, deferred):** `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb` still imports from `.closure`; per CLAUDE.md it's deferred. Do NOT edit.

- [ ] **Final grep gate:** `grep -rn "loop_closure.closure\|from .closure\|from \.\.closure" collab_splats/ tests/ evals/` → **zero hits** (the notebook is the only allowed remaining ref, and it's not matched by these paths).

### Task 6.6: Section-6 checkpoint

- [ ] Run `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -5` — baseline counts.
- [ ] `black collab_splats/geometry/loop_closure/ && isort collab_splats/geometry/loop_closure/`
- [ ] Commit:
```bash
git add -A
git commit -m "refactor(geometry): split closure.py into matching/graph/merge; LoopClosureConfig→wrapper"
```

---

## Section 7: `LoopClosure` wrapper renames

> Split `_run_lc_loop`'s per-window body to mirror VGGT-SLAM's `Solver.run_predictions`/`add_points`. The outer sweep loop stays a private `LoopClosure` method — NOT exposed (there is exactly one calling convention, the `BaseFeedforwardCreator` template method). `run_inference`/`load_model`/`setup_inference`/`postprocess`/`build_colmap`/`reconstruct`/`run`/`reproject` are unchanged.

**Files:** `collab_splats/geometry/loop_closure/wrapper.py`; re-check `tests/geometry/loop_closure/test_wrapper.py`.

### Task 7.1: Extract `run_predictions`

- [ ] **Step 1: Add `run_predictions`.** Factor the per-window body of `_run_lc_loop` (forward pass → build `Submap` → retrieval query → per-candidate verify + jump-check) into:
```python
def run_predictions(self, window, submaps, lc_submaps, **kwargs):
    """Forward-pass a window, build its Submap, detect+verify loop candidates.

    Direct analog of VGGT-SLAM Solver.run_predictions. Returns
    (submap, lc_submaps_new, loop_matches) where lc_submaps_new is the list
    (0 or more — max_loops_per_submap defaults to 5) of verified LC submaps
    from THIS window, and loop_matches is every candidate considered this
    window (accepted or rejected) so the caller can extend all_loop_candidates
    and derive the loops_found/verified counters from match.accepted.
    """
```
The body is lines ~234-372 of the current `_run_lc_loop` (from `end = min(...)` through building `submap`, `find_loop_closures`, the `for match in loop_matches:` verify/jump loop that appends to a *local* `lc_submaps_new` list). It must return `(submap, lc_submaps_new, loop_matches)`.

> **Correction (spec §7):** `run_predictions` returns `lc_submaps: list[Submap]` (0+), NOT a singular `lc_submap_or_None` — multiple accepted matches per window is the designed common case. It also returns `loop_matches: list` so the caller drives the pbar counters.

- [ ] **Step 2:** Keep the LC-submap id assignment consistent. The current code uses `submap_id=len(submaps) + len(lc_submaps)` when appending an LC submap — inside `run_predictions`, pass the running `lc_submaps` (all prior) plus the local count so ids stay globally unique. Verify the id formula still yields the same sequence as before the split (write it to match the pre-split values exactly).

### Task 7.2: Extract `add_points`

- [ ] **Step 1: Add `add_points`.**
```python
def add_points(self, submap, lc_submaps_new, submaps, lc_submaps):
    """Append the window's submap and extend the running LC-submap list.

    Bookkeeping only — pose-graph optimization is deferred to the batch
    run_pose_graph_optimization after the sweep (unlike Solver.add_points which
    wires edges incrementally). Name matches the role, not the implementation.
    """
    submaps.append(submap)
    lc_submaps.extend(lc_submaps_new)   # extend, not append — 0+ per window
```

- [ ] **Step 2: Rewrite the outer `_run_lc_loop` sweep** to call the two:
```python
for wi, start in enumerate(range(0, N, step)):
    submap, lc_new, loop_matches = self.run_predictions(views, ... , submaps, lc_submaps, **kwargs)
    self.add_points(submap, lc_new, submaps, lc_submaps)
    all_loop_candidates.extend(loop_matches)
    verified += sum(m.accepted for m in loop_matches)
    loops_found = verified
    pbar.update(1); pbar.set_postfix(loops=loops_found, verified=verified)
    end = min(start + K + O, N)
    if end >= N:
        break
```
The `end`/`break` termination and `pbar` stay in the outer sweep (sweep-termination/progress concerns, not per-window bookkeeping). Pass `end` out of `run_predictions` in the tuple if recomputing it in the outer loop is awkward — either is fine; recomputing `min(start+K+O, N)` is cheap.

> **Behavior must be identical.** The post-sweep block (setting `_lc_submaps`, `_lc_loop_submaps`, `_lc_precorrection_extrinsics` (Section-5.1 inlined form), `run_pose_graph_optimization`, `merge_submap_outputs`, `_ablate_loops`) is unchanged — the split is purely a rename of the per-window body.

### Task 7.3: Verify + commit

- [ ] **Step 1: Re-check the one direct test.** `tests/geometry/loop_closure/test_wrapper.py::test_lc_loop_passes_k_plus_overlap_to_forward` (line ~380) calls `wrapper._run_lc_loop()` directly and patches `BaseRetrievalExtractor` at `wrapper.py:414`. Its patch targets resolve through whichever module now hosts `find_loop_closures` (`matching.py`, post-Section-6) — the test patches `...wrapper.BaseRetrievalExtractor`, which is unaffected. Run it; if the window-slicing assertion still holds (it should — `run_predictions` does the same `views[start:end]` slice), no change needed. If the split changed the call signature the test reaches, update the test to the new internal structure while preserving its assertion.
- [ ] **Step 2:** Run `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_wrapper.py -q` — PASS.
- [ ] **Step 3:** Full LC suite: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q` — PASS.
- [ ] **Step 4:** `black collab_splats/geometry/loop_closure/wrapper.py && isort collab_splats/geometry/loop_closure/wrapper.py`
- [ ] **Step 5:** Commit `refactor(geometry): split _run_lc_loop into run_predictions/add_points (VGGT-SLAM naming)`.

---

## Final Verification

- [ ] **Full suite at baseline.** `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -10` — pass count equals Task 0 baseline minus exactly the tests deleted alongside dead code (`test_ate_utils_tum.py` cases migrated, `normalize_to_sl4`/`manifold="se3"` unit tests removed). Cross-check any residual failures against `docs/known-test-failures.md`.
- [ ] **No dangling references.** `grep -rn "ate_utils\|reconstruction_quality\|loop_closure.closure\|_assemble_precorrection\|normalize_to_sl4\|evals/runners\|evals\.runners" collab_splats/ evals/ tests/ CLAUDE.md` — only the deferred `slam_loop_closure.ipynb` may remain (and only for `.closure`).
- [ ] **Behavior-change ledger.** The only intended behavior deltas are: deleted dead branches (`normalize_to_sl4`, `manifold="se3"`) + their tests, and the `ate_utils`→`metrics.compute_ate` evo path (numerically identical — one fewer entry point, guarded by Task 2.3 Step 4). Everything else reproduces exactly.
- [ ] **Update memory.** Add a `project_` memory noting: closure.py dissolved into matching/graph/merge; `ate_utils`/`reconstruction_quality` gone; `evals/runners`→`evals/scripts`; downloaders → `evals/data/download.py`. Link `[[project_geometry_package]]`.

---

## Self-Review

**Spec coverage:** §1→Task 1.1 (+1.2 noted non-blocking); §2→2.1/2.2/2.3; §3→3.1/3.2 (optional, per §4 correction the sweep-driver dedup is out of scope); §4→4.1/4.2/4.3; §5→5.1/5.2/5.3 (`scale_method="none"` correctly preserved); §6→6.1-6.5 (+`eval.py` importer the spec missed); §7→7.1/7.2/7.3. All covered.

**Placeholder scan:** no "TBD"/"add appropriate X"; every code-editing step shows the code or the exact edit. The two `(optional)` Section-3 tasks are explicitly gated on an inspection step that can conclude "skip."

**Type/name consistency:** `run_predictions` returns `(submap, lc_submaps_new: list, loop_matches: list)` consistently (7.1↔7.2); `LoopClosureConfig` lazy-hook added to BOTH `__init__` files (6.4 Step 4 + Step 5); `dedup_overlap`/`merge_submap_outputs`→`.merge`, `run_pose_graph_optimization`/`umeyama_*`→`.graph`, `find_loop_closures`/`translation_jump_check`→`.matching` applied identically in source (6.1-6.4) and every test importer (6.5).

**Ordering hazards flagged:** 5.3↔6 (same functions move), 3.1↔6 (`umeyama_sim3` import path), 2.3-before-4 (`run_vggt_slam_lc.py` edited pre-move) — each has an inline note.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-19-evals-loop-closure-cleanup.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute in this session with checkpoints per section.

Which approach?
