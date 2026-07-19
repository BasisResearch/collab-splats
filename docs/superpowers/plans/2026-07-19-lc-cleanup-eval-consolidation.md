# LC Cleanup & Eval Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the `LoopClosure` wrapper's scripting path, remove confirmed dead code and stale comments from `collab_splats/geometry/loop_closure/`, add missing inline block comments per repo convention, and shrink `evals/` to a minimal reproducible set (delete one-off/absorbed investigation scripts, generalize the 3 scripts with ongoing regen value via CLI flags).

**Architecture:** No new abstractions. Section 1 adds documentation only. Section 2 is in-place edits to 6 existing files (remove dead constants, dedupe a section header, fix two stale comments, collapse one duplicate function, add missing block comments) — zero behavior change, verified by the existing test suite passing unchanged. Section 3 deletes 12 files and adds argparse flags to 3 files (defaults unchanged, so unparameterized invocations are unaffected), then rewrites `evals/README.md` to match.

**Tech Stack:** Python 3.11, pytest, GTSAM (SL4 pose graph), NumPy. No new dependencies.

**Guardrail:** every step must leave `/opt/venv/reconstruction/bin/python -m pytest tests/` passing with the exact same pass/fail set as the baseline (Task 1). If any step would require changing a test to keep it green, stop — that step is out of scope for this plan.

---

## File Structure

**Modify:**
- `collab_splats/geometry/loop_closure/wrapper.py` — add quickstart example to module docstring
- `docs/source/api/geometry.rst` — add quickstart snippet
- `collab_splats/geometry/loop_closure/closure.py` — remove dead sigma constants + stale comment; rename duplicate section header; add 2 missing block comments
- `collab_splats/geometry/loop_closure/eval.py` — rewrite stale docstring/comment; move `.closure` imports to top; delete `_umeyama_sim3`, rewire its one call site to `closure.umeyama_sim3`; add missing block comments
- `evals/runners/run_disparity_sweep.py` — `--backbone` flag; strip stale "Track A/B fix" debug language
- `evals/runners/run_cross_model_benchmark.py` — `--backbones`/`--framesets`/`--keyframe_lists`/`--submap_sizes` flags replacing `core_matrix()`
- `evals/runners/build_benchmark_table.py` — `--reference_backbone` flag
- `evals/README.md` — drop rows for deleted files, update usage examples, collapse Investigations A/B

**Delete (`git rm`):**
- `evals/runners/parity_trace.py`
- `evals/runners/our_solver_dump.py`
- `evals/runners/vggt_slam_solver_dump.py`
- `evals/runners/compare_solver_internals.py`
- `evals/runners/compare_vggt_outputs.py`
- `evals/diag_pose_graph.py`
- `evals/check_ate_methods.py`
- `evals/_ba_finding_eval.py`
- `evals/sweep_incremental_ba.py`
- `evals/plot_incremental_ba_sweep.py`
- `evals/incremental_ba_sweep.png`
- `evals/incremental_ba_sweep_200.png`

**Untouched:** `collab_splats/geometry/loop_closure/graph.py` (already exemplary comments, verified in Task 11), `collab_splats/geometry/loop_closure/submap.py` gets 1 small comment (Task 12) but its `TODO(spec-2)` stays, `collab_splats/geometry/loop_closure/__init__.py` (already minimal/self-explanatory), `evals/eval_gt.py`, `evals/eval_compare.py`, `evals/eval_suite.sh`, `evals/runners/run_lc_parity.py`, `evals/runners/lc_parity_common.py`, `evals/runners/run_vggt_slam.py`, `evals/runners/run_vggt_slam_lc.py`, `evals/runners/compare_loop_edges.py`, `evals/eval_similarity_calibration.py`, `evals/eval_multiview_conf.py`, `evals/baselines/`.

---

## Task 1: Baseline test suite

- [ ] **Step 1: Run the full suite and save the result as the baseline**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -30`

Expected: same pass/fail/xfail counts as the last known-green run (see `docs/known-test-failures.md` for pre-existing failures — note these, they must remain the *same* failures after every later task, not newly-introduced ones).

No commit for this step (no file changes).

---

## Task 2: Wrapper quickstart docstring (Section 1)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py:1-7`

- [ ] **Step 1: Add the quickstart example to the module docstring**

Current text (lines 1-7):
```python
"""LoopClosure wrapper around pointcloud creators.

This module deliberately depends on pointcloud result types (PointcloudResult,
FeedforwardResult, _raw_to_world_points) because it wraps feedforward creators.
That reverse dependency (geometry -> pointcloud) is why LoopClosure is lazily
exported via __getattr__ from the geometry and geometry.loop_closure __init__s.
"""
```

Replace with:
```python
"""LoopClosure wrapper around pointcloud creators.

Quickstart:
    from collab_splats.pointcloud import get_creator
    from collab_splats.geometry.loop_closure import LoopClosure, LoopClosureConfig

    base = get_creator("vggt_omega")()
    lc = LoopClosure(base, config=LoopClosureConfig())
    result = lc.reconstruct(image_dir="path/to/images", output_dir="path/to/out")

See docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb for the full
walkthrough (candidate matching, plots).

This module deliberately depends on pointcloud result types (PointcloudResult,
FeedforwardResult, _raw_to_world_points) because it wraps feedforward creators.
That reverse dependency (geometry -> pointcloud) is why LoopClosure is lazily
exported via __getattr__ from the geometry and geometry.loop_closure __init__s.
"""
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `/opt/venv/reconstruction/bin/python -c "from collab_splats.geometry.loop_closure import LoopClosure"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/geometry/loop_closure/wrapper.py
git commit -m "docs(geometry): add LoopClosure quickstart to wrapper docstring"
```

---

## Task 3: Quickstart snippet in geometry.rst (Section 1)

**Files:**
- Modify: `docs/source/api/geometry.rst:28-32`

- [ ] **Step 1: Add a Quickstart subsection before the wrapper automodule directive**

Current text (lines 28-32, end of file):
```rst
.. automodule:: collab_splats.geometry.loop_closure.wrapper
   :members:
   :show-inheritance:
```

Replace with:
```rst
Quickstart
----------

.. code-block:: python

   from collab_splats.pointcloud import get_creator
   from collab_splats.geometry.loop_closure import LoopClosure, LoopClosureConfig

   base = get_creator("vggt_omega")()
   lc = LoopClosure(base, config=LoopClosureConfig())
   result = lc.reconstruct(image_dir="path/to/images", output_dir="path/to/out")

See ``docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`` for the full
walkthrough (candidate matching, plots).

.. automodule:: collab_splats.geometry.loop_closure.wrapper
   :members:
   :show-inheritance:
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/source/api/geometry.rst
git commit -m "docs(geometry): add LoopClosure quickstart to API reference"
```

---

## Task 4: Remove dead sigma constants in closure.py (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/closure.py:140-147`

**Confirmed dead** (verified this session): zero usages of `_ODOM_INTRA_SIGMA_R`, `_ODOM_INTRA_SIGMA_T`, `_INTER_SIGMA_R`, `_INTER_SIGMA_T` anywhere in `collab_splats/` or `tests/` (grep for exact names returns only the definition lines). `eval.py::_classify_edges` classifies edges via `isinstance(nm, gtsam.noiseModel.Robust)`, not these constants — the comment claiming otherwise is also false. `graph.py::PoseGraph.__init__` hardcodes its own sigmas independently.

- [ ] **Step 1: Delete the constants and the stale comment**

Current text:
```python
########################################
####### Absorbed from pose_graph.py ####
########################################

# SE(3) noise constants — used by eval.py edge classification
_ODOM_INTRA_SIGMA_R, _ODOM_INTRA_SIGMA_T = 0.02, 0.05
_INTER_SIGMA_R, _INTER_SIGMA_T = 0.05, 0.20

########################################
####### Absorbed from retrieval.py #####
########################################
```

Replace with:
```python
########################################
####### Absorbed from pose_graph.py ####
########################################

########################################
####### Absorbed from retrieval.py #####
########################################
```

- [ ] **Step 2: Confirm no other reference exists**

Run: `grep -rn "_ODOM_INTRA_SIGMA\|_INTER_SIGMA" collab_splats/ tests/ evals/`
Expected: no matches.

- [ ] **Step 3: Run the loop_closure test subset**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q 2>&1 | tail -15`
Expected: same result as Task 1 baseline.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/geometry/loop_closure/closure.py
git commit -m "refactor(geometry): remove unused LC noise-sigma constants

Zero usages anywhere; graph.py's PoseGraph hardcodes its own sigmas.
The removed comment claiming eval.py used these was also false —
eval.py classifies edges via GTSAM noise-model type, not these values."
```

---

## Task 5: Fix duplicate section header in closure.py (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/closure.py` (the block starting at what is now, after Task 4's deletion, ~5 lines earlier than the original line 260 — locate by content, not line number)

The header `######## Absorbed from alignment.py #####` appears twice: once (correctly) above `umeyama_se3`/`umeyama_sim3` (the actual alignment/SVD math), and again above `dedup_overlap`/`translation_jump_check` — which are pose-merge and loop-sanity utilities, not alignment math. The second occurrence is a mislabeled copy, not real alignment.py content.

- [ ] **Step 1: Rename the second occurrence to describe what it actually precedes**

Current text:
```python
########################################
####### Absorbed from alignment.py #####
########################################


def dedup_overlap(  # noqa: F811 — shadows import; canonical copy lives here
```

Replace with:
```python
########################################
####### Pose merging & loop-jump utilities #####
########################################


def dedup_overlap(  # noqa: F811 — shadows import; canonical copy lives here
```

- [ ] **Step 2: Confirm exactly one "Absorbed from alignment.py" header remains**

Run: `grep -n "Absorbed from alignment.py" collab_splats/geometry/loop_closure/closure.py`
Expected: exactly 1 match (the one above `umeyama_se3`).

- [ ] **Step 3: Commit**

```bash
git add collab_splats/geometry/loop_closure/closure.py
git commit -m "docs(geometry): fix mislabeled duplicate section header in closure.py

dedup_overlap/translation_jump_check are merge/jump-check utilities,
not alignment.py content — the header was a mislabeled copy of the
umeyama_se3/umeyama_sim3 section above."
```

---

## Task 6: Add missing block comments in closure.py (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/closure.py` (`find_loop_closures`, `_loop_chain_relatives`)

Full-file audit (done as part of writing this plan) found the file already densely commented per repo convention (`_lc_anchor_scale`, `run_pose_graph_optimization`, `merge_submap_outputs`, `translation_jump_check` all already have block comments). Two functions have a docstring but no comment on their body logic:

- [ ] **Step 1: Add a comment to `find_loop_closures`'s nested-loop body**

Current text:
```python
    queue = LoopMatchQueue(max_size=max_loops, nms_frame_distance=nms_frame_distance)
    for q_idx in range(query_submap.retrieval_vectors.shape[0]):
        q_vec = query_submap.retrieval_vectors[q_idx]
        for past in past_submaps:
            dists = torch.cdist(q_vec.unsqueeze(0), past.retrieval_vectors).squeeze(0)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])
            if best_dist < lc_threshold:
```

Replace with:
```python
    queue = LoopMatchQueue(max_size=max_loops, nms_frame_distance=nms_frame_distance)
    # For each query frame, find the nearest-neighbor frame in each past submap by
    # L2 distance over DINO-SALAD retrieval vectors; keep it if under lc_threshold.
    for q_idx in range(query_submap.retrieval_vectors.shape[0]):
        q_vec = query_submap.retrieval_vectors[q_idx]
        for past in past_submaps:
            dists = torch.cdist(q_vec.unsqueeze(0), past.retrieval_vectors).squeeze(0)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])
            if best_dist < lc_threshold:
```

- [ ] **Step 2: Add a comment to `_loop_chain_relatives`'s matrix-chain body**

Current text:
```python
    eye = np.eye(4, dtype=np.float64)
    K_q = eye if K_q is None else K_q
    K_lc0 = eye if K_lc0 is None else K_lc0
    K_lc1 = eye if K_lc1 is None else K_lc1
    K_d = eye if K_d is None else K_d
    H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])
    H_inner = P_lc0.astype(np.float64) @ np.linalg.inv(P_lc1.astype(np.float64))
    H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])
    return H_rel_a, H_inner, H_rel_b
```

Replace with:
```python
    eye = np.eye(4, dtype=np.float64)
    K_q = eye if K_q is None else K_q
    K_lc0 = eye if K_lc0 is None else K_lc0
    K_lc1 = eye if K_lc1 is None else K_lc1
    K_d = eye if K_d is None else K_d
    # Anchor A: query → LC-frame-0 (identical image) — K change + scale fold s_a.
    H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])
    # Inner edge: the LC pair's own relative pose, no K change.
    H_inner = P_lc0.astype(np.float64) @ np.linalg.inv(P_lc1.astype(np.float64))
    # Anchor B: LC-frame-1 → detected (identical image) — K change + scale fold s_b.
    H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])
    return H_rel_a, H_inner, H_rel_b
```

- [ ] **Step 3: Run the loop_closure test subset**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q 2>&1 | tail -15`
Expected: same result as Task 1 baseline (comment-only change).

- [ ] **Step 4: Commit**

```bash
git add collab_splats/geometry/loop_closure/closure.py
git commit -m "docs(geometry): add block comments to find_loop_closures, _loop_chain_relatives"
```

---

## Task 7: Fix stale eval.py docstring and stub comment (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/eval.py:1-6, 106`

**Confirmed stale** (verified this session): `ate_translation`, `auc_at_threshold`, `umeyama_align` are fully implemented and live — imported by `evals/eval_gt.py` and exercised by `tests/geometry/loop_closure/test_auc_metric.py`, `tests/geometry/loop_closure/test_eval_metrics.py`, `tests/geometry/loop_closure/test_loop_closure_eval.py`, `tests/evals/test_eval_gt_helpers.py`, `tests/evals/test_loop_ablation_json.py`, `tests/evals/test_metrics_auc.py`.

- [ ] **Step 1: Rewrite the module docstring**

Current text:
```python
"""Loop-closure pose-graph evaluation helpers.

Provides instrumentation for the LC pose graph (per-iteration loss capture,
per-edge-type residual breakdown) and stubs for future ground-truth-based
metrics (ATE, RPE) that will be implemented when a GT pose dataset is sourced.
"""
```

Replace with:
```python
"""Loop-closure pose-graph evaluation helpers.

Provides instrumentation for the LC pose graph (per-iteration loss capture,
per-edge-type residual breakdown) plus ground-truth-based trajectory metrics
(ATE, RPE, AUC) used by evals/eval_gt.py and exercised in tests/evals/ and
tests/geometry/loop_closure/.
"""
```

- [ ] **Step 2: Fix the stale section-header comment**

Current text:
```python
# --- GT-phase stubs (implement when GT dataset arrives) ----------------------
```

Replace with:
```python
# --- Ground-truth trajectory metrics -----------------------------------------
```

- [ ] **Step 3: Commit**

```bash
git add collab_splats/geometry/loop_closure/eval.py
git commit -m "docs(geometry): fix stale eval.py docstring — GT metrics are implemented, not stubs"
```

---

## Task 8: Collapse duplicate umeyama_sim3 in eval.py (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/eval.py` (imports, `_umeyama_sim3` deletion, `auc_at_threshold` call site)

**Verified this session (numerically, not just by inspection):** `eval.py`'s private `_umeyama_sim3` (operates on transposed `(3, N)` arrays, computes scale via the Umeyama SVD-trace formula) and `closure.py`'s canonical `umeyama_sim3` (operates on `(M, 3)` arrays, computes scale via the variance-ratio formula) are **not bit-identical** — on noisy synthetic data they differ by ~0.03% in scale and ~2.5e-4 in translation. They agree exactly only in the noiseless case. This is safe to collapse because every test that touches `auc_at_threshold`/`ate_translation` output uses tolerance-based assertions (`pytest.approx(..., abs=1.0)`, `>= 99.0`, `< 10.0`, etc.) — none pins an exact value, so a ~0.03%-level formula difference does not change any test outcome. `_umeyama_sim3` has exactly one call site: `auc_at_threshold` (verified via grep — no test imports `_umeyama_sim3` directly).

- [ ] **Step 1: Move the `.closure` imports to the top of the file (repo convention: no inline imports inside functions except for optional heavy deps)**

Current top-of-file imports:
```python
from __future__ import annotations

import gtsam
import numpy as np

from .graph import PoseGraph
```

Replace with:
```python
from __future__ import annotations

import gtsam
import numpy as np

from .closure import umeyama_se3, umeyama_sim3
from .graph import PoseGraph
```

- [ ] **Step 2: Remove the now-redundant lazy import inside `umeyama_align`**

Current text:
```python
    from .closure import umeyama_se3

    R_pred = pred[:, :3, :3]
```

Replace with:
```python
    R_pred = pred[:, :3, :3]
```

- [ ] **Step 3: Remove the now-redundant lazy import inside `ate_translation`**

Current text:
```python
    # Align predicted positions to GT via Sim3 (scale + rotation + translation).
    # Must use Sim3 (not SE3) to match evo's correct_scale=True — VGGT depth predictions
    # carry an unknown global scale factor that SE3 alignment cannot remove.
    from .closure import umeyama_sim3

    s, R_align, t_align = umeyama_sim3(source=p_pred, target=p_gt)
```

Replace with:
```python
    # Align predicted positions to GT via Sim3 (scale + rotation + translation).
    # Must use Sim3 (not SE3) to match evo's correct_scale=True — VGGT depth predictions
    # carry an unknown global scale factor that SE3 alignment cannot remove.
    s, R_align, t_align = umeyama_sim3(source=p_pred, target=p_gt)
```

- [ ] **Step 4: Delete the private `_umeyama_sim3` function**

Current text:
```python
def _umeyama_sim3(source: np.ndarray, target: np.ndarray):
    """Sim3: c, R, t such that c * R @ source + t ≈ target. source/target: (3, N)."""
    mu_s = source.mean(axis=1, keepdims=True)
    mu_t = target.mean(axis=1, keepdims=True)
    var_s = np.square(source - mu_s).sum(axis=0).mean()
    cov = ((target - mu_t) @ (source - mu_s).T) / source.shape[1]
    U, D, VH = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2, 2] = -1
    c = float(np.trace(np.diag(D) @ S) / var_s)
    R = U @ S @ VH
    t = mu_t - c * R @ mu_s  # (3, 1)
    return c, R, t


def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

Replace with:
```python
def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

- [ ] **Step 5: Rewire the one call site in `auc_at_threshold`**

`closure.umeyama_sim3` takes `(M, 3)` arrays directly (no transpose needed — `centers_pred`/`centers_gt` are already `(N, 3)`) and returns `t` already flat as `(3,)` (no `.flatten()` needed), unlike the deleted private version.

Current text:
```python
    # Sim3 alignment: c * R_a @ centers_pred.T + t_a ≈ centers_gt.T
    c, R_a, t_a = _umeyama_sim3(centers_pred.T, centers_gt.T)
    t_a = t_a.flatten()
```

Replace with:
```python
    # Sim3 alignment: c * R_a @ centers_pred + t_a ≈ centers_gt
    c, R_a, t_a = umeyama_sim3(source=centers_pred, target=centers_gt)
```

- [ ] **Step 6: Confirm no remaining reference to the deleted function**

Run: `grep -rn "_umeyama_sim3" collab_splats/ tests/ evals/`
Expected: no matches.

- [ ] **Step 7: Run the eval-adjacent test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ tests/evals/ -q 2>&1 | tail -20`
Expected: same pass/fail set as Task 1 baseline.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/geometry/loop_closure/eval.py
git commit -m "refactor(geometry): collapse eval.py's private _umeyama_sim3 into closure.umeyama_sim3

One call site (auc_at_threshold). Verified numerically the two formulas
diverge only at the ~0.03%/noise level (variance-ratio vs SVD-trace scale) —
no test pins an exact value, all use tolerance-based assertions, so this
is behavior-preserving under the project's test suite."
```

---

## Task 9: Add missing block comments in eval.py (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/eval.py` (`capture_pose_graph_loss`, `rpe`)

Full-file audit found `_classify_edges`, `ate_translation`, `auc_at_threshold` already commented adequately. Two functions have a docstring but no body comments:

- [ ] **Step 1: Add a comment to `capture_pose_graph_loss`'s manual LM loop**

Current text:
```python
    optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph._graph, pose_graph._initial, params)

    iterations: list[float] = [float(optimizer.error())]
    prev_err = iterations[0]
    converged = False
    for _ in range(max_iterations):
        optimizer.iterate()
        err = float(optimizer.error())
        iterations.append(err)
        if prev_err > 0 and abs(prev_err - err) / max(prev_err, 1e-12) < relative_error_tol:
            converged = True
            break
        prev_err = err

    optimized = optimizer.values()
    groups = _classify_edges(pose_graph._graph)
```

Replace with:
```python
    optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph._graph, pose_graph._initial, params)

    # Manual iterate() loop (not optimizer.optimize()) so we can capture the cost
    # after every step; the outer max_iterations is the sole step-count authority
    # (GTSAM's own internal max-iteration param is intentionally left unset).
    iterations: list[float] = [float(optimizer.error())]
    prev_err = iterations[0]
    converged = False
    for _ in range(max_iterations):
        optimizer.iterate()
        err = float(optimizer.error())
        iterations.append(err)
        if prev_err > 0 and abs(prev_err - err) / max(prev_err, 1e-12) < relative_error_tol:
            converged = True
            break
        prev_err = err

    # Per-edge-type residual breakdown, before and after optimization.
    optimized = optimizer.values()
    groups = _classify_edges(pose_graph._graph)
```

- [ ] **Step 2: Add a comment to `rpe`'s relative-pose computation**

Current text:
```python
    if delta >= len(pred):
        raise ValueError(f"delta={delta} >= N={len(pred)}, no pose pairs available")
    rel_pred = np.linalg.inv(pred[:-delta]) @ pred[delta:]  # (N-δ, 4, 4)
    rel_gt = np.linalg.inv(gt[:-delta]) @ gt[delta:]  # (N-δ, 4, 4)
    err = np.linalg.inv(rel_gt) @ rel_pred  # (N-δ, 4, 4)

    t_err = np.linalg.norm(err[:, :3, 3], axis=1)
    cos_angle = np.clip((np.trace(err[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    r_err_deg = np.degrees(np.arccos(cos_angle))
```

Replace with:
```python
    if delta >= len(pred):
        raise ValueError(f"delta={delta} >= N={len(pred)}, no pose pairs available")
    # Relative pose between frame i and i+delta, for pred and gt independently,
    # then the error transform between the two relative poses.
    rel_pred = np.linalg.inv(pred[:-delta]) @ pred[delta:]  # (N-δ, 4, 4)
    rel_gt = np.linalg.inv(gt[:-delta]) @ gt[delta:]  # (N-δ, 4, 4)
    err = np.linalg.inv(rel_gt) @ rel_pred  # (N-δ, 4, 4)

    # Translation error is the error transform's norm; rotation error is its
    # geodesic angle via the standard trace formula.
    t_err = np.linalg.norm(err[:, :3, 3], axis=1)
    cos_angle = np.clip((np.trace(err[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    r_err_deg = np.degrees(np.arccos(cos_angle))
```

- [ ] **Step 3: Run the eval-adjacent test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ tests/evals/ -q 2>&1 | tail -20`
Expected: same result as Task 1 baseline.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/geometry/loop_closure/eval.py
git commit -m "docs(geometry): add block comments to capture_pose_graph_loss, rpe"
```

---

## Task 10: Add one comment in submap.py (Section 2)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/submap.py:44-46`

Audit found `submap.py` already has per-field comments on the dataclass and docstrings on every method. One block — the dehomogenize step in `get_world_points` — has no comment. The existing `TODO(spec-2)` comment (line 7) stays as-is (out of scope, tied to a different spec).

- [ ] **Step 1: Add a comment to the dehomogenize step**

Current text:
```python
        out_hom = (H @ hom.T).T  # (K*P, 4)
        w = out_hom[:, 3:4]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        return (out_hom[:, :3] / w).reshape(k, p, 3).astype(np.float32)
```

Replace with:
```python
        out_hom = (H @ hom.T).T  # (K*P, 4)
        # Dehomogenize by /w, guarding against near-zero w (projective transform
        # can push points toward the plane at infinity).
        w = out_hom[:, 3:4]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        return (out_hom[:, :3] / w).reshape(k, p, 3).astype(np.float32)
```

- [ ] **Step 2: Run the loop_closure test subset**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q 2>&1 | tail -15`
Expected: same result as Task 1 baseline.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/geometry/loop_closure/submap.py
git commit -m "docs(geometry): add dehomogenize comment to Submap.get_world_points"
```

---

## Task 11: Verify graph.py and __init__.py need no changes (Section 2)

**Files:** none modified — this is a documented verification step, not a code change.

- [ ] **Step 1: Confirm graph.py's comment coverage**

`graph.py` was re-read in full during this plan's preparation: every function (`decompose_camera`, `normalize_to_sl4`, `estimate_scale_pairwise`, `_pose3`, `PoseGraph.__init__`, `add_node`, `add_prior`, `add_sequential_edge`, `optimize`, `get_homography`) already has a docstring, and the non-trivial ones (`decompose_camera`, `PoseGraph.__init__`) already carry detailed inline comments (including a documented historical bug — the R vs Rᵀ convention that caused the original ATE gap). Section dividers (`# ---- node management ----`, `# ---- edges ----`, `# ---- optimization ----`) are already present. No changes needed.

`collab_splats/geometry/loop_closure/__init__.py` is a 48-line re-export module; its one piece of non-obvious logic (the lazy `__getattr__` for `LoopClosure`) already has a comment explaining the circular-import reason. No changes needed.

No commit for this step (no file changes).

---

## Task 12: Full suite checkpoint after Section 2

- [ ] **Step 1: Run the complete suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -30`
Expected: identical pass/fail/xfail counts to the Task 1 baseline. If anything differs, stop and diagnose before proceeding to Section 3 — Section 2 was scoped as pure removal + one call-site swap, so any new failure means something in this plan's understanding was wrong.

No commit for this step (no file changes; verification only).

---

## Task 13: Pre-deletion import/citation check (Section 3)

**Files:** none modified — this is a verification step gating Task 14.

- [ ] **Step 1: Grep for any import of the 10 non-binary delete candidates**

Run:
```bash
grep -rn "parity_trace\|our_solver_dump\|vggt_slam_solver_dump\|compare_solver_internals\|compare_vggt_outputs\|diag_pose_graph\|check_ate_methods\|_ba_finding_eval\|sweep_incremental_ba\|plot_incremental_ba_sweep" \
  --include="*.py" tests/ evals/ collab_splats/ | grep -v "^evals/runners/parity_trace.py:\|^evals/runners/our_solver_dump.py:\|^evals/runners/vggt_slam_solver_dump.py:\|^evals/runners/compare_solver_internals.py:\|^evals/runners/compare_vggt_outputs.py:\|^evals/diag_pose_graph.py:\|^evals/check_ate_methods.py:\|^evals/_ba_finding_eval.py:\|^evals/sweep_incremental_ba.py:\|^evals/plot_incremental_ba_sweep.py:"
```

Expected: no matches (i.e., no *other* file imports or references these modules by name). If a match appears, stop — that file is not safe to delete without also fixing the importer, which is out of scope for this plan; remove it from the delete list and flag it in the PR description instead.

- [ ] **Step 2: Grep for citations of the two PNGs**

Run: `grep -rln "incremental_ba_sweep" docs/superpowers/specs/ docs/ evals/README.md`
Expected: only `evals/README.md` (which Task 17 rewrites) — no spec doc embeds or links the images.

No commit for this step (no file changes; verification only).

---

## Task 14: Delete the 12 dead eval files (Section 3)

**Files:**
- Delete: `evals/runners/parity_trace.py`, `evals/runners/our_solver_dump.py`, `evals/runners/vggt_slam_solver_dump.py`, `evals/runners/compare_solver_internals.py`, `evals/runners/compare_vggt_outputs.py`, `evals/diag_pose_graph.py`, `evals/check_ate_methods.py`, `evals/_ba_finding_eval.py`, `evals/sweep_incremental_ba.py`, `evals/plot_incremental_ba_sweep.py`, `evals/incremental_ba_sweep.png`, `evals/incremental_ba_sweep_200.png`

- [ ] **Step 1: git rm all 12 files**

```bash
git rm evals/runners/parity_trace.py \
       evals/runners/our_solver_dump.py \
       evals/runners/vggt_slam_solver_dump.py \
       evals/runners/compare_solver_internals.py \
       evals/runners/compare_vggt_outputs.py \
       evals/diag_pose_graph.py \
       evals/check_ate_methods.py \
       evals/_ba_finding_eval.py \
       evals/sweep_incremental_ba.py \
       evals/plot_incremental_ba_sweep.py \
       evals/incremental_ba_sweep.png \
       evals/incremental_ba_sweep_200.png
```

- [ ] **Step 2: Run the full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -30`
Expected: identical pass/fail/xfail counts to the Task 1 baseline (Task 13 already confirmed nothing imports these).

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(evals): delete 12 investigation scripts whose findings are already recorded

LC↔SLAM parity-trace bug hunt (fixed, commit 1372ac2, findings in
2026-05-31-vggt-spark-stage-parity-findings.md) and incremental-BA tuning
(absorbed into eval_gt.py's ba/ba_track-density-N conditions) are done;
these scripts have no ongoing regen value. See evals/README.md (Task 17)
for the historical pointer that replaces their entries."
```

---

## Task 15: Generalize run_disparity_sweep.py (Section 3)

**Files:**
- Modify: `evals/runners/run_disparity_sweep.py`

- [ ] **Step 1: Add a `--backbone` flag and thread it through `run_our_pipeline`**

Current text:
```python
def run_our_pipeline(
    seq_dir: Path, keyframe_list: Path, condition: str, out_ate: Path
) -> dict:
    """Run our vggt_spark pipeline on the given keyframe list. Returns ATE dict."""
    out_ate.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _PYTHON, _EVAL_GT,
        "--dataset", "7scenes",
        "--seq_dir", seq_dir,
        "--backbone", "vggt_spark",
        "--conditions", condition,
        "--submap_size", 16,
        "--lc_scale_method", "none",
        "--keyframe_list", keyframe_list,
        "--output_ate", out_ate,
    ])
    return json.loads(out_ate.read_text())
```

Replace with:
```python
def run_our_pipeline(
    seq_dir: Path, keyframe_list: Path, condition: str, out_ate: Path, backbone: str = "vggt_spark"
) -> dict:
    """Run our pipeline on the given keyframe list. Returns ATE dict."""
    out_ate.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _PYTHON, _EVAL_GT,
        "--dataset", "7scenes",
        "--seq_dir", seq_dir,
        "--backbone", backbone,
        "--conditions", condition,
        "--submap_size", 16,
        "--lc_scale_method", "none",
        "--keyframe_list", keyframe_list,
        "--output_ate", out_ate,
    ])
    return json.loads(out_ate.read_text())
```

- [ ] **Step 2: Thread `backbone` through `sweep()`'s two call sites**

Current text:
```python
def sweep(seq_dir: Path, max_frames: int, start_disparity: int) -> int:
    """Run full disparity sweep. Returns 0 on full parity, 1 on first failure."""
```

Replace with:
```python
def sweep(seq_dir: Path, max_frames: int, start_disparity: int, backbone: str = "vggt_spark") -> int:
    """Run full disparity sweep. Returns 0 on full parity, 1 on first failure."""
```

Current text:
```python
        our_dir.mkdir(parents=True, exist_ok=True)
        our_baseline = run_our_pipeline(seq_dir, kf_list, "baseline", our_dir / "baseline_ate.json")
```

Replace with:
```python
        our_dir.mkdir(parents=True, exist_ok=True)
        our_baseline = run_our_pipeline(seq_dir, kf_list, "baseline", our_dir / "baseline_ate.json", backbone=backbone)
```

Current text:
```python
        slam_lc = run_slam_lc(seq_dir, d, max_frames, slam_dir)
        our_lc = run_our_pipeline(seq_dir, kf_list, "lc", our_dir / "lc_ate.json")
```

Replace with:
```python
        slam_lc = run_slam_lc(seq_dir, d, max_frames, slam_dir)
        our_lc = run_our_pipeline(seq_dir, kf_list, "lc", our_dir / "lc_ate.json", backbone=backbone)
```

- [ ] **Step 3: Add the `--backbone` CLI flag and pass it through in `main()`**

Current text:
```python
    parser.add_argument(
        "--start_disparity", type=int, default=50,
        help=(
            "Start sweep from this disparity level (inclusive). "
            "Levels are [50, 30, 20, 10, 0]; pass e.g. 30 to skip d=50. Default 50."
        ),
    )
    args = parser.parse_args()
    sys.exit(sweep(args.seq_dir, args.max_frames, args.start_disparity))
```

Replace with:
```python
    parser.add_argument(
        "--start_disparity", type=int, default=50,
        help=(
            "Start sweep from this disparity level (inclusive). "
            "Levels are [50, 30, 20, 10, 0]; pass e.g. 30 to skip d=50. Default 50."
        ),
    )
    parser.add_argument(
        "--backbone", type=str, default="vggt_spark",
        help="Feedforward backbone to compare against VGGT-SLAM. Default vggt_spark (the parity anchor).",
    )
    args = parser.parse_args()
    sys.exit(sweep(args.seq_dir, args.max_frames, args.start_disparity, backbone=args.backbone))
```

- [ ] **Step 4: Strip the stale "Track A/B fix" debug language from `_check_similarity_parity`**

Current text:
```python
    print(f"  VGGT-SLAM loop_closures: {slam_loops}")
    print("  → Check INFO log above for 'VGGT-SPARK image_match_ratio' lines from our LC run.")
    print("  → If our scores differ >0.05 from SLAM mean, Track A fix (Task 2) may not be applied.")
```

Replace with:
```python
    print(f"  VGGT-SLAM loop_closures: {slam_loops}")
    print("  → Check INFO log above for 'VGGT-SPARK image_match_ratio' lines from our LC run.")
```

- [ ] **Step 5: Strip the stale debug language from `sweep()`'s baseline-failure branch**

Current text:
```python
            print(f"  Keyframe list: {kf_list}")
            print("  → Both pipelines used identical frames — divergence is in inference/BA.")
            print("  → Enable Track B (H-matrix debug logging) to investigate.")
            return 1
```

Replace with:
```python
            print(f"  Keyframe list: {kf_list}")
            print("  → Both pipelines used identical frames — divergence is in inference/BA.")
            return 1
```

- [ ] **Step 6: Strip the stale debug language from `sweep()`'s LC-failure branch**

Current text:
```python
            print(f"  SLAM loop_closures={slam_loops}")
            if d == 50:
                print("  → Verify Track A fix (Task 2) is applied: VGGTSPARKCreator must use")
                print("    native compute_similarity=True path.")
                print("  → Check our INFO log for 'image_match_ratio' lines.")
            return 1
```

Replace with:
```python
            print(f"  SLAM loop_closures={slam_loops}")
            if d == 50:
                print("  → Check our INFO log for 'image_match_ratio' lines.")
            return 1
```

- [ ] **Step 7: Verify the script's help output and a dry parse**

Run: `/opt/venv/reconstruction/bin/python evals/runners/run_disparity_sweep.py --help`
Expected: shows `--seq_dir`, `--max_frames`, `--start_disparity`, `--backbone` (default `vggt_spark`), exit 0.

- [ ] **Step 8: Commit**

```bash
git add evals/runners/run_disparity_sweep.py
git commit -m "refactor(evals): generalize run_disparity_sweep.py's backbone via --backbone flag

Default stays vggt_spark so an unparameterized invocation reproduces
today's exact behavior. Also strips stale 'Track A/B fix' debug language
from a bug that was fixed in commit 1372ac2."
```

---

## Task 16: Generalize run_cross_model_benchmark.py (Section 3)

**Files:**
- Modify: `evals/runners/run_cross_model_benchmark.py`

`core_matrix()` hardcodes 4 backbones and 2 fixed keyframe-dir names (`slam_d10`, `slam_d20`) plus an optional `slam_d5_long`. Replace with CLI flags whose defaults reproduce the exact current matrix.

- [ ] **Step 1: Replace `core_matrix()` with a flag-driven `build_matrix()`**

Current text:
```python
def core_matrix(kf_dir: Path) -> list[RunSpec]:
    """Core matrix: 4 backbones × framesets × conditions (spec §Execution step 2)."""
    backbones = ["vggt_spark", "vggtx", "vggt_omega", "mapanything"]
    d10_kf = kf_dir / "slam_d10" / "selected_frames.txt"
    d20_kf = kf_dir / "slam_d20" / "selected_frames.txt"
    specs: list[RunSpec] = []
    for b in backbones:
        # Goal 1 — windowing cost: single-pass vs windowed baseline on short sets.
        specs.append(RunSpec(b, "slam_d10_single", None, ("baseline",), d10_kf))
        specs.append(RunSpec(b, "slam_d20_single", None, ("baseline",), d20_kf))
        # Windowed baseline + lc on d10 (2 submaps).
        specs.append(RunSpec(b, "slam_d10", 16, ("baseline", "lc"), d10_kf))
        # Goal 2 — LC benefit on the long, loop-closing set (only if it exists).
        long_kf = kf_dir / "slam_d5_long" / "selected_frames.txt"
        if long_kf.exists():
            specs.append(RunSpec(b, "slam_d5_long", 16, ("baseline", "lc"), long_kf))
    return specs
```

Replace with:
```python
def build_matrix(
    kf_dir: Path,
    backbones: list[str],
    framesets: list[str],
    submap_size: int,
) -> list[RunSpec]:
    """Build the benchmark matrix: backbones × framesets × conditions.

    Each frameset name is a subdirectory of kf_dir containing selected_frames.txt
    (e.g. "slam_d10" -> kf_dir/slam_d10/selected_frames.txt). For each frameset,
    emits a single-pass baseline run plus a windowed baseline+lc run, skipping
    framesets whose keyframe file doesn't exist.
    """
    specs: list[RunSpec] = []
    for b in backbones:
        for fs in framesets:
            kf = kf_dir / fs / "selected_frames.txt"
            if not kf.exists():
                continue
            # Windowing cost: single-pass vs windowed baseline on this frameset.
            specs.append(RunSpec(b, f"{fs}_single", None, ("baseline",), kf))
            specs.append(RunSpec(b, fs, submap_size, ("baseline", "lc"), kf))
    return specs
```

- [ ] **Step 2: Replace `main()`'s argparse + call site**

Current text:
```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path,
                    default=Path("evals/data/7scenes/chess/chess/seq-01"))
    ap.add_argument("--kf_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    ap.add_argument("--out_root", type=Path,
                    default=Path("evals/baselines/cross_model"))
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without running.")
    args = ap.parse_args()

    specs = core_matrix(args.kf_dir)
    cmds = build_commands(specs, args.seq_dir, args.out_root)
```

Replace with:
```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", type=Path,
                    default=Path("evals/data/7scenes/chess/chess/seq-01"))
    ap.add_argument("--kf_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    ap.add_argument("--out_root", type=Path,
                    default=Path("evals/baselines/cross_model"))
    ap.add_argument("--backbones", nargs="+",
                    default=["vggt_spark", "vggtx", "vggt_omega", "mapanything"],
                    help="Backbones to benchmark. Default: the 2026-05-31 4-backbone matrix.")
    ap.add_argument("--framesets", nargs="+",
                    default=["slam_d10", "slam_d20", "slam_d5_long"],
                    help="Frameset subdirs under --kf_dir (each needs selected_frames.txt). "
                         "Missing ones are skipped. Default: the 2026-05-31 frameset list.")
    ap.add_argument("--submap_size", type=int, default=16,
                    help="Submap size for the windowed baseline+lc runs. Default 16.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without running.")
    args = ap.parse_args()

    specs = build_matrix(args.kf_dir, args.backbones, args.framesets, args.submap_size)
    cmds = build_commands(specs, args.seq_dir, args.out_root)
```

- [ ] **Step 3: Verify the new default matrix matches the old one exactly**

Run: `/opt/venv/reconstruction/bin/python evals/runners/run_cross_model_benchmark.py --dry_run --kf_dir evals/baselines/disparity_sweep 2>&1 | head -20`
Expected: same command count and same `<backbone>__<frameset>__<sm>` output-dir names as running the old `core_matrix()` would have produced (4 backbones × [`slam_d10_single`, `slam_d20_single`, `slam_d10`, `slam_d5_long` if present] — note the new default frameset list folds the old special-cased `d10_single`/`d20_single` naming into the generic `{fs}_single` pattern, so confirm the printed dir names read `vggt_spark__slam_d10_single__single` etc., matching the original naming exactly).

- [ ] **Step 4: Commit**

```bash
git add evals/runners/run_cross_model_benchmark.py
git commit -m "refactor(evals): generalize run_cross_model_benchmark.py's matrix via CLI flags

core_matrix() hardcoded 4 backbones + 2 fixed frameset dir names.
build_matrix() takes --backbones/--framesets/--submap_size, defaulting
to the exact 2026-05-31 matrix so an unparameterized invocation is
unchanged."
```

---

## Task 17: Generalize build_benchmark_table.py (Section 3)

**Files:**
- Modify: `evals/runners/build_benchmark_table.py`

- [ ] **Step 1: Add `--reference_backbone` flag**

Current text:
```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("evals/baselines/cross_model"))
    ap.add_argument("--sweep_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    args = ap.parse_args()
    rows = assemble_rows(_load_runs(args.root), _load_slam_ate(args.sweep_dir),
                         reference_backbone="vggt_spark")
    print(render_markdown(rows))
    return 0
```

Replace with:
```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("evals/baselines/cross_model"))
    ap.add_argument("--sweep_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    ap.add_argument("--reference_backbone", type=str, default="vggt_spark",
                    help="Backbone used as the Δ-vs-reference baseline in the table. Default vggt_spark.")
    args = ap.parse_args()
    rows = assemble_rows(_load_runs(args.root), _load_slam_ate(args.sweep_dir),
                         reference_backbone=args.reference_backbone)
    print(render_markdown(rows))
    return 0
```

- [ ] **Step 2: Verify help output**

Run: `/opt/venv/reconstruction/bin/python evals/runners/build_benchmark_table.py --help`
Expected: shows `--root`, `--sweep_dir`, `--reference_backbone` (default `vggt_spark`), exit 0.

- [ ] **Step 3: Commit**

```bash
git add evals/runners/build_benchmark_table.py
git commit -m "refactor(evals): expose build_benchmark_table.py's reference_backbone as a CLI flag

Default stays vggt_spark; table generation over existing baselines/cross_model
data is unchanged."
```

---

## Task 18: Rewrite evals/README.md (Section 3)

**Files:**
- Modify: `evals/README.md`

- [ ] **Step 1: Drop the "Diagnostic" runners row and the deleted-script rows from the standalone-tools table**

Current text:
```markdown
**Diagnostic (LC↔SLAM parity — see Investigations §A):**
`compare_vggt_outputs.py` · `compare_solver_internals.py`.

## Other eval tools (standalone)

| file | role | status |
|---|---|---|
| `eval_similarity_calibration.py` | Sweep LC verify layer per backbone (DINO-SALAD pairs). Produced the per-model `_lc_layer_index` calibration. | keep (re-runnable tool) |
| `eval_multiview_conf.py` | Multiview-confidence eval across backbones (chess). | keep |
| `diag_pose_graph.py` | Compare our SL(4) pose-graph init vs VGGT-SLAM. | retained — Investigations §A |
| `check_ate_methods.py` | One-off: evo ATE vs our umeyama on SLAM poses. | retained — Investigations §A |
| `_ba_finding_eval.py` · `sweep_incremental_ba.py` · `plot_incremental_ba_sweep.py` · `incremental_ba_sweep*.png` | Incremental-BA add_size sweep investigation (ATE vs runtime). | retained — Investigations §B |
```

Replace with:
```markdown
## Other eval tools (standalone)

| file | role | status |
|---|---|---|
| `eval_similarity_calibration.py` | Sweep LC verify layer per backbone (DINO-SALAD pairs). Produced the per-model `_lc_layer_index` calibration. | keep (re-runnable tool) |
| `eval_multiview_conf.py` | Multiview-confidence eval across backbones (chess). | keep |
```

- [ ] **Step 2: Drop the deleted-file rows from the `runners/` active table**

Current text:
```markdown
**Active:**
| file | role |
|---|---|
| `run_vggt_slam.py` | Subprocess wrapper around `third_party/VGGT-SLAM/main.py`. **Use its defaults for the published-matching anchor** (see handoff below). |
| `run_vggt_slam_lc.py` | Run VGGT-SLAM on a sequence → dense TUM + ATE + loop count. Produced the long SLAM ref / loop probe. |
| `run_disparity_sweep.py` | Disparity-sweep parity harness (ours vs SLAM); generated `baselines/disparity_sweep/`. |
| `parity_trace.py` | **Canonical** stage-by-stage parity trace (vggt_spark LC vs VGGT-SLAM). Localizes a diverging stage. Uses the solver-dump helpers below. |
| `our_solver_dump.py` · `vggt_slam_solver_dump.py` | Per-boundary solver-internals dumps consumed by `parity_trace.py` / `run_disparity_sweep.py`. |
| `run_cross_model_benchmark.py` | **(2026-05-31)** Serial `eval_gt` matrix over 4 backbones × framesets. |
| `build_benchmark_table.py` | **(2026-05-31)** Aggregate `cross_model/*/metrics.json` → markdown table. |
| `compare_loop_edges.py` | Loop-edge composition diff (ours vs SLAM). Kept: `compose_slam_chain` imported by `tests/geometry/loop_closure/test_loop_edge_chain.py`. |
```

Replace with:
```markdown
**Active:**
| file | role |
|---|---|
| `run_vggt_slam.py` | Subprocess wrapper around `third_party/VGGT-SLAM/main.py`. **Use its defaults for the published-matching anchor** (see handoff below). |
| `run_vggt_slam_lc.py` | Run VGGT-SLAM on a sequence → dense TUM + ATE + loop count. Produced the long SLAM ref / loop probe. |
| `run_disparity_sweep.py` | Disparity-sweep parity harness (ours vs SLAM) for any backbone (`--backbone`, default `vggt_spark`); generated `baselines/disparity_sweep/`. |
| `run_cross_model_benchmark.py` | Serial `eval_gt` matrix over any backbones/framesets (`--backbones`, `--framesets`, `--submap_size`); defaults reproduce the 2026-05-31 4-backbone matrix. |
| `build_benchmark_table.py` | Aggregate `cross_model/*/metrics.json` → markdown table (`--reference_backbone`, default `vggt_spark`). |
| `compare_loop_edges.py` | Loop-edge composition diff (ours vs SLAM). Kept: `compose_slam_chain` imported by `tests/geometry/loop_closure/test_loop_edge_chain.py`. |
```

- [ ] **Step 3: Collapse Investigations A and B to short historical pointers**

Current text:
```markdown
### A. LC ↔ VGGT-SLAM parity (pose-extraction fix, commit `1372ac2`)
Goal: find why our LC trajectory diverged 17× from VGGT-SLAM. Root cause = `R` vs `Rᵀ`
in pose extraction. **Outcome:** fixed; vggt_spark baseline now matches SLAM.
Trail: `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md`.

| script | what it probed |
|---|---|
| `runners/parity_trace.py` | **Canonical** stage-by-stage trace (preprocess→forward→trajectory→scale→homographies). The tool to reach for. |
| `runners/our_solver_dump.py` · `runners/vggt_slam_solver_dump.py` | Per-boundary solver-internals dumps — **consumed by `parity_trace.py`** (don't move independently). |
| `runners/compare_solver_internals.py` | Per-boundary solver diff (manual precursor to parity_trace). |
| `runners/compare_vggt_outputs.py` | VGGT extrinsics under our vs SLAM image preprocessing (confirmed Δ=0). |
| `diag_pose_graph.py` | Our SL(4) pose-graph init vs VGGT-SLAM's. |
| `check_ate_methods.py` | Verified `evo` ATE == our umeyama on SLAM's own poses (sanity, passed). |

### B. Bundle-adjustment tuning
Goal: pick BA track-density / increment params. **Outcome:** folded into `eval_gt` `ba`
and `ba_track-density-N` conditions; CO3Dv2 notes in `EVAL_NOTES.md`.

| script | what it probed |
|---|---|
| `_ba_finding_eval.py` | Ad-hoc A/B over vis_thresh, track budget, fine_tracking. |
| `sweep_incremental_ba.py` · `plot_incremental_ba_sweep.py` | Incremental-BA add_size vs ATE/runtime; plots → `incremental_ba_sweep*.png`. |
```

Replace with:
```markdown
### A. LC ↔ VGGT-SLAM parity (pose-extraction fix, commit `1372ac2`)
Goal: find why our LC trajectory diverged 17× from VGGT-SLAM. Root cause = `R` vs `Rᵀ`
in pose extraction. **Outcome:** fixed; vggt_spark baseline now matches SLAM. The
investigation scripts (`parity_trace.py` and its solver-dump/comparison helpers) are
deleted — bug fixed, no ongoing regen value. Trail:
`docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md`.

### B. Bundle-adjustment tuning
Goal: pick BA track-density / increment params. **Outcome:** folded into `eval_gt` `ba`
and `ba_track-density-N` conditions — this capability is now native to the main runner,
so the sweep scripts are deleted. CO3Dv2 notes in `EVAL_NOTES.md`.
```

- [ ] **Step 4: Update the "Housekeeping" note at the end of the Investigations section (now dangling since the PNGs are deleted)**

Current text:
```markdown
> **Housekeeping:** `incremental_ba_sweep*.png` are committed binaries under `evals/`.
> Left in place per keep-all; if regenerating, prefer writing plots to the gitignored
> `results/` rather than the source tree.
```

Replace with:
```markdown
> **Housekeeping:** if re-running the incremental-BA sweep, write plots to the
> gitignored `results/` rather than the source tree.
```

- [ ] **Step 5: Update Investigations D's script list (2 of its 4 rows changed name/behavior)**

Current text:
```markdown
### D. Cross-model benchmark (2026-05-31)
Goal: rank backbones; separate windowing cost from LC benefit. **Outcome:**
`docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md`.

| script | what it probed |
|---|---|
| `runners/run_cross_model_benchmark.py` | Serial `eval_gt` matrix (4 backbones × framesets). |
| `runners/build_benchmark_table.py` | Aggregate `cross_model/*/metrics.json` → markdown. |
| `runners/run_disparity_sweep.py` | ours-vs-SLAM at min_disparity 10–50 → `baselines/disparity_sweep/`. |
| `runners/run_vggt_slam.py` · `runners/run_vggt_slam_lc.py` | VGGT-SLAM wrappers (anchor + long ref / loop probe). |
```

Replace with:
```markdown
### D. Cross-model benchmark (2026-05-31)
Goal: rank backbones; separate windowing cost from LC benefit. **Outcome:**
`docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md`. The frozen
2026-05-31 numbers live under `baselines/cross_model/`; to reproduce or extend
them for a different backbone/frameset combination, use the generalized scripts:

| script | what it probed |
|---|---|
| `runners/run_cross_model_benchmark.py` | Serial `eval_gt` matrix — any `--backbones` × `--framesets` (defaults reproduce the original 4-backbone matrix). |
| `runners/build_benchmark_table.py` | Aggregate `cross_model/*/metrics.json` → markdown (`--reference_backbone`). |
| `runners/run_disparity_sweep.py` | ours-vs-SLAM at min_disparity 10–50 for any `--backbone` → `baselines/disparity_sweep/`. |
| `runners/run_vggt_slam.py` · `runners/run_vggt_slam_lc.py` | VGGT-SLAM wrappers (anchor + long ref / loop probe). |
```

- [ ] **Step 6: Commit**

```bash
git add evals/README.md
git commit -m "docs(evals): rewrite README for the 12-file deletion + 3-script generalization

Drops rows for deleted investigation scripts, updates usage for the
3 generalized scripts, collapses Investigations A/B to historical
pointers now that their scripts are gone."
```

---

## Task 19: Final full-suite run + manual smoke test (Section 3)

- [ ] **Step 1: Run the complete suite one more time**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -30`
Expected: identical pass/fail/xfail counts to the Task 1 baseline.

- [ ] **Step 2: Smoke-test run_disparity_sweep.py's new flag (dry, no GPU work — just argument wiring)**

Run: `/opt/venv/reconstruction/bin/python -c "
import evals.runners.run_disparity_sweep as m
print(m.run_our_pipeline.__defaults__)
"`

(Run from the repo root with `PYTHONPATH=evals` if needed, matching the existing `sys.path.insert` pattern used by `tests/evals/`.) Expected: prints `('vggt_spark',)` confirming the default backbone is unchanged.

- [ ] **Step 3: Smoke-test run_cross_model_benchmark.py's dry_run matrix against the existing disparity_sweep baseline**

Run: `/opt/venv/reconstruction/bin/python evals/runners/run_cross_model_benchmark.py --dry_run --kf_dir evals/baselines/disparity_sweep`
Expected: prints one `eval_gt.py` command line per (backbone, frameset) combination, same count as before generalization (verified in Task 16 Step 3); exit 0.

- [ ] **Step 4: Smoke-test build_benchmark_table.py against the existing frozen baseline**

Run: `/opt/venv/reconstruction/bin/python evals/runners/build_benchmark_table.py --root evals/baselines/cross_model --sweep_dir evals/baselines/disparity_sweep`
Expected: prints the same markdown table as before the `--reference_backbone` flag was added (default `vggt_spark` unchanged); exit 0.

No commit for this step (verification only — if all four checks pass, this plan's work is complete).

---

## Self-Review

**1. Spec coverage:**
- Section 1 (wrapper scriptability) → Tasks 2, 3.
- Section 2 dead-code removal (sigma constants, duplicate header) → Tasks 4, 5.
- Section 2 "not dead, fix docs" (eval.py docstring) → Task 7.
- Section 2 duplication collapse (`_umeyama_sim3`) → Task 8.
- Section 2 comments (all 6 files) → Tasks 6, 9, 10, 11.
- Section 3 delete 12 files → Tasks 13, 14.
- Section 3 generalize 3 files → Tasks 15, 16, 17.
- Section 3 README rewrite → Task 18.
- Testing/validation plan (baseline, post-Section-2 checkpoint, post-Section-3 re-run + smoke test) → Tasks 1, 12, 19.
- Risks (missed import, CLI default drift, incorrect comment claims) → addressed by Task 13 (pre-deletion grep), Task 16 Step 3 + Task 19 Steps 2-4 (default-preservation checks), and by only writing comments for logic read verbatim during plan preparation (Tasks 6, 9, 10).

No spec requirement found without a corresponding task.

**2. Placeholder scan:** no "TBD"/"TODO"/"add appropriate X" phrasing in any step; every code-editing step shows exact current text and exact replacement text; every verification step gives the exact command and expected output.

**3. Type/signature consistency:** `run_our_pipeline(seq_dir, keyframe_list, condition, out_ate, backbone="vggt_spark")` (Task 15 Step 1) is called with matching keyword `backbone=` in Task 15 Step 2 and Task 19 Step 2. `sweep(seq_dir, max_frames, start_disparity, backbone="vggt_spark")` (Task 15 Step 2) is called with `backbone=args.backbone` in Task 15 Step 3, matching. `build_matrix(kf_dir, backbones, framesets, submap_size)` (Task 16 Step 1) is called as `build_matrix(args.kf_dir, args.backbones, args.framesets, args.submap_size)` in Task 16 Step 2, matching positionally. `umeyama_sim3(source, target, weights=None)` (already defined in `closure.py`, unmodified by this plan) is called as `umeyama_sim3(source=centers_pred, target=centers_gt)` in Task 8 Step 5 and `umeyama_sim3(source=p_pred, target=p_gt)` in Task 8 Step 3 (pre-existing call, just un-lazied) — both use keyword args matching the real signature.

---

## Execution Handoff

Two ways to run this:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
