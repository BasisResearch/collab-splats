# Cross-Model LC Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark all four feedforward backbones (`vggt_spark` reference, `vggtx`, `vggt_omega`, `mapanything`) on 7-Scenes chess across conditions/params, ranked by ATE/RPE/AUC against ground truth, separating windowing cost (single-pass vs windowed) from loop-closure benefit.

**Architecture:** Three small code changes (multi-threshold AUC, surface RPE-rot + AUC in eval_gt, SPARK-load guard), one orchestration runner that drives `eval_gt.py` over the matrix in tmux serially, plus SLAM reference generation (loop probe + denser/long refs). Results aggregate from each run's `metrics.json` into a committed table + analysis.

**Tech Stack:** Python 3.11 (`/opt/conda/envs/reconstruction/bin/python`), numpy, evo (ATE/RPE), pytest, tmux. Heavy GPU runs one model at a time (46 GB cgroup cap).

**Spec:** `docs/superpowers/specs/2026-05-31-cross-model-benchmark-design.md`

---

## File Structure

**Code (modify):**
- `collab_splats/pointcloud/loop_closure/eval.py` — `auc_at_threshold` gains multi-threshold support; dynamic `auc_{t}` keys.
- `evals/eval_gt.py` — compute/print/store AUC@{5,15,30} + RPE rotation.
- `collab_splats/pointcloud/feedforward/vggt_spark_creator.py` — `_load_model` purges cached `vggt`, asserts real SPARK loaded.

**Tests (modify/create):**
- `tests/pointcloud/test_auc_metric.py` — fix stale `per_frame_err`→`per_pair_err`; add multi-threshold test.
- `tests/pointcloud/feedforward/test_spark_load_guard.py` — new, unit-test the load-path assertion helper.

**Orchestration (create):**
- `evals/runners/run_cross_model_benchmark.py` — drives the matrix; writes per-run dirs under `evals/baselines/cross_model/`.
- `evals/runners/build_benchmark_table.py` — aggregates `metrics.json` → results markdown.

**Reference data (generate):**
- `evals/baselines/disparity_sweep/slam_dN_long/` — long-sequence SLAM ref (if loop probe passes).

**Deliverable (write):**
- `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md` — table + analysis.
- Raw JSONs committed under `evals/baselines/cross_model/`.

---

## Phase A — Code changes (TDD, fast, no GPU)

### Task 1: Multi-threshold AUC + fix stale test

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/eval.py:217-292`
- Test: `tests/pointcloud/test_auc_metric.py`

- [ ] **Step 1: Update the stale test and add a multi-threshold test**

The current file asserts `per_frame_err` (3 failing tests) — the metric now returns `per_pair_err`. Replace every `per_frame_err` with `per_pair_err`, and add a multi-threshold test. Apply these edits to `tests/pointcloud/test_auc_metric.py`:

Replace `test_auc_returns_required_keys`:
```python
def test_auc_returns_required_keys():
    pred = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert set(result) >= {"auc_30", "per_pair_err"}
```

Replace `test_per_frame_err_length_matches_input` (rename + key):
```python
def test_per_pair_err_present():
    N = 7
    pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert len(result["per_pair_err"]) == N * (N - 1)
```

In `test_auc_translation_direction_error_is_angular_not_l2`, change the final two assertions' `per_frame_err` → `per_pair_err` and drop the `len(...) == N` assertion (pair count is N*(N-1), not N):
```python
    assert result["auc_30"] >= 95.0, f"Expected ~100 for identical poses, got {result['auc_30']}"
    assert isinstance(result["per_pair_err"], list)
```

Add a new multi-threshold test at the end:
```python
def test_auc_multi_threshold_keys_and_monotonic():
    """thresholds=(5,15,30) returns auc_5/auc_15/auc_30; AUC is non-decreasing in threshold."""
    gt   = _make_poses([0, 10, 20, 30], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    pred = _make_poses([0, 12, 18, 33], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    result = auc_at_threshold(pred, gt, thresholds=(5.0, 15.0, 30.0))
    assert {"auc_5", "auc_15", "auc_30", "per_pair_err"} <= set(result)
    assert result["auc_5"] <= result["auc_15"] <= result["auc_30"] + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_auc_metric.py -q`
Expected: FAIL — `test_auc_multi_threshold_keys_and_monotonic` errors (`auc_at_threshold() got an unexpected keyword argument 'thresholds'`); the key-rename tests still fail until code keeps `per_pair_err` (already does) — they should now pass except the multi-threshold one.

- [ ] **Step 3: Refactor `auc_at_threshold` to multi-threshold**

In `collab_splats/pointcloud/loop_closure/eval.py`, change the signature and the AUC computation tail. Replace the signature line:
```python
def auc_at_threshold(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: tuple[float, ...] = (30.0,),
) -> dict:
```

Update the docstring `Returns:` line to:
```python
        {"auc_{t}": float in [0,100] for each t in thresholds, "per_pair_err": list[float]}
```

Replace the final histogram/return block (currently lines ~286-292, the `max_t = int(max_threshold_deg)` through `return {...}`) with:
```python
    # Histogram AUC at each requested threshold: integer 1° bins [0, t], cumsum mean.
    # err is computed once over all pairs; only the cumulative window changes per threshold.
    out: dict = {}
    for t in thresholds:
        max_t = int(t)
        histogram, _ = np.histogram(err, bins=np.arange(max_t + 1))
        normalized = histogram.astype(float) / len(err)
        out[f"auc_{max_t}"] = float(np.mean(np.cumsum(normalized)) * 100.0)
    out["per_pair_err"] = err.tolist()
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_auc_metric.py tests/evals/test_metrics_auc.py -q`
Expected: PASS (all). The default `thresholds=(30.0,)` keeps `auc_30` so `evals/metrics.py` and existing callers are unaffected.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/eval.py tests/pointcloud/test_auc_metric.py
git commit -m "feat(eval): multi-threshold pose AUC + fix stale per_pair_err test"
```

---

### Task 2: Surface RPE-rotation + AUC@{5,15,30} through eval_gt

**Files:**
- Modify: `evals/eval_gt.py:426-435` (compute/print), `evals/eval_gt.py:201-207` (`_save_outputs`)
- Test: `tests/evals/test_eval_gt_helpers.py`

- [ ] **Step 1: Write a failing test for `_save_outputs` storing all AUC thresholds**

`_save_outputs` currently stores only `auc_30`. Add a test that a metrics dict with `auc_5/15/30` is persisted. Append to `tests/evals/test_eval_gt_helpers.py`:
```python
def test_save_outputs_persists_all_auc_thresholds(tmp_path):
    import numpy as np
    from evals.eval_gt import _save_outputs
    poses = np.tile(np.eye(4), (3, 1, 1))
    metrics = {"baseline": {
        "ate": {"rmse": 0.01, "per_frame": np.zeros(3)},
        "rpe": {"trans_rmse": 0.02, "rot_rmse_deg": 1.5},
        "auc": {"auc_5": 10.0, "auc_15": 50.0, "auc_30": 80.0, "per_pair_err": [0.0]},
        "time_s": 1,
    }}
    _save_outputs(metrics, {"gt": poses, "baseline": poses}, tmp_path)
    import json
    saved = json.loads((tmp_path / "metrics.json").read_text())["baseline"]
    assert saved["auc"] == {"auc_5": 10.0, "auc_15": 50.0, "auc_30": 80.0}
    assert saved["rpe"]["rot_rmse_deg"] == 1.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py::test_save_outputs_persists_all_auc_thresholds -v`
Expected: FAIL (KeyError `auc` / stored value is `auc_30` scalar, not dict).

- [ ] **Step 3: Update `_save_outputs` to store the full AUC dict**

In `evals/eval_gt.py`, replace the per-condition block in `_save_outputs` (currently `"auc_30": m["auc"]["auc_30"],`) so the whole AUC dict minus `per_pair_err` is stored:
```python
    for cond, m in metrics.items():
        metrics_json[cond] = {
            "ate": {k: v for k, v in m["ate"].items() if k != "per_frame"},
            "rpe": m["rpe"],
            "auc": {k: v for k, v in m["auc"].items() if k != "per_pair_err"},
            "time_s": m.get("time_s", None),
        }
```

- [ ] **Step 4: Update the metric computation + prints to use 3 thresholds**

In `evals/eval_gt.py`, change the `auc_at_threshold(...)` call (line ~429) to request all thresholds:
```python
                "auc": auc_at_threshold(
                    np.linalg.inv(pred), np.linalg.inv(dataset.gt_poses),
                    thresholds=(5.0, 15.0, 30.0),
                ),
```
Replace the three print lines (433-435) with:
```python
            print(f"  ATE RMSE: {metrics[cond]['ate']['rmse']:.4f}m")
            print(f"  RPE trans/rot: {metrics[cond]['rpe']['trans_rmse']:.4f}m / "
                  f"{metrics[cond]['rpe']['rot_rmse_deg']:.3f}deg")
            print(f"  AUC@5/15/30: {metrics[cond]['auc']['auc_5']:.1f} / "
                  f"{metrics[cond]['auc']['auc_15']:.1f} / {metrics[cond]['auc']['auc_30']:.1f}")
```

- [ ] **Step 5: Run to verify pass + full helper suite green**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/eval_gt.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(eval): report RPE rotation + AUC@{5,15,30} in eval_gt metrics.json"
```

---

### Task 3: SPARK-load guard (kill the silent VGGT-X trap)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggt_spark_creator.py:60-108`
- Test: `tests/pointcloud/feedforward/test_spark_load_guard.py` (create)

**Why:** `vggt_spark_creator.py:20` does `from vggt.utils.pose_enc import ...` at module top — with **no** SPARK path inserted — so `vggt` is cached as VGGT-X before `_load_model` runs. The later `from vggt.models.vggt import VGGT` then returns cached VGGT-X. The INFO log says "loading from SPARK source tree" while actually loading VGGT-X. This task makes the wrong case a loud failure.

- [ ] **Step 1: Write a failing test for the path-assertion helper**

Create `tests/pointcloud/feedforward/test_spark_load_guard.py`:
```python
import pytest

from collab_splats.pointcloud.feedforward.vggt_spark_creator import (
    _assert_loaded_from_spark,
    _VGGT_SPARK_ROOT,
)


def test_accepts_file_under_spark_root():
    # A path inside the SPARK tree passes silently.
    _assert_loaded_from_spark(f"{_VGGT_SPARK_ROOT}/vggt/models/vggt.py")


def test_rejects_file_outside_spark_root():
    # An installed VGGT-X path (site-packages) must raise, not warn.
    with pytest.raises(RuntimeError, match="VGGT-X"):
        _assert_loaded_from_spark("/opt/conda/envs/reconstruction/lib/python3.11/site-packages/vggt/models/vggt.py")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_spark_load_guard.py -v`
Expected: FAIL (`ImportError: cannot import name '_assert_loaded_from_spark'`).

- [ ] **Step 3: Add the helper and wire it into `_load_model`**

In `collab_splats/pointcloud/feedforward/vggt_spark_creator.py`, add `import inspect` to the imports, and add this module-level helper after `_VGGT_SPARK_ROOT`:
```python
def _assert_loaded_from_spark(module_file: str) -> None:
    """Raise if the resolved vggt module is not under the SPARK source tree.

    Guards the silent import-cache trap where a previously-cached VGGT-X
    `vggt` package shadows the SPARK path insertion, so `vggt_spark` would
    silently run VGGT-X weights/code.
    """
    if not str(module_file).startswith(_VGGT_SPARK_ROOT):
        raise RuntimeError(
            f"vggt_spark loaded VGGT-X, not SPARK: resolved {module_file!r} "
            f"is outside {_VGGT_SPARK_ROOT!r}. The `vggt` package was import-cached "
            f"before _load_model inserted the SPARK path. Run vggt_spark in a fresh "
            f"process (or via parity_trace.py) before trusting its numbers."
        )
```

Then in `_load_model`, replace the import block so it purges any cached `vggt*` modules first, and assert after import. Replace the existing `_patched = ...` through the `from vggt.models.vggt import VGGT as VGGT_SPARK` block with:
```python
        # Purge any cached `vggt*` (e.g. VGGT-X imported at module top) so the
        # SPARK path insertion actually wins, then import SPARK.
        for _m in [m for m in list(sys.modules) if m == "vggt" or m.startswith("vggt.")]:
            del sys.modules[_m]
        _patched = _VGGT_SPARK_ROOT not in sys.path
        if _patched:
            sys.path.insert(0, _VGGT_SPARK_ROOT)
        try:
            from vggt.models.vggt import VGGT as VGGT_SPARK  # noqa: PLC0415
        finally:
            if _patched and _VGGT_SPARK_ROOT in sys.path:
                sys.path.remove(_VGGT_SPARK_ROOT)

        # Hard guard: confirm the real SPARK module loaded (not cached VGGT-X).
        _resolved = inspect.getfile(VGGT_SPARK)
        _assert_loaded_from_spark(_resolved)
        logger.info("VGGTSPARKCreator: loaded VGGT from %s", _resolved)
```

> Note: purging `vggt` from `sys.modules` invalidates the top-level `pose_encoding_to_extri_intri` symbol's module identity only if re-imported; it is already bound as a function reference at line 20 and keeps working. After `_load_model`, a subsequent `import vggt` re-imports VGGT-X fresh (SPARK path already removed), which is the intended session behavior.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_spark_load_guard.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggt_spark_creator.py tests/pointcloud/feedforward/test_spark_load_guard.py
git commit -m "fix(feedforward): hard-guard vggt_spark against silent VGGT-X import-cache fallback"
```

---

## Phase B — Orchestration scripts (no GPU to write; light smoke run)

### Task 4: Benchmark runner

**Files:**
- Create: `evals/runners/run_cross_model_benchmark.py`
- Test: `tests/evals/test_cross_model_runner.py` (create)

The runner builds the command list for the matrix and (unless `--dry_run`) invokes `eval_gt.py` once per `(backbone, frameset, submap_size)`, passing all requested `--conditions` in one call (eval_gt loops conditions internally, reusing the loaded model). Output goes to `evals/baselines/cross_model/<backbone>__<frameset>__sm<N>/`.

- [ ] **Step 1: Write a failing test for command construction**

Create `tests/evals/test_cross_model_runner.py`:
```python
from pathlib import Path

from evals.runners.run_cross_model_benchmark import build_commands, RunSpec


def test_build_commands_one_per_backbone_frameset():
    specs = [
        RunSpec(backbone="vggt_spark", frameset="slam_d10", submap_size=16,
                conditions=("baseline", "lc"), keyframe_list=Path("kf.txt")),
        RunSpec(backbone="vggtx", frameset="slam_d10_single", submap_size=None,
                conditions=("baseline",), keyframe_list=Path("kf.txt")),
    ]
    cmds = build_commands(specs, seq_dir=Path("/seq"), out_root=Path("/out"))
    assert len(cmds) == 2
    # windowed spark run includes submap flag and both conditions
    c0 = " ".join(cmds[0])
    assert "--backbone vggt_spark" in c0
    assert "--submap_size 16" in c0
    assert "--conditions baseline lc" in c0
    assert "--keyframe_list kf.txt" in c0
    # single-pass run omits --submap_size entirely
    c1 = " ".join(cmds[1])
    assert "--submap_size" not in c1
    assert "--backbone vggtx" in c1
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_cross_model_runner.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement the runner**

Create `evals/runners/run_cross_model_benchmark.py`:
```python
"""Drive the cross-model chess benchmark: one eval_gt.py call per (backbone, frameset).

Serial by design — heavy GPU runs, 46 GB cgroup cap, no parallel jobs. Each run's
metrics.json lands under evals/baselines/cross_model/ for build_benchmark_table.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PY = "/opt/conda/envs/reconstruction/bin/python"
EVAL_GT = Path(__file__).resolve().parents[1] / "eval_gt.py"


@dataclass
class RunSpec:
    """One eval_gt invocation: a backbone over a frameset at a submap size."""
    backbone: str
    frameset: str
    submap_size: int | None
    conditions: tuple[str, ...]
    keyframe_list: Path
    lc_scale_method: str = "rotation_only"


def _out_dir(out_root: Path, spec: RunSpec) -> Path:
    sm = "single" if spec.submap_size is None else f"sm{spec.submap_size}"
    return out_root / f"{spec.backbone}__{spec.frameset}__{sm}"


def build_commands(specs: list[RunSpec], seq_dir: Path, out_root: Path) -> list[list[str]]:
    """Build the argv list for each RunSpec."""
    cmds: list[list[str]] = []
    for s in specs:
        cmd = [
            PY, str(EVAL_GT),
            "--dataset", "7scenes",
            "--seq_dir", str(seq_dir),
            "--backbone", s.backbone,
            "--conditions", *s.conditions,
            "--lc_scale_method", s.lc_scale_method,
            "--keyframe_list", str(s.keyframe_list),
            "--output_dir", str(_out_dir(out_root, s)),
            "--output_ate", str(_out_dir(out_root, s) / "ate.json"),
        ]
        if s.submap_size is not None:
            cmd += ["--submap_size", str(s.submap_size)]
        cmds.append(cmd)
    return cmds


def core_matrix(kf_dir: Path) -> list[RunSpec]:
    """Core matrix: 4 backbones × framesets × conditions (spec §Execution step 2)."""
    backbones = ["vggt_spark", "vggtx", "vggt_omega", "mapanything"]
    specs: list[RunSpec] = []
    d10_kf = kf_dir / "slam_d10" / "selected_frames.txt"
    d20_kf = kf_dir / "slam_d20" / "selected_frames.txt"
    for b in backbones:
        # Goal 1 — windowing cost: single-pass vs windowed baseline on short sets.
        specs.append(RunSpec(b, "slam_d10_single", None, ("baseline",), d10_kf))
        specs.append(RunSpec(b, "slam_d20_single", None, ("baseline",), d20_kf))
        # Windowed baseline + lc on d10 (2 submaps).
        specs.append(RunSpec(b, "slam_d10", 16, ("baseline", "lc"), d10_kf))
    return specs


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
    args.out_root.mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(cmds, 1):
        print(f"\n[{i}/{len(cmds)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}) — continuing", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass + dry-run smoke**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_cross_model_runner.py -v`
Expected: PASS.
Run: `/opt/conda/envs/reconstruction/bin/python evals/runners/run_cross_model_benchmark.py --dry_run`
Expected: prints 12 commands (4 backbones × 3 runs each), no execution.

- [ ] **Step 5: Commit**

```bash
git add evals/runners/run_cross_model_benchmark.py tests/evals/test_cross_model_runner.py
git commit -m "feat(evals): cross-model benchmark runner (serial eval_gt matrix)"
```

---

### Task 5: Table builder

**Files:**
- Create: `evals/runners/build_benchmark_table.py`
- Test: `tests/evals/test_build_benchmark_table.py` (create)

Aggregates every `metrics.json` under `evals/baselines/cross_model/` into a markdown table with Δ-vs-spark and Δ-vs-SLAM (ATE) columns. SLAM ATE comes from `evals/baselines/disparity_sweep/slam_dN/metrics.json`.

- [ ] **Step 1: Write a failing test for row assembly**

Create `tests/evals/test_build_benchmark_table.py`:
```python
from evals.runners.build_benchmark_table import assemble_rows


def test_assemble_rows_computes_deltas():
    runs = {
        ("vggt_spark", "slam_d10", "baseline"): {"ate": {"rmse": 0.017},
            "rpe": {"trans_rmse": 0.01, "rot_rmse_deg": 1.0},
            "auc": {"auc_5": 50, "auc_15": 80, "auc_30": 90}},
        ("vggtx", "slam_d10", "baseline"): {"ate": {"rmse": 0.027},
            "rpe": {"trans_rmse": 0.02, "rot_rmse_deg": 2.0},
            "auc": {"auc_5": 40, "auc_15": 70, "auc_30": 85}},
    }
    slam_ate = {"slam_d10": 0.0176}
    rows = assemble_rows(runs, slam_ate, reference_backbone="vggt_spark")
    vggtx = [r for r in rows if r["backbone"] == "vggtx"][0]
    assert abs(vggtx["delta_vs_spark"] - 0.010) < 1e-6   # 0.027 - 0.017
    assert abs(vggtx["delta_vs_slam"] - (0.027 - 0.0176)) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_build_benchmark_table.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the builder**

Create `evals/runners/build_benchmark_table.py`:
```python
"""Aggregate cross_model/*/metrics.json into a markdown results table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assemble_rows(runs: dict, slam_ate: dict, reference_backbone: str) -> list[dict]:
    """Build per-(backbone,frameset,condition) rows with Δ-vs-spark / Δ-vs-SLAM (ATE)."""
    # Reference ATE per (frameset, condition) from the reference backbone.
    ref = {(fs, cond): m["ate"]["rmse"]
           for (b, fs, cond), m in runs.items() if b == reference_backbone}
    rows: list[dict] = []
    for (b, fs, cond), m in sorted(runs.items()):
        ate = m["ate"]["rmse"]
        rows.append({
            "backbone": b, "frameset": fs, "condition": cond,
            "ate": ate,
            "rpe_t": m["rpe"]["trans_rmse"], "rpe_r": m["rpe"]["rot_rmse_deg"],
            "auc_5": m["auc"]["auc_5"], "auc_15": m["auc"]["auc_15"],
            "auc_30": m["auc"]["auc_30"],
            "delta_vs_spark": ate - ref.get((fs, cond), float("nan")),
            "delta_vs_slam": ate - slam_ate.get(fs, float("nan")),
        })
    return rows


def _load_runs(root: Path) -> dict:
    """Read every cross_model/<backbone>__<frameset>__<sm>/metrics.json."""
    runs: dict = {}
    for mj in root.glob("*/metrics.json"):
        backbone, frameset, _sm = mj.parent.name.split("__")
        data = json.loads(mj.read_text())
        for cond, m in data.items():
            if cond.startswith("_"):
                continue
            runs[(backbone, frameset, cond)] = m
    return runs


def _load_slam_ate(sweep_dir: Path) -> dict:
    """Map frameset name → VGGT-SLAM ATE from disparity_sweep/slam_dN/metrics.json."""
    out: dict = {}
    for mj in sweep_dir.glob("slam_*/metrics.json"):
        out[mj.parent.name] = json.loads(mj.read_text()).get("ate_rmse", float("nan"))
    return out


def render_markdown(rows: list[dict]) -> str:
    """Render rows as a pipe table."""
    head = ("| backbone | frameset | cond | ATE | RPE-t | RPE-r° | AUC5 | AUC15 | AUC30 "
            "| Δspark | Δslam |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['backbone']} | {r['frameset']} | {r['condition']} | {r['ate']:.4f} "
            f"| {r['rpe_t']:.4f} | {r['rpe_r']:.2f} | {r['auc_5']:.1f} | {r['auc_15']:.1f} "
            f"| {r['auc_30']:.1f} | {r['delta_vs_spark']:+.4f} | {r['delta_vs_slam']:+.4f} |")
    return "\n".join(lines)


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


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/evals/test_build_benchmark_table.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/runners/build_benchmark_table.py tests/evals/test_build_benchmark_table.py
git commit -m "feat(evals): aggregate cross-model metrics into markdown table"
```

---

## Phase C — Execution (tmux, heavy GPU, one model at a time)

> All runs in tmux. After each, confirm the loaded-model line in stdout. Never run parallel GPU jobs (46 GB cap).

### Task 6: Sanity gate — vggt_spark @ d10 must reproduce ~0.017 m as real SPARK

**Files:** none (run + verify).

- [ ] **Step 1: Run the gate in tmux**

```bash
tmux new -s gate -d
tmux send-keys -t gate '/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_spark --conditions baseline \
  --submap_size 16 --lc_scale_method rotation_only \
  --keyframe_list evals/baselines/disparity_sweep/slam_d10/selected_frames.txt \
  --output_dir /tmp/gate_spark_d10 --output_ate /tmp/gate_spark_d10/ate.json \
  2>&1 | tee /tmp/gate_spark_d10.log' Enter
```

- [ ] **Step 2: Verify real SPARK loaded AND ATE ≈ 0.017**

```bash
grep "loaded VGGT from" /tmp/gate_spark_d10.log     # must contain third_party/vggt_spark
grep "ATE RMSE" /tmp/gate_spark_d10.log             # must be ~0.017m (vs SLAM 0.0176)
```
Expected: load path under `third_party/vggt_spark`; ATE in [0.015, 0.020] m.
**If the load guard raised RuntimeError** (VGGT-X loaded), the import-cache trap is live in-process — stop and report; do not proceed. **If ATE is far from 0.017**, stop: the pipeline regressed.

- [ ] **Step 3: Commit the gate artifact**

```bash
mkdir -p evals/baselines/cross_model/_gate
cp /tmp/gate_spark_d10/metrics.json evals/baselines/cross_model/_gate/spark_d10_metrics.json
git add evals/baselines/cross_model/_gate/spark_d10_metrics.json
git commit -m "test(evals): vggt_spark d10 sanity gate passes (real SPARK, ATE~0.017m)"
```

---

### Task 7: Loop probe — does chess seq-01 ever close a loop?

**Files:** none (run + decide).

- [ ] **Step 1: Run SLAM-LC on the long sequence in tmux**

```bash
tmux new -s probe -d
tmux send-keys -t probe '/opt/conda/envs/reconstruction/bin/python evals/runners/run_vggt_slam_lc.py \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --max_frames 1000 --min_disparity 5 --max_loops 1 \
  2>&1 | tee /tmp/loop_probe.log' Enter
```

- [ ] **Step 2: Read the closed-loop count**

```bash
grep -iE "num_loops|loop" /tmp/loop_probe.log | tail
```
Expected: a line reporting `get_num_loops()`.
- **If `> 0`** → proceed to Task 8 (generate long ref + extend matrix).
- **If `== 0`** → chess seq-01 does not revisit. Per spec decision, **record "LC-on-chess structurally untestable", skip Tasks 8–9's lc-on-long arm**, and note it in the results writeup (Task 11). Continue with goal-1 + windowing results only.

- [ ] **Step 3: Record the probe result**

```bash
mkdir -p evals/baselines/cross_model/_gate
cp /tmp/loop_probe.log evals/baselines/cross_model/_gate/loop_probe.log
git add evals/baselines/cross_model/_gate/loop_probe.log
git commit -m "test(evals): chess seq-01 loop probe result recorded"
```

---

### Task 8: Long SLAM reference (only if Task 7 probe > 0)

**Files:** generates `evals/baselines/disparity_sweep/slam_dN_long/`.

- [ ] **Step 1: Generate the long reference at the disparity that closed a loop**

Use the disparity from Task 7 (e.g. 5). Run `run_disparity_sweep.py` with raised `max_frames` so it emits `selected_frames.txt` + `metrics.json` for the long level:
```bash
tmux send-keys -t probe '/opt/conda/envs/reconstruction/bin/python evals/runners/run_disparity_sweep.py \
  --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --max_frames 1000 --start_disparity 5 2>&1 | tee /tmp/long_ref.log' Enter
```

- [ ] **Step 2: Confirm ≥3 submaps and ≥1 loop in the SLAM ref**

```bash
cat evals/baselines/disparity_sweep/slam_d5/metrics.json
```
Expected: `submaps >= 3`, `loop_closures >= 1`. Rename to the `_long` convention so the runner picks it up distinctly:
```bash
git mv evals/baselines/disparity_sweep/slam_d5 evals/baselines/disparity_sweep/slam_d5_long 2>/dev/null || \
  mv evals/baselines/disparity_sweep/slam_d5 evals/baselines/disparity_sweep/slam_d5_long
```

- [ ] **Step 3: Commit the long reference**

```bash
git add evals/baselines/disparity_sweep/slam_d5_long
git commit -m "data(evals): long SLAM reference (slam_d5_long, loops closed) for goal-2 arm"
```

---

### Task 9: Run the core matrix

**Files:** populates `evals/baselines/cross_model/`.

- [ ] **Step 1: (If long ref exists) extend the matrix in the runner**

Add the long frameset to `core_matrix()` in `evals/runners/run_cross_model_benchmark.py` — append inside the backbone loop:
```python
        # Goal 2 — LC benefit on the long, loop-closing set (only if it exists).
        long_kf = kf_dir / "slam_d5_long" / "selected_frames.txt"
        if long_kf.exists():
            specs.append(RunSpec(b, "slam_d5_long", 16, ("baseline", "lc"), long_kf))
```
Commit:
```bash
git add evals/runners/run_cross_model_benchmark.py
git commit -m "feat(evals): add long loop-closing frameset to benchmark matrix"
```

- [ ] **Step 2: Run the full matrix in tmux (serial, ~one model at a time)**

```bash
tmux new -s bench -d
tmux send-keys -t bench '/opt/conda/envs/reconstruction/bin/python \
  evals/runners/run_cross_model_benchmark.py 2>&1 | tee /tmp/bench.log' Enter
```
Monitor: `tmux capture-pane -t bench -p | tail -20`. Each run prints its loaded-model line, ATE/RPE/AUC.

- [ ] **Step 3: Verify every run's loaded model + completeness**

```bash
grep -L "Results written" evals/baselines/cross_model/*/.. 2>/dev/null  # sanity
for d in evals/baselines/cross_model/*/; do
  echo "$d"; grep -h "loaded VGGT from\|Loaded\|backbone" "$d"/../*.log 2>/dev/null | head -1
done
ls evals/baselines/cross_model/*/metrics.json | wc -l   # expect 12 (+3 if long set)
```
Expected: every `metrics.json` present; spark runs show the SPARK path.

- [ ] **Step 4: Commit raw results**

```bash
git add evals/baselines/cross_model
git commit -m "data(evals): cross-model core matrix raw metrics (chess)"
```

---

### Task 10: Adaptive layer sweep where LC hurts or diverges

**Files:** populates `evals/baselines/cross_model/_layersweep/`.

- [ ] **Step 1: Identify backbones where `lc` ATE ≥ `baseline` (windowed) or ATE far from spark**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/build_benchmark_table.py
```
Read the table. For any backbone where (lc ATE ≥ baseline ATE on a frameset that closed loops) OR (ATE Δ-vs-spark is large at matched params), it is a sweep candidate.

- [ ] **Step 2: Sweep ±a few LC layers for each candidate**

The LC layer is set per-creator via `_lc_layer_index`. For each candidate backbone `B` with default layer `L`, run at `{L-4, L-2, L, L+2, L+4}` (clamp to valid range) by setting the attribute before the run. Use a short driver in tmux per layer, e.g. for omega (default 16):
```bash
for LYR in 12 14 16 18 20; do
  tmux send-keys -t bench "OMEGA_LC_LAYER=$LYR /opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone vggt_omega --conditions baseline lc --submap_size 16 \
    --lc_scale_method rotation_only \
    --keyframe_list evals/baselines/disparity_sweep/slam_d10/selected_frames.txt \
    --output_dir evals/baselines/cross_model/_layersweep/omega_L\$LYR \
    --output_ate evals/baselines/cross_model/_layersweep/omega_L\$LYR/ate.json 2>&1 | tee -a /tmp/sweep_omega.log" Enter
done
```
> If `_lc_layer_index` is not env-overridable, add a one-line `--lc_layer` passthrough to `eval_gt.py` that sets `creator._lc_layer_index` before `reconstruct`, mirroring how `lc_scale_method` is threaded. Commit that small change first.

- [ ] **Step 3: Record whether the post-fix optimum moved**

Note, per candidate backbone, the layer minimizing lc ATE and whether it differs from the documented default. This feeds the writeup (re-validates / overturns the old 20/16/4 + "omega harmful / mapanything broken" notes).

- [ ] **Step 4: Commit sweep results**

```bash
git add evals/baselines/cross_model/_layersweep evals/eval_gt.py 2>/dev/null
git commit -m "data(evals): adaptive LC-layer sweep for backbones where LC hurt"
```

---

### Task 11: Expansion — BA + intermediate disparities (autonomous)

**Files:** extends `evals/baselines/cross_model/`.

- [ ] **Step 1: Add `ba` condition and d20/d30 reference framesets**

Extend `core_matrix()` to append a BA run per backbone and reference-column runs at d20/d30:
```python
        # Expansion: bundle adjustment on the short windowed set.
        specs.append(RunSpec(b, "slam_d10", 16, ("ba",), d10_kf))
        # Reference column at more disparities (baseline only).
        for lvl, kf in (("slam_d20", d20_kf), ("slam_d30", kf_dir / "slam_d30" / "selected_frames.txt")):
            specs.append(RunSpec(b, lvl, 16, ("baseline",), kf))
```
Commit, then re-run the runner in tmux (it overwrites/extends per-run dirs):
```bash
git add evals/runners/run_cross_model_benchmark.py
git commit -m "feat(evals): expansion matrix — BA + d20/d30 reference"
tmux send-keys -t bench '/opt/conda/envs/reconstruction/bin/python evals/runners/run_cross_model_benchmark.py 2>&1 | tee -a /tmp/bench.log' Enter
```

- [ ] **Step 2: Localize any backbone still diverging from spark**

For any backbone whose ATE stays far from spark at matched params, run the parity harness to pin the stage:
```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/parity_trace.py --side ours --min_disparity 10
# then --side slam, then --side diff; read the PASS/DIVERGE stage table
```
Record the diverging stage (forward / trajectory / boundary scale / homographies).

- [ ] **Step 3: Commit expansion results**

```bash
git add evals/baselines/cross_model
git commit -m "data(evals): expansion matrix raw metrics (BA + disparities)"
```

---

## Phase D — Results writeup

### Task 12: Write and commit the results doc

**Files:**
- Create: `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md`

- [ ] **Step 1: Generate the table**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/build_benchmark_table.py > /tmp/bench_table.md
```

- [ ] **Step 2: Write the results doc**

Create `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md` containing:
- The generated table (paste `/tmp/bench_table.md`), columns: backbone · frameset · condition · ATE · RPE-t · RPE-r° · AUC@{5,15,30} · Δ-vs-spark · Δ-vs-SLAM · loaded-model.
- **Analysis** sections (prose, grounded in the numbers):
  - *Sanity:* vggt_spark d10 baseline ATE vs SLAM 0.0176 (parity confirmed, real SPARK loaded).
  - *Best model per condition* (baseline / lc / ba).
  - *Goal 1 — windowing cost:* per backbone, `baseline(windowed) − single_pass` ATE; is windowing free or a tax? Cross-check with AUC (flat AUC + worse ATE ⇒ pure stitching/drift).
  - *Goal 2 — LC benefit:* per backbone, `lc − baseline` on the long loop-closing set — where LC helps vs hurts; explicitly re-validate or overturn the old "omega harmful / mapanything broken" notes (now measured post-pose-fix). **If the loop probe was 0**, state LC-on-chess is structurally untestable and recommend a looping sequence as follow-up.
  - *Layer calibration:* did the post-fix optimum move from 20/16/4? (Task 10 results.)
  - *Residual divergence:* any backbone still off spark + the parity_trace stage.

- [ ] **Step 3: Update the per-model LC layer memory if it changed**

If Task 10 overturned the documented layers or the harmful/broken notes, update `/root/.claude/projects/-workspace-collab-splats/memory/project_lc_layer_calibration.md` accordingly (and its MEMORY.md hook).

- [ ] **Step 4: Commit the deliverable**

```bash
git add docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md
git commit -m "docs(specs): cross-model LC benchmark results + analysis (chess)"
```

---

## Definition of Done (from spec)

- [ ] Sanity gate passed: vggt_spark ≈ 0.017 m at d10, real SPARK loaded and logged.
- [ ] Core matrix run; ATE/RPE/AUC tabulated with loaded-model verified per row.
- [ ] Goal 1 answered: windowing cost quantified per backbone (single_pass vs windowed baseline).
- [ ] Goal 2 answered: LC benefit on the long set, OR chess documented non-looping + arm stopped + follow-up recommended.
- [ ] Adaptive layer sweep run wherever LC hurt/diverged; results recorded; memory updated if changed.
- [ ] Findings written; any divergence traced to a stage via parity_trace.py.
- [ ] All new/changed code green: `/opt/conda/envs/reconstruction/bin/python -m pytest tests/`.
