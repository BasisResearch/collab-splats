# LC Parity Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-level parity harness (spec: `docs/superpowers/specs/2026-07-08-lc-parity-validation-design.md`) that validates our LC pipeline against upstream VGGT-SLAM 2.0 on 7-Scenes + TUM, including full-scene scaling sweeps — without changing any LC logic.

**Architecture:** Reuse existing runners: `evals/runners/run_vggt_slam_lc.py` (upstream reference, already exports `selected_frames.txt` keyframe list + metrics.json) and `evals/eval_gt.py` (ours, already consumes `--keyframe_list`, plots 3D trajectory vs GT). New code = one shared helper module, a scene-matrix driver, a gate/table aggregator, TUM support in the reference path, and stage-trace dumps. All heavy runs are serial subprocess calls (46.6 GB cgroup cap — tmux, never notebooks).

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, numpy, existing evals/ modules. No new dependencies.

**Constraints (from spec):**
- NO changes to LC logic in `closure.py` / `wrappers.py`. The only wrappers change allowed is attaching a `reject_reason` string to the existing `LoopMatch` objects at the three existing reject sites (pure metadata, no control flow).
- Backbone: `vggt_spark` only.
- Parity config: `submap_size=16`, `overlap=1`, `max_loops(=per submap)=1`, `conf_threshold=25.0`, `lc_thres=0.95`, `min_disparity=50`, full sequences (no frame cap).

**Output layout (all tasks write here):**
```
evals/baselines/lc_parity/<scene>/slam/           # Level 0: <scene>.tum, selected_frames.txt, metrics.json
evals/baselines/lc_parity/<scene>/ours/           # Level 1: metrics.json, ate.json, plots/, trajectories.npz
evals/baselines/lc_parity/<scene>/prefix_{25,50}/slam|ours/   # scaling sweep (100% = the full run above)
evals/baselines/lc_parity/_parity_table.md        # Level 1 gate report
```
Scene keys: `7s_chess`, `7s_fire`, `7s_heads`, `7s_office`, `7s_pumpkin`, `7s_redkitchen`, `7s_stairs`, `tum_fr1_desk`, `tum_fr1_room`, `tum_fr2_xyz`, `tum_fr3_office`.

---

### Task 1: Shared helpers — scene registry, keyframe prefix slicing, gate checks

**Files:**
- Create: `evals/runners/lc_parity_common.py`
- Test: `tests/evals/test_lc_parity_common.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for LC parity harness helpers."""
from pathlib import Path

import pytest

from evals.runners.lc_parity_common import (
    SCENES,
    check_gates,
    list_scene_images,
    slam_max_frames_for_prefix,
    slice_keyframes,
)


def test_scene_registry_complete():
    # 7 7-Scenes + 4 TUM, each with dataset type and long-scene flag
    assert len(SCENES) == 11
    assert SCENES["7s_chess"].dataset == "7scenes"
    assert SCENES["tum_fr1_room"].dataset == "tum"
    long_scenes = [k for k, s in SCENES.items() if s.scaling_sweep]
    assert sorted(long_scenes) == ["7s_office", "7s_redkitchen", "tum_fr1_room", "tum_fr3_office"]


def test_slice_keyframes_writes_prefix(tmp_path):
    kf = tmp_path / "selected_frames.txt"
    kf.write_text("\n".join(f"/data/frame-{i:06d}.color.png" for i in range(100)))
    out = slice_keyframes(kf, 0.25, tmp_path / "prefix_25.txt")
    lines = out.read_text().splitlines()
    assert len(lines) == 25
    assert lines[0].endswith("frame-000000.color.png")
    assert lines[-1].endswith("frame-000024.color.png")


def test_slice_keyframes_full_is_identity(tmp_path):
    kf = tmp_path / "selected_frames.txt"
    kf.write_text("\n".join(f"img_{i}.png" for i in range(10)))
    out = slice_keyframes(kf, 1.0, tmp_path / "prefix_100.txt")
    assert out.read_text() == kf.read_text()


def test_list_scene_images_7scenes_layout(tmp_path):
    # 7-Scenes: color frames directly in seq dir
    for i in range(3):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    (tmp_path / "frame-000000.pose.txt").touch()  # must be excluded
    imgs = list_scene_images(tmp_path)
    assert len(imgs) == 3
    assert all(p.suffix == ".png" and "color" in p.name for p in imgs)


def test_list_scene_images_tum_layout(tmp_path):
    # TUM: images under rgb/
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for ts in ("1305031102.175304", "1305031102.211214"):
        (rgb / f"{ts}.png").touch()
    (tmp_path / "groundtruth.txt").touch()
    imgs = list_scene_images(tmp_path)
    assert len(imgs) == 2
    assert imgs[0].parent.name == "rgb"
    assert imgs == sorted(imgs)


def test_slam_max_frames_for_prefix(tmp_path):
    # 10 source frames, keyframes are frames 0,3,6,9. 50% prefix = kf 0,3
    # → SLAM must process source frames up to index 3 → max_frames=4.
    for i in range(10):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("\n".join(
        str(tmp_path / f"frame-{i:06d}.color.png") for i in (0, 3)
    ))
    assert slam_max_frames_for_prefix(prefix, tmp_path) == 4


def test_check_gates_pass():
    slam = {"ate_rmse": 0.100, "loop_closures": 5}
    ours = {"ate_rmse": 0.104, "loops_applied": 5}
    g = check_gates(slam, ours)
    assert g["ate_pass"] and g["loops_pass"] and g["all_pass"]
    assert g["ate_delta"] == pytest.approx(0.004)


def test_check_gates_ate_absolute_fallback():
    # tiny ATEs: 5% of 20mm = 1mm, but 5mm absolute tolerance applies
    slam = {"ate_rmse": 0.020, "loop_closures": 0}
    ours = {"ate_rmse": 0.024, "loops_applied": 0}
    assert check_gates(slam, ours)["ate_pass"]


def test_check_gates_fail_on_loop_mismatch():
    slam = {"ate_rmse": 0.100, "loop_closures": 5}
    ours = {"ate_rmse": 0.100, "loops_applied": 0}
    g = check_gates(slam, ours)
    assert not g["loops_pass"] and not g["all_pass"]


def test_check_gates_missing_ate():
    g = check_gates({"ate_rmse": None, "loop_closures": 1}, {"ate_rmse": 0.1, "loops_applied": 1})
    assert not g["ate_pass"] and g["ate_delta"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_lc_parity_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.runners.lc_parity_common'`

- [ ] **Step 3: Write the implementation**

```python
"""Shared helpers for the LC parity harness (spec 2026-07-08-lc-parity-validation-design).

Scene registry, keyframe prefix slicing for scaling sweeps, and Level-1 gate checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

########## Scene registry ##########

# Gates from spec §Level 1
ATE_REL_TOL = 0.05    # ≤5% relative
ATE_ABS_TOL = 0.005   # or ≤5 mm absolute
PREFIX_FRACTIONS = (0.25, 0.50, 1.0)

_IMG_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class SceneSpec:
    """One validation scene: dataset type, seq path relative to evals/data/, sweep flag."""
    dataset: str            # eval_gt --dataset value ("7scenes" | "tum")
    rel_seq_dir: str        # relative to evals/data/
    scaling_sweep: bool = False  # include in 25/50/100% prefix stress (spec §scaling)


SCENES: dict[str, SceneSpec] = {
    "7s_chess":      SceneSpec("7scenes", "7scenes/chess/chess/seq-01"),
    "7s_fire":       SceneSpec("7scenes", "7scenes/fire/fire/seq-01"),
    "7s_heads":      SceneSpec("7scenes", "7scenes/heads/heads/seq-01"),
    "7s_office":     SceneSpec("7scenes", "7scenes/office/office/seq-01", scaling_sweep=True),
    "7s_pumpkin":    SceneSpec("7scenes", "7scenes/pumpkin/pumpkin/seq-01"),
    "7s_redkitchen": SceneSpec("7scenes", "7scenes/redkitchen/redkitchen/seq-01", scaling_sweep=True),
    "7s_stairs":     SceneSpec("7scenes", "7scenes/stairs/stairs/seq-01"),
    "tum_fr1_desk":  SceneSpec("tum", "tum/rgbd_dataset_freiburg1_desk"),
    "tum_fr1_room":  SceneSpec("tum", "tum/rgbd_dataset_freiburg1_room", scaling_sweep=True),
    "tum_fr2_xyz":   SceneSpec("tum", "tum/rgbd_dataset_freiburg2_xyz"),
    "tum_fr3_office": SceneSpec("tum", "tum/rgbd_dataset_freiburg3_long_office_household", scaling_sweep=True),
}


########## Image listing (7-Scenes flat vs TUM rgb/) ##########

def list_scene_images(seq_dir: Path) -> list[Path]:
    """Sorted source images for a scene; handles 7-Scenes flat and TUM rgb/ layouts."""
    rgb = seq_dir / "rgb"
    root = rgb if rgb.is_dir() else seq_dir
    return sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in _IMG_EXTS and ".depth" not in p.name
    )


########## Keyframe prefix slicing (scaling sweep) ##########

def slice_keyframes(kf_file: Path, fraction: float, out_file: Path) -> Path:
    """Write the first ceil(fraction*N) keyframe paths of kf_file to out_file."""
    lines = kf_file.read_text().splitlines()
    n = max(1, round(fraction * len(lines)))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines[:n]))
    return out_file


def slam_max_frames_for_prefix(prefix_file: Path, seq_dir: Path) -> int:
    """max_frames for a SLAM prefix re-run: source index of the last prefix keyframe + 1.

    Keyframe selection is deterministic (optical-flow tracker state depends only on
    frames seen so far), so SLAM on the first max_frames source frames selects
    exactly the prefix keyframes.
    """
    all_imgs = [str(p) for p in list_scene_images(seq_dir)]
    last_kf = prefix_file.read_text().splitlines()[-1]
    return all_imgs.index(last_kf) + 1


########## Level-1 gates ##########

def check_gates(slam_metrics: dict, ours_metrics: dict) -> dict:
    """Spec §Level 1 gates: ATE |Δ| ≤ 5% rel or ≤5mm abs; loop counts equal."""
    slam_ate = slam_metrics.get("ate_rmse")
    ours_ate = ours_metrics.get("ate_rmse")
    if slam_ate is None or ours_ate is None:
        ate_delta, ate_pass = None, False
    else:
        ate_delta = abs(ours_ate - slam_ate)
        ate_pass = ate_delta <= max(ATE_REL_TOL * slam_ate, ATE_ABS_TOL)
    slam_loops = slam_metrics.get("loop_closures")
    ours_loops = ours_metrics.get("loops_applied")
    loops_pass = slam_loops is not None and slam_loops == ours_loops
    return {
        "ate_delta": ate_delta,
        "ate_pass": ate_pass,
        "slam_loops": slam_loops,
        "ours_loops": ours_loops,
        "loops_pass": loops_pass,
        "all_pass": ate_pass and loops_pass,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_lc_parity_common.py -v`
Expected: 10 PASS

- [ ] **Step 5: Commit**

```bash
git add evals/runners/lc_parity_common.py tests/evals/test_lc_parity_common.py
git commit -m "feat(evals): LC parity harness helpers — scene registry, prefix slicing, gates"
```

---

### Task 2: TUM ground-truth support in `ate_utils`

`compute_ate_rmse` currently loads GT from 7-Scenes per-frame `*.pose.txt` files. TUM scenes ship `groundtruth.txt` (TUM format, timestamps at ~100Hz — associate by nearest timestamp to each keyframe's filename timestamp).

**Files:**
- Modify: `evals/ate_utils.py` (function `_load_gt_as_tum_trajectory`, starts line 11)
- Test: `tests/evals/test_ate_utils_tum.py`

- [ ] **Step 1: Read `evals/ate_utils.py` fully** (small file) to see the exact 7-Scenes loading code and the evo trajectory type it returns. The task below extends, not replaces.

- [ ] **Step 2: Write the failing test**

```python
"""TUM groundtruth support in ate_utils."""
from pathlib import Path

import numpy as np

from evals.ate_utils import _load_gt_as_tum_trajectory


def _make_tum_scene(tmp_path: Path) -> Path:
    (tmp_path / "rgb").mkdir()
    # 3 keyframes at timestamps 10.0, 10.5, 11.0
    for ts in ("10.000000", "10.500000", "11.000000"):
        (tmp_path / "rgb" / f"{ts}.png").touch()
    # GT at slightly offset timestamps (nearest-neighbour association required)
    gt_lines = ["# ts tx ty tz qx qy qz qw"]
    for i, ts in enumerate((9.99, 10.51, 11.01)):
        gt_lines.append(f"{ts} {i}.0 0.0 0.0 0.0 0.0 0.0 1.0")
    (tmp_path / "groundtruth.txt").write_text("\n".join(gt_lines))
    return tmp_path


def test_tum_gt_loaded_and_associated(tmp_path):
    seq = _make_tum_scene(tmp_path)
    frames = sorted((seq / "rgb").iterdir())
    traj = _load_gt_as_tum_trajectory(seq, selected_frames=frames)
    assert traj.num_poses == 3
    # x positions 0,1,2 from the associated GT rows
    np.testing.assert_allclose(traj.positions_xyz[:, 0], [0.0, 1.0, 2.0])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_ate_utils_tum.py -v`
Expected: FAIL (7-Scenes pose-file loader finds no `*.pose.txt`)

- [ ] **Step 4: Implement — branch on `groundtruth.txt` presence**

Inside `_load_gt_as_tum_trajectory`, before the 7-Scenes pose-file logic, add:

```python
    # TUM layout: groundtruth.txt with 'ts tx ty tz qx qy qz qw' rows.
    # Associate each selected frame (filename stem = capture timestamp) with the
    # nearest GT row — reuses the parser conventions of evals/datasets.py:_read_tum_groundtruth.
    gt_file = seq_dir / "groundtruth.txt"
    if gt_file.exists():
        rows = []
        for line in gt_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            rows.append((float(vals[0]), np.array([float(v) for v in vals[1:8]])))
        gt_ts = np.array([r[0] for r in rows])
        stamps, xyz, quat = [], [], []
        for f in selected_frames:
            t = float(Path(f).stem)
            j = int(np.argmin(np.abs(gt_ts - t)))
            stamps.append(t)
            xyz.append(rows[j][1][0:3])
            quat.append(rows[j][1][3:7])  # qx qy qz qw
        # evo PoseTrajectory3D wants (qw, qx, qy, qz)
        quat_wxyz = np.array(quat)[:, [3, 0, 1, 2]]
        return PoseTrajectory3D(
            positions_xyz=np.array(xyz),
            orientations_quat_wxyz=quat_wxyz,
            timestamps=np.array(stamps),
        )
```

(Adjust the constructor call to match whatever trajectory type the existing 7-Scenes branch returns — read it in Step 1 and keep the return type identical.)

- [ ] **Step 5: Run test + existing ate tests to verify pass and no regression**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_ate_utils_tum.py tests/evals/ -v -k "ate"`
Expected: PASS, no regressions

- [ ] **Step 6: Commit**

```bash
git add evals/ate_utils.py tests/evals/test_ate_utils_tum.py
git commit -m "feat(evals): TUM groundtruth.txt support in ate_utils GT loader"
```

---

### Task 3: TUM image layout in `run_vggt_slam_lc.py`

The upstream reference runner globs images directly from `seq_dir`; TUM keeps them in `rgb/`. Reuse `list_scene_images` from Task 1.

**Files:**
- Modify: `evals/runners/run_vggt_slam_lc.py` (the image-collection block: `all_images = [...]` glob over `seq_dir`)
- Test: covered by Task 1's `test_list_scene_images_tum_layout` (the runner imports the helper; no GPU test here)

- [ ] **Step 1: Replace the inline glob with the shared helper**

In `run_vggt_slam_lc.py`, replace the `all_images = [...]` block (filters `.png/.jpg/.jpeg`, excludes depth) with:

```python
    from evals.runners.lc_parity_common import list_scene_images
    all_images = [str(p) for p in list_scene_images(seq_dir)]
```

Keep the existing `max_frames` truncation and logging lines unchanged. Note: the module loads `ate_utils` via `importlib` because `evals/` has import quirks — if `from evals.runners...` fails at runtime, use the same `importlib.util.spec_from_file_location` pattern already present at the top of the file for `ate_utils`.

- [ ] **Step 2: Verify the runner still parses and dry-checks on chess**

Run: `/opt/venv/reconstruction/bin/python -c "import importlib.util as u; s=u.spec_from_file_location('m','evals/runners/run_vggt_slam_lc.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print('import ok')"`
Expected: `import ok` (model download may be triggered by module import — if so, run on GPU box; import must not crash)

- [ ] **Step 3: Commit**

```bash
git add evals/runners/run_vggt_slam_lc.py
git commit -m "feat(evals): TUM rgb/ image layout support in VGGT-SLAM reference runner"
```

---

### Task 4: Loop count + per-candidate gate decisions in our metrics

Level-1 gate needs `loops_applied` from our pipeline; Level-2 needs per-candidate decisions (accept / reject-ratio / reject-jump / drop-no-poses) — the benchmark runs persisted no logs, so this must be structured output. `eval_gt.py` already has the inspection state (`base._lc_loop_submaps`, `base._lc_all_matches`) after a run.

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py` — set `match.reject_reason` at the three existing reject sites (metadata only, NO control-flow change)
- Modify: `evals/eval_gt.py` — `_run_condition` returns loop stats; `_save_outputs` writes them into metrics.json + `lc_decisions.json`
- Test: `tests/evals/test_lc_decisions.py`

- [ ] **Step 1: Write the failing test for the serializer**

```python
"""LC decision serialization for the parity harness."""
from types import SimpleNamespace

from evals.eval_gt import _serialize_lc_decisions


def test_serialize_lc_decisions():
    matches = [
        SimpleNamespace(similarity_score=0.41, query_submap_id=3, detected_submap_id=0,
                        query_frame_idx=2, detected_frame_idx=7,
                        accepted=True, reject_reason=None),
        SimpleNamespace(similarity_score=0.80, query_submap_id=3, detected_submap_id=1,
                        query_frame_idx=0, detected_frame_idx=1,
                        accepted=False, reject_reason="no_joint_poses"),
    ]
    out = _serialize_lc_decisions(matches)
    assert out[0]["accepted"] is True and out[0]["reject_reason"] is None
    assert out[1]["reject_reason"] == "no_joint_poses"
    assert out[1]["l2_score"] == 0.80
    # loops_applied = accepted count
    assert sum(d["accepted"] for d in out) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_lc_decisions.py -v`
Expected: FAIL — `_serialize_lc_decisions` not defined

- [ ] **Step 3: Implement**

(a) `wrappers.py` — at each existing reject site inside the `for match in loop_matches:` block, set a reason string right where the current `console.log("✗ ...")` calls happen (and initialize on accept):

```python
# verify-ratio reject branch (existing "✗ Loop rejected" for ratio):
match.reject_reason = "verify_ratio"
# no-joint-poses branch (existing "✗ Loop skipped (no joint poses)"):
match.reject_reason = "no_joint_poses"
# jump-ratio reject branch (existing "✗ Loop rejected (jump ratio=...)"):
match.reject_reason = "jump_ratio"
# accepted branch (existing match.accepted = True):
match.reject_reason = None
```

If `LoopMatch` is a frozen/named type that rejects attribute assignment, it already accepts `match.accepted = True` at the accept site — so plain attribute assignment works; mirror that.

(b) `eval_gt.py` — add near `_save_outputs`:

```python
def _serialize_lc_decisions(matches: list) -> list[dict]:
    """Flatten LoopMatch inspection objects to JSON rows for the parity harness."""
    return [{
        "l2_score": float(m.similarity_score),
        "query_submap": int(m.query_submap_id),
        "detected_submap": int(m.detected_submap_id),
        "query_frame": int(m.query_frame_idx),
        "detected_frame": int(m.detected_frame_idx),
        "accepted": bool(getattr(m, "accepted", False)),
        "reject_reason": getattr(m, "reject_reason", None),
    } for m in matches]
```

In `_run_condition`, after the pipeline runs (creator is the `LoopClosure` wrapper for lc/baseline conditions), collect:

```python
    base = getattr(creator, "base", None)
    lc_stats = None
    if base is not None and hasattr(base, "_lc_all_matches"):
        decisions = _serialize_lc_decisions(base._lc_all_matches)
        lc_stats = {
            "loops_applied": len(getattr(base, "_lc_loop_submaps", []) or []),
            "candidates": len(decisions),
            "decisions": decisions,
        }
```

and thread `lc_stats` through to `_save_outputs`, which writes `loops_applied` + `candidates` into the per-condition entry of `metrics.json` and the full decision list to `output_dir / f"lc_decisions_{cond}.json"`.

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_lc_decisions.py tests/evals/ -v`
Expected: new test PASS, no regressions

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py evals/eval_gt.py tests/evals/test_lc_decisions.py
git commit -m "feat(evals): loops_applied + per-candidate LC decisions in eval_gt metrics"
```

---

### Task 5: Scene-matrix driver `run_lc_parity.py`

**Files:**
- Create: `evals/runners/run_lc_parity.py`
- Test: `tests/evals/test_run_lc_parity.py` (argv construction only — no GPU)

- [ ] **Step 1: Write the failing test**

```python
"""Driver argv construction for the LC parity matrix (no GPU)."""
from pathlib import Path

from evals.runners.run_lc_parity import build_scene_commands


def test_build_scene_commands_levels_0_and_1(tmp_path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "lc_parity"
    seq = data_root / "7scenes/chess/chess/seq-01"
    seq.mkdir(parents=True)
    cmds = build_scene_commands("7s_chess", data_root, out_root)
    # Level 0 first: run_vggt_slam_lc with LC on, full sequence, paper config
    slam = cmds[0]
    assert "run_vggt_slam_lc.py" in slam[1]
    assert "--max_loops" in slam and slam[slam.index("--max_loops") + 1] == "1"
    assert "--min_disparity" in slam and slam[slam.index("--min_disparity") + 1] == "50"
    assert "--max_frames" not in slam  # full sequence — no cap
    # Level 1: eval_gt consumes the Level-0 keyframe list, spark backbone, both conditions
    ours = cmds[1]
    assert "eval_gt.py" in ours[1]
    assert ours[ours.index("--backbone") + 1] == "vggt_spark"
    kf = Path(ours[ours.index("--keyframe_list") + 1])
    assert kf == out_root / "7s_chess/slam/selected_frames.txt"
    assert "baseline" in ours and "lc" in ours


def test_prefix_commands_only_for_sweep_scenes(tmp_path):
    from evals.runners.run_lc_parity import build_prefix_commands
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    (data_root / "7scenes/office/office/seq-01").mkdir(parents=True)
    # non-sweep scene → no prefix runs
    assert build_prefix_commands("7s_chess", data_root, out_root, n_keyframes=100) == []
    # sweep scene → 2 prefixes (25, 50) × 2 pipelines
    cmds = build_prefix_commands("7s_office", data_root, out_root, n_keyframes=100)
    assert len(cmds) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_run_lc_parity.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the driver**

```python
#!/usr/bin/env python
"""LC parity matrix driver — Levels 0/1 + scaling prefixes, one scene at a time.

Serial by design (46 GB cgroup cap). Resumable: a run whose metrics.json exists
is skipped. Run inside tmux, never a notebook:

    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py            # full matrix
    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py --scenes 7s_chess
    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py --dry_run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lc_parity_common import (  # same-dir import, matches run_cross_model_benchmark style
    PREFIX_FRACTIONS,
    SCENES,
    slam_max_frames_for_prefix,
    slice_keyframes,
)

PY = "/opt/venv/reconstruction/bin/python"
RUNNERS = Path(__file__).resolve().parent
SLAM_RUNNER = RUNNERS / "run_vggt_slam_lc.py"
EVAL_GT = RUNNERS.parent / "eval_gt.py"

# Paper/parity config — spec §Constraints
PARITY = ["--submap_size", "16", "--max_loops", "1", "--min_disparity", "50",
          "--conf_threshold", "25.0", "--lc_thres", "0.95"]


def _slam_cmd(seq_dir: Path, out_dir: Path, max_frames: int | None = None) -> list[str]:
    """Level-0 upstream reference run (full sequence unless prefix-capped)."""
    cmd = [PY, str(SLAM_RUNNER), "--seq_dir", str(seq_dir),
           "--out_tum", str(out_dir / "slam.tum"), *PARITY]
    if max_frames is not None:
        cmd += ["--max_frames", str(max_frames)]
    return cmd


def _ours_cmd(scene_key: str, seq_dir: Path, kf_list: Path, out_dir: Path) -> list[str]:
    """Level-1 run: our pipeline on the exact SLAM keyframes, spark backbone."""
    dataset = SCENES[scene_key].dataset
    return [PY, str(EVAL_GT), "--dataset", dataset, "--seq_dir", str(seq_dir),
            "--backbone", "vggt_spark", "--conditions", "baseline", "lc",
            "--submap_size", "16", "--keyframe_list", str(kf_list),
            "--output_dir", str(out_dir), "--output_ate", str(out_dir / "ate.json")]


def build_scene_commands(scene_key: str, data_root: Path, out_root: Path) -> list[list[str]]:
    """Level 0 + Level 1 argv for one scene."""
    spec = SCENES[scene_key]
    seq_dir = data_root / spec.rel_seq_dir
    slam_dir = out_root / scene_key / "slam"
    ours_dir = out_root / scene_key / "ours"
    return [
        _slam_cmd(seq_dir, slam_dir),
        _ours_cmd(scene_key, seq_dir, slam_dir / "selected_frames.txt", ours_dir),
    ]


def build_prefix_commands(scene_key: str, data_root: Path, out_root: Path,
                          n_keyframes: int) -> list[list[str]]:
    """Scaling-sweep argv (25/50%) for sweep scenes; 100% is the main run."""
    spec = SCENES[scene_key]
    if not spec.scaling_sweep:
        return []
    seq_dir = data_root / spec.rel_seq_dir
    full_kf = out_root / scene_key / "slam" / "selected_frames.txt"
    cmds: list[list[str]] = []
    for frac in PREFIX_FRACTIONS[:-1]:  # (0.25, 0.50) — 1.0 is the main run
        tag = f"prefix_{int(frac * 100)}"
        pdir = out_root / scene_key / tag
        prefix_kf = pdir / "keyframes.txt"
        if full_kf.exists():  # slicing needs the Level-0 output; dry_run tolerates absence
            slice_keyframes(full_kf, frac, prefix_kf)
            max_frames = slam_max_frames_for_prefix(prefix_kf, seq_dir)
        else:
            max_frames = 0  # placeholder in dry runs before Level 0 has produced keyframes
        cmds.append(_slam_cmd(seq_dir, pdir / "slam", max_frames=max_frames))
        cmds.append(_ours_cmd(scene_key, seq_dir, prefix_kf, pdir / "ours"))
    return cmds


def _done(cmd: list[str]) -> bool:
    """Skip completed runs: metrics.json already present in the run's output dir."""
    for flag in ("--out_tum", "--output_dir"):
        if flag in cmd:
            out = Path(cmd[cmd.index(flag) + 1])
            out_dir = out.parent if flag == "--out_tum" else out
            return (out_dir / "metrics.json").exists()
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("evals/data"))
    ap.add_argument("--out_root", type=Path, default=Path("evals/baselines/lc_parity"))
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    for scene in args.scenes:
        cmds = build_scene_commands(scene, args.data_root, args.out_root)
        # Prefix sweeps depend on the Level-0 keyframe list, so build after Level 0 ran
        for cmd in cmds:
            _run_one(cmd, args.dry_run)
        full_kf = args.out_root / scene / "slam" / "selected_frames.txt"
        n_kf = len(full_kf.read_text().splitlines()) if full_kf.exists() else 0
        for cmd in build_prefix_commands(scene, args.data_root, args.out_root, n_kf):
            _run_one(cmd, args.dry_run)
    return 0


def _run_one(cmd: list[str], dry_run: bool) -> None:
    if _done(cmd):
        print(f"[skip] {' '.join(cmd[2:6])} — metrics.json exists")
        return
    print(f"[run ] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
```

Note for the test: `build_scene_commands`/`build_prefix_commands` must be importable without running — the test imports via `evals.runners.run_lc_parity`; add `evals/runners/__init__.py`-compatible import or use the same `importlib` loading pattern as `tests/evals/test_run_vggt_slam.py` (read that test first and mirror its import mechanism).

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_run_lc_parity.py -v`
Expected: PASS

- [ ] **Step 5: Dry-run the full matrix**

Run: `/opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py --dry_run`
Expected: 22 Level-0/1 command lines (11 scenes × 2) printed with correct paths; prefix lines print for the 4 sweep scenes with `max_frames 0` placeholders (no Level-0 outputs yet); no subprocess launched.

- [ ] **Step 6: Commit**

```bash
git add evals/runners/run_lc_parity.py tests/evals/test_run_lc_parity.py
git commit -m "feat(evals): LC parity scene-matrix driver (Levels 0/1 + scaling prefixes)"
```

---

### Task 6: Gate report `build_parity_table.py`

**Files:**
- Create: `evals/runners/build_parity_table.py`
- Test: `tests/evals/test_build_parity_table.py`

- [ ] **Step 1: Write the failing test**

```python
"""Parity table aggregation from per-scene metrics.json fixtures."""
import json
from pathlib import Path

from evals.runners.build_parity_table import build_table


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def test_build_table_pass_and_fail_rows(tmp_path):
    # scene A: parity holds
    _write(tmp_path / "7s_chess/slam/metrics.json",
           {"ate_rmse": 0.020, "loop_closures": 3, "keyframes": 100, "submaps": 7})
    _write(tmp_path / "7s_chess/ours/metrics.json",
           {"lc": {"ate": {"rmse": 0.021}, "loops_applied": 3}})
    # scene B: LC blow-up
    _write(tmp_path / "7s_office/slam/metrics.json",
           {"ate_rmse": 0.030, "loop_closures": 10, "keyframes": 300, "submaps": 20})
    _write(tmp_path / "7s_office/ours/metrics.json",
           {"lc": {"ate": {"rmse": 0.600}, "loops_applied": 4}})
    table = build_table(tmp_path)
    assert "7s_chess" in table and "7s_office" in table
    chess_row = next(l for l in table.splitlines() if "7s_chess" in l)
    office_row = next(l for l in table.splitlines() if "7s_office" in l)
    assert "PASS" in chess_row
    assert "FAIL" in office_row


def test_build_table_skips_missing_scenes(tmp_path):
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 0})
    # no ours/ dir yet — row marked pending, no crash
    table = build_table(tmp_path)
    assert "pending" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_build_parity_table.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Aggregate LC parity runs into a markdown gate table.

    /opt/venv/reconstruction/bin/python evals/runners/build_parity_table.py \
        --root evals/baselines/lc_parity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_parity_common import SCENES, check_gates  # same-dir import (see Task 5 note)


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def build_table(root: Path) -> str:
    """One row per scene: SLAM ref vs ours-LC with gate verdicts."""
    lines = [
        "| scene | kf | submaps | SLAM ATE | SLAM loops | ours ATE (lc) | ours loops | ΔATE | gates |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for scene in SCENES:
        slam = _load(root / scene / "slam" / "metrics.json")
        ours_all = _load(root / scene / "ours" / "metrics.json")
        if slam is None:
            continue
        if ours_all is None or "lc" not in ours_all:
            lines.append(f"| {scene} | {slam.get('keyframes','?')} | {slam.get('submaps','?')} "
                         f"| {slam.get('ate_rmse')} | {slam.get('loop_closures')} "
                         f"| pending | pending | — | — |")
            continue
        ours = {"ate_rmse": ours_all["lc"]["ate"]["rmse"],
                "loops_applied": ours_all["lc"].get("loops_applied")}
        g = check_gates(slam, ours)
        verdict = "PASS" if g["all_pass"] else "FAIL"
        delta = f"{g['ate_delta']:.4f}" if g["ate_delta"] is not None else "—"
        lines.append(f"| {scene} | {slam.get('keyframes','?')} | {slam.get('submaps','?')} "
                     f"| {slam['ate_rmse']:.4f} | {slam['loop_closures']} "
                     f"| {ours['ate_rmse']:.4f} | {ours['loops_applied']} "
                     f"| {delta} | {verdict} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("evals/baselines/lc_parity"))
    args = ap.parse_args()
    table = build_table(args.root)
    out = args.root / "_parity_table.md"
    out.write_text(table + "\n")
    print(table)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Match the metrics.json schema `eval_gt._save_outputs` actually writes (per-condition dict with `ate.rmse`; Task 4 adds `loops_applied`) — read `_save_outputs` and adjust key paths if they differ from `ours_all["lc"]["ate"]["rmse"]`.

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_build_parity_table.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/runners/build_parity_table.py tests/evals/test_build_parity_table.py
git commit -m "feat(evals): LC parity gate table aggregator"
```

---

### Task 7: Scene download script

**Files:**
- Create: `evals/data/download_parity_scenes.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Download the LC-parity validation scenes (spec 2026-07-08-lc-parity-validation-design).
# Idempotent: skips anything already extracted. ~25 GB total — check disk first.
# 7-Scenes: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
# TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
set -euo pipefail
cd "$(dirname "$0")"

SEVEN_SCENES_BASE="https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
for scene in fire heads office pumpkin redkitchen stairs; do   # chess already local
    if [ -d "7scenes/${scene}/${scene}/seq-01" ]; then
        echo "[skip] 7scenes/${scene}"
        continue
    fi
    mkdir -p "7scenes/${scene}"
    echo "[get ] 7-Scenes ${scene}"
    wget -c "${SEVEN_SCENES_BASE}/${scene}.zip" -O "7scenes/${scene}.zip"
    unzip -q -o "7scenes/${scene}.zip" -d "7scenes/${scene}"
    # scene zips contain per-seq zips; extract seq-01 only (parity uses seq-01)
    unzip -q -o "7scenes/${scene}/${scene}/seq-01.zip" -d "7scenes/${scene}/${scene}"
done

TUM_BASE="https://cvg.cit.tum.de/rgbd/dataset"
declare -A TUM=(
    [freiburg1/rgbd_dataset_freiburg1_desk]=fr1_desk
    [freiburg1/rgbd_dataset_freiburg1_room]=fr1_room
    [freiburg2/rgbd_dataset_freiburg2_xyz]=fr2_xyz
    [freiburg3/rgbd_dataset_freiburg3_long_office_household]=fr3_office
)
mkdir -p tum
for path in "${!TUM[@]}"; do
    name="$(basename "${path}")"
    if [ -d "tum/${name}" ]; then
        echo "[skip] tum/${name}"
        continue
    fi
    echo "[get ] TUM ${TUM[$path]}"
    wget -c "${TUM_BASE}/${path}.tgz" -O "tum/${name}.tgz"
    tar -xzf "tum/${name}.tgz" -C tum/
done

echo "Done. Verify: ls 7scenes/*/*/seq-01 | head; ls tum/"
```

- [ ] **Step 2: Syntax check**

Run: `bash -n evals/data/download_parity_scenes.sh`
Expected: no output (clean parse)

- [ ] **Step 3: Commit**

```bash
git add evals/data/download_parity_scenes.sh
git commit -m "feat(evals): parity scene download script (7-Scenes + TUM)"
```

Note: actual download (~25 GB) happens at execution time in tmux, not during this task. If a Microsoft URL 404s, get the current per-scene links from the 7-Scenes project page and update the script; TUM URLs are long-term stable. Downloaded data is gitignored (`evals/data/`); archive to collab-data rclone after the matrix completes.

---

### Task 8: Level-2 stage-trace — composed loop-edge comparison

The retrieval/verify dumps come free from Task 4 (`lc_decisions_*.json`) and the existing SLAM-side `vggt_spark_similarity.json`. The missing piece is comparing loop-edge *constraints*: ours (single direct edge) vs upstream's composed chain.

**Files:**
- Create: `evals/runners/compare_loop_edges.py`
- Test: `tests/evals/test_compare_loop_edges.py`

- [ ] **Step 1: Write the failing test**

```python
"""Composed-chain loop-edge comparison (Level 2, spec §stage-trace item 3)."""
import numpy as np

from evals.runners.compare_loop_edges import compose_slam_chain, edge_divergence


def test_compose_slam_chain_matches_direct_relative():
    """Composing SLAM's 3-constraint chain q→LC0→LC1→d must equal the direct
    q→d relative in the graph's H_inner convention, when scales are identity."""
    rng = np.random.default_rng(7)

    def rand_se3():
        from scipy.spatial.transform import Rotation as R
        M = np.eye(4)
        M[:3, :3] = R.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        M[:3, 3] = rng.normal(size=3)
        return M

    P0, P1 = np.eye(4), rand_se3()      # LC-run poses (w2c), frame0 at origin
    h_rel_a = np.eye(4)                  # query→LC0 anchor (identical image, scale=I)
    h_inner = P0 @ np.linalg.inv(P1)     # LC0→LC1 from the LC VGGT run
    h_rel_b = np.eye(4)                  # LC1→detected anchor
    composed = compose_slam_chain(h_rel_a, h_inner, h_rel_b)
    np.testing.assert_allclose(composed, P0 @ np.linalg.inv(P1), atol=1e-12)


def test_edge_divergence_flags_inverted_edge():
    from scipy.spatial.transform import Rotation as R
    P1 = np.eye(4)
    P1[:3, :3] = R.from_euler("z", 30, degrees=True).as_matrix()
    P1[:3, 3] = [1.0, 0.0, 0.0]
    correct = np.eye(4) @ np.linalg.inv(P1)     # P0=I convention
    ours_current = np.linalg.inv(np.eye(4)) @ P1  # the confirmed-wrong formula
    d = edge_divergence(correct, ours_current)
    assert d["rot_deg"] > 10 and d["trans"] > 0.1  # far apart
    assert edge_divergence(correct, correct)["rot_deg"] < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_compare_loop_edges.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python
"""Level-2 loop-edge trace: compare our direct loop edges to VGGT-SLAM's composed chain.

SLAM inserts each loop as a 2-frame LC submap with 3 constraints
(query→LC0 scaled anchor, LC0→LC1 inner, LC1→detected scaled anchor);
ours is a single direct query→detected edge. Composing SLAM's chain yields
the equivalent direct relative — diffing the two quantifies the edge defects
(inversion, missing scale) per loop, per scene.
"""
from __future__ import annotations

import numpy as np


def compose_slam_chain(h_rel_a: np.ndarray, h_inner: np.ndarray,
                       h_rel_b: np.ndarray) -> np.ndarray:
    """Equivalent direct query→detected relative from SLAM's 3-edge chain."""
    return h_rel_a @ h_inner @ h_rel_b


def edge_divergence(h_ref: np.ndarray, h_test: np.ndarray) -> dict:
    """Rotation (deg) + translation (norm) gap between two relative constraints."""
    d = np.linalg.inv(h_ref) @ h_test
    # rotation angle from the closest-rotation part of the 3x3 block
    u, _, vt = np.linalg.svd(d[:3, :3])
    r = u @ vt
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "rot_deg": float(np.degrees(np.arccos(cos))),
        "trans": float(np.linalg.norm(d[:3, 3])),
        "det_ratio": float(np.linalg.det(h_test) / np.linalg.det(h_ref)),  # scale proxy
    }
```

(A CLI that loads per-run edge dumps can be added when the fix work starts; Level 2 needs the helpers + tests locked now so the fix PR can assert `edge_divergence ≈ 0` post-fix.)

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_compare_loop_edges.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/runners/compare_loop_edges.py tests/evals/test_compare_loop_edges.py
git commit -m "feat(evals): loop-edge composed-chain divergence helpers (Level 2)"
```

---

### Task 9: Chess integration smoke (GPU, tmux)

Validates the whole harness end-to-end on the one scene already local, including the prefix mechanism, before the ~25 GB download and the full matrix.

**Files:** none created — execution + verification only.

- [ ] **Step 1: Run Level 0 + Level 1 on chess (tmux)**

```bash
tmux new -s lc_parity -d
tmux send-keys -t lc_parity \
  "/opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py --scenes 7s_chess 2>&1 | tee /tmp/lc_parity_chess.log" Enter
```
Monitor with `tmux capture-pane -t lc_parity -p | tail -20`. Full chess seq-01 (1000 frames, disparity-50 keyframing) — expect O(1–2 h).

- [ ] **Step 2: Verify artifacts**

```bash
ls evals/baselines/lc_parity/7s_chess/slam/    # slam.tum, selected_frames.txt, metrics.json
ls evals/baselines/lc_parity/7s_chess/ours/    # metrics.json, ate.json, lc_decisions_lc.json, plots/
cat evals/baselines/lc_parity/7s_chess/slam/metrics.json
```
Expected: SLAM metrics show `loop_closures ≥ 1` on the full sequence; ours metrics contain `loops_applied` and per-candidate decisions; `plots/trajectory.png` shows ours + GT 3D overlay.

- [ ] **Step 3: Verify expected pre-fix behavior (this is the harness working, not failing)**

```bash
/opt/venv/reconstruction/bin/python evals/runners/build_parity_table.py
```
Expected: chess row present. Baseline ATE gate should PASS; `lc` condition expected FAIL with `loops_applied=0` and `lc_decisions_lc.json` showing `reject_reason: "no_joint_poses"` on every verified candidate — the first empirical log-confirmation of the no-op root cause.

- [ ] **Step 4: Exercise the prefix mechanism once on chess**

Temporarily run with chess forced into the sweep (no code change — pass prefix fractions via a manual invocation):
```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
from pathlib import Path
import subprocess
from evals.runners.lc_parity_common import slice_keyframes, slam_max_frames_for_prefix
root = Path("evals/baselines/lc_parity/7s_chess")
seq = Path("evals/data/7scenes/chess/chess/seq-01")
kf = slice_keyframes(root / "slam/selected_frames.txt", 0.25, root / "prefix_25/keyframes.txt")
print("prefix keyframes:", len(kf.read_text().splitlines()))
print("slam max_frames:", slam_max_frames_for_prefix(kf, seq))
EOF
```
Expected: prints ~25% of the full keyframe count and a plausible max_frames; confirms slicing + index mapping on real data.

- [ ] **Step 5: Commit results + note**

```bash
git add evals/baselines/lc_parity/7s_chess/*/metrics.json evals/baselines/lc_parity/_parity_table.md
git commit -m "eval(lc-parity): chess integration smoke — harness end-to-end, no-op cause log-confirmed"
```
(Only small JSON/md artifacts — TUM files, npz, plots stay gitignored per existing baselines convention; check `.gitignore` handling of `evals/baselines/` and force-add metrics.json if needed, matching `evals/baselines/cross_model/` precedent.)

---

### Task 10: Full matrix execution (background phase, after Task 9 review)

**Files:** none — execution guidance.

- [ ] **Step 1: Download scenes** — `bash evals/data/download_parity_scenes.sh` (tmux; verify ~25 GB free first with `df -h /workspace`).
- [ ] **Step 2: Run the matrix** — `tmux send-keys -t lc_parity "/opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py 2>&1 | tee /tmp/lc_parity_full.log" Enter`. Serial; resumable on interruption (skips completed runs). Expect a day-scale job.
- [ ] **Step 3: Build report** — `build_parity_table.py`; commit metrics + table as in Task 9.
- [ ] **Step 4: Write results doc** — `docs/superpowers/specs/2026-07-XX-lc-parity-results.md` mirroring the cross-model results doc format: gate table, scaling curves (ATE vs prefix %), loop counts, per-stage divergence findings, and the go/no-go for starting the fix work (spec Phase D).

---

## Self-Review Notes

- **Spec coverage:** Level 0 (Tasks 3, 5, 7), Level 1 + gates (Tasks 1, 4, 5, 6), scaling sweep (Tasks 1, 5, 9.4), Level 2 stage-trace (Tasks 4, 8), keyframe-transfer exactness (existing `--keyframe_list` + Task 1 index mapping test), gate-decision logging (Task 4), chess-first integration (Task 9), full matrix (Task 10). TUM support (Tasks 2, 3, 7). ✓
- **No LC logic changes:** only `reject_reason` metadata in wrappers (Task 4), matching the spec's allowance. ✓
- **Type consistency:** `SceneSpec.dataset` feeds `eval_gt --dataset`; `check_gates` consumes the exact metrics.json keys written by `run_vggt_slam_lc.py` (`ate_rmse`, `loop_closures`) and Task 4 (`loops_applied`). Task 6 rechecks the ours-side schema against `_save_outputs` at implementation time (flagged inline). ✓
- **Known risks:** (1) 7-Scenes download URLs may rot — Task 7 notes the fallback. (2) evals import mechanics are quirky (`importlib` patterns) — Tasks 3/5 flag mirroring the existing test/module import style rather than assuming package imports work.
