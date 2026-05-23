# Ground-Truth Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end evaluation harness that downloads 7-Scenes, runs VGGTXCreator under 3 conditions (baseline / +BA / +LC), and computes ATE + RPE against ground-truth poses.

**Architecture:** `evals/datasets.py` provides `get_dataset()` registry + `EvalDataset`; `collab_splats/pointcloud/loop_closure/eval.py` stubs are filled with `umeyama_align`, `ate_translation`, `rpe`; `evals/eval_gt.py` orchestrates per-condition inference + metrics; a notebook loads saved results for visualization.

**Tech Stack:** Python 3.10 (nerfstudio env), numpy, matplotlib, VGGTXCreator (vggtx backend), 7-Scenes dataset

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `evals/download_7scenes.sh` | Create | wget helper for 7-Scenes sequences |
| `evals/datasets.py` | Create | `EvalDataset` dataclass, `_load_7scenes`, `get_dataset()` |
| `tests/evals/test_datasets.py` | Create | Unit tests for datasets.py |
| `collab_splats/pointcloud/loop_closure/eval.py` | Modify (lines 121–168) | Fill `umeyama_align`, `ate_translation`, `rpe` stubs |
| `tests/pointcloud/test_eval_metrics.py` | Create | Unit tests for the three metric functions |
| `evals/eval_gt.py` | Create | CLI runner: loads dataset, runs 3 conditions, writes outputs |
| `tests/evals/test_eval_gt_helpers.py` | Create | Unit tests for runner helper functions |
| `docs/pointcloud/eval_7scenes_gt.ipynb` | Create | Load-only visualization notebook |

---

## Task 1: Download Script

**Files:**
- Create: `evals/download_7scenes.sh`

- [ ] **Step 1: Create the evals directory and download script**

```bash
mkdir -p /workspace/collab-splats/evals
```

Create `evals/download_7scenes.sh`:

```bash
#!/usr/bin/env bash
# Download a single 7-Scenes sequence.
# Usage: bash evals/download_7scenes.sh chess ./data/7scenes
# Scenes: chess fire heads office pumpkin redkitchen stairs
set -euo pipefail

SCENE="${1:-chess}"
DEST="${2:-./data/7scenes}"

declare -A URLS=(
    [chess]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/chess.zip"
    [fire]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/fire.zip"
    [heads]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/heads.zip"
    [office]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/office.zip"
    [pumpkin]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/pumpkin.zip"
    [redkitchen]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/redkitchen.zip"
    [stairs]="https://download.microsoft.com/download/2/8/5/285360d1-f16a-41fe-be72-45d4b3c72011/stairs.zip"
)

if [[ -z "${URLS[$SCENE]+x}" ]]; then
    echo "Unknown scene '$SCENE'. Choose from: ${!URLS[@]}"
    exit 1
fi

mkdir -p "$DEST"
ZIP="$DEST/$SCENE.zip"
echo "Downloading $SCENE → $ZIP"
wget -c "${URLS[$SCENE]}" -O "$ZIP"
echo "Extracting..."
unzip -q "$ZIP" -d "$DEST/$SCENE"
echo "Done. Data at $DEST/$SCENE/"
echo ""
echo "Expected structure:"
echo "  $DEST/$SCENE/seq-01/color/frame-000000.color.png"
echo "  $DEST/$SCENE/seq-01/pose/frame-000000.pose.txt"
```

- [ ] **Step 2: Make executable and verify help text**

```bash
chmod +x /workspace/collab-splats/evals/download_7scenes.sh
bash /workspace/collab-splats/evals/download_7scenes.sh badscene 2>&1 | grep "Unknown scene"
```

Expected: `Unknown scene 'badscene'`

- [ ] **Step 3: Commit**

```bash
git add evals/download_7scenes.sh
git commit -m "feat(evals): add 7-Scenes download helper script"
```

---

## Task 2: `evals/datasets.py`

**Files:**
- Create: `evals/datasets.py`
- Create: `tests/evals/__init__.py`
- Create: `tests/evals/test_datasets.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evals/__init__.py` (empty).

Create `tests/evals/test_datasets.py`:

```python
import numpy as np
import pytest
from pathlib import Path


def _make_seq(tmp_path: Path, n_frames: int, poses: list[np.ndarray] | None = None) -> Path:
    """Create a minimal synthetic 7-Scenes sequence directory."""
    color_dir = tmp_path / "color"
    pose_dir = tmp_path / "pose"
    color_dir.mkdir()
    pose_dir.mkdir()
    for i in range(n_frames):
        (color_dir / f"frame-{i:06d}.color.png").touch()
        p = poses[i] if poses else np.eye(4, dtype=np.float64)
        np.savetxt(pose_dir / f"frame-{i:06d}.pose.txt", p)
    return tmp_path


def test_get_dataset_unknown_raises():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    with pytest.raises(KeyError, match="unknown dataset"):
        get_dataset("nonexistent")


def test_load_7scenes_count(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_seq(tmp_path, n_frames=10)
    dataset = get_dataset("7scenes")(tmp_path, max_frames=10)
    assert len(dataset.images) == 10
    assert dataset.gt_poses.shape == (10, 4, 4)


def test_load_7scenes_max_frames(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_seq(tmp_path, n_frames=10)
    dataset = get_dataset("7scenes")(tmp_path, max_frames=5)
    assert len(dataset.images) == 5
    assert dataset.gt_poses.shape == (5, 4, 4)


def test_load_7scenes_pose_inverted(tmp_path):
    """7-Scenes poses are cam-to-world; loader must invert to world-to-cam."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 3] = [1.0, 2.0, 3.0]
    _make_seq(tmp_path, n_frames=1, poses=[cam_to_world])
    dataset = get_dataset("7scenes")(tmp_path, max_frames=1)
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 3] = [-1.0, -2.0, -3.0]
    np.testing.assert_allclose(dataset.gt_poses[0], expected, atol=1e-5)


def test_load_7scenes_dtype(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from datasets import get_dataset
    _make_seq(tmp_path, n_frames=3)
    dataset = get_dataset("7scenes")(tmp_path)
    assert dataset.gt_poses.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_datasets.py -v 2>&1 | tail -20
```

Expected: ImportError or ModuleNotFoundError (datasets.py doesn't exist yet)

- [ ] **Step 3: Implement `evals/datasets.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalDataset:
    images: list[Path]
    gt_poses: np.ndarray   # (N, 4, 4) world-to-cam float32


def _load_7scenes(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    seq_dir = Path(seq_dir)
    images = sorted((seq_dir / "color").glob("*.color.png"))[:max_frames]
    poses = np.stack([
        np.linalg.inv(
            np.loadtxt(seq_dir / "pose" / f"{p.stem.split('.')[0]}.pose.txt")
        )
        for p in images
    ]).astype(np.float32)
    return EvalDataset(images=images, gt_poses=poses)


_REGISTRY: dict[str, callable] = {
    "7scenes": _load_7scenes,
}


def get_dataset(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_datasets.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add evals/datasets.py tests/evals/__init__.py tests/evals/test_datasets.py
git commit -m "feat(evals): add EvalDataset + get_dataset() registry with 7-Scenes loader"
```

---

## Task 3: Fill `eval.py` Metric Stubs

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/eval.py` (lines 121–168)
- Create: `tests/pointcloud/test_eval_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pointcloud/test_eval_metrics.py`:

```python
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


def _make_poses(translations: np.ndarray) -> np.ndarray:
    """Build (N, 4, 4) world-to-cam poses with identity rotation and given translations."""
    N = len(translations)
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, :3, 3] = translations.astype(np.float32)
    return poses


def test_umeyama_align_identity():
    """When pred == gt, aligned_pred should equal pred (up to float tolerance)."""
    from collab_splats.pointcloud.loop_closure.eval import umeyama_align
    gt = _make_poses(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32))
    aligned, T = umeyama_align(gt.copy(), gt)
    np.testing.assert_allclose(aligned, gt, atol=1e-4)


def test_umeyama_align_pure_translation():
    """Pred shifted by constant offset; aligned_pred should match gt after alignment."""
    from collab_splats.pointcloud.loop_closure.eval import umeyama_align
    translations = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    gt = _make_poses(translations)
    # Pred is gt shifted by (5, 0, 0) in world space: cam_pos_world = R^T(-t)
    # For identity R, cam_pos_world = -t, so shifting t by -5 shifts cam_pos by +5
    shifted_t = translations.copy()
    shifted_t[:, 0] += 5.0  # shift cam translation component
    pred = _make_poses(-shifted_t)  # world-to-cam with shifted t
    aligned, _ = umeyama_align(pred, gt)
    # After alignment, camera positions should match gt
    def cam_pos(poses):
        R_ = poses[:, :3, :3]
        t_ = poses[:, :3, 3]
        return np.einsum("nij,nj->ni", R_.transpose(0, 2, 1), -t_)
    np.testing.assert_allclose(cam_pos(aligned), cam_pos(gt), atol=1e-3)


def test_ate_translation_perfect():
    """When pred == gt (after alignment), ATE RMSE should be near zero."""
    from collab_splats.pointcloud.loop_closure.eval import ate_translation
    gt = _make_poses(np.random.RandomState(0).randn(10, 3).astype(np.float32))
    result = ate_translation(gt.copy(), gt)
    assert result["rmse"] < 1e-4
    assert result["mean"] < 1e-4
    assert "per_frame" in result
    assert len(result["per_frame"]) == 10


def test_ate_translation_returns_correct_keys():
    from collab_splats.pointcloud.loop_closure.eval import ate_translation
    gt = _make_poses(np.zeros((5, 3), dtype=np.float32))
    result = ate_translation(gt.copy(), gt)
    assert set(result.keys()) >= {"rmse", "mean", "median", "max", "per_frame"}


def test_rpe_perfect():
    """When pred == gt, RPE trans and rot RMSE should be near zero."""
    from collab_splats.pointcloud.loop_closure.eval import rpe
    gt = _make_poses(np.linspace([0, 0, 0], [5, 0, 0], 10).astype(np.float32))
    result = rpe(gt.copy(), gt, delta=1)
    assert result["trans_rmse"] < 1e-4
    assert result["rot_rmse_deg"] < 1e-3


def test_rpe_returns_correct_keys():
    from collab_splats.pointcloud.loop_closure.eval import rpe
    gt = _make_poses(np.zeros((5, 3), dtype=np.float32))
    result = rpe(gt.copy(), gt)
    assert set(result.keys()) == {"trans_rmse", "rot_rmse_deg"}


def test_ate_nonzero_error():
    """Pred shifted uniformly; after Umeyama alignment, ATE should be near zero (translation-only drift is correctable)."""
    from collab_splats.pointcloud.loop_closure.eval import ate_translation
    gt_t = np.linspace([0, 0, 0], [4, 0, 0], 5).astype(np.float32)
    gt = _make_poses(gt_t)
    # Pred has a constant offset — Umeyama removes it, ATE → 0
    pred = _make_poses(gt_t + np.array([10, 0, 0], dtype=np.float32))
    result = ate_translation(pred, gt)
    assert result["rmse"] < 1e-3


def test_rpe_nonzero_translation_error():
    """Pred with accumulated drift has nonzero RPE even though ATE ~0 after alignment."""
    from collab_splats.pointcloud.loop_closure.eval import rpe
    # GT: uniform 1m steps; pred: steps grow by 0.1m each frame (drift)
    gt_steps = np.ones((9, 3), dtype=np.float32)
    gt_steps[:, 1:] = 0
    pred_steps = gt_steps.copy()
    pred_steps[:, 0] += np.arange(9, dtype=np.float32) * 0.1
    gt_t = np.vstack([[0, 0, 0], np.cumsum(gt_steps, axis=0)]).astype(np.float32)
    pred_t = np.vstack([[0, 0, 0], np.cumsum(pred_steps, axis=0)]).astype(np.float32)
    gt = _make_poses(gt_t)
    pred = _make_poses(pred_t)
    result = rpe(pred, gt)
    assert result["trans_rmse"] > 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_eval_metrics.py -v 2>&1 | tail -20
```

Expected: all FAILED with `NotImplementedError: Implement once GT pose dataset available`

- [ ] **Step 3: Fill the three stubs in `eval.py`**

Replace lines 121–168 in `collab_splats/pointcloud/loop_closure/eval.py`. Read the file first to confirm line numbers, then apply the following replacements:

Replace the `umeyama_align` body (the `raise NotImplementedError(...)` block):

```python
def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align predicted trajectory to ground truth via Umeyama SE(3).

    Args:
        pred: (N, 4, 4) predicted world-to-cam poses
        gt:   (N, 4, 4) ground-truth world-to-cam poses

    Returns:
        (aligned_pred, T_align): Umeyama-aligned predicted poses and the SE(3)
        transform applied.
    """
    from .alignment import umeyama_se3
    # Camera positions in world: for world-to-cam T, p_world = R^T @ (-t)
    R_pred = pred[:, :3, :3]
    t_pred = pred[:, :3, 3]
    p_pred = np.einsum("nij,nj->ni", R_pred.transpose(0, 2, 1), -t_pred)  # (N, 3)

    R_gt = gt[:, :3, :3]
    t_gt = gt[:, :3, 3]
    p_gt = np.einsum("nij,nj->ni", R_gt.transpose(0, 2, 1), -t_gt)  # (N, 3)

    # T_align: (4,4) such that p_gt ≈ T_align @ p_pred
    T_align = umeyama_se3(source=p_pred, target=p_gt)
    # Apply to full poses: aligned[i] = pred[i] @ inv(T_align)
    T_inv = np.linalg.inv(T_align)
    aligned = pred @ T_inv[None]   # (N, 4, 4)
    return aligned.astype(np.float32), T_align
```

Replace the `ate_translation` body:

```python
def ate_translation(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Absolute Trajectory Error on translation component.

    Args:
        pred: (N, 4, 4) predicted poses
        gt:   (N, 4, 4) ground-truth poses

    Returns:
        {'rmse', 'mean', 'median', 'max', 'per_frame'} after Umeyama alignment.
    """
    aligned, _ = umeyama_align(pred, gt)
    # Camera positions in world
    def _cam_pos(poses):
        R_ = poses[:, :3, :3]
        t_ = poses[:, :3, 3]
        return np.einsum("nij,nj->ni", R_.transpose(0, 2, 1), -t_)

    errs = np.linalg.norm(_cam_pos(aligned) - _cam_pos(gt), axis=1)
    return {
        "rmse":      float(np.sqrt((errs ** 2).mean())),
        "mean":      float(errs.mean()),
        "median":    float(np.median(errs)),
        "max":       float(errs.max()),
        "per_frame": errs,
    }
```

Replace the `rpe` body:

```python
def rpe(pred: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Relative Pose Error at frame stride delta.

    Args:
        pred:  (N, 4, 4) predicted poses
        gt:    (N, 4, 4) ground-truth poses
        delta: frame stride between compared pose pairs

    Returns:
        {'trans_rmse', 'rot_rmse_deg'} relative pose error statistics.
    """
    rel_pred = np.linalg.inv(pred[:-delta]) @ pred[delta:]   # (N-δ, 4, 4)
    rel_gt   = np.linalg.inv(gt[:-delta])   @ gt[delta:]     # (N-δ, 4, 4)
    err = np.linalg.inv(rel_gt) @ rel_pred                   # (N-δ, 4, 4)

    t_err = np.linalg.norm(err[:, :3, 3], axis=1)
    cos_angle = np.clip(
        (np.trace(err[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )
    r_err_deg = np.degrees(np.arccos(cos_angle))

    return {
        "trans_rmse":   float(np.sqrt((t_err ** 2).mean())),
        "rot_rmse_deg": float(np.sqrt((r_err_deg ** 2).mean())),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_eval_metrics.py -v
```

Expected: 8 PASSED

- [ ] **Step 5: Run full pointcloud test suite to check no regressions**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v 2>&1 | tail -30
```

Expected: same pass/fail counts as before (4 pre-existing failures unchanged)

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/eval.py tests/pointcloud/test_eval_metrics.py
git commit -m "feat(eval): implement umeyama_align, ate_translation, rpe metric stubs"
```

---

## Task 4: `evals/eval_gt.py` Runner

**Files:**
- Create: `evals/eval_gt.py`
- Create: `tests/evals/test_eval_gt_helpers.py`

- [ ] **Step 1: Write failing tests for helper functions**

Create `tests/evals/test_eval_gt_helpers.py`:

```python
import json
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))


def _make_poses(N=5):
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, 0, 3] = np.arange(N, dtype=np.float32)
    return poses


def test_prepare_image_dir_creates_symlinks(tmp_path):
    from eval_gt import _prepare_image_dir
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    images = []
    for i in range(3):
        p = src_dir / f"frame-{i:06d}.png"
        p.touch()
        images.append(p)
    result = _prepare_image_dir(images)
    try:
        links = sorted(result.iterdir())
        assert len(links) == 3
        assert all(l.is_symlink() for l in links)
        assert links[0].name == "000000.png"
        assert links[2].name == "000002.png"
    finally:
        shutil.rmtree(result)


def test_prepare_image_dir_ordered_by_index(tmp_path):
    """Symlinks are named 000000.png..N so VGGTXCreator sees correct frame order."""
    from eval_gt import _prepare_image_dir
    images = [tmp_path / f"z_{i}.png" for i in range(5)]
    for p in images:
        p.touch()
    result = _prepare_image_dir(images)
    try:
        names = sorted(p.name for p in result.iterdir())
        assert names == [f"{i:06d}.png" for i in range(5)]
    finally:
        shutil.rmtree(result)


def test_cam_positions_identity():
    """Identity world-to-cam → camera at origin."""
    from eval_gt import _cam_positions
    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    pos = _cam_positions(poses)
    np.testing.assert_allclose(pos, np.zeros((3, 3)), atol=1e-6)


def test_cam_positions_translation():
    """world-to-cam with t=(1,2,3) → cam at (-1,-2,-3) in world (identity R)."""
    from eval_gt import _cam_positions
    poses = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1))
    poses[0, :3, 3] = [1.0, 2.0, 3.0]
    pos = _cam_positions(poses)
    np.testing.assert_allclose(pos[0], [-1.0, -2.0, -3.0], atol=1e-6)


def test_save_outputs_writes_files(tmp_path):
    from eval_gt import _save_outputs
    gt = _make_poses(10)
    trajectories = {"gt": gt, "baseline": gt.copy(), "ba": gt.copy()}
    metrics = {
        "baseline": {"ate": {"rmse": 0.1, "mean": 0.09, "median": 0.08, "max": 0.2}, "rpe": {"trans_rmse": 0.01, "rot_rmse_deg": 0.5}},
        "ba":       {"ate": {"rmse": 0.05, "mean": 0.04, "median": 0.03, "max": 0.1}, "rpe": {"trans_rmse": 0.005, "rot_rmse_deg": 0.2}},
    }
    _save_outputs(metrics, trajectories, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "trajectories.npz").exists()
    assert (tmp_path / "plots" / "trajectory.png").exists()
    assert (tmp_path / "plots" / "ate_per_frame.png").exists()
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert "baseline" in loaded
    assert "rmse" in loaded["baseline"]["ate"]


def test_save_outputs_no_per_frame_in_json(tmp_path):
    """per_frame array must not appear in metrics.json (not JSON-serializable)."""
    from eval_gt import _save_outputs
    from collab_splats.pointcloud.loop_closure.eval import ate_translation
    gt = _make_poses(10)
    metrics = {
        "baseline": {
            "ate": ate_translation(gt.copy(), gt),
            "rpe": {"trans_rmse": 0.0, "rot_rmse_deg": 0.0},
        }
    }
    trajectories = {"gt": gt, "baseline": gt}
    _save_outputs(metrics, trajectories, tmp_path)
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    assert "per_frame" not in loaded["baseline"]["ate"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v 2>&1 | tail -15
```

Expected: ImportError (eval_gt.py doesn't exist yet)

- [ ] **Step 3: Implement `evals/eval_gt.py`**

```python
#!/usr/bin/env python
"""Ground-truth evaluation runner for collab-splats BA/LC pipelines.

Usage:
    python evals/eval_gt.py \\
        --dataset   7scenes \\
        --seq_dir   /data/7scenes/chess/seq-01 \\
        --output_dir ./eval_results/chess_seq01 \\
        --max_frames 500 \\
        --conditions baseline ba lc
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import get_dataset

from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.loop_closure.eval import ate_translation, rpe

_CONDITIONS: dict[str, dict] = {
    "baseline": {"use_ba": False, "enable_loop_closure": False},
    "ba":       {"use_ba": True,  "enable_loop_closure": False},
    "lc":       {"use_ba": False, "enable_loop_closure": True},
}

_COLORS = {"gt": "black", "baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green"}


def _prepare_image_dir(image_paths: list[Path]) -> Path:
    """Symlink image_paths into a fresh temp dir named 000000.png, 000001.png, ..."""
    tmp = Path(tempfile.mkdtemp(prefix="collab_eval_"))
    for i, src in enumerate(image_paths):
        (tmp / f"{i:06d}.png").symlink_to(src.resolve())
    return tmp


def _cam_positions(poses: np.ndarray) -> np.ndarray:
    """Convert (N,4,4) world-to-cam poses to (N,3) camera positions in world."""
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def _run_condition(name: str, image_dir: Path, output_dir: Path) -> np.ndarray:
    """Instantiate VGGTXCreator with condition flags, run, return (N,4,4) extrinsics."""
    flags = _CONDITIONS[name]
    creator = get_creator("vggtx")(**flags)
    creator.reconstruct(image_dir, output_dir)
    return creator.outputs.extrinsics  # (N, 4, 4) world-to-cam


def _save_outputs(
    metrics: dict,
    trajectories: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Write metrics.json, trajectories.npz, and two plot PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json — drop non-serializable per_frame array
    metrics_json = {}
    for cond, m in metrics.items():
        metrics_json[cond] = {
            "ate": {k: v for k, v in m["ate"].items() if k != "per_frame"},
            "rpe": m["rpe"],
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    # trajectories.npz
    npz_data = {"gt": trajectories["gt"]}
    for cond in _CONDITIONS:
        if cond in trajectories:
            npz_data[f"pred_{cond}"] = trajectories[cond]
    np.savez(output_dir / "trajectories.npz", **npz_data)

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_trajectory(trajectories, plots_dir / "trajectory.png")
    _plot_ate_per_frame(metrics, trajectories["gt"], plots_dir / "ate_per_frame.png")


def _plot_trajectory(trajectories: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, poses in trajectories.items():
        pos = _cam_positions(poses)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                label=name, color=_COLORS.get(name, "gray"),
                linewidth=2 if name == "gt" else 1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera Trajectory")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ate_per_frame(metrics: dict, gt: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for cond, m in metrics.items():
        per_frame = m["ate"].get("per_frame")
        if per_frame is None:
            continue
        rmse = m["ate"]["rmse"]
        ax.plot(per_frame, label=f"{cond} (RMSE={rmse:.3f}m)",
                color=_COLORS.get(cond, "gray"))
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.set_title("Per-frame Absolute Trajectory Error")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",     required=True, help="Dataset name, e.g. '7scenes'")
    parser.add_argument("--seq_dir",     type=Path, required=True, help="Path to sequence directory")
    parser.add_argument("--output_dir",  type=Path, required=True, help="Where to write results")
    parser.add_argument("--max_frames",  type=int, default=500)
    parser.add_argument("--conditions",  nargs="+", default=["baseline", "ba", "lc"],
                        choices=list(_CONDITIONS))
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)(args.seq_dir, max_frames=args.max_frames)
    print(f"Dataset: {args.dataset} | {len(dataset.images)} frames | {args.conditions}")

    tmp_image_dir = _prepare_image_dir(dataset.images)
    try:
        metrics: dict = {}
        trajectories: dict[str, np.ndarray] = {"gt": dataset.gt_poses}

        for cond in args.conditions:
            print(f"\n=== Condition: {cond} ===")
            pred = _run_condition(cond, tmp_image_dir, args.output_dir / cond)
            metrics[cond] = {
                "ate": ate_translation(pred, dataset.gt_poses),
                "rpe": rpe(pred, dataset.gt_poses),
            }
            trajectories[cond] = pred
            print(f"  ATE RMSE: {metrics[cond]['ate']['rmse']:.4f}m")
            print(f"  RPE trans RMSE: {metrics[cond]['rpe']['trans_rmse']:.4f}m")
    finally:
        shutil.rmtree(tmp_image_dir)

    _save_outputs(metrics, trajectories, args.output_dir)
    print(f"\nResults written to {args.output_dir}/")
    print(json.dumps(
        {c: {"ate_rmse": m["ate"]["rmse"], "rpe_trans": m["rpe"]["trans_rmse"]}
         for c, m in metrics.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v
```

Expected: 7 PASSED

- [ ] **Step 5: Smoke-test the CLI help**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py --help
```

Expected: prints usage with `--dataset`, `--seq_dir`, `--output_dir`, `--max_frames`, `--conditions`

- [ ] **Step 6: Commit**

```bash
git add evals/eval_gt.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(evals): add eval_gt.py runner with dataset-agnostic pipeline"
```

---

## Task 5: Visualization Notebook

**Files:**
- Create: `docs/pointcloud/eval_7scenes_gt.ipynb`

- [ ] **Step 1: Create the notebook**

Create `docs/pointcloud/eval_7scenes_gt.ipynb` with the following cells. The notebook is load-only — no inference runs here.

**Cell 1 (markdown):**
```markdown
# 7-Scenes GT Evaluation Results

Load `metrics.json` and `trajectories.npz` produced by `evals/eval_gt.py` and visualize
ATE/RPE comparisons across three conditions: baseline, BA, LC.
```

**Cell 2 (code) — configuration:**
```python
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Point this at the output directory from eval_gt.py
RESULTS_DIR = Path("../../eval_results/chess_seq01")

COLORS = {"baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green", "gt": "black"}
```

**Cell 3 (code) — metrics table:**
```python
metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())

import pandas as pd
rows = []
for cond, m in metrics.items():
    rows.append({
        "condition":    cond,
        "ATE RMSE (m)": round(m["ate"]["rmse"], 4),
        "ATE mean (m)": round(m["ate"]["mean"], 4),
        "ATE max (m)":  round(m["ate"]["max"], 4),
        "RPE trans RMSE (m)": round(m["rpe"]["trans_rmse"], 4),
        "RPE rot RMSE (deg)": round(m["rpe"]["rot_rmse_deg"], 4),
    })
pd.DataFrame(rows).set_index("condition")
```

**Cell 4 (code) — trajectory plot:**
```python
data = np.load(RESULTS_DIR / "trajectories.npz")

def cam_positions(poses):
    R, t = poses[:, :3, :3], poses[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
for key in data.files:
    label = key.replace("pred_", "")
    pos = cam_positions(data[key])
    lw = 2 if label == "gt" else 1
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label,
            color=COLORS.get(label, "gray"), linewidth=lw)
ax.legend()
ax.set_title("Camera Trajectory — Baseline vs BA vs LC vs GT")
plt.tight_layout()
plt.show()
```

**Cell 5 (code) — per-frame ATE:**
```python
# Load per-frame errors by recomputing from trajectories
import sys
sys.path.insert(0, "../../evals")
from datasets import get_dataset
from collab_splats.pointcloud.loop_closure.eval import ate_translation

gt = data["gt"]
fig, ax = plt.subplots(figsize=(14, 4))
for key in data.files:
    if key == "gt":
        continue
    cond = key.replace("pred_", "")
    result = ate_translation(data[key], gt)
    ax.plot(result["per_frame"], label=f"{cond} (RMSE={result['rmse']:.4f}m)",
            color=COLORS.get(cond, "gray"))
ax.set_xlabel("Frame")
ax.set_ylabel("ATE (m)")
ax.set_title("Per-frame Absolute Trajectory Error")
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 6 (markdown):**
```markdown
## Observations

*Fill in after running eval_gt.py on actual data.*
```

- [ ] **Step 2: Verify notebook parses without error**

```bash
cd /workspace/collab-splats
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.loads(open('docs/pointcloud/eval_7scenes_gt.ipynb').read())
print('cells:', len(nb['cells']))
print('OK')
"
```

Expected: `cells: 6` and `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/eval_7scenes_gt.ipynb
git commit -m "feat(docs): add eval_7scenes_gt notebook for trajectory + ATE visualization"
```

---

## Self-Review Checklist

- [x] All 4 required capabilities covered: download → load → run → metrics+plots
- [x] `get_dataset("7scenes")` syntax matches user request
- [x] `eval.py` stubs filled; `umeyama_se3` reused from `alignment.py`
- [x] 3 conditions: baseline / ba / lc — each with correct `use_ba` + `enable_loop_closure` flags
- [x] `reconstruct(image_dir, output_dir)` used (not `run()`) — confirmed from `base.py:36`
- [x] `creator.outputs.extrinsics` shape `(N,4,4)` — confirmed from `FeedforwardResult`
- [x] `per_frame` array stripped from `metrics.json` (not JSON-serializable)
- [x] Flat test functions throughout (no test classes) — matches feedback
- [x] `nerfstudio` env python used in all test commands
- [x] No placeholders or TBDs
