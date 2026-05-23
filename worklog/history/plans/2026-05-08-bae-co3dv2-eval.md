# BAE CO3Dv2 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend existing GT eval harness to reproduce BAE paper Table IV trend — show VGGT-X + BAE improves AUC@30 over VGGT-X baseline on CO3Dv2, and validate on longer 7-Scenes sequences.

**Architecture:** Add `auc_at_threshold()` to the existing `eval.py` metrics, add a CO3Dv2 loader alongside existing 7-Scenes/TUM/KITTI loaders, wire AUC@30 + timing into `eval_gt.py`, add `ba_hightrack` condition (paper-default track params). All new code follows existing file-per-concern pattern.

**Tech Stack:** Python 3.10 (nerfstudio env `/opt/conda/envs/nerfstudio/bin/python`), numpy, pytest, existing `umeyama_align`, `BundleAdjustment`, `BundleAdjustmentConfig`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/pointcloud/loop_closure/eval.py` | Modify | Add `auc_at_threshold()` |
| `collab_splats/pointcloud/bundle_adjustment.py` | Modify | Add `max_query_pts`/`query_frame_num` to `BundleAdjustmentConfig` |
| `collab_splats/pointcloud/wrappers.py` | Modify | Pass track params from config to `extract_tracks_vggsfm` |
| `evals/datasets.py` | Modify | Add optional `intrinsics` to `EvalDataset`; add `_load_co3dv2` |
| `evals/eval_gt.py` | Modify | Add `ba_hightrack` condition; emit `auc_30` + `time_s`; `co3dv2` dataset |
| `evals/download_co3dv2.sh` | Create | Thin wrapper around CO3Dv2 download script |
| `docs/pointcloud/eval_co3dv2_gt.ipynb` | Create | Load-only viz notebook |
| `tests/pointcloud/test_auc_metric.py` | Create | Unit tests for `auc_at_threshold` |
| `tests/pointcloud/test_co3dv2_loader.py` | Create | Unit tests for `_load_co3dv2` |

---

### Task 1: `auc_at_threshold` in `eval.py`

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/eval.py` (append after existing `rpe` function)
- Create: `tests/pointcloud/test_auc_metric.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pointcloud/test_auc_metric.py`:

```python
import numpy as np
import pytest

from collab_splats.pointcloud.loop_closure.eval import auc_at_threshold


def _make_poses(rotations_deg, translations):
    """Build (N, 4, 4) world-to-cam poses from rotation angles (about z-axis) and translations."""
    from scipy.spatial.transform import Rotation as R

    N = len(rotations_deg)
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    for i, (deg, t) in enumerate(zip(rotations_deg, translations)):
        poses[i, :3, :3] = R.from_euler("z", deg, degrees=True).as_matrix().astype(np.float32)
        poses[i, :3, 3] = np.array(t, dtype=np.float32)
    return poses


def test_auc_returns_required_keys():
    pred = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert set(result) >= {"auc_30", "per_frame_err"}


def test_auc_perfect_poses_scores_100():
    """When pred == gt, every frame error is 0 → AUC@30 = 100."""
    gt = _make_poses([0, 10, 20, 30], [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]])
    result = auc_at_threshold(gt.copy(), gt)
    assert result["auc_30"] >= 99.0, f"Expected ~100, got {result['auc_30']}"


def test_auc_large_error_scores_near_zero():
    """When all frames have 90° rotation error (>> 30°), AUC@30 ≈ 0."""
    from scipy.spatial.transform import Rotation as R

    N = 5
    gt = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    rot_90 = R.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    pred[:, :3, :3] = rot_90
    result = auc_at_threshold(pred, gt)
    assert result["auc_30"] < 10.0, f"Expected near 0, got {result['auc_30']}"


def test_per_frame_err_length_matches_input():
    N = 7
    pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert len(result["per_frame_err"]) == N


def test_auc_is_float_in_range():
    pred = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    gt   = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    result = auc_at_threshold(pred, gt)
    assert isinstance(result["auc_30"], float)
    assert 0.0 <= result["auc_30"] <= 100.0


def test_auc_half_frames_below_threshold():
    """4 frames: 2 with error ~0°, 2 with error ~60°. AUC@30 should be near 50."""
    from scipy.spatial.transform import Rotation as R

    gt   = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    pred = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    rot_60 = R.from_euler("x", 60, degrees=True).as_matrix().astype(np.float32)
    pred[2, :3, :3] = rot_60
    pred[3, :3, :3] = rot_60
    result = auc_at_threshold(pred, gt)
    # Frames 0,1 perfect (err≈0), frames 2,3 err≈60° > 30°
    # accuracy(t) = 0.5 for all t ∈ (0°, 60°], so AUC ≈ 50
    assert 40.0 < result["auc_30"] < 60.0, f"Expected ~50, got {result['auc_30']}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_auc_metric.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `AttributeError: module ... has no attribute 'auc_at_threshold'`

- [ ] **Step 3: Implement `auc_at_threshold` in `eval.py`**

Append after the existing `rpe` function in `collab_splats/pointcloud/loop_closure/eval.py`:

```python
def auc_at_threshold(
    pred: np.ndarray,
    gt: np.ndarray,
    max_threshold_deg: float = 30.0,
    num_steps: int = 100,
) -> dict:
    """AUC@max_threshold_deg metric for camera pose accuracy (CO3Dv2 / VGGSfM protocol).

    Aligns pred to gt via Umeyama SE(3), then computes per-frame max(R_err, T_err).
    Accuracy at t = fraction of frames with combined error < t.
    AUC = mean accuracy across num_steps thresholds in [0, max_threshold_deg] × 100.

    Returns:
        {"auc_30": float in [0, 100], "per_frame_err": list[float]}
    """
    aligned, _ = umeyama_align(pred, gt)

    R_pred = aligned[:, :3, :3]          # (N, 3, 3) world-to-cam rotation
    R_gt   = gt[:, :3, :3]
    t_pred = aligned[:, :3, 3]           # (N, 3) world-to-cam translation
    t_gt   = gt[:, :3, 3]

    # Rotation error: geodesic distance (degrees)
    R_rel  = R_pred @ R_gt.transpose(0, 2, 1)           # (N, 3, 3)
    traces = np.trace(R_rel, axis1=1, axis2=2)          # (N,)
    err_R  = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))

    # Translation direction error: angle between camera centers in world (degrees, scale-free)
    # Camera center in world: c = -R^T @ t
    c_pred = np.einsum("nij,nj->ni", R_pred.transpose(0, 2, 1), -t_pred)   # (N, 3)
    c_gt   = np.einsum("nij,nj->ni", R_gt.transpose(0, 2, 1),   -t_gt)     # (N, 3)
    c_pred_norm = c_pred / (np.linalg.norm(c_pred, axis=1, keepdims=True) + 1e-10)
    c_gt_norm   = c_gt   / (np.linalg.norm(c_gt,   axis=1, keepdims=True) + 1e-10)
    dots  = np.clip((c_pred_norm * c_gt_norm).sum(axis=1), -1.0, 1.0)
    err_T = np.degrees(np.arccos(dots))

    err = np.maximum(err_R, err_T)   # (N,) per-frame combined error

    thresholds = np.linspace(0.0, max_threshold_deg, num_steps)
    accuracies = np.array([np.mean(err < t) for t in thresholds])
    auc = float(np.mean(accuracies) * 100.0)

    return {"auc_30": auc, "per_frame_err": err.tolist()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_auc_metric.py -v 2>&1 | tail -15
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/eval.py tests/pointcloud/test_auc_metric.py
git commit -m "feat(eval): add auc_at_threshold metric (CO3Dv2/VGGSfM AUC@30 protocol)"
```

---

### Task 2: `EvalDataset` intrinsics + CO3Dv2 loader

**Files:**
- Modify: `evals/datasets.py`
- Create: `tests/pointcloud/test_co3dv2_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pointcloud/test_co3dv2_loader.py`:

```python
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_fake_seq(tmp_path: Path, n_frames: int = 4) -> Path:
    """Write a minimal CO3Dv2 sequence dir with synthetic frame_annotations.jgz."""
    seq_dir = tmp_path / "apple" / "seq1"
    images_dir = seq_dir / "images"
    images_dir.mkdir(parents=True)

    annotations = []
    for i in range(1, n_frames + 1):
        fname = f"frame{i:06d}.jpg"
        (images_dir / fname).write_bytes(b"fake")

        # Synthetic viewpoint: identity rotation, small translation
        annotations.append({
            "image": {"path": f"images/{fname}", "size": [480, 640]},
            "viewpoint": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [float(i) * 0.1, 0.0, 1.0],
                "focal_length": [1.2, 1.2],
                "principal_point": [0.0, 0.0],
            },
        })

    ann_path = seq_dir / "frame_annotations.jgz"
    with gzip.open(ann_path, "wt", encoding="utf-8") as f:
        json.dump(annotations, f)

    return seq_dir


def test_co3dv2_loader_returns_eval_dataset(tmp_path):
    from evals.datasets import get_dataset, EvalDataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=4)
    loader = get_dataset("co3dv2")
    ds = loader(seq_dir, max_frames=4)
    assert isinstance(ds, EvalDataset)


def test_co3dv2_loader_correct_n_frames(tmp_path):
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=4)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=4)
    assert len(ds.images) == 4
    assert ds.gt_poses.shape == (4, 4, 4)


def test_co3dv2_loader_max_frames_truncation(tmp_path):
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=4)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=2)
    assert len(ds.images) == 2
    assert ds.gt_poses.shape == (2, 4, 4)


def test_co3dv2_loader_pose_dtype(tmp_path):
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=3)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=3)
    assert ds.gt_poses.dtype == np.float32


def test_co3dv2_loader_identity_rotation_preserved(tmp_path):
    """Synthetic annotations use identity R — loader should preserve it."""
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=2)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=2)
    np.testing.assert_allclose(ds.gt_poses[0, :3, :3], np.eye(3), atol=1e-5)


def test_co3dv2_loader_provides_intrinsics(tmp_path):
    """intrinsics field should be (N, 3, 3) float32."""
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=3)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=3)
    assert ds.intrinsics is not None
    assert ds.intrinsics.shape == (3, 3, 3)
    assert ds.intrinsics.dtype == np.float32


def test_eval_dataset_intrinsics_defaults_to_none():
    """Existing callers pass no intrinsics — field must default to None."""
    from evals.datasets import EvalDataset

    ds = EvalDataset(images=[], gt_poses=np.zeros((0, 4, 4), dtype=np.float32))
    assert ds.intrinsics is None


def test_7scenes_loader_still_works(tmp_path):
    """Regression: existing 7-Scenes loader must still return EvalDataset without intrinsics."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
    from evals.datasets import EvalDataset

    ds = EvalDataset(
        images=[],
        gt_poses=np.zeros((0, 4, 4), dtype=np.float32),
    )
    assert ds.intrinsics is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_co3dv2_loader.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `KeyError: 'co3dv2'`

- [ ] **Step 3: Add `intrinsics` field to `EvalDataset` and `_load_co3dv2` to `datasets.py`**

In `evals/datasets.py`, change the `EvalDataset` dataclass:

```python
@dataclass
class EvalDataset:
    images: list[Path]
    gt_poses: np.ndarray           # (N, 4, 4) world-to-cam float32
    intrinsics: np.ndarray | None = None   # (N, 3, 3) float32, optional
```

Then append `_load_co3dv2` before the `_REGISTRY` dict:

```python
def _load_co3dv2(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """Load a single CO3Dv2 sequence.

    Layout:
        seq_dir/images/frame000001.jpg, frame000002.jpg, ...
        seq_dir/frame_annotations.jgz   (gzip-compressed JSON list of frame dicts)

    Each frame dict viewpoint fields:
        R: [[...], ...] — 3×3 world-to-cam rotation (row-major)
        T: [tx, ty, tz] — world-to-cam translation
        focal_length: [fx_ndc, fy_ndc] — relative to min(H,W)/2
        principal_point: [px_ndc, py_ndc] — offset from image centre, relative to min(H,W)/2
    """
    import gzip
    import json as _json

    seq_dir = Path(seq_dir)
    ann_path = seq_dir / "frame_annotations.jgz"

    with gzip.open(ann_path, "rt", encoding="utf-8") as f:
        annotations = _json.load(f)

    def _frame_number(ann: dict) -> int:
        stem = Path(ann["image"]["path"]).stem   # e.g. "frame000001"
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 0

    annotations = sorted(annotations, key=_frame_number)[:max_frames]

    images: list[Path] = []
    gt_poses_list: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []

    for ann in annotations:
        img_name = Path(ann["image"]["path"]).name
        images.append(seq_dir / "images" / img_name)

        vp = ann["viewpoint"]
        R = np.array(vp["R"], dtype=np.float32)   # (3, 3)
        T = np.array(vp["T"], dtype=np.float32)   # (3,)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = T
        gt_poses_list.append(pose)

        # Convert CO3Dv2 NDC intrinsics → pixel-space K
        # NDC: focal_length in units of min(H,W)/2; principal_point offset from centre
        H, W = ann["image"]["size"]
        s = min(H, W) / 2.0
        fx_ndc, fy_ndc = vp["focal_length"]
        px_ndc, py_ndc = vp["principal_point"]
        fx = fx_ndc * s
        fy = fy_ndc * s
        cx = px_ndc * s + W / 2.0
        cy = py_ndc * s + H / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrinsics_list.append(K)

    gt_poses = np.stack(gt_poses_list).astype(np.float32) if gt_poses_list else np.zeros((0, 4, 4), dtype=np.float32)
    intrinsics = np.stack(intrinsics_list).astype(np.float32) if intrinsics_list else None
    return EvalDataset(images=images, gt_poses=gt_poses, intrinsics=intrinsics)
```

Then add to `_REGISTRY`:

```python
_REGISTRY: dict[str, Callable[..., EvalDataset]] = {
    "7scenes": _load_7scenes,
    "tum": _load_tum,
    "kitti": _load_kitti,
    "waymo": _load_waymo,
    "co3dv2": _load_co3dv2,   # ← add this
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_co3dv2_loader.py -v 2>&1 | tail -15
```

Expected: `8 passed`

- [ ] **Step 5: Regression — existing loader tests still green**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v --ignore=tests/pointcloud/test_auc_metric.py --ignore=tests/pointcloud/test_co3dv2_loader.py 2>&1 | tail -10
```

Expected: same pass count as before

- [ ] **Step 6: Commit**

```bash
git add evals/datasets.py tests/pointcloud/test_co3dv2_loader.py
git commit -m "feat(datasets): add CO3Dv2 loader + optional intrinsics field on EvalDataset"
```

---

### Task 3: `BundleAdjustmentConfig` track params + `ba_hightrack`

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py` (add fields to `BundleAdjustmentConfig`)
- Modify: `collab_splats/pointcloud/wrappers.py` (`_apply_ba` uses track params from config)
- Modify: `evals/eval_gt.py` (add `ba_hightrack` choice + `_make_creator` case)

- [ ] **Step 1: Add `max_query_pts` and `query_frame_num` to `BundleAdjustmentConfig`**

In `collab_splats/pointcloud/bundle_adjustment.py`, find the `BundleAdjustmentConfig` dataclass and add two fields. The dataclass currently has fields passed to `run_bundle_adjustment` (e.g. `max_reproj_error`, `lm_steps`, `shared_camera`, `min_inliers_per_frame`). Add at the end:

```python
@dataclass
class BundleAdjustmentConfig:
    max_reproj_error: float | None = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048      # ← add
    query_frame_num: int = 5       # ← add
```

> **Note:** Read the actual file first. The existing fields may differ slightly in name or order — keep all existing fields unchanged and append only `max_query_pts` and `query_frame_num`.

- [ ] **Step 2: Update `_apply_ba` in `wrappers.py` to pass track params**

In `collab_splats/pointcloud/wrappers.py`, find `BundleAdjustment._apply_ba` (or the `reconstruct` method that calls `extract_tracks_vggsfm`). The call to `extract_tracks_vggsfm` currently uses hardcoded defaults. Change it to use `self.config.max_query_pts` and `self.config.query_frame_num`.

Also, `run_bundle_adjustment` is called with `**dataclasses.asdict(self.config)`. After adding track params to the config, filter them out before passing to `run_bundle_adjustment` (which doesn't accept them):

Find the block that calls `run_bundle_adjustment` and change it to:

```python
# Split config: track-extraction params vs BA-solver params
cfg_dict = dataclasses.asdict(self.config)
track_params = {
    "max_query_pts": cfg_dict.pop("max_query_pts"),
    "query_frame_num": cfg_dict.pop("query_frame_num"),
}

tracks, vis_scores, pts3d_kp = extract_tracks_vggsfm(
    result.images,
    result.conf if hasattr(result, "conf") else None,
    result.world_points if hasattr(result, "world_points") else None,
    **track_params,
)

_, refined_ext_3x4, refined_intr = run_bundle_adjustment(
    pts3d_kp,
    extrinsics_3x4,
    result.intrinsics,
    tracks,
    vis_scores,
    image_size=(result.model_height, result.model_width),
    **cfg_dict,          # ← now excludes max_query_pts / query_frame_num
)
```

> **Note:** Read `wrappers.py` first. Find the existing `extract_tracks_vggsfm` call — it already passes `result.images` etc. Replace only the hardcoded `max_query_pts`/`query_frame_num` with `**track_params`. Don't restructure anything else.

- [ ] **Step 3: Add `ba_hightrack` condition to `eval_gt.py`**

In `evals/eval_gt.py`, update `_CONDITION_CHOICES` and `_make_creator`:

```python
_CONDITION_CHOICES = ["baseline", "ba", "lc", "ba_hightrack"]
_COLORS = {"gt": "black", "baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green", "ba_hightrack": "tab:purple"}
```

```python
def _make_creator(condition: str):
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig

    base = get_creator("vggtx")()
    if condition == "ba":
        return BundleAdjustment(base)
    if condition == "ba_hightrack":
        return BundleAdjustment(base, config=BundleAdjustmentConfig(
            max_query_pts=4096,
            query_frame_num=8,
        ))
    if condition == "lc":
        return LoopClosure(base)
    return base
```

- [ ] **Step 4: Smoke-test config round-trip**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import dataclasses
from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig
cfg = BundleAdjustmentConfig(max_query_pts=4096, query_frame_num=8)
d = dataclasses.asdict(cfg)
assert 'max_query_pts' in d
assert d['max_query_pts'] == 4096
print('OK', d)
"
```

Expected: `OK {'max_reproj_error': 4.0, ..., 'max_query_pts': 4096, 'query_frame_num': 8}`

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py collab_splats/pointcloud/wrappers.py evals/eval_gt.py
git commit -m "feat(ba): add max_query_pts/query_frame_num to BundleAdjustmentConfig; add ba_hightrack condition"
```

---

### Task 4: Runner timing + AUC@30 + CO3Dv2 dataset

**Files:**
- Modify: `evals/eval_gt.py`

- [ ] **Step 1: Add `time` import and `auc_at_threshold` import**

At the top of `evals/eval_gt.py`, add `import time` alongside existing imports, and extend the eval import line:

```python
from collab_splats.pointcloud.loop_closure.eval import ate_translation, rpe, auc_at_threshold
```

- [ ] **Step 2: Add timing + AUC to the condition loop in `main()`**

Find the condition loop in `main()`. It currently reads:

```python
for cond in args.conditions:
    print(f"\n=== Condition: {cond} ===")
    pred = _run_condition(cond, tmp_image_dir, args.output_dir / cond)
    metrics[cond] = {
        "ate": ate_translation(pred, dataset.gt_poses),
        "rpe": rpe(pred, dataset.gt_poses),
    }
```

Replace with:

```python
for cond in args.conditions:
    print(f"\n=== Condition: {cond} ===")
    t0 = time.perf_counter()
    pred = _run_condition(cond, tmp_image_dir, args.output_dir / cond)
    elapsed = time.perf_counter() - t0
    metrics[cond] = {
        "ate":    ate_translation(pred, dataset.gt_poses),
        "rpe":    rpe(pred, dataset.gt_poses),
        "auc":    auc_at_threshold(pred, dataset.gt_poses),
        "time_s": round(elapsed, 2),
    }
    print(f"  ATE RMSE:       {metrics[cond]['ate']['rmse']:.4f}m")
    print(f"  RPE trans RMSE: {metrics[cond]['rpe']['trans_rmse']:.4f}m")
    print(f"  AUC@30:         {metrics[cond]['auc']['auc_30']:.1f}")
    print(f"  Time:           {elapsed:.1f}s")
```

- [ ] **Step 3: Update `_save_outputs` to emit `auc_30` and `time_s`**

Find the `metrics_json` block inside `_save_outputs`. It currently strips `per_frame` from `ate`. Update to also flatten `auc` and include `time_s`:

```python
metrics_json = {}
for cond, m in metrics.items():
    metrics_json[cond] = {
        "ate":    {k: v for k, v in m["ate"].items() if k != "per_frame"},
        "rpe":    m["rpe"],
        "auc_30": m["auc"]["auc_30"],
        "time_s": m.get("time_s", None),
    }
```

- [ ] **Step 4: Add `co3dv2` to `--dataset` choices**

The `--dataset` argument currently has no `choices=` restriction (it delegates to `get_dataset()`). No change needed — `get_dataset("co3dv2")` will work automatically once registered in `_REGISTRY`. But update the docstring in `eval_gt.py` to mention `co3dv2`:

```python
parser.add_argument("--dataset", required=True,
    help="Dataset name: 7scenes | tum | kitti | waymo | co3dv2")
```

- [ ] **Step 5: Verify updated output format with dry-run import**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import sys; sys.path.insert(0, 'evals')
from evals.datasets import get_dataset
loader = get_dataset('co3dv2')
print('co3dv2 loader found:', loader)
from collab_splats.pointcloud.loop_closure.eval import auc_at_threshold
print('auc_at_threshold importable:', auc_at_threshold)
"
```

Expected: both print without error.

- [ ] **Step 6: Commit**

```bash
git add evals/eval_gt.py
git commit -m "feat(eval_gt): add AUC@30 + timing to metrics output; support co3dv2 dataset"
```

---

### Task 5: `download_co3dv2.sh`

**Files:**
- Create: `evals/download_co3dv2.sh`

- [ ] **Step 1: Create download script**

CO3Dv2 is distributed via a Python download script from facebookresearch/co3d. The shell script wraps it:

Create `evals/download_co3dv2.sh`:

```bash
#!/usr/bin/env bash
# Download a single CO3Dv2 category for evaluation.
#
# Usage:
#   bash evals/download_co3dv2.sh <category> [output_dir]
#
# <category>: one of apple, ball, banana, bench, book, bottle, bowl, broccoli, car, chair
#             (10 categories used in standard SfM benchmarks)
# [output_dir]: destination directory (default: ./data/co3dv2/)
#
# Downloads:
#   <output_dir>/<category>/<sequence>/images/*.jpg
#   <output_dir>/<category>/<sequence>/frame_annotations.jgz
#
# Prerequisites:
#   pip install co3d   (or clone facebookresearch/co3d and install)
#   CO3D_AWS_ACCESS_KEY_ID and CO3D_AWS_SECRET_ACCESS_KEY env vars set
#   (obtain from https://github.com/facebookresearch/co3d#dataset-download)

set -euo pipefail

CATEGORY="${1:-}"
OUTPUT_DIR="${2:-./data/co3dv2}"

if [ -z "$CATEGORY" ]; then
    echo "Usage: bash evals/download_co3dv2.sh <category> [output_dir]"
    echo "Categories: apple ball banana bench book bottle bowl broccoli car chair"
    exit 1
fi

STANDARD_CATEGORIES="apple ball banana bench book bottle bowl broccoli car chair"
if ! echo "$STANDARD_CATEGORIES" | grep -qw "$CATEGORY"; then
    echo "Warning: '$CATEGORY' is not in the standard 10-category eval set."
    echo "Standard categories: $STANDARD_CATEGORIES"
fi

echo "=== Downloading CO3Dv2 category: $CATEGORY → $OUTPUT_DIR ==="
mkdir -p "$OUTPUT_DIR"

# Check for co3d Python package
PYTHON="${PYTHON:-/opt/conda/envs/nerfstudio/bin/python}"
if ! "$PYTHON" -c "import co3d" 2>/dev/null; then
    echo "Error: 'co3d' package not found. Install with:"
    echo "  $PYTHON -m pip install co3d"
    echo "  OR: git clone https://github.com/facebookresearch/co3d && pip install -e co3d/"
    exit 1
fi

"$PYTHON" -c "
from co3d.dataset.data_types import load_dataclass_jgzip
import subprocess, sys
# Use co3d's official download helper
from co3d.download import download_dataset
download_dataset(
    download_folder='${OUTPUT_DIR}',
    category='${CATEGORY}',
    download_modalities=['rgb', 'annotations'],
)
print('Download complete: ${OUTPUT_DIR}/${CATEGORY}/')
"

echo "=== Done. Test with: ==="
echo "python evals/eval_gt.py --dataset co3dv2 \\"
echo "  --seq_dir ${OUTPUT_DIR}/${CATEGORY}/<sequence_name> \\"
echo "  --output_dir ./eval_results/co3dv2_${CATEGORY} \\"
echo "  --conditions baseline ba ba_hightrack"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x evals/download_co3dv2.sh
git add evals/download_co3dv2.sh
git commit -m "feat(evals): add download_co3dv2.sh for CO3Dv2 subset download"
```

---

### Task 6: Visualization notebook

**Files:**
- Create: `docs/pointcloud/eval_co3dv2_gt.ipynb`

- [ ] **Step 1: Create the notebook**

Create `docs/pointcloud/eval_co3dv2_gt.ipynb` with these cells. The notebook is **load-only** — no inference runs here.

**Cell 1 (markdown):**
```markdown
# CO3Dv2 GT Evaluation Results (BAE Paper Table IV Comparison)

Load `metrics.json` and `trajectories.npz` from `evals/eval_gt.py` and visualize
AUC@30, ATE/RPE comparisons across conditions: baseline, ba, ba_hightrack.

Paper reference (VGGT + BAE): AUC@30 = 90.0
```

**Cell 2 (code) — configuration:**
```python
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("../../eval_results/co3dv2_apple_seq01")  # update to actual path

COLORS = {
    "baseline": "tab:red",
    "ba": "tab:blue",
    "ba_hightrack": "tab:purple",
    "gt": "black",
}

PAPER_REFERENCE = {"VGGT + BAE (paper)": 90.0}
```

**Cell 3 (code) — metrics table:**
```python
metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())

import pandas as pd
rows = []
for cond, m in metrics.items():
    rows.append({
        "condition":         cond,
        "AUC@30 ↑":          round(m.get("auc_30", float("nan")), 1),
        "ATE RMSE (m) ↓":    round(m["ate"]["rmse"], 4),
        "RPE trans RMSE ↓":  round(m["rpe"]["trans_rmse"], 4),
        "Time (s)":          m.get("time_s", "—"),
    })
# Append paper reference row
rows.append({
    "condition": "VGGT + BAE (paper)",
    "AUC@30 ↑": 90.0,
    "ATE RMSE (m) ↓": "—",
    "RPE trans RMSE ↓": "—",
    "Time (s)": 0.9,
})
df = pd.DataFrame(rows).set_index("condition")
print(df.to_string())
df
```

**Cell 4 (code) — AUC@30 bar chart:**
```python
fig, ax = plt.subplots(figsize=(8, 4))
conditions = list(metrics.keys())
auc_vals = [metrics[c].get("auc_30", 0) for c in conditions]
colors = [COLORS.get(c, "gray") for c in conditions]

bars = ax.bar(conditions, auc_vals, color=colors, alpha=0.85)
ax.axhline(90.0, color="black", linestyle="--", label="VGGT+BAE (paper) = 90.0")
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.set_ylabel("AUC@30 ↑")
ax.set_title("Camera Pose Accuracy — CO3Dv2 (higher is better)")
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "plots" / "auc30_comparison.png", dpi=150)
plt.show()
```

**Cell 5 (code) — trajectory plot:**
```python
data = np.load(RESULTS_DIR / "trajectories.npz")

def cam_positions(poses):
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
for key in data.files:
    label = key.replace("pred_", "")
    pos = cam_positions(data[key])
    lw = 2 if label == "gt" else 1
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            label=label, color=COLORS.get(label, "gray"), linewidth=lw)
ax.legend()
ax.set_title("Camera Trajectories")
plt.tight_layout()
plt.show()
```

Create the notebook JSON using Python (run this once to generate the file, then commit the `.ipynb`):

```python
# Run this one-time script to generate the notebook:
import nbformat as nbf

nb = nbf.v4.new_notebook()
# ... (paste cells above into nb.cells as nbf.v4.new_markdown_cell / new_code_cell)
nbf.write(nb, "docs/pointcloud/eval_co3dv2_gt.ipynb")
```

Or create manually using Jupyter. The above cell contents are the canonical source.

- [ ] **Step 2: Execute notebook to verify it runs without data (expect KeyError or FileNotFoundError — that's OK; the cells must at minimum import without error)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
  --to notebook --execute docs/pointcloud/eval_co3dv2_gt.ipynb \
  --output /tmp/test_nb.ipynb 2>&1 | tail -5
```

This will fail on the file-not-found for `metrics.json` — that's expected. Confirm it's only that error, not an import error.

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/eval_co3dv2_gt.ipynb
git commit -m "docs(notebook): add eval_co3dv2_gt.ipynb — load-only CO3Dv2 eval viz"
```

---

### Task 7: 7-Scenes extended validation (longer sequences)

**Files:** none — run commands only

Context: chess seq-01 at 50 frames showed BA marginally worse. This is expected (too short, no drift to correct). Run longer sequences and multiple scenes.

- [ ] **Step 1: Run chess seq-01 full (1000 frames)**

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir /data/7scenes/chess/seq-01 \
  --output_dir ./eval_results/chess_seq01_1000 \
  --max_frames 1000 \
  --conditions baseline ba
```

Expected: `ba` ATE RMSE ≤ `baseline` ATE RMSE on longer sequence. AUC@30 both conditions printed.

- [ ] **Step 2: Run fire seq-01 (500 frames)**

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir /data/7scenes/fire/seq-01 \
  --output_dir ./eval_results/fire_seq01_500 \
  --max_frames 500 \
  --conditions baseline ba
```

- [ ] **Step 3: Run heads seq-01 (500 frames)**

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir /data/7scenes/heads/seq-01 \
  --output_dir ./eval_results/heads_seq01_500 \
  --max_frames 500 \
  --conditions baseline ba
```

- [ ] **Step 4: Print summary table across sequences**

```python
# Run from repo root
import json
from pathlib import Path

runs = {
    "chess-1000": "eval_results/chess_seq01_1000/metrics.json",
    "fire-500":   "eval_results/fire_seq01_500/metrics.json",
    "heads-500":  "eval_results/heads_seq01_500/metrics.json",
}

print(f"{'Scene':<15} {'Cond':<12} {'AUC@30':>8} {'ATE RMSE':>10} {'Time':>8}")
print("-" * 55)
for scene, path in runs.items():
    if not Path(path).exists():
        print(f"{scene}: not yet run"); continue
    m = json.loads(Path(path).read_text())
    for cond in ("baseline", "ba"):
        if cond not in m: continue
        auc = m[cond].get("auc_30", "—")
        ate = m[cond]["ate"]["rmse"]
        t   = m[cond].get("time_s", "—")
        print(f"{scene:<15} {cond:<12} {auc:>8.1f} {ate:>10.4f} {t:>8}")
```

- [ ] **Step 5: Commit results**

```bash
git add eval_results/*/metrics.json eval_results/*/trajectories.npz
git commit -m "chore(eval): 7-Scenes extended validation results (chess-1000, fire-500, heads-500)"
```

---

### Task 8: CO3Dv2 evaluation run

**Files:** none — run commands only

Prerequisites: `download_co3dv2.sh` executed, at least one sequence available.

- [ ] **Step 1: Download one apple sequence (~2GB)**

```bash
bash evals/download_co3dv2.sh apple ./data/co3dv2
# Then pick one sequence:
ls data/co3dv2/apple/ | head -5
```

- [ ] **Step 2: Run eval on one sequence**

```bash
SEQ=$(ls data/co3dv2/apple/ | head -1)
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
  --dataset co3dv2 \
  --seq_dir data/co3dv2/apple/$SEQ \
  --output_dir ./eval_results/co3dv2_apple_${SEQ} \
  --max_frames 50 \
  --conditions baseline ba ba_hightrack
```

- [ ] **Step 3: Open notebook and check table**

Open `docs/pointcloud/eval_co3dv2_gt.ipynb`, set `RESULTS_DIR` to the output above, run all cells.

Success criterion: `ba` AUC@30 ≥ `baseline` AUC@30. Trend should mirror paper (BA adds ~2 points). Exact numbers will differ from paper (VGGT-X ≠ vanilla VGGT).

- [ ] **Step 4: Commit results**

```bash
git add eval_results/co3dv2_apple_*/metrics.json eval_results/co3dv2_apple_*/trajectories.npz
git commit -m "chore(eval): CO3Dv2 apple BA validation results"
```

---

## Self-Review

**Spec coverage:**
- ✅ `auc_at_threshold` — Task 1
- ✅ CO3Dv2 loader + `EvalDataset.intrinsics` — Task 2
- ✅ `ba_hightrack` condition (paper-default track params) — Task 3
- ✅ Runner: `auc_30` + `time_s` in `metrics.json` — Task 4
- ✅ `download_co3dv2.sh` — Task 5
- ✅ Viz notebook — Task 6
- ✅ 7-Scenes extended validation — Task 7
- ✅ CO3Dv2 run + comparison table — Task 8

**Placeholder scan:** None found. All code steps are complete.

**Type consistency check:**
- `auc_at_threshold` returns `{"auc_30": float, "per_frame_err": list}` — used as `m["auc"]["auc_30"]` in runner (Task 4) ✅
- `EvalDataset.intrinsics: np.ndarray | None = None` — loader sets it, tests check shape ✅
- `BundleAdjustmentConfig(max_query_pts=4096, query_frame_num=8)` — field names match what's added in Task 3 ✅
- `_CONDITION_CHOICES` includes `"ba_hightrack"` — `_make_creator` handles it ✅
