# Evals Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `evals/` to `datasets.py`, `gt_metrics.py`, `eval.py`, `configs/`, `README.md`; the grid runner drives `Reconstructor` with config overrides.

**Architecture:** A grid YAML names datasets and conditions (config overrides). `python -m evals.eval --config X` runs one subprocess per cell; each cell stages GT frames as an image dir, runs `Reconstructor.run_pipeline()`, reads poses (and depth) from `pointcloud.zarr`, scores them with evo ATE/RPE + our AUC + GT depth error, and copies every `*_quality_report.json` the stages wrote.

**Tech Stack:** numpy, evo 1.36 (`PoseTrajectory3D`, `metrics.APE/RPE`), mergedeep, matplotlib, pytest.

Spec: [2026-09-28-evals-release-cleanup-design.md](../specs/2026-09-28-evals-release-cleanup-design.md) · Rules: [017](../decisions/017-release-cleanup-rules.md)

**Worktree:** `/workspace/collab-splats/.worktrees/evals-release`, branch `clean/evals-release` off `clean/final` @ `2c0b2f06`. Every command is prefixed `cd $WT &&`.

**Gate** (after every commit):

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" \
  && cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/evals tests/pointcloud tests/geometry tests/wrapper -q -p no:cacheprovider
```

Never pipe through `tail`, never `--tb=no`. Commit with `git commit --only <paths>`; never amend/rebase/reset.

## Deviations from the spec (decided while planning)

- **`base:` block, not `stages:`.** A cell calls `run_pipeline()` with no stage list, so the config picks
  stages exactly as production does (`bundle_adjustment: true` adds `refine`). The grid's `base:` block
  (applied before each condition) turns semantics and mesh off. A `stages:` list would need the runner
  to mirror `run_pipeline`'s enable logic.
- **`EvalDataset.depth_paths`, not `gt_depth`.** 500 GT depth frames are ~600 MB; the runner reads them
  one at a time.
- **No `configs/sky_mask.yaml`.** The grid runs GT datasets (7-Scenes, TUM, CO3Dv2): none has sky, and
  the mesh report does not exist yet. The sky-mask A/B is two pipeline runs with `mesh.mask_sky`
  flipped; the sky-mask plan says so.
- **AUC drops its Sim3 alignment.** Its own comments prove alignment cancels in both relative rotation
  and normalized relative translation. `per_pair_err` (N² floats) is dropped from the output.

## File map

| path | action |
|---|---|
| `evals/datasets.py` | round 1 prose; round 2 trims to 3 loaders + `load_gt_depth` |
| `evals/gt_metrics.py` | create |
| `evals/eval.py` | create (replaces `evals/scripts/eval.py`) |
| `evals/configs/{7scenes,cross_model_chess}.yaml` | rewrite to override form |
| `evals/README.md` | rewrite |
| `evals/scripts/`, `metrics.py`, `trajectory_metrics.py`, `trajectory_io.py`, `pose_graph_diagnostics.py`, `envs/`, `data/` | delete |
| `tests/evals/*` | keep `test_datasets.py` (trimmed); add `test_gt_metrics.py`, `test_eval.py`; delete the rest |
| `tests/pointcloud/test_co3dv2_loader.py` | move into `tests/evals/test_datasets.py` |
| `tests/{pointcloud,geometry}/conftest.py` | drop `_fix_evals_import` if dead |
| `tests/test_docstring_contract.py` | cover `evals/*.py` |

---

### Task 0: Baseline gate

- [ ] **Step 1:** run the gate (tmux session `evals-baseline`, output in the session scratchpad). Record passed/failed/skipped/xfailed and the failing node ids. The `__file__` line must be inside `$WT`.

### Task 1: Round 1 — `datasets.py` prose

**Files:** Modify `evals/datasets.py`

- [ ] **Step 1:** add a module docstring and 017-shaped docstrings to every surviving def (`EvalDataset`, `_load_7scenes`, `_read_tum_assoc`, `_read_tum_groundtruth`, `_load_tum`, `_load_co3dv2`, `get_dataset`); rewrite the co3dv2 comment block as a header + bullets. Code untouched (loaders that die in Task 4 are left as-is).

```python
"""
Ground-truth eval datasets: frames plus world-to-camera poses per sequence.

- one loader per dataset type, looked up by `get_dataset`
- every loader returns frames in GT order, capped at max_frames
"""
```

- [ ] **Step 2: AST proof.** Strip every docstring statement both sides, compare dumps; then flip one constant and confirm the proof fails.

```bash
cd $WT && /opt/venv/reconstruction/bin/python - <<'EOF'
import ast, subprocess
def stripped(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
    return ast.dump(t)
old = subprocess.check_output(["git", "show", "HEAD:evals/datasets.py"], text=True)
new = open("evals/datasets.py").read()
print("EQUAL", stripped(old) == stripped(new))
print("MUTANT", stripped(old) == stripped(new.replace("0.02", "0.03")))
EOF
```

Expected: `EQUAL True`, `MUTANT False`.

- [ ] **Step 3:** gate; commit `docs(evals): datasets docstrings to 017 (round 1)`.

### Task 2: Delete one-offs

- [ ] **Step 1:** `git rm` `evals/scripts/{tri_angle_census,ba_start_at_gt,refit_at_fixed_poses,eval_run_backend,eval_multiview_conf,eval_similarity_calibration,eval_localization_parity}.py`, `evals/pose_graph_diagnostics.py`, `evals/envs/`, `evals/data/extract_waymo.py`, `tests/evals/test_pose_graph_diagnostics.py`, `tests/evals/test_eval_instantsfm.py` (tests `_run_instantsfm`), and any `tests/evals` file that imports only deleted modules (grep each first).
- [ ] **Step 2:** grep `collab_splats tests evals docs/source configs` for each deleted module name; only doc callers may remain (Task 9).
- [ ] **Step 3:** gate; commit `refactor(evals): delete finished-investigation scripts`.

### Task 3: Untrack results

- [ ] **Step 1:** `git rm --cached -r -q evals/results/chess_seq01 evals/baselines/`; confirm `.gitignore` covers both and files stay on disk (`ls`).
- [ ] **Step 2:** commit `chore(evals): untrack committed results and baselines`.

### Task 4: Datasets — three loaders, GT depth, CO3Dv2 raise

**Files:** Modify `evals/datasets.py`, `tests/evals/test_datasets.py`; delete `tests/pointcloud/test_co3dv2_loader.py` (moved)

- [ ] **Step 1: failing tests** appended to `tests/evals/test_datasets.py` (delete the kitti/waymo/video tests; keep 7scenes/tum):

```python
def _make_co3dv2_seq(tmp_path, n_frames=4, seq_name="seq1"):
    """Category dir holding frame_annotations.jgz and one sequence of images."""
    seq_dir = tmp_path / "apple" / seq_name
    (seq_dir / "images").mkdir(parents=True)
    annotations = []
    for i in range(1, n_frames + 1):
        fname = f"frame{i:06d}.jpg"
        (seq_dir / "images" / fname).write_bytes(b"fake")
        annotations.append({
            "sequence_name": seq_name,
            "image": {"path": f"apple/{seq_name}/images/{fname}", "size": [480, 640]},
            "viewpoint": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [i * 0.1, 0.0, 1.0],
                "focal_length": [1.2, 1.2],
                "principal_point": [0.0, 0.0],
            },
        })
    with gzip.open(seq_dir.parent / "frame_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(annotations, f)
    return seq_dir


def test_load_co3dv2_count_and_intrinsics(tmp_path):
    ds = get_dataset("co3dv2")(_make_co3dv2_seq(tmp_path), max_frames=3)
    assert len(ds.images) == 3
    assert ds.gt_poses.shape == (3, 4, 4) and ds.gt_poses.dtype == np.float32
    assert ds.intrinsics.shape == (3, 3, 3)


def test_load_co3dv2_pytorch3d_to_opencv(tmp_path):
    ds = get_dataset("co3dv2")(_make_co3dv2_seq(tmp_path, n_frames=2), max_frames=2)
    np.testing.assert_allclose(ds.gt_poses[0, :3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(ds.gt_poses[1, :3, 3], [-0.2, 0.0, 1.0], atol=1e-6)


def test_load_co3dv2_unknown_sequence_raises(tmp_path):
    seq_dir = _make_co3dv2_seq(tmp_path)
    other = seq_dir.parent / "seq2"
    (other / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="seq2"):
        get_dataset("co3dv2")(other, max_frames=4)


def test_load_7scenes_depth_paths(tmp_path):
    # existing 7scenes fixture helper writes frame-XXXXXX.{color.png,pose.txt}; add depth files
    ...  # use the file's existing 7scenes fixture; assert ds.depth_paths[i].name == f"frame-{i:06d}.depth.png"


def test_load_gt_depth_meters_and_invalid(tmp_path):
    raw = np.array([[1000, 65535], [0, 2500]], dtype=np.uint16)
    cv2.imwrite(str(tmp_path / "d.png"), raw)
    np.testing.assert_allclose(load_gt_depth(tmp_path / "d.png"), [[1.0, 0.0], [0.0, 2.5]])


def test_get_dataset_names_valid_types():
    with pytest.raises(KeyError, match="7scenes"):
        get_dataset("kitti")
```

(`test_load_7scenes_depth_paths` is filled from the existing 7-Scenes fixture in that file when implementing; it asserts `depth_paths` aligns with `images`.)

- [ ] **Step 2:** run `tests/evals/test_datasets.py` → new tests FAIL (`load_gt_depth` missing, co3dv2 falls back).
- [ ] **Step 3: implement.** Delete `_load_kitti`, `_load_waymo`, `_load_bicycle`, `_load_video` and their registry rows and imports; hoist `gzip`, `json`, `Rotation` to the top; add:

```python
@dataclass
class EvalDataset:
    """
    One sequence: frames and world-to-camera GT poses, in GT order.

    - intrinsics: CO3Dv2 only, (N, 3, 3) pixel K
    - depth_paths: 7-Scenes only, uint16 millimeter PNGs aligned with images
    """

    images: list[Path]
    gt_poses: np.ndarray  # (N, 4, 4) float32 world-to-camera
    intrinsics: np.ndarray | None = None
    depth_paths: list[Path] | None = None


def load_gt_depth(path: Path) -> np.ndarray:
    """
    7-Scenes depth frame in meters.

    - 65535 is the sensor's no-return code; it becomes 0 like every other invalid pixel

    Args:
        path: `*.depth.png`, uint16 millimeters.

    Returns:
        (H, W) float32 depth in meters, 0 where invalid.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = raw.astype(np.float32) / 1000.0
    depth[raw == 65535] = 0.0
    return depth
```

`_load_7scenes` sets `depth_paths=[seq_dir / f"{p.name.split('.')[0]}.depth.png" for p in images]`.
`_load_co3dv2` replaces the all-annotations fallback with:

```python
    if not annotations:
        raise ValueError(f"no annotations for sequence '{seq_name}' in {ann_path}")
```

and filters by `a.get("sequence_name") == seq_name` only (real CO3Dv2 always sets it; the path-prefix
match was a fixture artifact). `get_dataset` keeps its `KeyError` listing `sorted(_REGISTRY)`.

- [ ] **Step 4:** delete `tests/pointcloud/test_co3dv2_loader.py` (its fixture relied on the fallback; coverage now lives in `tests/evals/test_datasets.py`); gate green.
- [ ] **Step 5:** commit `refactor(evals): datasets to 7-Scenes, TUM, CO3Dv2; CO3Dv2 raises on unknown sequence`.

### Task 5: `gt_metrics.py`

**Files:** Create `evals/gt_metrics.py`, `tests/evals/test_gt_metrics.py`; delete `evals/{trajectory_metrics,metrics,trajectory_io}.py` and `tests/evals/{test_trajectory_metrics,test_metrics,test_metrics_auc,test_auc_metric,test_trajectory_io}.py`.

- [ ] **Step 1: failing tests**

```python
"""Tests for evals.gt_metrics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evals.gt_metrics import ate, auc_at_threshold, depth_error, rpe


def _trajectory(n=20, seed=0):
    """Random-walk camera-to-world poses."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(scale=0.3, size=(n, 3))).as_matrix()
    poses[:, :3, 3] = np.cumsum(rng.normal(scale=0.2, size=(n, 3)), axis=0)
    return poses


def _sim3(poses, s=2.5):
    """Apply a fixed similarity transform to camera-to-world poses."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.1, -0.4, 0.7]).as_matrix()
    T[:3, 3] = [1.0, -2.0, 0.5]
    out = T @ poses
    out[:, :3, 3] = s * (out[:, :3, 3] - T[:3, 3]) + T[:3, 3]
    return out


def test_ate_zero_on_identical():
    gt = _trajectory()
    assert ate(gt.copy(), gt)["rmse"] < 1e-9


def test_ate_sim3_invariant():
    gt = _trajectory()
    assert ate(_sim3(gt), gt)["rmse"] < 1e-6


def test_ate_detects_noise():
    gt = _trajectory()
    pred = gt.copy()
    pred[5, :3, 3] += 0.5
    out = ate(pred, gt)
    assert out["rmse"] > 0.05
    assert int(np.argmax(out["per_frame"])) == 5
    assert out["aligned_positions"].shape == (20, 3)


def test_rpe_zero_under_sim3():
    gt = _trajectory()
    out = rpe(_sim3(gt), gt)
    assert out["trans_rmse"] < 1e-6 and out["rot_rmse_deg"] < 1e-4


def test_auc_perfect_is_near_100():
    gt = _trajectory()
    # 1-degree bins: a perfect pair scores in bin 0, so AUC@30 = 100
    assert auc_at_threshold(_sim3(gt), gt, (30.0,))["auc_30"] == pytest.approx(100.0)


def test_auc_drops_with_rotation_error():
    gt = _trajectory()
    pred = gt.copy()
    pred[:10, :3, :3] = pred[:10, :3, :3] @ Rotation.from_rotvec([0, 0, np.radians(20)]).as_matrix()
    assert auc_at_threshold(pred, gt, (30.0,))["auc_30"] < 80.0


def test_depth_error_recovers_scale():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 4.0, size=(3, 8, 8)).astype(np.float32)
    gt[0, 0, 0] = 0.0
    out = depth_error(gt / 3.0, gt)
    assert out["scale"] == pytest.approx(3.0, rel=1e-5)
    assert out["median_rel_err"] < 1e-5
    assert out["coverage"] == pytest.approx(1.0)


def test_depth_error_no_overlap_raises():
    with pytest.raises(ValueError, match="no pixel"):
        depth_error(np.zeros((1, 4, 4)), np.ones((1, 4, 4)))
```

- [ ] **Step 2:** run → FAIL (`ModuleNotFoundError: evals.gt_metrics`).
- [ ] **Step 3: implement `evals/gt_metrics.py`**

```python
"""
Ground-truth metrics for a predicted trajectory and depth.

- poses are camera-to-world (N, 4, 4); ATE/RPE align with a Sim3 first (monocular scale)
- ATE/RPE are evo; AUC@k is the VGGT pairwise protocol (evo has none)
- reference-free quality is each stage's own report, not here
"""

from __future__ import annotations

import numpy as np
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D


def _evo_pair(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    """
    evo trajectories for (gt, pred), pred Sim3-aligned onto gt in place.
    """
    stamps = np.arange(len(gt_c2w), dtype=np.float64)
    ref = PoseTrajectory3D(poses_se3=list(gt_c2w.astype(np.float64)), timestamps=stamps)
    est = PoseTrajectory3D(poses_se3=list(pred_c2w.astype(np.float64)), timestamps=stamps)
    est.align(ref, correct_scale=True)
    return ref, est


def ate(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    """
    Absolute trajectory error on camera centers after Sim3 alignment.

    Args:
        pred_c2w: predicted camera-to-world poses, (N, 4, 4).
        gt_c2w: ground-truth camera-to-world poses, (N, 4, 4).

    Returns:
        rmse/mean/median/max in GT units, plus arrays `per_frame` (N,) and
        `aligned_positions` (N, 3).
    """
    ref, est = _evo_pair(pred_c2w, gt_c2w)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))

    stats = ape.get_all_statistics()
    return {
        "rmse": float(stats["rmse"]),
        "mean": float(stats["mean"]),
        "median": float(stats["median"]),
        "max": float(stats["max"]),
        "per_frame": np.asarray(ape.error),
        "aligned_positions": np.asarray(est.positions_xyz),
    }


def rpe(pred_c2w: np.ndarray, gt_c2w: np.ndarray, delta: int = 1) -> dict:
    """
    Relative pose error between frames `delta` apart, after Sim3 alignment.

    Args:
        pred_c2w: predicted camera-to-world poses, (N, 4, 4).
        gt_c2w: ground-truth camera-to-world poses, (N, 4, 4).
        delta: frame stride between compared pairs.

    Returns:
        `trans_rmse` in GT units and `rot_rmse_deg`.
    """
    ref, est = _evo_pair(pred_c2w, gt_c2w)

    out = {}
    for key, relation in (
        ("trans_rmse", metrics.PoseRelation.translation_part),
        ("rot_rmse_deg", metrics.PoseRelation.rotation_angle_deg),
    ):
        m = metrics.RPE(relation, delta=delta, delta_unit=metrics.Unit.frames, all_pairs=False)
        m.process_data((ref, est))
        out[key] = float(m.get_statistic(metrics.StatisticsType.rmse))
    return out


def auc_at_threshold(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray, thresholds: tuple[float, ...] = (30.0,)
) -> dict[str, float]:
    """
    Pairwise pose AUC, VGGT-X / VGGT-Long protocol.

    - every directed pair (i, j); error = max(relative-rotation angle, relative-translation direction angle)
    - direction angle is arccos(|cos|): scale- and sign-free, so no alignment is needed
    - AUC@t = mean of the cumulative 1-degree histogram over [0, t], in percent

    Args:
        pred_c2w: predicted camera-to-world poses, (N, 4, 4).
        gt_c2w: ground-truth camera-to-world poses, (N, 4, 4).
        thresholds: AUC cutoffs in degrees.

    Returns:
        `auc_<t>` in [0, 100] per threshold.
    """
    # Every directed pair: forward (i < j) and backward
    i1, i2 = np.triu_indices(len(gt_c2w), k=1)
    ii = np.concatenate([i1, i2])
    jj = np.concatenate([i2, i1])

    def _relative(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Relative rotation R_i^T R_j and camera-i-frame direction to camera j."""
        R, t = poses[:, :3, :3], poses[:, :3, 3]
        R_rel = np.einsum("nji,njk->nik", R[ii], R[jj])
        t_rel = np.einsum("nji,nj->ni", R[ii], t[jj] - t[ii])
        return R_rel, t_rel / (np.linalg.norm(t_rel, axis=1, keepdims=True) + 1e-15)

    R_pred, d_pred = _relative(pred_c2w.astype(np.float64))
    R_gt, d_gt = _relative(gt_c2w.astype(np.float64))

    # Rotation geodesic and translation-direction angle, both degrees
    traces = np.einsum("nij,nij->n", R_pred, R_gt)
    err_R = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
    err_T = np.degrees(np.arccos(np.clip(np.abs(np.sum(d_pred * d_gt, axis=1)), 0.0, 1.0)))
    err = np.maximum(err_R, err_T)

    # Histogram AUC per threshold over the same per-pair errors
    out = {}
    for t in thresholds:
        max_t = int(t)
        histogram, _ = np.histogram(err, bins=np.arange(max_t + 1))
        out[f"auc_{max_t}"] = float(np.mean(np.cumsum(histogram / len(err))) * 100.0)
    return out


def depth_error(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Predicted depth against GT after one median scale for the whole sequence.

    - a pixel counts only when both depths are positive
    - relative error is |s * pred - gt| / gt

    Args:
        pred: predicted depth on the GT pixel grid, any shape; 0 = no prediction.
        gt: GT depth in meters, same shape; 0 = invalid.

    Returns:
        scale, median_rel_err, p90_rel_err, frac_over_10pct, and coverage (share of valid GT
        pixels that also have a prediction).

    Raises:
        ValueError: no pixel has both depths.
    """
    gt_valid = gt > 0
    both = gt_valid & (pred > 0)
    if not both.any():
        raise ValueError("depth_error: no pixel has both a predicted and a GT depth")

    # One scale for the sequence: monocular depth is up to scale, like the poses
    scale = float(np.median(gt[both]) / np.median(pred[both]))
    rel = np.abs(scale * pred[both] - gt[both]) / gt[both]
    return {
        "scale": scale,
        "median_rel_err": float(np.median(rel)),
        "p90_rel_err": float(np.percentile(rel, 90)),
        "frac_over_10pct": float(np.mean(rel > 0.10)),
        "coverage": float(both.sum() / gt_valid.sum()),
    }
```

Note on `auc_at_threshold` trace: `trace(R_pred @ R_gt^T) = sum_ij R_pred_ij * R_gt_ij`, hence the `nij,nij` einsum.

- [ ] **Step 4:** run → PASS. Then parity check against the old code on a random trajectory (before deleting it): old `ate_translation(inv(pred), inv(gt))["rmse"]` equals new `ate` rmse to 1e-6; old `auc_at_threshold` `auc_30` equals new; old `rpe` equals new to 1e-5. Record numbers in the commit body.
- [ ] **Step 5:** delete the three old modules and their tests; gate; commit `refactor(evals): gt_metrics — evo ATE/RPE, AUC, GT depth error`.

### Task 6: Runner `evals/eval.py`

**Files:** Create `evals/eval.py`, `tests/evals/test_eval.py`; rewrite `evals/configs/7scenes.yaml`, `evals/configs/cross_model_chess.yaml`; delete `evals/scripts/` (incl. `eval_compare.py`, `__init__.py`) and `tests/evals/{test_eval_config,test_eval_compare,test_eval_gt_helpers}.py`.

- [ ] **Step 1: failing tests** — `tests/evals/test_eval.py`

```python
"""Tests for the evals grid runner; Reconstructor and zarr loading are stubbed."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from evals import eval as ev
from evals.datasets import EvalDataset


def _grid_yaml(tmp_path, **overrides):
    """Write a two-condition grid YAML over one 7-Scenes dataset."""
    raw = {
        "name": "t",
        "output_dir": str(tmp_path / "out"),
        "base": {"semantics": {"enabled": False}},
        "datasets": [{"name": "chess", "type": "7scenes", "seq_dir": "x", "max_frames": 5}],
        "conditions": {"omega": {"pointcloud": {"backend": "vggt_omega"}}, "lc": {"pointcloud": {"loop_closure": True}}},
    }
    raw.update(overrides)
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_build_grid_merges_base_and_override(tmp_path):
    cells = ev.build_grid(ev.load_eval_config(_grid_yaml(tmp_path)))
    assert [c.output_dir.name for c in cells] == ["chess__omega", "chess__lc"]
    assert cells[1].config == {"semantics": {"enabled": False}, "pointcloud": {"loop_closure": True}}


def test_load_eval_config_rejects_empty_conditions(tmp_path):
    with pytest.raises(ValueError, match="condition"):
        ev.load_eval_config(_grid_yaml(tmp_path, conditions={}))


def test_load_eval_config_rejects_unknown_dataset(tmp_path):
    ds = [{"name": "k", "type": "kitti", "seq_dir": "x", "max_frames": 5}]
    with pytest.raises(KeyError, match="kitti"):
        ev.load_eval_config(_grid_yaml(tmp_path, datasets=ds))


def test_stage_frames_gt_order(tmp_path):
    srcs = []
    for name in ["b.png", "a.png"]:
        (tmp_path / name).write_bytes(b"x")
        srcs.append(tmp_path / name)
    ev._stage_frames(srcs, tmp_path / "in")
    staged = sorted((tmp_path / "in").iterdir())
    assert [p.name for p in staged] == ["000000.png", "000001.png"]
    assert staged[0].resolve() == (tmp_path / "b.png").resolve()


def test_gt_permutation_reorders():
    paths = [Path("frame_000002"), Path("frame_000000"), Path("frame_000001")]
    assert ev._gt_permutation(paths, 3).tolist() == [1, 2, 0]


def test_gt_permutation_missing_frame_raises():
    with pytest.raises(ValueError, match="GT"):
        ev._gt_permutation([Path("frame_000000")], 2)


def test_pred_depth_at_gt_res_places_crop():
    out = ev._pred_depth_at_gt_res(np.ones((2, 2), np.float32), np.array([2, 1, 6, 5]), (6, 8))
    assert out[1:5, 2:6].min() == 1.0 and out.sum() == 16.0


def _fake_run(tmp_path, monkeypatch, n=6, write_report=True):
    """Stub the dataset, Reconstructor and zarr loader for one cell."""
    gt_c2w = np.tile(np.eye(4), (n, 1, 1))
    gt_c2w[:, 0, 3] = np.arange(n) * 0.1
    gt_c2w[:, 1, 3] = np.sin(np.arange(n))
    imgs = []
    for i in range(n):
        (tmp_path / f"{i}.png").write_bytes(b"x")
        imgs.append(tmp_path / f"{i}.png")
    ds = EvalDataset(images=imgs, gt_poses=np.linalg.inv(gt_c2w).astype(np.float32))
    monkeypatch.setattr(ev, "get_dataset", lambda _t: lambda *_a, **_k: ds)

    class FakeRecon:
        def __init__(self, config):
            self.config = {"splats": {"enabled": False}, **config}
            self.out = Path(config["output_path"])
            self.pointcloud_zarr = self.out / "vggt_omega" / "pointcloud.zarr"

        def run_pipeline(self):
            if write_report:
                (self.out / "vggt_omega").mkdir(parents=True)
                (self.out / "vggt_omega" / "reconstruction_quality_report.json").write_text('{"ok": 1}')

    # Reconstruction returns frames reversed; the runner must undo that
    rev = np.arange(n)[::-1]
    ff = SimpleNamespace(
        extrinsics=np.linalg.inv(gt_c2w[rev]).astype(np.float32),
        image_paths=[Path(f"frame_{i:06d}") for i in rev],
        depth=None,
        original_coords=None,
    )
    monkeypatch.setattr(ev, "Reconstructor", FakeRecon)
    monkeypatch.setattr(ev.FeedforwardResult, "load_zarr", classmethod(lambda cls, *a, **k: ff))
    return ev.EvalCell("chess", "7scenes", Path("x"), n, "omega", {}, tmp_path / "cell")


def test_run_cell_scores_reordered_poses(tmp_path, monkeypatch):
    cell = _fake_run(tmp_path, monkeypatch)
    m = ev.run_cell(cell)
    assert m["ate"]["rmse"] < 1e-6
    assert m["reports"]["reconstruction_quality_report"] == {"ok": 1}
    assert json.loads((cell.output_dir / "metrics.json").read_text())["condition"] == "omega"
    assert (cell.output_dir / "plots" / "trajectory.png").exists()


def test_run_cell_missing_report_raises(tmp_path, monkeypatch):
    cell = _fake_run(tmp_path, monkeypatch, write_report=False)
    with pytest.raises(ValueError, match="reconstruction_quality_report"):
        ev.run_cell(cell)


def test_run_grid_skips_done_cells_and_aggregates(tmp_path, monkeypatch):
    path = _grid_yaml(tmp_path)
    cells = ev.build_grid(ev.load_eval_config(path))
    done = cells[0].output_dir
    done.mkdir(parents=True)
    metrics = {"ate": {"rmse": 0.01}, "rpe": {"trans_rmse": 0.02, "rot_rmse_deg": 0.3}, "auc": {"auc_30": 90.0}, "depth": None}
    (done / "metrics.json").write_text(json.dumps(metrics))

    calls = []
    monkeypatch.setattr(ev.subprocess, "run", lambda cmd, check: calls.append(cmd))
    ev.run_grid(path)

    assert len(calls) == 1 and calls[0][-1] == "chess__lc"
    table = (tmp_path / "out" / "comparison.md").read_text()
    assert "chess__omega" in table and "0.0100" in table
```

- [ ] **Step 2:** run → FAIL (`cannot import name 'eval'` / missing attributes).
- [ ] **Step 3: implement `evals/eval.py`**

```python
"""
Ground-truth eval grid: datasets x conditions through the Reconstructor pipeline.

- a condition is a config override, merged over the grid's `base` block and configs/base.yaml
- one subprocess per cell for OOM isolation; a cell holding metrics.json is skipped
- run from the repo root: python -m evals.eval --config evals/configs/7scenes.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mergedeep import merge

from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud.feedforward import FeedforwardResult
from collab_splats.preproc.frames import frame_idx_from_path
from collab_splats.utils.io import write_json
from collab_splats.wrapper.reconstructor import Reconstructor
from evals.datasets import get_dataset, load_gt_depth
from evals.gt_metrics import ate, auc_at_threshold, depth_error, rpe

logger = logging.getLogger(__name__)

######## Constants

# AUC cutoffs reported per cell, degrees
_AUC_THRESHOLDS = (5.0, 15.0, 30.0)

# GT-depth pixel stride: pooling every pixel of 500 frames is ~150M values
_DEPTH_STRIDE = 4

######## Grid config


@dataclass
class EvalCell:
    """
    One grid cell: a dataset sequence under one condition.
    """

    dataset: str
    dataset_type: str
    seq_dir: Path
    max_frames: int
    condition: str
    config: dict[str, Any]  # base block + condition override; no input/output paths
    output_dir: Path


@dataclass
class EvalConfig:
    """
    A parsed grid YAML.
    """

    name: str
    output_dir: Path
    datasets: list[dict[str, Any]]
    conditions: dict[str, dict[str, Any]]
    base: dict[str, Any]


def load_eval_config(path: Path) -> EvalConfig:
    """
    Parse and validate a grid YAML.

    Args:
        path: grid YAML with name, output_dir, datasets, conditions and an optional base block.

    Returns:
        The parsed grid.

    Raises:
        ValueError: the grid has no datasets or no conditions.
        KeyError: a dataset names an unknown type.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not raw.get("conditions"):
        raise ValueError(f"{path}: a grid needs at least one condition")
    if not raw.get("datasets"):
        raise ValueError(f"{path}: a grid needs at least one dataset")

    # Resolve every loader now, so a typo fails before the first cell runs
    for ds in raw["datasets"]:
        get_dataset(ds["type"])

    return EvalConfig(
        name=raw["name"],
        output_dir=Path(raw["output_dir"]),
        datasets=raw["datasets"],
        conditions=raw["conditions"],
        base=raw.get("base") or {},
    )


def build_grid(cfg: EvalConfig) -> list[EvalCell]:
    """
    Expand a grid into cells, dataset-major.

    Args:
        cfg: the parsed grid.

    Returns:
        One cell per (dataset, condition), named `<dataset>__<condition>`.
    """
    cells = []
    for ds in cfg.datasets:
        for label, override in cfg.conditions.items():
            config = merge({}, copy.deepcopy(cfg.base), copy.deepcopy(override or {}))
            cells.append(
                EvalCell(
                    dataset=ds["name"],
                    dataset_type=ds["type"],
                    seq_dir=Path(ds["seq_dir"]),
                    max_frames=int(ds["max_frames"]),
                    condition=label,
                    config=config,
                    output_dir=cfg.output_dir / f"{ds['name']}__{label}",
                )
            )
    return cells


######## One cell


def _stage_frames(images: list[Path], input_dir: Path) -> None:
    """
    Symlink frames into input_dir as NNNNNN.<ext>, so filename order is GT order.

    - the pipeline names frame i `frame_{i:06d}`, so a pose's source index is its GT index
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(images):
        link = input_dir / f"{i:06d}{src.suffix.lower()}"
        if not link.is_symlink():
            link.symlink_to(Path(src).resolve())


def _gt_permutation(image_paths: list[Path], n_gt: int) -> np.ndarray:
    """
    Row of each GT frame in the reconstruction's frame order.

    - raises unless every GT frame is posed exactly once
    """
    gt_idx = np.array([frame_idx_from_path(p) for p in image_paths])
    if sorted(gt_idx.tolist()) != list(range(n_gt)):
        raise ValueError(f"reconstruction posed {len(gt_idx)} frames; GT has {n_gt} and every one must be posed once")
    return np.argsort(gt_idx)


def _pred_depth_at_gt_res(depth: np.ndarray, box: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """
    Model-res depth placed on the original frame through its crop box; 0 outside it.

    - nearest resize, so no depth is invented across edges
    - a box reaching past the frame (padding) is clipped to it
    """
    tl_x, tl_y, cr_x, cr_y = (int(round(float(v))) for v in box[:4])
    resized = cv2.resize(depth.astype(np.float32), (cr_x - tl_x, cr_y - tl_y), interpolation=cv2.INTER_NEAREST)

    # Paste the overlap of box and frame
    H, W = hw
    y0, x0, y1, x1 = max(tl_y, 0), max(tl_x, 0), min(cr_y, H), min(cr_x, W)
    canvas = np.zeros(hw, dtype=np.float32)
    canvas[y0:y1, x0:x1] = resized[y0 - tl_y : y1 - tl_y, x0 - tl_x : x1 - tl_x]
    return canvas


def _depth_metrics(ff: FeedforwardResult, perm: np.ndarray, depth_paths: list[Path]) -> dict[str, float]:
    """
    GT depth error over the sequence, pooled on a strided pixel grid.
    """
    preds, gts = [], []
    for gi, row in enumerate(perm):
        gt = load_gt_depth(depth_paths[gi])
        pred = _pred_depth_at_gt_res(np.asarray(ff.depth[row]), np.asarray(ff.original_coords[row]), gt.shape)
        preds.append(pred[::_DEPTH_STRIDE, ::_DEPTH_STRIDE])
        gts.append(gt[::_DEPTH_STRIDE, ::_DEPTH_STRIDE])
    return depth_error(np.stack(preds), np.stack(gts))


def _plot_cell(gt_c2w: np.ndarray, ate_out: dict, cell: EvalCell, plots_dir: Path) -> None:
    """
    Trajectory (GT vs Sim3-aligned prediction) and per-frame ATE PNGs.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 3D trajectory: GT against the aligned prediction
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    gt_pos, pred_pos = gt_c2w[:, :3, 3], ate_out["aligned_positions"]
    ax.plot(*gt_pos.T, label="gt", color="black", linewidth=2)
    ax.plot(*pred_pos.T, label=cell.condition, linewidth=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{cell.dataset} / {cell.condition}")
    ax.legend()
    fig.savefig(plots_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-frame ATE
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ate_out["per_frame"], label=f"{cell.condition} (RMSE={ate_out['rmse']:.3f} m)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.legend()
    fig.savefig(plots_dir / "ate_per_frame.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cell(cell: EvalCell) -> dict[str, Any]:
    """
    Reconstruct one cell through the pipeline and score it against ground truth.

    Args:
        cell: the grid cell to run.

    Returns:
        The payload written to `<cell>/metrics.json`.

    Raises:
        ValueError: the poses do not cover every GT frame once, or a stage report the
            config implies is missing.
    """
    dataset = get_dataset(cell.dataset_type)(cell.seq_dir, max_frames=cell.max_frames)
    input_dir = cell.output_dir / "input"
    run_dir = cell.output_dir / "run"
    _stage_frames(dataset.images, input_dir)

    # Run the pipeline as production does; the config picks the stages
    paths = {"input_path": str(input_dir), "output_path": str(run_dir)}
    recon = Reconstructor(merge({}, copy.deepcopy(cell.config), paths))
    t0 = time.perf_counter()
    recon.run_pipeline()
    elapsed = time.perf_counter() - t0

    # Poses in GT order; both sides camera-to-world for the metrics
    has_depth = dataset.depth_paths is not None
    ff = FeedforwardResult.load_zarr(
        recon.pointcloud_zarr,
        load_depth=has_depth,
        load_world_points=False,
        load_confidence=False,
        load_features=False,
        load_pixel_indices=False,
    )
    perm = _gt_permutation(ff.image_paths, len(dataset.images))
    pred_c2w = invert_poses(np.asarray(ff.extrinsics, dtype=np.float64)[perm])
    gt_c2w = invert_poses(dataset.gt_poses.astype(np.float64))

    # Ground-truth metrics
    ate_out = ate(pred_c2w, gt_c2w)
    depth = _depth_metrics(ff, perm, dataset.depth_paths) if has_depth else None

    # Every stage report the run wrote; the ones the config implies must be there
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.rglob("*_quality_report.json"))}
    expected = ["reconstruction_quality_report"]
    if recon.config["splats"]["enabled"]:
        expected.append("splats_quality_report")
    missing = [name for name in expected if name not in reports]
    if missing:
        raise ValueError(f"{cell.output_dir.name}: stages wrote no {missing}")

    payload = {
        "dataset": cell.dataset,
        "condition": cell.condition,
        "n_frames": len(dataset.images),
        "time_s": elapsed,
        "ate": {k: v for k, v in ate_out.items() if k not in ("per_frame", "aligned_positions")},
        "rpe": rpe(pred_c2w, gt_c2w),
        "auc": auc_at_threshold(pred_c2w, gt_c2w, _AUC_THRESHOLDS),
        "depth": depth,
        "reports": reports,
        "config": cell.config,
    }

    # Arrays for the notebook, then plots, then metrics.json last: it marks the cell done
    np.savez(
        cell.output_dir / "trajectories.npz",
        gt_c2w=gt_c2w,
        pred_c2w=pred_c2w,
        ate_per_frame=ate_out["per_frame"],
    )
    _plot_cell(gt_c2w, ate_out, cell, cell.output_dir / "plots")
    write_json(cell.output_dir / "metrics.json", payload)
    return payload


######## Grid


def _metric(section: dict | None, key: str) -> float:
    """
    One table value; nan when the section or key is absent.
    """
    value = (section or {}).get(key)
    return math.nan if value is None else float(value)


def _aggregate(output_dir: Path) -> None:
    """
    comparison.md and comparison.json over every finished cell.
    """
    rows = []
    for path in sorted(output_dir.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        rows.append(
            {
                "cell": path.parent.name,
                "ate_rmse": _metric(m["ate"], "rmse"),
                "rpe_trans": _metric(m["rpe"], "trans_rmse"),
                "rpe_rot_deg": _metric(m["rpe"], "rot_rmse_deg"),
                "auc_30": _metric(m["auc"], "auc_30"),
                "depth_med_rel": _metric(m.get("depth"), "median_rel_err"),
            }
        )

    lines = [
        "| cell | ATE RMSE | RPE trans | RPE rot deg | AUC@30 | depth med rel |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        values = " | ".join(f"{r[k]:.4f}" for k in ("ate_rmse", "rpe_trans", "rpe_rot_deg", "auc_30", "depth_med_rel"))
        lines.append(f"| {r['cell']} | {values} |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    write_json(output_dir / "comparison.json", rows)


def run_grid(config_path: Path, dry_run: bool = False) -> None:
    """
    Run every unfinished cell in its own subprocess, then aggregate.

    Args:
        config_path: grid YAML.
        dry_run: print each cell's command and run nothing.
    """
    config_path = Path(config_path).resolve()
    cfg = load_eval_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # One subprocess per cell; a finished cell is skipped
    for cell in build_grid(cfg):
        if (cell.output_dir / "metrics.json").exists():
            logger.info("skip %s: metrics.json exists", cell.output_dir.name)
            continue
        cmd = [sys.executable, "-m", "evals.eval", "--config", str(config_path), "--cell", cell.output_dir.name]
        if dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True)

    if not dry_run:
        _aggregate(cfg.output_dir)


def main() -> None:
    """
    CLI: `--config` runs the grid; `--cell` (internal) runs one cell in this process.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="grid YAML")
    parser.add_argument("--dry_run", action="store_true", help="print each cell command, run nothing")
    parser.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.cell is None:
        run_grid(args.config, dry_run=args.dry_run)
        return

    # Internal single-cell mode, launched by run_grid
    cells = {c.output_dir.name: c for c in build_grid(load_eval_config(args.config))}
    run_cell(cells[args.cell])


if __name__ == "__main__":
    main()
```

`print` in `run_grid --dry_run` is CLI output, not logging — allowed.

- [ ] **Step 4:** rewrite the configs.

`evals/configs/7scenes.yaml`:

```yaml
# 7-Scenes grid: three sequences x two backbones x baseline/BA/LC.
# Run from the repo root: python -m evals.eval --config evals/configs/7scenes.yaml
name: 7scenes
output_dir: evals/results/7scenes_grid

# Applied to every cell before its condition
base:
  semantics: {enabled: false}
  mesh: {enabled: false}

datasets:
  - {name: chess,  type: 7scenes, seq_dir: data/7scenes/chess/seq-01,  max_frames: 500}
  - {name: fire,   type: 7scenes, seq_dir: data/7scenes/fire/seq-01,   max_frames: 500}
  - {name: office, type: 7scenes, seq_dir: data/7scenes/office/seq-01, max_frames: 500}

# Label -> override merged over base
conditions:
  omega:     {pointcloud: {backend: vggt_omega}}
  omega_ba:  {pointcloud: {backend: vggt_omega, bundle_adjustment: true}}
  omega_lc:  {pointcloud: {backend: vggt_omega, loop_closure: true}}
  vggtx:     {pointcloud: {backend: vggtx}}
  vggtx_ba:  {pointcloud: {backend: vggtx, bundle_adjustment: true}}
  vggtx_lc:  {pointcloud: {backend: vggtx, loop_closure: true}}
```

`evals/configs/cross_model_chess.yaml`: same header/base, one chess dataset, conditions
`{vggtx, vggtx_lc, omega, omega_lc, mapanything, mapanything_lc}` in the same override form.

- [ ] **Step 5:** `git rm -r evals/scripts tests/evals/test_eval_config.py tests/evals/test_eval_compare.py tests/evals/test_eval_gt_helpers.py`; `--dry_run` both configs (prints 18 and 6 commands); gate green.
- [ ] **Step 6:** commit `refactor(evals): grid runner drives Reconstructor with config overrides`.

### Task 7: Delete splats / sky-mask scripts

Already inside `evals/scripts/` → removed by Task 6. If Task 6 left them (it removes the whole dir), this task is only:
- [ ] **Step 1:** delete `tests/evals/{test_eval_splats,test_analyze_splats}.py`; gate; commit `refactor(evals): drop splats and sky-mask scripts (stage reports cover them)`. (Order: do this with Task 6's `git rm` if the tests import `scripts.*` and would error at collection — then fold into Task 6's commit and say so in its body.)

### Task 8: Download tooling → README

- [ ] **Step 1:** `git rm evals/data/download_datasets.py tests/evals/test_download_7scenes.py`.
- [ ] **Step 2:** rewrite `evals/README.md` (~40 lines): purpose; layout; get data (7-Scenes `wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/<scene>.zip` + unzip seq zip into `data/7scenes/<scene>/seq-01`; TUM `wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz` + tar; CO3Dv2 via `facebookresearch/co3d` `download_dataset.py --download_categories <cat>`); grid YAML shape (`base`, `datasets`, `conditions`); run command; outputs per cell (`metrics.json`, `trajectories.npz`, `plots/`, `run/`) and `comparison.md`; metrics owned here vs stage reports. Verify each URL resolves with `curl -sI` before committing.
- [ ] **Step 3:** gate; commit `docs(evals): README — download recipe, grid config, outputs`.

### Task 9: Caller sweep

- [ ] **Step 1:** `grep -rn "evals/scripts\|scripts\.eval\|eval_gt\|eval_splats\|analyze_splats\|eval_sky_mask\|trajectory_metrics\|trajectory_io\|evals\.metrics\|pose_graph_diagnostics\|download_datasets\|extract_waymo" --include=*.py --include=*.md --include=*.ipynb --include=*.yaml . --exclude-dir=.worktrees --exclude-dir=results --exclude-dir=baselines --exclude-dir=superpowers`
- [ ] **Step 2:** fix each hit per the spec's caller table:
  - `CLAUDE.md`, `README.md`: `python -m evals.eval --config evals/configs/7scenes.yaml`; drop `--submap_size` advice.
  - `collab_splats/pointcloud/vda.py:89`: comment names `evals.eval`.
  - `docs/parity.md`: dead script paths → "removed in `<this branch's Task 2 SHA>`; last version at `2c0b2f06:<path>`".
  - `docs/known-test-failures.md`: drop entries for deleted tests.
  - notebooks (`ground_truth_evals`, `bundle_adjustment`, `train_splats`): edit JSON source cells with a small python script (`nbformat`), no re-execution; `train_splats` builds its splat inputs from the splats API instead of `eval_splats.inputs_from_pointcloud_zarr`.
  - sky-mask plan: append the A/B note (two runs, `mesh.mask_sky` flipped; numbers wait for the mesh report).
- [ ] **Step 3:** re-run Step 1 grep → only history (CHANGELOG, specs/plans, decisions) remains. Gate + `tests/docs` if present. Commit `docs: point callers at evals.eval`.

### Task 10: Conftest workaround

- [ ] **Step 1:** `tests/__init__.py` exists, so `tests/evals` imports as `tests.evals` and cannot shadow top-level `evals`. Delete `_fix_evals_import` + its docstring paragraph from `tests/pointcloud/conftest.py` and `tests/geometry/conftest.py` (and unused `sys` imports).
- [ ] **Step 2:** run each dir alone AND the full gate (`tests/evals` first then `tests/pointcloud`, and reversed order via explicit path order) — all green. If anything fails with an `evals` import error, revert the file and note why.
- [ ] **Step 3:** commit `test: drop dead evals-shadowing workaround`.

### Task 11: Docstring contract covers `evals`

- [ ] **Step 1:** in `tests/test_docstring_contract.py` add a source list for top-level `evals/*.py` alongside `PACKAGES`, and make the `relative_to` at line ~467 relative to `ROOT` for those files (so ids read `evals/eval.py`).
- [ ] **Step 2:** run it → fix any violation in `evals/*.py` (prose only); run again → PASS.
- [ ] **Step 3:** gate; commit `test(contract): cover evals/`.

### Task 12: Live cell, graph, report

- [ ] **Step 1:** in tmux, a one-cell grid YAML in the scratchpad (chess, `max_frames: 500`, condition `omega: {pointcloud: {backend: vggt_omega}}`, `output_dir` in scratchpad): `cd $WT && PYTHONPATH=$WT python -m evals.eval --config <yaml>`. Compare ATE RMSE with `cross_model_chess` `chess__vggt_omega__baseline` = **0.0151** (old runner, `submap_size 50`). Report both; a difference is expected if the pipeline's defaults differ from the old flags — say which.
- [ ] **Step 2:** `cd $WT && graphify update .`
- [ ] **Step 3:** final gate; report counts vs Task 0 and `git diff --stat clean/final..HEAD -- evals tests/evals`. Do not merge.

---

## Self-review

- Spec coverage: round 1 (Task 1), steps 1–7 (Tasks 2–8), caller sweep (9), testing incl. contract + live cell (10–12). Deviations listed at top.
- Types: `EvalCell` fields used identically in tests and runner; `ate` returns `per_frame`/`aligned_positions` consumed by `_plot_cell` and stripped before JSON; `_gt_permutation` returns rows in GT order used for both poses and depth.
- No placeholders except `test_load_7scenes_depth_paths`, which depends on that file's existing 7-Scenes fixture helper (read it first; assert `depth_paths` names match `images`).
