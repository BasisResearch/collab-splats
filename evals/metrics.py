"""Thin wrapper around `evo` (Zhang & Scaramuzza) ATE/RPE.

Both VGGT-SLAM and VGGT-Long publish numbers via evo, so we use it as the
metric source of truth. All inputs are TUM trajectory files; conversion from
KITTI 3×4-flat is handled by `trajectory_io.kitti_file_to_w2c` then
`trajectory_io.write_tum`.

Alignment knob:
    "none"  — no alignment (use when both trajectories share a metric frame)
    "se3"   — rigid alignment (correct for our SE(3) baseline / BA conditions)
    "sim3"  — similarity alignment (correct for monocular Sim(3) outputs:
              our LC condition, VGGT-Long, VGGT-SLAM)

The CLI equivalents are ``-a`` (se3) and ``-as`` (sim3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from evo.core import metrics as _evo_metrics
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.tools import file_interface

# Ensure collab_splats module is importable from evals scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ALIGN_CHOICES = ("none", "se3", "sim3")


def _load_pair(pred_path: Path | str, gt_path: Path | str):
    traj_ref = file_interface.read_tum_trajectory_file(str(gt_path))
    traj_est = file_interface.read_tum_trajectory_file(str(pred_path))
    return sync.associate_trajectories(traj_ref, traj_est)


def _apply_alignment(traj_ref, traj_est, align: str) -> None:
    if align not in _ALIGN_CHOICES:
        raise ValueError(f"align must be one of {_ALIGN_CHOICES}, got {align!r}")
    if align == "none":
        return
    traj_est.align(traj_ref, correct_scale=(align == "sim3"))


def _stats_dict(metric) -> dict[str, float]:
    s = metric.get_all_statistics()
    return {
        "rmse":   float(s["rmse"]),
        "mean":   float(s["mean"]),
        "median": float(s["median"]),
        "max":    float(s["max"]),
        "std":    float(s["std"]),
    }


def compute_ate(
    pred_path: Path | str,
    gt_path: Path | str,
    align: str = "se3",
) -> dict[str, float]:
    """Absolute Trajectory Error (translation part), evo APE.

    Returns a dict with keys: rmse, mean, median, max, std (all metres).
    """
    traj_ref, traj_est = _load_pair(pred_path, gt_path)
    _apply_alignment(traj_ref, traj_est, align)

    ape = _evo_metrics.APE(PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    return _stats_dict(ape)


def compute_rpe(
    pred_path: Path | str,
    gt_path: Path | str,
    align: str = "se3",
    delta: int = 1,
) -> dict[str, float]:
    """Relative Pose Error at frame stride `delta`.

    Returns a dict with `trans_rmse` (metres) and `rot_rmse_deg` (degrees).
    """
    traj_ref, traj_est = _load_pair(pred_path, gt_path)
    _apply_alignment(traj_ref, traj_est, align)

    rpe_t = _evo_metrics.RPE(
        PoseRelation.translation_part, delta=delta, delta_unit=Unit.frames,
        all_pairs=False,
    )
    rpe_t.process_data((traj_ref, traj_est))

    rpe_r = _evo_metrics.RPE(
        PoseRelation.rotation_angle_deg, delta=delta, delta_unit=Unit.frames,
        all_pairs=False,
    )
    rpe_r.process_data((traj_ref, traj_est))

    return {
        "trans_rmse":   float(rpe_t.get_statistic(_evo_metrics.StatisticsType.rmse)),
        "rot_rmse_deg": float(rpe_r.get_statistic(_evo_metrics.StatisticsType.rmse)),
    }


def compute_auc(
    pred_path: Path | str,
    gt_path: Path | str,
    align: str = "sim3",
    max_threshold_deg: float = 30.0,
) -> dict:
    """AUC@max_threshold_deg (CO3Dv2/VGGSfM protocol).

    Loads TUM files, aligns, converts to (N,4,4) pose arrays, delegates to
    collab_splats.pointcloud.loop_closure.eval.auc_at_threshold.

    Returns dict with 'auc_30' (float in [0,100]) and 'per_frame_err' (list[float]).
    """
    from collab_splats.pointcloud.loop_closure.eval import auc_at_threshold

    traj_ref, traj_est = _load_pair(pred_path, gt_path)
    _apply_alignment(traj_ref, traj_est, align)

    # evo stores poses_se3 as list of (4,4) arrays — stack to (N,4,4) cam-to-world
    pred_poses = traj_est.poses_se3
    gt_poses = traj_ref.poses_se3
    pred_c2w = np.stack(pred_poses).astype(np.float32)
    gt_c2w = np.stack(gt_poses).astype(np.float32)

    # auc_at_threshold expects world-to-cam poses; invert
    pred_w2c = np.linalg.inv(pred_c2w)
    gt_w2c = np.linalg.inv(gt_c2w)

    result = auc_at_threshold(pred_w2c, gt_w2c, max_threshold_deg=max_threshold_deg)
    return {
        "auc_30": result["auc_30"],
        "per_frame_err": result["per_frame_err"],
    }
