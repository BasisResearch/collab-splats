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

from collab_splats.geometry.transforms import project_to_so3


def _evo_pair(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    """
    evo trajectories (gt, pred), pred Sim3-aligned onto gt.

    - rotations snapped to SO(3): evo rejects 7-Scenes GT, off-orthonormal by ~1e-4
    """
    gt_c2w = gt_c2w.astype(np.float64)
    pred_c2w = pred_c2w.astype(np.float64)
    gt_c2w[:, :3, :3] = project_to_so3(gt_c2w[:, :3, :3])
    pred_c2w[:, :3, :3] = project_to_so3(pred_c2w[:, :3, :3])

    stamps = np.arange(len(gt_c2w), dtype=np.float64)
    ref = PoseTrajectory3D(poses_se3=list(gt_c2w), timestamps=stamps)
    est = PoseTrajectory3D(poses_se3=list(pred_c2w), timestamps=stamps)
    est.align(ref, correct_scale=True)
    return ref, est


def ate(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    """
    Absolute trajectory error on camera centers after Sim3 alignment.

    Args:
        pred_c2w: predicted camera-to-world poses, (N, 4, 4).
        gt_c2w: ground-truth camera-to-world poses, (N, 4, 4).

    Returns:
        rmse, mean, median and max in GT units, plus arrays `per_frame` (N,) and
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


def rpe(pred_c2w: np.ndarray, gt_c2w: np.ndarray, delta: int = 1) -> dict[str, float]:
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

    # One evo RPE per relation, same pairs
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
    pred_c2w: np.ndarray, gt_c2w: np.ndarray, thresholds: tuple[float, ...] = (30.0,), n_frames: int | None = None
) -> dict[str, float]:
    """
    Pairwise pose AUC, VGGT-X / VGGT-Long protocol.

    - every directed pair (i, j); error = max(relative-rotation angle, direction angle)
    - direction angle is arccos(|cos|): scale- and sign-free, so no alignment is needed
    - AUC@t = mean of the cumulative 1-degree histogram over [0, t], in percent
    - n_frames > N: pairs touching an unregistered frame count as failures, as COLMAP's
      benchmark scores them, so dropping frames cannot raise the AUC

    Args:
        pred_c2w: predicted camera-to-world poses, (N, 4, 4).
        gt_c2w: ground-truth camera-to-world poses, (N, 4, 4).
        thresholds: AUC cutoffs in degrees.
        n_frames: GT frame count, registered or not; None means N.

    Returns:
        `auc_<t>` in [0, 100] per threshold.
    """
    # Every directed pair: forward (i < j) and backward
    i1, i2 = np.triu_indices(len(gt_c2w), k=1)
    ii = np.concatenate([i1, i2])
    jj = np.concatenate([i2, i1])

    def _relative(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Relative rotation R_i^T R_j and unit direction to camera j in camera i's frame.
        """
        R, t = poses[:, :3, :3], poses[:, :3, 3]
        R_rel = np.einsum("nji,njk->nik", R[ii], R[jj])
        t_rel = np.einsum("nji,nj->ni", R[ii], t[jj] - t[ii])
        return R_rel, t_rel / (np.linalg.norm(t_rel, axis=1, keepdims=True) + 1e-15)

    R_pred, d_pred = _relative(pred_c2w.astype(np.float64))
    R_gt, d_gt = _relative(gt_c2w.astype(np.float64))

    # Rotation geodesic, trace(R_pred @ R_gt^T), and direction angle, both degrees
    traces = np.einsum("nij,nij->n", R_pred, R_gt)
    err_R = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))
    err_T = np.degrees(np.arccos(np.clip(np.abs(np.sum(d_pred * d_gt, axis=1)), 0.0, 1.0)))
    err = np.maximum(err_R, err_T)

    # Histogram AUC per threshold; the denominator is every directed GT pair
    n = len(gt_c2w) if n_frames is None else n_frames
    n_pairs = n * (n - 1)
    out = {}
    for t in thresholds:
        max_t = int(t)
        histogram, _ = np.histogram(err, bins=np.arange(max_t + 1))
        out[f"auc_{max_t}"] = float(np.mean(np.cumsum(histogram / n_pairs)) * 100.0)
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
        scale, median_rel_err, p90_rel_err, frac_over_10pct, and coverage (share of valid
        GT pixels that also have a prediction).

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
