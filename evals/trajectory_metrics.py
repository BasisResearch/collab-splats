"""
Ground-truth trajectory metrics: ATE, RPE and pairwise AUC.

- used by evals/scripts/eval.py, evals/scripts/ba_start_at_gt.py and evals/metrics.py
- ATE and AUC align to ground truth first (Umeyama); RPE is alignment-free
"""

from __future__ import annotations

import numpy as np

from collab_splats.geometry.transforms import umeyama_se3, umeyama_sim3


def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align predicted trajectory to ground truth via Umeyama SE(3).

    Args:
        pred: (N, 4, 4) predicted world-to-cam poses
        gt:   (N, 4, 4) ground-truth world-to-cam poses

    Returns:
        (aligned_pred, T_align): Umeyama-aligned predicted poses and the SE(3)
        transform applied.
    """
    R_pred = pred[:, :3, :3]
    t_pred = pred[:, :3, 3]
    p_pred = np.einsum("nij,nj->ni", R_pred.transpose(0, 2, 1), -t_pred)  # (N, 3)

    R_gt = gt[:, :3, :3]
    t_gt = gt[:, :3, 3]
    p_gt = np.einsum("nij,nj->ni", R_gt.transpose(0, 2, 1), -t_gt)  # (N, 3)

    T_align = umeyama_se3(source=p_pred, target=p_gt)
    T_inv = np.linalg.inv(T_align)
    aligned = pred @ T_inv[None]
    return aligned.astype(np.float32), T_align


def ate_translation(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Absolute Trajectory Error on translation component.

    Args:
        pred: (N, 4, 4) predicted poses
        gt:   (N, 4, 4) ground-truth poses

    Returns:
        {'rmse', 'mean', 'median', 'max', 'per_frame'} after Umeyama alignment.
    """

    # Extract camera positions in world from world-to-cam poses
    def _cam_pos(poses: np.ndarray) -> np.ndarray:
        R_ = poses[:, :3, :3]
        t_ = poses[:, :3, 3]
        return np.einsum("nij,nj->ni", R_.transpose(0, 2, 1), -t_)

    p_pred = _cam_pos(pred)  # (N, 3)
    p_gt = _cam_pos(gt)  # (N, 3)

    # Align predicted positions to GT via Sim3 (scale + rotation + translation).
    # Must use Sim3 (not SE3) to match evo's correct_scale=True — VGGT depth predictions
    # carry an unknown global scale factor that SE3 alignment cannot remove.
    s, R_align, t_align = umeyama_sim3(source=p_pred, target=p_gt)
    p_aligned = (s * R_align @ p_pred.T).T + t_align  # (N, 3)

    errs = np.linalg.norm(p_aligned - p_gt, axis=1)
    return {
        "rmse": float(np.sqrt((errs**2).mean())),
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "max": float(errs.max()),
        "per_frame": errs,
    }


def rpe(pred: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Relative Pose Error at frame stride delta.

    Args:
        pred:  (N, 4, 4) predicted poses
        gt:    (N, 4, 4) ground-truth poses
        delta: frame stride between compared pose pairs

    Returns:
        {'trans_rmse', 'rot_rmse_deg'} relative pose error statistics.
    """
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

    return {
        "trans_rmse": float(np.sqrt((t_err**2).mean())),
        "rot_rmse_deg": float(np.sqrt((r_err_deg**2).mean())),
    }


def auc_at_threshold(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: tuple[float, ...] = (30.0,),
) -> dict:
    """AUC@max_threshold_deg — VGGT-X / VGGT-Long pairwise protocol.

    Aligns pred to gt via Umeyama Sim3 on camera centers. Computes rotation
    and translation direction errors across all N*(N-1) directed pairs
    (both i→j and j→i for symmetry). AUC = mean of cumulative histogram at
    1° integer bins up to max_threshold_deg.

    Rotation error: SO3 geodesic angle between relative rotations.
    Translation error: arccos(|cos θ|) between normalized relative translations
    (scale-free, sign-ambiguity-free).

    Args:
        pred: (N, 4, 4) predicted cam-to-world poses (t column = camera center)
        gt:   (N, 4, 4) ground-truth cam-to-world poses
    Returns:
        {"auc_{t}": float in [0, 100] for each t in thresholds, "per_pair_err": list[float]}
    """
    R_pred = pred[:, :3, :3]
    t_pred = pred[:, :3, 3]  # camera centers (c2w convention)
    R_gt = gt[:, :3, :3]
    t_gt = gt[:, :3, 3]  # camera centers (c2w convention)

    # Camera centers = translation column directly (c2w convention)
    centers_pred = t_pred
    centers_gt = t_gt

    # Sim3 alignment: c * R_a @ centers_pred + t_a ≈ centers_gt
    c, R_a, t_a = umeyama_sim3(source=centers_pred, target=centers_gt)

    # Apply alignment: R_aligned[i] = R_a @ R_pred[i], t_aligned[i] = c*R_a@t_pred[i] + t_a
    # For relative rotation R_i^T@R_j, R_a cancels. For relative translation direction,
    # both R_a and t_a cancel after normalisation — so alignment affects neither metric.
    R_aligned = np.einsum("ij,njk->nik", R_a, R_pred)  # (N, 3, 3)
    t_aligned = c * np.einsum("ij,nj->ni", R_a, t_pred) + t_a  # (N, 3)

    # All N*(N-1) directed pairs: forward (i1→i2) + backward (i2→i1)
    N = len(pred)
    i1, i2 = np.triu_indices(N, k=1)
    ii = np.concatenate([i1, i2])
    jj = np.concatenate([i2, i1])

    # Relative rotation: R_i^T @ R_j  (R_a cancels: (R_a@R_i)^T @ (R_a@R_j) = R_i^T@R_j)
    R_rel_pred = np.einsum("nij,njk->nik", R_aligned[ii].transpose(0, 2, 1), R_aligned[jj])
    R_rel_gt = np.einsum("nij,njk->nik", R_gt[ii].transpose(0, 2, 1), R_gt[jj])

    # Rotation error: geodesic angle
    R_err_mat = np.einsum("nij,nkj->nik", R_rel_pred, R_rel_gt)  # R_rel_pred @ R_rel_gt^T
    traces = np.trace(R_err_mat, axis1=1, axis2=2)
    err_R = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))

    # Relative translation direction in camera i's frame: R_i^T @ (C_j - C_i)
    # For c2w: R^T maps world→camera; t is camera center. t_a cancels in delta.
    t_rel_pred = np.einsum("nij,nj->ni", R_aligned[ii].transpose(0, 2, 1), t_aligned[jj] - t_aligned[ii])
    t_rel_gt = np.einsum("nij,nj->ni", R_gt[ii].transpose(0, 2, 1), t_gt[jj] - t_gt[ii])

    # Translation direction error: arccos(|cos θ|) — scale-free, sign-ambiguity-free
    t_pred_n = t_rel_pred / (np.linalg.norm(t_rel_pred, axis=1, keepdims=True) + 1e-15)
    t_gt_n = t_rel_gt / (np.linalg.norm(t_rel_gt, axis=1, keepdims=True) + 1e-15)
    dots = np.clip(np.sum(t_pred_n * t_gt_n, axis=1), -1.0, 1.0)
    err_T = np.degrees(np.arccos(np.abs(dots)))

    err = np.maximum(err_R, err_T)

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
