"""Loop-closure pose-graph evaluation helpers.

Provides instrumentation for the LC pose graph (per-iteration loss capture,
per-edge-type residual breakdown) and stubs for future ground-truth-based
metrics (ATE, RPE) that will be implemented when a GT pose dataset is sourced.
"""
from __future__ import annotations

import gtsam
import numpy as np

from .graph import PoseGraph


def _classify_edges(graph: gtsam.NonlinearFactorGraph) -> dict[str, list[int]]:
    """Partition factor indices by edge type.

    Loop edges use Robust(Huber) wrapping around BetweenFactorSL4.
    Sequential edges use Diagonal noise BetweenFactorSL4.
    Prior factors (anchor) are skipped.
    """
    groups: dict[str, list[int]] = {"sequential": [], "loop": []}
    for i in range(graph.size()):
        factor = graph.at(i)
        if not isinstance(factor, gtsam.BetweenFactorSL4):
            continue  # skip PriorFactorSL4 and others
        nm = factor.noiseModel()
        if isinstance(nm, gtsam.noiseModel.Robust):
            groups["loop"].append(i)
        else:
            groups["sequential"].append(i)
    return groups


def _per_edge_error(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    edge_groups: dict[str, list[int]],
) -> dict[str, float]:
    """Sum graph.at(i).error(values) over each edge group."""
    return {
        group_name: float(sum(graph.at(i).error(values) for i in indices))
        for group_name, indices in edge_groups.items()
    }


def capture_pose_graph_loss(
    pose_graph: PoseGraph,
    max_iterations: int = 100,
    relative_error_tol: float = 1e-5,
) -> dict:
    """Re-run LM optimization with per-iteration cost capture.

    Does NOT mutate pose_graph. Builds a fresh LM optimizer from
    pose_graph._graph + pose_graph._initial, iterates manually via
    optimizer.iterate(), records optimizer.error() each step.

    Iteration count is governed solely by the outer loop (max_iterations).
    GTSAM's internal max-iteration param is intentionally not set so that
    the manual loop is the sole authority on step count.

    Returns:
        iterations: list[float]            graph.error() per LM iter (incl. initial)
        per_edge_initial: dict[str, float] {'sequential', 'loop'} initial residuals
        per_edge_final: dict[str, float]   {'sequential', 'loop'} final residuals
        optimized_values: gtsam.Values     optimizer result
        converged: bool                    True if relative_error_tol was reached
        n_iterations: int                  number of iterate() calls (= len(iterations) - 1)
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(relative_error_tol)

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        pose_graph._graph, pose_graph._initial, params
    )

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
    per_edge_initial = _per_edge_error(pose_graph._graph, pose_graph._initial, groups)
    per_edge_final = _per_edge_error(pose_graph._graph, optimized, groups)

    return {
        "iterations": iterations,
        "per_edge_initial": per_edge_initial,
        "per_edge_final": per_edge_final,
        "optimized_values": optimized,
        "converged": converged,
        "n_iterations": len(iterations) - 1,
    }


# --- GT-phase stubs (implement when GT dataset arrives) ----------------------


def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align predicted trajectory to ground truth via Umeyama SE(3).

    Args:
        pred: (N, 4, 4) predicted world-to-cam poses
        gt:   (N, 4, 4) ground-truth world-to-cam poses

    Returns:
        (aligned_pred, T_align): Umeyama-aligned predicted poses and the SE(3)
        transform applied.
    """
    from .closure import umeyama_se3
    # Camera positions in world: for world-to-cam T, p_world = R^T @ (-t)
    R_pred = pred[:, :3, :3]
    t_pred = pred[:, :3, 3]
    p_pred = np.einsum("nij,nj->ni", R_pred.transpose(0, 2, 1), -t_pred)  # (N, 3)

    R_gt = gt[:, :3, :3]
    t_gt = gt[:, :3, 3]
    p_gt = np.einsum("nij,nj->ni", R_gt.transpose(0, 2, 1), -t_gt)  # (N, 3)

    # SE(3) only (no scale) — preserves metric scale of pred poses.
    # Use umeyama_sim3 if scale normalization is needed.
    # T_align: (4,4) such that p_gt ≈ T_align @ p_pred
    T_align = umeyama_se3(source=p_pred, target=p_gt)
    # Apply to full poses: aligned[i] = pred[i] @ inv(T_align)
    T_inv = np.linalg.inv(T_align)
    aligned = pred @ T_inv[None]   # (N, 4, 4)
    return aligned.astype(np.float32), T_align


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

    R_pred = aligned[:, :3, :3]
    R_gt   = gt[:, :3, :3]
    t_pred = aligned[:, :3, 3]
    t_gt   = gt[:, :3, 3]

    # Rotation error: trace-based angle between relative rotations
    R_rel  = R_pred @ R_gt.transpose(0, 2, 1)
    traces = np.trace(R_rel, axis1=1, axis2=2)
    err_R  = np.degrees(np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)))

    # Translation error: angular direction error in degrees (scale-free)
    # Camera position: c = -R^T @ t (for world-to-cam pose)
    c_pred = np.einsum("nij,nj->ni", R_pred.transpose(0, 2, 1), -t_pred)
    c_gt   = np.einsum("nij,nj->ni", R_gt.transpose(0, 2, 1),   -t_gt)
    c_pred_norm = c_pred / (np.linalg.norm(c_pred, axis=1, keepdims=True) + 1e-10)
    c_gt_norm   = c_gt   / (np.linalg.norm(c_gt,   axis=1, keepdims=True) + 1e-10)
    dots  = np.clip((c_pred_norm * c_gt_norm).sum(axis=1), -1.0, 1.0)
    err_T = np.degrees(np.arccos(dots))

    # Combined error: max of rotation and translation
    err = np.maximum(err_R, err_T)

    thresholds = np.linspace(0.0, max_threshold_deg, num_steps)
    accuracies = np.array([np.mean(err < t) for t in thresholds])
    auc = float(np.mean(accuracies) * 100.0)

    return {"auc_30": auc, "per_frame_err": err.tolist()}
