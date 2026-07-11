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
    """Align predicted trajectory to ground truth via Umeyama SE(3).

    Args:
        pred: (N, 4, 4) predicted world-to-cam poses
        gt:   (N, 4, 4) ground-truth world-to-cam poses

    Returns:
        (aligned_pred, T_align): Umeyama-aligned predicted poses and the SE(3)
        transform applied.
    """
    from .closure import umeyama_se3

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
    from .closure import umeyama_sim3

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
    rel_pred = np.linalg.inv(pred[:-delta]) @ pred[delta:]  # (N-δ, 4, 4)
    rel_gt = np.linalg.inv(gt[:-delta]) @ gt[delta:]  # (N-δ, 4, 4)
    err = np.linalg.inv(rel_gt) @ rel_pred  # (N-δ, 4, 4)

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

    # Sim3 alignment: c * R_a @ centers_pred.T + t_a ≈ centers_gt.T
    c, R_a, t_a = _umeyama_sim3(centers_pred.T, centers_gt.T)
    t_a = t_a.flatten()

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
