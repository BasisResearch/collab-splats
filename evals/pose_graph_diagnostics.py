"""
Loop-closure pose-graph diagnostics: loss capture.

- capture_pose_graph_loss: per-iteration LM cost and per-edge-type residuals
"""

from __future__ import annotations

import gtsam

from collab_splats.geometry.loop_closure.graph import PoseGraph

########################################################################
########## Pose-graph loss capture #####################################
########################################################################


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

    # Manual iterate() loop (not optimizer.optimize()) so we can capture the cost
    # after every step; the outer max_iterations is the sole step-count authority
    # (GTSAM's own internal max-iteration param is intentionally left unset).
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

    # Per-edge-type residual breakdown, before and after optimization.
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
