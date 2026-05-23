from __future__ import annotations

import logging
import numpy as np
import torch
import gtsam
from gtsam import BetweenFactorPose3, PriorFactorPose3

from .submap import Submap

log = logging.getLogger(__name__)

# SE(3) noise: 6-DOF [rotation (3) + translation (3)]
# gtsam-develop SL(4) (15-DOF) requires Python 3.11 wheels; SE(3) is equivalent
# for rigid camera poses and available in gtsam 4.2.
# Starting sigmas — ADR 005 trigger field tuning.
ODOM_INTRA_SIGMA_R, _ODOM_INTRA_SIGMA_T = 0.02, 0.05  # tight: VGGT/MapAnything intra-window
INTER_SIGMA_R, _INTER_SIGMA_T = 0.05, 0.20            # looser at submap boundary
_LOOP_INTRA_SIGMA_R, _LOOP_INTRA_SIGMA_T = 0.10, 0.30  # rare
_LOOP_INTER_SIGMA_R, _LOOP_INTER_SIGMA_T = 0.15, 0.50  # loosest; Huber handles outliers
_ANCHOR_NOISE_SIGMA = 1e-6
_HUBER_K = 1.0
_LOOP_DOWNWEIGHT_ALPHA = 0.1


def _noise(sigma: float) -> gtsam.noiseModel.Base:
    return gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * sigma)


def _noise6(sigma_r: float, sigma_t: float) -> gtsam.noiseModel.Diagonal:
    """Diagonal SE(3) noise: [σ_r]*3 + [σ_t]*3."""
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_r] * 3 + [sigma_t] * 3, dtype=np.float64))


def _pose3(mat: np.ndarray) -> gtsam.Pose3:
    """Convert (4, 4) world-to-cam matrix to gtsam.Pose3."""
    R = gtsam.Rot3(mat[:3, :3].astype(np.float64))
    t = gtsam.Point3(mat[:3, 3].astype(np.float64))
    return gtsam.Pose3(R, t)


def _relative_pose3(pose_i: np.ndarray, pose_j: np.ndarray) -> gtsam.Pose3:
    """Compute SE(3) relative transform: inv(pose_i) @ pose_j."""
    p_i = _pose3(pose_i)
    p_j = _pose3(pose_j)
    return p_i.between(p_j)


class PoseGraph:
    """GTSAM SE(3) pose graph for loop closure trajectory correction.

    Uses Pose3/BetweenFactorPose3 (SE(3), 6 DOF). Adapted from
    MIT-SPARK/VGGT-SLAM graph.py (which uses SL(4), 15 DOF — unavailable
    on Python 3.10). Frame keys: gtsam.symbol('x', global_frame_index).
    """

    def __init__(self):
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        # Three-way noise split (F6, item 17). Same-name `_intra_noise` aliases
        # `_odom_intra_noise` — odom edges are the original intra-submap edges.
        self._odom_intra_noise = _noise6(ODOM_INTRA_SIGMA_R, _ODOM_INTRA_SIGMA_T)
        self._intra_noise = self._odom_intra_noise  # alias for backward-compat in tests
        self._inter_noise = _noise6(INTER_SIGMA_R, _INTER_SIGMA_T)
        self._loop_intra_noise = _noise6(_LOOP_INTRA_SIGMA_R, _LOOP_INTRA_SIGMA_T)
        self._loop_inter_noise = _noise6(_LOOP_INTER_SIGMA_R, _LOOP_INTER_SIGMA_T)
        self._anchor_noise = _noise(_ANCHOR_NOISE_SIGMA)
        self._frame_offset: dict[int, int] = {}  # submap_id → global frame start index
        self._submaps: list[Submap] = []
        self._total_frames = 0

    def add_submaps(self, submaps: list[Submap], overlap_frames: int = 4) -> None:
        """Add submaps: intra-submap edges + inter-submap edges (F1) + chained init values (F2)."""
        from collab_splats.pointcloud.loop_closure.alignment import overlap_region_align

        accumulated_T = np.eye(4, dtype=np.float32)
        accumulated_Ts: list[np.ndarray] = []

        for i, submap in enumerate(submaps):
            if i > 0:
                T_local = overlap_region_align(submaps[i - 1], submap, overlap_frames)
                accumulated_T = accumulated_T @ T_local
            accumulated_Ts.append(accumulated_T.copy())

            start = self._total_frames
            self._frame_offset[submap.submap_id] = start
            self._submaps.append(submap)
            k = submap.poses.shape[0]
            inv_T = np.linalg.inv(accumulated_T).astype(np.float32)

            for local_i, pose in enumerate(submap.poses):
                global_idx = start + local_i
                key = gtsam.symbol('x', global_idx)

                world_pose = pose @ inv_T  # F2: express initial value in common world frame
                self._initial.insert(key, _pose3(world_pose))

                if global_idx == 0:
                    self._graph.add(PriorFactorPose3(key, _pose3(world_pose), self._anchor_noise))

                if local_i > 0:
                    prev_world_pose = submap.poses[local_i - 1] @ inv_T
                    prev_key = gtsam.symbol('x', global_idx - 1)
                    rel = _relative_pose3(prev_world_pose, world_pose)
                    self._graph.add(BetweenFactorPose3(prev_key, key, rel, self._intra_noise))

            # F1: inter-submap edge
            if i > 0:
                last_prev_key = gtsam.symbol('x', start - 1)
                first_curr_key = gtsam.symbol('x', start)
                prev_inv_T = np.linalg.inv(accumulated_Ts[-2]).astype(np.float32)
                last_prev_world = submaps[i - 1].poses[-1] @ prev_inv_T
                first_curr_world = submap.poses[0] @ inv_T
                rel_inter = _relative_pose3(last_prev_world, first_curr_world)
                self._graph.add(BetweenFactorPose3(last_prev_key, first_curr_key, rel_inter, self._inter_noise))

            self._total_frames += k

    def add_loop_edges(self, lc_submaps: list[Submap]) -> None:
        """Add loop closure constraints from verified LC submaps.

        Each LC submap has exactly 2 frames: [query_frame, detected_frame].
        Applies per-edge translation-magnitude down-weighting (item 16) +
        Huber kernel wrap (item 10). Same-submap loops use `_loop_intra_noise`,
        cross-submap loops use `_loop_inter_noise` (F6, item 17).
        """
        for lc in lc_submaps:
            if lc.poses.shape[0] != 2:
                log.warning("LC submap %d has %d frames, expected 2 — skipping", lc.submap_id, lc.poses.shape[0])
                continue

            q_key = self._find_global_key(lc.image_paths[0])
            d_key = self._find_global_key(lc.image_paths[1])
            if q_key is None or d_key is None:
                log.warning("LC submap %d: could not resolve global keys — skipping", lc.submap_id)
                continue

            rel = _relative_pose3(lc.poses[0], lc.poses[1])

            base = self._loop_intra_noise if self._same_submap(lc) else self._loop_inter_noise
            t_norm = float(np.linalg.norm(lc.poses[1][:3, 3] - lc.poses[0][:3, 3]))
            scale = 1.0 + _LOOP_DOWNWEIGHT_ALPHA * (t_norm ** 2)
            scaled_sigmas = base.sigmas() * scale
            scaled = gtsam.noiseModel.Diagonal.Sigmas(scaled_sigmas)
            robust = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(_HUBER_K),
                scaled,
            )
            self._graph.add(BetweenFactorPose3(q_key, d_key, rel, robust))

    def _same_submap(self, lc: Submap) -> bool:
        """True if both LC frames belong to the same parent submap (rare but possible)."""
        a, b = lc.image_paths[0], lc.image_paths[1]
        for s in self._submaps:
            if a in s.image_paths and b in s.image_paths:
                return True
        return False

    def _find_global_key(self, image_path) -> int | None:
        for submap in self._submaps:
            for local_i, p in enumerate(submap.image_paths):
                if p == image_path:
                    global_idx = self._frame_offset[submap.submap_id] + local_i
                    return gtsam.symbol('x', global_idx)
        return None

    def optimize(self) -> dict[int, np.ndarray]:
        """Run Levenberg-Marquardt. Returns {submap_id: corrected_poses (K, 4, 4)}."""
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial, params)
            result = optimizer.optimize()
        except Exception as e:
            log.warning("GTSAM optimization failed: %s — returning uncorrected poses", e)
            return {s.submap_id: s.poses for s in self._submaps}

        corrected: dict[int, np.ndarray] = {}
        for submap in self._submaps:
            start = self._frame_offset[submap.submap_id]
            k = submap.poses.shape[0]
            poses = np.stack([
                result.atPose3(gtsam.symbol('x', start + i)).matrix().astype(np.float32)
                for i in range(k)
            ])
            corrected[submap.submap_id] = poses
        return corrected


# ---------------------------------------------------------------------------
# Sim(3) pose graph — pypose-based, replaces GTSAM SE(3) for LC optimization.
# PoseGraph (GTSAM) kept above as upgrade path to SL(4).
# ---------------------------------------------------------------------------


class Sim3PoseGraph(torch.nn.Module):
    """Pose graph on Sim(3) manifold optimized via pypose LM.

    One node per submap. Sequential and loop edges are BetweenFactor residuals:
    log(T_j^{-1} ⊗ T_i ⊗ T_ij) → 0 at optimum.

    Node format: pp.Sim3 data layout = [t(3), q(4, xyzw), s(1)], shape (8,).
    """

    def __init__(self, initial_poses: np.ndarray) -> None:
        """
        Args:
            initial_poses: (N, 8) float32 in pp.Sim3 data format.
        """
        import pypose as pp
        super().__init__()
        self.poses = pp.Parameter(
            pp.Sim3(torch.from_numpy(initial_poses.copy()).float()),
        )

    def forward(
        self,
        ii: torch.Tensor,
        jj: torch.Tensor,
        T_ij: "pp.Sim3",
        weights: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Compute between-factor residuals.

        Args:
            ii: (E,) long — source node indices.
            jj: (E,) long — target node indices.
            T_ij: (E,) pp.Sim3 — relative transforms.
            weights: optional (E,) per-edge weights.

        Returns:
            (E, 7) residual tensor; zero at optimum.
        """
        residuals = (self.poses[jj].Inv() @ self.poses[ii] @ T_ij).Log()
        if weights is not None:
            residuals = residuals * weights[:, None]
        return residuals

    def optimized_poses(self) -> np.ndarray:
        """Returns (N, 8) float32 optimized Sim3 data after optimization."""
        return self.poses.data.detach().cpu().numpy()
