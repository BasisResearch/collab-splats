"""
Levenberg-Marquardt bundle adjustment over VGGSfM tracks.

- BundleAdjustmentConfig: solver, filter and track-extraction settings
- BundleAdjustment: refines poses and focal from arrays, K at model resolution
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from zarr.codecs import BloscCodec

import pypose as pp
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.pysolvers import PCG
from vggt.dependency.track_predict import predict_tracks

from collab_splats.geometry.projection import project
from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    invert_poses,
    project_to_so3,
    umeyama_sim3,
)

__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]

logger = logging.getLogger(__name__)


########################################################################
# Configuration
########################################################################


@dataclass
class BundleAdjustmentConfig:
    """
    LM bundle adjustment settings, grouped by the step that reads them.

    - tracks: VGGSfM extraction and its cache
    - filter: which observations enter the solve
    - solve: the LM problem itself
    - runtime: where it runs

    Args:
        max_query_pts: extraction query points (upstream demo default).
        query_frame_num: extraction query frames (upstream demo default).
        fine_tracking: VGGSfM fine refinement; coarse-only tracks are ~1-2 px off.
        tracks_cache_dir: zarr track-cache dir; None always extracts.
        vis_thresh: min VGGSfM visibility score for an observation.
        max_reproj_error: pre-solve pixel reprojection gate; None skips the filter.
        min_inliers_per_frame: frames below this inlier count sit out the solve.
        lm_steps: LM iterations per solve, all run, no early stop.
        shared_camera: one focal per scene (per-frame K spread is model noise); False fits one per frame.
        increment_size: 0 = one global solve; 1..N-1 = frames added per incremental solve.
        device: CUDA device ("cuda", "cuda:1"), None = auto; CPU unsupported (bae LM is CUDA-only).
    """

    # Tracks
    max_query_pts: int = 4096
    query_frame_num: int = 8
    fine_tracking: bool = True
    tracks_cache_dir: Path | None = None

    # Filter
    vis_thresh: float = 0.2
    max_reproj_error: float | None = 4.0
    min_inliers_per_frame: int = 64

    # Solve
    lm_steps: int = 40
    shared_camera: bool = True
    increment_size: int = 0

    # Runtime
    device: str | None = None


########################################################################
# BundleAdjustment
########################################################################


class BundleAdjustment:
    """
    Refines camera poses and focal via VGGSfM tracks and LM bundle adjustment.

    - refines arrays whose K is at model resolution
    - the refine stage refuses sfm results
    - does not reproject points; the caller re-derives them under the new poses
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        """
        Store the settings and start an empty loss history.

        - config None uses the defaults
        """
        self.config = config or BundleAdjustmentConfig()

        # Populated per _optimize() call with that call's per-step LM losses
        self.loss_history: list[list[float]] = []

    def extract_tracks(
        self,
        images: np.ndarray,
        confidence: np.ndarray,
        world_points: np.ndarray,
        image_paths: list | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        VGGSfM tracks for the frames; cached under tracks_cache_dir when one is set.

        - no cache dir or no image_paths: always extracts, nothing is cached
        - a key mismatch or an unreadable store is deleted and rebuilt

        Args:
            images: (N, 3, H, W) model-grid frames, the track source.
            confidence: (N, H, W) per-pixel confidence for query-point sampling.
            world_points: (N, H, W, 3) the tracks are lifted from.
            image_paths: frame paths for the cache key; None skips the cache.

        Returns:
            tracks (N, P, 2), vis_scores (N, P) and pts3d_tracks (P, 3), all float32.
        """
        cfg = self.config

        # No cache configured, or nothing to key it on
        if cfg.tracks_cache_dir is None or not image_paths:
            return extract_tracks_vggsfm(images, confidence, world_points, cfg)

        # Cache location and the key a valid store must carry
        cache_path = Path(cfg.tracks_cache_dir) / "tracks.zarr"
        expected_key = _compute_tracks_cache_key(image_paths, world_points, cfg)

        # Attempt to read from existing cache; validate key before using
        if cache_path.exists():
            try:
                store = zarr.open(str(cache_path), mode="r")
                if store.attrs.get("cache_key") == expected_key:
                    logger.info("Track cache hit (skipping VGGSfM extraction): %s", cache_path)
                    return (
                        store["tracks"][:],
                        store["vis_scores"][:],
                        store["pts3d_tracks"][:],
                    )
                logger.warning("Track cache key mismatch, re-extracting: %s", cache_path)

            # Unreadable store: rebuild it
            # - missing array: KeyError
            # - bad or missing metadata: JSONDecodeError / GroupNotFoundError (ValueError, OSError)
            except (KeyError, ValueError, OSError) as exc:
                logger.warning("Track cache unreadable (%s), re-extracting: %s", exc, cache_path)
            shutil.rmtree(cache_path)

        # Extract and persist to zarr cache
        tracks, vis_scores, pts3d_tracks = extract_tracks_vggsfm(images, confidence, world_points, cfg)

        # One lz4 chunk per array; the key stamp makes the store valid
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(cache_path), mode="w")
        for key, arr in [("tracks", tracks), ("vis_scores", vis_scores), ("pts3d_tracks", pts3d_tracks)]:
            store.create_array(key, data=arr, chunks=arr.shape, compressors=lz4)
        store.attrs["cache_key"] = expected_key
        logger.debug("Track cache saved: %s", cache_path)

        return tracks, vis_scores, pts3d_tracks

    def _refine_global(
        self,
        extrinsics: np.ndarray,
        tracks: np.ndarray,
        vis_scores: np.ndarray,
        pts3d_tracks: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One BA solve over all N frames at once.

        - (N, 4, 4) world-to-cam in; returns refined (N, 3, 4) extrinsics and (N, 3, 3) K
        """
        _, refined_extrinsics, refined_intrinsics = self._optimize(
            pts3d_tracks,
            extrinsics[:, :3, :],
            intrinsics,
            tracks,
            vis_scores,
        )
        return refined_extrinsics, refined_intrinsics

    def _refine_incremental(
        self,
        extrinsics: np.ndarray,
        tracks: np.ndarray,
        vis_scores: np.ndarray,
        pts3d_tracks: np.ndarray,
        intrinsics: np.ndarray,
        increment_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Grow the frame set by increment_size per step, warm-starting each BA from the last.

        - all steps share one track extraction; each solves tracks[:k]
        - (N, 4, 4) world-to-cam in; returns refined (N, 3, 4) extrinsics and (N, 3, 3) K
        """
        N = len(extrinsics)

        # Initialize warm state from the input poses
        refined_extrinsics = extrinsics[:, :3, :].copy()
        refined_intrinsics = intrinsics.copy()

        # Step sequence ends at exactly N; a 1-frame window cannot be adjusted, so it is skipped
        steps = sorted({min(k, N) for k in range(increment_size, N + increment_size, increment_size)} - {1})

        for k in steps:
            logger.info("Incremental BA: %d/%d frames registered", k, N)
            _, refined_ext_k, refined_intr_k = self._optimize(
                pts3d_tracks,
                refined_extrinsics[:k].copy(),  # warm start from previous step
                refined_intrinsics[:k].copy(),
                tracks[:k],
                vis_scores[:k],
            )
            refined_extrinsics[:k] = refined_ext_k
            refined_intrinsics[:k] = refined_intr_k

        return refined_extrinsics, refined_intrinsics

    def refine(
        self,
        images: np.ndarray,
        confidence: np.ndarray,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        image_paths: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine camera poses and focal against VGGSfM tracks.

        - K must be at model resolution
        - points are not touched; the caller re-derives them under the new poses

        Args:
            images: (N, 3, H, W) model-grid frames, the track source.
            confidence: (N, H, W) per-pixel confidence for query-point sampling.
            world_points: (N, H, W, 3) the tracks are lifted from.
            extrinsics: (N, 4, 4) world-to-cam start poses.
            intrinsics: (N, 3, 3) model-resolution K.
            image_paths: frame paths for the track-cache key; None skips the cache.

        Returns:
            (N, 4, 4) refined extrinsics and (N, 3, 3) refined intrinsics.

        Raises:
            ValueError: fewer than 2 frames or 2 points stay active after filtering, in the global
                solve or any incremental window.
            RuntimeError: the resolved device is not CUDA (bae LM is CUDA-only).
        """
        self.loss_history = []
        N = len(images)
        logger.info(
            "BA refine start: %d frames (increment_size=%d, lm_steps=%d, max_reproj_error=%s)",
            N,
            self.config.increment_size,
            self.config.lm_steps,
            self.config.max_reproj_error,
        )

        # Load cached tracks or extract via VGGSfM (one extraction shared across all k-steps)
        tracks, vis_scores, pts3d_tracks = self.extract_tracks(images, confidence, world_points, image_paths)
        logger.info("Tracks ready: %d points across %d frames", tracks.shape[1], tracks.shape[0])

        # One global solve, or an incremental one when increment_size < N
        increment_size = self.config.increment_size
        if increment_size == 0 or increment_size >= N:
            logger.info("Global BA over %d frames", N)
            refined_extrinsics, refined_intrinsics = self._refine_global(
                extrinsics,
                tracks,
                vis_scores,
                pts3d_tracks,
                intrinsics,
            )
        else:
            refined_extrinsics, refined_intrinsics = self._refine_incremental(
                extrinsics,
                tracks,
                vis_scores,
                pts3d_tracks,
                intrinsics,
                increment_size,
            )

        # Report the final loss when a curve was captured
        if self.loss_history and self.loss_history[-1]:
            logger.info("BA refine done: final loss %.6e", self.loss_history[-1][-1])
        else:
            logger.info("BA refine done")
        return extrinsics_to_homogeneous(refined_extrinsics), refined_intrinsics

    def _optimize(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        tracks: np.ndarray,
        vis_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refine 3D points and camera poses with Levenberg-Marquardt BA.

        - appends this solve's per-step losses to self.loss_history
        - frames the inlier gate drops are carried by the active-set Sim(3), not refined
        - (N, 3, 4) world-to-cam in; returns pts3d (P, 3) float64, extrinsics (N, 3, 4) and K (N, 3, 3) float32
        """
        cfg = self.config
        max_reproj = cfg.max_reproj_error
        n_steps = cfg.lm_steps

        # Work on copies
        refined_extrinsics = extrinsics.astype(np.float32).copy()
        refined_intrinsics = intrinsics.astype(np.float32).copy()
        refined_pts3d = pts3d.astype(np.float64).copy()

        # Visibility gate + reprojection filter + frame/landmark drops (upstream order)
        vis = _filter_observations(
            vis_scores,
            tracks,
            pts3d,
            extrinsics,
            intrinsics,
            vis_thresh=cfg.vis_thresh,
            max_reproj=max_reproj,
            min_inliers_per_frame=cfg.min_inliers_per_frame,
        )

        # Build flat index arrays for active keyframes and landmarks
        active_frames = np.where(vis.any(1))[0]  # (K,)
        active_pts = np.where(vis.any(0))[0]  # (L,)

        if len(active_frames) < 2 or len(active_pts) < 2:
            raise ValueError(
                f"BA: too few active frames/points after filtering ({len(active_frames)} frames, "
                f"{len(active_pts)} points); need >= 2 of each"
            )

        # bae LM is CUDA-only (CuSparse spgemm); checked after the observation guard so it stays CPU-testable
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "cuda" not in device:
            raise RuntimeError(
                f"Bundle adjustment requires a CUDA device (got {device!r}). "
                "The bae LM optimizer uses a CUDA-only sparse matmul (CuSparse); "
                "CPU bundle adjustment is not supported."
            )

        # Flatten the active observations into (frame, point) index pairs
        frame_idx, pt_idx = np.where(vis[np.ix_(active_frames, active_pts)])
        logger.info(
            "LM optimize: %d/%d frames, %d/%d points, %d observations, up to %d steps",
            len(active_frames),
            vis.shape[0],
            len(active_pts),
            vis.shape[1],
            len(frame_idx),
            n_steps,
        )
        global_frame_idx = active_frames[frame_idx]
        global_pt_idx = active_pts[pt_idx]
        obs_2d = tracks[global_frame_idx, global_pt_idx].astype(np.float64)  # (M, 2)

        # Build SE3 camera tensor from (K, 3, 4) extrinsics; pad to (K, 4, 4) for mat2SE3
        ext_sub = extrinsics[active_frames].astype(np.float64)
        ext_4x4 = extrinsics_to_homogeneous(ext_sub)

        # Snap rotations to the nearest orthogonal matrix (SVD, det > 0)
        # - float32 rotations from some backends fail pypose's mat2SE3 check
        # - already-orthogonal rotations pass through unchanged
        ext_4x4[:, :3, :3] = project_to_so3(ext_4x4[:, :3, :3])
        cameras_se3 = pp.mat2SE3(torch.tensor(ext_4x4, dtype=torch.float64, device=device))

        # SIMPLE_PINHOLE: average fx/fy as single focal length per camera
        focal = ((intrinsics[active_frames, 0, 0] + intrinsics[active_frames, 1, 1]) / 2.0).astype(np.float64)
        focal_tensor = torch.tensor(focal, dtype=torch.float64, device=device).unsqueeze(-1)  # (K, 1)
        principal_points = torch.tensor(
            intrinsics[active_frames, :2, 2].astype(np.float64),
            dtype=torch.float64,
            device=device,
        )  # (K, 2)
        pts3d_tensor = torch.tensor(
            pts3d[active_pts].astype(np.float64),
            dtype=torch.float64,
            device=device,
        )  # (L, 3)

        # Concatenate focal length into camera params tensor for per-camera case
        if cfg.shared_camera:
            cam_params = cameras_se3.data  # (K, 7)
            shared_focal = focal_tensor.mean(0, keepdim=True)  # (1, 1)
        else:
            cam_params = torch.cat([cameras_se3.data, focal_tensor], dim=-1)  # (K, 8)
            shared_focal = None

        # Assemble observation index tensors
        obs_2d_t = torch.tensor(obs_2d, dtype=torch.float64, device=device)
        cam_idx = torch.tensor(frame_idx, dtype=torch.long, device=device)
        pt_idx_t = torch.tensor(pt_idx, dtype=torch.long, device=device)

        # Observation dict shared by both optimization paths
        input_dict = {
            "points_2d": obs_2d_t,
            "camera_indices": cam_idx,
            "point_indices": pt_idx_t,
            "principal_points": principal_points,
        }

        # Optimize reprojection residuals with Levenberg-Marquardt
        with torch.enable_grad():
            model = _BAModel(cam_params, pts3d_tensor, shared_focal, cfg.shared_camera)
            strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
            optimizer = LM(
                model,
                strategy=strategy,
                solver=_get_default_solver(device=device),
                reject=10,
            )

            # Bind target=None so bae LM.step can call the model
            # - pypose>=0.7 RobustModel.forward requires target; bae passes none
            # - residuals then fall back to the model output, the intended objective
            optimizer.model.forward = functools.partial(type(optimizer.model).forward, optimizer.model, target=None)

            # Manual LM loop over all n_steps, not pypose's StopOnPlateau
            # - StopOnPlateau stops on any rejected trust-region step (scheduler.py:153-155)
            # - its absolute 1e-3 plateau test never fires at this loss scale
            loss_hist: list[float] = []
            for i in range(n_steps):
                step_loss = optimizer.step(input=input_dict)
                loss_hist.append(float(step_loss))
                logger.info("LM step %d/%d: loss=%.6e", i + 1, n_steps, float(step_loss))
            self.loss_history.append(loss_hist)

        # Recover (3, 4) extrinsics from optimized SE3 quaternion representation
        opt_cam = model.pose.data.detach().cpu().numpy()  # (K, 7) or (K, 8)
        opt_pts = model.pts.data.detach().cpu().numpy()  # (L, 3)

        opt_se3 = pp.SE3(torch.tensor(opt_cam[:, :7], dtype=torch.float64))
        opt_extrinsics_3x4 = opt_se3.matrix().numpy()[:, :3, :]
        refined_extrinsics[active_frames] = opt_extrinsics_3x4.astype(np.float32)
        refined_pts3d[active_pts] = opt_pts.astype(np.float64)

        # Carry frames the inlier gate dropped into the refined gauge
        # - see _carry_dropped_frames
        n_dropped = vis.shape[0] - len(active_frames)
        if n_dropped:
            inactive = np.setdiff1d(np.arange(vis.shape[0]), active_frames)
            refined_extrinsics, gauge_scale = _carry_dropped_frames(refined_extrinsics, extrinsics, active_frames)
            if gauge_scale is None:
                logger.warning(
                    "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s) and left "
                    "in the pre-BA gauge — fewer than 3 active frames, cannot estimate it",
                    n_dropped,
                    vis.shape[0],
                    cfg.min_inliers_per_frame,
                    inactive.tolist(),
                )
            else:
                logger.warning(
                    "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s); carried "
                    "by the active-set Sim(3) (scale %.6f) but not refined",
                    n_dropped,
                    vis.shape[0],
                    cfg.min_inliers_per_frame,
                    inactive.tolist(),
                    gauge_scale,
                )

        # Write optimized focal lengths back to K
        # - shared focal goes to every frame, dropped ones included, so K is never mixed
        # - per-frame focal goes to active frames only
        if cfg.shared_camera and model.shared_intr is not None:
            focal_val = float(model.shared_intr.data.detach().cpu().numpy().mean())
            refined_intrinsics[:, 0, 0] = focal_val
            refined_intrinsics[:, 1, 1] = focal_val
        elif not cfg.shared_camera:
            opt_focal = opt_cam[:, 7]
            refined_intrinsics[active_frames, 0, 0] = opt_focal
            refined_intrinsics[active_frames, 1, 1] = opt_focal

        return refined_pts3d, refined_extrinsics, refined_intrinsics


########################################################################
# BA model
########################################################################


@map_transform
def _reproject_per_camera(pts: torch.Tensor, cam_params: torch.Tensor, principal_point: torch.Tensor) -> torch.Tensor:
    """
    Pinhole projection with a per-camera focal.

    - @map_transform vectorizes it for the bae LM Jacobian
    - cam_params: SE3 (7) plus focal (1); every argument has one row per observation
    """
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * cam_params[..., 7:] + principal_point


@map_transform
def _reproject_shared(
    pts: torch.Tensor, cam_params: torch.Tensor, principal_point: torch.Tensor, focal: torch.Tensor
) -> torch.Tensor:
    """
    Pinhole projection with one focal shared by every camera.

    - @map_transform vectorizes it for the bae LM Jacobian
    - cam_params: SE3 (7); focal: the shared focal broadcast; one row per observation
    """
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * focal + principal_point


class _BAModel(nn.Module):
    """
    Reprojection residual module for pypose Levenberg-Marquardt optimization.

    - parameters: pose (SE3, plus focal per camera), pts, and shared_intr when the focal is shared
    """

    def __init__(
        self,
        cam_params: torch.Tensor,
        pts_3d: torch.Tensor,
        shared_focal: torch.Tensor | None,
        shared_camera: bool,
    ) -> None:
        """
        Wrap poses, landmarks and the optional shared focal as trackable parameters.

        - cam_params: (K, 7) SE3, or (K, 8) SE3 plus focal; pts_3d: (L, 3)
        - shared_focal: (1, 1) when shared_camera, else None
        """
        super().__init__()

        # Trackable parameters for the bae sparse Jacobian
        # - trim_SE3_grad: 7-D SE3 pose takes a 6-DoF se3 step, trailing focal steps as-is
        self.pose = nn.Parameter(TrackingTensor(cam_params))
        self.pts = nn.Parameter(TrackingTensor(pts_3d))
        self.pose.trim_SE3_grad = True
        self.shared_intr = nn.Parameter(TrackingTensor(shared_focal)) if shared_focal is not None else None
        self.shared_camera = shared_camera

    def forward(
        self,
        points_2d: torch.Tensor,
        camera_indices: torch.Tensor,
        point_indices: torch.Tensor,
        principal_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reprojection residuals, predicted minus observed, for all M observations.

        - points_2d (M, 2) observed pixels; camera_indices / point_indices (M,) rows into pose / pts
        - principal_points: (K, 2), one per camera; returns (M, 2)
        """
        # Gather the landmark, camera and principal point of each observation
        pts = self.pts[point_indices]
        cam = self.pose[camera_indices]
        ctr = principal_points[camera_indices]

        if self.shared_camera:
            # Broadcast shared focal to match observation count (M,)
            focal = self.shared_intr[torch.zeros_like(camera_indices)]
            pts_proj = _reproject_shared(pts, cam, ctr, focal)
        else:
            pts_proj = _reproject_per_camera(pts, cam, ctr)

        return pts_proj - points_2d


def _get_default_solver(device: str | None = None) -> Any:
    """
    Sparse linear solver for bae LM: CuDSS when CUDA is requested and available, else PCG.

    - device None auto-detects CUDA; "cpu" always gives PCG
    - a CUDA request without a working CuDSS logs a warning and falls back to PCG
    """
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver

            return CuDirectSparseSolver()
        except (ImportError, RuntimeError) as exc:
            logger.warning("BA: CuDSS solver unavailable (%r), falling back to PCG", exc)
    return PCG()


########################################################################
# Helpers
########################################################################


def _filter_observations(
    vis_scores: np.ndarray,
    tracks: np.ndarray,
    pts3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    vis_thresh: float,
    max_reproj: float | None,
    min_inliers_per_frame: int,
) -> np.ndarray:
    """
    Boolean (N, P) observation mask for the BA solve; True enters the solve.

    - order: visibility gate, reprojection filter, frame min-inlier drop, landmark drop (<2 views or out of range)
    - order follows upstream VGGT demo_colmap
    - max_reproj None skips the reprojection filter
    """
    # Visibility gate: keep observations the tracker is confident about
    vis = vis_scores > vis_thresh

    # Remove observations with high reprojection error under the current poses
    if max_reproj is not None:
        points = torch.as_tensor(pts3d, dtype=torch.float64)
        world_to_cam = torch.as_tensor(extrinsics, dtype=torch.float64)
        K = torch.as_tensor(intrinsics, dtype=torch.float64)
        projected = [project(points, world_to_cam[i], K[i]) for i in range(len(K))]
        proj2d = torch.stack([pixels for pixels, _ in projected])  # (N, P, 2)
        proj2d = proj2d.numpy()
        depth = torch.stack([cam[:, 2] for _, cam in projected])  # (N, P)
        depth = depth.numpy()

        # Behind-camera points get large sentinel projection so they fail the threshold
        proj2d[depth <= 0] = 1e6
        reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)
        vis[~(reproj_err <= max_reproj)] = False  # NaN-safe: a NaN point fails the gate

    # Drop frames with too few inliers, before the landmark check
    # - landmark counts then see surviving frames only, so 1-view points drop
    vis[vis.sum(1) < min_inliers_per_frame] = False

    # Drop points seen from fewer than 2 (surviving) frames and points outside valid world range
    seen_enough = vis.sum(0) >= 2
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    vis[:, ~(seen_enough & in_range)] = False

    return vis


def _carry_dropped_frames(
    refined_extrinsics: np.ndarray,
    extrinsics: np.ndarray,
    active_frames: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """
    Transform frames outside active_frames by the Sim(3) the active set underwent.

    - BA fixes no frame and no scale, so the refined set can drift as a whole
    - refined_extrinsics: (N, 3, 4), active rows refined; extrinsics: the pre-BA poses
    - returns (carried copy, Sim(3) scale), or (input, None) when nothing dropped or < 3 active frames
    """
    inactive = np.setdiff1d(np.arange(refined_extrinsics.shape[0]), active_frames)
    if len(inactive) == 0 or len(active_frames) < 3:
        return refined_extrinsics, None

    # Estimate the world-gauge Sim(3) from how the active cameras' centers moved
    src_c = invert_poses(extrinsics_to_homogeneous(extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    dst_c = invert_poses(extrinsics_to_homogeneous(refined_extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    s, R_g, t_g = umeyama_sim3(src_c, dst_c)

    # Apply the gauge to each dropped world-to-cam [R|t]
    # - world gauge X' = s R_g X + t_g maps [R|t] to [R R_g^T | s t - R R_g^T t_g]
    # - puts the dropped camera's center at s R_g C + t_g, the same map the points took
    out = refined_extrinsics.copy()
    R_in = refined_extrinsics[inactive, :, :3].astype(np.float64)
    t_in = refined_extrinsics[inactive, :, 3].astype(np.float64)
    R_new = R_in @ R_g.T.astype(np.float64)
    out[inactive, :, :3] = R_new.astype(np.float32)
    out[inactive, :, 3] = (s * t_in - np.einsum("nij,j->ni", R_new, t_g.astype(np.float64))).astype(np.float32)
    return out, float(s)


def _compute_tracks_cache_key(image_paths: list, world_points: np.ndarray, cfg: BundleAdjustmentConfig) -> str:
    """
    SHA-256 track-cache key over image paths, world points and extraction config.

    - any frame, world_points or track-setting change invalidates the key; image_paths order does not
    - world_points digest: two backbones over the same images must not share tracks; ~1-3 s at 300 frames
    """
    meta = {
        "image_paths": sorted(str(p) for p in image_paths),
        "world_points": hashlib.sha256(np.ascontiguousarray(world_points).tobytes()).hexdigest(),
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
        "fine_tracking": cfg.fine_tracking,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


def extract_tracks_vggsfm(
    images: torch.Tensor,
    conf: torch.Tensor | None,
    world_points: np.ndarray | None,
    cfg: BundleAdjustmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict cross-frame 2D tracks via VGGSfM (ALIKED+SP keypoints), with the config's knobs.

    - no cache; BundleAdjustment.extract_tracks adds one
    - conf None: no confidence-guided queries; world_points None: no 3D lifting; both: pad to square

    Args:
        images: (N, 3, H, W) frames, tensor or float32 numpy.
        conf: (N, H, W) or (N, 1, H, W) per-pixel confidence, or None.
        world_points: (N, H, W, 3) points the tracks are lifted from, or None.
        cfg: track-extraction knobs (max_query_pts, query_frame_num, fine_tracking, ...).

    Returns:
        tracks (N, P, 2) pixels, vis_scores (N, P) in [0, 1] and pts3d (P, 3) world, all float32.
    """
    logger.info(
        "Extracting VGGSfM tracks: %d frames, max_query_pts=%d, query_frame_num=%d, fine_tracking=%s (slow step)",
        len(images),
        cfg.max_query_pts,
        cfg.query_frame_num,
        cfg.fine_tracking,
    )
    target_device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # numpy images: sfm's align_depth stores float32 arrays, not tensors
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Move images to target_device unconditionally
    # - predict_tracks places the tracker on images.device and never relocates
    images = images.to(target_device)
    img_device = images.device

    # VGGSfM tracker uses grid_sample; not implemented for BFloat16 on CUDA
    images = images.float()

    # Confidence on the image device, normalized to (N, H, W)
    # - predict_tracks does not accept (N, 1, H, W)
    conf_tensor: torch.Tensor | None = None
    if conf is not None:
        conf_tensor = conf.to(img_device)
        if conf_tensor.ndim == 4 and conf_tensor.shape[1] == 1:
            conf_tensor = conf_tensor.squeeze(1)

    # World points as a tensor on the image device
    pts3d_tensor: torch.Tensor | None = None
    if world_points is not None:
        pts3d_tensor = torch.tensor(world_points, dtype=images.dtype, device=img_device)

    # VGGSfM asserts H == W when both conf and world_points are provided; pad the shorter dim
    if conf_tensor is not None and pts3d_tensor is not None:
        H_img, W_img = images.shape[-2], images.shape[-1]
        if H_img != W_img:
            size = max(H_img, W_img)
            pad_h, pad_w = size - H_img, size - W_img
            images = F.pad(images, (0, pad_w, 0, pad_h))
            conf_tensor = F.pad(conf_tensor, (0, pad_w, 0, pad_h))
            pts3d_tensor = F.pad(pts3d_tensor, (0, 0, 0, pad_w, 0, pad_h))

    # Predict tracks with conf and points on CPU, without grad
    # - CPU: VGGSfM mixes CPU numpy indexing with GPU tensors internally
    # - no_grad: pred_track would otherwise retain a gradient that breaks .numpy()
    with torch.no_grad():
        pred_tracks, pred_vis_scores, _pred_confs, pred_pts3d, _pred_colors = predict_tracks(
            images,
            conf=conf_tensor.cpu() if conf_tensor is not None else None,
            points_3d=pts3d_tensor.cpu() if pts3d_tensor is not None else None,
            max_query_pts=cfg.max_query_pts,
            query_frame_num=cfg.query_frame_num,
            fine_tracking=cfg.fine_tracking,
        )

    # Return float32 numpy arrays
    tracks = np.asarray(pred_tracks).astype(np.float32)
    vis_scores = np.asarray(pred_vis_scores).astype(np.float32)
    pts3d = np.asarray(pred_pts3d).astype(np.float32)
    return tracks, vis_scores, pts3d
