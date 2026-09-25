"""
Levenberg-Marquardt bundle adjustment over VGGSfM tracks.

- BundleAdjustmentConfig: solver, filter and track-extraction settings
- BundleAdjustment: refines poses and focal of a `pointcloud.zarr` result
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.pysolvers import PCG
from vggt.dependency.projection import project_3D_points_np
from vggt.dependency.track_predict import predict_tracks
from zarr.codecs import BloscCodec

from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    invert_poses,
    umeyama_sim3,
)

if TYPE_CHECKING:
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]

logger = logging.getLogger(__name__)


def _compute_tracks_cache_key(result: "FeedforwardResult", cfg: "BundleAdjustmentConfig") -> str:
    """SHA-256 track-cache key over image paths and extraction config."""
    meta = {
        "image_paths": sorted(str(p) for p in (result.image_paths or [])),
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
        "fine_tracking": cfg.fine_tracking,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


########################################################
########## Configuration ##############################
########################################################


@dataclass
class BundleAdjustmentConfig:
    """
    LM bundle adjustment settings.

    - increment_size: 0 runs one global BA; 1..N-1 grows the frame set by that many per solve
    - shared_camera: True fits one focal for the scene; False fits one focal per frame
    - max_reproj_error=None: skips the pre-solve reprojection filter
    """

    max_reproj_error: float = 4.0  # pixel reprojection gate before the solve; lower drops more observations
    lm_steps: int = 40  # LM iterations per solve; all are run, no early stop
    shared_camera: bool = True  # one physical camera per scene; per-frame K spread is model noise
    vis_thresh: float = 0.2  # min VGGSfM visibility score for an observation to enter BA
    fine_tracking: bool = True  # VGGSfM fine refinement stage (upstream always on; coarse-only ~1-2px error)
    min_inliers_per_frame: int = 64  # frames with fewer inlier observations are left out of the solve
    max_query_pts: int = 4096  # track extraction: max query points (upstream demo default)
    query_frame_num: int = 8  # track extraction: number of query frames (upstream demo default)
    device: str | None = None  # CUDA device (e.g. "cuda", "cuda:1"); None = auto. CPU unsupported (bae LM is CUDA-only)
    increment_size: int = 0  # frames added per step; 0 = disabled (global BA); 1..N-1 = incremental
    tracks_cache_dir: Path | None = None  # zarr cache dir for tracks; None = always extract


########################################################
########## BundleAdjustment ###########################
########################################################


class BundleAdjustment:
    """
    Refines camera poses and focal via VGGSfM tracks and LM bundle adjustment.

    - refines a `pointcloud.zarr` result whose K is at model resolution
    - the refine stage refuses sfm results
    - does not reproject points; call creator.reproject(result) afterwards if needed
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        self.config = config or BundleAdjustmentConfig()
        # Populated per _optimize() call with that call's per-step LM losses
        self.loss_history: list[list[float]] = []

    def _load_or_extract_tracks(self, result: "FeedforwardResult") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tracks, vis_scores, pts3d_tracks) from the zarr cache or a fresh VGGSfM extraction."""
        cfg = self.config

        def _extract() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            logger.info(
                "Extracting VGGSfM tracks: %d frames, max_query_pts=%d, query_frame_num=%d, fine_tracking=%s (slow step)",
                len(result.images),
                cfg.max_query_pts,
                cfg.query_frame_num,
                cfg.fine_tracking,
            )
            return _extract_tracks_vggsfm(
                result.images,
                result.confidence,
                result.world_points,
                max_query_pts=cfg.max_query_pts,
                query_frame_num=cfg.query_frame_num,
                fine_tracking=cfg.fine_tracking,
                device=cfg.device,
            )

        if cfg.tracks_cache_dir is None or not result.image_paths:
            return _extract()

        cache_path = Path(cfg.tracks_cache_dir) / "tracks.zarr"
        expected_key = _compute_tracks_cache_key(result, cfg)

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
            # unreadable store: missing array -> KeyError
            # bad/missing metadata -> JSONDecodeError / GroupNotFoundError (ValueError, OSError)
            except (KeyError, ValueError, OSError) as exc:
                logger.warning("Track cache unreadable (%s), re-extracting: %s", exc, cache_path)
            shutil.rmtree(cache_path)

        # Extract and persist to zarr cache
        tracks, vis_scores, pts3d_tracks = _extract()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(cache_path), mode="w")
        for key, arr in [("tracks", tracks), ("vis_scores", vis_scores), ("pts3d_tracks", pts3d_tracks)]:
            store.create_array(key, data=arr, chunks=arr.shape, compressors=lz4)
        store.attrs["cache_key"] = expected_key
        logger.debug("Track cache saved: %s", cache_path)

        return tracks, vis_scores, pts3d_tracks

    def _refine_allonce(
        self,
        result: "FeedforwardResult",
        tracks: np.ndarray,
        vis_scores: np.ndarray,
        pts3d_tracks: np.ndarray,
        intrinsics_model: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one BA over all N frames at once."""
        extrinsics_3x4 = result.extrinsics[:, :3, :]
        _, refined_extrinsics, refined_intrinsics_model = self._optimize(
            pts3d_tracks,
            extrinsics_3x4,
            intrinsics_model,
            tracks,
            vis_scores,
        )
        return refined_extrinsics, refined_intrinsics_model

    def _refine_incremental(
        self,
        result: "FeedforwardResult",
        tracks: np.ndarray,
        vis_scores: np.ndarray,
        pts3d_tracks: np.ndarray,
        intrinsics_model: np.ndarray,
        increment_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grow the frame set by increment_size per step, warm-starting each BA from the last."""
        N = len(result.images)
        # Initialize warm state from the input poses
        refined_extrinsics = result.extrinsics[:, :3, :].copy()
        refined_intrinsics = intrinsics_model.copy()

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

    def refine(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """
        Refine camera poses and focal of a result.

        - points, colors and pixel_indices are unchanged; call creator.reproject(result) after

        Args:
            result: reconstruction with images, confidence, world_points, extrinsics, K,
                original_coords and image_paths (track-cache key).

        Returns:
            A copy of `result` with refined (N, 4, 4) extrinsics and (N, 3, 3) intrinsics.

        Raises:
            ValueError: K is at original resolution, or fewer than 2 frames or 2 points stay
                active after filtering in the global solve or any incremental window.
            RuntimeError: the resolved device is not CUDA (bae LM is CUDA-only).
        """
        self.loss_history = []
        logger.info(
            "BA refine start: %d frames (increment_size=%d, lm_steps=%d, max_reproj_error=%s)",
            len(result.images),
            self.config.increment_size,
            self.config.lm_steps,
            self.config.max_reproj_error,
        )

        # VGGSfM tracks live on the model grid, so K must too; checked before the slow extraction
        _check_model_resolution(result.intrinsics, result.images, result.original_coords)
        intrinsics_model = result.intrinsics

        # Load cached tracks or extract via VGGSfM (one extraction shared across all k-steps)
        tracks, vis_scores, pts3d_tracks = self._load_or_extract_tracks(result)
        logger.info("Tracks ready: %d points across %d frames", tracks.shape[1], tracks.shape[0])

        N = len(result.images)
        increment_size = self.config.increment_size
        if increment_size == 0 or increment_size >= N:
            logger.info("All-at-once BA over %d frames", N)
            refined_extrinsics, refined_intrinsics_model = self._refine_allonce(
                result,
                tracks,
                vis_scores,
                pts3d_tracks,
                intrinsics_model,
            )
        else:
            refined_extrinsics, refined_intrinsics_model = self._refine_incremental(
                result,
                tracks,
                vis_scores,
                pts3d_tracks,
                intrinsics_model,
                increment_size,
            )

        refined_extrinsics_4x4 = extrinsics_to_homogeneous(refined_extrinsics)
        # Report the final loss when a curve was captured
        if self.loss_history and self.loss_history[-1]:
            logger.info("BA refine done: final loss %.6e", self.loss_history[-1][-1])
        else:
            logger.info("BA refine done")
        return replace(result, extrinsics=refined_extrinsics_4x4, intrinsics=refined_intrinsics_model)

    def _optimize(
        self,
        pts3d: np.ndarray,  # (P, 3)    initial 3D keypoint positions
        extrinsics: np.ndarray,  # (N, 3, 4) world-to-camera poses
        intrinsics: np.ndarray,  # (N, 3, 3) camera intrinsics
        tracks: np.ndarray,  # (N, P, 2) 2D pixel observations from VGGSfM
        vis_scores: np.ndarray,  # (N, P)    visibility scores in [0, 1]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refine 3D points and camera poses with Levenberg-Marquardt BA.

        Returns:
            refined_pts3d:        (P, 3)    float64
            refined_extrinsics:   (N, 3, 4) float32
            refined_intrinsics:   (N, 3, 3) float32

        Raises:
            ValueError: fewer than 2 frames or 2 points stay active after filtering.
            RuntimeError: the resolved device is not CUDA (bae LM is CUDA-only).
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
        U, _, Vt = np.linalg.svd(ext_4x4[:, :3, :3])
        U[np.linalg.det(U @ Vt) < 0, :, -1] *= -1
        ext_4x4[:, :3, :3] = U @ Vt
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
        # - BA fixes no frame and no scale, so the active set can drift as a whole
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


########################################################
########## Helpers ####################################
########################################################


def _check_model_resolution(intrinsics: np.ndarray, images: Any, original_coords: np.ndarray) -> None:
    """
    Raise unless K is at model (depth) resolution, the `pointcloud.zarr` contract.

    - model-res K has 2·cx ≈ W_model
    - original-res K has cx ≈ the crop center, so 2·cx ≈ tl_x + cr_x in original pixels
    - raises when 2·cx is closer to tl_x + cr_x than to W_model
    - checks frame 0 only
    - only a `pointcloud.zarr` written before the model-res K contract fails

    Args:
        intrinsics: (N, 3, 3) stored K.
        images: (N, 3, H, W) model-resolution images.
        original_coords: (N, 6) crop boxes `[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`.

    Raises:
        ValueError: K is at original resolution.
    """
    W_model = float(images.shape[-1])
    tl_x, cr_x = float(original_coords[0, 0]), float(original_coords[0, 2])
    cx2 = 2.0 * float(intrinsics[0, 0, 2])
    if abs(cx2 - W_model) > abs(cx2 - (tl_x + cr_x)):
        raise ValueError(
            "BA: intrinsics are at original resolution, not model resolution — this "
            "pointcloud.zarr predates the model-res K contract; re-run the pointcloud stage"
        )


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
    Boolean (N, P) observation mask for the BA solve.

    - order: visibility gate, reprojection filter, frame min-inlier drop, landmark drop (<2 views or out of range)
    - order follows upstream VGGT demo_colmap
    """
    # Visibility gate: keep observations the tracker is confident about
    vis = vis_scores > vis_thresh

    # Remove observations with high reprojection error under the current poses
    if max_reproj is not None:
        proj2d, proj_cam = project_3D_points_np(pts3d, extrinsics, intrinsics)
        # Behind-camera points get large sentinel projection so they fail the threshold
        behind = proj_cam[:, 2, :] <= 0
        proj2d = proj2d.copy()
        proj2d[behind] = 1e6
        reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)
        vis[reproj_err > max_reproj] = False

    # Drop frames with too few inliers, before the landmark check
    # - landmark counts then see surviving frames only, so 1-view points drop
    vis[vis.sum(1) < min_inliers_per_frame] = False

    # Drop points seen from fewer than 2 (surviving) frames and points outside valid world range
    seen_enough = vis.sum(0) >= 2
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    vis[:, ~(seen_enough & in_range)] = False

    return vis


def _get_default_solver(device: str | None = None) -> Any:
    """
    Sparse linear solver for bae LM: CuDSS when CUDA is requested and available, else PCG.

    - a CUDA request without a working CuDSS logs a warning and falls back to PCG

    Args:
        device: resolved device string; None auto-detects CUDA, "cpu" always gives PCG.

    Returns:
        A bae solver instance.
    """
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver

            return CuDirectSparseSolver()
        except (ImportError, RuntimeError) as exc:
            logger.warning("BA: CuDSS solver unavailable (%r), falling back to PCG", exc)
    return PCG()


def _carry_dropped_frames(
    refined_extrinsics: np.ndarray,
    extrinsics: np.ndarray,
    active_frames: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """
    Transform frames outside active_frames by the Sim(3) the active set underwent.

    - BA fixes no frame and no scale, so the refined set can drift as a whole
    - without this, dropped frames stay in the pre-BA gauge

    Args:
        refined_extrinsics: (N, 3, 4) poses with active rows already refined.
        extrinsics: (N, 3, 4) original pre-BA poses.
        active_frames: (K,) indices refined by the solve.

    Returns:
        (extrinsics, scale): a copy with dropped frames carried, and the Sim(3) scale.
        scale is None when nothing was dropped or the gauge could not be estimated.
    """
    inactive = np.setdiff1d(np.arange(refined_extrinsics.shape[0]), active_frames)
    if len(inactive) == 0 or len(active_frames) < 3:
        return refined_extrinsics, None

    # Estimate the world-gauge Sim(3) from how the active cameras' centers moved
    src_c = invert_poses(extrinsics_to_homogeneous(extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    dst_c = invert_poses(extrinsics_to_homogeneous(refined_extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    s, R_g, t_g = umeyama_sim3(src_c, dst_c)

    # World gauge X' = s R_g X + t_g maps a world-to-cam [R|t] to [R R_g^T | s t - R R_g^T t_g],
    # which puts the dropped camera's center at s R_g C + t_g — the same map the points took.
    out = refined_extrinsics.copy()
    R_in = refined_extrinsics[inactive, :, :3].astype(np.float64)
    t_in = refined_extrinsics[inactive, :, 3].astype(np.float64)
    R_new = R_in @ R_g.T.astype(np.float64)
    out[inactive, :, :3] = R_new.astype(np.float32)
    out[inactive, :, 3] = (s * t_in - np.einsum("nij,j->ni", R_new, t_g.astype(np.float64))).astype(np.float32)
    return out, float(s)


def _extract_tracks_vggsfm(
    images: torch.Tensor,
    conf: torch.Tensor | None,
    world_points: np.ndarray | None,
    *,
    max_query_pts: int,
    query_frame_num: int,
    fine_tracking: bool,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict cross-frame 2D tracks via VGGSfM (ALIKED+SP keypoints).

    Returns:
        tracks:     (N, P, 2) float32 — 2D pixel coords per frame per point.
        vis_scores: (N, P)    float32 — visibility score in [0, 1].
        pts3d:      (P, 3)    float32 — world-space 3D positions at keypoints.
    """
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # numpy images: sfm's result_from_reconstruction stores float32 arrays, not tensors
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Move images to target_device unconditionally
    # - predict_tracks places the tracker on images.device and never relocates
    images = images.to(target_device)
    img_device = images.device
    # VGGSfM tracker uses grid_sample; not implemented for BFloat16 on CUDA
    images = images.float()

    conf_tensor: torch.Tensor | None = None
    if conf is not None:
        conf_tensor = conf.to(img_device)
        # Normalize to (N, H, W) — predict_tracks does not accept (N, 1, H, W)
        if conf_tensor.ndim == 4 and conf_tensor.shape[1] == 1:
            conf_tensor = conf_tensor.squeeze(1)

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

    # Move conf/pts to CPU: VGGSfM mixes CPU numpy indexing with GPU tensors internally.
    # no_grad prevents pred_track from retaining a gradient that breaks .numpy()
    with torch.no_grad():
        pred_tracks, pred_vis_scores, _pred_confs, pred_pts3d, _pred_colors = predict_tracks(
            images,
            conf=conf_tensor.cpu() if conf_tensor is not None else None,
            points_3d=pts3d_tensor.cpu() if pts3d_tensor is not None else None,
            max_query_pts=max_query_pts,
            query_frame_num=query_frame_num,
            fine_tracking=fine_tracking,
        )

    tracks = np.asarray(pred_tracks).astype(np.float32)
    vis_scores = np.asarray(pred_vis_scores).astype(np.float32)
    pts3d = np.asarray(pred_pts3d).astype(np.float32)
    return tracks, vis_scores, pts3d


@map_transform
def _reproject_per_camera(pts: torch.Tensor, cam_params: torch.Tensor, principal_point: torch.Tensor) -> torch.Tensor:
    """Per-element pinhole projection with per-camera focal; @map_transform vectorizes for bae LM Jacobian."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * cam_params[..., 7:] + principal_point


@map_transform
def _reproject_shared(
    pts: torch.Tensor, cam_params: torch.Tensor, principal_point: torch.Tensor, focal: torch.Tensor
) -> torch.Tensor:
    """Pinhole projection with a shared focal; @map_transform vectorizes it for the bae LM Jacobian."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * focal + principal_point


class _BAModel(nn.Module):
    """Reprojection residual module for pypose Levenberg-Marquardt optimization."""

    def __init__(
        self,
        cam_params: torch.Tensor,  # (K, 7) SE3 or (K, 8) SE3+focal
        pts_3d: torch.Tensor,  # (L, 3) 3D landmark positions
        shared_focal: torch.Tensor | None,  # (1, 1) if shared_camera, else None
        shared_camera: bool,
    ) -> None:
        super().__init__()
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
        """Return reprojection residuals (predicted - observed) for all M observations."""
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
