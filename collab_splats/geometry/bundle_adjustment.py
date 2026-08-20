"""Bundle adjustment for collab-splats.

Public API:
- BundleAdjustmentConfig: configuration dataclass
- BundleAdjustment:       refines camera poses via VGGSfM tracks + LM bundle adjustment
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

from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses, umeyama_sim3

if TYPE_CHECKING:
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]

logger = logging.getLogger(__name__)

# Sentinel for "caller did not pass this kwarg" — distinguishes explicit None (skip
# reprojection filter) from omitted (fall back to config default).
_UNSET = object()


def _compute_tracks_cache_key(result: "FeedforwardResult", cfg: "BundleAdjustmentConfig") -> str:
    """SHA-256 key for track cache invalidation — covers image paths + extraction config."""
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
    """Configuration for LM bundle adjustment."""

    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = True  # one physical camera per scene; per-frame K spread is model noise
    vis_thresh: float = 0.2  # min VGGSfM visibility score for an observation to enter BA
    fine_tracking: bool = True  # VGGSfM fine refinement stage (upstream always on; coarse-only ~1-2px error)
    min_inliers_per_frame: int = 64
    max_query_pts: int = 4096  # track extraction: max query points (upstream demo default)
    query_frame_num: int = 8  # track extraction: number of query frames (upstream demo default)
    device: str | None = None  # CUDA device (e.g. "cuda", "cuda:1"); None = auto. CPU unsupported (bae LM is CUDA-only)
    increment_size: int = 0  # frames added per step; 0 = disabled (global BA); 1..N-1 = incremental
    # Sweep results (chess seq-01): increment_size ≈ N//10 is Pareto-optimal (N=50→3, N=200→20).
    # increment_size=1 diverges. Default 0 = global BA; set explicitly to opt into incremental.
    tracks_cache_dir: Path | None = None  # zarr cache dir for tracks; None = always extract


########################################################
########## BundleAdjustment ###########################
########################################################


class BundleAdjustment:
    """Refines camera poses via VGGSfM track extraction + LM bundle adjustment.

    Method-agnostic: works with any FeedforwardResult regardless of source creator.
    Does NOT reproject pts3d — call creator.reproject(result) after if needed.
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        self.config = config or BundleAdjustmentConfig()
        # Populated per _optimize() call with that call's per-step LM losses
        self._last_loss_history: list[list[float]] = []

    def _load_or_extract_tracks(self, result: "FeedforwardResult") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (tracks, vis_scores, pts3d_tracks) from zarr cache or VGGSfM extraction."""
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
            except Exception as exc:
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
        """Run BA on all N frames simultaneously (current all-at-once behaviour)."""
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
        """Grow the registered frame set by increment_size per step; warm-start each BA from prior step."""
        N = len(result.images)
        # Initialise warm state from feedforward poses
        refined_extrinsics = result.extrinsics[:, :3, :].copy()
        refined_intrinsics = intrinsics_model.copy()

        # Build step sequence: increment_size, 2*increment_size, ..., N (always ends at exactly N)
        steps = sorted({min(k, N) for k in range(increment_size, N + increment_size, increment_size)})

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
        """Refine camera poses; return updated FeedforwardResult with new extrinsics/intrinsics.

        points, colors, and pixel_indices are unchanged — call creator.reproject(result)
        after to re-extract points from refined poses.
        """
        self._last_loss_history = []
        logger.info(
            "BA refine start: %d frames (increment_size=%d, lm_steps=%d, max_reproj_error=%s)",
            len(result.images),
            self.config.increment_size,
            self.config.lm_steps,
            self.config.max_reproj_error,
        )

        # Load cached tracks or extract via VGGSfM (one extraction shared across all k-steps)
        tracks, vis_scores, pts3d_tracks = self._load_or_extract_tracks(result)
        logger.info("Tracks ready: %d points across %d frames", tracks.shape[1], tracks.shape[0])

        # Bring intrinsics to model space if needed — VGGSfM tracks are model-res; creators
        # already store model-res K (the guard makes this a no-op), legacy original-res K is scaled
        intrinsics_model, sx, sy, tl_x, tl_y = _scale_intrinsics_to_model(
            result.intrinsics,
            result.images,
            result.original_coords,
        )

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

        # Rescale refined intrinsics back to original-image space
        refined_intrinsics = refined_intrinsics_model.copy()
        refined_intrinsics[:, 0, 0] /= sx
        refined_intrinsics[:, 1, 1] /= sy
        refined_intrinsics[:, 0, 2] = refined_intrinsics_model[:, 0, 2] / sx + tl_x
        refined_intrinsics[:, 1, 2] = refined_intrinsics_model[:, 1, 2] / sy + tl_y

        refined_extrinsics_4x4 = extrinsics_to_homogeneous(refined_extrinsics)
        # Report final loss when a curve was captured — the single number a console watcher needs
        if self._last_loss_history and self._last_loss_history[-1]:
            logger.info("BA refine done: final loss %.6e", self._last_loss_history[-1][-1])
        else:
            logger.info("BA refine done")
        return replace(result, extrinsics=refined_extrinsics_4x4, intrinsics=refined_intrinsics)

    def _optimize(
        self,
        pts3d: np.ndarray,  # (P, 3)    initial 3D keypoint positions
        extrinsics: np.ndarray,  # (N, 3, 4) world-to-camera poses
        intrinsics: np.ndarray,  # (N, 3, 3) camera intrinsics
        tracks: np.ndarray,  # (N, P, 2) 2D pixel observations from VGGSfM
        vis_scores: np.ndarray,  # (N, P)    visibility scores in [0, 1]
        *,
        max_reproj_error: float | None = _UNSET,  # override self.config if provided; None = skip filter
        lm_steps: int | None = None,  # override self.config if provided
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine 3D points and camera poses with Levenberg-Marquardt BA.

        Returns:
            refined_pts3d:        (P, 3)    float64
            refined_extrinsics:   (N, 3, 4) float32
            refined_intrinsics:   (N, 3, 3) float32
        """
        cfg = self.config
        # _UNSET means "caller didn't pass" — fall back to config; explicit None means "skip filter"
        max_reproj = cfg.max_reproj_error if max_reproj_error is _UNSET else max_reproj_error
        n_steps = lm_steps if lm_steps is not None else cfg.lm_steps

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
            logger.warning(
                "BA skipped: too few active frames/points after filtering (%d frames, %d points)",
                len(active_frames),
                len(active_pts),
            )
            return refined_pts3d, refined_extrinsics, refined_intrinsics

        # bae LM optimizer is CUDA-only (CuSparse spgemm); reject CPU with a clear message.
        # Done here, after early-exit, so the no-device early-exit path stays CPU-runnable.
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
        # Project rotations to the nearest orthogonal matrix (SVD, det>0 enforced) — some
        # backends (LoGeR se3 refinement) emit float32 rotations that fail pypose's
        # mat2SE3 orthogonality check; already-orthogonal rotations pass through unchanged.
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

        # Observation dict shared by both optimisation paths
        input_dict = {
            "points_2d": obs_2d_t,
            "camera_indices": cam_idx,
            "point_indices": pt_idx_t,
            "principal_points": principal_points,
        }

        # Optimise reprojection residuals with Levenberg-Marquardt
        with torch.enable_grad():
            model = _BAModel(cam_params, pts3d_tensor, shared_focal, cfg.shared_camera)
            strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
            optimizer = LM(
                model,
                strategy=strategy,
                solver=_get_default_solver(device=device),
                reject=10,
            )
            # bae LM.step calls self.model(input) with no target, but pypose>=0.7
            # RobustModel.forward requires target positionally — bind target=None on the
            # instance (residuals then fall back to squared model output, the intended
            # objective). Without this every LM step raises TypeError.
            optimizer.model.forward = functools.partial(
                type(optimizer.model).forward, optimizer.model, target=None
            )

            # Manual LM loop, running all n_steps. pypose's StopOnPlateau is unusable here:
            # it aborts the whole optimization as soon as bae's LM needs a single
            # trust-region damping retry (scheduler.py:153-155 stops on reject_count > 0,
            # which optimizer.py increments on any rejected trial step), and its plateau
            # test is an absolute difference against 1e-3 that never fires at our loss
            # scale (~1e5-1e6). It terminated after 1-2 of 40 steps in practice.
            loss_hist: list[float] = []
            for i in range(n_steps):
                step_loss = optimizer.step(input=input_dict)
                loss_hist.append(float(step_loss))
                logger.info("LM step %d/%d: loss=%.6e", i + 1, n_steps, float(step_loss))
            self._last_loss_history.append(loss_hist)

        # Recover (3, 4) extrinsics from optimised SE3 quaternion representation
        opt_cam = model.pose.data.detach().cpu().numpy()  # (K, 7) or (K, 8)
        opt_pts = model.pts.data.detach().cpu().numpy()  # (L, 3)

        opt_se3 = pp.SE3(torch.tensor(opt_cam[:, :7], dtype=torch.float64))
        opt_extrinsics_3x4 = opt_se3.matrix().numpy()[:, :3, :]
        refined_extrinsics[active_frames] = opt_extrinsics_3x4.astype(np.float32)
        refined_pts3d[active_pts] = opt_pts.astype(np.float64)

        # Frames the inlier gate dropped keep their pre-BA pose, which sits in the pre-BA
        # gauge — BA fixes no frame and no scale, so the refined set can drift as a whole.
        # Carry them by the Sim(3) the active set underwent so they stay consistent.
        n_dropped = vis.shape[0] - len(active_frames)
        if n_dropped:
            inactive = np.setdiff1d(np.arange(vis.shape[0]), active_frames)
            refined_extrinsics, gauge_scale = _carry_dropped_frames(
                refined_extrinsics, extrinsics, active_frames
            )
            if gauge_scale is None:
                logger.warning(
                    "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s) and left "
                    "in the pre-BA gauge — fewer than 3 active frames, cannot estimate it",
                    n_dropped, vis.shape[0], cfg.min_inliers_per_frame, inactive.tolist(),
                )
            else:
                logger.warning(
                    "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s); carried "
                    "by the active-set Sim(3) (scale %.6f) but not refined",
                    n_dropped, vis.shape[0], cfg.min_inliers_per_frame, inactive.tolist(),
                    gauge_scale,
                )

        # Write optimised focal lengths back to intrinsics matrix. The shared focal goes to
        # every frame, including dropped ones: a stale per-frame focal surviving in an
        # otherwise single-camera reconstruction is the mixed-K state shared_camera removes.
        # The per-frame branch keeps active_frames — there is no scene-wide value to assign.
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


def _scale_intrinsics_to_model(
    intrinsics: np.ndarray,
    images: Any,
    original_coords: np.ndarray | None,
) -> tuple[np.ndarray, float, float, float, float]:
    """Scale intrinsics from original-image space to model-resolution space.

    VGGSfM tracks are predicted on model-resolution images; intrinsics stored in
    FeedforwardResult are at original-image resolution. Scaling them to model space
    ensures reprojection error is computed in the same coordinate system as the tracks.

    Returns:
        (intrinsics_model, sx, sy, tl_x, tl_y)
        sx/sy and tl_x/tl_y are the scale + crop-offset needed to invert the transform.
    """
    if images is None:
        return intrinsics, 1.0, 1.0, 0.0, 0.0

    H_model = float(images.shape[-2])
    W_model = float(images.shape[-1])

    if original_coords is not None:
        # Crop-aware: original_coords = [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
        # The cropped region was resized to (W_model, H_model).
        tl_x = float(original_coords[0, 0])
        tl_y = float(original_coords[0, 1])
        cr_x = float(original_coords[0, 2])
        cr_y = float(original_coords[0, 3])
        # All current creators decode K at model resolution (original-res decode removed —
        # see vggtx._forward). Detect which space K lives in by comparing the principal
        # point against the two candidate optical centres: model-res K has 2·cx ≈ W_model,
        # original-res K has 2·cx ≈ tl_x + cr_x (crop centre in original pixels). Scaling
        # a model-res K would double-apply the crop transform and corrupt reprojection.
        cx2 = 2.0 * float(intrinsics[0, 0, 2])
        if abs(cx2 - W_model) <= abs(cx2 - (tl_x + cr_x)):
            return intrinsics, 1.0, 1.0, 0.0, 0.0
        crop_w = cr_x - tl_x
        crop_h = cr_y - tl_y
        sx = W_model / crop_w
        sy = H_model / crop_h
    else:
        # No crop info: estimate from principal point (cx ≈ orig_w/2, cy ≈ orig_h/2)
        tl_x, tl_y = 0.0, 0.0
        sx = W_model / max(float(intrinsics[0, 0, 2]) * 2, 1.0)
        sy = H_model / max(float(intrinsics[0, 1, 2]) * 2, 1.0)

    intr = intrinsics.copy()
    intr[:, 0, 0] = intrinsics[:, 0, 0] * sx
    intr[:, 1, 1] = intrinsics[:, 1, 1] * sy
    intr[:, 0, 2] = (intrinsics[:, 0, 2] - tl_x) * sx
    intr[:, 1, 2] = (intrinsics[:, 1, 2] - tl_y) * sy
    return intr, sx, sy, tl_x, tl_y


def _scale_intrinsics_to_original(
    intrinsics: np.ndarray,
    sx: float,
    sy: float,
    tl_x: float,
    tl_y: float,
) -> np.ndarray:
    """Invert _scale_intrinsics_to_model: model-resolution K back to original-image space.

    It takes the (sx, sy, tl_x, tl_y) the forward returns, so the pair stays ONE transform
    rather than two open-coded copies that can drift. The crop origin is added back AFTER the
    scale is undone, mirroring the forward subtracting it BEFORE applying the scale — an origin
    term that cancels algebraically on a centred crop and is wrong on every other one.

    Args:
        intrinsics: (..., 3, 3) K on the model grid.
        sx, sy:     the forward's model/crop scale factors, divided out here.
        tl_x, tl_y: the crop's top-left corner in original pixels.
    """
    intr = np.array(intrinsics, dtype=np.float64)
    intr[..., 0, 0] = intr[..., 0, 0] / sx
    intr[..., 1, 1] = intr[..., 1, 1] / sy
    intr[..., 0, 2] = intr[..., 0, 2] / sx + tl_x
    intr[..., 1, 2] = intr[..., 1, 2] / sy + tl_y
    return intr


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
    """Return bool (N, P) observation mask: visibility gate, reprojection filter,
    frame min-inlier drop, then landmark >=2-obs/in-range drop (upstream demo_colmap order)."""
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

    # Drop under-constrained frames (too few inliers for reliable pose update) BEFORE the
    # landmark check below, so a landmark's observation count reflects only surviving
    # frames — otherwise single-observation landmarks (depth-unconstrained) can slip through
    vis[vis.sum(1) < min_inliers_per_frame] = False

    # Drop points seen from fewer than 2 (surviving) frames and points outside valid world range
    seen_enough = vis.sum(0) >= 2
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    vis[:, ~(seen_enough & in_range)] = False

    return vis


def _get_default_solver(device: str | None = None) -> Any:
    """Return CuDSS when CUDA is requested and available, else PCG.

    Args:
        device: resolved device string; ``None`` auto-detects CUDA.
                ``"cpu"`` always returns PCG even when CUDA is present.
    """
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver

            return CuDirectSparseSolver()
        except (ImportError, RuntimeError):
            pass
    return PCG()


def _carry_dropped_frames(
    refined_extrinsics: np.ndarray,
    extrinsics: np.ndarray,
    active_frames: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """Transform frames outside active_frames by the Sim(3) the active set underwent.

    BA fixes no frame and no scale, so the refined set can drift as a whole. A frame the
    inlier gate dropped keeps its pre-BA pose and would otherwise sit in the pre-BA gauge,
    inconsistent with every refined frame around it.

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

    # Estimate the world-gauge Sim(3) from how the active cameras' centres moved
    src_c = invert_poses(extrinsics_to_homogeneous(extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    dst_c = invert_poses(extrinsics_to_homogeneous(refined_extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    s, R_g, t_g = umeyama_sim3(src_c, dst_c)

    # World gauge X' = s R_g X + t_g maps a world-to-cam [R|t] to [R R_g^T | s t - R R_g^T t_g],
    # which puts the dropped camera's centre at s R_g C + t_g — the same map the points took.
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
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
    fine_tracking: bool = True,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict cross-frame 2D tracks via VGGSfM (ALIKED+SP keypoints).

    Returns:
        tracks:     (N, P, 2) float32 — 2D pixel coords per frame per point.
        vis_scores: (N, P)    float32 — visibility score in [0, 1].
        pts3d:      (P, 3)    float32 — world-space 3D positions at keypoints.
    """
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Accept numpy array input from windowed LC path (raw_outputs stores merged images as numpy)
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Always move to target_device regardless of input type — predict_tracks uses images.device
    # for tracker model placement and does NOT self-relocate. Without this unconditional .to(),
    # a CPU torch.Tensor input would leave the tracker on CPU even when target_device is CUDA.
    images = images.to(target_device)
    img_device = images.device
    # VGGSfM tracker uses grid_sample; not implemented for BFloat16 on CUDA
    images = images.float()

    conf_tensor: torch.Tensor | None = None
    if conf is not None:
        conf_tensor = conf.to(img_device)
        # Normalise to (N, H, W) — predict_tracks does not accept (N, 1, H, W)
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
def _reproject_per_camera(pts, cam_params, principal_point):
    """Per-element pinhole projection with per-camera focal; @map_transform vectorises for bae LM Jacobian."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * cam_params[..., 7:] + principal_point


@map_transform
def _reproject_shared(pts, cam_params, principal_point, focal):
    """Per-element pinhole projection with shared focal length; @map_transform vectorises for bae LM Jacobian."""
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return pts_2d * focal + principal_point


class _BAModel(nn.Module):
    """Reprojection residual module for pypose Levenberg-Marquardt optimisation."""

    def __init__(
        self,
        cam_params: torch.Tensor,  # (K, 7) SE3 or (K, 8) SE3+focal
        pts_3d: torch.Tensor,  # (L, 3) 3D landmark positions
        shared_focal: torch.Tensor | None,  # (1, 1) if shared_camera, else None
        shared_camera: bool,
    ):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(cam_params))
        self.pts = nn.Parameter(TrackingTensor(pts_3d))
        self.pose.trim_SE3_grad = True
        self.shared_intr = nn.Parameter(TrackingTensor(shared_focal)) if shared_focal is not None else None
        self.shared_camera = shared_camera

    def forward(self, points_2d, camera_indices, point_indices, principal_points):
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
