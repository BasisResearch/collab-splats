"""
Levenberg-Marquardt bundle adjustment over VGGSfM or matcher tracks, depth and brightness.

- BundleAdjustmentConfig: settings
- BundleAdjustment: refines poses and focal; K at model resolution
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from zarr.codecs import BloscCodec

import pypose as pp
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.optim.strategy import TrustRegion
from bae.utils.pypose_ambient_grad import install_pypose_ambient_grad_monkeypatch
from bae.utils.pysolvers import PCG
from vggt.dependency.track_predict import predict_tracks

from collab_splats.geometry.photometric import (
    photometric_residual,
    photometric_samples,
)
from collab_splats.geometry.projection import project
from collab_splats.geometry.tracks import build_tracks
from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    project_to_so3,
    rescale_intrinsics,
)
from collab_splats.localization.extractors import LocalMatcher
from collab_splats.utils.torch_utils import (
    batch_iterator,
    get_device,
    pytorch_gc,
    to_numpy,
)

__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]

logger = logging.getLogger(__name__)

# bae ambient-grad pose Jacobians, as upstream rgbd runs them; process-wide, read per call
os.environ["BAE_USE_PYPOSE_AMBIENT_GRAD"] = "1"
install_pypose_ambient_grad_monkeypatch()


########################################################################
# Configuration
########################################################################


@dataclass
class BundleAdjustmentConfig:
    """
    LM bundle adjustment settings, grouped by the step that reads them.

    Args:
        max_query_pts: extraction query points (upstream demo default).
        query_frame_num: extraction query frames (upstream demo default).
        fine_tracking: VGGSfM fine refinement; ~1-2 px sharper tracks, but 34 GB RSS at 50 frames.
        tracks_cache_dir: zarr track-cache dir; None always extracts.
        track_source: "vggsfm" predicts tracks; "xfeat" / "loma" build matcher star tracks (geometry/tracks.py).
        vis_thresh: min VGGSfM visibility score for an observation.
        max_reproj_error: pre-solve pixel reprojection gate; None skips the filter.
        min_inliers_per_frame: frames below this inlier count sit out the solve.
        lm_steps: max LM steps for a solve without photometric; photometric runs its fixed schedule instead.
        lm_tol: relative loss drop below which an LM step counts as stalled.
        lm_patience: stalled steps in a row that end the solve, or one photometric re-sample.
        shared_camera: one focal per scene (per-frame K spread is model noise); False: one per frame.
        refine_focal: solve for focal; False holds the input mean focal fixed (shared across cameras).
        dtype: solve precision, "float32" or "float64".
        increment_size: 0 = one global solve; 1..N-1 = frames added per incremental solve.
        use_photometric: brightness matching between overlapping frames (global solve only).
        use_depth: track camera z against feedforward depth.
        depth_sigma: relative depth error weighted like 1 px of reprojection error.
        device: CUDA device ("cuda", "cuda:1"), None = auto; CPU unsupported (bae LM is CUDA-only).
    """

    # Tracks
    max_query_pts: int = 4096
    query_frame_num: int = 8
    fine_tracking: bool = False
    tracks_cache_dir: Path | None = None
    track_source: Literal["vggsfm", "xfeat", "loma"] = "vggsfm"

    # Filter
    vis_thresh: float = 0.2
    max_reproj_error: float | None = 4.0
    min_inliers_per_frame: int = 64

    # Solve
    lm_steps: int = 40
    lm_tol: float = 1e-4
    lm_patience: int = 2
    shared_camera: bool = True
    refine_focal: bool = True
    dtype: Literal["float32", "float64"] = "float32"
    increment_size: int = 0
    use_photometric: bool = True
    use_depth: bool = True
    depth_sigma: float = 0.01

    # Runtime
    device: str | None = None

    def __post_init__(self) -> None:
        """
        Refuse a dtype or term combination the solver cannot run.

        - photometric reads images, which the incremental solve does not carry
        """
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"BundleAdjustmentConfig.dtype must be 'float32' or 'float64', got {self.dtype!r}")

        if self.use_photometric and self.increment_size > 0:
            raise ValueError("BundleAdjustmentConfig: use_photometric needs increment_size 0 (global solve)")

        if self.track_source not in ("vggsfm", "xfeat", "loma"):
            raise ValueError(
                f"BundleAdjustmentConfig.track_source must be vggsfm, xfeat or loma, got {self.track_source!r}"
            )


########################################################################
# BundleAdjustment
########################################################################


class BundleAdjustment:
    """
    Refines camera poses and focal against VGGSfM or matcher tracks.

    - points untouched; the caller re-derives them under the new poses
    - reports per refine: loss_history, alignment_scale, losses
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        """
        Store the settings; reports start empty.
        """
        self.config = config or BundleAdjustmentConfig()

        # Populated per _optimize() call with that call's per-step LM losses
        self.loss_history: list[list[float]] = []

        # Last solve's reports: alignment scale, final loss per term
        self.alignment_scale: float | None = None
        self.losses: dict[str, float] = {}

    def extract_tracks(
        self,
        images: np.ndarray,
        confidence: np.ndarray,
        world_points: np.ndarray,
        image_paths: list | None,
        *,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        frame_paths: list[Path] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracks for the frames from the configured source; cached under tracks_cache_dir when set.

        - no cache dir or no image_paths: always extracts, nothing is cached
        - a key mismatch or an unreadable store is deleted and rebuilt

        Args:
            images: (N, 3, H, W) model-grid frames, the track source.
            confidence: (N, H, W) per-pixel confidence for query-point sampling.
            world_points: (N, H, W, 3) the tracks are lifted from.
            image_paths: frame paths for the cache key; None skips the cache.
            extrinsics: (N, 4, 4) world-to-cam; matcher sources only.
            intrinsics: (N, 3, 3) model-grid K; matcher sources only.
            frame_paths: full-res store frame per model frame; matcher sources only.

        Returns:
            tracks (N, P, 2), vis_scores (N, P) and pts3d_tracks (P, 3), all float32.

        Raises:
            ValueError: a matcher source without extrinsics, intrinsics or frame_paths.
        """
        cfg = self.config

        # No cache configured, or nothing to key it on
        if cfg.tracks_cache_dir is None or not image_paths:
            return self._extract_uncached(images, confidence, world_points, extrinsics, intrinsics, frame_paths)

        # Cache location and the key a valid store must carry
        cache_path = Path(cfg.tracks_cache_dir) / "tracks.zarr"
        expected_key = _compute_tracks_cache_key(image_paths, world_points, cfg)

        # Attempt to read from existing cache; validate key before using
        if cache_path.exists():
            try:
                store = zarr.open(str(cache_path), mode="r")
                if store.attrs.get("cache_key") == expected_key:
                    logger.info("Track cache hit (skipping extraction): %s", cache_path)
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
        tracks, vis_scores, pts3d_tracks = self._extract_uncached(
            images, confidence, world_points, extrinsics, intrinsics, frame_paths
        )

        # One lz4 chunk per array; the key stamp makes the store valid
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(cache_path), mode="w")
        for key, arr in [("tracks", tracks), ("vis_scores", vis_scores), ("pts3d_tracks", pts3d_tracks)]:
            store.create_array(key, data=arr, chunks=arr.shape, compressors=lz4)
        store.attrs["cache_key"] = expected_key
        logger.debug("Track cache saved: %s", cache_path)

        return tracks, vis_scores, pts3d_tracks

    def _extract_uncached(
        self,
        images: np.ndarray,
        confidence: np.ndarray,
        world_points: np.ndarray,
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
        frame_paths: list[Path] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracks from the configured source, uncached.

        - matcher sources need poses, model-grid K and full-res frame paths; confidence is unused there
        """
        cfg = self.config

        if cfg.track_source == "vggsfm":
            return extract_tracks_vggsfm(images, confidence, world_points, cfg)

        # Matcher tracks read prior poses and full-res frames
        if extrinsics is None or intrinsics is None or not frame_paths:
            raise ValueError(f"BA track_source {cfg.track_source!r} needs extrinsics, intrinsics and frame_paths")

        matcher = LocalMatcher(cfg.track_source)
        tracks = build_tracks(matcher, images, frame_paths, world_points, extrinsics, intrinsics)

        # Free the matcher before the bae solve claims the GPU
        del matcher
        pytorch_gc()
        return tracks

    def _refine_incremental(
        self,
        extrinsics: np.ndarray,
        tracks: np.ndarray,
        vis_scores: np.ndarray,
        pts3d_tracks: np.ndarray,
        intrinsics: np.ndarray,
        increment_size: int,
        depth: np.ndarray | None,
        confidence: np.ndarray,
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
            refined_ext_k, refined_intr_k = self._optimize(
                pts3d_tracks,
                refined_extrinsics[:k].copy(),  # warm start from previous step
                refined_intrinsics[:k].copy(),
                tracks[:k],
                vis_scores[:k],
                images=None,
                depth=None if depth is None else depth[:k],
                confidence=confidence[:k],
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
        depth: np.ndarray | None = None,
        frame_paths: list[Path] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine camera poses and focal; K must be at model resolution.

        - reprojection always; depth and photometric per the config's use_* flags
        - the result is moved back onto the input poses (see _align_to_input_poses)

        Args:
            images: (N, 3, H, W) model-grid frames, the track source.
            confidence: (N, H, W) per-pixel confidence for query-point sampling.
            world_points: (N, H, W, 3) the tracks are lifted from.
            extrinsics: (N, 4, 4) world-to-cam start poses.
            intrinsics: (N, 3, 3) model-resolution K.
            image_paths: frame paths for the track-cache key; None skips the cache.
            depth: (N, H, W) model-grid z-depth; needed by use_depth and use_photometric.
            frame_paths: full-res store frame per model frame; read by matcher track sources.

        Returns:
            (N, 4, 4) refined extrinsics and (N, 3, 3) refined intrinsics.

        Raises:
            ValueError: < 2 frames or points active after filtering; or depth None with a depth term on.
            RuntimeError: the resolved device is not CUDA (bae LM is CUDA-only).
        """
        # Depth and photometric terms read feedforward depth
        if depth is None and (self.config.use_depth or self.config.use_photometric):
            raise ValueError("BA: use_depth / use_photometric need depth; pass depth= or turn both off")

        self.loss_history = []
        self.losses = {}
        self.alignment_scale = None
        N = len(images)
        logger.info(
            "BA refine start: %d frames (increment_size=%d, lm_steps=%d, max_reproj_error=%s)",
            N,
            self.config.increment_size,
            self.config.lm_steps,
            self.config.max_reproj_error,
        )

        # Load cached tracks or extract from the configured source (one extraction shared across k-steps)
        tracks, vis_scores, pts3d_tracks = self.extract_tracks(
            images,
            confidence,
            world_points,
            image_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_paths=frame_paths,
        )
        logger.info("Tracks ready: %d points across %d frames", tracks.shape[1], tracks.shape[0])

        # Numpy copies for the depth and photometric terms; tracks above take the tensors as given
        images = to_numpy(images)
        confidence = to_numpy(confidence)

        # One global solve, or an incremental one when increment_size < N
        increment_size = self.config.increment_size
        if increment_size == 0 or increment_size >= N:
            logger.info("Global BA over %d frames", N)
            refined_extrinsics, refined_intrinsics = self._optimize(
                pts3d_tracks,
                extrinsics[:, :3, :],
                intrinsics,
                tracks,
                vis_scores,
                images=images,
                depth=depth,
                confidence=confidence,
            )
        else:
            refined_extrinsics, refined_intrinsics = self._refine_incremental(
                extrinsics,
                tracks,
                vis_scores,
                pts3d_tracks,
                intrinsics,
                increment_size,
                depth,
                confidence,
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
        images: np.ndarray | None,
        depth: np.ndarray | None,
        confidence: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One LM solve over tracks, plus depth and photometric when on.

        - extrinsics in: (N, 3, 4) world-to-cam; frames the inlier gate drops keep their input poses
        - photometric: 3 image scales coarse to fine, (3, 2, 1) re-samples x 5 IRLS steps
        - early stop: lm_patience steps in a row below lm_tol end the solve, or the current re-sample
        - returns extrinsics (N, 3, 4) and K (N, 3, 3)
        """
        cfg = self.config
        max_reproj = cfg.max_reproj_error
        n_steps = cfg.lm_steps
        N = len(extrinsics)

        # Work on copies
        refined_extrinsics = extrinsics.astype(np.float32).copy()
        refined_intrinsics = intrinsics.astype(np.float32).copy()

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

        # Solve precision for every tensor below
        dtype = getattr(torch, cfg.dtype)

        # Flatten the active observations into (frame, point) index pairs
        frame_idx, pt_idx = np.where(vis[np.ix_(active_frames, active_pts)])

        # Step budget: lm_steps without photometric, else the photometric schedule; both stop early on a stall
        steps = "photometric schedule" if cfg.use_photometric else f"up to {n_steps} steps"
        logger.info(
            "LM optimize: %d/%d frames, %d/%d points, %d observations, %s",
            len(active_frames),
            vis.shape[0],
            len(active_pts),
            vis.shape[1],
            len(frame_idx),
            steps,
        )
        global_frame_idx = active_frames[frame_idx]
        global_pt_idx = active_pts[pt_idx]
        obs_2d = tracks[global_frame_idx, global_pt_idx].astype(np.float64)  # (M, 2)

        # Depth target per observation: nearest feedforward depth, weight sqrt(conf) / (D * depth_sigma)
        target_z = np.zeros(len(obs_2d))
        weight_z = np.zeros(len(obs_2d))

        if cfg.use_depth:
            H, W = depth.shape[1:]
            col = np.clip(np.floor(obs_2d[:, 0]).astype(int), 0, W - 1)
            row = np.clip(np.floor(obs_2d[:, 1]).astype(int), 0, H - 1)
            obs_depth = depth[global_frame_idx, row, col].astype(np.float64)
            obs_conf = confidence[global_frame_idx, row, col] / np.median(confidence)
            valid = obs_depth > 0
            weight_z[valid] = np.sqrt(obs_conf[valid]) / (obs_depth[valid] * cfg.depth_sigma)
            target_z = obs_depth

        # Build SE3 camera tensor from (K, 3, 4) extrinsics; pad to (K, 4, 4) for mat2SE3
        ext_sub = extrinsics[active_frames].astype(np.float64)
        ext_4x4 = extrinsics_to_homogeneous(ext_sub)

        # Snap rotations to SO(3): float32 rotations from some backends fail pypose's mat2SE3 check
        ext_4x4[:, :3, :3] = project_to_so3(ext_4x4[:, :3, :3])
        ext_t = torch.tensor(ext_4x4, dtype=dtype, device=device)
        cameras_se3 = pp.mat2SE3(ext_t)

        # SIMPLE_PINHOLE: average fx/fy as single focal length per camera
        focal = ((intrinsics[active_frames, 0, 0] + intrinsics[active_frames, 1, 1]) / 2.0).astype(np.float64)
        focal_tensor = torch.tensor(focal, dtype=dtype, device=device).unsqueeze(-1)  # (K, 1)
        principal_points = torch.tensor(
            intrinsics[active_frames, :2, 2].astype(np.float64),
            dtype=dtype,
            device=device,
        )  # (K, 2)
        pts3d_tensor = torch.tensor(
            pts3d[active_pts].astype(np.float64),
            dtype=dtype,
            device=device,
        )  # (L, 3)

        # Concatenate focal length into camera params tensor for per-camera case
        if cfg.shared_camera or not cfg.refine_focal:
            cam_params = cameras_se3.data  # (K, 7)
            shared_focal = focal_tensor.mean(0, keepdim=True)  # (1, 1)
        else:
            cam_params = torch.cat([cameras_se3.data, focal_tensor], dim=-1)  # (K, 8)
            shared_focal = None

        # Assemble observation index tensors
        cam_idx = torch.tensor(frame_idx, dtype=torch.long, device=device)
        pt_idx_t = torch.tensor(pt_idx, dtype=torch.long, device=device)

        # Track-row targets [u, v, D] and weights [1, 1, depth weight]
        target = np.column_stack([obs_2d, target_z])
        weight = np.column_stack([np.ones_like(obs_2d), weight_z])

        # Observation dict the model reads each step
        input_dict = {
            "camera_indices": cam_idx,
            "point_indices": pt_idx_t,
            "principal_points": principal_points,
            "target": torch.tensor(target, dtype=dtype, device=device),
            "weight": torch.tensor(weight, dtype=dtype, device=device),
        }

        # Image scales for photometric, coarse to fine: (gray, valid-masked mean depth, K, re-samples)
        levels = []

        if cfg.use_photometric:
            rgb = torch.tensor(images[active_frames], dtype=dtype, device=device)
            gray = rgb.mean(1, keepdim=True)
            depth_t = torch.tensor(depth[active_frames], dtype=dtype, device=device)[:, None]
            valid_t = (depth_t > 0).to(dtype)
            full_hw = np.array(gray.shape[-2:])

            for factor, n_relin in ((4, 3), (2, 2), (1, 1)):
                gray_l = F.avg_pool2d(gray, factor)[:, 0]
                valid_l = F.avg_pool2d(valid_t, factor)[:, 0]
                depth_l = F.avg_pool2d(depth_t * valid_t, factor)[:, 0] / valid_l.clamp(min=1e-6)

                # Pooled pixel j covers full pixels [f j, f j + f): K scales by exactly 1/f, the floor only crops
                level_hw = full_hw / factor
                K_l = rescale_intrinsics(intrinsics[active_frames], full_hw, level_hw)
                K_l = torch.tensor(K_l, dtype=dtype, device=device)
                levels.append((gray_l, depth_l, K_l, n_relin))

        # Optimize the stacked residual with Levenberg-Marquardt
        with torch.enable_grad():
            model = _BAModel(cam_params, pts3d_tensor, shared_focal, refine_focal=cfg.refine_focal)

            # bae TrustRegion: non-positive predicted drop counts as rejected (pypose's does not)
            strategy = TrustRegion(up=2.0, down=0.5**4, max=1e6)

            # Matrix-free normal equations: J^T J never formed, PCG on the operator
            pcg = PCG(tol=1e-4, maxiter=250)
            optimizer = LM(
                model,
                strategy=strategy,
                solver=lambda A, b: pcg(A, b).view(-1, 1),
                reject=10,
                matrix_free_normal=True,
            )

            # Bind target=None: pypose>=0.7 RobustModel.forward needs it, bae LM.step passes none
            optimizer.model.forward = functools.partial(type(optimizer.model).forward, optimizer.model, target=None)

            # Manual LM loop: pypose StopOnPlateau stops on any rejected step
            loss_hist: list[float] = []

            if not levels:
                torch.cuda.empty_cache()
                stalled = 0

                for _ in range(n_steps):
                    loss, drop = _lm_step(optimizer, input_dict)
                    loss_hist.append(loss)
                    logger.info("LM step %d: loss=%.6e drop=%.2e", len(loss_hist), loss, drop)

                    # Stalled steps in a row end the solve
                    stalled = stalled + 1 if drop < cfg.lm_tol else 0

                    if stalled >= cfg.lm_patience:
                        break

            # Photometric schedule: per image scale, re-sample the pixels, then 5 IRLS-reweighted steps
            n_draws = 0

            for gray_l, depth_l, K_l, n_relin in levels:

                # Hand torch's cached blocks back once per scale: warp allocates the Jacobian outside torch's pool
                torch.cuda.empty_cache()

                for _ in range(n_relin):

                    # Re-sample pixel brightness and image slope at the current poses and focal
                    pose_se3 = pp.SE3(model.pose.data[:, :7].detach())
                    w2c = pose_se3.matrix()
                    K_cur = K_l.clone()

                    # Scale K's focal by the live focal over the input one: shared, or per camera in pose column 7
                    if model.shared_intr is not None:
                        K_cur[:, [0, 1], [0, 1]] *= float(model.shared_intr.data) / focal_tensor
                    else:
                        K_cur[:, [0, 1], [0, 1]] *= model.pose.data[:, 7:8].detach() / focal_tensor

                    # One seed per re-sample, so every draw picks fresh pixels yet repeats run to run
                    with torch.no_grad():
                        samples = photometric_samples(w2c, gray_l, depth_l, K_cur, seed=n_draws)

                    n_draws += 1

                    if samples is None:
                        logger.warning("BA: too few photometric samples at %s, scale skipped", gray_l.shape)
                        break

                    input_dict["photometric"] = samples
                    base_weight = samples["weight"].clone()
                    stalled = 0

                    for _ in range(5):

                        # IRLS Huber on the photometric samples: delta = 1.5 x median normalized residual
                        with torch.no_grad():
                            samples["weight"] = base_weight
                            r = model(**input_dict)[len(obs_2d) :, 0].tensor().abs() / base_weight
                            delta = 1.5 * r.median().clamp(min=1e-9)
                            samples["weight"] = base_weight * (delta / r.clamp(min=delta)).sqrt()

                        optimizer.loss = optimizer.model.loss(input_dict, None)
                        loss, drop = _lm_step(optimizer, input_dict)
                        loss_hist.append(loss)
                        logger.info("LM step %d: loss=%.6e drop=%.2e", len(loss_hist), loss, drop)

                        # Stalled steps in a row end this re-sample; the next re-sample still runs
                        stalled = stalled + 1 if drop < cfg.lm_tol else 0

                        if stalled >= cfg.lm_patience:
                            break

            # No LM step ran (lm_steps 0, or no photometric scale had samples): only the alignment below applies
            if not loss_hist:
                logger.warning("BA: no LM step ran; refine is a no-op apart from the alignment to input poses")

            self.loss_history.append(loss_hist)

            # Final loss per term, sliced from the stacked residual
            with torch.no_grad():
                final = model(**input_dict).tensor()

            M = len(obs_2d)
            self.losses = {"reprojection": float(final[:M, :2].square().sum())}

            if cfg.use_depth:
                self.losses["depth"] = float(final[:M, 2].square().sum())

            if cfg.use_photometric:
                self.losses["photometric"] = float(final[M:, 0].square().sum())

        # Recover (3, 4) extrinsics from optimized SE3 quaternion representation
        opt_cam = model.pose.data.detach().cpu().numpy()  # (K, 7) or (K, 8)
        opt_se3 = pp.SE3(torch.tensor(opt_cam[:, :7], dtype=torch.float64))
        opt_extrinsics_3x4 = opt_se3.matrix().numpy()[:, :3, :]
        refined_extrinsics[active_frames] = opt_extrinsics_3x4.astype(np.float32)

        # Move refined cameras back onto the input poses; scale fixed when depth sets it
        fix_scale = cfg.use_photometric or cfg.use_depth
        refined_extrinsics, self.alignment_scale = _align_to_input_poses(
            refined_extrinsics, extrinsics, active_frames, fix_scale=fix_scale
        )
        logger.info("BA: alignment to input poses, scale %s", self.alignment_scale)

        # Log frames the inlier gate dropped
        n_dropped = N - len(active_frames)

        if n_dropped:
            logger.warning(
                "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s); kept at their input poses",
                n_dropped,
                N,
                cfg.min_inliers_per_frame,
                np.setdiff1d(np.arange(N), active_frames).tolist(),
            )

        # Focal back to K: shared to every frame (dropped too), per-frame to active frames only
        if cfg.shared_camera or not cfg.refine_focal:
            focal_val = float(model.shared_intr.data.detach().cpu().numpy().mean())
            refined_intrinsics[:, 0, 0] = focal_val
            refined_intrinsics[:, 1, 1] = focal_val
        else:
            opt_focal = opt_cam[:, 7]
            refined_intrinsics[active_frames, 0, 0] = opt_focal
            refined_intrinsics[active_frames, 1, 1] = opt_focal

        return refined_extrinsics, refined_intrinsics


########################################################################
# BA model
########################################################################


@map_transform
def _reproject_per_camera(pts: torch.Tensor, cam_params: torch.Tensor, principal_point: torch.Tensor) -> torch.Tensor:
    """
    Pinhole projection with a per-camera focal, plus camera z.

    - @map_transform vectorizes it for the bae LM Jacobian
    - cam_params: SE3 (7) plus focal (1); every argument has one row per observation
    - returns [u, v, z]
    """
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return torch.cat([pts_2d * cam_params[..., 7:] + principal_point, pts_cam[..., 2:]], dim=-1)


@map_transform
def _reproject_shared(
    pts: torch.Tensor, cam_params: torch.Tensor, principal_point: torch.Tensor, focal: torch.Tensor
) -> torch.Tensor:
    """
    Pinhole projection with one focal shared by every camera, plus camera z.

    - @map_transform vectorizes it for the bae LM Jacobian
    - cam_params: SE3 (7); focal: the shared focal broadcast; one row per observation
    - returns [u, v, z]
    """
    pts_cam = pp.SE3(cam_params[..., :7]).Act(pts)
    pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
    return torch.cat([pts_2d * focal + principal_point, pts_cam[..., 2:]], dim=-1)


class _BAModel(nn.Module):
    """
    Residual module for bae LM: track residuals (reprojection + depth), then photometric.

    - every residual block is 3 wide so they stack
    """

    def __init__(
        self,
        cam_params: torch.Tensor,
        pts_3d: torch.Tensor,
        shared_focal: torch.Tensor | None,
        *,
        refine_focal: bool,
    ) -> None:
        """
        Wrap poses, points and the optional shared focal as bae parameters.

        - cam_params: (K, 7) SE3, or (K, 8) with focal; pts_3d: (L, 3); shared_focal: (1, 1) or None
        - refine_focal False: shared focal is a constant buffer, outside the Jacobian
        """
        super().__init__()

        # Trackable parameters for the bae sparse Jacobian
        # - trim_SE3_grad: 7-D SE3 pose takes a 6-DoF se3 step, trailing focal steps as-is
        self.pose = nn.Parameter(TrackingTensor(cam_params))
        self.pose.trim_SE3_grad = True
        self.pts = nn.Parameter(TrackingTensor(pts_3d))

        if shared_focal is not None and not refine_focal:
            self.register_buffer("shared_intr", shared_focal)
        elif shared_focal is not None:
            self.shared_intr = nn.Parameter(TrackingTensor(shared_focal))
        else:
            self.shared_intr = None

    def forward(
        self,
        camera_indices: torch.Tensor,
        point_indices: torch.Tensor,
        principal_points: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        photometric: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Residuals: [du, dv, depth] per track observation, then [r, 0, 0] per photometric sample.

        - target / weight (M, 3): [u, v, D] and [1, 1, depth weight]
        - photometric: photometric_samples output, or None
        """
        # Track residuals: reprojection pixels and camera z against depth
        pts = self.pts[point_indices]
        cam = self.pose[camera_indices]
        ctr = principal_points[camera_indices]

        if self.shared_intr is not None:
            focal = self.shared_intr[torch.zeros_like(camera_indices)]
            pred = _reproject_shared(pts, cam, ctr, focal)
        else:
            pred = _reproject_per_camera(pts, cam, ctr)

        blocks = [(pred - target) * weight]

        # Photometric residuals: brightness of each sampled pixel against its match in the other frame
        if photometric is not None:
            residual = photometric_residual(
                self.pose[photometric["i_idx"]],
                self.pose[photometric["j_idx"]],
                photometric["x_i"],
                photometric["K_j"],
                photometric["I_i"],
                photometric["I_j"],
                photometric["gx_j"],
                photometric["gy_j"],
                photometric["uv_j"],
                photometric["weight"],
            )
            blocks.append(residual)

        return torch.cat(blocks) if len(blocks) > 1 else blocks[0]


def _lm_step(optimizer: LM, input_dict: dict[str, Any]) -> tuple[float, float]:
    """
    One bae LM step that never keeps a loss increase; returns (loss, relative drop).

    - bae keeps the last rejected step once its reject budget runs out; undone here
    - drop = (loss before - loss after) / loss before; an undone step drops 0
    """
    params = [p for p in optimizer.model.parameters() if p.requires_grad]
    before = [p.data.clone() for p in params]
    loss = float(optimizer.step(input=input_dict))

    # Undo a step that raised the loss
    if loss > float(optimizer.last):
        for p, b in zip(params, before):
            p.data.copy_(b)

        optimizer.loss = optimizer.last
        loss = float(optimizer.last)
        logger.warning("BA: LM step raised the loss after %d rejects, undone", optimizer.reject_count)

    # Relative drop against the loss before the step; zero when there is no prior loss
    last = float(optimizer.last)
    drop = (last - loss) / last if last > 0 else 0.0
    return loss, drop


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
    batch_size: int = 1_000_000,
) -> np.ndarray:
    """
    Boolean (N, P) observation mask for the BA solve; True enters the solve.

    - order: visibility gate, reprojection filter, frame min-inlier drop, landmark drop
    - landmark drop: < 2 views or out of range
    - order follows upstream VGGT demo_colmap
    - max_reproj None skips the reprojection filter
    - reprojection checks only gated (frame, point) pairs, float64 on get_device(), batch_size pairs at a time
    """
    # Visibility gate: keep observations the tracker is confident about
    vis = vis_scores > vis_thresh

    # Remove observations with high reprojection error under the current poses
    if max_reproj is not None:
        frame_idx, point_idx = np.nonzero(vis)
        device = get_device()
        points = torch.as_tensor(pts3d, dtype=torch.float64, device=device)
        world_to_cam = torch.as_tensor(extrinsics, dtype=torch.float64, device=device)
        K = torch.as_tensor(intrinsics, dtype=torch.float64, device=device)
        pairs = [torch.as_tensor(a, device=device) for a in (frame_idx, point_idx, tracks[frame_idx, point_idx])]
        inliers = [torch.zeros(0, dtype=torch.bool, device=device)]

        # Each batch projects its pairs as one-point cameras; behind-camera and NaN fail the gate
        for frames, pair_points, observed in batch_iterator(batch_size, *pairs):
            pixels, points_cam = project(points[pair_points][:, None], world_to_cam[frames], K[frames])
            err = (pixels[:, 0] - observed.double()).norm(dim=-1)
            inliers.append((err <= max_reproj) & (points_cam[:, 0, 2] > 0))

        reproj_ok = torch.cat(inliers)
        reproj_ok = to_numpy(reproj_ok)
        vis[frame_idx[~reproj_ok], point_idx[~reproj_ok]] = False

    # Drop frames with too few inliers, before the landmark check
    # - landmark counts then see surviving frames only, so 1-view points drop
    vis[vis.sum(1) < min_inliers_per_frame] = False

    # Drop points seen from fewer than 2 (surviving) frames and points outside valid world range
    seen_enough = vis.sum(0) >= 2
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    vis[:, ~(seen_enough & in_range)] = False

    return vis


def _align_to_input_poses(
    refined_extrinsics: np.ndarray,
    extrinsics: np.ndarray,
    active_frames: np.ndarray,
    *,
    fix_scale: bool,
    min_spread: float = 1e-12,
) -> tuple[np.ndarray, float | None]:
    """
    Move the refined cameras, as one rigid body, back onto the input cameras.

    - BA pins no camera or scale, so the whole solution drifts; this undoes that
    - rotation: mean per-camera rotation change; positions alone miss a spin about a straight path
    - fix_scale: skip the scale fit; depth already set the scale
    - refined center spread (sum of squares) below min_spread: no scale to fit, s = 1
    - returns the moved copy and its scale s; < 3 active frames: (input, None)
    """
    if len(active_frames) < 3:
        return refined_extrinsics, None

    # Refined and input world-to-cam poses over the active frames
    R_ref = refined_extrinsics[active_frames, :, :3].astype(np.float64)
    t_ref = refined_extrinsics[active_frames, :, 3].astype(np.float64)
    R_inp = extrinsics[active_frames, :, :3].astype(np.float64)
    t_inp = extrinsics[active_frames, :, 3].astype(np.float64)

    # Rotation: average of R_inp^T R_ref, snapped to the nearest rotation
    R_g = project_to_so3(np.einsum("nji,njk->ik", R_inp, R_ref))

    # Scale and translation from camera centers C = -R^T t under that rotation
    ctr_ref = -np.einsum("nji,nj->ni", R_ref, t_ref)
    ctr_inp = -np.einsum("nji,nj->ni", R_inp, t_inp)
    ref_dev = (ctr_ref - ctr_ref.mean(0)) @ R_g.T
    inp_dev = ctr_inp - ctr_inp.mean(0)
    spread = float((ref_dev**2).sum())
    s = 1.0 if fix_scale or spread < min_spread else float((ref_dev * inp_dev).sum() / spread)
    t_g = ctr_inp.mean(0) - s * R_g @ ctr_ref.mean(0)

    # Move the world by X' = s R_g X + t_g: [R|t] -> [R R_g^T | s t - R R_g^T t_g]
    R_new = R_ref @ R_g.T
    t_new = s * t_ref - R_new @ t_g
    out = refined_extrinsics.copy()
    out[active_frames, :, :3] = R_new
    out[active_frames, :, 3] = t_new
    return out, s


def _compute_tracks_cache_key(image_paths: list, world_points: np.ndarray, cfg: BundleAdjustmentConfig) -> str:
    """
    SHA-256 track-cache key over image paths, world points and extraction config.

    - any frame, world_points or track-setting change invalidates it; image_paths order does not
    - world_points digest: two backbones over one image set must not share tracks
    - digest cost ~1-3 s at 300 frames
    """
    meta = {
        "image_paths": sorted(str(p) for p in image_paths),
        "world_points": hashlib.sha256(np.ascontiguousarray(world_points).tobytes()).hexdigest(),
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
        "fine_tracking": cfg.fine_tracking,
        "track_source": cfg.track_source,
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
