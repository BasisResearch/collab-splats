"""
Levenberg-Marquardt bundle adjustment over VGGSfM or matcher tracks, depth and brightness.

- BundleAdjustmentConfig: settings
- BundleAdjustment: refines poses and focal; K at model resolution
- importing sets BAE_USE_PYPOSE_AMBIENT_GRAD=1 process-wide
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pypose as pp
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.optim.optimizer import Schur
from bae.optim.strategy import TrustRegion
from bae.utils.pypose_ambient_grad import install_pypose_ambient_grad_monkeypatch
from bae.utils.pysolvers import PCG

from collab_splats.geometry.photometric import (
    KEYS,
    photometric_residual,
    photometric_samples,
)
from collab_splats.geometry.projection import reprojection_error
from collab_splats.geometry.tracks import extract_tracks
from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    project_to_so3,
    rescale_intrinsics,
    shift_intrinsics,
)
from collab_splats.utils.torch_utils import (
    batch_iterator,
    get_device,
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

    Tracks:
        - track_source: "vggsfm" predicts tracks; "xfeat" / "loma" build matcher star tracks
        - tracks_cache_dir: zarr track-cache dir; None always extracts
        - track_kwargs: source-specific settings forwarded to extract_tracks; empty takes their defaults
          (vggsfm: max_query_pts, query_frame_num, fine_tracking; matchers: retrieval, seed_fraction, window, ...)

    Filter:
        - vis_thresh: min track score for an observation (VGGSfM visibility; matcher tracks score 1.0)
        - max_reproj_error: pre-solve pixel reprojection gate; None skips the filter
        - min_inliers_per_frame: frames below this inlier count sit out the solve

    Solve:
        - solver: "schur" eliminates points (camera-only PCG, less GPU); "lm" solves the joint system
        - dtype: solve precision, "float32" or "float64"
        - lm_steps: max LM steps without photometric; photometric runs its fixed schedule
        - lm_tol: relative loss drop below which an LM step counts as stalled
        - lm_patience: stalled steps in a row that end the solve, or one photometric re-sample
        - increment_size: 0 = one global solve; 1..N-1 = frames added per incremental solve
        - shared_camera: one focal per scene (per-frame K spread is model noise); False: one per frame
        - refine_focal: solve for focal; False holds the input mean focal fixed (shared across cameras)

    Terms:
        - use_depth: track camera z against feedforward depth
        - depth_sigma: relative depth error weighted like 1 px of reprojection error
        - use_photometric: brightness matching between overlapping frames (global solve only)

    Runtime:
        - device: solve CUDA device ("cuda", "cuda:1"), None = auto; tracks run on get_device()
    """

    # Tracks
    track_source: Literal["vggsfm", "xfeat", "loma"] = "xfeat"
    tracks_cache_dir: Path | None = None
    track_kwargs: dict[str, Any] = field(default_factory=dict)

    # Filter
    vis_thresh: float = 0.2
    max_reproj_error: float | None = 4.0
    min_inliers_per_frame: int = 64

    # Solve
    solver: Literal["schur", "lm"] = "schur"
    dtype: Literal["float32", "float64"] = "float64"
    lm_steps: int = 40
    lm_tol: float = 1e-4
    lm_patience: int = 2
    increment_size: int = 0
    shared_camera: bool = True
    refine_focal: bool = False

    # Terms
    use_depth: bool = True
    depth_sigma: float = 0.01
    use_photometric: bool = True

    # Runtime
    device: str | None = None

    def __post_init__(self) -> None:
        """
        Refuse a dtype or term combination the solver cannot run.

        - photometric reads images, which the incremental solve does not carry
        - Schur splits cameras and points only; a refined shared focal is a third block
        """
        # Allowed-value fields
        for name, allowed in (
            ("dtype", ("float32", "float64")),
            ("track_source", ("vggsfm", "xfeat", "loma")),
            ("solver", ("schur", "lm")),
        ):
            value = getattr(self, name)

            # Reject a value outside its allowed set
            if value not in allowed:
                raise ValueError(
                    f"BundleAdjustmentConfig.{name} must be one of {allowed}, got {value!r}"
                )

        # Photometric reads images, which the incremental solve does not carry
        if self.use_photometric and self.increment_size > 0:
            raise ValueError(
                "BundleAdjustmentConfig: use_photometric needs increment_size 0 (global solve)"
            )

        # Schur splits cameras and points only; a refined shared focal is a third block
        if self.solver == "schur" and self.refine_focal and self.shared_camera:
            raise ValueError(
                "BundleAdjustmentConfig: solver 'schur' cannot refine a shared focal; use 'lm' or freeze it"
            )


########################################################################
# BundleAdjustment
########################################################################


class BundleAdjustment:
    """
    Refines camera poses and focal against VGGSfM or matcher tracks.

    - points untouched; the caller re-derives them under the new poses
    - reports: loss_history (one list per solve, kept across refine calls), alignment_scale, losses
    """

    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        """
        Store the settings; reports start empty.

        Args:
            config: BA settings; None takes the defaults.
        """
        self.config = config or BundleAdjustmentConfig()

        # Populated per _optimize() call with that call's per-step LM losses
        self.loss_history: list[list[float]] = []

        # Last solve's reports: alignment scale, final loss per term
        self.alignment_scale: float | None = None
        self.losses: dict[str, float] = {}

    def refine(
        self,
        images: np.ndarray | torch.Tensor,
        confidence: np.ndarray | torch.Tensor,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        depth: np.ndarray,
        frame_paths: list[Path] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine camera poses and focal; K must be at model resolution.

        - reprojection always; depth and photometric per the config's use_* flags
        - the result is moved back onto the input poses (see _align_to_input_poses)

        Args:
            images: (N, 3, H, W) model-grid frames in [0, 1]; VGGSfM tracks and the photometric term.
            confidence: (N, H, W) per-pixel confidence; VGGSfM query sampling and the depth-term weight.
            world_points: (N, H, W, 3) the tracks take their 3D points from.
            extrinsics: (N, 4, 4) world-to-cam start poses.
            intrinsics: (N, 3, 3) model-resolution K.
            depth: (N, H, W) model-grid z-depth; read by use_depth and use_photometric.
            frame_paths: full-res store frame per model frame; the track-cache key, and matcher sources' input.

        Returns:
            (N, 4, 4) refined extrinsics and (N, 3, 3) refined intrinsics.

        Raises:
            ValueError: < 2 frames or points active after filtering, or from extract_tracks: a matcher
                source without frame_paths, a cropped-aspect frame, or a bad retrieval / seed_fraction.
            TypeError: a track_kwargs key the track_source does not take.
            RuntimeError: the resolved device is not CUDA (bae LM is CUDA-only).
        """
        cfg = self.config
        N = len(images)
        logger.info("BA refine start: %d frames, %s", N, cfg)

        # Tracks from the configured source; one extraction serves every incremental step
        frame, track, xy, score, pts3d_tracks = extract_tracks(
            images,
            confidence,
            world_points,
            source=cfg.track_source,
            frame_paths=frame_paths,
            cache_dir=cfg.tracks_cache_dir,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            **cfg.track_kwargs,
        )
        logger.info(
            "Tracks ready: %d observations, %d points", len(frame), len(pts3d_tracks)
        )

        # Numpy copies for the depth and photometric terms; tracks above take the tensors as given
        images = to_numpy(images)
        confidence = to_numpy(confidence)

        # Solve windows of the first k frames warm-started from the last; steps end at N, 1-frame windows skipped
        step = cfg.increment_size or N
        steps = [k for k in range(step, N, step) if k > 1] + [N]
        refined_extrinsics = extrinsics[:, :3].copy()
        refined_intrinsics = intrinsics.copy()

        # Each window refines the poses and K of its first k frames
        for k in steps:
            logger.info("BA solve over %d/%d frames", k, N)
            in_window = frame < k
            refined_extrinsics[:k], refined_intrinsics[:k] = self._optimize(
                pts3d_tracks,
                refined_extrinsics[:k],
                refined_intrinsics[:k],
                frame[in_window],
                track[in_window],
                xy[in_window],
                score[in_window],
                images=images[:k],
                depth=depth[:k],
                confidence=confidence[:k],
            )

        # Back to (N, 4, 4) world-to-camera
        logger.info("BA refine done: losses %s", self.losses)
        return extrinsics_to_homogeneous(refined_extrinsics), refined_intrinsics

    def _optimize(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        frame: np.ndarray,
        track: np.ndarray,
        xy: np.ndarray,
        score: np.ndarray,
        images: np.ndarray,
        depth: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One LM solve over flat track observations, plus depth and photometric when on.

        - extrinsics in: (N, 3, 4) world-to-cam; frames the inlier gate drops keep their input poses
        - photometric: 3 image scales coarse to fine, (3, 2, 1) re-samples x 5 IRLS steps
        - early stop: lm_patience steps in a row below lm_tol end the solve, or the current re-sample
        - returns extrinsics (N, 3, 4) and K (N, 3, 3)
        """
        cfg = self.config
        N = len(extrinsics)

        # Work on copies
        refined_extrinsics = extrinsics.astype(np.float32)
        refined_intrinsics = intrinsics.astype(np.float32)

        # 1. Filter observations: visibility gate, reprojection filter, frame/landmark drops (upstream order)
        keep = _filter_observations(
            frame,
            track,
            xy,
            score,
            pts3d,
            extrinsics,
            intrinsics,
            vis_thresh=cfg.vis_thresh,
            max_reproj=cfg.max_reproj_error,
            min_inliers_per_frame=cfg.min_inliers_per_frame,
        )

        # Active keyframes and landmarks, and each kept observation's index into them
        frame_k, track_k = frame[keep], track[keep]
        active_frames = np.unique(frame_k)
        active_pts = np.unique(track_k)
        frame_idx = cast(np.ndarray, np.searchsorted(active_frames, frame_k))
        pt_idx = np.searchsorted(active_pts, track_k)
        obs_2d = xy[keep].astype(np.float64)

        # Fewer than 2 frames or points leaves the solve underdetermined
        if len(active_frames) < 2 or len(active_pts) < 2:
            raise ValueError(
                f"BA: too few active frames/points after filtering ({len(active_frames)} frames, "
                f"{len(active_pts)} points); need >= 2 of each"
            )

        # bae LM is CUDA-only (CuSparse spgemm); checked after the observation guard so it stays CPU-testable
        device = cfg.device or get_device()

        # Refuse a CPU device
        if "cuda" not in device:
            raise RuntimeError(
                f"Bundle adjustment requires a CUDA device (got {device!r}). "
                "The bae LM optimizer uses a CUDA-only sparse matmul (CuSparse); "
                "CPU bundle adjustment is not supported."
            )

        # Solve precision for every tensor below
        dtype = getattr(torch, cfg.dtype)

        # Active frames, points and observations against the input
        counts = (len(active_frames), N, len(active_pts), len(pts3d), len(frame_idx))
        logger.info("LM optimize: %d/%d frames, %d/%d points, %d observations", *counts)

        # 2. Observation tensors and depth weights: nearest feedforward depth, weight sqrt(conf) / (D * depth_sigma)
        target_z = np.zeros(len(obs_2d))
        weight_z = np.zeros(len(obs_2d))

        # Depth rows read the nearest depth pixel; zero weight when depth is off
        if cfg.use_depth:
            H, W = depth.shape[1:]
            col = np.clip(np.floor(obs_2d[:, 0]).astype(int), 0, W - 1)
            row = np.clip(np.floor(obs_2d[:, 1]).astype(int), 0, H - 1)
            obs_depth = depth[frame_k, row, col].astype(np.float64)
            obs_conf = confidence[frame_k, row, col] / np.median(confidence)
            valid = obs_depth > 0
            weight_z[valid] = np.sqrt(obs_conf[valid]) / (
                obs_depth[valid] * cfg.depth_sigma
            )
            target_z = obs_depth

        # Build SE3 camera tensor from (K, 3, 4) extrinsics; pad to (K, 4, 4) for mat2SE3
        ext_sub = extrinsics[active_frames].astype(np.float64)
        ext_4x4 = extrinsics_to_homogeneous(ext_sub)

        # Snap rotations to SO(3): float32 rotations from some backends fail pypose's mat2SE3 check
        ext_4x4[:, :3, :3] = project_to_so3(ext_4x4[:, :3, :3])
        ext_t = torch.tensor(ext_4x4, dtype=dtype, device=device)
        cameras_se3 = pp.mat2SE3(ext_t)

        # SIMPLE_PINHOLE: average fx/fy as single focal length per camera
        focal = (
            intrinsics[active_frames, 0, 0] + intrinsics[active_frames, 1, 1]
        ) / 2.0
        focal_tensor = torch.tensor(
            focal[:, None], dtype=dtype, device=device
        )  # (K, 1)
        principal_points = torch.tensor(
            intrinsics[active_frames, :2, 2], dtype=dtype, device=device
        )  # (K, 2)
        pts3d_tensor = torch.tensor(
            pts3d[active_pts], dtype=dtype, device=device
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

        # 3. Photometric pyramid, coarse to fine: (gray, valid-masked mean depth, K, re-samples)
        levels = []

        if cfg.use_photometric:
            rgb = torch.tensor(images[active_frames], dtype=dtype, device=device)
            gray = rgb.mean(1, keepdim=True)
            depth_t = torch.tensor(depth[active_frames], dtype=dtype, device=device)[
                :, None
            ]
            valid_t = (depth_t > 0).to(dtype)
            full_hw = np.array(gray.shape[-2:])

            # Per scale: pooled gray, valid-weighted depth and rescaled K
            for factor, n_relin in ((4, 3), (2, 2), (1, 1)):
                gray_l = F.avg_pool2d(gray, factor)[:, 0]
                valid_l = F.avg_pool2d(valid_t, factor)[:, 0]
                depth_l = F.avg_pool2d(depth_t * valid_t, factor)[:, 0] / valid_l.clamp(
                    min=1e-6
                )

                # Pooled pixel j covers full pixels [f j, f j + f): pixel-center K scales via the corner convention
                level_hw = full_hw / factor
                K_l = shift_intrinsics(intrinsics[active_frames], (0.5, 0.5))
                K_l = rescale_intrinsics(K_l, full_hw, level_hw)
                K_l = shift_intrinsics(K_l, (-0.5, -0.5))
                K_l = torch.tensor(K_l, dtype=dtype, device=device)
                levels.append((gray_l, depth_l, K_l, n_relin))

        # 4. Solver over the stacked residual
        with torch.enable_grad():
            model = _BAModel(
                cam_params, pts3d_tensor, shared_focal, refine_focal=cfg.refine_focal
            )

            # bae TrustRegion: non-positive predicted drop counts as rejected (pypose's does not)
            strategy = TrustRegion(up=2.0, down=0.5**4, max=1e6)

            # Matrix-free normal equations: J^T J never formed, PCG on the operator (Schur reads tol/maxiter off it)
            pcg = PCG(tol=1e-4, maxiter=250)

            # Schur takes the PCG as is; joint LM wants its step as a column
            if cfg.solver == "schur":
                optimizer_cls, solver = Schur, pcg
            else:
                optimizer_cls, solver = LM, lambda A, b: pcg(A, b).view(-1, 1)

            # Optimizer over the model; up to 10 rejected steps per step call
            optimizer = optimizer_cls(
                model,
                strategy=strategy,
                solver=solver,
                reject=10,
                matrix_free_normal=True,
            )

            # Pin target=None: pypose>=0.7 RobustModel.forward needs it; LM passes none, Schur passes it positionally
            forward = functools.partial(
                type(optimizer.model).forward, optimizer.model, target=None
            )
            optimizer.model.forward = lambda input, target=None: forward(input)

            # 5. LM / IRLS loop, manual: pypose StopOnPlateau stops on any rejected step
            loss_hist: list[float] = []
            params = [p for p in optimizer.model.parameters() if p.requires_grad]

            # Photometric draws so far; seeds each re-sample
            n_draws = 0

            # Without photometric one pass of up to lm_steps; with it, per scale and re-sample 5 IRLS steps
            for gray_l, depth_l, K_l, n_relin in levels or [(None, None, None, 1)]:
                # Hand torch's cached blocks back once per scale: warp allocates the Jacobian outside torch's pool
                torch.cuda.empty_cache()

                # One pass per re-sample; lm_steps without photometric, 5 with it
                for _ in range(n_relin):
                    n_steps = cfg.lm_steps

                    # Re-sample pixel brightness and image slope at the current poses and focal
                    if gray_l is not None:
                        pose_se3 = pp.SE3(model.pose.data[:, :7].detach())
                        w2c = pose_se3.matrix()
                        K_cur = K_l.clone()

                        # Scale K's focal by the live focal over the input one
                        K_cur[:, [0, 1], [0, 1]] *= model.focal() / focal_tensor

                        # One seed per re-sample, so every draw picks fresh pixels yet repeats run to run
                        with torch.no_grad():
                            samples = photometric_samples(
                                w2c, gray_l, depth_l, K_cur, seed=n_draws
                            )

                        n_draws += 1

                        if samples is None:
                            logger.warning(
                                "BA: too few photometric samples at %s, scale skipped",
                                gray_l.shape,
                            )
                            break

                        input_dict["photometric"] = samples
                        base_weight = samples["weight"].clone()
                        n_steps = 5

                    # Consecutive steps under lm_tol
                    stalled = 0

                    # LM steps on the current samples
                    for _ in range(n_steps):
                        # IRLS Huber on the photometric samples: delta = 1.5 x median normalized residual
                        if gray_l is not None:
                            assert samples is not None

                            with torch.no_grad():
                                samples["weight"] = base_weight
                                r = (
                                    model(**input_dict)[len(obs_2d) :, 0].tensor().abs()
                                    / base_weight
                                )
                                delta = 1.5 * r.median().clamp(min=1e-9)
                                samples["weight"] = (
                                    base_weight * (delta / r.clamp(min=delta)).sqrt()
                                )

                            # Loss at the current parameters, under the new weights
                            optimizer.loss = optimizer.model.loss(input_dict, None)

                        # One bae LM step; it keeps its last rejected step once the reject budget runs out
                        before = [p.data.clone() for p in params]
                        loss = optimizer.step(input=input_dict)
                        loss = float(loss)

                        # Undo a step that raised the loss
                        if loss > float(optimizer.last):
                            for p, b in zip(params, before):
                                p.data.copy_(b)

                            optimizer.loss = optimizer.last
                            loss = float(optimizer.last)
                            logger.warning(
                                "BA: LM step raised the loss after %d rejects, undone",
                                optimizer.reject_count,
                            )

                        # Relative drop against the loss before the step; zero when there is no prior loss
                        last = float(optimizer.last)
                        drop = (last - loss) / last if last > 0 else 0.0
                        loss_hist.append(loss)
                        logger.info(
                            "LM step %d: loss=%.6e drop=%.2e",
                            len(loss_hist),
                            loss,
                            drop,
                        )

                        # Stalled steps in a row end the solve, or this re-sample (the next one still runs)
                        stalled = stalled + 1 if drop < cfg.lm_tol else 0

                        if stalled >= cfg.lm_patience:
                            break

            # No LM step ran (lm_steps 0, or no photometric scale had samples): only the alignment below applies
            if not loss_hist:
                logger.warning(
                    "BA: no LM step ran; refine is a no-op apart from the alignment to input poses"
                )

            # Keep this solve's per-step losses
            self.loss_history.append(loss_hist)

            # 6. Losses, poses, alignment, focal: final loss per term, sliced from the stacked residual
            with torch.no_grad():
                final = model(**input_dict).tensor()

            # Track rows first, photometric rows after
            M = len(obs_2d)
            self.losses = {"reprojection": float(final[:M, :2].square().sum())}

            if cfg.use_depth:
                self.losses["depth"] = float(final[:M, 2].square().sum())

            if cfg.use_photometric:
                self.losses["photometric"] = float(final[M:, 0].square().sum())

        # Recover (3, 4) extrinsics from optimized SE3 quaternion representation
        opt_cam = to_numpy(model.pose.data).astype(np.float64)  # (K, 7) or (K, 8)
        opt_se3 = pp.SE3(torch.from_numpy(opt_cam[:, :7]))
        refined_extrinsics[active_frames] = opt_se3.matrix().numpy()[:, :3]

        # Move refined cameras back onto the input poses; scale fixed when depth sets it
        fix_scale = cfg.use_photometric or cfg.use_depth
        refined_extrinsics, self.alignment_scale = _align_to_input_poses(
            refined_extrinsics, extrinsics, active_frames, fix_scale=fix_scale
        )
        logger.info("BA: alignment to input poses, scale %s", self.alignment_scale)

        # Log frames the inlier gate dropped
        dropped = np.setdiff1d(np.arange(N), active_frames)

        # Warn only when the gate dropped a frame
        if len(dropped):
            logger.warning(
                "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s); kept at their input poses",
                len(dropped),
                N,
                cfg.min_inliers_per_frame,
                dropped.tolist(),
            )

        # Focal back to K: shared to every frame (dropped too), per-frame to active frames only
        focal_out = model.focal()[:, 0]
        focal_out = to_numpy(focal_out)

        if model.shared_intr is not None:
            refined_intrinsics[:, [0, 1], [0, 1]] = focal_out[0]
        else:
            refined_intrinsics[active_frames, 0, 0] = focal_out
            refined_intrinsics[active_frames, 1, 1] = focal_out

        return refined_extrinsics, refined_intrinsics


########################################################################
# BA model
########################################################################


@map_transform
def _reproject(
    pts: torch.Tensor,
    cam_params: torch.Tensor,
    principal_point: torch.Tensor,
    *shared_focal: torch.Tensor,
) -> torch.Tensor:
    """
    Pinhole projection plus camera z; returns [u, v, z].

    - @map_transform vectorizes it for the bae LM Jacobian; every argument has one row per observation
    - cam_params: SE3 (7), plus the per-camera focal (1) when no shared focal is passed
    """
    focal = shared_focal[0] if shared_focal else cam_params[..., 7:]
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

        # Trackable parameters; trim_SE3_grad: SE3 takes a 6-DoF step, a trailing focal steps as-is
        self.pose = nn.Parameter(TrackingTensor(cam_params))
        self.pose.trim_SE3_grad = True
        self.pts = nn.Parameter(TrackingTensor(pts_3d))

        # Shared focal: fixed buffer, trainable parameter, or none (focal in pose column 7)
        if shared_focal is not None and not refine_focal:
            self.register_buffer("shared_intr", shared_focal)
        elif shared_focal is not None:
            self.shared_intr = nn.Parameter(TrackingTensor(shared_focal))
        else:
            self.shared_intr = None

    def focal(self) -> torch.Tensor:
        """
        (K, 1) live focal per camera, detached: the shared one repeated, or pose column 7.
        """
        if self.shared_intr is None:
            return self.pose.data[:, 7:8]

        return self.shared_intr.data.expand(len(self.pose), 1)

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
        """
        # Track residuals: reprojection pixels and camera z against depth
        pts = self.pts[point_indices]
        cam = self.pose[camera_indices]
        ctr = principal_points[camera_indices]

        # Shared focal indexed per observation; per-frame focal rides in the pose
        shared_focal = (
            ()
            if self.shared_intr is None
            else (self.shared_intr[torch.zeros_like(camera_indices)],)
        )
        pred = _reproject(pts, cam, ctr, *shared_focal)

        # Weighted track residual block
        blocks = [(pred - target) * weight]

        # Photometric residuals: brightness of each sampled pixel against its match in the other frame
        if photometric is not None:
            pose_i = self.pose[photometric["i_idx"]]
            pose_j = self.pose[photometric["j_idx"]]
            residual = photometric_residual(
                pose_i, pose_j, *(photometric[k] for k in KEYS)
            )
            blocks.append(residual)

        return torch.cat(blocks)


########################################################################
# Helpers
########################################################################


def _filter_observations(
    frame: np.ndarray,
    track: np.ndarray,
    xy: np.ndarray,
    score: np.ndarray,
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
    (M,) observation mask for the BA solve; True enters the solve.

    - order: score gate, reprojection filter, frame min-inlier drop, landmark drop (upstream VGGT demo_colmap)
    - landmark drop: < 2 surviving views or out of range
    - max_reproj None skips the reprojection filter
    - reprojection: float64 on get_device(), batch_size observations at a time
    """
    # Score gate: keep observations the tracker is confident about
    keep = score > vis_thresh

    # Remove observations with high reprojection error under the current poses
    if max_reproj is not None:
        rows = np.flatnonzero(keep)
        device = get_device()
        points = torch.as_tensor(pts3d, dtype=torch.float64, device=device)
        world_to_cam = torch.as_tensor(extrinsics, dtype=torch.float64, device=device)
        K = torch.as_tensor(intrinsics, dtype=torch.float64, device=device)
        inliers = [np.zeros(0, bool)]

        # Reprojection error per batch against max_reproj
        for (batch,) in batch_iterator(batch_size, rows):
            f = torch.as_tensor(frame[batch], device=device)
            t = torch.as_tensor(track[batch], device=device)
            observed = torch.as_tensor(xy[batch], dtype=torch.float64, device=device)
            err = reprojection_error(points[t], world_to_cam[f], K[f], observed)
            ok = to_numpy(err <= max_reproj)
            inliers.append(ok)

        # Scatter batch inliers back into the score-gated rows
        keep[rows] = np.concatenate(inliers)

    # Drop frames with too few inliers first, so landmark counts see surviving frames only
    per_frame = np.bincount(frame[keep], minlength=len(extrinsics))
    keep &= per_frame[frame] >= min_inliers_per_frame

    # Drop points seen from fewer than 2 surviving frames and points outside valid world range
    per_track = np.bincount(track[keep], minlength=len(pts3d))
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    good = (per_track >= 2) & in_range
    keep &= good[track]
    return keep


def _align_to_input_poses(
    refined_extrinsics: np.ndarray,
    extrinsics: np.ndarray,
    active_frames: np.ndarray,
    *,
    fix_scale: bool,
) -> tuple[np.ndarray, float | None]:
    """
    Move the refined cameras, as one rigid body, back onto the input cameras.

    - BA pins no camera or scale, so the whole solution drifts; this undoes that
    - rotation: mean per-camera rotation change; positions alone miss a spin about a straight path
    - fix_scale: skip the scale fit; depth already set the scale
    - refined center spread (sum of squares) below 1e-12: no scale to fit, s = 1
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
    s = (
        1.0
        if fix_scale or spread < 1e-12
        else float((ref_dev * inp_dev).sum() / spread)
    )
    t_g = ctr_inp.mean(0) - s * R_g @ ctr_ref.mean(0)

    # Move the world by X' = s R_g X + t_g: [R|t] -> [R R_g^T | s t - R R_g^T t_g]
    R_new = R_ref @ R_g.T
    t_new = s * t_ref - R_new @ t_g
    out = refined_extrinsics.copy()
    out[active_frames, :, :3] = R_new
    out[active_frames, :, 3] = t_new
    return out, s
