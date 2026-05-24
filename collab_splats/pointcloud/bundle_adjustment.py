"""Bundle adjustment for collab-splats.

Public API:
- BundleAdjustmentConfig: configuration dataclass
- BundleAdjustment:       refines camera poses via VGGSfM tracks + LM bundle adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pypose as pp
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.projection import project_3D_points_np

if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult

__all__ = ["BundleAdjustment", "BundleAdjustmentConfig"]

# Sentinel for "caller did not pass this kwarg" — distinguishes explicit None (skip
# reprojection filter) from omitted (fall back to config default).
_UNSET = object()


########################################################
########## Configuration ##############################
########################################################


@dataclass
class BundleAdjustmentConfig:
    """Configuration for LM bundle adjustment."""

    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048       # track extraction: max query points
    query_frame_num: int = 5        # track extraction: number of query frames
    device: str | None = None       # target device; None = auto (CUDA if available, else CPU)


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

    def refine(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Refine camera poses; return updated FeedforwardResult with new extrinsics/intrinsics.

        pts3d, colors, and pixel_indices are unchanged — call creator.reproject(result)
        after to re-extract pts3d from refined poses.
        """
        cfg = self.config
        extrinsics_3x4 = result.extrinsics[:, :3, :]

        # Extract 2D tracks across all frames via VGGSfM
        tracks, vis_scores, pts3d_tracks = _extract_tracks_vggsfm(
            result.images, result.conf, result.world_points,
            max_query_pts=cfg.max_query_pts,
            query_frame_num=cfg.query_frame_num,
            device=cfg.device,
        )

        # Refine poses and intrinsics via LM bundle adjustment
        _, refined_extrinsics, refined_intrinsics = self._optimize(
            pts3d_tracks, extrinsics_3x4, result.intrinsics, tracks, vis_scores,
        )

        # Pad (N, 3, 4) extrinsics back to (N, 4, 4) for FeedforwardResult convention
        n = refined_extrinsics.shape[0]
        bottom_row = np.tile([[0, 0, 0, 1]], (n, 1, 1)).astype(np.float32)
        refined_extrinsics_4x4 = np.concatenate([refined_extrinsics, bottom_row], axis=1)
        return replace(result, extrinsics=refined_extrinsics_4x4, intrinsics=refined_intrinsics)

    def _optimize(
        self,
        pts3d: np.ndarray,          # (P, 3)    initial 3D keypoint positions
        extrinsics: np.ndarray,     # (N, 3, 4) world-to-camera poses
        intrinsics: np.ndarray,     # (N, 3, 3) camera intrinsics
        tracks: np.ndarray,         # (N, P, 2) 2D pixel observations from VGGSfM
        vis_scores: np.ndarray,     # (N, P)    visibility scores in [0, 1]
        *,
        max_reproj_error: float | None = _UNSET,  # override self.config if provided; None = skip filter
        lm_steps: int | None = None,              # override self.config if provided
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine 3D points and camera poses with Levenberg-Marquardt BA.

        Returns:
            refined_pts3d:        (P, 3)    float64
            refined_extrinsics:   (N, 3, 4) float32
            refined_intrinsics:   (N, 3, 3) float32
        """
        cfg = self.config
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # _UNSET means "caller didn't pass" — fall back to config; explicit None means "skip filter"
        max_reproj = cfg.max_reproj_error if max_reproj_error is _UNSET else max_reproj_error
        n_steps = lm_steps if lm_steps is not None else cfg.lm_steps

        # Work on copies; convert vis_scores (float) to bool visibility mask
        vis = vis_scores.astype(bool).copy()
        refined_extrinsics = extrinsics.astype(np.float32).copy()
        refined_intrinsics = intrinsics.astype(np.float32).copy()
        refined_pts3d = pts3d.astype(np.float64).copy()

        # Remove observations with high reprojection error under the current poses
        if max_reproj is not None:
            proj2d, proj_cam = project_3D_points_np(pts3d, extrinsics, intrinsics)
            # Behind-camera points get large sentinel projection so they fail the threshold
            behind = proj_cam[:, 2, :] <= 0
            proj2d = proj2d.copy()
            proj2d[behind] = 1e6
            reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)
            vis[reproj_err > max_reproj] = False

        # Drop points seen from fewer than 2 frames and points outside valid world range
        seen_enough = vis.sum(0) >= 2
        in_range = (np.abs(pts3d) < 3000).all(axis=-1)
        vis[:, ~(seen_enough & in_range)] = False

        # Drop under-constrained frames (too few inliers for reliable pose update)
        vis[vis.sum(1) < cfg.min_inliers_per_frame] = False

        # Build flat index arrays for active keyframes and landmarks
        active_frames = np.where(vis.any(1))[0]    # (K,)
        active_pts = np.where(vis.any(0))[0]       # (L,)

        if len(active_frames) < 2 or len(active_pts) < 2:
            return refined_pts3d, refined_extrinsics, refined_intrinsics

        frame_idx, pt_idx = np.where(vis[np.ix_(active_frames, active_pts)])
        global_frame_idx = active_frames[frame_idx]
        global_pt_idx = active_pts[pt_idx]
        obs_2d = tracks[global_frame_idx, global_pt_idx].astype(np.float64)   # (M, 2)

        # Build SE3 camera tensor from (K, 3, 4) extrinsics; pad to (K, 4, 4) for mat2SE3
        ext_sub = extrinsics[active_frames].astype(np.float64)
        ext_4x4 = np.concatenate(
            [ext_sub, np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float64), (len(active_frames), 1, 1))],
            axis=1,
        )
        cameras_se3 = pp.mat2SE3(torch.tensor(ext_4x4, dtype=torch.float64, device=device))

        # SIMPLE_PINHOLE: average fx/fy as single focal length per camera
        focal = (
            (intrinsics[active_frames, 0, 0] + intrinsics[active_frames, 1, 1]) / 2.0
        ).astype(np.float64)
        focal_tensor = torch.tensor(focal, dtype=torch.float64, device=device).unsqueeze(-1)   # (K, 1)
        principal_points = torch.tensor(
            intrinsics[active_frames, :2, 2].astype(np.float64),
            dtype=torch.float64, device=device,
        )  # (K, 2)
        pts3d_tensor = torch.tensor(
            pts3d[active_pts].astype(np.float64), dtype=torch.float64, device=device,
        )  # (L, 3)

        # Concatenate focal length into camera params tensor for per-camera case
        if cfg.shared_camera:
            cam_params = cameras_se3.data                             # (K, 7)
            shared_focal = focal_tensor.mean(0, keepdim=True)        # (1, 1)
        else:
            cam_params = torch.cat([cameras_se3.data, focal_tensor], dim=-1)   # (K, 8)
            shared_focal = None

        # Assemble observation index tensors
        obs_2d_t = torch.tensor(obs_2d, dtype=torch.float64, device=device)
        cam_idx = torch.tensor(frame_idx, dtype=torch.long, device=device)
        pt_idx_t = torch.tensor(pt_idx, dtype=torch.long, device=device)

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
            scheduler = pp.optim.scheduler.StopOnPlateau(
                optimizer, steps=n_steps, patience=3, decreasing=1e-3, verbose=False,
            )
            scheduler.optimize(input={
                "points_2d": obs_2d_t,
                "camera_indices": cam_idx,
                "point_indices": pt_idx_t,
                "principal_points": principal_points,
            })

        # Recover (3, 4) extrinsics from optimised SE3 quaternion representation
        opt_cam = model.pose.data.detach().cpu().numpy()     # (K, 7) or (K, 8)
        opt_pts = model.pts.data.detach().cpu().numpy()      # (L, 3)

        opt_se3 = pp.SE3(torch.tensor(opt_cam[:, :7], dtype=torch.float64))
        opt_extrinsics_3x4 = opt_se3.matrix().numpy()[:, :3, :]
        refined_extrinsics[active_frames] = opt_extrinsics_3x4.astype(np.float32)
        refined_pts3d[active_pts] = opt_pts.astype(np.float64)

        # Write optimised focal lengths back to intrinsics matrix
        if cfg.shared_camera and model.shared_intr is not None:
            focal_val = float(model.shared_intr.data.detach().cpu().numpy().mean())
            refined_intrinsics[active_frames, 0, 0] = focal_val
            refined_intrinsics[active_frames, 1, 1] = focal_val
        elif not cfg.shared_camera:
            opt_focal = opt_cam[:, 7]
            refined_intrinsics[active_frames, 0, 0] = opt_focal
            refined_intrinsics[active_frames, 1, 1] = opt_focal

        return refined_pts3d, refined_extrinsics, refined_intrinsics


########################################################
########## Helpers ####################################
########################################################


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
    from bae.utils.pysolvers import PCG
    return PCG()


def _extract_tracks_vggsfm(
    images: torch.Tensor,
    conf: torch.Tensor | None,
    world_points: np.ndarray | None,
    *,
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
    fine_tracking: bool = False,
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
        images = torch.from_numpy(images).to(target_device)

    # predict_tracks inherits device from images.device — it does NOT self-relocate
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
        cam_params: torch.Tensor,            # (K, 7) SE3 or (K, 8) SE3+focal
        pts_3d: torch.Tensor,                # (L, 3) 3D landmark positions
        shared_focal: torch.Tensor | None,   # (1, 1) if shared_camera, else None
        shared_camera: bool,
    ):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(cam_params))
        self.pts = nn.Parameter(TrackingTensor(pts_3d))
        self.pose.trim_SE3_grad = True
        self.shared_intr = (
            nn.Parameter(TrackingTensor(shared_focal)) if shared_focal is not None else None
        )
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
