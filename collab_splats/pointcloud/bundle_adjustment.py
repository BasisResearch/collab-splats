"""Bundle adjustment utilities for collab-splats.

Two public functions:
- extract_tracks_vggsfm: Predict cross-frame 2D tracks via VGGSfM tracker.
- run_bundle_adjustment: Run BAE LM bundle adjustment to refine poses and points.

Heavy dependencies (torch, pypose, bae, vggt) are imported lazily inside the
functions so this module can be imported even if those packages are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["BundleAdjustmentConfig", "extract_tracks_vggsfm", "run_bundle_adjustment"]


########################################################
########## Configuration ##############################
########################################################
@dataclass
class BundleAdjustmentConfig:
    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048      # track extraction: max query points
    query_frame_num: int = 5       # track extraction: number of query frames
    device: "str | None" = None    # target device; None = auto (CUDA if available, else CPU)


########################################################
########## Solver selection ############################
########################################################
def _get_default_solver(device: str | None = None) -> Any:
    """Return CuDSS when CUDA is requested and available, else PCG.

    Args:
        device: resolved device string; ``None`` auto-detects CUDA.
                ``"cpu"`` always returns PCG even when CUDA is present.
    """
    import torch
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in resolved:
        try:
            from bae.sparse.solve import CuDirectSparseSolver
            return CuDirectSparseSolver()
        except (ImportError, RuntimeError):
            pass
    from bae.utils.pysolvers import PCG
    return PCG()


########################################################
########## Track extraction ############################
########################################################
def extract_tracks_vggsfm(
    images: "torch.Tensor",  # (N, 3, H, W) float — normalised RGB
    conf: "torch.Tensor | None",  # (N, H, W) confidence scores
    world_points: "np.ndarray | None",  # (N, H, W, 3) for confidence-guided sampling
    *,
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
    fine_tracking: bool = False,
    device: "str | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict cross-frame 2D tracks via VGGSfM (ALIKED+SP keypoints).

    Args:
        device: Target device for the tracker (e.g. ``"cuda"``, ``"cpu"``).
                ``None`` auto-selects CUDA when available, CPU otherwise.
                Tensor inputs keep their existing device; only numpy inputs are
                moved to this device.

    Returns:
        tracks:     (N, P, 2) float32 — 2D pixel coords per frame per point.
        vis_scores: (N, P)    float32 — visibility score in [0, 1].
        pts3d:      (P, 3)    float32 — world-space 3D positions at keypoints.
    """
    import torch
    from vggt.dependency.track_predict import predict_tracks

    # Resolve target device: caller-supplied > auto-detect CUDA > CPU.
    # predict_tracks inherits device from images.device — it does NOT self-relocate.
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Accept either a torch.Tensor or a numpy array (the windowed LC path stores
    # merged images as numpy in raw_outputs["images"]).
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).to(target_device)

    device = images.device
    # VGGSfM tracker uses grid_sample which is not implemented for BFloat16 on CUDA.
    # VGGT-X runs in bfloat16; cast to float32 here so the tracker always gets fp32.
    images = images.float()

    # Prepare optional confidence tensor — predict_tracks expects (S, H, W)
    conf_tensor: "torch.Tensor | None" = None
    if conf is not None:
        conf_tensor = conf.to(device)
        # Ensure shape is (N, H, W), not (N, 1, H, W)
        if conf_tensor.ndim == 4 and conf_tensor.shape[1] == 1:
            conf_tensor = conf_tensor.squeeze(1)

    # Prepare optional world_points tensor — predict_tracks expects (S, H, W, 3)
    points_3d_tensor: "torch.Tensor | None" = None
    if world_points is not None:
        points_3d_tensor = torch.tensor(world_points, dtype=images.dtype, device=device)

    # VGGSfM track_predict asserts height == width when conf+world_points are both
    # provided.  VGGT-X outputs non-square feature maps (e.g. 350×518).  Pad the
    # short spatial dimension with zeros so existing pixel coords are unchanged.
    if conf_tensor is not None and points_3d_tensor is not None:
        import torch.nn.functional as F
        H_img, W_img = images.shape[-2], images.shape[-1]
        if H_img != W_img:
            size = max(H_img, W_img)
            pad_h, pad_w = size - H_img, size - W_img
            images = F.pad(images, (0, pad_w, 0, pad_h))
            conf_tensor = F.pad(conf_tensor, (0, pad_w, 0, pad_h))
            # points_3d: (N, H, W, 3) — last dim is channel, pad H and W
            points_3d_tensor = F.pad(points_3d_tensor, (0, 0, 0, pad_w, 0, pad_h))

    # VGGSfM's _forward_on_query builds pred_color as numpy then indexes it with
    # a CUDA valid_mask derived from conf — fails if conf is on GPU.  Move conf
    # and world_points to CPU; images can stay on device for the tracker itself.
    # Wrap in no_grad: VGGSfM tracker doesn't guard itself and pred_track retains
    # grad, causing .numpy() to raise "requires grad" error.
    with torch.no_grad():
        (
            pred_tracks,
            pred_vis_scores,
            _pred_confs,
            pred_points_3d,
            _pred_colors,
        ) = predict_tracks(
            images,
            conf=conf_tensor.cpu() if conf_tensor is not None else None,
            points_3d=points_3d_tensor.cpu() if points_3d_tensor is not None else None,
            max_query_pts=max_query_pts,
            query_frame_num=query_frame_num,
            fine_tracking=fine_tracking,
        )

    # predict_tracks already concatenates per-query-frame results internally and
    # returns np.ndarrays:
    #   pred_tracks:     (N, P, 2)
    #   pred_vis_scores: (N, P)
    #   pred_points_3d:  (P, 3)
    tracks = np.asarray(pred_tracks).astype(np.float32)
    vis_scores = np.asarray(pred_vis_scores).astype(np.float32)
    pts3d = np.asarray(pred_points_3d).astype(np.float32)

    return tracks, vis_scores, pts3d


########################################################
########## Bundle adjustment (LM) #####################
########################################################
def run_bundle_adjustment(
    points3d: np.ndarray,       # (P, 3)   initial 3D point positions
    extrinsics: np.ndarray,     # (N, 3, 4) world2cam [R|t]
    intrinsics: np.ndarray,     # (N, 3, 3) camera intrinsics K
    tracks: np.ndarray,         # (N, P, 2) 2D pixel observations
    vis_mask: np.ndarray,       # (N, P)    bool visibility mask
    image_size: tuple[int, int],  # (H, W)
    *,
    max_reproj_error: float | None = 4.0,
    lm_steps: int = 40,
    shared_camera: bool = False,
    min_inliers_per_frame: int = 64,
    solver=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BAE LM bundle adjustment to refine 3D points and camera poses.

    Returns:
        refined_points3d:   (P, 3)    float64 — refined world-space points.
        refined_extrinsics: (N, 3, 4) float32 — refined world-to-camera poses.
        refined_intrinsics: (N, 3, 3) float32 — refined intrinsics.
    """
    import torch
    import torch.nn as nn
    import pypose as pp
    from bae.autograd.function import TrackingTensor, map_transform
    from bae.optim import LM

    # rotate_quat: apply SE3 pose to 3D points (world → camera frame).
    # Not present in the installed bae version; implement via pypose SE3.Act.
    def rotate_quat(pts, pose):
        """Apply SE3 transform to pts. Equivalent to bae.utils.ba.rotate_quat."""
        return pose.Act(pts)
    from vggt.dependency.projection import project_3D_points_np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Work on copies so we don't mutate the caller's arrays.
    vis_mask = vis_mask.copy().astype(bool)
    refined_extrinsics = extrinsics.astype(np.float32).copy()
    refined_intrinsics = intrinsics.astype(np.float32).copy()
    refined_points3d = points3d.astype(np.float64).copy()

    ########################################################
    ########## Step 1: Filter by reprojection error ############
    ########################################################
    if max_reproj_error is not None:
        proj2d, proj_cam = project_3D_points_np(points3d, extrinsics, intrinsics)  # (N,P,2), (N,3,P)
        # Behind-camera points: reference (demo_colmap.py) sets projection to 1e6
        # so the reproj-error filter rejects them. Without this, ghost projections
        # of behind-camera points can pass a small-error check.
        behind = proj_cam[:, 2, :] <= 0   # (N, P)
        proj2d = proj2d.copy()
        proj2d[behind] = 1e6
        reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)               # (N, P)
        vis_mask[reproj_err > max_reproj_error] = False

    ########################################################
    ########## Step 2: Filter degenerate observations #####
    ########################################################
    valid_mask = vis_mask.sum(0) >= 2                              # (P,) — at least 2 views
    point_inrange = (np.abs(points3d) < 3000).all(axis=-1)        # (P,)
    vis_mask[:, ~(valid_mask & point_inrange)] = False
    # COLMAP/VGGSfM convention: frames with fewer than min_inliers_per_frame observations
    # are dropped; too few inliers make the per-frame pose update numerically unreliable.
    vis_mask[vis_mask.sum(1) < min_inliers_per_frame] = False

    ########################################################
    ########## Step 3: Build flat index tensors ###########
    ########################################################
    unique_keyframe = np.where(vis_mask.any(1))[0]   # (K,)
    unique_landmark = np.where(vis_mask.any(0))[0]   # (L,)

    if len(unique_keyframe) < 2 or len(unique_landmark) < 2:
        # Not enough observations — return inputs unchanged
        return refined_points3d, refined_extrinsics, refined_intrinsics

    frame_idx, point_idx = np.where(
        vis_mask[np.ix_(unique_keyframe, unique_landmark)]
    )  # indices into unique_keyframe / unique_landmark

    # Global frame / point indices for observation lookup
    global_frame_idx = unique_keyframe[frame_idx]
    global_point_idx = unique_landmark[point_idx]

    observations = tracks[global_frame_idx, global_point_idx].astype(np.float64)  # (M, 2)

    ########################################################
    ########## Step 4: Build camera params and points #####
    ########################################################
    ext_sub = extrinsics[unique_keyframe].astype(np.float64)   # (K, 3, 4)

    # Pad (3,4) → (4,4) so mat2SE3 can consume it
    ext_4x4 = np.concatenate(
        [ext_sub, np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float64), (len(unique_keyframe), 1, 1))],
        axis=1,
    )  # (K, 4, 4)

    cameras_se3 = pp.mat2SE3(
        torch.tensor(ext_4x4, dtype=torch.float64, device=device)
    )  # LieTensor (K,)

    # SIMPLE_PINHOLE model: single focal length — average fx/fy (loses aspect ratio for
    # non-square pixels, but cameras with square pixels are assumed here).
    f = (
        (intrinsics[unique_keyframe, 0, 0] + intrinsics[unique_keyframe, 1, 1]) / 2.0
    ).astype(np.float64)  # (K,)
    f_tensor = torch.tensor(f, dtype=torch.float64, device=device).unsqueeze(-1)  # (K, 1)

    center = torch.tensor(
        intrinsics[unique_keyframe, :2, 2].astype(np.float64),
        dtype=torch.float64,
        device=device,
    )  # (K, 2)

    points_tensor = torch.tensor(
        points3d[unique_landmark].astype(np.float64),
        dtype=torch.float64,
        device=device,
    )  # (L, 3)

    if shared_camera:
        f_mean = f_tensor.mean(0, keepdim=True)  # (1, 1)
        cameras_params = cameras_se3.data          # (K, 7)
        shared_intr = f_mean
    else:
        cameras_params = torch.cat([cameras_se3.data, f_tensor], dim=-1)  # (K, 8)
        shared_intr = None

    observations_t = torch.tensor(observations, dtype=torch.float64, device=device)  # (M, 2)
    camera_indices_t = torch.tensor(frame_idx, dtype=torch.long, device=device)      # (M,)
    point_indices_t = torch.tensor(point_idx, dtype=torch.long, device=device)       # (M,)

    ########################################################
    ########## Step 5: Define ReprojNonBatched module #####
    ########################################################

    # Define reprojection function without optional None args — bae's vmap cannot
    # handle None as a batched input, so we use two separate signatures.
    if shared_camera:
        @map_transform
        def reproject_simple_pinhole(pts, cam_params, ctr, shared_f):
            pts_cam = rotate_quat(pts, pp.SE3(cam_params[..., :7]))
            pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
            return pts_2d * shared_f + ctr
    else:
        @map_transform
        def reproject_simple_pinhole(pts, cam_params, ctr):
            pts_cam = rotate_quat(pts, pp.SE3(cam_params[..., :7]))
            pts_2d = pts_cam[..., :2] / pts_cam[..., 2].unsqueeze(-1)
            return pts_2d * cam_params[..., 7:] + ctr

    class ReprojNonBatched(nn.Module):
        def __init__(self, cam_params, pts_3d, shared_f=None):
            super().__init__()
            self.pose = nn.Parameter(TrackingTensor(cam_params))
            self.points_3d = nn.Parameter(TrackingTensor(pts_3d))
            self.pose.trim_SE3_grad = True
            self.shared_intr = (
                nn.Parameter(TrackingTensor(shared_f)) if shared_f is not None else None
            )

        def forward(self, points_2d, camera_indices, point_indices, ctr):
            cam = self.pose
            pts = self.points_3d
            if self.shared_intr is not None:
                shared_f = self.shared_intr[torch.zeros_like(camera_indices)]
                pts_proj = reproject_simple_pinhole(
                    pts[point_indices], cam[camera_indices], ctr[camera_indices], shared_f
                )
            else:
                pts_proj = reproject_simple_pinhole(
                    pts[point_indices], cam[camera_indices], ctr[camera_indices]
                )
            return pts_proj - points_2d

    ########################################################
    ########## Step 6: Optimise with LM ###################
    ########################################################
    with torch.enable_grad():
        model = ReprojNonBatched(cameras_params, points_tensor, shared_intr)

        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
        optimizer = LM(model, strategy=strategy, solver=solver if solver is not None else _get_default_solver(), reject=10)
        scheduler = pp.optim.scheduler.StopOnPlateau(
            optimizer, steps=lm_steps, patience=3, decreasing=1e-3, verbose=False
        )

        input_dict = {
            "points_2d": observations_t,
            "camera_indices": camera_indices_t,
            "point_indices": point_indices_t,
            "ctr": center,
        }
        scheduler.optimize(input=input_dict)

    ########################################################
    ########## Step 7: Extract results #####################
    ########################################################
    opt_cam = model.pose.data.detach().cpu().numpy()    # (K, 7) or (K, 8)
    opt_pts = model.points_3d.data.detach().cpu().numpy()  # (L, 3)

    # Recover (3, 4) extrinsics from SE3 quaternion representation
    opt_se3 = pp.SE3(torch.tensor(opt_cam[:, :7], dtype=torch.float64))
    opt_mat = opt_se3.matrix().numpy()  # (K, 4, 4)
    opt_ext_3x4 = opt_mat[:, :3, :]    # (K, 3, 4)

    refined_extrinsics[unique_keyframe] = opt_ext_3x4.astype(np.float32)
    refined_points3d[unique_landmark] = opt_pts.astype(np.float64)

    # Update focal lengths in intrinsics
    if shared_camera and model.shared_intr is not None:
        focal_val = float(model.shared_intr.data.detach().cpu().numpy().mean())
        refined_intrinsics[unique_keyframe, 0, 0] = focal_val
        refined_intrinsics[unique_keyframe, 1, 1] = focal_val
    elif not shared_camera:
        opt_focal = opt_cam[:, 7]  # (K,)
        refined_intrinsics[unique_keyframe, 0, 0] = opt_focal
        refined_intrinsics[unique_keyframe, 1, 1] = opt_focal

    return refined_points3d, refined_extrinsics, refined_intrinsics
