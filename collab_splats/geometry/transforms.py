"""
NumPy/SciPy camera geometry shared across the pipeline.

- pose algebra: extrinsics_to_homogeneous, invert_poses, transform_points, project_to_so3
- intrinsics: intrinsics_4x4, rescale_intrinsics, shift_intrinsics, decompose_camera
- fits: estimate_intrinsics_from_points, fit_dominant_plane
- point-set alignment: umeyama_se3, umeyama_sim3
- OpenCV camera axes throughout: X right, Y down, Z forward (COLMAP, VGGT-X, BA)
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import open3d as o3d
from numpy.typing import ArrayLike
from scipy.linalg import rq
from torch import Tensor

from collab_splats.utils.torch_utils import full_fp32_matmul

########################################################################
# Geometry helpers
########################################################################


def extrinsics_to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    """
    Append a [0, 0, 0, 1] row: (N, 3, 4) -> (N, 4, 4) or (3, 4) -> (4, 4).

    Args:
        extrinsics: (N, 3, 4) or (3, 4) pose matrices.

    Returns:
        (N, 4, 4) or (4, 4) homogeneous poses, same dtype as the input.
    """
    # Broadcast [0, 0, 0, 1] row stacked under the poses
    bottom = np.zeros((*extrinsics.shape[:-2], 1, 4), dtype=extrinsics.dtype)
    bottom[..., 3] = 1
    return np.concatenate([extrinsics, bottom], axis=-2)


def invert_poses(poses: np.ndarray) -> np.ndarray:
    """
    Closed-form SE(3) inverse via R^T and -R^T t.

    - a rotation's inverse is its transpose, so [R|t]^-1 = [R^T | -R^T t]
    - any leading batch shape: (4, 4), (N, 4, 4), (B, N, 4, 4)
    - assumes valid rigid-body transforms; no general matrix inverse

    Args:
        poses: (..., 4, 4) rigid-body transforms.

    Returns:
        (..., 4, 4) inverse transforms.
    """
    # Invert rotation and translation blocks, then reassemble the 4x4
    R = poses[..., :3, :3]
    t = poses[..., :3, 3:]
    R_inv = np.swapaxes(R, -1, -2)  # R^T
    t_inv = -(R_inv @ t)  # -R^T t
    out = np.zeros_like(poses)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:] = t_inv
    out[..., 3, 3] = 1.0
    return out


def transform_points(
    points: np.ndarray | Tensor, T: np.ndarray | Tensor
) -> np.ndarray | Tensor:
    """
    Points moved by a rigid transform: R @ p + t.

    - reads only the 3x4 block; a (B, 4, 4) batch maps (P, 3) or (B, P, 3) points to (B, P, 3)
    - numpy or torch, not mixed; torch is differentiable and runs in full fp32, as TF32 would round R @ p
    - the TF32 toggle waits while another thread holds hold_matmul_precision

    Args:
        points: (..., 3) points, or (P, 3) / (B, P, 3) with a pose batch.
        T: (4, 4) or (3, 4) rigid transform, or a (B, 4, 4) / (B, 3, 4) batch of them.

    Returns:
        (..., 3) transformed points; (B, P, 3) for a pose batch.
    """
    # Full fp32 for torch: TF32, which a mapanything import enables, rounds the matmul; numpy ignores it
    precision = nullcontext() if isinstance(points, np.ndarray) else full_fp32_matmul()

    with precision:
        # One pose: apply its 3x4 block directly
        if T.ndim == 2:
            return points @ T[:3, :3].T + T[:3, 3]

        # Pose batch: one matmul per pose, translation broadcast over the points
        return points @ T[..., :3, :3].swapaxes(-1, -2) + T[..., None, :3, 3]


def extract_intrinsics(K: np.ndarray) -> tuple[float, float, float, float]:
    """
    Focal lengths and principal point as Python floats.

    Args:
        K: (3, 3) camera intrinsics matrix.

    Returns:
        (fx, fy, cx, cy) in pixels.
    """
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def intrinsics_4x4(K: np.ndarray) -> np.ndarray:
    """
    Embed (..., 3, 3) K in the top-left of (..., 4, 4) identities.

    - output dtype follows K; cast K first to pick it

    Args:
        K: (..., 3, 3) intrinsics.

    Returns:
        (..., 4, 4) with K top-left and 1 at [3, 3].
    """
    # Identities tiled over K's batch dims, K written top-left
    out = np.tile(np.eye(4, dtype=K.dtype), K.shape[:-2] + (1, 1))
    out[..., :3, :3] = K
    return out


def rescale_intrinsics(
    K: ArrayLike, src_hw: ArrayLike, dst_hw: ArrayLike
) -> np.ndarray:
    """
    Map K from one pixel grid to another of a different size, as an image resize does.

    - row 0 (fx, skew, cx) scales by dst_w / src_w, row 1 (fy, cy) by dst_h / src_h
    - pixel-corner convention (cx = W / 2 is the center), as COLMAP and gsplat use
    - pixel-center K (as PointcloudResult stores): shift by +0.5, rescale, shift by -0.5
    - a crop is a separate shift_intrinsics call; K follows the image ops in the same order
    - output takes K's shape; the hw leading dims broadcast into K's: (2,) for any K, (N, 2) with (N, 3, 3)

    Args:
        K: (..., 3, 3) intrinsics on the source grid.
        src_hw: (..., 2) source grid (height, width).
        dst_hw: (..., 2) destination grid (height, width).

    Returns:
        (..., 3, 3) float64 K on the destination grid.

    Raises:
        ValueError: an hw's leading dims don't broadcast into K's, or a height or width is
            non-positive or NaN.
    """
    # Output always takes K's shape; a bad hw shape fails the in-place multiply below
    src = np.asarray(src_hw, dtype=np.float64)
    dst = np.asarray(dst_hw, dtype=np.float64)
    out = np.array(K, dtype=np.float64)

    # Positive-form check so a NaN size is rejected too: any comparison against NaN is False
    bad = ~(src > 0) | ~(dst > 0)

    if np.any(bad):
        raise ValueError(
            f"grid sizes must be positive, got src_hw {src.tolist()} dst_hw {dst.tolist()}"
        )

    # Per-axis dst / src scale on the x and y rows
    scale = dst / src
    out[..., 0, :] *= scale[..., 1, None]
    out[..., 1, :] *= scale[..., 0, None]
    return out


def shift_intrinsics(K: ArrayLike, offset_xy: ArrayLike) -> np.ndarray:
    """
    Move K's principal point by a pixel offset, as a crop or its undo does.

    - crop at top-left (tl_x, tl_y): shift by -tl; undo the crop: shift by +tl
    - focal lengths and skew are unchanged
    - output takes K's shape; offset_xy's leading dims broadcast into K's

    Args:
        K: (..., 3, 3) intrinsics.
        offset_xy: (..., 2) (dx, dy) added to (cx, cy).

    Returns:
        (..., 3, 3) float64 K with the shifted principal point.

    Raises:
        ValueError: offset_xy's leading dims don't broadcast into K's.
    """
    # Output always takes K's shape; a bad offset shape fails the in-place add below
    offset = np.asarray(offset_xy, dtype=np.float64)
    out = np.array(K, dtype=np.float64)
    out[..., 0, 2] += offset[..., 0]
    out[..., 1, 2] += offset[..., 1]
    return out


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """
    Nearest rotation to each 3x3 matrix, by SVD, with the reflection case flipped.

    - U @ Vt is orthogonal but may have det -1; negating U's last column fixes it
    - bit-identical to U @ Vt when det is already +1

    Args:
        R: (..., 3, 3) near-rotation matrices.

    Returns:
        (..., 3, 3) rotations, det +1.
    """
    # SVD projection, flipping U's last column wherever det(U @ Vt) is -1
    U, _, Vt = np.linalg.svd(R)
    U[..., :, -1] *= np.where(np.linalg.det(U @ Vt) < 0, -1.0, 1.0)[..., None]
    return U @ Vt


def decompose_camera(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    RQ-decompose a 3x4 or 4x4 projection matrix into (K, R, t, scale).

    - port of MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/slam_utils.py:45-83; no orthogonal snap
    - R is camera-to-world (upstream default), t is world-to-camera (upstream no_inverse); C = -R @ t
    - callers store [R.T | t] as a world-to-cam pose; see Submap.get_all_poses_world

    Args:
        P: 3x4 projection matrix, or 4x4 (divided by P[-1, -1], last row dropped).

    Returns:
        (K with K[2, 2] = 1, camera-to-world R, world-to-cam t, scale = K[2, 2] before normalizing).

    Raises:
        ValueError: P is not (3, 4) after the 4x4 strip.
    """
    # Normalize a 4x4 by P[-1, -1] and drop its last row
    P = np.array(P, dtype=np.float64)

    if P.shape[0] != 3:
        P = P / P[-1, -1]
        P = P[:3, :]

    if P.shape != (3, 4):
        raise ValueError(f"expected (3,4) after strip, got {P.shape}")

    # RQ-split the left 3x3 into upper-triangular K and rotation R
    M = P[:, :3]
    K, R = rq(M)

    # Positive diagonal on K: flip each negative column of K with the matching row of R
    for axis in range(3):
        if K[axis, axis] < 0:
            K[:, axis] *= -1
            R[axis, :] *= -1

    scale = float(K[2, 2])

    # Default branch's camera-to-world R, no_inverse branch's world-to-cam t; see the docstring
    R = np.linalg.inv(R)
    t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    return K, R, t, scale


########################################################################
# Intrinsics and floor-plane fits
########################################################################


def _compute_weighted_median(
    values: np.ndarray, weights: np.ndarray, max_n: int = 50_000
) -> float | None:
    """
    Confidence-weighted median, subsampled above `max_n` with a seeded RNG.

    - values must be finite; a non-finite entry skews the result without showing
    - prior art: github.com/PolyCam/LoGeR @ 5d7c1a7, `run_loger.py:167`
    - values, weights: (M,); weights non-negative
    - None for empty input or zero total weight
    """
    # Empty input has no median
    if len(values) == 0:
        return None

    # Cap the argsort over the pooled H*W*N population; fixed seed keeps the subsample reproducible
    if len(values) > max_n:
        idx = np.random.default_rng(42).choice(len(values), max_n, replace=False)
        values, weights = values[idx], weights[idx]

    # Sort by value, then walk the cumulative weight to the halfway mass
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumw = np.cumsum(weights, dtype=np.float64)

    # Zero total weight has no midpoint; report no estimate
    if cumw[-1] <= 0:
        return None

    return float(values[np.searchsorted(cumw, cumw[-1] / 2.0)])


def estimate_intrinsics_from_points(
    local_points: np.ndarray, conf: np.ndarray, conf_threshold: float = 0.1
) -> np.ndarray:
    """
    Fit one shared pinhole K to a camera-frame pointmap by confidence-weighted median.

    - for backends with no intrinsics head; per pixel fx = u_c * Z / X (fy likewise), pooled over all frames
    - one K per batch, fx and fy distinct, principal point centered
    - ported from PolyCam/LoGeR @ 5d7c1a7, run_loger.py:180-254, without its _snap_square_pixels (:195)

    Args:
        local_points: (N, H, W, 3) camera-frame points, channel 2 being depth.
        conf: (N, H, W) or (N, H, W, 1) confidence already activated into [0, 1], not logits.
        conf_threshold: minimum confidence for a pixel to contribute; lower admits more pixels.

    Returns:
        (3, 3) float32 K with principal point ((W - 1) / 2, (H - 1) / 2).

    Raises:
        RuntimeError: too few pixels survive to fit either focal; there is no fallback.
    """
    n, h, w, _ = local_points.shape

    # Drop a trailing singleton confidence channel
    if conf.ndim == 4:
        conf = conf.squeeze(-1)

    # Centered pixel grid: fixes the principal point, so return K, not (fx, fy)
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )

    # Invert the pinhole model per pixel: u_c = fx * X / Z  =>  fx = u_c * Z / X
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    valid = (
        (z > 1e-3) & (np.abs(x) > 1e-6) & (np.abs(y) > 1e-6) & (conf > conf_threshold)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        fx_per_pixel = uu * z / x
        fy_per_pixel = vv * z / y

    # Valid per-pixel focals and their confidence weights
    fx_vals, fy_vals = fx_per_pixel[valid], fy_per_pixel[valid]
    weights = conf[valid]

    # Focal bounds (157-6 degree FOV): upper drops +inf, lower drops -inf, NaN fails both
    ok_fx = (fx_vals > w * 0.1) & (fx_vals < w * 10)
    ok_fy = (fy_vals > h * 0.1) & (fy_vals < h * 10)
    fx = _compute_weighted_median(fx_vals[ok_fx], weights[ok_fx])
    fy = _compute_weighted_median(fy_vals[ok_fy], weights[ok_fy])

    # Fail loudly: a plausible-but-wrong fallback K fails silently downstream
    if (
        fx is None
        or fy is None
        or not np.isfinite(fx)
        or not np.isfinite(fy)
        or fx <= 0
        or fy <= 0
    ):
        raise RuntimeError(
            f"Pinhole intrinsics fit failed over {n} frames: "
            f"{int(valid.sum())}/{valid.size} pixels passed the validity mask "
            f"({int((conf > conf_threshold).sum())} cleared conf_threshold={conf_threshold}), "
            f"of which {int(ok_fx.sum())} survived the fx bounds and {int(ok_fy.sum())} the fy bounds "
            f"(fx={fx}, fy={fy}). No fallback focal is applied by design."
        )

    # Keep fx and fy distinct: per-axis resizing yields non-square pixels
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0], [0.0, fy, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC floor plane, as the rigid transform that puts it at Z-up, z=0.

    - open3d segment_plane over the full cloud; largest inlier set is taken as the floor

    Args:
        points: (N, 3) point cloud.

    Returns:
        R: (3, 3) rotation taking the floor normal onto [0, 0, 1].
        t: (3,) translation placing the floor at z=0 after that rotation.
    """
    # RANSAC plane normal · x + d_norm = 0, with a unit normal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    n_mag = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / n_mag
    d_norm = d / n_mag

    # Ensure normal points upward (positive Z component after alignment)
    if normal[2] < 0:
        normal = -normal
        d_norm = -d_norm

    # Rodrigues rotation of the normal onto +z; the flip above rules out the antiparallel case
    axis = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        R = np.eye(3)
    else:
        axis /= axis_norm
        angle = np.arccos(np.clip(normal[2], -1.0, 1.0))
        cross = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)

    # After R the floor sits at z = -d_norm; translate by d_norm to bring it to z = 0
    t = np.array([0.0, 0.0, d_norm], dtype=np.float64)
    return R, t


########################################################################
# Point-set alignment
########################################################################


def _umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    *,
    with_scale: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Umeyama core shared by both solvers: scale, float64 rotation and the two weighted means.

    - source, target (M, 3); weights (M,) non-negative, None for uniform
    - scale is 1.0 without with_scale; multiplying by 1.0 leaves the cross-covariance bit-identical
    - raises ValueError on fewer than 3 correspondences or zero total weight
    """
    # Reject too few points or zero weight, then normalize
    M = source.shape[0]

    if M < 3:
        raise ValueError(f"Umeyama alignment needs at least 3 correspondences, got {M}")

    w = (
        np.ones(M, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    w_sum = w.sum()

    if w_sum < 1e-9:
        raise ValueError("Umeyama alignment got zero total weight")

    w = w / w_sum

    # Center both sets on their weighted means
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # Scale as the ratio of weighted RMS spreads; coincident source points keep scale 1
    s = 1.0

    if with_scale:
        scale_src = float(np.sqrt((w * (src_c**2).sum(axis=1)).sum()))
        scale_tgt = float(np.sqrt((w * (tgt_c**2).sum(axis=1)).sum()))
        s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    # Rotation from the SVD of the weighted cross-covariance; D forces det(R) = +1
    H = (src_c * s * w[:, None]).T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    return s, R, mu_src, mu_tgt


def umeyama_se3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Closed-form rigid alignment of point sets by weighted SVD, without scale.

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: (M,) non-negative weights; uniform if None.

    Returns:
        (4, 4) float32 homogeneous T such that target ≈ T @ source.

    Raises:
        ValueError: fewer than 3 correspondences, or zero total weight.
    """
    # Rotation from the shared core, translation from the weighted means
    _, R, mu_src, mu_tgt = _umeyama(source, target, weights, with_scale=False)
    t = mu_tgt - R @ mu_src

    # Pack into a float32 homogeneous transform
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Closed-form similarity alignment of point sets by weighted Umeyama, with scale.

    - coincident source points give scale 1 by design: a stationary camera has one center

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: (M,) non-negative weights; uniform if None.

    Returns:
        (s, R, t): float scale, (3, 3) float32 rotation, (3,) float32 translation
        such that target ≈ s * R @ source + t.

    Raises:
        ValueError: fewer than 3 correspondences, or zero total weight.
    """
    # Translation from the float32 rotation, as callers apply it
    s, R, mu_src, mu_tgt = _umeyama(source, target, weights, with_scale=True)
    R = R.astype(np.float32)
    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t
