"""
NumPy/SciPy camera geometry shared across the pipeline.

- OpenCV camera axes: X right, Y down, Z forward (COLMAP, VGGT-X, BA)
- OpenGL camera axes: X right, Y up, Z backward (nerfstudio)
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.linalg import rq
from torch import Tensor

from collab_splats.utils.torch_utils import full_fp32_matmul

########################################################################
# Constants
########################################################################

# Camera axis convention flip (OpenCV <-> OpenGL)
# - diag(1, -1, -1, 1)
# - self-inverse: applying it twice returns the original
OPENGL_TO_OPENCV: np.ndarray = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

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
    single = extrinsics.ndim == 2  # (3,4) → (4,4)
    if single:
        extrinsics = extrinsics[np.newaxis]  # (1,3,4)
    n = extrinsics.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=extrinsics.dtype), (n, 1, 1))  # (N,1,4)
    out = np.concatenate([extrinsics, bottom], axis=1)  # (N,4,4)
    return out[0] if single else out


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


def transform_points(points: np.ndarray | Tensor, T: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    Points moved by a rigid transform: R @ p + t.

    - reads only the 3x4 block, so a (4, 4) bottom row must be [0, 0, 0, 1]
    - numpy or torch, not mixed; dtype follows the library's promotion of the inputs
    - torch: differentiable in both arguments
    - a (B, 4, 4) batch maps (P, 3) or (B, P, 3) points to (B, P, 3)
    - computes forward in full fp32 whatever the matmul precision: TF32 would round R @ p
    - TF32 is forced off only for torch inputs; the toggle waits while another thread holds hold_matmul_precision

    Args:
        points: (..., 3) points, or (P, 3) / (B, P, 3) with a pose batch.
        T: (4, 4) or (3, 4) rigid transform, or a (B, 4, 4) / (B, 3, 4) batch of them.

    Returns:
        (..., 3) transformed points; (B, P, 3) for a pose batch.
    """
    # numpy ignores torch's matmul precision, so skip the toggle
    if isinstance(points, np.ndarray):
        return _rigid_transform(points, T)

    # Full fp32 precision: TF32, which a mapanything import enables, rounds the matmul
    with full_fp32_matmul():
        return _rigid_transform(points, T)


def _rigid_transform(points: np.ndarray | Tensor, T: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    R @ p + t for one pose or a pose batch, at the caller's matmul precision.
    """
    # One pose: the original expression, bit-identical for existing callers
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
    out = np.tile(np.eye(4, dtype=K.dtype), K.shape[:-2] + (1, 1))
    out[..., :3, :3] = K
    return out


def rescale_intrinsics(K: np.ndarray, src_hw: np.ndarray, dst_hw: np.ndarray) -> np.ndarray:
    """
    Map K from one pixel grid to another of a different size, as an image resize does.

    - row 0 (fx, skew, cx) scales by dst_w / src_w, row 1 (fy, cy) by dst_h / src_h
    - pixel-corner convention (cx = W / 2 is the centre), as COLMAP, VGGT and MapAnything use
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
    # Output always takes K's shape; the hw leading dims must broadcast into it
    src = np.asarray(src_hw, dtype=np.float64)
    dst = np.asarray(dst_hw, dtype=np.float64)
    out = np.array(K, dtype=np.float64)
    for hw in (src, dst):
        if np.broadcast_shapes(hw.shape[:-1], out.shape[:-2]) != out.shape[:-2]:
            raise ValueError(f"hw leading dims {hw.shape[:-1]} do not broadcast into K leading dims {out.shape[:-2]}")

    # Positive-form check so a NaN size is rejected too: any comparison against NaN is False
    bad = ~(src > 0) | ~(dst > 0)
    if np.any(bad):
        raise ValueError(f"grid sizes must be positive, got src_hw {src.tolist()} dst_hw {dst.tolist()}")

    # Per-axis dst / src scale on the x and y rows
    scale = dst / src
    out[..., 0, :] *= scale[..., 1, None]
    out[..., 1, :] *= scale[..., 0, None]
    return out


def shift_intrinsics(K: np.ndarray, offset_xy: np.ndarray) -> np.ndarray:
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
    # Output always takes K's shape; the offset's leading dims must broadcast into it
    offset = np.asarray(offset_xy, dtype=np.float64)
    out = np.array(K, dtype=np.float64)
    if np.broadcast_shapes(offset.shape[:-1], out.shape[:-2]) != out.shape[:-2]:
        raise ValueError(
            f"offset leading dims {offset.shape[:-1]} do not broadcast into K leading dims {out.shape[:-2]}"
        )

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
    U, _, Vt = np.linalg.svd(R)
    U[..., :, -1] *= np.where(np.linalg.det(U @ Vt) < 0, -1.0, 1.0)[..., None]
    return U @ Vt


def decompose_camera(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    RQ-decompose a 3x4 or 4x4 projection matrix into (K, R, t, scale).

    - port of MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/slam_utils.py:45-83; no orthogonal snap, as upstream
    - upstream default (:76-78): R = inv(R_rq) camera-to-world, t = -R @ inv(K) @ P[:, 3] = center C
    - upstream ``no_inverse=True`` (:79-80): R = R_rq world-to-camera, t = inv(K) @ P[:, 3]
    - ours mixes them: default branch's R, ``no_inverse`` branch's t; C = -R @ t
    - callers store [R.T | t], a world-to-cam pose; see PoseGraph.extract_extrinsics
    - intrinsics ratio inv(K_a) @ K_b is I for a shared camera; RQ strips K from K @ inv(H)

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

    # ensure positive diagonal on K (per-column sign fix)
    if K[0, 0] < 0:
        K[:, 0] *= -1
        R[0, :] *= -1
    if K[1, 1] < 0:
        K[:, 1] *= -1
        R[1, :] *= -1
    if K[2, 2] < 0:
        K[:, 2] *= -1
        R[2, :] *= -1
    scale = float(K[2, 2])

    # Default branch's camera-to-world R, no_inverse branch's world-to-cam t; see the docstring
    R = np.linalg.inv(R)
    t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    return K, R, t, scale


########################################################################
# Intrinsics estimation
########################################################################


def _compute_weighted_median(values: np.ndarray, weights: np.ndarray, max_n: int = 50_000) -> float | None:
    """
    Confidence-weighted median, subsampled above `max_n` with a seeded RNG.

    - values must be finite; a non-finite entry skews the result without showing
    - prior art: github.com/PolyCam/LoGeR @ 5d7c1a7, `run_loger.py:167`
    - values, weights: (M,); weights non-negative
    - None for empty input or zero total weight
    """
    if len(values) == 0:
        return None

    # Cap the argsort over the pooled H*W*N population
    # - fixed seed keeps the subsample reproducible
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

    - for backends with no intrinsics head; a best-fit pinhole, not a recovered ground truth
    - per pixel fx = u_c * Z / X (fy likewise), pooled over all frames
    - one K for the batch (one camera, one resolution); fx and fy distinct, principal point centered
    - prior art: github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:180-192 (_focal_from_frame),
      :206-254 (estimate_focal_lengths); not done: its _snap_square_pixels (:195)
    - vendored fork differs: github.com/Junyi42/LoGeR @ 7685b7a, eval/relpose/launch.py:528-534
      (dust3r weiszfeld, one focal)
    - fit is approximate: ray field, github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:772-775

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
    if conf.ndim == 4:
        conf = conf.squeeze(-1)

    # Centered pixel grid: fixes the principal point, so return K, not (fx, fy)
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )

    # Invert the pinhole model per pixel: u_c = fx * X / Z  =>  fx = u_c * Z / X
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    valid = (z > 1e-3) & (np.abs(x) > 1e-6) & (np.abs(y) > 1e-6) & (conf > conf_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_per_pixel = uu * z / x
        fy_per_pixel = vv * z / y

    fx_vals, fy_vals = fx_per_pixel[valid], fy_per_pixel[valid]
    weights = conf[valid]

    # Bounds keep focals in a 157-6 degree FOV band; X or Y near 0 makes u_c * Z / X explode
    # - both sides load-bearing: upper drops +inf, lower drops -inf, NaN fails both
    ok_fx = (fx_vals > w * 0.1) & (fx_vals < w * 10)
    ok_fy = (fy_vals > h * 0.1) & (fy_vals < h * 10)
    fx = _compute_weighted_median(fx_vals[ok_fx], weights[ok_fx])
    fy = _compute_weighted_median(fy_vals[ok_fy], weights[ok_fy])

    # Fail loudly: a plausible-but-wrong fallback K fails silently downstream
    if fx is None or fy is None or not np.isfinite(fx) or not np.isfinite(fy) or fx <= 0 or fy <= 0:
        raise RuntimeError(
            f"Pinhole intrinsics fit failed over {n} frames: "
            f"{int(valid.sum())}/{valid.size} pixels passed the validity mask "
            f"({int((conf > conf_threshold).sum())} cleared conf_threshold={conf_threshold}), "
            f"of which {int(ok_fx.sum())} survived the fx bounds and {int(ok_fy.sum())} the fy bounds "
            f"(fx={fx}, fy={fy}). No fallback focal is applied by design."
        )

    # Keep fx and fy distinct: per-axis resizing yields non-square pixels
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0],
         [0.0, fy, (h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Geodesic rotation angle from the trace formula.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        Rotation angle in degrees, in [0, 180].
    """
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Rotation R with R @ src ≈ dst, by Rodrigues' formula.

    - inputs are normalized first
    - antiparallel inputs give a 180 degree turn about an arbitrary perpendicular axis

    Args:
        src: (3,) vector to rotate from.
        dst: (3,) vector to rotate to.

    Returns:
        (3, 3) rotation matrix; identity when src and dst are parallel.
    """
    # Normalize inputs to ensure unit vectors
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    # Compute rotation axis via cross product
    axis = np.cross(src, dst)
    axis_norm = np.linalg.norm(axis)

    # Parallel (identity) or antiparallel (180° rotation around arbitrary perp axis)
    if axis_norm < 1e-6:
        if np.dot(src, dst) > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)

        # Rodrigues for 180°: R = 2 * axis @ axis.T - I
        return -np.eye(3) + 2 * np.outer(axis, axis)

    # General case: Rodrigues' rotation formula
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(src, dst), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC floor plane, as the rigid transform that puts it at Z-up, z=0.

    - open3d segment_plane over the full cloud; largest inlier set is taken as the floor
    - no heuristic percentile

    Args:
        points: (N, 3) point cloud.

    Returns:
        R: (3, 3) rotation taking the floor normal onto [0, 0, 1].
        t: (3,) translation placing the floor at z=0 after that rotation.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    n_mag = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / n_mag
    d_norm = d / n_mag  # plane: normal · x + d_norm = 0; floor at z = -d_norm after rotation

    # Ensure normal points upward (positive Z component after alignment)
    if normal[2] < 0:
        normal = -normal
        d_norm = -d_norm

    R = rotation_align_vectors(normal, np.array([0.0, 0.0, 1.0]))
    # After R, floor is at z = -d_norm. Translate by d_norm to bring to z = 0.
    t = np.array([0.0, 0.0, d_norm])
    return R.astype(np.float64), t.astype(np.float64)


########################################################################
# Point-set alignment
########################################################################


def _umeyama_weights(source: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """
    Validated, normalized per-point weights shared by both Umeyama solvers.

    - source (M, 3); weights (M,) non-negative, None for uniform
    - returns (M,) float64 summing to 1
    - raises ValueError on fewer than 3 correspondences or zero total weight
    """
    # Reject too few points or zero weight, then normalize
    M = source.shape[0]
    if M < 3:
        raise ValueError(f"Umeyama alignment needs at least 3 correspondences, got {M}")
    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        raise ValueError("Umeyama alignment got zero total weight")
    return w / w_sum


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
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (4, 4) float32 homogeneous T such that target ≈ T @ source.

    Raises:
        ValueError: fewer than 3 correspondences, or zero total weight.
    """
    # Reject degenerate input; normalize the weights
    w = _umeyama_weights(source, weights)

    # Center both sets on their weighted means
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # Rotation from the SVD of the weighted cross-covariance; D forces det(R) = +1
    H = (src_c * w[:, None]).T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_tgt - R @ mu_src

    # Pack into a float32 homogeneous transform
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
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
        source: (M, 3) float32/64 points in source frame.
        target: (M, 3) float32/64 corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (s, R, t): float scale, (3, 3) float32 rotation, (3,) float32 translation
        such that target ≈ s * R @ source + t.

    Raises:
        ValueError: fewer than 3 correspondences, or zero total weight.
    """
    # Reject degenerate input; normalize the weights
    w = _umeyama_weights(source, weights)

    # Center both sets on their weighted means
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # Scale as the ratio of weighted RMS spreads
    scale_src = float(np.sqrt((w * (src_c**2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c**2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    # Rotation from the SVD of the weighted cross-covariance; D forces det(R) = +1
    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t
