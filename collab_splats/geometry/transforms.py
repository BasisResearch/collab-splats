"""Pure-numpy camera geometry utilities shared across the pipeline.

Conventions:
  OpenCV camera axes:  X right, Y down,  Z forward  (COLMAP, VGGT-X, BA)
  OpenGL camera axes:  X right, Y up,    Z backward  (nerfstudio, splats)
"""

from __future__ import annotations

import numpy as np

########################################################################
########## Constants ###################################################
########################################################################

# Camera axis convention flip (OpenCV ↔ OpenGL). Self-inverse: applying
# twice returns to original. diag(1, -1, -1, 1).
OPENGL_TO_OPENCV: np.ndarray = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

########################################################################
########## Geometry helpers ############################################
########################################################################


def extrinsics_to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    """Append [0,0,0,1] row to convert (N,3,4)→(N,4,4) or (3,4)→(4,4).

    Output dtype matches input dtype.
    """
    single = extrinsics.ndim == 2  # (3,4) → (4,4)
    if single:
        extrinsics = extrinsics[np.newaxis]  # (1,3,4)
    n = extrinsics.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=extrinsics.dtype), (n, 1, 1))  # (N,1,4)
    out = np.concatenate([extrinsics, bottom], axis=1)  # (N,4,4)
    return out[0] if single else out


def invert_poses(poses: np.ndarray) -> np.ndarray:
    """Closed-form SE3 inverse: (...,4,4) → (...,4,4).

    Works on any leading batch shape: (4,4), (N,4,4), (B,N,4,4).
    Uses R^T, -R^T@t — numerically exact for valid rotation matrices and
    faster than np.linalg.inv. Assumes poses are valid rigid-body transforms.
    """
    R = poses[..., :3, :3]
    t = poses[..., :3, 3:]
    R_inv = np.swapaxes(R, -1, -2)  # R^T
    t_inv = -(R_inv @ t)  # -R^T t
    out = np.zeros_like(poses)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:] = t_inv
    out[..., 3, 3] = 1.0
    return out


def extract_intrinsics(K: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from a (3,3) camera intrinsics matrix."""
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


########################################################################
########## Intrinsics estimation #######################################
########################################################################


def _compute_weighted_median(values: np.ndarray, weights: np.ndarray, max_n: int = 50_000) -> float | None:
    """Confidence-weighted median, subsampled above ``max_n`` with a seeded RNG.

    Textbook definition — sort by value, walk the cumulative weight, return the value
    at half the total mass.  Prior art for using one to reduce per-pixel focal
    estimates: github.com/PolyCam/LoGeR @ 5d7c1a7, ``run_loger.py:167``.  Returns
    ``None`` for an empty input, or for weights carrying no positive mass, so the
    caller can raise rather than invent a value.

    Values must be finite, and callers filter them before calling.  A non-finite entry
    does not poison the result visibly, it skews it: ``np.argsort`` sorts ``+inf`` and
    ``NaN`` to the tail (biasing the result upward) and ``-inf`` to the head (biasing
    it downward), so either way the return is a plausible finite number that a
    downstream ``np.isfinite`` check waves through.
    """
    if len(values) == 0:
        return None

    # A weighted median needs a full argsort, and the pooled per-pixel population is
    # H*W*N — 76.5M values at 300 frames, 255M at the 1000-frame sequences the LoGeR
    # backend exists for.  The cap bounds that; the fixed seed keeps it reproducible.
    if len(values) > max_n:
        idx = np.random.default_rng(42).choice(len(values), max_n, replace=False)
        values, weights = values[idx], weights[idx]

    # Sort by value, then walk the cumulative weight to the halfway mass.
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumw = np.cumsum(weights, dtype=np.float64)

    # All-zero weights carry no mass to bisect; searchsorted would return index 0 and
    # hand back the smallest value as if it were an estimate. Report "no estimate".
    if cumw[-1] <= 0:
        return None
    return float(values[np.searchsorted(cumw, cumw[-1] / 2.0)])


def rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix R such that R @ src ≈ dst.

    Args:
        src: (3,) unit vector to rotate from.
        dst: (3,) unit vector to rotate to.
    Returns:
        (3, 3) rotation matrix. Identity if src ≈ dst or antiparallel fallback.
    """
    # Normalize inputs to ensure unit vectors
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    # Compute rotation axis via cross product
    axis = np.cross(src, dst)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        # Parallel (identity) or antiparallel (180° rotation around arbitrary perp axis)
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
