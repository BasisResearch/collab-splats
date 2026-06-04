"""Shared PyVista/colour helpers for the dashboard viewer."""

from __future__ import annotations

import logging

import matplotlib.cm as cm
import numpy as np

from collab_splats.utils.visualization import (
    PCD_KWARGS,
    VIZ_KWARGS,
    pointcloud_to_polydata,
)

__all__ = [
    "pointcloud_to_polydata",
    "apply_viridis",
    "compute_view_transform",
    "PCD_KWARGS",
    "VIZ_KWARGS",
]

logger = logging.getLogger(__name__)


def apply_viridis(sims: np.ndarray) -> np.ndarray:
    """Map similarity scores (P,) to viridis RGB uint8 (P, 3)."""
    s_min, s_max = sims.min(), sims.max()
    if s_max > s_min:
        normalized = (sims - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(sims)
    rgba = cm.viridis(normalized)
    return (rgba[:, :3] * 255).astype(np.uint8)


########################################################################
# View normalization (display-only orientation + scale)
########################################################################


def _rotation_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping unit vector src onto unit vector dst (Rodrigues)."""
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    # Already aligned.
    if c > 1.0 - 1e-8:
        return np.eye(3)
    # Antiparallel: 180° about any axis perpendicular to src.
    if c < -1.0 + 1e-8:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    # General case.
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def compute_view_transform(
    points: np.ndarray,
    extrinsics: np.ndarray | None = None,
    target_radius: float = 0.7,
    percentile: float = 95.0,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Display-only similarity transform (4x4) that recenters, up-aligns and scales a scene.

    Centers on the bounding-box midpoint of the inlier points (percentile-clipped about the
    median, so flyers don't drag the center off — a plain median sits below the visual center
    for skewed clouds and renders the scene shifted up). Rotates the mean camera up-vector onto
    up_axis (viewer +Z by default) and scales the percentile-th radius to target_radius.
    Feedforward extrinsics are world-to-camera under OpenCV (Y-down) so camera up in world is
    -R_w2c[1]. Rotation is identity when no extrinsics are given or the mean up-vector is
    degenerate (e.g. a nadir orbit whose ups cancel). target_radius < 1 leaves margin so the
    fixed VIZ_KWARGS camera (distance 3, ~30° FOV) frames the scene without clipping. Does NOT
    mutate the scene — render-time only.
    """
    pts = np.asarray(points, dtype=np.float64)

    # Center on the inlier bbox midpoint: clip flyers by radius about the median, then take the
    # geometric mid of what's left -> the visible volume sits centered in the camera frame.
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    inliers = pts[d <= np.percentile(d, percentile)]
    if len(inliers) == 0:
        inliers = pts
    center = (inliers.min(axis=0) + inliers.max(axis=0)) / 2.0

    # Up-align: mean camera up in world = -R_w2c[1, :]; skip rotation if degenerate.
    R = np.eye(3)
    if extrinsics is not None and len(extrinsics) > 0:
        E = np.asarray(extrinsics, dtype=np.float64)
        mean_up = -E[:, 1, :3].mean(axis=0)
        if np.linalg.norm(mean_up) > 1e-6:
            R = _rotation_align(mean_up, np.asarray(up_axis, dtype=np.float64))
        else:
            logger.warning("view transform: degenerate mean camera up-vector; skipping rotation")

    # Scale: percentile radius about center -> target_radius (robust to flyer points).
    radii = np.linalg.norm(pts - center, axis=1)
    r = float(np.percentile(radii, percentile))
    s = target_radius / r if r > 1e-9 else 1.0

    # Compose homogeneous T: p' = s * R @ (p - center).
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = -s * (R @ center)
    return T
