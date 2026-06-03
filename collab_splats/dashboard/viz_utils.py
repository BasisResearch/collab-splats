"""Shared PyVista/colour helpers for the dashboard viewer."""

from __future__ import annotations

import matplotlib.cm as cm
import numpy as np
import pyvista as pv


def pointcloud_to_polydata(pts3d: np.ndarray, **point_data) -> pv.PolyData:
    """Convert pts3d + named scalar arrays to a PyVista PolyData.

    Args:
        pts3d: (P, 3) float32 world-space XYZ
        **point_data: named scalar arrays to attach as PyVista point arrays.
            e.g. RGB=colors, features=feat_arr, similarity=scores
    """
    cloud = pv.PolyData(pts3d.copy())
    for k, v in point_data.items():
        cloud[k] = v
    return cloud


def apply_viridis(sims: np.ndarray) -> np.ndarray:
    """Map similarity scores (P,) to viridis RGB uint8 (P, 3)."""
    s_min, s_max = sims.min(), sims.max()
    if s_max > s_min:
        normalized = (sims - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(sims)
    rgba = cm.viridis(normalized)
    return (rgba[:, :3] * 255).astype(np.uint8)
