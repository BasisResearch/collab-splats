"""Shared PyVista/colour helpers for the dashboard viewer."""

from __future__ import annotations

import matplotlib.cm as cm
import numpy as np

from collab_splats.utils.visualization import pointcloud_to_polydata

__all__ = ["pointcloud_to_polydata", "apply_viridis"]


def apply_viridis(sims: np.ndarray) -> np.ndarray:
    """Map similarity scores (P,) to viridis RGB uint8 (P, 3)."""
    s_min, s_max = sims.min(), sims.max()
    if s_max > s_min:
        normalized = (sims - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(sims)
    rgba = cm.viridis(normalized)
    return (rgba[:, :3] * 255).astype(np.uint8)
