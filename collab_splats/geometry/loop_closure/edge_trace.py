"""Level-2 loop-edge trace: compare our direct loop edges to VGGT-SLAM's composed chain.

SLAM inserts each loop as a 2-frame LC submap with 3 constraints
(query→LC0 scaled anchor, LC0→LC1 inner, LC1→detected scaled anchor);
ours is a single direct query→detected edge. Composing SLAM's chain yields
the equivalent direct relative — diffing the two quantifies the edge defects
(inversion, missing scale) per loop, per scene. Helpers only; a CLI over
per-run edge dumps lands with the fix work.
"""

from __future__ import annotations

import numpy as np


def compose_slam_chain(h_rel_a: np.ndarray, h_inner: np.ndarray, h_rel_b: np.ndarray) -> np.ndarray:
    """Equivalent direct query→detected relative from SLAM's 3-edge chain."""
    return h_rel_a @ h_inner @ h_rel_b


def edge_divergence(h_ref: np.ndarray, h_test: np.ndarray) -> dict:
    """Rotation (deg) + translation (norm) gap between two relative constraints."""
    d = np.linalg.inv(h_ref) @ h_test
    # rotation angle from the closest-rotation part of the 3x3 block
    u, _, vt = np.linalg.svd(d[:3, :3])
    r = u @ vt
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "rot_deg": float(np.degrees(np.arccos(cos))),
        "trans": float(np.linalg.norm(d[:3, 3])),
        "det_ratio": float(np.linalg.det(h_test) / np.linalg.det(h_ref)),  # scale proxy
    }
