"""CPU test for the finite-difference depth normal used by evals/scripts/analyze_splats.py."""

import numpy as np

from evals.scripts.analyze_splats import depth_to_normal


def test_depth_to_normal_planar_depth_matches_plane_normal():
    """A tilted plane's z-depth yields the plane normal (camera-facing) at every interior pixel."""
    height, width = 24, 32
    intrinsics = np.array([[40.0, 0.0, 16.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]])

    # Plane n . p = d with n tilted off the optical axis; z-depth solved per pixel from the ray
    plane_normal = np.array([0.2, -0.1, 1.0])
    plane_normal /= np.linalg.norm(plane_normal)
    plane_offset = 2.0
    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.stack([(cols - 16.0) / 40.0, (rows - 12.0) / 40.0, np.ones_like(cols)], axis=-1)
    depth = plane_offset / (rays @ plane_normal)

    normals = depth_to_normal(depth, intrinsics)

    # Camera-facing convention: the plane normal here points away (+z), so expect its negation
    expected = -plane_normal
    interior = normals[1:-1, 1:-1]
    assert np.allclose(interior, expected, atol=1e-3)
    assert np.all(normals[0] == 0) and np.all(normals[:, 0] == 0)
