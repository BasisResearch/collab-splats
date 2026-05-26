import numpy as np
import pytest
from collab_splats.utils.visualization import create_camera_frustum_pyvista


def test_frustum_apex_at_origin_for_identity_w2c():
    """Identity w2c → camera at world origin → apex at [0, 0, 0]."""
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    assert np.allclose(frustum.points[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_frustum_apex_at_camera_position():
    """Camera translated to [1, 2, 3] → apex at [1, 2, 3] in world space."""
    # w2c with camera at world [1, 2, 3]: R=I, t = -R @ p = [-1, -2, -3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 3] = [-1.0, -2.0, -3.0]
    frustum = create_camera_frustum_pyvista(w2c)
    assert np.allclose(frustum.points[0], [1.0, 2.0, 3.0], atol=1e-5)


def test_frustum_near_plane_at_positive_z_for_identity():
    """OpenCV convention: identity w2c → camera looks +Z → near/far at +Z."""
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    # Vertices 1-4: near plane; 5-8: far plane
    assert np.all(frustum.points[1:5, 2] > 0), "near plane z must be positive"
    assert np.all(frustum.points[5:9, 2] > 0), "far plane z must be positive"


def test_frustum_rectangles_closed():
    """Near and far plane rectangles must be closed (48 total line array entries)."""
    # Closed rects: near=[5,1,2,3,4,1] + far=[5,5,6,7,8,5] = 12 entries total
    # Unclosed rects: near=[4,1,2,3,4] + far=[4,5,6,7,8] = 10 entries total
    # Other lines (apex→near×4, apex→far×4, near→far×4) = 36 entries
    # Total closed = 48, unclosed = 46
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    assert len(frustum.lines) == 48, (
        f"Expected 48 line array entries (closed rects), got {len(frustum.lines)}"
    )
