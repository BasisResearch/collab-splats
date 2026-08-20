import numpy as np

from collab_splats.dashboard.viz_utils import (
    PCD_KWARGS,
    apply_viridis,
    compute_view_transform,
    pointcloud_to_polydata,
)


def test_pcd_kwargs_uses_flat_points():
    """Spheres at sub-pixel point_size are invisible; flat GL points are far cheaper for 500k pts."""
    assert PCD_KWARGS.get("render_points_as_spheres") is False
    assert PCD_KWARGS.get("point_size") == 2.0
    # Base keys must survive the override spread (RGB scalar binding drives all recolors).
    assert PCD_KWARGS.get("scalars") == "RGB"
    assert PCD_KWARGS.get("rgb") is True


def test_polydata_has_points():
    pts = np.random.rand(10, 3).astype(np.float32)
    rgb = (np.random.rand(10, 3) * 255).astype(np.uint8)
    poly = pointcloud_to_polydata(pts, RGB=rgb)
    assert poly.n_points == 10


def test_apply_viridis_shape_and_dtype():
    sims = np.linspace(-1.0, 1.0, 12).astype(np.float32)
    rgb = apply_viridis(sims)
    assert rgb.shape == (12, 3)
    assert rgb.dtype == np.uint8


########################################################################
# compute_view_transform
########################################################################


def _apply(T, pts):
    """Apply a 4x4 homogeneous transform to (P, 3) points."""
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (h @ T.T)[:, :3]


def _bbox_center(pts):
    """Geometric midpoint of a point set's bounding box."""
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


def test_view_transform_centers_on_origin():
    # Inlier bbox center (flyers clipped by radius) maps to the origin.
    pts = np.random.rand(200, 3).astype(np.float32) + np.array([10.0, 5.0, -3.0])
    T = compute_view_transform(pts, extrinsics=None, percentile=95.0)
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    inliers = pts[d <= np.percentile(d, 95.0)]
    assert np.allclose(_apply(T, _bbox_center(inliers)[None]), 0.0, atol=1e-4)


def test_view_transform_scales_to_target_radius():
    # 100x-inflated cloud -> 95th-pct radius maps to target_radius regardless of input scale.
    pts = (np.random.rand(500, 3).astype(np.float32) - 0.5) * 100.0
    T = compute_view_transform(pts, extrinsics=None, target_radius=0.7, percentile=95.0)
    out = _apply(T, pts)
    # Radii are measured about the origin, because T maps its center onto it. Measuring about
    # _bbox_center(out) instead compares against a DIFFERENT center: the transform scales the
    # percentile radius about the inlier bbox midpoint (viz_utils.py:100), while the bbox of
    # `out` includes the clipped flyers. The gap between the two moves with the random draw,
    # which made this assertion fail for ~25% of unseeded clouds at atol 1e-3. Against the
    # right center the identity is exact, so the tolerance can be float-precision tight.
    r = np.percentile(np.linalg.norm(out, axis=1), 95.0)
    assert abs(r - 0.7) < 1e-6


def test_view_transform_aligns_mean_camera_up_to_plus_z():
    # Two identity-rotation w2c cams -> camera up in world = -Y. T must map -Y onto +Z.
    extr = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    pts = np.random.rand(50, 3).astype(np.float32)
    T = compute_view_transform(pts, extrinsics=extr)
    up_world = np.array([0.0, -1.0, 0.0])
    d = T[:3, :3] @ up_world
    d /= np.linalg.norm(d)
    assert np.allclose(d, [0.0, 0.0, 1.0], atol=1e-6)


def test_view_transform_degenerate_up_skips_rotation():
    # Opposite camera ups cancel to ~0 -> rotation part is identity (scale only, no flip).
    e0 = np.eye(4)
    e1 = np.eye(4)
    e1[1, :3] = [0.0, -1.0, 0.0]  # second cam up cancels the first
    extr = np.stack([e0, e1]).astype(np.float32)
    pts = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 4.0
    T = compute_view_transform(pts, extrinsics=extr)
    # Rotation must be a pure scaling of identity (no off-diagonal mixing).
    off_diag = T[:3, :3] - np.diag(np.diag(T[:3, :3]))
    assert np.allclose(off_diag, 0.0, atol=1e-6)


def test_view_transform_no_extrinsics_is_identity_rotation():
    pts = (np.random.rand(50, 3).astype(np.float32) - 0.5) * 4.0
    T = compute_view_transform(pts, extrinsics=None)
    off_diag = T[:3, :3] - np.diag(np.diag(T[:3, :3]))
    assert np.allclose(off_diag, 0.0, atol=1e-6)
