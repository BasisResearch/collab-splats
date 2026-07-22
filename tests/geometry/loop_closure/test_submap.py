"""Fat Submap (ported from VGGT-SLAM) — dense points/colors/conf + world-frame reads."""

import numpy as np

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.submap import Submap


def _dense_submap(sid=0, S=2, H=4, W=5):
    rng = np.random.default_rng(sid)
    return Submap(
        submap_id=sid,
        frames=None,
        poses=np.tile(np.eye(4, dtype=np.float32), (S, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (S, 1, 1)),
        retrieval_vectors=rng.standard_normal((S, 8)).astype(np.float32),
        image_paths=[f"f{i}.jpg" for i in range(S)],
        points=rng.standard_normal((S, H, W, 3)).astype(np.float32),
        colors=(rng.random((S, H, W, 3)) * 255).astype(np.uint8),
        conf=rng.random((S, H, W)).astype(np.float32),
        frame_start=sid * S,
    )


def test_dense_fields_and_conf_threshold():
    s = _dense_submap()
    assert s.points.shape == (2, 4, 5, 3)
    assert s.colors.dtype == np.uint8
    assert s.conf_threshold is not None  # percentile-derived


def test_get_points_in_world_frame_conf_masked():
    s = _dense_submap()
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    pts = s.get_points_in_world_frame(pg)
    cols = s.get_points_colors()
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert cols.shape[0] == pts.shape[0]  # index-aligned points/colors
    assert pts.shape[0] <= 2 * 4 * 5  # conf mask may drop points


def test_get_all_poses_world_matches_extract_extrinsics():
    s = _dense_submap()
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    per_submap = s.get_all_poses_world(pg)  # (S,4,4)
    full = pg.extract_extrinsics(total_frames=s.points.shape[0])  # (S,4,4)
    np.testing.assert_allclose(per_submap, full, atol=1e-5)


def _rot_z(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.eye(4, dtype=np.float32)
    R[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return R


def _rotated_submap(sid=0, S=3, H=4, W=5):
    """Submap with non-identity per-frame rotations (R != R.T) + non-identity K."""
    rng = np.random.default_rng(sid)
    poses = np.stack([_rot_z(15.0 * (i + 1)) for i in range(S)]).astype(np.float32)
    for i in range(S):
        poses[i, :3, 3] = [0.05 * i, 0.02 * i, 0.1 * (sid * S + i)]
    K = np.array([[200.0, 0, 48], [0, 200.0, 32], [0, 0, 1]], dtype=np.float32)
    return Submap(
        submap_id=sid,
        frames=None,
        poses=poses,
        intrinsics=np.tile(K, (S, 1, 1)),
        retrieval_vectors=rng.standard_normal((S, 8)).astype(np.float32),
        image_paths=[f"f{i}.jpg" for i in range(S)],
        points=rng.standard_normal((S, H, W, 3)).astype(np.float32),
        colors=(rng.random((S, H, W, 3)) * 255).astype(np.uint8),
        conf=rng.random((S, H, W)).astype(np.float32),
        frame_start=sid * S,
    )


def test_get_all_poses_world_convention_nonidentity():
    """Non-identity rotation: get_all_poses_world must match extract_extrinsics (R.T convention)."""
    s = _rotated_submap()
    # Sanity: the fixture genuinely has R != R.T (else the test proves nothing).
    assert np.max(np.abs(s.poses[1, :3, :3] - s.poses[1, :3, :3].T)) > 0.1
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    per_submap = s.get_all_poses_world(pg)
    full = pg.extract_extrinsics(total_frames=s.points.shape[0])
    np.testing.assert_allclose(per_submap, full, atol=1e-5)
