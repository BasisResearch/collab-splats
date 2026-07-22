"""GraphMap (ported from VGGT-SLAM) — submap collection + scene access."""

import numpy as np
import pytest

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.map import GraphMap
from collab_splats.geometry.loop_closure.submap import Submap


def _submap(sid, is_lc=False, k=2):
    return Submap(
        submap_id=sid,
        frames=None,
        poses=np.tile(np.eye(4, dtype=np.float32), (k, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=np.zeros((k, 8), dtype=np.float32),
        image_paths=[f"s{sid}_{i}.jpg" for i in range(k)],
        is_lc_submap=is_lc,
        frame_start=sid * k,
    )


def test_add_get_and_len():
    m = GraphMap()
    assert len(m) == 0
    m.add_submap(_submap(0))
    m.add_submap(_submap(1))
    assert len(m) == 2
    assert m.get_submap(1).submap_id == 1


def test_ordered_and_largest_key():
    m = GraphMap()
    for sid in (2, 0, 1):
        m.add_submap(_submap(sid))
    assert [s.submap_id for s in m.ordered_submaps_by_key()] == [0, 1, 2]
    assert m.get_largest_key() == 2


def test_largest_key_ignores_lc_submaps():
    m = GraphMap()
    m.add_submap(_submap(0))
    m.add_submap(_submap(1))
    m.add_submap(_submap(5, is_lc=True))  # loop-closure submap
    assert m.get_largest_key(ignore_loop_closure_submaps=True) == 1
    assert m.get_latest_submap(ignore_loop_closure_submaps=True).submap_id == 1


def test_get_largest_key_empty_is_none():
    assert GraphMap().get_largest_key() is None


def test_get_latest_submap_empty_raises():
    with pytest.raises(ValueError):
        GraphMap().get_latest_submap()


def test_get_latest_submap_all_lc_ignored_raises():
    m = GraphMap()
    m.add_submap(_submap(3, is_lc=True))
    with pytest.raises(ValueError):
        m.get_latest_submap(ignore_loop_closure_submaps=True)


def _dense(sid, S=2, H=3, W=4, is_lc=False):
    rng = np.random.default_rng(sid)
    poses = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    for i in range(S):
        poses[i, :3, 3] = [0.0, 0.0, 0.1 * (sid * S + i)]
    return Submap(
        submap_id=sid,
        frames=None,
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (S, 1, 1)),
        retrieval_vectors=np.zeros((S, 8), dtype=np.float32),
        image_paths=[f"s{sid}_{i}.jpg" for i in range(S)],
        points=rng.standard_normal((S, H, W, 3)).astype(np.float32),
        colors=(rng.random((S, H, W, 3)) * 255).astype(np.uint8),
        conf=rng.random((S, H, W)).astype(np.float32),
        is_lc_submap=is_lc,
        frame_start=sid * S,
    )


def test_get_world_pointcloud_concats_submap_reads():
    subs = [_dense(0), _dense(1)]
    pg = PoseGraph()
    for s in subs:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()
    m = GraphMap()
    for s in subs:
        m.add_submap(s)
    points, colors = m.get_world_pointcloud(pg)
    # expected: concat of each submap's own reads, in submap-id order
    exp_p = np.vstack([s.get_points_in_world_frame(pg) for s in subs])
    exp_c = np.vstack([s.get_points_colors() for s in subs])
    np.testing.assert_allclose(points, exp_p)
    np.testing.assert_array_equal(colors, exp_c)
    assert points.shape[0] == colors.shape[0]  # index-aligned


def test_get_world_pointcloud_overlap_dedups_leading_frames():
    """overlap drops each non-first submap's leading overlap frames (first-writer dedup)."""
    subs = [_dense(0), _dense(1)]
    pg = PoseGraph()
    for s in subs:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()
    m = GraphMap()
    for s in subs:
        m.add_submap(s)
    points, colors = m.get_world_pointcloud(pg, overlap=1)
    # Submap 0 keeps all frames; submap 1 drops its leading frame (owned by submap 0).
    exp_p = np.vstack([subs[0].get_points_in_world_frame(pg), subs[1].get_points_in_world_frame(pg, skip_first=1)])
    exp_c = np.vstack([subs[0].get_points_colors(), subs[1].get_points_colors(skip_first=1)])
    np.testing.assert_allclose(points, exp_p)
    np.testing.assert_array_equal(colors, exp_c)
    # Fewer points than the no-dedup concat (overlap frame removed).
    full, _ = m.get_world_pointcloud(pg)
    assert points.shape[0] < full.shape[0]


def test_get_world_pointcloud_skips_lc_submaps():
    subs = [_dense(0), _dense(1, is_lc=True)]  # submap 1 is a loop-closure submap
    pg = PoseGraph()
    pg.add_submap(subs[0], overlap_frames=1)
    pg.optimize()
    m = GraphMap()
    for s in subs:
        m.add_submap(s)
    points, _ = m.get_world_pointcloud(pg)
    exp_p = subs[0].get_points_in_world_frame(pg)  # only the non-LC submap
    np.testing.assert_allclose(points, exp_p)


def test_get_world_pointcloud_skips_points_none_submaps():
    """Submaps with points=None (MapAnything degraded path) contribute no cloud."""
    dense = _dense(0)
    degraded = _submap(1)  # no dense points/conf populated
    pg = PoseGraph()
    pg.add_submap(dense, overlap_frames=1)
    pg.optimize()
    m = GraphMap()
    m.add_submap(dense)
    m.add_submap(degraded)
    points, colors = m.get_world_pointcloud(pg)
    exp_p = dense.get_points_in_world_frame(pg)  # only the dense submap
    np.testing.assert_allclose(points, exp_p)
    assert points.shape[0] == colors.shape[0]


def test_get_corrected_extrinsics_matches_extract():
    subs = [_dense(0), _dense(1)]
    pg = PoseGraph()
    for s in subs:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()
    m = GraphMap()
    for s in subs:
        m.add_submap(s)
    N = sum(s.points.shape[0] for s in subs)
    np.testing.assert_allclose(m.get_corrected_extrinsics(pg, N), pg.extract_extrinsics(N), atol=1e-6)
