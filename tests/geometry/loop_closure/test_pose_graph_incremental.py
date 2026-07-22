"""Incremental PoseGraph == batch per-submap cadence (parity lock)."""

import numpy as np
import pytest

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.submap import Submap
from tests.geometry.loop_closure._helpers import drive_pose_graph


def _regular_submap(sid: int, k: int, seed: int) -> Submap:
    """Deterministic submap: identity-ish poses + random dense points."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    # small forward translation per frame so inter-frame relatives are non-trivial
    for i in range(k):
        poses[i, :3, 3] = [0.0, 0.0, 0.1 * (sid * k + i)]
    P = 64
    return Submap(
        submap_id=sid,
        frames=np.zeros((k, 3, 4, 4), dtype=np.float32),
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=rng.standard_normal((k, 8)).astype(np.float32),
        image_paths=[f"s{sid}_f{i}.jpg" for i in range(k)],
        world_points=rng.standard_normal((k, P, 3)).astype(np.float32),
        world_points_conf=np.full((k, P), 50.0, dtype=np.float32),
        frame_start=sid * k,
    )


@pytest.fixture
def two_submaps():
    return [_regular_submap(0, 4, 1), _regular_submap(1, 4, 2)]


def test_monolith_golden_shape(two_submaps):
    out = drive_pose_graph(two_submaps, lc_submaps=[], total_frames=8, overlap_frames=1)
    assert out.shape == (8, 4, 4)
    # first node is identity (prior)
    np.testing.assert_allclose(out[0], np.eye(4), atol=1e-5)


def test_add_submap_matches_monolith_sequential(two_submaps):
    """Incremental add_submap + optimize per submap == monolith (no loops)."""
    golden = drive_pose_graph(two_submaps, lc_submaps=[], total_frames=8, overlap_frames=1)

    pg = PoseGraph()
    for s in two_submaps:
        pg.add_submap(s, overlap_frames=1, conf_threshold=25.0, scale_method="rotation_only")
        pg.optimize()
    incremental = pg.extract_extrinsics(total_frames=8)

    np.testing.assert_allclose(incremental, golden, atol=1e-6)


def _submap_nonident_K(sid: int, k: int, seed: int) -> Submap:
    """Submap with non-identity intrinsics + translated poses (exercises scale path)."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    for i in range(k):
        poses[i, :3, 3] = [0.05 * i, 0.0, 0.1 * (sid * k + i)]
    K = np.array([[200.0, 0, 48], [0, 200.0, 32], [0, 0, 1]], dtype=np.float32)
    P = 64
    return Submap(
        submap_id=sid,
        frames=None,
        poses=poses,
        intrinsics=np.tile(K, (k, 1, 1)),
        retrieval_vectors=rng.standard_normal((k, 8)).astype(np.float32),
        image_paths=[f"s{sid}_f{i}.jpg" for i in range(k)],
        world_points=rng.standard_normal((k, P, 3)).astype(np.float32),
        world_points_conf=np.full((k, P), 50.0, dtype=np.float32),
        frame_start=sid * k,
    )


@pytest.mark.parametrize("scale_method", ["se3", "rotation_only", "pairwise_dist"])
def test_incremental_matches_monolith_nonident_K(scale_method):
    """Non-identity K + translated poses exercise the inter-submap scale estimator,
    so incremental/monolith parity here actually covers scale_method regressions
    (with identity K, T=inv(K_prev)@K_curr=I and se3/rotation_only are indistinguishable)."""
    subs = [_submap_nonident_K(0, 4, 1), _submap_nonident_K(1, 4, 2)]
    golden = drive_pose_graph(subs, lc_submaps=[], total_frames=8, overlap_frames=1, scale_method=scale_method)
    pg = PoseGraph()
    for s in subs:
        pg.add_submap(s, overlap_frames=1, scale_method=scale_method)
        pg.optimize()
    incremental = pg.extract_extrinsics(total_frames=8)
    np.testing.assert_allclose(incremental, golden, atol=1e-6)


def _lc_submap(sid: int, q_path: str, d_path: str) -> Submap:
    """2-frame loop-closure submap tying q_path→d_path."""
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    return Submap(
        submap_id=sid,
        frames=np.zeros((2, 3, 4, 4), dtype=np.float32),
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        retrieval_vectors=np.zeros((2, 8), dtype=np.float32),
        image_paths=[q_path, d_path],
        is_lc_submap=True,
        world_points=np.zeros((2, 64, 3), dtype=np.float32),
        world_points_conf=np.full((2, 64), 50.0, dtype=np.float32),
    )


def test_add_loop_edge_matches_monolith(two_submaps):
    """Incremental build + deferred loop edges == monolith with the same loop."""
    lc = _lc_submap(99, "s1_f1.jpg", "s0_f1.jpg")
    golden = drive_pose_graph(two_submaps, lc_submaps=[lc], total_frames=8, overlap_frames=1)

    pg = PoseGraph()
    for s in two_submaps:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()
    pg.add_loop_edge(lc, self_submaps=two_submaps, conf_threshold=25.0, scale_method="rotation_only")
    pg.optimize()
    incremental = pg.extract_extrinsics(total_frames=8)

    np.testing.assert_allclose(incremental, golden, atol=1e-6)
