"""Fat Submap (ported from VGGT-SLAM) — dense points/colors/conf + world-frame reads."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.submap import Submap
from tests.geometry.loop_closure._helpers import graph_extrinsics


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


def test_conf_percentile_sets_conf_threshold():
    """conf_threshold = percentile(conf, conf_percentile) + 1e-6, as VGGT-SLAM vggt_slam/submap.py:40."""
    conf = np.arange(1, 101, dtype=np.float32).reshape(1, 10, 10)
    kw = dict(
        submap_id=0,
        poses=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32)[None],
        retrieval_vectors=np.zeros((1, 8), dtype=np.float32),
        image_paths=["f0.jpg"],
        conf=conf,
    )
    assert Submap(**kw).conf_threshold == np.percentile(conf, 25.0) + 1e-6
    assert (
        Submap(**kw, conf_percentile=60.0).conf_threshold
        == np.percentile(conf, 60.0) + 1e-6
    )


def test_get_all_poses_world_matches_graph_extrinsics():
    s = _dense_submap()
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    per_submap = s.get_all_poses_world(pg)  # (S,4,4)
    full = graph_extrinsics(pg, s.points.shape[0])  # (S,4,4)
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
    """Non-identity rotation: get_all_poses_world must match graph_extrinsics (R.T convention)."""
    s = _rotated_submap()
    # Sanity: the fixture genuinely has R != R.T (else the test proves nothing).
    assert np.max(np.abs(s.poses[1, :3, :3] - s.poses[1, :3, :3].T)) > 0.1
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    per_submap = s.get_all_poses_world(pg)
    full = graph_extrinsics(pg, s.points.shape[0])
    np.testing.assert_allclose(per_submap, full, atol=1e-5)


def _numpy_world_grid(s, pg):
    """
    Per-frame float64 numpy homography of a submap's grid, the float64 numpy reference.
    """
    expected = np.empty(s.points.shape, dtype=np.float32)
    node_ids = pg.submap_node_ids(s.submap_id)

    for i in range(s.points.shape[0]):
        H_node = pg.get_homography(node_ids[i]).astype(np.float64)
        H = H_node @ s.poses[i].astype(np.float64)
        flat = s.points[i].reshape(-1, 3).astype(np.float64)
        hom = (H @ np.hstack([flat, np.ones((flat.shape[0], 1))]).T).T
        w = np.where(np.abs(hom[:, 3:4]) < 1e-10, 1e-10, hom[:, 3:4])
        expected[i] = (hom[:, :3] / w).reshape(s.points.shape[1:])

    return expected


def test_get_world_grid_matches_numpy_homography():
    """
    float32 world grid equals the per-frame float64 numpy homography to float32 rounding.
    """
    s = _dense_submap(S=3)
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()

    expected = _numpy_world_grid(s, pg)
    got = s.get_world_grid(pg, chunk=2)

    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 only exists on CUDA")
@pytest.mark.usefixtures("matmul_precision")
def test_get_world_grid_ignores_global_tf32():
    """
    A process-wide TF32 setting (mapanything enables it) neither rounds the world grid nor leaks.
    """
    s = _dense_submap(S=4, H=64, W=96)
    s.points = s.points * 10.0

    # Rotated, translated frames so the homographies are not identity
    for i in range(1, 4):
        c, n = np.cos(0.3 * i), np.sin(0.3 * i)
        s.poses[i, :3, :3] = [[c, -n, 0.0], [n, c, 0.0], [0.0, 0.0, 1.0]]
        s.poses[i, :3, 3] = [0.7 * i, -0.4 * i, 0.2 * i]

    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()

    # Reference under full fp32, then the tested run under TF32
    torch.set_float32_matmul_precision("highest")
    exact = s.get_world_grid(pg)
    torch.set_float32_matmul_precision("high")
    under_tf32 = s.get_world_grid(pg)
    assert torch.get_float32_matmul_precision() == "high"

    np.testing.assert_array_equal(under_tf32, exact)


def _moving_submaps(S=4, H=6, W=8):
    """
    Two overlapping submaps of a moving camera over a plane.

    - points in each submap's frame-0 camera
    """
    K = np.array([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]])
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1
    )

    # Global world-to-cam poses: translate along x, yaw a little
    G = []
    for j in range(2 * S - 1):
        g = _rot_z(3.0 * j).astype(np.float64)
        g[:3, 3] = [-0.2 * j, 0.05 * j, 0.0]
        G.append(g)

    submaps = []
    for sid, start in enumerate([0, S - 1]):
        # Submap-frame poses and plane points at depth 2 per camera, lifted into the frame-0 camera
        P = np.stack([G[start + i] @ np.linalg.inv(G[start]) for i in range(S)])
        cam = rays * 2.0
        pts = np.stack([(cam - P[i, :3, 3]) @ P[i, :3, :3] for i in range(S)])
        submaps.append(
            Submap(
                submap_id=sid,
                frames=None,
                poses=P.astype(np.float32),
                intrinsics=np.tile(K.astype(np.float32), (S, 1, 1)),
                retrieval_vectors=np.zeros((S, 8), dtype=np.float32),
                image_paths=[f"f{start + i}.jpg" for i in range(S)],
                points=pts.astype(np.float32),
                colors=np.zeros((S, H, W, 3), dtype=np.uint8),
                conf=np.ones((S, H, W), dtype=np.float32),
                frame_start=start,
            )
        )
    return submaps, K, np.stack([u, v], -1)


def test_world_grid_reprojects_under_extracted_extrinsics():
    """Each frame's world grid lands on its own pixels under graph_extrinsics, in every submap."""
    submaps, K, uv = _moving_submaps()
    pg = PoseGraph()
    for s in submaps:
        pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    ext = graph_extrinsics(pg, submaps[-1].frame_start + len(submaps[-1].poses))

    for s in submaps:
        grid = s.get_world_grid(pg).astype(np.float64)
        for i in range(len(s.poses)):
            E = ext[s.frame_start + i].astype(np.float64)
            p = (grid[i] @ E[:3, :3].T + E[:3, 3]) @ K.T
            np.testing.assert_allclose(p[..., :2] / p[..., 2:], uv, atol=1e-2)


def test_submap_node_ids_count_overlap_and_loop_carriers():
    """Node ids count overlap frames and loop carriers, so they run ahead of frame indices; returned as a copy."""
    submaps, _, _ = _moving_submaps()
    pg = PoseGraph()

    for s in submaps:
        pg.add_submap(s, overlap_frames=1)

    # Loop carrier on global frames 0 and 6, then a third window starting at frame 6
    first, second = submaps
    lc = replace(
        first,
        submap_id=2,
        poses=np.stack([first.poses[0], second.poses[3]]),
        intrinsics=first.intrinsics[:2],
        retrieval_vectors=None,
        frames=None,
        image_paths=["f0.jpg", "f6.jpg"],
        points=None,
        colors=None,
        conf=None,
    )
    pg.add_loop_edge(lc)
    third = replace(
        second,
        submap_id=3,
        frame_start=6,
        image_paths=[f"f{6 + i}.jpg" for i in range(4)],
    )
    pg.add_submap(third, overlap_frames=1)

    # Overlap frame and the two carrier nodes take ids, so node ids are not frame indices
    assert pg.submap_node_ids(1) == [4, 5, 6, 7]
    assert pg.submap_node_ids(3) == [10, 11, 12, 13]

    # Mutating the returned list leaves the graph untouched
    ids = pg.submap_node_ids(0)
    ids.append(99)
    assert pg.submap_node_ids(0) == [0, 1, 2, 3]
