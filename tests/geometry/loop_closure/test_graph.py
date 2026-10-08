from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.linalg import rq
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.geometry.loop_closure import graph as graph_mod
from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.transforms import decompose_camera
from tests.geometry.loop_closure._helpers import drive_pose_graph, graph_extrinsics


def test_decompose_camera_round_trip():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    R = ScipyR.from_euler("y", 15, degrees=True).as_matrix()
    t = np.array([0.1, -0.2, 0.5])
    P34 = K @ np.hstack([R, t[:, None]])  # (3, 4)
    K_out, R_out, t_out, scale = decompose_camera(P34)
    assert np.allclose(K_out[:3, :3] / K_out[0, 0], K / K[0, 0], atol=1e-6)
    assert np.allclose(np.abs(R_out), np.abs(R), atol=1e-5)
    assert np.allclose(t_out, t, atol=1e-5)


def test_decompose_camera_accepts_4x4():
    K = np.eye(3, dtype=np.float64) * 400.0
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3)
    P34 = K @ np.hstack([R, t[:, None]])
    P44 = np.vstack([P34, [0, 0, 0, 1]])
    K_out, R_out, t_out, scale = decompose_camera(P44)
    assert K_out.shape[0] == 3


def test_decompose_camera_keeps_a_reflection_like_upstream():
    """
    det<0 input returns det<0 R: upstream has no snap and no reflection fix.
    """
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    refl = ScipyR.from_euler("y", 15, degrees=True).as_matrix() @ np.diag(
        [1.0, 1.0, -1.0]
    )
    P34 = K @ np.hstack([refl, np.zeros((3, 1))])
    _, R_out, _, _ = decompose_camera(P34)
    assert np.linalg.det(R_out) < 0
    assert np.allclose(R_out, refl.T, atol=1e-10)


def test_decompose_camera_returns_the_unsnapped_rq_rotation_like_upstream():
    """
    R and t bit-equal upstream's arithmetic: inv of RQ's sign-fixed R, no U @ Vt snap.
    """
    # Upstream vggt_slam/slam_utils.py:45-83 arithmetic on a generic camera
    K = np.array([[512.3, 0.7, 301.9], [0, 498.1, 247.3], [0, 0, 1]], dtype=np.float64)
    R = ScipyR.from_euler("xyz", [23.0, -41.0, 67.0], degrees=True).as_matrix()
    P34 = 1.7 * K @ np.hstack([R, np.array([[0.3], [-1.2], [2.9]])])
    K_rq, R_rq = rq(P34[:, :3])
    signs = np.sign(np.diag(K_rq))
    K_rq, R_rq = K_rq * signs[None, :], R_rq * signs[:, None]

    # A snap moves R by ~1e-16, so only bit-equality can see it
    _, R_out, t_out, _ = decompose_camera(P34)
    assert np.array_equal(R_out, np.linalg.inv(R_rq))
    assert np.array_equal(t_out, np.linalg.inv(K_rq) @ P34[:, 3])


########################################
########## PoseGraph tests #############
########################################


def _identity_H() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _translate_H(tx: float, ty: float, tz: float) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [tx, ty, tz]
    return H


def _anchored_graph(Hs: list[np.ndarray]) -> PoseGraph:
    """PoseGraph over one submap whose frame nodes start at Hs, the first held at identity by the prior."""
    k = len(Hs)
    assert np.array_equal(Hs[0], np.eye(4))
    poses = np.linalg.inv(np.stack(Hs)).astype(np.float32)
    submap = Submap(
        submap_id=0,
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        image_paths=[Path(f"f{i}.png") for i in range(k)],
    )
    pg = PoseGraph()
    pg.add_submap(submap, overlap_frames=0)
    return pg


def test_sl4_add_homography_initializes():
    pg = PoseGraph()
    pg.add_homography(0, _identity_H())
    pg.add_homography(1, _translate_H(0.1, 0, 0))
    np.testing.assert_allclose(pg.get_homography(0), _identity_H(), atol=1e-9)
    np.testing.assert_allclose(pg.get_homography(1), _translate_H(0.1, 0, 0), atol=1e-9)


def test_sl4_sequential_edge_optimize():
    # Translations have det 1, already SL(4); inserts SL4-normalize regardless
    H0 = _identity_H()
    H1 = _translate_H(0.1, 0, 0)
    pg = _anchored_graph([H0, H1])
    np.testing.assert_allclose(pg.get_homography(1), H1, atol=1e-6)
    pg.optimize()
    H0_out = pg.get_homography(0)
    assert H0_out.shape == (4, 4)
    assert np.allclose(H0_out, H0, atol=0.05)


def test_sl4_loop_edge_no_crash():
    Hs = [_translate_H(i * 0.1, 0, 0) for i in range(3)]
    pg = _anchored_graph(Hs)
    # Loop-chain edges share the sequential-edge API and Gaussian noise
    pg.add_between_factor(2, 0, np.linalg.inv(Hs[2]) @ Hs[0])
    pg.optimize()  # must not raise
    for i in range(3):
        assert np.isfinite(pg.get_homography(i)).all()


def test_get_homography_post_optimize():
    pg = _anchored_graph([_identity_H()])
    pg.optimize()
    H_out = pg.get_homography(0)
    assert H_out.shape == (4, 4)
    assert np.isfinite(H_out).all()


########################################
####### incremental PoseGraph drive ####
########################################


def _make_real_submap(submap_id: int, k: int = 4, frame_start: int = 0) -> Submap:
    rng = np.random.default_rng(submap_id)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, 3] = (rng.standard_normal((k, 3)) * 0.05).astype(np.float32)
    intrinsics = np.tile(np.diag([400.0, 400.0, 1.0]).astype(np.float32), (k, 1, 1))
    points = rng.standard_normal((k, 4, 5, 3)).astype(np.float32)
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 8, 8),
        poses=poses,
        intrinsics=intrinsics,
        retrieval_vectors=torch.zeros(k, 32),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
        frame_start=frame_start,
        points=points,
    )


def _set_overlap_points(
    submap: Submap, points: np.ndarray, conf: np.ndarray, conf_threshold: float = 25.0
) -> None:
    """
    Give every frame of submap the same (N, 3) points and (N,) conf as a (1, N) dense grid.

    - conf_threshold is set raw, so the confidence tiers below read 50 / 1 / 0 against 25
    """
    k = submap.poses.shape[0]
    submap.points = np.tile(points.reshape(1, 1, -1, 3), (k, 1, 1, 1)).astype(
        np.float32
    )
    submap.conf = np.tile(conf.reshape(1, 1, -1), (k, 1, 1)).astype(np.float32)
    submap.conf_threshold = conf_threshold


def test_incremental_pose_graph_returns_correct_shape():
    k = 4
    submaps = [
        _make_real_submap(0, k=k, frame_start=0),
        _make_real_submap(1, k=k, frame_start=k),
    ]
    result = drive_pose_graph(
        submaps,
        lc_submaps=[],
        total_frames=k * 2,
        overlap_frames=1,
    )
    assert result.shape == (k * 2, 4, 4)
    assert np.isfinite(result).all()


########################################
####### error handling #################
########################################


def test_decompose_camera_rejects_bad_shape():
    with pytest.raises(ValueError, match="expected"):
        decompose_camera(np.ones((2, 4)))


def _optimizer_raising(exc):
    def build(*args, **kwargs):
        raise exc

    return build


def test_optimize_keeps_initial_values_on_gtsam_runtime_error(monkeypatch):
    pg = PoseGraph()
    H0 = _translate_H(0.3, 0, 0)
    pg.add_homography(0, H0)
    before = pg.get_homography(0)
    monkeypatch.setattr(
        graph_mod,
        "gtsam",
        SimpleNamespace(
            LevenbergMarquardtParams=lambda: None,
            LevenbergMarquardtOptimizer=_optimizer_raising(
                RuntimeError("indeterminant system")
            ),
        ),
    )
    pg.optimize()
    monkeypatch.undo()
    assert np.array_equal(pg.get_homography(0), before)


def test_optimize_propagates_non_gtsam_errors(monkeypatch):
    pg = PoseGraph()
    pg.add_homography(0, _identity_H())
    monkeypatch.setattr(
        graph_mod,
        "gtsam",
        SimpleNamespace(
            LevenbergMarquardtParams=lambda: None,
            LevenbergMarquardtOptimizer=_optimizer_raising(TypeError("bad argument")),
        ),
    )
    with pytest.raises(TypeError, match="bad argument"):
        pg.optimize()


########################################
####### confidence-mask floor ##########
########################################


# Confidence groups voting scales 2 / 4 / 8: A both confident, B prior-only, C low on both sides
N_A, N_B, N_C = 20, 60, 120


def _conf_group_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Curr/prior camera-local points plus confidences for the three groups.

    Returns:
        curr points, prior points (both (N, 3)), curr confidence, prior confidence (both (N,)).
    """
    rng = np.random.default_rng(7)
    n = N_A + N_B + N_C
    prior = rng.standard_normal((n, 3)) + np.array([0, 0, 5.0])
    scale = np.concatenate([np.full(N_A, 2.0), np.full(N_B, 4.0), np.full(N_C, 8.0)])
    curr = prior / scale[:, None]

    # Joint confidence on A only; prior confidence on A+B; C passes only a `> 0` test
    curr_conf = np.concatenate(
        [np.full(N_A, 50.0), np.zeros(N_B), np.ones(N_C)]
    ).astype(np.float32)
    prior_conf = np.concatenate([np.full(N_A + N_B, 50.0), np.ones(N_C)]).astype(
        np.float32
    )
    return curr, prior, curr_conf, prior_conf


def _submap_pair_with_conf_groups():
    """Two submaps whose single overlap frame carries the three confidence groups."""
    curr, prior, curr_conf, prior_conf = _conf_group_points()
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses[1, 0, 3] = 0.1  # a baseline inside s1, so its scale shows in frame 2
    _set_overlap_points(s0, prior, prior_conf)
    _set_overlap_points(s1, curr, curr_conf)
    return s0, s1


def _drive(pg, submaps):
    for s in submaps:
        pg.add_submap(s, 1)
        pg.optimize()
    return graph_extrinsics(pg, 3)


def test_min_conf_points_sets_the_confidence_mask_floor():
    """
    Each floor picks its own mask on the sequential edge.

    - 100 (default): joint 20 and prior 80 both fall short -> every point
    - 30: joint falls short, prior clears -> the prior-only mask
    - 10: joint clears -> the joint mask
    """
    s0, s1 = _submap_pair_with_conf_groups()
    default = _drive(PoseGraph(), [s0, s1])
    explicit = _drive(PoseGraph(min_conf_points=100), [s0, s1])
    prior_only = _drive(PoseGraph(min_conf_points=30), [s0, s1])
    joint = _drive(PoseGraph(min_conf_points=10), [s0, s1])
    assert np.array_equal(default, explicit)
    assert not np.allclose(default, prior_only)
    assert not np.allclose(prior_only, joint)
    assert not np.allclose(default, joint)


def test_sequential_edge_last_tier_drops_zero_confidence_points():
    """
    Joint and prior tiers both short -> prior > 0, not every point, on the sequential edge.

    - A (20): conf 50 on both sides, scale 2
    - C (60): prior conf 1, scale 4
    - Z (100): prior conf 0, scale 8; would win the median if kept
    """
    rng = np.random.default_rng(11)
    n_a, n_c, n_z = 20, 60, 100
    prior = rng.standard_normal((n_a + n_c + n_z, 3)) + np.array([0, 0, 5.0])
    scale = np.concatenate([np.full(n_a, 2.0), np.full(n_c, 4.0), np.full(n_z, 8.0)])
    curr = prior / scale[:, None]
    prior_conf = np.concatenate(
        [np.full(n_a, 50.0), np.ones(n_c), np.zeros(n_z)]
    ).astype(np.float32)
    curr_conf = np.concatenate([np.full(n_a, 50.0), np.ones(n_c + n_z)]).astype(
        np.float32
    )

    # Two identity-pose submaps sharing one overlap frame
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    _set_overlap_points(s0, prior, prior_conf)
    _set_overlap_points(s1, curr, curr_conf)

    # s1's frame-0 initial H_w = diag(s, s, s, 1), read before optimize; SL4 det-1 rescale, so s = [0, 0] / [3, 3]
    pg = PoseGraph(min_conf_points=100)
    for sm in (s0, s1):
        pg.add_submap(sm, 1)
    H_w = pg.get_homography(pg.submap_node_ids(1)[0])
    assert H_w[0, 0] / H_w[3, 3] == pytest.approx(4.0, rel=1e-5)


def test_sequential_edge_recovers_a_known_2x_scale():
    """
    Sequential edge scales s1 by the 2x ratio between the two submaps' overlap points.
    """
    rng = np.random.default_rng(12)
    prior = rng.standard_normal((50, 3)) + np.array([0, 0, 5.0])
    curr = prior / 2.0
    conf = np.full(50, 50.0, dtype=np.float32)

    # Two identity-pose submaps sharing one overlap frame, all points confident
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    _set_overlap_points(s0, prior, conf)
    _set_overlap_points(s1, curr, conf)

    # s1's frame-0 initial H_w = diag(s, s, s, 1), read before optimize; SL4 det-1 rescale, so ratio is s
    pg = PoseGraph()
    for sm in (s0, s1):
        pg.add_submap(sm, 1)
    H_w = pg.get_homography(pg.submap_node_ids(1)[0])
    assert H_w[0, 0] / H_w[3, 3] == pytest.approx(2.0, rel=1e-5)


def _sequential_scale(s0: Submap, s1: Submap, min_conf_points: int = 100) -> float:
    """
    Scale folded into s1's frame-0 initial value, read before any optimize.

    - shared K and identity s1 frame 0: H_w = H_prev_last @ diag(s, s, s, 1)
    - gtsam.SL4 rescales to det 1, so s is a ratio of the undone H_prev_last
    """
    pg = PoseGraph(min_conf_points=min_conf_points)
    for sm in (s0, s1):
        pg.add_submap(sm, 1)
    H_prev = pg.get_homography(pg.submap_node_ids(0)[-1])
    H_scale = np.linalg.inv(H_prev) @ pg.get_homography(pg.submap_node_ids(1)[0])
    return float(H_scale[0, 0] / H_scale[3, 3])


def test_sequential_edge_reads_the_prior_frame_in_the_overlap_camera():
    """
    Prior points move into the overlap frame's camera before the norm ratio.

    - s0's last camera sits 1 unit back along z; its points are c2w @ camera-local
    - back-transformed by that camera's w2c, the prior norms equal curr's: scale 1
    - VGGT-SLAM's frame-0 read (solver.py:142) would give median(|c2w @ x| / |x|), far from 1
    """
    rng = np.random.default_rng(13)
    cam = rng.standard_normal((200, 3)) + np.array([0, 0, 5.0])
    conf = np.full(200, 50.0, dtype=np.float32)

    # s0: last frame translated; its dense points in s0's frame-0 camera
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s0.poses[1, 2, 3] = 1.0
    c2w = np.linalg.inv(s0.poses[1].astype(np.float64))
    prior = (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]
    _set_overlap_points(s0, prior, conf)
    _set_overlap_points(s1, cam, conf)

    # Fixture sanity: the frame-0 read is visibly biased, so the assert can tell them apart
    upstream = np.median(np.linalg.norm(prior, axis=1) / np.linalg.norm(cam, axis=1))
    assert abs(upstream - 1.0) > 0.1
    assert _sequential_scale(s0, s1) == pytest.approx(1.0, rel=1e-5)


def test_sequential_edge_gates_by_the_prior_submaps_conf_percentile():
    """
    Confidence gate is the prior's 25th percentile of conf + 1e-6 (vggt_slam/submap.py:40), not raw 25.

    - L (250 pts): conf 0.2, scale 8; H (150 pts): conf 0.9, scale 2
    - percentile gate 0.2 + 1e-6 keeps H only -> scale 2
    - a raw 25 gate keeps nobody, falls through to prior > 0 -> the L majority, scale 8
    """
    rng = np.random.default_rng(14)
    n_l, n_h = 250, 150
    prior = rng.standard_normal((n_l + n_h, 3)) + np.array([0, 0, 5.0])
    scale = np.concatenate([np.full(n_l, 8.0), np.full(n_h, 2.0)])
    curr = prior / scale[:, None]
    conf = np.concatenate([np.full(n_l, 0.2), np.full(n_h, 0.9)]).astype(np.float32)

    # Two identity-pose submaps; conf_threshold derived from conf, as production
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    for sm, pts in ((s0, prior), (s1, curr)):
        grid = np.tile(pts.reshape(1, 1, -1, 3), (k, 1, 1, 1)).astype(np.float32)
        sm.set_dense_points(
            grid,
            np.zeros(grid.shape, dtype=np.uint8),
            np.tile(conf.reshape(1, 1, -1), (k, 1, 1)),
        )

    assert s0.conf_threshold == pytest.approx(0.2 + 1e-6)
    assert _sequential_scale(s0, s1) == pytest.approx(2.0, rel=1e-5)


def test_sequential_edge_without_dense_points_warns_and_uses_scale_1(caplog):
    """A submap with no dense points gets scale 1.0 and one warning."""
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.points = None
    with caplog.at_level(
        logging.WARNING, logger="collab_splats.geometry.loop_closure.graph"
    ):
        s = _sequential_scale(s0, s1)
    assert s == pytest.approx(1.0, rel=1e-6)
    assert (
        sum("no usable overlap points" in r.getMessage() for r in caplog.records) == 1
    )


def _single_frame_submap(sid: int, points: np.ndarray, conf: np.ndarray) -> Submap:
    """One-frame submap at the identity pose with identity intrinsics."""
    return Submap(
        submap_id=sid,
        frames=None,
        poses=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32)[None],
        retrieval_vectors=np.zeros((1, 8), dtype=np.float32),
        image_paths=[f"s{sid}_f0.jpg"],
        points=points.reshape(1, 1, -1, 3).astype(np.float32),
        conf=conf.reshape(1, 1, -1).astype(np.float32),
        conf_threshold=25.0,
    )


@pytest.mark.parametrize(
    "min_conf_points, expected", [(100, 8.0), (30, 4.0), (10, 2.0)]
)
def testcalculate_pairwise_frame_scale_confidence_floor_picks_the_mask(
    min_conf_points, expected
):
    """100: prior > 0 fallback (all); 30: prior-only mask (A+B); 10: joint mask (A)."""
    curr, prior, curr_conf, prior_conf = _conf_group_points()
    lc = _single_frame_submap(9, curr, curr_conf)
    reg = _single_frame_submap(0, prior, prior_conf)
    s = graph_mod.calculate_pairwise_frame_scale(lc, 0, reg, 0, min_conf_points)
    assert s == pytest.approx(expected, rel=1e-5)


def testcalculate_pairwise_frame_scale_gates_by_the_prior_submaps_conf_percentile():
    """
    Anchor gate is the prior's percentile threshold, not raw 25; groups as the sequential test.

    - L (250 pts): conf 0.2, scale 8; H (150 pts): conf 0.9, scale 2
    - percentile gate 0.2 + 1e-6 keeps H -> 2; raw 25 falls through to prior > 0 -> 8
    """
    rng = np.random.default_rng(15)
    n_l, n_h = 250, 150
    prior = rng.standard_normal((n_l + n_h, 3)) + np.array([0, 0, 5.0])
    scale = np.concatenate([np.full(n_l, 8.0), np.full(n_h, 2.0)])
    curr = prior / scale[:, None]
    conf = np.concatenate([np.full(n_l, 0.2), np.full(n_h, 0.9)]).astype(np.float32)

    # conf_threshold derived from conf on both single-frame submaps
    lc = _single_frame_submap(9, curr, conf)
    reg = _single_frame_submap(0, prior, conf)
    for sm in (lc, reg):
        sm.conf_threshold = None
        sm.__post_init__()
    assert reg.conf_threshold == pytest.approx(0.2 + 1e-6)
    assert graph_mod.calculate_pairwise_frame_scale(
        lc, 0, reg, 0, 100
    ) == pytest.approx(2.0, rel=1e-5)
