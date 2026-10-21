from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.geometry.loop_closure import graph as graph_mod
from collab_splats.geometry.loop_closure.graph import (
    PoseGraph,
    calculate_pairwise_frame_scale,
)
from collab_splats.geometry.loop_closure.submap import Submap
from tests.geometry.loop_closure._helpers import drive_pose_graph, graph_extrinsics

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


########################################
####### scale fits and H_w init #######
########################################


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
        points=rng.standard_normal((k, 8, P // 8, 3)).astype(np.float32),
        conf=np.full((k, 8, P // 8), 50.0, dtype=np.float32),
        frame_start=sid * k,
    )


@pytest.fixture
def two_submaps():
    return [_regular_submap(0, 4, 1), _regular_submap(1, 4, 2)]


def _lc_submap(sid: int, q_path: str, d_path: str) -> Submap:
    """2-frame loop-closure submap tying q_path→d_path."""
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    return Submap(
        submap_id=sid,
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        image_paths=[q_path, d_path],
        points=np.zeros((2, 8, 8, 3), dtype=np.float32),
        conf=np.full((2, 8, 8), 50.0, dtype=np.float32),
    )


def test_min_conf_points_reaches_sequential_and_both_anchor_scale_fits(
    two_submaps, monkeypatch
):
    """PoseGraph(min_conf_points=...) reaches the sequential scale fit and both loop-anchor fits."""
    seen = []
    real = graph_mod.calculate_pairwise_frame_scale

    def spy(*args):
        seen.append(args[-1])
        return real(*args)

    monkeypatch.setattr(graph_mod, "calculate_pairwise_frame_scale", spy)
    lc = _lc_submap(99, "s1_f1.jpg", "s0_f1.jpg")
    pg = PoseGraph(min_conf_points=10)
    for s in two_submaps:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()
    assert seen == [10]

    seen.clear()
    pg.add_loop_edge(lc)
    assert seen == [10, 10]


def _one_frame_submap(
    submap_id, pts, pose=None, conf=None, conf_threshold=None
) -> Submap:
    """1-frame submap with K = I; P points laid out as a (1, 1, P, 3) grid."""
    n = len(pts)
    pose = np.eye(4) if pose is None else pose
    return Submap(
        submap_id=submap_id,
        poses=pose[None].astype(np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        image_paths=[f"frame_{submap_id}.png"],
        points=np.asarray(pts, dtype=np.float32).reshape(1, 1, n, 3),
        conf=None
        if conf is None
        else np.asarray(conf, dtype=np.float32).reshape(1, 1, n),
        conf_threshold=conf_threshold,
    )


def _frame_scale(
    curr_pts,
    prior_pts,
    curr_pose=None,
    curr_conf=None,
    prior_conf=None,
    conf_threshold=None,
) -> float:
    """calculate_pairwise_frame_scale between two 1-frame submaps, min_conf_points 10."""
    curr = _one_frame_submap(1, curr_pts, curr_pose, curr_conf, conf_threshold)
    prior = _one_frame_submap(0, prior_pts, None, prior_conf, conf_threshold)
    return calculate_pairwise_frame_scale(curr, 0, prior, 0, min_conf_points=10)


def test_frame_scale_known_ratio():
    rng = np.random.default_rng(1)
    X = rng.random((50, 3))
    assert abs(_frame_scale(X, X * 2.5) - 2.5) < 0.01


def test_frame_scale_degenerate_source_falls_back_to_one():
    # Every source point at the origin: no valid ratio, scale 1.0
    assert _frame_scale(np.zeros((5, 3)), np.ones((5, 3))) == 1.0


def test_scale_estimation_survives_intersubmap_rotation():
    """Scale estimation moves each frame's points into its own camera first.

    The key failure mode: curr world origin is far from prev world origin.
    Skipping the camera move leaves curr_pts at their small curr-world magnitudes while
    prev_pts are large → scale wildly off.  Applying the full w2c pose
    (rotation + translation) first recovers the correct 1/true_scale ratio.

    Why 1/true_scale?  H_scale = diag([scale, scale, scale, 1]) is right-multiplied
    into H_w.  When curr world is at true_scale times prev world, the SL4 homography
    for the overlap frame satisfies P_prev = P_curr @ diag([true_scale,...,1]), so
    H_scale must be diag([1/true_scale,...,1]) — i.e. scale = 1/true_scale.
    """
    rng = np.random.default_rng(42)
    true_scale = 3.0

    # Canonical scene: small cluster of points near the origin
    N = 200
    X_scene = rng.standard_normal((N, 3)) * 0.1

    # Prev world: scene at [D, 0, 0], overlap-frame camera at the origin (P_prev_ov = I)
    D = 10.0
    X_prev = X_scene + np.array([D, 0.0, 0.0])
    P_prev_ov = np.eye(4, dtype=np.float64)

    # Curr world: same points as true_scale * R_w @ X_scene, near the curr origin (prev [D, 0, 0])
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()
    X_curr = true_scale * (R_w @ X_scene.T).T

    # Overlap-frame camera in curr world: true_scale * R_w @ (0 - scene_offset) = true_scale * R_w @ [-D, 0, 0]
    C_cam_curr = true_scale * (R_w @ np.array([-D, 0.0, 0.0]))
    R_cam_curr = R_w.T  # compensate for world rotation
    t_cam_curr = -R_cam_curr @ C_cam_curr
    P_curr_ov = np.eye(4, dtype=np.float64)
    P_curr_ov[:3, :3] = R_cam_curr
    P_curr_ov[:3, 3] = t_cam_curr

    # Without the camera move (identity curr pose) vs with the full w2c pose
    assert np.array_equal(P_prev_ov, np.eye(4))
    scale_old = _frame_scale(X_curr, X_prev)
    scale_new = _frame_scale(X_curr, X_prev, curr_pose=P_curr_ov)

    # With the camera move: scale = 1/true_scale (the H_scale correction for a 3× curr world)
    expected = 1.0 / true_scale
    assert abs(scale_new - expected) / expected < 0.05, (
        f"New scale {scale_new:.4f} far from expected {expected:.4f}"
    )
    # Without it: X_curr near origin (~0.3) vs X_prev far (~10), so scale_old >> expected
    assert scale_old > 5.0, (
        f"Unmoved points should give a large wrong scale, got {scale_old:.3f}"
    )


def test_confidence_masking_reduces_scale_noise():
    """Confidence filtering excludes noisy low-conf points from scale estimation.

    The scale is median(||prior[i]|| / ||curr[i]||) over the masked points.
    When curr world is world_scale× larger than prev world, the function
    returns ~1/world_scale (the SL4 correction factor).

    Setup: 20 good points (conf=50, ratio=0.5) + 80 noisy points (conf=5,
    ratio≈50, far from truth). Noisy majority corrupts the unmasked median
    but the joint mask (conf>25) selects only the 20 good points.
    """
    rng = np.random.default_rng(42)
    world_scale = 2.0
    expected_scale = 1.0 / world_scale  # 0.5

    N_good = 20
    N_noisy = 80  # majority — enough to shift unmasked median far from truth

    # Good points: prev far from origin, curr = world_scale × prev → ratio = 0.5
    X_prev_good = rng.standard_normal((N_good, 3)) + np.array([10.0, 0.0, 0.0])
    X_curr_in_prev_good = world_scale * X_prev_good  # ratio ||prev||/||curr|| = 0.5

    # Noisy points: large curr norms, small prev norms → ratio ≈ 50 (far from 0.5)
    X_prev_noisy = rng.standard_normal((N_noisy, 3)) * 0.01 + np.array([0.1, 0.0, 0.0])
    X_curr_in_prev_noisy = rng.standard_normal((N_noisy, 3)) * 5.0 + np.array(
        [10.0, 0.0, 0.0]
    )

    X_prev = np.vstack([X_prev_good, X_prev_noisy])
    X_curr_in_prev = np.vstack([X_curr_in_prev_good, X_curr_in_prev_noisy])

    # Confidence: good=50 (above threshold 25), noisy=5 (below threshold)
    conf_threshold = 25.0
    conf_prev = np.array([50.0] * N_good + [5.0] * N_noisy, dtype=np.float32)
    conf_curr = np.array([50.0] * N_good + [5.0] * N_noisy, dtype=np.float32)

    # Without conf: 80 noisy points dominate the median, pulling it away from 0.5
    scale_unmasked = _frame_scale(X_curr_in_prev, X_prev)

    # With conf on both sides: the joint tier (both > threshold) keeps only the 20 good points
    joint_mask = (conf_curr > conf_threshold) & (conf_prev > conf_threshold)
    assert joint_mask.sum() == N_good, (
        f"Should have exactly {N_good} good points in mask"
    )
    scale_masked = _frame_scale(
        X_curr_in_prev,
        X_prev,
        curr_conf=conf_curr,
        prior_conf=conf_prev,
        conf_threshold=conf_threshold,
    )

    err_masked = abs(scale_masked - expected_scale) / expected_scale
    err_unmasked = abs(scale_unmasked - expected_scale) / expected_scale

    assert err_masked < 0.05, (
        f"Masked scale {scale_masked:.3f} far from expected {expected_scale:.3f}"
    )
    assert err_unmasked > err_masked, (
        f"Masking should improve estimate: masked_err={err_masked:.3f}, "
        f"unmasked_err={err_unmasked:.3f}"
    )


def _make_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _make_submap(
    poses: np.ndarray, world_points: np.ndarray, submap_id: int = 0
) -> Submap:
    k = poses.shape[0]
    # Submap.points is a dense (K, H, W, 3) grid; lay any (K, P, 3) input out as H = 1
    pts = np.asarray(world_points, dtype=np.float32).reshape(k, 1, -1, 3)
    return Submap(
        submap_id=submap_id,
        frames=None,
        poses=poses.astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=np.zeros((k, 64), dtype=np.float32),
        image_paths=[f"frame_{i:04d}.png" for i in range(k)],
        frame_start=submap_id * k,
        points=pts,
    )


def test_hw_formula_integration_two_submaps_rotated():
    """Integration: 2-submap PGO with 30° rotation between world frames.
    New H_w correctly initializes the first frame of submap 2 with the rotation.
    Check: first frame of submap 2 in output has rotation close to R_w (not identity).
    """
    rng = np.random.default_rng(0)
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()
    k = 2

    # Prev submap: identity camera (overlap frame = last frame = identity)
    poses_prev = np.stack(
        [_make_w2c(np.eye(3), np.array([i * 0.1, 0.0, 0.0])) for i in range(k)]
    )
    # Curr submap: first frame (overlap) has 30° rotation in curr world
    poses_curr = np.stack(
        [_make_w2c(R_w, np.array([i * 0.1, 0.0, 0.0])) for i in range(k)]
    )

    wp_prev = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1
    wp_curr = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1

    prev_sub = _make_submap(poses_prev.astype(np.float32), wp_prev, submap_id=0)
    curr_sub = _make_submap(poses_curr.astype(np.float32), wp_curr, submap_id=1)

    total_frames = k + (k - 1)  # 3 with overlap=1
    result = drive_pose_graph(
        [prev_sub, curr_sub], lc_submaps=[], total_frames=total_frames, overlap_frames=1
    )

    assert result.shape == (total_frames, 4, 4)
    # First frame (reference) should be near identity
    assert np.allclose(result[0], np.eye(4), atol=0.15), (
        f"Frame 0 should be near identity, got\n{result[0]}"
    )
    # T-based H_w: R_w in H_opt and local_proj cancels to I; the old K-only H_w left R_w visible
    R_out = result[
        k, :3, :3
    ]  # unique frame of curr submap (starts at frame_start=submap_id*k=k)
    rot_vs_identity = np.linalg.norm(R_out - np.eye(3), "fro")
    assert rot_vs_identity < 0.3, (
        f"With correct H_w, rotation should cancel in extraction (got rot_vs_identity={rot_vs_identity:.3f})"
    )
