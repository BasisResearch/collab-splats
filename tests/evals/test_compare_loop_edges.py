"""Composed-chain loop-edge comparison (Level 2, spec §stage-trace item 3)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from collab_splats.geometry.loop_closure.edge_trace import (
    compose_slam_chain,
    edge_divergence,
)


def test_compose_slam_chain_matches_direct_relative():
    """Composing SLAM's 3-constraint chain q→LC0→LC1→d must equal the direct
    q→d relative in the graph's H_inner convention, when scales are identity."""
    rng = np.random.default_rng(7)

    def rand_se3():
        M = np.eye(4)
        M[:3, :3] = R.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        M[:3, 3] = rng.normal(size=3)
        return M

    P0, P1 = np.eye(4), rand_se3()  # LC-run poses (w2c), frame0 at origin
    h_rel_a = np.eye(4)  # query→LC0 anchor (identical image, scale=I)
    h_inner = P0 @ np.linalg.inv(P1)  # LC0→LC1 from the LC VGGT run
    h_rel_b = np.eye(4)  # LC1→detected anchor
    composed = compose_slam_chain(h_rel_a, h_inner, h_rel_b)
    np.testing.assert_allclose(composed, P0 @ np.linalg.inv(P1), atol=1e-12)


def test_edge_divergence_flags_inverted_edge():
    """A confirmed-wrong inverted-edge formula must read as far apart on both axes."""
    P1 = np.eye(4)
    P1[:3, :3] = R.from_euler("z", 30, degrees=True).as_matrix()
    P1[:3, 3] = [1.0, 0.0, 0.0]
    correct = np.eye(4) @ np.linalg.inv(P1)  # P0=I convention
    ours_current = np.linalg.inv(np.eye(4)) @ P1  # the confirmed-wrong formula
    d = edge_divergence(correct, ours_current)
    assert d["rot_deg"] > 10 and d["trans"] > 0.1  # far apart
    assert edge_divergence(correct, correct)["rot_deg"] < 1e-9


def test_edge_divergence_det_ratio_pins_scale_proxy():
    """det_ratio isolates the missing-scale defect: scaling translation by s cubes det."""
    P1 = np.eye(4)
    P1[:3, :3] = R.from_euler("z", 30, degrees=True).as_matrix()
    P1[:3, 3] = [1.0, 0.0, 0.0]
    reference = np.eye(4) @ np.linalg.inv(P1)
    scaled = reference.copy()
    s = 2.0
    scaled[:3, :3] *= s  # same rotation direction + translation, scale=s block
    assert edge_divergence(reference, scaled)["det_ratio"] == pytest.approx(8.0)
