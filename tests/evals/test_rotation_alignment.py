"""Tests for evals/rotation_alignment.py — the fit-free invariant and the camera-offset fit."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evals.rotation_alignment import (
    fit_camera_offset,
    invariant_turn_angle_error_deg,
    relative_rotation_error_deg,
    relative_rotations,
)


def _random_trajectory(n=40, seed=0):
    """A smooth-ish random rotation sequence standing in for a reconstruction."""
    rng = np.random.default_rng(seed)
    # Cumulative small rotations, so consecutive frames turn by a realistic few degrees.
    steps = Rotation.from_rotvec(rng.normal(scale=0.15, size=(n, 3)))
    out = [np.eye(3)]
    for step in steps[1:]:
        out.append(out[-1] @ step.as_matrix())
    return np.stack(out)


def test_relative_rotations_compose_back_to_the_sequence():
    traj = _random_trajectory(n=6)
    rel = relative_rotations(traj)
    # R[i] @ rel[i] must reproduce R[i+1]; this pins the operand order, which a transpose
    # slip would silently invert.
    np.testing.assert_allclose(traj[:-1] @ rel, traj[1:], atol=1e-12)


def test_invariant_is_exactly_zero_under_an_arbitrary_camera_offset():
    """The whole point: turn ANGLE survives conjugation, so a pure offset scores zero."""
    traj = _random_trajectory()
    offset = Rotation.from_rotvec([0.7, -1.3, 2.1]).as_matrix()
    conjugated = np.einsum("ij,njk,kl->nil", offset.T, traj, offset)
    err = invariant_turn_angle_error_deg(traj, conjugated)
    assert err.max() < 1e-9


def test_invariant_is_exactly_zero_under_a_world_frame_change():
    """A global left-multiplication is the other unknown; it must also cancel."""
    traj = _random_trajectory()
    world = Rotation.from_rotvec([-0.4, 0.9, 0.2]).as_matrix()
    err = invariant_turn_angle_error_deg(traj, np.einsum("ij,njk->nik", world, traj))
    assert err.max() < 1e-9


def test_invariant_detects_a_genuinely_different_turn():
    """It must not be vacuously zero — a scaled turn is a real disagreement."""
    traj = _random_trajectory()
    rel = Rotation.from_matrix(relative_rotations(traj))
    # Rebuild a trajectory whose steps turn 1.5x as far; angles now genuinely differ.
    scaled = [np.eye(3)]
    for rotvec in rel.as_rotvec():
        scaled.append(scaled[-1] @ Rotation.from_rotvec(1.5 * rotvec).as_matrix())
    err = invariant_turn_angle_error_deg(traj, np.stack(scaled))
    assert np.median(err) > 1.0


def test_fit_camera_offset_recovers_a_known_offset():
    traj = _random_trajectory()
    offset = Rotation.from_rotvec([0.3, -0.8, 1.7]).as_matrix()
    # est = ref conjugated by `offset`, which is the relation fit_camera_offset inverts.
    est = np.einsum("ij,njk,kl->nil", offset, traj, offset.T)
    recovered = fit_camera_offset(est, traj)
    residual = Rotation.from_matrix(recovered.T @ offset).magnitude()
    assert np.degrees(residual) < 0.5


def test_relative_rotation_error_is_near_zero_for_a_pure_offset():
    """A reconstruction differing only by camera convention must score ~0 after the fit."""
    traj = _random_trajectory()
    offset = Rotation.from_rotvec([1.1, 0.2, -0.6]).as_matrix()
    est = np.einsum("ij,njk,kl->nil", offset, traj, offset.T)
    angles, _ = relative_rotation_error_deg(est, traj)
    assert np.median(angles) < 0.1


def test_relative_rotation_error_never_falls_below_the_invariant_floor():
    """The self-check the CLI relies on: the fitted error cannot beat the fit-free bound."""
    est, ref = _random_trajectory(seed=1), _random_trajectory(seed=2)
    angles, _ = relative_rotation_error_deg(est, ref)
    floor = invariant_turn_angle_error_deg(est, ref)
    assert np.median(angles) >= np.median(floor) - 1e-6


def test_fit_camera_offset_rejects_mismatched_pose_counts():
    with pytest.raises(ValueError, match="pose count mismatch"):
        fit_camera_offset(_random_trajectory(n=10), _random_trajectory(n=9))
