"""Tests for gopro_telemetry.py — GPMF stream loading and frame-aligned sampling.

Archived alongside the module under test; not collected by the default suite (`testpaths`
is `./tests`). Run explicitly: `pytest docs/benchmarks/scripts/ -p no:randomly`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

# This archive is outside the package and outside `testpaths`, so the module under test is
# not importable by name — put its directory on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gopro_telemetry import (
    GOPRO_FPS,
    ORIENTATION_MODES,
    load_gps_enu,
    load_image_orientation,
    reference_rotations_c2w,
    sample_reference_at_frames,
)


@pytest.fixture
def telemetry(tmp_path):
    """A synthetic GPMF table: orientation every sample, GPS every third (a slower stream)."""
    n = 120
    times = np.arange(n) / GOPRO_FPS
    rng = np.random.default_rng(0)
    cori = Rotation.from_rotvec(rng.normal(scale=0.3, size=(n, 3))).as_quat()[:, [3, 0, 1, 2]]
    iori = Rotation.from_rotvec(rng.normal(scale=0.2, size=(n, 3))).as_quat()[:, [3, 0, 1, 2]]
    df = pd.DataFrame({"source_time": times})
    for name, quat in (("cori", cori), ("iori", iori)):
        for k, axis in enumerate("wxyz"):
            df[f"{name}_{axis}"] = quat[:, k]
    # GPS arrives at a lower rate; absent samples are NaN, which dropna must remove.
    for col, value in (("gps_lat", 51.5), ("gps_lon", -0.12), ("gps_alt", 30.0)):
        df[col] = np.nan
        df.loc[::3, col] = value + np.arange(len(df.loc[::3])) * 1e-4
    path = tmp_path / "telemetry.parquet"
    df.to_parquet(path)
    return path


def test_load_image_orientation_rejects_an_unknown_mode(telemetry):
    with pytest.raises(ValueError, match="unknown orientation mode"):
        load_image_orientation(telemetry, mode="not_a_mode")


@pytest.mark.parametrize("mode", ORIENTATION_MODES)
def test_every_mode_returns_unit_quaternions(telemetry, mode):
    """Composing two streams drifts the norm; downstream rotation conversion needs unit."""
    times, quats = load_image_orientation(telemetry, mode)
    assert len(times) == len(quats) == 120
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)


def test_composition_order_matters(telemetry):
    """iori.cori and cori.iori are different rotations — quaternion product is not commutative.

    This is the distinction the whole EIS convention search rests on (they scored 0.288 vs
    1.624 deg on real data), so a swapped argument in _quat_multiply must not go unnoticed.
    """
    _, iori_cori = load_image_orientation(telemetry, "iori_cori")
    _, cori_iori = load_image_orientation(telemetry, "cori_iori")
    assert np.abs(iori_cori - cori_iori).max() > 1e-3


def test_conjugate_modes_differ_from_their_plain_counterparts(telemetry):
    _, plain = load_image_orientation(telemetry, "cori_iori")
    _, conj = load_image_orientation(telemetry, "cori_iori_conj")
    assert np.abs(plain - conj).max() > 1e-3


def test_load_gps_enu_drops_absent_samples_and_centres_the_origin(telemetry):
    times, enu = load_gps_enu(telemetry)
    # 40 of 120 rows carry GPS; the rest are NaN and must not survive.
    assert len(times) == 40 == len(enu)
    assert np.isfinite(enu).all()
    # East/north are referenced to the mean lat/lon, so each averages to zero; altitude is
    # referenced to the first sample instead.
    np.testing.assert_allclose(enu[:, :2].mean(axis=0), 0.0, atol=1e-6)
    assert enu[0, 2] == 0.0


def test_sample_reference_at_frames_aligns_by_frame_index(telemetry):
    """Frame index maps to telemetry time through the video rate — no interpolation of quats."""
    frame_idx = np.array([0, 30, 60])
    times, positions, quats = sample_reference_at_frames(telemetry, frame_idx, "cori")
    np.testing.assert_allclose(times, frame_idx / GOPRO_FPS)
    assert positions.shape == (3, 3) and quats.shape == (3, 4)
    # Nearest-sample selection must return the exact stored quaternion, not a blend of two.
    _, stored = load_image_orientation(telemetry, "cori")
    for row, idx in enumerate(frame_idx):
        np.testing.assert_allclose(quats[row], stored[idx], atol=1e-12)


def test_reference_rotations_are_transposed_relative_to_the_raw_quaternion(telemetry):
    """The GPMF quaternion is world-from-camera; c2w is its transpose (measured, see docstring)."""
    frame_idx = np.array([0, 10, 20])
    rot_c2w = reference_rotations_c2w(telemetry, frame_idx, "cori")
    _, _, quats = sample_reference_at_frames(telemetry, frame_idx, "cori")
    raw = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(rot_c2w, np.transpose(raw, (0, 2, 1)), atol=1e-12)
    # Guard against the transpose being a no-op on a symmetric fixture, which would make the
    # assertion above vacuous.
    assert np.abs(raw - np.transpose(raw, (0, 2, 1))).max() > 1e-3
