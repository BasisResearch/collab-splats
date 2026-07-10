"""Data-prep unit tests for the LC correction visualizer (no HTML rendering)."""

import sys
from pathlib import Path

import numpy as np

# `run_lc_parity.py`-style same-dir import: put evals/runners itself on sys.path
# (mirrors tests/evals/test_run_lc_parity.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))

from visualize_lc_correction import (
    apply_sim3,
    build_trajectory_figure,
    load_tum,
    loop_frame_indices,
    sim3_align,
)


def _write_tum(path: Path, n: int = 5) -> None:
    lines = ["# ts tx ty tz qx qy qz qw"]
    for i in range(n):
        lines.append(f"{float(i)} {i * 0.1} {i * 0.2} {i * 0.3} 0.0 0.0 0.0 1.0")
    path.write_text("\n".join(lines))


def test_load_tum_shapes_and_values(tmp_path):
    tum = tmp_path / "traj.tum"
    _write_tum(tum, n=5)
    ts, xyz, quat = load_tum(tum)
    assert ts.shape == (5,)
    assert xyz.shape == (5, 3)
    assert quat.shape == (5, 4)
    np.testing.assert_allclose(ts, np.arange(5, dtype=np.float64))
    np.testing.assert_allclose(xyz[3], [0.3, 0.6, 0.9])
    np.testing.assert_allclose(quat[:, 3], 1.0)  # identity qw column


def test_sim3_align_recovers_known_transform():
    # Ground truth = known Sim(3) applied to a random trajectory; alignment must undo it.
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    theta = 0.4
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    s, t = 2.5, np.array([1.0, -2.0, 0.5])
    target = apply_sim3(source, s, R, t)
    aligned, (s_est, R_est, t_est) = sim3_align(source, target)
    np.testing.assert_allclose(aligned, target, atol=1e-5)
    assert abs(s_est - s) < 1e-5
    np.testing.assert_allclose(R_est, R, atol=1e-5)


def test_loop_frame_indices_accepted_only_and_clamped():
    decisions = [
        {"query_submap": 2, "query_frame": 2, "detected_submap": 0, "detected_frame": 10, "accepted": True},
        {"query_submap": 3, "query_frame": 7, "detected_submap": 1, "detected_frame": 16, "accepted": False},
        {"query_submap": 5, "query_frame": 15, "detected_submap": 4, "detected_frame": 0, "accepted": True},
    ]
    pairs = loop_frame_indices(decisions, submap_size=16, n_frames=90)
    # Rejected decision excluded; global index = submap_id * submap_size + frame_idx.
    assert pairs == [(34, 10), (89, 64)]  # 5*16+15=95 clamped to n_frames-1=89


def test_trajectory_figure_traces_and_legendonly_chords():
    # Three visible trajectory traces plus a legend-hidden loop-chord trace.
    xyz = np.linspace(0.0, 1.0, 30).reshape(10, 3)
    fig = build_trajectory_figure(xyz, xyz + 0.1, xyz + 0.2, loop_pairs=[(0, 9), (2, 7)])
    names = [tr.name for tr in fig.data]
    assert names == ["GT", "baseline", "LC", "loop closures (click to show)"]
    assert all(tr.visible is None for tr in fig.data[:3])  # default-visible
    assert fig.data[3].visible == "legendonly"
    # One None-gap segment per loop pair: 3 vertices per chord.
    assert len(fig.data[3].x) == 6


def test_trajectory_figure_without_loops_has_three_traces():
    xyz = np.zeros((4, 3))
    fig = build_trajectory_figure(xyz, xyz, xyz, loop_pairs=[])
    assert [tr.name for tr in fig.data] == ["GT", "baseline", "LC"]
