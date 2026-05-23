"""Tests for evals/runners/run_vggt_slam.py — Py3.11-blocked stub."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))


def test_run_vggt_slam_raises_env_blocked():
    from run_vggt_slam import EnvBlocked, run_vggt_slam

    with pytest.raises(EnvBlocked, match="Python 3.11"):
        run_vggt_slam(
            image_dir=Path("/data/tum/fr1_desk/rgb"),
            output_tum=Path("/tmp/never.tum"),
        )


def test_env_blocked_message_includes_remediation():
    from run_vggt_slam import EnvBlocked, run_vggt_slam

    with pytest.raises(EnvBlocked) as excinfo:
        run_vggt_slam(image_dir=Path("/x"), output_tum=Path("/y.tum"))
    msg = str(excinfo.value)
    assert "conda create" in msg
    assert "vggt-slam" in msg
    assert "main.py" in msg


def test_env_blocked_is_a_runtime_error():
    """Catch-all `except RuntimeError` should also catch EnvBlocked."""
    from run_vggt_slam import EnvBlocked

    assert issubclass(EnvBlocked, RuntimeError)


def test_pending_sentinels_exist_for_planned_sequences():
    """The .pending sentinel files match what eval_compare.py expects to skip."""
    repo = Path(__file__).resolve().parents[2]
    pending_dir = repo / "evals" / "baselines" / "vggt_slam"
    assert pending_dir.is_dir()

    expected_seqs = {
        # 9 TUM fr1 (matches VGGT-SLAM's eval_tum.sh)
        "fr1_desk", "fr1_desk2", "fr1_360", "fr1_floor", "fr1_plant",
        "fr1_room", "fr1_rpy", "fr1_teddy", "fr1_xyz",
        # 7-Scenes representative trio
        "chess_seq01", "fire_seq01", "office_seq01",
    }
    found = {p.stem for p in pending_dir.glob("*.pending")}
    missing = expected_seqs - found
    assert not missing, f"missing pending sentinels: {missing}"
