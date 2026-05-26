"""Tests for evals/runners/run_vggt_slam.py — real subprocess wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))


def test_vggtslam_dir_exists():
    from run_vggt_slam import VGGTSLAM_DIR
    assert VGGTSLAM_DIR.is_dir(), f"VGGT-SLAM not found at {VGGTSLAM_DIR}"


def test_vggtslam_main_exists():
    from run_vggt_slam import VGGTSLAM_DIR
    assert (VGGTSLAM_DIR / "main.py").is_file()


def test_baselines_dir_exists():
    from run_vggt_slam import BASELINES_DIR
    assert BASELINES_DIR.is_dir()


def test_run_vggt_slam_missing_submodule(tmp_path, monkeypatch):
    """Raises FileNotFoundError when VGGT-SLAM submodule not present."""
    import run_vggt_slam as m
    monkeypatch.setattr(m, "VGGTSLAM_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="VGGT-SLAM submodule not found"):
        m.run_vggt_slam(tmp_path, tmp_path / "out.tum")
