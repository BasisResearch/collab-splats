"""Tests for evals/scripts/run_vggt_slam.py — real subprocess wrapper.

Covers both modes: the no-LC baseline (--max_loops 0) and the loop-closure run
(--max_loops >0, which also emits selected_frames.txt for eval.py parity).
CLI surface only — real runs shell out to heavy/GPU VGGT-SLAM.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[2] / "evals" / "scripts" / "run_vggt_slam.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "scripts"))


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


def test_run_vggt_slam_default_max_loops_is_zero():
    """Default is the published no-LC baseline (max_loops=0)."""
    import inspect

    import run_vggt_slam as m

    assert inspect.signature(m.run_vggt_slam).parameters["max_loops"].default == 0


def test_cli_help_exposes_both_mode_surfaces():
    """--help exits 0 and documents the merged surface: baseline + LC flags/aliases."""
    out = subprocess.run([sys.executable, str(RUNNER), "--help"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    # Baseline surface + eval_suite.sh aliases
    assert "--image_dir" in out.stdout
    assert "--output" in out.stdout
    assert "--submap_size" in out.stdout
    # LC surface folded in from run_vggt_slam_lc.py
    assert "--max_loops" in out.stdout
    assert "--min_disparity" in out.stdout
    assert "--out_tum" in out.stdout  # alias for --output
    assert "--seq_dir" in out.stdout  # alias for --image_dir
    assert "--image_list" in out.stdout  # TUM GT-gap parity restriction
