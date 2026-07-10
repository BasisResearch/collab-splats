"""run_vggt_slam_lc CLI: --image_list flag exists (subprocess — in-process import of
the runner would prepend third_party/VGGT-SLAM to sys.path, whose own evals/ dir
shadows evals.runners for later-collected test modules)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[2] / "evals" / "runners" / "run_vggt_slam_lc.py"


def test_cli_exposes_image_list_flag():
    # --help exits 0 and documents --image_list (frame-universe restriction for TUM parity)
    out = subprocess.run([sys.executable, str(RUNNER), "--help"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "--image_list" in out.stdout
