"""Subprocess wrapper around VGGT-SLAM's main.py.

Runs VGGT-SLAM using the current Python interpreter (reconstruction env,
Python 3.11 — satisfies VGGT-SLAM's SL(4)/GTSAM requirement).

On success, copies the output TUM file to output_tum and removes the
corresponding .pending sentinel from evals/baselines/vggt_slam/ if present.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VGGTSLAM_DIR = REPO_ROOT / "third_party" / "VGGT-SLAM"
BASELINES_DIR = REPO_ROOT / "evals" / "baselines" / "vggt_slam"


def run_vggt_slam(
    image_dir: Path,
    output_tum: Path,
    submap_size: int = 16,
    python: str | None = None,
) -> Path:
    """Run VGGT-SLAM on image_dir; write trajectory to output_tum.

    Args:
        image_dir: Directory of input images (sorted, no GT required).
        output_tum: Destination path for the TUM trajectory file.
        submap_size: VGGT-SLAM submap window size (default 16).
        python: Python binary to use. Defaults to sys.executable.

    Returns:
        output_tum path on success.
    """
    if not VGGTSLAM_DIR.is_dir():
        raise FileNotFoundError(
            f"VGGT-SLAM submodule not found at {VGGTSLAM_DIR}. "
            "Run: git submodule update --init third_party/VGGT-SLAM"
        )
    py = python or sys.executable
    log_path = output_tum.with_suffix(".vggtslam.txt")
    cmd = [
        py,
        str(VGGTSLAM_DIR / "main.py"),
        "--image_folder", str(image_dir),
        "--max_loops", "1",
        "--min_disparity", "50",
        "--conf_threshold", "25",
        "--lc_thres", "0.95",
        "--submap_size", str(submap_size),
        "--log_results",
        "--skip_dense_log",
        "--log_path", str(log_path),
    ]
    output_tum.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=VGGTSLAM_DIR)
    if not log_path.is_file():
        raise RuntimeError(
            f"VGGT-SLAM finished but log not found at {log_path}. "
            "Check --log_path handling in VGGT-SLAM main.py."
        )
    shutil.copy2(log_path, output_tum)
    # Remove .pending sentinel for this sequence if present
    results_seq = output_tum.parent.name  # e.g. "chess_seq01"
    sentinel = BASELINES_DIR / f"{results_seq}.pending"
    if sentinel.is_file():
        sentinel.unlink()
        print(f"  Removed sentinel: {sentinel}")
    return output_tum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", type=Path, required=True,
                    help="Directory of input images for VGGT-SLAM")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output TUM trajectory path")
    ap.add_argument("--submap_size", type=int, default=16,
                    help="VGGT-SLAM submap size (default 16)")
    ap.add_argument("--python", type=str, default=None,
                    help="Python binary (default: sys.executable)")
    args = ap.parse_args()
    run_vggt_slam(args.image_dir, args.output, submap_size=args.submap_size, python=args.python)
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
