"""Subprocess wrapper around VGGT-SLAM's ``main.py`` (BLOCKED until Py3.11).

VGGT-SLAM (MIT-SPARK) targets the SL(4) manifold optimizer in GTSAM and pins
Python 3.11. Our nerfstudio env runs Python 3.10. Until that env is upgraded,
this runner refuses to execute and instead raises :class:`EnvBlocked` with the
exact remediation steps. Phase-2 (``evals/eval_compare.py``) discovers
``evals/baselines/vggt_slam/<seq>.pending`` sentinel files and reports the
method as ``"status": "pending"`` instead of running this stub.

When the env upgrade lands, replace the body of :func:`run_vggt_slam` with a
real subprocess invocation modeled on
``third_party/VGGT-SLAM/evals/eval_tum.sh``: ``python main.py --image_folder
<rgb_dir> --max_loops 1 --log_results --log_path <path>``. VGGT-SLAM writes
TUM format directly (``map.write_poses_to_file(kitti_format=False)``), so the
adapter just needs to copy/symlink that file into the results dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VGGTSLAM_DIR = REPO_ROOT / "third_party" / "VGGT-SLAM"


class EnvBlocked(RuntimeError):
    """VGGT-SLAM cannot run because the project env lacks Python 3.11 / SL(4)."""


def run_vggt_slam(
    image_dir: Path,
    output_tum: Path,
    python: str | None = None,
) -> Path:
    """Stub: always raises :class:`EnvBlocked` until the env is upgraded.

    Parameters mirror :func:`evals.runners.run_vggt_long.run_vggt_long` so the
    real implementation can drop in without changing call sites.
    """
    raise EnvBlocked(
        "VGGT-SLAM requires Python 3.11 + SL(4) manifold support (GTSAM). "
        "Steps to unblock:\n"
        "  1. conda create -n vggt-slam python=3.11\n"
        "  2. conda activate vggt-slam\n"
        "  3. cd third_party/VGGT-SLAM && pip install -e .\n"
        "  4. Replace the body of run_vggt_slam with:\n"
        "       cmd = [python, str(VGGTSLAM_DIR / 'main.py'),\n"
        "              '--image_folder', str(image_dir),\n"
        "              '--max_loops', '1', '--log_results',\n"
        "              '--log_path', str(output_tum)]\n"
        "       subprocess.run(cmd, check=True, cwd=VGGTSLAM_DIR)\n"
        "  5. Drop the corresponding .pending sentinel under "
        "evals/baselines/vggt_slam/.\n"
        f"Inputs were image_dir={image_dir}, output_tum={output_tum}, python={python}."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--python", type=str, default=None)
    args = ap.parse_args()
    try:
        run_vggt_slam(args.image_dir, args.output, args.python)
    except EnvBlocked as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
