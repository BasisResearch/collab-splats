"""Sweep incremental BA add_size values and plot accuracy vs runtime.

Runs eval_gt.py for each add_size, saves per-condition metrics.json,
then generates the Pareto plot via plot_incremental_ba_sweep.py.

Usage:
    python evals/sweep_incremental_ba.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --max_frames 50 \\
        --add_sizes 1 2 3 5 7 10 15 25

Output:
    evals/results/incremental_ba_sweep/<add_size>/metrics.json
    evals/incremental_ba_sweep.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
EVALS_DIR = REPO_ROOT / "evals"
RESULTS_BASE = REPO_ROOT / "evals" / "results"

_DEFAULT_ADD_SIZES = [1, 2, 3, 5, 7, 10, 15, 25]


def run_condition(seq_dir: Path, output_dir: Path, max_frames: int, add_size: int) -> None:
    """Run eval_gt.py for one add_size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(EVALS_DIR / "eval_gt.py"),
        "--dataset", "7scenes",
        "--seq_dir", str(seq_dir),
        "--output_dir", str(output_dir),
        "--max_frames", str(max_frames),
        "--conditions", f"incremental_ba-{add_size}",
    ]
    print(f"\n=== add_size={add_size} ===")
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq_dir", type=Path, required=True, help="Path to sequence directory")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--add_sizes", type=int, nargs="+", default=_DEFAULT_ADD_SIZES,
                        help="add_size values to sweep")
    parser.add_argument("--output_base", type=Path,
                        default=RESULTS_BASE / "incremental_ba_sweep",
                        help="Base dir for per-add_size result subdirs")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip add_size if metrics.json already exists")
    args = parser.parse_args(argv)

    for add_size in args.add_sizes:
        out_dir = args.output_base / str(add_size)
        if args.skip_existing and (out_dir / "metrics.json").exists():
            print(f"=== add_size={add_size} — skipping (already exists) ===")
            continue
        run_condition(args.seq_dir, out_dir, args.max_frames, add_size)

    # Generate plot — import inline so script works even if matplotlib missing at import time
    print("\n=== Generating plot ===")
    sys.path.insert(0, str(REPO_ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "plot_sweep", EVALS_DIR / "plot_incremental_ba_sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
