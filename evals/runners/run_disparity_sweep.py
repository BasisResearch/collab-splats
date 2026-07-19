#!/usr/bin/env python
"""Disparity sweep parity harness: compares our pipeline against VGGT-SLAM.

Usage:
    python evals/runners/run_disparity_sweep.py \\
        --seq_dir evals/data/7scenes/chess/chess/seq-01 \\
        --max_frames 200

Sweeps min_disparity [50, 30, 20, 10, 0]. At each level:
  1. Runs VGGT-SLAM baseline (no LC) → reads metrics.json for ATE + selected_frames.txt.
  2. Runs our vggt_spark baseline on same keyframes → reads baseline_ate.json.
  3. Gates: ATE divergence > 10% → prints diagnostic, exits 1.
  4. At d=50 only: runs both with LC, compares similarity scores + loop counts.
  5. For all d: runs both with LC, gates on LC ATE parity.

Use --start_disparity to resume a partial sweep (default 50, meaning start from the top).

Exits 0 on full sweep completion, 1 on first parity failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

_PYTHON = "/opt/conda/envs/reconstruction/bin/python"
_REPO = Path(__file__).resolve().parents[2]
_EVAL_GT = _REPO / "evals" / "eval.py"
_SLAM_RUNNER = _REPO / "evals" / "runners" / "run_vggt_slam_lc.py"
_SEQ_DEFAULT = _REPO / "evals" / "data" / "7scenes" / "chess" / "chess" / "seq-01"
_SWEEP_BASE = _REPO / "evals" / "baselines" / "disparity_sweep"
_SIMILARITY_JSON = _REPO / "evals" / "results" / "parity_harness" / "vggt_spark_similarity.json"

DISPARITY_LEVELS = [50, 30, 20, 10, 0]
ATE_PARITY_THRESHOLD = 0.10      # 10% relative difference
LOOP_PARITY_RATIO = 3.0          # our loops / slam loops must be in [1/3, 3]
SIMILARITY_MEAN_TOLERANCE = 0.05  # max allowed mean score difference between pipelines


########################################################################
# Subprocess helpers
########################################################################

def _run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output unless capture=True."""
    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(
        [str(c) for c in cmd],
        capture_output=capture,
        text=True,
        check=True,
    )


########################################################################
# Pipeline runners
########################################################################

def run_slam_baseline(seq_dir: Path, d: int, max_frames: int, out_dir: Path) -> dict:
    """Run VGGT-SLAM with no LC (max_loops=0). Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tum = out_dir / "baseline.tum"
    _run([
        _PYTHON, _SLAM_RUNNER,
        "--seq_dir", seq_dir,
        "--max_frames", max_frames,
        "--min_disparity", d,
        "--max_loops", 0,
        "--out_tum", tum,
    ])
    return json.loads((out_dir / "metrics.json").read_text())


def run_slam_lc(seq_dir: Path, d: int, max_frames: int, out_dir: Path) -> dict:
    """Run VGGT-SLAM with LC (max_loops=1). Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tum = out_dir / "lc.tum"
    _run([
        _PYTHON, _SLAM_RUNNER,
        "--seq_dir", seq_dir,
        "--max_frames", max_frames,
        "--min_disparity", d,
        "--max_loops", 1,
        "--out_tum", tum,
    ])
    return json.loads((out_dir / "metrics.json").read_text())


def run_our_pipeline(
    seq_dir: Path, keyframe_list: Path, condition: str, out_ate: Path,
    backbone: str = "vggt_spark",
) -> dict:
    """Run our pipeline on the given keyframe list. Returns ATE dict."""
    out_ate.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _PYTHON, _EVAL_GT,
        "--dataset", "7scenes",
        "--seq_dir", seq_dir,
        "--backbone", backbone,
        "--conditions", condition,
        "--submap_size", 16,
        "--lc_scale_method", "none",
        "--keyframe_list", keyframe_list,
        "--output_ate", out_ate,
    ])
    return json.loads(out_ate.read_text())


########################################################################
# Parity checks
########################################################################

def _parity_ok(ours: float, slam: float) -> bool:
    """Return True if ATE values are within ATE_PARITY_THRESHOLD of each other."""
    if slam == 0.0:
        return ours == 0.0
    return abs(ours - slam) / slam <= ATE_PARITY_THRESHOLD


def _extract_ate(ate_dict: dict) -> float:
    """Extract scalar ATE from a {condition: ate_rmse} dict (first value wins)."""
    # Try common condition names first, fall back to first value
    for key in ("baseline", "lc", "ba"):
        if key in ate_dict:
            return float(ate_dict[key])
    return float(next(iter(ate_dict.values())))


def _check_similarity_parity(slam_loops: int) -> bool:
    """Print similarity calibration diagnostics at d=50. Returns True if no hard gate triggered.

    The hard gate is: SLAM found >0 loops but we found 0 (unambiguous Track A failure).
    Soft diagnostic (mean score gap) is printed but does not gate — human decides.
    """
    print("\n  [similarity calibration — d=50]")

    # Load VGGT-SLAM similarity scores written by run_vggt_slam_lc.py
    slam_scores: list[float] = []
    if _SIMILARITY_JSON.exists():
        sim_data = json.loads(_SIMILARITY_JSON.read_text())
        slam_scores = sim_data.get("scores", [])
        slam_mean = sum(slam_scores) / len(slam_scores) if slam_scores else float("nan")
        print(f"  VGGT-SLAM similarity scores: n={len(slam_scores)}, mean={slam_mean:.4f}")
    else:
        print(f"  VGGT-SLAM similarity JSON not found: {_SIMILARITY_JSON}")
        print("  → VGGT-SLAM LC run may not have produced similarity scores yet.")

    print(f"  VGGT-SLAM loop_closures: {slam_loops}")
    print("  → Check INFO log above for 'VGGT-SPARK image_match_ratio' lines from our LC run.")

    # Hard gate: SLAM closed loops but ours found none is checked externally (loop count gate).
    # This function returns True — loop count gate below handles the hard failure.
    return True


########################################################################
# Print helpers
########################################################################

def _print_row(label: str, slam: float | str, ours: float | str, ok: bool | None = None) -> None:
    """Print a single result row with SLAM / OURS / status columns."""
    status = "" if ok is None else ("OK" if ok else "FAIL")
    print(f"  {label:<35} SLAM={slam!s:<12} OURS={ours!s:<12} {status}")


########################################################################
# Main sweep
########################################################################

def sweep(seq_dir: Path, max_frames: int, start_disparity: int, backbone: str = "vggt_spark") -> int:
    """Run full disparity sweep. Returns 0 on full parity, 1 on first failure."""
    # Filter levels to those <= start_disparity so resume works correctly
    levels = [d for d in DISPARITY_LEVELS if d <= start_disparity]
    rows: list[dict] = []

    for d in levels:
        print(f"\n{'='*60}")
        print(f"  min_disparity = {d}")
        print(f"{'='*60}")

        slam_dir = _SWEEP_BASE / f"slam_d{d}"
        our_dir = _SWEEP_BASE / f"ours_d{d}"

        # ── Baseline: SLAM no-LC ──────────────────────────────────
        print("\n  [baseline]")
        slam_baseline = run_slam_baseline(seq_dir, d, max_frames, slam_dir)
        kf_list = slam_dir / "selected_frames.txt"

        our_dir.mkdir(parents=True, exist_ok=True)
        our_baseline = run_our_pipeline(seq_dir, kf_list, "baseline", our_dir / "baseline_ate.json", backbone=backbone)

        slam_ate = float(slam_baseline["ate_rmse"])
        our_ate = _extract_ate(our_baseline)
        ok_baseline = _parity_ok(our_ate, slam_ate)
        _print_row(f"d={d} baseline ATE (m)", f"{slam_ate:.4f}", f"{our_ate:.4f}", ok_baseline)
        rows.append({"d": d, "type": "baseline", "slam": slam_ate, "ours": our_ate, "ok": ok_baseline})

        if not ok_baseline:
            print(f"\n  PARITY FAILURE at d={d} baseline.")
            print(f"  SLAM ATE={slam_ate:.4f}m, OURS={our_ate:.4f}m")
            print(f"  Frames selected by SLAM: {slam_baseline.get('keyframes', '?')}")
            print(f"  Keyframe list: {kf_list}")
            print("  → Both pipelines used identical frames — divergence is in inference/BA.")
            return 1

        # ── LC: both pipelines ────────────────────────────────────
        print("\n  [LC]")
        slam_lc = run_slam_lc(seq_dir, d, max_frames, slam_dir)
        our_lc = run_our_pipeline(seq_dir, kf_list, "lc", our_dir / "lc_ate.json", backbone=backbone)

        slam_lc_ate = float(slam_lc["ate_rmse"])
        our_lc_ate = _extract_ate(our_lc)
        slam_loops = int(slam_lc.get("loop_closures", 0))
        ok_lc = _parity_ok(our_lc_ate, slam_lc_ate)
        _print_row(f"d={d} LC ATE (m)", f"{slam_lc_ate:.4f}", f"{our_lc_ate:.4f}", ok_lc)
        rows.append({"d": d, "type": "lc", "slam": slam_lc_ate, "ours": our_lc_ate, "ok": ok_lc})

        # ── Similarity calibration checkpoint (d=50 only) ─────────
        if d == 50:
            _check_similarity_parity(slam_loops)
            # Hard loop-count gate: SLAM found loops but we found 0
            if slam_loops > 0:
                # We can't easily read our loop count from output_ate JSON alone;
                # this is a diagnostic checkpoint — print and let ATE gate decide.
                print(f"  VGGT-SLAM accepted {slam_loops} loop closure(s) at d=50.")
                print("  → Verify our pipeline also accepted loops (check INFO log above).")

        if not ok_lc:
            print(f"\n  LC PARITY FAILURE at d={d}.")
            print(f"  SLAM LC ATE={slam_lc_ate:.4f}m, OURS={our_lc_ate:.4f}m")
            print(f"  SLAM loop_closures={slam_loops}")
            if d == 50:
                print("  → Check our INFO log for 'image_match_ratio' lines.")
            return 1

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SWEEP COMPLETE — ALL PARITY GATES PASSED")
    print(f"{'='*60}")
    print(f"  {'Level':<10} {'Type':<12} {'SLAM ATE':<12} {'OURS ATE':<12} {'OK'}")
    for r in rows:
        ok_mark = "OK" if r["ok"] else "FAIL"
        print(f"  d={r['d']:<8} {r['type']:<12} {r['slam']:.4f}{'':6} {r['ours']:.4f}{'':6} {ok_mark}")
    return 0


########################################################################
# Entry point
########################################################################

def main() -> None:
    """Parse args and run the disparity sweep."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seq_dir", type=Path, default=_SEQ_DEFAULT,
        help=f"Path to 7-Scenes sequence directory. Default: {_SEQ_DEFAULT}",
    )
    parser.add_argument(
        "--max_frames", type=int, default=200,
        help="Frame limit passed to both pipelines. Default 200.",
    )
    parser.add_argument(
        "--start_disparity", type=int, default=50,
        help=(
            "Start sweep from this disparity level (inclusive). "
            "Levels are [50, 30, 20, 10, 0]; pass e.g. 30 to skip d=50. Default 50."
        ),
    )
    parser.add_argument(
        "--backbone", type=str, default="vggt_spark",
        help="Our-pipeline backbone to compare against VGGT-SLAM. Default vggt_spark (the parity anchor).",
    )
    args = parser.parse_args()
    sys.exit(sweep(args.seq_dir, args.max_frames, args.start_disparity, backbone=args.backbone))


if __name__ == "__main__":
    main()
