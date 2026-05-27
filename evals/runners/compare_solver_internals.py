#!/usr/bin/env python
"""Compare per-boundary solver internals from VGGT-SLAM and our LC pipeline.

Loads both JSON dumps (from vggt_slam_solver_dump.py and our_solver_dump.py),
produces a per-boundary diff table and writes boundary_diff.json.

Usage:
    python evals/runners/compare_solver_internals.py \\
        --slam_dump evals/results/parity_harness/vggt_slam_internals.json \\
        --our_dump  evals/results/parity_harness/our_internals.json \\
        --out_json  evals/results/parity_harness/boundary_diff.json \\
        --flag_threshold 0.01
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _frob(a: list | None, b: list | None) -> float | None:
    """Frobenius norm of (np.array(a) - np.array(b)). Returns None if either is None."""
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.array(a) - np.array(b), "fro"))


def compare(
    slam_dump: Path,
    our_dump: Path,
    out_json: Path,
    flag_threshold: float = 0.01,
) -> None:
    """Load both dumps, compute diffs, print table, write JSON."""
    try:
        slam_data = json.loads(slam_dump.read_text())
        our_data = json.loads(our_dump.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file not found: {exc.filename}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed JSON: {exc}") from exc

    slam_bounds = {e["boundary_idx"]: e for e in slam_data["boundaries"]}
    our_bounds = {e["boundary_idx"]: e for e in our_data["boundaries"]}
    all_indices = sorted(set(slam_bounds) | set(our_bounds))

    # Format helper
    def _fmt(v: float | None) -> str:
        return f"{v:12.4f}" if v is not None else f"{'N/A':>12}"

    # Header
    print(
        f"\n{'Bnd':>4}  {'delta_scale':>12}  {'delta_H_w':>12}  "
        f"{'delta_T':>10}  {'delta_H_opt':>12}  {'note':>4}"
    )
    print("-" * 65)

    rows = []
    for idx in all_indices:
        slam_e = slam_bounds.get(idx)
        our_e = our_bounds.get(idx)

        ds = None
        if slam_e and our_e:
            ds = abs(float(slam_e.get("scale") or 0) - float(our_e.get("scale") or 0))

        d_hw = _frob(slam_e.get("H_w") if slam_e else None,
                     our_e.get("H_w") if our_e else None)
        d_t = _frob(slam_e.get("T") if slam_e else None,
                    our_e.get("T") if our_e else None)
        d_ho = _frob(slam_e.get("H_opt") if slam_e else None,
                     our_e.get("H_opt") if our_e else None)

        flag = "←" if (d_hw is not None and d_hw > flag_threshold) else ""

        print(
            f"{idx:>4d}  {_fmt(ds)}  {_fmt(d_hw)}  {_fmt(d_t)}  {_fmt(d_ho)}  {flag}"
        )

        rows.append({
            "boundary_idx": idx,
            "delta_scale": ds,
            "delta_H_w_frob": d_hw,
            "delta_T_frob": d_t,
            "delta_H_opt_frob": d_ho,
            "flagged": bool(flag),
        })

    # Overall stats
    hw_vals = [r["delta_H_w_frob"] for r in rows if r["delta_H_w_frob"] is not None]
    if hw_vals:
        print(f"\ndelta_H_w — mean: {np.mean(hw_vals):.4f}  max: {np.max(hw_vals):.4f}")
        flagged = sum(1 for r in rows if r["flagged"])
        print(f"Flagged boundaries (delta_H_w > {flag_threshold}): {flagged}/{len(rows)}")

    # Write JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "slam_config": slam_data.get("config"),
        "our_config": our_data.get("config"),
        "flag_threshold": flag_threshold,
        "boundaries": rows,
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Written: {out_json}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Compare per-boundary solver internals from VGGT-SLAM and our pipeline."
    )
    parser.add_argument(
        "--slam_dump",
        default="evals/results/parity_harness/vggt_slam_internals.json",
    )
    parser.add_argument(
        "--our_dump",
        default="evals/results/parity_harness/our_internals.json",
    )
    parser.add_argument(
        "--out_json",
        default="evals/results/parity_harness/boundary_diff.json",
    )
    parser.add_argument(
        "--flag_threshold",
        type=float,
        default=0.01,
        help="delta_H_w threshold above which a boundary is flagged (default: 0.01).",
    )
    args = parser.parse_args()

    compare(
        slam_dump=Path(args.slam_dump),
        our_dump=Path(args.our_dump),
        out_json=Path(args.out_json),
        flag_threshold=args.flag_threshold,
    )


if __name__ == "__main__":
    main()
