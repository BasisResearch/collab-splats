"""Aggregate cross_model/*/metrics.json into a markdown results table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assemble_rows(runs: dict, slam_ate: dict, reference_backbone: str) -> list[dict]:
    """Build per-(backbone,frameset,condition) rows with Δ-vs-spark / Δ-vs-SLAM (ATE)."""
    # Reference ATE per (frameset, condition) from the reference backbone.
    ref = {(fs, cond): m["ate"]["rmse"]
           for (b, fs, cond), m in runs.items() if b == reference_backbone}
    rows: list[dict] = []
    for (b, fs, cond), m in sorted(runs.items()):
        ate = m["ate"]["rmse"]
        rows.append({
            "backbone": b, "frameset": fs, "condition": cond,
            "ate": ate,
            "rpe_t": m["rpe"]["trans_rmse"], "rpe_r": m["rpe"]["rot_rmse_deg"],
            "auc_5": m["auc"]["auc_5"], "auc_15": m["auc"]["auc_15"],
            "auc_30": m["auc"]["auc_30"],
            "delta_vs_spark": ate - ref.get((fs, cond), float("nan")),
            "delta_vs_slam": ate - slam_ate.get(fs, float("nan")),
        })
    return rows


def _load_runs(root: Path) -> dict:
    """Read every cross_model/<backbone>__<frameset>__<sm>/metrics.json."""
    runs: dict = {}
    for mj in root.glob("*/metrics.json"):
        backbone, frameset, _sm = mj.parent.name.split("__")
        data = json.loads(mj.read_text())
        for cond, m in data.items():
            if cond.startswith("_"):
                continue
            runs[(backbone, frameset, cond)] = m
    return runs


def _load_slam_ate(sweep_dir: Path) -> dict:
    """Map frameset name → VGGT-SLAM ATE from disparity_sweep/slam_dN/metrics.json."""
    out: dict = {}
    for mj in sweep_dir.glob("slam_*/metrics.json"):
        out[mj.parent.name] = json.loads(mj.read_text()).get("ate_rmse", float("nan"))
    return out


def render_markdown(rows: list[dict]) -> str:
    """Render rows as a pipe table."""
    head = ("| backbone | frameset | cond | ATE | RPE-t | RPE-r° | AUC5 | AUC15 | AUC30 "
            "| Δspark | Δslam |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['backbone']} | {r['frameset']} | {r['condition']} | {r['ate']:.4f} "
            f"| {r['rpe_t']:.4f} | {r['rpe_r']:.2f} | {r['auc_5']:.1f} | {r['auc_15']:.1f} "
            f"| {r['auc_30']:.1f} | {r['delta_vs_spark']:+.4f} | {r['delta_vs_slam']:+.4f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("evals/baselines/cross_model"))
    ap.add_argument("--sweep_dir", type=Path,
                    default=Path("evals/baselines/disparity_sweep"))
    ap.add_argument("--reference_backbone", type=str, default="vggt_spark",
                    help="Backbone used as the Δ-vs-reference baseline in the table. Default vggt_spark.")
    args = ap.parse_args()
    rows = assemble_rows(_load_runs(args.root), _load_slam_ate(args.sweep_dir),
                         reference_backbone=args.reference_backbone)
    print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
