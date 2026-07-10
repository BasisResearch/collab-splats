#!/usr/bin/env python
"""Aggregate LC parity runs into a markdown gate table.

    /opt/venv/reconstruction/bin/python evals/runners/build_parity_table.py \
        --root evals/baselines/lc_parity
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
from pathlib import Path

import numpy as np

from lc_loop_pr import loop_precision_recall
from lc_parity_common import SCENES, check_gates, check_lc_harmless, check_scaling_gate

# trajectory_io lives in evals/, not evals/runners/, and a plain `from trajectory_io
# import read_tum` would resolve to an installed `evals` pip package rather than the
# local module (same shadowing problem documented in run_vggt_slam_lc.py) — load it
# by file path instead, mirroring that runner's importlib pattern.
_traj_io_spec = _ilu.spec_from_file_location("trajectory_io", Path(__file__).resolve().parents[1] / "trajectory_io.py")
_traj_io_mod = _ilu.module_from_spec(_traj_io_spec)
_traj_io_spec.loader.exec_module(_traj_io_mod)
read_tum = _traj_io_mod.read_tum

# Scaling-sweep prefix points (spec §scaling); 100% is the main scene row.
_PREFIX_TAGS = (("prefix_25", "25%"), ("prefix_50", "50%"))


def _load(p: Path) -> dict | None:
    """Parse a metrics.json if present, else None (partial/pending run)."""
    return json.loads(p.read_text()) if p.exists() else None


def _ours_lc_metrics(ours_all: dict) -> dict | None:
    """Extract {'ate_rmse', 'loops_applied'} from eval_gt's metrics.json (lc condition)."""
    lc = ours_all.get("lc")
    if not lc:
        return None
    return {
        "ate_rmse": lc.get("ate", {}).get("rmse"),
        "loops_applied": lc.get("loops_applied"),
    }


def _ours_baseline_ate(ours_all: dict) -> float | None:
    """Extract ours' baseline-condition ATE RMSE from eval_gt's metrics.json."""
    base = ours_all.get("baseline")
    return base.get("ate", {}).get("rmse") if base else None


def _fmt(x: float | None) -> str:
    """4dp float or an em-dash for missing values."""
    return f"{x:.4f}" if x is not None else "—"


def max_translation_deviation(slam_tum: Path, ours_tum: Path) -> float | None:
    """Max per-frame translation deviation (m) between SLAM and ours-LC trajectories.

    Both files list the identical ordered keyframe sequence — ours consumes SLAM's own
    keyframe list via --keyframe_list — so rows are associated by position rather than
    by the literal timestamp column: SLAM's writer (vendored `solver.map.write_poses_
    to_file`) timestamps rows with the source-video frame index (e.g. 0, 51, 95, ...),
    while ours' timestamps are the sequential position in the filtered keyframe set
    (0, 1, 2, ...) — the two conventions only coincide at frame 0, so literal timestamp
    equality would silently degrade to a single-frame (trivial) comparison. Upstream's
    solver can also emit one duplicate row at a submap boundary (a repeated timestamp);
    those are dropped before pairing. Each trajectory is re-expressed relative to its
    own frame 0 (T_i' = inv(T_0) @ T_i) to remove any global rigid offset before
    comparing. Translation-only — rotation deviation is deferred to a future pass.
    """
    if not slam_tum.exists() or not ours_tum.exists():
        return None
    slam_poses, slam_ts = read_tum(slam_tum)
    ours_poses, _ = read_tum(ours_tum)
    if len(slam_poses) == 0 or len(ours_poses) == 0:
        return None
    # Drop consecutive duplicate SLAM timestamps (submap-boundary repeat)
    keep = [i for i in range(len(slam_ts)) if i == 0 or slam_ts[i] != slam_ts[i - 1]]
    slam_poses = slam_poses[keep]
    n = min(len(slam_poses), len(ours_poses))
    if n == 0:
        return None
    slam_rel = np.einsum("ij,njk->nik", np.linalg.inv(slam_poses[0]), slam_poses[:n])
    ours_rel = np.einsum("ij,njk->nik", np.linalg.inv(ours_poses[0]), ours_poses[:n])
    diffs = np.linalg.norm(slam_rel[:, :3, 3] - ours_rel[:, :3, 3], axis=1)
    return float(diffs.max())


def _lc_tum_path(ours_dir: Path) -> Path | None:
    """Find the LC-condition TUM file inside an ours_<backbone> dir.

    eval_gt.py names TUM files by a per-backbone prefix (spark/omega/mapanything/vggtx),
    not a fixed string, so glob for it rather than hardcoding "spark_lc.tum".
    """
    matches = sorted(ours_dir.glob("*_lc.tum"))
    return matches[0] if matches else None


def _loop_pr_cells(ours_dir: Path) -> tuple[str, str]:
    """GT-verified loop P/R cells for one arm; em-dashes when inputs are missing or degenerate.

    Computed live from lc_decisions_lc.json + gt.tum + metrics.json _config (lc_loop_pr);
    a zero denominator (no labeled accepted loops / no GT opportunities) also yields "—".
    """
    try:
        pr = loop_precision_recall(ours_dir)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return "—", "—"
    p, r = pr["precision"], pr["recall"]
    return (f"{p:.2f}" if p is not None else "—"), (f"{r:.2f}" if r is not None else "—")


def _row(label: str, backbone: str, slam: dict, ours_all: dict | None, ours_dir: Path, level_dir: Path,
        scale_ate_25: float | None = None) -> str:
    """One markdown row; 'pending' cells when a metric's run hasn't happened yet.

    Base and LC conditions get separate gate columns (base gate/lc gate) rather than
    one combined verdict: pre-fix, baseline is expected to pass while LC is expected to
    fail, and a single column would hide that signal. Loop-count equality only applies
    to the LC gate (baseline never runs loop closure). The scaling-sweep verdict
    (check_scaling_gate), when applicable, is appended as a suffix on the LC gate cell.

    Gate semantics differ by backbone (spec §Cross-model sweep): vggt_spark has an
    upstream reference at identical weights, so base/lc gate cells are PASS/FAIL vs
    SLAM. Other backbones have no upstream reference — ATE-vs-SLAM is reported (ΔATE
    column) but not gated; base gate is "—" (ungated by design) and lc gate is
    HARMLESS/HARMFUL from check_lc_harmless (LC vs that backbone's own baseline).
    """
    is_spark = backbone == "vggt_spark"
    kf, sm = slam.get("keyframes", "?"), slam.get("submaps", "?")
    prefix = f"| {label} | {kf} | {sm} | {_fmt(slam.get('ate_rmse'))} | {slam.get('loop_closures')} "
    if ours_all is None:
        return prefix + "| pending | pending | pending | — | — | — | — | pending | pending |"

    base_ate = _ours_baseline_ate(ours_all)
    if base_ate is None:
        base_gate = "pending"
    elif is_spark:
        g_base = check_gates(slam, {"ate_rmse": base_ate}, check_loops=False)
        base_gate = "PASS" if g_base["all_pass"] else "FAIL"
    else:
        base_gate = "—"  # no upstream reference for this backbone — reported, not gated

    lc = _ours_lc_metrics(ours_all)
    lc_ate_str, ours_loops_str, delta_str, lc_gate = "pending", "pending", "—", "pending"
    if lc is not None:
        g_lc = check_gates(slam, lc, check_loops=True)  # ΔATE-vs-SLAM is reported for every backbone
        lc_ate_str = _fmt(lc["ate_rmse"])
        ours_loops_str = str(lc["loops_applied"])
        delta_str = _fmt(g_lc["ate_delta"])
        if is_spark:
            lc_gate = "PASS" if g_lc["all_pass"] else "FAIL"
        else:
            harmless = check_lc_harmless(base_ate, lc["ate_rmse"])
            lc_gate = "pending" if harmless is None else ("HARMLESS" if harmless else "HARMFUL")
        scale_ok = check_scaling_gate(scale_ate_25, lc["ate_rmse"])
        if scale_ok is not None:
            lc_gate += " / scale:OK" if scale_ok else " / scale:BLOWUP"

    ours_tum = _lc_tum_path(ours_dir)
    max_dt = max_translation_deviation(level_dir / "slam" / "slam.tum", ours_tum) if ours_tum else None
    loop_p, loop_r = _loop_pr_cells(ours_dir)
    return (
        prefix + f"| {_fmt(base_ate)} | {lc_ate_str} | {ours_loops_str} | {delta_str} "
        f"| {_fmt(max_dt)} | {loop_p} | {loop_r} | {base_gate} | {lc_gate} |"
    )


def build_table(root: Path) -> str:
    """One row per (scene, backbone) + per prefix sweep point: SLAM ref vs ours-base/LC."""
    lines = [
        "| scene | kf | submaps | SLAM ATE | SLAM loops | ours ATE (base) | ours ATE (lc) "
        "| ours loops | ΔATE | max Δt | loop P | loop R | base gate | lc gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scene, spec in SCENES.items():
        scene_dir = root / scene
        slam = _load(scene_dir / "slam" / "metrics.json")
        if slam is None:
            continue
        # Discover per-backbone runs present on disk rather than assuming a fixed list
        # — a sweep may be mid-flight, or run with a --backbones subset.
        for ours_dir in sorted(scene_dir.glob("ours_*")):
            backbone = ours_dir.name[len("ours_"):]
            # Scaling gate (spec §scaling) needs this backbone's prefix_25 ours-LC ATE
            # up front, since its verdict is a suffix on the MAIN (100%) row's lc-gate
            # cell, built below.
            scale_ate_25 = None
            if spec.scaling_sweep:
                p25_ours = _load(scene_dir / "prefix_25" / ours_dir.name / "metrics.json")
                p25_lc = _ours_lc_metrics(p25_ours) if p25_ours else None
                scale_ate_25 = p25_lc["ate_rmse"] if p25_lc else None
            label = f"{scene} [{backbone}]"
            lines.append(_row(label, backbone, slam, _load(ours_dir / "metrics.json"), ours_dir, scene_dir,
                              scale_ate_25))
            # Scaling-sweep prefix rows (25/50%); 100% is the main row above. Prefix rows
            # never carry the scaling-gate suffix themselves — only the main row does.
            for tag, pct in _PREFIX_TAGS:
                pdir = scene_dir / tag
                pslam = _load(pdir / "slam" / "metrics.json")
                if pslam is None:
                    continue
                p_ours_dir = pdir / ours_dir.name
                lines.append(_row(f"{scene}@{pct} [{backbone}]", backbone, pslam,
                                  _load(p_ours_dir / "metrics.json"), p_ours_dir, pdir))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2] / "evals" / "baselines" / "lc_parity"
    )
    args = ap.parse_args()
    table = build_table(args.root)
    out = args.root / "_parity_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)  # robust to invocation before any runs exist
    out.write_text(table + "\n")
    print(table)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
