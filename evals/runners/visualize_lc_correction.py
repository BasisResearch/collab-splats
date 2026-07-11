#!/usr/bin/env python
"""Interactive HTML visualizer for loop-closure error correction in LC parity run dirs.

Produces one self-contained offline HTML per `ours_<backbone>` run dir: GT,
baseline (no-LC), and LC trajectories Sim(3)-aligned into the GT frame, with
accepted-loop chords available as a legend toggle.

Usage:
    python visualize_lc_correction.py --run_dir <.../ours_vggt_omega>
    python visualize_lc_correction.py --root <.../lc_parity_d5_postfix>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Reuse eval's own Sim(3) Umeyama (matches evo align+correct_scale used for metrics.json ATE)
from collab_splats.geometry.loop_closure.closure import umeyama_sim3

logger = logging.getLogger(__name__)

########################################
############ Plot constants ############
########################################

# CVD-validated categorical colors (dataviz six-checks: worst adjacent ΔE 19.4);
# gray is intentionally recessive for the GT reference trace.
COLOR_GT = "#8a8a8a"
COLOR_BASELINE = "#c9184a"
COLOR_LC = "#1b9e77"
COLOR_LOOP = "#b28900"

########################################
########## Data-prep functions #########
########################################


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a TUM trajectory file into (timestamps (N,), positions (N,3), quats xyzw (N,4))."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(v) for v in line.split()]
        rows.append(vals[:8])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a Sim(3) transform (target = s * R @ p + t) to an (N, 3) array."""
    R64 = np.asarray(R, dtype=np.float64)
    return (s * (R64 @ np.asarray(points, dtype=np.float64).T)).T + np.asarray(t, dtype=np.float64)


def sim3_align(source: np.ndarray, target: np.ndarray):
    """Sim(3)-align source positions to target; returns (aligned (N,3), (s, R, t))."""
    s, R, t = umeyama_sim3(source=source, target=target)
    return apply_sim3(source, s, R, t), (s, R, t)


def loop_frame_indices(decisions: list[dict], submap_size: int, n_frames: int) -> list[tuple[int, int]]:
    """Map accepted loop decisions to global (query_idx, detected_idx) keyframe indices.

    Global index = submap_id * submap_size + frame_idx: wrappers.py windows stride by
    submap_size with frame_start = submap_id * submap_size, and per-submap frame lists
    index from frame_start (overlap frames spill past the stride, hence the clamp).
    """
    pairs = []
    for d in decisions:
        if not d.get("accepted", False):
            continue
        q = min(d["query_submap"] * submap_size + d["query_frame"], n_frames - 1)
        det = min(d["detected_submap"] * submap_size + d["detected_frame"], n_frames - 1)
        pairs.append((q, det))
    return pairs


########################################
############ Figure builder ############
########################################


def build_trajectory_figure(
    gt_xyz: np.ndarray,
    baseline_xyz: np.ndarray,
    lc_xyz: np.ndarray,
    loop_pairs: list[tuple[int, int]],
) -> go.Figure:
    """Three aligned trajectories plus legend-toggled accepted-loop chords."""
    fig = go.Figure()

    # Trajectory lines (all Sim3-aligned into the GT frame)
    for xyz, name, color in [
        (gt_xyz, "GT", COLOR_GT),
        (baseline_xyz, "baseline", COLOR_BASELINE),
        (lc_xyz, "LC", COLOR_LC),
    ]:
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
            line=dict(color=color, width=4), name=name,
        ))

    # Loop chords on the LC trajectory, hidden until toggled via the legend
    if loop_pairs:
        xs, ys, zs = [], [], []
        for q, d in loop_pairs:
            xs += [lc_xyz[q, 0], lc_xyz[d, 0], None]
            ys += [lc_xyz[q, 1], lc_xyz[d, 1], None]
            zs += [lc_xyz[q, 2], lc_xyz[d, 2], None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=COLOR_LOOP, width=3, dash="dash"),
            name="loop closures (click to show)",
            visible="legendonly",
        ))

    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(aspectmode="data"),
        legend=dict(itemsizing="constant"),
    )
    return fig


########################################
############# HTML assembly ############
########################################

_PAGE_CSS = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #222; }
h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
p.stats { margin: 0.2rem 0 0.8rem; font-size: 0.95rem; }
p.caption { font-size: 0.85rem; color: #555; max-width: 70rem; }
"""


def build_html(title: str, stats_html: str, fig: go.Figure) -> str:
    """Assemble the single self-contained page: header + one plotly figure."""
    return "".join([
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>',
        f"<style>{_PAGE_CSS}</style></head><body>",
        f"<h1>{title}</h1>",
        f'<p class="stats">{stats_html}</p>',
        '<p class="caption">GT gray, baseline (no LC) red, LC green — each trajectory '
        "independently Sim(3) Umeyama-aligned to GT (same convention as the eval's ATE). "
        "Dashed chords joining the two frames of each accepted loop are hidden by default; "
        "click the legend entry to show them.</p>",
        fig.to_html(full_html=False, include_plotlyjs=True),
        "</body></html>",
    ])


########################################
############### Pipeline ###############
########################################


def visualize_run(run_dir: Path) -> Path | None:
    """Build <run_dir>/plots/lc_correction.html; returns the output path or None if skipped."""
    run_dir = Path(run_dir)

    # Locate the backbone TUM pair; both conditions plus GT are required
    baseline_tums = sorted(run_dir.glob("*_baseline.tum"))
    if not baseline_tums or not (run_dir / "gt.tum").exists():
        logger.warning("%s: missing gt.tum or *_baseline.tum; skipping", run_dir)
        return None
    prefix = baseline_tums[0].name[: -len("_baseline.tum")]
    lc_tum = run_dir / f"{prefix}_lc.tum"
    if not lc_tum.exists():
        logger.warning("%s: missing %s_lc.tum; skipping", run_dir, prefix)
        return None

    # Run config + headline metrics (tolerate absence: loop chords then unavailable)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    cfg = metrics.get("_config", {})
    submap_size = int(cfg.get("submap_size", 0))

    # Load trajectories and align each into the GT frame (Sim3, eval convention)
    _, gt_xyz, _ = load_tum(run_dir / "gt.tum")
    _, base_xyz_raw, _ = load_tum(baseline_tums[0])
    _, lc_xyz_raw, _ = load_tum(lc_tum)
    n = gt_xyz.shape[0]
    if base_xyz_raw.shape[0] != n or lc_xyz_raw.shape[0] != n:
        logger.warning("%s: trajectory length mismatch (gt=%d base=%d lc=%d); skipping",
                       run_dir, n, base_xyz_raw.shape[0], lc_xyz_raw.shape[0])
        return None
    base_xyz, _ = sim3_align(base_xyz_raw, gt_xyz)
    lc_xyz, _ = sim3_align(lc_xyz_raw, gt_xyz)

    # Accepted loop pairs -> global keyframe indices
    loop_pairs: list[tuple[int, int]] = []
    decisions_path = run_dir / "lc_decisions_lc.json"
    if submap_size > 0 and decisions_path.exists():
        loop_pairs = loop_frame_indices(json.loads(decisions_path.read_text()), submap_size, n)
    else:
        logger.warning("%s: no submap_size/lc_decisions_lc.json; skipping loop chords", run_dir)

    fig = build_trajectory_figure(gt_xyz, base_xyz, lc_xyz, loop_pairs)

    # Header stats from metrics.json
    scene, backbone = run_dir.parent.name, run_dir.name
    base_ate = metrics.get("baseline", {}).get("ate", {}).get("rmse")
    lc_ate = metrics.get("lc", {}).get("ate", {}).get("rmse")
    loops_applied = metrics.get("lc", {}).get("loops_applied")
    stats = []
    if base_ate is not None and lc_ate is not None:
        stats.append(f"ATE RMSE: baseline <b>{base_ate:.4f} m</b> → LC <b>{lc_ate:.4f} m</b> "
                     f"({(lc_ate - base_ate) / base_ate * 100:+.1f}%)")
    if loops_applied is not None:
        stats.append(f"loops applied: <b>{loops_applied}</b>")
    stats.append(f"keyframes: <b>{n}</b>, submap_size: <b>{submap_size or '?'}</b>")

    html = build_html(f"LC correction — {scene} / {backbone}", " · ".join(stats), fig)
    out = run_dir / "plots" / "lc_correction.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def find_run_dirs(root: Path) -> list[Path]:
    """All ours_* dirs under root that have both a baseline and an LC trajectory."""
    dirs = []
    for d in sorted(root.rglob("ours_*")):
        if d.is_dir() and (d / "gt.tum").exists() and list(d.glob("*_baseline.tum")) and list(d.glob("*_lc.tum")):
            dirs.append(d)
    return dirs


def main() -> None:
    """CLI entry point: single run dir or batch over a matrix/probe root."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=Path, help="single ours_<backbone> run dir")
    group.add_argument("--root", type=Path, help="matrix/probe root; batch over every eligible ours_* dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dirs = [args.run_dir] if args.run_dir else find_run_dirs(args.root)
    if not run_dirs:
        logger.warning("no eligible run dirs found")
    written = [p for d in run_dirs if (p := visualize_run(d)) is not None]
    logger.info("done: %d/%d HTML files written", len(written), len(run_dirs))


if __name__ == "__main__":
    main()
