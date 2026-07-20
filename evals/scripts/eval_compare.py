#!/usr/bin/env python
"""Phase-2 unified comparison runner for the GT eval harness.

Phase-1 runners (``eval.py`` for our methods, ``run_vggt_long.py`` /
``run_vggt_slam.py`` for external baselines) drop a per-method TUM trajectory
into ``evals/results/<dataset_seq>/`` next to a ``gt.tum`` reference. This
phase-2 runner ingests that directory and emits a single ``metrics.json``
holding ATE + RPE for every method, plus a Markdown summary on stdout.

Usage:
    python evals/scripts/eval_compare.py --results-dir evals/results/chess_seq01

Default alignment per method (override via ``--align-overrides``):

    gt               skipped (it IS the reference)
    omega_baseline   sim3  (VGGT-Omega baseline, monocular Sim(3))
    omega_ba         sim3  (VGGT-Omega + bundle adjustment)
    omega_lc         sim3  (VGGT-Omega + loop closure, monocular Sim(3))
    vggtx_baseline   sim3  (VGGT-X baseline, monocular Sim(3))
    vggtx_lc         sim3  (VGGT-X + loop closure, monocular Sim(3))
    vggt_slam        sim3  (VGGT-SLAM internal LC, monocular)
    <unknown>        sim3  (mono assumption + warning)

A method that is queued but not yet computed appears as a sentinel
``<method>.pending`` file; it is recorded as ``{"status": "pending"}`` and no
metric is run. Any non-``.tum`` non-``.pending`` file in the directory aborts
the run — the directory is expected to be clean.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import compute_ate, compute_rpe, compute_auc

logger = logging.getLogger(__name__)

_DEFAULT_ALIGN: dict[str, str] = {
    # All monocular feedforward methods use sim3 — matches VGGT-SLAM's evo_ape -as protocol.
    # Even baseline/BA have scale ambiguity; SE(3) alignment would be unfair.
    "omega_baseline": "sim3",
    "omega_ba": "sim3",
    "omega_lc": "sim3",
    "vggtx_baseline": "sim3",
    "vggtx_lc": "sim3",
    "vggt_slam": "sim3",
    "vggt_long": "sim3",
    # legacy names (backward compat)
    "ours_baseline": "sim3",
    "ours_ba": "sim3",
    "ours_lc": "sim3",
}
_FALLBACK_ALIGN = "sim3"


def _parse_overrides(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--align-overrides entry must be name=align, got {raw!r}")
        name, align = raw.split("=", 1)
        name = name.strip()
        align = align.strip()
        if align not in ("none", "se3", "sim3"):
            raise ValueError(f"override align must be one of none|se3|sim3, got {align!r}")
        out[name] = align
    return out


def _resolve_align(method: str, overrides: dict[str, str]) -> str:
    if method in overrides:
        return overrides[method]
    if method in _DEFAULT_ALIGN:
        return _DEFAULT_ALIGN[method]
    logger.warning(
        "Unknown method %r — defaulting to align=%s (mono assumption). "
        "Pass --align-overrides %s=<se3|sim3|none> to silence.",
        method,
        _FALLBACK_ALIGN,
        method,
    )
    return _FALLBACK_ALIGN


def scan_results_dir(results_dir: Path, gt_path: Path) -> tuple[dict[str, Path], set[str]]:
    """Return (method_name → tum path) and a set of pending method names.

    Raises if any non-.tum/non-.pending file is found, or if names collide.
    """
    methods: dict[str, Path] = {}
    pending: set[str] = set()
    for entry in sorted(results_dir.iterdir()):
        if entry.is_dir():
            continue
        if entry.name == "metrics.json":
            continue  # prior run output, will be overwritten
        if entry == gt_path:
            continue
        if entry.suffix == ".tum":
            stem = entry.stem
            if stem == "gt":
                continue  # GT lives elsewhere or has already been excluded
            if stem in methods:
                raise ValueError(f"duplicate method name {stem!r} in {results_dir}")
            methods[stem] = entry
        elif entry.suffix == ".pending":
            pending.add(entry.stem)
        elif entry.suffix in (".json", ".npz", ".png", ".jpg", ".log"):
            continue  # sidecar files — skip silently
        else:
            raise ValueError(
                f"unexpected file {entry.name!r} in {results_dir} — "
                "only *.tum, *.pending, and sidecar files are allowed"
            )
    overlap = methods.keys() & pending
    if overlap:
        raise ValueError(f"method names appear as both .tum and .pending: {sorted(overlap)}")
    return methods, pending


def format_markdown(methods: dict[str, dict]) -> str:
    header = (
        "| method | status | align | ATE RMSE | RPE trans | RPE rot° | AUC@30 | loop_res↓ | chamfer_ratio↓ |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for name in sorted(methods.keys()):
        body = methods[name]
        status = body.get("status", "?")
        if status != "ok":
            lines.append(f"| {name} | {status} | - | - | - | - | - | - | - |")
            continue
        a = body["ate"]
        r = body["rpe"]
        auc_val = body.get("auc", {}).get("auc_30", float("nan"))
        al = body.get("alignment") or {}
        loop_before = al.get("loop_match_residual", {}).get("mean_before", None)
        loop_after = al.get("loop_match_residual", {}).get("mean_after", None)
        chamfer_before = al.get("pointcloud_chamfer", {}).get("mean_before", None)
        chamfer_after = al.get("pointcloud_chamfer", {}).get("mean_after", None)
        loop_str = (
            f"{loop_before:.3f}→{loop_after:.3f}" if (loop_before is not None and loop_after is not None) else "null"
        )
        chamfer_ratio = (
            (chamfer_after / chamfer_before) if (chamfer_before and chamfer_after and chamfer_before > 0) else None
        )
        chamfer_str = f"{chamfer_ratio:.3f}" if chamfer_ratio is not None else "null"
        lines.append(
            f"| {name} | {status} | {body['align']} | "
            f"{a['rmse']:.4f} | {r['trans_rmse']:.4f} | {r['rot_rmse_deg']:.4f} | "
            f"{auc_val:.1f} | {loop_str} | {chamfer_str} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, required=True, help="Directory holding gt.tum + per-method .tum files"
    )
    parser.add_argument("--gt-path", type=Path, default=None, help="Override GT path (default: <results-dir>/gt.tum)")
    parser.add_argument(
        "--align-overrides", nargs="*", default=None, help="Per-method alignment overrides, e.g. ours_baseline=sim3"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    gt_path: Path = args.gt_path if args.gt_path is not None else results_dir / "gt.tum"
    if not gt_path.is_file():
        raise FileNotFoundError(f"gt trajectory not found at {gt_path} — phase-2 needs a 'gt.tum' file")

    overrides = _parse_overrides(args.align_overrides)
    methods, pending = scan_results_dir(results_dir, gt_path)

    out: dict[str, dict] = {}
    for name in sorted(pending):
        out[name] = {"status": "pending"}

    for name, tum_path in methods.items():
        align = _resolve_align(name, overrides)
        ate = compute_ate(tum_path, gt_path, align=align)
        rpe = compute_rpe(tum_path, gt_path, align=align, delta=1)
        auc = compute_auc(tum_path, gt_path)
        alignment_path = results_dir / f"{name}_alignment.json"
        alignment = json.loads(alignment_path.read_text()) if alignment_path.is_file() else None
        out[name] = {
            "status": "ok",
            "align": align,
            "ate": ate,
            "rpe": rpe,
            "auc": auc,
            "alignment": alignment,
        }

    payload = {"methods": out}
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    print(format_markdown(out))
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()


########################################################################
# Backward-compatible private aliases (pre-rename callers/tests)
########################################################################

_scan_results_dir = scan_results_dir
_format_markdown = format_markdown


########################################################################
# Grid aggregation (config-driven eval.py)
########################################################################


def collect_grid_metrics(output_root: Path) -> list[dict]:
    """Read every <cell>/metrics.json under a grid output root into flat rows.

    Each single-cell metrics.json is keyed by condition (plus a ``_config``
    block); one row is emitted per condition, tagged with its cell dir name.
    """
    rows = []
    for mj in sorted(output_root.glob("*/metrics.json")):
        data = json.loads(mj.read_text())
        for cond, m in data.items():
            if cond == "_config" or not isinstance(m, dict) or "ate" not in m:
                continue
            rows.append(m | {"_cell": mj.parent.name, "_condition": cond})
    return rows


def format_markdown_rows(rows: list[dict]) -> str:
    """Render grid rows (cell + ATE/RPE/AUC) as a markdown table."""
    header = "| cell | ATE RMSE | RPE trans | RPE rot deg | AUC@30 |\n" "|---|---|---|---|---|"
    lines = [header]
    for r in sorted(rows, key=lambda x: x.get("_cell", "")):
        ate = r.get("ate", {}).get("rmse", float("nan"))
        rpe_t = r.get("rpe", {}).get("trans_rmse", float("nan"))
        rpe_r = r.get("rpe", {}).get("rot_rmse_deg", float("nan"))
        auc = r.get("auc", {}).get("auc_30", float("nan"))
        lines.append(f"| {r.get('_cell','?')} | {ate:.4f} | {rpe_t:.4f} | {rpe_r:.4f} | {auc:.4f} |")
    return "\n".join(lines)
