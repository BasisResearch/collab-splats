"""
Ground-truth eval grid: datasets x conditions through the Reconstructor pipeline.

- a condition is a config override, merged over the grid's `base` block and configs/base.yaml
- one subprocess per cell for OOM isolation; a cell holding eval_metrics.json is skipped
- run from the repo root: python -m evals.eval --config evals/configs/7scenes.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mergedeep import merge

from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.preproc.frames import frame_idx_from_path
from collab_splats.utils.io import write_json
from collab_splats.wrapper.reconstructor import Reconstructor
from evals.datasets import get_dataset, load_gt_depth
from evals.gt_metrics import ate, auc_at_threshold, depth_error, rpe

logger = logging.getLogger(__name__)

######## Constants

# AUC cutoffs reported per cell, degrees
_AUC_THRESHOLDS = (5.0, 15.0, 30.0)

######## Grid config


@dataclass
class EvalCell:
    """
    One grid cell: a dataset sequence under one condition.
    """

    dataset: str
    dataset_type: str
    seq_dir: Path
    max_frames: int
    condition: str
    config: dict[str, Any]  # base block + condition override; no input/output paths
    output_dir: Path


@dataclass
class EvalConfig:
    """
    A parsed grid YAML.
    """

    name: str
    output_dir: Path
    datasets: list[dict[str, Any]]
    conditions: dict[str, dict[str, Any]]
    base: dict[str, Any]


def load_eval_config(path: Path) -> EvalConfig:
    """
    Parse and validate a grid YAML.

    Args:
        path: grid YAML with name, output_dir, datasets, conditions and an optional base block.

    Returns:
        The parsed grid.

    Raises:
        ValueError: the grid has no datasets or no conditions.
        KeyError: a dataset names an unknown type.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not raw.get("conditions"):
        raise ValueError(f"{path}: a grid needs at least one condition")
    if not raw.get("datasets"):
        raise ValueError(f"{path}: a grid needs at least one dataset")

    # Resolve every loader now, so a typo fails before the first cell runs
    for ds in raw["datasets"]:
        get_dataset(ds["type"])

    return EvalConfig(
        name=raw["name"],
        output_dir=Path(raw["output_dir"]),
        datasets=raw["datasets"],
        conditions=raw["conditions"],
        base=raw.get("base") or {},
    )


def build_grid(cfg: EvalConfig) -> list[EvalCell]:
    """
    Expand a grid into cells, dataset-major.

    Args:
        cfg: the parsed grid.

    Returns:
        One cell per (dataset, condition), named `<dataset>__<condition>`.
    """
    cells = []
    for ds in cfg.datasets:
        for label, override in cfg.conditions.items():
            config = merge({}, copy.deepcopy(cfg.base), copy.deepcopy(override or {}))
            cells.append(
                EvalCell(
                    dataset=ds["name"],
                    dataset_type=ds["type"],
                    seq_dir=Path(ds["seq_dir"]),
                    max_frames=int(ds["max_frames"]),
                    condition=label,
                    config=config,
                    output_dir=cfg.output_dir / f"{ds['name']}__{label}",
                )
            )
    return cells


######## One cell


def _stage_frames(images: list[Path], input_dir: Path) -> None:
    """
    Symlink frames into input_dir as NNNNNN.<ext>, so filename order is GT order.

    - the pipeline names frame i `frame_{i:06d}`, so a pose's source index is its GT index
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(images):
        link = input_dir / f"{i:06d}{src.suffix.lower()}"
        if not link.is_symlink():
            link.symlink_to(Path(src).resolve())


def _gt_match(image_paths: list[Path], n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruction rows in GT order, and the GT index each one matches, by frame name.

    - an sfm backend may leave frames unregistered: those GT frames have no row
    - raises on a frame posed twice or a name outside the GT range
    """
    gt_idx = np.array([frame_idx_from_path(p) for p in image_paths])
    if len(set(gt_idx.tolist())) != len(gt_idx) or not all(0 <= i < n_gt for i in gt_idx):
        raise ValueError(f"reconstruction frames {sorted(gt_idx.tolist())[:10]} do not match {n_gt} GT frames once")
    rows = np.argsort(gt_idx)
    return rows, gt_idx[rows]


def _pred_depth_at_gt_res(depth: np.ndarray, box: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """
    Model-res depth placed on the original frame through its crop box; 0 outside it.

    - nearest resize, so no depth is invented across edges
    - a box reaching past the frame (padding) is clipped to it
    """
    tl_x, tl_y, cr_x, cr_y = (int(round(float(v))) for v in box[:4])
    resized = cv2.resize(depth.astype(np.float32), (cr_x - tl_x, cr_y - tl_y), interpolation=cv2.INTER_NEAREST)

    # Paste the overlap of box and frame
    H, W = hw
    y0, x0, y1, x1 = max(tl_y, 0), max(tl_x, 0), min(cr_y, H), min(cr_x, W)
    canvas = np.zeros(hw, dtype=np.float32)
    canvas[y0:y1, x0:x1] = resized[y0 - tl_y : y1 - tl_y, x0 - tl_x : x1 - tl_x]
    return canvas


def _depth_metrics(
    ff: PointcloudResult, rows: np.ndarray, gt_idx: np.ndarray, depth_paths: list[Path], stride: int = 4
) -> dict[str, float]:
    """
    GT depth error over the registered frames, pooled on a strided pixel grid.

    - stride: pooling every pixel of 500 frames is ~150M values
    """
    preds, gts = [], []
    for row, gi in zip(rows, gt_idx):
        gt = load_gt_depth(depth_paths[gi])
        pred = _pred_depth_at_gt_res(np.asarray(ff.depth[row]), np.asarray(ff.original_coords[row]), gt.shape)
        preds.append(pred[::stride, ::stride])
        gts.append(gt[::stride, ::stride])
    return depth_error(np.stack(preds), np.stack(gts))


def _plot_cell(gt_c2w: np.ndarray, ate_out: dict, cell: EvalCell, plots_dir: Path) -> None:
    """
    Trajectory (GT vs Sim3-aligned prediction) and per-frame ATE PNGs.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 3D trajectory: GT against the aligned prediction
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    gt_pos, pred_pos = gt_c2w[:, :3, 3], ate_out["aligned_positions"]
    ax.plot(*gt_pos.T, label="gt", color="black", linewidth=2)
    ax.plot(*pred_pos.T, label=cell.condition, linewidth=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{cell.dataset} / {cell.condition}")
    ax.legend()
    fig.savefig(plots_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-frame ATE
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ate_out["per_frame"], label=f"{cell.condition} (RMSE={ate_out['rmse']:.3f} m)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.legend()
    fig.savefig(plots_dir / "ate_per_frame.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cell(cell: EvalCell) -> dict[str, Any]:
    """
    Reconstruct one cell through the pipeline and score it against ground truth.

    Args:
        cell: the grid cell to run.

    Returns:
        The payload written to `<cell>/eval_metrics.json`.

    Raises:
        ValueError: a frame name matches no GT frame or matches one twice, or a stage
            report the config implies is missing.
    """
    dataset = get_dataset(cell.dataset_type)(cell.seq_dir, max_frames=cell.max_frames)
    input_dir = cell.output_dir.resolve() / "input"
    run_dir = cell.output_dir.resolve() / "run"
    _stage_frames(dataset.images, input_dir)

    # Run the pipeline as production does; the config picks the stages
    paths = {"input_path": str(input_dir), "output_path": str(run_dir)}
    recon = Reconstructor(merge({}, copy.deepcopy(cell.config), paths))
    t0 = time.perf_counter()
    recon.run_pipeline()
    elapsed = time.perf_counter() - t0

    # Poses matched to GT by frame name; both sides camera-to-world for the metrics
    # - unregistered frames: ATE/RPE/depth over the registered ones, reported as N/M
    # - AUC counts every pair touching an unregistered frame as a failure (COLMAP benchmark)
    has_depth = dataset.depth_paths is not None
    ff = PointcloudResult.load_zarr(
        recon.pointcloud_zarr,
        load_depth=has_depth,
        load_world_points=False,
        load_confidence=False,
        load_pixel_indices=False,
    )
    rows, gt_idx = _gt_match(ff.image_paths, len(dataset.images))
    pred_c2w = invert_poses(np.asarray(ff.extrinsics, dtype=np.float64)[rows])
    gt_c2w = invert_poses(dataset.gt_poses.astype(np.float64)[gt_idx])

    # Ground-truth metrics
    ate_out = ate(pred_c2w, gt_c2w)
    depth = _depth_metrics(ff, rows, gt_idx, dataset.depth_paths) if has_depth else None

    # Every stage report the run wrote; the ones the config implies must be there
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.rglob("*_quality_report.json"))}
    expected = ["reconstruction_quality_report"]
    if recon.config["splats"]["enabled"]:
        expected.append("splats_quality_report")
    missing = [name for name in expected if name not in reports]
    if missing:
        raise ValueError(f"{cell.output_dir.name}: stages wrote no {missing}")

    payload = {
        "dataset": cell.dataset,
        "condition": cell.condition,
        "n_frames": len(dataset.images),
        "registered_frames": len(rows),
        "time_s": elapsed,
        "ate": {k: v for k, v in ate_out.items() if k not in ("per_frame", "aligned_positions")},
        "rpe": rpe(pred_c2w, gt_c2w),
        "auc": auc_at_threshold(pred_c2w, gt_c2w, _AUC_THRESHOLDS, n_frames=len(dataset.images)),
        "depth": depth,
        "reports": reports,
        "config": cell.config,
    }

    # Arrays for the notebook, then plots, then eval_metrics.json last: it marks the cell done
    np.savez(
        cell.output_dir / "trajectories.npz",
        gt_c2w=gt_c2w,
        gt_idx=gt_idx,
        pred_c2w=pred_c2w,
        ate_per_frame=ate_out["per_frame"],
        aligned_positions=ate_out["aligned_positions"],
    )
    _plot_cell(gt_c2w, ate_out, cell, cell.output_dir / "plots")
    write_json(cell.output_dir / "eval_metrics.json", payload)
    return payload


######## Grid


def _metric(section: dict | None, key: str) -> float:
    """
    One table value; nan when the section or key is absent.
    """
    value = (section or {}).get(key)
    return math.nan if value is None else float(value)


def _aggregate(output_dir: Path) -> None:
    """
    comparison.md and comparison.json over every finished cell.
    """
    rows = []
    for path in sorted(output_dir.glob("*/eval_metrics.json")):
        m = json.loads(path.read_text())
        rows.append(
            {
                "cell": path.parent.name,
                "registered": f"{m['registered_frames']}/{m['n_frames']}",
                "ate_rmse": _metric(m["ate"], "rmse"),
                "rpe_trans": _metric(m["rpe"], "trans_rmse"),
                "rpe_rot_deg": _metric(m["rpe"], "rot_rmse_deg"),
                "auc_30": _metric(m["auc"], "auc_30"),
                "depth_med_rel": _metric(m.get("depth"), "median_rel_err"),
            }
        )

    lines = [
        "| cell | registered | ATE RMSE | RPE trans | RPE rot deg | AUC@30 | depth med rel |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        values = " | ".join(f"{r[k]:.4f}" for k in ("ate_rmse", "rpe_trans", "rpe_rot_deg", "auc_30", "depth_med_rel"))
        lines.append(f"| {r['cell']} | {r['registered']} | {values} |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    write_json(output_dir / "comparison.json", rows)


def run_grid(config_path: Path, dry_run: bool = False) -> None:
    """
    Run every unfinished cell in its own subprocess, then aggregate.

    Args:
        config_path: grid YAML.
        dry_run: print each cell's command and run nothing.
    """
    config_path = Path(config_path).resolve()
    cfg = load_eval_config(config_path)
    if not dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # One subprocess per cell; a finished cell is skipped
    for cell in build_grid(cfg):
        if (cell.output_dir / "eval_metrics.json").exists():
            logger.info("skip %s: eval_metrics.json exists", cell.output_dir.name)
            continue
        cmd = [sys.executable, "-m", "evals.eval", "--config", str(config_path), "--cell", cell.output_dir.name]
        if dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True)

    if not dry_run:
        _aggregate(cfg.output_dir)


def main() -> None:
    """
    CLI: `--config` runs the grid; `--cell` (internal) runs one cell in this process.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="grid YAML")
    parser.add_argument("--dry_run", action="store_true", help="print each cell command, run nothing")
    parser.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.cell is None:
        run_grid(args.config, dry_run=args.dry_run)
        return

    # Internal single-cell mode, launched by run_grid
    cells = {c.output_dir.name: c for c in build_grid(load_eval_config(args.config))}
    run_cell(cells[args.cell])


if __name__ == "__main__":
    main()
