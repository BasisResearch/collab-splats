#!/usr/bin/env python
"""Ground-truth evaluation runner for collab-splats BA/LC pipelines.

Single-cell usage:
    python evals/eval.py \\
        --dataset   7scenes \\
        --seq_dir   /data/7scenes/chess/seq-01 \\
        --output_dir ./eval_results/chess_seq01 \\
        --max_frames 500 \\
        --conditions baseline ba lc

Grid usage (YAML-driven; serial, resume-on-metrics.json, then aggregate):
    python evals/eval.py --config evals/configs/7scenes.yaml
    python evals/eval.py --config evals/configs/7scenes.yaml --dry_run

For long sequences that exceed GPU memory in a single forward pass, use
``--submap_size N`` to enable windowed inference.  ``baseline`` becomes
windowed VGGT-X (LC pipeline with loop detection disabled) and ``ba``
wraps that with bundle adjustment.  ``lc`` always uses the full LC loop
regardless of this flag.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import Any
from datetime import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import get_dataset
from trajectory_io import read_tum
from eval_compare import collect_grid_metrics, format_markdown_rows

from collab_splats.pointcloud import get_creator
from collab_splats.geometry import BundleAdjustment, BundleAdjustmentConfig
from collab_splats.geometry.loop_closure.eval import ate_translation, rpe, auc_at_threshold
from collab_splats.geometry.loop_closure import LoopClosureConfig
from collab_splats.geometry.loop_closure.wrapper import LoopClosure

_FIXED_CONDITIONS = {"baseline", "ba", "lc"}
_COLORS = {"gt": "black", "baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green", "vggt_slam": "tab:orange"}


########################################################################
# Config-driven grid (YAML)
########################################################################


@dataclass
class EvalCell:
    """One grid cell: a backbone x condition over one dataset at fixed params."""

    dataset_name: str
    dataset_type: str
    seq_dir: Path
    keyframe_list: Path | None
    backbone: str
    condition: str
    submap_size: int | None
    max_frames: int | None
    lc_layer: int | None
    output_dir: Path


@dataclass
class EvalConfig:
    """Flat, declarative eval experiment loaded from YAML."""

    name: str
    datasets: list[dict]
    backbones: list[str]
    conditions: list[str]
    output_dir: Path
    submap_size: int | None = None
    max_frames: int | None = None
    lc_layer: int | None = None


def load_eval_config(path: Path) -> EvalConfig:
    """Parse a flat experiment YAML into an EvalConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return EvalConfig(
        name=raw["name"],
        datasets=raw["datasets"],
        backbones=raw["backbones"],
        conditions=raw["conditions"],
        output_dir=Path(raw["output_dir"]),
        submap_size=raw.get("submap_size"),
        max_frames=raw.get("max_frames"),
        lc_layer=raw.get("lc_layer"),
    )


def build_grid(cfg: EvalConfig) -> list[EvalCell]:
    """Expand the config into one EvalCell per (dataset x backbone x condition)."""
    cells = []
    for ds, backbone in itertools.product(cfg.datasets, cfg.backbones):
        for cond in cfg.conditions:
            cells.append(
                EvalCell(
                    dataset_name=ds["name"],
                    # Loader key for --dataset; defaults to the label when the
                    # entry's name already IS the loader type (e.g. "7scenes").
                    dataset_type=ds.get("type", ds["name"]),
                    seq_dir=Path(ds["seq_dir"]),
                    keyframe_list=(Path(ds["keyframe_list"]) if ds.get("keyframe_list") else None),
                    backbone=backbone,
                    condition=cond,
                    submap_size=cfg.submap_size,
                    max_frames=cfg.max_frames,
                    lc_layer=cfg.lc_layer,
                    output_dir=cfg.output_dir,
                )
            )
    return cells


def _validate_condition(cond: str) -> None:
    """Raise ValueError if cond is not a recognised condition string."""
    if cond in _FIXED_CONDITIONS:
        return
    m = re.fullmatch(r"ba_track-density-(\d+)", cond)
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError(f"ba_track-density-{{N}} requires N > 0, got {cond!r}")
        return
    m2 = re.fullmatch(r"incremental_ba-(\d+)", cond)
    if m2:
        n = int(m2.group(1))
        if n <= 0:
            raise ValueError(f"incremental_ba-{{N}} requires N > 0, got {cond!r}")
        return
    raise ValueError(
        f"Unknown condition {cond!r}. "
        f"Valid: {sorted(_FIXED_CONDITIONS)} or ba_track-density-{{N}} "
        f"or incremental_ba-{{N}} (e.g. incremental_ba-5)"
    )


def _default_output_dir(dataset: str, seq_dir: Path) -> Path:
    """Auto-generate a timestamped output directory under evals/results/."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("evals/results") / dataset / seq_dir.name / f"run-{ts}"


def _prepare_image_dir(image_paths: list[Path]) -> Path:
    """Symlink image_paths into a fresh temp dir named 000000.png, 000001.png, ..."""
    tmp = Path(tempfile.mkdtemp(prefix="collab_eval_"))
    for i, src in enumerate(image_paths):
        (tmp / f"{i:06d}.png").symlink_to(src.resolve())
    return tmp


def _cam_positions(poses: np.ndarray) -> np.ndarray:
    """Convert (N,4,4) world-to-cam poses to (N,3) camera positions in world."""
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def _write_tum(path: Path, poses_w2c: np.ndarray) -> None:
    """Write TUM trajectory: 'timestamp tx ty tz qx qy qz qw' (camera-to-world)."""
    from scipy.spatial.transform import Rotation

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, w2c in enumerate(poses_w2c):
        c2w = np.linalg.inv(w2c.astype(np.float64))
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [qx, qy, qz, qw]
        lines.append(f"{i:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} " f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    path.write_text("\n".join(lines) + "\n")


_BACKBONE_PREFIX = {"vggt_omega": "omega", "vggtx": "vggtx", "mapanything": "mapanything", "vggt_spark": "spark"}


def _make_creator(
    condition: str,
    submap_size: int | None = None,
    backbone: str = "vggt_omega",
    lc_scale_method: str = "se3",
    max_loops_per_submap: int | None = None,
):
    """Build a (creator, ba_config) pair for the given condition.

    Returns (creator, None) when no bundle adjustment is needed.
    Returns (creator, BundleAdjustmentConfig) when BA should run after postprocess.
    """
    # Optional LoopClosureConfig override shared by every construction below;
    # None = keep the LoopClosureConfig class default.
    _lc_extra = {} if max_loops_per_submap is None else {"max_loops_per_submap": max_loops_per_submap}
    base = get_creator(backbone)()
    if condition == "lc":
        lc_cfg = LoopClosureConfig(
            scale_method=lc_scale_method, **_lc_extra, **({} if submap_size is None else {"submap_size": submap_size})
        )
        return LoopClosure(base, config=lc_cfg), None
    m = re.fullmatch(r"ba_track-density-(\d+)", condition)
    if m:
        n = int(m.group(1))
        cfg = BundleAdjustmentConfig(
            max_query_pts=n,
            query_frame_num=max(5, n // 512),
        )
        if submap_size is not None:
            _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_retrieval_threshold=0.0, **_lc_extra)
            windowed = LoopClosure(base, config=_no_lc_cfg)
            return windowed, cfg
        return base, cfg
    m2 = re.fullmatch(r"incremental_ba-(\d+)", condition)
    if m2:
        increment_size = int(m2.group(1))
        cfg = BundleAdjustmentConfig(increment_size=increment_size)
        if submap_size is not None:
            _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_retrieval_threshold=0.0, **_lc_extra)
            windowed = LoopClosure(base, config=_no_lc_cfg)
            return windowed, cfg
        return base, cfg
    if submap_size is not None:
        # Windowed mode: LC pipeline with detection disabled so baseline = windowed VGGT-X
        _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_retrieval_threshold=0.0, **_lc_extra)
        windowed = LoopClosure(base, config=_no_lc_cfg)
        if condition == "ba":
            return windowed, BundleAdjustmentConfig()
        return windowed, None  # baseline
    # Default: single-pass (short sequences that fit in GPU memory)
    if condition == "ba":
        return base, BundleAdjustmentConfig()
    return base, None  # baseline


def _run_condition(
    name: str,
    image_dir: Path,
    output_dir: Path,
    submap_size: int | None = None,
    backbone: str = "vggt_omega",
    lc_scale_method: str = "se3",
    max_loops_per_submap: int | None = None,
) -> tuple[np.ndarray, Any]:
    """Run condition, return (extrinsics (N,4,4), creator)."""
    creator, ba_cfg = _make_creator(
        name,
        submap_size=submap_size,
        backbone=backbone,
        lc_scale_method=lc_scale_method,
        max_loops_per_submap=max_loops_per_submap,
    )
    if ba_cfg is None:
        creator.reconstruct(image_dir, output_dir)
    else:
        # Run inference, refine poses with BA, reproject pts3d, then write COLMAP output
        ba = BundleAdjustment(ba_cfg)
        creator.load_model()
        creator.setup_inference(image_dir)
        creator.run_inference()
        creator.postprocess()
        creator.outputs = ba.refine(creator.outputs).reproject()
        creator.build_colmap(output_dir)
    if creator.outputs is None:
        raise RuntimeError(f"Condition '{name}' produced no outputs")
    return creator.outputs.extrinsics, creator


def _save_outputs(
    metrics: dict,
    trajectories: dict[str, np.ndarray],
    output_dir: Path,
    config: dict | None = None,
) -> None:
    """Write metrics.json, trajectories.npz, and two plot PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json — drop non-serializable per_frame array; include run config
    metrics_json: dict = {}
    if config:
        metrics_json["_config"] = config
    for cond, m in metrics.items():
        metrics_json[cond] = {
            "ate": {k: v for k, v in m["ate"].items() if k != "per_frame"},
            "rpe": m["rpe"],
            "auc": {k: v for k, v in m["auc"].items() if k != "per_pair_err"},
            "time_s": m.get("time_s", None),
        }
        # LC summary: number of accepted loop-closure submaps applied.
        if "n_loops_applied" in m:
            metrics_json[cond]["n_loops_applied"] = m["n_loops_applied"]
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    # trajectories.npz — poses + pre-computed per-frame ATE for notebook
    npz_data = {"gt": trajectories["gt"]}
    for cond in metrics:
        if cond in trajectories:
            npz_data[f"pred_{cond}"] = trajectories[cond]
        per_frame = metrics[cond]["ate"].get("per_frame")
        if per_frame is not None:
            npz_data[f"ate_per_frame_{cond}"] = per_frame
    np.savez(output_dir / "trajectories.npz", **npz_data)

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_trajectory(trajectories, plots_dir / "trajectory.png")
    _plot_ate_per_frame(metrics, trajectories["gt"], plots_dir / "ate_per_frame.png")


def _plot_trajectory(trajectories: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, poses in trajectories.items():
        pos = _cam_positions(poses)
        ax.plot(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            label=name,
            color=_COLORS.get(name, "gray"),
            linewidth=2 if name == "gt" else 1,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Camera Trajectory")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ate_per_frame(metrics: dict, gt: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for cond, m in metrics.items():
        per_frame = m["ate"].get("per_frame")
        if per_frame is None:
            continue
        rmse = m["ate"]["rmse"]
        ax.plot(per_frame, label=f"{cond} (RMSE={rmse:.3f}m)", color=_COLORS.get(cond, "gray"))
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.set_title("Per-frame Absolute Trajectory Error")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Grid wrapper: when --config is set, the single-cell args below are not
    # required — each cell is expanded and launched as its own single-cell run.
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment config (dataset x backbone x condition grid). "
        "When set, runs the grid serially with resume-on-metrics.json and post-grid aggregation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="With --config: print the planned per-cell command for each cell and run nothing.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name: 7scenes | tum | kitti | waymo | co3dv2")
    parser.add_argument("--seq_dir", type=Path, default=None, help="Path to sequence directory")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write results " "(default: evals/results/{dataset}/{seq_name}/run-{timestamp})",
    )
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument(
        "--submap_size",
        type=int,
        default=None,
        help="Frames per window for windowed inference. Required for sequences "
        "too long for single-pass GPU inference (e.g. >200 frames). "
        "baseline→windowed VGGT-X, ba→windowed+BA, lc→full LC pipeline.",
    )
    parser.add_argument(
        "--backbone",
        choices=["vggtx", "vggt_omega", "mapanything", "vggt_spark"],
        default="vggt_omega",
        help="Feedforward backbone. Output TUM files are prefixed: vggt_omega→omega_*, vggtx→vggtx_*, mapanything→mapanything_*, vggt_spark→spark_*",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "ba", "lc"],
        help="Conditions: baseline | ba | lc | ba_track-density-{N}",
    )
    parser.add_argument(
        "--lc_scale_method",
        choices=["se3", "rotation_only", "pairwise_dist", "none"],
        default="se3",
        help="Inter-submap scale estimation method for lc condition. "
        "se3=current (full SE3, biased), rotation_only=VGGT-SLAM style, "
        "pairwise_dist=translation-invariant fix, none=skip scale (always 1.0).",
    )
    parser.add_argument(
        "--lc_layer",
        type=int,
        default=None,
        help="Override the per-backbone LC verify layer (_lc_layer_index) for layer sweeps.",
    )
    parser.add_argument(
        "--max_loops_per_submap",
        type=int,
        default=None,
        help="Override LoopClosureConfig.max_loops_per_submap (default None = keep class "
        "default). Pass 1 for VGGT-SLAM parity runs (upstream caps at 1 loop/submap).",
    )
    parser.add_argument(
        "--keyframe_list",
        type=Path,
        default=None,
        help="Path to selected_frames.txt from run_vggt_slam_lc.py. "
        "When set, filters the dataset to only these frames (matched by filename) "
        "so all models run on the exact same keyframes as VGGT-SLAM.",
    )
    parser.add_argument(
        "--slam_tum",
        type=Path,
        default=None,
        help="Path to an upstream VGGT-SLAM TUM trajectory (e.g. .../slam/slam.tum). "
        "When set, overlays it (label 'vggt_slam') in the 3D trajectory plot only "
        "— excluded from ATE/RPE/AUC metrics and from trajectories.npz.",
    )
    # Internal flag: run exactly one condition as a subprocess and write results
    # to --_result_file as JSON.  Not part of the public API.
    parser.add_argument(
        "--output_ate",
        type=Path,
        default=None,
        help="If set, write {condition: ate_rmse} JSON to this path after all conditions complete.",
    )
    parser.add_argument("--_condition", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_image_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_result_file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _subprocess_mode(args: argparse.Namespace) -> None:
    """Run one condition and write {extrinsics, time_s, backbone, n_loops_applied} JSON."""
    _validate_condition(args._condition)
    backbone = getattr(args, "backbone", "vggt_omega")
    t0 = time.perf_counter()
    pred, creator = _run_condition(
        args._condition,
        args._image_dir,
        args.output_dir / args._condition,
        submap_size=args.submap_size,
        backbone=backbone,
        lc_scale_method=getattr(args, "lc_scale_method", "se3"),
        max_loops_per_submap=getattr(args, "max_loops_per_submap", None),
    )
    elapsed = time.perf_counter() - t0

    # Loop-closure summary (only field LC now exposes).
    n_loops_applied: int | None = None
    if hasattr(creator, "base") and hasattr(creator.base, "n_loops_applied"):
        n_loops_applied = int(creator.base.n_loops_applied)

    args._result_file.write_text(
        json.dumps(
            {
                "extrinsics": pred.tolist(),
                "time_s": round(elapsed, 2),
                "backbone": backbone,
                "n_loops_applied": n_loops_applied,
            }
        )
    )
    print(f"  time={elapsed:.1f}s")


def _build_cell_command(cell: EvalCell, cell_dir: Path) -> list[str]:
    """Build the single-cell eval.py CLI command for one grid cell.

    Reuses the existing single-cell path verbatim — one --backbone over one
    --conditions into the cell's own --output_dir, which writes metrics.json.
    """
    cmd = [
        sys.executable,
        __file__,
        "--dataset",
        cell.dataset_type,
        "--seq_dir",
        str(cell.seq_dir),
        "--output_dir",
        str(cell_dir),
        "--backbone",
        cell.backbone,
        "--conditions",
        cell.condition,
    ]
    if cell.max_frames is not None:
        cmd += ["--max_frames", str(cell.max_frames)]
    if cell.submap_size is not None:
        cmd += ["--submap_size", str(cell.submap_size)]
    if cell.lc_layer is not None:
        cmd += ["--lc_layer", str(cell.lc_layer)]
    if cell.keyframe_list is not None:
        cmd += ["--keyframe_list", str(cell.keyframe_list)]
    return cmd


def _run_grid(config_path: Path, dry_run: bool = False) -> None:
    """Load a YAML experiment, run each cell serially (resume), then aggregate."""
    cfg = load_eval_config(config_path)
    grid = build_grid(cfg)
    print(f"Grid '{cfg.name}': {len(grid)} cells -> {cfg.output_dir}")

    for cell in grid:
        cell_dir = cfg.output_dir / f"{cell.dataset_name}__{cell.backbone}__{cell.condition}"
        # Resume: a cell that already wrote metrics.json is considered done.
        if (cell_dir / "metrics.json").exists():
            print(f"  SKIP (resume): {cell_dir.name}")
            continue
        cmd = _build_cell_command(cell, cell_dir)
        if dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
            continue
        print(f"  RUN: {cell_dir.name}")
        subprocess.run(cmd, check=True)

    # Post-grid aggregation — read every cell's metrics.json into one table.
    if dry_run:
        return
    rows = collect_grid_metrics(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "comparison.md").write_text(format_markdown_rows(rows))
    (cfg.output_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"Aggregated {len(rows)} cells -> {cfg.output_dir}/comparison.md")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    # ── config-grid wrapper ────────────────────────────────────────────────────
    if args.config is not None:
        _run_grid(args.config, dry_run=args.dry_run)
        return

    if args.dataset is None or args.seq_dir is None:
        parser.error("--dataset and --seq_dir are required unless --config is given")

    for cond in args.conditions or []:
        _validate_condition(cond)
    if args._condition is not None:
        _validate_condition(args._condition)

    # Optional LC verify-layer override for sweeps — set the ClassVar on the
    # backbone creator class (applies in both orchestrator and subprocess leaf).
    if getattr(args, "lc_layer", None) is not None:
        get_creator(args.backbone)._lc_layer_index = args.lc_layer

    # ── subprocess leaf ────────────────────────────────────────────────────────
    if args._condition is not None:
        _subprocess_mode(args)
        return

    # ── auto-name output dir if not provided ──────────────────────────────────
    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.dataset, args.seq_dir)
        print(f"Output directory: {args.output_dir}")

    # ── orchestrator ───────────────────────────────────────────────────────────
    # When keyframe_list is set, load all frames so high-index keyframes aren't truncated
    _load_max = args.max_frames if args.keyframe_list is None else 100_000
    dataset = get_dataset(args.dataset)(args.seq_dir, max_frames=_load_max)

    if args.keyframe_list is not None:
        allowed_basenames = {Path(p).name for p in args.keyframe_list.read_text().splitlines() if p.strip()}
        indices = [i for i, p in enumerate(dataset.images) if Path(p).name in allowed_basenames]
        # Guard against a silent subset run: if the dataset loader dropped any requested
        # keyframe (e.g. TUM's groundtruth-gap filter in datasets.py:_load_tum), SLAM's
        # selected_frames.txt can name frames that never made it into `dataset.images`, and
        # this filter would then silently run on fewer frames than SLAM did.
        loaded_basenames = {Path(p).name for p in dataset.images}
        missing = sorted(allowed_basenames - loaded_basenames)
        if missing:
            raise ValueError(
                f"--keyframe_list requested {len(allowed_basenames)} keyframes but "
                f"{len(missing)} are missing from the loaded dataset (first 5: {missing[:5]}) — "
                "keyframes dropped by dataset loader (e.g. TUM GT-gap filter) — SLAM and ours "
                "would run different frames; parity run aborted."
            )
        from datasets import EvalDataset

        dataset = EvalDataset(
            images=[dataset.images[i] for i in indices],
            gt_poses=dataset.gt_poses[indices],
            intrinsics=dataset.intrinsics[indices] if dataset.intrinsics is not None else None,
        )
        print(f"keyframe_list: filtered to {len(indices)} frames from {args.keyframe_list.name}")

    mode = f"windowed(submap_size={args.submap_size})" if args.submap_size else "single-pass"
    print(f"Dataset: {args.dataset} | {len(dataset.images)} frames | {args.conditions} | mode={mode}")

    tmp_image_dir = _prepare_image_dir(dataset.images)
    result_dir = Path(tempfile.mkdtemp(prefix="collab_eval_results_"))
    try:
        metrics: dict = {}
        trajectories: dict[str, np.ndarray] = {"gt": dataset.gt_poses}

        for cond in args.conditions:
            print(f"\n=== Condition: {cond} ===")
            result_file = result_dir / f"{cond}.json"

            # Each condition runs in its own subprocess so the GPU starts clean.
            # The heavy VGGT-1B model (~8 GB) and VGGSfM tracker are fully freed
            # between conditions — Python GC cannot guarantee this in-process.
            cmd = [
                sys.executable,
                __file__,
                "--dataset",
                args.dataset,
                "--seq_dir",
                str(args.seq_dir),
                "--output_dir",
                str(args.output_dir),
                "--max_frames",
                str(args.max_frames),
                "--backbone",
                args.backbone,
                "--_condition",
                cond,
                "--_image_dir",
                str(tmp_image_dir),
                "--_result_file",
                str(result_file),
            ]
            if args.submap_size is not None:
                cmd += ["--submap_size", str(args.submap_size)]
            if getattr(args, "lc_scale_method", "se3") != "se3":
                cmd += ["--lc_scale_method", args.lc_scale_method]
            if getattr(args, "lc_layer", None) is not None:
                cmd += ["--lc_layer", str(args.lc_layer)]
            if getattr(args, "max_loops_per_submap", None) is not None:
                cmd += ["--max_loops_per_submap", str(args.max_loops_per_submap)]

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - t0

            result_json = json.loads(result_file.read_text())
            pred = np.array(result_json["extrinsics"], dtype=np.float32)
            time_s = result_json.get("time_s", round(elapsed, 2))

            metrics[cond] = {
                "ate": ate_translation(pred, dataset.gt_poses),
                "rpe": rpe(pred, dataset.gt_poses),
                "auc": auc_at_threshold(
                    np.linalg.inv(pred),
                    np.linalg.inv(dataset.gt_poses),
                    thresholds=(5.0, 15.0, 30.0),
                ),
                "time_s": time_s,
            }
            # LC summary: number of accepted loop-closure submaps applied.
            n_loops_applied = result_json.get("n_loops_applied")
            if n_loops_applied is not None:
                metrics[cond]["n_loops_applied"] = int(n_loops_applied)
                print(f"  Loops applied: {n_loops_applied}")
            trajectories[cond] = pred
            print(f"  ATE RMSE: {metrics[cond]['ate']['rmse']:.4f}m")
            print(
                f"  RPE trans/rot: {metrics[cond]['rpe']['trans_rmse']:.4f}m / "
                f"{metrics[cond]['rpe']['rot_rmse_deg']:.3f}deg"
            )
            print(
                f"  AUC@5/15/30: {metrics[cond]['auc']['auc_5']:.1f} / "
                f"{metrics[cond]['auc']['auc_15']:.1f} / {metrics[cond]['auc']['auc_30']:.1f}"
            )
            print(f"  Time: {time_s}s")
    finally:
        shutil.rmtree(tmp_image_dir, ignore_errors=True)
        shutil.rmtree(result_dir, ignore_errors=True)

    # Write TUM files for eval_compare.py phase-2 runner
    prefix = _BACKBONE_PREFIX.get(args.backbone, args.backbone)
    _write_tum(args.output_dir / "gt.tum", dataset.gt_poses)
    for cond, poses in trajectories.items():
        if cond == "gt":
            continue
        _write_tum(args.output_dir / f"{prefix}_{cond}.tum", poses)

    # Optional upstream SLAM reference — plot overlay only. Added after the TUM-write
    # loop above (so it isn't re-serialized to its own <prefix>_vggt_slam.tum; the
    # source file at --slam_tum is already on disk) and after the metrics loop (so it
    # never enters ate/rpe/auc computation). _save_outputs' npz loop iterates `metrics`
    # keys, not `trajectories` keys, so this extra entry is naturally excluded from
    # trajectories.npz too — it only reaches _plot_trajectory's full trajectories dict.
    if args.slam_tum is not None:
        slam_poses, _ = read_tum(args.slam_tum)
        trajectories["vggt_slam"] = slam_poses

    _save_outputs(
        metrics,
        trajectories,
        args.output_dir,
        config={
            "backbone": args.backbone,
            "max_frames": args.max_frames,
            "submap_size": args.submap_size,
            "dataset": args.dataset,
            "seq_dir": str(args.seq_dir),
        },
    )
    print(f"\nResults written to {args.output_dir}/")
    print(
        json.dumps(
            {c: {"ate_rmse": m["ate"]["rmse"], "rpe_trans": m["rpe"]["trans_rmse"]} for c, m in metrics.items()},
            indent=2,
        )
    )

    # Write per-condition ATE RMSE to JSON file if requested
    if args.output_ate is not None:
        ate_by_condition = {c: m["ate"]["rmse"] for c, m in metrics.items()}
        args.output_ate.parent.mkdir(parents=True, exist_ok=True)
        args.output_ate.write_text(json.dumps(ate_by_condition, indent=2))


if __name__ == "__main__":
    main()
