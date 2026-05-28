#!/usr/bin/env python
"""Ground-truth evaluation runner for collab-splats BA/LC pipelines.

Usage:
    python evals/eval_gt.py \\
        --dataset   7scenes \\
        --seq_dir   /data/7scenes/chess/seq-01 \\
        --output_dir ./eval_results/chess_seq01 \\
        --max_frames 500 \\
        --conditions baseline ba lc

For long sequences that exceed GPU memory in a single forward pass, use
``--submap_size N`` to enable windowed inference.  ``baseline`` becomes
windowed VGGT-X (LC pipeline with loop detection disabled) and ``ba``
wraps that with bundle adjustment.  ``lc`` always uses the full LC loop
regardless of this flag.
"""
from __future__ import annotations

import argparse
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import get_dataset

from collab_splats.pointcloud import BundleAdjustment, BundleAdjustmentConfig, get_creator
from collab_splats.pointcloud.loop_closure.eval import ate_translation, rpe, auc_at_threshold
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
from collab_splats.pointcloud.wrappers import LoopClosure

_FIXED_CONDITIONS = {"baseline", "ba", "lc"}
_COLORS = {"gt": "black", "baseline": "tab:red", "ba": "tab:blue", "lc": "tab:green"}


def _validate_condition(cond: str) -> None:
    """Raise ValueError if cond is not a recognised condition string."""
    if cond in _FIXED_CONDITIONS:
        return
    m = re.fullmatch(r"ba_track-density-(\d+)", cond)
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError(
                f"ba_track-density-{{N}} requires N > 0, got {cond!r}"
            )
        return
    m2 = re.fullmatch(r"incremental_ba-(\d+)", cond)
    if m2:
        n = int(m2.group(1))
        if n <= 0:
            raise ValueError(
                f"incremental_ba-{{N}} requires N > 0, got {cond!r}"
            )
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
        lines.append(
            f"{i:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    path.write_text("\n".join(lines) + "\n")


_BACKBONE_PREFIX = {"vggt_omega": "omega", "vggtx": "vggtx", "mapanything": "mapanything"}


def _make_creator(condition: str, submap_size: int | None = None, backbone: str = "vggt_omega"):
    """Build a (creator, ba_config) pair for the given condition.

    Returns (creator, None) when no bundle adjustment is needed.
    Returns (creator, BundleAdjustmentConfig) when BA should run after postprocess.
    """
    base = get_creator(backbone)()
    if condition == "lc":
        return LoopClosure(base), None
    m = re.fullmatch(r"ba_track-density-(\d+)", condition)
    if m:
        n = int(m.group(1))
        cfg = BundleAdjustmentConfig(
            max_query_pts=n,
            query_frame_num=max(5, n // 512),
        )
        if submap_size is not None:
            _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_retrieval_threshold=0.0)
            windowed = LoopClosure(base, config=_no_lc_cfg)
            return windowed, cfg
        return base, cfg
    m2 = re.fullmatch(r"incremental_ba-(\d+)", condition)
    if m2:
        increment_size = int(m2.group(1))
        cfg = BundleAdjustmentConfig(increment_size=increment_size)
        if submap_size is not None:
            _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_retrieval_threshold=0.0)
            windowed = LoopClosure(base, config=_no_lc_cfg)
            return windowed, cfg
        return base, cfg
    if submap_size is not None:
        # Windowed mode: LC pipeline with detection disabled so baseline = windowed VGGT-X
        _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_retrieval_threshold=0.0)
        windowed = LoopClosure(base, config=_no_lc_cfg)
        if condition == "ba":
            return windowed, BundleAdjustmentConfig()
        return windowed, None  # baseline
    # Default: single-pass (short sequences that fit in GPU memory)
    if condition == "ba":
        return base, BundleAdjustmentConfig()
    return base, None  # baseline


def _run_condition(
    name: str, image_dir: Path, output_dir: Path,
    submap_size: int | None = None,
    backbone: str = "vggt_omega",
) -> tuple[np.ndarray, Any]:
    """Run condition, return (extrinsics (N,4,4), creator)."""
    creator, ba_cfg = _make_creator(name, submap_size=submap_size, backbone=backbone)
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
            "auc_30": m["auc"]["auc_30"],
            "time_s": m.get("time_s", None),
        }
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
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                label=name, color=_COLORS.get(name, "gray"),
                linewidth=2 if name == "gt" else 1)
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
        ax.plot(per_frame, label=f"{cond} (RMSE={rmse:.3f}m)",
                color=_COLORS.get(cond, "gray"))
    ax.set_xlabel("Frame")
    ax.set_ylabel("ATE (m)")
    ax.set_title("Per-frame Absolute Trajectory Error")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name: 7scenes | tum | kitti | waymo | co3dv2")
    parser.add_argument("--seq_dir",    type=Path, required=True,
                        help="Path to sequence directory")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where to write results "
                             "(default: evals/results/{dataset}/{seq_name}/run-{timestamp})")
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--submap_size", type=int, default=None,
                        help="Frames per window for windowed inference. Required for sequences "
                             "too long for single-pass GPU inference (e.g. >200 frames). "
                             "baseline→windowed VGGT-X, ba→windowed+BA, lc→full LC pipeline.")
    parser.add_argument(
        "--backbone", choices=["vggtx", "vggt_omega", "mapanything"], default="vggt_omega",
        help="Feedforward backbone. Output TUM files are prefixed: vggt_omega→omega_*, vggtx→vggtx_*, mapanything→mapanything_*",
    )
    parser.add_argument("--conditions", nargs="+", default=["baseline", "ba", "lc"],
                        help="Conditions: baseline | ba | lc | ba_track-density-{N}")
    parser.add_argument(
        "--keyframe_list", type=Path, default=None,
        help="Path to selected_frames.txt from run_vggt_slam_lc.py. "
             "When set, filters the dataset to only these frames (matched by filename) "
             "so all models run on the exact same keyframes as VGGT-SLAM.",
    )
    # Internal flag: run exactly one condition as a subprocess and write results
    # to --_result_file as JSON.  Not part of the public API.
    parser.add_argument("--_condition",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_image_dir",   type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_result_file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _subprocess_mode(args: argparse.Namespace) -> None:
    """Run one condition and write {extrinsics, time_s, backbone, alignment} JSON."""
    _validate_condition(args._condition)
    backbone = getattr(args, "backbone", "vggt_omega")
    t0 = time.perf_counter()
    pred, creator = _run_condition(
        args._condition, args._image_dir, args.output_dir / args._condition,
        submap_size=args.submap_size,
        backbone=backbone,
    )
    elapsed = time.perf_counter() - t0

    # Compute submap alignment metrics for LC conditions
    alignment: dict = {}
    if hasattr(creator, "base") and hasattr(creator.base, "_lc_submaps"):
        try:
            from reconstruction_quality import compute_alignment_metrics
            alignment = compute_alignment_metrics(creator)
        except Exception as exc:
            print(f"  WARNING: alignment metrics failed: {exc}")

    args._result_file.write_text(json.dumps({
        "extrinsics": pred.tolist(),
        "time_s": round(elapsed, 2),
        "backbone": backbone,
        "alignment": alignment,
    }))
    print(f"  time={elapsed:.1f}s")


def main() -> None:
    args = _build_parser().parse_args()

    for cond in (args.conditions or []):
        _validate_condition(cond)
    if args._condition is not None:
        _validate_condition(args._condition)

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
        allowed_basenames = {
            Path(p).name
            for p in args.keyframe_list.read_text().splitlines()
            if p.strip()
        }
        indices = [i for i, p in enumerate(dataset.images) if Path(p).name in allowed_basenames]
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
                sys.executable, __file__,
                "--dataset",    args.dataset,
                "--seq_dir",    str(args.seq_dir),
                "--output_dir", str(args.output_dir),
                "--max_frames", str(args.max_frames),
                "--backbone",   args.backbone,
                "--_condition",   cond,
                "--_image_dir",   str(tmp_image_dir),
                "--_result_file", str(result_file),
            ]
            if args.submap_size is not None:
                cmd += ["--submap_size", str(args.submap_size)]

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - t0

            result_json = json.loads(result_file.read_text())
            pred = np.array(result_json["extrinsics"], dtype=np.float32)
            time_s = result_json.get("time_s", round(elapsed, 2))

            # Write alignment JSON for LC conditions that produced submap metrics
            alignment = result_json.get("alignment", {})
            if alignment:
                prefix = _BACKBONE_PREFIX.get(args.backbone, args.backbone)
                alignment_path = args.output_dir / f"{prefix}_{cond}_alignment.json"
                alignment_path.parent.mkdir(parents=True, exist_ok=True)
                alignment_path.write_text(json.dumps(alignment, indent=2))
                print(f"  Alignment JSON: {alignment_path}")

            metrics[cond] = {
                "ate": ate_translation(pred, dataset.gt_poses),
                "rpe": rpe(pred, dataset.gt_poses),
                "auc": auc_at_threshold(np.linalg.inv(pred), np.linalg.inv(dataset.gt_poses)),
                "time_s": time_s,
            }
            trajectories[cond] = pred
            print(f"  ATE RMSE: {metrics[cond]['ate']['rmse']:.4f}m")
            print(f"  RPE trans RMSE: {metrics[cond]['rpe']['trans_rmse']:.4f}m")
            print(f"  AUC@30: {metrics[cond]['auc']['auc_30']:.1f}")
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

    _save_outputs(metrics, trajectories, args.output_dir, config={
        "backbone": args.backbone,
        "max_frames": args.max_frames,
        "submap_size": args.submap_size,
        "dataset": args.dataset,
        "seq_dir": str(args.seq_dir),
    })
    print(f"\nResults written to {args.output_dir}/")
    print(json.dumps(
        {c: {"ate_rmse": m["ate"]["rmse"], "rpe_trans": m["rpe"]["trans_rmse"]}
         for c, m in metrics.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
