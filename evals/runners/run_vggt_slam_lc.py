#!/usr/bin/env python
"""Run VGGT-SLAM on a 7-Scenes sequence, write dense TUM trajectory + ATE metrics.

Usage (full-sequence baseline, matching VGGT-SLAM paper defaults):
    python evals/runners/run_vggt_slam_lc.py \\
        --seq_dir data/7scenes/chess/seq-01

Usage (legacy 200-frame LC comparison):
    python evals/runners/run_vggt_slam_lc.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \\
        --max_frames 200 --min_disparity 0 --max_loops 1

Default args match VGGT-SLAM paper: submap_size=16, min_disparity=50, max_loops=0 (no LC).
Saves selected_frames.txt alongside TUM so eval_gt.py --keyframe_list can use same frames.
"""
from __future__ import annotations

# ── VGGT-SPARK shadow ─────────────────────────────────────────────────────────
# Shadow the installed vggt package (VGGT-X based) with VGGT-SPARK, which adds
# native compute_similarity=True support to VGGT.forward(). Must come before any
# vggt import. Affects this process only — reconstruction env is unaffected.
import sys
from pathlib import Path as _Path
_vggt_spark = str(_Path(__file__).resolve().parents[2] / "third_party" / "vggt_spark")
if _vggt_spark not in sys.path:
    sys.path.insert(0, _vggt_spark)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm

# Add repo root and VGGT-SLAM to path
_repo_root = str(Path(__file__).resolve().parents[2])
_slam_root = str(Path(__file__).resolve().parents[2] / "third_party" / "VGGT-SLAM")
for _p in (_repo_root, _slam_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT  # resolves to VGGT-SPARK via sys.path shadow above
import importlib.util as _ilu
_ate_spec = _ilu.spec_from_file_location(
    "ate_utils", Path(__file__).resolve().parents[2] / "evals" / "ate_utils.py"
)
_ate_mod = _ilu.module_from_spec(_ate_spec)
_ate_spec.loader.exec_module(_ate_mod)
compute_ate_rmse = _ate_mod.compute_ate_rmse

# lc_parity_common shares the same shadowing problem as ate_utils above (installed
# `evals` package would otherwise win), so load it the same way. Register in
# sys.modules before exec: lc_parity_common's SceneSpec dataclass resolves its
# (stringified, via `from __future__ import annotations`) field annotations
# through sys.modules[cls.__module__], which is unset until we do this.
_lc_common_spec = _ilu.spec_from_file_location(
    "lc_parity_common", Path(__file__).resolve().parent / "lc_parity_common.py"
)
_lc_common_mod = _ilu.module_from_spec(_lc_common_spec)
sys.modules["lc_parity_common"] = _lc_common_mod
_lc_common_spec.loader.exec_module(_lc_common_mod)
collect_frames = _lc_common_mod.collect_frames

logger = logging.getLogger(__name__)


def run_vggt_slam_lc(
    seq_dir: Path,
    out_tum: Path,
    max_frames: int | None = None,   # None = all frames (VGGT-SLAM paper default)
    submap_size: int = 16,
    overlapping_window_size: int = 1,
    conf_threshold: float = 25.0,
    max_loops: int = 0,              # 0 = no LC; 1 for LC runs
    min_disparity: float = 50.0,     # VGGT-SLAM paper default
    lc_thres: float = 0.95,
    image_list: Path | None = None,  # restrict frame universe (TUM GT-gap parity)
) -> None:
    """Run VGGT-SLAM pipeline and write TUM trajectory + ATE metrics + keyframe list."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    solver = Solver(
        init_conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        vis_voxel_size=None,
    )

    logger.info("Loading VGGT model...")
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    _vggt = VGGT()
    _vggt.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    _vggt.eval()
    _vggt = _vggt.to(dtype).to(device)

    model = _vggt  # VGGT-SPARK natively handles compute_similarity=True

    # Capture image_match_ratio each time model is called with compute_similarity=True.
    # Forward hook fires on every call; filter by presence of "image_match_ratio" key.
    _spark_similarity_log: list[float] = []

    def _capture_similarity(_module: torch.nn.Module, _inp: tuple, output: dict) -> None:
        if isinstance(output, dict) and output.get("image_match_ratio") is not None:
            ratio = float(output["image_match_ratio"])
            _spark_similarity_log.append(ratio)
            logger.info("VGGT-SPARK image_match_ratio: %.4f (threshold 0.85, accept if >=)", ratio)

    model.register_forward_hook(_capture_similarity)

    # Collect and sort images (7-Scenes flat or TUM rgb/ layout); optional image_list
    # restriction first, then max_frames cap (None = full sequence)
    all_images = collect_frames(seq_dir, image_list=image_list, max_frames=max_frames)
    logger.info("Found %d images (image_list=%s, max_frames=%s)",
                len(all_images), image_list, max_frames)

    # Optical flow keyframe selection + submap processing (mirrors main.py)
    image_names_subset: list[str] = []
    selected_image_paths: list[str] = []   # all accepted keyframes, for export
    image_count = 0

    for image_name in tqdm(all_images, desc="Frames"):
        img = cv2.imread(image_name)
        if img is None:
            continue
        # vis_flow=False; bypass filter entirely when min_disparity=0
        enough_disparity = (
            min_disparity <= 0.0
            or solver.flow_tracker.compute_disparity(img, min_disparity, False)
        )
        if enough_disparity:
            image_names_subset.append(image_name)
            selected_image_paths.append(image_name)
            image_count += 1

        # Process submap when full or on last frame
        if (len(image_names_subset) == submap_size + overlapping_window_size
                or image_name == all_images[-1]):
            if not image_names_subset:
                continue
            # clip_model/clip_preprocess=None: skip semantic embeddings (not needed for pose eval)
            predictions = solver.run_predictions(
                image_names_subset, model, max_loops,
                clip_model=None, clip_preprocess=None,
            )
            # solver.add_points() calls .numpy() on frames_lc; numpy rejects BFloat16.
            # Cast to float32 here rather than touching vendored solver.py.
            if predictions.get("frames_lc") is not None:
                predictions["frames_lc"] = predictions["frames_lc"].to(torch.float32)
            solver.add_points(predictions)
            solver.graph.optimize()
            # Keep last overlapping_window_size frames for next submap continuity
            image_names_subset = image_names_subset[-overlapping_window_size:]

    logger.info("Processed %d keyframes, %d submaps", image_count, solver.map.get_num_submaps())
    logger.info("Loop closures: %d", solver.graph.get_num_loops())

    # Write dense TUM trajectory (kitti_format=False → TUM format with timestamps)
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    solver.map.write_poses_to_file(str(out_tum), solver.graph, kitti_format=False)
    logger.info("Written: %s", out_tum)

    # Save selected keyframe paths — eval_gt.py --keyframe_list uses this for parity runs
    kf_path = out_tum.parent / "selected_frames.txt"
    kf_path.write_text("\n".join(selected_image_paths))
    logger.info("Keyframes (%d) → %s", len(selected_image_paths), kf_path)

    # Compute ATE against 7-Scenes GT
    ate_rmse: float | None = None
    try:
        ate_rmse = compute_ate_rmse(out_tum, seq_dir, selected_frames_path=kf_path)
        logger.info("ATE RMSE: %.6f m", ate_rmse)
    except Exception as exc:
        logger.warning("ATE computation failed: %s", exc)

    metrics_out = out_tum.parent / "metrics.json"
    metrics_out.write_text(json.dumps({
        "ate_rmse": ate_rmse,
        "keyframes": len(selected_image_paths),
        "submaps": solver.map.get_num_submaps(),
        "loop_closures": solver.graph.get_num_loops(),
        "min_disparity": min_disparity,
        "max_frames": max_frames,
    }, indent=2))
    logger.info("Metrics → %s", metrics_out)

    # Write VGGT-SPARK similarity scores for comparison against our cross_frame_attention_ratio.
    # Lives next to this run's own metrics.json (not a shared path) so parallel/prefix
    # scene runs don't clobber each other's similarity dumps.
    out_similarity = out_tum.parent / "vggt_spark_similarity.json"
    out_similarity.parent.mkdir(parents=True, exist_ok=True)
    out_similarity.write_text(json.dumps({
        "model": "VGGT-SPARK (VGGT-1B weights)",
        "metric": "image_match_ratio",
        "threshold": 0.85,
        "note": "computed via attention K/Q in aggregator._process_global_attention",
        "scores": _spark_similarity_log,
    }, indent=2))
    logger.info("VGGT-SPARK similarity scores (%d LC calls) → %s", len(_spark_similarity_log), out_similarity)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Run VGGT-SLAM with LC on a 7-Scenes sequence, write dense TUM."
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory.")
    parser.add_argument(
        "--out_tum",
        default="evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum",
        help="Output TUM path.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Max frames. Default None = full sequence. Pass 200 for legacy 200-frame runs.",
    )
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument(
        "--max_loops", type=int, default=0,
        help="Max LC per submap. 0 = disable (default). 1 for LC runs.",
    )
    parser.add_argument(
        "--min_disparity",
        type=float,
        default=50.0,
        help="Optical-flow disparity threshold for keyframe selection. "
             "VGGT-SLAM paper default=50. Use 0 to accept all frames.",
    )
    parser.add_argument(
        "--lc_thres", type=float, default=0.95,
        help="DINO-SALAD retrieval threshold for LC candidates (VGGT-SLAM default 0.95).",
    )
    parser.add_argument(
        "--image_list", type=Path, default=None,
        help="File of allowed frame basenames (one per line). Restricts keyframe "
             "selection to these frames — required for TUM parity runs, where the "
             "eval-side dataset loader drops GT-gap frames.",
    )
    args = parser.parse_args()

    run_vggt_slam_lc(
        seq_dir=Path(args.seq_dir),
        out_tum=Path(args.out_tum),
        max_frames=args.max_frames,
        submap_size=args.submap_size,
        conf_threshold=args.conf_threshold,
        max_loops=args.max_loops,
        min_disparity=args.min_disparity,
        lc_thres=args.lc_thres,
        image_list=args.image_list,
    )


if __name__ == "__main__":
    main()
