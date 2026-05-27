#!/usr/bin/env python
"""Run VGGT-SLAM with loop closure on a 7-Scenes sequence, write dense TUM.

Usage:
    python evals/runners/run_vggt_slam_lc.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \\
        --max_frames 200

Fixed pipeline args (matching VGGT-SLAM paper defaults):
    submap_size=16, overlapping_window_size=1, conf_threshold=25.0,
    max_loops=1, min_disparity=0

Note — min_disparity=0 (not paper default of 50): with min_disparity=50, chess_seq01
produces zero loop closure candidates, making comparison vacuous. Setting 0 accepts all
frames so LC is actually triggered. Both this script and our_solver_dump.py use 0 so
frame selection is identical — comparison isolates algorithm parity, not keyframe selection.
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
import glob
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

logger = logging.getLogger(__name__)


def run_vggt_slam_lc(
    seq_dir: Path,
    out_tum: Path,
    max_frames: int = 200,
    submap_size: int = 16,
    overlapping_window_size: int = 1,
    conf_threshold: float = 25.0,
    max_loops: int = 1,  # LC enabled — for end-to-end ATE comparison with Phase 1
    min_disparity: float = 0.0,
    lc_thres: float = 0.95,
) -> None:
    """Run full VGGT-SLAM pipeline with LC and write dense TUM trajectory."""
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
        if isinstance(output, dict) and "image_match_ratio" in output:
            ratio = float(output["image_match_ratio"])
            _spark_similarity_log.append(ratio)
            logger.info("VGGT-SPARK image_match_ratio: %.4f (threshold 0.85, accept if >=)", ratio)

    model.register_forward_hook(_capture_similarity)

    # Collect and sort images, apply max_frames limit
    all_images = [
        f for f in glob.glob(str(seq_dir / "*"))
        if "depth" not in Path(f).name.lower()
        and "txt" not in Path(f).name.lower()
        and "db" not in Path(f).name.lower()
        and Path(f).suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    all_images = utils.sort_images_by_number(all_images)[:max_frames]
    logger.info("Found %d images (limited to %d)", len(all_images), max_frames)

    # Optical flow keyframe selection + submap processing (mirrors main.py)
    image_names_subset: list[str] = []
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
                import torch as _torch
                predictions["frames_lc"] = predictions["frames_lc"].to(_torch.float32)
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

    # Write VGGT-SPARK similarity scores for comparison against our cross_frame_attention_ratio
    out_similarity = out_tum.parent.parent.parent / "results" / "parity_harness" / "vggt_spark_similarity.json"
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
        default="evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum",
        help="Output TUM path.",
    )
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument("--max_loops", type=int, default=1)
    parser.add_argument(
        "--min_disparity",
        type=float,
        default=0.0,
        help="Min optical-flow disparity for keyframe selection (0 = accept all frames).",
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
    )


if __name__ == "__main__":
    main()
