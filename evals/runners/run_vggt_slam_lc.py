#!/usr/bin/env python
"""Run VGGT-SLAM with loop closure on a 7-Scenes sequence, write dense TUM.

Usage:
    python evals/runners/run_vggt_slam_lc.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \\
        --max_frames 200

Fixed pipeline args (matching VGGT-SLAM paper defaults):
    submap_size=16, overlapping_window_size=1, conf_threshold=25.0,
    max_loops=1, min_disparity=50
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
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
from vggt.models.vggt import VGGT


def run_vggt_slam_lc(
    seq_dir: Path,
    out_tum: Path,
    max_frames: int = 200,
    submap_size: int = 16,
    overlapping_window_size: int = 1,
    conf_threshold: float = 25.0,
    max_loops: int = 1,
    min_disparity: float = 50.0,
    lc_thres: float = 0.95,
) -> None:
    """Run full VGGT-SLAM pipeline with LC and write dense TUM trajectory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Solver handles its own dtype selection internally per device capability
    solver = Solver(
        init_conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        vis_voxel_size=None,
    )

    print("Loading VGGT model...")
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(torch.bfloat16)
    model = model.to(device)

    # Collect and sort images, apply max_frames limit
    all_images = [
        f for f in glob.glob(str(seq_dir / "*"))
        if "depth" not in Path(f).name.lower()
        and "txt" not in Path(f).name.lower()
        and "db" not in Path(f).name.lower()
        and Path(f).suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    all_images = utils.sort_images_by_number(all_images)[:max_frames]
    print(f"Found {len(all_images)} images (limited to {max_frames})")

    # Optical flow keyframe selection + submap processing (mirrors main.py)
    image_names_subset: list[str] = []
    image_count = 0

    for image_name in tqdm(all_images, desc="Frames"):
        img = cv2.imread(image_name)
        # vis_flow=False: no display during batch run
        enough_disparity = solver.flow_tracker.compute_disparity(img, min_disparity, False)
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
            solver.add_points(predictions)
            solver.graph.optimize()
            # Keep last overlapping_window_size frames for next submap continuity
            image_names_subset = image_names_subset[-overlapping_window_size:]

    print(f"Processed {image_count} keyframes, {solver.map.get_num_submaps()} submaps")
    print(f"Loop closures: {solver.graph.get_num_loops()}")

    # Write dense TUM trajectory (kitti_format=False → TUM format with timestamps)
    out_tum.parent.mkdir(parents=True, exist_ok=True)
    solver.map.write_poses_to_file(str(out_tum), solver.graph, kitti_format=False)
    print(f"Written: {out_tum}")


def main() -> None:
    """CLI entry point."""
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
    args = parser.parse_args()

    run_vggt_slam_lc(
        seq_dir=Path(args.seq_dir),
        out_tum=Path(args.out_tum),
        max_frames=args.max_frames,
        submap_size=args.submap_size,
        conf_threshold=args.conf_threshold,
        max_loops=args.max_loops,
    )


if __name__ == "__main__":
    main()
