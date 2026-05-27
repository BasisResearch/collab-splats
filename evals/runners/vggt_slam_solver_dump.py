#!/usr/bin/env python
"""Run VGGT-SLAM (max_loops=0 for sequential only) and dump per-boundary solver internals to JSON.

Captures per inter-submap boundary: T, scale, H_overlap, H_w, H_opt.
H_opt is read after graph.optimize() completes the full pipeline.
Scale extracted algebraically: H_w = H_overlap @ T @ diag(s,s,s,1) → s = (inv(T)@inv(H_overlap)@H_w)[0,0]

Usage:
    python evals/runners/vggt_slam_solver_dump.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --out_json evals/results/parity_harness/vggt_slam_internals.json \\
        --max_frames 200

Output schema:
    {
      "config": {"seq_dir": ..., "max_frames": 200, "submap_size": 16, "conf_threshold": 25.0},
      "boundaries": [
        {"boundary_idx": 0, "submap_id": N, "scale": float,
         "T": [[4x4]], "H_overlap": [[4x4]], "H_w": [[4x4]], "H_opt": [[4x4]]}
      ],
      "final_poses": [[[4x4]], ...]
    }
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Insert repo root and VGGT-SLAM root into sys.path before any vggt_slam imports.
_repo_root = str(Path(__file__).resolve().parents[2])
_slam_root = str(Path(__file__).resolve().parents[2] / "third_party" / "VGGT-SLAM")
for _p in (_repo_root, _slam_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


########################################################################
# DumpSolver: Solver subclass that captures inter-submap boundary data
########################################################################


class DumpSolver(Solver):
    """Solver subclass that captures per-boundary internals during add_edge.

    Overrides add_edge to snapshot T, H_overlap, H_w, and scale at each
    sequential inter-submap boundary. finalize_dumps() appends H_opt after
    the final graph.optimize() call completes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Accumulates one dict per inter-submap boundary.
        self._boundary_dumps: list[dict] = []

    def add_edge(
        self,
        submap_id_curr: int,
        frame_id_curr: int,
        submap_id_prev: int | None = None,
        frame_id_prev: int | None = None,
        is_loop_closure: bool = False,
    ) -> None:
        """Override to capture inter-submap boundary data before/after super call."""
        # Capture pre-call values for sequential inter-submap edges only.
        _capture = submap_id_prev is not None and not is_loop_closure
        if _capture:
            curr_sub = self.map.get_submap(submap_id_curr)
            prior_sub = self.map.get_submap(submap_id_prev)
            overlap_node = submap_id_prev + frame_id_prev

            # H_overlap: graph estimate at prior submap's overlapping frame.
            H_overlap = self.graph.get_homography(overlap_node).copy()

            # T: relative pose — same computation as solver's P_temp (no scale yet).
            T = np.linalg.inv(prior_sub.proj_mats[-1]) @ curr_sub.proj_mats[0]

        super().add_edge(
            submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure
        )

        if _capture:
            # H_w: graph estimate for the first node of the new submap, set by super().
            H_w = self.graph.get_homography(submap_id_curr + frame_id_curr).copy()

            # Extract scale algebraically: H_w = H_overlap @ T @ diag(s,s,s,1).
            try:
                H_scale = np.linalg.inv(T) @ np.linalg.inv(H_overlap) @ H_w
                scale = float(H_scale[0, 0])
            except np.linalg.LinAlgError:
                scale = float("nan")

            self._boundary_dumps.append({
                "boundary_idx": len(self._boundary_dumps),
                "submap_id": submap_id_curr,
                "scale": scale,
                "T": T.tolist(),
                "H_overlap": H_overlap.tolist(),
                "H_w": H_w.tolist(),
                # H_opt filled in by finalize_dumps() after graph.optimize().
            })

    def finalize_dumps(self) -> None:
        """Capture post-optimization H_opt for each boundary's first node.

        Must be called after the final graph.optimize() to get the optimized
        homography at the first frame of each new submap.
        """
        for entry in self._boundary_dumps:
            # First frame of this submap has node_id == submap_id + 0.
            node_id = entry["submap_id"]
            try:
                entry["H_opt"] = self.graph.get_homography(node_id).tolist()
            except (KeyError, IndexError, RuntimeError) as exc:
                logger.warning("H_opt capture failed for submap %s: %s", entry["submap_id"], exc)
                entry["H_opt"] = None

    def get_final_poses_c2w(self) -> list[list]:
        """Return list of (4, 4) c2w pose lists for all non-LC frames.

        Mirrors map.get_all_cam_matricies(graph, give_camera_mat=True).
        """
        all_poses = []
        for submap in self.map.ordered_submaps_by_key():
            # Skip loop-closure submaps — they are auxiliary, not trajectory frames.
            if submap.get_lc_status():
                continue
            poses_world = submap.get_all_poses_world(self.graph, give_camera_mat=True)
            for pose in poses_world:
                all_poses.append(np.array(pose).tolist())
        return all_poses


########################################################################
# Pipeline runner
########################################################################


def run_dump(
    seq_dir: Path,
    out_json: Path,
    max_frames: int = 200,
    submap_size: int = 16,
    conf_threshold: float = 25.0,
    max_loops: int = 0,  # Sequential edges only — isolates stitching without LC
) -> None:
    """Run VGGT-SLAM pipeline and write boundary dump JSON.

    Args:
        seq_dir: Directory containing sequential image frames.
        out_json: Output JSON file path for boundary internals.
        max_frames: Maximum number of frames to process.
        submap_size: Number of frames per submap.
        conf_threshold: Confidence threshold for point filtering (percentile).
        max_loops: Maximum loop closures (0 = sequential only).
    """
    # Set up device and dtype.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
    else:
        dtype = torch.float32

    # Build solver with dump capture enabled.
    solver = DumpSolver(
        init_conf_threshold=conf_threshold,
        lc_thres=0.95,
        vis_voxel_size=None,
    )

    # Load VGGT model.
    logger.info("Loading VGGT model...")
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(dtype).to(device)

    # Collect and sort image paths; exclude depth/annotation files.
    _image_exts = {".png", ".jpg", ".jpeg"}
    all_images = [
        f
        for f in glob.glob(str(seq_dir / "*"))
        if Path(f).suffix.lower() in _image_exts
        and "depth" not in Path(f).name.lower()
    ]
    all_images = sorted(all_images)[:max_frames]
    logger.info("Found %d images (limited to %d)", len(all_images), max_frames)

    # Run VGGT-SLAM submap loop; mirrors main.py from VGGT-SLAM.
    overlapping_window_size = 1
    image_names_subset: list[str] = []

    for image_name in tqdm(all_images, desc="Frames"):
        # Accumulate frames that pass the disparity / motion filter.
        img = cv2.imread(image_name)
        if img is None:
            continue
        if solver.flow_tracker.compute_disparity(img, 50.0, False):
            image_names_subset.append(image_name)

        # Flush a complete submap when we have enough frames, or on the last image.
        submap_full = len(image_names_subset) == submap_size + overlapping_window_size
        is_last = image_name == all_images[-1]
        if submap_full or is_last:
            if not image_names_subset:
                continue
            predictions = solver.run_predictions(
                image_names_subset,
                model,
                max_loops,
                clip_model=None,
                clip_preprocess=None,
            )
            solver.add_points(predictions)
            solver.graph.optimize()
            # Carry overlap window into the next submap.
            image_names_subset = image_names_subset[-overlapping_window_size:]

    # Capture post-optimization H_opt for all boundaries.
    solver.finalize_dumps()
    final_poses_c2w = solver.get_final_poses_c2w()

    # Write JSON output.
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seq_dir": str(seq_dir),
            "max_frames": max_frames,
            "submap_size": submap_size,
            "conf_threshold": conf_threshold,
        },
        "boundaries": solver._boundary_dumps,
        "final_poses": final_poses_c2w,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("Written %d boundaries → %s", len(solver._boundary_dumps), out_json)


########################################################################
# CLI
########################################################################


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Run VGGT-SLAM and dump per-boundary solver internals to JSON."
    )
    parser.add_argument("--seq_dir", required=True, help="Path to image sequence directory.")
    parser.add_argument(
        "--out_json",
        default="evals/results/parity_harness/vggt_slam_internals.json",
        help="Output JSON path for boundary dump.",
    )
    parser.add_argument("--max_frames", type=int, default=200, help="Max frames to process.")
    parser.add_argument("--submap_size", type=int, default=16, help="Frames per submap.")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=25.0,
        help="Point confidence threshold (percentile).",
    )
    args = parser.parse_args()

    run_dump(
        seq_dir=Path(args.seq_dir),
        out_json=Path(args.out_json),
        max_frames=args.max_frames,
        submap_size=args.submap_size,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
