"""Diagnostic: trace run_pose_graph_optimization step-by-step for d=10 test case.

Prints at each stage:
  - submap metadata (n_frames, world_points present?)
  - scale estimation inputs and result
  - H_overlap, T, H_scale, H_w at each boundary
  - H_inner chain (first 3 frames of each submap)
  - pre- and post-opt homographies for first 3 frames
  - decompose_camera output (R, t) for first 3 frames
"""
import sys
import numpy as np

sys.path.insert(0, "/workspace/collab-splats")

# ── Monkey-patch run_pose_graph_optimization to be verbose ──────────────────

from collab_splats.pointcloud.loop_closure import closure as _closure_mod
from collab_splats.pointcloud.loop_closure.closure import (
    run_pose_graph_optimization as _orig_pgo,
)
from collab_splats.pointcloud.loop_closure.graph import (
    PoseGraph as _SL4PoseGraph, decompose_camera, estimate_scale_pairwise
)
import collab_splats.pointcloud.loop_closure.closure as closure_mod

# ── Run the eval pipeline with a debug_out hook ──────────────────────────────

import json
from pathlib import Path

# Load keyframe list
kf_file = Path("evals/baselines/disparity_sweep/slam_d10/selected_frames.txt")
keyframes = [l.strip() for l in kf_file.read_text().splitlines() if l.strip()]
print(f"Keyframes: {len(keyframes)} frames")

# We'll monkeypatch run_pose_graph_optimization to print steps
_orig = closure_mod.run_pose_graph_optimization

def _verbose_pgo(submaps, lc_submaps, total_frames, overlap_frames,
                 manifold="sl4", conf_threshold=25.0, scale_method="se3",
                 debug_out=None):
    print(f"\n{'='*60}")
    print(f"run_pose_graph_optimization: {len(submaps)} submaps, overlap={overlap_frames}")
    print(f"  scale_method={scale_method}, conf_threshold={conf_threshold}")
    for i, s in enumerate(submaps):
        wp_shape = s.world_points.shape if s.world_points is not None else None
        print(f"  submap[{i}]: id={s.submap_id} frames={s.poses.shape[0]} "
              f"start={s.frame_start} wp={wp_shape} "
              f"poses[0]≈I={np.allclose(s.poses[0], np.eye(4), atol=0.01)}")
    print()

    # Run with debug_out
    my_debug = []
    result = _orig(submaps, lc_submaps, total_frames, overlap_frames,
                   manifold=manifold, conf_threshold=conf_threshold,
                   scale_method=scale_method, debug_out=my_debug)

    print(f"\n--- Debug entries ({len(my_debug)} boundaries) ---")
    for entry in my_debug:
        sid = entry.get("submap_id")
        T = entry.get("T")
        scale = entry.get("scale")
        H_overlap = entry.get("H_overlap")
        H_w = entry.get("H_w")
        H_opt = entry.get("H_opt")
        print(f"\n  Submap {sid} boundary:")
        print(f"    scale = {scale:.6f}")
        print(f"    T diag = {np.diag(T)[:4] if T is not None else None}")
        if H_overlap is not None:
            print(f"    H_overlap[0,:4] = {H_overlap[0]}")
        if H_w is not None:
            print(f"    H_w[0,:4] = {H_w[0]}")
            print(f"    det(H_w) = {np.linalg.det(H_w):.6f}")
        if H_opt is not None:
            print(f"    H_opt (post-PGO)[0,:4] = {H_opt[0]}")

    # Print first 3 corrected poses per submap
    print(f"\n--- Corrected extrinsics (first 3 frames per submap region) ---")
    for i, s in enumerate(submaps):
        start = s.frame_start
        k = min(3, s.poses.shape[0])
        print(f"  submap[{i}] (global frames {start}..{start+k-1}):")
        for fi in range(k):
            g = start + fi
            R = result[g, :3, :3]
            t = result[g, :3, 3]
            print(f"    frame {g}: t={t}, R_trace={np.trace(R):.4f}")

    return result

closure_mod.run_pose_graph_optimization = _verbose_pgo

# ── Actually run the pipeline ─────────────────────────────────────────────────
import evals.eval_gt as eg

print("\nRunning eval pipeline with verbose PGO...\n")
sys.argv = [
    "eval_gt.py",
    "--dataset", "7scenes",
    "--seq_dir", "evals/data/7scenes/chess/chess/seq-01",
    "--backbone", "vggt_spark",
    "--conditions", "baseline",
    "--submap_size", "16",
    "--lc_scale_method", "rotation_only",
    "--keyframe_list", str(kf_file),
    "--output_ate", "/tmp/d10_debug.json",
]
eg.main()

print("\n--- ATE ---")
print(json.loads(Path("/tmp/d10_debug.json").read_text()))
