#!/usr/bin/env python
"""Compare evo ATE (SLAM's method) vs our umeyama_align on SLAM's own poses."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from evals.ate_utils import compute_ate_rmse

seq_dir = Path("evals/data/7scenes/chess/chess/seq-01")
slam_tum = Path("evals/baselines/disparity_sweep/slam_d50/baseline.tum")
kf_list = Path("evals/baselines/disparity_sweep/slam_d50/selected_frames.txt")

ate_evo = compute_ate_rmse(slam_tum, seq_dir, selected_frames_path=kf_list)
print(f"SLAM poses via evo ATE: {ate_evo:.4f}m")

# Now test our ate_translation on SLAM's poses
from collab_splats.geometry.loop_closure.eval import ate_translation
import numpy as np
from scipy.spatial.transform import Rotation

# Load SLAM TUM (cam-to-world: tx ty tz qx qy qz qw)
poses_c2w = []
with open(slam_tum) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        poses_c2w.append(T)

poses_c2w = np.stack(poses_c2w)
# Convert to world-to-cam for our ate_translation
poses_w2c = np.linalg.inv(poses_c2w)

# Load GT world-to-cam for same frames
selected = [Path(p) for p in kf_list.read_text().splitlines() if p.strip()]
gt_w2c = []
for f in selected:
    pose_file = seq_dir / f"{f.stem.split('.')[0]}.pose.txt"
    c2w = np.loadtxt(pose_file).astype(np.float32)
    gt_w2c.append(np.linalg.inv(c2w))
gt_w2c = np.stack(gt_w2c)

result = ate_translation(poses_w2c, gt_w2c)
print(f"SLAM poses via our ate_translation (fixed): {result['rmse']:.4f}m")
print(f"Expected ~0.0224m — match = {abs(result['rmse'] - ate_evo) < 0.005}")
