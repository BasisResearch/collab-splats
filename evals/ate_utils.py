"""ATE computation against 7-Scenes GT poses using evo."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _load_gt_as_tum_trajectory(seq_dir: Path, selected_frames: list[Path] | None = None):
    """Load 7-Scenes GT cam-to-world poses as an evo PoseTrajectory3D.

    If selected_frames is given, load poses for exactly those files (in order)
    and assign sequential timestamps 0, 1, 2, … — matching VGGT-SLAM's TUM scheme.
    Otherwise load all frame-*.pose.txt in the sequence directory.
    """
    from evo.core.trajectory import PoseTrajectory3D

    if selected_frames is not None:
        pose_files = [
            seq_dir / f"{Path(f).stem.split('.')[0]}.pose.txt"
            for f in selected_frames
        ]
        # Use actual frame number as timestamp — matches VGGT-SLAM TUM convention
        # (VGGT-SLAM writes frame-XXXXXX index, not sequential submap index)
        timestamps = [
            int(re.search(r"(\d+)", Path(f).stem).group(1)) for f in selected_frames
        ]
    else:
        pose_files_unsorted = list(seq_dir.glob("frame-*.pose.txt"))
        pose_files = sorted(
            pose_files_unsorted,
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )
        timestamps = [
            int(re.search(r"(\d+)", p.stem).group(1)) for p in pose_files
        ]

    positions, quats = [], []
    for pf in pose_files:
        c2w = np.loadtxt(pf)  # (4,4) cam-to-world
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # xyzw
        positions.append(t)
        quats.append(q)

    return PoseTrajectory3D(
        positions_xyz=np.array(positions, dtype=np.float64),
        orientations_quat_wxyz=np.array(quats, dtype=np.float64)[:, [3, 0, 1, 2]],  # xyzw→wxyz
        timestamps=np.array(timestamps, dtype=np.float64),
    )


def compute_ate_rmse(
    tum_path: Path,
    seq_dir: Path,
    selected_frames_path: Path | None = None,
) -> float:
    """Umeyama-aligned ATE RMSE (metres) between a TUM trajectory and 7-Scenes GT.

    tum_path: TUM file written by VGGT-SLAM (cam-to-world, sequential timestamps).
    seq_dir:  7-Scenes sequence directory containing frame-*.pose.txt files.
    selected_frames_path: path to selected_frames.txt from run_vggt_slam_lc.py.
        When provided, GT uses the same N frames with matching sequential timestamps.
    """
    from evo.tools import file_interface
    from evo.core import metrics, sync
    import evo.main_ape as main_ape

    traj_est = file_interface.read_tum_trajectory_file(str(tum_path))

    selected: list[Path] | None = None
    if selected_frames_path is not None:
        selected = [Path(p) for p in selected_frames_path.read_text().splitlines() if p.strip()]

    traj_ref = _load_gt_as_tum_trajectory(seq_dir, selected_frames=selected)

    traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.5)

    result = main_ape.ape(
        traj_ref_sync,
        traj_est_sync,
        metrics.PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return float(result.stats["rmse"])
