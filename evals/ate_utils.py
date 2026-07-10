"""ATE computation against 7-Scenes GT poses using evo."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _load_gt_as_tum_trajectory(seq_dir: Path, selected_frames: list[Path] | None = None):
    """Load 7-Scenes or TUM RGB-D GT cam-to-world poses as an evo PoseTrajectory3D.

    If selected_frames is given, load poses for exactly those files (in order);
    timestamps are derived from each frame's filename — the parsed stem for TUM
    scenes, the embedded frame index for 7-Scenes — matching VGGT-SLAM's own
    pred timestamps. Otherwise load all frame-*.pose.txt in the sequence directory.

    TUM RGB-D scenes (detected via a groundtruth.txt file) instead ship one
    groundtruth.txt of 'ts tx ty tz qx qy qz qw' rows at ~100Hz that do not line
    up with image timestamps, so each frame is associated with its nearest GT
    row by timestamp.
    """
    from evo.core.trajectory import PoseTrajectory3D

    gt_file = seq_dir / "groundtruth.txt"
    if gt_file.exists():
        # TUM layout: parse 'ts tx ty tz qx qy qz qw' rows (cam-to-world), skipping
        # comments/blanks — same conventions as evals/datasets.py:_read_tum_groundtruth.
        gt_rows = []
        for line in gt_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            gt_rows.append((float(vals[0]), np.array([float(v) for v in vals[1:8]], dtype=np.float64)))
        gt_timestamps = np.array([t for t, _ in gt_rows], dtype=np.float64)
        gt_vals = np.stack([v for _, v in gt_rows])  # (G, 7): tx ty tz qx qy qz qw

        # Frame filename stem IS the capture timestamp (e.g. "10.500000.png"); this is
        # exactly what VGGT-SLAM's own frame_id regex extracts (submap.py:set_frame_ids)
        # and writes per-row via write_poses_to_file, so reusing it here — rather than the
        # (slightly offset) raw GT row timestamp — keeps pred/GT timestamps aligned.
        frames = list(selected_frames) if selected_frames is not None else sorted((seq_dir / "rgb").glob("*.png"))
        timestamps = [float(Path(f).stem) for f in frames]

        # Nearest-neighbour associate each frame timestamp to a GT row, skipping frames
        # whose nearest GT row is >0.02s away (mocap gap) — same threshold as
        # datasets.py:_load_tum's rgb/GT matching — so a gap frame isn't associated with
        # bogus GT; sync.associate_trajectories (by timestamp) then tolerates the missing row.
        positions, quats, kept_timestamps = [], [], []
        for ts in timestamps:
            idx = int(np.argmin(np.abs(gt_timestamps - ts)))
            if abs(gt_timestamps[idx] - ts) > 0.02:
                continue
            tx, ty, tz, qx, qy, qz, qw = gt_vals[idx]
            positions.append([tx, ty, tz])
            quats.append([qx, qy, qz, qw])
            kept_timestamps.append(ts)

        return PoseTrajectory3D(
            positions_xyz=np.array(positions, dtype=np.float64),
            orientations_quat_wxyz=np.array(quats, dtype=np.float64)[:, [3, 0, 1, 2]],  # xyzw→wxyz
            timestamps=np.array(kept_timestamps, dtype=np.float64),
        )

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
        When provided, GT uses the same N frames with matching filename-derived timestamps.
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
