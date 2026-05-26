#!/usr/bin/env python
"""Diagnose per-frame trajectory parity between our LC pipeline and a VGGT-SLAM TUM file.

Usage:
    python evals/runners/diagnose_lc_parity.py \\
        --seq_dir    /data/7scenes/chess/seq-01 \\
        --vggt_slam_tum /results/vggt_slam_chess.tum \\
        --max_frames 500 \\
        --submap_size 100

What it outputs:
    - Per-frame table: frame_idx | gt_t | ours_t_err(m) | slam_t_err(m) | ours_r_err(deg) | slam_r_err(deg)
    - Per-submap ATE summary: which submap boundaries accumulate the most error
    - Overall ATE RMSE for both trajectories vs GT

Both trajectories are aligned to GT via SE(3) (umeyama_se3, no scale) on
camera-centre translations before error computation.

This script imports cleanly without GPU / model weights present.  The
``diagnose()`` function accepts pre-computed (N,4,4) world-to-cam pose arrays
so it can be called offline — no live inference is performed here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Allow running directly from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collab_splats.pointcloud.loop_closure.closure import umeyama_se3

logger = logging.getLogger(__name__)

########################################
####### TUM I/O helpers ################
########################################


def _load_tum(path: str | Path) -> np.ndarray:
    """Parse a TUM trajectory file into (N, 4, 4) world-to-cam matrices.

    TUM format: one pose per line — timestamp tx ty tz qx qy qz qw
    Lines starting with '#' and blank lines are ignored.
    The file stores camera-to-world (c2w); we invert to world-to-cam (w2c).

    Returns:
        (N, 4, 4) float32 world-to-cam homogeneous matrices.
    """
    path = Path(path)
    c2w_list: list[np.ndarray] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        # Build c2w matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        c2w_list.append(T)
    if not c2w_list:
        raise ValueError(f"No valid poses found in TUM file: {path}")
    # Invert c2w → w2c
    c2w = np.stack(c2w_list)  # (N, 4, 4)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    return w2c


########################################
####### Geometry helpers ###############
########################################


def _rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance in degrees between two rotation matrices R1 and R2."""
    # Relative rotation: dR = R1^T @ R2
    dR = R1.T @ R2
    # Clamp trace to valid range for arccos
    trace = float(np.trace(dR))
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _camera_centres(w2c: np.ndarray) -> np.ndarray:
    """Extract camera centres (3,) from (N, 4, 4) world-to-cam matrices.

    centre = -R^T t  where w2c = [R | t]
    """
    R = w2c[:, :3, :3]   # (N, 3, 3)
    t = w2c[:, :3, 3]    # (N, 3)
    # c = -R^T t = R^T @ (-t)
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)  # (N, 3)


def _align_to_gt(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> np.ndarray:
    """SE(3)-align pred camera centres to GT camera centres via umeyama_se3.

    Returns:
        pred_aligned: (N, 4, 4) world-to-cam after applying the alignment transform.
    """
    pred_centres = _camera_centres(pred_w2c)  # (N, 3)
    gt_centres = _camera_centres(gt_w2c)      # (N, 3)

    # Compute SE(3): target ≈ T @ source  →  gt ≈ T_align @ pred
    T_align = umeyama_se3(pred_centres, gt_centres)  # (4, 4)

    # Apply alignment to each pose: new_w2c = pred_w2c @ T_align^{-1}
    T_align_inv = np.linalg.inv(T_align.astype(np.float64)).astype(np.float32)
    aligned = pred_w2c @ T_align_inv[None]  # broadcast (N, 4, 4)
    return aligned


########################################
####### Per-submap ATE table ###########
########################################


def _submap_ate_table(
    ours_t_errs: np.ndarray,
    slam_t_errs: np.ndarray,
    submap_size: int,
) -> list[dict]:
    """Group per-frame translation errors into submap buckets.

    Returns a list of dicts with keys: submap_idx, start_frame, end_frame,
    ours_ate_rmse, slam_ate_rmse, ours_max, slam_max.
    """
    N = len(ours_t_errs)
    rows: list[dict] = []
    for sm_idx, start in enumerate(range(0, N, submap_size)):
        end = min(start + submap_size, N)
        o_chunk = ours_t_errs[start:end]
        s_chunk = slam_t_errs[start:end]
        rows.append({
            "submap_idx": sm_idx,
            "start_frame": start,
            "end_frame": end - 1,
            "ours_ate_rmse": float(np.sqrt(np.mean(o_chunk ** 2))),
            "slam_ate_rmse": float(np.sqrt(np.mean(s_chunk ** 2))),
            "ours_max": float(np.max(o_chunk)),
            "slam_max": float(np.max(s_chunk)),
        })
    return rows


########################################
####### Main diagnostic logic ##########
########################################


def diagnose(
    seq_dir: str | Path,
    vggt_slam_tum: str | Path,
    max_frames: int = 500,
    submap_size: int | None = None,
    ours_w2c: np.ndarray | None = None,
) -> None:
    """Compare our LC pipeline trajectory to a VGGT-SLAM TUM file against GT.

    Args:
        seq_dir: Path to a 7-Scenes sequence directory (*.color.png + *.pose.txt).
        vggt_slam_tum: Path to the VGGT-SLAM TUM trajectory file.
        max_frames: Maximum number of frames to evaluate.
        submap_size: If set, also prints a per-submap ATE summary table.
        ours_w2c: Optional pre-computed (N, 4, 4) world-to-cam poses for our
            pipeline.  If None, a placeholder identity trajectory is used and
            a warning is printed (offline / import-check mode).
    """
    seq_dir = Path(seq_dir)

    # Load GT from 7-Scenes layout: *.color.png + matching *.pose.txt
    images = sorted(seq_dir.glob("*.color.png"))[:max_frames]
    if not images:
        raise FileNotFoundError(f"No *.color.png files found in {seq_dir}")
    gt_poses = np.stack([
        np.linalg.inv(
            np.loadtxt(seq_dir / f"{p.stem.split('.')[0]}.pose.txt")
        )
        for p in images
    ]).astype(np.float32)
    N = len(gt_poses)
    logger.info("Loaded %d GT frames from %s", N, seq_dir)

    # Load VGGT-SLAM TUM trajectory
    slam_w2c_full = _load_tum(vggt_slam_tum)
    if len(slam_w2c_full) < N:
        logger.warning(
            "VGGT-SLAM TUM has %d poses but GT has %d frames; trimming GT.",
            len(slam_w2c_full),
            N,
        )
        N = len(slam_w2c_full)
        gt_poses = gt_poses[:N]
    slam_w2c = slam_w2c_full[:N]

    # Handle ours_w2c (offline mode: identity placeholder)
    if ours_w2c is None:
        logger.warning(
            "ours_w2c not provided — using identity placeholder. "
            "Pass pre-computed poses from the LC pipeline for real results."
        )
        ours_w2c = np.broadcast_to(np.eye(4, dtype=np.float32)[None], (N, 4, 4)).copy()
    else:
        ours_w2c = np.asarray(ours_w2c, dtype=np.float32)[:N]

    # SE(3)-align both trajectories to GT camera centres
    ours_aligned = _align_to_gt(ours_w2c, gt_poses)
    slam_aligned = _align_to_gt(slam_w2c, gt_poses)

    # Compute per-frame translation and rotation errors
    gt_centres = _camera_centres(gt_poses)           # (N, 3)
    ours_centres = _camera_centres(ours_aligned)     # (N, 3)
    slam_centres = _camera_centres(slam_aligned)     # (N, 3)

    ours_t_errs = np.linalg.norm(ours_centres - gt_centres, axis=1)  # (N,)
    slam_t_errs = np.linalg.norm(slam_centres - gt_centres, axis=1)  # (N,)

    ours_r_errs = np.array([
        _rotation_error_deg(ours_aligned[i, :3, :3], gt_poses[i, :3, :3])
        for i in range(N)
    ])
    slam_r_errs = np.array([
        _rotation_error_deg(slam_aligned[i, :3, :3], gt_poses[i, :3, :3])
        for i in range(N)
    ])

    # Print per-frame table header
    print(
        f"\n{'Frame':>6}  {'GT_t':>8}  "
        f"{'Ours_t(m)':>10}  {'SLAM_t(m)':>10}  "
        f"{'Ours_R(deg)':>12}  {'SLAM_R(deg)':>12}"
    )
    print("-" * 70)
    for i in range(N):
        gt_t = np.linalg.norm(gt_centres[i])
        print(
            f"{i:>6d}  {gt_t:>8.4f}  "
            f"{ours_t_errs[i]:>10.4f}  {slam_t_errs[i]:>10.4f}  "
            f"{ours_r_errs[i]:>12.3f}  {slam_r_errs[i]:>12.3f}"
        )

    # Print overall ATE RMSE
    ours_ate = float(np.sqrt(np.mean(ours_t_errs ** 2)))
    slam_ate = float(np.sqrt(np.mean(slam_t_errs ** 2)))
    print("\n" + "=" * 70)
    print(f"Overall ATE RMSE (translation):")
    print(f"  Ours : {ours_ate:.4f} m")
    print(f"  SLAM : {slam_ate:.4f} m")

    # Per-submap ATE table
    if submap_size is not None:
        rows = _submap_ate_table(ours_t_errs, slam_t_errs, submap_size)
        print(f"\nPer-submap ATE (submap_size={submap_size}):")
        print(
            f"{'SM':>4}  {'Start':>6}  {'End':>6}  "
            f"{'Ours_ATE':>10}  {'SLAM_ATE':>10}  "
            f"{'Ours_Max':>10}  {'SLAM_Max':>10}"
        )
        print("-" * 65)
        for r in rows:
            print(
                f"{r['submap_idx']:>4d}  {r['start_frame']:>6d}  {r['end_frame']:>6d}  "
                f"{r['ours_ate_rmse']:>10.4f}  {r['slam_ate_rmse']:>10.4f}  "
                f"{r['ours_max']:>10.4f}  {r['slam_max']:>10.4f}"
            )


########################################
####### CLI entry point ################
########################################


def main() -> None:
    """Parse args and run the trajectory parity diagnostic."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Per-frame LC vs VGGT-SLAM vs GT trajectory comparison."
    )
    parser.add_argument(
        "--seq_dir",
        required=True,
        help="Path to 7-Scenes sequence directory (*.color.png + *.pose.txt).",
    )
    parser.add_argument(
        "--vggt_slam_tum",
        required=True,
        help="Path to the VGGT-SLAM TUM trajectory file (c2w, 8-col format).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum number of frames to evaluate (default: 500).",
    )
    parser.add_argument(
        "--submap_size",
        type=int,
        default=None,
        help="If set, print a per-submap ATE summary table with this window size.",
    )
    args = parser.parse_args()

    diagnose(
        seq_dir=args.seq_dir,
        vggt_slam_tum=args.vggt_slam_tum,
        max_frames=args.max_frames,
        submap_size=args.submap_size,
    )


if __name__ == "__main__":
    main()
