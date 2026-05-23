"""Trajectory I/O for the eval harness.

Canonical on-disk format: TUM (`timestamp tx ty tz qx qy qz qw`), space-separated,
camera pose expressed in the world frame (i.e., camera-to-world transform).

Canonical in-memory format: numpy `(N, 4, 4)` SE(3) homogeneous matrices in the
**world-to-camera** convention (matches what `VGGTXCreator.outputs.extrinsics`
emits and what `eval_gt.py` already passes around). Conversion to TUM's
camera-to-world convention happens at write/read time.

Datasets without true timing (7-Scenes, KITTI) record frame index as the
timestamp in seconds (0.0, 1.0, 2.0, ...). evo handles this transparently.

KITTI Odometry pose files are 3×4 cam-to-world matrices flattened row-major,
one frame per line. Both TUM and KITTI are cam-to-world; the
internal-to-disk inversion is the only convention boundary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# --- internal helpers --------------------------------------------------------


def _invert_se3(poses: np.ndarray) -> np.ndarray:
    """Invert (N, 4, 4) SE(3) poses (handles batched w2c↔c2w)."""
    out = np.tile(np.eye(4, dtype=poses.dtype), (poses.shape[0], 1, 1))
    R_in = poses[:, :3, :3]
    t_in = poses[:, :3, 3]
    R_out = R_in.transpose(0, 2, 1)
    out[:, :3, :3] = R_out
    out[:, :3, 3] = -np.einsum("nij,nj->ni", R_out, t_in)
    return out


def _check_poses(poses: np.ndarray) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N, 4, 4), got {poses.shape}")


# --- TUM ---------------------------------------------------------------------


def write_tum(
    path: Path | str,
    poses_w2c: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> None:
    """Write world-to-cam poses as a TUM trajectory file (cam-to-world on disk).

    Args:
        path: output path; parent directories are created.
        poses_w2c: (N, 4, 4) world-to-cam SE(3) matrices.
        timestamps: optional (N,) seconds. Defaults to frame index 0..N-1.
    """
    _check_poses(poses_w2c)
    n = poses_w2c.shape[0]
    if timestamps is None:
        timestamps = np.arange(n, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.shape != (n,):
        raise ValueError(
            f"timestamps shape {timestamps.shape} != poses count ({n},)"
        )

    poses_c2w = _invert_se3(poses_w2c.astype(np.float64))
    t = poses_c2w[:, :3, 3]
    quat = R.from_matrix(poses_c2w[:, :3, :3]).as_quat()  # (N, 4) [x, y, z, w]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{timestamps[i]:.9f} "
        f"{t[i, 0]:.9f} {t[i, 1]:.9f} {t[i, 2]:.9f} "
        f"{quat[i, 0]:.9f} {quat[i, 1]:.9f} {quat[i, 2]:.9f} {quat[i, 3]:.9f}"
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")


def read_tum(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a TUM trajectory file (cam-to-world on disk) → world-to-cam.

    Returns:
        poses_w2c: (N, 4, 4) world-to-cam SE(3).
        timestamps: (N,) float64 seconds.
    """
    raw = Path(path).read_text().splitlines()
    rows: list[list[float]] = []
    for ln in raw:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) != 8:
            raise ValueError(
                f"TUM line must have 8 columns, got {len(toks)}: {ln!r}"
            )
        rows.append([float(x) for x in toks])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 4, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    timestamps = arr[:, 0]
    t = arr[:, 1:4]
    quat = arr[:, 4:8]  # [qx, qy, qz, qw] — scipy native order

    poses_c2w = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
    poses_c2w[:, :3, :3] = R.from_quat(quat).as_matrix()
    poses_c2w[:, :3, 3] = t
    poses_w2c = _invert_se3(poses_c2w)
    return poses_w2c, timestamps


# --- KITTI Odometry ----------------------------------------------------------


def kitti_3x4_flat_to_w2c(text: str) -> np.ndarray:
    """Parse KITTI 3×4-flat (cam-to-world) text → (N, 4, 4) world-to-cam.

    Each non-empty, non-comment line must contain 12 floats (row-major 3×4).
    Whitespace-tolerant; trailing newline OK.
    """
    rows: list[list[float]] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) != 12:
            raise ValueError(
                f"KITTI pose line must have 12 floats, got {len(toks)}: {ln!r}"
            )
        rows.append([float(x) for x in toks])
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 4)

    poses_c2w = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
    poses_c2w[:, :3, :4] = arr
    return _invert_se3(poses_c2w)


def kitti_file_to_w2c(path: Path | str) -> np.ndarray:
    """Read a KITTI Odometry pose file → (N, 4, 4) world-to-cam."""
    return kitti_3x4_flat_to_w2c(Path(path).read_text())


def w2c_to_kitti_3x4_flat(poses_w2c: np.ndarray) -> str:
    """Serialize world-to-cam (N, 4, 4) → KITTI 3×4-flat (cam-to-world) text."""
    _check_poses(poses_w2c)
    poses_c2w = _invert_se3(poses_w2c.astype(np.float64))
    flat = poses_c2w[:, :3, :4].reshape(-1, 12)
    return "\n".join(" ".join(f"{x:.9f}" for x in row) for row in flat) + "\n"
