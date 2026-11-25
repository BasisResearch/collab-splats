"""
Ground-truth eval datasets: frames plus world-to-camera poses per sequence.

- one loader per dataset type, looked up by `get_dataset`
- every loader returns frames in GT order, capped at max_frames
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class EvalDataset:
    """
    One sequence: frames and world-to-camera GT poses, in GT order.

    - intrinsics: CO3Dv2 only, (N, 3, 3) pixel K
    - depth_paths: 7-Scenes only, uint16 millimeter PNGs aligned with images
    """

    images: list[Path]
    gt_poses: np.ndarray  # (N, 4, 4) float32 world-to-camera
    intrinsics: np.ndarray | None = None
    depth_paths: list[Path] | None = None


def load_gt_depth(path: Path) -> np.ndarray:
    """
    7-Scenes depth frame in meters.

    - 65535 is the sensor's no-return code; it becomes 0 like every other invalid pixel

    Args:
        path: `*.depth.png`, uint16 millimeters.

    Returns:
        (H, W) float32 depth in meters, 0 where invalid.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = raw.astype(np.float32) / 1000.0
    depth[raw == 65535] = 0.0
    return depth


def _load_7scenes(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """
    7-Scenes sequence: `frame-NNNNNN.color.png` with a camera-to-world `.pose.txt` each.
    """
    seq_dir = Path(seq_dir)
    images = sorted(seq_dir.glob("*.color.png"))[:max_frames]
    poses = np.stack([np.linalg.inv(np.loadtxt(seq_dir / f"{p.stem.split('.')[0]}.pose.txt")) for p in images]).astype(
        np.float32
    )
    depth_paths = [seq_dir / f"{p.name.split('.')[0]}.depth.png" for p in images]
    return EvalDataset(images=images, gt_poses=poses, depth_paths=depth_paths)


def _read_tum_assoc(path: Path) -> list[tuple[float, str]]:
    """
    TUM list file as (timestamp, first value) pairs; `#` lines skipped.
    """
    out: list[tuple[float, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append((float(parts[0]), parts[1]))
    return out


def _read_tum_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    """
    TUM groundtruth.txt as (timestamp, camera-to-world 4x4 float64) pairs.

    - line format: timestamp tx ty tz qx qy qz qw
    """
    out: list[tuple[float, np.ndarray]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts = float(parts[0])
        tx, ty, tz = (float(x) for x in parts[1:4])
        qx, qy, qz, qw = (float(x) for x in parts[4:8])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out.append((ts, T))
    return out


def _load_tum(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """
    TUM RGB-D sequence: each rgb.txt frame paired with the nearest GT pose.

    - frames with no GT pose within 0.02 s are dropped
    """
    seq_dir = Path(seq_dir)
    rgb_entries = _read_tum_assoc(seq_dir / "rgb.txt")
    gt_entries = _read_tum_groundtruth(seq_dir / "groundtruth.txt")
    if not gt_entries:
        raise ValueError(f"No groundtruth entries in {seq_dir / 'groundtruth.txt'}")
    gt_ts = np.array([t for t, _ in gt_entries], dtype=np.float64)
    gt_T = np.stack([T for _, T in gt_entries])  # (G, 4, 4)
    images: list[Path] = []
    poses: list[np.ndarray] = []
    for ts, fname in rgb_entries[:max_frames]:
        idx = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(gt_ts[idx] - ts) > 0.02:
            continue
        images.append(seq_dir / fname)
        poses.append(np.linalg.inv(gt_T[idx]))
    gt_poses = np.stack(poses).astype(np.float32) if poses else np.zeros((0, 4, 4), dtype=np.float32)
    return EvalDataset(images=images, gt_poses=gt_poses)


def _load_co3dv2(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """
    CO3Dv2 sequence: `images/` plus the category's frame_annotations.jgz.

    - annotations: seq_dir or its category parent
    - viewpoint R, T: PyTorch3D, x_cam = x_world @ R + T, axes +x left, +y up
    - to OpenCV w2c: R_cv = S @ R.T, T_cv = S @ T, S = diag(-1, -1, 1)
    - focal_length, principal_point: NDC, units of min(H, W) / 2
    """
    seq_dir = Path(seq_dir)

    # Annotations sit beside the sequence or, as CO3Dv2 ships them, in the category dir
    ann_path = seq_dir / "frame_annotations.jgz"
    if not ann_path.exists():
        ann_path = seq_dir.parent / "frame_annotations.jgz"
    if not ann_path.exists():
        raise FileNotFoundError(f"frame_annotations.jgz not found in {seq_dir} or {seq_dir.parent}")

    with gzip.open(ann_path, "rt", encoding="utf-8") as f:
        all_annotations = json.load(f)

    # Keep this sequence's frames; the category file holds every sequence
    seq_name = seq_dir.name
    annotations = [a for a in all_annotations if a.get("sequence_name") == seq_name]
    if not annotations:
        raise ValueError(f"no annotations for sequence '{seq_name}' in {ann_path}")

    def _frame_number(ann: dict) -> int:
        stem = Path(ann["image"]["path"]).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else 0

    annotations = sorted(annotations, key=_frame_number)[:max_frames]

    images: list[Path] = []
    gt_poses_list: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []

    for ann in annotations:
        img_name = Path(ann["image"]["path"]).name
        images.append(seq_dir / "images" / img_name)

        vp = ann["viewpoint"]
        R = np.array(vp["R"], dtype=np.float32)
        T = np.array(vp["T"], dtype=np.float32)
        # PyTorch3D to OpenCV camera axes
        # - x_cam_cv = S @ x_cam_p3d, S = diag(-1, -1, 1)
        # - so R_cv = S @ R.T, T_cv = S @ T
        _S = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _S[:, None] * R.T  # equivalent to diag(S) @ R.T
        pose[:3, 3] = _S * T
        gt_poses_list.append(pose)

        # NDC intrinsics to pixel K; NDC y points up, image y down
        H, W = ann["image"]["size"]
        s = min(H, W) / 2.0
        fx_ndc, fy_ndc = vp["focal_length"]
        px_ndc, py_ndc = vp["principal_point"]
        fx = fx_ndc * s
        fy = fy_ndc * s
        cx = px_ndc * s + W / 2.0
        cy = -py_ndc * s + H / 2.0  # PyTorch3D NDC y points UP; image y points DOWN
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrinsics_list.append(K)

    gt_poses = np.stack(gt_poses_list).astype(np.float32) if gt_poses_list else np.zeros((0, 4, 4), dtype=np.float32)
    intrinsics = np.stack(intrinsics_list).astype(np.float32) if intrinsics_list else None
    return EvalDataset(images=images, gt_poses=gt_poses, intrinsics=intrinsics)


_REGISTRY: dict[str, Callable[..., EvalDataset]] = {
    "7scenes": _load_7scenes,
    "tum": _load_tum,
    "co3dv2": _load_co3dv2,
}


def get_dataset(name: str) -> Callable[..., EvalDataset]:
    """
    Loader for a dataset type.

    Args:
        name: dataset type key.

    Returns:
        Loader taking (seq_dir, max_frames).

    Raises:
        KeyError: unknown type; the message lists valid ones.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
