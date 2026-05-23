from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from collab_splats.utils.frame_sampling import extract_video_frames, sample_frames_fps


@dataclass
class EvalDataset:
    images: list[Path]
    gt_poses: np.ndarray   # (N, 4, 4) world-to-cam float32
    intrinsics: np.ndarray | None = None   # (N, 3, 3) float32, optional


def _load_7scenes(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    seq_dir = Path(seq_dir)
    images = sorted(seq_dir.glob("*.color.png"))[:max_frames]
    poses = np.stack([
        np.linalg.inv(
            np.loadtxt(seq_dir / f"{p.stem.split('.')[0]}.pose.txt")
        )
        for p in images
    ]).astype(np.float32)
    return EvalDataset(images=images, gt_poses=poses)


def _read_tum_assoc(path: Path) -> list[tuple[float, str]]:
    """Read a TUM-format association file. Lines: 'timestamp value [...]', '#' comments."""
    out: list[tuple[float, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append((float(parts[0]), parts[1]))
    return out


def _read_tum_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    """Read TUM groundtruth.txt: 'timestamp tx ty tz qx qy qz qw' (camera-to-world).

    Returns (timestamp, c2w 4x4 float64) pairs.
    """
    from scipy.spatial.transform import Rotation
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


def _load_kitti(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """Load a KITTI Odometry sequence.

    Layout (per official KITTI Odometry):
        seq_dir/image_2/{000000.png, 000001.png, ...}   (left color camera)
        seq_dir/poses.txt                                 (3x4 cam-to-world per line)
            OR
        seq_dir/../../poses/{NN}.txt                      (alt layout: split poses dir)

    Pose convention is documented in `evals/trajectory_io.py:kitti_3x4_flat_to_w2c`,
    which inverts cam-to-world to world-to-cam for us.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trajectory_io import kitti_file_to_w2c

    seq_dir = Path(seq_dir)
    images = sorted((seq_dir / "image_2").glob("*.png"))[:max_frames]

    primary = seq_dir / "poses.txt"
    alt = seq_dir.parent.parent / "poses" / f"{seq_dir.name}.txt"
    if primary.exists():
        poses_path = primary
    elif alt.exists():
        poses_path = alt
    else:
        raise FileNotFoundError(
            f"KITTI poses file not found. Tried: {primary} and {alt}"
        )

    poses = kitti_file_to_w2c(poses_path)[:max_frames].astype(np.float32)
    return EvalDataset(images=images, gt_poses=poses)


def _load_waymo(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """Load a Waymo Open Dataset segment that has been pre-extracted from .tfrecord.

    Waymo's tfrecord format requires the ``waymo-open-dataset`` pip wheel which
    pins TensorFlow and conflicts with the nerfstudio env's torch/CUDA stack.
    To keep this loader light, we expect the sequence to have been extracted by
    ``evals/runners/extract_waymo.py`` (run in a sidecar env) into a flat
    on-disk layout::

        seq_dir/images/{000000.png, 000001.png, ...}   (front camera)
        seq_dir/groundtruth.txt                         (TUM format, c2w camera)

    Image i is paired 1:1 with line i of groundtruth.txt — the extraction
    sidecar guarantees that ordering.
    """
    seq_dir = Path(seq_dir)
    images = sorted((seq_dir / "images").glob("*.png"))[:max_frames]
    gt_entries = _read_tum_groundtruth(seq_dir / "groundtruth.txt")
    if len(gt_entries) < len(images):
        raise ValueError(
            f"Waymo seq {seq_dir} has {len(images)} images but only "
            f"{len(gt_entries)} groundtruth entries — extraction is misaligned"
        )
    gt_T = np.stack([T for _, T in gt_entries[: len(images)]])
    poses = np.linalg.inv(gt_T).astype(np.float32)
    return EvalDataset(images=images, gt_poses=poses)


def _load_co3dv2(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """Load a single CO3Dv2 sequence.

    Layout:
        seq_dir/images/frame000001.jpg, ...
        seq_dir/frame_annotations.jgz   (gzip-compressed JSON list of frame dicts)

    Each frame dict viewpoint fields:
        R: [[...], ...] — 3x3 rotation in PyTorch3D row-major convention
                          (x_cam = x_world @ R + T; left-handed axes: +x=left, +y=up, +z=fwd)
        T: [tx, ty, tz] — world-to-cam translation in PyTorch3D camera space
        focal_length: [fx_ndc, fy_ndc] — relative to min(H,W)/2
        principal_point: [px_ndc, py_ndc] — offset from image centre, relative to min(H,W)/2

    Converted to OpenCV w2c convention: R_cv = S @ R.T, T_cv = S @ T
    where S = diag(-1,-1,1) maps PyTorch3D→OpenCV camera axes.
    """
    import gzip
    import json as _json

    seq_dir = Path(seq_dir)
    # CO3Dv2 stores frame_annotations.jgz at the category level (parent dir)
    # when the download is a single-sequence subset — fall back automatically.
    ann_path = seq_dir / "frame_annotations.jgz"
    if not ann_path.exists():
        ann_path = seq_dir.parent / "frame_annotations.jgz"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"frame_annotations.jgz not found in {seq_dir} or {seq_dir.parent}"
        )

    with gzip.open(ann_path, "rt", encoding="utf-8") as f:
        all_annotations = _json.load(f)

    # Filter to frames belonging to this sequence (by sequence_name field or path prefix)
    seq_name = seq_dir.name
    annotations = [
        a for a in all_annotations
        if a.get("sequence_name") == seq_name
        or Path(a["image"]["path"]).parts[0] == seq_name
    ]
    if not annotations:
        # fall back: use all annotations (single-sequence file)
        annotations = all_annotations

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
        # CO3Dv2 / PyTorch3D convention: x_cam = x_world @ R + T  (row-major, left-handed axes)
        # PyTorch3D camera axes: +x=left, +y=up, +z=forward
        # OpenCV camera axes:    +x=right, +y=down, +z=forward
        # Let S = diag(-1,-1,1).  Then x_cam_cv = S @ x_cam_p3d.
        # Substituting: x_cam_cv = S @ (R.T @ x_world + T)
        #   => R_cv = S @ R.T,  T_cv = S @ T
        _S = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _S[:, None] * R.T   # equivalent to diag(S) @ R.T
        pose[:3, 3] = _S * T
        gt_poses_list.append(pose)

        # Convert CO3Dv2 NDC intrinsics → pixel-space K
        # NDC: focal_length in units of min(H,W)/2; principal_point offset from centre
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


def _load_bicycle(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """Load a LLFF/bicycle-format sequence from the images_4/ subdirectory."""
    # Glob all pngs in images_4/ sorted lexicographically, then cap at max_frames
    images = sorted((seq_dir / "images_4").glob("*.png"))[:max_frames]
    # GT poses not available for LLFF format; zeros placeholder preserves EvalDataset shape
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))


def _load_video(seq_dir: Path, max_frames: int = 500, fps: float = 1.0) -> EvalDataset:
    """Sample frames from a video file at the requested fps, write to a sidecar _frames/ dir."""
    # Output frames into <stem>_frames/ sibling directory; created if absent
    frames_dir = seq_dir.parent / (seq_dir.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Sample keyframes then write to disk so EvalDataset receives file paths
    frames, indices = sample_frames_fps(str(seq_dir), fps=fps)
    images = extract_video_frames(str(seq_dir), indices[:max_frames], frames_dir)
    # GT poses not available for raw video; zeros placeholder
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))


_REGISTRY: dict[str, Callable[..., EvalDataset]] = {
    "7scenes": _load_7scenes,
    "tum": _load_tum,
    "kitti": _load_kitti,
    "waymo": _load_waymo,
    "co3dv2": _load_co3dv2,
    "bicycle": _load_bicycle,
    "video": _load_video,
}


def get_dataset(name: str) -> Callable[..., EvalDataset]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
