"""
Keyframe store: a COLMAP-style images/ directory of PNGs.

- images/frame_NNNNNN.png, named by source frame index, lossless, RGB at both boundaries
- selection config lives in <backend>/run_config.yaml, per-frame quality in video_quality_report.json
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from collab_splats.utils.io import read_image

logger = logging.getLogger(__name__)

########################
# Constants
########################

# Image extensions every frame directory reader accepts
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


########################
# Store API
########################


def frame_idx_from_path(path: Path | str) -> int:
    """
    Source frame index encoded in a frame filename.

    - reads the numeric tail after the last underscore: frame_000042 -> 42, IMG_1234 -> 1234

    Args:
        path: path whose stem ends in the source index.

    Returns:
        The source video frame index.
    """
    return int(Path(path).stem.split("_")[-1])


def frame_paths(dir: Path | str, idxs: Sequence[int] | None = None) -> list[Path]:
    """
    Image paths in a frame directory, by source frame index or in filename order.

    - idxs select by source frame_idx, never by row position

    Args:
        dir: directory holding frame_NNNNNN.<ext> images.
        idxs: source frame indices, in the order wanted; None takes every image.

    Returns:
        Image paths; empty when the directory is missing or holds none.

    Raises:
        KeyError: when idxs names a frame_idx the directory does not hold.
    """
    dir = Path(dir)
    paths = sorted(p for p in dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) if dir.is_dir() else []

    if idxs is None:
        return paths

    # Map source index to path, refusing any index the directory lacks
    by_idx = {frame_idx_from_path(p): p for p in paths}
    missing = [int(i) for i in idxs if int(i) not in by_idx]

    if missing:
        raise KeyError(f"frame_paths: frame_idx {missing[:5]} not in {dir}")

    return [by_idx[int(i)] for i in idxs]


def write_frames(
    dir: Path | str,
    frames: Sequence[np.ndarray] | np.ndarray,
    idxs: Sequence[int],
    *,
    png_compression: int = 1,
) -> list[Path]:
    """
    Write frames as PNGs named by their source frame index.

    Args:
        dir: images directory to create; stale frame images in it are removed first.
        frames: RGB uint8 (H, W, 3) frames, one per index.
        idxs: source video frame index of each frame.
        png_compression: cv2 PNG level 0-9; higher is smaller and slower.

    Returns:
        Written image paths, in idxs order.

    Raises:
        ValueError: frames and idxs differ in length, or idxs is empty.
    """
    dir = Path(dir)

    if len(frames) != len(idxs):
        raise ValueError(f"write_frames: {len(frames)} frames against {len(idxs)} indices")

    if not len(idxs):
        raise ValueError("write_frames: no frames selected")

    dir.mkdir(parents=True, exist_ok=True)

    # Remove a previous run's frames so frame_paths never serves them as this run's
    for stale in frame_paths(dir):
        stale.unlink()

    # Store is RGB at the boundary; cv2 writes BGR
    paths: list[Path] = []
    for frame, idx in zip(frames, idxs):
        path = dir / f"frame_{int(idx):06d}.png"
        cv2.imwrite(
            str(path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
        )
        paths.append(path)

    logger.info("frames: wrote %d PNGs to %s", len(paths), dir)
    return paths


def read_frames(dir: Path | str, idxs: Sequence[int] | None = None, *, workers: int = 8) -> np.ndarray:
    """
    Read frames from an images directory as one RGB stack.

    - decoded on a thread pool straight into one preallocated stack; cv2 releases the GIL

    Args:
        dir: images directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices to read, in the order given; None reads every
            frame in filename order.
        workers: decode threads.

    Returns:
        (N, H, W, 3) uint8 RGB.

    Raises:
        FileNotFoundError: when no frame resolves: idxs is None over an empty directory, or idxs is empty.
        KeyError: when idxs names a frame_idx the directory does not hold.
    """
    paths = frame_paths(dir, idxs)

    if not paths:
        raise FileNotFoundError(f"read_frames: no frame images in {dir}")

    # First frame sets the stack's shape and dtype
    first = read_image(paths[0])
    stack = np.empty((len(paths), *first.shape), first.dtype)
    stack[0] = first

    def _load(i: int) -> None:
        """
        Decode frame i into its slot of the stack.
        """
        stack[i] = read_image(paths[i])

    # Remaining frames decode on the pool; a worker's exception re-raises here
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_load, range(1, len(paths))))

    return stack
