"""
Canonical keyframe store: a COLMAP-style images/ directory plus frames.json.

- the preprocess stage decodes a video once and writes images/frame_NNNNNN.png,
  lossless at PNG compression 1
- frames.json sits beside it, holding the selection records and provenance COLMAP has
  no slot for
- every pixel consumer reads the directory; path-locked consumers take the directory
  itself, so nothing stages a second copy
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

########################
# Constants
########################

SCHEMA_VERSION = 2

# The repo's single image-extension listing — reconstructor and feedforward both defer here
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

_MANIFEST_NAME = "frames.json"

# OpenCV's default. Level 9 costs 10x the time for 11% of the size (measured, spec 2.2).
_PNG_COMPRESSION = 1


########################
# Helpers
########################


def _manifest_path(dir: Path | str) -> Path:
    """
    frames.json, which sits beside the images directory rather than inside it.
    """
    return Path(dir).parent / _MANIFEST_NAME


def _jsonable(value: Any) -> Any:
    """
    numpy scalar or NaN -> a plain JSON value (NaN becomes null).
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if np.isnan(value) else value
    return value


########################
# Store API
########################


def frame_idx_from_path(path: Path | str) -> int:
    """
    Source frame index encoded in a frame_{idx:06d}.<ext> filename.

    Args:
        path: path whose stem ends in the zero-padded source index.

    Returns:
        The source video frame index.
    """
    return int(Path(path).stem.split("_")[-1])


def frame_paths(dir: Path | str) -> list[Path]:
    """
    Image paths in a frame directory, in filename order.

    Args:
        dir: directory holding frame_NNNNNN.<ext> images.

    Returns:
        Sorted image paths; empty when the directory is missing or holds none.
    """
    dir = Path(dir)
    if not dir.is_dir():
        return []
    return sorted(p for p in dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def write_frames(
    dir: Path | str,
    frames: Sequence[np.ndarray] | np.ndarray,
    records: Sequence[dict],
    provenance: dict,
) -> list[Path]:
    """
    Write frames as PNGs and the manifest beside them.

    Args:
        dir: images directory to create; stale frame images in it are removed first.
        frames: RGB uint8 (H, W, 3) frames, one per record.
        records: selection records, each carrying an int 'frame_idx' (source index).
        provenance: descriptive dict stamped into frames.json.

    Returns:
        Written image paths, in record order.
    """
    dir = Path(dir)
    if len(frames) != len(records):
        raise ValueError(f"write_frames: {len(frames)} frames against {len(records)} records")
    if not records or "frame_idx" not in records[0]:
        raise ValueError("write_frames: every record must contain 'frame_idx' (source video index)")

    dir.mkdir(parents=True, exist_ok=True)

    # A re-run selecting fewer frames must not leave the previous run's extras behind,
    # where frame_paths would serve them as if they were this run's selection
    for stale in frame_paths(dir):
        stale.unlink()

    # Store is RGB at the boundary; cv2 writes BGR
    paths: list[Path] = []
    for frame, record in zip(frames, records):
        path = dir / f"frame_{int(record['frame_idx']):06d}.png"
        cv2.imwrite(
            str(path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION],
        )
        paths.append(path)

    # Row-oriented: a reader wants one frame's record, not one column
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "frames": [{k: _jsonable(v) for k, v in record.items()} for record in records],
    }
    _manifest_path(dir).write_text(json.dumps(manifest, indent=2))

    logger.info("frames: wrote %d PNGs to %s", len(paths), dir)
    return paths


def read_frames(dir: Path | str, idxs: Sequence[int] | None = None) -> np.ndarray:
    """
    Read frames from an images directory as one RGB stack.

    Args:
        dir: images directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices to read, in the order given; None reads every
            frame in filename order.

    Returns:
        (N, H, W, 3) uint8 RGB.
    """
    paths = frame_paths(dir)
    if not paths:
        raise FileNotFoundError(f"read_frames: no frame images in {dir}")

    # idxs select by source frame_idx, never by row position — a caller holding a
    # frame_idx from a record must not have to know where it landed in the directory
    if idxs is not None:
        by_idx = {frame_idx_from_path(p): p for p in paths}
        missing = [int(i) for i in idxs if int(i) not in by_idx]
        if missing:
            raise KeyError(f"read_frames: frame_idx {missing[:5]} not in {dir}")
        paths = [by_idx[int(i)] for i in idxs]

    return np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])


def read_manifest(dir: Path | str) -> dict:
    """
    Selection records and provenance written beside an images directory.

    Args:
        dir: images directory; frames.json sits in its parent.

    Returns:
        {'schema_version', 'provenance', 'frames'}.
    """
    path = _manifest_path(dir)
    if not path.exists():
        raise FileNotFoundError(
            f"read_manifest: {path} not found. A scene written before this format holds "
            "frames.zarr — convert it with scripts/migrate_frames_zarr.py."
        )
    return json.loads(path.read_text())
