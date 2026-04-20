"""
Preprocessing Utilities for Video and Image Processing

This module provides generic utility functions for preprocessing video and image data,
including rotation detection, frame loading, and other common operations.

These utilities are used by the optical flow frame selector and other preprocessing modules.

Main Components:
1. Video rotation detection and correction
2. Frame loading utilities
3. Image I/O helpers

Usage:
    from preproc_utils import get_video_rotation, rotate_frame, load_frame

    # Detect and apply rotation
    rotation = get_video_rotation('video.mp4')
    frame = load_frame(cap, frame_idx)
    corrected_frame = rotate_frame(frame, rotation)
"""

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np


# ============================================================================
# Video Stream Utilities
# ============================================================================

@contextmanager
def video_only_stream(video_path: Union[str, Path]):
    """
    Context manager that yields a path to a temporary video-only copy of the input.

    Strips audio and telemetry streams (e.g. GoPro GPMF) before handing the
    file to OpenCV. This prevents the FFmpeg packet read limit error that occurs
    when a multi-stream container has too many non-video packets between video
    frames relative to OPENCV_FFMPEG_READ_ATTEMPTS (default 4096).

    Uses stream copy (-c:v copy), so no re-encoding takes place.

    Args:
        video_path: Path to the source video file

    Yields:
        Path to a temporary video-only file (deleted on context exit)

    Example:
        with video_only_stream('GH010210.MP4') as clean:
            cap = cv2.VideoCapture(str(clean))
    """
    video_path = Path(video_path)
    tmp_path = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=video_path.suffix)
        os.close(fd)
        tmp_path = Path(tmp)

        subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-map', '0:v:0',
                '-c:v', 'copy',
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
        )
        yield tmp_path
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


# ============================================================================
# Video Rotation Detection and Correction
# ============================================================================

def get_video_rotation(video_path: Union[str, Path]) -> Optional[int]:
    """
    Detect video rotation from metadata using ffprobe.

    Many drone videos are recorded in portrait mode but have rotation metadata
    indicating they should be displayed rotated. This function reads that metadata.

    Args:
        video_path: Path to the video file

    Returns:
        cv2.ROTATE_* constant (90_CLOCKWISE, 180, or 90_COUNTERCLOCKWISE) or None
    """
    try:
        # Use ffprobe to get rotation metadata
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        metadata = json.loads(result.stdout)

        # Check for rotation in video stream
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                rotation = stream.get('tags', {}).get('rotate', '0')
                rotation = int(rotation)

                # Map rotation degrees to cv2.rotate codes
                if rotation == 90:
                    return cv2.ROTATE_90_CLOCKWISE
                elif rotation == 180:
                    return cv2.ROTATE_180
                elif rotation == 270:
                    return cv2.ROTATE_90_COUNTERCLOCKWISE
    except Exception:
        pass

    return None


def rotate_frame(frame: np.ndarray, rotation_code: Optional[int]) -> np.ndarray:
    """
    Apply rotation to a frame if rotation_code is not None.

    Args:
        frame: BGR image (H, W, 3)
        rotation_code: cv2.ROTATE_* constant or None

    Returns:
        Rotated frame or original frame if rotation_code is None
    """
    if rotation_code is not None and frame is not None:
        return cv2.rotate(frame, rotation_code)
    return frame


# ============================================================================
# Frame Loading Utilities
# ============================================================================

def load_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load a specific frame from an open video capture.

    Args:
        cap: OpenCV VideoCapture object
        frame_idx: Frame index to load

    Returns:
        BGR frame (H, W, 3) or None if loading failed
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get basic information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video metadata (fps, frame_count, width, height, etc.)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info


# ============================================================================
# Image I/O Utilities
# ============================================================================

def get_image_paths(
    directory: Path,
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Get sorted image paths from a directory.

    Args:
        directory: Directory containing images
        extensions: List of file extensions to include (default: common image formats)

    Returns:
        Sorted list of image paths
    """
    if extensions is None:
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    paths = []
    for ext in extensions:
        paths.extend(directory.glob(ext))

    return sorted(paths)


def save_frame(
    frame: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save a frame to disk with optional quality setting.

    Args:
        frame: BGR image to save
        output_path: Output file path
        quality: JPEG quality (0-100, higher is better)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), frame)
