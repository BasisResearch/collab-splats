"""Video decode and probe: the only module that shells out to ffmpeg/ffprobe.

Holds no measurement and no selection logic, so both preproc.qa and
preproc.sampling can depend on it without a cycle.
"""

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


########################################################################
# Video metadata / decoding (ffmpeg + ffprobe only)
########################################################################


def _require_ffmpeg() -> None:
    """Raise if ffmpeg/ffprobe are missing — the only supported decode backend."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH; install ffmpeg (e.g. `apt install ffmpeg`)")


def _rotation_degrees(stream: dict) -> int:
    """CW display rotation from an ffprobe stream dict.

    Two metadata locations: legacy tags.rotate (CW), and Display Matrix side
    data (modern GoPro/iPhone; ffprobe reports CCW, convert with (-rot) % 360).
    """
    rotate = stream.get("tags", {}).get("rotate")
    if rotate:
        return int(rotate) % 360
    for sd in stream.get("side_data_list", []):
        if sd.get("side_data_type") == "Display Matrix" and sd.get("rotation") is not None:
            return int(-sd["rotation"]) % 360
    return 0


def get_video_info(video_path: str) -> dict:
    """Return video metadata via ffprobe.

    Keys: total_frames (int), fps (float), duration_s (float), width (int),
    height (int). Width/height are display dims (rotation applied), matching
    the frames the decode functions yield. All zeros if the file can't be probed.
    """
    _require_ffmpeg()
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    # -count_packets gives a reliable frame count when nb_frames is absent
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-count_packets", str(video_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe failed for %s", video_path, exc_info=True)
        return zeros
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        # Frame rate arrives as a ratio string, e.g. "30000/1001"
        num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else 0.0
        total = int(s.get("nb_frames") or s.get("nb_read_packets") or 0)
        width, height = int(s.get("width") or 0), int(s.get("height") or 0)
        # Report display dims: ffmpeg auto-rotates output, so 90/270 swaps W/H
        if _rotation_degrees(s) in (90, 270):
            width, height = height, width
        duration_s = total / fps if fps > 0 else 0.0
        return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}
    return zeros


def _probe_dims(video_path: str) -> tuple[int, int]:
    """Display (width, height) via a cheap ffprobe — no packet count / full demux."""
    _require_ffmpeg()
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-select_streams",
                "v:0",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe dims failed for %s", video_path, exc_info=True)
        return 0, 0
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        w, h = int(s.get("width") or 0), int(s.get("height") or 0)
        # Match get_video_info: 90/270 rotation swaps display W/H
        if _rotation_degrees(s) in (90, 270):
            w, h = h, w
        return w, h
    return 0, 0


def _iter_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield every frame as BGR uint8 HWC via one ffmpeg rawvideo pipe.

    ffmpeg applies rotation metadata itself, so yielded dims always match
    get_video_info's display dims.
    """
    w, h = _probe_dims(str(video_path))
    if w == 0 or h == 0:
        return
    cmd = ["ffmpeg", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    try:
        # Read fixed-size frames until the pipe runs dry
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def _iter_selected_frames(video_path: str, indices: list[int], w: int, h: int) -> Iterator[np.ndarray]:
    """Yield BGR frames for the given source indices via one ffmpeg select pass.

    ffmpeg decodes in a single streaming pipe (in C) but a `select` filter emits
    only the requested frame numbers, in ascending source order — so Python
    touches len(indices) frames, not the whole video. The caller zips the yields
    with sorted(indices) to key frames by source index.
    """
    _require_ffmpeg()
    if not indices:
        return
    # select='eq(n\,i)+eq(n\,j)+...' passes only these frame numbers; -vsync 0
    # keeps them 1:1 (no constant-frame-rate resampling / duplication).
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select={expr}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-an",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    try:
        # Read fixed-size frames until the pipe runs dry
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


########################################################################
# Frame I/O
########################################################################


def _seek_frame(
    video_path: str | Path,
    frame_idx: int,
    *,
    fps: float,
    w: int,
    h: int,
    total: int = 0,
) -> np.ndarray:
    """Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    Caller supplies pre-probed fps/w/h/total so a batch of seeks probes the video
    only once. Seeks by timestamp (O(1) in frame depth).
    """
    _require_ffmpeg()
    if not fps or not w or not h:
        raise ValueError(f"cannot seek {video_path}: missing fps/width/height")
    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"_seek_frame: frame {frame_idx} out of range for {video_path}")
    # Seek to the frame midpoint, not its start: PTS float rounding can otherwise land
    # the demuxer just past the target timestamp and decode frame N+1 instead of N.
    seek_s = max(frame_idx - 0.5, 0) / fps
    # -ss before -i = input seek (demuxer-level); rawvideo pipe avoids a temp file.
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{seek_s:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    raw = proc.stdout
    if len(raw) < w * h * 3:
        err = proc.stderr.decode(errors="replace")[-500:]
        raise ValueError(f"_seek_frame: frame {frame_idx} not found in {video_path}: {err}")
    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()


def extract_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    """Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    Exact on constant-frame-rate video; may land one frame off near keyframes on
    VFR sources.
    """
    info = get_video_info(str(video_path))
    return _seek_frame(
        video_path,
        frame_idx,
        fps=info["fps"],
        w=info["width"],
        h=info["height"],
        total=info["total_frames"],
    )
