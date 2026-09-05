"""
Video decode and probe: the only module that shells out to ffmpeg/ffprobe.

Holds no measurement logic and no quality-driven selection, so both preproc.qa
and preproc.sampling can depend on it without a cycle. The constant-rate index
grid lives here rather than in sampling because sampling delegates to it.

Colour convention: iter_frames yields BGR (what cv2 wants), extract_frame
returns RGB (what its consumers store). Both are uint8 HWC.
"""

import itertools
import json
import logging
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


########################################################################
# Probe
########################################################################


def _require_ffmpeg() -> None:
    """
    Raise if ffmpeg/ffprobe are missing — the only supported decode backend.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH; install ffmpeg (e.g. `apt install ffmpeg`)")


def _rotation_degrees(stream: dict) -> int:
    """
    CW display rotation from an ffprobe stream dict.

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


def get_video_info(video_path: str | Path, *, count_frames: bool = True) -> dict:
    """
    Video metadata via ffprobe.

    - Keys: total_frames, fps, duration_s, width, height. All zeros if unprobeable.
    - Width/height are DISPLAY dims (rotation applied), matching what decode yields.
    - count_frames=False skips the -count_packets full demux, which is the whole
      cost of this call on a long video. total_frames then comes from the
      container's nb_frames and is 0 when the container does not carry it.
    """
    _require_ffmpeg()
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}

    # -count_packets demuxes the whole file for a reliable count when nb_frames
    # is absent; -select_streams v:0 keeps the cheap path to one stream.
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0", "-show_streams"]
    if count_frames:
        cmd.append("-count_packets")
    cmd.append(str(video_path))

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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

        # ffmpeg auto-rotates its output, so 90/270 swaps the display W/H
        if _rotation_degrees(s) in (90, 270):
            width, height = height, width

        duration_s = total / fps if fps > 0 else 0.0
        return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}

    return zeros


########################################################################
# Decode
########################################################################


def iter_frames(
    video_path: str | Path,
    *,
    indices: Sequence[int] | None = None,
    start: int = 0,
    count: int | None = None,
    info: dict | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, BGR uint8 HWC) from one ffmpeg rawvideo pipe.

    Three modes, one pipe and one cleanup path:

    - no arguments — every frame, in order.
    - indices=[...] — only those source frames, via a `select` filter. ffmpeg
      still demuxes from frame 0, so this is cheap in Python but not in IO.
    - start=/count= — a contiguous range via an INPUT seek, so the demuxer skips
      everything before it. This is the mode range-parallel workers use; the
      `select` filter cannot serve them because it would have every worker demux
      the whole file.

    indices and start/count are mutually exclusive. Pass info= to reuse a probe.
    """
    if indices is not None and (start or count is not None):
        raise ValueError("iter_frames: pass either indices= or start=/count=, not both")

    _require_ffmpeg()

    # count_frames=False: decoding never needs the total, and the full demux it
    # costs is the single most expensive thing this module does.
    info = info if info is not None else get_video_info(video_path, count_frames=False)
    w, h = info["width"], info["height"]
    if w == 0 or h == 0:
        return

    cmd = ["ffmpeg", "-v", "error"]

    if indices is not None:
        if len(indices) == 0:
            return

        # select='eq(n\,i)+eq(n\,j)+...' emits only these frame numbers; -vsync 0
        # keeps them 1:1 (no constant-frame-rate resampling or duplication).
        ordered = sorted({int(i) for i in indices})
        expr = "+".join(f"eq(n\\,{i})" for i in ordered)
        cmd += ["-i", str(video_path), "-vf", f"select={expr}", "-vsync", "0"]
        index_source: Iterator[int] = iter(ordered)
    else:
        # Seek to the frame midpoint, not its start: PTS float rounding can
        # otherwise land the demuxer past the target and start at frame N+1.
        if start:
            fps = info["fps"]
            if not fps:
                raise ValueError(f"iter_frames: cannot seek {video_path} without fps")
            cmd += ["-ss", f"{max(start - 0.5, 0) / fps:.6f}"]

        cmd += ["-i", str(video_path)]
        if count is not None:
            cmd += ["-frames:v", str(int(count))]
        index_source = itertools.count(start)

    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3

    try:
        # Read fixed-size frames until the pipe runs dry
        for frame_idx in index_source:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield frame_idx, np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def context_indices(video_path: str | Path, *, target_fps: float, info: dict | None = None) -> list[int]:
    """
    Source frame indices on a constant-rate grid at target_fps.

    - `sampling.sample_fps` calls this for its own targets, so a keyframe grid and a
      context grid built at the same rate agree frame-for-frame and keyframes are a
      subset by construction.
    - Stride floors at 1: a rate above the source rate cannot sample sub-frame.
    """
    # target_fps is the contract here, so an absent one is a config error, not a default
    if target_fps is None or target_fps <= 0:
        raise ValueError(f"context_indices needs a positive target_fps, got {target_fps!r}")

    # Reuse a caller's probe when given — a fresh one costs an ffprobe subprocess
    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]
    if total == 0:
        return []

    # Stride floors at 1 — a rate above the source rate cannot sample sub-frame
    native_fps = info["fps"] or 30.0
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, total, step))


def extract_frame(video_path: str | Path, frame_idx: int, *, info: dict | None = None) -> np.ndarray:
    """
    Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    - Exact on constant-frame-rate video; may land one frame off near keyframes
      on VFR sources.
    - Pass info= (a get_video_info dict) to hoist the probe out of a loop —
      probing per call is ~8x the cost of the decode itself.
    """
    _require_ffmpeg()
    info = info if info is not None else get_video_info(video_path)
    fps, w, h, total = info["fps"], info["width"], info["height"], info["total_frames"]

    if not fps or not w or not h:
        raise ValueError(f"cannot seek {video_path}: missing fps/width/height")
    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"extract_frame: frame {frame_idx} out of range for {video_path}")

    # Seek to the frame midpoint — same PTS-rounding guard as iter_frames.
    # -ss before -i is an input seek (demuxer-level); the rawvideo pipe avoids a temp file.
    seek_s = max(frame_idx - 0.5, 0) / fps
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
        raise ValueError(f"extract_frame: frame {frame_idx} not found in {video_path}: {err}")

    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()
