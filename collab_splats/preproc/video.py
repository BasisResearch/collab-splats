"""
Video decode and probe.

Metadata comes from a PyAV container parse and every decode runs in-process —
no subprocess, no full demux, and nothing required on PATH.

Holds no measurement logic and no quality-driven selection, so both preproc.qa
and preproc.sampling can depend on it without a cycle. The constant-rate index
grid lives here rather than in sampling because sampling delegates to it.

Colour convention: iter_frames yields BGR (what cv2 wants), extract_frame
returns RGB (what its consumers store). Both are uint8 HWC.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Clockwise display rotation -> the cv2 op that applies it. Keyed the way
# _rotation_degrees reports, so 0 (and anything unrecognised) means "no rotate".
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


########################################################################
# Probe
########################################################################


def _rotation_degrees(container: av.container.InputContainer) -> int:
    """
    Clockwise display rotation, read off the first decoded frame.

    - PyAV exposes no stream-level side data, so a container's display matrix only
      reaches Python attached to a decoded frame.
    - VideoFrame.rotation is counter-clockwise (ffprobe's side-data convention);
      this returns clockwise, matching the legacy tags.rotate field.
    - Decoding advances the container, so call this after reading stream metadata.
    """
    for frame in container.decode(video=0):
        return int(-frame.rotation) % 360

    return 0


def get_video_info(video_path: str | Path) -> dict:
    """
    Video metadata from a PyAV container parse.

    - the count is exact and cheap: the container carries it
    - so no cheap/expensive split, and no full demux to opt out of

    Args:
        video_path: source video.

    Returns:
        {"total_frames": int, "fps": float, "duration_s": float, "width": int,
        "height": int}. Width/height are DISPLAY dims (rotation applied), matching
        what decode yields. Every value is zero if the container cannot be probed.
    """
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            # codec_context carries the STORED dimensions, before display rotation
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            width, height = stream.codec_context.width, stream.codec_context.height

            # Frame count comes from the container when it carries one
            # - stream.frames is nb_frames, exact for mp4/mov/avi — every format this repo ingests
            # - mkv / mpeg-ts omit it, report 0, and fall back to duration x rate
            total = int(stream.frames)
            if total == 0 and container.duration:
                total = int(round(container.duration / av.time_base * fps))

            # ffmpeg auto-rotates its decode output, so 90/270 swaps the display W/H
            if _rotation_degrees(container) in (90, 270):
                width, height = height, width
    except Exception:
        logger.debug("PyAV could not probe %s", video_path, exc_info=True)
        return zeros

    duration_s = total / fps if fps > 0 else 0.0
    return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}


########################################################################
# Decode
########################################################################


def _frame_index(frame: av.VideoFrame, stream: av.video.stream.VideoStream, fps: float) -> int:
    """
    Source frame index of a decoded frame, read off its presentation timestamp.

    - Only needed after a seek: the decoder resumes at the keyframe at or before
      the target, so a count of decoded frames is no longer the absolute index.
    - Assumes a constant frame rate — the same assumption the ffmpeg input seek
      this replaces already made.
    """
    if frame.pts is None:
        raise ValueError("iter_frames: cannot seek a stream whose frames carry no presentation timestamps")

    return round(float((frame.pts - (stream.start_time or 0)) * stream.time_base) * fps)


def _upright(frame: av.VideoFrame) -> np.ndarray:
    """
    Decoded frame as BGR uint8 HWC, turned to its display orientation.

    - PyAV hands back the STORED orientation. The ffmpeg binary auto-rotated for
      us, so the display matrix has to be applied here or a portrait clip comes
      back landscape and contradicts the dims get_video_info reports.
    - frame.rotation is counter-clockwise, so it is negated to index the
      clockwise codes — the same convention _rotation_degrees returns.
    """
    bgr = frame.to_ndarray(format="bgr24")
    code = _ROTATE_CODES.get(int(-frame.rotation) % 360)

    return bgr if code is None else cv2.rotate(bgr, code)


def iter_frames(
    video_path: str | Path,
    *,
    indices: Sequence[int] | None = None,
    start: int = 0,
    count: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, BGR uint8 HWC) from one in-process PyAV decode pass.

    - three modes, one decode loop
    - passed nothing, walks the whole video
    - `indices` and `start`/`count` each select a subset and are mutually exclusive

    Args:
        video_path: source video. An unopenable path yields nothing rather than
            raising — the samplers lean on that, the same way a zeros probe returns.
        indices: only these source frames, deduped and ascending. One linear scan,
            no seeking: a seek per index lands on a keyframe and re-decodes forward
            from it, which is slower than reading straight through.
        start: first frame of a contiguous window, reached by a container seek so
            the demuxer skips everything before it. The mode range-parallel workers
            use; scanning to the window instead would have every worker decode the
            whole file.
        count: length of that window; None runs to the end of the video.

    Returns:
        An iterator of (frame_idx, (H, W, 3) uint8 BGR) in ascending frame_idx
        order, in display orientation.
    """
    if indices is not None and (start or count is not None):
        raise ValueError("iter_frames: pass either indices= or start=/count=, not both")

    wanted = sorted({int(i) for i in indices}) if indices is not None else None
    if wanted is not None and not wanted:
        return

    # The scan stops at the last frame anyone asked for; None runs to the end
    if wanted is not None:
        last = wanted[-1]
    elif count is not None:
        last = start + count - 1
    else:
        last = None

    # An unopenable path yields nothing rather than raising — the ffmpeg pipe
    # returned early on a zeros probe and the samplers still lean on that
    try:
        container = av.open(str(video_path))
    except Exception:
        logger.debug("PyAV could not open %s", video_path, exc_info=True)
        return

    with container:
        stream = container.streams.video[0]

        # Decode is the whole cost here, so let PyAV use every core it is allowed
        stream.thread_type = "AUTO"

        fps = float(stream.average_rate) if stream.average_rate else 0.0
        cursor = 0
        resync = False

        # Contiguous windows keep the input seek the ffmpeg pipe had
        # - a backward seek (the default) lands on the keyframe at or before the target
        # - the loop below drops whatever precedes `start`
        if wanted is None and start:
            if not fps:
                raise ValueError(f"iter_frames: cannot seek {video_path} without fps")

            container.seek(int(start / fps / stream.time_base) + (stream.start_time or 0), stream=stream)
            resync = True

        pending = iter(wanted) if wanted is not None else None
        target = next(pending, None) if pending is not None else None

        for frame in container.decode(stream):
            # A seek leaves the counter meaningless, so the first frame out of the
            # decoder re-establishes it from its own timestamp
            if resync:
                cursor = _frame_index(frame, stream, fps)
                resync = False

            if last is not None and cursor > last:
                break

            # Scattered indices walk their sorted list; a window is a range test
            if wanted is not None:
                take = cursor == target
                if take:
                    target = next(pending, None)
            else:
                take = cursor >= start

            if take:
                yield cursor, _upright(frame)

            cursor += 1


def extract_frame(video_path: str | Path, frame_idx: int, *, info: dict | None = None) -> np.ndarray:
    """
    Decode one frame by seeking to it.

    Args:
        video_path: source video.
        frame_idx: source frame index.
        info: a get_video_info dict, to skip this call's own probe.

    Returns:
        (H, W, 3) uint8 RGB, in display orientation.
    """
    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]

    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"extract_frame: frame {frame_idx} out of range for {video_path} ({total} frames)")

    # A one-frame window: iter_frames seeks to the keyframe at or before frame_idx and
    # decodes forward from there, which is why one frame does not cost a full scan
    for _index, bgr in iter_frames(video_path, start=frame_idx, count=1):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError(f"extract_frame: decode of {video_path} ended before frame {frame_idx}")
