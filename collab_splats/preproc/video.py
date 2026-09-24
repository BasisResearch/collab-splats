"""
Video probe and decode, in-process via PyAV.

- iter_frames yields BGR (for cv2); extract_frame returns RGB; both uint8 HWC
- frames come out in display orientation; get_video_info reports display dims
- no selection logic here, so qa and sampling both import it without a cycle
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import av
import cv2
import numpy as np

# Clockwise display rotation -> the cv2 op that applies it
# - keyed the way _rotation_degrees reports; 0 or unrecognized means "no rotate"
# - PyAV's frame.rotation is counter-clockwise, hence the `-frame.rotation` negations
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

    - PyAV exposes the display matrix only on decoded frames
    - advances the container: call it after reading stream metadata
    """
    for frame in container.decode(video=0):
        return int(-frame.rotation) % 360

    return 0


def get_video_info(video_path: str | Path) -> dict:
    """
    Frame count, rate, duration and display size from a container parse.

    Args:
        video_path: source video.

    Returns:
        {"total_frames", "fps", "duration_s", "width", "height"}; width/height are display
        dims (rotation applied).

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the container cannot be probed or reports no frames or no rate.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"get_video_info: {video_path} does not exist")

    # IndexError is what streams.video[0] raises on a file with no video stream
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

            # Decode output is rotated to display orientation, so 90/270 swaps the display W/H
            if _rotation_degrees(container) in (90, 270):
                width, height = height, width
    except (av.error.FFmpegError, IndexError) as exc:
        raise ValueError(f"get_video_info: cannot probe {video_path}: {exc}") from exc

    if total == 0 or fps == 0:
        raise ValueError(f"get_video_info: {video_path} reports {total} frames at {fps} fps")

    return {"total_frames": total, "fps": fps, "duration_s": total / fps, "width": width, "height": height}


########################################################################
# Decode
########################################################################


def _frame_index(frame: av.VideoFrame, stream: av.video.stream.VideoStream, fps: float) -> int:
    """
    Source frame index of a decoded frame, from its presentation timestamp.

    - needed after a seek, where the decode count is no longer the absolute index
    - assumes a constant frame rate
    """
    if frame.pts is None:
        raise ValueError("iter_frames: cannot seek a stream whose frames carry no presentation timestamps")

    # PyAV reports None when the container has no start time; ffmpeg treats that as 0
    start_time = 0 if stream.start_time is None else stream.start_time

    return round(float((frame.pts - start_time) * stream.time_base) * fps)


def _upright(frame: av.VideoFrame) -> np.ndarray:
    """
    Decoded frame as BGR uint8 HWC in display orientation.

    - PyAV returns the stored orientation; the display rotation is applied here
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
    - a generator: every raise below surfaces at the first `next()`, not at the call

    Args:
        video_path: source video.
        indices: only these source frames, deduped and ascending; one linear scan, no seek.
        start: first frame of a contiguous window, reached by a container seek.
        count: length of that window; None runs to the end of the video.

    Returns:
        An iterator of (frame_idx, (H, W, 3) uint8 BGR) in ascending frame_idx
        order, in display orientation.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file cannot be decoded, both modes are passed, or a window seek has
            no fps to seek by.
    """
    if indices is not None and (start or count is not None):
        raise ValueError("iter_frames: pass either indices= or start=/count=, not both")

    if not Path(video_path).exists():
        raise FileNotFoundError(f"iter_frames: {video_path} does not exist")

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

    try:
        container = av.open(str(video_path))
    except av.error.FFmpegError as exc:
        raise ValueError(f"iter_frames: cannot decode {video_path}: {exc}") from exc

    with container:
        # IndexError is what streams.video[0] raises on a file with no video stream
        try:
            stream = container.streams.video[0]
        except IndexError as exc:
            raise ValueError(f"iter_frames: cannot decode {video_path}: {exc}") from exc

        # Decode is the whole cost here, so let PyAV use every core it is allowed
        stream.thread_type = "AUTO"

        fps = float(stream.average_rate) if stream.average_rate else 0.0
        cursor = 0
        resync = False

        # Contiguous window: seek to the keyframe at or before `start`
        # - the loop below drops frames before `start`
        if wanted is None and start:
            if not fps:
                raise ValueError(f"iter_frames: cannot seek {video_path} without fps")

            start_time = 0 if stream.start_time is None else stream.start_time
            container.seek(int(start / fps / stream.time_base) + start_time, stream=stream)
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

    Raises:
        FileNotFoundError: the path does not exist (from the probe or the decode).
        ValueError: the video cannot be probed or decoded, or frame_idx is out of range.
    """
    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]

    if frame_idx < 0 or frame_idx >= total:
        raise ValueError(f"extract_frame: frame {frame_idx} out of range for {video_path} ({total} frames)")

    # A one-frame window: iter_frames seeks to the keyframe at or before frame_idx and
    # decodes forward from there, which is why one frame does not cost a full scan
    for _index, bgr in iter_frames(video_path, start=frame_idx, count=1):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise ValueError(f"extract_frame: decode of {video_path} ended before frame {frame_idx}")
