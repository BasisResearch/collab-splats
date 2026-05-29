from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from collab_splats.utils.frame_sampling import get_video_info, sample_frames_fps, sample_frames_optical_flow
from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/preprocess")


########################################################################
# Info endpoint
########################################################################

@router.get("/info")
async def video_info() -> JSONResponse:
    """Return video metadata and session state for the preprocess tab."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    info: dict = {"ok": True, "output_dir": str(s.output_dir)}
    if s.video_path and s.video_path.exists():
        meta = get_video_info(str(s.video_path))
        info["video"] = meta
    # Count extracted frames if frames dir exists
    frames_dir = s.output_dir / "frames"
    if frames_dir.is_dir():
        jpgs = sorted(frames_dir.glob("*.jpg"))
        info["frames_extracted"] = len(jpgs)
        info["frames_dir"] = str(frames_dir)
    return JSONResponse(info)


########################################################################
# Frame writing helper
########################################################################

def _write_frames(frames: list[np.ndarray], output_dir: Path) -> Path:
    """Write RGB numpy frames as JPEGs to output_dir/frames/. Return frames dir."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        out = frames_dir / f"frame_{i:06d}.jpg"
        Image.fromarray(frame).save(str(out), format="JPEG", quality=95)
    return frames_dir


########################################################################
# SSE extraction generator
########################################################################

def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


async def _extract_sse(method: str, max_frames: int, min_disparity: float) -> AsyncIterator[str]:
    """Async generator: run frame extraction in thread, yield SSE JSON lines."""
    s = get_session()
    if s.video_path is None or not s.video_path.exists():
        yield _sse({"type": "error", "msg": "No video loaded"})
        return
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No output directory"})
        return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def progress(current: int, total: int) -> None:
        pct = int(current / max(total, 1) * 100)
        loop.call_soon_threadsafe(
            queue.put_nowait, {"type": "progress", "pct": pct, "msg": f"{current}/{total} frames"}
        )

    def run() -> None:
        try:
            if method == "optical_flow":
                frames, _ = sample_frames_optical_flow(
                    str(s.video_path), max_frames=max_frames,
                    min_disparity=min_disparity, on_progress=progress, verbose=False,
                )
            else:
                # Balanced FPS: derive target fps from desired frame count and duration
                info = get_video_info(str(s.video_path))
                dur = info.get("duration_s") or (info["total_frames"] / (info.get("fps") or 30.0))
                target_fps = max_frames / max(dur, 1.0)
                frames, _ = sample_frames_fps(
                    str(s.video_path), fps=target_fps, max_frames=max_frames,
                    on_progress=progress, verbose=False,
                )
            _write_frames(frames, s.output_dir)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "done", "msg": f"{len(frames)} frames extracted", "count": len(frames)},
            )
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": str(exc)})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Drain the queue until done/error
    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


########################################################################
# Extract endpoint
########################################################################

@router.get("/extract")
async def extract_frames(method: str = "optical_flow", max_frames: int = 200, min_disparity: float = 50.0):
    """SSE endpoint: extract frames from session video."""
    return StreamingResponse(
        _extract_sse(method, max_frames, min_disparity),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
