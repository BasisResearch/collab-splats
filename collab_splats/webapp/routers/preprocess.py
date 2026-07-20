from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse

from collab_splats.preproc import get_video_info, sample_frames
from collab_splats.preproc.frame_store import FrameStore
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
    # Count extracted frames from the canonical frames.zarr store
    frames_zarr = s.output_dir / "frames.zarr"
    if frames_zarr.exists():
        info["frames_extracted"] = len(FrameStore.open(frames_zarr))
        info["frames_zarr"] = str(frames_zarr)
    return JSONResponse(info)


########################################################################
# Frame serving route (reads frames.zarr, encodes JPG on demand)
########################################################################


@router.get("/frame/{idx}")
async def frame_jpeg(idx: int) -> Response:
    """Return the idx-th selected keyframe as on-demand-encoded JPEG bytes."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"}, status_code=404)
    frames_zarr = s.output_dir / "frames.zarr"
    if not frames_zarr.exists():
        return JSONResponse({"ok": False, "error": "No frames.zarr for this session"}, status_code=404)
    store = FrameStore.open(frames_zarr)
    if not 0 <= idx < len(store):
        return JSONResponse({"ok": False, "error": f"idx {idx} out of range (0..{len(store) - 1})"}, status_code=404)
    jpg = cv2.imencode(".jpg", cv2.cvtColor(store.image(idx), cv2.COLOR_RGB2BGR))[1].tobytes()
    return Response(content=jpg, media_type="image/jpeg")


########################################################################
# Frame writing helpers
########################################################################


def _write_frames_zarr(
    frames: list[np.ndarray], records: list[dict], output_dir: Path, *, video_path: Path, method: str, max_frames: int
) -> None:
    """Write the canonical frames.zarr (decode-once keyframe store) for on-demand JPG serving."""
    prov = {
        "video_path": str(video_path),
        "video_mtime": video_path.stat().st_mtime,
        "method": method,
        "max_frames": max_frames,
    }
    FrameStore.create(output_dir / "frames.zarr", frames, records, provenance=prov)


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
            # Dispatch and window derivation live in the preproc library now
            sampling_method = "optical_flow" if method == "optical_flow" else "uniform"
            frames, records = sample_frames(
                str(s.video_path),
                method=sampling_method,
                max_frames=max_frames,
                min_disparity=min_disparity,
                on_progress=progress,
            )
            # Write frames.zarr — the sole persistent frame store. Served on demand by
            # GET /api/preprocess/frame/{idx}; reconstruct.py opens it as a FrameStore.
            _write_frames_zarr(
                frames,
                records,
                s.output_dir,
                video_path=s.video_path,
                method=sampling_method,
                max_frames=max_frames,
            )
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
