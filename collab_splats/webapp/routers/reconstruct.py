from __future__ import annotations

import asyncio
import json
import threading
import traceback
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/reconstruct")


########################################################################
# Status endpoint
########################################################################


@router.get("/status")
async def status() -> JSONResponse:
    """Return current reconstruction state for the active session."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    # frames.zarr is the sole persistent frame store
    frames_zarr = s.output_dir / "frames.zarr"
    frame_count = len(FrameStore.open(frames_zarr)) if frames_zarr.exists() else 0
    backend_dir = s.output_dir / s.creator
    zarr_exists = (backend_dir / "feedforward.zarr").exists()
    return JSONResponse(
        {
            "ok": True,
            "has_frames": frame_count > 0,
            "frame_count": frame_count,
            "zarr_exists": zarr_exists,
            "creator": s.creator,
            "conf": s.conf,
        }
    )


########################################################################
# SSE helpers
########################################################################


def _sse(data: dict) -> str:
    """Encode a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(data)}\n\n"


async def _run_sse() -> AsyncIterator[str]:
    """Run feedforward reconstruction in a thread; stream log events via SSE."""
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"})
        return
    frames_zarr = s.output_dir / "frames.zarr"
    if not frames_zarr.exists() or len(FrameStore.open(frames_zarr)) == 0:
        yield _sse({"type": "error", "msg": "No frames found — run Preprocess first"})
        return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    creator_name = s.creator
    conf = s.conf
    output_dir = s.output_dir

    def run() -> None:
        """Worker: instantiate creator, run inference, save zarr."""
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "log", "msg": f"Backend: {creator_name}  conf: {conf}"},
            )
            # Instantiate the requested feedforward creator
            if creator_name == "vggtx":
                from collab_splats.pointcloud.feedforward import VGGTXCreator

                creator = VGGTXCreator(conf_threshold=conf)
            elif creator_name == "mapanything":
                from collab_splats.pointcloud.feedforward import MapAnythingCreator

                creator = MapAnythingCreator(confidence_percentile=conf)
            elif creator_name == "vggt_omega":
                from collab_splats.pointcloud.feedforward import VGGTOmegaCreator

                creator = VGGTOmegaCreator(conf_threshold=conf)
            else:
                raise ValueError(f"Unknown creator: {creator_name!r}")

            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "log", "msg": f"Starting {creator_name} inference…"},
            )
            # Run inference; outputs land in backend_dir. creator.reconstruct accepts a
            # FrameStore (Task 5) and temp-exports frames itself for path-locked preprocessing.
            backend_dir = output_dir / creator_name
            creator.reconstruct(FrameStore.open(frames_zarr), backend_dir)
            ff = creator.outputs
            if ff is None:
                raise RuntimeError("Creator produced no outputs")

            # Persist result as zarr for downstream tabs
            zarr_path = backend_dir / "feedforward.zarr"
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "log", "msg": f"Saving {zarr_path.name}…"},
            )
            ff.save_zarr(zarr_path)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "done",
                    "msg": f"Done. {len(ff.points):,} points.",
                    "zarr_path": str(zarr_path),
                },
            )
        except Exception as exc:
            tb = traceback.format_exc()
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "msg": f"{exc}\n{tb}"},
            )

    threading.Thread(target=run, daemon=True).start()

    # Relay queue events to the SSE stream until done/error
    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


########################################################################
# Run endpoint
########################################################################


@router.get("/run")
async def run_reconstruction():
    """SSE endpoint: run feedforward reconstruction."""
    return StreamingResponse(
        _run_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
