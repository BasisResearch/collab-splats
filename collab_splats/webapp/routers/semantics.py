from __future__ import annotations

import asyncio
import json
import threading
import traceback
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/semantics")


@router.get("/methods")
async def list_methods() -> JSONResponse:
    """Return registered semantic extractor names."""
    from collab_splats.semantics.features import BaseFeatureExtractor
    methods = sorted(BaseFeatureExtractor._registry.keys()) if hasattr(BaseFeatureExtractor, "_registry") else ["dinov2"]
    return JSONResponse({"ok": True, "methods": methods})


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _run_sse() -> AsyncIterator[str]:
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"}); return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    extractor_name = s.extractor
    output_dir = s.output_dir
    backend_dir = output_dir / s.creator

    def run() -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Extractor: {extractor_name}"})
            from collab_splats.semantics.features import BaseFeatureExtractor
            extractor = BaseFeatureExtractor.get(extractor_name)()
            zarr_path = backend_dir / "feedforward.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"feedforward.zarr not found at {zarr_path}")
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult
            ff = FeedforwardResult.load_zarr(zarr_path, load_images=True)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Running feature extraction…"})
            features_path = output_dir / "features" / extractor_name
            extractor.extract(ff, features_path)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": f"Features saved to {features_path.name}"})
        except Exception as exc:
            tb = traceback.format_exc()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": f"{exc}\n{tb}"})

    threading.Thread(target=run, daemon=True).start()
    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("done", "error"):
            break


@router.get("/run")
async def run_semantics():
    return StreamingResponse(
        _run_sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
