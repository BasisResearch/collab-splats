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


@router.get("/status")
async def feature_status() -> JSONResponse:
    """Return which extractors have already-cached lifted features for the current session."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "cached": []})
    backend_dir = s.output_dir / s.creator
    sem_dir = backend_dir / "semantics"
    cached = sorted(
        p.name for p in sem_dir.iterdir()
        if p.is_dir() and (p / "features.zarr").exists()
    ) if sem_dir.is_dir() else []
    return JSONResponse({"ok": True, "cached": cached, "creator": s.creator})


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
    # Features are stored at {backend}/semantics/{extractor}/features.zarr
    cache_dir = backend_dir / "semantics" / extractor_name
    features_zarr = cache_dir / "features.zarr"

    def run() -> None:
        try:
            # Load from cache if it already exists — never re-extract by default
            if features_zarr.exists():
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"✓ Features cached: {cache_dir.name}"})
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": f"Loaded from cache ({cache_dir.relative_to(output_dir)})"})
                return

            zarr_path = backend_dir / "feedforward.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"feedforward.zarr not found at {zarr_path}")

            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Extractor: {extractor_name}"})
            from collab_splats.semantics.features import BaseFeatureExtractor  # noqa: PLC0415
            extractor = BaseFeatureExtractor.get(extractor_name)()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Running feature extraction…"})
            # extract_and_cache_from_zarr handles images loading + caching internally
            extractor.extract_and_cache_from_zarr(zarr_path, cache_dir, skip_existing=True)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": f"Features saved to {cache_dir.name}"})
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
async def run_semantics(extractor: str = ""):
    """SSE endpoint: extract features. Uses session extractor if param not provided."""
    s = get_session()
    if extractor:
        s.extractor = extractor
    return StreamingResponse(
        _run_sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
