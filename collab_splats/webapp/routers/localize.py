from __future__ import annotations

import asyncio
import json
import threading
import traceback
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/localize")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.get("/sample_query")
async def sample_query_image() -> JSONResponse:
    """Return path to an out-of-sample image from a different scene for query testing."""
    import glob as _glob  # noqa: PLC0415

    s = get_session()
    _OUTPUTS = Path("/workspace/outputs")

    # Find a frame from ANY scene that is NOT the current session's scene
    current = s.output_dir.name if s.output_dir else ""
    for scene_dir in sorted(_OUTPUTS.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name == current:
            continue
        # Prefer frames/ directory (webapp-extracted)
        frames_dir = scene_dir / "frames"
        if frames_dir.is_dir():
            jpgs = sorted(frames_dir.glob("frame_*.jpg"))
            if jpgs:
                mid = jpgs[len(jpgs) // 2]
                return JSONResponse({"ok": True, "path": str(mid), "scene": scene_dir.name})
        # Fall back to images/ directory (old pipeline)
        images_dir = scene_dir / "images"
        if images_dir.is_dir():
            jpgs = sorted(images_dir.glob("frame_*.jpg"))
            if jpgs:
                mid = jpgs[len(jpgs) // 2]
                return JSONResponse({"ok": True, "path": str(mid), "scene": scene_dir.name})
    return JSONResponse({"ok": False, "error": "No other scenes found"})


@router.get("/methods")
async def list_methods() -> JSONResponse:
    """Return available localization methods (subdirs with feedforward.zarr)."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "methods": []})
    methods = [
        p.name for p in s.output_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    ]
    return JSONResponse({"ok": True, "methods": sorted(methods)})


async def _run_sse(query_path: str) -> AsyncIterator[str]:
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"}); return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    method = s.localize_method
    extractor = s.localize_extractor
    output_dir = s.output_dir

    def run() -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": f"Method: {method}  extractor: {extractor}"})
            from collab_splats.pointcloud.localization import DinoSaladExtractor
            loc = DinoSaladExtractor()
            zarr_path = output_dir / method / "feedforward.zarr"
            index_dir = output_dir / method / "localization_index" / extractor
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Building index…"})
            loc.build_index(zarr_path, index_dir)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Querying…"})
            results = loc.query(Path(query_path), index_dir, top_k=5)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "msg": "Localization complete", "results": results})
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
async def run_localize(query_path: str = ""):
    return StreamingResponse(
        _run_sse(query_path), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
