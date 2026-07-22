from __future__ import annotations

import asyncio
import json
import threading
import traceback
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/localize")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.get("/sample_query")
async def sample_query_image() -> JSONResponse:
    """Return path to an out-of-sample image from a different scene for query testing."""
    s = get_session()
    _OUTPUTS = Path("/workspace/outputs")

    # Find a scene that is NOT the current session's scene and has a frames.zarr store
    current = s.output_dir.name if s.output_dir else ""
    for scene_dir in sorted(_OUTPUTS.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name == current:
            continue
        frames_zarr = scene_dir / "frames.zarr"
        if not frames_zarr.exists():
            continue
        store = FrameStore.open(frames_zarr)
        if len(store) == 0:
            continue
        # Export the middle frame to a derived cache jpg — frames.zarr is the canonical source,
        # but the localize query flow needs a real file path to hand back to the client.
        mid = len(store) // 2
        query_path = scene_dir / "query_sample.jpg"
        cv2.imwrite(str(query_path), cv2.cvtColor(store.image(mid), cv2.COLOR_RGB2BGR))
        return JSONResponse({"ok": True, "path": str(query_path), "scene": scene_dir.name})
    return JSONResponse({"ok": False, "error": "No other scenes found"})


@router.get("/methods")
async def list_methods() -> JSONResponse:
    """Return available localization methods (subdirs with feedforward.zarr)."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "methods": []})
    methods = [p.name for p in s.output_dir.iterdir() if p.is_dir() and (p / "feedforward.zarr").exists()]
    return JSONResponse({"ok": True, "methods": sorted(methods)})


async def _run_sse(query_path: str, extractor_name: str = "xfeat") -> AsyncIterator[str]:
    s = get_session()
    if s.output_dir is None:
        yield _sse({"type": "error", "msg": "No session loaded"})
        return

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    method = s.localize_method or s.creator
    output_dir = s.output_dir

    def run() -> None:
        try:
            import numpy as np  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415

            from collab_splats.localization import (  # noqa: PLC0415
                CameraLocalizer,
                DiskExtractor,
                XFeatExtractor,
            )
            from collab_splats.pointcloud.feedforward.base import (  # noqa: PLC0415
                FeedforwardResult,
            )

            zarr_path = output_dir / method / "feedforward.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"feedforward.zarr not found at {zarr_path}")

            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "log", "msg": f"Method: {method}  extractor: {extractor_name}"}
            )

            # Build/load local feature extractor
            ext = XFeatExtractor() if extractor_name.lower().startswith("xfeat") else DiskExtractor()

            # Progress callback for index building
            def _progress(i: int, total: int) -> None:
                pct = int(i / max(total, 1) * 100)
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "progress", "pct": pct, "msg": f"Building index {i}/{total}"}
                )

            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Loading feedforward result…"})
            ff = FeedforwardResult.load_zarr(zarr_path, load_images=True, load_world_points=True)

            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Building/loading local feature index…"})
            # frames.zarr sits at the session root (sibling of every backend's method dir), shared
            # across methods. Boundary adapter: canonical store → (images, ids) core objects; the
            # lazy genexpr does zero reads on a cache hit, one partial-read per frame on a miss.
            frames_zarr = output_dir / "frames.zarr"
            store = FrameStore.open(frames_zarr)
            frame_indices = store.frame_indices()
            images = (store.image_by_frame_idx(fi) for fi in frame_indices)
            ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
            localizer = CameraLocalizer.from_feedforward(
                ff,
                images=images,
                ids=ids,
                extractor=ext,
                zarr_path=zarr_path,
                progress_callback=_progress,
            )

            # Load query image
            if not query_path or not Path(query_path).exists():
                raise FileNotFoundError(f"Query image not found: {query_path}")
            query_img = np.array(PILImage.open(query_path).convert("RGB"))

            # Use mean intrinsics from reconstruction as approximation for query
            query_K = ff.intrinsics.mean(0).astype(np.float32)

            loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "msg": "Localizing query image…"})
            result = localizer.localize(query_img, query_K)

            pose_info = "pose not found"
            if result.pose is not None:
                pos = result.pose[:3, 3].tolist()
                pose_info = f"position [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], {result.n_inliers} inliers"

            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "type": "done",
                    "msg": f"Localization: {pose_info}",
                    "n_inliers": result.n_inliers,
                    "pose": result.pose.tolist() if result.pose is not None else None,
                },
            )
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
async def run_localize(query_path: str = "", extractor: str = "xfeat"):
    return StreamingResponse(
        _run_sse(query_path, extractor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
