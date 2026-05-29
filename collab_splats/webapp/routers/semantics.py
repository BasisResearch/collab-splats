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


@router.get("/queryable")
async def is_queryable() -> JSONResponse:
    """Return whether current extractor supports text queries."""
    s = get_session()
    try:
        from collab_splats.semantics.features.base import BaseQueryableExtractor  # noqa: PLC0415
        ext_cls = BaseQueryableExtractor.get(s.extractor)
        return JSONResponse({"ok": True, "queryable": True, "extractor": s.extractor})
    except Exception:
        return JSONResponse({"ok": True, "queryable": False, "extractor": s.extractor})


@router.get("/frame_viz")
async def frame_viz(idx: int = 0) -> JSONResponse:
    """Compute per-frame features, return PCA image + frame URL."""
    import base64, io  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})

    frames_dir = s.output_dir / "frames"
    jpgs = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.is_dir() else []
    if not jpgs:
        return JSONResponse({"ok": False, "error": "No extracted frames found"})

    idx = max(0, min(idx, len(jpgs) - 1))
    frame_path = jpgs[idx]
    # Build /outputs-relative URL
    frame_url = "/outputs" + str(frame_path).replace("/workspace/outputs", "")

    try:
        from PIL import Image as PILImage  # noqa: PLC0415
        from collab_splats.semantics.features import BaseFeatureExtractor  # noqa: PLC0415

        extractor = BaseFeatureExtractor.get(s.extractor)()
        img = PILImage.open(str(frame_path)).convert("RGB")
        import numpy as _np  # noqa: PLC0415
        img_array = _np.array(img)

        # Run forward pass on single frame
        with torch.no_grad():
            feats_list = extractor.forward([img_array])  # list of tensors
        if not feats_list:
            return JSONResponse({"ok": False, "error": "Extractor returned no features"})

        feats = feats_list[0]  # (H', W', D) or (D, H', W') tensor
        if feats.ndim == 3 and feats.shape[0] < feats.shape[-1]:
            feats = feats.permute(1, 2, 0)  # (H', W', D)

        # PCA → RGB via features_to_rgb
        pca_rgb = extractor.features_to_rgb(feats)  # (H', W', 3) uint8
        # Resize PCA to match frame image dimensions for consistent display
        pca_img = PILImage.fromarray(pca_rgb.astype(_np.uint8)).resize(
            (img.width, img.height), PILImage.BILINEAR
        )
        buf = io.BytesIO()
        pca_img.save(buf, format="PNG")
        pca_b64 = base64.b64encode(buf.getvalue()).decode()

        return JSONResponse({
            "ok": True,
            "frame_url": frame_url,
            "pca_b64": pca_b64,
            "n_frames": len(jpgs),
            "frame_idx": idx,
        })
    except Exception as exc:
        import traceback  # noqa: PLC0415
        return JSONResponse({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})


@router.get("/query_frame")
async def query_frame(idx: int = 0, text: str = "", neg: str = "") -> JSONResponse:
    """Run text query on a single frame; return similarity heatmap as PNG base64."""
    import base64, io  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    if not text.strip():
        return JSONResponse({"ok": False, "error": "No query text"})

    frames_dir = s.output_dir / "frames"
    jpgs = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.is_dir() else []
    if not jpgs:
        return JSONResponse({"ok": False, "error": "No frames"})

    idx = max(0, min(idx, len(jpgs) - 1))
    frame_path = jpgs[idx]

    try:
        from PIL import Image as PILImage  # noqa: PLC0415
        from collab_splats.semantics.features.base import BaseQueryableExtractor  # noqa: PLC0415
        import matplotlib.cm as _cm  # noqa: PLC0415

        extractor = BaseQueryableExtractor.get(s.extractor)()
        img = PILImage.open(str(frame_path)).convert("RGB")
        img_array = _np.array(img)

        with torch.no_grad():
            feats_list = extractor.forward([img_array])
        feats = feats_list[0]
        if feats.ndim == 3 and feats.shape[0] < feats.shape[-1]:
            feats = feats.permute(1, 2, 0)  # (H', W', D)

        H, W, D = feats.shape
        flat = feats.reshape(-1, D)  # (H*W, D)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_normed = flat / norms

        text_vec = extractor.encode_text([text])[0].detach().cpu()

        # Check if compressor is available
        comp_dir = s.output_dir / s.creator / "semantics" / s.extractor / "compressor.pt"
        if comp_dir.is_dir():
            from collab_splats.semantics.compression import FeatureAutoencoder  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: PLC0415
            comp = FeatureAutoencoder.load(comp_dir); comp.eval()
            with torch.no_grad():
                flat_normed = F.normalize(comp.per_point_encode(flat_normed), dim=-1)
                text_vec = F.normalize(comp.per_point_encode(text_vec.unsqueeze(0)), dim=-1).squeeze(0)

        sims = (flat_normed @ text_vec.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        if neg.strip():
            neg_vec = extractor.encode_text([neg])[0].detach().cpu()
            if comp_dir.is_dir():
                with torch.no_grad():
                    neg_vec = F.normalize(comp.per_point_encode(neg_vec.unsqueeze(0)), dim=-1).squeeze(0)
            sims = sims - (flat_normed @ neg_vec.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        sims_img = sims.reshape(H, W)
        mn, mx = sims_img.min(), sims_img.max()
        sims_norm = (sims_img - mn) / max(mx - mn, 1e-8)
        heatmap = (_cm.get_cmap("viridis")(sims_norm)[:, :, :3] * 255).astype(_np.uint8)
        heatmap_pil = PILImage.fromarray(heatmap).resize((img.width, img.height), PILImage.BILINEAR)
        buf = io.BytesIO()
        heatmap_pil.save(buf, format="PNG")
        return JSONResponse({
            "ok": True,
            "heatmap_b64": base64.b64encode(buf.getvalue()).decode(),
            "frame_idx": idx,
        })
    except Exception as exc:
        import traceback  # noqa: PLC0415
        return JSONResponse({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})


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
