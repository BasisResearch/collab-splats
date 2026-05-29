from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from collab_splats.webapp.state import get_session

_OUTPUTS_DIR = Path("/workspace/outputs")

router = APIRouter(prefix="/api/session")


@router.get("/list")
async def list_sessions() -> JSONResponse:
    """Return sorted names of output dirs that contain run_config.yaml."""
    if not _OUTPUTS_DIR.is_dir():
        return JSONResponse({"ok": False, "sessions": []})
    sessions = sorted(
        p.name for p in _OUTPUTS_DIR.iterdir()
        if p.is_dir() and (p / "run_config.yaml").exists()
    )
    return JSONResponse({"ok": True, "sessions": sessions})


@router.post("/load")
async def load_session(body: dict[str, Any]) -> JSONResponse:
    """Load an existing output directory as the current session."""
    raw = body.get("output_dir", "")
    out_dir = Path(raw)
    if not out_dir.is_dir():
        return JSONResponse({"ok": False, "error": f"Not a directory: {raw}"})

    session = get_session()
    session.output_dir = out_dir

    # Try to read video_path from run_config.yaml
    config_file = out_dir / "run_config.yaml"
    video_path = None
    if config_file.exists():
        try:
            cfg = yaml.safe_load(config_file.read_text())
            raw_vp = cfg.get("video_path") or cfg.get("input_path")
            if raw_vp and Path(raw_vp).exists():
                video_path = Path(raw_vp)
        except Exception:
            pass
    session.video_path = video_path

    # Auto-detect creator: first backend subdir with feedforward.zarr
    available_backends = sorted(
        p.name for p in out_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    ) if out_dir.is_dir() else []
    if available_backends:
        session.creator = available_backends[0]

    # Auto-detect extractor: first subdir present under features/
    features_dir = out_dir / "features"
    available_extractors = sorted(
        p.name for p in features_dir.iterdir() if p.is_dir()
    ) if features_dir.is_dir() else []
    if available_extractors:
        session.extractor = available_extractors[0]

    return JSONResponse({
        "ok": True,
        "output_dir": str(out_dir),
        "video_path": str(video_path) if video_path else None,
        "name": out_dir.name,
        "creator": session.creator,
        "extractor": session.extractor,
        "available_backends": available_backends,
        "available_extractors": available_extractors,
    })


@router.get("/current")
async def current_session() -> JSONResponse:
    """Return current session state (all serialisable fields)."""
    s = get_session()
    return JSONResponse({
        "output_dir": str(s.output_dir) if s.output_dir else None,
        "video_path": str(s.video_path) if s.video_path else None,
        "creator": s.creator,
        "conf": s.conf,
        "extractor": s.extractor,
        "localize_method": s.localize_method,
        "localize_extractor": s.localize_extractor,
    })


@router.get("/video")
async def serve_video() -> FileResponse:
    """Stream the session video file."""
    s = get_session()
    if s.video_path is None or not s.video_path.exists():
        return JSONResponse({"error": "No video loaded"}, status_code=404)
    return FileResponse(str(s.video_path), media_type="video/mp4")


@router.post("/update")
async def update_session(body: dict[str, Any]) -> JSONResponse:
    """Partial update of session config (creator, conf, extractor, etc.)."""
    s = get_session()
    for key in ("creator", "conf", "extractor", "localize_method", "localize_extractor"):
        if key in body:
            setattr(s, key, body[key])
    return JSONResponse({"ok": True})
