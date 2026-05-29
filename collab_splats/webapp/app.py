from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"
_OUTPUTS_DIR = Path("/workspace/outputs")


def create_app() -> FastAPI:
    """Instantiate the FastAPI application."""
    app = FastAPI(title="collab-splats webapp")

    # Serve static app files (HTML, CSS, JS)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Serve reconstruction outputs (frames, video, PLY) directly as files
    if _OUTPUTS_DIR.exists():
        app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (_STATIC_DIR / "index.html").read_text()

    return app
