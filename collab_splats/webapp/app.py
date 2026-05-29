from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from collab_splats.webapp.routers import preprocess as preprocess_router
from collab_splats.webapp.routers import reconstruct as reconstruct_router
from collab_splats.webapp.routers import session as session_router
from collab_splats.webapp.routers import visualize as visualize_router

_STATIC_DIR = Path(__file__).parent / "static"
_OUTPUTS_DIR = Path("/workspace/outputs")


def create_app() -> FastAPI:
    """Instantiate the FastAPI application."""
    app = FastAPI(title="collab-splats webapp")

    # Mount API routers
    app.include_router(session_router.router)
    app.include_router(preprocess_router.router)
    app.include_router(reconstruct_router.router)
    app.include_router(visualize_router.router)

    # Serve static app files (HTML, CSS, JS)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Serve reconstruction outputs (frames, video, PLY) directly as files
    if _OUTPUTS_DIR.exists():
        app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")

    _index_html = (_STATIC_DIR / "index.html").read_text()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _index_html

    return app
