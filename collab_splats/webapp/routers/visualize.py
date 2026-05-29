from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from collab_splats.webapp.state import get_session

router = APIRouter(prefix="/api/visualize")


@router.get("/status")
async def status() -> JSONResponse:
    """Return URLs for available pointcloud and mesh PLY files."""
    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})
    backend_dir = s.output_dir / s.creator
    ply = backend_dir / "sparse_pc.ply"
    mesh = backend_dir / "mesh" / "mesh.ply"
    # Build /outputs-relative URLs for the browser to fetch
    base = str(s.output_dir).replace("/workspace/outputs", "")
    return JSONResponse({
        "ok": True,
        "ply_url": f"/outputs{base}/{s.creator}/sparse_pc.ply" if ply.exists() else None,
        "mesh_url": f"/outputs{base}/{s.creator}/mesh/mesh.ply" if mesh.exists() else None,
        "creator": s.creator,
    })
