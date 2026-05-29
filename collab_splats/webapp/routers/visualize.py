from __future__ import annotations

import json
from pathlib import Path

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

    # Scan all backend subdirs that contain a sparse_pc.ply
    available = sorted(
        p.name for p in s.output_dir.iterdir()
        if p.is_dir() and (p / "sparse_pc.ply").exists()
    ) if s.output_dir.is_dir() else []

    # Use current creator if it has a PLY; fall back to first available
    creator = s.creator if (s.output_dir / s.creator / "sparse_pc.ply").exists() else (available[0] if available else s.creator)

    backend_dir = s.output_dir / creator
    ply = backend_dir / "sparse_pc.ply"
    mesh = backend_dir / "mesh" / "mesh.ply"
    base = str(s.output_dir).replace("/workspace/outputs", "")

    # Load ground plane transform from transforms.json if present
    ground_plane = None
    transforms_path = backend_dir / "transforms.json"
    if transforms_path.exists():
        try:
            data = json.loads(transforms_path.read_text())
            gp = data.get("ground_plane")
            if gp and "R" in gp and "t" in gp:
                ground_plane = {"R": gp["R"], "t": gp["t"]}
        except Exception:
            pass

    return JSONResponse({
        "ok": True,
        "ply_url": f"/outputs{base}/{creator}/sparse_pc.ply" if ply.exists() else None,
        "mesh_url": f"/outputs{base}/{creator}/mesh/mesh.ply" if mesh.exists() else None,
        "creator": creator,
        "available_backends": available,
        "ground_plane": ground_plane,
    })
