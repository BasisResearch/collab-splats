from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from collab_splats.webapp.state import get_session

# Viridis colormap (256 entries, precomputed)
_VIRIDIS_LUT: np.ndarray | None = None

def _viridis(sims: np.ndarray) -> np.ndarray:
    """Map (N,) float32 similarity scores → (N, 3) uint8 RGB via viridis."""
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is None:
        import matplotlib.cm as cm  # noqa: PLC0415
        lut = cm.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]
        _VIRIDIS_LUT = (lut * 255).astype(np.uint8)
    mn, mx = sims.min(), sims.max()
    if mx - mn < 1e-8:
        idx = np.full(len(sims), 128, dtype=np.int32)
    else:
        idx = ((sims - mn) / (mx - mn) * 255).clip(0, 255).astype(np.int32)
    return _VIRIDIS_LUT[idx]

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


@router.get("/similarity")
async def query_similarity(pos: str, neg: str = "", extractor: str = "talk2dino") -> JSONResponse:
    """Compute per-point semantic similarity to a text query; return viridis RGB as base64."""
    import torch  # noqa: PLC0415
    import zarr  # noqa: PLC0415

    s = get_session()
    if s.output_dir is None:
        return JSONResponse({"ok": False, "error": "No session loaded"})

    creator = s.creator
    feat_zarr = s.output_dir / creator / "semantics" / extractor / "features.zarr"
    if not feat_zarr.exists():
        return JSONResponse({"ok": False, "error": f"No lifted features at {feat_zarr}"})

    try:
        # Load L2-normalised lifted features (P, D)
        store = zarr.open(str(feat_zarr), mode="r")
        feats = store["features"][:]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        lifted = (feats / np.maximum(norms, 1e-8)).astype(np.float32)

        # Lazy-load extractor for text encoding
        from collab_splats.semantics.features.base import BaseQueryableExtractor  # noqa: PLC0415
        ext = BaseQueryableExtractor.get(extractor)()

        def _encode(text: str) -> np.ndarray:
            raw = ext.encode_text([text])[0].detach().cpu()
            # Load compressor if present and project into latent space
            comp_dir = s.output_dir / creator / "semantics" / extractor / "compressor.pt"
            if comp_dir.is_dir():
                from collab_splats.semantics.compression import FeatureAutoencoder  # noqa: PLC0415
                import torch.nn.functional as F  # noqa: PLC0415
                comp = FeatureAutoencoder.load(comp_dir)
                comp.eval()
                with torch.no_grad():
                    raw = F.normalize(comp.per_point_encode(raw.unsqueeze(0)), dim=-1).squeeze(0)
            vec = raw.numpy().astype(np.float32)
            n = np.linalg.norm(vec)
            return vec / n if n > 1e-8 else vec

        pos_vec = _encode(pos)
        sims = lifted @ pos_vec

        if neg.strip():
            neg_vec = _encode(neg.strip())
            sims = sims - lifted @ neg_vec

        colors = _viridis(sims)  # (P, 3) uint8
        encoded = base64.b64encode(colors.tobytes()).decode()
        return JSONResponse({"ok": True, "colors_b64": encoded, "n_points": int(len(colors))})

    except Exception as exc:
        import traceback  # noqa: PLC0415
        return JSONResponse({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})
