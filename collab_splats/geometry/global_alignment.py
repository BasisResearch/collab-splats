"""VGGT-X native global alignment: feature matching + joint BA over poses/intrinsics.

Parked utility. Not wired into any creator — VGGTXCreator's call site is commented
out (see ``pointcloud/feedforward/vggtx.py``). Kept as an alternative to the LM
bundle adjustment in ``bundle_adjustment.py`` for a future accuracy comparison
(bae-vggt-parity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from vggt.dependency.global_alignment import extract_matches, pose_optimization  # type: ignore

    _HAS_VGGT = True
except ImportError:
    _HAS_VGGT = False


def run_global_alignment(
    raw_outputs: Dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_paths: List[Path],
    colmap_dir: Optional[Path] = None,
    max_query_pts: int = 4096,
    shared_camera: bool = False,
    lambda_depth: float = 0.0,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine camera poses via feature matching and bundle adjustment.

    Two-stage pipeline:
      1. extract_matches   — build 2D-3D correspondences across all image pairs.
                             Result is cached to ``colmap_dir/matches.pt``.
                             Set ``overwrite=False`` to reload from cache, useful
                             when tuning ``lambda_depth`` without re-running inference.
      2. pose_optimization — joint BA over extrinsics and intrinsics.

    Args:
        raw_outputs:   Forward-pass output dict. Required keys:
                         'images'     — (N, C, H, W) float tensor
                         'depth'      — (N, H, W) float32 depth map
                         'depth_conf' — (N, H, W) float32 per-pixel confidence
        extrinsic:     (N, 3, 4) or (N, 4, 4) float32 world-to-camera matrices.
        intrinsic:     (N, 3, 3) float32 camera intrinsics K.
        image_paths:   Ordered image Paths; basenames used by COLMAP matchers.
        colmap_dir:    Working directory for ``matches.pt``. Created if absent.
                       Defaults to ``Path(".")``.
        max_query_pts: Max 2D-3D correspondences per image for PnP.
        shared_camera: True if all images share one camera model.
        lambda_depth:  Depth regularization weight in BA. 0 = reprojection only.
        overwrite:     When False, load ``matches.pt`` from disk if it exists.

    Returns:
        Tuple of (refined_extrinsic, refined_intrinsic), same shapes as inputs.
    """
    if not _HAS_VGGT:
        raise ImportError("VGGT-X not installed. " "pip install git+https://github.com/Linketic/VGGT-X.git")

    colmap_dir = Path(colmap_dir) if colmap_dir else Path(".")
    colmap_dir.mkdir(parents=True, exist_ok=True)
    matches_path = colmap_dir / "matches.pt"

    # extract_matches requires bare filenames (no parent paths) for the match table.
    image_basenames = [p.name for p in image_paths]

    if not overwrite and matches_path.exists():
        # Reload cached correspondences — skip the expensive matching step.
        match_outputs = torch.load(matches_path)
    else:
        # Build 2D-3D correspondences across all image pairs and cache to disk.
        match_outputs = extract_matches(
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            base_image_path_list=image_basenames,
            max_query_pts=max_query_pts,
        )
        torch.save(match_outputs, matches_path)

    # Run joint bundle adjustment over extrinsics and intrinsics.
    extrinsic_refined, intrinsic_refined = pose_optimization(
        match_outputs=match_outputs,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        images=raw_outputs["images"],
        depth_conf=raw_outputs["depth_conf"],
        base_image_path_list=image_basenames,
        target_scene_dir=colmap_dir,
        shared_intrinsics=shared_camera,
        depth_maps=raw_outputs["depth"],
        lambda_depth=lambda_depth,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return extrinsic_refined, intrinsic_refined
