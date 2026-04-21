"""VGGT-X pointcloud creation module.

Bundles all VGGT-specific logic: load model, preprocess, inference,
unproject depth, optional global alignment, and filter.

Ported from stage/vggt_utils.py. Internal helpers (_run_vggt_inference,
_unproject_and_filter_points, _run_global_alignment) are GPU-only and raise
NotImplementedError when VGGT-X is not installed; they are mocked in unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional VGGT-X imports (GPU path only)
# ---------------------------------------------------------------------------
try:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_ratio
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    _HAS_VGGT = True
except ImportError:
    _HAS_VGGT = False

try:
    from streamvggt.models.streamvggt import StreamVGGT  # noqa: F401
    _HAS_STREAM_VGGT = True
except ImportError:
    _HAS_STREAM_VGGT = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_vggt(
    image_dir: Path,
    colmap_dir: Path,
    model_name: str,
    *,
    use_global_alignment: bool = False,
    chunk_size: int = 256,
    conf_threshold: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Path], np.ndarray, int, int]:
    """Load + preprocess + inference + unproject + optional global alignment + filter.

    Args:
        image_dir: Directory containing input images.
        colmap_dir: Output directory for COLMAP reconstruction.
        model_name: HuggingFace model name (e.g. "facebook/VGGT-1B").
        use_global_alignment: If True, run cross-camera pose refinement.
        chunk_size: Chunk size for chunked inference (memory efficiency).
        conf_threshold: Depth confidence percentile threshold (0–100).

    Returns:
        (pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h)
        - pts3d:          (P, 3) float32 world-space points
        - colors:         (P, 3) uint8 RGB
        - extrinsics:     (N, 3, 4) float32
        - intrinsics:     (N, 3, 3) float32  [downsampled to VGGT resolution]
        - image_paths:    list of N Path objects
        - original_coords: (N, 6) float32  [0,0,new_w,new_h,orig_w,orig_h]
        - model_w:        int  model output width
        - model_h:        int  model output height
    """
    if not _is_vggt_available():
        raise RuntimeError(
            "VGGT-X not installed. "
            "pip install git+https://github.com/Linketic/VGGT-X.git"
        )

    vggt_data = _run_vggt_inference(image_dir, colmap_dir, model_name, chunk_size=chunk_size)

    extrinsic = vggt_data["extrinsic"]
    intrinsic = vggt_data["instrinsics_downsampled"]  # note: typo preserved from stage

    extrinsic, intrinsic = _maybe_run_global_alignment(
        vggt_data, extrinsic, intrinsic, colmap_dir,
        use_global_alignment=use_global_alignment,
    )

    pts3d, colors = _unproject_and_filter_points(
        depth=vggt_data["depth"],
        depth_conf=vggt_data["depth_conf"],
        images=vggt_data["images"],
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        conf_threshold=conf_threshold,
    )

    model_h: int = int(vggt_data["depth"].shape[1])
    model_w: int = int(vggt_data["depth"].shape[2])
    return (
        pts3d,
        colors,
        extrinsic,
        intrinsic,
        vggt_data["image_paths"],
        vggt_data["original_coords"],
        model_w,
        model_h,
    )


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _is_vggt_available() -> bool:
    """Check if VGGT-X (or StreamVGGT) is installed.

    Adapts stage/vggt_utils.py: _HAS_VGGT / is_vggt_available() L104–111.
    """
    return _HAS_VGGT or _HAS_STREAM_VGGT


# ---------------------------------------------------------------------------
# Global alignment wrapper
# ---------------------------------------------------------------------------

def _maybe_run_global_alignment(
    vggt_data: Dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    colmap_dir: Path,
    *,
    use_global_alignment: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run global alignment if requested, else return cameras unchanged."""
    if not use_global_alignment:
        return extrinsic, intrinsic
    refined_ext, refined_int, _ = _run_global_alignment(
        images=vggt_data["images"],
        image_paths=vggt_data["image_paths"],
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_conf=vggt_data["depth_conf"],
        colmap_dir=colmap_dir,
        depth_map=vggt_data["depth"],
    )
    return refined_ext, refined_int


# ---------------------------------------------------------------------------
# Internal helpers (GPU path — ported from stage/vggt_utils.py)
# ---------------------------------------------------------------------------

def _run_vggt_inference(
    image_dir: Path,
    colmap_dir: Path,
    model_name: str,
    chunk_size: int = 256,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load model + preprocess + forward pass.

    Adapts stage/vggt_utils.py:_run_vggt_inference L974–1119.

    Returns dict with keys:
        images, extrinsic, instrinsics (typo preserved), instrinsics_downsampled,
        depth, depth_conf, image_paths, original_coords.
    """
    if not _HAS_VGGT:
        raise RuntimeError(
            "VGGT-X not installed. "
            "pip install git+https://github.com/Linketic/VGGT-X.git"
        )

    from PIL import Image as PILImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    # Load VGGT model
    model = VGGT.from_pretrained(model_name, chunk_size=chunk_size)
    model.eval()
    model = model.to(device, dtype=dtype)

    # Find and sort image files
    image_dir = Path(image_dir)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])

    # Prepare paths as strings for VGGT preprocessing
    image_names = [str(p) for p in image_paths]

    # Preprocess: keep aspect ratio, scale largest dim to 518
    img_load_resolution = 518
    images, original_coords = load_and_preprocess_images_ratio(image_names, img_load_resolution)

    images = images.to(device, dtype=dtype)
    original_coords = original_coords.to(device)

    width, height = original_coords[0, -2:]
    image_shape = images.shape[-2:]  # (H, W) at VGGT resolution

    # Forward pass (no grad)
    with torch.no_grad():
        predictions = model(images.unsqueeze(0))

    # Decode pose encodings at two resolutions
    extrinsic_ds, intrinsic_ds = pose_encoding_to_extri_intri(
        predictions["pose_enc"], image_shape
    )
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], [width, height]
    )

    # Move to CPU float32
    extrinsic = extrinsic.cpu().float().numpy().squeeze(0)
    intrinsic = intrinsic.cpu().float().numpy().squeeze(0)
    intrinsic_downsampled = intrinsic_ds.cpu().float().numpy().squeeze(0)
    depth_map = predictions["depth"].squeeze(0).cpu().float().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()

    # Free GPU
    if torch.cuda.is_available():
        model = model.cpu()
        del model
        torch.cuda.empty_cache()

    return {
        "images": images,
        "extrinsic": extrinsic,
        "instrinsics": intrinsic,                      # typo preserved from stage
        "instrinsics_downsampled": intrinsic_downsampled,  # typo preserved from stage
        "depth": depth_map,
        "depth_conf": depth_conf,
        "image_paths": image_paths,
        "original_coords": original_coords.cpu().float().numpy(),
    }


def _unproject_and_filter_points(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    images: Any,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    conf_threshold: float = 50.0,
    max_points: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject depth to world-space points and filter by confidence.

    Adapts:
    - stage/vggt_utils.py: unproject_depth_map_to_point_map call (L~220)
    - stage/vggt_utils.py: _filter_and_prepare_points_for_pycolmap L1888–1942

    Args:
        depth:        (N, H, W) float32 depth maps.
        depth_conf:   (N, H, W) float32 confidence maps.
        images:       (N, 3, H, W) tensor or array of preprocessed images.
        extrinsic:    (N, 3, 4) or (N, 4, 4) camera extrinsics.
        intrinsic:    (N, 3, 3) camera intrinsics.
        conf_threshold: Confidence percentile (0–100) or raw value (0–1).
        max_points:   Maximum number of output points (random subsample if exceeded).

    Returns:
        pts3d:  (P, 3) float32 world-space points.
        colors: (P, 3) uint8 RGB.
    """
    if not _HAS_VGGT:
        raise RuntimeError(
            "VGGT-X not installed. "
            "pip install git+https://github.com/Linketic/VGGT-X.git"
        )

    # Unproject depth -> (N, H, W, 3) world-space point map
    points3d = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    # Extract RGB colors from images tensor
    if hasattr(images, "cpu"):
        images_np = images.cpu().float().numpy()
    else:
        images_np = np.asarray(images, dtype=np.float32)
    # images_np: (N, 3, H, W) -> (N, H, W, 3)
    colors_np = images_np.transpose(0, 2, 3, 1)

    # Confidence threshold: if > 1 treat as percentile, else raw value
    if conf_threshold > 1.0:
        threshold_val = float(np.percentile(depth_conf, conf_threshold))
    else:
        threshold_val = float(conf_threshold)

    conf_mask = depth_conf >= threshold_val  # (N, H, W) bool

    # Subsample if too many points
    n_true = int(conf_mask.sum())
    if n_true > max_points:
        conf_mask = randomly_limit_trues(conf_mask, max_points)

    pts_out = points3d[conf_mask].astype(np.float32)           # (P, 3)
    colors_out = (colors_np[conf_mask] * 255).astype(np.uint8) # (P, 3)

    return pts_out, colors_out


def _run_global_alignment(
    images: Any,
    image_paths: List[Path],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_conf: np.ndarray,
    colmap_dir: Path,
    depth_map: Optional[np.ndarray] = None,
    max_query_pts: int = 4096,
    shared_camera: bool = False,
    verbose: bool = False,
    lambda_depth: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Refine camera poses via feature matching + pose optimization.

    Adapts stage/vggt_utils.py:_run_global_alignment L1943–2035.

    Returns:
        (extrinsic_refined, intrinsic_refined, match_outputs)
    """
    if not _HAS_VGGT:
        raise RuntimeError(
            "VGGT-X not installed. "
            "pip install git+https://github.com/Linketic/VGGT-X.git"
        )

    from vggt.dependency.global_alignment import extract_matches, pose_optimization  # type: ignore

    colmap_dir = Path(colmap_dir)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    image_basenames = [p.name for p in image_paths]

    matches_path = colmap_dir / "matches.pt"
    match_outputs = extract_matches(
        images=images,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        base_image_path_list=image_basenames,
        max_query_pts=max_query_pts,
    )
    torch.save(match_outputs, matches_path)

    extrinsic_refined, intrinsic_refined = pose_optimization(
        match_outputs=match_outputs,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        images=images,
        depth_conf=depth_conf,
        base_image_path_list=image_basenames,
        target_scene_dir=colmap_dir,
        shared_intrinsics=shared_camera,
        depth_maps=depth_map,
        lambda_depth=lambda_depth,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return extrinsic_refined, intrinsic_refined, match_outputs
