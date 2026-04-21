# collab_splats/pointcloud/_mapanything.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import voxel_downsample_point_cloud


def run_mapanything(
    image_dir: Path,
    model_name: str,
    *,
    confidence_percentile: float = 35.0,
    use_multiview_confidence: bool = True,
    minibatch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], np.ndarray, int, int]:
    """Load + preprocess + inference + collect pts3d + voxel downsample.

    Returns:
        (pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h)
    """
    model = _load_mapanything_model(model_name)
    views, image_paths, original_coords = _load_and_preprocess_images(image_dir)
    model_w: int = views[0]["img"].shape[-1]
    model_h: int = views[0]["img"].shape[-2]
    outputs = _run_mapanything_inference(
        model, views,
        confidence_percentile=confidence_percentile,
        use_multiview_confidence=use_multiview_confidence,
        minibatch_size=minibatch_size,
    )
    pts3d, colors, extrinsics, intrinsics = _collect_pts3d_from_outputs(outputs)
    pts3d, colors = voxel_downsample_point_cloud(pts3d, colors)
    return pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h


def _load_mapanything_model(model_name: str) -> Any:
    """Load MapAnything model. Adapts stage/mapanything_utils.py:L57–123."""
    try:
        from mapanything import MapAnything
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


def _load_and_preprocess_images(
    image_dir: Path,
) -> tuple[list[dict], list[Path], np.ndarray]:
    """Load images, return views, image_paths, original_coords (N,6)."""
    from PIL import Image as PILImage
    try:
        from mapanything.utils.image import load_images
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix in exts)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    views = load_images([str(p) for p in image_paths])
    model_h: int = views[0]["img"].shape[-2]
    model_w: int = views[0]["img"].shape[-1]

    original_coords = np.array(
        [
            [0, 0, model_w, model_h, PILImage.open(p).width, PILImage.open(p).height]
            for p in image_paths
        ],
        dtype=np.float32,
    )
    return views, image_paths, original_coords


def _run_mapanything_inference(
    model: Any,
    views: list[dict],
    *,
    confidence_percentile: float,
    use_multiview_confidence: bool,
    minibatch_size: int,
) -> list[dict]:
    """Run model.infer(). Adapts stage/mapanything_utils.py:L208–312."""
    import torch
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=minibatch_size,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            use_multiview_confidence=use_multiview_confidence,
            confidence_percentile=confidence_percentile,
        )
    return outputs


def _collect_pts3d_from_outputs(
    outputs: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pts3d, colors, extrinsics, intrinsics from MapAnything output dicts.

    Adapts data collection loop from stage/mapanything_utils.py:export_predictions_to_colmap_internal L761–830.

    Returns:
        pts3d: (P, 3) float32 world-space points (all frames concatenated)
        colors: (P, 3) uint8 RGB
        extrinsics: (N, 3, 4) float32 world2cam [R|t]
        intrinsics: (N, 3, 3) float32 K matrices
    """
    try:
        from mapanything.utils.geometry import closed_form_pose_inverse
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    extrinsics_list: list[np.ndarray] = []

    for pred in outputs:
        # 3D points and confidence/validity mask
        pts3d = pred["pts3d"][0].cpu().numpy()          # (H, W, 3)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

        # Filter by valid depth (camera Z > 0)
        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
        valid_depth_mask = depth_z > 0
        combined_mask = mask & valid_depth_mask

        # Colors from denormalized image [0, 1] -> uint8
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3) in [0, 1]
        colors = (img_no_norm * 255).astype(np.uint8)

        all_points.append(pts3d[combined_mask])
        all_colors.append(colors[combined_mask])

        # Intrinsics (K matrix)
        intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())

        # Convert cam2world -> world2cam [R|t] (3, 4)
        cam2world = pred["camera_poses"][0].cpu().numpy()  # (4, 4)
        world2cam = closed_form_pose_inverse(cam2world[None])[0]  # (4, 4)
        extrinsics_list.append(world2cam[:3, :4])

    pts3d_all = np.concatenate(all_points, axis=0)          # (P, 3)
    colors_all = np.concatenate(all_colors, axis=0)         # (P, 3)
    intrinsics = np.stack(intrinsics_list)                   # (N, 3, 3)
    extrinsics = np.stack(extrinsics_list)                   # (N, 3, 4)

    return pts3d_all, colors_all, extrinsics, intrinsics
