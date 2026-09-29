"""
Metric depth for the sfm path: estimate it with VDA, then align it to the sfm model's scale.

- estimate_depth: Video-Depth-Anything metric depth per keyframe, cached under depth_vda/
- align_depth: sfm COLMAP model + VDA depth -> PointcloudResult at the COLMAP world scale
- weights: Metric-Video-Depth-Anything-Large, license cc-by-nc-4.0 (non-commercial)
- VDA inference follows https://github.com/DepthAnything/Video-Depth-Anything @ 4f5ae23, run.py:45-57
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from huggingface_hub import hf_hub_download

from collab_splats.geometry.projection import unproject
from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    rescale_intrinsics,
    shift_intrinsics,
)
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.utils.torch_utils import get_device, pytorch_gc, vendored_path

logger = logging.getLogger(__name__)

# Location of the Video-Depth-Anything source that setup.sh clones
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"


########################################################################
# Depth estimation
########################################################################


def estimate_depth(
    frames: np.ndarray,
    out_dir: Path,
    names: list[str],
    *,
    depth_width: int = 518,
    fp32: bool = False,
) -> np.ndarray:
    """
    VDA metric depth for each keyframe, written in the layout InstantSfM reads.

    - writes out_dir/depth_vda/images/npy/<stem>.npy, one map per frame
    - a complete cache (one npy per stem) is loaded as is; depth_width is not re-applied
    - a partial cache is deleted and recomputed
    - maps are nearest-resized to depth_width, so depth edges never blend

    Args:
        frames: (N, H, W, 3) uint8 RGB, in images/ order.
        out_dir: run directory; depth_vda/ is written under it.
        names: one filename per frame, in `frames` order.
        depth_width: output map width in pixels; 518 matches the feedforward model grid.
        fp32: run inference in float32 instead of fp16 autocast.

    Returns:
        (N, h, depth_width) float32 metric depth, the maps on disk.

    Raises:
        ValueError: names and frames differ in length.
    """
    # Check there is exactly one name per frame
    if len(names) != len(frames):
        raise ValueError(f"names ({len(names)}) and frames ({len(frames)}) must align one-to-one")

    # Work out where the cached depth maps live
    depth_dir = Path(out_dir) / "depth_vda"
    npy_dir = depth_dir / "images" / "npy"
    stems = [Path(n).stem for n in names]

    # If every depth map is already cached, load them and return
    if _depth_cache_complete(npy_dir, names):
        logger.info("VDA depth exists at %s (%d maps) — loading", npy_dir, len(stems))

        return np.stack([np.load(npy_dir / f"{s}.npy") for s in stems])

    # Otherwise clear any partial cache and load the model
    shutil.rmtree(depth_dir, ignore_errors=True)
    device = get_device()
    model = _load_vda_model(device)

    # Predict metric depth for the whole video
    logger.info("VDA metric inference: %d frames (writing %d maps)", len(frames), len(names))
    depths, _fps = model.infer_video_depth(frames, target_fps=1.0, input_size=518, device=device, fp32=fp32)
    depths = np.asarray(depths, dtype=np.float32)

    # Free GPU memory, since the rest of this runs on the CPU
    del model
    pytorch_gc()

    # Shrink each depth map to depth_width and save one file per frame
    h, w = depths.shape[1:3]
    depth_hw = (int(round(depth_width * h / w)), depth_width)
    npy_dir.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(names), depth_hw[0], depth_hw[1]), dtype=np.float32)

    for i, (stem, depth) in enumerate(zip(stems, depths, strict=True)):
        small = cv2.resize(depth, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST)
        np.save(npy_dir / f"{stem}.npy", small)
        out[i] = small

    logger.info("VDA depths written: %s (%d maps @ %dx%d)", npy_dir, len(names), depth_hw[1], depth_hw[0])

    return out


########################################################################
# Alignment
########################################################################


def align_depth(
    reconstruction: pycolmap.Reconstruction,
    depths: np.ndarray,
    images: np.ndarray,
    names: list[str],
    *,
    min_obs: int = 20,
) -> tuple[PointcloudResult, dict]:
    """
    PointcloudResult at COLMAP world scale from an sfm model and VDA depth maps.

    - the depth grid is the model grid: model_intrinsics, images and pixel_indices use it
    - `intrinsics` is the COLMAP K on the keyframe grid
    - depth is scaled per frame to the COLMAP world: median of COLMAP / VDA track depths
    - frames with fewer than min_obs track depths use the global median scale
    - confidence is None: sfm has no per-pixel confidence

    Args:
        reconstruction: sfm model; its sorted image names equal the stems of `names`.
        depths: (N, h, w) VDA metric depth.
        images: (N, H, W, 3) uint8 RGB at keyframe resolution.
        names: keyframe filenames, sorted, all registered in `reconstruction`.
        min_obs: valid track depths a frame needs for its own scale.

    Returns:
        (result, attrs); attrs holds the per-frame scales for save_zarr.

    Raises:
        RuntimeError: the model did not register every frame.
        ValueError: names, depths, images or camera sizes disagree, or no frame reaches min_obs.
    """
    stems = [Path(n).stem for n in names]

    # Check that the reconstruction has a pose for every frame
    if len(reconstruction.images) != len(stems):
        raise RuntimeError(
            f"the sfm model registered {len(reconstruction.images)}/{len(stems)} frames — names "
            "must be subset to the registered frames first"
        )

    # Check that the frame names match the reconstruction's image names in order
    images_sorted = sorted(reconstruction.images.values(), key=lambda im: im.name)
    registered = [im.name for im in images_sorted]

    if registered != stems:
        raise ValueError(
            f"registered image names do not match the requested frames in order (first "
            f"registered: {registered[0]}, first expected: {stems[0]}); the frame store and "
            "the reconstruction describe different runs."
        )

    name_to_row = {name: row for row, name in enumerate(registered)}
    depths = np.asarray(depths, dtype=np.float32)
    n, h, w = depths.shape

    # Check there is one depth map per name
    if len(stems) != n:
        raise ValueError(f"{len(stems)} image names for {n} depth maps — rows would misalign")

    # Check there is one image per depth map
    if len(images) != n:
        raise ValueError(f"{len(images)} frames for {n} depth maps — rows would misalign")

    # Collect each frame's world-to-camera pose as a 4x4 matrix
    w2c_stack = np.stack([im.cam_from_world().matrix() for im in images_sorted])
    extrinsics = extrinsics_to_homogeneous(w2c_stack)
    extrinsics = extrinsics.astype(np.float32)

    # Check that the COLMAP cameras have the same resolution as the frames
    orig_h, orig_w = images.shape[1:3]
    cam_dims = {
        (reconstruction.cameras[im.camera_id].width, reconstruction.cameras[im.camera_id].height)
        for im in images_sorted
    }

    if cam_dims != {(orig_w, orig_h)}:
        raise ValueError(
            f"COLMAP camera resolution {sorted(cam_dims)} does not match the frames "
            f"({orig_w}x{orig_h}); the keyframe images / SIFT database came from a different store."
        )

    # Get the COLMAP intrinsics and the scale from frame size to depth-map size
    sx, sy = w / orig_w, h / orig_h
    colmap_intrinsics = np.stack([reconstruction.cameras[im.camera_id].calibration_matrix() for im in images_sorted])

    # Scale the intrinsics down to the depth-map size
    intrinsics = rescale_intrinsics(colmap_intrinsics, (orig_h, orig_w), (h, w)).astype(np.float32)

    # Find a scale per frame that matches the predicted depth to the COLMAP depth
    scales = np.full(n, np.nan)
    pooled_ratios: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []

    for row, image in enumerate(images_sorted):
        # Get the COLMAP depth of every 3D point seen in this frame
        observations = [p for p in image.points2D if p.has_point3D()]

        if not observations:
            continue

        xyz = np.stack([reconstruction.points3D[p.point3D_id].xyz for p in observations])
        d_colmap = (image.cam_from_world() * xyz)[:, 2]

        # Look up the predicted depth at each point's pixel
        xy = np.stack([p.xy for p in observations])
        u = np.floor(xy[:, 0] * sx).astype(np.int64)
        v = np.floor(xy[:, 1] * sy).astype(np.int64)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d_vda = np.zeros(len(observations))
        d_vda[in_bounds] = depths[row, v[in_bounds], u[in_bounds]]

        # Compute depth ratios where both depths are positive
        valid = in_bounds & (d_vda > 0) & (d_colmap > 0)

        if not valid.any():
            continue

        ratios = d_colmap[valid] / d_vda[valid]
        pooled_ratios.append(ratios)
        pooled_rows.append(np.full(len(ratios), row))

        if len(ratios) >= min_obs:
            scales[row] = np.median(ratios)

    # Fail if no frame has enough points to fit a scale
    fitted = ~np.isnan(scales)

    if not fitted.any():
        raise ValueError(
            f"depth alignment: no frame has >= {min_obs} valid track observations — "
            "the reconstruction is too sparse to align VDA depth to the COLMAP world."
        )

    # Give frames with too few points the median scale of the others
    global_scale = float(np.median(scales[fitted]))
    fallback_frames = [registered[i] for i in np.flatnonzero(~fitted)]

    if fallback_frames:
        logger.warning(
            "depth alignment: %d frames under %d obs (first: %s) — using global scale",
            len(fallback_frames),
            min_obs,
            fallback_frames[0],
        )

    scales[~fitted] = global_scale

    # Log how well one global scale fits compared with per-frame scales
    ratios_all = np.concatenate(pooled_ratios)
    rows_all = np.concatenate(pooled_rows).astype(np.int64)
    logger.info(
        "depth alignment: global scale %.4f, ratio p10/p50/p90 %s -> %s, %d fallback frames",
        global_scale,
        [round(float(x), 4) for x in np.percentile(ratios_all / global_scale, [10, 50, 90])],
        [round(float(x), 4) for x in np.percentile(ratios_all / scales[rows_all], [10, 50, 90])],
        len(fallback_frames),
    )

    # Rescale each depth map by its frame's scale
    depths = (depths * scales[:, None, None]).astype(np.float32)

    # Collect the COLMAP 3D points that were seen in at least one image
    point3d_ids = sorted(pid for pid, p in reconstruction.points3D.items() if len(p.track.elements) > 0)
    points = np.array([reconstruction.points3D[pid].xyz for pid in point3d_ids], dtype=np.float32).reshape(-1, 3)
    colors = np.array([reconstruction.points3D[pid].color for pid in point3d_ids], dtype=np.uint8).reshape(-1, 3)
    pixel_indices = _pixel_indices_from_reconstruction(
        reconstruction, point3d_ids, name_to_row, scale_x=sx, scale_y=sy, depth_hw=(h, w)
    )

    # Resize the images to the depth-map size
    images_arr = np.stack([cv2.resize(images[i], (w, h), interpolation=cv2.INTER_AREA) for i in range(n)])
    images_arr = images_arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    # Shift K half a pixel: COLMAP pixel centers sit at +0.5, unproject samples at integer pixels
    centered_intrinsics = shift_intrinsics(intrinsics, (-0.5, -0.5)).astype(np.float32)

    # Turn the rescaled depth maps into 3D points
    depth_t = torch.from_numpy(depths)
    world_to_cam = torch.from_numpy(extrinsics)
    intrinsics_t = torch.from_numpy(centered_intrinsics)
    world_points = unproject(depth_t, world_to_cam, intrinsics_t)
    world_points = world_points.numpy()

    # Mark each frame as uncropped
    original_coords = np.tile(np.array([0, 0, orig_w, orig_h, orig_w, orig_h], dtype=np.float32), (n, 1))

    # Package everything, keeping both the full-size and depth-size intrinsics
    result = PointcloudResult(
        points=points,
        colors=colors,
        extrinsics=extrinsics,
        intrinsics=colmap_intrinsics.astype(np.float32),
        model_intrinsics=intrinsics,
        image_paths=[Path(im.name) for im in images_sorted],
        original_coords=original_coords,
        model_width=w,
        model_height=h,
        images=torch.from_numpy(images_arr),
        world_points=world_points,
        depth=depths,
        pixel_indices=pixel_indices,
    )

    # Record the scales used, to be saved with the result
    attrs = {
        "depth_scale": "colmap",
        "depth_scales": [float(s) for s in scales],
        "depth_scale_fallback_frames": fallback_frames,
    }

    return result, attrs


########################################################################
# Private helpers
########################################################################


def _load_vda_model(device: str) -> torch.nn.Module:
    """
    VDA metric vitl model on `device`, weights from the Hugging Face hub.

    - separate from estimate_depth so tests can stub it without a GPU
    """
    # Import Video-Depth-Anything from its cloned source folder
    with vendored_path(VDA_ROOT, "run setup.sh (clones Video-Depth-Anything at 4f5ae23)"):
        from video_depth_anything.video_depth import VideoDepthAnything

    # Download the model weights from Hugging Face
    try:
        ckpt = hf_hub_download(
            repo_id="depth-anything/Metric-Video-Depth-Anything-Large",
            filename="metric_video_depth_anything_vitl.pth",
        )
    except Exception as exc:
        raise RuntimeError(
            "VDA metric checkpoint unavailable (depth-anything/Metric-Video-Depth-Anything-Large, "
            f"~1.5 GB, cached under HF_HOME): {exc}"
        ) from exc

    # Build the large metric-depth model and load the weights
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)

    return model.to(device).eval()


def _depth_cache_complete(npy_dir: Path, names: list[str]) -> bool:
    """
    Whether the VDA depth cache holds exactly one .npy per requested stem.

    - a missing stem or a leftover from another keyframe set is a miss
    """
    return npy_dir.is_dir() and {p.stem for p in npy_dir.glob("*.npy")} == {Path(n).stem for n in names}


def _pixel_indices_from_reconstruction(
    recon: pycolmap.Reconstruction,
    point3d_ids: list[int],
    name_to_row: dict[str, int],
    scale_x: float,
    scale_y: float,
    depth_hw: tuple[int, int],
) -> np.ndarray:
    """
    (P, 3) int32 [frame, row, col] source pixel of each point, from COLMAP tracks.

    - uses the point's first track observation, scaled to the depth grid
    - sfm has no dense source pixel, so the observing keypoint stands in
    """
    h, w = depth_hw
    out = np.zeros((len(point3d_ids), 3), dtype=np.int32)

    # Use the first image that saw each point, and its pixel on the depth map
    for i, pid in enumerate(point3d_ids):
        elem = recon.points3D[pid].track.elements[0]
        image = recon.images[elem.image_id]
        xy = image.points2D[elem.point2D_idx].xy
        col = np.floor(xy[0] * scale_x)
        col = int(np.clip(col, 0, w - 1))
        row = np.floor(xy[1] * scale_y)
        row = int(np.clip(row, 0, h - 1))
        out[i] = (name_to_row[image.name], row, col)

    return out
