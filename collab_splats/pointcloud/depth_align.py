# collab_splats/pointcloud/depth_align.py
"""
Build a FeedforwardResult from an InstantSfM COLMAP model + VDA depth, at the COLMAP world scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from vggt.utils.geometry import unproject_depth_map_to_point_map

from .feedforward.base import FeedforwardResult

logger = logging.getLogger(__name__)


########################################################
########## Track-observation correspondences ###########
########################################################


def _tracked_point3d_ids(recon: pycolmap.Reconstruction) -> list[int]:
    """
    Sorted point3D ids that carry at least one observation.

    - InstantSfM exports sub-min-track-length points with EMPTY tracks (the writer consistency
      patch drops their unverifiable observations); no observation means no pixel provenance,
      so the result tail excludes them.
    """
    return sorted(pid for pid, p in recon.points3D.items() if len(p.track.elements) > 0)


def _pixel_indices_from_reconstruction(
    recon: pycolmap.Reconstruction,
    point3d_ids: list[int],
    name_to_row: dict[str, int],
    scale_x: float,
    scale_y: float,
    depth_hw: tuple[int, int],
) -> np.ndarray:
    """
    Synthesize (P, 3) int32 [frame_row, row, col] pixel indices from COLMAP tracks.

    - First track observation per point3D; keypoint xy is original-res, scaled to the depth
      grid and clamped in-bounds.
    - lift_features requires pixel_indices; SfM results have no dense source pixel, so the
      observing keypoint is the honest substitute.
    """
    h, w = depth_hw
    out = np.zeros((len(point3d_ids), 3), dtype=np.int32)

    # One observation per point: the first track element's keypoint, scaled + clamped
    for i, pid in enumerate(point3d_ids):
        elem = recon.points3D[pid].track.elements[0]
        image = recon.images[elem.image_id]
        xy = image.points2D[elem.point2D_idx].xy
        col = min(max(int(xy[0] * scale_x), 0), w - 1)
        row = min(max(int(xy[1] * scale_y), 0), h - 1)
        out[i] = (name_to_row[image.name], row, col)

    return out


def _depth_correspondences(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Per-frame (d_colmap, d_vda) pairs from track observations, both positive and in bounds.

    - Each points2D carrying a point3D gives an exact pixel plus that point's z in the camera
      frame; the pixel is rescaled from native camera resolution to the depth grid and
      nearest-sampled into VDA depth.
    - Returns one (d_colmap, d_vda) tuple per row of `depth`, in `image_names` order; a frame
      with no usable observation gets a pair of empty arrays.
    """
    # Row order is the caller's: one name per depth row, and every name registered
    if len(image_names) != depth.shape[0]:
        raise ValueError(f"{len(image_names)} image names for {depth.shape[0]} depth maps — rows would misalign")
    name_to_image = {image.name: image for image in reconstruction.images.values()}
    missing = [name for name in image_names if name not in name_to_image]
    if missing:
        raise ValueError(f"{len(missing)} image names not in reconstruction (first: {missing[0]})")

    _n_frames, grid_h, grid_w = depth.shape
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for row, name in enumerate(image_names):
        image = name_to_image[name]
        camera = reconstruction.cameras[image.camera_id]

        # Track observations: exact 2D pixel + the observed point's depth in this view
        observations = [p for p in image.points2D if p.has_point3D()]
        if not observations:
            pairs.append((np.zeros(0), np.zeros(0)))
            continue
        xyz = np.stack([reconstruction.points3D[p.point3D_id].xyz for p in observations])
        cam_from_world = image.cam_from_world().matrix()
        d_colmap = (xyz @ cam_from_world[:3, :3].T + cam_from_world[:3, 3])[:, 2]

        # Rescale native pixels to the depth grid (the localization ref_px bug class —
        # native-res keypoints indexed into a model-res grid), then nearest-sample
        xy = np.stack([p.xy for p in observations])
        u = np.rint(xy[:, 0] * (grid_w / camera.width)).astype(np.int64)
        v = np.rint(xy[:, 1] * (grid_h / camera.height)).astype(np.int64)
        in_bounds = (u >= 0) & (u < grid_w) & (v >= 0) & (v < grid_h)
        d_vda = np.zeros(len(observations))
        d_vda[in_bounds] = depth[row, v[in_bounds], u[in_bounds]]

        # Keep pairs with positive depth on both sides
        valid = in_bounds & (d_vda > 0) & (d_colmap > 0)
        pairs.append((d_colmap[valid], d_vda[valid]))

    return pairs


def _fit_depth_scales(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
    *,
    min_obs: int,
) -> tuple[np.ndarray, dict]:
    """
    Per-frame scale factors aligning VDA depth to the reconstruction's world scale.

    - s_i = median(d_colmap / d_vda) per frame; frames with fewer than `min_obs` valid pairs
      inherit the global median of the fitted scales; zero fitted frames raises.
    - Returns (scales, stats): (N,) float64 depth multipliers, and a stats dict with the global
      scale, fallback frames, and the pooled ratio spread before/after alignment (the
      after-spread is the unit-level success check).
    """
    n_frames = depth.shape[0]
    scales = np.full(n_frames, np.nan)
    pooled_ratios: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []

    # One robust scale per frame from its track observations; below the obs floor, don't fit
    for row, (d_colmap, d_vda) in enumerate(_depth_correspondences(reconstruction, image_names, depth)):
        if len(d_colmap) == 0:
            continue
        ratios = d_colmap / d_vda
        pooled_ratios.append(ratios)
        pooled_rows.append(np.full(len(ratios), row))
        if len(ratios) >= min_obs:
            scales[row] = np.median(ratios)

    fitted = ~np.isnan(scales)
    if not fitted.any():
        raise ValueError(
            f"depth alignment: no frame has >= {min_obs} valid track observations — "
            "the reconstruction is too sparse to align VDA depth to the COLMAP world."
        )

    # Thin frames inherit the scene answer (below the obs floor: don't fit, inherit)
    global_scale = float(np.median(scales[fitted]))
    fallback_frames = [image_names[i] for i in np.flatnonzero(~fitted)]
    if fallback_frames:
        logger.warning(
            "depth alignment: %d frames under %d obs (first: %s) — using global scale",
            len(fallback_frames),
            min_obs,
            fallback_frames[0],
        )
    scales[~fitted] = global_scale

    # Pooled spread: before = one global scale for all frames, after = per-frame scales
    ratios_all = np.concatenate(pooled_ratios)
    rows_all = np.concatenate(pooled_rows).astype(np.int64)
    stats = {
        "global_scale": global_scale,
        "n_fallback": len(fallback_frames),
        "fallback_frames": fallback_frames,
        "ratio_p10_p50_p90_before": [float(x) for x in np.percentile(ratios_all / global_scale, [10, 50, 90])],
        "ratio_p10_p50_p90_after": [float(x) for x in np.percentile(ratios_all / scales[rows_all], [10, 50, 90])],
    }
    return scales, stats


########################################################
########## COLMAP model -> FeedforwardResult ###########
########################################################


def result_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    depths: np.ndarray,
    images: np.ndarray,
    names: list[str],
    *,
    min_obs: int = 20,
) -> tuple[FeedforwardResult, dict]:
    """
    Build a COLMAP-scale FeedforwardResult from an InstantSfM model + VDA depth maps.

    - the depth grid is the result's model resolution: K, images and pixel_indices are scaled to
      it (pairing original-res K with model-res depth is the 2026-08-11 mesh-regression class)
    - depth is rescaled to the COLMAP world before anything is derived from it, so the zarr and
      the model share one scale (splat depth targets, mesh fusion, localization lookup)
    - confidence / mv_* stay absent — SfM has no learned per-pixel confidence

    Args:
        reconstruction: Registered InstantSfM model; its image stems must equal `names`' stems.
        depths:         (N, h, w) VDA metric depth.
        images:         (N, H, W, 3) uint8 RGB at keyframe resolution.
        names:          Keyframe filenames (frame_NNNNNN.png), in registration order.
        min_obs:        Minimum valid track-depth pairs a frame needs to get its own scale.

    Returns:
        (result, attrs) — attrs is the alignment provenance for save_zarr.
    """
    stems = [Path(n).stem for n in names]

    # Every requested frame must be registered — a partial model leaves rows without poses
    if len(reconstruction.images) != len(stems):
        raise RuntimeError(
            f"InstantSfM registered {len(reconstruction.images)}/{len(stems)} frames — partial "
            "registration is not supported; re-run with more overlap"
        )

    # Two orderings must agree or the rows silently misalign
    # - depths/images rows follow `names`; model-derived arrays follow sorted image name
    # - the pipeline's frame_NNNNNN naming already guarantees it, so a mismatch is a real bug
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

    # `images` is indexed row-by-row against that count below: too few frames is an opaque
    # IndexError out of the resize, too many are silently dropped
    if len(images) != n:
        raise ValueError(f"{len(images)} frames for {n} depth maps — rows would misalign")

    # Poses: cam_from_world (w2c) as homogeneous 4x4
    extrinsics = np.stack(
        [np.vstack([im.cam_from_world().matrix(), [0.0, 0.0, 0.0, 1.0]]) for im in images_sorted]
    ).astype(np.float32)

    # COLMAP K is at keyframe (original) resolution, rescale it to the depth grid
    # - the COLMAP cameras must be at the frames' resolution
    # - otherwise the keyframe set / SIFT DB came from a different store
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
    sx, sy = w / orig_w, h / orig_h
    intrinsics = np.stack([reconstruction.cameras[im.camera_id].calibration_matrix() for im in images_sorted])
    intrinsics = intrinsics.astype(np.float32)
    intrinsics[:, 0, :] *= sx
    intrinsics[:, 1, :] *= sy

    # Align VDA depth to the COLMAP world FIRST — world_points below must come from the aligned
    # depth (t is not scale-invariant, so scaled world points would be wrong).
    scales, stats = _fit_depth_scales(reconstruction, registered, depths, min_obs=min_obs)
    logger.info(
        "depth alignment: global scale %.4f, ratio p10/p50/p90 %s -> %s, %d fallback frames",
        stats["global_scale"],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_before"]],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_after"]],
        stats["n_fallback"],
    )
    depths = (depths * scales[:, None, None]).astype(np.float32)

    # Sparse points in point3D-id order; pixel_indices from each point's first observation.
    # Observation-less points (InstantSfM's sub-min-track-length exports) are dropped.
    point3d_ids = _tracked_point3d_ids(reconstruction)
    points = np.array([reconstruction.points3D[pid].xyz for pid in point3d_ids], dtype=np.float32).reshape(-1, 3)
    colors = np.array([reconstruction.points3D[pid].color for pid in point3d_ids], dtype=np.uint8).reshape(-1, 3)
    pixel_indices = _pixel_indices_from_reconstruction(
        reconstruction, point3d_ids, name_to_row, scale_x=sx, scale_y=sy, depth_hw=(h, w)
    )

    # RGB at depth res as (N, 3, H, W) float32 in [0, 1] — the feedforward images convention
    images_arr = np.stack([cv2.resize(images[i], (w, h), interpolation=cv2.INTER_AREA) for i in range(n)])
    images_arr = images_arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    # Dense world points by unprojecting the ALIGNED depth through the rescaled K and w2c poses
    world_points = unproject_depth_map_to_point_map(depths[..., None], extrinsics[:, :3, :], intrinsics).astype(
        np.float32
    )

    # No crop: the depth grid is a full-frame resize, so the crop box is the whole original frame
    # in ORIGINAL pixels — [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], the loger convention.
    original_coords = np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]] * n, dtype=np.float32)

    result = FeedforwardResult(
        points=points,
        colors=colors,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[Path(im.name) for im in images_sorted],
        original_coords=original_coords,
        model_width=w,
        model_height=h,
        images=images_arr,  # numpy float32 on purpose — save_zarr accepts it; no torch tensor needed
        world_points=world_points,
        depth=depths,
        pixel_indices=pixel_indices,
    )
    attrs = {
        "depth_scale": "colmap",
        "depth_scales": [float(s) for s in scales],
        "depth_scale_fallback_frames": stats["fallback_frames"],
    }
    return result, attrs
