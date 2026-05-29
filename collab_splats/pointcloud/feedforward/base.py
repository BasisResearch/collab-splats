"""Shared types, helpers, and abstract pipeline for feedforward pointcloud creators.

Provides:
  FeedforwardResult                        — typed output dataclass for all feedforward backends
  _raw_to_world_points                     — unproject depth maps to subsampled world-space grids
  build_pycolmap_reconstruction            — build a pycolmap Reconstruction from pts+cameras
  _rescale_reconstruction_to_original_dims — rescale camera params from model resolution to original
  BaseFeedforwardCreator                   — abstract 5-step template-method pipeline
"""
from __future__ import annotations

import copy
import time
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import zarr
from rich.console import Console
from zarr.codecs import BloscCodec

from ..base import BasePointcloudCreator, CoordinateFrame, PointcloudResult
from ..utils import cross_frame_attention_ratio, reproject_pixels
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses

console = Console()


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class FeedforwardResult:
    """Typed output from a feedforward creator's ``_postprocess`` step.

    All arrays are float32 unless noted. Shapes assume N images and P output points.
    Optional fields ``images``, ``confidence``, ``world_points`` are always populated by
    feedforward creators and consumed by ``BundleAdjustment``.
    """

    points: np.ndarray           # (P, 3) float32 — world-space XYZ points
    colors: np.ndarray           # (P, 3) uint8 — RGB, range [0, 255]
    extrinsics: np.ndarray       # (N, 4, 4) float32 — world-to-camera homogeneous transform
                                 #   rows 0-2: [R|t], row 3: [0, 0, 0, 1]
    intrinsics: np.ndarray       # (N, 3, 3) float32 — camera intrinsics K
    image_paths: list[Path]      # length N — source image paths, ordered to match extrinsics
    original_coords: np.ndarray  # (N, 6) float32 — [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
                                 #   tl = model crop top-left in original pixels
                                 #   cr = model crop bottom-right in original pixels
                                 #   orig_w/h = full original image dimensions
    model_width: int             # model inference resolution width (pixels)
    model_height: int            # model inference resolution height (pixels)
    # Populated by feedforward creators; consumed by BundleAdjustment wrapper.
    images: "torch.Tensor | None" = None        # (N, 3, H, W) normalised RGB for track extraction
    confidence: "torch.Tensor | None" = None      # (N, H, W) confidence scores
    world_points: "np.ndarray | None" = None     # (N, H, W, 3) world-space points per pixel
    depth: "np.ndarray | None" = None            # (N, H, W) float32 depth maps (normalised to 3-D across backends)
    features: "np.ndarray | None" = None      # (P, D) float32 — feature vector per point, index-aligned with points
    pixel_indices: "np.ndarray | None" = None  # (P, 3) int32 — [frame_id, row, col] source pixel for each point
    _zarr_path: "Path | None" = field(default=None, init=False, repr=False, compare=False)

    def save(self, path: Path) -> None:
        """Save to compressed .npz. images/confidence/world_points excluded (too large)."""
        # Collect required arrays into a flat dict for np.savez_compressed
        arrays: dict = dict(
            points=self.points,
            colors=self.colors,
            extrinsics=self.extrinsics,
            intrinsics=self.intrinsics,
            original_coords=self.original_coords,
            image_paths=np.array([str(p) for p in self.image_paths]),
            model_width=np.array(self.model_width),
            model_height=np.array(self.model_height),
        )
        # Append optional arrays if present
        if self.features is not None:
            arrays["features"] = self.features
        if self.pixel_indices is not None:
            arrays["pixel_indices"] = self.pixel_indices
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "FeedforwardResult":
        """Load from .npz saved by save(). images/confidence/world_points will be None."""
        # Deserialize all saved keys; optional keys default to None if absent
        d = np.load(path, allow_pickle=False)
        return cls(
            points=d["points"],
            colors=d["colors"],
            extrinsics=d["extrinsics"],
            intrinsics=d["intrinsics"],
            original_coords=d["original_coords"],
            image_paths=[Path(str(p)) for p in d["image_paths"]],
            model_width=int(d["model_width"]),
            model_height=int(d["model_height"]),
            features=d["features"] if "features" in d else None,
            pixel_indices=d["pixel_indices"] if "pixel_indices" in d else None,
        )

    def save_zarr(self, path: Path) -> None:
        """Save to a zarr v3 store with lz4 compression.

        Unlike save(), this backend also persists world_points (chunked by frame),
        confidence (chunked by frame), and images (tensor → numpy, chunked by frame).
        images is excluded from load_zarr to avoid loading large tensors inadvertently.

        Args:
            path: Directory path for the zarr store (created if absent).
        """
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(path), mode="w")

        # Store scalar metadata and image paths in attrs
        store.attrs["image_paths"] = [str(p) for p in self.image_paths]
        store.attrs["model_width"] = self.model_width
        store.attrs["model_height"] = self.model_height

        # Save required arrays with lz4 compression
        for name, arr in (
            ("points", self.points),
            ("colors", self.colors),
            ("extrinsics", self.extrinsics),
            ("intrinsics", self.intrinsics),
            ("original_coords", self.original_coords),
        ):
            store.create_array(name, data=arr, chunks=arr.shape, compressors=lz4)

        # Save optional dense arrays
        if self.features is not None:
            store.create_array("features", data=self.features, chunks=self.features.shape, compressors=lz4)
        if self.pixel_indices is not None:
            store.create_array("pixel_indices", data=self.pixel_indices, chunks=self.pixel_indices.shape, compressors=lz4)

        # Save depth (N, H, W) chunked by frame
        if self.depth is not None:
            chunks = (1, self.depth.shape[1], self.depth.shape[2])
            store.create_array("depth", data=self.depth, chunks=chunks, compressors=lz4)

        # Save world_points chunked by frame: (1, H, W, 3)
        if self.world_points is not None:
            wp = self.world_points  # (N, H, W, 3)
            chunks = (1, wp.shape[1], wp.shape[2], wp.shape[3])
            store.create_array("world_points", data=wp, chunks=chunks, compressors=lz4)

        # Save confidence (N, H, W) chunked by frame; cast bfloat16 → float32 (zarr limitation).
        if self.confidence is not None:
            conf_np = self.confidence
            if isinstance(conf_np, torch.Tensor):
                if conf_np.dtype == torch.bfloat16:
                    conf_np = conf_np.to(torch.float32)
                conf_np = conf_np.detach().cpu().numpy()
            chunks = (1, conf_np.shape[1], conf_np.shape[2])
            store.create_array("confidence", data=conf_np, chunks=chunks, compressors=lz4)

        # Save images (tensor → numpy) chunked by frame: (1, 3, H, W)
        # Cast bfloat16 → float32 first; zarr/numpy do not support bfloat16.
        if self.images is not None:
            imgs = self.images
            if isinstance(imgs, torch.Tensor):
                if imgs.dtype == torch.bfloat16:
                    imgs = imgs.to(torch.float32)
                imgs = imgs.detach().cpu().numpy()
            chunks = (1, imgs.shape[1], imgs.shape[2], imgs.shape[3])
            store.create_array("images", data=imgs, chunks=chunks, compressors=lz4)

    @classmethod
    def load_zarr(cls, path: Path, load_images: bool = False) -> "FeedforwardResult":
        """Load from a zarr v3 store saved by save_zarr().

        images defaults to None (large tensor; skipped to avoid accidental loads).
        Pass load_images=True to restore the (N, 3, H, W) tensor — required for
        post-load feature lifting so the extractor sees the same FOV as the depth map.
        confidence is restored as a torch.Tensor if present in the store.

        Args:
            path: Directory path of the zarr store.
            load_images: If True and "images" is in the store, load and return as a torch.Tensor.

        Returns:
            FeedforwardResult with all persisted fields restored.
        """
        store = zarr.open(str(path), mode="r")
        attrs = dict(store.attrs)

        # Load required arrays
        pts3d = store["points"][:]
        colors = store["colors"][:]
        extrinsics = store["extrinsics"][:]
        intrinsics = store["intrinsics"][:]
        original_coords = store["original_coords"][:]
        image_paths = [Path(p) for p in attrs["image_paths"]]
        model_width = int(attrs["model_width"])
        model_height = int(attrs["model_height"])

        # Load optional arrays; absent keys → None
        features = store["features"][:] if "features" in store else None
        pixel_indices = store["pixel_indices"][:] if "pixel_indices" in store else None
        world_points = store["world_points"][:] if "world_points" in store else None
        depth = store["depth"][:] if "depth" in store else None
        _conf_key = "confidence" if "confidence" in store else ("conf" if "conf" in store else None)
        confidence = torch.from_numpy(store[_conf_key][:]) if _conf_key else None
        # Opt-in image load: skipped by default to avoid pulling large tensor into memory
        images = (
            torch.from_numpy(store["images"][:])
            if load_images and "images" in store
            else None
        )

        result = cls(
            points=pts3d,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            original_coords=original_coords,
            image_paths=image_paths,
            model_width=model_width,
            model_height=model_height,
            features=features,
            pixel_indices=pixel_indices,
            world_points=world_points,
            depth=depth,
            images=images,
            confidence=confidence,
        )
        result._zarr_path = Path(path)
        return result

    def reproject(self) -> "FeedforwardResult":
        """Re-project points under current extrinsics using stored source pixels and depth.

        ``intrinsics``, ``depth``, and ``pixel_indices`` all live in model-resolution
        space — no scaling required.
        """
        if self.depth is None or self.pixel_indices is None:
            raise ValueError(
                "reproject() requires depth and pixel_indices; load via load_zarr() "
                "or ensure the creator's _postprocess populated both fields."
            )
        # Reproject stored source pixels under current extrinsics — deterministic,
        # point set stays index-aligned with colors and features.
        pts3d = reproject_pixels(
            self.depth,
            self.pixel_indices,
            self.extrinsics[:, :3, :],
            self.intrinsics,
        )
        return replace(self, points=pts3d)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _raw_to_world_points(raw: dict, subsample: int = 8) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract world-space 3D points from raw _forward output dict.

    Builds a subsampled grid of world-space points from depth + intrinsics + extrinsics.
    Used by BundleAdjustment for track extraction — NOT the final point cloud (which uses
    full-resolution unprojection in each backend's _postprocess). subsample=8 balances
    coverage vs memory for long sequences.

    Args:
        raw:       Dict with keys 'depth', 'extrinsic', 'intrinsics_downsampled',
                   and optionally 'depth_conf'.
        subsample: Pixel stride for the output grid. Higher = fewer points.

    Returns:
        (all_pts, all_conf) — (K, P, 3) world-space points and (K, P) confidence,
        or (None, None) if required keys are absent.
    """
    # Guard: return None if required depth/pose/intrinsics keys are absent
    if not all(k in raw for k in ("depth", "extrinsic", "intrinsics_downsampled")):
        return None, None

    # Unpack depth, intrinsics, extrinsics, and optional confidence
    depth = raw["depth"]
    intr = raw["intrinsics_downsampled"]
    extr_3x4 = raw["extrinsic"]
    conf_map = raw.get("depth_conf")

    if depth.ndim == 4:
        depth = depth.squeeze(-1)  # VGGT-X returns (K, H, W, 1); MapAnything (K, H, W)
    K, H, W = depth.shape

    # Invert extrinsics (world2cam) to get cam2world transforms for unprojection
    extr_4x4 = extrinsics_to_homogeneous(extr_3x4)
    cam2world = invert_poses(extr_4x4.astype(np.float64)).astype(np.float32)

    # Build subsampled pixel grid (us × vs) for sparse world-point extraction
    us = np.arange(0, W, subsample)
    vs = np.arange(0, H, subsample)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    P = len(uu)

    # Pre-allocate output buffers
    all_pts = np.zeros((K, P, 3), dtype=np.float32)
    all_conf = np.zeros((K, P), dtype=np.float32) if conf_map is not None else None

    # Unproject each frame's sampled pixels to world space via depth + intrinsics + cam2world
    for ki in range(K):
        z = depth[ki][vv, uu]
        fx, fy = intr[ki, 0, 0], intr[ki, 1, 1]
        cx, cy = intr[ki, 0, 2], intr[ki, 1, 2]
        x_c = (uu - cx) * z / fx
        y_c = (vv - cy) * z / fy
        pts_cam = np.stack([x_c, y_c, z, np.ones_like(z)], axis=-1)
        all_pts[ki] = (cam2world[ki] @ pts_cam.T).T[:, :3]
        if conf_map is not None and all_conf is not None:
            all_conf[ki] = conf_map[ki][vv, uu]

    return all_pts, all_conf


def compute_multiview_depth_confidence(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_masks: Optional[np.ndarray] = None,
    abs_thresh: float = 0.0,
    rel_thresh: float = 0.05,
    device: str = "cuda",
) -> np.ndarray:
    """Geometric cross-view depth consistency confidence per pixel.

    For each source pixel, projects it into all other frames and checks whether
    the reprojected and sampled depths agree within abs_thresh + rel_thresh * depth.
    Returns per-pixel inlier ratio across overlapping views, in [0, 1].

    Args:
        depth:       (N, H, W) float32 Z-depth per frame.
        intrinsics:  (N, 3, 3) float32 pinhole intrinsics in pixel units.
        extrinsics:  (N, 4, 4) float32 world-to-cam transforms.
        depth_masks: (N, H, W) bool — source pixels to include; None = all valid depth.
        abs_thresh:  Absolute depth tolerance (depth units). 0.0 for non-metric depth.
        rel_thresh:  Relative depth tolerance as fraction of expected depth.
        device:      Torch device for computation.
    """
    dev = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    N, H, W = depth.shape

    depth_t = torch.from_numpy(depth.astype(np.float32)).to(dev)
    K = torch.from_numpy(intrinsics.astype(np.float32)).to(dev)
    E = torch.from_numpy(extrinsics.astype(np.float32)).to(dev)
    cam2world = torch.linalg.inv(E)

    # Build pixel grid [x, y, 1] for each pixel in (H, W)
    rows, cols = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=dev),
        torch.arange(W, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    pixel_h = torch.stack(
        [cols, rows, torch.ones(H, W, device=dev)], dim=-1
    ).reshape(-1, 3)  # (H*W, 3) [x, y, 1]

    inlier_sum = torch.zeros(N, H, W, dtype=torch.float32, device=dev)
    valid_sum  = torch.zeros(N, H, W, dtype=torch.float32, device=dev)

    # Pre-allocate homogeneous padding — reused across all (i, j) pairs
    ones_hw1 = torch.ones(H * W, 1, dtype=torch.float32, device=dev)

    # Pre-convert depth_masks to a GPU bool tensor to avoid per-iteration H2D copies
    if depth_masks is not None:
        depth_masks_t = torch.from_numpy(depth_masks.astype(bool)).to(dev)  # (N, H, W)
    else:
        depth_masks_t = None

    for i in range(N):
        # Unproject source pixels to world space via cam-i intrinsics and pose
        K_i_inv = torch.linalg.inv(K[i])
        cam_rays = (K_i_inv @ pixel_h.T).T                          # (H*W, 3)
        src_d = depth_t[i].reshape(-1, 1)                           # (H*W, 1)
        src_valid = (src_d > 0).squeeze(-1)                          # (H*W,)
        if depth_masks_t is not None:
            src_valid = src_valid & depth_masks_t[i].reshape(-1)

        pts_cam_i = cam_rays * src_d                                 # (H*W, 3)
        pts_cam_h = torch.cat(
            [pts_cam_i, ones_hw1], dim=-1
        )                                                             # (H*W, 4)
        pts_world = (cam2world[i] @ pts_cam_h.T).T[:, :3]           # (H*W, 3)

        for j in range(N):
            if i == j:
                continue

            # Project world points into frame j; compute expected depth and pixel coords
            pts_world_h = torch.cat(
                [pts_world, ones_hw1], dim=-1
            )
            pts_cam_j = (E[j] @ pts_world_h.T).T[:, :3]             # (H*W, 3)

            expected_d = pts_cam_j[:, 2]                             # (H*W,) Z in cam-j
            in_front = expected_d > 0

            proj_j = (K[j] @ pts_cam_j.T).T                         # (H*W, 3)
            z_j = proj_j[:, 2:3].clamp(min=1e-6)
            px_j = proj_j[:, :2] / z_j                              # (H*W, 2)

            # Normalise pixel coords to [-1, 1] for grid_sample
            px_norm = torch.stack(
                [px_j[:, 0] / (W - 1) * 2 - 1,
                 px_j[:, 1] / (H - 1) * 2 - 1],
                dim=-1,
            )                                                         # (H*W, 2)
            in_bounds = (
                (px_norm[:, 0] >= -1) & (px_norm[:, 0] <= 1)
                & (px_norm[:, 1] >= -1) & (px_norm[:, 1] <= 1)
            )
            valid_ij = src_valid & in_front & in_bounds              # (H*W,)

            # Sample frame-j depth at projected locations using bilinear interpolation
            # Reshape to (1, H, W, 2): px_norm is ordered as the flattened source meshgrid,
            # so grid[0, r, c, :] = the normalised target coord where source pixel (r, c) projects.
            grid = px_norm.reshape(1, H, W, 2)
            sampled_d = F.grid_sample(
                depth_t[j].unsqueeze(0).unsqueeze(0),               # (1, 1, H, W)
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze()                                               # (H, W)
            sampled_d_flat = sampled_d.reshape(-1)                   # (H*W,)

            # Count inliers: depth agreement within abs + rel tolerance
            tol = abs_thresh + rel_thresh * expected_d.abs()
            inlier = (
                (torch.abs(expected_d - sampled_d_flat) < tol)
                & valid_ij
                & (sampled_d_flat > 0)
            )

            inlier_sum[i] += inlier.reshape(H, W).float()
            valid_sum[i]  += valid_ij.reshape(H, W).float()

    # Pixels with no overlapping views → confidence = 0
    mv_conf = torch.where(
        valid_sum > 0,
        inlier_sum / valid_sum.clamp(min=1.0),
        torch.zeros_like(inlier_sum),
    )
    return mv_conf.cpu().numpy().astype(np.float32)


# ── COLMAP reconstruction builders ────────────────────────────────────────────

def build_pycolmap_reconstruction(
    pts3d: np.ndarray,
    colors: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: list[str],
    camera_model: str = "PINHOLE",
) -> pycolmap.Reconstruction:
    """Build a pycolmap Reconstruction from pointcloud + camera data.

    Creates one camera and one image per entry in ``image_names``.  Points are
    added as free 3D points with no ``Point2D`` track observations — feedforward
    methods do not produce feature matches, so there are no 2D-3D correspondences
    to record.  This means the reconstruction is valid for writing to disk and
    converting to ``transforms.json``, but cannot be used as input to COLMAP BA.

    Args:
        pts3d:        (P, 3) float32 or float64 world-space point positions.
        colors:       (P, 3) uint8 or float32 [0,1] RGB colors.
                      Float inputs are clipped and scaled to uint8 automatically.
        extrinsics:   (N, 3, 4) or (N, 4, 4) float32 world-to-camera matrices.
                      The first 3 rows are used; the 4th row is ignored.
        intrinsics:   (N, 3, 3) float32 camera intrinsics K per image.
        image_width:  Width in pixels of the model inference resolution.
        image_height: Height in pixels of the model inference resolution.
        image_names:  Length-N list of image filenames (basename only, no path).
        camera_model: pycolmap camera model string.
                      ``"PINHOLE"`` — params [fx, fy, cx, cy].
                      ``"SIMPLE_PINHOLE"`` — params [f, cx, cy], f = mean(fx, fy).

    Returns:
        pycolmap.Reconstruction with cameras, images, and 3D points.
        Call ``_rescale_reconstruction_to_original_dimensions`` before writing
        to disk if the model resolution differs from the original image size.
    """
    recon = pycolmap.Reconstruction()
    exts = extrinsics[:, :3, :] if extrinsics.shape[1] == 4 else extrinsics

    # Convert colors to uint8
    colors_u8 = (
        colors if colors.dtype == np.uint8
        else (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    )

    # Add points — feedforward has no 2D feature tracks, so Track() is empty
    for xyz, rgb in zip(pts3d, colors_u8):
        recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb)

    # Add one camera + image per frame
    for i, name in enumerate(image_names):
        camera_id = i + 1
        image_id = i + 1

        K = intrinsics[i]
        if camera_model == "PINHOLE":
            params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        else:  # SIMPLE_PINHOLE
            params = [(K[0, 0] + K[1, 1]) / 2.0, K[0, 2], K[1, 2]]

        camera = pycolmap.Camera(
            model=camera_model,
            width=image_width,
            height=image_height,
            params=params,
            camera_id=camera_id,
        )
        # add_camera_with_trivial_rig creates a matching rig entry required by
        # pycolmap >=4.0 before calling add_image_with_trivial_frame.
        recon.add_camera_with_trivial_rig(camera)

        R = exts[i, :3, :3].astype(np.float64)
        t = exts[i, :3, 3].astype(np.float64)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)

        image = pycolmap.Image(name=name, camera_id=camera_id, image_id=image_id)
        # add_image_with_trivial_frame creates image + frame and registers the
        # pose atomically — required by pycolmap >=4.0 (cam_from_world is now
        # read-only; it lives on the Frame, not the Image).
        recon.add_image_with_trivial_frame(image, cam_from_world)

    return recon


def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: list[Path],
    original_image_sizes: np.ndarray,
    image_size: tuple[int, int],
    shared_camera: bool = False,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> Any:
    """Rescale a reconstruction from model resolution to original image dimensions.

    Feedforward models run inference at a fixed resolution (e.g. 518px). This
    function maps camera intrinsics and image dimensions back to the original
    image size so the reconstruction is metrically consistent with the source data.

    Args:
        reconstruction:             pycolmap Reconstruction object.
        image_paths:                List of Path objects for the images.
        original_image_sizes:       (N, 6) array with format
                                    [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
        image_size:                 Model inference resolution as (width, height).
        shared_camera:              If True, use a single shared camera for all images.
        shift_point2d_to_original_res: If True, shift Point2D observations to original res.
        verbose:                    Print progress if True.
    """
    if verbose:
        original_width, original_height = original_image_sizes[0, -2:]
        console.log(
            f"Rescaling reconstruction from {image_size[0]}x{image_size[1]} "
            f"to original dimensions"
        )
        console.log(f"  Original image sizes (WxH): {int(original_width)}x{int(original_height)}")

    # Initialise per-loop shared-camera bookkeeping (used only when shared_camera=True)
    rescale_camera = True
    shared_intrinsics = None
    shared_width = None
    shared_height = None

    # Rescale intrinsics and image dimensions for each frame
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]

        pyimage.name = image_paths[pyimageid - 1].name

        pred_params = copy.deepcopy(pycamera.params)

        real_image_size = original_image_sizes[pyimageid - 1, -2:]
        # scale_x/scale_y: ratio of original image size to model inference size.
        # Multiplying camera params by these factors maps from model-resolution
        # pixel coordinates back to original-resolution pixel coordinates.
        scale_x = real_image_size[0] / image_size[0]
        scale_y = real_image_size[1] / image_size[1]

        if rescale_camera and (not shared_camera or shared_intrinsics is None):
            if pycamera.model.name == "SIMPLE_PINHOLE":
                pred_params[0] *= max(scale_x, scale_y)
            elif pycamera.model.name in ("PINHOLE", "OPENCV", "RADIAL", "OPENCV_FISHEYE"):
                pred_params[0] *= scale_x
                pred_params[1] *= scale_y

            pred_params[-2] *= scale_x
            pred_params[-1] *= scale_y

            if shared_camera:
                shared_intrinsics = pred_params
                shared_width = int(real_image_size[0])
                shared_height = int(real_image_size[1])

                pycamera.params = shared_intrinsics
                pycamera.width = shared_width
                pycamera.height = shared_height
            else:
                pycamera.params = pred_params
                pycamera.width = int(real_image_size[0])
                pycamera.height = int(real_image_size[1])

        # Propagate shared intrinsics to subsequent frames when shared_camera=True
        if shared_camera and shared_intrinsics is not None:
            pycamera.params = shared_intrinsics
            pycamera.width = shared_width
            pycamera.height = shared_height

        # Shift Point2D observations from model-resolution to original-resolution coords
        if shift_point2d_to_original_res:
            top_left = original_image_sizes[pyimageid - 1, :2]
            scale_x = real_image_size[0] / image_size[0]
            scale_y = real_image_size[1] / image_size[1]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * np.array([scale_x, scale_y])

    if verbose:
        console.log("Rescaled reconstruction to original dimensions")

    return reconstruction


# ── Abstract pipeline ─────────────────────────────────────────────────────────

@dataclass
class BaseFeedforwardCreator(BasePointcloudCreator):
    """Template Method pipeline for feedforward pointcloud creators.

    Runs a fixed 5-step pipeline: load_model → setup_inference → run_inference
    → postprocess → build_colmap.  Subclasses implement the four abstract methods
    below; the base class handles device detection, GPU memory cleanup, state
    storage, and COLMAP reconstruction (shared across all feedforward methods).

    Abstract methods — contract each subclass must satisfy:

    ``_load_model(device: str) -> Any``
        Load the model from a pretrained checkpoint, move to ``device``, set to
        eval mode, and return it.  Do not store GPU state outside the returned
        model object.

    ``_preprocess(image_dir: Path) -> tuple[Any, list[Path], np.ndarray]``
        Load and preprocess images from ``image_dir``.  Return:
          views           — model-specific input batch (tensor or list of dicts)
          image_paths     — ordered list of image Paths (length N)
          original_coords — (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]

    ``_forward(model, views, **kwargs) -> Any``
        Run the model forward pass under ``torch.no_grad()``.  Return raw outputs
        as a dict or any structure ``_postprocess`` expects.  The base class calls
        ``torch.cuda.empty_cache()`` immediately after this returns.

    ``_postprocess(raw_outputs, **kwargs) -> FeedforwardResult``
        Convert raw model outputs to a ``FeedforwardResult``.  This is where
        depth unprojection, confidence filtering, and optional postprocessing
        (e.g., global alignment) happen.

    ``_verify_loop_candidate(frame1, frame2, verify_match_ratio) -> tuple[bool, np.ndarray | None]``
        Re-run model on a 2-frame pair to verify a loop closure candidate.
        Return (accepted, fresh_poses_2x4x4) or (False, None) if rejected.

    Attributes:
        camera_model: pycolmap camera model string for COLMAP reconstruction.
                      Use ``"PINHOLE"`` (fx, fy, cx, cy) or ``"SIMPLE_PINHOLE"``
                      (f, cx, cy — single focal length).

    Inspection-only attrs set after run_inference() when LC enabled:
        _lc_submaps: list[Submap]        submaps built during LC inference
        _lc_loop_submaps: list[Submap]   verified loop-closure submaps (each 2 frames)
        _lc_overlap_frames: int          cfg.submap_overlap value used
        _lc_all_matches: list[LoopMatch] all post-NMS candidates; .accepted=True for accepted ones
    Consumer: collab_splats.pointcloud.loop_closure.eval.capture_pose_graph_loss
    These are not stable API; refactor cautiously.
    """

    # Threshold for cross_frame_attention_ratio LC verification gate.
    # 0.85 matches VGGT-SPARK calibration (tested at _lc_layer_index=20).
    default_verify_match_ratio: ClassVar[float] = 0.85

    # Global block index to tap for Q/K in _verify_loop_candidate.
    # VGGT-SPARK uses target_layer=20 (of 24 global blocks); -1 (last) gives
    # systematically lower scores (~0.66 vs ~1.02) due to different attention distribution.
    _lc_layer_index: ClassVar[int] = 20

    # Token offset passed to cross_frame_attention_ratio — skip special tokens
    # that precede patch tokens. VGGT has 5 (1 camera + 4 register). Models
    # without special tokens (e.g. MapAnything) should override to 0.
    _lc_token_offset: ClassVar[int] = 5

    def _lc_collate_outputs(self, raw: Any) -> Any:
        """Aggregate _forward outputs for use in _run_lc_loop.

        Default: no-op for models that return a flat dict (VGGT-X, VGGT-Omega).
        Override for models that return list[dict] (MapAnything).
        """
        return raw

    camera_model: str = "PINHOLE"
    max_points: int = 500_000

    model: Any = field(default=None, init=False, repr=False)
    views: Any = field(default=None, init=False, repr=False)
    image_paths: list[Path] | None = field(default=None, init=False, repr=False)
    original_coords: np.ndarray | None = field(default=None, init=False, repr=False)
    raw_outputs: Any = field(default=None, init=False, repr=False)
    outputs: FeedforwardResult | None = field(default=None, init=False, repr=False)

    # ── Pipeline orchestration ────────────────────────────────────────────────

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.load_model()
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        return self.build_colmap(output_dir)

    def run(self, image_dir: Path, device: str | None = None) -> FeedforwardResult:
        """Run the full inference pipeline and return outputs.

        Convenience wrapper for load_model → setup_inference → run_inference → postprocess.
        Use reconstruct() instead if you also need COLMAP output written to disk.
        """
        self.load_model(device=device)
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        return self.outputs

    def load_model(self, device: str | None = None) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        console.log(f"Loading model ({device})...")
        self.model = self._load_model(device)
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")

    def setup_inference(self, image_dir: Path) -> None:
        t0 = time.perf_counter()
        console.log("Preprocessing images...")
        self.views, self.image_paths, self.original_coords = self._preprocess(Path(image_dir))
        console.log(f"  → {len(self.image_paths)} images  done in {time.perf_counter() - t0:.1f}s")

    def run_inference(self, **kwargs: Any) -> None:
        t0 = time.perf_counter()
        console.log("Running inference...")
        self.raw_outputs = self._forward(self.model, self.views, **kwargs)
        # Clear GPU cache after forward pass to free memory before postprocessing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")

    def postprocess(self, **kwargs: Any) -> None:
        t0 = time.perf_counter()
        console.log("Postprocessing...")
        self.outputs = self._postprocess(self.raw_outputs, **kwargs)
        n_pts = len(self.outputs.points)
        console.log(f"  → {n_pts:,} pts  done in {time.perf_counter() - t0:.1f}s")

    def build_colmap(self, output_dir: Path) -> PointcloudResult:
        t0 = time.perf_counter()
        console.log("Building COLMAP reconstruction...")
        o = self.outputs
        # Build pycolmap Reconstruction from points + cameras at model resolution
        recon = build_pycolmap_reconstruction(
            o.points, o.colors, o.extrinsics, o.intrinsics,
            o.model_width, o.model_height,
            [p.name for p in o.image_paths],
            camera_model=self.camera_model,
        )
        # Rescale intrinsics and image dims back to original image resolution
        recon = _rescale_reconstruction_to_original_dimensions(
            recon, o.image_paths, o.original_coords,
            (o.model_width, o.model_height),
        )
        # Write binary COLMAP reconstruction to disk and export transforms.json
        sparse_dir = Path(output_dir) / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        recon.write_binary(str(sparse_dir))
        self._write_transforms(sparse_dir, Path(output_dir))
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=o.image_paths,
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _load_model(self, device: str) -> Any:
        ...

    @abstractmethod
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        ...

    @abstractmethod
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        ...

    @abstractmethod
    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Capture cross-frame transformer activations via a per-call forward hook.

        Register a forward hook on the QKV projection of the cross-frame attention
        block at ``layer_index``, run the model on ``frames``, capture activations,
        then remove the hook.  The hook is removed in a finally block — guaranteed
        even if the forward raises.

        Args:
            frames:      (N, C, H, W) preprocessed frames on the correct device.
            layer_index: Cross-frame block index.  -1 = last block (default).
                         VGGTx indexes into aggregator.global_blocks;
                         MapAnything into info_sharing.self_attention_blocks.
            **kwargs:    Backend-specific forward kwargs.
                         MapAnything: minibatch_size (int),
                                      memory_efficient_inference (bool).
                         VGGTx: unused.

        Returns:
            dict with at minimum:
              "q":  (B, heads, N_tokens, head_dim) — query projections
              "k":  (B, heads, N_tokens, head_dim) — key projections
            VGGTx additionally includes:
              "poses": (2, 4, 4) float32 np.ndarray — pre-decoded camera extrinsics.
            MapAnything omits "poses"; _verify_loop_candidate returns None for poses.
        """
        ...

    def _verify_loop_candidate(
        self,
        frame1: Any,
        frame2: Any,
        verify_match_ratio: float = 0.85,
        layer_index: int = -1,
        **kwargs: Any,
    ) -> tuple[bool, Any]:
        """Verify a loop closure candidate via cross-frame attention gate.

        Args:
            frame1, frame2:      Preprocessed frames (C, H, W).
            verify_match_ratio:  Accept threshold (default 0.85, matches VGGT-SPARK).
            layer_index:         Transformer block to tap (default -1 = last).
            **kwargs:            Forwarded to extract_intermediate_features (e.g.
                                 minibatch_size=2 for MapAnything).

        Returns:
            (accepted, poses_or_None).  poses is (2, 4, 4) float32 np.ndarray when
            the backend includes a "poses" key; None otherwise — caller uses submap poses.
        """
        # Use model-calibrated layer; override layer_index arg if provided explicitly.
        effective_layer = self._lc_layer_index if layer_index == -1 else layer_index
        # Run the model once to capture cross-frame activations
        features = self.extract_intermediate_features(
            torch.stack([frame1, frame2]), layer_index=effective_layer, **kwargs
        )
        # Compute the cross-frame attention ratio gate using model's token offset
        ratio = cross_frame_attention_ratio(
            features["k"], features["q"], token_offset=self._lc_token_offset
        )
        if ratio < verify_match_ratio:
            return False, None
        # "poses" is optional — VGGTx includes it (pre-decoded), MapAnything does not
        return True, features.get("poses")

    def reproject(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Re-extract pts3d/colors via full depth unprojection under refined poses.

        Uses self.raw_outputs from the last run() or run_inference() call.
        Prefer result.reproject() when depth and pixel_indices are populated —
        no creator state needed, deterministic point set.
        """
        pts3d, colors = self._reproject(
            self.raw_outputs, result.extrinsics[:, :3, :], result.intrinsics
        )
        return replace(result, points=pts3d, colors=colors)

    @abstractmethod
    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space point cloud using refined camera poses.

        Called by reproject() after BundleAdjustment refines extrinsics.
        Each backend re-projects its raw depth/point data under the new poses.

        Args:
            raw_outputs:    Raw model outputs stored from _forward().
            extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices.
            intrinsics:     (N, 3, 3) refined camera intrinsics.

        Returns:
            (pts3d, colors) — (P, 3) float32 and (P, 3) uint8.
        """
        ...
