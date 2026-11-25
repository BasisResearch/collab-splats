"""
Helpers the splat trainer calls around its step loop.

- scene geometry: scale, point spacing, normalization
- coarse-to-fine: downscale schedule and per-view targets, cached per factor
- view schedule: shuffled epochs of view indices
"""

import random
from collections.abc import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

########################################
# Scene geometry
########################################


def compute_scene_scale(cam_to_world: Tensor, *, margin: float = 1.1) -> float:
    """
    Scene extent: the farthest camera from the camera centroid, times a margin.

    Args:
        cam_to_world: (N, 4, 4) camera-to-world poses.
        margin: multiplier on the spread (1.1 as in gsplat).

    Returns:
        Scene extent, in the units of `cam_to_world`.
    """
    positions = cam_to_world[:, :3, 3]
    centroid = positions.mean(0)
    spread = (positions - centroid).norm(dim=-1).max()
    return float(spread) * margin


def knn_spacing(points: np.ndarray, k: int) -> np.ndarray:
    """
    Per-point RMS distance to its k nearest neighbors (itself excluded).

    Args:
        points: (N, 3) positions.
        k: neighbors per point.

    Returns:
        (N,) spacing, in the units of `points`.
    """
    neighbor_dists, _ = (
        NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    )
    return np.sqrt((neighbor_dists[:, 1:] ** 2).mean(-1))


def scene_normalization(cam_to_world: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Center and scale that fit the cameras into a unit box, as in splatfacto.

    - center: mean camera position; scale: 1 / largest camera offset (L-inf)
    - no rotation applied

    Args:
        cam_to_world: (N, 4, 4) camera-to-world poses in world units.

    Returns:
        (center (3,) float32, scale float).
    """
    positions = cam_to_world[:, :3, 3]
    center = positions.mean(0)
    spread = float(np.abs(positions - center).max())

    if spread <= 0:
        raise ValueError("splats: cannot normalize a scene whose cameras coincide")

    return center.astype(np.float32), 1.0 / spread


def denormalize_cameras(cam_to_world: Tensor, center: np.ndarray, scale: float) -> None:
    """
    Undo `scene_normalization` on camera translations, in place.

    Args:
        cam_to_world: (N, 4, 4) normalized poses; rewritten in world units.
        center: (3,) center from `scene_normalization`.
        scale: scale from `scene_normalization`.

    Returns:
        None; `cam_to_world` is modified in place.
    """
    center_t = torch.as_tensor(center, dtype=torch.float32, device=cam_to_world.device)

    with torch.no_grad():
        cam_to_world[:, :3, 3] = cam_to_world[:, :3, 3] / scale + center_t


########################################
# Coarse-to-fine views
########################################


def downscale_factor(step: int, num_downscales: int, resolution_schedule: int) -> int:
    """
    Image downscale factor at a training step, halving toward full resolution.

    Args:
        step: current training step.
        num_downscales: halvings at step 0; 0 turns the schedule off.
        resolution_schedule: steps between halvings.

    Returns:
        Integer divisor; 1 once the schedule has run out.
    """
    if num_downscales <= 0:
        return 1

    return 2 ** max(0, num_downscales - step // resolution_schedule)


def downscale_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Image shrunk by an integer factor (bilinear); the input itself at factor 1.

    - does not touch intrinsics; use `rescale_intrinsics` for those

    Args:
        image: (H, W, 3) uint8 image.
        factor: integer divisor from `downscale_factor`.

    Returns:
        (H // factor, W // factor, 3) image.
    """
    if factor == 1:
        return image

    height, width = image.shape[:2]
    return cv2.resize(
        image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR
    )


def prepare_target(image: np.ndarray, depth: np.ndarray | None, device: str) -> dict:
    """
    One view's training targets as tensors on `device`.

    Args:
        image: (H, W, 3) uint8 image.
        depth: (h, w) depth target at any resolution, 0 = no target, or None.
        device: torch device string.

    Returns:
        {"rgb": (1, H, W, 3) in [0, 1], "depth": (1, H, W, 1) or None}.
    """
    rgb = torch.from_numpy(image).to(device).float()[None] / 255.0

    if depth is None:
        return {"rgb": rgb, "depth": None}

    height, width = image.shape[:2]
    depth_nchw = torch.from_numpy(depth).to(device)[None, None]

    # Nearest, not bilinear, so "no target" zeros don't blend
    depth_nchw = F.interpolate(depth_nchw, size=(height, width), mode="nearest")
    return {"rgb": rgb, "depth": depth_nchw.permute(0, 2, 3, 1)}


def cached_target(
    cache: dict[int, dict[int, dict]],
    images: np.ndarray,
    depth_targets: np.ndarray | None,
    view: int,
    factor: int,
    device: str,
) -> dict:
    """
    One view's training targets at a downscale factor, built once and kept on `device`.

    - same `downscale_image` + `prepare_target` as a per-step build, so identical tensors
    - holds one factor at a time: a new factor drops the previous factor's targets

    Args:
        cache: {factor: {view: target}}, owned by the caller and updated in place.
        images: (n_views, H, W, 3) uint8 frames.
        depth_targets: (n_views, h, w) depth at any resolution, 0 = no target, or None.
        view: row of `images`.
        factor: integer divisor from `downscale_factor`.
        device: torch device string.

    Returns:
        The `prepare_target` dict for this view at this factor.
    """
    # A new factor frees the previous factor's tensors
    if factor not in cache:
        cache.clear()
        cache[factor] = {}

    targets = cache[factor]

    # Build on first visit only
    if view not in targets:
        image = downscale_image(images[view], factor)
        depth = None if depth_targets is None else depth_targets[view]
        targets[view] = prepare_target(image, depth, device)

    return targets[view]


########################################
# View schedule
########################################


def view_order(n_views: int, *, seed: int = 42) -> Iterator[int]:
    """
    Endless stream of view indices: each epoch visits every view once, shuffled.

    - ported from nerfstudio @ 50e0e3c, full_images_datamanager.py:152-161 and :396-399
    - order reversed vs upstream, so not identical seed for seed

    Args:
        n_views: number of training views.
        seed: RNG seed.

    Yields:
        View indices in [0, n_views), forever.
    """
    rng = random.Random(seed)

    while True:
        order = list(range(n_views))
        rng.shuffle(order)
        yield from reversed(order)
