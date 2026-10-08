# collab_splats/utils/image.py
"""
Image helpers: PIL coercion, guided depth upsampling, hole filling.

- open_image, resize_image: PIL coercion and aspect-preserving resize
- upsample_depths: model-res depth onto the original-res RGB grid (kornia guided filter)
- fill_missing_pixels: push-pull fill of unknown pixels (texture atlas, hull ground heights)
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from kornia.filters import guided_blur
from PIL import Image

from collab_splats.utils.torch_utils import get_device

########################################################
########## Normalization constants #####################
########################################################

# ImageNet stats — DINOv2/DINOv3 training preprocessing, and DINO-SALAD retrieval
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP stats — from maskclip_onnx/clip.py _transform()
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def open_image(image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    """Coerce various image representations to a PIL Image.

    Args:
        image: File path (str or Path), numpy array, or PIL Image.

    Returns:
        PIL Image instance.

    Raises:
        ValueError: If *image* is an unsupported type.
    """
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def resize_image(image: Image.Image, longest_edge: int) -> Image.Image:
    """Resize maintaining aspect ratio so the longest edge equals *longest_edge*.

    Args:
        image: PIL Image to resize.
        longest_edge: Target pixel length for the longest dimension.

    Returns:
        Resized PIL Image.
    """
    width, height = image.size
    ratio = longest_edge / max(width, height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height), Image.BILINEAR)


########################################################
########## Depth upsampling ############################
########################################################


def _guided_upsample_depth(
    depth: np.ndarray,
    rgb_full: np.ndarray,
    crop_box: np.ndarray,
    device: torch.device,
    radius: int | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Upsample one model-res depth map into its crop region of the original-res RGB canvas.

    - crop_box (tl_x, tl_y, cr_x, cr_y) in original pixels; radius None = ~2 × the upsample factor
    - masked pixels stay 0, canvas outside the crop is 0
    - kornia guided_blur on device: He et al., gray guide, reflect-101 border
    """
    H, W = rgb_full.shape[:2]
    tl_x, tl_y, cr_x, cr_y = (int(round(v)) for v in crop_box)
    cw, ch = cr_x - tl_x, cr_y - tl_y
    if cw <= 0 or ch <= 0:
        raise ValueError(
            f"Degenerate crop box {crop_box} — original_coords are corrupt"
        )
    if tl_x < 0 or tl_y < 0 or cr_x > W or cr_y > H:
        raise ValueError(f"Crop box {crop_box} lies outside the {H}x{W} canvas")

    # Nearest resize of depth and validity to crop size — blocky but never invents values
    depth_nn = cv2.resize(depth, (cw, ch), interpolation=cv2.INTER_NEAREST)
    valid_nn = (depth_nn > 0).astype(np.float32)

    # Gray guide in [0, 1] from the original-res crop; radius spans ~2x the upsample factor
    guide = (
        cv2.cvtColor(rgb_full[tl_y:cr_y, tl_x:cr_x], cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        / 255.0
    )

    # Zero-mean guide: offset-invariant filter, less float32 cancellation in a and b
    guide = guide - 0.5

    if radius is None:
        radius = max(1, int(np.ceil(2 * cw / depth.shape[1])))

    # Validity-weighted filtering: depth and validity as two channels of one guided filter
    guide_t = torch.as_tensor(guide, device=device)[None, None]
    src_t = torch.as_tensor(np.stack([depth_nn * valid_nn, valid_nn]), device=device)[
        None
    ]
    num, den = guided_blur(guide_t, src_t, 2 * radius + 1, eps)[0]
    filtered = torch.where(den > 1e-6, num / den.clamp(min=1e-6), 0.0)

    # The guide must never resurrect deleted depth, and depth must stay non-negative
    filtered[src_t[0, 1] == 0] = 0.0
    filtered = filtered.clamp(min=0.0).cpu().numpy()

    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[tl_y:cr_y, tl_x:cr_x] = filtered
    return canvas


def upsample_depths(
    depths: np.ndarray, rgbs: np.ndarray, crop_boxes: np.ndarray
) -> np.ndarray:
    """
    Guided-filter upsample model-res depth maps onto their original-res RGB frames.

    - filters on get_device(), one frame at a time

    Args:
        depths: (N, h, w) model-res depth, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 original-res frames; H, W set the output size.
        crop_boxes: (N, 4) [tl_x, tl_y, cr_x, cr_y] model crops in original pixels
            (original_coords[:, :4]).

    Returns:
        (N, H, W) float32 depth at frame resolution.
    """
    depths = np.asarray(depths)
    rgbs = np.asarray(rgbs)
    crop_boxes = np.asarray(crop_boxes)
    if not (len(depths) == len(rgbs) == len(crop_boxes)):
        raise ValueError(
            f"{len(depths)} depths, {len(rgbs)} rgbs, {len(crop_boxes)} crop boxes"
        )

    # One guided upsample per frame into a preallocated stack
    device = torch.device(get_device())
    n, H, W = len(depths), rgbs.shape[1], rgbs.shape[2]
    out = np.zeros((n, H, W), dtype=np.float32)

    for i in range(n):
        out[i] = _guided_upsample_depth(
            np.asarray(depths[i], dtype=np.float32), rgbs[i], crop_boxes[i], device
        )

    return out


########################################################
########## Hole filling ################################
########################################################


def fill_missing_pixels(image: np.ndarray, known: np.ndarray) -> np.ndarray:
    """
    Push-pull pyramid fill: every unknown pixel takes the smooth continuation of the known ones.

    - fills any distance from a known pixel; the coarsest pyramid level is never empty
    - known pixels come back unchanged
    - with nothing known the result is zero

    Args:
        image: (H, W) or (H, W, C) values; unknown pixels are ignored.
        known: (H, W) bool, True where image holds a real value.

    Returns:
        float32 array shaped like image.
    """

    # Weight broadcast over channels when the image has them
    def expand(weight: np.ndarray) -> np.ndarray:
        return weight[..., None] if image.ndim == 3 else weight

    # Pull: repeatedly halve value-sum and weight-sum
    values = [np.ascontiguousarray(image * expand(known), dtype=np.float32)]
    weights = [known.astype(np.float32)]

    while min(values[-1].shape[:2]) > 1:
        values.append(cv2.pyrDown(values[-1]))
        weights.append(cv2.pyrDown(weights[-1]))

    # Push: normalize each level and let the coarser result show through wherever weight is missing
    out = values[-1] / np.maximum(expand(weights[-1]), 1e-8)

    for level in range(len(values) - 2, -1, -1):
        h, w = values[level].shape[:2]
        up = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        weight = expand(np.clip(weights[level], 0.0, 1.0))
        out = values[level] / np.maximum(weight, 1e-8) * weight + up * (1.0 - weight)

    return out.astype(np.float32)
