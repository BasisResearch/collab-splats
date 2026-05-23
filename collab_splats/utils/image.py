# collab_splats/utils/image.py
"""Pure PIL image utilities — no torch dependency."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


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
