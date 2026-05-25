"""I/O utilities: image collection and path helpers."""
from __future__ import annotations

from pathlib import Path

_IMAGE_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")


def collect_image_paths(
    image_dir: Path,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> list[Path]:
    """Return sorted image paths from image_dir; raises ValueError if empty."""
    image_dir = Path(image_dir)
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(image_dir.glob(f"*.{ext}"))
    # Sort and deduplicate (glob order is not guaranteed)
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(
            f"No images found in {image_dir} (extensions: {extensions})"
        )
    return paths
