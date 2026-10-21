"""Shared fixtures for pointcloud tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

########################################################################
########## Crop boxes through each backend's _preprocess ###############
########################################################################


def _frames(sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    """Black (h, w, 3) frames for these (w, h) sizes."""
    return [np.zeros((h, w, 3), dtype=np.uint8) for w, h in sizes]


def _frame_files(frames: list[np.ndarray], dir: Path) -> list[Path]:
    """Frames written as frame_NNNNNN.png under dir: the paths _preprocess reads."""
    dir.mkdir(parents=True, exist_ok=True)
    paths = [dir / f"frame_{i:06d}.png" for i in range(len(frames))]
    for frame, path in zip(frames, paths):
        Image.fromarray(np.ascontiguousarray(frame)).save(path)
    return paths


def _sized_paths(sizes: list[tuple[int, int]]) -> tuple[list[Path], MagicMock]:
    """
    Fake frame paths plus a PIL.Image stand-in whose open(path).size is that frame's (w, h).

    - boxes need only the frame sizes; thousands of real multi-megapixel PNGs would take minutes
    """
    paths = [Path(f"{w}x{h}_{i}.png") for i, (w, h) in enumerate(sizes)]
    size_of = dict(zip(paths, sizes))
    image = MagicMock()
    image.open.side_effect = lambda p: SimpleNamespace(size=size_of[Path(p)])
    return paths, image


def _vggtx_boxes(sizes: list[tuple[int, int]]) -> np.ndarray:
    """(N, 6) boxes VGGTXCreator._preprocess returns for frames of these (w, h) sizes."""
    paths, image = _sized_paths(sizes)
    with (
        patch("collab_splats.pointcloud.feedforward.base.Image", image),
        patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images"),
    ):
        _, coords = VGGTXCreator()._preprocess(paths)
    return coords


def _omega_boxes(sizes: list[tuple[int, int]]) -> np.ndarray:
    """(N, 6) boxes VGGTOmegaCreator._preprocess returns for frames of these (w, h) sizes."""
    paths, image = _sized_paths(sizes)
    with (
        patch("collab_splats.pointcloud.feedforward.base.Image", image),
        patch(
            "collab_splats.pointcloud.feedforward.vggt_omega.load_and_preprocess_images",
            side_effect=lambda chunk, **kw: torch.zeros(len(chunk), 3, 16, 16),
        ),
    ):
        _, coords = VGGTOmegaCreator()._preprocess(paths)
    return coords
