"""Shared fixtures for pointcloud tests.

Also ensures the project-root evals/ package is importable as `evals.datasets`,
not shadowed by tests/evals/ (which pytest adds to sys.path as a package).
The eviction hack is deliberately duplicated per consuming subtree rather than
hoisted into tests/conftest.py: an autouse eviction of `evals` modules at root
scope would also run inside tests/evals/ itself and break that suite's
self-imports (mirrored in tests/geometry/conftest.py).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

_PROJECT_ROOT = str(Path(__file__).parents[2])


@pytest.fixture(autouse=True)
def _fix_evals_import():
    """Evict tests/evals from sys.modules so project-root evals/ is used."""
    # Insert project root before tests/ directory
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    elif sys.path[0] != _PROJECT_ROOT:
        sys.path.remove(_PROJECT_ROOT)
        sys.path.insert(0, _PROJECT_ROOT)

    # Evict any cached evals module that points to tests/evals/
    for mod in list(sys.modules):
        if mod == "evals" or mod.startswith("evals."):
            m = sys.modules[mod]
            f = getattr(m, "__file__", "") or ""
            if "/tests/evals" in f or (not f and "/collab-splats/evals" not in f):
                del sys.modules[mod]

    yield


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
        patch("collab_splats.pointcloud.feedforward.vggtx.Image", image),
        patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images"),
    ):
        _, coords = VGGTXCreator()._preprocess(paths)
    return coords


def _omega_boxes(sizes: list[tuple[int, int]]) -> np.ndarray:
    """(N, 6) boxes VGGTOmegaCreator._preprocess returns for frames of these (w, h) sizes."""
    paths, image = _sized_paths(sizes)
    with (
        patch("collab_splats.pointcloud.feedforward.vggt_omega.Image", image),
        patch("collab_splats.pointcloud.feedforward.vggt_omega.load_and_preprocess_images"),
    ):
        _, coords = VGGTOmegaCreator()._preprocess(paths)
    return coords
