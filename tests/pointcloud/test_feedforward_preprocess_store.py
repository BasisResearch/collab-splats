"""BaseFeedforwardCreator reads inference frames straight from a scene's images/ files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
from collab_splats.preproc import frames as fr


@dataclass
class _EchoCreator(BaseFeedforwardCreator):
    """Minimal creator whose _preprocess decodes the files and echoes their dims."""

    def _load_model(self, device: str) -> Any:
        return object()

    def _preprocess(self, paths: list[Path]) -> Any:
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        coords = np.array(
            [[0, 0, f.shape[1], f.shape[0], f.shape[1], f.shape[0]] for f in frames],
            dtype=np.float32,
        )
        return frames, coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        return {}

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> PointcloudResult:
        raise NotImplementedError


# Gappy source indices: a quality filter drops frames, so row position and frame_idx
# diverge and a label taken from the wrong one is visible here.
_FRAME_IDXS = [0, 3, 7, 9]


def _make_images_dir(tmp_path) -> Path:
    """Write a scene images/ directory whose frame indices are non-contiguous."""
    frames = [np.full((32, 48, 3), idx, dtype=np.uint8) for idx in _FRAME_IDXS]
    images_dir = tmp_path / "images"
    fr.write_frames(images_dir, frames, _FRAME_IDXS)
    return images_dir


def test_setup_inference_labels_are_file_stems(tmp_path):
    creator = _EchoCreator()

    creator.setup_inference(fr.frame_paths(_make_images_dir(tmp_path)))

    # Labels are the filename stems — never the row position, which would misjoin poses to frames
    assert [p.name for p in creator.image_paths] == [
        f"frame_{i:06d}" for i in _FRAME_IDXS
    ]
    assert [f.shape for f in creator.views] == [(32, 48, 3)] * 4
    assert creator.original_coords.shape == (4, 6)


def test_setup_inference_reads_pixels_in_filename_order(tmp_path):
    """Frame N's pixels land at row N — a reorder would shift every pose against its frame."""
    creator = _EchoCreator()

    creator.setup_inference(fr.frame_paths(_make_images_dir(tmp_path)))

    # _make_images_dir paints each frame with its own source index
    assert [int(f[0, 0, 0]) for f in creator.views] == _FRAME_IDXS


def test_colmap_image_names_stable_extensionless(tmp_path):
    """Extension-less frame_{idx:06d} labels are valid, stable COLMAP image names."""
    names = [f"frame_{i:06d}" for i in (0, 1)]
    result = PointcloudResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=np.stack([np.eye(4, dtype=np.float32)] * 2),
        intrinsics=np.stack([np.eye(3, dtype=np.float32)] * 2),
        model_intrinsics=np.stack([np.eye(3, dtype=np.float32)] * 2),
        image_paths=[tmp_path / name for name in names],
        original_coords=np.array([[0, 0, 64, 64, 64, 64]] * 2, dtype=np.float32),
        model_width=64,
        model_height=64,
    )

    recon = result.to_colmap()

    got = sorted(recon.images[i].name for i in recon.images)
    assert got == names
