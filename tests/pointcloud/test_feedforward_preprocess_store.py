"""BaseFeedforwardCreator decodes inference frames from a FrameStore in memory."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    build_pycolmap_reconstruction,
)
from collab_splats.pointcloud.feedforward.base import _decode_dir_to_frames
from collab_splats.preproc.frame_store import FrameStore


@dataclass
class _EchoCreator(BaseFeedforwardCreator):
    """Minimal creator whose _preprocess echoes frame_idx labels + frame dims."""

    def _load_model(self, device: str) -> Any:
        return object()

    def _preprocess(self, frames, frame_idxs):
        image_paths = [Path(f"frame_{idx:06d}") for idx in frame_idxs]
        coords = np.array([[0, 0, f.shape[1], f.shape[0], f.shape[1], f.shape[0]] for f in frames], dtype=np.float32)
        return frames, image_paths, coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        return {}

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        raise NotImplementedError

    def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
        return {}

    def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)


def _make_store(tmp_path) -> FrameStore:
    frames = [np.full((32, 48, 3), i, dtype=np.uint8) for i in range(4)]
    records = [{"frame_idx": i, "blur_score": 1.0} for i in range(4)]
    FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance={"video_path": "v"})
    return FrameStore.open(tmp_path / "frames.zarr")


def test_setup_inference_accepts_frame_store(tmp_path):
    store = _make_store(tmp_path)
    creator = _EchoCreator()

    creator.setup_inference(store)

    # Labels are the store's source frame indices, formatted stably for COLMAP
    assert [p.name for p in creator.image_paths] == [f"frame_{i:06d}" for i in range(4)]
    # Frames are decoded from the store (no temp files) at their native dims
    assert creator.views.shape == (4, 32, 48, 3)
    assert creator.original_coords.shape == (4, 6)


def test_setup_inference_accepts_zarr_path(tmp_path):
    _make_store(tmp_path)
    creator = _EchoCreator()

    creator.setup_inference(tmp_path / "frames.zarr")

    assert [p.name for p in creator.image_paths] == [f"frame_{i:06d}" for i in range(4)]


def test_setup_inference_accepts_image_dir(tmp_path):
    """Legacy eval path: an image dir is PIL-decoded with integer sort-order labels."""
    from PIL import Image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        Image.fromarray(np.full((16, 24, 3), i, dtype=np.uint8)).save(img_dir / f"{i:04d}.png")
    creator = _EchoCreator()

    creator.setup_inference(img_dir)

    assert [p.name for p in creator.image_paths] == [f"frame_{i:06d}" for i in range(3)]
    assert creator.views[0].shape == (16, 24, 3)


def test_decode_dir_to_frames_matches_legacy_sorted_glob(tmp_path):
    """Frame order + content must match the legacy sorted-iterdir the old _preprocess used.

    A reorder here would silently shift eval frames (poses index-aligned to GT) → ATE drift.
    """
    from PIL import Image

    # Write out-of-order names with distinct pixel content, plus a non-image to be rejected
    names = ["frame-000002.png", "frame-000000.png", "frame-000001.png"]
    for i, name in enumerate(names):
        Image.fromarray(np.full((8, 8, 3), (i + 1) * 40, dtype=np.uint8)).save(tmp_path / name)
    (tmp_path / "notes.txt").write_text("ignore me")

    # Legacy order: sorted Path objects filtered by lowercase suffix (vggtx/omega _preprocess)
    legacy_paths = sorted(p for p in tmp_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    legacy_frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in legacy_paths]

    frames, labels = _decode_dir_to_frames(tmp_path)

    assert labels == list(range(len(legacy_paths)))
    assert len(frames) == len(legacy_frames)
    for got, want in zip(frames, legacy_frames):
        assert np.array_equal(got, want)  # same order → same pixel content


def test_colmap_image_names_stable_extensionless(tmp_path):
    """Extension-less frame_{idx:06d} labels are valid, stable COLMAP image names."""
    names = [f"frame_{i:06d}" for i in (0, 1)]
    pts = np.zeros((1, 3), dtype=np.float32)
    colors = np.zeros((1, 3), dtype=np.uint8)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)
    intrinsics = np.stack([np.eye(3, dtype=np.float32)] * 2)

    recon = build_pycolmap_reconstruction(
        pts, colors, extrinsics, intrinsics, image_width=64, image_height=64, image_names=names
    )

    got = sorted(recon.images[i].name for i in recon.images)
    assert got == names
