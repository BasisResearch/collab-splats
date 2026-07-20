"""BaseFeedforwardCreator reads inference frames from a FrameStore via temp export."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    FeedforwardResult,
)
from collab_splats.preproc.frame_store import FrameStore


@dataclass
class _PathEchoCreator(BaseFeedforwardCreator):
    """Minimal creator whose _preprocess just returns the sorted exported image paths."""

    def _load_model(self, device: str) -> Any:
        return object()

    def _preprocess(self, image_dir: Path):
        paths = sorted(Path(image_dir).glob("*.jpg"))
        return None, paths, None

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


def test_preprocess_from_store_exports_and_delegates(tmp_path):
    store = _make_store(tmp_path)
    creator = _PathEchoCreator()

    views, image_paths, coords = creator._preprocess_from_store(store)

    # Exported files are source-idx named and delegated to path-based _preprocess
    assert [p.name for p in image_paths] == [f"frame_{i:06d}.jpg" for i in range(4)]
    # Temp export is held on the instance so files survive the whole run
    assert creator._frame_export is not None
    assert all(p.exists() for p in image_paths)


def test_setup_inference_accepts_frame_store(tmp_path):
    store = _make_store(tmp_path)
    creator = _PathEchoCreator()

    creator.setup_inference(store)

    assert [p.name for p in creator.image_paths] == [f"frame_{i:06d}.jpg" for i in range(4)]


def test_setup_inference_accepts_zarr_path(tmp_path):
    _make_store(tmp_path)
    creator = _PathEchoCreator()

    creator.setup_inference(tmp_path / "frames.zarr")

    assert [p.name for p in creator.image_paths] == [f"frame_{i:06d}.jpg" for i in range(4)]
