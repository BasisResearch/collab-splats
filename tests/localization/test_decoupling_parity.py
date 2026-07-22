"""Decoupling parity: (images, ids) index build == direct per-frame extraction."""

import numpy as np
import torch

from collab_splats.localization.extractors import DiskExtractor
from collab_splats.localization.localizer import CameraLocalizer


def test_index_features_match_direct_extraction():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(3)]
    ids = [f"frame_{i:06d}.jpg" for i in range(3)]
    extractor = DiskExtractor()

    # Reference: extract each frame directly (the pure operation).
    direct = [extractor.extract(f) for f in frames]

    # Under test: build the index via the decoupled (images, ids) path.
    loc = CameraLocalizer(
        world_points=np.zeros((3, 96, 128, 3), np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        images=iter(frames),
        ids=ids,
        extractor=extractor,
    )

    assert len(loc._frame_features) == 3
    for got, want in zip(loc._frame_features, direct):
        assert torch.equal(got.keypoints, want.keypoints)
        assert torch.equal(got.descriptors, want.descriptors)
    assert loc.image_paths == ids
