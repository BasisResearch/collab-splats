"""Decoupling parity: (images, ids) index build == direct per-frame extraction."""

import numpy as np
import torch

from collab_splats.localization.extractors import LocalFeatures
from collab_splats.localization.localizer import CameraLocalizer


class _ContentExtractor:
    """Deterministic stub whose output DEPENDS on image content.

    Keypoints/descriptors are derived from pixel values, so the parity equality
    below fails if the index build feeds different frames (or a different order)
    than direct extraction — a content-independent stub would pass vacuously.
    """

    def extract(self, image):
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        # 8 keypoints whose coordinates hash local pixel intensity
        ys = (img[:, :, 0].mean(axis=1).argsort()[:8]).astype(np.float32)
        xs = (img[:, :, 1].mean(axis=0).argsort()[:8]).astype(np.float32)
        kpts = torch.from_numpy(np.stack([xs % w, ys % h], axis=1))
        # Descriptors summarise per-channel statistics around each keypoint row
        desc = torch.from_numpy(
            np.stack([img[int(y) % h, int(x) % w] / 255.0 for x, y in zip(xs, ys)]).astype(np.float32)
        )
        return LocalFeatures(keypoints=kpts, descriptors=desc)


def test_index_features_match_direct_extraction():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(3)]
    ids = [f"frame_{i:06d}.jpg" for i in range(3)]
    extractor = _ContentExtractor()

    # Reference: extract each frame directly (the pure operation).
    direct = [extractor.extract(f) for f in frames]
    # Sanity: the stub is content-sensitive — different frames give different features
    assert not torch.equal(direct[0].keypoints, direct[1].keypoints) or not torch.equal(
        direct[0].descriptors, direct[1].descriptors
    )

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
