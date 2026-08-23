"""Absent-confidence seams: mesh fusion, feature lifting, splats depth targets tolerate
FeedforwardResult.confidence=None (SfM-derived backends, e.g. instantsfm, carry none)."""

import numpy as np
import torch

from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import lift_features


def _result_no_confidence():
    """Minimal FeedforwardResult with confidence=None: 2 frames, 8x8 model res."""
    n, h, w = 2, 8, 8
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32),
        model_width=w,
        model_height=h,
        images=np.zeros((n, 3, h, w), dtype=np.float32),
        depth=np.ones((n, h, w), dtype=np.float32),
    )


def test_tsdf_inputs_skip_masking_when_confidence_absent(caplog):
    """conf_percentile set + no confidence -> proceed unmasked with a log, not ValueError."""
    result = _result_no_confidence()
    with caplog.at_level("INFO"):
        out = _feedforward_to_tsdf_inputs(result, conf_percentile=20)
    assert out is not None
    depths, _, _, _ = out
    np.testing.assert_array_equal(depths, result.depth)  # unmasked
    assert any("no confidence" in r.message for r in caplog.records)


def test_lift_features_uniform_weights_when_confidence_absent():
    """lift_features must not assert on confidence and should fall back to uniform weights."""
    result = _result_no_confidence()
    result.pixel_indices = np.zeros((5, 3), dtype=np.int32)
    fmaps = [torch.zeros(4, 8, 8) for _ in range(2)]
    feats = lift_features(fmaps, result)
    assert feats.shape == (5, 4)
