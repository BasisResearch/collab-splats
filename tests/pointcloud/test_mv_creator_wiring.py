"""
Multiview filter wiring in the feedforward _postprocess.
"""

from unittest.mock import patch

import numpy as np

from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from tests.pointcloud._stubs import vggt_raw_outputs


def _run(creator: VGGTXCreator, raw: dict | None = None) -> int:
    """
    Point count after _postprocess over the stub raw dict.
    """
    raw = vggt_raw_outputs(n=3, h=8, w=10) if raw is None else raw
    creator.image_paths = [f"frame_{i:06d}" for i in range(3)]
    creator.original_coords = np.tile(
        np.array([0, 0, 10, 8, 10, 8], np.float32), (3, 1)
    )
    return len(creator._postprocess(raw).points)


def test_min_views_zero_is_off():
    with patch(
        "collab_splats.pointcloud.feedforward.base.multiview_depth_confidence"
    ) as mock_mv:
        _run(VGGTXCreator(min_views=0))
    mock_mv.assert_not_called()


def test_min_views_filters_on_disagreement():
    assert _run(VGGTXCreator(min_views=1, mv_rel_thresh=1e-9)) < _run(
        VGGTXCreator(min_views=0)
    )


def _disjoint_views() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two frames whose cameras sit 1000 units apart, so neither sees the other's pixels.
    """
    depth = np.ones((2, 8, 10), np.float32)
    intrinsics = np.tile(
        np.array([[10, 0, 5], [0, 10, 4], [0, 0, 1]], np.float32), (2, 1, 1)
    )
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    extrinsics[1, 0, 3] = 1000.0
    return depth, intrinsics, extrinsics


def test_unseen_pixels_are_kept():
    depth, intrinsics, extrinsics = _disjoint_views()
    on = VGGTXCreator(min_views=2)._multiview_mask(depth, intrinsics, extrinsics)
    off = VGGTXCreator(min_views=0)._multiview_mask(depth, intrinsics, extrinsics)
    assert on.sum() == off.sum() == depth.size


def test_zero_depth_is_dropped_with_the_filter_off():
    raw = vggt_raw_outputs(n=3, h=8, w=10)
    raw["depth"][0, 0, 0] = 0.0
    assert _run(VGGTXCreator(min_views=0), raw) == 3 * 8 * 10 - 1
