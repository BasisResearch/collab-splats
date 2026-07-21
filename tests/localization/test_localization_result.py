"""LocalizationResult.ranked_ref_frames: pure inlier-count ranking over result fields."""

import numpy as np

from collab_splats.localization.localizer import LocalizationResult


def _make(ref_frame_indices, inlier_mask):
    return LocalizationResult(
        pose=None,
        n_inliers=int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
        n_correspondences=len(ref_frame_indices) if ref_frame_indices is not None else 0,
        pts2d=None,
        pts3d_matched=None,
        pts2d_ref=None,
        inlier_mask=inlier_mask,
        ref_frame_indices=ref_frame_indices,
    )


def test_ranked_orders_by_inlier_count_desc():
    ref = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
    mask = np.array([True, True, True, True, True, True])
    res = _make(ref, mask)
    assert res.ranked_ref_frames == [2, 0, 1]


def test_ranked_excludes_zero_inlier_frames():
    ref = np.array([0, 1, 1], dtype=np.int32)
    mask = np.array([False, True, True])
    res = _make(ref, mask)
    assert res.ranked_ref_frames == [1]


def test_ranked_empty_when_no_result():
    assert _make(None, None).ranked_ref_frames == []
    ref = np.array([0, 1], dtype=np.int32)
    assert _make(ref, None).ranked_ref_frames == []


def test_ranked_empty_when_no_inliers():
    ref = np.array([0, 1, 2], dtype=np.int32)
    mask = np.array([False, False, False])
    assert _make(ref, mask).ranked_ref_frames == []


def test_ranked_breaks_ties_by_lowest_index():
    ref = np.array([2, 0, 1], dtype=np.int32)  # one inlier each
    mask = np.array([True, True, True])
    assert _make(ref, mask).ranked_ref_frames == [0, 1, 2]
