"""Tests for subsample_points (mask-form random count cap) and confidence_mask."""

import numpy as np

from collab_splats.pointcloud.utils import confidence_mask, subsample_points


def _sparse_mask(shape=(4, 16, 16), p=0.5):
    return np.random.default_rng(1).random(shape) < p


def test_subsample_points_caps_within_input_trues():
    mask = _sparse_mask()
    keep = subsample_points(mask, 100)
    assert keep.shape == mask.shape and keep.dtype == bool
    assert keep.sum() == 100
    assert not (keep & ~mask).any()


def test_subsample_points_noop_under_budget():
    mask = _sparse_mask()
    keep = subsample_points(mask, int(mask.sum()))
    assert keep is mask


def test_subsample_points_is_deterministic_and_leaves_global_rng():
    mask = np.ones((4, 8, 8), bool)
    state = np.random.get_state()
    a = subsample_points(mask, 50)
    b = subsample_points(mask, 50)
    assert a.sum() == 50 and np.array_equal(a, b)
    assert np.array_equal(np.random.get_state()[1], state[1])


def test_subsample_points_seed_changes_draw():
    mask = np.ones(1000, bool)
    assert not np.array_equal(
        subsample_points(mask, 100, seed=0), subsample_points(mask, 100, seed=1)
    )


def test_subsample_points_matches_seeded_choice():
    """
    Draw pinned to rng(seed).choice over the True positions, so feedforward parity holds.
    """
    mask = _sparse_mask()
    idx = np.flatnonzero(mask)
    expected = np.zeros(mask.size, bool)
    expected[np.random.default_rng(3).choice(idx, size=40, replace=False)] = True
    np.testing.assert_array_equal(
        subsample_points(mask, 40, seed=3), expected.reshape(mask.shape)
    )


def test_confidence_mask_global_percentile_strict():
    """Keep = strictly above the global cutoff, shape-agnostic."""
    conf = np.array([[0.0, 0.0], [1.0, 1.0]])  # p50 cutoff = 0.5
    keep = confidence_mask(conf, 50.0)
    np.testing.assert_array_equal(keep, [[False, False], [True, True]])


def test_confidence_mask_uniform_confidence_keeps_all(caplog):
    """Uniform conf: nothing is strictly above the cutoff → keep everything, never delete-all."""
    keep = confidence_mask(np.full((3, 4), 0.7), 50.0)
    assert keep.all() and keep.shape == (3, 4)
    assert "nothing above" in caplog.text


def test_confidence_mask_ties_at_the_max_keep_only_the_max():
    """p100, or a saturated max: the fallback keeps the max pixels, never the unfiltered map."""
    conf = np.array([0.1, 0.5, 0.9, 0.9])
    np.testing.assert_array_equal(
        confidence_mask(conf, 100.0), [False, False, True, True]
    )
    np.testing.assert_array_equal(
        confidence_mask(conf, 60.0), [False, False, True, True]
    )
