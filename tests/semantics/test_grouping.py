from pathlib import Path

import numpy as np
import pytest

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.semantics.grouping import (
    _mask_point_sets,
    _merge_masks,
    lift_segments,
    nearest_labels,
)

########################################################
########## Synthetic scene #############################
########################################################

# Plane z = 5 seen by N cameras sliding along x; f = 40 on a 64x32 grid
N, H, W, F_PX, Z = 10, 32, 64, 40.0, 5.0
CAM_X = np.linspace(-0.5, 0.5, N)


def _scene() -> PointcloudResult:
    """
    Plane grid of points, x in [-2, 2], y in [-1, 1], with exact depth in every frame.
    """
    xs, ys = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-1, 1, 41))
    points = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, Z)], 1)

    # World-to-camera: identity rotation, camera at (cx, 0, 0)
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    extrinsics[:, 0, 3] = -CAM_X
    K = np.array([[F_PX, 0, W / 2], [0, F_PX, H / 2], [0, 0, 1]], dtype=np.float32)

    return PointcloudResult(
        points=points.astype(np.float32),
        colors=np.zeros((len(points), 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=None,
        model_intrinsics=np.tile(K, (N, 1, 1)),
        image_paths=[Path(f"frame_{i}.png") for i in range(N)],
        original_coords=np.tile(np.array([0, 0, W, H, W, H], np.float32), (N, 1)),
        model_width=W,
        model_height=H,
        depth=np.full((N, H, W), Z, dtype=np.float32),
    )


def _world_x(i: int) -> np.ndarray:
    """
    World x of every pixel column of frame i, broadcast to (H, W).
    """
    u = np.arange(W)

    return np.broadcast_to((u - W / 2) * Z / F_PX + CAM_X[i], (H, W))


def _masks(*ranges: tuple[float, float]):
    """
    frame_masks callable: one mask per (lo, hi) world-x range, in every frame.
    """

    def frame_masks(i: int) -> np.ndarray:
        x = _world_x(i)

        return np.stack([(x >= lo) & (x < hi) for lo, hi in ranges])

    return frame_masks


LEFT, RIGHT = (-9.0, 0.0), (0.0, 9.0)


def _two_class_features(result: PointcloudResult) -> np.ndarray:
    """
    e0 on the left half, e1 on the right half.
    """
    features = np.zeros((len(result.points), 2), dtype=np.float32)
    left = result.points[:, 0] < 0
    features[left, 0] = 1
    features[~left, 1] = 1

    return features


########################################################
########## lift_segments ###############################
########################################################


def test_same_object_merges_across_views():
    result = _scene()
    groups = lift_segments(_masks(LEFT, RIGHT), result)

    # Two groups, split at x = 0, each seen in every frame
    x = result.points[:, 0]
    assert len(groups.n_frames) == 2
    assert (groups.n_frames == N).all()
    assert len(np.unique(groups.labels[x < -0.2])) == 1
    assert len(np.unique(groups.labels[x > 0.2])) == 1
    assert groups.labels[x < -0.2][0] != groups.labels[x > 0.2][0]
    assert (groups.parent == -1).all()


def test_contained_group_gets_parent():
    result = _scene()
    groups = lift_segments(_masks(LEFT, RIGHT, (-9.0, 9.0)), result)

    # Whole-plane group is the parent of both halves; points keep the halves
    assert len(groups.n_frames) == 3
    whole = np.flatnonzero(groups.parent == -1)
    assert len(whole) == 1
    halves = np.flatnonzero(groups.parent >= 0)
    assert (groups.parent[halves] == whole[0]).all()
    assert not (groups.labels == whole[0]).any()


def test_fill_labels_unclaimed_points():
    result = _scene()
    x = result.points[:, 0]
    strip = np.abs(x) < 0.15

    for features in (None, _two_class_features(result)):
        groups = lift_segments(
            _masks((-9.0, -0.3), (0.3, 9.0)), result, features=features
        )

        # Strip between the masks is filled, each side from its own half
        assert not groups.labeled[strip].any()
        assert groups.labeled[np.abs(x) > 0.5].all()
        assert (groups.labels >= 0).all()
        left = groups.labels[x < -0.5][0]
        right = groups.labels[x > 0.5][0]
        assert (groups.labels[strip & (x < -0.05)] == left).all()
        assert (groups.labels[strip & (x > 0.05)] == right).all()


def test_feature_rows_must_match_points():
    result = _scene()

    with pytest.raises(ValueError, match="rows"):
        lift_segments(_masks(LEFT), result, features=np.zeros((3, 2), np.float32))


########################################################
########## _merge_masks ################################
########################################################


def _merge(frame_masks, result, features=None, **kw) -> np.ndarray:
    """
    Group id per mask for a scene; masks are numbered frame-major in range order.
    """
    A, V, mask_frame = _mask_point_sets(frame_masks, result, 0.05, 40)
    mask_features = None if features is None else np.asarray(A @ features)
    params = {"tau": 0.6, "dup_iou": 0.7, "veto_cos": 0.5, "min_points": 40, **kw}

    return _merge_masks(A, V, mask_frame, mask_features, **params)


def test_cannot_link_blocks_same_frame_chaining():
    result = _scene()

    # Frame 0 splits the left half in two; later frames see it whole
    def frame_masks(i: int) -> np.ndarray:
        if i == 0:
            return _masks((-9.0, -1.0), (-1.0, 0.0))(i)

        return _masks(LEFT)(i)

    # Both split masks link to the whole left group at tau 0.4
    gid = _merge(frame_masks, result, tau=0.4)
    assert gid[0] != gid[1]

    gid = _merge(frame_masks, result, tau=0.4, dup_iou=0.0)
    assert gid[0] == gid[1]


def test_feature_veto_blocks_merge_across_objects():
    result = _scene()
    frame_masks = _masks(LEFT, RIGHT, (-1.0, 1.0))
    features = _two_class_features(result)

    # Cannot-link off and tau 0.4: overlap alone chains the center into both halves
    gid = _merge(frame_masks, result, tau=0.4, dup_iou=0.0)
    assert len(np.unique(gid)) == 1

    # Center's mean feature is 0.71 from either half: a 0.8 veto keeps three groups
    gid = _merge(frame_masks, result, features, tau=0.4, dup_iou=0.0, veto_cos=0.8)
    assert len(np.unique(gid)) == 3


########################################################
########## nearest_labels ##############################
########################################################


def test_nearest_labels_cutoff_is_scale_relative():
    points = np.stack([np.arange(100) * 0.01, np.zeros(100), np.zeros(100)], 1)
    labels = (np.arange(100) >= 50).astype(np.int32)
    queries = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.5, 0.2, 0.0]])

    # Spacing 0.01: 0.2 away is past 5x spacing, so unlabeled
    out = nearest_labels(queries, points, labels)
    assert out.tolist() == [0, 1, -1]

    # Same geometry scaled 100x labels the same way
    out = nearest_labels(queries * 100, points * 100, labels)
    assert out.tolist() == [0, 1, -1]
