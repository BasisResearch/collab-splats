"""Tests for load_lifted_normed module-level helper."""

from unittest.mock import patch

import numpy as np
import torch

from collab_splats.dashboard.viewer import load_lifted_normed
from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def test_load_lifted_normed_normalises():
    fake_lifted = torch.tensor([[3.0, 4.0], [0.0, 2.0]])  # norms 5, 2
    with (
        patch("collab_splats.pointcloud.utils.lift_features", return_value=fake_lifted),
        patch("collab_splats.dashboard.viewer._load_feature_maps", return_value=["fm"]),
    ):
        out = load_lifted_normed(result=object(), semantics_dir=object())
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_load_lifted_normed_reloads_dense_on_demand(tmp_path):
    """A lean display result must be re-hydrated from its zarr before lifting."""
    # Dense-bearing store: depth + confidence + pixel_indices persisted on disk.
    n, p, h, w = 2, 5, 4, 4
    full = FeedforwardResult(
        points=np.zeros((p, 3), dtype=np.float32),
        colors=np.zeros((p, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        image_paths=[tmp_path / f"{i:05d}.jpg" for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
        depth=np.ones((n, h, w), dtype=np.float32),
        confidence=torch.ones((n, h, w)),
        pixel_indices=np.zeros((p, 3), dtype=np.int32),
    )
    store = tmp_path / "feedforward.zarr"
    full.save_zarr(store)
    lean = FeedforwardResult.load_zarr(
        store,
        load_depth=False,
        load_world_points=False,
        load_confidence=False,
        load_features=False,
        load_pixel_indices=False,
    )

    seen = {}

    def fake_lift(feature_maps, result):
        # Record whether the lift saw the re-hydrated dense members.
        seen["dense"] = all(getattr(result, f) is not None for f in ("depth", "confidence", "pixel_indices"))
        return torch.zeros((p, 2))

    with (
        patch("collab_splats.pointcloud.utils.lift_features", fake_lift),
        patch("collab_splats.dashboard.viewer._load_feature_maps", return_value=[]),
    ):
        load_lifted_normed(lean, tmp_path)
    assert seen["dense"] is True


def test_ensure_lifted_self_upgrades_legacy_scene(tmp_path, monkeypatch):
    """A successful on-demand lift persists lifted_normed.npy so the scene upgrades."""
    import numpy as np

    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.operation_log import OperationLog
    from collab_splats.dashboard.viewer import SplitViewer

    lifted = np.ones((5, 4), dtype=np.float32)
    monkeypatch.setattr(viewer_mod, "load_lifted_normed", lambda result, sem: lifted)
    op_log = OperationLog()
    viewer = SplitViewer(off_screen=True, op_log=op_log)
    viewer._result = object()  # anything non-None; the lift itself is stubbed
    viewer._semantics_dir = tmp_path
    viewer.ensure_lifted(op_log)
    saved = tmp_path / "lifted_normed.npy"
    assert saved.exists()
    np.testing.assert_array_equal(np.load(saved), lifted)
    assert any("scene upgraded" in line for line in op_log.log_lines)


def test_ensure_lifted_failure_does_not_write_npy(tmp_path, monkeypatch):
    import collab_splats.dashboard.viewer as viewer_mod
    from collab_splats.dashboard.viewer import SplitViewer

    def boom(result, sem):
        raise RuntimeError("missing pixel_indices")

    monkeypatch.setattr(viewer_mod, "load_lifted_normed", boom)
    viewer = SplitViewer(off_screen=True)
    viewer._result = object()
    viewer._semantics_dir = tmp_path
    viewer.ensure_lifted(None)
    assert viewer._lifted_normed is None
    assert not (tmp_path / "lifted_normed.npy").exists()
