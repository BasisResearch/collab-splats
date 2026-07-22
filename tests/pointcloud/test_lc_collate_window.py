"""Test that _lc_collate_outputs uses window-specific view context, not first-K frames."""
from unittest.mock import patch
import numpy as np
import pytest
import torch


def _make_mock_processed_view(tag: str):
    # Postprocess is patched out in these tests; only _tag is inspected.
    return {"img": torch.zeros(1, 3, 224, 224), "_tag": tag}


def test_lc_collate_uses_window_views_not_full_sequence():
    """_lc_collate_outputs must use the window view context stored by _forward,
    not self._processed_views[:N] which always points at the first K frames."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator._processed_views = [_make_mock_processed_view(f"full_{i}") for i in range(10)]
    window_views = [_make_mock_processed_view(f"win_{i}") for i in range(3)]
    creator._lc_window_views = window_views

    captured_views = []

    def fake_postprocess(raw_list, views_ctx, apply_mask):
        captured_views.extend(views_ctx)
        # Include depth_z/conf — collation emits them as 'depth'/'depth_conf';
        # img_no_norm is the [0,1] denormalized image collation stacks into 'colors'
        return [
            {"camera_poses": [torch.eye(4).unsqueeze(0)],
             "intrinsics": [torch.eye(3).unsqueeze(0)],
             "img_no_norm": torch.full((1, 4, 4, 3), 0.5),
             "depth_z": torch.ones(1, 4, 4, 1),
             "conf": torch.ones(1, 4, 4)}
            for _ in raw_list
        ]

    raw_list = [
        {"pts3d_cam": torch.zeros(1, 3), "pts3d": torch.zeros(1, 3)}
        for _ in range(3)
    ]

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        side_effect=fake_postprocess,
    ), patch(
        "collab_splats.pointcloud.feedforward.mapanything.invert_poses",
        return_value=np.eye(4),
    ):
        out = creator._lc_collate_outputs(raw_list)

    assert all(v["_tag"].startswith("win_") for v in captured_views), (
        f"Expected window views (win_*), got: {[v['_tag'] for v in captured_views]}"
    )
    assert creator._lc_window_views is None
    # Geometry keys collated from depth_z/conf with per-frame stacking
    assert out["depth"].shape == (3, 4, 4, 1)
    assert out["depth_conf"].shape == (3, 4, 4)


def test_lc_collate_raises_when_window_views_not_set():
    """_lc_collate_outputs must raise if _lc_window_views is None (broken call order)."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator._processed_views = [_make_mock_processed_view("full_0")]
    creator._lc_window_views = None

    raw_list = [{"pts3d_cam": torch.zeros(1, 3), "pts3d": torch.zeros(1, 3)}]

    with pytest.raises(RuntimeError, match="_lc_window_views is None"):
        creator._lc_collate_outputs(raw_list)
