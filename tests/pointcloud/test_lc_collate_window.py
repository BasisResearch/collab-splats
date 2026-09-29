"""
An LC window forward postprocesses against that window's views, not the full sequence.
"""

from unittest.mock import MagicMock, patch

import torch

from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator


def _make_mock_processed_view(tag: str) -> dict:
    """Processed view carrying only a tag; postprocess is patched out."""
    return {"img": torch.zeros(1, 3, 4, 4), "_tag": tag}


def test_lc_window_forward_uses_window_views_unmasked():
    """A views slice other than self.views is preprocessed here and stacked without the mask."""
    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator.minibatch_size = 1
    creator.views = [object() for _ in range(10)]
    creator._processed_views = [_make_mock_processed_view(f"full_{i}") for i in range(10)]
    window_views = [_make_mock_processed_view(f"win_{i}") for i in range(3)]

    # Model on CPU returning one raw pred per window view
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.forward.return_value = [{"pts3d_cam": torch.zeros(1, 3), "pts3d": torch.zeros(1, 3)} for _ in range(3)]

    captured = {}

    def fake_postprocess(preds, views, **kwargs):
        captured["views"] = views
        captured["kwargs"] = kwargs
        return [
            {
                "camera_poses": torch.eye(4).unsqueeze(0),
                "intrinsics": torch.eye(3).unsqueeze(0),
                "img_no_norm": torch.full((1, 4, 4, 3), 0.5),
                "depth_z": torch.ones(1, 4, 4, 1),
                "conf": torch.ones(1, 4, 4),
            }
            for _ in preds
        ]

    with (
        patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference", lambda v: v),
        patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference", lambda v: v),
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            side_effect=fake_postprocess,
        ),
    ):
        out = creator._forward(model, window_views)

    assert [v["_tag"] for v in captured["views"]] == ["win_0", "win_1", "win_2"]
    assert captured["kwargs"] == {"apply_mask": False}
    assert "mask" not in out
    assert out["depth"].shape == (3, 4, 4, 1)
    assert out["depth_conf"].shape == (3, 4, 4)
