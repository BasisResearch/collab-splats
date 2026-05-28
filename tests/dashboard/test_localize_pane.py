from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import panel as pn
import pytest

pn.extension()


def _make_scene_panel():
    from collab_splats.dashboard.panes.localize import LocalizeScenePanel
    pts3d = np.zeros((10, 3), dtype=np.float32)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 4)
    return LocalizeScenePanel(pts3d=pts3d, extrinsics=extrinsics, image_paths=[])


def test_scene_panel_constructs():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
    assert panel is not None


def test_scene_panel_reset_clears_highlight():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
        panel._highlighted_query_idx = 2
        panel._highlighted_ref_idx = 1
        panel.reset()
    assert panel._highlighted_query_idx is None
    assert panel._highlighted_ref_idx is None


def test_scene_panel_highlight_sets_indices():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
        panel.highlight(query_ext=np.eye(4), ref_ext=np.eye(4), query_idx=3, ref_idx=1)
    assert panel._highlighted_query_idx == 3
    assert panel._highlighted_ref_idx == 1


from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.localize import LocalizePane
from collab_splats.dashboard.state import AppState


def _make_pane():
    state = AppState()
    op_log = OperationLog()
    return LocalizePane(state=state, op_log=op_log), state, op_log


def test_localize_pane_run_btn_disabled_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._run_btn.disabled is True


def test_localize_pane_run_btn_still_disabled_without_query_image(tmp_path):
    # output_dir set but no query image — run still disabled
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert pane._run_btn.disabled is True


def test_localize_pane_method_dropdown_empty_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._method_dd.options == []


def test_localize_pane_method_dropdown_populated_when_zarr_found(tmp_path):
    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert "vggtx" in pane._method_dd.options


def test_localize_pane_run_btn_enabled_when_all_conditions_met(tmp_path):
    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    pane._query_input.value = str(tmp_path / "query.jpg")
    assert pane._run_btn.disabled is False


def test_run_localize_calls_localizer_and_updates_corr_info(tmp_path):
    """_run_localize() calls localizer.localize() and updates _corr_info on success."""
    from collab_splats.pointcloud.localization import LocalizationResult

    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()

    pane, state, op_log = _make_pane()
    state.output_dir = tmp_path
    pane._query_input.value = str(tmp_path / "query.jpg")

    # Create a fake query image file
    fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "query.jpg"), fake_img)

    mock_ff = MagicMock()
    mock_ff.points = np.zeros((5, 3), dtype=np.float32)
    mock_ff.extrinsics = np.stack([np.eye(4)] * 3).astype(np.float32)
    mock_ff.intrinsics = np.stack([np.eye(3)] * 3).astype(np.float32)
    mock_ff.image_paths = [tmp_path / f"f{i}.jpg" for i in range(3)]

    loc_result = LocalizationResult(
        pts2d=np.zeros((10, 2), dtype=np.float32),
        pts3d_matched=np.zeros((10, 3), dtype=np.float32),
        inlier_mask=np.ones(10, dtype=bool),
        pose=np.eye(4, dtype=np.float32),
        pts2d_ref=np.zeros((10, 2), dtype=np.float32),
        ref_frame_indices=np.zeros(10, dtype=np.int32),
        n_correspondences=10,
        n_inliers=10,
    )

    mock_localizer = MagicMock()
    mock_localizer.localize.return_value = loc_result

    with patch("collab_splats.dashboard.panes.localize.FeedforwardResult") as MockFF, \
         patch("collab_splats.dashboard.panes.localize.CameraLocalizer") as MockCL, \
         patch("collab_splats.dashboard.panes.localize.LocalizeScenePanel"), \
         patch("collab_splats.dashboard.panes.localize._render_correspondences_to_png", return_value=None):
        MockFF.load_zarr.return_value = mock_ff
        MockCL.from_feedforward.return_value = mock_localizer
        pane._run_localize(
            method="vggtx",
            extractor_name="DISK+LightGlue",
            query_path=tmp_path / "query.jpg",
            warp_corners=False,
        )

    mock_localizer.localize.assert_called_once()
    # On success path (pose is non-None), corr_info must show inlier count
    assert "inliers" in pane._corr_info.object
    assert "✗" not in pane._corr_info.object  # should not be in failure state


def test_batch_run_populates_table(tmp_path):
    """_run_batch() appends one row per image to _batch_table."""
    from collab_splats.pointcloud.localization import LocalizationResult

    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()

    # Create fake query images
    for name in ["q1.jpg", "q2.jpg"]:
        cv2.imwrite(str(tmp_path / name), np.zeros((100, 100, 3), dtype=np.uint8))

    pane, state, _ = _make_pane()
    state.output_dir = tmp_path

    mock_ff = MagicMock()
    mock_ff.points = np.zeros((5, 3), dtype=np.float32)
    mock_ff.extrinsics = np.stack([np.eye(4)] * 3).astype(np.float32)
    mock_ff.intrinsics = np.stack([np.eye(3)] * 3).astype(np.float32)
    mock_ff.image_paths = [tmp_path / f"f{i}.jpg" for i in range(3)]

    success_loc = LocalizationResult(
        pts2d=np.zeros((10, 2), dtype=np.float32),
        pts3d_matched=np.zeros((10, 3), dtype=np.float32),
        inlier_mask=np.ones(10, dtype=bool),
        pose=np.eye(4, dtype=np.float32),
        pts2d_ref=np.zeros((10, 2), dtype=np.float32),
        ref_frame_indices=np.zeros(10, dtype=np.int32),
        n_correspondences=10,
        n_inliers=10,
    )

    mock_localizer = MagicMock()
    mock_localizer.localize.return_value = success_loc

    with patch("collab_splats.dashboard.panes.localize.FeedforwardResult") as MockFF, \
         patch("collab_splats.dashboard.panes.localize.CameraLocalizer") as MockCL, \
         patch("collab_splats.dashboard.panes.localize.LocalizeScenePanel"):
        MockFF.load_zarr.return_value = mock_ff
        MockCL.from_feedforward.return_value = mock_localizer
        pane._run_batch(
            method="vggtx",
            extractor_name="DISK+LightGlue",
            folder_path=tmp_path,
        )

    assert len(pane._batch_table.value) == 2
    assert list(pane._batch_table.value["status"]) == ["✓", "✓"]
