from unittest.mock import MagicMock, patch
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
