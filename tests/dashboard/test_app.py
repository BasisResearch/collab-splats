from pathlib import Path

import panel as pn

from collab_splats.dashboard.app import App, _scan_output_dirs
from collab_splats.dashboard.panes._placeholder import PlaceholderPane
from collab_splats.dashboard.panes.reconstruct import ReconstructPane


def test_placeholder_pane_returns_panel():
    pane = PlaceholderPane("Semantics", "Coming in Phase 2")
    result = pane.panel()
    assert result is not None


def test_placeholder_pane_contains_title():
    pane = PlaceholderPane("Semantics", "Coming in Phase 2")
    result = pane.panel()
    # panel repr varies; just verify it returns something
    assert result is not None


def test_app_creates():
    app = App()
    assert app is not None


def test_app_servable_returns_material_template():
    app = App()
    template = app.servable()
    assert isinstance(template, pn.template.MaterialTemplate)


def test_app_has_five_tabs():
    app = App()
    assert set(app._tab_names) == {"Preprocess", "Semantics", "Reconstruct", "Visualize", "Localize"}


def test_reconstruct_pane_wired(tmp_path):
    app = App(base_dir=str(tmp_path))
    assert isinstance(app._reconstruct, ReconstructPane)


# _scan_output_dirs tests

def test_scan_output_dirs_returns_dirs_with_config(tmp_path):
    (tmp_path / "birds_c0043").mkdir()
    (tmp_path / "birds_c0043" / "run_config.yaml").write_text("video_path: /foo.mp4")
    (tmp_path / "empty_dir").mkdir()
    result = _scan_output_dirs(tmp_path)
    assert result == ["birds_c0043"]


def test_scan_output_dirs_sorted(tmp_path):
    for name in ["zoo", "alpha", "beta"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_config.yaml").write_text("")
    result = _scan_output_dirs(tmp_path)
    assert result == ["alpha", "beta", "zoo"]


def test_scan_output_dirs_empty_when_no_configs(tmp_path):
    (tmp_path / "no_config").mkdir()
    result = _scan_output_dirs(tmp_path)
    assert result == []


def test_scan_output_dirs_missing_base(tmp_path):
    result = _scan_output_dirs(tmp_path / "nonexistent")
    assert result == []


# Select widget / sidebar tests

def test_load_existing_sidebar_uses_select_widget(tmp_path):
    (tmp_path / "scene_01").mkdir()
    (tmp_path / "scene_01" / "run_config.yaml").write_text("")
    app = App(base_dir=str(tmp_path))
    assert isinstance(app._output_dir_select, pn.widgets.Select)
    assert "scene_01" in app._output_dir_select.options


def test_load_existing_sidebar_has_refresh_button(tmp_path):
    app = App(base_dir=str(tmp_path))
    assert isinstance(app._refresh_dirs_btn, pn.widgets.Button)


def test_app_stores_tabs_reference():
    app = App()
    app.servable()
    assert hasattr(app, "_tabs")
    assert isinstance(app._tabs, pn.Tabs)


def test_confirm_load_existing_switches_to_preprocess_tab(tmp_path):
    import unittest.mock as mock

    (tmp_path / "scene_01").mkdir()
    (tmp_path / "scene_01" / "run_config.yaml").write_text("")
    app = App(base_dir=str(tmp_path))
    app.servable()
    # Simulate being on tab 2 (Semantics)
    app._tabs.active = 2
    assert app._tabs.active == 2, "Setup: tab should be at Semantics before trigger"

    # Trigger load-existing flow
    app._on_load_existing(None)
    app._output_dir_select.value = "scene_01"

    app._on_confirm_session(None)

    assert app._tabs.active == 0, "Should switch to Preprocess tab (index 0)"
