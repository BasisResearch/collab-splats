"""Smoke tests for SemanticsDashboard — no browser, no CUDA required."""

import pytest

pn = pytest.importorskip("panel", reason="panel not installed")
from pathlib import Path
from collab_splats.dashboard.semantics import SemanticsDashboard


def test_instantiation_no_crash(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    assert dashboard is not None


def test_create_layout_returns_material_template(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    layout = dashboard.create_layout()
    assert isinstance(layout, pn.template.MaterialTemplate)


def test_refresh_species_empty_dir(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    assert dashboard.species_select.options == []


def test_refresh_species_populates_with_videos(tmp_path):
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    assert "birds" in dashboard.species_select.options


def test_watch_chain_populates_dates(tmp_path):
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    dashboard.species_select.value = "birds"
    assert "2024-02-06" in dashboard.date_select.options


def test_watch_chain_populates_videos(tmp_path):
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    dashboard.species_select.value = "birds"
    dashboard.date_select.value = "2024-02-06"
    assert "C0043" in dashboard.video_select.options



def test_image_panes_have_max_size(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    for pane in [
        dashboard.current_frame_pane,
        dashboard.feature_overlay_pane,
        dashboard.seg_output_pane,
    ]:
        assert pane.max_width == 640
        assert pane.max_height == 480


def test_mode_badges_contain_correct_labels(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard.create_layout()
    assert "Training Mode" in dashboard._training_badge.object
    assert "Explore Mode" in dashboard._explore_badge.object


def test_sampling_mode_dd_exists_and_enabled(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    assert hasattr(dashboard, "sampling_mode_dd")
    assert dashboard.sampling_mode_dd.disabled is False
    assert dashboard.sampling_mode_dd.value == "FPS"


def test_sampling_mode_dd_enabled_in_task7():
    dashboard = SemanticsDashboard()
    assert dashboard.sampling_mode_dd.disabled is False


def test_min_disparity_slider_exists_and_initially_hidden():
    dashboard = SemanticsDashboard()
    assert hasattr(dashboard, "min_disparity_slider")
    assert dashboard.min_disparity_slider.visible is False
    assert dashboard.min_disparity_slider.value == 50.0


def test_explore_inner_tabs_exist(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    app = dashboard.create_layout()

    # flatten all panel objects to find pn.Tabs instances
    def collect(obj, found=None):
        if found is None:
            found = []
        found.append(obj)
        if hasattr(obj, "objects"):
            for o in obj.objects:
                collect(o, found)
        return found

    # MaterialTemplate exposes its contents via .main (a ListLike); seed from there
    all_objs = []
    for obj in app.main.objects:
        collect(obj, all_objs)
    tabs_objs = [o for o in all_objs if isinstance(o, pn.Tabs)]
    assert len(tabs_objs) >= 2, "Expected outer Tabs + inner Tabs for Explore"
    # find tabs with ① as first tab name
    inner = [t for t in tabs_objs if any("①" in str(n) for n in t._names)]
    assert len(inner) == 1, "Expected one inner Tabs with ① Frames"
    assert len(inner[0]) == 4, "Expected 4 sub-tabs in Explore"


def test_frame_sampling_module_importable():
    from collab_splats.utils.frame_sampling import sample_frames_fps, sample_frames_optical_flow

    assert callable(sample_frames_fps)
    assert callable(sample_frames_optical_flow)


def test_sample_frames_fps_empty_path_returns_empty():
    from collab_splats.utils.frame_sampling import sample_frames_fps

    frames, indices = sample_frames_fps("/nonexistent/video.mp4", fps=1.0)
    assert frames == []
    assert indices == []


def test_sample_frames_optical_flow_empty_path_returns_empty():
    from collab_splats.utils.frame_sampling import sample_frames_optical_flow

    result = sample_frames_optical_flow("/nonexistent/video.mp4")
    assert result == []
