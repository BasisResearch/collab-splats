# tests/dashboard/test_app.py
from unittest.mock import MagicMock, patch

from collab_splats.dashboard.app import SplatsApp


def _app(tmp_path):
    source = MagicMock()
    source.list_sessions.return_value = ["2026_05_07"]
    source.list_videos.return_value = ["clip_03.mp4"]
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        return SplatsApp(base_dir=tmp_path, source=source), source


def test_app_populates_sessions(tmp_path):
    app, source = _app(tmp_path)
    assert app.session_select.options == ["2026_05_07"]


def test_selecting_session_lists_videos(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    assert "clip_03.mp4" in app.video_select.options


def test_run_button_spawns_pipeline(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    with (
        patch("collab_splats.dashboard.app.threading.Thread") as thread,
        patch.object(app, "_ensure_local_video", return_value=tmp_path / "clip_03.mp4"),
    ):
        app._on_run(event=None, force=True)
    thread.assert_called_once()


def test_run_loads_cache_without_recompute(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    # Simulate a cached result on disk
    out = tmp_path / "2026_05_07" / "clip_03" / "feedforward.zarr"
    out.mkdir(parents=True)
    with patch("collab_splats.dashboard.app.threading.Thread") as thread, patch.object(app, "_load_outputs") as load:
        app._on_run(event=None, force=False)
    thread.assert_not_called()  # no recompute
    load.assert_called_once()  # loaded from cache


def test_force_rerun_recomputes_even_when_cached(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    out = tmp_path / "2026_05_07" / "clip_03" / "feedforward.zarr"
    out.mkdir(parents=True)
    with (
        patch("collab_splats.dashboard.app.threading.Thread") as thread,
        patch.object(app, "_ensure_local_video", return_value=tmp_path / "clip_03.mp4"),
    ):
        app._on_run(event=None, force=True)
    thread.assert_called_once()  # recompute despite cache


def test_min_disparity_visibility_tracks_sampling(tmp_path):
    app, _ = _app(tmp_path)
    # Hidden under balanced (fps) sampling; shown only for optical_flow.
    assert app.min_disparity.visible is False
    app.sampling.value = "optical_flow"
    assert app.min_disparity.visible is True
    app.sampling.value = "balanced"
    assert app.min_disparity.visible is False


def test_run_query_button_forwards_parsed_terms(tmp_path):
    app, _ = _app(tmp_path)
    app.pos_query.value = "chair, stool"
    app.neg_query.value = "floor"
    app._on_query(event=None)
    app._viewer.query.assert_called_once()
    kwargs = app._viewer.query.call_args.kwargs
    assert kwargs["positive"] == ["chair", "stool"]
    assert kwargs["negative"] == ["floor"]


def test_view(tmp_path):
    app, _ = _app(tmp_path)
    assert app.view() is not None
