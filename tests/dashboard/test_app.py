# tests/dashboard/test_app.py
from unittest.mock import MagicMock, patch

from collab_splats.dashboard.app import SplatsApp


class _RecordingWorker:
    """Captures submitted jobs WITHOUT running them — proves work is deferred off-loop."""

    def __init__(self):
        self.submitted = []

    def submit(self, job_fn, on_done, doc):
        self.submitted.append((job_fn, on_done, doc))


def _app(tmp_path):
    source = MagicMock()
    source.list_sessions.return_value = ["2026_05_07"]
    source.list_videos.return_value = ["clip_03.mp4"]
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._session_thread.join(timeout=5)  # deterministic options for tests
    return app, source


def test_app_populates_sessions(tmp_path):
    app, source = _app(tmp_path)
    assert app.session_select.options == ["2026_05_07"]


def test_selecting_session_lists_videos(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    assert "clip_03.mp4" in app.video_select.options


def test_run_button_submits_pipeline_job(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    app._on_run(event=None, force=True)
    assert len(app._gpu.submitted) == 1  # pipeline deferred to the worker
    assert app.run_btn.disabled  # busy while running


def test_run_loads_cache_without_recompute(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    out = tmp_path / "2026_05_07" / "clip_03" / "feedforward.zarr"
    out.mkdir(parents=True)
    with patch.object(app, "_load_outputs") as load:
        app._on_run(event=None, force=False)
    load.assert_called_once()  # loaded from cache, no recompute job


def test_force_rerun_submits_even_when_cached(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    (tmp_path / "2026_05_07" / "clip_03" / "feedforward.zarr").mkdir(parents=True)
    app._on_run(event=None, force=True)
    assert len(app._gpu.submitted) == 1  # recompute despite cache


def test_min_disparity_visibility_tracks_sampling(tmp_path):
    app, _ = _app(tmp_path)
    # Hidden under balanced (fps) sampling; shown only for optical_flow.
    assert app.min_disparity.visible is False
    app.sampling.value = "optical_flow"
    assert app.min_disparity.visible is True
    app.sampling.value = "balanced"
    assert app.min_disparity.visible is False


def test_query_submits_score_job_with_parsed_terms(tmp_path):
    app, _ = _app(tmp_path)
    app.pos_query.value = "chair, stool"
    app.neg_query.value = "floor"
    app._on_query(event=None)
    # Deferred to the worker, not scored inline.
    assert len(app._gpu.submitted) == 1
    assert app.run_query_btn.disabled


def test_query_on_done_renders_colors(tmp_path):
    import numpy as np

    app, _ = _app(tmp_path)
    app.pos_query.value = "chair"
    app._on_query(event=None)
    _job, on_done, _doc = app._gpu.submitted[0]
    colors = np.zeros((3, 3), dtype=np.uint8)
    on_done(colors)
    app._viewer.render_query.assert_called_once_with(colors)
    assert not app.run_query_btn.disabled  # re-enabled after render


def test_run_app_serves_with_hardening(tmp_path):
    with (
        patch("collab_splats.dashboard.app._ensure_display"),
        patch("collab_splats.dashboard.app.pn.extension"),
        patch("collab_splats.dashboard.app.GpuWorker") as worker_cls,
        patch("collab_splats.dashboard.app.pn.serve") as serve,
    ):
        from collab_splats.dashboard.app import run_app

        run_app(host="127.0.0.1", port=9999, base_dir=str(tmp_path), websocket_origin=None)
    worker_cls.assert_called_once()  # one shared worker for all sessions
    kwargs = serve.call_args.kwargs
    assert kwargs["session_token_expiration"] >= 1800
    assert kwargs["websocket_origin"] == ["127.0.0.1:9999", "localhost:9999"]


def test_view(tmp_path):
    app, _ = _app(tmp_path)
    assert app.view() is not None


from collab_splats.dashboard.gpu_worker import GpuWorker


def test_set_busy_toggles_action_buttons(tmp_path):
    app, _ = _app(tmp_path)
    app._set_busy(True)
    assert app.run_btn.disabled and app.force_btn.disabled and app.run_query_btn.disabled
    app._set_busy(False)
    assert not app.run_btn.disabled and not app.force_btn.disabled and not app.run_query_btn.disabled


def test_has_max_display_points_widget(tmp_path):
    app, _ = _app(tmp_path)
    assert app.max_display_points.value == 150_000


def test_app_uses_injected_gpu_worker(tmp_path):
    worker = MagicMock(spec=GpuWorker)
    source = MagicMock()
    source.list_sessions.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    assert app._gpu is worker


def test_refresh_sessions_runs_off_loop(tmp_path):
    source = MagicMock()
    source.list_sessions.return_value = ["a", "b"]
    with (
        patch("collab_splats.dashboard.app.SplitViewer"),
        patch("collab_splats.dashboard.app.threading.Thread") as thread,
    ):
        SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    thread.assert_called()  # listing dispatched to a background thread, not inline on the loop


def _recording_app(tmp_path):
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_sessions.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    return app, worker


def test_load_outputs_defers_heavy_work_to_worker(tmp_path):
    app, worker = _recording_app(tmp_path)
    out = tmp_path / "s" / "clip" / "feedforward.zarr"
    out.mkdir(parents=True)
    app._load_outputs("s", "clip")
    # The handler must NOT render inline; it enqueues exactly one job.
    app._viewer.load.assert_not_called()
    assert len(worker.submitted) == 1
    assert callable(worker.submitted[0][0])  # job_fn deferred to the worker


def test_load_outputs_on_done_renders_into_viewer(tmp_path):
    app, worker = _recording_app(tmp_path)
    (tmp_path / "s" / "clip" / "feedforward.zarr").mkdir(parents=True)
    app._load_outputs("s", "clip")
    _job, on_done, _doc = worker.submitted[0]
    sentinel = ("result", None, "lifted")
    on_done(sentinel)
    app._viewer.load.assert_called_once()
    kwargs = app._viewer.load.call_args.kwargs
    assert kwargs["max_points"] == app.max_display_points.value
