# tests/dashboard/test_app.py
import threading
from unittest.mock import MagicMock, patch

from collab_splats.dashboard.app import SplatsApp
from collab_splats.dashboard.localize import SceneCache
from collab_splats.dashboard.sources import PULL_EXCLUDES


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
    if getattr(app, "_video_list_thread", None):
        app._video_list_thread.join(timeout=5)
    assert "clip_03.mp4" in app.video_select.options


def test_on_session_lists_videos_off_loop(tmp_path):
    """Selecting a session must not call rclone list_videos on the calling (IOLoop) thread."""
    app, _source = _app(tmp_path)
    calling_thread = threading.current_thread().name
    ran_on = {}
    orig = app._source.list_videos

    def tracking_list(sess):
        ran_on["thread"] = threading.current_thread().name
        return orig(sess)

    app._source.list_videos = tracking_list
    app._on_session(type("E", (), {"new": "2026_05_07"})())
    if getattr(app, "_video_list_thread", None):
        app._video_list_thread.join(timeout=5)
    assert ran_on["thread"] != calling_thread


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


def test_load_outputs_pull_excludes_dense_arrays(tmp_path):
    """The splats load must skip GBs of dense arrays the viewer never reads."""
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.options = ["clip_03.mp4"]

    app._load_outputs("2026_05_07", "clip_03")
    job_fn, _on_done, _doc = app._gpu.submitted[-1]

    source.pull_processed.reset_mock()
    # feedforward.zarr absent -> job pulls; assert it forwards the exclude set.
    try:
        job_fn()
    except Exception:
        pass  # load_zarr will fail on the empty tmp tree; we only assert the pull call
    _args, kwargs = source.pull_processed.call_args
    assert kwargs.get("excludes") == PULL_EXCLUDES or (len(_args) >= 4 and _args[3] == PULL_EXCLUDES)


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
    assert app.max_display_points.value == 500_000


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


def _recording_app(tmp_path, cache=None):
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_sessions.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker, cache=cache)
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


def test_load_job_reports_pull_progress_to_op_log(tmp_path):
    """rclone --stats lines must drive op_log.update_progress so the bar shows a live %."""
    app, _worker = _recording_app(tmp_path)
    seen_pct = []
    app._op_log.update_progress = lambda pct, message="", log=True: seen_pct.append(pct)

    # pull_processed invokes on_line with a stats line carrying 42%.
    def fake_pull(session, stem, out, excludes=(), on_line=None):
        if on_line:
            on_line("Transferred: 1 GiB / 2 GiB, 42%, 10 MiB/s")
        (out / "feedforward.zarr").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("stop before load_zarr")

    app._source.pull_processed = fake_pull
    app._load_outputs("2026_05_07", "clip_03")
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    try:
        job_fn()
    except Exception:
        pass
    assert 42 in seen_pct


def test_load_outputs_on_done_renders_into_viewer(tmp_path):
    app, worker = _recording_app(tmp_path)
    (tmp_path / "s" / "clip" / "feedforward.zarr").mkdir(parents=True)
    app._load_outputs("s", "clip")
    _job, on_done, _doc = worker.submitted[0]
    # Job returns (result, mesh_path, semantics_dir, lifted_normed). lifted_normed=None here means
    # an older run with no cached features; the viewer falls back to lazy lifting on first query.
    sentinel = ("result", None, "semdir", None)
    on_done(sentinel)
    app._viewer.load.assert_called_once()
    kwargs = app._viewer.load.call_args.kwargs
    assert kwargs["semantics_dir"] == "semdir"
    assert kwargs["lifted_normed"] is None
    assert kwargs["max_points"] == app.max_display_points.value


def test_reselecting_loaded_scene_skips_reload(tmp_path):
    """Reselecting the already-displayed scene must not enqueue another load job."""
    app, worker = _recording_app(tmp_path, cache=SceneCache())
    app._current_scene = ("2026_05_07", "clip_03")  # pretend it is displayed
    before = len(app._gpu.submitted)
    app._load_outputs("2026_05_07", "clip_03")
    assert len(app._gpu.submitted) == before  # short-circuited, no new job


def test_force_run_invalidates_scene_cache(tmp_path):
    cache = SceneCache()
    cache.put(("2026_05_07", "clip_03"), "loaded", object())
    cache.put(("2026_05_07", "clip_03"), "mesh", object())  # LocalizePage's kind, same key shape
    app, _worker = _recording_app(tmp_path, cache=cache)
    app._current_scene = ("2026_05_07", "clip_03")
    app._invalidate_scene("2026_05_07", "clip_03")
    # Every kind for the scene is GONE (not tombstoned) — incl. LocalizePage's mesh.
    assert cache.get(("2026_05_07", "clip_03"), "loaded") is None
    assert cache.get(("2026_05_07", "clip_03"), "mesh") is None
    assert app._current_scene is None  # a post-run reload must not be short-circuited


def test_loaded_cache_evicts_beyond_last_three(tmp_path):
    """Only the last N 'loaded' tuples stay resident (each can hold GBs)."""
    cache = SceneCache()
    app, _worker = _recording_app(tmp_path, cache=cache)
    for stem in ["a", "b", "c", "d"]:
        cache.put(("s", stem), "loaded", stem)
        app._remember_loaded(("s", stem))
    assert cache.get(("s", "a"), "loaded") is None  # oldest evicted
    assert cache.get(("s", "b"), "loaded") == "b"
    assert cache.get(("s", "d"), "loaded") == "d"


def test_density_change_busts_reselect_shortcircuit(tmp_path):
    """Changing max_display_points must allow a reselect to re-render at the new density."""
    app, _worker = _recording_app(tmp_path)
    app._current_scene = ("s", "clip")
    app.max_display_points.value = 123_000
    assert app._current_scene is None


def test_load_job_returns_cached_value_without_pull(tmp_path):
    """Second load of a scene must come from the SceneCache, not rclone + zarr."""
    cache = SceneCache()
    sentinel = ("result", None, "semdir", None)
    cache.put(("2026_05_07", "clip_03"), "loaded", sentinel)
    app, worker = _recording_app(tmp_path, cache=cache)
    app._load_outputs("2026_05_07", "clip_03")
    job_fn, _on_done, _doc = worker.submitted[-1]
    assert job_fn() is sentinel
    app._source.pull_processed.assert_not_called()


def test_load_does_not_eager_load_lifted_normed(tmp_path, monkeypatch):
    """The display load must not np.load lifted features before any query is issued."""
    import numpy as np

    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    app, worker = _recording_app(tmp_path)
    # Fake local scene: feedforward.zarr present (skips the pull) + cached lifted features.
    out = tmp_path / "s" / "clip"
    (out / "feedforward.zarr").mkdir(parents=True)
    sem_dir = out / "semantics"
    sem_dir.mkdir()
    np.save(sem_dir / "lifted_normed.npy", np.zeros((4, 2), dtype=np.float32))
    monkeypatch.setattr(FeedforwardResult, "load_zarr", lambda p, **kwargs: object())

    # Count every np.load between enqueue and job completion — must stay zero.
    loaded = {"n": 0}
    real_load = np.load

    def counting_load(*a, **k):
        loaded["n"] += 1
        return real_load(*a, **k)

    monkeypatch.setattr(np, "load", counting_load)
    app._load_outputs("s", "clip")
    job_fn, _on_done, _doc = worker.submitted[-1]
    job_fn()
    assert loaded["n"] == 0


def test_persist_state_is_debounced(tmp_path, monkeypatch):
    """Rapid widget changes coalesce into a single disk write, not one per event."""
    app, _source = _app(tmp_path)
    writes = {"n": 0}
    monkeypatch.setattr(
        type(app._state_path),
        "write_text",
        lambda self, text: writes.__setitem__("n", writes["n"] + 1),
    )

    for _ in range(5):
        app._persist_state()
    assert writes["n"] == 0  # nothing written synchronously
    app._flush_state()
    assert writes["n"] == 1  # one coalesced write
    app._flush_state()
    assert writes["n"] == 1  # not dirty -> no second write


def test_warm_heavy_stack_imports_localizer_and_pipeline(monkeypatch):
    """Warm thread must front-load the localizer + mesh/pipeline stacks, not just feedforward."""
    from collab_splats.dashboard import app as app_mod

    imported = []
    monkeypatch.setattr(
        app_mod.importlib, "import_module", lambda name, *a, **k: imported.append(name)
    )
    app_mod._warm_heavy_stack()
    assert any("localization.localizer" in n for n in imported)
    assert any("dashboard.pipeline" in n for n in imported)
    assert any("semantics.features.base" in n for n in imported)
