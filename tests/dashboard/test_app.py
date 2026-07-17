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
        self.busy = False  # mirrored by _sync_busy in on_done handlers

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


def test_load_outputs_logs_steps(tmp_path, monkeypatch):
    """A cold load logs pull/read steps with elapsed times in the op log."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    app, worker = _recording_app(tmp_path)

    # feedforward.zarr absent -> job pulls; fake pull materialises the zarr dir.
    def fake_pull(session, stem, out, excludes=(), on_line=None):
        (out / "feedforward.zarr").mkdir(parents=True, exist_ok=True)

    app._source.pull_processed = fake_pull
    monkeypatch.setattr(FeedforwardResult, "load_zarr", lambda p, **kwargs: object())
    app._load_outputs("s", "v")
    job_fn, _on_done, _doc = worker.submitted[-1]
    job_fn()
    joined = "\n".join(app._op_log.log_lines)
    assert "pulling from server" in joined
    assert "reading feedforward.zarr" in joined and "done (" in joined


def test_density_change_logs_hint(tmp_path):
    """Changing display density logs a reselect-to-apply hint in the op log."""
    app, _src = _app(tmp_path)
    app.max_display_points.value = 250_000
    assert any("display density 250,000" in line for line in app._op_log.log_lines)


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


def test_set_busy_disables_all_mutating_widgets(tmp_path):
    app, _ = _app(tmp_path)
    app._set_busy(True)
    for w in (
        app.run_btn,
        app.force_btn,
        app.run_query_btn,
        app.view_mode,
        app.normalize_view,
        app.session_select,
        app.video_select,
    ):
        assert w.disabled
    assert "busy" in app.busy_note.object
    app._set_busy(False)
    assert not app.run_btn.disabled
    assert app.busy_note.object == ""


def test_sync_busy_follows_worker_flag(tmp_path):
    app, _ = _app(tmp_path)
    app._gpu.busy = True
    app._sync_busy()
    assert app.run_btn.disabled
    app._gpu.busy = False
    app._sync_busy()
    assert not app.run_btn.disabled


def test_on_run_without_selection_logs_error(tmp_path):
    app, _source = _app(tmp_path)
    app.session_select.options = []
    app.video_select.options = []
    app._on_run(None, force=False)
    assert any("select a session" in line for line in app._op_log.log_lines)


def test_on_run_remote_check_runs_off_loop(tmp_path, monkeypatch):
    """has_processed must not block the click handler; load fires from the off-loop apply."""
    app, _source = _app(tmp_path)
    monkeypatch.setattr(app._source, "has_processed", lambda *a: True)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda s, st: loads.append((s, st)))
    app._suppress_autoload = True
    app.session_select.options = ["s"]
    app.session_select.value = "s"
    # Join the video-list thread so its options apply can't race the manual ones below.
    if getattr(app, "_video_list_thread", None):
        app._video_list_thread.join(timeout=5)
    app.video_select.options = ["v.mp4"]
    app.video_select.value = "v.mp4"
    app._suppress_autoload = False
    app._on_run(None, force=False)
    app._cache_check_thread.join(timeout=5)
    assert loads == [("s", "v")]


def _select(app, session, videos, value):
    """Set session/video selection without firing autoload watchers (deterministic)."""
    app._suppress_autoload = True
    app.session_select.options = [session]
    app.session_select.value = session
    # Join the video-list thread so its options apply can't race the manual ones below.
    if getattr(app, "_video_list_thread", None):
        app._video_list_thread.join(timeout=5)
    app.video_select.options = videos
    app.video_select.value = value
    app._suppress_autoload = False


def test_double_click_during_check_fires_single_run(tmp_path, monkeypatch):
    """A second Run click during a pending server check must not queue a second pipeline."""
    app, _source = _app(tmp_path)
    _select(app, "s", ["v.mp4"], "v.mp4")
    starts = []
    monkeypatch.setattr(app, "_start_run", lambda *a: starts.append(a))
    gate = threading.Event()

    def slow_check(*_a):
        gate.wait(timeout=5)
        return False

    monkeypatch.setattr(app._source, "has_processed", slow_check)
    app._on_run(None, force=False)
    first = app._cache_check_thread
    # Check window: widgets locked, and the worker-flag poll must not unlock them.
    assert app.run_btn.disabled
    app._sync_busy()
    assert app.run_btn.disabled
    app._on_run(None, force=False)  # second click while the first check is unresolved
    gate.set()
    first.join(timeout=5)
    app._cache_check_thread.join(timeout=5)
    assert len(starts) == 1  # stale first verdict bailed; only the latest fired


def test_stale_video_meta_apply_skipped(tmp_path, monkeypatch):
    """A slow frame-count probe for a superseded video must not clobber the bound."""
    import collab_splats.preproc as preproc

    app, _source = _app(tmp_path)
    _select(app, "s", ["a.mp4", "b.mp4"], "b.mp4")
    app._suppress_autoload = True  # probe called directly below; keep watchers quiet
    monkeypatch.setattr(app, "_ensure_local_video", lambda s, n: tmp_path / n)
    monkeypatch.setattr(preproc, "get_video_info", lambda p: {"total_frames": 777})
    # Late probe for a.mp4 lands while b.mp4 is selected -> dropped.
    app._update_max_frames_bound("s", "a.mp4")
    app._video_meta_thread.join(timeout=5)
    assert "777" not in app.max_frames.name
    # Probe matching the current selection applies normally.
    app._update_max_frames_bound("s", "b.mp4")
    app._video_meta_thread.join(timeout=5)
    assert app.max_frames.end == 777


def _mode_event(new):
    return type("E", (), {"new": new})()


def test_view_mode_mesh_defers_load_to_worker(tmp_path):
    """Switching to mesh must not touch the viewer on the IOLoop; set_mode runs in on_done."""
    app, worker = _recording_app(tmp_path)
    app._viewer.active_query.return_value = None  # no active query -> no re-score
    app._on_view_mode(_mode_event("mesh"))
    app._viewer.set_mode.assert_not_called()  # no VTK mutation before the mesh is resident
    assert len(worker.submitted) == 1
    assert app.view_mode.disabled  # busy for the switch window
    job_fn, on_done, _doc = worker.submitted[0]
    job_fn()  # worker: materialise the mesh polydata
    app._viewer.ensure_mesh_polydata.assert_called_once()
    on_done(None)
    app._viewer.set_mode.assert_called_once_with("mesh")
    assert not app.view_mode.disabled  # re-enabled after the switch


def test_view_mode_mesh_populates_shared_cache(tmp_path):
    """The worker-loaded mesh polydata lands in the shared SceneCache for LocalizePage."""
    cache = SceneCache()
    app, worker = _recording_app(tmp_path, cache=cache)
    app._current_scene = ("s", "clip")
    app._viewer.active_query.return_value = None
    app._viewer.ensure_mesh_polydata.return_value = True
    app._on_view_mode(_mode_event("mesh"))
    job_fn, _on_done, _doc = worker.submitted[0]
    job_fn()
    # preloaded came from the (empty) cache; the loaded polydata was put back under "mesh".
    kwargs = app._viewer.ensure_mesh_polydata.call_args.kwargs
    assert kwargs["preloaded"] is None
    assert cache.get(("s", "clip"), "mesh") is app._viewer.mesh_polydata()


def test_view_mode_switch_rescores_active_query_on_worker(tmp_path):
    """An active query without cached colours for the new mode re-scores in the same job."""
    import numpy as np

    app, worker = _recording_app(tmp_path)
    app._viewer.active_query.return_value = (["chair"], [], "talk2dino")
    app._viewer.cached_query_colors.return_value = None
    colors = np.zeros((3, 3), dtype=np.uint8)
    app._viewer.score_query.return_value = colors
    app._on_view_mode(_mode_event("mesh"))
    job_fn, on_done, _doc = worker.submitted[0]
    res = job_fn()
    app._viewer.score_query.assert_called_once()
    on_done(res)
    app._viewer.set_mode.assert_called_once_with("mesh")
    app._viewer.render_query.assert_called_once_with(colors)


def test_video_options_marks_processed_scenes():
    from collab_splats.dashboard.app import _video_options

    opts = _video_options(["a.mp4", "b.mp4"], {"a"})
    assert opts == {"a.mp4 ✓": "a.mp4", "b.mp4": "b.mp4"}


def test_session_switch_same_video_name_still_loads(tmp_path, monkeypatch):
    """Both sessions hold the same filename: no value event fires, load must still happen."""
    app, _src = _app(tmp_path)
    monkeypatch.setattr(app._source, "list_videos", lambda s: ["C0043.mp4"])
    monkeypatch.setattr(app._source, "list_processed_stems", lambda s: [])
    loads = []
    monkeypatch.setattr(app, "_autoload_current", lambda: loads.append(app.session_select.value))
    app._suppress_autoload = True
    app.session_select.options = ["s1", "s2"]
    app.session_select.value = "s1"
    app._video_list_thread.join(timeout=5)
    app._suppress_autoload = False
    app.session_select.value = "s2"  # same video name -> Select value unchanged, no watcher
    app._video_list_thread.join(timeout=5)
    assert "s2" in loads


def test_load_outputs_inflight_dedupe(tmp_path, monkeypatch):
    """Two requests for the same scene while its load is in flight enqueue exactly one job."""
    app, _src = _app(tmp_path)
    submitted = []
    monkeypatch.setattr(app._gpu, "submit", lambda job, on_done, doc: submitted.append(job))
    app._load_outputs("s", "v")
    app._load_outputs("s", "v")  # in flight -> dropped
    assert len(submitted) == 1


def test_score_query_targets_requested_mode(tmp_path):
    """Mode switches score in the TARGET feature space, not the not-yet-switched current one."""
    import numpy as np

    from collab_splats.dashboard.viewer import SplitViewer

    viewer = SplitViewer(off_screen=True)
    viewer.mode = "mesh"  # outgoing mode at job time
    viewer._result = type("R", (), {"colors": np.zeros((10, 3), dtype=np.uint8), "points": np.zeros((10, 3))})()
    viewer._lifted_normed = None  # no features -> plain RGB fallback, but cache slot matters
    colors = viewer.score_query(positive=["x"], mode="pointcloud")
    assert len(colors) == 10  # point-space fallback, not a mesh-vertex array


def test_render_query_length_mismatch_falls_back(tmp_path):
    """Stale wrong-length colours must not crash the fast-path recolor."""
    import numpy as np

    from collab_splats.dashboard.operation_log import OperationLog
    from collab_splats.dashboard.viewer import SplitViewer

    op_log = OperationLog()
    viewer = SplitViewer(off_screen=True, op_log=op_log)
    pts = np.random.rand(20, 3).astype(np.float32)
    cols = np.zeros((20, 3), dtype=np.uint8)
    viewer.load(type("R", (), {"points": pts, "colors": cols, "extrinsics": None})(), mesh_path=None)
    stale = np.zeros((7, 3), dtype=np.uint8)  # wrong length (e.g. mesh-vertex colours)
    viewer.render_query(stale)  # must not raise
    assert any("don't match" in line for line in op_log.log_lines)


def test_view_mode_failure_snaps_radio_back(tmp_path):
    """A failed mesh load must reset the radio to the displayed mode (no dead 'mesh' state)."""
    app, worker = _recording_app(tmp_path)
    app._viewer.active_query.return_value = None
    app._viewer.mode = "pointcloud"
    app.view_mode.value = "mesh"  # fires the watcher -> switch job submitted
    assert len(worker.submitted) == 1
    _job_fn, on_done, _doc = worker.submitted[0]
    on_done(RuntimeError("mesh read failed"))
    # Radio snapped back to what is displayed, without re-firing the watcher (no new job).
    assert app.view_mode.value == "pointcloud"
    assert len(worker.submitted) == 1
    assert any(line.startswith("ERROR") for line in app._op_log.log_lines)


def test_warm_heavy_stack_imports_localizer_and_pipeline(monkeypatch):
    """Warm thread must front-load the localizer + mesh/pipeline stacks, not just feedforward."""
    from collab_splats.dashboard import app as app_mod

    imported = []
    monkeypatch.setattr(app_mod.importlib, "import_module", lambda name, *a, **k: imported.append(name))
    app_mod._warm_heavy_stack()
    assert any("localization.localizer" in n for n in imported)
    assert any("dashboard.pipeline" in n for n in imported)
    assert any("semantics.features.base" in n for n in imported)
