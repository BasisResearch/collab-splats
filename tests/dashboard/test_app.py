# tests/dashboard/test_app.py
import threading
from unittest.mock import MagicMock, patch

import yaml

from collab_splats.dashboard.app import SplatsApp
from collab_splats.dashboard.localize import SceneCache
from collab_splats.remote import PULL_EXCLUDES

# Flat curated scene id: YYYY_MM_DD-PARENTFOLDER-VIDEONAME, one video inside.
SCENE = "2026_05_07-birds-clip_03"
OTHER = "2026_05_07-birds-clip_04"


class _RecordingWorker:
    """Captures submitted jobs WITHOUT running them — proves work is deferred off-loop."""

    def __init__(self):
        self.submitted = []
        self.busy = False  # mirrored by _sync_busy in on_done handlers

    def submit(self, job_fn, on_done, doc):
        self.submitted.append((job_fn, on_done, doc))


def _app(tmp_path):
    source = MagicMock()
    source.list_scenes.return_value = [SCENE]
    source.list_processed_scenes.return_value = []
    source.scene_video.return_value = "clip_03.mp4"
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._scene_thread.join(timeout=5)  # deterministic options for tests
    return app, source


def _select(app, scene):
    """Set the scene selection without firing autoload watchers (deterministic)."""
    app._suppress_autoload = True
    app.scene_select.options = [scene]
    app.scene_select.value = scene
    app._suppress_autoload = False


def test_app_populates_scenes(tmp_path):
    app, source = _app(tmp_path)
    # Blank-first: populating options must not auto-select (and auto-load) a scene.
    assert app.scene_select.options == {"— select a scene —": "", SCENE: SCENE}
    assert not app.scene_select.value


def test_refresh_scenes_lists_off_loop(tmp_path):
    """Both rclone listings must run on the background thread, not the calling (IOLoop) thread."""
    calling_thread = threading.current_thread().name
    ran_on = {}
    source = MagicMock()
    source.list_scenes.side_effect = lambda: ran_on.setdefault("curated", threading.current_thread().name) and []
    source.list_processed_scenes.side_effect = (
        lambda: ran_on.setdefault("processed", threading.current_thread().name) and []
    )
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._scene_thread.join(timeout=5)
    assert ran_on["curated"] != calling_thread
    assert ran_on["processed"] != calling_thread


def test_run_button_submits_pipeline_job(tmp_path):
    app, source = _app(tmp_path)
    app.scene_select.value = SCENE
    app._on_run(event=None, force=True)
    assert len(app._gpu.submitted) == 1  # pipeline deferred to the worker
    assert app.run_btn.disabled  # busy while running


def test_run_loads_cache_without_recompute(tmp_path):
    app, source = _app(tmp_path)
    app.scene_select.value = SCENE
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    with patch.object(app, "_load_outputs") as load:
        app._on_run(event=None, force=False)
    load.assert_called_once()  # loaded from cache, no recompute job


def test_force_rerun_submits_even_when_cached(tmp_path):
    app, source = _app(tmp_path)
    app.scene_select.value = SCENE
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    app._on_run(event=None, force=True)
    assert len(app._gpu.submitted) == 1  # recompute despite cache


def test_load_outputs_pull_excludes_dense_arrays(tmp_path):
    """The splats load must skip GBs of dense arrays the viewer never reads."""
    app, source = _app(tmp_path)

    app._load_outputs(SCENE)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]

    source.pull_processed.reset_mock()
    # pointcloud.zarr absent -> job pulls; assert it forwards the exclude set.
    try:
        job_fn()
    except Exception:
        pass  # load_zarr will fail on the empty tmp tree; we only assert the pull call
    _args, kwargs = source.pull_processed.call_args
    assert kwargs.get("excludes") == PULL_EXCLUDES


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
    source.list_scenes.return_value = []
    source.list_processed_scenes.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker)
    assert app._gpu is worker


def test_refresh_scenes_runs_off_loop(tmp_path):
    source = MagicMock()
    source.list_scenes.return_value = ["a", "b"]
    source.list_processed_scenes.return_value = []
    with (
        patch("collab_splats.dashboard.app.SplitViewer"),
        patch("collab_splats.dashboard.app.threading.Thread") as thread,
    ):
        SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    thread.assert_called()  # listing dispatched to a background thread, not inline on the loop


def _recording_app(tmp_path, cache=None):
    worker = _RecordingWorker()
    source = MagicMock()
    source.list_scenes.return_value = []
    source.list_processed_scenes.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=worker, cache=cache)
    return app, worker


def test_load_outputs_defers_heavy_work_to_worker(tmp_path):
    app, worker = _recording_app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    app._load_outputs(SCENE)
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
    def fake_pull(scene, out, excludes=(), on_line=None):
        if on_line:
            on_line("Transferred: 1 GiB / 2 GiB, 42%, 10 MiB/s")
        (out / "pointcloud.zarr").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("stop before load_zarr")

    app._source.pull_processed = fake_pull
    app._load_outputs(SCENE)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    try:
        job_fn()
    except Exception:
        pass
    assert 42 in seen_pct


def test_load_outputs_on_done_renders_into_viewer(tmp_path):
    app, worker = _recording_app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    app._load_outputs(SCENE)
    _job, on_done, _doc = worker.submitted[0]
    # Job returns (result, mesh_path, semantics_dir, point_features). point_features=None here means
    # an older run with no cached features; the viewer falls back to lazy lifting on first query.
    sentinel = ("result", None, "semdir", None)
    on_done(sentinel)
    app._viewer.load.assert_called_once()
    kwargs = app._viewer.load.call_args.kwargs
    assert kwargs["semantics_dir"] == "semdir"
    assert kwargs["point_features"] is None
    assert kwargs["max_points"] == app.max_display_points.value


########
# Production (Reconstructor) scene layout on the READ path — item 9
########


def _write_features_zarr(sem_dir, extractor="talk2dino"):
    """Minimal lifted per-point store at an arbitrary semantics dir."""
    import numpy as np
    import zarr

    sem_dir.mkdir(parents=True, exist_ok=True)
    zarr.open(str(sem_dir / f"{extractor}_lifted.zarr"), mode="w")["features"] = np.zeros((3, 4), dtype=np.float32)
    return sem_dir


def _run_load_job(app, scene):
    """Run the queued load job with FeedforwardResult.load_zarr stubbed out."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    app._load_outputs(scene)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    with patch.object(FeedforwardResult, "load_zarr", return_value="result"):
        return job_fn()


def test_load_outputs_resolves_the_flat_semantics_layout(tmp_path):
    """The load job reports the flat semantics dir, matching the flat pointcloud.zarr it gated on.

    Both halves of a loadable scene are flat, because the dashboard browses its own output. A
    backend-keyed (published) scene is not loadable at all — load_zarr raises on the pointcloud
    long before semantics is consulted — so there is no layout to unify here.
    """
    app, _source = _app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    flat = _write_features_zarr(tmp_path / SCENE / "semantics")

    _result, _mesh, semantics_dir, _features = _run_load_job(app, SCENE)
    assert semantics_dir == flat


def test_load_outputs_semantics_is_none_when_scene_has_none(tmp_path):
    """No flat semantics dir -> None, which viewer.load already tolerates."""
    app, _source = _app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)

    _result, _mesh, semantics_dir, _features = _run_load_job(app, SCENE)
    assert semantics_dir is None


def test_reselecting_loaded_scene_skips_reload(tmp_path):
    """Reselecting the already-displayed scene must not enqueue another load job."""
    app, worker = _recording_app(tmp_path, cache=SceneCache())
    app._current_scene = SCENE  # pretend it is displayed
    before = len(app._gpu.submitted)
    app._load_outputs(SCENE)
    assert len(app._gpu.submitted) == before  # short-circuited, no new job


def test_force_run_invalidates_scene_cache(tmp_path):
    cache = SceneCache()
    cache.put(SCENE, "loaded", object())
    cache.put(SCENE, "mesh", object())  # LocalizePage's kind, same key shape
    app, _worker = _recording_app(tmp_path, cache=cache)
    app._current_scene = SCENE
    app._invalidate_scene(SCENE)
    # Every kind for the scene is GONE (not tombstoned) — incl. LocalizePage's mesh.
    assert cache.get(SCENE, "loaded") is None
    assert cache.get(SCENE, "mesh") is None
    assert app._current_scene is None  # a post-run reload must not be short-circuited


def test_loaded_cache_evicts_beyond_last_three(tmp_path):
    """Only the last N 'loaded' tuples stay resident (each can hold GBs)."""
    cache = SceneCache()
    app, _worker = _recording_app(tmp_path, cache=cache)
    for scene in ["a", "b", "c", "d"]:
        cache.put(scene, "loaded", scene)
        app._remember_loaded(scene)
    assert cache.get("a", "loaded") is None  # oldest evicted
    assert cache.get("b", "loaded") == "b"
    assert cache.get("d", "loaded") == "d"


def test_density_change_busts_reselect_shortcircuit(tmp_path):
    """Changing max_display_points must allow a reselect to re-render at the new density."""
    app, _worker = _recording_app(tmp_path)
    app._current_scene = SCENE
    app.max_display_points.value = 123_000
    assert app._current_scene is None


def test_load_outputs_logs_steps(tmp_path, monkeypatch):
    """A cold load logs pull/read steps with elapsed times in the op log."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    app, worker = _recording_app(tmp_path)

    # pointcloud.zarr absent -> job pulls; fake pull materialises the zarr dir.
    def fake_pull(scene, out, excludes=(), on_line=None):
        (out / "pointcloud.zarr").mkdir(parents=True, exist_ok=True)

    app._source.pull_processed = fake_pull
    monkeypatch.setattr(FeedforwardResult, "load_zarr", lambda p, **kwargs: object())
    app._load_outputs(SCENE)
    job_fn, _on_done, _doc = worker.submitted[-1]
    job_fn()
    joined = "\n".join(app._op_log.log_lines)
    assert "pulling from server" in joined
    assert "reading pointcloud.zarr" in joined and "done (" in joined


def test_density_change_logs_hint(tmp_path):
    """Changing display density logs a reselect-to-apply hint in the op log."""
    app, _src = _app(tmp_path)
    app.max_display_points.value = 250_000
    assert any("display density 250,000" in line for line in app._op_log.log_lines)


def test_load_job_returns_cached_value_without_pull(tmp_path):
    """Second load of a scene must come from the SceneCache, not rclone + zarr."""
    cache = SceneCache()
    sentinel = ("result", None, "semdir", None)
    cache.put(SCENE, "loaded", sentinel)
    app, worker = _recording_app(tmp_path, cache=cache)
    app._load_outputs(SCENE)
    job_fn, _on_done, _doc = worker.submitted[-1]
    assert job_fn() is sentinel
    app._source.pull_processed.assert_not_called()


def test_load_does_not_eager_load_features(tmp_path, monkeypatch):
    """The display load must not read the cached point features before any query is issued."""
    import zarr

    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    app, worker = _recording_app(tmp_path)
    # Fake local scene: pointcloud.zarr present (skips the pull) + cached lifted features.
    out = tmp_path / SCENE
    (out / "pointcloud.zarr").mkdir(parents=True)
    sem_dir = out / "semantics"
    sem_dir.mkdir()
    (sem_dir / "talk2dino_lifted.zarr").mkdir()  # contents irrelevant — nothing may open it yet
    monkeypatch.setattr(FeedforwardResult, "load_zarr", lambda p, **kwargs: object())

    # Any eager read of the cached features (decode or on-demand lift) opens a store under
    # semantics/ — count those between enqueue and job completion; must stay zero.
    opened = {"n": 0}
    real_open = zarr.open

    def counting_open(path, *a, **k):
        if "semantics" in str(path):
            opened["n"] += 1
        return real_open(path, *a, **k)

    monkeypatch.setattr(zarr, "open", counting_open)
    app._load_outputs(SCENE)
    job_fn, _on_done, _doc = worker.submitted[-1]
    job_fn()
    assert opened["n"] == 0


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
        app.scene_select,
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
    app.scene_select.options = []
    app._on_run(None, force=False)
    assert any("select a scene" in line for line in app._op_log.log_lines)


def test_on_run_remote_check_runs_off_loop(tmp_path, monkeypatch):
    """has_processed must not block the click handler; load fires from the off-loop apply."""
    app, _source = _app(tmp_path)
    monkeypatch.setattr(app._source, "has_processed", lambda *a: True)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda s: loads.append(s))
    _select(app, SCENE)
    app._on_run(None, force=False)
    app._cache_check_thread.join(timeout=5)
    assert loads == [SCENE]


def test_double_click_during_check_fires_single_run(tmp_path, monkeypatch):
    """A second Run click during a pending server check must not queue a second pipeline."""
    app, _source = _app(tmp_path)
    _select(app, SCENE)
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
    """A slow frame-count probe for a superseded scene must not clobber the bound."""
    import collab_splats.preproc as preproc

    app, _source = _app(tmp_path)
    _select(app, SCENE)
    app._suppress_autoload = True  # probe called directly below; keep watchers quiet
    monkeypatch.setattr(app, "_ensure_local_video", lambda s: tmp_path / f"{s}.mp4")
    monkeypatch.setattr(preproc, "get_video_info", lambda p: {"total_frames": 777})
    # Late probe for OTHER lands while SCENE is selected -> dropped.
    app._update_max_frames_bound(OTHER)
    app._video_meta_thread.join(timeout=5)
    assert "777" not in app.max_frames.name
    # Probe matching the current selection applies normally.
    app._update_max_frames_bound(SCENE)
    app._video_meta_thread.join(timeout=5)
    assert app.max_frames.end == 777


def test_ensure_local_video_names_the_file_from_the_scene_listing(tmp_path):
    """fetch_video resolves the name itself; the exists() fast path needs it up front."""
    app, source = _app(tmp_path)
    local = tmp_path / SCENE / "clip_03.mp4"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    assert app._ensure_local_video(SCENE) == local
    source.fetch_video.assert_not_called()  # already local -> no download


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
    app._current_scene = SCENE
    app._viewer.active_query.return_value = None
    app._viewer.ensure_mesh_polydata.return_value = True
    app._on_view_mode(_mode_event("mesh"))
    job_fn, _on_done, _doc = worker.submitted[0]
    job_fn()
    # preloaded came from the (empty) cache; the loaded polydata was put back under "mesh".
    kwargs = app._viewer.ensure_mesh_polydata.call_args.kwargs
    assert kwargs["preloaded"] is None
    assert cache.get(SCENE, "mesh") is app._viewer.mesh_polydata()


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


def test_scene_options_marks_processed_scenes_with_blank_default():
    from collab_splats.dashboard.app import _scene_options

    opts = _scene_options(["a", "b"], {"a"})
    assert opts == {"— select a scene —": "", "a ✓": "a", "b": "b"}
    assert next(iter(opts.values())) == ""  # blank entry first -> nothing auto-selected


def test_scene_options_includes_processed_only_scenes():
    """Processed scenes whose curated source dir is gone still appear."""
    from collab_splats.dashboard.app import _scene_options

    opts = _scene_options(["a"], {"a", "orphan"})
    assert opts["orphan ✓ (no source video)"] == "orphan"  # the load path needs only the scene id


def test_refresh_scenes_marks_processed_scenes_in_the_real_dropdown(tmp_path):
    """The processed listing must actually reach the widget: ✓ markers and processed-only
    entries are the user's only signal that a scene has outputs."""
    source = MagicMock()
    source.list_scenes.return_value = [SCENE]
    source.list_processed_scenes.return_value = [SCENE, OTHER]
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._scene_thread.join(timeout=5)
    assert app.scene_select.options == {
        "— select a scene —": "",
        f"{SCENE} ✓": SCENE,
        f"{OTHER} ✓ (no source video)": OTHER,
    }


def test_restore_selection_rejects_a_scene_missing_from_both_buckets(tmp_path):
    """A stale .dashboard_state.yaml scene must not be set: assigning a value outside the
    widget's options inside a next-tick callback breaks the dropdown."""
    (tmp_path / ".dashboard_state.yaml").write_text(yaml.safe_dump({"scene_select": "2020_01_01-gone-clip"}))
    source = MagicMock()
    source.list_scenes.return_value = [SCENE]
    source.list_processed_scenes.return_value = []
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._scene_thread.join(timeout=5)
    assert app._restore_selection([SCENE], set()) is False  # nothing to restore
    assert not app.scene_select.value  # blank entry still selected
    assert "2020_01_01-gone-clip" not in app.scene_select.options.values()


def test_restore_selection_accepts_a_processed_only_scene(tmp_path):
    """A scene whose curated video is gone but whose outputs remain is still restorable."""
    from collab_splats.dashboard.app import _scene_options

    app, _src = _app(tmp_path)
    app._suppress_autoload = True  # restoring must not trip the load watchers
    app.scene_select.options = _scene_options([], {OTHER})
    app._state["scene_select"] = OTHER
    assert app._restore_selection([], {OTHER}) is True
    assert app.scene_select.value == OTHER
    app._suppress_autoload = False


def test_autoload_current_noop_on_blank_selection(tmp_path, monkeypatch):
    """Blank selection ('— select a scene —') must probe nothing: no ffprobe, no rclone."""
    app, source = _app(tmp_path)
    probed = []
    monkeypatch.setattr(app, "_update_max_frames_bound", probed.append)
    monkeypatch.setattr(app, "_load_outputs", lambda scene: probed.append(scene))
    source.has_processed.reset_mock()
    app.scene_select.value = ""  # blank entry
    app._autoload_current()
    assert probed == []
    source.has_processed.assert_not_called()


def test_restore_selection_sets_scene_without_autoloading(tmp_path):
    """A page reload restores the persisted scene but must not kick off a load nobody asked for."""
    (tmp_path / ".dashboard_state.yaml").write_text(yaml.safe_dump({"scene_select": SCENE}))
    source = MagicMock()
    source.list_scenes.return_value = [SCENE]
    source.list_processed_scenes.return_value = []
    with (
        patch("collab_splats.dashboard.app.SplitViewer"),
        patch.object(SplatsApp, "_load_outputs") as load,
    ):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
        app._scene_thread.join(timeout=5)
    assert app.scene_select.value == SCENE
    load.assert_not_called()


def test_scene_listing_failure_shows_retry_hint(tmp_path):
    """Total listing failure surfaces a retry hint instead of a silently empty dropdown."""

    def boom():
        raise RuntimeError("rclone down")

    source = MagicMock()
    source.list_scenes.side_effect = boom
    source.list_processed_scenes.side_effect = boom
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._scene_thread.join(timeout=5)
    assert any("listing failed" in label for label in app.scene_select.options)


def test_partial_listing_failure_still_populates(tmp_path):
    """Curated-only (processed listing down) beats an empty dropdown."""
    source = MagicMock()
    source.list_scenes.return_value = [SCENE]
    source.list_processed_scenes.side_effect = RuntimeError("rclone down")
    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())
    app._scene_thread.join(timeout=5)
    assert SCENE in app.scene_select.options.values()


def test_picking_a_scene_triggers_the_load_path(tmp_path, monkeypatch):
    """Explicit-select UX: the load path fires only when the user picks a scene."""
    app, _src = _app(tmp_path)
    picked = []
    monkeypatch.setattr(app, "_autoload_current", lambda: picked.append(app.scene_select.value))
    app.scene_select.value = SCENE
    assert picked == [SCENE]


def test_load_outputs_inflight_dedupe(tmp_path, monkeypatch):
    """Two requests for the same scene while its load is in flight enqueue exactly one job."""
    app, _src = _app(tmp_path)
    submitted = []
    monkeypatch.setattr(app._gpu, "submit", lambda job, on_done, doc: submitted.append(job))
    app._load_outputs(SCENE)
    app._load_outputs(SCENE)  # in flight -> dropped
    assert len(submitted) == 1


def test_score_query_targets_requested_mode(tmp_path):
    """Mode switches score in the TARGET feature space, not the not-yet-switched current one."""
    import numpy as np

    from collab_splats.dashboard.viewer import SplitViewer

    viewer = SplitViewer(off_screen=True)
    viewer.mode = "mesh"  # outgoing mode at job time
    viewer._result = type("R", (), {"colors": np.zeros((10, 3), dtype=np.uint8), "points": np.zeros((10, 3))})()
    viewer._point_features = None  # no features -> plain RGB fallback, but cache slot matters
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


def test_ensure_lift_inputs_pulls_missing_dense_members(tmp_path, monkeypatch):
    """Legacy scene (no lifted store, dense arrays excluded by the pull) fetches them."""
    app, _src = _app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    pulls = []
    monkeypatch.setattr(
        app._source, "pull_zarr_members", lambda scene, out, members, on_line=None: pulls.append(members)
    )
    app._ensure_lift_inputs(SCENE)
    assert pulls and "pixel_indices" in pulls[0]
    assert any("fetching dense arrays" in line for line in app._op_log.log_lines)


def _write_cached_features(sem_dir, *, weights=True, extractor="talk2dino"):
    """Write a semantics dir the way the pipeline does: <extractor>_lifted.zarr (+ _ae.pt)."""
    import numpy as np

    from collab_splats.semantics.compression import FeatureAutoencoder
    from collab_splats.semantics.utils import write_point_features

    ae = FeatureAutoencoder(input_dim=8, latent_dim=4)
    write_point_features(sem_dir, extractor, np.zeros((4, 4), dtype=np.float32), ae)
    if not weights:
        (sem_dir / f"{extractor}_ae.pt").unlink()  # half-written pair: codes nothing can decode


def test_ensure_lift_inputs_skips_when_lifted_cached(tmp_path, monkeypatch):
    """A cached lifted store means the lift never runs -> no dense-array fetch."""
    app, _src = _app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    _write_cached_features(tmp_path / SCENE / "semantics")
    pulls = []
    monkeypatch.setattr(app._source, "pull_zarr_members", lambda *a, **k: pulls.append(a))
    app._ensure_lift_inputs(SCENE)
    assert pulls == []


def test_ensure_lift_inputs_refetches_when_cached_features_lack_weights(tmp_path, monkeypatch):
    """Orphaned codes are not a usable cache: fetch the dense members so the re-lift can run.

    Reporting the scene as cached on the lifted store alone strands it — the lift it needs has
    no inputs, and the unreadable cache is never rewritten.
    """
    app, _src = _app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    _write_cached_features(tmp_path / SCENE / "semantics", weights=False)
    pulls = []
    monkeypatch.setattr(
        app._source, "pull_zarr_members", lambda scene, out, members, on_line=None: pulls.append(members)
    )
    assert app._ensure_lift_inputs(SCENE) is True
    assert pulls and "pixel_indices" in pulls[0]


def test_ensure_lift_inputs_skips_when_members_present(tmp_path, monkeypatch):
    """Dense members already on disk (fresh local run) -> no fetch."""
    app, _src = _app(tmp_path)
    zarr_dir = tmp_path / SCENE / "pointcloud.zarr"
    for member in ("pixel_indices", "depth", "confidence"):
        (zarr_dir / member).mkdir(parents=True)
    pulls = []
    monkeypatch.setattr(app._source, "pull_zarr_members", lambda *a, **k: pulls.append(a))
    app._ensure_lift_inputs(SCENE)
    assert pulls == []


def test_cleanup_lift_inputs_removes_members_after_lift(tmp_path):
    """Fetched dense members are deleted once the cached lifted pair exists."""
    app, _src = _app(tmp_path)
    out = tmp_path / SCENE
    for member in ("pixel_indices", "depth", "confidence"):
        d = out / "pointcloud.zarr" / member
        d.mkdir(parents=True)
        (d / "chunk").write_bytes(b"x" * 10)
    _write_cached_features(out / "semantics")
    app._cleanup_lift_inputs(SCENE)
    assert not (out / "pointcloud.zarr" / "pixel_indices").exists()
    assert not (out / "pointcloud.zarr" / "depth").exists()
    assert any("dense arrays" in line and "freed" in line for line in app._op_log.log_lines)


def test_cleanup_lift_inputs_keeps_members_when_weights_missing(tmp_path):
    """Codes without their weights are not a completed lift — keep the only inputs a retry has."""
    app, _src = _app(tmp_path)
    out = tmp_path / SCENE
    d = out / "pointcloud.zarr" / "depth"
    d.mkdir(parents=True)
    (d / "chunk").write_bytes(b"x")
    _write_cached_features(out / "semantics", weights=False)
    app._cleanup_lift_inputs(SCENE)
    assert (out / "pointcloud.zarr" / "depth").exists()


def test_cleanup_lift_inputs_keeps_members_when_lift_failed(tmp_path):
    """No cached lifted store (lift failed) -> members stay so a retry can run."""
    app, _src = _app(tmp_path)
    out = tmp_path / SCENE
    d = out / "pointcloud.zarr" / "depth"
    d.mkdir(parents=True)
    (d / "chunk").write_bytes(b"x")
    app._cleanup_lift_inputs(SCENE)
    assert (out / "pointcloud.zarr" / "depth").exists()


def test_ensure_lift_inputs_reports_fetch(tmp_path, monkeypatch):
    """Returns True only when a fetch actually happened."""
    app, _src = _app(tmp_path)
    (tmp_path / SCENE / "pointcloud.zarr").mkdir(parents=True)
    monkeypatch.setattr(app._source, "pull_zarr_members", lambda *a, **k: None)
    assert app._ensure_lift_inputs(SCENE) is True
    _write_cached_features(tmp_path / SCENE / "semantics")
    assert app._ensure_lift_inputs(SCENE) is False  # features cached -> no fetch


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
