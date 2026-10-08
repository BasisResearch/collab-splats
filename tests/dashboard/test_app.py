"""Tests for SplatsApp: sidebar wiring, scene listing, run/load/query jobs, persisted state."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from collab_splats.dashboard.app import (
    PULL_EXCLUDES,
    SceneCache,
    SplatsApp,
    _scene_options,
)
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.reconstructor import Reconstructor

# Flat curated scene id: YYYY_MM_DD-PARENTFOLDER-VIDEONAME, one video inside
SCENE = "2026_05_07-birds-clip_03"
OTHER = "2026_05_07-birds-clip_04"
BACKEND = "vggt_omega"


########
# Helpers
########


class _RecordingWorker:
    """
    Captures submitted jobs without running them.

    - proves work is deferred off the IOLoop; tests call job_fn / on_done themselves
    """

    def __init__(self):
        self.submitted = []
        self.busy = False

    def submit(self, job_fn, on_done, doc):
        self.submitted.append((job_fn, on_done, doc))


class _InlineThread:
    """Runs a thread target synchronously on start(), so detached work is deterministic."""

    def __init__(self, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def _source(scenes=(SCENE,), processed=()):
    """Mock SceneSource listing `scenes` / `processed`, with nothing on the server to pull."""
    source = MagicMock()
    source.list_scenes.return_value = list(scenes)
    source.list_processed_scenes.return_value = list(processed)
    source.scene_video.return_value = "clip_03.mp4"
    source.has_processed.return_value = False
    return source


def _app(tmp_path, source=None):
    """SplatsApp over a mocked viewer and a recording worker, with the scene listing finished."""
    source = source if source is not None else _source()

    with patch("collab_splats.dashboard.app.SplitViewer"):
        app = SplatsApp(base_dir=tmp_path, source=source, gpu_worker=_RecordingWorker())

    app._scene_thread.join(timeout=5)
    return app, source


def _select(app, scene):
    """Set the scene selection without firing autoload watchers."""
    app._suppress_autoload = True
    app.scene_select.options = [scene]
    app.scene_select.value = scene
    app._suppress_autoload = False


def _done_backend(out, backend=BACKEND, extractor=None):
    """Fake a finished pointcloud stage (zarr + COLMAP model) for one backend, optionally its run_config."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / backend / "pointcloud.zarr").mkdir(parents=True)
    (out / backend / "colmap" / "sparse" / "0").mkdir(parents=True)

    if extractor is not None:
        config = {"semantics": {"extractor": extractor}}
        (out / backend / "run_config.yaml").write_text(yaml.safe_dump(config))


def _mode_event(new):
    return type("E", (), {"new": new})()


########
# Layout and busy state
########


def test_view_hosts_the_operations_console(tmp_path):
    """The op-log console renders under the viewer and shows new log lines on the next tick."""
    app, _ = _app(tmp_path)
    template = app.view()
    app._op_log.append_line("hello console")
    app._tick()
    assert "hello console" in app._progress.object
    assert any(obj is app._console for obj in template.main[0])


def test_set_busy_disables_all_mutating_widgets(tmp_path):
    app, _ = _app(tmp_path)
    app._set_busy(True)
    widgets = (
        app.run_btn,
        app.force_btn,
        app.run_query_btn,
        app.view_mode,
        app.normalize_view,
        app.scene_select,
        app.backend,
        app.stages,
        app.overrides,
    )

    for w in widgets:
        assert w.disabled

    assert "busy" in app.busy_note.object
    app._set_busy(False)

    for w in widgets:
        assert not w.disabled

    assert app.busy_note.object == ""


def test_sync_busy_follows_worker_flag(tmp_path):
    app, _ = _app(tmp_path)
    app._gpu.busy = True
    app._sync_busy()
    assert app.run_btn.disabled
    app._gpu.busy = False
    app._sync_busy()
    assert not app.run_btn.disabled


########
# Scene listing and selection
########


def test_app_populates_scenes(tmp_path):
    """Populating options must not auto-select (and auto-load) a scene."""
    app, _ = _app(tmp_path)
    assert app.scene_select.options == {"— select a scene —": "", SCENE: SCENE}
    assert not app.scene_select.value


def test_refresh_scenes_lists_off_loop(tmp_path):
    """Both rclone listings run on the background thread, not the calling (IOLoop) thread."""
    calling_thread = threading.current_thread().name
    ran_on = {}
    source = _source()
    source.list_scenes.side_effect = lambda: ran_on.setdefault("curated", threading.current_thread().name) and []
    source.list_processed_scenes.side_effect = (
        lambda: ran_on.setdefault("processed", threading.current_thread().name) and []
    )
    _app(tmp_path, source)
    assert ran_on["curated"] != calling_thread
    assert ran_on["processed"] != calling_thread


def test_scene_options_marks_processed_scenes_with_blank_default():
    opts = _scene_options(["a", "b"], {"a"})
    assert opts == {"— select a scene —": "", "a ✓": "a", "b": "b"}
    assert next(iter(opts.values())) == ""


def test_scene_options_includes_processed_only_scenes():
    """Processed scenes whose curated source dir is gone still appear; loading needs only the id."""
    opts = _scene_options(["a"], {"a", "orphan"})
    assert opts["orphan ✓ (no source video)"] == "orphan"


def test_refresh_scenes_marks_processed_scenes_in_the_real_dropdown(tmp_path):
    """The processed listing reaches the widget: ✓ markers and processed-only entries."""
    app, _ = _app(tmp_path, _source(processed=(SCENE, OTHER)))
    assert app.scene_select.options == {
        "— select a scene —": "",
        f"{SCENE} ✓": SCENE,
        f"{OTHER} ✓ (no source video)": OTHER,
    }


def test_scene_listing_failure_shows_retry_hint(tmp_path):
    """Total listing failure surfaces a retry hint instead of a silently empty dropdown."""
    source = _source()
    source.list_scenes.side_effect = RuntimeError("rclone down")
    source.list_processed_scenes.side_effect = RuntimeError("rclone down")
    app, _ = _app(tmp_path, source)
    assert any("listing failed" in label for label in app.scene_select.options)


def test_partial_listing_failure_still_populates(tmp_path):
    """Curated-only (processed listing down) beats an empty dropdown."""
    source = _source()
    source.list_processed_scenes.side_effect = RuntimeError("rclone down")
    app, _ = _app(tmp_path, source)
    assert SCENE in app.scene_select.options.values()


def test_restore_selection_rejects_a_scene_missing_from_both_buckets(tmp_path):
    """A stale persisted scene is not set: a value outside the options breaks the dropdown."""
    (tmp_path / ".dashboard_state.yaml").write_text(yaml.safe_dump({"scene_select": "2020_01_01-gone-clip"}))
    app, _ = _app(tmp_path)
    assert app._restore_selection([SCENE], set()) is False
    assert not app.scene_select.value
    assert "2020_01_01-gone-clip" not in app.scene_select.options.values()


def test_restore_selection_accepts_a_processed_only_scene(tmp_path):
    """A scene whose curated video is gone but whose outputs remain is still restorable."""
    app, _ = _app(tmp_path)
    app._suppress_autoload = True
    app.scene_select.options = _scene_options([], {OTHER})
    app._state["scene_select"] = OTHER
    assert app._restore_selection([], {OTHER}) is True
    assert app.scene_select.value == OTHER
    app._suppress_autoload = False


def test_restore_selection_sets_scene_without_autoloading(tmp_path):
    """A browser reload restores the persisted scene but does not start a load nobody asked for."""
    (tmp_path / ".dashboard_state.yaml").write_text(yaml.safe_dump({"scene_select": SCENE}))

    with patch.object(SplatsApp, "_load_outputs") as load:
        app, _ = _app(tmp_path)

    assert app.scene_select.value == SCENE
    load.assert_not_called()


def test_autoload_current_noop_on_blank_selection(tmp_path, monkeypatch):
    """The blank entry probes nothing: no load job, no rclone call."""
    app, source = _app(tmp_path)
    probed = []
    monkeypatch.setattr(app, "_load_outputs", lambda scene, backend: probed.append(scene))
    source.has_processed.reset_mock()
    app.scene_select.value = ""
    app._autoload_current()
    assert probed == []
    source.has_processed.assert_not_called()


def test_picking_a_scene_triggers_the_load_path(tmp_path, monkeypatch):
    """The load path fires only when the user picks a scene."""
    app, _ = _app(tmp_path)
    picked = []
    monkeypatch.setattr(app, "_autoload_current", lambda: picked.append(app.scene_select.value))
    app.scene_select.value = SCENE
    assert picked == [SCENE]


def test_switching_backend_reloads(tmp_path, monkeypatch):
    """Changing the backend dropdown loads (scene, new backend)."""
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda scene, backend: loads.append((scene, backend)))
    app.backend.value = "instantsfm"
    assert loads == [(SCENE, "instantsfm")]


########
# Persisted state
########


def test_persist_state_is_debounced(tmp_path, monkeypatch):
    """Rapid widget changes coalesce into a single disk write, not one per event."""
    app, _ = _app(tmp_path)
    writes = {"n": 0}
    monkeypatch.setattr(
        type(app._state_path),
        "write_text",
        lambda self, text: writes.__setitem__("n", writes["n"] + 1),
    )

    for _ in range(5):
        app._persist_state()

    assert writes["n"] == 0
    app._flush_state()
    assert writes["n"] == 1
    app._flush_state()
    assert writes["n"] == 1


def test_persisted_state_round_trips_into_a_fresh_app(tmp_path):
    """Flushed widget values come back on the next page load, normalize_view included in the viewer."""
    app, _ = _app(tmp_path)
    app.scene_select.value = SCENE
    app.backend.value = "instantsfm"
    app.stages.value = ["pointcloud", "mesh"]
    app.overrides.value = "semantics:\n  enabled: false\n"
    app.pos_query.value = "chair"
    app.neg_query.value = "floor"
    app.max_display_points.value = 123_000
    app.normalize_view.value = False
    app._flush_state()
    restored, _ = _app(tmp_path)

    assert restored.scene_select.value == SCENE
    assert restored.backend.value == "instantsfm"
    assert restored.stages.value == ["pointcloud", "mesh"]
    assert restored.overrides.value == "semantics:\n  enabled: false\n"
    assert restored.pos_query.value == "chair"
    assert restored.neg_query.value == "floor"
    assert restored.max_display_points.value == 123_000
    assert restored.normalize_view.value is False
    restored._viewer.set_normalize_view.assert_called_once_with(False)


def test_persisted_stages_drop_unknown_names(tmp_path):
    (tmp_path / ".dashboard_state.yaml").write_text(yaml.safe_dump({"stages": ["preproc", "bogus"]}))
    app, _ = _app(tmp_path)
    assert app.stages.value == ["preproc"]


########
# Reconstructor
########


def test_reconstructor_reads_the_recorded_run_config(tmp_path):
    """The recorded extractor wins, so outputs['semantics'] names the store that run wrote."""
    app, _ = _app(tmp_path)
    _done_backend(tmp_path / SCENE, extractor="dinov2")
    rec = app._reconstructor(SCENE, BACKEND)
    assert rec.outputs["semantics"] == tmp_path / SCENE / BACKEND / "semantics" / "dinov2_lifted.zarr"
    assert rec.config["output_path"] == str(tmp_path / SCENE)
    assert rec.done("pointcloud")


def test_reconstructor_picks_the_method_from_the_registry(tmp_path):
    app, _ = _app(tmp_path)
    assert app._reconstructor(SCENE, "instantsfm").config["pointcloud"]["method"] == "sfm"
    assert app._reconstructor(SCENE, "vggt_omega").config["pointcloud"]["method"] == "feedforward"


def test_backend_dropdown_lists_every_registered_backend(tmp_path):
    app, _ = _app(tmp_path)
    assert {"vggt_omega", "instantsfm"} <= set(app.backend.options)


def test_ensure_local_video_reuses_a_local_copy(tmp_path):
    """A video already at base_dir/<scene>/<listed name> is returned without a download."""
    app, source = _app(tmp_path)
    local = tmp_path / SCENE / "clip_03.mp4"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    assert app._ensure_local_video(SCENE) == local
    source.fetch_video.assert_not_called()


def test_ensure_local_video_fetches_into_the_scene_dir(tmp_path):
    """A missing video downloads into base_dir/<scene>/, where the next run finds it."""
    app, source = _app(tmp_path)
    source.fetch_video.side_effect = lambda scene, dest_dir, on_line=None: dest_dir / "clip_03.mp4"
    assert app._ensure_local_video(SCENE) == tmp_path / SCENE / "clip_03.mp4"
    assert source.fetch_video.call_args.args == (SCENE, tmp_path / SCENE)


########
# Run
########


def test_run_button_submits_pipeline_job(tmp_path):
    """Run always enqueues, even for a done backend; the job itself decides what to skip."""
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    _done_backend(tmp_path / SCENE)
    app._on_run(event=None, force=False)
    assert len(app._gpu.submitted) == 1
    assert app.run_btn.disabled


def test_on_run_without_selection_logs_error(tmp_path):
    app, _ = _app(tmp_path)
    app.scene_select.options = []
    app._on_run(None, force=False)
    assert any("select a scene" in line for line in app._op_log.log_lines)


def test_run_click_while_busy_says_so(tmp_path):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    app._gpu.busy = True
    app._on_run(event=None, force=False)
    assert app._gpu.submitted == []
    assert "busy — wait for the current job" in app._op_log.log_lines


@pytest.mark.parametrize(
    "overrides, message",
    [
        ("- just\n- a list\n", "overrides must be a YAML mapping"),
        ("semantics: [unclosed\n", "overrides are not valid YAML"),
    ],
    ids=["not_a_mapping", "invalid_yaml"],
)
def test_run_rejects_bad_overrides(tmp_path, overrides, message):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    app.overrides.value = overrides
    app._on_run(event=None, force=False)
    assert app._gpu.submitted == []
    assert any(message in line for line in app._op_log.log_lines)


def test_run_job_writes_run_config_runs_the_stages_then_pushes(tmp_path, monkeypatch):
    """Run = CLI's three steps (build Reconstructor, dump config, run stages), then a push."""
    app, source = _app(tmp_path)
    _select(app, SCENE)

    # preproc done -> no video fetch
    (tmp_path / SCENE / "images").mkdir(parents=True)
    app.backend.value = "instantsfm"
    app.stages.value = ["pointcloud"]
    app.overrides.value = "semantics:\n  enabled: false\n"
    calls = []
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: calls.append((stages, overwrite)))
    monkeypatch.setattr(app, "_push_async", lambda scene: calls.append(("push", scene)))

    app._on_run(event=None, force=True)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    job_fn()

    assert calls == [(["pointcloud"], True), ("push", SCENE)]
    written = yaml.safe_load(Reconstructor.run_config_path(tmp_path / SCENE, "instantsfm").read_text())
    assert written["pointcloud"]["method"] == "sfm"
    assert written["output_path"] == str(tmp_path / SCENE)
    assert written["semantics"]["enabled"] is False
    source.scene_video.assert_not_called()


def test_push_uploads_the_scene_dir_and_logs_done(tmp_path, monkeypatch):
    app, source = _app(tmp_path)
    monkeypatch.setattr("collab_splats.dashboard.app.threading", SimpleNamespace(Thread=_InlineThread))
    app._push_async(SCENE)
    assert source.push_outputs.call_args.args == (tmp_path / SCENE, SCENE)
    assert any(line.startswith("push: done") for line in app._op_log.log_lines)


def test_push_failure_is_logged_not_raised(tmp_path, monkeypatch):
    """Outputs are already on local disk, so a failed upload only logs."""
    app, source = _app(tmp_path)
    monkeypatch.setattr("collab_splats.dashboard.app.threading", SimpleNamespace(Thread=_InlineThread))
    source.push_outputs.side_effect = RuntimeError("bucket down")
    app._push_async(SCENE)
    assert "push: FAILED (bucket down)" in app._op_log.log_lines


@pytest.mark.parametrize("preproc_done, force", [(False, False), (True, True)], ids=["not_done", "force_rerun"])
def test_run_of_preproc_fetches_the_video(tmp_path, monkeypatch, preproc_done, force):
    """Preproc reads the fetched video, whether it never ran or a force re-run rebuilds images/."""
    app, _ = _app(tmp_path)
    _select(app, SCENE)

    if preproc_done:
        (tmp_path / SCENE / "images").mkdir(parents=True)

    video = tmp_path / "clip_03.mp4"
    monkeypatch.setattr(app, "_ensure_local_video", lambda scene: video)

    # Real run(); only the preproc stage body is replaced, recording the input it would read
    app.stages.value = ["preproc"]
    seen = []
    monkeypatch.setattr(Reconstructor, "preproc", lambda self, **kwargs: seen.append(self.config["input_path"]))
    monkeypatch.setattr(app, "_push_async", lambda scene: None)

    app._on_run(event=None, force=force)
    app._gpu.submitted[-1][0]()

    assert seen == [str(video)]
    written = yaml.safe_load(Reconstructor.run_config_path(tmp_path / SCENE, app.backend.value).read_text())
    assert written["input_path"] == str(video)


def test_run_without_preproc_never_fetches_the_video(tmp_path, monkeypatch):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    fetches = []
    monkeypatch.setattr(app, "_ensure_local_video", lambda scene: fetches.append(scene))
    app.stages.value = ["mesh"]
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: None)
    monkeypatch.setattr(app, "_push_async", lambda scene: None)

    app._on_run(event=None, force=True)
    app._gpu.submitted[-1][0]()

    assert fetches == []


def test_run_with_no_stages_runs_every_enabled_stage(tmp_path, monkeypatch):
    """An empty Stages box passes None, which run() reads as every stage the config enables."""
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    (tmp_path / SCENE / "images").mkdir(parents=True)
    app.stages.value = []
    monkeypatch.setattr(app, "_ensure_local_video", lambda scene: tmp_path / "clip_03.mp4")
    calls = []
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: calls.append(stages))
    monkeypatch.setattr(app, "_push_async", lambda scene: None)

    app._on_run(event=None, force=False)
    app._gpu.submitted[-1][0]()

    assert calls == [None]


def test_run_pulls_server_outputs_without_dense_excludes_before_running(tmp_path, monkeypatch):
    """A backend not done locally comes down in full, dense zarr members included, before run()."""
    app, source = _app(tmp_path)
    _select(app, SCENE)
    source.has_processed.return_value = True
    (tmp_path / SCENE / "images").mkdir(parents=True)
    events = []
    source.pull_processed.side_effect = lambda *a, **kw: events.append(("pull", kw["excludes"]))
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: events.append(("run",)))
    monkeypatch.setattr(app, "_push_async", lambda scene: None)

    app._on_run(event=None, force=False)
    app._gpu.submitted[-1][0]()

    assert [e[0] for e in events] == ["pull", "run"]
    excludes = events[0][1]
    assert not set(PULL_EXCLUDES) & set(excludes)
    assert f"/{app.backend.value}/**" not in excludes


def test_run_skips_the_pull_when_the_backend_is_done_locally(tmp_path, monkeypatch):
    """Local outputs are never overwritten by a possibly stale server copy."""
    app, source = _app(tmp_path)
    _select(app, SCENE)
    source.has_processed.return_value = True
    _done_backend(tmp_path / SCENE, backend=app.backend.value)
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: None)
    monkeypatch.setattr(app, "_push_async", lambda scene: None)

    app._on_run(event=None, force=False)
    app._gpu.submitted[-1][0]()

    source.pull_processed.assert_not_called()


def test_run_dumps_the_config_pulled_from_the_server(tmp_path, monkeypatch):
    """The Reconstructor is rebuilt after the pull, so the server's run_config.yaml carries over."""
    app, source = _app(tmp_path)
    _select(app, SCENE)
    source.has_processed.return_value = True
    (tmp_path / SCENE / "images").mkdir(parents=True)
    backend = app.backend.value

    # The mocked pull lands a server-side run_config.yaml with a non-default value
    def pull(scene, dest, **kw):
        path = Reconstructor.run_config_path(dest, backend)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"preproc": {"frame_selection": "optical_flow"}}))

    source.pull_processed.side_effect = pull
    monkeypatch.setattr(Reconstructor, "run", lambda self, stages, overwrite: None)
    monkeypatch.setattr(app, "_push_async", lambda scene: None)

    app._on_run(event=None, force=False)
    app._gpu.submitted[-1][0]()

    written = yaml.safe_load(Reconstructor.run_config_path(tmp_path / SCENE, backend).read_text())
    assert written["preproc"]["frame_selection"] == "optical_flow"


def test_run_on_done_suggests_force_when_output_exists(tmp_path):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    app._on_run(event=None, force=False)
    _job, on_done, _doc = app._gpu.submitted[-1]
    on_done(ValueError("stage 'pointcloud' output already exists; pass overwrite=True to replace it"))
    assert any("use Force re-run" in line for line in app._op_log.log_lines)


def test_run_on_done_reloads_the_backend(tmp_path, monkeypatch):
    app, _ = _app(tmp_path)
    _select(app, SCENE)
    loads = []
    monkeypatch.setattr(app, "_load_outputs", lambda scene, backend: loads.append((scene, backend)))
    app._on_run(event=None, force=False)
    _job, on_done, _doc = app._gpu.submitted[-1]
    on_done(None)
    assert loads == [(SCENE, app.backend.value)]


def test_run_on_done_drops_every_cached_kind_for_the_backend(tmp_path):
    """Fresh outputs drop the backend's loaded and mesh entries; other backends keep theirs."""
    app, _ = _app(tmp_path)
    cache = app._cache
    cache.put((SCENE, BACKEND), "loaded", object())
    cache.put((SCENE, BACKEND), "mesh", object())
    other = object()
    cache.put((SCENE, "instantsfm"), "loaded", other)
    app._current_key = (SCENE, BACKEND)
    app._start_run(SCENE, BACKEND, ["mesh"], {}, overwrite=True)
    _job_fn, on_done, _doc = app._gpu.submitted[-1]
    on_done(None)
    assert cache.get((SCENE, BACKEND), "loaded") is None
    assert cache.get((SCENE, BACKEND), "mesh") is None
    assert app._current_key is None
    assert cache.get((SCENE, "instantsfm"), "loaded") is other


########
# Load
########


def test_loaded_cache_evicts_beyond_last_three():
    """Only the last N 'loaded' tuples stay resident (each can hold GBs)."""
    cache = SceneCache()

    for scene in ["a", "b", "c", "d"]:
        cache.put(scene, "loaded", scene)

    assert cache.get("a", "loaded") is None
    assert cache.get("b", "loaded") == "b"
    assert cache.get("d", "loaded") == "d"


def test_load_outputs_defers_heavy_work_to_worker(tmp_path):
    """The handler enqueues exactly one job and does not render inline."""
    app, _ = _app(tmp_path)
    _done_backend(tmp_path / SCENE)
    app._load_outputs(SCENE, BACKEND)
    app._viewer.load.assert_not_called()
    assert len(app._gpu.submitted) == 1


def test_load_outputs_inflight_dedupe(tmp_path):
    """Two requests for the same scene while its load is in flight enqueue exactly one job."""
    app, _ = _app(tmp_path)
    app._load_outputs(SCENE, BACKEND)
    app._load_outputs(SCENE, BACKEND)
    assert len(app._gpu.submitted) == 1


def test_reselecting_loaded_scene_skips_reload(tmp_path):
    """Reselecting the already-displayed scene does not enqueue another load job."""
    app, _ = _app(tmp_path)
    app._current_key = (SCENE, BACKEND)
    app._load_outputs(SCENE, BACKEND)
    assert app._gpu.submitted == []


@pytest.mark.parametrize("online", [True, False], ids=["not_on_server", "offline"])
def test_load_job_reports_a_backend_not_run(tmp_path, online):
    """No pointcloud locally or remotely (or no rclone) -> None, and on_done shows the hint."""
    app, source = _app(tmp_path)
    source.check_available.return_value = online

    if not online:
        source.has_processed.side_effect = RuntimeError("rclone is not available")

    app._load_outputs(SCENE, BACKEND)
    job_fn, on_done, _doc = app._gpu.submitted[-1]
    res = job_fn()
    on_done(res)

    assert res is None
    source.pull_processed.assert_not_called()
    app._viewer.load.assert_not_called()
    assert any(f"{BACKEND} not run for {SCENE}" in line for line in app._op_log.log_lines)
    assert not app._op_log.is_running


def test_load_job_reads_a_done_backend(tmp_path, monkeypatch):
    """A done backend loads its result, mesh path and lifted store, cached under (scene, backend)."""
    app, source = _app(tmp_path)
    out = tmp_path / SCENE
    _done_backend(out)
    lifted = out / BACKEND / "semantics" / "talk2dino_lifted.zarr"
    lifted.mkdir(parents=True)
    monkeypatch.setattr(PointcloudResult, "load_zarr", lambda p, **kwargs: "result")
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    value = job_fn()
    assert value == ("result", None, lifted, "talk2dino")
    assert app._cache.get((SCENE, BACKEND), "loaded") is value
    source.pull_processed.assert_not_called()


def test_load_job_reports_mesh_and_no_lifted_store(tmp_path, monkeypatch):
    """A mesh.ply on disk is reported; a backend with no lifted store reports None."""
    app, _ = _app(tmp_path)
    _done_backend(tmp_path / SCENE)
    rec = app._reconstructor(SCENE, BACKEND)
    rec.outputs["mesh"].write_bytes(b"ply")
    monkeypatch.setattr(PointcloudResult, "load_zarr", lambda p, **kwargs: "result")
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    value = job_fn()
    assert value[1] == rec.outputs["mesh"]
    assert value[2] is None


def test_load_job_returns_cached_value_without_pull(tmp_path):
    """A second load of a scene comes from the SceneCache, not rclone + zarr."""
    sentinel = ("result", None, "semdir", "talk2dino")
    app, source = _app(tmp_path)
    app._cache.put((SCENE, BACKEND), "loaded", sentinel)
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    assert job_fn() is sentinel
    source.pull_processed.assert_not_called()


def test_load_job_pulls_only_the_chosen_backend(tmp_path):
    """The server has the scene but not this backend: one pull scoped to it, minus dense arrays, then None."""
    app, source = _app(tmp_path)
    source.has_processed.return_value = True
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    assert job_fn() is None
    source.pull_processed.assert_called_once()
    excludes = source.pull_processed.call_args.kwargs["excludes"]
    assert "/instantsfm/**" in excludes
    assert f"/{BACKEND}/**" not in excludes
    assert set(PULL_EXCLUDES) <= set(excludes)


def test_load_job_reports_pull_progress_to_op_log(tmp_path):
    """rclone --stats lines drive the op-log progress, so the bar shows a live percent."""
    app, source = _app(tmp_path)

    # The pull reports 42% then fails, ending the job before any zarr read
    def fake_pull(scene, out, excludes=(), on_line=None):
        on_line("Transferred: 1 GiB / 2 GiB, 42%, 10 MiB/s")
        raise RuntimeError("stop before load_zarr")

    source.pull_processed.side_effect = fake_pull
    source.has_processed.return_value = True
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]

    with pytest.raises(RuntimeError, match="stop before load_zarr"):
        job_fn()

    assert app._op_log.progress == 42


def test_load_outputs_logs_steps(tmp_path, monkeypatch):
    """A cold load logs pull/read steps with elapsed times in the op log."""
    app, source = _app(tmp_path)

    # pointcloud.zarr absent -> job pulls; the fake pull materializes the zarr dir
    source.pull_processed.side_effect = lambda scene, out, excludes=(), on_line=None: _done_backend(out)
    source.has_processed.return_value = True
    monkeypatch.setattr(PointcloudResult, "load_zarr", lambda p, **kwargs: object())
    app._load_outputs(SCENE, BACKEND)
    job_fn, _on_done, _doc = app._gpu.submitted[-1]
    job_fn()
    joined = "\n".join(app._op_log.log_lines)
    assert "pulling from server" in joined
    assert "reading pointcloud.zarr" in joined and "done (" in joined


def test_load_outputs_on_done_renders_into_viewer(tmp_path):
    app, _ = _app(tmp_path)
    _done_backend(tmp_path / SCENE)
    app._load_outputs(SCENE, BACKEND)
    _job, on_done, _doc = app._gpu.submitted[0]

    # Job value is (result, mesh_path, lifted_store, extractor_name)
    result = MagicMock(points=np.zeros((10, 3)))
    on_done((result, None, "semdir", "talk2dino"))
    app._viewer.load.assert_called_once()
    kwargs = app._viewer.load.call_args.kwargs
    assert kwargs["lifted_store"] == "semdir"
    assert kwargs["max_points"] == app.max_display_points.value


def test_density_change_reloads_the_displayed_scene(tmp_path):
    """Changing max_display_points re-renders the displayed scene at the new density."""
    app, _ = _app(tmp_path)
    app._current_key = (SCENE, BACKEND)
    app.max_display_points.value = 123_000
    assert len(app._gpu.submitted) == 1
    assert app._loading_key == (SCENE, BACKEND)


def test_density_change_without_a_scene_loads_nothing(tmp_path):
    app, _ = _app(tmp_path)
    app.max_display_points.value = 123_000
    assert app._gpu.submitted == []


########
# View mode
########


def test_view_mode_mesh_defers_load_to_worker(tmp_path):
    """Switching to mesh leaves the viewer alone on the IOLoop; set_mode runs in on_done."""
    app, _ = _app(tmp_path)
    app._viewer.active_query.return_value = None
    app._on_view_mode(_mode_event("mesh"))
    app._viewer.set_mode.assert_not_called()
    assert len(app._gpu.submitted) == 1
    assert app.view_mode.disabled

    # Worker materializes the mesh polydata; on_done renders the new mode
    job_fn, on_done, _doc = app._gpu.submitted[0]
    job_fn()
    app._viewer.ensure_mesh_polydata.assert_called_once()
    on_done(None)
    app._viewer.set_mode.assert_called_once_with("mesh")
    assert not app.view_mode.disabled


def test_view_mode_mesh_populates_shared_cache(tmp_path):
    """The worker-loaded mesh polydata lands in the shared SceneCache."""
    app, _ = _app(tmp_path)
    app._current_key = (SCENE, BACKEND)
    app._viewer.active_query.return_value = None
    app._viewer.ensure_mesh_polydata.return_value = True
    app._on_view_mode(_mode_event("mesh"))
    job_fn, _on_done, _doc = app._gpu.submitted[0]
    job_fn()

    # preloaded came from the (empty) cache; the loaded polydata was put back under "mesh"
    kwargs = app._viewer.ensure_mesh_polydata.call_args.kwargs
    assert kwargs["preloaded"] is None
    assert app._cache.get((SCENE, BACKEND), "mesh") is app._viewer.mesh_polydata()


def test_view_mode_switch_rescores_active_query_on_worker(tmp_path):
    """An active query without cached colors for the new mode re-scores in the same job."""
    app, _ = _app(tmp_path)
    app._viewer.active_query.return_value = (["chair"], [], "talk2dino")
    app._viewer.cached_query_colors.return_value = None
    colors = np.zeros((3, 3), dtype=np.uint8)
    app._viewer.score_query.return_value = colors
    app._on_view_mode(_mode_event("mesh"))
    job_fn, on_done, _doc = app._gpu.submitted[0]
    res = job_fn()
    app._viewer.score_query.assert_called_once()
    on_done(res)
    app._viewer.set_mode.assert_called_once_with("mesh")
    app._viewer.render_query.assert_called_once_with(colors)


def test_view_mode_failure_snaps_radio_back(tmp_path):
    """A failed mesh load resets the radio to the displayed mode without queueing another switch."""
    app, _ = _app(tmp_path)
    app._viewer.active_query.return_value = None
    app._viewer.mode = "pointcloud"
    app.view_mode.value = "mesh"
    assert len(app._gpu.submitted) == 1
    _job_fn, on_done, _doc = app._gpu.submitted[0]
    on_done(RuntimeError("mesh read failed"))
    assert app.view_mode.value == "pointcloud"
    assert len(app._gpu.submitted) == 1
    assert any(line.startswith("ERROR") for line in app._op_log.log_lines)


########
# Query
########


def test_query_submits_score_job_with_parsed_terms(tmp_path):
    """Comma-separated terms are split and stripped before they reach score_query on the worker."""
    app, _ = _app(tmp_path)
    app.pos_query.value = "chair, stool"
    app.neg_query.value = "floor"
    app._on_query(event=None)
    assert len(app._gpu.submitted) == 1
    assert app.run_query_btn.disabled
    app._viewer.score_query.assert_not_called()

    job_fn, _on_done, _doc = app._gpu.submitted[0]
    job_fn()
    app._viewer.score_query.assert_called_once_with(positive=["chair", "stool"], negative=["floor"], extractor_name="")


def test_query_on_done_renders_colors_and_finishes_the_op(tmp_path):
    """A successful query renders and ends its op, so later scene picks are not ignored as busy."""
    app, _ = _app(tmp_path)
    app.pos_query.value = "chair"
    app._on_query(event=None)
    _job, on_done, _doc = app._gpu.submitted[0]
    colors = np.zeros((3, 3), dtype=np.uint8)
    on_done(colors)
    app._viewer.render_query.assert_called_once_with(colors)
    assert not app.run_query_btn.disabled
    assert not app._op_log.is_running


def test_query_on_done_without_a_scene_finishes_without_rendering(tmp_path):
    """score_query returns None when no scene is loaded: nothing to render, op still ends."""
    app, _ = _app(tmp_path)
    app._on_query(event=None)
    _job, on_done, _doc = app._gpu.submitted[0]
    on_done(None)
    app._viewer.render_query.assert_not_called()
    assert not app._op_log.is_running


def test_query_on_done_exception_logs_the_error(tmp_path):
    app, _ = _app(tmp_path)
    app._on_query(event=None)
    _job, on_done, _doc = app._gpu.submitted[0]
    on_done(RuntimeError("extractor failed"))
    app._viewer.render_query.assert_not_called()
    assert not app._op_log.is_running
    assert any("extractor failed" in line and line.startswith("ERROR") for line in app._op_log.log_lines)
