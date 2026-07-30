"""Remote driver: order of operations, verify gate, cleanup, failure isolation, CLI wiring."""

import fnmatch
import importlib.util
import logging
from pathlib import Path

import pytest

from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE

_DRIVER = Path(__file__).parent.parent.parent / "docs" / "examples" / "run_pipeline_remote.py"

# Distinguishable sentinels for every pipeline arg the driver forwards to batch.run_scene.
# Each has a different type/value from its neighbours, so swapping any two adjacent
# positional slots (e.g. --stages read as overwrite) trips an assertion in the fake.
_CONFIG_DIR = Path("/sentinel/config-dir")
_OVERRIDE_CONFIG = {"sentinel": "override-config"}
_STAGES = ["preproc", "pointcloud"]
_OVERWRITE = True


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_pipeline_remote", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeSource:
    def __init__(self, scenes=("s1",), verify=True, video_ext=".mp4", available=True, fail_fetch=()):
        self._scenes = list(scenes)
        self._verify = verify
        self._video_ext = video_ext
        # available stands in for the liveness re-probe: False = rclone has stopped working.
        self._available = available
        self._fail_fetch = set(fail_fetch)
        self.calls = []
        # op -> the on_line progress callback the driver passed (None if it passed none)
        self.on_lines = {}

    def list_scenes(self):
        self.calls.append(("list_scenes",))
        return list(self._scenes)

    def check_available(self):
        self.calls.append(("check_available",))
        return self._available

    def fetch_video(self, scene, dest_dir, on_line=None):
        self.calls.append(("fetch", scene))
        self.on_lines["fetch"] = on_line
        if scene in self._fail_fetch:
            raise RuntimeError("rclone copyto failed (exit 1): 401 unauthorized")
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        video = dest_dir / f"{scene}{self._video_ext}"
        video.write_bytes(b"video")
        return video

    def push_outputs(self, local_dir, scene, on_line=None):
        self.calls.append(("push", scene))
        self.on_lines["push"] = on_line

    def verify_push(self, local_dir, scene, on_line=None):
        self.calls.append(("verify", scene))
        self.on_lines["verify"] = on_line
        return self._verify


@pytest.fixture()
def driver():
    return _load_driver()


def _patch_run_scene(driver, monkeypatch, fail_on=()):
    """Replace batch.run_scene with a fake that pins each positional arg to its sentinel."""

    def _run_scene(video, output_root, config_dir, override_config, stages, overwrite, name=None):
        # Arg-order guard: a reordered call site lands a sentinel in the wrong slot
        assert Path(video).is_file(), f"video slot got {video!r}"
        assert Path(output_root).is_dir(), f"output_root slot got {output_root!r}"
        assert config_dir == _CONFIG_DIR, f"config_dir slot got {config_dir!r}"
        assert override_config == _OVERRIDE_CONFIG, f"override_config slot got {override_config!r}"
        assert stages == _STAGES, f"stages slot got {stages!r}"
        assert overwrite is _OVERWRITE, f"overwrite slot got {overwrite!r}"
        assert name is not None, "name= must pin the output dir to the scene id"
        if name in fail_on:
            raise RuntimeError("reconstruction failed")
        out = Path(output_root) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "sparse_pc.ply").write_bytes(b"ply")
        return out, None

    monkeypatch.setattr(driver.batch, "run_scene", _run_scene)


def _run(driver, src, scenes, output_root, keep_local=False):
    """Call run_remote with the sentinel pipeline args every test shares."""
    return driver.run_remote(
        src, scenes, output_root, _CONFIG_DIR, _OVERRIDE_CONFIG, _STAGES, _OVERWRITE, keep_local=keep_local
    )


########
# Order of operations, verify gate, cleanup
########


def test_happy_path_order_is_fetch_run_push_verify_delete(driver, monkeypatch, tmp_path):
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, ["s1"], tmp_path)
    assert code == 0
    assert [c[0] for c in src.calls] == ["fetch", "push", "verify"]
    assert not (tmp_path / "s1").exists()


def test_all_processes_every_listed_scene(driver, monkeypatch, tmp_path):
    src = _FakeSource(scenes=("s1", "s2"))
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == 0
    assert [c for c in src.calls if c[0] == "push"] == [("push", "s1"), ("push", "s2")]


def test_failed_verify_keeps_local_data(driver, monkeypatch, tmp_path):
    src = _FakeSource(verify=False)
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, ["s1"], tmp_path)
    assert code == 1
    assert (tmp_path / "s1" / "sparse_pc.ply").exists()


def test_keep_local_skips_deletion(driver, monkeypatch, tmp_path):
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, ["s1"], tmp_path, keep_local=True)
    assert code == 0
    assert (tmp_path / "s1" / "sparse_pc.ply").exists()


def test_empty_bucket_exits_two_without_fetching(driver, monkeypatch, tmp_path):
    """--all against an empty bucket is a usage-level no-op, not a scene failure."""
    src = _FakeSource(scenes=())
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == 2
    assert [c[0] for c in src.calls] == ["list_scenes"]


def test_every_source_call_gets_a_progress_callback(driver, monkeypatch, tmp_path):
    """fetch/push/verify all stream rclone output through on_line; silence is a regression."""
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)
    _run(driver, src, ["s1"], tmp_path)
    assert set(src.on_lines) == {"fetch", "push", "verify"}
    assert all(callable(cb) for cb in src.on_lines.values()), src.on_lines


########
# Batch continuation: one bad scene must not truncate the run
########


def test_reconstruction_failure_does_not_abort_the_batch(driver, monkeypatch, tmp_path):
    src = _FakeSource(scenes=("s1", "s2"))
    _patch_run_scene(driver, monkeypatch, fail_on=("s1",))
    code = _run(driver, src, None, tmp_path)
    assert code == 1
    assert ("push", "s1") not in src.calls
    assert ("push", "s2") in src.calls


def test_verify_failure_does_not_abort_the_batch(driver, monkeypatch, tmp_path):
    """A flaky rclone check on scene 1 must not silently skip every later scene."""
    src = _FakeSource(scenes=("s1", "s2"), verify=False)
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == 1
    assert [c for c in src.calls if c[0] == "verify"] == [("verify", "s1"), ("verify", "s2")]


def test_delete_failure_does_not_abort_the_batch(driver, monkeypatch, tmp_path):
    """A scene dir that survives deletion is a FAIL for that scene only, not a batch stop."""
    src = _FakeSource(scenes=("s1", "s2"))
    _patch_run_scene(driver, monkeypatch)
    # rmtree silently swallows errors, so simulate the swallowed failure with a no-op
    monkeypatch.setattr(driver.shutil, "rmtree", lambda *a, **k: None)
    code = _run(driver, src, None, tmp_path)
    assert code == 1
    assert [c for c in src.calls if c[0] == "verify"] == [("verify", "s1"), ("verify", "s2")]
    assert (tmp_path / "s2" / "sparse_pc.ply").exists()


########
# Mid-batch transport fault: abort loudly instead of failing every remaining scene
########


def test_transport_fault_aborts_the_batch(driver, monkeypatch, tmp_path):
    """A dead rclone must stop the run. Exception type cannot distinguish this from a bad scene — a
    dead remote surfaces as a bare RuntimeError out of _run_streaming, same as a pipeline error —
    so the driver re-probes liveness, and only an unreachable remote aborts."""
    src = _FakeSource(scenes=("s1", "s2", "s3"), available=False, fail_fetch=("s1",))
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == driver.EXIT_REMOTE_UNAVAILABLE
    # s2 and s3 were never touched at all — not fetched, not reconstructed.
    assert [c for c in src.calls if c[0] == "fetch"] == [("fetch", "s1")]
    assert not [c for c in src.calls if c[0] == "push"]


def test_transport_fault_during_push_aborts_before_the_next_reconstruction(driver, monkeypatch, tmp_path):
    """The dominant mid-batch case: the reconstruction succeeded and the push is what died."""

    class _PushDies(_FakeSource):
        def push_outputs(self, local_dir, scene, on_line=None):
            self.calls.append(("push", scene))
            raise RuntimeError("rclone push failed (exit 1): 403 forbidden")

    src = _PushDies(scenes=("s1", "s2", "s3"), available=False)
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == driver.EXIT_REMOTE_UNAVAILABLE
    assert [c for c in src.calls if c[0] == "push"] == [("push", "s1")]
    assert [c for c in src.calls if c[0] == "fetch"] == [("fetch", "s1")]


def test_scene_failure_with_a_healthy_remote_still_continues(driver, monkeypatch, tmp_path):
    """The probe succeeding is what separates the two cases: a bad video or an OOM costs one scene."""
    src = _FakeSource(scenes=("s1", "s2"), available=True)
    _patch_run_scene(driver, monkeypatch, fail_on=("s1",))
    code = _run(driver, src, None, tmp_path)
    assert code == driver.EXIT_SCENE_FAILED
    assert ("push", "s2") in src.calls
    assert ("check_available",) in src.calls, "a failed scene must consult the probe"


def test_healthy_batch_never_probes_liveness(driver, monkeypatch, tmp_path):
    """The probe is a failure-path question; a clean run must not spend a request on it."""
    src = _FakeSource(scenes=("s1", "s2"))
    _patch_run_scene(driver, monkeypatch)
    assert _run(driver, src, None, tmp_path) == driver.EXIT_OK
    assert ("check_available",) not in src.calls


def test_aborted_batch_reports_which_scenes_never_ran(driver, monkeypatch, tmp_path, caplog):
    """The summary must still print, and un-attempted scenes must not be blamed as failures — they
    are safe to retry, and an operator reading 20 FAILs would go looking for 20 causes."""
    src = _FakeSource(scenes=("s1", "s2", "s3"), available=False, fail_fetch=("s1",))
    _patch_run_scene(driver, monkeypatch)
    with caplog.at_level(logging.INFO):
        _run(driver, src, None, tmp_path)
    messages = [r.getMessage() for r in caplog.records]
    assert any("Summary" in m for m in messages), messages
    # s2/s3 are reported as skipped, and distinctly from s1 which actually failed.
    skipped = [m for m in messages if "SKIPPED" in m]
    assert any("s2" in m for m in skipped) and any("s3" in m for m in skipped), messages
    assert not any("SKIPPED" in m and "s1" in m for m in messages)
    # And the abort itself is loud, naming rclone as the cause.
    aborts = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("rclone" in m.lower() and "abort" in m.lower() for m in aborts), aborts


def test_verify_failure_with_a_dead_remote_aborts_the_batch(driver, monkeypatch, tmp_path):
    """A transport fault can arrive on the verify path too, and there it never looks like one.

    verify_push swallows RuntimeError/OSError internally and returns a bare False, so credentials
    expiring *during verify* reach the driver indistinguishable from a content mismatch. Without a
    probe on that path the row reads "push verification failed" (wrong cause), the run exits 1 —
    which configs/README.md maps to "a data problem" — and one more scene is marched into the fault.
    """
    src = _FakeSource(scenes=("s1", "s2", "s3"), verify=False, available=False)
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == driver.EXIT_REMOTE_UNAVAILABLE
    assert ("check_available",) in src.calls, "a verify failure must consult the probe"
    assert [c for c in src.calls if c[0] == "verify"] == [("verify", "s1")]
    assert [c for c in src.calls if c[0] == "fetch"] == [("fetch", "s1")]


def test_delete_failure_with_a_dead_remote_aborts_the_batch(driver, monkeypatch, tmp_path):
    """The cleanup path is the third failure exit and shares the same fault mode."""
    src = _FakeSource(scenes=("s1", "s2"), available=False)
    _patch_run_scene(driver, monkeypatch)
    # rmtree swallows errors, so a surviving dir is how a failed delete actually presents
    monkeypatch.setattr(driver.shutil, "rmtree", lambda *a, **k: None)
    code = _run(driver, src, None, tmp_path)
    assert code == driver.EXIT_REMOTE_UNAVAILABLE
    assert ("check_available",) in src.calls, "a delete failure must consult the probe"
    assert [c for c in src.calls if c[0] == "fetch"] == [("fetch", "s1")]


def test_a_dead_remote_at_batch_start_exits_three_not_one(driver, monkeypatch, tmp_path):
    """list_scenes raises by design on any rclone failure — uncaught it exited 1 with a traceback.

    That made EXIT_REMOTE_UNAVAILABLE unreachable at batch start: an unattended runner facing an
    unconfigured remote was told "some scenes failed" instead of "infrastructure is down".
    """

    class _ListDies(_FakeSource):
        def list_scenes(self):
            self.calls.append(("list_scenes",))
            raise RuntimeError("rclone lsjson collab-data:environments-curated failed (exit 3)")

    src = _ListDies()
    _patch_run_scene(driver, monkeypatch)
    code = _run(driver, src, None, tmp_path)
    assert code == driver.EXIT_REMOTE_UNAVAILABLE
    assert [c[0] for c in src.calls] == ["list_scenes"]


def test_abort_exit_code_is_distinct_from_a_scene_failure(driver):
    """cron needs to tell "infrastructure died" from "some scenes failed" without parsing the log."""
    codes = (driver.EXIT_OK, driver.EXIT_SCENE_FAILED, driver.EXIT_NOTHING_TO_DO, driver.EXIT_REMOTE_UNAVAILABLE)
    assert len(set(codes)) == len(codes), codes
    assert driver.EXIT_OK == 0 and driver.EXIT_REMOTE_UNAVAILABLE != 0


########
# Cleanup failure reporting
########


def test_failed_delete_is_not_reported_as_success(driver, monkeypatch, tmp_path, caplog):
    """A verified push plus a surviving local dir is a FAIL: the disk was not reclaimed."""
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)
    monkeypatch.setattr(driver.shutil, "rmtree", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        code = _run(driver, src, ["s1"], tmp_path)
    assert code == 1
    scene_dir = tmp_path / "s1"
    assert scene_dir.exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(scene_dir) in msg for msg in warnings), warnings


def test_rmtree_errors_are_swallowed_and_reported_as_a_delete_failure(driver, monkeypatch, tmp_path, caplog):
    """ignore_errors=True is load-bearing: an unremovable dir is a scene FAIL, not a raised OSError."""
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)

    # Stand-in for real rmtree semantics: it raises unless the caller opted out of errors
    def _rmtree(path, ignore_errors=False, **kwargs):
        if not ignore_errors:
            raise OSError("permission denied")

    monkeypatch.setattr(driver.shutil, "rmtree", _rmtree)
    with caplog.at_level(logging.INFO):
        code = _run(driver, src, ["s1"], tmp_path)
    assert code == 1
    # The summary must name the delete failure, not surface a bare OSError message
    messages = [r.getMessage() for r in caplog.records]
    assert any("local dir remains" in msg for msg in messages), messages
    assert not any("permission denied" in msg for msg in messages), messages


def test_the_deleted_dir_is_the_one_that_was_verified(driver, monkeypatch, tmp_path):
    """The delete must target the path the verify gate actually passed on, not a re-derived twin.

    `output_root / scene` and run_scene's returned `out` are equal only because scene_output_dir
    happens to build them the same way. For a destructive delete that coincidence is not a contract.
    """
    removed = []
    verified = []

    class _RecordingSource(_FakeSource):
        def verify_push(self, local_dir, scene, on_line=None):
            verified.append(Path(local_dir))
            return super().verify_push(local_dir, scene, on_line=on_line)

    # run_scene owns the output layout: hand back a dir that is deliberately NOT output_root / scene
    def _run_scene(video, output_root, config_dir, override_config, stages, overwrite, name=None):
        out = Path(output_root) / name / "backend-out"
        out.mkdir(parents=True, exist_ok=True)
        return out, None

    monkeypatch.setattr(driver.batch, "run_scene", _run_scene)
    monkeypatch.setattr(driver.shutil, "rmtree", lambda p, **k: removed.append(Path(p)))
    _run(driver, _RecordingSource(), ["s1"], tmp_path)
    assert removed == verified == [tmp_path / "s1" / "backend-out"]


def test_reconstruction_failure_warns_which_local_dir_was_retained(driver, monkeypatch, tmp_path, caplog):
    """Partial state is only actionable if the operator is told where it is."""
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch, fail_on=("s1",))
    with caplog.at_level(logging.WARNING):
        code = _run(driver, src, ["s1"], tmp_path)
    assert code == 1
    scene_dir = tmp_path / "s1"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(scene_dir) in msg for msg in warnings), warnings


########
# The fetched video lives in the pushed dir, so PUSH_EXCLUDES is what keeps it local
########


@pytest.mark.parametrize(
    "video_ext",
    [".mp4", ".MP4", ".Mp4", ".mov", ".MOV", ".Mov", ".avi", ".AVI", ".Avi"],
)
def test_fetched_video_lands_in_the_pushed_dir_and_is_push_excluded(driver, monkeypatch, tmp_path, video_ext):
    src = _FakeSource(video_ext=video_ext)
    _patch_run_scene(driver, monkeypatch)
    _run(driver, src, ["s1"], tmp_path, keep_local=True)
    # fetch_video wrote <output_root>/s1/s1<ext>, and run_scene returned that same dir as `out`
    video = tmp_path / "s1" / f"s1{video_ext}"
    assert video.exists()
    # fnmatchcase stands in for rclone's case-sensitive glob over the scene-relative name
    assert any(fnmatch.fnmatchcase(video.name, p) for p in PUSH_EXCLUDES), video.name
    # A real output with a similar name must still be pushed
    assert not any(fnmatch.fnmatchcase("sparse_pc.ply", p) for p in PUSH_EXCLUDES)


########
# main(): argparse wiring and exit code
########


def _patch_main(driver, monkeypatch, argv, code=0):
    """Drive main() with a fake argv, a stubbed SceneSource and a run_remote spy."""
    recorded = {}
    monkeypatch.setattr("sys.argv", argv)
    # Never construct a real SceneSource: that would build an rclone client
    monkeypatch.setattr(driver, "SceneSource", lambda *a, **k: object())

    def _run_remote(**kwargs):
        recorded.update(kwargs)
        return code

    monkeypatch.setattr(driver, "run_remote", _run_remote)
    return recorded


def test_main_forwards_named_scenes_verbatim(driver, monkeypatch, tmp_path):
    """A named-scene invocation must never widen to the whole bucket."""
    recorded = _patch_main(
        driver,
        monkeypatch,
        [
            "run_pipeline_remote.py",
            "--output-root",
            str(tmp_path),
            "--stages",
            "preproc,mesh",
            "2026_07_20-birds-C0043",
            "2026_07_21-rats-C0100",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        driver.main()
    assert exc.value.code == 0
    assert recorded["scenes"] == ["2026_07_20-birds-C0043", "2026_07_21-rats-C0100"]
    assert recorded["stages"] == ["preproc", "mesh"]
    assert recorded["overwrite"] is False
    assert recorded["keep_local"] is False


def test_main_all_flag_means_scenes_none(driver, monkeypatch, tmp_path):
    recorded = _patch_main(driver, monkeypatch, ["run_pipeline_remote.py", "--output-root", str(tmp_path), "--all"])
    with pytest.raises(SystemExit) as exc:
        driver.main()
    assert exc.value.code == 0
    assert recorded["scenes"] is None


def test_main_requires_scenes_or_all(driver, monkeypatch, tmp_path):
    recorded = _patch_main(driver, monkeypatch, ["run_pipeline_remote.py", "--output-root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        driver.main()
    # argparse usage errors exit 2, and run_remote must never have been reached
    assert exc.value.code == 2
    assert recorded == {}


@pytest.mark.parametrize("code", [0, 1])
def test_main_propagates_the_run_remote_exit_code(driver, monkeypatch, tmp_path, code):
    """A failed batch must exit non-zero or cron/CI reports green on a broken run."""
    argv = ["run_pipeline_remote.py", "--output-root", str(tmp_path), "2026_07_20-birds-C0043"]
    _patch_main(driver, monkeypatch, argv, code=code)
    with pytest.raises(SystemExit) as exc:
        driver.main()
    assert exc.value.code == code


########
# CLI scene ids are joined straight onto output_root — validate them like --all's are
########


@pytest.mark.parametrize("bad", ["../x", "sceneA", "2026_07_20", "/abs/path"])
def test_main_rejects_a_scene_id_that_is_not_a_curated_dir_name(driver, monkeypatch, tmp_path, bad):
    """Explicit ids skipped the SCENE_ID_RE filter that --all discovery gets, so "../x" escaped
    output_root on a path join."""
    recorded = _patch_main(driver, monkeypatch, ["run_pipeline_remote.py", "--output-root", str(tmp_path), bad])
    with pytest.raises(SystemExit) as exc:
        driver.main()
    # argparse usage errors exit 2, and run_remote must never have been reached
    assert exc.value.code == 2
    assert recorded == {}


def test_main_accepts_every_id_that_discovery_would_yield(driver, monkeypatch, tmp_path):
    """The CLI validator and list_scenes must share one regex, or --all could surface an id that an
    explicit re-run of the same scene then refuses."""
    scene = "2026_07_20-birds-C0043"
    assert SCENE_ID_RE.match(scene), "fixture must be an id discovery would yield"
    recorded = _patch_main(driver, monkeypatch, ["run_pipeline_remote.py", "--output-root", str(tmp_path), scene])
    with pytest.raises(SystemExit) as exc:
        driver.main()
    assert exc.value.code == 0
    assert recorded["scenes"] == [scene]
