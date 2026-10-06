"""
reconstruct local / remote argument handling and batch flow with a stubbed Reconstructor.
"""

import argparse
import copy
import fnmatch
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from collab_splats import __main__ as cli
from collab_splats.reconstructor import Reconstructor
from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE

BASE_YAML = Path(__file__).parents[2] / "configs" / "base.yaml"
SCENE = "2026_07_20-birds-C0043"

PULLED_CONFIG = {
    "input_path": "/on/another/machine/C0043.MP4",
    "output_path": "/on/another/machine/out",
    "preproc": {"frame_selection": "uniform", "max_frames": 250},
    "pointcloud": {"method": "feedforward", "backend": "vggtx"},
    "mesh": {"enabled": True, "voxel_size": 0.02},
    "localization": {"enabled": True, "matcher": "xfeat"},
}


class StubRecon:
    """
    Records the config and run args; fails when the input file name starts with "fail".
    """

    made = []

    def __init__(self, config, base_config=None):
        self.config = {**config, "pointcloud": {"backend": "vggt_omega", **config.get("pointcloud", {})}}
        self.base_config = base_config
        self.viewer = None
        StubRecon.made.append(self)

    run_config_path = staticmethod(Reconstructor.run_config_path)

    def run(self, stages=None, overwrite=False):
        self.ran = (stages, overwrite)
        name = Path(self.config["input_path"]).name
        if name.startswith("fail"):
            raise RuntimeError("boom")


class FakeSource:
    """
    SceneSource stand-in: records calls and writes the files a real transfer would land.
    """

    def __init__(
        self,
        scenes=("s1",),
        verify=True,
        video_ext=".mp4",
        available=True,
        fail_fetch=(),
        processed=True,
        run_config=PULLED_CONFIG,
    ):
        self.scenes = list(scenes)
        self.verify = verify
        self.video_ext = video_ext
        self.available = available
        self.fail_fetch = set(fail_fetch)
        self.processed = processed
        self.run_config = run_config
        self.calls = []
        self.on_lines = {}

    def list_scenes(self):
        self.calls.append(("list_scenes",))
        return list(self.scenes)

    def list_processed_scenes(self):
        self.calls.append(("list_processed_scenes",))
        return [SCENE]

    def check_available(self):
        self.calls.append(("check_available",))
        return self.available

    def fetch_video(self, scene, dest_dir, on_line=None):
        self.calls.append(("fetch", scene))
        self.on_lines["fetch"] = on_line
        if scene in self.fail_fetch:
            raise RuntimeError("rclone copyto failed (exit 1): 401 unauthorized")

        dest_dir.mkdir(parents=True, exist_ok=True)
        video = dest_dir / f"{scene}{self.video_ext}"
        video.write_bytes(b"video")
        return video

    def has_processed(self, scene):
        self.calls.append(("has_processed", scene))
        return self.processed

    def pull_processed(self, scene, dest_dir, on_line=None):
        self.calls.append(("pull", scene))
        dest_dir.mkdir(parents=True, exist_ok=True)
        if self.run_config is not None:
            run_cfg = Reconstructor.run_config_path(dest_dir, self.run_config["pointcloud"]["backend"])
            run_cfg.parent.mkdir(parents=True, exist_ok=True)
            run_cfg.write_text(yaml.dump(self.run_config))

        return dest_dir

    def push_outputs(self, local_dir, scene, on_line=None):
        self.calls.append(("push", scene))
        self.on_lines["push"] = on_line

    def verify_push(self, local_dir, scene):
        self.calls.append(("verify", scene))
        return self.verify


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    StubRecon.made = []
    monkeypatch.setattr(cli, "Reconstructor", StubRecon)


def _args(tmp_path, stages=None, keep_local=False, overrides=None):
    """
    Namespace carrying what main() hands the remote runner.
    """
    return argparse.Namespace(
        output_root=tmp_path,
        stages=stages,
        overwrite=False,
        keep_local=keep_local,
        base_config=None,
        overrides=overrides or {},
    )


def _ops(source, op):
    """
    The calls of one kind a fake source recorded.
    """
    return [c for c in source.calls if c[0] == op]


########################################
# local
########################################


def test_local_writes_each_input_to_output_root_by_stem(tmp_path):
    video = tmp_path / "C0043.MP4"
    video.touch()
    code = cli.main(["local", str(video), "--output-root", str(tmp_path / "out")])
    assert code == cli.EXIT_OK
    assert StubRecon.made[0].config["output_path"] == str(tmp_path / "out" / "C0043")
    assert (tmp_path / "out" / "C0043" / "vggt_omega" / "run_config.yaml").exists()


def test_local_frame_directory_writes_to_its_folder_name(tmp_path):
    frames = tmp_path / "scene.v2"
    frames.mkdir()
    cli.main(["local", str(frames), "--output-root", str(tmp_path / "out")])
    assert StubRecon.made[0].config["output_path"] == str(tmp_path / "out" / "scene.v2")


def test_local_refuses_two_inputs_sharing_a_stem(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.mp4").touch()
    (tmp_path / "b" / "x.mov").touch()
    argv = ["local", str(tmp_path / "a" / "x.mp4"), str(tmp_path / "b" / "x.mov"), "--output-root", str(tmp_path)]
    with pytest.raises(SystemExit):
        cli.main(argv)

    assert StubRecon.made == []


def test_local_continues_past_a_failing_input(tmp_path):
    good, bad = tmp_path / "good.mp4", tmp_path / "fail.mp4"
    good.touch()
    bad.touch()
    code = cli.main(["local", str(bad), str(good), "--output-root", str(tmp_path / "o")])
    assert code == cli.EXIT_SCENE_FAILED
    assert len(StubRecon.made) == 2


def test_set_overrides_parse_yaml_values_on_dotted_keys(tmp_path):
    video = tmp_path / "v.mp4"
    video.touch()
    sets = ["--set", "mesh.voxel_size=0.02", "--set", "semantics.enabled=false"]
    cli.main(["local", str(video), "--output-root", str(tmp_path), *sets])
    config = StubRecon.made[0].config
    assert config["mesh"]["voxel_size"] == 0.02
    assert config["semantics"]["enabled"] is False


def test_set_without_equals_is_a_usage_error(tmp_path, capsys):
    # argparse exit 2 with the message on stderr, never a traceback
    video = tmp_path / "v.mp4"
    video.touch()
    with pytest.raises(SystemExit) as exc:
        cli.main(["local", str(video), "--output-root", str(tmp_path), "--set", "mesh.voxel_size"])

    assert exc.value.code == 2
    assert "key=value" in capsys.readouterr().err
    assert StubRecon.made == []


def test_set_under_a_scalar_is_a_usage_error(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        cli.main(["local", str(video), "--output-root", str(tmp_path), "--set", "a=1", "--set", "a.b=2"])

    assert exc.value.code == 2
    assert "a.b" in capsys.readouterr().err
    assert StubRecon.made == []


def test_missing_config_file_is_a_usage_error(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    missing = tmp_path / "nope.yaml"
    with pytest.raises(SystemExit) as exc:
        cli.main(["local", str(video), "--output-root", str(tmp_path), "--config", str(missing)])

    assert exc.value.code == 2
    assert str(missing) in capsys.readouterr().err
    assert StubRecon.made == []


def test_local_refuses_missing_inputs_before_any_run(tmp_path, capsys):
    video = tmp_path / "v.mp4"
    video.touch()
    missing = tmp_path / "nope.mp4"
    with pytest.raises(SystemExit) as exc:
        cli.main(["local", str(video), str(missing), "--output-root", str(tmp_path)])

    assert exc.value.code == 2
    assert str(missing) in capsys.readouterr().err
    assert StubRecon.made == []


def test_overrides_are_not_mutated_across_runs(tmp_path):
    # Every scene merges into a fresh dict; the shared overrides stay as parsed
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for video in videos:
        video.touch()

    overrides = {"mesh": {"voxel_size": 0.02}, "pointcloud": {"method": "feedforward"}}
    before = copy.deepcopy(overrides)
    args = argparse.Namespace(
        inputs=videos,
        output_root=tmp_path / "out",
        stages=None,
        overwrite=False,
        keep_viewer=False,
        base_config=None,
        overrides=overrides,
    )
    cli._run_local(args)
    cli._run_local(args)
    assert overrides == before

    # The remote path merges the same overrides into each scene
    remote_args = _args(tmp_path / "remote", overrides=overrides)
    cli._run_remote(FakeSource(scenes=("s1", "s2")), [], remote_args)
    cli._run_remote(FakeSource(scenes=("s1", "s2")), [], remote_args)
    assert overrides == before


def test_config_file_merges_under_set(tmp_path):
    video = tmp_path / "v.mp4"
    video.touch()
    override = tmp_path / "o.yaml"
    override.write_text(yaml.safe_dump({"mesh": {"voxel_size": 0.05, "texture": True}}))
    argv = ["local", str(video), "--output-root", str(tmp_path), "--config", str(override)]
    cli.main([*argv, "--set", "mesh.voxel_size=0.02"])
    assert StubRecon.made[0].config["mesh"] == {"voxel_size": 0.02, "texture": True}


def test_stages_are_split_and_passed(tmp_path):
    video = tmp_path / "v.mp4"
    video.touch()
    cli.main(["local", str(video), "--output-root", str(tmp_path), "--stages", "preproc, pointcloud", "--overwrite"])
    assert StubRecon.made[0].ran == (["preproc", "pointcloud"], True)


def test_base_config_reaches_the_reconstructor(tmp_path):
    video = tmp_path / "v.mp4"
    video.touch()
    base = tmp_path / "base.yaml"
    cli.main(["local", str(video), "--output-root", str(tmp_path), "--base-config", str(base)])
    assert StubRecon.made[0].base_config == base


def test_run_config_is_rewritten_with_the_config_that_ran(tmp_path):
    # A pulled scene ships a stale run_config.yaml; it must be replaced, not kept
    video = tmp_path / "v.mp4"
    video.touch()
    out = tmp_path / "out" / "v" / "vggt_omega"
    out.mkdir(parents=True)
    (out / "run_config.yaml").write_text(yaml.dump({"mesh": {"voxel_size": 0.99}}))

    cli.main(["local", str(video), "--output-root", str(tmp_path / "out"), "--set", "mesh.voxel_size=0.777"])
    recorded = yaml.safe_load((out / "run_config.yaml").read_text())
    assert recorded == StubRecon.made[0].config
    assert recorded["mesh"]["voxel_size"] == 0.777


def test_keep_viewer_serves_the_last_viewer(tmp_path, monkeypatch):
    served = []

    class _Viewer:
        def serve_forever(self):
            served.append(True)

    # Only the last reconstructor's viewer is kept alive
    def _run_scene(config, args):
        recon = StubRecon(config)
        recon.viewer = _Viewer()
        return recon

    monkeypatch.setattr(cli, "_run_scene", _run_scene)
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for video in videos:
        video.touch()

    cli.main(["local", *map(str, videos), "--output-root", str(tmp_path), "--keep-viewer"])
    assert served == [True]


def test_without_keep_viewer_a_viewer_never_blocks(tmp_path, monkeypatch):
    class _Viewer:
        def serve_forever(self):
            raise AssertionError("serve_forever blocks forever")

    def _run_scene(config, args):
        recon = StubRecon(config)
        recon.viewer = _Viewer()
        return recon

    monkeypatch.setattr(cli, "_run_scene", _run_scene)
    video = tmp_path / "a.mp4"
    video.touch()
    assert cli.main(["local", str(video), "--output-root", str(tmp_path)]) == cli.EXIT_OK


def test_keep_viewer_without_a_viewer_returns(tmp_path):
    video = tmp_path / "fail.mp4"
    video.touch()
    code = cli.main(["local", str(video), "--output-root", str(tmp_path), "--keep-viewer"])
    assert code == cli.EXIT_SCENE_FAILED


def test_keep_viewer_on_a_successful_run_without_a_viewer_returns_ok(tmp_path):
    # StubRecon.viewer is None: the run succeeds and nothing blocks
    video = tmp_path / "v.mp4"
    video.touch()
    code = cli.main(["local", str(video), "--output-root", str(tmp_path), "--keep-viewer"])
    assert code == cli.EXIT_OK
    assert StubRecon.made[0].viewer is None


########################################
# remote: argument handling
########################################


@pytest.mark.parametrize("bad", ["../escape", "/abs/path", ".hidden", "a/b"])
def test_remote_rejects_unsafe_scene_ids(tmp_path, monkeypatch, bad):
    # Never build a real SceneSource: that constructs an rclone client
    monkeypatch.setattr(cli, "SceneSource", FakeSource)
    with pytest.raises(SystemExit) as exc:
        cli.main(["remote", bad, "--output-root", str(tmp_path)])

    assert exc.value.code == 2


def test_remote_needs_ids_or_all(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SceneSource", FakeSource)
    with pytest.raises(SystemExit) as exc:
        cli.main(["remote", "--output-root", str(tmp_path)])

    assert exc.value.code == 2


def test_remote_unknown_stage_is_a_usage_error_before_any_fetch(tmp_path, monkeypatch, capsys):
    sources = []
    monkeypatch.setattr(cli, "SceneSource", lambda: sources.append(FakeSource()) or sources[-1])
    with pytest.raises(SystemExit) as exc:
        cli.main(["remote", "--all", "--stages", "mesh,msh", "--output-root", str(tmp_path)])

    assert exc.value.code == 2
    assert "msh" in capsys.readouterr().err
    assert [c for s in sources for c in s.calls] == []
    assert StubRecon.made == []


def test_remote_scenes_with_all_is_a_usage_error(tmp_path, monkeypatch, capsys):
    sources = []
    monkeypatch.setattr(cli, "SceneSource", lambda: sources.append(FakeSource()) or sources[-1])
    with pytest.raises(SystemExit) as exc:
        cli.main(["remote", SCENE, "--all", "--output-root", str(tmp_path)])

    assert exc.value.code == 2
    assert "not both" in capsys.readouterr().err
    assert sources == []


def test_remote_forwards_named_scenes_verbatim(tmp_path, monkeypatch):
    # A named-scene run never widens to the whole bucket; discovery's ids pass the CLI check
    recorded = {}
    monkeypatch.setattr(cli, "SceneSource", FakeSource)
    monkeypatch.setattr(cli, "_run_remote", lambda source, scenes, args: recorded.update(scenes=scenes) or 1)
    assert SCENE_ID_RE.match(SCENE)

    code = cli.main(["remote", SCENE, "2026_07_21-rats-C0100", "--output-root", str(tmp_path)])
    assert code == 1
    assert recorded["scenes"] == [SCENE, "2026_07_21-rats-C0100"]


def test_remote_all_runs_the_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SceneSource", lambda: FakeSource(scenes=("s1", "s2")))
    assert cli.main(["remote", "--all", "--output-root", str(tmp_path)]) == cli.EXIT_OK
    assert len(StubRecon.made) == 2


def test_exit_codes_are_distinct():
    codes = (cli.EXIT_OK, cli.EXIT_SCENE_FAILED, cli.EXIT_NOTHING_TO_DO, cli.EXIT_REMOTE_UNAVAILABLE)
    assert len(set(codes)) == len(codes)
    assert cli.EXIT_OK == 0


########################################
# remote: fetch, run, push, verify, delete
########################################


def test_happy_path_order_is_fetch_run_push_verify_delete(tmp_path):
    source = FakeSource()
    assert cli._run_remote(source, ["s1"], _args(tmp_path)) == cli.EXIT_OK
    assert [c[0] for c in source.calls] == ["fetch", "push", "verify"]
    assert StubRecon.made[0].config["input_path"] == str(tmp_path / "s1" / "s1.mp4")
    assert not (tmp_path / "s1").exists()


def test_every_transfer_gets_a_progress_callback(tmp_path):
    source = FakeSource()
    cli._run_remote(source, ["s1"], _args(tmp_path))
    assert set(source.on_lines) == {"fetch", "push"}
    assert all(callable(cb) for cb in source.on_lines.values())


def test_all_processes_every_listed_scene(tmp_path):
    source = FakeSource(scenes=("s1", "s2"))
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_OK
    assert _ops(source, "push") == [("push", "s1"), ("push", "s2")]


def test_failed_verify_keeps_local_data(tmp_path):
    source = FakeSource(verify=False)
    assert cli._run_remote(source, ["s1"], _args(tmp_path)) == cli.EXIT_SCENE_FAILED
    assert (tmp_path / "s1" / "vggt_omega" / "run_config.yaml").exists()


def test_keep_local_skips_deletion(tmp_path):
    source = FakeSource()
    assert cli._run_remote(source, ["s1"], _args(tmp_path, keep_local=True)) == cli.EXIT_OK
    assert (tmp_path / "s1" / "vggt_omega" / "run_config.yaml").exists()


def test_empty_bucket_is_nothing_to_do(tmp_path):
    source = FakeSource(scenes=())
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_NOTHING_TO_DO
    assert [c[0] for c in source.calls] == ["list_scenes"]


def test_a_dead_remote_at_batch_start_is_remote_unavailable(tmp_path):
    class _ListDies(FakeSource):
        def list_scenes(self):
            self.calls.append(("list_scenes",))
            raise RuntimeError("rclone lsjson failed (exit 3)")

    source = _ListDies()
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_REMOTE_UNAVAILABLE
    assert [c[0] for c in source.calls] == ["list_scenes"]


def test_leaf_rerun_lists_the_processed_bucket(tmp_path):
    source = FakeSource()
    cli._run_remote(source, [], _args(tmp_path, stages=["mesh"]))
    assert ("list_processed_scenes",) in source.calls
    assert ("list_scenes",) not in source.calls


def test_full_run_lists_the_curated_bucket(tmp_path):
    source = FakeSource()
    cli._run_remote(source, [], _args(tmp_path, stages=["preproc", "pointcloud"]))
    assert ("list_scenes",) in source.calls
    assert ("list_processed_scenes",) not in source.calls


########################################
# remote: failure isolation and abort
########################################


def test_reconstruction_failure_does_not_abort_the_batch(tmp_path):
    source = FakeSource(scenes=("fail-s1", "s2"))
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_SCENE_FAILED
    assert _ops(source, "push") == [("push", "s2")]
    assert ("check_available",) in source.calls


def test_verify_failure_does_not_abort_the_batch(tmp_path):
    source = FakeSource(scenes=("s1", "s2"), verify=False)
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_SCENE_FAILED
    assert _ops(source, "verify") == [("verify", "s1"), ("verify", "s2")]


def test_delete_failure_does_not_abort_the_batch(tmp_path, monkeypatch):
    # rmtree swallows errors, so a no-op stands in for a failed delete
    source = FakeSource(scenes=("s1", "s2"))
    monkeypatch.setattr(cli.shutil, "rmtree", lambda *a, **k: None)
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_SCENE_FAILED
    assert _ops(source, "verify") == [("verify", "s1"), ("verify", "s2")]
    assert (tmp_path / "s2" / "vggt_omega" / "run_config.yaml").exists()


def test_healthy_batch_never_probes_liveness(tmp_path):
    source = FakeSource(scenes=("s1", "s2"))
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_OK
    assert ("check_available",) not in source.calls


def test_transport_fault_aborts_the_batch(tmp_path):
    source = FakeSource(scenes=("s1", "s2", "s3"), available=False, fail_fetch=("s1",))
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_REMOTE_UNAVAILABLE
    assert _ops(source, "fetch") == [("fetch", "s1")]
    assert _ops(source, "push") == []


def test_transport_fault_during_push_aborts_before_the_next_scene(tmp_path):
    class _PushDies(FakeSource):
        def push_outputs(self, local_dir, scene, on_line=None):
            self.calls.append(("push", scene))
            raise RuntimeError("rclone push failed (exit 1): 403 forbidden")

    source = _PushDies(scenes=("s1", "s2", "s3"), available=False)
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_REMOTE_UNAVAILABLE
    assert _ops(source, "push") == [("push", "s1")]
    assert _ops(source, "fetch") == [("fetch", "s1")]


def test_verify_failure_with_a_dead_remote_aborts_the_batch(tmp_path):
    # verify_push swallows transport faults into False, so the probe must run on that path too
    source = FakeSource(scenes=("s1", "s2", "s3"), verify=False, available=False)
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_REMOTE_UNAVAILABLE
    assert _ops(source, "verify") == [("verify", "s1")]


def test_delete_failure_with_a_dead_remote_aborts_the_batch(tmp_path, monkeypatch):
    source = FakeSource(scenes=("s1", "s2"), available=False)
    monkeypatch.setattr(cli.shutil, "rmtree", lambda *a, **k: None)
    assert cli._run_remote(source, [], _args(tmp_path)) == cli.EXIT_REMOTE_UNAVAILABLE
    assert _ops(source, "fetch") == [("fetch", "s1")]


def test_aborted_batch_reports_skipped_scenes(tmp_path, caplog):
    # Un-attempted scenes are SKIPPED, never FAIL: they are safe to retry
    source = FakeSource(scenes=("s1", "s2", "s3"), available=False, fail_fetch=("s1",))
    with caplog.at_level(logging.INFO):
        cli._run_remote(source, [], _args(tmp_path))

    messages = [r.getMessage() for r in caplog.records]
    skipped = [m for m in messages if "SKIPPED" in m]
    assert any("Summary" in m for m in messages)
    assert any("s2" in m for m in skipped) and any("s3" in m for m in skipped)
    assert not any("s1" in m for m in skipped)

    # The abort itself is loud and names rclone
    errors = [r.getMessage().lower() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("rclone" in m and "abort" in m for m in errors)


def test_failed_delete_is_not_reported_as_success(tmp_path, monkeypatch, caplog):
    source = FakeSource()
    monkeypatch.setattr(cli.shutil, "rmtree", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        assert cli._run_remote(source, ["s1"], _args(tmp_path)) == cli.EXIT_SCENE_FAILED

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(tmp_path / "s1") in m for m in warnings)


def test_rmtree_errors_are_swallowed_and_reported_as_a_delete_failure(tmp_path, monkeypatch, caplog):
    # ignore_errors=True is load-bearing: an unremovable dir is a FAIL row, not a raised OSError
    def _rmtree(path, ignore_errors=False, **kwargs):
        if not ignore_errors:
            raise OSError("permission denied")

    monkeypatch.setattr(cli.shutil, "rmtree", _rmtree)
    with caplog.at_level(logging.INFO):
        assert cli._run_remote(FakeSource(), ["s1"], _args(tmp_path)) == cli.EXIT_SCENE_FAILED

    messages = [r.getMessage() for r in caplog.records]
    assert any("local dir remains" in m for m in messages)
    assert not any("permission denied" in m for m in messages)


def test_the_deleted_dir_is_the_one_that_was_verified(tmp_path, monkeypatch):
    verified = []
    removed = []

    class _RecordingSource(FakeSource):
        def verify_push(self, local_dir, scene):
            verified.append(local_dir)
            return super().verify_push(local_dir, scene)

    monkeypatch.setattr(cli.shutil, "rmtree", lambda p, **k: removed.append(p))
    cli._run_remote(_RecordingSource(), ["s1"], _args(tmp_path))
    assert removed == verified == [tmp_path / "s1"]


def test_reconstruction_failure_names_the_retained_dir(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert cli._run_remote(FakeSource(), ["fail-s1"], _args(tmp_path)) == cli.EXIT_SCENE_FAILED

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(tmp_path / "fail-s1") in m for m in warnings)


@pytest.mark.parametrize("video_ext", [".mp4", ".MP4", ".Mp4", ".mov", ".MOV", ".Mov", ".avi", ".AVI", ".Avi"])
def test_fetched_video_lands_in_the_pushed_dir_and_is_push_excluded(tmp_path, video_ext):
    cli._run_remote(FakeSource(video_ext=video_ext), ["s1"], _args(tmp_path, keep_local=True))
    video = tmp_path / "s1" / f"s1{video_ext}"
    assert video.exists()

    # fnmatchcase stands in for rclone's case-sensitive glob; a real output must still push
    assert any(fnmatch.fnmatchcase(video.name, p) for p in PUSH_EXCLUDES)
    assert not any(fnmatch.fnmatchcase("sparse_pc.ply", p) for p in PUSH_EXCLUDES)


########################################
# remote: leaf re-run from processed outputs
########################################


def test_leaf_stages_pull_from_processed(tmp_path):
    source = FakeSource()
    config = cli._prepare_scene(source, SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))
    assert ("pull", SCENE) in source.calls
    assert _ops(source, "fetch") == []

    # Stages not being re-run keep the pulled provenance; paths point at this run
    assert config["pointcloud"]["backend"] == "vggtx"
    assert config["preproc"]["max_frames"] == 250
    assert config["output_path"] == str(tmp_path / SCENE)


@pytest.mark.parametrize("stages", [None, ["pointcloud", "mesh"]])
def test_any_upstream_stage_fetches_the_curated_video(tmp_path, stages):
    source = FakeSource()
    overrides = {"mesh": {"voxel_size": 0.001}}
    config = cli._prepare_scene(source, SCENE, tmp_path / SCENE, _args(tmp_path, stages=stages, overrides=overrides))
    assert config["input_path"] == str(tmp_path / SCENE / f"{SCENE}.mp4")
    assert config["mesh"] == {"voxel_size": 0.001}
    assert "pointcloud" not in config
    assert _ops(source, "pull") == []


def test_unprocessed_scene_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no processed outputs"):
        cli._prepare_scene(FakeSource(processed=False), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))


def test_pull_without_run_config_raises(tmp_path):
    with pytest.raises(ValueError, match="recorded backends \\[\\]"):
        cli._prepare_scene(FakeSource(run_config=None), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))


def test_two_recorded_backends_need_an_explicit_pick(tmp_path):
    # A scene compared across backends holds one record per backend
    other = Reconstructor.run_config_path(tmp_path / SCENE, "colmap")
    other.parent.mkdir(parents=True)
    other.write_text(yaml.dump({"pointcloud": {"backend": "colmap"}}))

    with pytest.raises(ValueError, match="colmap.*vggtx"):
        cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))


def test_explicit_backend_picks_its_own_record(tmp_path):
    other = Reconstructor.run_config_path(tmp_path / SCENE, "colmap")
    other.parent.mkdir(parents=True)
    other.write_text(yaml.dump({"pointcloud": {"backend": "colmap"}, "preproc": {"max_frames": 9}}))

    args = _args(tmp_path, stages=["mesh"], overrides={"pointcloud": {"backend": "colmap"}})
    config = cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, args)
    assert config["pointcloud"]["backend"] == "colmap"
    assert config["preproc"]["max_frames"] == 9


def test_explicit_backend_without_a_record_raises(tmp_path):
    args = _args(tmp_path, stages=["mesh"], overrides={"pointcloud": {"backend": "vggt_omega"}})
    with pytest.raises(FileNotFoundError, match="vggt_omega"):
        cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, args)


def test_empty_run_config_file_reads_as_empty(tmp_path):
    # A 0-byte file loads as None; it must read as an empty record, not crash on None
    class _EmptyRunConfig(FakeSource):
        def pull_processed(self, scene, dest_dir, on_line=None):
            self.calls.append(("pull", scene))
            run_cfg = Reconstructor.run_config_path(dest_dir, "vggtx")
            run_cfg.parent.mkdir(parents=True, exist_ok=True)
            run_cfg.write_bytes(b"")
            return dest_dir

    config = cli._prepare_scene(_EmptyRunConfig(), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))
    assert config["output_path"] == str(tmp_path / SCENE)


def test_matching_backend_is_accepted(tmp_path):
    args = _args(tmp_path, stages=["mesh"], overrides={"pointcloud": {"backend": "vggtx"}})
    config = cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, args)
    assert config["pointcloud"]["backend"] == "vggtx"


def test_override_wins_over_pulled(tmp_path):
    args = _args(tmp_path, stages=["mesh"], overrides={"mesh": {"voxel_size": 0.001}})
    config = cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, args)
    assert config["mesh"] == {"voxel_size": 0.001}


def test_localize_drops_the_localization_section(tmp_path):
    config = cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["localize"]))
    assert "localization" not in config
    assert config["mesh"]["voxel_size"] == 0.02


def test_dropped_section_is_refilled_by_base_yaml_end_to_end(tmp_path):
    # The real Reconstructor must refill the dropped mesh section from base.yaml
    config = cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))
    rec = Reconstructor(config)

    base_mesh = yaml.safe_load(BASE_YAML.read_text())["mesh"]
    assert rec.config["mesh"] == base_mesh
    assert rec.config["mesh"]["voxel_size"] != PULLED_CONFIG["mesh"]["voxel_size"]

    # Stages not being re-run keep the pulled scene's provenance, not base.yaml's
    assert rec.config["pointcloud"]["backend"] == "vggtx"
    assert rec.config["preproc"]["max_frames"] == 250


def test_rerun_plan_is_logged(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        cli._prepare_scene(FakeSource(), SCENE, tmp_path / SCENE, _args(tmp_path, stages=["mesh"]))

    assert "mesh" in caplog.text and "vggtx" in caplog.text


########################################
# Entry point
########################################


def test_local_help_exits_zero():
    # The console script's module entry point parses and prints usage without running anything
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "collab_splats", "local", "--help"], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--output-root" in result.stdout
