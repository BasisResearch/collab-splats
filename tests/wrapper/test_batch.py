"""batch.py: output-dir derivation, config build, failure isolation."""

from pathlib import Path

import pytest
import yaml

from collab_splats.wrapper import batch

########
# Input discovery
########


def test_collect_videos_expands_directories(tmp_path):
    d = tmp_path / "clips"
    d.mkdir()
    (d / "a.mp4").touch()
    (d / "b.MOV").touch()
    (d / "notes.txt").touch()
    assert batch.collect_videos([d]) == [d / "a.mp4", d / "b.MOV"]


def test_collect_videos_passes_files_through(tmp_path):
    f = tmp_path / "x.mp4"
    f.touch()
    assert batch.collect_videos([f]) == [f]


def test_collect_videos_is_sorted_deterministically(tmp_path):
    """iterdir yields in scandir order — creation order here is reverse-sorted on purpose."""
    d = tmp_path / "clips"
    d.mkdir()
    for name in ("z_last.mp4", "m_mid.MOV", "a_first.avi"):
        (d / name).touch()
    assert batch.collect_videos([d]) == [d / "a_first.avi", d / "m_mid.MOV", d / "z_last.mp4"]


def test_collect_videos_warns_on_empty_directory(tmp_path, caplog):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "notes.txt").touch()
    with caplog.at_level("WARNING"):
        assert batch.collect_videos([d]) == []
    assert "No videos" in caplog.text


########
# Output-dir derivation
########


def test_scene_output_dir_uses_date_parent(tmp_path):
    video = tmp_path / "2026-07-20" / "C0043.MP4"
    assert batch.scene_output_dir(video, tmp_path / "out") == tmp_path / "out" / "2026_07_20" / "C0043"


def test_scene_output_dir_falls_back_to_stem(tmp_path):
    assert batch.scene_output_dir(tmp_path / "C0043.MP4", tmp_path / "out") == tmp_path / "out" / "C0043"


def test_scene_output_dir_accepts_explicit_name(tmp_path):
    """Remote scenes name their own output dir — the curated dir name IS the scene id."""
    video = tmp_path / "2026-07-20" / "C0043.MP4"
    out = batch.scene_output_dir(video, tmp_path / "out", name="2026_07_20-birds-C0043")
    assert out == tmp_path / "out" / "2026_07_20-birds-C0043"


def test_scene_output_dir_ignores_underscore_scene_ids(tmp_path):
    """_DATE_RE matches local hyphen date dirs only; underscore scene ids fall back to stem."""
    video = tmp_path / "2026_05_07-birds-clip_03" / "clip_03.MP4"
    assert batch.scene_output_dir(video, tmp_path / "out") == tmp_path / "out" / "clip_03"


def test_scene_output_dir_finds_date_dir_above_direct_parent(tmp_path):
    """Real layouts nest the video below the session dir, so the whole parents chain is walked."""
    video = tmp_path / "fieldwork" / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    assert batch.scene_output_dir(video, tmp_path / "out") == tmp_path / "out" / "2024_02_06" / "C0043"


def test_scene_output_dir_accepts_string_output_root():
    """Drivers read output_root from YAML/CLI, i.e. as a str — it must be coerced to Path."""
    assert batch.scene_output_dir("/data/2026-07-20/C0043.MP4", "/out") == Path("/out/2026_07_20/C0043")


def test_scene_output_dir_requires_full_date_dir_name(tmp_path):
    """A dir merely starting with a date (2026-07-20-birds) is not a session dir."""
    video = tmp_path / "2026-07-20-birds" / "C0043.MP4"
    assert batch.scene_output_dir(video, tmp_path / "out") == tmp_path / "out" / "C0043"


def test_scene_output_dir_picks_nearest_date_dir(tmp_path):
    """Two date-like dirs in one path: the one nearest the video wins (parents order)."""
    video = tmp_path / "2025-01-01" / "sub" / "2026-07-20" / "C1.MP4"
    assert batch.scene_output_dir(video, tmp_path / "out") == tmp_path / "out" / "2026_07_20" / "C1"


########
# Per-scene config
########


def test_build_scene_config_sets_paths(tmp_path):
    """Called without config_dir — that param is gone from the signature."""
    video = tmp_path / "2026-07-20" / "C0043.MP4"
    config = batch.build_scene_config(video, tmp_path / "out")
    assert config["input_path"] == str(video)
    assert config["output_path"] == str(tmp_path / "out" / "2026_07_20" / "C0043")


def test_build_scene_config_sets_paths_only(tmp_path):
    """No manual base merge here: only the paths this function sets are present."""
    config = batch.build_scene_config(tmp_path / "scene.MP4", tmp_path / "out")
    assert set(config) == {"input_path", "output_path"}


def test_build_scene_config_honours_name(tmp_path):
    video = tmp_path / "2026-07-20" / "C0043.MP4"
    config = batch.build_scene_config(video, tmp_path / "out", name="2026_07_20-birds-C0043")
    assert config["output_path"] == str(tmp_path / "out" / "2026_07_20-birds-C0043")


def test_build_scene_config_deep_merges_override(tmp_path):
    """A shallow update would drop sibling nested keys — mergedeep must preserve them."""
    override = {"pointcloud": {"method": "vggt_omega", "viz": {"enabled": True}}, "semantics": {"enabled": True}}
    config = batch.build_scene_config(tmp_path / "C0043.MP4", tmp_path / "out", override_config=override)
    assert config["pointcloud"] == {"method": "vggt_omega", "viz": {"enabled": True}}
    assert config["semantics"] == {"enabled": True}


def test_build_scene_config_deep_copies_nested_override(tmp_path):
    """run_all reuses one override dict for every scene, so nested dicts must be copied, not
    aliased — a shallow update satisfies value equality but leaks scene 1's writes into 2..N."""
    override = {"pointcloud": {"viz": {"enabled": True}}}
    config = batch.build_scene_config(tmp_path / "C0043.MP4", tmp_path / "out", override_config=override)
    assert config["pointcloud"] is not override["pointcloud"]
    assert config["pointcloud"]["viz"] is not override["pointcloud"]["viz"]


def test_build_scene_config_does_not_mutate_override(tmp_path):
    override = {"pointcloud": {"method": "vggt_omega"}}
    batch.build_scene_config(tmp_path / "C0043.MP4", tmp_path / "out", override_config=override)
    assert "input_path" not in override


########
# Batch driver
########


def _patch_run_scene(monkeypatch, fail_on=()):
    """Replace run_scene with a recorder that never builds a real Reconstructor."""
    calls = []

    def _run_scene(video, output_root, config_dir, override_config, stages, overwrite, name=None):
        calls.append((Path(video).name, name))
        if Path(video).name in fail_on:
            raise RuntimeError("boom")
        return output_root / Path(video).stem, None

    monkeypatch.setattr(batch, "run_scene", _run_scene)
    return calls


def test_run_all_returns_zero_when_all_succeed(tmp_path, monkeypatch):
    calls = _patch_run_scene(monkeypatch)
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
    code = batch.run_all(videos, tmp_path / "out", None, None, None, False)
    assert code == 0
    assert [name for name, _ in calls] == ["a.mp4", "b.mp4", "c.mp4"]


def test_run_all_isolates_one_failure(tmp_path, monkeypatch):
    """A failing video must not abort the batch; exit code flips to 1."""
    calls = _patch_run_scene(monkeypatch, fail_on={"b.mp4"})
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
    code = batch.run_all(videos, tmp_path / "out", None, None, None, False)
    assert code == 1
    assert [name for name, _ in calls] == ["a.mp4", "b.mp4", "c.mp4"]


def test_run_all_passes_no_name(tmp_path, monkeypatch):
    """Local driver relies on date-parent derivation, so name stays unset."""
    calls = _patch_run_scene(monkeypatch)
    batch.run_all([tmp_path / "a.mp4"], tmp_path / "out", None, None, None, False)
    assert calls == [("a.mp4", None)]


def _patch_run_scene_with_viewer(monkeypatch):
    """Replace run_scene with one whose Reconstructor exposes a viewer recording serve_forever."""
    served = []

    class _Viewer:
        def serve_forever(self):
            served.append(True)

    class _Recon:
        viewer = _Viewer()

    def _run_scene(video, output_root, config_dir, override_config, stages, overwrite, name=None):
        return output_root / Path(video).stem, _Recon()

    monkeypatch.setattr(batch, "run_scene", _run_scene)
    return served


def test_run_all_keep_viewer_serves_last_reconstructor(tmp_path, monkeypatch):
    served = _patch_run_scene_with_viewer(monkeypatch)
    code = batch.run_all([tmp_path / "a.mp4"], tmp_path / "out", None, None, None, False, keep_viewer=True)
    assert code == 0
    assert served == [True]


def test_run_all_without_keep_viewer_does_not_block_on_viewer(tmp_path, monkeypatch):
    """Default batch runs must return even when a viewer exists — serve_forever blocks forever."""
    served = _patch_run_scene_with_viewer(monkeypatch)
    code = batch.run_all([tmp_path / "a.mp4"], tmp_path / "out", None, None, None, False)
    assert code == 0
    assert served == []


def test_run_all_keep_viewer_without_viewer_is_noop(tmp_path, monkeypatch):
    """Every video failed / viz disabled → keep_viewer logs and returns, never crashes."""
    _patch_run_scene(monkeypatch, fail_on={"a.mp4"})
    code = batch.run_all([tmp_path / "a.mp4"], tmp_path / "out", None, None, None, False, keep_viewer=True)
    assert code == 1


########
# Single-scene runner
########


def _patch_reconstructor(monkeypatch):
    """Replace Reconstructor with a fake recording config_dir + run_pipeline args."""
    ran = {}

    class _FakeReconstructor:
        def __init__(self, config, config_dir=None):
            self.config = dict(config)
            ran["config_dir"] = config_dir

        def run_pipeline(self, stages=None, overwrite=False):
            ran["stages"] = stages
            ran["overwrite"] = overwrite

    monkeypatch.setattr(batch, "Reconstructor", _FakeReconstructor)
    return ran, _FakeReconstructor


def test_run_scene_writes_run_config_and_returns_reconstructor(tmp_path, monkeypatch):
    """run_scene persists run_config.yaml then returns (output_path, Reconstructor)."""
    out_dir = tmp_path / "out" / "C0043"
    ran, _FakeReconstructor = _patch_reconstructor(monkeypatch)
    output_path, recon = batch.run_scene(
        tmp_path / "C0043.MP4", tmp_path / "out", None, None, ["preproc"], False, name="C0043"
    )

    assert output_path == out_dir
    assert isinstance(recon, _FakeReconstructor)
    assert ran == {"config_dir": batch.DEFAULT_CONFIG_DIR, "stages": ["preproc"], "overwrite": False}
    with open(out_dir / "run_config.yaml") as f:
        assert yaml.safe_load(f)["output_path"] == str(out_dir)


def test_run_scene_overwrite_governs_stale_run_config(tmp_path, monkeypatch):
    """overwrite decides whether an existing run_config.yaml is refreshed or left stale."""
    _patch_reconstructor(monkeypatch)
    out_dir = tmp_path / "out" / "C0043"
    out_dir.mkdir(parents=True)
    run_cfg = out_dir / "run_config.yaml"
    run_cfg.write_text("stale: true\n")

    # overwrite=False keeps whatever is already on disk
    batch.run_scene(tmp_path / "C0043.MP4", tmp_path / "out", None, None, None, False, name="C0043")
    assert yaml.safe_load(run_cfg.read_text()) == {"stale": True}

    # overwrite=True rewrites it with the config actually used for this run
    batch.run_scene(tmp_path / "C0043.MP4", tmp_path / "out", None, None, None, True, name="C0043")
    refreshed = yaml.safe_load(run_cfg.read_text())
    assert refreshed["output_path"] == str(out_dir)
    assert "stale" not in refreshed


def test_video_exts_is_public():
    assert batch.VIDEO_EXTS == {".mp4", ".mov", ".avi"}
