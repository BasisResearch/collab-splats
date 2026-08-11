"""prepare_scene/discover_scenes: leaf-only routing, failure modes, config merge."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from collab_splats.remote.rerun import discover_scenes, prepare_scene

SCENE = "2026_07_20-birds-C0043"

PULLED_CONFIG = {
    "input_path": "/on/another/machine/C0043.MP4",
    "output_path": "/on/another/machine/out",
    "preprocessing": {"frame_selection": "uniform", "max_frames": 250},
    "pointcloud": {"method": "feedforward", "backend": "vggtx"},
    "mesh": {"enabled": True, "voxel_size": 0.02},
}


class _FakeSource:
    """SceneSource stand-in: records calls, writes the files a real pull would land."""

    def __init__(self, processed=True, run_config=PULLED_CONFIG, processed_scenes=(SCENE,)):
        self._processed = processed
        self._run_config = run_config
        self._processed_scenes = list(processed_scenes)
        self.calls = []

    def has_processed(self, scene):
        self.calls.append(("has_processed", scene))
        return self._processed

    def pull_processed(self, scene, dest_dir, on_line=None):
        self.calls.append(("pull_processed", scene, Path(dest_dir)))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if self._run_config is not None:
            (dest / "run_config.yaml").write_text(yaml.dump(self._run_config))
        return dest

    def fetch_video(self, scene, dest_dir, on_line=None):
        self.calls.append(("fetch_video", scene, Path(dest_dir)))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        video = dest / "C0043.MP4"
        video.touch()
        return video

    def list_scenes(self):
        self.calls.append(("list_scenes",))
        return ["curated-only-scene", SCENE]

    def list_processed_scenes(self):
        self.calls.append(("list_processed_scenes",))
        return self._processed_scenes


########
# Routing
########


def test_leaf_stages_pull_from_processed(tmp_path):
    source = _FakeSource()
    video, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)
    assert video is None
    assert ("pull_processed", SCENE, tmp_path / SCENE) in source.calls
    assert not any(c[0] == "fetch_video" for c in source.calls)
    # Provenance for stages NOT being re-run survives verbatim.
    assert config["pointcloud"]["backend"] == "vggtx"
    assert config["preprocessing"]["max_frames"] == 250


def test_any_upstream_stage_fetches_the_curated_video(tmp_path):
    """pointcloud is not a leaf, so the whole scene is rebuilt from the video."""
    source = _FakeSource()
    video, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["pointcloud", "mesh"], None)
    assert video == tmp_path / SCENE / "C0043.MP4"
    assert config is None
    assert not any(c[0] == "pull_processed" for c in source.calls)


def test_stages_none_fetches_the_curated_video(tmp_path):
    source = _FakeSource()
    video, _ = prepare_scene(source, SCENE, tmp_path / SCENE, None, None)
    assert video == tmp_path / SCENE / "C0043.MP4"


def test_override_config_passes_through_on_the_curated_path(tmp_path):
    source = _FakeSource()
    override = {"mesh": {"voxel_size": 0.001}}
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, None, override)
    assert config is override


########
# Failure modes — each raises before any compute, landing as a FAIL row in the driver
########


def test_unprocessed_scene_raises(tmp_path):
    source = _FakeSource(processed=False)
    with pytest.raises(FileNotFoundError, match="no processed outputs"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)


def test_pull_without_run_config_raises(tmp_path):
    source = _FakeSource(run_config=None)
    with pytest.raises(FileNotFoundError, match="run_config.yaml"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)


def test_backend_mismatch_raises(tmp_path):
    """Config is authoritative and never silently retargeted: disagreeing with the data is fatal."""
    source = _FakeSource()
    override = {"pointcloud": {"backend": "vggt_omega"}}
    with pytest.raises(ValueError, match="vggtx"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], override)


def test_matching_backend_is_accepted(tmp_path):
    source = _FakeSource()
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], {"pointcloud": {"backend": "vggtx"}})
    assert config["pointcloud"]["backend"] == "vggtx"


########
# Config merge
########


def test_rerun_stage_section_is_dropped_so_base_yaml_supplies_it(tmp_path):
    source = _FakeSource()
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)
    assert "mesh" not in config  # Reconstructor merges base.yaml's mesh section over the gap


def test_override_config_wins_over_pulled(tmp_path):
    source = _FakeSource()
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], {"mesh": {"voxel_size": 0.001}})
    assert config["mesh"]["voxel_size"] == 0.001


def test_localize_drops_the_localization_section(tmp_path):
    """Stage name and config section differ only for localize."""
    source = _FakeSource(run_config={**PULLED_CONFIG, "localization": {"enabled": True, "extractor": "disk"}})
    _, config = prepare_scene(source, SCENE, tmp_path / SCENE, ["localize"], None)
    assert "localization" not in config
    assert config["mesh"]["voxel_size"] == 0.02  # untouched stage keeps its provenance


def test_plan_is_logged(tmp_path, caplog):
    source = _FakeSource()
    with caplog.at_level("INFO"):
        prepare_scene(source, SCENE, tmp_path / SCENE, ["mesh"], None)
    assert "mesh" in caplog.text and "vggtx" in caplog.text


########
# --all bucket choice
########


def test_discover_scenes_lists_processed_for_a_leaf_rerun():
    source = _FakeSource()
    assert discover_scenes(source, ["mesh"]) == [SCENE]
    assert ("list_processed_scenes",) in source.calls


def test_discover_scenes_lists_curated_otherwise():
    source = _FakeSource()
    assert discover_scenes(source, None) == ["curated-only-scene", SCENE]
    assert discover_scenes(source, ["preproc", "pointcloud"]) == ["curated-only-scene", SCENE]
    assert not any(c[0] == "list_processed_scenes" for c in source.calls)


def test_importing_the_remote_package_stays_light():
    """rerun imports the pipeline; dashboard/operation_log imports this package on the fast bind."""
    # Pins why discover_scenes/prepare_scene are absent from collab_splats.remote.__all__:
    # re-exporting them here drags torch into the dashboard's light import path.
    code = (
        "import sys; import collab_splats.remote; "
        "assert 'torch' not in sys.modules, 'collab_splats.remote import pulled torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
