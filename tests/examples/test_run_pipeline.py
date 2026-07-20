import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Load run_pipeline.py by path (docs/examples is not a package)
_MODULE_PATH = Path(__file__).parent.parent.parent / "docs" / "examples" / "run_pipeline.py"
_spec = importlib.util.spec_from_file_location("run_pipeline", _MODULE_PATH)
run_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["run_pipeline"] = run_pipeline
_spec.loader.exec_module(run_pipeline)


def _write_configs(tmp_path):
    """Create a minimal configs/ dir with base.yaml."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    base = {
        "input_path": None,
        "output_path": None,
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega"},
        "semantics": {"enabled": False},
        "localization": {"enabled": False, "extractor": "loma"},
    }
    (cfg_dir / "base.yaml").write_text(yaml.dump(base))
    return cfg_dir


########################################
# collect_videos — file + directory expansion
########################################


def test_collect_videos_passes_files_through(tmp_path):
    v1, v2 = tmp_path / "a.MP4", tmp_path / "b.mov"
    v1.touch()
    v2.touch()
    got = run_pipeline.collect_videos([v1, v2])
    assert [p.name for p in got] == ["a.MP4", "b.mov"]


def test_collect_videos_expands_directory(tmp_path):
    d = tmp_path / "clips"
    d.mkdir()
    (d / "c1.mp4").touch()
    (d / "c2.MOV").touch()
    (d / "notes.txt").touch()  # non-video ignored
    got = run_pipeline.collect_videos([d])
    assert sorted(p.name for p in got) == ["c1.mp4", "c2.MOV"]


def test_collect_videos_mixed_and_empty_dir(tmp_path):
    v = tmp_path / "solo.avi"
    v.touch()
    empty = tmp_path / "empty"
    empty.mkdir()
    got = run_pipeline.collect_videos([v, empty])
    assert [p.name for p in got] == ["solo.avi"]  # empty dir contributes nothing


########################################
# scene_output_dir — date-based layout matching live outputs/
########################################


def test_scene_output_dir_uses_parent_date(tmp_path):
    video = Path("/data/fieldwork/birds/2024-02-06/SplatsSD/C0043.MP4")
    out = run_pipeline.scene_output_dir(video, "/out")
    assert out == Path("/out/2024_02_06/C0043")


def test_scene_output_dir_falls_back_to_stem(tmp_path):
    video = Path("/data/misc/clip.mp4")  # no date dir in path
    out = run_pipeline.scene_output_dir(video, "/out")
    assert out == Path("/out/clip")


########################################
# build_scene_config — path wiring + override merge
########################################


def test_build_scene_config_maps_paths(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    video = Path("/data/birds/2024-02-06/SplatsSD/C0043.MP4")
    config = run_pipeline.build_scene_config(video, "/out", cfg_dir)
    assert config["input_path"] == "/data/birds/2024-02-06/SplatsSD/C0043.MP4"
    assert config["output_path"] == "/out/2024_02_06/C0043"
    assert config["pointcloud"]["backend"] == "vggt_omega"  # base preserved


def test_build_scene_config_merges_override(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    override = {"localization": {"enabled": True}}
    config = run_pipeline.build_scene_config(Path("/data/x.mp4"), "/out", cfg_dir, override)
    assert config["localization"]["enabled"] is True
    assert config["localization"]["extractor"] == "loma"  # base preserved


########################################
# run_all — per-video execution + failure isolation
########################################


def test_run_all_runs_pipeline_per_video(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    v1.touch()
    v2.touch()
    fake = MagicMock()
    fake.config = {"output_path": str(tmp_path / "out" / "a")}
    with patch.object(run_pipeline, "Reconstructor", return_value=fake) as R:
        code = run_pipeline.run_all(
            [v1, v2],
            output_root=tmp_path / "out",
            config_dir=cfg_dir,
            override_config=None,
            stages=None,
            overwrite=False,
        )
    assert R.call_count == 2
    assert fake.run_pipeline.call_count == 2
    assert code == 0


def test_run_all_continues_on_failure(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    v1.touch()
    v2.touch()
    fake = MagicMock()
    fake.config = {"output_path": str(tmp_path / "out" / "x")}
    fake.run_pipeline.side_effect = [RuntimeError("boom"), None]
    with patch.object(run_pipeline, "Reconstructor", return_value=fake):
        code = run_pipeline.run_all(
            [v1, v2],
            output_root=tmp_path / "out",
            config_dir=cfg_dir,
            override_config=None,
            stages=None,
            overwrite=False,
        )
    assert fake.run_pipeline.call_count == 2  # did not abort after first failure
    assert code == 1  # non-zero because one failed
