import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Load run_scenes.py by path (docs/examples is not a package)
_MODULE_PATH = Path(__file__).parent.parent.parent / "docs" / "examples" / "run_scenes.py"
_spec = importlib.util.spec_from_file_location("run_scenes", _MODULE_PATH)
run_scenes = importlib.util.module_from_spec(_spec)
sys.modules["run_scenes"] = run_scenes
_spec.loader.exec_module(run_scenes)


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


def test_build_scene_config_maps_paths(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    video = Path("/data/birds/C0043.MP4")
    config = run_scenes.build_scene_config(video, "/out", cfg_dir)
    assert config["input_path"] == "/data/birds/C0043.MP4"
    assert config["output_path"] == "/out/C0043"
    assert config["pointcloud"]["backend"] == "vggt_omega"  # base preserved


def test_build_scene_config_merges_override(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    override = {"localization": {"enabled": True}}
    config = run_scenes.build_scene_config(Path("/data/x.mp4"), "/out", cfg_dir, override)
    assert config["localization"]["enabled"] is True
    assert config["localization"]["extractor"] == "loma"  # base preserved


def test_run_scenes_runs_pipeline_per_video(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    v1.touch()
    v2.touch()
    fake = MagicMock()
    fake.config = {"output_path": str(tmp_path / "out" / "a")}
    with patch.object(run_scenes, "Reconstructor", return_value=fake) as R:
        code = run_scenes.run_all(
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


def test_run_scenes_continues_on_failure(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    v1.touch()
    v2.touch()
    fake = MagicMock()
    fake.config = {"output_path": str(tmp_path / "out" / "x")}
    fake.run_pipeline.side_effect = [RuntimeError("boom"), None]
    with patch.object(run_scenes, "Reconstructor", return_value=fake):
        code = run_scenes.run_all(
            [v1, v2],
            output_root=tmp_path / "out",
            config_dir=cfg_dir,
            override_config=None,
            stages=None,
            overwrite=False,
        )
    assert fake.run_pipeline.call_count == 2  # did not abort after first failure
    assert code == 1  # non-zero because one failed
