"""Tests for the run_pipeline.py example entrypoint."""

import importlib.util
from pathlib import Path

# Load the example script by file path (it lives under docs/, not an importable package)
_spec = importlib.util.spec_from_file_location(
    "run_pipeline",
    Path(__file__).parent.parent.parent / "docs" / "examples" / "run_pipeline.py",
)
run_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pipeline)


def test_build_scene_config_sets_paths_only(tmp_path):
    """build_scene_config sets input/output paths and defers all defaults to Reconstructor."""
    video = tmp_path / "scene.MP4"
    video.touch()
    config_dir = Path(__file__).parent.parent.parent / "configs"
    cfg = run_pipeline.build_scene_config(video, tmp_path / "out", config_dir)
    assert cfg["input_path"] == str(video)
    assert cfg["output_path"].endswith("scene")
    # No manual base merge here anymore: only the paths this function sets are present
    assert set(cfg) == {"input_path", "output_path"}


def test_build_scene_config_carries_overrides(tmp_path):
    """Shared --config overrides pass through; base defaults still come from Reconstructor."""
    video = tmp_path / "scene.MP4"
    video.touch()
    config_dir = Path(__file__).parent.parent.parent / "configs"
    override = {"pointcloud": {"backend": "mapanything"}}
    cfg = run_pipeline.build_scene_config(video, tmp_path / "out", config_dir, override)
    assert cfg["pointcloud"]["backend"] == "mapanything"
    assert cfg["input_path"] == str(video)
