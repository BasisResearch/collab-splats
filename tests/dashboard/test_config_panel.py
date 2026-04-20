"""Tests for ConfigPanel YAML load/save/round-trip."""
from pathlib import Path
import yaml
import pytest
from collab_splats.dashboard.config_panel import ConfigPanel

CONFIGS_DIR = Path(__file__).parents[2] / "docs" / "splats" / "configs"


def test_config_panel_loads_base_defaults():
    panel = ConfigPanel(configs_dir=CONFIGS_DIR)
    panel.load_from_yaml(None)
    assert panel.method == "rade-features"
    assert panel.sfm_tool == "hloc"
    assert abs(panel.frame_proportion - 0.25) < 1e-9
    assert panel.min_frames == 100


def test_config_panel_loads_existing_yaml(tmp_path):
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    dataset_yaml = datasets_dir / "birds_date-02062024_video-C0043.yaml"
    dataset_yaml.write_text(
        "method: rade-gs\nframe_proportion: 0.10\nfile_path: /workspace/test.MP4\n"
    )
    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(dataset_yaml)
    assert panel.method == "rade-gs"
    assert abs(panel.frame_proportion - 0.10) < 1e-9
    assert panel.sfm_tool == "hloc"  # not overridden


def test_config_panel_save_writes_overrides_only(tmp_path):
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    (tmp_path / "datasets").mkdir()
    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(None)
    panel.method = "rade-gs"
    out = tmp_path / "datasets" / "test_dataset.yaml"
    panel.save_to_yaml(out)
    written = yaml.safe_load(out.read_text())
    assert written.get("method") == "rade-gs"
    assert "frame_proportion" not in written  # unchanged — omitted


def test_config_panel_round_trip(tmp_path):
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    (tmp_path / "datasets").mkdir()
    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(None)
    panel.method = "splatfacto"
    panel.sfm_tool = "colmap"
    panel.frame_proportion = 0.5
    panel.min_frames = 200
    out = tmp_path / "datasets" / "roundtrip.yaml"
    panel.save_to_yaml(out)
    panel2 = ConfigPanel(configs_dir=tmp_path)
    panel2.load_from_yaml(out)
    assert panel2.method == "splatfacto"
    assert panel2.sfm_tool == "colmap"
    assert abs(panel2.frame_proportion - 0.5) < 1e-9
    assert panel2.min_frames == 200


def test_config_panel_to_splatter_config(tmp_path):
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    (tmp_path / "datasets").mkdir()
    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(None)
    config = panel.to_splatter_config(
        file_path=tmp_path / "video.MP4",
        output_path=tmp_path / "output",
    )
    assert config["file_path"] == str(tmp_path / "video.MP4")
    assert config["method"] == "rade-features"
    assert config["input_type"] == "video"
    assert config["output_path"] == str(tmp_path / "output")
    assert abs(config["frame_proportion"] - 0.25) < 1e-9
    assert config["min_frames"] == 100
