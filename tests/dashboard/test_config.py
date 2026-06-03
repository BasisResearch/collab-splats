from pathlib import Path

from collab_splats.dashboard.config import RunConfig


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.sampling_method == "balanced"
    assert cfg.max_frames == 50
    assert cfg.env_model == "vggt_omega"
    assert cfg.conf_threshold == 50.0
    assert cfg.semantic_extractor == "talk2dino"
    assert cfg.query == ""
    assert cfg.mesh_voxel_size == 0.01


def test_runconfig_yaml_roundtrip(tmp_path: Path):
    cfg = RunConfig(env_model="mapanything", conf_threshold=35.0, query="chair")
    cfg.frame_indices = [0, 5, 10]
    path = tmp_path / "run_config.yaml"
    cfg.to_yaml(path, video_ref="reconstruction/2026_05_07/clip_03.mp4")
    loaded = RunConfig.from_yaml(path)
    assert loaded.env_model == "mapanything"
    assert loaded.conf_threshold == 35.0
    assert loaded.query == "chair"
    assert loaded.frame_indices == [0, 5, 10]
    assert loaded.video_ref == "reconstruction/2026_05_07/clip_03.mp4"
