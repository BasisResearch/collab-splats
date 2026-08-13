from pathlib import Path

from collab_splats.dashboard.config import RunConfig


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.sampling_method == "fps"
    assert cfg.fps == 1.0
    assert cfg.max_frames == 100
    assert cfg.env_model == "vggt_omega"
    assert cfg.conf_threshold == 50.0  # vggt_omega native default (matches notebook)
    assert cfg.semantic_extractor == "talk2dino"
    assert cfg.query_positive == ""
    assert cfg.query_negative == "background, sky"
    assert cfg.mesh_voxel_size == 0.005
    assert cfg.mesh_sdf_trunc == 0.02
    assert cfg.mesh_depth_trunc == 1.0
    assert cfg.mesh_clean_repair is False


def test_runconfig_yaml_roundtrip(tmp_path: Path):
    cfg = RunConfig(env_model="mapanything", conf_threshold=35.0, query_positive="chair")
    cfg.frame_indices = [0, 5, 10]
    path = tmp_path / "run_config.yaml"
    cfg.to_yaml(path, video_ref="2026_05_07-birds-clip_03/clip_03.mp4")
    loaded = RunConfig.from_yaml(path)
    assert loaded.env_model == "mapanything"
    assert loaded.conf_threshold == 35.0
    assert loaded.query_positive == "chair"
    assert loaded.frame_indices == [0, 5, 10]
    assert loaded.video_ref == "2026_05_07-birds-clip_03/clip_03.mp4"
