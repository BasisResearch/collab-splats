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
    assert cfg.mesh_depth_trunc == 1.0


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


def test_runconfig_from_yaml_ignores_retired_keys(tmp_path: Path):
    """A run_config.yaml written before mesh_sdf_trunc/mesh_clean_repair were retired must still load.

    These files are permanent per-scene provenance — _stamp_db_provenance reads one for every
    already-processed scene, so a TypeError on an unknown key breaks the whole back catalogue.
    """
    path = tmp_path / "run_config.yaml"
    path.write_text("env_model: mapanything\nmesh_sdf_trunc: 0.02\nmesh_clean_repair: false\n")
    loaded = RunConfig.from_yaml(path)
    assert loaded.env_model == "mapanything"
    assert not hasattr(loaded, "mesh_sdf_trunc")
