# tests/dashboard/test_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from collab_splats.dashboard import pipeline as pl
from collab_splats.dashboard.config import RunConfig


def _fake_frames(n=3):
    return [np.zeros((4, 4, 3), np.uint8) for _ in range(n)], [0, 1, 2]


def test_run_pipeline_orders_steps_and_pushes(tmp_path):
    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig(env_model="vggt_omega", semantic_extractor="talk2dino")
    video = tmp_path / "clip_03.mp4"
    video.write_bytes(b"x")

    fake_result = MagicMock()
    creator = MagicMock()
    creator.outputs = fake_result

    with (
        patch.object(pl, "sample_frames_fps", return_value=_fake_frames()),
        patch.object(pl, "get_video_info", return_value={"duration_s": 3.0, "fps": 30}),
        patch.object(pl, "_write_frames_zarr") as wz,
        patch.object(pl, "_write_frames_jpegs", return_value=tmp_path / "frames"),
        patch.object(pl, "_build_creator", return_value=creator),
        patch.object(pl, "pointcloud_to_mesh") as mesh,
        patch.object(pl, "_extract_semantics") as sem,
    ):
        out = pl.run_pipeline(
            video_path=video,
            session="2026_05_07",
            stem="clip_03",
            config=cfg,
            op_log=op_log,
            source=source,
            base_dir=tmp_path / "outputs",
        )

    assert out == tmp_path / "outputs" / "2026_05_07" / "clip_03"
    wz.assert_called_once()
    creator.reconstruct.assert_called_once()
    fake_result.save_zarr.assert_called_once()
    mesh.assert_called_once()
    sem.assert_called_once()
    cfg_path = out / "run_config.yaml"
    assert cfg_path.exists()
    loaded = RunConfig.from_yaml(cfg_path)
    assert loaded.frame_indices == [0, 1, 2]
    source.push_outputs.assert_called_with(out, "2026_05_07", "clip_03")
    op_log.finish_op.assert_called_once()


def test_run_pipeline_does_not_push_on_failure(tmp_path):
    import pytest

    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    with (
        patch.object(pl, "sample_frames_fps", return_value=_fake_frames()),
        patch.object(pl, "get_video_info", return_value={"duration_s": 3.0, "fps": 30}),
        patch.object(pl, "_write_frames_zarr"),
        patch.object(pl, "_write_frames_jpegs", return_value=tmp_path / "frames"),
        patch.object(pl, "_build_creator", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError):
            pl.run_pipeline(
                video_path=video,
                session="s",
                stem="clip",
                config=cfg,
                op_log=op_log,
                source=source,
                base_dir=tmp_path / "o",
            )

    source.push_outputs.assert_not_called()
    op_log.error_op.assert_called_once()


def test_build_creator_maps_models():
    with patch.object(pl, "VGGTOmegaCreator") as omega:
        pl._build_creator("vggt_omega", 50.0)
        omega.assert_called_with(conf_threshold=50.0)
    with patch.object(pl, "VGGTXCreator") as vx:
        pl._build_creator("vggtx", 35.0)
        vx.assert_called_with(conf_threshold=35.0)
    with patch.object(pl, "MapAnythingCreator") as ma:
        pl._build_creator("mapanything", 35.0)
        ma.assert_called_with(confidence_percentile=35.0)
