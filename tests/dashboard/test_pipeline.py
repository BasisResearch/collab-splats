# tests/dashboard/test_pipeline.py
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np

from collab_splats.dashboard import pipeline as pl
from collab_splats.dashboard.config import RunConfig


def _fake_frames(n=3):
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(n)]
    records = [{"frame_idx": i, "blur_score": 200.0} for i in range(n)]
    return frames, records


class _InlineThread:
    """Stand-in for threading.Thread that runs the target synchronously on start()."""

    def __init__(self, target=None, daemon=None, **kw):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


def test_run_pipeline_orders_steps_and_pushes(tmp_path):
    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig(env_model="vggt_omega", semantic_extractor="talk2dino")
    video = tmp_path / "clip_03.mp4"
    video.write_bytes(b"x")

    fake_result = MagicMock()
    fake_result.points = list(range(10))  # len() used in the pointcloud step log line
    creator = MagicMock()
    # Pipeline decomposes run() into load_model→setup_inference→run_inference→postprocess;
    # result comes from creator.outputs after postprocess.
    creator.outputs = fake_result

    with (
        patch.object(pl, "sample_frames", return_value=_fake_frames()),
        patch.object(pl, "_write_frames_zarr") as wz,
        patch.object(pl, "_write_frames_jpegs", return_value=tmp_path / "frames"),
        patch.object(pl, "_build_creator", return_value=creator),
        patch.object(pl, "pointcloud_to_mesh") as mesh,
        patch.object(pl, "_extract_semantics") as sem,
        patch.object(pl, "_lift_and_compress") as liftc,
        patch.object(pl.threading, "Thread", _InlineThread),
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
    creator.load_model.assert_called_once()
    creator.setup_inference.assert_called_once()
    creator.run_inference.assert_called_once()
    creator.postprocess.assert_called_once()
    fake_result.save_zarr.assert_called_once()
    mesh.assert_called_once()
    liftc.assert_called_once()
    sem.assert_called_once()
    cfg_path = out / "run_config.yaml"
    assert cfg_path.exists()
    loaded = RunConfig.from_yaml(cfg_path)
    assert loaded.frame_indices == [0, 1, 2]
    source.push_outputs.assert_called_with(out, "2026_05_07", "clip_03", on_line=ANY)
    op_log.finish_op.assert_called_once()


def test_run_pipeline_does_not_push_on_failure(tmp_path):
    import pytest

    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    with (
        patch.object(pl, "sample_frames", return_value=_fake_frames()),
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


def test_transfer_mesh_features_writes_npy(tmp_path):
    import numpy as np
    import open3d as o3d

    from collab_splats.dashboard import pipeline

    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh_tsdf.ply"), mesh)

    sem_dir = tmp_path / "semantics"
    sem_dir.mkdir()
    np.save(sem_dir / "lifted_normed.npy", np.eye(3, 2, dtype=np.float32))

    class _Result:
        points = verts.copy()

    pipeline._transfer_mesh_features(_Result(), tmp_path, k=1, sdf_trunc=0.5)

    assert (mesh_dir / "vertex_features.npy").exists()
    out = np.load(mesh_dir / "vertex_features.npy")
    assert out.shape == (3, 2)
