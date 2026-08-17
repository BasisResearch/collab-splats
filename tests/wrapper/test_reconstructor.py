import importlib.util
import shutil
import subprocess
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pycolmap
import pytest
import torch
import yaml
from mergedeep import merge

from collab_splats.mesh.tsdf import Open3DTSDFFusion
from collab_splats.pointcloud.feedforward.base import (
    FeedforwardResult,
    build_pycolmap_reconstruction,
)
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor

# Import ConfigLoader directly from config.py to avoid wrapper/__init__.py
spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).parent.parent.parent / "collab_splats" / "wrapper" / "config.py"
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
ConfigLoader = config_module.ConfigLoader


def test_config_load_base_defaults(tmp_path):
    base = {
        "preprocessing": {"frame_selection": "fps", "fps": 1.0, "min_frames": 300},
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "bundle_adjustment": False, "loop_closure": False},
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 1024},
        "mesh": {"enabled": False, "voxel_size": 0.01, "sdf_trunc": 0.04, "depth_trunc": 1.0},
        "localization": {"enabled": False, "matcher": "loma"},
        "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
    }
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()
    ds = {"input_path": "/data/video.mp4", "output_path": "/data/out"}
    (tmp_path / "datasets" / "test.yaml").write_text(yaml.dump(ds))

    loader = ConfigLoader(tmp_path)
    config = loader.load("test")

    assert config["input_path"] == "/data/video.mp4"
    assert config["pointcloud"]["method"] == "feedforward"
    assert config["pointcloud"]["backend"] == "vggtx"
    assert config["semantics"]["enabled"] is False


def test_config_dataset_override_merges(tmp_path):
    base = {
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "bundle_adjustment": False},
        "semantics": {"enabled": False, "extractor": "dinov2"},
    }
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()
    ds = {
        "input_path": "/data/v.mp4",
        "output_path": "/out",
        "pointcloud": {"backend": "mapanything", "bundle_adjustment": True},
    }
    (tmp_path / "datasets" / "ds.yaml").write_text(yaml.dump(ds))

    loader = ConfigLoader(tmp_path)
    config = loader.load("ds")

    assert config["pointcloud"]["backend"] == "mapanything"
    assert config["pointcloud"]["bundle_adjustment"] is True
    assert config["pointcloud"]["method"] == "feedforward"  # base preserved


def test_config_runtime_overrides(tmp_path):
    base = {"pointcloud": {"method": "feedforward", "backend": "vggtx"}}
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "ds.yaml").write_text(yaml.dump({"input_path": "/v.mp4", "output_path": "/o"}))

    loader = ConfigLoader(tmp_path)
    config = loader.load("ds", overrides={"pointcloud": {"backend": "vggt_omega"}})

    assert config["pointcloud"]["backend"] == "vggt_omega"


def test_config_missing_dataset_raises(tmp_path):
    base = {"pointcloud": {"method": "feedforward"}}
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()

    loader = ConfigLoader(tmp_path)
    with pytest.raises(ValueError, match="Dataset config not found"):
        loader.load("nonexistent")


def _make_config(tmp_path, overrides=None):
    """Build minimal valid config dict for tests."""
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "preprocessing": {"frame_selection": "uniform", "fps": 1.0, "min_frames": 10},
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "bundle_adjustment": False,
            "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 512},
        "mesh": {"enabled": False, "voxel_size": 0.01, "sdf_trunc": 0.04, "depth_trunc": 1.0},
        "localization": {"enabled": False},
        "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
    }
    if overrides:
        config = merge({}, config, overrides)
    return config


def test_no_inline_defaults_in_source():
    """No cfg.get(key, default) with a VALUE default remains — base.yaml is the only source.

    Structural {}/[] defaults (e.g. validate_config's config.get("pointcloud", {}) on a raw
    partial config) are allowed.
    """
    import re
    from pathlib import Path

    src = Path("collab_splats/wrapper/reconstructor.py").read_text()
    # Capture the default expression of each two-arg .get("key", <default>)
    defaults = re.findall(r"\.get\(\s*['\"][^'\"]+['\"]\s*,\s*([^)]+)\)", src)
    # Structural {}/[] defaults (validate_config's standalone guards) are allowed; value defaults are not
    offenders = [d.strip() for d in defaults if d.strip() not in ("{}", "[]")]
    assert offenders == [], f"inline value defaults still present: {offenders}"


def test_extract_frames_dispatches_per_frame_selection(tmp_path, monkeypatch):
    """Each frame_selection value reaches its own sampler with only its own knobs."""
    from collab_splats.wrapper import reconstructor as R

    calls = {}

    def fake_sample_frames(path, **kwargs):
        calls.clear()
        calls.update(kwargs)
        return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

    monkeypatch.setattr(R, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})

    out = tmp_path / "out"
    video = tmp_path / "v.mp4"
    video.touch()

    # fps: rate + both band bounds
    R._extract_frames(video, out / "a.zarr", "fps", 2.0, 5, 50)
    assert calls == {"method": "fps", "fps": 2.0, "min_frames": 5, "max_frames": 50}

    # uniform: max_frames is the count; no fps, no floor
    R._extract_frames(video, out / "b.zarr", "uniform", None, 5, 50)
    assert calls == {"method": "uniform", "max_frames": 50}

    # optical_flow: max_frames caps the selector
    R._extract_frames(video, out / "c.zarr", "optical_flow", None, 5, 50)
    assert calls == {"method": "optical_flow", "max_frames": 50}


def test_extract_frames_rejects_unknown_selection(tmp_path, monkeypatch):
    from collab_splats.wrapper import reconstructor as R

    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    video = tmp_path / "v.mp4"
    video.touch()
    with pytest.raises(ValueError, match="frame_selection"):
        R._extract_frames(video, tmp_path / "out.zarr", "balanced", None, 5, 50)


def test_extract_frames_records_fps_in_provenance(tmp_path, monkeypatch):
    """frames.zarr must record the rate a scene was sampled at, not just the cap."""
    from collab_splats.wrapper import reconstructor as R

    monkeypatch.setattr(
        R,
        "sample_frames",
        lambda path, **kw: ([np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]),
    )
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})

    video = tmp_path / "v.mp4"
    video.touch()
    R._extract_frames(video, tmp_path / "out" / "frames.zarr", "fps", 2.0, None, 50)

    store = R.FrameStore.open(tmp_path / "out" / "frames.zarr")
    prov = dict(store._store.attrs["provenance"])
    assert prov["fps"] == 2.0
    assert prov["method"] == "fps"
    # fps must be a staleness key, or changing the rate silently reuses old frames
    assert store.is_stale({**prov, "fps": 4.0})


def test_from_config_file_removed():
    """The dead, dataset-based from_config_file constructor is gone."""
    assert not hasattr(Reconstructor, "from_config_file")


def test_reconstructor_init(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.config["pointcloud"]["backend"] == "vggtx"


def test_init_fills_defaults_from_base_yaml(tmp_path):
    """A partial config gets missing keys filled from configs/base.yaml."""
    partial = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
    }
    rec = Reconstructor(partial)
    # min_frames is null in base.yaml so fps is honoured literally, NOT a stale code default
    assert rec.config["preprocessing"]["min_frames"] is None
    # fps comes from base.yaml (1.0), and fps is the default selection method
    assert rec.config["preprocessing"]["fps"] == 1.0
    assert rec.config["preprocessing"]["frame_selection"] == "fps"
    # backend comes from base.yaml (vggt_omega)
    assert rec.config["pointcloud"]["backend"] == "vggt_omega"


def test_init_user_override_wins_over_base(tmp_path):
    """User-supplied value overrides the base.yaml default."""
    partial = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"backend": "mapanything"},
    }
    rec = Reconstructor(partial)
    assert rec.config["pointcloud"]["backend"] == "mapanything"
    # sibling keys still filled from base
    assert rec.config["pointcloud"]["method"] == "feedforward"


def test_reconstructor_validate_missing_input_path(tmp_path):
    config = _make_config(tmp_path)
    del config["input_path"]
    with pytest.raises(ValueError, match="input_path"):
        Reconstructor.validate_config(config)


def test_reconstructor_validate_missing_output_path(tmp_path):
    config = _make_config(tmp_path)
    del config["output_path"]
    with pytest.raises(ValueError, match="output_path"):
        Reconstructor.validate_config(config)


def test_reconstructor_validate_bad_backend(tmp_path):
    config = _make_config(tmp_path, {"pointcloud": {"method": "feedforward", "backend": "badmodel"}})
    with pytest.raises(ValueError, match="backend"):
        Reconstructor.validate_config(config)


def test_validate_config_ignores_mesher(tmp_path):
    """mesh.mesher is no longer a validated knob — its absence never raises."""
    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
    }
    # Reaches validate via __init__ (base-merged); must not raise on missing mesher
    rec = Reconstructor(config)
    assert "mesher" not in rec.config["mesh"]


def test_valid_meshers_symbol_removed():
    """_VALID_MESHERS constant is gone."""
    from collab_splats.wrapper import reconstructor as R

    assert not hasattr(R, "_VALID_MESHERS")


def test_reconstructor_backend_dir(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.backend_dir == tmp_path / "out" / "vggtx"


def test_reconstructor_no_images_dir(tmp_path):
    """images/ JPG dir is retired — frames.zarr is the sole persistent frame store."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert not hasattr(rec, "images_dir")
    assert rec.frames_zarr == tmp_path / "out" / "frames.zarr"


def test_reconstructor_semantics_cache_dir(tmp_path):
    """The 2D cache is scene-level: it depends on the frames only, not on the backend."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.semantics_cache_dir == tmp_path / "out" / "semantics"


def test_preprocess_skips_if_frames_zarr_exists(tmp_path):
    """Skip extraction when frames.zarr already exists and overwrite=False."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.frames_zarr.mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        result = rec.preprocess(overwrite=False)

    mock_extract.assert_not_called()
    assert result == rec.frames_zarr


def test_preprocess_runs_if_frames_zarr_missing(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        mock_extract.return_value = 1
        result = rec.preprocess(overwrite=False)

    mock_extract.assert_called_once()
    assert result == rec.frames_zarr


def test_preprocess_overwrite_reruns(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.frames_zarr.mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        mock_extract.return_value = 1
        rec.preprocess(overwrite=True)

    mock_extract.assert_called_once()


def _make_mock_pointcloud_result(tmp_path):
    """Minimal PointcloudResult mock for testing — avoids importing collab_splats.pointcloud."""
    result = MagicMock()
    result.reconstruction = pycolmap.Reconstruction()  # empty, no images
    result.image_paths = [tmp_path / "images" / "frame_0001.jpg"]
    # Match the real PointcloudResult.points/.colors shape for an empty reconstruction
    # (pointcloud/base.py) — a bare MagicMock's default __len__/__iter__ produces a
    # malformed (0,) array instead of (0, 3), which write_pointcloud_ply rejects.
    result.points = np.zeros((0, 3), dtype=np.float32)
    result.colors = np.zeros((0, 3), dtype=np.uint8)
    return result


def test_build_pointcloud_skips_if_colmap_and_zarr_exist(tmp_path):
    """Skip rebuild only when BOTH colmap/sparse/0/cameras.bin and feedforward.zarr exist."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    colmap_dir = rec.backend_dir / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True)
    (colmap_dir / "cameras.bin").touch()
    # The pointcloud zarr marker is also required; colmap alone no longer skips.
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    with (
        patch("collab_splats.wrapper.reconstructor._run_feedforward") as mock_ff,
        patch.object(rec, "_load_pointcloud_from_disk", return_value=mock_result),
    ):
        rec.build_pointcloud(overwrite=False)

    mock_ff.assert_not_called()


def test_reconstructor_bundle_adjustment_raises(tmp_path):
    """bundle_adjustment=True at the Reconstructor level raises NotImplementedError (was a silent no-op)."""
    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"backend": "vggtx", "bundle_adjustment": True},
    }
    rec = Reconstructor(config)
    with patch("collab_splats.wrapper.reconstructor._run_feedforward") as mock_ff:
        mock_ff.return_value = _make_mock_pointcloud_result(tmp_path)
        with pytest.raises(NotImplementedError, match="bundle_adjustment"):
            rec.build_pointcloud(overwrite=True)


def test_build_pointcloud_feedforward_vggtx(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(mock_result, None)) as mock_ff:
        result = rec.build_pointcloud(overwrite=True)

    mock_ff.assert_called_once()
    assert result is mock_result
    assert rec.pointcloud is mock_result


def test_build_pointcloud_method_dir_routing(tmp_path):
    """mapanything backend → out/mapanything/."""
    config = _make_config(tmp_path, {"pointcloud": {"backend": "mapanything"}})
    rec = Reconstructor(config)
    assert rec.backend_dir == tmp_path / "out" / "mapanything"


def test_build_pointcloud_nerfstudio_dispatches_correctly(tmp_path):
    """Test that method='nerfstudio' dispatches to _run_nerfstudio."""
    config = _make_config(
        tmp_path,
        {
            "pointcloud": {"method": "nerfstudio", "backend": "vggtx"},
            "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
        },
    )
    rec = Reconstructor(config)

    with (
        patch.object(rec, "_run_nerfstudio") as mock_ns,
        patch("collab_splats.wrapper.reconstructor._run_feedforward") as mock_ff,
    ):
        mock_ns.return_value = _make_mock_pointcloud_result(tmp_path)
        rec.build_pointcloud(overwrite=True)

    # _run_nerfstudio was called, not _run_feedforward
    mock_ns.assert_called_once()
    mock_ff.assert_not_called()


def test_extract_semantics_uses_feature_cache(tmp_path):
    """2D feature extraction skipped when cache exists."""
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    rec.pointcloud = mock_result

    # Pre-populate 2D cache so extraction is skipped
    cache_path = rec.semantics_cache_dir / "dinov2.zarr"
    cache_path.mkdir(parents=True)

    with (
        patch("collab_splats.wrapper.reconstructor._extract_2d_features") as mock_2d,
        patch("collab_splats.wrapper.reconstructor._lift_and_save") as mock_lift,
    ):
        mock_lift.return_value = rec.backend_dir / "semantics"
        rec.extract_semantics(result=mock_result, overwrite=False)

    mock_2d.assert_not_called()  # cache hit — no extraction


def test_extract_semantics_skips_if_lifted_exists(tmp_path):
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    lifted_dir = rec.backend_dir / "semantics"
    lifted_dir.mkdir(parents=True)
    (lifted_dir / "dinov2_lifted.zarr").mkdir()

    with (
        patch("collab_splats.wrapper.reconstructor._extract_2d_features") as mock_2d,
        patch("collab_splats.wrapper.reconstructor._lift_and_save") as mock_lift,
    ):
        rec.extract_semantics(overwrite=False)

    mock_2d.assert_not_called()
    mock_lift.assert_not_called()


def test_extract_2d_features_reads_zarr_directly(tmp_path):
    """_extract_2d_features delegates straight to extract_and_cache_from_zarr — no temp-JPG bridge."""
    from collab_splats.wrapper import reconstructor as rec_mod

    frames_zarr = tmp_path / "frames.zarr"
    cache_dir = tmp_path / "semantics"
    sentinel = cache_dir / "dinov2.zarr"

    # Fake extractor records the call and returns a sentinel cache path
    class _FakeExtractor:
        def __init__(self):
            self.calls = []

        def extract_and_cache_from_zarr(self, frames_zarr_path, cache_dir):
            self.calls.append((frames_zarr_path, cache_dir))
            return sentinel

    fake = _FakeExtractor()

    with (
        patch.object(rec_mod, "_get_extractor", return_value=fake) as mock_get,
        patch.object(rec_mod, "FrameStore") as mock_fs,
    ):
        result = rec_mod._extract_2d_features("dinov2", frames_zarr, cache_dir)

    # Extractor resolved by name, then fed the frames.zarr path + cache dir directly. The dir
    # is used as given: the extractor names the store, so no per-extractor subdir is joined on.
    mock_get.assert_called_once_with("dinov2")
    assert fake.calls == [(frames_zarr, cache_dir)]
    # No temp-export bridge: FrameStore is never touched
    mock_fs.open.assert_not_called()
    # Sentinel cache path is propagated back unchanged
    assert result == sentinel


def _run_lift_and_save(tmp_path, n_components, dim=32, n_points=6):
    """Drive the real _lift_and_save with the heavy lift/loader stubbed. Returns the output dir."""
    import torch
    import zarr

    from collab_splats.wrapper import reconstructor as rec_mod

    # 2D feature cache the writer reads: (N, D, H_p, W_p)
    cache = zarr.open(str(tmp_path / "dinov2.zarr"), mode="w")
    cache["features"] = np.zeros((2, dim, 2, 2), dtype=np.float32)
    (tmp_path / "feedforward.zarr").mkdir()
    out_dir = tmp_path / "semantics"

    with (
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", MagicMock()),
        patch("collab_splats.pointcloud.utils.lift_features", return_value=torch.rand(n_points, dim)),
    ):
        # Both training args are required; these tests assert file layout, so no early stop
        rec_mod._lift_and_save(
            "dinov2",
            tmp_path / "dinov2.zarr",
            tmp_path / "feedforward.zarr",
            out_dir,
            n_components,
            target_cosine=None,
            max_epochs=1,
        )
    return out_dir


def test_lift_and_save_writes_weights_beside_codes(tmp_path):
    """The real writer lands semantics/<extractor>_ae.pt — not a compressor.pt directory.

    FeatureAutoencoder.save() mkdirs the path it is given, so passing a filename silently
    creates a DIRECTORY of that name and no consumer can load the weights.
    """
    import zarr

    out_dir = _run_lift_and_save(tmp_path, n_components=8)

    assert (out_dir / "dinov2_ae.pt").is_file()
    assert not (out_dir / "compressor.pt").exists()
    store = zarr.open(str(out_dir / "dinov2_lifted.zarr"), mode="r")
    assert np.asarray(store["features"]).shape == (6, 8)  # latent codes
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 8}


def test_lift_and_save_uncompressed_writes_full_dim_and_no_weights(tmp_path):
    """n_components: null is supported: full-dim codes, no autoencoder, attrs say so."""
    import zarr

    out_dir = _run_lift_and_save(tmp_path, n_components=None)

    assert not (out_dir / "dinov2_ae.pt").exists()
    store = zarr.open(str(out_dir / "dinov2_lifted.zarr"), mode="r")
    assert np.asarray(store["features"]).shape == (6, 32)
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 32}


def test_mesh_skips_if_ply_exists(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf"}})
    rec = Reconstructor(config)
    mesh_path = rec.backend_dir / "mesh.ply"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.touch()

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        result = rec.mesh(overwrite=False)

    mock_mesh.assert_not_called()
    assert result == mesh_path


def test_mesh_skip_check_matches_tsdf_writer_filename(tmp_path):
    """mesh()'s skip-check fires on the file the real TSDF writer actually laid down.

    Neither filename is hardcoded here: the mesher writes the file and mesh() looks for it,
    so the test breaks if either side renames the mesh independently of the other.
    """
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf"}})
    rec = Reconstructor(config)

    # Genuine write path: a tiny synthetic TSDF run produces the mesh file itself
    fusion = Open3DTSDFFusion(output_dir=rec.backend_dir, clean_repair=False)
    depths = np.ones((2, 32, 32), dtype=np.float32)
    rgbs = np.full((2, 32, 32, 3), 0.5, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)
    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(2, axis=0)
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 32.0
    intrinsics[:, 0, 2] = intrinsics[:, 1, 2] = 16.0
    written = fusion.create(depths, rgbs, c2w, intrinsics).mesh_path
    # Pin the writer side separately: without this, a writer regression surfaces below as the
    # same "No PointcloudResult available" fall-through as a reader regression, hiding which broke
    assert written.exists()

    # Skip must short-circuit on that exact file rather than re-running TSDF
    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        result = rec.mesh(overwrite=False)

    mock_mesh.assert_not_called()
    assert result == written


def test_mesh_runs_tsdf(tmp_path):
    config = _make_config(
        tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04}}
    )
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    rec.pointcloud = mock_result

    # feedforward.zarr must exist for the pre-flight check in mesh()
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        mock_mesh.return_value = rec.backend_dir / "mesh.ply"
        result = rec.mesh(result=mock_result, overwrite=True)

    mock_mesh.assert_called_once()


def test_mesh_forwards_clean_repair_from_config(tmp_path):
    """clean_repair is reachable from a config file — the CLI/remote path is the one that meshes.

    Before this key existed, only the dashboard could ask for cleanup, so a config-driven run had
    no way to turn it on.
    """
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf", "clean_repair": True}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        mock_mesh.return_value = rec.backend_dir / "mesh.ply"
        rec.mesh(result=mock_result, overwrite=True)

    assert mock_mesh.call_args.kwargs["clean_repair"] is True


def test_mesh_clean_repair_defaults_off(tmp_path):
    """base.yaml is the sole default source, and the default must not cost every run a second pass."""
    rec = Reconstructor(_make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf"}}))
    mock_result = _make_mock_pointcloud_result(tmp_path)
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        mock_mesh.return_value = rec.backend_dir / "mesh.ply"
        rec.mesh(result=mock_result, overwrite=True)

    assert mock_mesh.call_args.kwargs["clean_repair"] is False


def _tsdf_mesh_doubles(n_colmap=2, n_zarr=2, model_hw=(8, 8)):
    """(PointcloudResult double with original-res K, FeedforwardResult with model-res K)."""
    H, W = model_hw

    # COLMAP camera after _rescale_reconstruction_to_original_dimensions: 2x the model grid
    result = MagicMock()
    result.extrinsics = np.eye(4, dtype=np.float32)[None].repeat(n_colmap, axis=0)
    result.extrinsics[:, 0, 3] = 7.0  # distinctive, so the two pose sources are separable
    K_orig = np.eye(3, dtype=np.float32)[None].repeat(n_colmap, axis=0)
    K_orig[:, 0, 0] = K_orig[:, 1, 1] = 2.0 * W
    K_orig[:, 0, 2] = W  # cx = W → principal point outside a W-wide image
    K_orig[:, 1, 2] = H
    result.intrinsics = K_orig

    K_model = np.eye(3, dtype=np.float32)[None].repeat(n_zarr, axis=0)
    K_model[:, 0, 0] = K_model[:, 1, 1] = float(W)
    K_model[:, 0, 2] = W / 2
    K_model[:, 1, 2] = H / 2
    ff = FeedforwardResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n_zarr, axis=0),
        intrinsics=K_model,
        image_paths=[Path(f"frame_{i:04d}.png") for i in range(n_zarr)],
        original_coords=np.tile([0, 0, W, H, W, H], (n_zarr, 1)).astype(np.float32),
        model_width=W,
        model_height=H,
        images=torch.zeros((n_zarr, 3, H, W), dtype=torch.float32),
        depth=np.ones((n_zarr, H, W), dtype=np.float32),
    )
    return result, ff


def test_run_tsdf_mesh_fuses_zarr_intrinsics_not_colmap(tmp_path):
    """Regression: COLMAP K is original-res, zarr depth is model-res. Fuse with the zarr K."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles()
    mock_creator = MagicMock()
    with (
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=ff),
        patch("collab_splats.mesh.get_mesh_creator", return_value=mock_creator),
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            clean_repair=False,
        )

    fused_K = mock_creator.create.call_args[0][3]
    np.testing.assert_allclose(fused_K, ff.intrinsics)
    assert not np.allclose(fused_K, result.intrinsics)


def test_run_tsdf_mesh_uses_colmap_poses(tmp_path):
    """COLMAP stays the pose authority — BA and LC corrections land there, not in the zarr."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles()
    mock_creator = MagicMock()
    with (
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=ff),
        patch("collab_splats.mesh.get_mesh_creator", return_value=mock_creator),
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            clean_repair=False,
        )

    fused_c2w = mock_creator.create.call_args[0][2]
    np.testing.assert_allclose(fused_c2w, np.linalg.inv(result.extrinsics), atol=1e-5)


def test_run_tsdf_mesh_raises_on_frame_count_mismatch(tmp_path):
    """Stage re-runs can pair a COLMAP dir with a feedforward.zarr from a different run."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles(n_colmap=3, n_zarr=2)
    with (
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=ff),
        pytest.raises(ValueError, match="feedforward.zarr"),
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            clean_repair=False,
        )


def test_run_tsdf_mesh_passes_clean_repair_to_the_fusion(tmp_path):
    """The flag has to survive the last hop too — mesh() → _run_tsdf_mesh → the mesh creator."""
    from collab_splats.wrapper.reconstructor import _run_tsdf_mesh

    result, ff = _tsdf_mesh_doubles()
    mock_creator = MagicMock()
    with (
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=ff),
        patch("collab_splats.mesh.get_mesh_creator", return_value=mock_creator) as mock_get,
    ):
        _run_tsdf_mesh(
            result=result,
            feedforward_zarr=tmp_path / "feedforward.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=1.0,
            clean_repair=True,
        )

    assert mock_get.call_args.kwargs["clean_repair"] is True
    assert mock_get.call_args.kwargs["depth_trunc"] == 1.0


def test_run_pipeline_calls_stages_in_order(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.frames_zarr
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh") or tmp_path

    rec.run_pipeline(stages=["preproc", "pointcloud", "semantics", "mesh"])
    assert calls == ["preproc", "pointcloud", "semantics", "mesh"]


def test_run_pipeline_subset(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.frames_zarr
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run_pipeline(stages=["preproc", "pointcloud"])
    assert calls == ["preproc", "pointcloud"]
    assert "semantics" not in calls
    assert "mesh" not in calls


def test_run_pipeline_dep_validation(tmp_path):
    """semantics requires pointcloud to have run first."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with pytest.raises(ValueError, match="pointcloud"):
        rec.run_pipeline(stages=["semantics"])


def test_run_pipeline_dep_satisfied_by_existing_output(tmp_path):
    """`--stages pointcloud` alone runs when a prior preprocess's frames.zarr exists on disk."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    # Preprocess output already present → dependency is satisfied without re-running it.
    rec.frames_zarr.mkdir(parents=True, exist_ok=True)
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run_pipeline(stages=["pointcloud"])
    assert calls == ["pointcloud"]


def test_run_pipeline_missing_dep_output_still_raises(tmp_path):
    """pointcloud with no frames.zarr and no preproc stage → hard error."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with pytest.raises(ValueError, match="preproc"):
        rec.run_pipeline(stages=["pointcloud"])


def test_run_pipeline_default_uses_config_enabled(tmp_path):
    config = _make_config(
        tmp_path,
        {
            "semantics": {"enabled": True, "extractor": "dinov2"},
            "mesh": {"enabled": False},
        },
    )
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.frames_zarr
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path

    rec.run_pipeline()  # no stages arg — uses config
    assert "semantics" in calls
    assert "mesh" not in calls


def test_splatter_emits_deprecation_warning(tmp_path):
    # splatter.py has a hard top-level nerfstudio import; skip if import fails
    try:
        from collab_splats.wrapper.splatter import Splatter
    except (ImportError, ModuleNotFoundError):
        pytest.skip("nerfstudio not importable in this env")
    config = {
        "file_path": str(tmp_path / "video.mp4"),
        "method": "rade-features",
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            Splatter(config)
        except Exception:
            pass  # Splatter init may fail due to missing deps; warning should still fire
    assert any(
        "deprecated" in str(warning.message).lower() for warning in w
    ), f"No deprecation warning found in: {[str(x.message) for x in w]}"
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


########################################
# Localization stage
########################################


def test_localize_in_stage_order_and_deps():
    from collab_splats.wrapper import reconstructor as R

    assert "localize" in R._STAGE_ORDER
    assert R._STAGE_DEPS["localize"] == ["pointcloud"]


def test_run_pipeline_auto_includes_localize_when_enabled(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    called = []
    with (
        patch.object(rec, "preprocess"),
        patch.object(rec, "build_pointcloud", return_value=None),
        patch.object(rec, "build_localization_db", side_effect=lambda **k: called.append("localize")),
    ):
        rec.run_pipeline()
    assert called == ["localize"]


def test_run_pipeline_omits_localize_when_disabled(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": False}})
    rec = Reconstructor(config)
    called = []
    with (
        patch.object(rec, "preprocess"),
        patch.object(rec, "build_pointcloud", return_value=None),
        patch.object(rec, "build_localization_db", side_effect=lambda **k: called.append("localize")),
    ):
        rec.run_pipeline()
    assert called == []


def test_localize_without_pointcloud_raises(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    with pytest.raises(ValueError, match="requires 'pointcloud'"):
        rec.run_pipeline(stages=["localize"])


def test_build_localization_db_missing_zarr_raises(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    with pytest.raises(FileNotFoundError, match="feedforward.zarr"):
        rec.build_localization_db()


def test_build_localization_db_skips_when_exists(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    ff = rec.backend_dir / "feedforward.zarr"
    ff.mkdir(parents=True)
    with (
        patch.object(R, "_localization_db_exists", return_value=True),
        patch.object(R, "_build_localization_db") as build,
    ):
        out = rec.build_localization_db(overwrite=False)
    build.assert_not_called()
    assert out == ff


########################################
# Viz wiring (P6.2)
########################################


def test_base_yaml_has_viz_defaults(tmp_path):
    """base.yaml exposes pointcloud.viz.enabled (default False) and pointcloud.viz.port (default 8080)."""
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
    }
    rec = Reconstructor(config)
    assert rec.config["pointcloud"]["viz"]["enabled"] is False
    assert rec.config["pointcloud"]["viz"]["port"] == 8080


def test_build_pointcloud_passes_viz_config_to_run_feedforward(tmp_path):
    """build_pointcloud reads pointcloud.viz.enabled/port strictly and forwards to _run_feedforward."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": True, "viz": {"enabled": True, "port": 9001}}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(mock_result, None)) as mock_ff:
        rec.build_pointcloud(overwrite=True)

    _, kwargs = mock_ff.call_args
    assert kwargs["loop_closure"] is True
    assert kwargs["viz_enabled"] is True
    assert kwargs["viz_port"] == 9001


def test_build_pointcloud_exposes_viewer_when_viz_enabled(tmp_path):
    """reconstructor.viewer is the Viewer instance _run_feedforward created, once build_pointcloud returns."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": True, "viz": {"enabled": True, "port": 9001}}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    mock_viewer = MagicMock()

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(mock_result, mock_viewer)):
        rec.build_pointcloud(overwrite=True)

    assert rec.viewer is mock_viewer


def test_reconstructor_viewer_none_when_viz_disabled(tmp_path):
    """reconstructor.viewer stays None both before build_pointcloud and after, when viz is disabled."""
    config = _make_config(tmp_path)  # viz.enabled defaults False
    rec = Reconstructor(config)
    assert rec.viewer is None

    mock_result = _make_mock_pointcloud_result(tmp_path)
    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(mock_result, None)):
        rec.build_pointcloud(overwrite=True)

    assert rec.viewer is None


def test_run_feedforward_attaches_viewer_when_enabled(tmp_path):
    """_run_feedforward attaches a Viewer to the LoopClosure creator when loop_closure + viz_enabled."""
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock()
    mock_lc_instance = MagicMock(outputs=None)

    with (
        patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=mock_creator),
        patch("collab_splats.geometry.loop_closure.wrapper.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
        patch.object(R, "FrameStore"),
    ):
        R._run_feedforward(
            backend="vggtx",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure=True,
            viz_enabled=True,
            viz_port=9999,
            max_points=500_000,
            use_multiview_confidence=False,
        )

    mock_lc_cls.assert_called_once_with(base=mock_creator, config=None)
    mock_viewer_cls.assert_called_once_with(port=9999)
    assert mock_lc_instance.viz is mock_viewer_cls.return_value


def test_run_feedforward_builds_lc_config_from_dict(tmp_path):
    """A dict loop_closure builds a LoopClosureConfig from its knobs and passes it through."""
    from collab_splats.geometry.loop_closure.wrapper import LoopClosureConfig
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock()
    mock_lc_instance = MagicMock(outputs=None)

    with (
        patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=mock_creator),
        patch("collab_splats.geometry.loop_closure.wrapper.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
        patch.object(R, "FrameStore"),
    ):
        R._run_feedforward(
            backend="vggtx",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure={"enabled": True, "submap_size": 32, "submap_overlap": 2},
            viz_enabled=False,
            viz_port=8080,
            max_points=500_000,
            use_multiview_confidence=False,
        )

    # LoopClosure got a config carrying the dict knobs
    _, kwargs = mock_lc_cls.call_args
    cfg = kwargs["config"]
    assert isinstance(cfg, LoopClosureConfig)
    assert (cfg.submap_size, cfg.submap_overlap) == (32, 2)


def test_run_feedforward_dict_enabled_false_skips_lc(tmp_path):
    """loop_closure={'enabled': False} runs the bare creator — no LoopClosure wrap."""
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock(outputs=None)

    with (
        patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=mock_creator),
        patch("collab_splats.geometry.loop_closure.wrapper.LoopClosure") as mock_lc_cls,
        patch.object(R, "FrameStore"),
    ):
        R._run_feedforward(
            backend="vggtx",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure={"enabled": False, "submap_size": 32},
            viz_enabled=False,
            viz_port=8080,
            max_points=500_000,
            use_multiview_confidence=False,
        )

    mock_lc_cls.assert_not_called()


def test_run_feedforward_invalid_lc_knob_raises(tmp_path):
    """An unknown loop_closure knob fails loud with the valid-key list."""
    from collab_splats.wrapper import reconstructor as R

    with (
        patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=MagicMock()),
        patch.object(R, "FrameStore"),
    ):
        with pytest.raises(ValueError, match="Invalid pointcloud.loop_closure knob"):
            R._run_feedforward(
                backend="vggtx",
                frames_zarr=tmp_path / "frames.zarr",
                output_dir=tmp_path / "out",
                loop_closure={"bogus_knob": 1},
                viz_enabled=False,
                viz_port=8080,
                max_points=500_000,
                use_multiview_confidence=False,
            )


def test_run_feedforward_no_viewer_when_viz_disabled(tmp_path):
    """No Viewer instantiated when viz_enabled=False, even with loop_closure=True."""
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock()
    mock_lc_instance = MagicMock(outputs=None)

    with (
        patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=mock_creator),
        patch("collab_splats.geometry.loop_closure.wrapper.LoopClosure", return_value=mock_lc_instance),
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
        patch.object(R, "FrameStore"),
    ):
        R._run_feedforward(
            backend="vggtx",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure=True,
            viz_enabled=False,
            viz_port=8080,
            max_points=500_000,
            use_multiview_confidence=False,
        )

    mock_viewer_cls.assert_not_called()


def test_run_feedforward_no_loop_closure_no_viewer(tmp_path):
    """loop_closure=False never wraps the creator or attaches viz, even if viz_enabled=True."""
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock(outputs=None)

    with (
        patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=mock_creator),
        patch("collab_splats.geometry.loop_closure.wrapper.LoopClosure") as mock_lc_cls,
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
        patch.object(R, "FrameStore"),
    ):
        R._run_feedforward(
            backend="vggtx",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure=False,
            viz_enabled=True,
            viz_port=8080,
            max_points=500_000,
            use_multiview_confidence=False,
        )

    mock_lc_cls.assert_not_called()
    mock_viewer_cls.assert_not_called()


def test_build_localization_db_runs_when_missing(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    ff = rec.backend_dir / "feedforward.zarr"
    ff.mkdir(parents=True)
    with (
        patch.object(R, "_localization_db_exists", return_value=False),
        patch.object(R, "_build_localization_db") as build,
    ):
        rec.build_localization_db(overwrite=False)
    # top_k comes from base.yaml's localization.top_k default (pairwise/vismatch fan-out)
    build.assert_called_once_with(ff, "loma", rec.frames_zarr, top_k=8)


########################################
# Leaf-stage re-run (stage-rerun-from-processed)
########################################


def test_leaf_stages_derived_from_dep_graph():
    """LEAF_STAGES is whatever nothing depends on — not a hardcoded list."""
    from collab_splats.wrapper import reconstructor as R

    # Recomputed from the graph rather than compared to a literal: adding a stage that consumes
    # mesh output must move mesh out of LEAF_STAGES, and a frozen literal would not notice.
    expected = {s for s in R._STAGE_ORDER if not any(s in deps for deps in R._STAGE_DEPS.values())}
    assert R.LEAF_STAGES == expected
    # Today's graph, spelled out so a failure above reads as a real change rather than a typo.
    assert expected == {"semantics", "mesh", "localize", "verify"}


def _seed_disk_reconstruction(rec, frame_idxs, image_names):
    """Write a frames.zarr and a COLMAP reconstruction the way a finished run leaves them."""
    FrameStore.create(
        rec.frames_zarr,
        [np.zeros((4, 6, 3), dtype=np.uint8) for _ in frame_idxs],
        [{"frame_idx": fi} for fi in frame_idxs],
        provenance={"video_path": "v.mp4"},
    )
    n = len(frame_idxs)
    recon = build_pycolmap_reconstruction(
        pts3d=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=np.stack([np.eye(4, dtype=np.float32)] * n),
        intrinsics=np.stack([np.eye(3, dtype=np.float32)] * n),
        image_width=6,
        image_height=4,
        image_names=image_names,
    )
    colmap_dir = rec.backend_dir / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    recon.write(str(colmap_dir))


def test_load_pointcloud_from_disk_matches_registered_image_names(tmp_path):
    """Loading a finished reconstruction off disk — the whole basis of a leaf-stage re-run."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    # Sparse source indices, as a quality-gated selection always produces: a loader that assumed
    # row position rather than frame_idx would survive a contiguous 0,1,2 store.
    frame_idxs = [2, 16]
    # Names built the way the creators build them — Path(f"frame_{idx:06d}"), no extension
    # (vggt_omega.py, vggtx.py, mapanything.py) — rather than re-spelled by hand here.
    _seed_disk_reconstruction(rec, frame_idxs, [Path(f"frame_{i:06d}").name for i in frame_idxs])

    result = rec._load_pointcloud_from_disk()
    assert [p.name for p in result.image_paths] == ["frame_000002", "frame_000016"]
    # extrinsics is where a naming mismatch actually bites: it looks every image_path up by name.
    # Every other test mocks this loader, so nothing else exercises the round trip.
    assert result.extrinsics.shape == (2, 4, 4)


def test_load_pointcloud_from_disk_rejects_a_disagreeing_store(tmp_path):
    """A store the reconstruction does not describe must name both, not raise a bare KeyError."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    # The reconstruction registers a frame the store never selected, and vice versa.
    _seed_disk_reconstruction(rec, [2, 16], ["frame_000002", "frame_000099"])

    with pytest.raises(ValueError, match="frame_000016"):
        rec._load_pointcloud_from_disk()


def test_stage_output_exists_mesh(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec._stage_output_exists("mesh") is False
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    (rec.backend_dir / "mesh.ply").touch()
    assert rec._stage_output_exists("mesh") is True


def test_stage_output_exists_semantics_is_per_extractor(tmp_path):
    """The marker is this run's extractor — another extractor's lifted store must not satisfy it."""
    from collab_splats.semantics.compression import lifted_store_path

    config = _make_config(tmp_path, {"semantics": {"extractor": "dinov2"}})
    rec = Reconstructor(config)
    sem_dir = rec.backend_dir / "semantics"
    sem_dir.mkdir(parents=True)
    lifted_store_path(sem_dir, "talk2dino").mkdir()
    assert rec._stage_output_exists("semantics") is False
    lifted_store_path(sem_dir, "dinov2").mkdir()
    assert rec._stage_output_exists("semantics") is True


def test_stage_output_exists_localize(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    # No zarr at all: absent, and _localization_db_exists must not even be consulted.
    assert rec._stage_output_exists("localize") is False
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True)
    with patch.object(R, "_localization_db_exists", return_value=True):
        assert rec._stage_output_exists("localize") is True
    with patch.object(R, "_localization_db_exists", return_value=False):
        assert rec._stage_output_exists("localize") is False


def _seed_pointcloud_markers(rec):
    """Make _stage_output_exists('pointcloud') true without running the stage."""
    colmap_dir = rec.backend_dir / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    (colmap_dir / "cameras.bin").touch()
    (rec.backend_dir / "feedforward.zarr").mkdir(parents=True, exist_ok=True)


def test_mesh_resolves_result_from_disk_when_not_in_memory(tmp_path):
    """`--stages mesh` on a pulled scene: self.pointcloud is None but COLMAP is on disk."""
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    loaded = _make_mock_pointcloud_result(tmp_path)
    rec._load_pointcloud_from_disk = lambda: loaded

    with patch.object(R, "_run_tsdf_mesh", return_value=rec.backend_dir / "mesh.ply") as run_mesh:
        rec.mesh()

    assert run_mesh.call_args.kwargs["result"] is loaded
    assert rec.pointcloud is loaded  # cached, so a second leaf stage does not re-read COLMAP


def test_mesh_without_pointcloud_on_disk_still_raises(tmp_path):
    """Nothing in memory and nothing on disk is still a hard error, not a silent skip."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    with pytest.raises(ValueError, match="No PointcloudResult"):
        rec.mesh()


def test_extract_semantics_resolves_result_from_disk(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"semantics": {"extractor": "dinov2"}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    loaded = _make_mock_pointcloud_result(tmp_path)
    rec._load_pointcloud_from_disk = lambda: loaded
    # 2D cache hit so the extractor never loads; only the lift is exercised.
    rec.semantics_cache_dir.mkdir(parents=True, exist_ok=True)
    (rec.semantics_cache_dir / "dinov2.zarr").mkdir()

    with patch.object(R, "_lift_and_save", return_value=rec.backend_dir / "semantics") as lift:
        rec.extract_semantics()

    lift.assert_called_once()
    assert rec.pointcloud is loaded


def test_run_pipeline_refuses_named_stage_whose_output_exists(tmp_path):
    """A named no-op must fail loudly: a remote re-run would otherwise pull GBs and push nothing."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()

    with pytest.raises(ValueError, match="already exists"):
        rec.run_pipeline(stages=["mesh"])


def test_run_pipeline_named_stage_with_overwrite_runs(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()
    calls = []
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh")

    rec.run_pipeline(stages=["mesh"], overwrite=True)
    assert calls == ["mesh"]


def test_run_pipeline_named_upstream_stages_resume_instead_of_refusing(tmp_path):
    """`--stages preproc,pointcloud,localize` retried after a localize failure must not refuse."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    # Both upstream stages already complete on disk; only the named leaf still has work.
    rec.frames_zarr.mkdir(parents=True, exist_ok=True)
    _seed_pointcloud_markers(rec)
    calls = []
    rec.preprocess = lambda overwrite=False: calls.append("preproc")
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud")
    rec.build_localization_db = lambda overwrite=False: calls.append("localize")

    # Reaching localize at all is the assertion: the refusal is scoped to leaf stages, so the
    # two completed non-leaf stages fall through to their own skip-checks as before.
    rec.run_pipeline(stages=["preproc", "pointcloud", "localize"])
    assert calls == ["preproc", "pointcloud", "localize"]


def test_run_pipeline_config_derived_stages_still_skip_silently(tmp_path):
    """stages=None comes from config enabled flags — resume behaviour must not become an error."""
    config = _make_config(tmp_path, {"mesh": {"enabled": True}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()
    calls = []
    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.frames_zarr
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud")
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh")

    rec.run_pipeline()  # must not raise
    assert calls == ["preproc", "pointcloud", "mesh"]
