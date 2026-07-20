import importlib.util
import shutil
import subprocess
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pycolmap
import pytest
import yaml
from mergedeep import merge

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
        "preprocessing": {"frame_selection": "fps", "frame_proportion": 0.1, "min_frames": 300},
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "bundle_adjustment": False, "loop_closure": False},
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 1024},
        "mesh": {"enabled": False, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04},
        "localization": {"enabled": False, "extractor": "dinosalad"},
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
        "preprocessing": {"frame_selection": "fps", "frame_proportion": 0.1, "min_frames": 10},
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "bundle_adjustment": False,
            "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 512},
        "mesh": {"enabled": False, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04},
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


def test_extract_frames_uniform_is_default_branch(tmp_path, monkeypatch):
    """frame_selection='uniform' takes the uniform sampler branch (not optical_flow)."""
    from collab_splats.wrapper import reconstructor as R

    calls = {}

    def fake_sample_frames(path, method, max_frames):
        calls["method"] = method
        return [np.zeros((4, 4, 3), dtype=np.uint8)], None

    def fake_video_info(path):
        return {"total_frames": 100}

    monkeypatch.setattr(R, "sample_frames", fake_sample_frames)
    monkeypatch.setattr(R, "get_video_info", fake_video_info)

    out = tmp_path / "out"
    video = tmp_path / "v.mp4"
    video.touch()
    R._extract_frames(video, out, "uniform", 0.1, 5, 50)
    assert calls["method"] == "uniform"


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
    # min_frames comes from base.yaml (150), NOT a stale code default (300)
    assert rec.config["preprocessing"]["min_frames"] == 150
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


def test_reconstructor_images_dir(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.images_dir == tmp_path / "out" / "images"


def test_reconstructor_features_dir(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.features_dir == tmp_path / "out" / "features"


def test_preprocess_skips_if_images_exist(tmp_path):
    """Skip extraction when images/ already populated and overwrite=False."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    images_dir = rec.images_dir
    images_dir.mkdir(parents=True)
    (images_dir / "frame_0001.jpg").touch()

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        result = rec.preprocess(overwrite=False)

    mock_extract.assert_not_called()
    assert result == images_dir


def test_preprocess_runs_if_images_missing(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        mock_extract.return_value = [rec.images_dir / "frame_0001.jpg"]
        rec.images_dir.mkdir(parents=True)
        (rec.images_dir / "frame_0001.jpg").touch()
        result = rec.preprocess(overwrite=False)

    assert result == rec.images_dir


def test_preprocess_overwrite_reruns(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)
    (rec.images_dir / "frame_0001.jpg").touch()

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        mock_extract.return_value = [rec.images_dir / "frame_0001.jpg"]
        rec.preprocess(overwrite=True)

    mock_extract.assert_called_once()


def _make_mock_pointcloud_result(tmp_path):
    """Minimal PointcloudResult mock for testing — avoids importing collab_splats.pointcloud."""
    result = MagicMock()
    result.reconstruction = pycolmap.Reconstruction()  # empty, no images
    result.image_paths = [tmp_path / "images" / "frame_0001.jpg"]
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

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=mock_result) as mock_ff:
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
    cache_path = rec.features_dir / "dinov2" / "dinov2.zarr"
    cache_path.mkdir(parents=True)

    with (
        patch("collab_splats.wrapper.reconstructor._extract_2d_features") as mock_2d,
        patch("collab_splats.wrapper.reconstructor._lift_and_save") as mock_lift,
    ):
        mock_lift.return_value = rec.backend_dir / "semantics" / "dinov2"
        rec.extract_semantics(result=mock_result, overwrite=False)

    mock_2d.assert_not_called()  # cache hit — no extraction


def test_extract_semantics_skips_if_lifted_exists(tmp_path):
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    lifted_dir = rec.backend_dir / "semantics" / "dinov2"
    lifted_dir.mkdir(parents=True)
    (lifted_dir / "features.zarr").mkdir()

    with (
        patch("collab_splats.wrapper.reconstructor._extract_2d_features") as mock_2d,
        patch("collab_splats.wrapper.reconstructor._lift_and_save") as mock_lift,
    ):
        rec.extract_semantics(overwrite=False)

    mock_2d.assert_not_called()
    mock_lift.assert_not_called()


def test_mesh_skips_if_ply_exists(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf"}})
    rec = Reconstructor(config)
    mesh_path = rec.backend_dir / "mesh" / "mesh.ply"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.touch()

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        result = rec.mesh(overwrite=False)

    mock_mesh.assert_not_called()
    assert result == mesh_path


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
        mock_mesh.return_value = rec.backend_dir / "mesh" / "mesh.ply"
        result = rec.mesh(result=mock_result, overwrite=True)

    mock_mesh.assert_called_once()


def test_run_pipeline_calls_stages_in_order(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh") or tmp_path

    rec.run_pipeline(stages=["preprocess", "pointcloud", "semantics", "mesh"])
    assert calls == ["preprocess", "pointcloud", "semantics", "mesh"]


def test_run_pipeline_subset(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run_pipeline(stages=["preprocess", "pointcloud"])
    assert calls == ["preprocess", "pointcloud"]
    assert "semantics" not in calls
    assert "mesh" not in calls


def test_run_pipeline_dep_validation(tmp_path):
    """semantics requires pointcloud to have run first."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with pytest.raises(ValueError, match="pointcloud"):
        rec.run_pipeline(stages=["semantics"])


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

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
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
    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
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
    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
    rec = Reconstructor(config)
    with pytest.raises(FileNotFoundError, match="feedforward.zarr"):
        rec.build_localization_db()


def test_build_localization_db_skips_when_exists(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
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


def test_build_localization_db_runs_when_missing(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma", "radius": 8.0}})
    rec = Reconstructor(config)
    ff = rec.backend_dir / "feedforward.zarr"
    ff.mkdir(parents=True)
    with (
        patch.object(R, "_localization_db_exists", return_value=False),
        patch.object(R, "_build_localization_db") as build,
    ):
        rec.build_localization_db(overwrite=False)
    build.assert_called_once_with(ff, "loma", 8.0)
