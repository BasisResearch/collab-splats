import importlib.util
import json
import types
import weakref
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import cv2
import numpy as np
import pycolmap
import pytest
import torch
import yaml
import zarr
from mergedeep import merge

from collab_splats.geometry import metrics
from collab_splats.mesh import create_tsdf_mesh
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.preproc import frames as fr
from collab_splats.preproc.undistort import calibrate_camera
from collab_splats.wrapper import reconstructor as R
from collab_splats.wrapper.reconstructor import Reconstructor
from tests.wrapper._stubs import minimal_feedforward_result

# Import ConfigLoader directly from config.py to avoid wrapper/__init__.py
spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).parent.parent.parent / "collab_splats" / "wrapper" / "config.py"
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
ConfigLoader = config_module.ConfigLoader


def test_config_load_base_defaults(tmp_path):
    base = {
        "preproc": {"frame_selection": "fps", "fps": 1.0, "min_frames": 300},
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "bundle_adjustment": False, "loop_closure": False},
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 1024},
        "mesh": {"enabled": False, "voxel_size": 0.01, "depth_trunc": 1.0},
        "localization": {"enabled": False, "matcher": "loma"},
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
        "preproc": {"frame_selection": "uniform", "fps": 1.0, "min_frames": 10},
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "bundle_adjustment": False,
            "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 512},
        "mesh": {"enabled": False, "voxel_size": 0.01, "depth_trunc": 1.0},
        "localization": {"enabled": False},
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
    """
    Each frame_selection value reaches its own sampler with only its own knobs.
    """
    from collab_splats.wrapper import reconstructor as R

    calls = {}

    def _recorder(name):
        def fake(path, **kwargs):
            calls.clear()
            calls["sampler"] = name
            calls.update(kwargs)
            return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

        return fake

    # The report is the samplers' shared input, not the dispatch under test — stub it
    # and assert every branch forwards the same object.
    report = {"frames": {}}
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: report)
    monkeypatch.setattr(R, "sample_fps", _recorder("fps"))
    monkeypatch.setattr(R, "sample_uniform", _recorder("uniform"))
    monkeypatch.setattr(R, "sample_optical_flow", _recorder("optical_flow"))
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    # Plots are not the dispatch under test, and the stub report has no columns
    monkeypatch.setattr(
        R,
        "preproc_viz",
        types.SimpleNamespace(
            **{
                name: lambda *a, **k: None
                for name in (
                    "plot_photometric",
                    "plot_motion",
                )
            }
        ),
    )

    out = tmp_path / "out"
    video = tmp_path / "v.mp4"
    video.touch()

    # fps: rate + both band bounds
    R.extract_frames(video, out / "a" / "images", "fps", 2.0, 5, 50)
    assert calls == {
        "sampler": "fps",
        "fps": 2.0,
        "min_frames": 5,
        "max_frames": 50,
        "report": report,
        "quality": None,
        "on_empty_slot": "rescue",
    }

    # uniform: max_frames is the count, with no fps and no floor — it spreads exactly
    # that many picks over the eligible pool
    R.extract_frames(video, out / "b" / "images", "uniform", None, 5, 50)
    assert calls == {
        "sampler": "uniform",
        "max_frames": 50,
        "report": report,
        "quality": None,
    }

    # optical_flow: max_frames caps the selector
    R.extract_frames(video, out / "c" / "images", "optical_flow", None, 5, 50)
    assert calls == {"sampler": "optical_flow", "max_frames": 50, "report": report, "quality": None}


def test_extract_frames_forwards_quality_overrides_to_every_sampler(tmp_path, monkeypatch):
    """
    preproc.quality is a shared knob: each branch hands it to its own sampler.
    """
    from collab_splats.wrapper import reconstructor as R

    seen = []

    def fake(path, **kwargs):
        seen.append(kwargs.get("quality"))
        return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    for name in ("sample_fps", "sample_uniform", "sample_optical_flow"):
        monkeypatch.setattr(R, name, fake)

    # Plots are not under test here, and the stub report carries no columns to draw
    monkeypatch.setattr(
        R,
        "preproc_viz",
        types.SimpleNamespace(plot_photometric=lambda *a, **k: None, plot_motion=lambda *a, **k: None),
    )

    video = tmp_path / "v.mp4"
    video.touch()
    overrides = {"sharpness_k": 1.0, "max_clipped_frac": 0.1}
    for i, selection in enumerate(("fps", "uniform", "optical_flow")):
        R.extract_frames(video, tmp_path / str(i) / "images", selection, 2.0, 5, 50, quality=overrides)

    assert seen == [overrides] * 3


def test_extract_frames_forwards_on_empty_slot_to_fps_only(tmp_path, monkeypatch):
    seen = {}

    def fake(name):
        def _f(path, **kwargs):
            seen[name] = kwargs
            return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

        return _f

    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    for name in ("sample_fps", "sample_uniform", "sample_optical_flow"):
        monkeypatch.setattr(R, name, fake(name))

    # Plots are not under test here, and the stub report carries no columns to draw
    monkeypatch.setattr(
        R, "preproc_viz", types.SimpleNamespace(plot_photometric=lambda *a, **k: None, plot_motion=lambda *a, **k: None)
    )

    video = tmp_path / "v.mp4"
    video.touch()
    for i, selection in enumerate(("fps", "uniform", "optical_flow")):
        R.extract_frames(video, tmp_path / str(i) / "images", selection, 2.0, 5, 50, on_empty_slot="drop")

    assert seen["sample_fps"]["on_empty_slot"] == "drop"
    assert "on_empty_slot" not in seen["sample_uniform"] and "on_empty_slot" not in seen["sample_optical_flow"]


def test_extract_frames_rejects_unknown_selection(tmp_path, monkeypatch):
    from collab_splats.wrapper import reconstructor as R

    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
    video = tmp_path / "v.mp4"
    video.touch()
    with pytest.raises(ValueError, match="frame_selection"):
        R.extract_frames(video, tmp_path / "out" / "images", "balanced", None, 5, 50)


def test_extract_frames_records_fps_in_provenance(tmp_path, monkeypatch):
    """
    frames.json must record the rate and empty-slot policy a scene was sampled with, not just the cap.
    """
    from collab_splats.wrapper import reconstructor as R

    monkeypatch.setattr(
        R,
        "sample_fps",
        lambda path, **kw: ([np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]),
    )
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    # Plots are not the dispatch under test, and the stub report has no columns
    monkeypatch.setattr(
        R,
        "preproc_viz",
        types.SimpleNamespace(
            **{
                name: lambda *a, **k: None
                for name in (
                    "plot_photometric",
                    "plot_motion",
                )
            }
        ),
    )

    video = tmp_path / "v.mp4"
    video.touch()
    R.extract_frames(video, tmp_path / "out" / "images", "fps", 2.0, None, 50, on_empty_slot="drop")

    prov = fr.read_manifest(tmp_path / "out" / "images")["provenance"]
    assert prov["fps"] == 2.0
    assert prov["method"] == "fps"
    assert prov["on_empty_slot"] == "drop"


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
    assert rec.config["preproc"]["min_frames"] is None
    # fps comes from base.yaml (2.0), and fps is the default selection method
    assert rec.config["preproc"]["fps"] == 2.0
    assert rec.config["preproc"]["frame_selection"] == "fps"
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
    """images/ is the one keyframe store — nothing derives a frames.zarr path any more."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.images_dir == tmp_path / "out" / "images"
    assert not hasattr(rec, "frames_zarr")


def test_reconstructor_semantics_cache_dir(tmp_path):
    """The 2D cache is scene-level: it depends on the frames only, not on the backend."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.semantics_cache_dir == tmp_path / "out" / "semantics"


def test_preprocess_skips_if_images_dir_exists(tmp_path):
    """Skip extraction when images/ already exists and overwrite=False."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor.extract_frames") as mock_extract:
        result = rec.preprocess(overwrite=False)

    mock_extract.assert_not_called()
    assert result == rec.images_dir


def test_preprocess_runs_if_images_dir_missing(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with patch("collab_splats.wrapper.reconstructor.extract_frames") as mock_extract:
        mock_extract.return_value = 1
        result = rec.preprocess(overwrite=False)

    mock_extract.assert_called_once()
    assert result == rec.images_dir


def test_preprocess_overwrite_reruns(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor.extract_frames") as mock_extract:
        mock_extract.return_value = 1
        rec.preprocess(overwrite=True)

    mock_extract.assert_called_once()


def _make_mock_pointcloud_result(tmp_path):
    """Minimal PointcloudResult mock for testing — avoids importing collab_splats.pointcloud."""
    result = MagicMock()
    result.image_paths = [tmp_path / "images" / "frame_0001.jpg"]
    # Match the real PointcloudResult.points/.colors shape for an empty point set
    # (pointcloud/base.py) — a bare MagicMock's default __len__/__iter__ produces a
    # malformed (0,) array instead of the (0, 3) that this result's consumers
    # (outlier_mask, the splats train call) expect.
    result.points = np.zeros((0, 3), dtype=np.float32)
    result.colors = np.zeros((0, 3), dtype=np.uint8)
    return result


def test_build_pointcloud_skips_if_colmap_and_zarr_exist(tmp_path):
    """Skip rebuild only when BOTH the colmap model dir and pointcloud.zarr exist."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.colmap_model_dir.mkdir(parents=True)
    # The pointcloud zarr marker is also required; colmap alone no longer skips.
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    with (
        patch("collab_splats.wrapper.reconstructor._run_feedforward") as mock_ff,
        patch.object(rec, "_load_pointcloud_from_disk", return_value=mock_result),
    ):
        rec.build_pointcloud(overwrite=False)

    mock_ff.assert_not_called()


def test_build_pointcloud_feedforward_vggtx(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=(mock_result, None)) as mock_ff:
        result = rec.build_pointcloud(overwrite=True)

    mock_ff.assert_called_once()
    assert result is mock_result
    assert rec.pointcloud is mock_result


def test_build_pointcloud_frees_the_dense_arrays_after_save(tmp_path):
    """
    self.pointcloud is light once pointcloud.zarr is written: no dense array stays alive.
    """
    rec = Reconstructor(_make_config(tmp_path))
    refs = {}

    # A real result held on self.outputs, as the LoopClosure wrapper leaves it
    # - weakrefs only: the test itself must not keep a dense array alive
    class _Creator:
        def __init__(self, **kwargs):
            self.outputs = None

        def create_pointcloud(self, source, out_dir, model_dir):
            self.outputs = minimal_feedforward_result()
            self.outputs.world_points = np.zeros((2, 8, 8, 3), np.float32)
            refs["creator"] = weakref.ref(self)
            for name in ("depth", "images", "world_points"):
                refs[name] = weakref.ref(getattr(self.outputs, name))
            return self.outputs

    with patch.object(R, "get_creator", return_value=_Creator), patch.object(R, "pytorch_gc"):
        rec.build_pointcloud(overwrite=True)

    # Freed by refcount alone, no gc.collect: nothing, creator included, still holds them
    assert {name: ref() is None for name, ref in refs.items()} == dict.fromkeys(refs, True)
    assert rec.pointcloud.depth is None and rec.pointcloud.world_points is None
    assert rec.pointcloud.points.shape == (5, 3)

    # The dense arrays live on in the zarr the downstream stages read
    assert zarr.open(str(rec.pointcloud_zarr), mode="r")["depth"].shape == (2, 8, 8)


def test_build_pointcloud_method_dir_routing(tmp_path):
    """mapanything backend → out/mapanything/."""
    config = _make_config(tmp_path, {"pointcloud": {"backend": "mapanything"}})
    rec = Reconstructor(config)
    assert rec.backend_dir == tmp_path / "out" / "mapanything"


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


def test_extract_2d_features_forwards_the_images_dir(tmp_path):
    """_extract_2d_features hands the scene's images/ dir straight to extract_feature_cache."""
    from collab_splats.wrapper import reconstructor as rec_mod

    images_dir = tmp_path / "images"
    cache_dir = tmp_path / "semantics"
    sentinel = cache_dir / "dinov2.zarr"
    extractor = object()
    calls = []

    with (
        patch.object(rec_mod, "_get_extractor", return_value=extractor) as mock_get,
        patch.object(
            rec_mod,
            "extract_feature_cache",
            lambda ext, images, cache: calls.append((ext, images, cache)) or sentinel,
        ),
    ):
        result = rec_mod._extract_2d_features("dinov2", images_dir, cache_dir)

    # Extractor resolved by name, then fed the images/ dir + cache dir directly. The dir is
    # used as given: the extractor names the store, so no per-extractor subdir is joined on.
    mock_get.assert_called_once_with("dinov2")
    assert calls == [(extractor, images_dir, cache_dir)]
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
    (tmp_path / "pointcloud.zarr").mkdir()
    out_dir = tmp_path / "semantics"

    with (
        patch("collab_splats.pointcloud.base.PointcloudResult.load_zarr", MagicMock()),
        # Patch the reconstructor's own binding: lift_features is imported at module scope,
        # so patching collab_splats.pointcloud.utils would not reach the name it calls.
        patch.object(rec_mod, "lift_features", return_value=torch.rand(n_points, dim)),
    ):
        # Both training args are required; these tests assert file layout, so no early stop
        rec_mod._lift_and_save(
            "dinov2",
            tmp_path / "dinov2.zarr",
            tmp_path / "pointcloud.zarr",
            out_dir,
            n_components,
            target_cosine=None,
            max_epochs=1,
            images_dir=tmp_path / "images",
        )
    return out_dir


def test_lift_and_save_writes_weights_beside_codes(tmp_path):
    """The real writer lands semantics/<extractor>_ae.pt — not a compressor.pt directory.

    FeatureAutoencoder.save() takes the weights FILE path and mkdirs its parent. The earlier
    dir+extractor form mkdir'd whatever it was handed, so passing a filename silently created
    a DIRECTORY of that name and no consumer could load the weights.
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
    config = _make_config(tmp_path, {"mesh": {"enabled": True}})
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
    config = _make_config(tmp_path, {"mesh": {"enabled": True}})
    rec = Reconstructor(config)

    # Genuine write path: a tiny synthetic TSDF run produces the mesh file itself
    depths = np.ones((2, 32, 32), dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)
    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(2, axis=0)
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 32.0
    intrinsics[:, 0, 2] = intrinsics[:, 1, 2] = 16.0
    written = create_tsdf_mesh(
        depths,
        np.full((2, 32, 32, 3), 128, np.uint8),
        c2w,
        intrinsics,
        rec.backend_dir,
        voxel_size=0.05,
        depth_trunc=2.0,
    )
    # Pin the writer side separately: without this, a writer regression surfaces below as the
    # same "No PointcloudResult available" fall-through as a reader regression, hiding which broke
    assert written.exists()

    # Skip must short-circuit on that exact file rather than re-running TSDF
    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        result = rec.mesh(overwrite=False)

    mock_mesh.assert_not_called()
    assert result == written


def test_mesh_runs_tsdf(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "voxel_size": 0.01}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    rec.pointcloud = mock_result

    # pointcloud.zarr must exist for the pre-flight check in mesh()
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        mock_mesh.return_value = rec.backend_dir / "mesh.ply"
        rec.mesh(result=mock_result, overwrite=True)

    mock_mesh.assert_called_once()


def _tsdf_mesh_doubles(n_colmap=2, n_zarr=2, model_hw=(8, 8)):
    """(in-session result double with full-res K, PointcloudResult loaded from the zarr)."""
    H, W = model_hw

    # Full-res K: 2x the model grid
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
    ff = PointcloudResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n_zarr, axis=0),
        intrinsics=None,
        model_intrinsics=K_model,
        image_paths=[Path(f"frame_{i:04d}.png") for i in range(n_zarr)],
        original_coords=np.tile([0, 0, 2 * W, 2 * H, 2 * W, 2 * H], (n_zarr, 1)).astype(np.float32),
        model_width=W,
        model_height=H,
        images=torch.zeros((n_zarr, 3, H, W), dtype=torch.float32),
        depth=np.ones((n_zarr, H, W), dtype=np.float32),
    )
    return result, ff


def test_run_tsdf_mesh_fuses_full_res_intrinsics(tmp_path, monkeypatch):
    """
    Model-res depth is lifted onto the frame grid, so the full-res `intrinsics` is the right K.

    Pairing the model grid's depth with the original grid's K is the 2026-08-11 collapse bug
    (5.06M -> 75k vertices); this pins the pairing that fixed it.
    """
    result, ff = _tsdf_mesh_doubles(model_hw=(16, 16))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    monkeypatch.setattr(R, "upsample_depths", lambda d, r, b: np.ones((2, 32, 32), np.float32))
    fuse = MagicMock(return_value=tmp_path / "mesh.ply")
    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())

    R._run_tsdf_mesh(
        result=result,
        pointcloud_zarr=tmp_path / "pointcloud.zarr",
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        voxel_size=0.01,
        depth_trunc=2.0,
    )
    np.testing.assert_array_equal(fuse.call_args.args[3], result.intrinsics)


def _tsdf_mesh_fuse_kwargs(tmp_path, monkeypatch, **overrides):
    """
    Run _run_tsdf_mesh over doubles and return the kwargs create_tsdf_mesh was called with.
    """
    result, ff = _tsdf_mesh_doubles(model_hw=(16, 16))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    monkeypatch.setattr(R, "upsample_depths", lambda d, r, b: np.ones((2, 32, 32), np.float32))
    fuse = MagicMock(return_value=tmp_path / "mesh.ply")
    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())

    R._run_tsdf_mesh(
        result=result,
        pointcloud_zarr=tmp_path / "pointcloud.zarr",
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        voxel_size=0.01,
        depth_trunc=2.0,
        **overrides,
    )
    return fuse.call_args.kwargs


def test_run_tsdf_mesh_passes_use_convex_hull_to_cleaning(tmp_path, monkeypatch):
    """
    mesh.use_convex_hull reaches clean_repair_mesh, and is off unless asked for.
    """
    for flag in (False, True):
        overrides = {"use_convex_hull": True} if flag else {}
        _tsdf_mesh_fuse_kwargs(tmp_path, monkeypatch, **overrides)
        assert R.clean_repair_mesh.call_args.kwargs == {"use_convex_hull": flag}


def test_run_tsdf_mesh_sdf_trunc_mult_scales_the_truncation_band(tmp_path, monkeypatch):
    """
    sdf_trunc reaches create_tsdf_mesh as sdf_trunc_mult x voxel_size, defaulting to 4x.

    - the multiplier, not voxel_size, sets the thin-structure floor: a TSDF cancels anything
      thinner than 2 x sdf_trunc, so the default band is 8 voxels wide
    - measured on GH010229 at voxel 0.2 (scene diagonal 262 world units): the default floor is
      1.60 units, while the voxel grid alone would resolve 0.40
    """
    default = _tsdf_mesh_fuse_kwargs(tmp_path, monkeypatch)
    assert default["sdf_trunc"] == pytest.approx(0.04)

    narrow = _tsdf_mesh_fuse_kwargs(tmp_path, monkeypatch, sdf_trunc_mult=1.5)
    assert narrow["sdf_trunc"] == pytest.approx(0.015)
    assert narrow["voxel_size"] == pytest.approx(0.01)


def test_run_tsdf_mesh_uses_result_poses(tmp_path, monkeypatch):
    """
    Fusion poses are the passed result's extrinsics — in-session, refined or loop-closed.
    """
    result, ff = _tsdf_mesh_doubles(model_hw=(16, 16))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    monkeypatch.setattr(R, "upsample_depths", lambda d, r, b: np.ones((2, 32, 32), np.float32))
    fuse = MagicMock(return_value=tmp_path / "mesh.ply")
    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())

    R._run_tsdf_mesh(
        result=result,
        pointcloud_zarr=tmp_path / "pointcloud.zarr",
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        voxel_size=0.01,
        depth_trunc=2.0,
    )

    np.testing.assert_allclose(fuse.call_args.args[2], np.linalg.inv(result.extrinsics), atol=1e-5)


def test_run_tsdf_mesh_raises_on_frame_count_mismatch(tmp_path, monkeypatch):
    """
    Stage re-runs can pair a COLMAP dir with a pointcloud.zarr from a different run.
    """
    result, ff = _tsdf_mesh_doubles(n_colmap=3, n_zarr=2, model_hw=(16, 16))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))

    with pytest.raises(ValueError, match="pointcloud.zarr"):
        R._run_tsdf_mesh(
            result=result,
            pointcloud_zarr=tmp_path / "pointcloud.zarr",
            output_dir=tmp_path,
            images_dir=tmp_path / "images",
            voxel_size=0.01,
            depth_trunc=2.0,
        )


def test_run_tsdf_mesh_masks_depth_by_confidence(tmp_path, monkeypatch):
    """
    conf_percentile zeroes the depth under the percentile before it reaches the fusion.
    """
    result, ff = _tsdf_mesh_doubles(model_hw=(16, 16))
    ff.confidence = np.tile(np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16), (2, 1, 1))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    seen = {}

    def spy_upsample(depths, rgbs, boxes):
        seen["depths"] = depths
        return np.ones((2, 32, 32), np.float32)

    monkeypatch.setattr(R, "upsample_depths", spy_upsample)
    monkeypatch.setattr(R, "create_tsdf_mesh", MagicMock(return_value=tmp_path / "mesh.ply"))
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())

    R._run_tsdf_mesh(
        result=result,
        pointcloud_zarr=tmp_path / "pointcloud.zarr",
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        voxel_size=0.01,
        depth_trunc=2.0,
        conf_percentile=20,
    )

    # The bottom 20% of a linear ramp is zeroed, the rest is untouched
    masked = seen["depths"]
    assert (masked == 0).mean() == pytest.approx(0.2, abs=0.02)
    assert masked.max() == ff.depth.max()


def test_run_pipeline_calls_stages_in_order(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh") or tmp_path

    rec.run_pipeline(stages=["preproc", "pointcloud", "semantics", "mesh"])
    assert calls == ["preproc", "pointcloud", "semantics", "mesh"]


def test_run_pipeline_subset(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.images_dir
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
    """`--stages pointcloud` alone runs when a prior preprocess's images/ exists on disk."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    # Preprocess output already present → dependency is satisfied without re-running it.
    rec.images_dir.mkdir(parents=True, exist_ok=True)
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run_pipeline(stages=["pointcloud"])
    assert calls == ["pointcloud"]


def test_run_pipeline_missing_dep_output_still_raises(tmp_path):
    """pointcloud with no images/ and no preproc stage → hard error."""
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

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path
    # report is always on and has no config flag, so a config-derived run always includes it
    rec.reconstruction_quality_report = (
        lambda overwrite=False: calls.append("reconstruction_quality_report") or tmp_path
    )

    rec.run_pipeline()  # no stages arg — uses config
    assert "semantics" in calls
    assert "mesh" not in calls


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
        patch.object(rec, "reconstruction_quality_report"),  # always on, and it would resolve a real reconstruction
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
        patch.object(rec, "reconstruction_quality_report"),  # always on, and it would resolve a real reconstruction
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
    with pytest.raises(FileNotFoundError, match="pointcloud.zarr"):
        rec.build_localization_db()


def test_build_localization_db_skips_when_exists(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    pc_zarr = rec.backend_dir / "pointcloud.zarr"
    pc_zarr.mkdir(parents=True)
    with (
        patch.object(R, "_localization_db_exists", return_value=True),
        patch.object(R, "_build_localization_db") as build,
    ):
        out = rec.build_localization_db(overwrite=False)
    build.assert_not_called()
    assert out == pc_zarr


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
        patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=mock_creator)),
        patch("collab_splats.wrapper.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
    ):
        R._run_feedforward(
            backend="vggtx",
            images_dir=tmp_path / "images",
            output_dir=tmp_path / "out",
            model_dir=tmp_path / "model",
            loop_closure=True,
            viz_enabled=True,
            viz_port=9999,
            max_points=500_000,
            min_views=0,
            mv_rel_thresh=0.01,
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
        patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=mock_creator)),
        patch("collab_splats.wrapper.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ):
        R._run_feedforward(
            backend="vggtx",
            images_dir=tmp_path / "images",
            output_dir=tmp_path / "out",
            model_dir=tmp_path / "model",
            loop_closure={"enabled": True, "submap_size": 32, "submap_overlap": 2},
            viz_enabled=False,
            viz_port=8080,
            max_points=500_000,
            min_views=0,
            mv_rel_thresh=0.01,
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
        patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=mock_creator)),
        patch("collab_splats.wrapper.reconstructor.LoopClosure") as mock_lc_cls,
    ):
        R._run_feedforward(
            backend="vggtx",
            images_dir=tmp_path / "images",
            output_dir=tmp_path / "out",
            model_dir=tmp_path / "model",
            loop_closure={"enabled": False, "submap_size": 32},
            viz_enabled=False,
            viz_port=8080,
            max_points=500_000,
            min_views=0,
            mv_rel_thresh=0.01,
        )

    mock_lc_cls.assert_not_called()


def test_run_feedforward_invalid_lc_knob_raises(tmp_path):
    """An unknown loop_closure knob fails loud with the valid-key list."""
    from collab_splats.wrapper import reconstructor as R

    with (patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=MagicMock())),):
        with pytest.raises(ValueError, match="Invalid pointcloud.loop_closure knob"):
            R._run_feedforward(
                backend="vggtx",
                images_dir=tmp_path / "images",
                output_dir=tmp_path / "out",
                model_dir=tmp_path / "model",
                loop_closure={"bogus_knob": 1},
                viz_enabled=False,
                viz_port=8080,
                max_points=500_000,
                min_views=0,
                mv_rel_thresh=0.01,
            )


def test_run_feedforward_no_viewer_when_viz_disabled(tmp_path):
    """No Viewer instantiated when viz_enabled=False, even with loop_closure=True."""
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock()
    mock_lc_instance = MagicMock(outputs=None)

    with (
        patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=mock_creator)),
        patch("collab_splats.wrapper.reconstructor.LoopClosure", return_value=mock_lc_instance),
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
    ):
        R._run_feedforward(
            backend="vggtx",
            images_dir=tmp_path / "images",
            output_dir=tmp_path / "out",
            model_dir=tmp_path / "model",
            loop_closure=True,
            viz_enabled=False,
            viz_port=8080,
            max_points=500_000,
            min_views=0,
            mv_rel_thresh=0.01,
        )

    mock_viewer_cls.assert_not_called()


def test_run_feedforward_no_loop_closure_no_viewer(tmp_path):
    """loop_closure=False never wraps the creator or attaches viz, even if viz_enabled=True."""
    from collab_splats.wrapper import reconstructor as R

    mock_creator = MagicMock(outputs=None)

    with (
        patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=mock_creator)),
        patch("collab_splats.wrapper.reconstructor.LoopClosure") as mock_lc_cls,
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
    ):
        R._run_feedforward(
            backend="vggtx",
            images_dir=tmp_path / "images",
            output_dir=tmp_path / "out",
            model_dir=tmp_path / "model",
            loop_closure=False,
            viz_enabled=True,
            viz_port=8080,
            max_points=500_000,
            min_views=0,
            mv_rel_thresh=0.01,
        )

    mock_lc_cls.assert_not_called()
    mock_viewer_cls.assert_not_called()


def test_build_localization_db_runs_when_missing(tmp_path):
    from collab_splats.wrapper import reconstructor as R

    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    pc_zarr = rec.backend_dir / "pointcloud.zarr"
    pc_zarr.mkdir(parents=True)
    with (
        patch.object(R, "_localization_db_exists", return_value=False),
        patch.object(R, "_build_localization_db") as build,
    ):
        rec.build_localization_db(overwrite=False)
    # top_k comes from base.yaml's localization.top_k default (pairwise/vismatch fan-out)
    build.assert_called_once_with(pc_zarr, "loma", rec.images_dir, top_k=8, overwrite=False)


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
    assert expected == {"refine", "semantics", "splats", "mesh", "localize", "reconstruction_quality_report"}


def _seed_disk_reconstruction(rec, frame_idxs):
    """Write the pointcloud.zarr a finished run leaves, rows in the given frame order."""
    n = len(frame_idxs)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * n)
    extrinsics[:, 2, 3] = np.arange(n, dtype=np.float32)
    PointcloudResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=None,
        model_intrinsics=np.stack([np.eye(3, dtype=np.float32)] * n),
        image_paths=[Path(f"frame_{fi:06d}") for fi in frame_idxs],
        original_coords=np.array([[0, 0, 6, 4, 6, 4]] * n, dtype=np.float32),
        model_width=6,
        model_height=4,
        depth=np.ones((n, 4, 6), dtype=np.float32),
    ).save_zarr(rec.pointcloud_zarr)


def test_load_pointcloud_from_disk_keeps_zarr_row_order(tmp_path):
    """Loading a finished reconstruction off disk — the whole basis of a leaf-stage re-run."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    # Sparse, non-sorted source indices: a loader that re-sorted rows would scramble the poses
    _seed_disk_reconstruction(rec, [16, 2])

    result = rec._load_pointcloud_from_disk()
    assert [p.name for p in result.image_paths] == ["frame_000016", "frame_000002"]
    np.testing.assert_array_equal(result.extrinsics[:, 2, 3], [0.0, 1.0])

    # The light load leaves the dense per-frame arrays on disk
    assert result.depth is None


def test_stage_output_exists_mesh(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec._stage_output_exists("mesh") is False
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    (rec.backend_dir / "mesh.ply").touch()
    assert rec._stage_output_exists("mesh") is True


def test_stage_output_exists_pointcloud_needs_zarr_and_colmap(tmp_path):
    """
    pointcloud.zarr alone is a partial run; the COLMAP export completes the stage.
    """
    rec = Reconstructor(_make_config(tmp_path))
    rec.pointcloud_zarr.mkdir(parents=True)
    assert rec._stage_output_exists("pointcloud") is False

    rec.colmap_model_dir.mkdir(parents=True)
    assert rec._stage_output_exists("pointcloud") is True


def test_stage_output_exists_semantics_is_per_extractor(tmp_path):
    """The marker is this run's extractor — another extractor's lifted store must not satisfy it."""
    from collab_splats.semantics.utils import lifted_store_path

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
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    with patch.object(R, "_localization_db_exists", return_value=True):
        assert rec._stage_output_exists("localize") is True
    with patch.object(R, "_localization_db_exists", return_value=False):
        assert rec._stage_output_exists("localize") is False


def _seed_pointcloud_markers(rec):
    """Make _stage_output_exists('pointcloud') true without running the stage."""
    rec.colmap_model_dir.mkdir(parents=True, exist_ok=True)
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True, exist_ok=True)


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
    rec.images_dir.mkdir(parents=True, exist_ok=True)
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
    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud")
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh")
    rec.reconstruction_quality_report = lambda overwrite=False: calls.append("reconstruction_quality_report")

    rec.run_pipeline()  # must not raise
    assert calls == ["preproc", "pointcloud", "mesh", "reconstruction_quality_report"]


def test_base_yaml_mesh_has_fidelity_keys():
    """
    The mesh block is nine keys and nothing else — every knob a user can reach.
    """
    cfg = yaml.safe_load((Path(__file__).parents[2] / "configs" / "base.yaml").read_text())
    assert set(cfg["mesh"]) == {
        "enabled",
        "source",
        "voxel_size",
        "sdf_trunc_mult",
        "depth_trunc",
        "conf_percentile",
        "mask_sky",
        "texture",
        "use_convex_hull",
    }
    assert cfg["mesh"]["source"] == "feedforward"
    assert cfg["mesh"]["texture"] is False


########################################
# Report stage
########################################


def test_report_is_appended_with_no_config_boolean_to_turn_it_off(tmp_path):
    """Always on, deliberately against repo precedent, so nothing may gate it.

    Every other diagnostic ships behind a default-false flag, and the one boolean this stage
    would have had is the boolean that keeps it off. The config here disables everything that
    HAS a flag — semantics, mesh, localize, BA — so an appended report is the only
    thing that can follow pointcloud.
    """
    config = _make_config(
        tmp_path,
        {
            "semantics": {"enabled": False},
            "mesh": {"enabled": False},
            "localization": {"enabled": False},
            "pointcloud": {"bundle_adjustment": False},
        },
    )
    rec = Reconstructor(config)
    calls = []
    rec.preprocess = lambda overwrite=False: calls.append("preproc") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud")
    rec.reconstruction_quality_report = lambda overwrite=False: calls.append("reconstruction_quality_report")

    rec.run_pipeline()  # stages=None: the config-derived list
    assert calls == ["preproc", "pointcloud", "reconstruction_quality_report"]


def test_report_stage_dispatches_to_the_report_method_and_forwards_overwrite(tmp_path):
    """A stage in _STAGE_ORDER with no dispatch branch is a silent no-op."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    with patch.object(rec, "reconstruction_quality_report") as report:
        rec.run_pipeline(stages=["reconstruction_quality_report"], overwrite=True)
    report.assert_called_once_with(overwrite=True)


def test_report_output_marker_is_report_json_in_the_backend_dir(tmp_path):
    """The marker makes --stages reconstruction_quality_report refuse existing output."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    assert rec._stage_output_exists("reconstruction_quality_report") is False
    (rec.backend_dir / "reconstruction_quality_report.json").write_text("{}")
    assert rec._stage_output_exists("reconstruction_quality_report") is True
    with pytest.raises(ValueError, match="already exists"):
        rec.run_pipeline(stages=["reconstruction_quality_report"])


def _save_tiny_zarr(rec, n=3, with_confidence=True, confidence=None):
    """A slanted-plane pointcloud.zarr at rec.pointcloud_zarr, so the report stage runs for real."""
    if confidence is None and with_confidence:
        confidence = np.full((n, 16, 16), 0.9, np.float32)
    hw = 16
    rows = np.arange(hw, dtype=np.float32)[:, None]
    base = np.broadcast_to(3.0 + 0.1 * rows, (hw, hw)).astype(np.float32)
    extrinsics = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k in range(n):
        extrinsics[k][0, 3] = -0.15 * k  # sideways baseline, so pairs have parallax
    K = np.array([[50.0, 0, hw / 2], [0, 50.0, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    PointcloudResult(
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
        extrinsics=extrinsics,
        intrinsics=None,
        model_intrinsics=np.stack([K] * n),
        image_paths=[Path(f"frame_{4 * k:06d}.png") for k in range(n)],
        original_coords=np.tile(np.array([4, 2, 20, 18, 32, 24], dtype=np.float32), (n, 1)),
        model_width=hw,
        model_height=hw,
        depth=np.stack([base * (1.0 + 0.01 * k) for k in range(n)]),
        confidence=confidence,
    ).save_zarr(rec.pointcloud_zarr)


def _texture_frames(n=3, width=32, height=24):
    """Textured RGB keyframes on the fixture's original canvas, shifted per frame."""
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    out = []
    for k in range(n):
        x = xx + 1.5 * k
        rgb = np.stack([127 + 100 * np.sin(0.7 * x) * np.cos(0.5 * yy),
                        127 + 90 * np.cos(0.45 * x + 0.3 * yy),
                        127 + 80 * np.sin(0.25 * x - 0.6 * yy)], axis=-1)
        out.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.stack(out)


def test_report_json_is_the_columnar_contract(tmp_path):
    """Top-level keys and scene block; nan ships as null and no .json.tmp is left."""
    rec = Reconstructor(_make_config(tmp_path))
    rec._resolve_result = lambda: object()
    _save_tiny_zarr(rec, with_confidence=False)

    out = rec.reconstruction_quality_report()

    text = out.read_text()
    report = json.loads(text)
    assert set(report) == {
        "scene",
        "frames",
        "depth_pairs",
        "depth_residual_histogram",
        "photometric_pairs",
    }
    assert report["scene"]["n_frames"] == 3
    assert report["scene"]["model_resolution"] == "16x16"
    assert report["scene"]["image_width"] == 32
    assert report["frames"]["frame_idx"] == [0, 4, 8]
    assert report["frames"]["confidence_median"] == [None, None, None]
    assert "NaN" not in text
    assert not out.with_suffix(".json.tmp").exists()


def test_report_refuses_a_stale_old_format_report(tmp_path):
    """A pre-columnar report on disk must raise on reuse, not be served as current."""
    rec = Reconstructor(_make_config(tmp_path))
    out = rec.backend_dir / "reconstruction_quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"measurements": {}}')

    with pytest.raises(ValueError, match="stale"):
        rec.reconstruction_quality_report()


def test_report_writes_nan_as_null(tmp_path):
    """An all-nan confidence frame has a nan median; the file must carry null, never a bare NaN."""
    rec = Reconstructor(_make_config(tmp_path))
    rec._resolve_result = lambda: object()
    confidence = np.full((3, 16, 16), 0.9, np.float32)
    confidence[1] = np.nan
    _save_tiny_zarr(rec, confidence=confidence)

    text = rec.reconstruction_quality_report().read_text()

    assert "NaN" not in text
    assert json.loads(text)["frames"]["confidence_median"] == [pytest.approx(0.9), None, pytest.approx(0.9)]


def test_report_runs_photometric_when_images_exist(tmp_path, monkeypatch):
    """Keyframes present: photometric_pairs is a filled table, not null."""
    rec = Reconstructor(_make_config(tmp_path))
    rec._resolve_result = lambda: object()
    _save_tiny_zarr(rec)
    monkeypatch.setattr(fr, "frame_paths", lambda d: [Path(f"frame_{4 * k:06d}.png") for k in range(3)])
    monkeypatch.setattr(fr, "read_frames", lambda d: _texture_frames())

    report = json.loads(rec.reconstruction_quality_report().read_text())
    assert report["photometric_pairs"]["idx1"]


def test_report_raises_when_photometric_raises(tmp_path, monkeypatch):
    """A failing measurement fails the report; nothing is caught."""
    rec = Reconstructor(_make_config(tmp_path))
    rec._resolve_result = lambda: object()
    _save_tiny_zarr(rec)
    monkeypatch.setattr(fr, "frame_paths", lambda d: [Path(f"frame_{4 * k:06d}.png") for k in range(3)])
    monkeypatch.setattr(fr, "read_frames", lambda d: _texture_frames())

    def _boom(*args, **kwargs):
        raise RuntimeError("photometric exploded")

    monkeypatch.setattr(metrics, "compute_photometric_ncc", _boom)
    with pytest.raises(RuntimeError, match="photometric exploded"):
        rec.reconstruction_quality_report()


def test_report_skips_without_overwrite_and_never_touches_the_reconstruction(tmp_path):
    """Skip is checked BEFORE the result is resolved, so a re-run costs nothing."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    out = rec.backend_dir / "reconstruction_quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"frames": {}}')

    def _explode():
        raise AssertionError("_resolve_result must not run when the report already exists")

    rec._resolve_result = _explode
    assert rec.reconstruction_quality_report() == out
    assert out.read_text() == '{"frames": {}}'


def test_stale_preprocessing_key_is_refused(tmp_path):
    """
    A pre-2026-08-22 config carries `preprocessing:`; it must raise, not be ignored.
    """
    from collab_splats.wrapper.reconstructor import Reconstructor

    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega", "loop_closure": False},
        "preprocessing": {"frame_selection": "fps", "fps": 1.0},
    }

    with pytest.raises(ValueError, match="renamed to 'preproc'"):
        Reconstructor.validate_config(config)


def test_nerfstudio_method_rejected():
    """
    pointcloud.method=nerfstudio is gone — validation only knows feedforward and sfm.
    """
    from collab_splats.wrapper.reconstructor import _VALID_METHODS

    assert _VALID_METHODS == {"feedforward", "sfm"}
    assert not hasattr(Reconstructor, "_run_nerfstudio")


def test_extract_frames_writes_video_quality_pngs(tmp_path, monkeypatch):
    """
    The video branch renders both report PNGs beside images/; the dir branch none.
    """
    from collab_splats.wrapper import reconstructor as R

    rng = np.random.default_rng(0)
    n = 20
    columns = (
        "blur",
        "laplacian",
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    )
    report = {
        "video": {"path": "/data/clip.mp4", "fps": 10.0, "total_frames": n, "width": 64, "height": 48},
        "params": {"motion_stride": 2},
        "frames": {"frame_idx": list(range(n)), **{k: rng.uniform(0, 1, n).tolist() for k in columns}},
        "pairs": {
            "frame_idx_a": list(range(0, 20, 2)),
            "frame_idx_b": list(range(2, 22, 2)),
            "n_matches": [10] * 10,
            "translation_px": [1.0] * 9 + [None],
            "parallax": [0.5] * 9 + [None],
        },
    }
    two_frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 2
    two_records = [{"frame_idx": 3, "blur_score": 1.0}, {"frame_idx": 7, "blur_score": 1.0}]
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: report)
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": n})
    monkeypatch.setattr(R, "sample_fps", lambda path, **kw: (two_frames, two_records))

    out = tmp_path / "scene"
    video = tmp_path / "v.mp4"
    video.touch()
    R.extract_frames(video, out / "images", "fps", 1.0, None, 50)

    expected = {"photometric.png", "motion.png"}
    assert {p.name for p in out.glob("*.png")} == expected
    for name in expected:
        assert (out / name).read_bytes()[:4] == b"\x89PNG"

    # Image directory: no report, no plots
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    out2 = tmp_path / "scene2"
    R.extract_frames(img_dir, out2 / "images", "fps", 1.0, None, 50)
    assert list(out2.glob("*.png")) == []


def test_undistort_provenance_records_the_camera_not_a_profile(tmp_path, monkeypatch):
    # Provenance carries pycolmap's cameras and no roi: the alpha=0 crop is gone, so
    # there is no crop offset left for a reader to reapply to the principal point.
    camera = pycolmap.Camera(model="OPENCV", width=64, height=48, params=[60.0, 60.0, 32.0, 24.0, -0.2, 0.0, 0.0, 0.0])

    # autospec, not a bare lambda: calibrate_camera takes the images DIRECTORY, and a
    # signature-blind stub stays green through a wrong call site
    stub = create_autospec(calibrate_camera, return_value=camera)
    monkeypatch.setattr(R, "calibrate_camera", stub)

    prov = {}
    out = R._apply_undistortion(np.zeros((3, 48, 64, 3), np.uint8), tmp_path, prov)

    stub.assert_called_once_with(tmp_path)
    assert "roi" not in prov["undistort"] and "profile" not in prov["undistort"]
    assert prov["undistort"]["camera"]["model"] == "OPENCV"
    assert prov["undistort"]["undistorted_camera"]["model"] == "PINHOLE"
    assert out.shape[1:3] == (
        prov["undistort"]["undistorted_camera"]["height"],
        prov["undistort"]["undistorted_camera"]["width"],
    )
