import contextlib
import hashlib
import itertools
import json
import re
import weakref
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import pytest
import torch
import yaml
import zarr
from mergedeep import merge

from collab_splats import reconstructor as R
from collab_splats.geometry import metrics
from collab_splats.geometry.loop_closure.wrapper import LoopClosureConfig
from collab_splats.mesh import create_tsdf_mesh
from collab_splats.pointcloud import utils as pointcloud_utils
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.preproc import frames as fr
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
from collab_splats.reconstructor import STAGES, Reconstructor
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.store import write_feature_cache, write_point_features
from tests.reconstructor._stubs import minimal_feedforward_result, stub_creator_cls


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
    src = Path("collab_splats/reconstructor.py").read_text()
    # Capture the default expression of each two-arg .get("key", <default>)
    defaults = re.findall(r"\.get\(\s*['\"][^'\"]+['\"]\s*,\s*([^)]+)\)", src)
    # Structural {}/[] defaults are allowed; so is ocr_lens's model_id, a constructor default not in base.yaml
    allowed = ("{}", "[]", '"llava-hf/llava-v1.6-vicuna-7b-hf"')
    offenders = [d.strip() for d in defaults if d.strip() not in allowed]
    assert offenders == [], f"inline value defaults still present: {offenders}"


def _video_reconstructor(tmp_path, preproc):
    """Reconstructor over an empty video file, with the given preproc overrides."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "video.mp4"
    video.touch()
    return Reconstructor(_make_config(tmp_path, {"preproc": preproc}))


def _stub_plots(monkeypatch):
    """Plots are not under test here, and a stub report carries no columns to draw."""
    monkeypatch.setattr(R, "plot_photometric", lambda *a, **k: None)
    monkeypatch.setattr(R, "plot_motion", lambda *a, **k: None)


def test_preproc_dispatches_per_frame_selection(tmp_path, monkeypatch):
    """
    Each frame_selection value reaches its own sampler with only its own knobs.
    """
    calls = {}

    def _recorder(name):
        def fake(path, **kwargs):
            calls.clear()
            calls["sampler"] = name
            calls.update(kwargs)
            return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

        return fake

    # Stub the shared report and assert every branch forwards the same object
    report = {"frames": {}}
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: report)
    monkeypatch.setattr(R, "sample_fps", _recorder("fps"))
    monkeypatch.setattr(R, "sample_uniform", _recorder("uniform"))
    monkeypatch.setattr(R, "sample_optical_flow", _recorder("optical_flow"))
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    _stub_plots(monkeypatch)

    # fps: rate + both band bounds
    rec = _video_reconstructor(
        tmp_path / "a", {"frame_selection": "fps", "fps": 2.0, "min_frames": 5, "max_frames": 50}
    )
    rec.preproc()
    quality = rec.config["preproc"]["quality"]
    workers = rec.config["preproc"]["n_workers"]
    assert calls == {
        "sampler": "fps",
        "workers": workers,
        "fps": 2.0,
        "min_frames": 5,
        "max_frames": 50,
        "report": report,
        "quality": quality,
        "on_empty_slot": "rescue",
    }

    # uniform: max_frames is the count, with no fps and no floor
    rec = _video_reconstructor(tmp_path / "b", {"frame_selection": "uniform", "min_frames": 5, "max_frames": 50})
    rec.preproc()
    assert calls == {"sampler": "uniform", "workers": workers, "max_frames": 50, "report": report, "quality": quality}

    # optical_flow: max_frames caps the selector
    rec = _video_reconstructor(tmp_path / "c", {"frame_selection": "optical_flow", "min_frames": 5, "max_frames": 50})
    rec.preproc()
    assert calls == {"sampler": "optical_flow", "max_frames": 50, "report": report, "quality": quality}


def test_preproc_forwards_quality_overrides_to_every_sampler(tmp_path, monkeypatch):
    """
    preproc.quality is a shared knob: each branch hands it to its own sampler.
    """
    seen = []

    def fake(path, **kwargs):
        seen.append(kwargs["quality"])
        return [np.zeros((4, 4, 3), dtype=np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    for name in ("sample_fps", "sample_uniform", "sample_optical_flow"):
        monkeypatch.setattr(R, name, fake)

    _stub_plots(monkeypatch)

    overrides = {"sharpness_k": 1.0, "max_clipped_frac": 0.1}
    for i, selection in enumerate(("fps", "uniform", "optical_flow")):
        rec = _video_reconstructor(tmp_path / str(i), {"frame_selection": selection, "quality": overrides})
        rec.preproc()

    assert seen == [overrides] * 3


def test_preproc_forwards_on_empty_slot_to_fps_only(tmp_path, monkeypatch):
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

    _stub_plots(monkeypatch)

    for i, selection in enumerate(("fps", "uniform", "optical_flow")):
        rec = _video_reconstructor(tmp_path / str(i), {"frame_selection": selection, "on_empty_slot": "drop"})
        rec.preproc()

    assert seen["sample_fps"]["on_empty_slot"] == "drop"
    assert "on_empty_slot" not in seen["sample_uniform"] and "on_empty_slot" not in seen["sample_optical_flow"]


def test_preproc_rejects_unknown_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})

    rec = _video_reconstructor(tmp_path, {"frame_selection": "balanced"})
    with pytest.raises(ValueError, match="frame_selection"):
        rec.preproc()


def test_preproc_refuses_an_empty_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": 100})
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"frames": {}})
    monkeypatch.setattr(R, "sample_fps", lambda path, **kw: ([], []))

    rec = _video_reconstructor(tmp_path, {"frame_selection": "fps"})
    with pytest.raises(ValueError, match="0 of 100 frames selected"):
        rec.preproc()


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


def test_validate_config_ignores_mesher(tmp_path):
    """mesh.mesher is no longer a validated knob — its absence never raises."""
    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
    }
    # Reaches validate via __init__ (base-merged); must not raise on missing mesher
    rec = Reconstructor(config)
    assert "mesher" not in rec.config["mesh"]


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


def test_run_skips_preproc_if_images_dir_exists(tmp_path):
    """Skip extraction when images/ already exists and overwrite=False."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)

    with patch.object(Reconstructor, "preproc") as mock_preproc:
        rec.run(["preproc"])

    mock_preproc.assert_not_called()


def test_run_calls_preproc_if_images_dir_missing(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with patch.object(Reconstructor, "preproc") as mock_preproc:
        rec.run(["preproc"])

    mock_preproc.assert_called_once()


def test_preprocess_overwrite_reruns(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)

    with patch.object(Reconstructor, "preproc") as mock_preproc:
        rec.run(["preproc"], overwrite=True)

    mock_preproc.assert_called_once()


def _make_mock_pointcloud_result(tmp_path):
    """Minimal PointcloudResult mock for testing — avoids importing collab_splats.pointcloud."""
    result = MagicMock()
    result.image_paths = [tmp_path / "images" / "frame_0001.jpg"]
    # Empty (0, 3) points/colors, as consumers expect; a bare MagicMock yields a malformed (0,) array
    result.points = np.zeros((0, 3), dtype=np.float32)
    result.colors = np.zeros((0, 3), dtype=np.uint8)
    return result


def test_run_skips_pointcloud_if_colmap_and_zarr_exist(tmp_path):
    """Skip rebuild only when BOTH the colmap model dir and pointcloud.zarr exist."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)
    rec.colmap_model_dir.mkdir(parents=True)

    # The pointcloud zarr marker is also required; colmap alone no longer skips.
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True)

    with patch("collab_splats.reconstructor.get_creator") as mock_get:
        rec.run(["pointcloud"])

    mock_get.assert_not_called()


def test_pointcloud_stage_feedforward_vggtx(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    creator_cls = stub_creator_cls(mock_result)

    with patch("collab_splats.reconstructor.get_creator", return_value=creator_cls) as mock_get:
        rec.pointcloud()

    # Creator built from the shared knobs, preproc's frames (none without preproc) and the backend block
    mock_get.assert_called_once_with("vggtx")
    pc = rec.config["pointcloud"]
    assert creator_cls.call_args.kwargs == {
        "max_points": pc["max_points"],
        "min_views": pc["min_views"],
        "mv_rel_thresh": pc["mv_rel_thresh"],
        "clean": False,
        "frames": None,
        **pc["vggtx"],
    }
    creator_cls.return_value.create_pointcloud.assert_called_once_with(
        rec.images_dir, rec.backend_dir, rec.colmap_model_dir
    )
    mock_result.save_zarr.assert_called_once_with(
        rec.pointcloud_zarr, extra_attrs={"method": "feedforward", "backend": "vggtx"}
    )
    mock_result.write_ply.assert_called_once_with(rec.backend_dir / "sparse_pc.ply")
    assert rec._result is None


def test_pointcloud_stage_frees_the_dense_arrays_after_save(tmp_path):
    """
    Nothing dense stays alive once pointcloud.zarr is written.
    """
    rec = Reconstructor(_make_config(tmp_path))
    refs = {}

    # A real result held on self.outputs, tracked by weakrefs only so the test keeps nothing alive
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
        rec.pointcloud()

    # Freed by refcount alone, no gc.collect: nothing, creator included, still holds them
    assert {name: ref() is None for name, ref in refs.items()} == dict.fromkeys(refs, True)
    assert rec.result.points.shape == (5, 3)

    # The dense arrays live on in the zarr the downstream stages read
    assert zarr.open(str(rec.pointcloud_zarr), mode="r")["depth"].shape == (2, 8, 8)


def test_pointcloud_stage_method_dir_routing(tmp_path):
    """mapanything backend → out/mapanything/."""
    config = _make_config(tmp_path, {"pointcloud": {"backend": "mapanything"}})
    rec = Reconstructor(config)
    assert rec.backend_dir == tmp_path / "out" / "mapanything"


def _touch_frames(rec, frame_idxs):
    """Empty images/ frames: frame_paths only lists and sorts, nothing decodes them."""
    rec.images_dir.mkdir(parents=True, exist_ok=True)

    for fi in frame_idxs:
        (rec.images_dir / f"frame_{fi:06d}.png").touch()


def _seed_codes(rec, name, maps, latent_dim=None, extractor_kwargs=None):
    """A valid <name>_codes.zarr over rec's images/, as a finished extract + encode leaves it."""
    path = rec.semantics_cache_dir / f"{name}_codes.zarr"
    attrs = {
        "extractor": name,
        "patch_size": 14,
        "n_frames": len(maps),
        "extractor_kwargs": extractor_kwargs or {},
        "latent_dim": latent_dim,
    }
    write_feature_cache(path, iter(torch.from_numpy(m) for m in maps), len(maps), attrs)
    return path


def test_semantics_valid_cache_skips_the_extractor(tmp_path):
    """A codes store valid for this extractor, kwargs and width is lifted without building the extractor."""
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    _touch_frames(rec, [0, 1])
    _seed_disk_reconstruction(rec, [0, 1])
    codes_path = _seed_codes(rec, "dinov2", [np.zeros((4, 2, 2), np.float16)] * 2)

    with (
        patch.object(R, "BaseFeatureExtractor") as extractor_base,
        patch.object(R, "write_feature_cache") as write,
        patch.object(R, "lift_features", return_value=torch.zeros(1, 4)),
        patch.object(R, "write_point_features"),
    ):
        rec.semantics()

    extractor_base.get.assert_not_called()
    write.assert_not_called()
    assert codes_path.exists()


def test_semantics_invalid_cache_extracts_with_extractor_kwargs(tmp_path):
    """semantics.extractor_kwargs reaches the extractor build and the codes store's attrs."""
    semantics = {"enabled": True, "extractor": "dinov2", "extractor_kwargs": {"layer": 20}, "n_components": None}
    rec = _semantics_rec(tmp_path, semantics)

    with _stub_extraction() as extractor_base:
        rec.semantics()

    extractor_base.get.assert_called_once_with("dinov2")
    extractor_base.get.return_value.assert_called_once_with(layer=20)
    attrs = dict(zarr.open(str(rec.semantics_cache_dir / "dinov2_codes.zarr"), mode="r").attrs)
    assert attrs["extractor_kwargs"] == {"layer": 20}
    assert attrs["latent_dim"] is None


def test_run_refuses_semantics_if_lifted_exists(tmp_path):
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    lifted_dir = rec.backend_dir / "semantics"
    lifted_dir.mkdir(parents=True)
    (lifted_dir / "dinov2_lifted.zarr").mkdir()

    with (
        patch.object(R, "write_feature_cache") as write,
        patch.object(R, "lift_features") as lift,
        pytest.raises(ValueError, match="already exists"),
    ):
        rec.run(["semantics"])

    write.assert_not_called()
    lift.assert_not_called()


def _semantics_rec(tmp_path, semantics):
    """A Reconstructor over two images/ frames and a one-point reconstruction."""
    rec = Reconstructor(_make_config(tmp_path, {"semantics": semantics}))
    _touch_frames(rec, [0, 1])
    _seed_disk_reconstruction(rec, [0, 1])
    return rec


@contextlib.contextmanager
def _stub_extraction(dim=32):
    """
    Extractor, image decode and lift stubbed; frame k's map is constant k + 1 over (dim, 2, 2).

    - lift returns frame 0's first cell repeated per target row, so its width is the loader's
    """
    calls = []

    def forward(frames):
        first = len(calls)
        calls.extend(frames)
        return [torch.full((dim, 2, 2), float(first + j + 1)) for j in range(len(frames))]

    def lift(frame_features, result):
        return frame_features(0)[:, 0, 0].float().cpu().repeat(len(result.points), 1)

    with (
        patch.object(R, "BaseFeatureExtractor") as extractor_base,
        patch.object(R, "read_image", return_value=np.zeros((4, 4, 3), np.uint8)),
        patch.object(R, "lift_features", side_effect=lift),
    ):
        extractor = extractor_base.get.return_value.return_value
        extractor.forward.side_effect = forward
        extractor.patch_size = 14
        yield extractor_base


COMPRESSED = {"extractor": "dinov2", "n_components": 8, "max_epochs": 1, "target_cosine": None}


def test_semantics_encodes_every_frame_and_deletes_the_features(tmp_path, monkeypatch):
    rec = _semantics_rec(tmp_path, COMPRESSED)
    written = []
    real_write = R.write_feature_cache
    monkeypatch.setattr(
        R, "write_feature_cache", lambda path, *a, **k: (written.append(path), real_write(path, *a, **k))
    )

    with _stub_extraction():
        rec.semantics()

    features_path = rec.semantics_cache_dir / "dinov2_features.zarr"
    codes_path = rec.semantics_cache_dir / "dinov2_codes.zarr"
    assert written == [features_path, codes_path]
    assert not features_path.exists()

    codes = zarr.open(str(codes_path), mode="r")["features"]
    assert codes.shape == (2, 8, 2, 2) and codes.dtype == np.float16
    assert (codes_path / "autoencoder.pt").is_file()

    # Row k holds frame k's encoding; the stub's frame k is constant k + 1
    ae = FeatureAutoencoder.load(codes_path / "autoencoder.pt").cpu()

    for k in range(2):
        with torch.no_grad():
            expected = ae.encode(torch.full((32, 2, 2), float(k + 1))).numpy()

        np.testing.assert_allclose(codes[k].astype(np.float32), expected, rtol=1e-2, atol=1e-3)


def test_semantics_writes_weights_inside_the_lifted_store(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    out_dir = rec.backend_dir / "semantics"
    assert rec.done("semantics")
    assert (out_dir / "dinov2_lifted.zarr" / "autoencoder.pt").is_file()
    store = zarr.open(str(out_dir / "dinov2_lifted.zarr"), mode="r")
    assert store["features"].shape == (1, 8) and store["features"].dtype == np.float16
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 8, "extractor": "dinov2", "extractor_kwargs": {}}


def test_semantics_second_run_neither_extracts_nor_fits(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    with _stub_extraction() as extractor_base, patch.object(FeatureAutoencoder, "fit") as fit:
        rec.semantics()

    extractor_base.get.assert_not_called()
    fit.assert_not_called()


def test_semantics_n_components_change_re_extracts(tmp_path):
    rec = _semantics_rec(tmp_path, COMPRESSED)

    with _stub_extraction():
        rec.semantics()

    rec.config["semantics"]["n_components"] = 4

    with _stub_extraction() as extractor_base:
        rec.semantics()

    extractor_base.get.assert_called_once()
    stored = FeatureAutoencoder.load(rec.semantics_cache_dir / "dinov2_codes.zarr" / "autoencoder.pt")
    assert stored.latent_dim == 4


def test_semantics_uncompressed_writes_full_width_codes_without_features(tmp_path, monkeypatch):
    rec = _semantics_rec(tmp_path, {"extractor": "dinov2", "n_components": None})
    written = []
    real_write = R.write_feature_cache
    monkeypatch.setattr(
        R, "write_feature_cache", lambda path, *a, **k: (written.append(path), real_write(path, *a, **k))
    )

    with _stub_extraction():
        rec.semantics()

    codes_path = rec.semantics_cache_dir / "dinov2_codes.zarr"
    assert written == [codes_path]
    assert zarr.open(str(codes_path), mode="r")["features"].shape == (2, 32, 2, 2)
    assert not (codes_path / "autoencoder.pt").exists()


def test_load_frame_maps_pointcloud_frame_to_store_row(tmp_path):
    """_load_frame reads store row rows[i]; with an AE it returns (r, H_p, W_p) codes."""
    data = np.stack([np.full((4, 2, 2), i, np.float16) for i in range(3)])
    cache = zarr.open(str(tmp_path / "c.zarr"), mode="w")
    cache["features"] = data
    features = cache["features"]

    raw = R._load_frame(features, [2, 0], None, 0)
    assert raw.dtype == torch.float16 and float(raw[0, 0, 0]) == 2.0

    ae = FeatureAutoencoder(input_dim=4, latent_dim=2)
    codes = R._load_frame(features, [2, 0], ae, 0)
    assert codes.shape == (2, 2, 2) and not codes.requires_grad

    # Frame 0 is store row 2 (all 2s), not row 0 (all 0s)
    with torch.no_grad():
        from_row2 = ae.encode(torch.full((4, 2, 2), 2.0))
        from_row0 = ae.encode(torch.zeros(4, 2, 2))
    assert torch.allclose(codes, from_row2)
    assert not torch.allclose(from_row2, from_row0)


def test_semantics_uncompressed_writes_full_dim_and_no_weights(tmp_path):
    """n_components: null is supported: full-dim codes, no autoencoder, attrs say so."""
    rec = _semantics_rec(tmp_path, {"extractor": "dinov2", "n_components": None})

    with _stub_extraction():
        rec.semantics()

    out_dir = rec.backend_dir / "semantics"
    assert not (out_dir / "dinov2_lifted.zarr" / "autoencoder.pt").exists()
    store = zarr.open(str(out_dir / "dinov2_lifted.zarr"), mode="r")
    assert np.asarray(store["features"]).shape == (1, 32)
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 32, "extractor": "dinov2", "extractor_kwargs": {}}


def test_run_skips_mesh_if_ply_exists(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True}})
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()
    calls = []
    rec.reconstruction_quality_report = lambda: calls.append("reconstruction_quality_report")

    with patch("collab_splats.reconstructor.create_tsdf_mesh") as mock_fuse:
        rec.run()

    mock_fuse.assert_not_called()
    assert calls == ["reconstruction_quality_report"]


def test_mesh_done_check_matches_tsdf_writer_filename(tmp_path):
    """done("mesh") fires on the file the real TSDF writer actually laid down.

    Neither filename is hardcoded here: the mesher writes the file and done() looks for it,
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
    # Pin the writer side separately, so a writer regression is not read as a reader one
    assert written.exists()

    # The marker is that exact file, so run() skips the stage instead of re-running TSDF
    assert rec.outputs["mesh"] == written
    assert rec.done("mesh") is True


def test_mesh_runs_tsdf(tmp_path, monkeypatch):
    fuse = _mesh_fuse(tmp_path, monkeypatch, _tsdf_mesh_ff(model_hw=(16, 16)))

    fuse.assert_called_once()
    R.clean_repair_mesh.assert_called_once()


def _tsdf_mesh_ff(n=2, model_hw=(8, 8)):
    """
    PointcloudResult as loaded off pointcloud.zarr: model-grid depth, full-res K, distinctive poses.
    """
    H, W = model_hw

    # Full-res K: 2x the model grid, principal point outside a W-wide image
    K_orig = np.eye(3, dtype=np.float32)[None].repeat(n, axis=0)
    K_orig[:, 0, 0] = K_orig[:, 1, 1] = 2.0 * W
    K_orig[:, 0, 2] = W
    K_orig[:, 1, 2] = H

    # Model K on the model grid
    K_model = np.eye(3, dtype=np.float32)[None].repeat(n, axis=0)
    K_model[:, 0, 0] = K_model[:, 1, 1] = float(W)
    K_model[:, 0, 2] = W / 2
    K_model[:, 1, 2] = H / 2

    # Translated poses, so a c2w/w2c mix-up is visible
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(n, axis=0)
    extrinsics[:, 0, 3] = 7.0

    return PointcloudResult(
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=K_orig,
        model_intrinsics=K_model,
        image_paths=[Path(f"frame_{i:06d}.png") for i in range(n)],
        original_coords=np.tile([0, 0, 2 * W, 2 * H, 2 * W, 2 * H], (n, 1)).astype(np.float32),
        model_width=W,
        model_height=H,
        depth=np.ones((n, H, W), dtype=np.float32),
    )


def _unit_upsample(depths, rgbs, boxes):
    """
    Frame-grid lift stand-in: unit depth on the 32x32 frame grid.
    """
    return np.ones((2, 32, 32), np.float32)


def _mesh_fuse(tmp_path, monkeypatch, ff, upsample=_unit_upsample, **mesh_overrides):
    """
    Run rec.mesh() over a zarr double and return the create_tsdf_mesh mock.

    - upsample stands in for the frame-grid lift inside frame_depths
    """
    mesh_cfg = {"enabled": True, "source": "feedforward", "voxel_size": 0.01, **mesh_overrides}
    rec = Reconstructor(_make_config(tmp_path, {"mesh": mesh_cfg}))

    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    monkeypatch.setattr(pointcloud_utils, "upsample_depths", upsample)
    fuse = MagicMock(return_value=tmp_path / "mesh.ply")
    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())
    monkeypatch.setattr(R, "prepare_mesh", lambda mesh, **kw: mesh)

    rec.mesh()

    return fuse


def test_mesh_fuses_full_res_intrinsics(tmp_path, monkeypatch):
    """
    Model-res depth is lifted onto the frame grid, so the full-res `intrinsics` is the right K.

    Pairing the model grid's depth with the original grid's K is the 2026-08-11 collapse bug
    (5.06M -> 75k vertices); this pins the pairing that fixed it.
    """
    ff = _tsdf_mesh_ff(model_hw=(16, 16))

    fuse = _mesh_fuse(tmp_path, monkeypatch, ff)

    np.testing.assert_array_equal(fuse.call_args.args[3], ff.intrinsics)


def test_mesh_passes_use_convex_hull_to_cleaning(tmp_path, monkeypatch):
    """
    mesh.use_convex_hull reaches clean_repair_mesh, and is on unless turned off.
    """
    for flag in (False, True):
        overrides = {} if flag else {"use_convex_hull": False}
        _mesh_fuse(tmp_path, monkeypatch, _tsdf_mesh_ff(model_hw=(16, 16)), **overrides)
        assert R.clean_repair_mesh.call_args.kwargs == {"use_convex_hull": flag}


def test_mesh_sdf_trunc_mult_scales_the_truncation_band(tmp_path, monkeypatch):
    """
    sdf_trunc reaches create_tsdf_mesh as sdf_trunc_mult x voxel_size, defaulting to 4x.

    - the multiplier, not voxel_size, sets the thin-structure floor: a TSDF cancels anything
      thinner than 2 x sdf_trunc, so the default band is 8 voxels wide
    - measured on GH010229 at voxel 0.2 (scene diagonal 262 world units): the default floor is
      1.60 units, while the voxel grid alone would resolve 0.40
    """
    default = _mesh_fuse(tmp_path, monkeypatch, _tsdf_mesh_ff(model_hw=(16, 16))).call_args.kwargs
    assert default["sdf_trunc"] == pytest.approx(0.04)

    narrow = _mesh_fuse(tmp_path, monkeypatch, _tsdf_mesh_ff(model_hw=(16, 16)), sdf_trunc_mult=1.5)
    assert narrow.call_args.kwargs["sdf_trunc"] == pytest.approx(0.015)
    assert narrow.call_args.kwargs["voxel_size"] == pytest.approx(0.01)


def test_mesh_uses_zarr_poses(tmp_path, monkeypatch):
    """
    Fusion poses are the zarr's extrinsics inverted to c2w — refined or loop-closed on disk.
    """
    ff = _tsdf_mesh_ff(model_hw=(16, 16))

    fuse = _mesh_fuse(tmp_path, monkeypatch, ff)

    np.testing.assert_allclose(fuse.call_args.args[2], np.linalg.inv(ff.extrinsics), atol=1e-5)


def test_mesh_masks_depth_by_confidence(tmp_path, monkeypatch):
    """
    conf_percentile zeroes the depth under the percentile before it reaches the fusion.
    """
    ff = _tsdf_mesh_ff(model_hw=(16, 16))
    ff.confidence = np.tile(np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16), (2, 1, 1))
    seen = {}

    def spy_upsample(depths, rgbs, boxes):
        seen["depths"] = depths
        return np.ones((2, 32, 32), np.float32)

    _mesh_fuse(tmp_path, monkeypatch, ff, upsample=spy_upsample, conf_percentile=20)

    # The bottom 20% of a linear ramp is zeroed, the rest is untouched
    masked = seen["depths"]
    assert (masked == 0).mean() == pytest.approx(0.2, abs=0.02)
    assert masked.max() == ff.depth.max()


def _run_prepared_mesh(tmp_path, monkeypatch, texture):
    """
    Run rec.mesh() with fusion writing a real mesh; return (rec, cleaned, prepared, prepare, texture mocks).
    """
    mesh_cfg = {"enabled": True, "source": "feedforward", "voxel_size": 0.01, "texture": texture}
    rec = Reconstructor(_make_config(tmp_path, {"mesh": mesh_cfg}))
    ff = _tsdf_mesh_ff(model_hw=(16, 16))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    monkeypatch.setattr(pointcloud_utils, "upsample_depths", _unit_upsample)

    # Fusion writes a real sphere to mesh.ply; cleaning is a no-op on it
    cleaned = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)
    mesh_path = rec.backend_dir / "mesh.ply"

    def fuse(*args, **kwargs):
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_path), cleaned)
        return mesh_path

    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())

    # prepare_mesh returns a box, so mesh.ply's triangle count tells which mesh was written
    prepared = o3d.geometry.TriangleMesh.create_box()
    prepare = MagicMock(return_value=prepared)
    monkeypatch.setattr(R, "prepare_mesh", prepare)
    texture_mesh = MagicMock()
    monkeypatch.setattr(R, "create_texture_mesh", texture_mesh)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    rec.mesh()

    return rec, cleaned, prepared, prepare, texture_mesh


def test_mesh_writes_the_prepared_mesh_without_texture(tmp_path, monkeypatch):
    """mesh.ply is prepare_mesh's output even with texture off, prepared from the cleaned mesh."""
    rec, cleaned, prepared, prepare, texture_mesh = _run_prepared_mesh(tmp_path, monkeypatch, texture=False)

    assert len(prepare.call_args.args[0].triangles) == len(cleaned.triangles)
    assert prepare.call_args.kwargs == {"voxel_size": 0.01, "smooth_iterations": 0}
    written = o3d.io.read_triangle_mesh(str(rec.backend_dir / "mesh.ply"))
    assert len(written.triangles) == len(prepared.triangles)
    texture_mesh.assert_not_called()


def test_mesh_textures_the_prepared_mesh_behind_the_cleaned_occluder(tmp_path, monkeypatch):
    """Texturing unwraps the same mesh.ply geometry; the unfilled cleaned mesh is the occluder."""
    rec, cleaned, prepared, _, texture_mesh = _run_prepared_mesh(tmp_path, monkeypatch, texture=True)

    mesh, occluder, out_dir = texture_mesh.call_args.args[:3]
    assert mesh is prepared
    assert len(occluder.triangles) == len(cleaned.triangles)
    assert out_dir == rec.backend_dir / "texture"
    assert texture_mesh.call_args.kwargs == {"voxel_size": 0.01}


def test_mesh_texture_without_cuda_raises_before_fusion(tmp_path, monkeypatch):
    """texture on a CPU-only machine fails at stage start, not after fusing and cleaning."""
    mesh_cfg = {"enabled": True, "source": "feedforward", "voxel_size": 0.01, "texture": True}
    rec = Reconstructor(_make_config(tmp_path, {"mesh": mesh_cfg}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    load = MagicMock()
    fuse = MagicMock()
    monkeypatch.setattr(PointcloudResult, "load_zarr", load)
    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)

    with pytest.raises(RuntimeError, match="mesh.texture"):
        rec.mesh()

    load.assert_not_called()
    fuse.assert_not_called()


def test_run_calls_stages_in_order(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preproc = lambda **kwargs: calls.append("preproc") or rec.images_dir
    rec.pointcloud = lambda: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.semantics = lambda: calls.append("semantics") or tmp_path
    rec.mesh = lambda: calls.append("mesh") or tmp_path

    rec.run(["preproc", "pointcloud", "semantics", "mesh"])
    assert calls == ["preproc", "pointcloud", "mesh", "semantics"]


def test_run_subset(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preproc = lambda **kwargs: calls.append("preproc") or rec.images_dir
    rec.pointcloud = lambda: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run(["preproc", "pointcloud"])
    assert calls == ["preproc", "pointcloud"]
    assert "semantics" not in calls
    assert "mesh" not in calls


def test_run_dep_validation(tmp_path):
    """semantics requires pointcloud to have run first."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with pytest.raises(ValueError, match="pointcloud"):
        rec.run(["semantics"])


def test_run_dep_satisfied_by_existing_output(tmp_path):
    """`--stages pointcloud` alone runs when a prior preprocess's images/ exists on disk."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    # Preprocess output already present → dependency is satisfied without re-running it.
    rec.images_dir.mkdir(parents=True, exist_ok=True)
    rec.pointcloud = lambda: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run(["pointcloud"])
    assert calls == ["pointcloud"]


def test_run_missing_dep_output_still_raises(tmp_path):
    """pointcloud with no images/ and no preproc stage → hard error."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with pytest.raises(ValueError, match="preproc"):
        rec.run(["pointcloud"])


def test_run_default_uses_config_enabled(tmp_path):
    config = _make_config(
        tmp_path,
        {
            "semantics": {"enabled": True, "extractor": "dinov2"},
            "mesh": {"enabled": False},
        },
    )
    rec = Reconstructor(config)
    calls = []

    rec.preproc = lambda **kwargs: calls.append("preprocess") or rec.images_dir
    rec.pointcloud = lambda: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.semantics = lambda: calls.append("semantics") or tmp_path
    # report is always on and has no config flag, so a config-derived run always includes it
    rec.reconstruction_quality_report = lambda: calls.append("reconstruction_quality_report") or tmp_path

    rec.run()  # no stages arg — uses config
    assert "semantics" in calls
    assert "mesh" not in calls


########################################
# Localization stage
########################################


def test_localize_in_stage_order_and_deps():
    assert STAGES["localize"] == ("pointcloud",)


def test_run_auto_includes_localize_when_enabled(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    called = []
    with (
        patch.object(rec, "preproc"),
        patch.object(rec, "pointcloud", return_value=None),
        patch.object(rec, "localize", side_effect=lambda: called.append("localize")),
        patch.object(rec, "reconstruction_quality_report"),  # always on, and it would resolve a real reconstruction
    ):
        rec.run()
    assert called == ["localize"]


def test_run_omits_localize_when_disabled(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": False}})
    rec = Reconstructor(config)
    called = []
    with (
        patch.object(rec, "preproc"),
        patch.object(rec, "pointcloud", return_value=None),
        patch.object(rec, "localize", side_effect=lambda: called.append("localize")),
        patch.object(rec, "reconstruction_quality_report"),  # always on, and it would resolve a real reconstruction
    ):
        rec.run()
    assert called == []


def test_localize_without_pointcloud_raises(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    with pytest.raises(ValueError, match="requires 'pointcloud'"):
        rec.run(["localize"])


def test_localize_missing_zarr_raises(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    with pytest.raises(FileNotFoundError, match="pointcloud.zarr"):
        rec.localize()

    # The failed rebuild must not leave an empty store that done("pointcloud") would trust
    assert not rec.pointcloud_zarr.exists()


def test_run_refuses_localize_when_db_exists(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)

    with (
        patch.object(R, "localization_db_exists", return_value=True),
        patch.object(R, "_build_localization_db") as build,
        pytest.raises(ValueError, match="already exists"),
    ):
        rec.run(["localize"])

    build.assert_not_called()


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


def test_pointcloud_stage_attaches_viewer_when_lc_and_viz_enabled(tmp_path):
    """Loop closure + viz wraps the creator, attaches a live Viewer, and keeps it on reconstructor.viewer."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": True, "viz": {"enabled": True, "port": 9001}}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    mock_lc_instance.create_pointcloud.return_value = _make_mock_pointcloud_result(tmp_path)

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
    ):
        rec.pointcloud()

    mock_lc_cls.assert_called_once_with(base=creator_cls.return_value, config=None, ba=None)
    mock_viewer_cls.assert_called_once_with(port=9001)
    assert mock_lc_instance.viz is mock_viewer_cls.return_value
    assert mock_lc_instance.config.loop_edge_timing == "live"
    assert rec.viewer is mock_viewer_cls.return_value


def test_reconstructor_viewer_none_when_viz_disabled(tmp_path):
    """reconstructor.viewer stays None both before the pointcloud stage and after, when viz is disabled."""
    config = _make_config(tmp_path)  # viz.enabled defaults False
    rec = Reconstructor(config)
    assert rec.viewer is None

    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    with patch("collab_splats.reconstructor.get_creator", return_value=creator_cls):
        rec.pointcloud()

    assert rec.viewer is None


def test_pointcloud_stage_passes_window_ba_config_and_attrs(tmp_path):
    """BA + LC: LoopClosure gets a BA config with the window_ba cache dir; the zarr records window_ba."""
    config = _make_config(
        tmp_path,
        {
            "pointcloud": {
                "loop_closure": {"submap_size": 32},
                "bundle_adjustment": {"enabled": True, "dtype": "float64"},
            }
        },
    )
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    result = _make_mock_pointcloud_result(tmp_path)
    mock_lc_instance.create_pointcloud.return_value = result
    mock_lc_instance.window_ba = [{"start": 0, "focal": np.float32(368.5), "loss_final": float("nan")}]

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ):
        rec.pointcloud()

    ba_cfg = mock_lc_cls.call_args.kwargs["ba"]
    assert ba_cfg.dtype == "float64"
    assert ba_cfg.tracks_cache_dir == rec.backend_dir / "window_ba"

    attrs = result.save_zarr.call_args.kwargs["extra_attrs"]
    assert attrs["window_ba"] == [{"start": 0, "focal": 368.5, "loss_final": None}]


def test_pointcloud_stage_lc_without_ba_passes_none(tmp_path):
    """LC alone: LoopClosure gets ba=None and the zarr attrs carry no window_ba."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": True}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    result = _make_mock_pointcloud_result(tmp_path)
    mock_lc_instance.create_pointcloud.return_value = result

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ):
        rec.pointcloud()

    assert mock_lc_cls.call_args.kwargs["ba"] is None
    assert "window_ba" not in result.save_zarr.call_args.kwargs["extra_attrs"]


def test_pointcloud_stage_builds_lc_config_from_dict(tmp_path):
    """A dict loop_closure builds a LoopClosureConfig from its knobs and passes it through."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": {"submap_size": 32, "submap_overlap": 2}}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    mock_lc_instance.create_pointcloud.return_value = _make_mock_pointcloud_result(tmp_path)

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ):
        rec.pointcloud()

    # LoopClosure got a config carrying the dict knobs
    lc_config = mock_lc_cls.call_args.kwargs["config"]
    assert isinstance(lc_config, LoopClosureConfig)
    assert (lc_config.submap_size, lc_config.submap_overlap) == (32, 2)


def test_pointcloud_stage_dict_enabled_false_skips_lc(tmp_path):
    """loop_closure={'enabled': False} runs the bare creator — no LoopClosure wrap."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": {"enabled": False, "submap_size": 32}}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure") as mock_lc_cls,
    ):
        rec.pointcloud()

    mock_lc_cls.assert_not_called()
    creator_cls.return_value.create_pointcloud.assert_called_once()


def test_pointcloud_stage_no_viewer_when_viz_disabled(tmp_path):
    """No Viewer instantiated when viz is off, even with loop closure on."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": True}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    mock_lc_instance.create_pointcloud.return_value = _make_mock_pointcloud_result(tmp_path)

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance),
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
    ):
        rec.pointcloud()

    mock_viewer_cls.assert_not_called()


def test_pointcloud_stage_no_loop_closure_no_viewer(tmp_path):
    """loop_closure off never wraps the creator or attaches viz, even with viz on."""
    config = _make_config(tmp_path, {"pointcloud": {"viz": {"enabled": True}}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure") as mock_lc_cls,
        patch("collab_splats.viewer.Viewer") as mock_viewer_cls,
    ):
        rec.pointcloud()

    mock_lc_cls.assert_not_called()
    mock_viewer_cls.assert_not_called()


def test_localize_stage_builds_the_db(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    pc_zarr = rec.backend_dir / "pointcloud.zarr"
    pc_zarr.mkdir(parents=True)
    with (
        patch.object(R, "localization_db_exists", return_value=False),
        patch.object(R, "_build_localization_db") as build,
    ):
        rec.localize()
    build.assert_called_once_with(pc_zarr, "loma", rec.images_dir)


########################################
# Leaf-stage re-run
########################################


def test_leaf_stages_derived_from_dep_graph():
    """LEAF_STAGES is whatever nothing depends on — not a hardcoded list."""
    # Recompute from the graph, so a new consumer of a stage moves it out of LEAF_STAGES
    expected = {s for s in R.STAGES if not any(s in deps for deps in R.STAGES.values())}
    assert R.LEAF_STAGES == expected

    # Today's graph spelled out, so a failure above reads as a real change
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


def test_result_keeps_zarr_row_order(tmp_path):
    """Loading a finished reconstruction off disk — the whole basis of a leaf-stage re-run."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    # Sparse, non-sorted source indices: a loader that re-sorted rows would scramble the poses
    _seed_disk_reconstruction(rec, [16, 2])

    result = rec.result
    assert [p.name for p in result.image_paths] == ["frame_000016", "frame_000002"]
    np.testing.assert_array_equal(result.extrinsics[:, 2, 3], [0.0, 1.0])

    # The light load leaves the dense per-frame arrays on disk
    assert result.depth is None


def test_done_mesh(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.done("mesh") is False
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    (rec.backend_dir / "mesh.ply").touch()
    assert rec.done("mesh") is True


def test_done_pointcloud_needs_zarr_and_colmap(tmp_path):
    """
    pointcloud.zarr alone is a partial run; the COLMAP export completes the stage.
    """
    rec = Reconstructor(_make_config(tmp_path))
    rec.pointcloud_zarr.mkdir(parents=True)
    assert rec.done("pointcloud") is False

    rec.colmap_model_dir.mkdir(parents=True)
    assert rec.done("pointcloud") is True


def test_done_semantics_is_per_extractor(tmp_path):
    """The marker is this run's extractor — another extractor's lifted store must not satisfy it."""
    config = _make_config(tmp_path, {"semantics": {"extractor": "dinov2"}})
    rec = Reconstructor(config)
    sem_dir = rec.backend_dir / "semantics"
    sem_dir.mkdir(parents=True)
    (sem_dir / "talk2dino_lifted.zarr").mkdir()
    assert rec.done("semantics") is False
    (sem_dir / "dinov2_lifted.zarr").mkdir()
    assert rec.done("semantics") is True


def _seed_mesh(rec):
    """A two-triangle quad as the backend's mesh.ply, its last vertex behind the cameras; returns its vertices."""
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, -1]], dtype=np.float64)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(np.array([[0, 1, 2], [1, 3, 2]]))
    )
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(rec.outputs["mesh"]), mesh)
    return vertices.astype(np.float32)


def _mesh_semantics_rec(tmp_path, extractor, extractor_kwargs=None, dim=4):
    """Uncompressed codes seeded (no extraction), a one-point reconstruction and a quad mesh.ply."""
    semantics = {
        "enabled": True,
        "extractor": extractor,
        "extractor_kwargs": extractor_kwargs or {},
        "n_components": None,
    }
    rec = _semantics_rec(tmp_path, semantics)
    _seed_codes(rec, extractor, [np.ones((dim, 2, 2), np.float16)] * 2, extractor_kwargs=extractor_kwargs)
    return rec, _seed_mesh(rec)


def _arange_lift(calls):
    """
    lift_features stub: records (result, num_classes, frame 0's map), returns arange rows of the lifted width.

    - points behind the cameras (z < 0) get a zero row, as the real lift leaves unobserved vertices
    """

    def lift(frame_features, result, num_classes=None):
        frame = frame_features(0)
        calls.append((result, num_classes, frame))
        width = num_classes or frame.shape[0]
        rows = torch.arange(len(result.points) * width, dtype=torch.float32).reshape(-1, width)
        rows[torch.from_numpy(result.points[:, 2] < 0)] = 0
        return rows

    return lift


def test_semantics_lifts_queryable_codes_onto_mesh_vertices(tmp_path):
    rec, vertices = _mesh_semantics_rec(tmp_path, "talk2dino")
    calls = []

    with patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    # Second lift is the vertices, over the same cameras, no source pixel
    vertex_cloud = calls[1][0]
    np.testing.assert_array_equal(vertex_cloud.points, vertices)
    assert vertex_cloud.pixel_indices is None

    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert store["vertex_features"].dtype == np.float16
    expected = np.arange(16, dtype=np.float16).reshape(4, 4)
    expected[3] = 0
    np.testing.assert_array_equal(store["vertex_features"][:], expected)

    # The vertex behind the cameras stays all zero
    assert not store["vertex_features"][3].any()
    assert store.attrs["mesh_sha256"] == hashlib.sha256(rec.outputs["mesh"].read_bytes()).hexdigest()
    assert rec.done("semantics")


def test_semantics_without_vertex_mode_writes_points_only(tmp_path):
    rec, _ = _mesh_semantics_rec(tmp_path, "dinov2")
    calls = []

    with patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert len(calls) == 1
    assert "vertex_features" not in store and "vertex_word_ids" not in store
    assert "mesh_sha256" not in store.attrs
    assert rec.done("semantics")


def test_semantics_without_mesh_writes_points_only(tmp_path):
    rec = _semantics_rec(tmp_path, {"extractor": "talk2dino", "n_components": None})
    _seed_codes(rec, "talk2dino", [np.ones((4, 2, 2), np.float16)] * 2)
    calls = []

    with patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert len(calls) == 1 and "vertex_features" not in store and "mesh_sha256" not in store.attrs


@contextlib.contextmanager
def _stub_lens(n_words=70):
    """
    Processor, vocabulary, decoder and word_probabilities stubbed.

    - patch j's probabilities are arange(n_words) rolled by 7 * j, normalized: word (k + 7j) % n_words ranks k-th
    """
    vocab = MagicMock()
    vocab.words = [f"w{i}" for i in range(n_words)]

    def probabilities(states, decoder, vocab, ae=None):
        ramp = torch.arange(n_words, dtype=torch.float32)
        rows = torch.stack([ramp.roll(7 * j) for j in range(len(states))])
        yield rows / rows.sum(dim=1, keepdim=True), None

    with (
        patch.object(R, "load_processor") as processor,
        patch.object(R, "word_vocabulary", return_value=vocab),
        patch.object(R, "load_decoder") as decoder,
        patch.object(R, "word_probabilities", side_effect=probabilities) as decode,
    ):
        yield processor, decoder, decode


def test_semantics_stores_each_vertex_top_64_words(tmp_path):
    rec, _ = _mesh_semantics_rec(tmp_path, "ocr_lens", extractor_kwargs={"model_id": "local/llava"})
    calls = []

    with (
        _stub_lens() as (processor, decoder, decode),
        patch.object(R, "lift_features", side_effect=_arange_lift(calls)),
    ):
        rec.semantics()

    # The extracting checkpoint's lens; one decode per frame; one word lift over the vocabulary
    processor.assert_called_once_with("local/llava")
    decoder.assert_called_once_with("local/llava")
    assert decode.call_count == 2
    assert calls[1][1] == 70

    # Frame 0's indexed map: patch (r, c) lists word (69 - m + 7 * (2r + c)) % 70 at rank m
    ids, vals = calls[1][2]
    assert ids.shape == (64, 2, 2) and vals.shape == (64, 2, 2)

    for r, c in itertools.product(range(2), range(2)):
        expected_ids = [(69 - m + 7 * (2 * r + c)) % 70 for m in range(64)]
        assert ids[:, r, c].cpu().tolist() == expected_ids

    # Each seen vertex keeps the top-64 of its lifted row, descending
    expected = torch.arange(4 * 70, dtype=torch.float32).reshape(4, 70).topk(64, dim=1)
    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert store["vertex_word_ids"].dtype == np.int16 and store["vertex_word_probs"].dtype == np.float16
    np.testing.assert_array_equal(store["vertex_word_ids"][:3], expected.indices.numpy()[:3])
    np.testing.assert_array_equal(store["vertex_word_probs"][:3], expected.values.numpy()[:3].astype(np.float16))

    # The vertex behind the cameras has no word mass
    assert store["vertex_word_probs"][3].max() == 0
    assert store.attrs["words"] == [f"w{i}" for i in range(70)]
    assert "vertex_features" not in store


def test_done_semantics_is_stale_once_mesh_ply_changes(tmp_path):
    rec = Reconstructor(_make_config(tmp_path, {"semantics": {"extractor": "dinov2"}}))
    _seed_mesh(rec)
    sha = hashlib.sha256(rec.outputs["mesh"].read_bytes()).hexdigest()
    write_point_features(rec.outputs["semantics"], np.zeros((1, 2), np.float32), None, attrs={"mesh_sha256": sha})
    assert rec.done("semantics") is True

    rec.outputs["mesh"].write_bytes(rec.outputs["mesh"].read_bytes() + b"\n")
    assert rec.done("semantics") is False


def test_done_semantics_without_a_recorded_mesh_stays_done(tmp_path):
    # Points-only store (dinov2, or lifted before any mesh): nothing on the mesh to go stale
    rec = Reconstructor(_make_config(tmp_path, {"semantics": {"extractor": "dinov2"}}))
    write_point_features(rec.outputs["semantics"], np.zeros((1, 2), np.float32), None)
    _seed_mesh(rec)
    assert rec.done("semantics") is True


def test_done_localize(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "matcher": "loma"}})
    rec = Reconstructor(config)
    # No zarr at all: absent, and localization_db_exists must not even be consulted.
    assert rec.done("localize") is False
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    with patch.object(R, "localization_db_exists", return_value=True):
        assert rec.done("localize") is True
    with patch.object(R, "localization_db_exists", return_value=False):
        assert rec.done("localize") is False


def _seed_pointcloud_markers(rec):
    """Make done('pointcloud') true without running the stage."""
    rec.colmap_model_dir.mkdir(parents=True, exist_ok=True)
    (rec.backend_dir / "pointcloud.zarr").mkdir(parents=True, exist_ok=True)


def test_mesh_reads_frames_and_poses_from_the_zarr_on_disk(tmp_path):
    """
    `--stages mesh` on a pulled scene: rows, frames and poses all follow pointcloud.zarr.
    """
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "texture": False}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)

    # Sparse, non-sorted rows; each frame's pixels carry its own frame_idx
    _seed_disk_reconstruction(rec, [16, 2])
    frame_stack = np.stack([np.full((4, 6, 3), fi, np.uint8) for fi in (2, 16)])
    fr.write_frames(rec.images_dir, frame_stack, [2, 16])

    with (
        patch.object(R, "create_tsdf_mesh", return_value=rec.backend_dir / "mesh.ply") as fuse,
        patch.object(R, "clean_repair_mesh"),
        patch.object(R, "prepare_mesh", side_effect=lambda mesh, **kw: mesh),
    ):
        rec.mesh()

    depths, rgbs, c2w = fuse.call_args.args[:3]
    assert depths.shape == (2, 4, 6)
    assert rgbs[:, 0, 0, 0].tolist() == [16, 2]
    np.testing.assert_allclose(c2w[:, 2, 3], [0.0, -1.0], atol=1e-6)


def test_mesh_without_pointcloud_on_disk_still_raises(tmp_path):
    """Nothing in memory and nothing on disk is still a hard error, not a silent skip."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    with pytest.raises(FileNotFoundError, match="pointcloud.zarr"):
        rec.mesh()


def test_semantics_reads_pointcloud_zarr_from_disk(tmp_path):
    config = _make_config(tmp_path, {"semantics": {"extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    _touch_frames(rec, [0, 1, 2])
    _seed_disk_reconstruction(rec, [2, 0])

    # Scene codes row r is constant r; the writer stubbed so only the pick runs
    _seed_codes(rec, "dinov2", [np.full((4, 2, 2), r, np.float16) for r in range(3)])

    with (
        patch.object(R, "BaseFeatureExtractor"),
        patch.object(R, "lift_features", return_value=torch.zeros(1, 4)) as lift,
        patch.object(R, "write_point_features"),
    ):
        rec.semantics()

    # The lift gets the zarr's frames, in the zarr's order, and the result read off disk
    frame_features, ff = lift.call_args.args
    assert [float(frame_features(i)[0, 0, 0]) for i in range(2)] == [2.0, 0.0]
    assert [p.name for p in ff.image_paths] == ["frame_000002", "frame_000000"]


def test_run_named_upstream_stages_resume_instead_of_refusing(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    # Both upstream stages already complete on disk; only the named leaf still has work.
    rec.images_dir.mkdir(parents=True, exist_ok=True)
    _seed_pointcloud_markers(rec)
    calls = []
    rec.preproc = lambda **kwargs: calls.append("preproc")
    rec.pointcloud = lambda: calls.append("pointcloud")
    rec.localize = lambda: calls.append("localize")

    # The refusal is scoped to leaf stages: the two completed non-leaf stages are skipped
    rec.run(["preproc", "pointcloud", "localize"])
    assert calls == ["localize"]


def test_run_config_derived_stages_still_skip_silently(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True}})
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)
    _seed_pointcloud_markers(rec)
    (rec.backend_dir / "mesh.ply").touch()
    calls = []
    rec.preproc = lambda **kwargs: calls.append("preproc") or rec.images_dir
    rec.pointcloud = lambda: calls.append("pointcloud")
    rec.mesh = lambda: calls.append("mesh")
    rec.reconstruction_quality_report = lambda: calls.append("reconstruction_quality_report")

    # Must not raise: every done stage, leaf mesh included, is skipped
    rec.run()
    assert calls == ["reconstruction_quality_report"]


def test_base_yaml_mesh_has_fidelity_keys():
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
        "smooth_iterations",
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
    rec.preproc = lambda **kwargs: calls.append("preproc") or rec.images_dir
    rec.pointcloud = lambda: calls.append("pointcloud")
    rec.reconstruction_quality_report = lambda: calls.append("reconstruction_quality_report")

    rec.run()  # stages=None: the config-derived list
    assert calls == ["preproc", "pointcloud", "reconstruction_quality_report"]


def test_report_stage_dispatches_to_the_report_method(tmp_path):
    """A stage in STAGES whose method is not dispatched is a silent no-op."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    with patch.object(rec, "reconstruction_quality_report") as report:
        rec.run(["reconstruction_quality_report"], overwrite=True)
    report.assert_called_once_with()


def test_report_output_marker_is_report_json_in_the_backend_dir(tmp_path):
    """The marker makes --stages reconstruction_quality_report refuse existing output."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    _seed_pointcloud_markers(rec)
    assert rec.done("reconstruction_quality_report") is False
    (rec.backend_dir / "reconstruction_quality_report.json").write_text("{}")
    assert rec.done("reconstruction_quality_report") is True
    with pytest.raises(ValueError, match="already exists"):
        rec.run(["reconstruction_quality_report"])


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
        rgb = np.stack(
            [
                127 + 100 * np.sin(0.7 * x) * np.cos(0.5 * yy),
                127 + 90 * np.cos(0.45 * x + 0.3 * yy),
                127 + 80 * np.sin(0.25 * x - 0.6 * yy),
            ],
            axis=-1,
        )
        out.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.stack(out)


def test_report_json_is_the_columnar_contract(tmp_path):
    """Top-level keys and scene block; nan ships as null and no .json.tmp is left."""
    rec = Reconstructor(_make_config(tmp_path))
    _save_tiny_zarr(rec, with_confidence=False)

    rec.reconstruction_quality_report()

    out = rec.outputs["reconstruction_quality_report"]
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


def test_report_scene_block_records_min_pair_overlap(tmp_path):
    """The pruning knob is part of how the numbers were made, so the report carries it."""
    rec = Reconstructor(_make_config(tmp_path, {"reconstruction_quality_report": {"min_pair_overlap": 0.05}}))
    _save_tiny_zarr(rec, with_confidence=False)

    rec.reconstruction_quality_report()
    report = json.loads(rec.outputs["reconstruction_quality_report"].read_text())
    assert report["scene"]["min_pair_overlap"] == 0.05


def test_report_scene_block_defaults_min_pair_overlap_to_zero(tmp_path):
    """A config without the block still runs: base.yaml supplies 0.0, every pair kept."""
    rec = Reconstructor(_make_config(tmp_path))
    _save_tiny_zarr(rec, with_confidence=False)

    rec.reconstruction_quality_report()
    report = json.loads(rec.outputs["reconstruction_quality_report"].read_text())
    assert report["scene"]["min_pair_overlap"] == 0.0


def test_report_writes_nan_as_null(tmp_path):
    """An all-nan confidence frame has a nan median; the file must carry null, never a bare NaN."""
    rec = Reconstructor(_make_config(tmp_path))
    confidence = np.full((3, 16, 16), 0.9, np.float32)
    confidence[1] = np.nan
    _save_tiny_zarr(rec, confidence=confidence)

    rec.reconstruction_quality_report()
    text = rec.outputs["reconstruction_quality_report"].read_text()

    assert "NaN" not in text
    assert json.loads(text)["frames"]["confidence_median"] == [pytest.approx(0.9), None, pytest.approx(0.9)]


def test_report_runs_photometric_when_images_exist(tmp_path, monkeypatch):
    """Keyframes present: photometric_pairs is a filled table, not null."""
    rec = Reconstructor(_make_config(tmp_path))
    _save_tiny_zarr(rec)
    monkeypatch.setattr(fr, "frame_paths", lambda d: [Path(f"frame_{4 * k:06d}.png") for k in range(3)])
    monkeypatch.setattr(fr, "read_frames", lambda d, idxs: _texture_frames())

    rec.reconstruction_quality_report()
    report = json.loads(rec.outputs["reconstruction_quality_report"].read_text())
    assert report["photometric_pairs"]["idx1"]


def test_report_hands_photometric_the_uint8_frames_uncast(tmp_path, monkeypatch):
    """A float32 cast of every 1080p frame quadruples the stack; the stage passes uint8 as read."""
    rec = Reconstructor(_make_config(tmp_path))
    _save_tiny_zarr(rec)
    monkeypatch.setattr(fr, "frame_paths", lambda d: [Path(f"frame_{4 * k:06d}.png") for k in range(3)])
    monkeypatch.setattr(fr, "read_frames", lambda d, idxs: _texture_frames())
    real = metrics.compute_photometric_ncc
    seen = []

    def spy(images, *args, **kwargs):
        seen.append(images.dtype)
        return real(images, *args, **kwargs)

    monkeypatch.setattr(metrics, "compute_photometric_ncc", spy)
    rec.reconstruction_quality_report()
    assert seen == [np.uint8]


def test_report_raises_when_photometric_raises(tmp_path, monkeypatch):
    """A failing measurement fails the report; nothing is caught."""
    rec = Reconstructor(_make_config(tmp_path))
    _save_tiny_zarr(rec)
    monkeypatch.setattr(fr, "frame_paths", lambda d: [Path(f"frame_{4 * k:06d}.png") for k in range(3)])
    monkeypatch.setattr(fr, "read_frames", lambda d, idxs: _texture_frames())

    def _boom(*args, **kwargs):
        raise RuntimeError("photometric exploded")

    monkeypatch.setattr(metrics, "compute_photometric_ncc", _boom)
    with pytest.raises(RuntimeError, match="photometric exploded"):
        rec.reconstruction_quality_report()


def test_run_skips_the_report_and_never_touches_the_reconstruction(tmp_path):
    """Skip is checked BEFORE the stage runs, so a config-driven re-run costs nothing."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)
    _seed_pointcloud_markers(rec)
    out = rec.backend_dir / "reconstruction_quality_report.json"
    out.write_text('{"frames": {}}')

    with patch.object(Reconstructor, "reconstruction_quality_report") as report:
        rec.run()

    report.assert_not_called()
    assert out.read_text() == '{"frames": {}}'


def test_preproc_writes_video_quality_pngs(tmp_path, monkeypatch):
    """
    The video branch renders both report PNGs beside images/; the dir branch none.
    """
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

    rec = _video_reconstructor(tmp_path / "v", {"frame_selection": "fps", "fps": 1.0, "min_frames": None})
    rec.preproc()

    out = rec.images_dir.parent
    expected = {"photometric.png", "motion.png"}
    assert {p.name for p in out.glob("*.png")} == expected
    for name in expected:
        assert (out / name).read_bytes()[:4] == b"\x89PNG"

    # Image directory: no report, no plots
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    rec = Reconstructor(_make_config(tmp_path / "d", {"input_path": str(img_dir)}))
    rec.preproc()
    assert list(rec.images_dir.parent.glob("*.png")) == []


def test_undistort_rewrites_frames_at_the_undistorted_camera_dims(tmp_path, monkeypatch):
    camera = pycolmap.Camera(model="OPENCV", width=64, height=48, params=[60.0, 60.0, 32.0, 24.0, -0.2, 0.0, 0.0, 0.0])

    # Autospec so a wrong call site fails: calibrate_camera takes the images directory
    stub = create_autospec(calibrate_camera, return_value=camera)
    monkeypatch.setattr(R, "calibrate_camera", stub)

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"frame_{i:06d}.png"), np.zeros((48, 64, 3), np.uint8))

    rec = Reconstructor(_make_config(tmp_path, {"input_path": str(img_dir), "preproc": {"undistort": True}}))
    rec.preproc()

    stub.assert_called_once_with(rec.images_dir)

    # Same calibration, recomputed here, names the framing the stage must have written
    _, undistorted_camera = undistort_frames(np.zeros((1, 48, 64, 3), np.uint8), camera)
    assert fr.read_frames(rec.images_dir).shape[1:3] == (undistorted_camera.height, undistorted_camera.width)
