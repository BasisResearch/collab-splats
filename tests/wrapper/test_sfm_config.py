"""Tests for InstantSfM config surface: backend allowlist, BA/refine rejection."""

from pathlib import Path

import pytest
import yaml

from collab_splats.wrapper.reconstructor import _SFM_BACKENDS, Reconstructor

BASE_YAML = Path(__file__).parents[2] / "configs" / "base.yaml"


def _base_config():
    """
    Load configs/base.yaml as a plain dict, with dummy required top-level fields.

    base.yaml itself carries `input_path: null` / `output_path: null` (they're filled in
    at Reconstructor construction time from the caller's config) — validate_config's
    required-field check rejects None, so tests that call it directly need placeholders.
    """
    with open(BASE_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["input_path"] = "dummy.mp4"
    cfg["output_path"] = "dummy_out"
    return cfg


def test_instantsfm_is_valid_sfm_backend():
    """
    instantsfm joins colmap/hloc in the sfm backend allowlist.
    """
    assert "instantsfm" in _SFM_BACKENDS


def test_sfm_instantsfm_config_validates():
    """
    method: sfm, backend: instantsfm is a valid, constructible config.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    Reconstructor.validate_config(cfg)  # must not raise


def test_sfm_rejects_bundle_adjustment():
    """
    bundle_adjustment is InstantSfM's own job — refused at validation.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    cfg["pointcloud"]["bundle_adjustment"] = True
    with pytest.raises(ValueError, match="bundle_adjustment"):
        Reconstructor.validate_config(cfg)


def test_instantsfm_features_allowlist():
    """
    pointcloud.instantsfm.features must be in the installed-version allowlist.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    cfg["pointcloud"]["instantsfm"]["features"] = "superglue"
    with pytest.raises(ValueError, match="features"):
        Reconstructor.validate_config(cfg)


def test_base_yaml_has_instantsfm_block():
    """
    base.yaml carries the instantsfm sub-block with its documented defaults.
    """
    cfg = _base_config()
    assert cfg["pointcloud"]["instantsfm"] == {
        "features": "colmap",
        "retriangulation": False,
        "depth_align": "scale",
        "random_seed": None,
    }


def test_refine_poses_refuses_sfm_method(tmp_path):
    """
    refine_poses refuses outright when pointcloud.method is sfm.

    Cheap to construct directly since the guard is the first line of the method —
    no pointcloud.zarr or heavy creator setup needed.
    """
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "sfm", "backend": "instantsfm"},
    }
    r = Reconstructor(config)
    with pytest.raises(ValueError, match="refine_poses"):
        r.refine_poses()


def test_sfm_rejects_loop_closure():
    """
    loop_closure is a sequential-submap mechanism; InstantSfM is a global mapper — refused.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    cfg["pointcloud"]["loop_closure"] = True
    with pytest.raises(ValueError, match="loop_closure"):
        Reconstructor.validate_config(cfg)


def test_run_sfm_colmap_hloc_still_not_implemented(tmp_path):
    """
    _run_sfm only implements backend: instantsfm; colmap/hloc raise NotImplementedError naming it.
    """
    cfg = _base_config()
    cfg["input_path"] = str(tmp_path / "video.mp4")
    cfg["output_path"] = str(tmp_path / "out")
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "colmap"
    r = Reconstructor(cfg)
    with pytest.raises(NotImplementedError, match="instantsfm"):
        r._run_sfm()
