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


def test_instantsfm_is_the_only_sfm_backend():
    """
    instantsfm is the sole sfm backend on the allowlist — colmap/hloc are not wired into _run_sfm.
    """
    assert _SFM_BACKENDS == {"instantsfm"}


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


def test_base_yaml_has_instantsfm_block():
    """
    base.yaml carries the instantsfm sub-block with its documented defaults.
    """
    cfg = _base_config()
    assert cfg["pointcloud"]["instantsfm"] == {
        "retriangulation": False,
        "random_seed": None,
        "min_num_view_per_track": None,
    }


def test_base_yaml_mesh_sdf_trunc_mult_default_is_four():
    """
    base.yaml carries mesh.sdf_trunc_mult at Open3D's noisy-sensor default.
    """
    assert _base_config()["mesh"]["sdf_trunc_mult"] == 4.0


def test_sdf_trunc_mult_below_one_is_rejected_at_config_load():
    """
    A truncation band narrower than a voxel punctures the surface, so it is refused.
    """
    for bad in (0.9, 0, -1, "1.5", True):
        cfg = _base_config()
        cfg["mesh"]["sdf_trunc_mult"] = bad
        with pytest.raises(ValueError, match=r"mesh.sdf_trunc_mult must be a number >= 1.0"):
            Reconstructor.validate_config(cfg)

    # 1.0 is the floor, and the thin-structure settings this key exists for are above it
    for good in (1.0, 1.5, 2, 4.0):
        cfg = _base_config()
        cfg["mesh"]["sdf_trunc_mult"] = good
        Reconstructor.validate_config(cfg)  # must not raise


def test_base_yaml_mesh_bands_default_is_null():
    """
    base.yaml ships one TSDF volume; banding is opt-in.
    """
    assert _base_config()["mesh"]["bands"] is None


def test_mesh_bands_that_do_not_partition_depth_are_rejected_at_config_load():
    """
    A gap between bands loses the geometry inside it and an overlap double-surfaces it.
    """
    gap = [
        {"depth_min": 0, "depth_trunc": 2, "voxel_size": 0.01},
        {"depth_min": 3, "depth_trunc": 8, "voxel_size": 0.04},
    ]
    for bad in ([], gap, [{"depth_min": 0, "depth_trunc": 2}], [{"depth_min": 0, "depth_trunc": 2, "voxel_size": 0}]):
        cfg = _base_config()
        cfg["mesh"]["bands"] = bad
        with pytest.raises(ValueError, match=r"mesh.bands invalid"):
            Reconstructor.validate_config(cfg)

    # null and a contiguous ascending list both pass
    for good in (None, [{"depth_min": 0, "depth_trunc": 2, "voxel_size": 0.01}], gap[:1] + [dict(gap[1], depth_min=2)]):
        cfg = _base_config()
        cfg["mesh"]["bands"] = good
        Reconstructor.validate_config(cfg)  # must not raise


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


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
def test_sfm_rejects_unwired_backends_at_config_load(backend):
    """
    colmap/hloc are refused at validation, before any stage runs — not mid-run after preproc.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = backend
    with pytest.raises(ValueError, match="pointcloud.backend"):
        Reconstructor.validate_config(cfg)


def test_run_sfm_still_guards_non_instantsfm_backends(tmp_path):
    """
    _run_sfm keeps its own NotImplementedError as defence in depth behind validate_config.

    Constructed with instantsfm (the allowlist rejects anything else) then mutated, since the
    guard now only fires for a direct _run_sfm call, never through a validated config.
    """
    cfg = _base_config()
    cfg["input_path"] = str(tmp_path / "video.mp4")
    cfg["output_path"] = str(tmp_path / "out")
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "instantsfm"
    r = Reconstructor(cfg)
    r.config["pointcloud"]["backend"] = "colmap"
    with pytest.raises(NotImplementedError, match="instantsfm"):
        r._run_sfm()
