"""Tests for the sfm config surface: backend allowlist, sub-blocks, BA/LC/refine rejection."""

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


def _sfm_cfg(backend, **block):
    """
    base.yaml with method: sfm, the given backend, and `block` merged into its sub-block.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = backend
    cfg["pointcloud"][backend].update(block)
    return cfg


def test_sfm_backends_are_the_sfm_creators():
    """
    The allowlist is every creator _run_sfm dispatches through SFM_CREATORS.
    """
    assert _SFM_BACKENDS == {"instantsfm", "colmap", "hloc"}


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
def test_sfm_backend_validates_at_config_load(backend):
    cfg = Reconstructor.validate_config(_sfm_cfg(backend))
    assert cfg["pointcloud"]["backend"] == backend


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
def test_sfm_refuses_bundle_adjustment_and_loop_closure(backend):
    for key in ("bundle_adjustment", "loop_closure"):
        cfg = _sfm_cfg(backend)
        cfg["pointcloud"][key] = True
        with pytest.raises(ValueError, match=key):
            Reconstructor.validate_config(cfg)


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


def test_bad_colmap_block_fails_config_load():
    """
    The sub-block bounds check is reachable through validate_config, not only directly.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "colmap"
    cfg["pointcloud"]["colmap"]["overlap"] = 0
    with pytest.raises(ValueError, match="pointcloud.colmap.overlap"):
        Reconstructor.validate_config(cfg)


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
@pytest.mark.parametrize("pairing", ["sequential", "retrieval", "sequential+retrieval", "exhaustive"])
def test_every_pairing_validates(backend, pairing):
    Reconstructor._validate_sfm_block(_sfm_cfg(backend, pairing=pairing)["pointcloud"], backend)


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
@pytest.mark.parametrize(
    "key,bad",
    [
        ("pairing", "spatial"),
        ("pairing", None),
        ("overlap", 0),
        ("overlap", True),
        ("overlap", 2.0),
        ("num_retrieved", 0),
        ("num_retrieved", False),
        ("num_threads", -1),
        ("num_threads", "8"),
        ("min_registered_frac", 0),
        ("min_registered_frac", 1.5),
        ("min_registered_frac", True),
        ("min_registered_frac", "0.5"),
        ("typo_key", 1),
    ],
)
def test_bad_sfm_block_values_are_rejected(backend, key, bad):
    with pytest.raises(ValueError, match=f"pointcloud.{backend}"):
        Reconstructor._validate_sfm_block(_sfm_cfg(backend, **{key: bad})["pointcloud"], backend)


@pytest.mark.parametrize("key", ["retrieval_conf", "feature_conf", "matcher_conf"])
@pytest.mark.parametrize("bad", ["", None, 3])
def test_hloc_conf_keys_must_be_non_empty_strings(key, bad):
    with pytest.raises(ValueError, match=f"pointcloud.hloc.{key}"):
        Reconstructor._validate_sfm_block(_sfm_cfg("hloc", **{key: bad})["pointcloud"], "hloc")


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
def test_min_registered_frac_is_one_floor_for_every_backend(backend):
    for good in (1.0, 1, 0.01):
        Reconstructor._validate_sfm_block(_sfm_cfg(backend, min_registered_frac=good)["pointcloud"], backend)
    with pytest.raises(ValueError, match=f"pointcloud.{backend}.min_registered_frac"):
        Reconstructor._validate_sfm_block(_sfm_cfg(backend, min_registered_frac=1.5)["pointcloud"], backend)


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
@pytest.mark.parametrize("bad", [0, -1, True, "8", 2.0])
def test_num_threads_is_checked_for_every_backend(backend, bad):
    Reconstructor._validate_sfm_block(_sfm_cfg(backend, num_threads=3)["pointcloud"], backend)
    with pytest.raises(ValueError, match=f"pointcloud.{backend}.num_threads"):
        Reconstructor._validate_sfm_block(_sfm_cfg(backend, num_threads=bad)["pointcloud"], backend)


def test_colmap_rejects_hloc_only_keys():
    with pytest.raises(ValueError, match="pointcloud.colmap"):
        Reconstructor._validate_sfm_block(_sfm_cfg("colmap", feature_conf="sift")["pointcloud"], "colmap")


########################################################################
# instantsfm knob bounds at config load
########################################################################


def _validated_sfm_config(**instantsfm):
    """
    base.yaml merged into a method:sfm config with the given instantsfm overrides, validated.
    """
    config = {
        "input_path": "dummy.mp4",
        "output_path": "dummy_out",
        "pointcloud": {"method": "sfm", "backend": "instantsfm", "instantsfm": dict(instantsfm)},
    }
    return Reconstructor(config).config


def test_random_seed_out_of_range_is_rejected_at_config_load():
    # np.random.seed rejects this, but InstantSfM only reads the value after the SIFT +
    # exhaustive-matching pass — the whole run would burn first
    with pytest.raises(ValueError, match=r"random_seed must be null or an int"):
        _validated_sfm_config(random_seed=-1)

    with pytest.raises(ValueError, match=r"random_seed must be null or an int"):
        _validated_sfm_config(random_seed=2**32)


def test_random_seed_accepts_null_and_an_in_range_int():
    assert _validated_sfm_config(random_seed=None)["pointcloud"]["instantsfm"]["random_seed"] is None
    assert _validated_sfm_config(random_seed=2**32 - 1)["pointcloud"]["instantsfm"]["random_seed"] == 2**32 - 1


def test_min_num_view_per_track_below_two_is_rejected_at_config_load():
    # A track needs two views to triangulate; 1 and 0 produce no geometry, and the value is
    # read only after the SIFT + exhaustive-matching pass
    for bad in (1, 0, -1, 2.5):
        with pytest.raises(ValueError, match=r"min_num_view_per_track must be null or an int >= 2"):
            _validated_sfm_config(min_num_view_per_track=bad)


def test_min_num_view_per_track_accepts_null_and_an_int_at_or_above_two():
    cfg = _validated_sfm_config(min_num_view_per_track=None)
    assert cfg["pointcloud"]["instantsfm"]["min_num_view_per_track"] is None
    assert _validated_sfm_config(min_num_view_per_track=2)["pointcloud"]["instantsfm"]["min_num_view_per_track"] == 2
    assert _validated_sfm_config(min_num_view_per_track=6)["pointcloud"]["instantsfm"]["min_num_view_per_track"] == 6


@pytest.mark.parametrize("key", ["retriangulate", "use_depths"])
def test_instantsfm_block_rejects_unknown_keys(key):
    with pytest.raises(ValueError, match=rf"pointcloud.instantsfm has unknown keys \['{key}'\]"):
        Reconstructor.validate_config(_sfm_cfg("instantsfm", **{key: True}))
