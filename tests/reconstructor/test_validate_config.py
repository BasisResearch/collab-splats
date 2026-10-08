"""
Config-load refusals owned by Reconstructor.validate_config.
"""

from pathlib import Path

import pytest
import yaml

from collab_splats.reconstructor import Reconstructor

BASE_YAML = Path(__file__).parents[2] / "configs" / "base.yaml"


def _cfg(tmp_path, **pointcloud):
    """
    Minimal config dict; pointcloud kwargs merged over base.yaml defaults.
    """
    return {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": pointcloud,
    }


def _base_config():
    """
    configs/base.yaml as a plain dict, with placeholder required paths.
    """
    cfg = yaml.safe_load(BASE_YAML.read_text())
    cfg["input_path"] = "dummy.mp4"
    cfg["output_path"] = "dummy_out"
    return cfg


def _sfm_cfg(backend):
    """
    base.yaml with method: sfm and the given backend.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = backend
    return cfg


########################################
# Required fields, method and backend
########################################


@pytest.mark.parametrize("field", ["input_path", "output_path"])
def test_missing_required_path_is_refused(field):
    cfg = _base_config()
    del cfg[field]

    with pytest.raises(ValueError, match=field):
        Reconstructor.validate_config(cfg)


def test_unknown_method_is_refused(tmp_path):
    # pointcloud.method=nerfstudio is gone; only feedforward and sfm dispatch
    with pytest.raises(ValueError, match="pointcloud.method must be one of"):
        Reconstructor(_cfg(tmp_path, method="nerfstudio"))


def test_unknown_feedforward_backend_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pointcloud.backend must be one of"):
        Reconstructor(_cfg(tmp_path, method="feedforward", backend="badmodel"))


def test_backend_of_the_other_method_is_refused(tmp_path):
    # colmap is an sfm creator, so feedforward refuses it
    with pytest.raises(ValueError, match="for method 'feedforward'"):
        Reconstructor(_cfg(tmp_path, method="feedforward", backend="colmap"))


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
def test_sfm_backend_validates_at_config_load(backend):
    cfg = Reconstructor.validate_config(_sfm_cfg(backend))
    assert cfg["pointcloud"]["backend"] == backend


########################################
# Loop closure normalization
########################################


def test_unknown_loop_closure_knob_is_refused(tmp_path):
    # An unknown loop_closure knob fails loud at config load, naming the key
    with pytest.raises(ValueError, match="pointcloud.loop_closure has unknown keys \\['bogus'\\]"):
        Reconstructor(_cfg(tmp_path, loop_closure={"bogus": 1}))


def test_loop_closure_is_normalized_to_a_dict(tmp_path):
    # A bool, or a knob dict without `enabled`, becomes a dict carrying `enabled`
    on = Reconstructor(_cfg(tmp_path, loop_closure=True))
    off = Reconstructor(_cfg(tmp_path, loop_closure=False))
    knobs = Reconstructor(_cfg(tmp_path, loop_closure={"submap_size": 32}))

    assert on.config["pointcloud"]["loop_closure"] == {"enabled": True}
    assert off.config["pointcloud"]["loop_closure"] == {"enabled": False}
    assert knobs.config["pointcloud"]["loop_closure"] == {"enabled": True, "submap_size": 32}


########################################
# Combinations no stage can run
########################################


@pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
def test_sfm_refuses_bundle_adjustment_and_loop_closure(backend):
    for key in ("bundle_adjustment", "loop_closure"):
        cfg = _sfm_cfg(backend)
        cfg["pointcloud"][key] = True

        with pytest.raises(ValueError, match=key):
            Reconstructor.validate_config(cfg)


def test_loop_closure_with_loger_is_refused(tmp_path):
    # Refused at construction, before any inference or images/ read
    with pytest.raises(ValueError, match="loop_closure is not supported with backend 'loger'"):
        Reconstructor(_cfg(tmp_path, backend="loger", loop_closure=True))


def test_loop_closure_with_loger_is_refused_in_dict_form(tmp_path):
    # A knob dict without `enabled` means enabled, so it is refused too
    with pytest.raises(ValueError, match="backend 'loger'"):
        Reconstructor(_cfg(tmp_path, backend="loger", loop_closure={"submap_size": 16}))


def test_loop_closure_disabled_by_dict_is_not_refused_for_loger(tmp_path):
    # {"enabled": False} is a truthy object with falsy intent; the refusal reads the normalized flag
    rec = Reconstructor(_cfg(tmp_path, backend="loger", loop_closure={"enabled": False}))
    assert rec.config["pointcloud"]["loop_closure"]["enabled"] is False


########################################
# base.yaml
########################################


def test_base_config_path_is_read(tmp_path):
    # A caller-supplied base.yaml replaces configs/base.yaml as the defaults
    cfg = _base_config()
    cfg["mesh"]["voxel_depth_px"] = 0.123
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(cfg))

    rec = Reconstructor({"input_path": "v.mp4", "output_path": str(tmp_path / "out")}, base_config=base)
    assert rec.config["mesh"]["voxel_depth_px"] == 0.123


def test_base_yaml_mesh_sdf_trunc_mult_default_is_four():
    # base.yaml carries mesh.sdf_trunc_mult at Open3D's noisy-sensor default
    assert _base_config()["mesh"]["sdf_trunc_mult"] == 4.0
