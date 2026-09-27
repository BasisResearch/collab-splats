"""Removed geometric verification: the old stage and flag refuse, published configs still load."""

from pathlib import Path

import pytest
import yaml

from collab_splats.remote.sources import PUSH_EXCLUDES
from collab_splats.wrapper.reconstructor import (
    _STAGE_DEPS,
    _STAGE_ORDER,
    LEAF_STAGES,
    Reconstructor,
)

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_verify_is_not_a_stage():
    """The old stage name is gone from the graph."""
    assert "verify" not in _STAGE_ORDER
    assert "verify" not in _STAGE_DEPS
    assert "verify" not in LEAF_STAGES


def _config(tmp_path, **overrides):
    """A minimal valid config for Reconstructor construction."""
    return {"input_path": str(tmp_path / "v.mp4"), "output_path": str(tmp_path / "out"), **overrides}


def test_naming_the_verify_stage_raises(tmp_path):
    """--stages verify raises, saying it was removed, before any work."""
    rec = Reconstructor(_config(tmp_path))
    with pytest.raises(ValueError, match=r"geometric verification was removed"):
        rec.run_pipeline(stages=["verify"])
    with pytest.raises(ValueError, match=r"geometric verification was removed"):
        rec.run_pipeline(stages=["pointcloud", "verify"])


def test_geometric_verification_true_raises(tmp_path):
    """The removed flag set true raises at construction."""
    with pytest.raises(ValueError, match=r"geometric verification was removed"):
        Reconstructor(_config(tmp_path, pointcloud={"geometric_verification": True}))


def test_geometric_verification_false_is_still_accepted(tmp_path):
    """Every published run_config.yaml carries the flag false; a leaf re-run must not break on it."""
    rec = Reconstructor(_config(tmp_path, pointcloud={"geometric_verification": False}))
    assert rec.config["pointcloud"]["geometric_verification"] is False


def test_base_yaml_has_no_verification_keys():
    """Neither the old flag nor the unshipped report knob survives in the defaults."""
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    assert "reconstruction_quality_report" not in cfg
    assert "geometric_verification" not in cfg["pointcloud"]


def test_database_db_not_pushed():
    """Older scenes still carry colmap/database.db — a rebuildable local artifact, never pushed."""
    assert "/*/colmap/database.db" in PUSH_EXCLUDES
