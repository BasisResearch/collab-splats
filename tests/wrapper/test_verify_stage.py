"""Verify-stage wiring: leaf-stage registration + config default."""

from pathlib import Path

import yaml

from collab_splats.wrapper.reconstructor import _STAGE_DEPS, _STAGE_ORDER, LEAF_STAGES

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_verify_is_a_leaf_stage():
    """verify is registered, depends only on pointcloud, and is re-runnable on its own."""
    assert "verify" in _STAGE_ORDER
    assert _STAGE_DEPS["verify"] == ["pointcloud"]
    assert "verify" in LEAF_STAGES


def test_geometric_verification_defaults_off():
    """Ships off until the first measured report (spec: Validation gates the default)."""
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    assert cfg["pointcloud"]["geometric_verification"] is False


def test_database_db_not_pushed():
    """database.db is a rebuildable local artifact — excluded from GCS pushes."""
    from collab_splats.remote.sources import PUSH_EXCLUDES

    assert "/*/colmap/database.db" in PUSH_EXCLUDES
