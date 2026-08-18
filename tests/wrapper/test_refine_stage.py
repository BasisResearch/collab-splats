"""Tests for the refine stage: config validation, stage registration, refine_poses."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.wrapper.reconstructor import (
    _STAGE_DEPS,
    _STAGE_ORDER,
    LEAF_STAGES,
    Reconstructor,
)


def _cfg(tmp_path, **pointcloud):
    """Minimal valid config dict; pointcloud kwargs merged over base.yaml defaults."""
    return {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": pointcloud,
    }


def test_validate_config_rejects_ba_with_lc_bool(tmp_path):
    """bundle_adjustment + loop_closure=true must fail loud at construction."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))


def test_validate_config_rejects_ba_with_lc_dict(tmp_path):
    """Dict-form loop_closure ({'enabled': ...} implicit true) is rejected too."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure={"submap_size": 16}))


def test_validate_config_allows_ba_without_lc(tmp_path):
    """BA alone constructs fine — the old NotImplementedError is gone."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=False))
    assert r.config["pointcloud"]["bundle_adjustment"] is True
