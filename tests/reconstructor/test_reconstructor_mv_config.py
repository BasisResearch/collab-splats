"""min_views and mv_rel_thresh reach the creator the same way max_points does."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from collab_splats.reconstructor import Reconstructor
from tests.reconstructor._stubs import stub_creator_cls

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _run(tmp_path, **pointcloud):
    """Run the pointcloud stage with the creator class patched out; returns that class."""
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"backend": "vggt_omega", **pointcloud},
    }
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(MagicMock())

    with patch("collab_splats.reconstructor.get_creator", return_value=creator_cls):
        rec.pointcloud()

    return creator_cls


@pytest.mark.parametrize("min_views", [0, 2])
def test_multiview_knobs_forwarded(tmp_path, min_views):
    """Both knobs are passed through verbatim, alongside max_points."""
    creator_cls = _run(
        tmp_path, max_points=1234, min_views=min_views, mv_rel_thresh=0.03
    )
    assert creator_cls.call_args.kwargs["min_views"] == min_views
    assert creator_cls.call_args.kwargs["mv_rel_thresh"] == 0.03
    assert creator_cls.call_args.kwargs["max_points"] == 1234


def test_base_yaml_declares_the_key():
    """Reconstructor reads pc_cfg strictly, so both keys must exist in base.yaml; off by default."""
    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    assert cfg["pointcloud"]["min_views"] == 0
    assert cfg["pointcloud"]["mv_rel_thresh"] == 0.01
