"""min_views and mv_rel_thresh reach the creator the same way max_points does."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from collab_splats.wrapper.reconstructor import _run_feedforward

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _run(tmp_path, **kwargs):
    """Call _run_feedforward with the creator class patched out."""
    with patch("collab_splats.wrapper.reconstructor.get_creator") as get:
        mock_cls = get.return_value
        _run_feedforward(
            backend="vggt_omega",
            images_dir=tmp_path / "images",
            output_dir=tmp_path,
            model_dir=tmp_path / "model",
            loop_closure=False,
            viz_enabled=False,
            viz_port=0,
            **kwargs,
        )
    return mock_cls


@pytest.mark.parametrize("min_views", [0, 2])
def test_multiview_knobs_forwarded(tmp_path, min_views):
    """Both knobs are passed through verbatim, alongside max_points."""
    mock_cls = _run(tmp_path, max_points=1234, min_views=min_views, mv_rel_thresh=0.03)
    assert mock_cls.call_args.kwargs["min_views"] == min_views
    assert mock_cls.call_args.kwargs["mv_rel_thresh"] == 0.03
    assert mock_cls.call_args.kwargs["max_points"] == 1234


def test_base_yaml_declares_the_key():
    """Reconstructor reads pc_cfg strictly, so both keys must exist in base.yaml; off by default."""
    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    assert cfg["pointcloud"]["min_views"] == 0
    assert cfg["pointcloud"]["mv_rel_thresh"] == 0.01
