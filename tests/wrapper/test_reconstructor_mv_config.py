"""use_multiview_confidence reaches the creator the same way max_points does."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from collab_splats.wrapper.reconstructor import _run_feedforward

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _run(tmp_path, **kwargs):
    """Call _run_feedforward with the creator class patched out."""
    with patch("collab_splats.pointcloud.feedforward.VGGTOmegaCreator") as mock_cls:
        mock_cls.return_value.outputs = None
        _run_feedforward(
            backend="vggt_omega",
            images_dir=tmp_path / "images",
            output_dir=tmp_path,
            loop_closure=False,
            viz_enabled=False,
            viz_port=0,
            **kwargs,
        )
    return mock_cls


@pytest.mark.parametrize("flag", [True, False])
def test_use_multiview_confidence_forwarded(tmp_path, flag):
    """The flag is passed through verbatim, alongside max_points."""
    mock_cls = _run(tmp_path, max_points=1234, use_multiview_confidence=flag)
    assert mock_cls.call_args.kwargs["use_multiview_confidence"] is flag
    assert mock_cls.call_args.kwargs["max_points"] == 1234


def test_base_yaml_declares_the_key():
    """Reconstructor reads pc_cfg strictly, so the key must exist in base.yaml."""
    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    assert cfg["pointcloud"]["use_multiview_confidence"] is False
