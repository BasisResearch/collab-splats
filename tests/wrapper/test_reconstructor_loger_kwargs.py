"""The per-backend config block reaches the creator via build_pointcloud, not just _run_feedforward."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from collab_splats.wrapper.reconstructor import Reconstructor

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class _StopAtFeedforward(Exception):
    """Sentinel raised by the recorder to halt build_pointcloud at the call site under test."""


def _loger_block() -> dict:
    """The shipping pointcloud.loger block, read straight off base.yaml."""
    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    return cfg["pointcloud"]["loger"]


def test_loger_block_reaches_run_feedforward_as_creator_kwargs(tmp_path):
    """build_pointcloud forwards pointcloud.<backend> verbatim as creator_kwargs."""
    # Read the expected values from the same file the Reconstructor merges, so the test pins the
    # passthrough rather than a snapshot of the numbers a concurrent tuning pass may change.
    expected = _loger_block()
    assert expected, "configs/base.yaml has no pointcloud.loger block — nothing left to pass through"

    # Reconstructor.__init__ deep-merges over the shipping base.yaml, so only the backend override
    # is needed; pc_cfg is read with strict key access and a hand-rolled config would KeyError.
    recon = Reconstructor(
        {
            "input_path": str(tmp_path / "video.mp4"),
            "output_path": str(tmp_path / "out"),
            "pointcloud": {"backend": "loger"},
        }
    )

    # Record the kwargs, then raise: build_pointcloud continues into cleaning and a PLY re-export,
    # and a mock return value would let that tail run against a mock instead of stopping here.
    captured: dict = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _StopAtFeedforward

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", _recorder):
        with pytest.raises(_StopAtFeedforward):
            recon.build_pointcloud()

    # build_pointcloud short-circuits when the pointcloud stage output already exists, so prove the
    # call site was actually reached before trusting anything in `captured`.
    assert captured, "build_pointcloud returned without calling _run_feedforward"

    # .get() so a dropped key reads as a named assertion, not a bare KeyError in the test itself.
    arrived = captured.get("creator_kwargs")
    assert arrived == expected, (
        f"creator_kwargs arrived as {arrived!r}; expected the base.yaml pointcloud.loger block {expected!r}"
    )
    assert arrived.get("window_size") == expected["window_size"], (
        f"window_size arrived as {arrived.get('window_size')!r}, expected {expected['window_size']!r}"
    )


def test_base_yaml_declares_the_loger_block():
    """Guards the precondition above: an empty block would make the passthrough test vacuous."""
    block = _loger_block()
    assert block.get("window_size") is not None, f"pointcloud.loger.window_size missing from base.yaml; got {block!r}"
    assert block.get("variant") is not None, f"pointcloud.loger.variant missing from base.yaml; got {block!r}"
