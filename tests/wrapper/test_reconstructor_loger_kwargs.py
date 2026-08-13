"""The per-backend config block reaches the creator via build_pointcloud, not just _run_feedforward."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from collab_splats.wrapper.reconstructor import Reconstructor

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class _StopAtFeedforward(Exception):
    """Sentinel raised in place of _run_feedforward to halt build_pointcloud at the call site."""


def _loger_block() -> dict:
    """The shipping pointcloud.loger block, read straight off base.yaml; {} if absent."""
    # Total rather than strict on purpose. Both realistic regressions — the key deleted, or
    # `loger:` left present-but-empty (yaml yields None) — would otherwise raise KeyError or
    # AttributeError here, so the callers' named assertions would never run and would pin
    # nothing. Returning {} routes both into a real AssertionError instead.
    cfg = yaml.safe_load((CONFIGS / "base.yaml").read_text())
    return (cfg.get("pointcloud") or {}).get("loger") or {}


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

    # Raise instead of returning: build_pointcloud continues into cleaning and a PLY re-export
    # (clean.enabled is true in base.yaml), which would then run against a mock. The mock records
    # the call for us, so no hand-rolled recorder is needed.
    with patch(
        "collab_splats.wrapper.reconstructor._run_feedforward", side_effect=_StopAtFeedforward
    ) as ff:
        with pytest.raises(_StopAtFeedforward):
            recon.build_pointcloud()

    # .get() so a dropped key reads as a named assertion, not a bare KeyError in the test itself.
    arrived = ff.call_args.kwargs.get("creator_kwargs")
    assert arrived == expected, (
        f"creator_kwargs arrived as {arrived!r}; expected the base.yaml pointcloud.loger block {expected!r}"
    )


def test_base_yaml_declares_the_loger_block():
    """A dropped key is otherwise symptomless: the creator's own defaults already match base.yaml."""
    # Measured: all five of LoGeRCreator's dataclass defaults equal the values base.yaml ships
    # (LoGeR_star, 32, 3, 0, 50.0). So deleting the block changes no behaviour and raises nothing —
    # it silently turns the whole config surface into a no-op, and only this test would notice.
    block = _loger_block()
    assert block.get("window_size") is not None, f"pointcloud.loger.window_size missing from base.yaml; got {block!r}"
    assert block.get("variant") is not None, f"pointcloud.loger.variant missing from base.yaml; got {block!r}"
