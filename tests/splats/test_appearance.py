"""
AppearanceModule: identity at init, per-image affine, regulariser via render["appearance"].
"""

import pytest
import torch

from collab_splats.splats.appearance import AppearanceModule
from collab_splats.splats.losses import OPTIONAL_LOSSES, appearance_reg_loss
from collab_splats.splats.trainer import SplatsConfig


def test_identity_at_init():
    module = AppearanceModule(4)
    rgb = torch.rand(1, 8, 8, 3)
    out = module(rgb, torch.tensor([2]))
    assert torch.equal(out, rgb)


def test_affine_per_channel_and_only_addressed_row_gets_gradient():
    module = AppearanceModule(4)
    with torch.no_grad():
        module.params.weight[1] = torch.tensor([1.0, 0.0, -0.5, 0.1, 0.0, 0.0])
    rgb = torch.full((1, 2, 2, 3), 0.5)
    out = module(rgb, torch.tensor([1]))
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.1, 0.5, 0.25]))

    out.sum().backward()
    grad = module.params.weight.grad
    assert grad[1].abs().sum() > 0
    assert grad[[0, 2, 3]].abs().sum() == 0


def test_appearance_reg_reads_render_params_or_skips():
    assert OPTIONAL_LOSSES["appearance_reg"] is appearance_reg_loss
    assert appearance_reg_loss({}, {}, None, 1.0) is None
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert appearance_reg_loss({"appearance": params}, {}, None, 1.0).item() == pytest.approx(1.0 / 6)


def test_config_appearance_fields():
    cfg = SplatsConfig.from_dict({"appearance_opt": True, "appearance_lr": 2e-3})
    assert cfg.appearance_opt and cfg.appearance_lr == 2e-3
    assert SplatsConfig().appearance_opt is False
