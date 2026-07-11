"""Shared fakes for feedforward LC tests."""
import types

import torch

# _FakeQKV / _FakeMapAnythingModel moved verbatim from test_verify_lc_data.py
# (the copy in test_mapanything_creator.py was byte-identical).
# Consumers import these classes directly rather than via fixtures — deliberate:
# the constructors take args (canned preds), which zero-arg fixtures can't supply.


class _FakeQKV(torch.nn.Module):
    """Real nn.Module so register_forward_hook works; never actually called."""

    def forward(self, x):
        return x


class _FakeMapAnythingModel(torch.nn.Module):
    """Minimal stand-in: info_sharing block tree + forward returning canned preds."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds
        self._param = torch.nn.Parameter(torch.zeros(1))
        attn = types.SimpleNamespace(num_heads=2, qkv=_FakeQKV())
        block = types.SimpleNamespace(attn=attn)
        self.info_sharing = types.SimpleNamespace(self_attention_blocks=[block])

    def forward(self, views, memory_efficient_inference=False, minibatch_size=1):
        return self._preds
