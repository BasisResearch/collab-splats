"""
Reconstructor test fixtures.
"""

import pytest

import collab_splats.reconstructor as reconstructor


@pytest.fixture(autouse=True)
def _offline_hf_access(monkeypatch):
    """
    No Hub call from run(): the gated-weights preflight is stubbed unless a test asks for it.
    """
    monkeypatch.setattr(reconstructor, "check_hf_access", lambda repo, filename: None)
