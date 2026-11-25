"""Shared fixtures for preproc tests."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def noise_gray():
    """240x320 uniform noise — maximum high-frequency content, the sharp end of any blur ladder.

    Shared across the session, so treat it as read-only: score it, blur a copy,
    never write into it.
    """
    rng = np.random.default_rng(0)
    return (rng.random((240, 320)) * 255).astype(np.uint8)
