"""Tests for load_lifted_normed module-level helper."""

from unittest.mock import patch

import numpy as np
import torch

from collab_splats.dashboard.viewer import load_lifted_normed


def test_load_lifted_normed_normalises():
    fake_lifted = torch.tensor([[3.0, 4.0], [0.0, 2.0]])  # norms 5, 2
    with patch("collab_splats.dashboard.viewer.lift_features", return_value=fake_lifted), \
         patch("collab_splats.dashboard.viewer._load_feature_maps", return_value=["fm"]):
        out = load_lifted_normed(result=object(), semantics_dir=object())
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
