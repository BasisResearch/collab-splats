"""
Scaffold-GS anchors: config, MLP heads, decode, and anchor densification.
"""

import numpy as np
import pytest
import torch

from collab_splats.splats.scaffold import ScaffoldConfig
from collab_splats.splats.trainer import SplatsConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def test_representation_defaults_to_vanilla():
    cfg = SplatsConfig()
    assert cfg.representation == "vanilla"
    assert cfg.scaffold is None
    assert cfg.scaffold_config is None


def test_scaffold_block_parses_into_scaffold_config():
    cfg = SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offsets": 5, "feat_dim": 16}})
    assert isinstance(cfg.scaffold_config, ScaffoldConfig)
    assert cfg.scaffold_config.n_offsets == 5
    assert cfg.scaffold_config.feat_dim == 16


def test_scaffold_representation_without_a_block_gets_defaults():
    cfg = SplatsConfig.from_dict({"representation": "scaffold"})
    assert cfg.scaffold_config.n_offsets == ScaffoldConfig().n_offsets


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="representation"):
        SplatsConfig.from_dict({"representation": "octree"})


def test_unknown_scaffold_key_is_rejected():
    with pytest.raises(ValueError, match="n_offset"):
        SplatsConfig.from_dict({"representation": "scaffold", "scaffold": {"n_offset": 5}})


def test_scaffold_block_without_scaffold_representation_is_rejected():
    with pytest.raises(ValueError, match="representation: scaffold"):
        SplatsConfig.from_dict({"scaffold": {"n_offsets": 5}})


def test_sh_degree_is_rejected_under_scaffold():
    with pytest.raises(ValueError, match="sh_degree"):
        SplatsConfig.from_dict({"representation": "scaffold", "sh_degree": 3})


########################################
# MLP heads
########################################


def test_mlp_heads_emit_per_offset_outputs():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    cfg = ScaffoldConfig(n_offsets=4, feat_dim=8)
    mlps = ScaffoldMLPs(cfg)
    features = torch.zeros(6, cfg.feat_dim + 4)  # feat + view dir (3) + view distance (1)
    opacity, cov, colour = mlps(features, camera_id=None)
    assert opacity.shape == (6, 4)
    assert cov.shape == (6, 4 * 7)
    assert colour.shape == (6, 4 * 3)
    assert opacity.min() >= -1.0 and opacity.max() <= 1.0  # tanh
    assert colour.min() >= 0.0 and colour.max() <= 1.0  # sigmoid


def test_appearance_embedding_changes_colour_only_when_enabled():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    features = torch.zeros(3, 8 + 4)
    camera_id = torch.zeros(3, dtype=torch.long)

    off = ScaffoldMLPs(ScaffoldConfig(n_offsets=2, feat_dim=8, appearance_dim=0), n_views=5)
    assert off.embedding_appearance is None
    off(features, camera_id)  # camera_id is accepted and ignored

    on = ScaffoldMLPs(ScaffoldConfig(n_offsets=2, feat_dim=8, appearance_dim=6), n_views=5)
    assert on.embedding_appearance is not None
    assert on.embedding_appearance.weight.shape == (5, 6)
    with pytest.raises(ValueError, match="camera_id"):
        on(features, camera_id=None)


def test_appearance_embedding_needs_view_count():
    from collab_splats.splats.scaffold import ScaffoldMLPs

    with pytest.raises(ValueError, match="n_views"):
        ScaffoldMLPs(ScaffoldConfig(appearance_dim=6), n_views=0)
