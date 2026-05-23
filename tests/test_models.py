import torch
import pytest

from nerfstudio.data.scene_box import SceneBox

from collab_splats.nerfstudio.models.rade_gs import RadegsModelConfig, RadegsModel
from collab_splats.nerfstudio.models.rade_features import (
    RadegsFeaturesModelConfig,
    RadegsFeaturesModel,
)


def make_scene_box(aabb_scale: float = 1.0) -> SceneBox:
    return SceneBox(
        aabb=torch.tensor(
            [
                [-aabb_scale, -aabb_scale, -aabb_scale],
                [aabb_scale, aabb_scale, aabb_scale],
            ],
            dtype=torch.float32,
        )
    )


def make_features_metadata(channels: int = 8, height: int = 4, width: int = 4):
    return {
        "feature_type": "maskclip",
        "feature_dims": {
            "maskclip": (channels, height, width),
            "dinov2": (channels, height, width),
        },
    }


@pytest.fixture
def scene_box():
    return make_scene_box()


@pytest.fixture
def features_metadata():
    return make_features_metadata()


def test_radegs_model(scene_box):
    """Test RadeGS model instantiation."""
    cfg = RadegsModelConfig(output_depth_during_training=False)
    cfg.sh_degree = 0
    model = RadegsModel(cfg, scene_box=scene_box, num_train_data=1)
    assert model is not None


def test_radegs_features_model(scene_box, features_metadata):
    """Test RadeGS Features model instantiation."""
    cfg = RadegsFeaturesModelConfig(output_depth_during_training=False)
    cfg.sh_degree = 0
    model = RadegsFeaturesModel(
        cfg,
        scene_box=scene_box,
        num_train_data=1,
        metadata=features_metadata,
    )
    assert model is not None


def test_populate_text_encoder_talk2dino(scene_box):
    """populate_text_encoder wires up similarity_fx for talk2dino feature type."""
    from unittest.mock import patch, MagicMock
    import torch.nn as nn

    class _FakeEncoder(nn.Module):
        def score_queries(self, *args, **kwargs):
            pass

    mock_encoder_cls = MagicMock(return_value=_FakeEncoder())

    cfg = RadegsFeaturesModelConfig(output_depth_during_training=False)
    cfg.sh_degree = 0

    metadata = {
        "feature_type": "talk2dino",
        "feature_dims": {
            "talk2dino": (8, 4, 4),
        },
    }

    with patch(
        "collab_splats.nerfstudio.models.rade_features.BaseFeatureExtractor.get",
        return_value=mock_encoder_cls,
    ):
        model = RadegsFeaturesModel(
            cfg,
            scene_box=scene_box,
            num_train_data=1,
            metadata=metadata,
        )

    assert model.similarity_fx is not None
    assert model.similarity_fx == model.text_encoder.score_queries


def test_populate_text_encoder_non_queryable(scene_box):
    """Non-queryable feature types leave similarity_fx as None."""
    cfg = RadegsFeaturesModelConfig(output_depth_during_training=False)
    cfg.sh_degree = 0

    metadata = {
        "feature_type": "dinov2",
        "feature_dims": {
            "dinov2": (8, 4, 4),
        },
    }

    model = RadegsFeaturesModel(
        cfg,
        scene_box=scene_box,
        num_train_data=1,
        metadata=metadata,
    )

    assert model.similarity_fx is None
