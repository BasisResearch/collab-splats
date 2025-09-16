import torch
import pytest

from nerfstudio.data.scene_box import SceneBox

from collab_splats.models.rade_gs_model import RadegsModelConfig, RadegsModel
from collab_splats.models.rade_features_model import (
    RadegsFeaturesModelConfig,
    RadegsFeaturesModel,
)

# Disable TorchDynamo to prevent Transformer/PyTorch import errors in CI
torch._dynamo.disable()

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
        "feature_type": "clip-vit",
        "feature_dims": {
            "clip-vit": (channels, height, width),
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
