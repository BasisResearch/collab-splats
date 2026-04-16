"""Smoke tests verifying collab_splats.nerfstudio public API is importable."""


def test_model_imports():
    from collab_splats.nerfstudio import RadegsModel, RadegsModelConfig
    from collab_splats.nerfstudio import RadegsFeaturesModel, RadegsFeaturesModelConfig
    assert RadegsModel is not None
    assert RadegsModelConfig is not None
    assert RadegsFeaturesModel is not None
    assert RadegsFeaturesModelConfig is not None


def test_model_loading_import():
    from collab_splats.nerfstudio import load_checkpoint
    assert callable(load_checkpoint)


def test_trainer_config_import():
    from collab_splats.nerfstudio.trainer_config import _TrainerConfig
    assert _TrainerConfig is not None


def test_datamanager_import():
    from collab_splats.nerfstudio.datamanagers.features import (
        FeatureSplattingDataManager,
        FeatureSplattingDataManagerConfig,
    )
    assert FeatureSplattingDataManager is not None
    assert FeatureSplattingDataManagerConfig is not None


def test_method_configs_import():
    from collab_splats.nerfstudio.method_configs import rade_gs_method, rade_features_method
    assert rade_gs_method is not None
    assert rade_features_method is not None
