"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

# Custom configs
from collab_splats.models.rade_features_model import RadegsFeaturesModelConfig
from collab_splats.datamanagers.features_datamanager import (
    FeatureSplattingDataManagerConfig,
)
from collab_splats.utils.trainer_config import _TrainerConfig

rade_features_method = MethodSpecification(
    config=_TrainerConfig(
        method_name="rade-features",  # TODO: rename to your own model
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FeatureSplattingDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=RadegsFeaturesModelConfig(
                use_depth_normal_loss=True,
                depth_normal_lambda=0.05,
                depth_ratio=0.6,
                sh_degree=0,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "distill_features": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-4,
                    max_steps=10000,
                ),
            },
            "decoder": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="RaDe-Features combines RaDe-GS with feature splatting.",
)
