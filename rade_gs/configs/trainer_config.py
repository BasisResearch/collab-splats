from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
from pathlib import Path

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.trainer import TrainerConfig

@dataclass
class _ExperimentConfig(ExperimentConfig):
    """Patched experiment config to disable viewer."""
    vis: Literal[ 
        "viewer", "wandb", "tensorboard", "comet", "viewer+wandb", "viewer+tensorboard", "viewer+comet", "viewer_legacy", None
    ] = None
    """Which visualizer to use.""" 

class _TrainerConfig(TrainerConfig):
    """Patched trainer config to disable viewer."""
    vis: Literal[ 
        "viewer", "wandb", "tensorboard", "comet", "viewer+wandb", "viewer+tensorboard", "viewer+comet", "viewer_legacy", None
    ] = None
    """Which visualizer to use.""" 