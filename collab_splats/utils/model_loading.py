"""Utility functions for loading trained nerfstudio models."""

from pathlib import Path
from typing import Tuple, Union


def load_checkpoint(
    config_path: Union[str, Path],
    test_mode: str = "inference",
) -> Tuple:
    """
    Load a trained nerfstudio model checkpoint.

    Args:
        config_path: Path to the config.yml file from a trained model
        test_mode: Evaluation mode - "test", "val", or "inference" (default)

    Returns:
        Tuple of (config, pipeline, checkpoint_path, step)

    Example:
        >>> from collab_splats.utils import load_checkpoint
        >>> config, pipeline, ckpt_path, step = load_checkpoint("outputs/scene/rade-gs/config.yml")
        >>> model = pipeline.model
        >>> outputs = model.get_outputs(camera)
    """
    # Import here to avoid circular dependency during plugin discovery
    from nerfstudio.utils.eval_utils import eval_setup

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return eval_setup(config_path, test_mode=test_mode)
