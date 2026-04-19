"""Configuration loading utilities for Splatter workflows."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
from mergedeep import merge


class ConfigLoader:
    """
    Load and merge hierarchical YAML configurations.

    Supports inheritance with the following priority (highest to lowest):
    1. Runtime overrides (passed programmatically)
    2. Dataset config
    3. Base config

    Each config file should contain all sections (preprocess, training, meshing).
    Dataset configs inherit from base and can override any values.
    """

    def __init__(self, config_dir: Union[str, Path]):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing:
                - base.yaml (default configuration)
                - datasets/ (dataset-specific configs)
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise ValueError(f"Config directory not found: {config_dir}")

        base_path = self.config_dir / "base.yaml"
        if not base_path.exists():
            raise ValueError(f"base.yaml not found in {config_dir}")

        self.base_config = self._load_yaml(base_path)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        # Use mergedeep for clean deep merging
        return merge({}, base, override)

    def load(
        self,
        dataset: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load and merge configuration.

        Args:
            dataset: Dataset config name (from datasets/ subdirectory)
            overrides: Optional dictionary of runtime overrides

        Returns:
            Merged configuration dictionary with all sections:
            - preprocess: SfM and preprocessing settings
            - training: Model training parameters
            - meshing: Mesh generation settings
            - file_path, method, etc: Top-level splatter settings

        Raises:
            ValueError: If dataset config not found
        """
        # Start with base
        config = self.base_config.copy()

        # Merge dataset config
        dataset_path = self.config_dir / "datasets" / f"{dataset}.yaml"
        if not dataset_path.exists():
            raise ValueError(
                f"Dataset config not found: {dataset_path}\n"
                f"Available datasets: {self.list_datasets()}"
            )
        dataset_config = self._load_yaml(dataset_path)
        config = self._deep_merge(config, dataset_config)

        # Apply runtime overrides
        if overrides:
            config = self._deep_merge(config, overrides)

        return config

    def list_datasets(self) -> list:
        """
        List available dataset configs.

        Returns:
            List of available dataset names
        """
        datasets_dir = self.config_dir / "datasets"
        if not datasets_dir.exists():
            return []
        return sorted([f.stem for f in datasets_dir.glob("*.yaml")])


def parse_cli_overrides(override_strings: list) -> Dict[str, Any]:
    """
    Parse command-line override strings into a config dict.

    Supports nested keys using dot notation.

    Args:
        override_strings: List of 'key=value' or 'section.key=value' strings

    Returns:
        Dictionary of overrides

    Example:
        >>> parse_cli_overrides(['method=rade-gs', 'preprocess.sfm_tool=colmap'])
        {'method': 'rade-gs', 'preprocess': {'sfm_tool': 'colmap'}}
    """
    overrides: Dict[str, Any] = {}
    for override in override_strings:
        if "=" not in override:
            raise ValueError(f"Invalid override: '{override}'. Expected 'key=value'")

        key, value = override.split("=", 1)

        # Type conversion
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
            value = float(value) if "." in value else int(value)

        # Handle nested keys (e.g., 'preprocess.sfm_tool=colmap')
        keys = key.split(".")
        current = overrides
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    return overrides
