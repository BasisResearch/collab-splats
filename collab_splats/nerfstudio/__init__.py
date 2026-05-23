__all__ = [
    "RadegsModel",
    "RadegsModelConfig",
    "RadegsFeaturesModel",
    "RadegsFeaturesModelConfig",
    "load_checkpoint",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading nerfstudio/CUDA unless actually needed."""
    if name in ("RadegsModel", "RadegsModelConfig", "RadegsFeaturesModel", "RadegsFeaturesModelConfig"):
        from .models import RadegsModel, RadegsModelConfig, RadegsFeaturesModel, RadegsFeaturesModelConfig

        _g = {
            "RadegsModel": RadegsModel,
            "RadegsModelConfig": RadegsModelConfig,
            "RadegsFeaturesModel": RadegsFeaturesModel,
            "RadegsFeaturesModelConfig": RadegsFeaturesModelConfig,
        }
        return _g[name]
    if name == "load_checkpoint":
        from .utils.model_loading import load_checkpoint

        return load_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
