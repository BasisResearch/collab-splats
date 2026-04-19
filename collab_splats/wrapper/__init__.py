__all__ = [
    "SplatterConfig",
    "Splatter",
]


def __getattr__(name: str):  # type: ignore[override]
    if name in ("Splatter", "SplatterConfig"):
        from .splatter import Splatter, SplatterConfig
        globals()["Splatter"] = Splatter
        globals()["SplatterConfig"] = SplatterConfig
        return globals()[name]
    if name in ("ConfigLoader", "parse_cli_overrides"):
        from .config import ConfigLoader, parse_cli_overrides
        globals()["ConfigLoader"] = ConfigLoader
        globals()["parse_cli_overrides"] = parse_cli_overrides
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
