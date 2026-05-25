from __future__ import annotations


def __getattr__(name: str):
    if name in ("SemanticsDashboard", "build_app", "run_app"):
        from .semantics import SemanticsDashboard, build_app, run_app  # noqa: F401

        globals()["SemanticsDashboard"] = SemanticsDashboard
        globals()["build_app"] = build_app
        globals()["run_app"] = run_app
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SemanticsDashboard", "build_app", "run_app"]
