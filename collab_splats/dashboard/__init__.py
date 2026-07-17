# collab_splats/dashboard/__init__.py
"""Splats dashboard package.

Exports resolve lazily (PEP 562): SplatsApp/GpuWorker pull the heavy torch/pyvista
stack, and the fast-binding CLI (serve.py) must be able to import light members
(OperationLog) and bind the HTTP server before that stack loads.
"""

from typing import Any

# Public name -> (module, attribute). run_app points at the fast-binding server.
_LAZY_EXPORTS = {
    "SplatsApp": ("collab_splats.dashboard.app", "SplatsApp"),
    "run_app": ("collab_splats.dashboard.serve", "run_app"),
    "GpuWorker": ("collab_splats.dashboard.gpu_worker", "GpuWorker"),
    "OperationLog": ("collab_splats.dashboard.operation_log", "OperationLog"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve exports on first access so importing the package stays light."""
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list:
    return sorted(__all__)
