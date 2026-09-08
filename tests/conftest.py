"""pytest configuration and session-level fixtures.

Shims and environment fixes that must run before any test module is imported.
"""

# ── sys.path: make top-level project dir importable (for evals.* etc.) ───────
import pathlib as _pathlib
import sys as _sys

_project_root = _pathlib.Path(__file__).parents[1]
if str(_project_root) not in _sys.path:
    _sys.path.insert(0, str(_project_root))

# ── pkg_resources.packaging shim ─────────────────────────────────────────────
# setuptools>=71 removed pkg_resources.packaging as a submodule. maskclip_onnx
# (and some other old packages) do `from pkg_resources import packaging`, which
# raises ImportError on modern setuptools. Inject the standalone `packaging`
# library into the pkg_resources namespace so those imports resolve cleanly.
# This is safe: pkg_resources previously just re-exported packaging, so the API
# is identical.
import importlib
import sys

# pkg_resources ships with setuptools, which is only pinned (<70) by the `splatting`
# extra. A `collab_splats[semantics]`-only env has no maskclip_onnx to patch and may
# have a setuptools new enough to have dropped pkg_resources entirely, so treat the
# shim as best-effort rather than letting conftest import-fail the whole suite.
try:
    import pkg_resources
except ImportError:
    pkg_resources = None

if pkg_resources is not None and not hasattr(pkg_resources, "packaging"):
    import packaging as _packaging
    pkg_resources.packaging = _packaging
    # Also make `from pkg_resources import packaging` work via the module dict
    sys.modules["pkg_resources.packaging"] = _packaging
    for _submod in ("version", "specifiers", "requirements", "markers", "utils"):
        _full = f"packaging.{_submod}"
        if _full in sys.modules:
            sys.modules[f"pkg_resources.packaging.{_submod}"] = sys.modules[_full]
        else:
            try:
                sys.modules[f"pkg_resources.packaging.{_submod}"] = importlib.import_module(_full)
            except ImportError:
                pass
