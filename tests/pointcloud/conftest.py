"""Shared fixtures for pointcloud tests.

Also ensures the project-root evals/ package is importable as `evals.datasets`,
not shadowed by tests/evals/ (which pytest adds to sys.path as a package).
The eviction hack is deliberately duplicated per consuming subtree rather than
hoisted into tests/conftest.py: an autouse eviction of `evals` modules at root
scope would also run inside tests/evals/ itself and break that suite's
self-imports (mirrored in tests/geometry/conftest.py).
"""
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).parents[2])


@pytest.fixture(autouse=True)
def _fix_evals_import():
    """Evict tests/evals from sys.modules so project-root evals/ is used."""
    # Insert project root before tests/ directory
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    elif sys.path[0] != _PROJECT_ROOT:
        sys.path.remove(_PROJECT_ROOT)
        sys.path.insert(0, _PROJECT_ROOT)

    # Evict any cached evals module that points to tests/evals/
    for mod in list(sys.modules):
        if mod == "evals" or mod.startswith("evals."):
            m = sys.modules[mod]
            f = getattr(m, "__file__", "") or ""
            if "/tests/evals" in f or (not f and "/collab-splats/evals" not in f):
                del sys.modules[mod]

    yield
