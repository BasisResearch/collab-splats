"""
Import style contract: absolute imports only, grouped by the pyproject isort sections.

- groups: stdlib, general third-party, model/method upstreams (known_models), then our own code
"""

import ast
from pathlib import Path

import isort
import pytest

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("collab_splats", "evals", "tests")
FILES = sorted(p for root in ROOTS for p in (REPO / root).rglob("*.py"))
CONFIG = isort.Config(settings_path=str(REPO))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_relative_imports(path):
    tree = ast.parse(path.read_text())
    relative = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level]
    assert not relative, f"relative imports at lines {relative}; use absolute collab_splats.* imports"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_imports_sorted_into_sections(path):
    assert isort.check_file(str(path), config=CONFIG), "run isort; imports are out of section order"
