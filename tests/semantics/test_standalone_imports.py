"""
Semantics-only install contract: heavy reconstruction deps stay out of the semantics import graph.

- fresh-process checks: importing a module must not load the named heavy dependency
"""

import subprocess
import sys

import pytest

# (module imported in a fresh process, top-level module that must stay unloaded)
LAZY = [
    ("collab_splats.geometry", "bae"),
]


@pytest.mark.parametrize(("module", "heavy"), LAZY, ids=lambda v: str(v))
def test_fresh_import_skips_heavy_dep(module, heavy):
    code = f"import sys, {module}; print({heavy!r} in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", f"import {module} loaded {heavy}"
