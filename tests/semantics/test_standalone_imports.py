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
    ("collab_splats.preproc", "pycolmap"),
    ("collab_splats.semantics.segmentation", "mobile_sam"),
    ("collab_splats.geometry.transforms", "open3d"),
    ("collab_splats.semantics.lifting", "open3d"),
]


@pytest.mark.parametrize(("module", "heavy"), LAZY, ids=lambda v: str(v))
def test_fresh_import_skips_heavy_dep(module, heavy):
    code = f"import sys, {module}; print({heavy!r} in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", f"import {module} loaded {heavy}"


# Heavy reconstruction deps no semantics module may load
HEAVY = ("bae", "pypose", "pycolmap", "mobile_sam", "open3d", "gsplat", "collab_data")


def test_every_semantics_module_skips_heavy_deps():
    code = (
        "import pkgutil, sys, collab_splats.semantics as s\n"
        "for m in pkgutil.walk_packages(s.__path__, s.__name__ + '.'):\n"
        "    __import__(m.name)\n"
        f"print(sorted(set({HEAVY!r}) & set(sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "[]", proc.stdout
