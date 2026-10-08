"""
Fresh-process imports of the packages whose import graph crosses geometry and localization.

- geometry.tracks imports localization; localizer imports pointcloud only under TYPE_CHECKING
"""

import subprocess
import sys

import pytest

MODULES = (
    "collab_splats.pointcloud",
    "collab_splats.geometry",
    "collab_splats.geometry.tracks",
    "collab_splats.localization",
    "collab_splats.reconstructor",
)


@pytest.mark.parametrize("module", MODULES)
def test_fresh_import(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, proc.stderr
