"""Verify the dashboard can be imported without triggering NumPy ABI warnings."""
import subprocess
import sys

import pytest


def test_dashboard_import_no_numpy_warning():
    """Importing the dashboard must not produce a UserWarning about NumPy."""
    # First check if panel is available — if not, skip rather than fail.
    check = subprocess.run(
        [sys.executable, "-c", "import panel"],
        capture_output=True,
    )
    if check.returncode != 0:
        pytest.skip("panel not installed — skipping NumPy ABI warning check")

    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c",
         "from collab_splats.dashboard import App"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Dashboard import raised UserWarning (likely NumPy ABI):\n{result.stderr}"
    )


@pytest.mark.xfail(reason="splatter.py has top-level 'import torch'; needs lazy import fix")
def test_collab_splats_init_no_torch():
    """collab_splats top-level __init__ must not import torch at module load time."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import collab_splats; import sys; assert 'torch' not in sys.modules, "
         "'torch was imported at collab_splats import time'"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
