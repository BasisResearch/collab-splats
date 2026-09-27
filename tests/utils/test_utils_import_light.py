"""
The utils package imports without torch, so io helpers stay usable torch-free.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _import_without_torch(module: str) -> subprocess.CompletedProcess:
    """
    Import `module` in a fresh interpreter where `import torch` raises.
    """
    code = f"import sys; sys.modules['torch'] = None; import {module}"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_utils_package_imports_without_torch():
    proc = _import_without_torch("collab_splats.utils")
    assert proc.returncode == 0, proc.stderr
