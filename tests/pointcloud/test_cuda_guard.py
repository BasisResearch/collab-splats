import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_without_gpu(module):
    """Import `module` in a fresh interpreter with every GPU hidden; return the finished process."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": REPO}
    return subprocess.run([sys.executable, "-c", f"import {module}"], env=env, cwd=REPO, capture_output=True, text=True)


def test_pointcloud_import_without_gpu_names_the_requirement():
    """No GPU: a clear ImportError from our package, not VGGT-X's CUDA warmup traceback."""
    proc = _import_without_gpu("collab_splats.pointcloud")

    assert proc.returncode != 0
    assert "ImportError: collab_splats.pointcloud needs a CUDA GPU" in proc.stderr
    assert "warmup_gelu_fused" not in proc.stderr


def test_mesh_imports_without_gpu():
    """The mesh package stays importable on CPU; only texturing needs CUDA."""
    proc = _import_without_gpu("collab_splats.mesh")

    assert proc.returncode == 0, proc.stderr
