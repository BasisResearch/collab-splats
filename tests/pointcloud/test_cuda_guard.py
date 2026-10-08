import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_without_gpu(code):
    """Run `code` in a fresh interpreter with every GPU hidden; return the finished process."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": REPO}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=REPO, capture_output=True, text=True
    )


def test_pipeline_imports_without_gpu():
    """No GPU: pointcloud, the reconstructor and mesh import; VGGT-X's CUDA warmup is not reached."""
    proc = _run_without_gpu(
        "import collab_splats.pointcloud, collab_splats.reconstructor, collab_splats.mesh"
    )

    assert proc.returncode == 0, proc.stderr


def test_vggtx_load_without_gpu_names_the_requirement():
    """No GPU: loading the VGGT-X model raises our clear error, not VGGT-X's warmup traceback."""
    proc = _run_without_gpu(
        "from collab_splats.pointcloud import VGGTXCreator; VGGTXCreator()._load_model('cpu')"
    )

    assert proc.returncode != 0
    assert "RuntimeError: the vggtx backend needs a CUDA GPU" in proc.stderr
    assert "warmup_gelu_fused" not in proc.stderr
