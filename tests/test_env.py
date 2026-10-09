"""
Environment gate: torch/CUDA build, numpy 2, the bae and gsplat pins, the pycolmap 4 API.
"""

import importlib.metadata as importlib_metadata
import inspect
import sysconfig
import tomllib
from pathlib import Path

import numpy as np
import pycolmap
import torch

import bae
import gsplat
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.sparse.solve import CuDirectSparseSolver
from bae.utils.pysolvers import PCG
from gsplat.losses import (
    depth_l1_loss,
    normal_cosine_loss,
    opacity_reg_loss,
    scale_reg_loss,
    ssim_loss,
)
from gsplat.strategy import MCMCStrategy

from collab_splats.splats import GSPLAT_COMMIT


def test_torch_version():
    assert torch.__version__.startswith("2.5"), (
        f"Expected torch 2.5.x, got {torch.__version__}"
    )
    assert "cu121" in torch.__version__, (
        f"Expected cu121 build, got {torch.__version__}"
    )


def test_cuda_version():
    assert torch.cuda.is_available(), "CUDA not available"
    assert "12.1" in torch.version.cuda, f"Expected CUDA 12.1, got {torch.version.cuda}"


def test_numpy_not_downgraded():
    major, minor = [int(x) for x in np.__version__.split(".")[:2]]
    assert major >= 2, (
        f"numpy was downgraded below 2.0 — got {np.__version__}. "
        "Check pyproject.toml numpy>=1.26 constraint."
    )


def test_bae_installed_version():
    """
    bae must be 0.2.5 (pypose/bae git source) installed in the active venv.
    """
    # Version pin: bae 0.2.5 is the source pinned in [tool.uv.sources] (pypose/bae@0.2.5)
    assert importlib_metadata.version("bae") == "0.2.5", (
        f"bae must be 0.2.5, got {importlib_metadata.version('bae')}"
    )

    # Location: under the running interpreter's site-packages, not a hardcoded venv path
    site = sysconfig.get_path("purelib")
    assert bae.__file__ and bae.__file__.startswith(site), (
        f"bae not installed under active venv site-packages ({site}): {bae.__file__}"
    )


def test_gsplat_upstream_pinned():
    """
    gsplat is upstream nerfstudio-project/gsplat @ d2f5c0f: gsplat.losses, 2DGS, MCMC and extra_signals exist.
    """
    losses = (
        depth_l1_loss,
        normal_cosine_loss,
        opacity_reg_loss,
        scale_reg_loss,
        ssim_loss,
    )
    assert all(callable(fn) for fn in losses)
    assert inspect.isclass(MCMCStrategy)
    assert hasattr(gsplat, "rasterization_2dgs")
    assert "extra_signals" in inspect.signature(gsplat.rasterization).parameters


def test_gsplat_commit_constant_matches_pyproject():
    """
    collab_splats.splats.GSPLAT_COMMIT must match the [tool.uv.sources] gsplat rev in pyproject.toml.
    """
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    gsplat_source = pyproject["tool"]["uv"]["sources"]["gsplat"]
    assert gsplat_source["rev"] == GSPLAT_COMMIT


def test_bae_use_cudss():
    """
    bae key symbols importable: TrackingTensor, PCG solver, LM optimiser.
    """
    assert all(
        symbol is not None for symbol in (TrackingTensor, map_transform, LM, PCG)
    )


def test_pycolmap_api_surface():
    """
    pycolmap 4.x API: all constructors and new 4.x methods present and callable.
    """
    recon = pycolmap.Reconstruction()
    pycolmap.Track()

    # One pinhole camera on a trivial rig
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=224,
        height=224,
        params=[200.0, 200.0, 112.0, 112.0],
        camera_id=1,
    )
    recon.add_camera_with_trivial_rig(camera)

    # One image at the identity pose on a trivial frame
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    rot = pycolmap.Rotation3d(R)
    cam_from_world = pycolmap.Rigid3d(rot, t)
    image = pycolmap.Image(name="frame_0000.jpg", camera_id=1, image_id=1)
    recon.add_image_with_trivial_frame(image, cam_from_world)

    assert len(recon.images) == 1, f"Expected 1 image, got {len(recon.images)}"
    assert len(recon.cameras) == 1, f"Expected 1 camera, got {len(recon.cameras)}"


def test_bae_cudss_importable():
    """
    bae built with USE_CUDSS=1: CuDirectSparseSolver must import and instantiate.
    """
    assert torch.cuda.is_available(), "CUDA not available — CuDSS build meaningless"
    solver = CuDirectSparseSolver()
    assert solver is not None
