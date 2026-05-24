"""cu121 migration verification — Phases 1, 2, 3.

Run:
    /opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_cu121_migration.py -v

All tests are hard gates: any failure blocks the migration merge.
"""
import sys

import numpy as np
import pytest


# ── Phase 1: Environment guards ───────────────────────────────────────────────


def test_python_version():
    assert sys.version_info >= (3, 11), (
        f"Expected Python >= 3.11, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_torch_version():
    import torch
    assert torch.__version__.startswith("2.4"), (
        f"Expected torch 2.4.x, got {torch.__version__}"
    )
    assert "cu121" in torch.__version__, (
        f"Expected cu121 build, got {torch.__version__}"
    )


def test_cuda_version():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    assert "12.1" in torch.version.cuda, (
        f"Expected CUDA 12.1, got {torch.version.cuda}"
    )


def test_numpy_not_downgraded():
    major, minor = [int(x) for x in np.__version__.split(".")[:2]]
    assert major >= 2, (
        f"numpy was downgraded below 2.0 — got {np.__version__}. "
        "Check pyproject.toml numpy>=1.26 constraint."
    )


def test_scipy_version():
    import scipy
    major, minor = [int(x) for x in scipy.__version__.split(".")[:2]]
    assert (major, minor) >= (1, 17), (
        f"Expected scipy >= 1.17, got {scipy.__version__}"
    )


def test_pycolmap_version():
    import pycolmap
    ver = pycolmap.__version__
    major = int(ver.split(".")[0])
    assert major >= 4, f"Expected pycolmap >= 4.0, got {ver}"


def test_bae_editable_install():
    import bae
    bae_file = bae.__file__
    assert bae_file is not None
    assert "/opt/bae" in bae_file, (
        f"bae not from /opt/bae editable install: {bae_file}"
    )


def test_viser_version():
    import viser
    parts = [int(x) for x in viser.__version__.split(".")[:3]]
    major, minor, patch = parts[0], parts[1], parts[2] if len(parts) > 2 else 0
    assert (major, minor, patch) >= (0, 2, 23), (
        f"Expected viser >= 0.2.23, got {viser.__version__}"
    )


# ── Phase 2: Import sweep ─────────────────────────────────────────────────────


def test_import_collab_splats_top_level():
    """import collab_splats must succeed — catches _patch_mapanything_torch_compat crash."""
    import collab_splats  # noqa: F401


def test_import_all_modules():
    """All collab_splats submodules must import without error."""
    modules = [
        "collab_splats.pointcloud",
        "collab_splats.pointcloud.base",
        "collab_splats.pointcloud.bundle_adjustment",
        "collab_splats.pointcloud.feedforward",
        "collab_splats.pointcloud.feedforward.base",
        "collab_splats.pointcloud.feedforward.vggtx",
        "collab_splats.pointcloud.feedforward.mapanything",
        "collab_splats.pointcloud.localization",
        "collab_splats.pointcloud.sfm",
        "collab_splats.pointcloud.utils",
        "collab_splats.pointcloud.wrappers",
        "collab_splats.pointcloud.loop_closure",
        "collab_splats.pointcloud.loop_closure.alignment",
        "collab_splats.pointcloud.loop_closure.closure",
        "collab_splats.pointcloud.loop_closure.pose_graph",
        "collab_splats.pointcloud.loop_closure.retrieval",
        "collab_splats.pointcloud.loop_closure.submap",
        "collab_splats.semantics",
        "collab_splats.semantics.features",
        "collab_splats.semantics.compression",
        "collab_splats.semantics.segmentation",
        "collab_splats.semantics.utils",
        "collab_splats.mesh",
        "collab_splats.mesh.base",
        "collab_splats.mesh.poisson",
        "collab_splats.mesh.tsdf",
        "collab_splats.mesh.utils",
        "collab_splats.nerfstudio.method_configs.rade_gs",
        "collab_splats.nerfstudio.method_configs.rade_features",
        "collab_splats.nerfstudio.models.rade_gs",
        "collab_splats.nerfstudio.models.rade_features",
        "collab_splats.utils",
        "collab_splats.utils.torch_utils",
        "collab_splats.utils.frame_sampling",
        "collab_splats.nerfstudio.utils.camera_utils",
        "collab_splats.wrapper.splatter",
        # collab_splats.dashboard excluded: requires 'panel' which is not installed
    ]
    import importlib
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    assert not failed, "Import failures:\n" + "\n".join(failed)


def test_flagged_package_imports():
    """Packages flagged for numpy 2.x compat — all must import cleanly.

    These were explicitly flagged in the migration spec as having unknown numpy 2.x
    compatibility. All must resolve before Phase 4.
    """
    packages = {
        "pyntcloud": "import pyntcloud",
        "mobile_sam": "import mobile_sam",
        "maskclip_onnx": "import maskclip_onnx",
        "uniception": "import uniception",
        "meshlib": "import meshlib.mrmeshpy as mrmeshpy",
    }
    failed = []
    for name, stmt in packages.items():
        try:
            exec(stmt)
        except Exception as e:
            failed.append(f"{name}: {e}")
    assert not failed, (
        "Flagged packages failed to import — investigate numpy 2.x / torch 2.4 compat:\n"
        + "\n".join(failed)
    )


# ── Phase 3: Migration smoke tests ────────────────────────────────────────────


def test_gsplat_rade_fork():
    """rasterization_2dgs_inria_wrapper is a rade-specific symbol absent in official gsplat."""
    from gsplat import rasterization_2dgs_inria_wrapper  # noqa: F401


def test_gsplat_not_overwritten():
    """gsplat installed is the rade fork, not official PyPI gsplat 1.4.0."""
    import gsplat
    assert hasattr(gsplat, "rasterization_2dgs_inria_wrapper"), (
        f"gsplat at {gsplat.__file__} is missing rade-specific symbol — "
        "nerfstudio install may have overwritten gsplat-rade fork"
    )


def test_gtsam_sl4_manifold():
    """gtsam-develop 4.3a1 provides SL4 manifold; PyPI gtsam 4.2.1 does not."""
    from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4  # noqa: F401


def test_bae_use_cudss():
    """bae key symbols importable: TrackingTensor, PCG solver, LM optimiser."""
    import pypose  # must import before bae
    from bae.autograd.function import TrackingTensor, map_transform  # noqa: F401
    from bae.utils.pysolvers import PCG  # noqa: F401
    from bae.optim import LM  # noqa: F401


def test_bae_cuda_backend():
    """bae active with CUDA 12.1 / torch 2.4."""
    import pypose  # noqa: F401
    import bae  # noqa: F401
    import torch
    assert torch.__version__.startswith("2.4"), f"Wrong torch: {torch.__version__}"
    assert "12.1" in torch.version.cuda, f"Wrong CUDA: {torch.version.cuda}"


def test_nerfstudio_installed_local():
    """nerfstudio is the local /workspace/nerfstudio install, not a PyPI package."""
    import nerfstudio.field_components.activations as _ns_probe
    ns_file = _ns_probe.__file__
    assert ns_file is not None
    assert "/workspace/nerfstudio" in ns_file, (
        f"nerfstudio loaded from unexpected location: {ns_file}. "
        "Expected /workspace/nerfstudio — PyPI nerfstudio may have been installed."
    )


def test_mapanything_compat_patch_safe():
    """_patch_mapanything_torch_compat must not raise RuntimeError at import time."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator  # noqa: F401


def test_pycolmap_api_surface():
    """pycolmap 4.0.4 API: all constructors and new 4.x methods present and callable."""
    import pycolmap
    import numpy as np

    recon = pycolmap.Reconstruction()
    track = pycolmap.Track()

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=224,
        height=224,
        params=[200.0, 200.0, 112.0, 112.0],
        camera_id=1,
    )
    recon.add_camera_with_trivial_rig(camera)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    rot = pycolmap.Rotation3d(R)
    cam_from_world = pycolmap.Rigid3d(rot, t)

    image = pycolmap.Image(name="frame_0000.jpg", camera_id=1, image_id=1)
    recon.add_image_with_trivial_frame(image, cam_from_world)

    assert len(recon.images) == 1, f"Expected 1 image, got {len(recon.images)}"
    assert len(recon.cameras) == 1, f"Expected 1 camera, got {len(recon.cameras)}"


def test_nerfstudio_method_configs():
    """rade-gs and rade-features appear in nerfstudio's method registry."""
    import collab_splats.nerfstudio.method_configs.rade_gs  # noqa: F401
    import collab_splats.nerfstudio.method_configs.rade_features  # noqa: F401
    from nerfstudio.configs.method_configs import all_methods
    assert "rade-gs" in all_methods, (
        f"rade-gs missing from nerfstudio registry. Keys: {sorted(all_methods)}"
    )
    assert "rade-features" in all_methods, (
        f"rade-features missing from nerfstudio registry. Keys: {sorted(all_methods)}"
    )


def test_collab_data_installed():
    """collab_data package must be installed (BasisResearch private dep)."""
    import collab_data  # noqa: F401


def test_nerfstudio_patched_deps():
    """nerfacc and timm resolve to patched (unpinned) versions."""
    import nerfacc
    import timm
    nerfacc_parts = [int(x) for x in nerfacc.__version__.split(".")[:3]]
    assert tuple(nerfacc_parts) >= (0, 5, 2), (
        f"Expected nerfacc >= 0.5.2, got {nerfacc.__version__}"
    )
    timm_parts = [int(x) for x in timm.__version__.split(".")[:2]]
    assert tuple(timm_parts) >= (0, 6), (
        f"Expected timm >= 0.6.7, got {timm.__version__}"
    )


def test_splatfacto_uses_gsplat_rade():
    """nerfstudio.models.splatfacto imports without error and sees gsplat-rade."""
    import nerfstudio.models.splatfacto  # noqa: F401
    from gsplat import rasterization_2dgs_inria_wrapper  # noqa: F401
