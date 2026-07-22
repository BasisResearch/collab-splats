"""cu121 migration verification — Phases 1, 2, 3.

Run:
    /opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py -v

All tests are hard gates: any failure blocks the migration merge.
"""

import importlib.metadata as importlib_metadata
import importlib.util
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest

# ── Phase 1: Environment guards ───────────────────────────────────────────────


def test_python_version():
    assert sys.version_info >= (
        3,
        11,
    ), f"Expected Python >= 3.11, got {sys.version_info.major}.{sys.version_info.minor}"


def test_torch_version():
    import torch

    assert torch.__version__.startswith("2.5"), f"Expected torch 2.5.x, got {torch.__version__}"
    assert "cu121" in torch.__version__, f"Expected cu121 build, got {torch.__version__}"


def test_cuda_version():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    assert "12.1" in torch.version.cuda, f"Expected CUDA 12.1, got {torch.version.cuda}"


def test_numpy_not_downgraded():
    major, minor = [int(x) for x in np.__version__.split(".")[:2]]
    assert major >= 2, (
        f"numpy was downgraded below 2.0 — got {np.__version__}. " "Check pyproject.toml numpy>=1.26 constraint."
    )


def test_scipy_version():
    import scipy

    major, minor = [int(x) for x in scipy.__version__.split(".")[:2]]
    assert (major, minor) >= (1, 17), f"Expected scipy >= 1.17, got {scipy.__version__}"


def test_pycolmap_version():
    import pycolmap

    ver = pycolmap.__version__
    major = int(ver.split(".")[0])
    assert major >= 4, f"Expected pycolmap >= 4.0, got {ver}"


def test_bae_installed_version():
    """bae must be 0.2.4 (pypose/bae git source) installed in the active venv."""
    import bae

    # Version pin: bae 0.2.4 is the source pinned in [tool.uv.sources] (pypose/bae@0.2.4)
    assert importlib_metadata.version("bae") == "0.2.4", f"bae must be 0.2.4, got {importlib_metadata.version('bae')}"
    # Location: must live under the running interpreter's site-packages (venv-agnostic —
    # works for conda /opt/conda/... and uv /opt/venv/..., not a hardcoded path).
    site = sysconfig.get_path("purelib")
    assert bae.__file__ and bae.__file__.startswith(
        site
    ), f"bae not installed under active venv site-packages ({site}): {bae.__file__}"


def test_viser_version():
    import viser

    parts = [int(x) for x in viser.__version__.split(".")[:3]]
    major, minor, patch = parts[0], parts[1], parts[2] if len(parts) > 2 else 0
    assert (major, minor, patch) >= (0, 2, 23), f"Expected viser >= 0.2.23, got {viser.__version__}"


# ── Phase 2: Import sweep ─────────────────────────────────────────────────────


def test_import_collab_splats_top_level():
    """import collab_splats must succeed — catches _patch_mapanything_torch_compat crash."""
    import collab_splats  # noqa: F401


def test_import_all_modules():
    """All collab_splats submodules must import without error."""
    modules = [
        "collab_splats.pointcloud",
        "collab_splats.pointcloud.base",
        "collab_splats.geometry.bundle_adjustment",
        "collab_splats.pointcloud.feedforward",
        "collab_splats.pointcloud.feedforward.base",
        "collab_splats.pointcloud.feedforward.vggtx",
        "collab_splats.pointcloud.feedforward.mapanything",
        "collab_splats.localization",
        "collab_splats.pointcloud.sfm",
        "collab_splats.pointcloud.utils",
        "collab_splats.geometry.loop_closure.wrapper",
        "collab_splats.geometry.loop_closure",
        "collab_splats.geometry.loop_closure.matching",
        "collab_splats.geometry.loop_closure.graph",
        "collab_splats.geometry.loop_closure.submap",
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
        "collab_splats.preproc",
        "collab_splats.preproc.sampling",
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
    assert not failed, "Flagged packages failed to import — investigate numpy 2.x / torch 2.4 compat:\n" + "\n".join(
        failed
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
    """bae active with CUDA 12.1 / torch 2.5."""
    import pypose  # noqa: F401
    import bae  # noqa: F401
    import torch

    assert torch.__version__.startswith("2.5"), f"Wrong torch: {torch.__version__}"
    assert "12.1" in torch.version.cuda, f"Wrong CUDA: {torch.version.cuda}"


def test_nerfstudio_installed_local():
    """nerfstudio loads from the active venv site-packages (BasisResearch fork via pyproject)."""
    # tests/nerfstudio dir shadows the real nerfstudio package; temporarily remove it from sys.path
    tests_dir = Path(__file__).parent
    orig_path = sys.path.copy()
    try:
        # Clear nerfstudio from sys.modules cache to force reimport from site-packages
        sys.modules.pop("nerfstudio", None)
        sys.modules.pop("nerfstudio.field_components.activations", None)
        # Remove tests/ from sys.path to allow site-packages nerfstudio to be found
        sys.path = [p for p in sys.path if Path(p).resolve() != tests_dir.resolve()]
        import nerfstudio.field_components.activations as _ns_probe

        ns_file = _ns_probe.__file__
    finally:
        sys.path = orig_path

    assert ns_file is not None
    # Location check: must resolve under the running interpreter's site-packages
    # (venv-agnostic — conda or uv), not a stale /workspace/nerfstudio or PyPI build.
    site = sysconfig.get_path("purelib")
    assert ns_file.startswith(site), (
        f"nerfstudio loaded from unexpected location: {ns_file}. "
        f"Expected site-packages under the active venv ({site}) — "
        "stale /workspace/nerfstudio or PyPI nerfstudio may be on sys.path."
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

    assert "rade-gs" in all_methods, f"rade-gs missing from nerfstudio registry. Keys: {sorted(all_methods)}"
    assert (
        "rade-features" in all_methods
    ), f"rade-features missing from nerfstudio registry. Keys: {sorted(all_methods)}"


@pytest.mark.skipif(
    importlib.util.find_spec("collab_data") is None,
    reason="collab-data is a private dep installed at deploy (git creds); absent in creds-less builds",
)
def test_collab_data_installed():
    """collab_data package must be installed (BasisResearch private dep)."""
    import collab_data  # noqa: F401


def test_nerfstudio_patched_deps():
    """nerfacc and timm resolve to patched (unpinned) versions."""
    import nerfacc
    import timm

    nerfacc_parts = [int(x) for x in nerfacc.__version__.split(".")[:3]]
    assert tuple(nerfacc_parts) >= (0, 5, 2), f"Expected nerfacc >= 0.5.2, got {nerfacc.__version__}"
    timm_parts = [int(x) for x in timm.__version__.split(".")[:2]]
    assert tuple(timm_parts) >= (0, 6), f"Expected timm >= 0.6.7, got {timm.__version__}"


def test_splatfacto_uses_gsplat_rade():
    """nerfstudio.models.splatfacto imports without error and sees gsplat-rade."""
    import nerfstudio.models.splatfacto  # noqa: F401
    from gsplat import rasterization_2dgs_inria_wrapper  # noqa: F401


def test_bae_cudss_importable():
    """bae built with USE_CUDSS=1: CuDirectSparseSolver must import and instantiate."""
    import pypose  # noqa: F401 — must precede bae imports
    from bae.sparse.solve import CuDirectSparseSolver
    import torch

    assert torch.cuda.is_available(), "CUDA not available — CuDSS build meaningless"
    solver = CuDirectSparseSolver()
    assert solver is not None
