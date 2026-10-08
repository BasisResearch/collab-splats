def test_bae_imports():
    import bae  # noqa: F401
    import pypose  # noqa: F401


def test_bae_cuda_version():
    import torch

    cuda_version = torch.version.cuda
    assert cuda_version is not None
    major = int(cuda_version.split(".")[0])
    assert major >= 12, f"Expected CUDA >= 12, got {cuda_version}"


def test_bundle_adjustment_module_loads():
    from collab_splats.geometry.bundle_adjustment import (  # noqa: F401
        BundleAdjustmentConfig,
    )
