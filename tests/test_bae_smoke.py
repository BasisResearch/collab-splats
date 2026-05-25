def test_bae_imports():
    import pypose
    import bae


def test_bae_cuda_version():
    import torch
    cuda_version = torch.version.cuda
    assert cuda_version is not None
    major = int(cuda_version.split(".")[0])
    assert major >= 12, f"Expected CUDA >= 12, got {cuda_version}"


def test_bundle_adjustment_module_loads():
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig
