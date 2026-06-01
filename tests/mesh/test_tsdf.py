import numpy as np
import pytest
from pathlib import Path


def _synthetic_frames(N=3, H=32, W=32):
    """Synthetic data: flat depth plane at 1m, random colors, identity poses."""
    depths = np.ones((N, H, W), dtype=np.float32)
    rgbs = (np.random.rand(N, H, W, 3) * 0.5 + 0.25).astype(np.float32)
    # Cam-to-world: identity + small x-translations
    c2w = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    for i in range(N):
        c2w[i, 0, 3] = i * 0.05
    # Intrinsics: ~45 deg fov
    f = float(W)
    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = f
    intrinsics[:, 1, 1] = f
    intrinsics[:, 0, 2] = W / 2.0
    intrinsics[:, 1, 2] = H / 2.0
    return depths, rgbs, c2w, intrinsics


def test_open3d_tsdf_returns_mesh_result(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion
    from collab_splats.mesh.base import MeshResult

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    result = creator.create(depths, rgbs, c2w, intrinsics)
    assert isinstance(result, MeshResult)


def test_open3d_tsdf_writes_ply(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    result = creator.create(depths, rgbs, c2w, intrinsics)
    assert result.mesh_path.exists()
    assert result.mesh_path.suffix == ".ply"


def test_open3d_tsdf_clean_repair_raises_meshlib_incompatible(tmp_path):
    """clean_repair=True is hard-disabled: the installed meshlib's addPartByMask
    API is incompatible (tsdf.py:98-103), so create() raises AssertionError
    pointing the user to clean_repair=False. Accepted-environment behavior —
    do NOT install meshlib, do NOT change production.
    """
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=True)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    with pytest.raises(AssertionError, match="clean_repair disabled"):
        creator.create(depths, rgbs, c2w, intrinsics)


def test_open3d_tsdf_creates_output_dir(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    nested = tmp_path / "a" / "b" / "c"
    creator = Open3DTSDFFusion(output_dir=nested, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    creator.create(depths, rgbs, c2w, intrinsics)
    assert nested.exists()
