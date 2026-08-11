import numpy as np
import pytest


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
    from collab_splats.mesh.base import MeshResult
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

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


def test_open3d_tsdf_clean_repair_runs_and_keeps_one_mesh_ply(tmp_path):
    """clean_repair=True cleans in place — no second filename for readers to probe for.

    Every reader in the repo (Reconstructor's skip-check, splatter, the dashboard, the remote
    push) looks for `mesh.ply`; a separate `mesh_clean.ply` would need a precedence rule in each.
    """
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=True)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    result = creator.create(depths, rgbs, c2w, intrinsics)

    assert result.mesh_path == tmp_path / "mesh.ply"
    assert result.mesh_path.exists()
    assert not (tmp_path / "mesh_clean.ply").exists()
    assert sorted(p.name for p in tmp_path.glob("*.ply")) == ["mesh.ply"]


def test_open3d_tsdf_defaults_to_no_clean_repair(tmp_path):
    """The default must be the value that works — it is what the Reconstructor path gets.

    `_run_tsdf_mesh` constructs the fusion without passing clean_repair, so a True default that
    raised made every config-driven mesh stage die after fusing every frame.
    """
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path)
    assert creator.clean_repair is False
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    assert creator.create(depths, rgbs, c2w, intrinsics).mesh_path.exists()


def test_open3d_tsdf_creates_output_dir(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    nested = tmp_path / "a" / "b" / "c"
    creator = Open3DTSDFFusion(output_dir=nested, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    creator.create(depths, rgbs, c2w, intrinsics)
    assert nested.exists()


def test_open3d_tsdf_rejects_rgb_in_0_255_range(tmp_path):
    """rgbs is documented [0, 1]; [0, 255] silently fuses a black mesh, so refuse it."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        creator.create(depths, rgbs * 255.0, c2w, intrinsics)
