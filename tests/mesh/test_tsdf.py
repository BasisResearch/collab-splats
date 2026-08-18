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
    """The default must be the value that works — cleanup is opt-in.

    A True default that raised would make every caller who omits the flag die after fusing
    every frame. `_run_tsdf_mesh` forwards it explicitly; notebooks and evals often do not.
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


########
# uint8 RGB passthrough + principal-point guard (native-resolution TSDF fusion)
########


def _flat_scene(h=32, w=32):
    """Two identity-pose frames looking at a flat plane at depth 1."""
    depths = np.ones((2, h, w), dtype=np.float32)
    c2w = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    K = np.tile(
        np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32), (2, 1, 1)
    )
    return depths, c2w, K


def test_create_uint8_rgb_matches_float(tmp_path):
    """uint8 RGB fuses to the same mesh as the equivalent [0,1] float RGB."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    depths, c2w, K = _flat_scene()
    rgb_u8 = np.full((2, 32, 32, 3), 200, dtype=np.uint8)
    rgb_f = rgb_u8.astype(np.float32) / 255.0

    m1 = Open3DTSDFFusion(output_dir=tmp_path / "a", voxel_size=0.05, sdf_trunc=0.2)
    m2 = Open3DTSDFFusion(output_dir=tmp_path / "b", voxel_size=0.05, sdf_trunc=0.2)
    p1 = m1.create(depths, rgb_u8, c2w, K).mesh_path
    p2 = m2.create(depths, rgb_f, c2w, K).mesh_path
    assert p1.read_bytes() == p2.read_bytes()


def test_create_principal_point_outside_grid_raises(tmp_path):
    """Original-res K paired with model-res depth must fail loudly, not fuse a collapsed mesh."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    depths, c2w, K = _flat_scene()
    K = K.copy()
    K[:, 0, 2] = 500.0  # cx far outside the 32-px grid
    rgbs = np.zeros((2, 32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="[Pp]rincipal point"):
        Open3DTSDFFusion(output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.2).create(
            depths, rgbs, c2w, K
        )


def test_create_fuses_adapter_native_output(tmp_path):
    """The native-res adapter's uint8 output fuses without the float-range guard firing."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
    from tests.mesh.test_utils import _tiny_ff_result

    class FakeStore:
        def __len__(self):
            return 2

        def images(self):
            return np.full((2, 16, 16, 3), 128, dtype=np.uint8)

    ff = _tiny_ff_result()
    native_K = np.tile(
        np.array([[16, 0, 8], [0, 16, 8], [0, 0, 1]], dtype=np.float32), (2, 1, 1)
    )
    depths, rgbs, c2w, K = _feedforward_to_tsdf_inputs(
        ff, frame_store=FakeStore(), native_intrinsics=native_K
    )
    result = Open3DTSDFFusion(output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.2).create(
        depths, rgbs, c2w, K
    )
    assert result.mesh_path.exists()
