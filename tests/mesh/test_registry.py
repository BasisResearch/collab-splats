import pytest
from pathlib import Path


def test_get_mesh_creator_open3d_tsdf(tmp_path):
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = get_mesh_creator("open3d_tsdf", output_dir=tmp_path)
    assert isinstance(creator, Open3DTSDFFusion)
    assert creator.output_dir == tmp_path


def test_get_mesh_creator_depth_normal_poisson(tmp_path):
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.poisson import DepthNormalPoisson

    creator = get_mesh_creator("depth_normal_poisson", output_dir=tmp_path)
    assert isinstance(creator, DepthNormalPoisson)


def test_get_mesh_creator_gaussians_poisson(tmp_path):
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.poisson import GaussiansPoisson

    creator = get_mesh_creator("gaussians_poisson", output_dir=tmp_path)
    assert isinstance(creator, GaussiansPoisson)


def test_get_mesh_creator_unknown_raises():
    from collab_splats.mesh import get_mesh_creator

    with pytest.raises(ValueError, match="Unknown mesh method"):
        get_mesh_creator("nonexistent", output_dir=Path("/tmp"))


def test_get_mesh_creator_passes_kwargs(tmp_path):
    from collab_splats.mesh import get_mesh_creator

    creator = get_mesh_creator("open3d_tsdf", output_dir=tmp_path, voxel_size=0.05)
    assert creator.voxel_size == 0.05
