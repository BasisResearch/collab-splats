from __future__ import annotations

from pathlib import Path

from collab_splats.mesh.base import BaseMeshCreator, MeshResult
from collab_splats.mesh.poisson import DepthNormalPoisson, GaussiansPoisson
from collab_splats.mesh.tsdf import Open3DTSDFFusion
from collab_splats.mesh.utils import pointcloud_to_mesh

REGISTRY: dict[str, type[BaseMeshCreator]] = {
    "open3d_tsdf": Open3DTSDFFusion,
    "Open3DTSDFFusion": Open3DTSDFFusion,
    "depth_normal_poisson": DepthNormalPoisson,
    "DepthNormalPoisson": DepthNormalPoisson,
    "gaussians_poisson": GaussiansPoisson,
    "GaussiansPoisson": GaussiansPoisson,
}


def get_mesh_creator(method: str, output_dir: Path, **kwargs) -> BaseMeshCreator:
    if method not in REGISTRY:
        raise ValueError(f"Unknown mesh method {method!r}. Choose from: {sorted(REGISTRY)}")
    return REGISTRY[method](output_dir=Path(output_dir), **kwargs)


__all__ = [
    "get_mesh_creator",
    "pointcloud_to_mesh",
    "BaseMeshCreator",
    "MeshResult",
    "Open3DTSDFFusion",
    "DepthNormalPoisson",
    "GaussiansPoisson",
    "REGISTRY",
]
