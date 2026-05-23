# Design Spec: `collab_splats/mesh/` Module

**Date:** 2026-04-28
**Branch:** `refactor/core-modules`
**Status:** Approved

---

## Context

`collab_splats/utils/mesh.py` (1764 lines, ported from dn-splatter) has two problems:

1. **Hard nerfstudio dependency at import time** — `from nerfstudio.models.splatfacto import SplatfactoModel`, `eval_setup`, `Cameras`, `OrientedBox` are top-level imports. The module crashes to import without nerfstudio installed.
2. **Entangled concerns** — every exporter does (a) load trained model from checkpoint, (b) extract arrays from it, (c) fuse into mesh. Steps (a)+(b) are nerfstudio-specific; step (c) is generic geometry.

Only `Open3DTSDFFusion` is currently used. `TSDFFusion` (vdbfusion), `LevelSetExtractor`, and `MarchingCubesMesh` are dropped. `DepthNormalPoisson` and `GaussiansPoisson` kept as stubs for future use.

---

## Architecture

```
collab_splats/mesh/
  __init__.py       # get_mesh_creator(), MeshResult re-export
  base.py           # BaseMeshCreator, MeshResult — zero nerfstudio dep
  tsdf.py           # Open3DTSDFFusion — array-in/mesh-out
  poisson.py        # DepthNormalPoisson + GaussiansPoisson (stubs)
  utils.py          # clean_repair_mesh, normals2vertex, features2vertex,
                    #   find_depth_edges, align_geometry_floor, get_floor_plane,
                    #   mesh_clustering

collab_splats/nerfstudio/
  mesh_adapter.py   # NEW: extract_mesh_inputs(pipeline) -> arrays

collab_splats/utils/
  mesh.py           # DELETED — one caller (splatter.py:439) updated
```

All nerfstudio imports are contained in `nerfstudio/mesh_adapter.py`. `collab_splats/mesh/` has no nerfstudio dependency.

---

## `base.py`

```python
@dataclass
class MeshResult:
    mesh_path: Path
    pcd_path: Path | None = None

@dataclass
class BaseMeshCreator:
    output_dir: Path

    def create(
        self,
        depths: np.ndarray,      # (N, H, W) float32, meters
        rgbs: np.ndarray,        # (N, H, W, 3) float32 [0, 1]
        c2w: np.ndarray,         # (N, 4, 4) cam-to-world, OpenCV convention
        intrinsics: np.ndarray,  # (N, 3, 3)
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError
```

No `eval_setup`, no checkpoint paths, no nerfstudio types.

---

## `tsdf.py` — `Open3DTSDFFusion`

Dataclass with config fields mirroring current `Open3DTSDFFusion`:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `output_dir` | `Path` | — | required |
| `voxel_size` | `float` | `0.01` | TSDF voxel size |
| `sdf_trunc` | `float` | `0.04` | truncation distance |
| `depth_scale` | `float` | `1.0` | depth unit scale |
| `clean_repair` | `bool` | `True` | run meshlib post-process |
| `max_hole_size` | `float` | `3.0` | passed to `clean_repair_mesh` |
| `max_edge_splits` | `int` | `10000` | passed to `clean_repair_mesh` |
| `use_largest` | `bool` | `False` | keep only largest component |

`create()` logic:
1. Build `o3d.pipelines.integration.ScalableTSDFVolume`
2. Iterate `N` frames — build RGBD from `rgbs[i]` + `depths[i]`, call `volume.integrate()` with extrinsic = `inv(c2w[i])`
3. `volume.extract_triangle_mesh()` → write `{output_dir}/mesh_tsdf.ply`
4. If `clean_repair=True` and `mm` available: run `clean_repair_mesh`, write `{output_dir}/mesh_tsdf_clean.ply`
5. Return `MeshResult(mesh_path=..., pcd_path=None)`

meshlib import guarded: `try: import meshlib.mrmeshpy as mm; _MM_AVAILABLE = True; except ImportError: _MM_AVAILABLE = False`. If `clean_repair=True` and not available, warn and skip.

---

## `poisson.py` — Stubs

```python
@dataclass
class DepthNormalPoisson(BaseMeshCreator):
    """Backproject rendered depth+normals → Poisson reconstruction."""
    poisson_depth: int = 9

    def create(self, depths, rgbs, c2w, intrinsics, normals=None, **kwargs) -> MeshResult:
        raise NotImplementedError("DepthNormalPoisson not yet implemented")

@dataclass
class GaussiansPoisson(BaseMeshCreator):
    """Direct Gaussian means+normals → Poisson reconstruction."""
    poisson_depth: int = 9

    def create(self, depths, rgbs, c2w, intrinsics, means=None, normals=None, **kwargs) -> MeshResult:
        raise NotImplementedError("GaussiansPoisson not yet implemented")
```

---

## `utils.py`

Verbatim move from `utils/mesh.py` (no logic changes):
- `clean_repair_mesh(mesh_path, max_hole_size, max_edge_splits, use_largest)`
- `_filter_mesh_components(mesh, use_largest)` (internal helper)
- `normals2vertex(mesh_vertices, points, normals, k, sdf_trunc)`
- `features2vertex(mesh_vertices, points, features, k, sdf_trunc)`
- `find_depth_edges(depth_im, threshold, dilation_itr)`
- `align_geometry_floor(geometry, floor_normal, floor_point)`
- `get_floor_plane(pcd)`
- `mesh_clustering(pcd, ...)`

meshlib guard at top: `try: import meshlib.mrmeshpy as mm; except ImportError: mm = None`. Functions that use `mm` check at call time and raise `ImportError` with install hint.

---

## `__init__.py`

```python
from collab_splats.mesh.base import BaseMeshCreator, MeshResult
from collab_splats.mesh.tsdf import Open3DTSDFFusion
from collab_splats.mesh.poisson import DepthNormalPoisson, GaussiansPoisson

REGISTRY: dict[str, type[BaseMeshCreator]] = {
    "open3d_tsdf": Open3DTSDFFusion,
    "depth_normal_poisson": DepthNormalPoisson,
    "gaussians_poisson": GaussiansPoisson,
}

def get_mesh_creator(method: str, output_dir: Path, **kwargs) -> BaseMeshCreator:
    if method not in REGISTRY:
        raise ValueError(f"Unknown mesh method {method!r}. Choose from: {sorted(REGISTRY)}")
    return REGISTRY[method](output_dir=Path(output_dir), **kwargs)
```

---

## `nerfstudio/mesh_adapter.py`

```python
def extract_mesh_inputs(pipeline) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (depths, rgbs, c2w, intrinsics) arrays from a trained nerfstudio pipeline.

    Returns:
        depths:     (N, H, W) float32
        rgbs:       (N, H, W, 3) float32
        c2w:        (N, 4, 4) float32, cam-to-world OpenCV
        intrinsics: (N, 3, 3) float32
    """
```

All `eval_setup`, `SplatfactoModel`, `Cameras` imports live here. Splatter calls this then passes arrays to `get_mesh_creator(...).create(...)`.

---

## Splatter integration

`collab_splats/wrapper/splatter.py:439` currently: `from collab_splats.utils import mesh`

Replace with:
```python
from collab_splats.mesh import get_mesh_creator
from collab_splats.nerfstudio.mesh_adapter import extract_mesh_inputs
```

`SplatterConfig` gains optional `mesh_method: str = "open3d_tsdf"`.

---

## Tests

| File | What |
|------|------|
| `tests/mesh/test_tsdf.py` | `Open3DTSDFFusion.create()` with synthetic depth/rgb/pose arrays (no GPU, no nerfstudio) |
| `tests/mesh/test_utils.py` | `features2vertex`, `normals2vertex`, `find_depth_edges` |
| `tests/mesh/test_registry.py` | `get_mesh_creator` returns correct type; bad method raises `ValueError` |

Adapter tests skipped without nerfstudio env (same pattern as existing GPU smoke tests).

---

## What is dropped

| Item | Reason |
|------|--------|
| `TSDFFusion` | requires `vdbfusion`; superseded by `Open3DTSDFFusion` |
| `LevelSetExtractor` | uses non-standard model methods (`compute_level_surface_points`); not generalizable |
| `MarchingCubesMesh` | uses `model.get_density` + `get_closest_gaussians`; not generalizable |
| `GSMeshExporter` base class | replaced by `BaseMeshCreator` |
| `entrypoint()` tyro CLI | unused; drop |

---

## Out of scope

- Implementing `DepthNormalPoisson` or `GaussiansPoisson` bodies
- Feedforward meshing (mapanything direct mesh) — separate concern, stays in `stage/` for now
- Talk2DINO / MaskCLIP feature evaluation
