# Mesh Texture Bake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a final, opt-in texturing step to the `mesh` stage that turns `mesh.ply` into an error-bounded decimated, UV-atlased, albedo + world-normal textured OBJ and a subdivided vertex-colour `mesh_baked.ply`, leaving `mesh.ply` untouched.

**Architecture:** One new module `collab_splats/mesh/texture.py` holds the whole chain as small pure functions (`decimate`, `measure_deviation`, `project_textures`, `export_obj`, `bake_vertex_colors`) plus the orchestrator `texture_mesh(...)`. `_run_tsdf_mesh` in the wrapper calls it once after the mesh is written, for either `mesh.source`. Images come from `splats.zarr/rgb` (`source: splats`) or `frames.zarr` rows matching `splats.zarr.attrs["image_ids"]` (`source: frames`); poses/K/normals always come from `splats.zarr`. Config nests under `mesh.texture`. Deviation bound is in `voxel_size` units (scene-relative rule).

**Tech Stack:** Open3D legacy (cleanup, Taubin, vertex normals) + Open3D tensor (`compute_uvatlas`, `project_images_to_albedo`, `RaycastingScene.compute_distance`), `meshoptimizer==0.2.30a0` (`simplify` with `SIMPLIFY_ERROR_ABSOLUTE`), trimesh (`TextureVisuals` OBJ export, subdivide, `to_color`), PIL, zarr.

**Spec:** `docs/superpowers/specs/2026-08-29-mesh-texture-bake-design.md`

**Deviations from spec (recorded here):** (a) report omits "UV chart count" — Open3D's `compute_uvatlas` does not expose it; texel fill fraction stays. (b) the spec hooks texturing inside `mesh_from_tsdf_inputs`. That function only receives the fused arrays — it has no splats.zarr, no normals, no frames.zarr. The hook lives in `_run_tsdf_mesh` (wrapper) instead, which has all four. Output layout, config, and behaviour are exactly as specified.

**Repo rules that apply to every task**
- Python: `/opt/venv/reconstruction/bin/python` (never bare `python`).
- Commits: `git add -f <paths>` (docs/superpowers is force-added) then `git commit --only <paths> -m ...` — the worktree has foreign uncommitted changes from other sessions (`CLAUDE.md`, `collab_splats/remote/rerun.py`, `pyproject.toml`, `uv.lock`, tests, docs). Never `git add -A`, never `git commit -a`.
- Style: imports at top, `logging` not print, one-line docstrings with `"""` on their own lines, block comments, flat test functions.
- Do not run repo-wide `black .` (venv black is newer than the pin) — format only files you touched: `black <file>` and `isort <file>`.
- Never import pymeshlab anywhere in this package (segfaults alongside open3d).

---

## File map

| Path | Role |
|---|---|
| `collab_splats/mesh/texture.py` | **Create.** Whole texturing chain + bake-back. |
| `collab_splats/mesh/utils.py` | **Modify.** Add `_texture_images(splats_zarr, frames_zarr, source)` next to `_splats_to_tsdf_inputs`. |
| `collab_splats/wrapper/reconstructor.py:599-708` | **Modify.** `_run_tsdf_mesh` gains `texture: dict | None`, calls `texture_mesh` after the mesh exists; `Reconstructor.mesh()` passes `mesh_cfg["texture"]`. |
| `configs/base.yaml:145-165` | **Modify.** `mesh.texture` block. |
| `configs/README.md:366-370` | **Modify.** Option table rows. |
| `docs/examples/texture_mesh.py` | **Create.** Standalone CLI: mesh.ply + splats.zarr [+ frames.zarr] → `texture/<source>/`. |
| `pyproject.toml` | **Modify.** `meshoptimizer==0.2.30a0` pin. |
| `tests/mesh/test_texture.py` | **Create.** Flat tests. |
| `docs/superpowers/CHANGELOG.md`, `CLAUDE.md` | **Modify.** Entry + in-flight line. |

---

### Task 1: Dependency pin

**Files:**
- Modify: `pyproject.toml` (dependencies list, near `"trimesh",` line 54)

- [ ] **Step 1: Install into the venv**

Run:
```bash
uv pip install --python /opt/venv/reconstruction/bin/python meshoptimizer==0.2.30a0
/opt/venv/reconstruction/bin/python -c "import meshoptimizer as mo; print(mo.SIMPLIFY_ERROR_ABSOLUTE)"
```
Expected: `4`

- [ ] **Step 2: Pin in pyproject**

Add one line to the `dependencies = [` list, directly after `"trimesh",`:
```toml
"meshoptimizer==0.2.30a0",  # error-bounded QEM decimation (mesh.texture); alpha tag on PyPI, pin exact
```

- [ ] **Step 3: Commit (pyproject only — uv.lock has foreign uncommitted edits; lock refresh is deferred until that lands)**

```bash
git add pyproject.toml
git commit --only pyproject.toml -m "build(mesh): pin meshoptimizer==0.2.30a0 for error-bounded decimation"
```

---

### Task 2: `decimate` + `measure_deviation`

**Files:**
- Create: `collab_splats/mesh/texture.py`
- Create: `tests/mesh/test_texture.py`

- [ ] **Step 1: Write the failing tests**

`tests/mesh/test_texture.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import trimesh
from PIL import Image

from collab_splats.mesh.texture import decimate, measure_deviation

# Later tasks extend this import line as they add functions:
#   Task 3 → export_obj, project_textures; Task 4 → bake_vertex_colors; Task 5 → texture_mesh;
#   Task 6 → `import zarr` and `from collab_splats.mesh.utils import _texture_images`

########################################################################################
# Fixtures
########################################################################################


def _dense_plane(n: int = 60, noise: float = 0.0, seed: int = 0) -> o3d.geometry.TriangleMesh:
    """
    Unit plane in z=0 tessellated n×n, optional gaussian z-noise (marching-cubes-like input).
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([xs.ravel(), ys.ravel(), rng.normal(0, noise, n * n)], axis=1)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    return m


def _sphere(res: int = 40) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=res)


########################################################################################
# decimate / measure_deviation
########################################################################################


def test_decimate_respects_absolute_bound():
    m = _dense_plane(noise=0.002)
    bound = 0.01
    out, result_error = decimate(m, bound)
    dev = measure_deviation(m, out)
    assert len(out.triangles) < len(m.triangles)
    assert dev["p99"] <= 1.5 * bound
    assert result_error <= bound + 1e-6


def test_decimate_keeps_curvature_relative_to_planes():
    plane = _dense_plane(n=60)
    sphere = _sphere(res=40)
    bound = 0.01
    p, _ = decimate(plane, bound)
    s, _ = decimate(sphere, bound)
    # A plane collapses to a handful of triangles; a sphere must keep many to stay in bound
    assert len(p.triangles) < 0.05 * len(plane.triangles)
    assert len(s.triangles) > len(p.triangles)


def test_decimate_never_moves_vertices():
    m = _dense_plane(n=20, noise=0.001)
    out, _ = decimate(m, 0.01)
    v_in = {tuple(np.round(x, 9)) for x in np.asarray(m.vertices)}
    assert all(tuple(np.round(x, 9)) in v_in for x in np.asarray(out.vertices))
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q 2>&1 | tail -3`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'collab_splats.mesh.texture'`

- [ ] **Step 3: Create the module with `decimate` and `measure_deviation`**

`collab_splats/mesh/texture.py`:
```python
"""
Texture pass for the mesh stage: error-bounded decimation, UV atlas, albedo + world-normal
projection, textured OBJ, and a subdivided vertex-colour PLY bake-back.

- Runs after mesh.ply exists; never modifies mesh.ply.
- Outputs land in ``<backend_dir>/texture/<source>/``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import meshoptimizer as mo
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)

########################################################################################
# Geometry pass
########################################################################################


def decimate(
    mesh: o3d.geometry.TriangleMesh, max_error: float
) -> tuple[o3d.geometry.TriangleMesh, float]:
    """
    QEM decimation that stops at an absolute surface-deviation bound (scene units).

    - meshoptimizer ``simplify`` with ``SIMPLIFY_ERROR_ABSOLUTE``: vertices are only removed,
      never moved, so the result is a strict subset of the input vertex set.
    - ``target_index_count=3`` means "as few triangles as the bound allows".
    - Returns (decimated mesh, meshoptimizer's own result_error in scene units).
    """
    v = np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float32)
    idx = np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.uint32).ravel()
    dst = np.zeros_like(idx)
    err = np.zeros(1, dtype=np.float32)
    n = mo.simplify(
        dst,
        idx,
        v,
        target_index_count=3,
        target_error=float(max_error),
        options=mo.SIMPLIFY_ERROR_ABSOLUTE,
        result_error=err,
    )

    # Rebuild as a legacy mesh and drop the vertices no triangle references any more
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(dst[:n].reshape(-1, 3).astype(np.int32)),
    )
    out.remove_unreferenced_vertices()
    return out, float(err[0])


def measure_deviation(
    reference: o3d.geometry.TriangleMesh,
    decimated: o3d.geometry.TriangleMesh,
    voxel_size: float | None = None,
) -> dict:
    """
    Distance from every reference vertex to the decimated surface (scene units).

    - p50/p90/p99/max, and the fraction beyond one voxel when ``voxel_size`` is given.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(decimated))
    q = o3d.core.Tensor(np.asarray(reference.vertices), dtype=o3d.core.float32)
    d = scene.compute_distance(q).numpy()
    report = {
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
    }
    if voxel_size is not None:
        report["frac_over_voxel"] = float((d > voxel_size).mean())
    return report
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q 2>&1 | tail -3`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/mesh/texture.py tests/mesh/test_texture.py && isort collab_splats/mesh/texture.py tests/mesh/test_texture.py
git add collab_splats/mesh/texture.py tests/mesh/test_texture.py
git commit --only collab_splats/mesh/texture.py tests/mesh/test_texture.py -m "feat(mesh): error-bounded decimation + deviation measurement for texture pass"
```

---

### Task 3: `project_textures` + `export_obj`

**Files:**
- Modify: `collab_splats/mesh/texture.py`
- Modify: `tests/mesh/test_texture.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/mesh/test_texture.py`:
```python
########################################################################################
# project_textures / export_obj
########################################################################################


def _box_and_camera():
    """
    Unit box at origin, one camera 5 units down +z looking at it, 256² image, K f=300.
    """
    box = o3d.geometry.TriangleMesh.create_box().translate([-0.5, -0.5, -0.5])
    c2w = np.eye(4)
    c2w[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    c2w[:3, 3] = [0, 0, 5]
    K = np.array([[300, 0, 128], [0, 300, 128], [0, 0, 1]], dtype=np.float64)
    return box, c2w[None], K[None]


def test_project_textures_albedo_and_normal():
    box, c2w, K = _box_and_camera()
    img = np.zeros((1, 256, 256, 3), np.uint8)
    img[..., 0] = 255  # solid red
    nrm = np.zeros((1, 256, 256, 3), np.float32)
    nrm[..., 2] = 1.0  # +z world normal everywhere
    tm, uv, albedo, normal = project_textures(box, img, nrm, c2w, K, tex_size=128)
    assert uv.shape == (len(box.triangles), 3, 2)
    assert albedo.shape == (128, 128, 3) and albedo.dtype == np.uint8
    filled = albedo.reshape(-1, 3).any(axis=1)
    assert filled.mean() > 0.05
    assert (albedo[filled][:, 0] == 255).all() and (albedo[filled][:, 1:] == 0).all()
    # normal encoded (n+1)/2*255 → +z maps to (128,128,255) ± 1
    assert normal.shape == (128, 128, 3) and normal.dtype == np.uint8
    nz = normal.reshape(-1, 3)[filled]
    assert np.abs(nz[:, 2].astype(int) - 255).max() <= 1
    assert np.abs(nz[:, :2].astype(int) - 127).max() <= 1


def test_project_textures_no_normals():
    box, c2w, K = _box_and_camera()
    img = np.full((1, 256, 256, 3), 200, np.uint8)
    _, _, albedo, normal = project_textures(box, img, None, c2w, K, tex_size=64)
    assert normal is None and albedo.shape == (64, 64, 3)


def test_export_obj_writes_textured_obj(tmp_path):
    box, c2w, K = _box_and_camera()
    img = np.zeros((1, 256, 256, 3), np.uint8)
    img[..., 0] = 255
    nrm = np.zeros((1, 256, 256, 3), np.float32)
    nrm[..., 2] = 1.0
    _, uv, albedo, normal = project_textures(box, img, nrm, c2w, K, tex_size=64)
    obj = export_obj(box, uv, albedo, normal, tmp_path)
    assert obj == tmp_path / "mesh.obj"
    assert (tmp_path / "albedo.png").exists()
    assert (tmp_path / "normal_world.png").exists()
    mtl = (tmp_path / "mesh.mtl").read_text()
    assert "map_Kd albedo.png" in mtl and "map_bump normal_world.png" in mtl
    assert "Kd 1.0" in mtl or "Kd 1 1 1" in mtl or "Kd 1.00000000 1.00000000 1.00000000" in mtl
    back = trimesh.load(obj, process=False, force="mesh")
    assert isinstance(back.visual, trimesh.visual.TextureVisuals)
    assert len(back.faces) == len(box.triangles)


def test_export_obj_without_normal(tmp_path):
    box, c2w, K = _box_and_camera()
    img = np.full((1, 256, 256, 3), 200, np.uint8)
    _, uv, albedo, _ = project_textures(box, img, None, c2w, K, tex_size=64)
    export_obj(box, uv, albedo, None, tmp_path)
    assert not (tmp_path / "normal_world.png").exists()
    assert "map_bump" not in (tmp_path / "mesh.mtl").read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'project_textures'` (after extending the import line at the top of the test file to `decimate, export_obj, measure_deviation, project_textures`)

- [ ] **Step 3: Implement**

Append to `collab_splats/mesh/texture.py` (after `measure_deviation`):
```python
########################################################################################
# Texture pass
########################################################################################


def _project(
    tm: o3d.t.geometry.TriangleMesh,
    images: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    tex_size: int,
) -> np.ndarray:
    """
    Visibility-weighted projection of (N,H,W,3) uint8 images into the mesh's UV atlas.
    """
    w2c = np.linalg.inv(c2w)
    tex = tm.project_images_to_albedo(
        [o3d.t.geometry.Image(np.ascontiguousarray(im)) for im in images],
        [o3d.core.Tensor(np.ascontiguousarray(k, dtype=np.float64)) for k in intrinsics],
        [o3d.core.Tensor(np.ascontiguousarray(e, dtype=np.float64)) for e in w2c],
        tex_size,
    )
    return tex.as_tensor().numpy()


def project_textures(
    mesh: o3d.geometry.TriangleMesh,
    images: np.ndarray,
    normals: np.ndarray | None,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    tex_size: int,
) -> tuple[o3d.t.geometry.TriangleMesh, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    UV-atlas the mesh and project albedo (+ optional world-space normal map) into it.

    - ``images`` (N,H,W,3) uint8; ``normals`` (N,H,W,3) float world-frame unit vectors or None.
    - ``c2w`` (N,4,4), ``intrinsics`` (N,3,3) at image resolution.
    - Returns (tensor mesh with ``triangle.texture_uvs``, uvs (F,3,2), albedo uint8, normal uint8|None).
    - Normal map is encoded ``(n+1)/2*255`` after renormalising the projected average.
    """
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    tm.compute_uvatlas(size=tex_size)
    uv = tm.triangle.texture_uvs.numpy().copy()

    albedo = _project(tm, images, c2w, intrinsics, tex_size)

    # Normals ride the same projector as uint8; decode, renormalise, re-encode
    normal = None
    if normals is not None:
        enc = np.clip((normals + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
        proj = _project(tm, enc, c2w, intrinsics, tex_size).astype(np.float32) / 255.0 * 2.0 - 1.0
        norm = np.linalg.norm(proj, axis=-1, keepdims=True)
        proj = np.where(norm > 1e-6, proj / np.maximum(norm, 1e-6), 0.0)
        normal = np.clip((proj + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    return tm, uv, albedo, normal


def export_obj(
    mesh: o3d.geometry.TriangleMesh,
    uv: np.ndarray,
    albedo: np.ndarray,
    normal: np.ndarray | None,
    out_dir: Path,
) -> Path:
    """
    Write mesh.obj + mesh.mtl + albedo.png [+ normal_world.png] with per-corner UVs.

    - trimesh needs one vertex per triangle corner to carry per-face UVs, so vertices are
      split (V[F]); the OBJ carries the same face count as the input.
    - Open3D's OBJ writers omit ``map_Kd``, hence trimesh here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    vs = v[f].reshape(-1, 3)
    fs = np.arange(len(vs)).reshape(-1, 3)

    # Kd=1 so viewers show the texture unattenuated (trimesh's default Kd is 0.4)
    Image.fromarray(albedo).save(out_dir / "albedo.png")
    mat = trimesh.visual.material.SimpleMaterial(
        image=Image.open(out_dir / "albedo.png"), diffuse=[255, 255, 255, 255]
    )
    tm = trimesh.Trimesh(vs, fs, process=False)
    tm.visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), material=mat)
    obj = out_dir / "mesh.obj"
    tm.export(obj, mtl_name="mesh.mtl")

    # trimesh names the texture after the material; rename to albedo.png and patch the MTL.
    # Append map_bump by hand — trimesh has no normal-map slot on SimpleMaterial.
    mtl = out_dir / "mesh.mtl"
    text = mtl.read_text()
    for p in out_dir.glob("material*.png"):
        text = text.replace(p.name, "albedo.png")
        p.unlink()
    if normal is not None:
        Image.fromarray(normal).save(out_dir / "normal_world.png")
        text = text.rstrip("\n") + "\nmap_bump normal_world.png\n"
    mtl.write_text(text)
    return obj
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q 2>&1 | tail -3`
Expected: all project/export tests PASS. If the `Kd` assertion fails, print the MTL and adjust the assertion to the exact `Kd` line trimesh writes for `diffuse=[255,255,255,255]` (must be 1.0-valued, not 0.4).

- [ ] **Step 5: Commit**

```bash
black collab_splats/mesh/texture.py tests/mesh/test_texture.py && isort collab_splats/mesh/texture.py tests/mesh/test_texture.py
git add collab_splats/mesh/texture.py tests/mesh/test_texture.py
git commit --only collab_splats/mesh/texture.py tests/mesh/test_texture.py -m "feat(mesh): UV atlas + albedo/normal projection + textured OBJ export"
```

---

### Task 4: `bake_vertex_colors` (ported from collab-data)

**Files:**
- Modify: `collab_splats/mesh/texture.py`
- Modify: `tests/mesh/test_texture.py`

- [ ] **Step 1: Add failing tests (ported from collab-data `tests/track_reprojection/test_bake.py`)**

Append:
```python
########################################################################################
# bake_vertex_colors (ported: collab-data d059661 tests/track_reprojection/test_bake.py)
########################################################################################

RED, GREEN, BLUE, YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)


def _quad_obj(tmp_path):
    """
    Unit quad in y=0 mapped onto a 2×2 texture; uv (0,0) is the LAST image row → BLUE.
    """
    img = np.array([[RED, GREEN], [BLUE, YELLOW]], dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / "tex.png")
    obj = tmp_path / "quad.obj"
    obj.with_suffix(".mtl").write_text("newmtl m\nmap_Kd tex.png\n")
    obj.write_text(
        "mtllib quad.mtl\nusemtl m\n"
        "v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
    )
    return obj


def test_bake_texture_onto_vertices(tmp_path):
    out = bake_vertex_colors(_quad_obj(tmp_path), target_faces=0)
    assert out == tmp_path / "quad_baked.ply"
    mesh = trimesh.load(out, process=False)
    assert len(mesh.faces) == 2
    colors = {
        tuple(v[[0, 2]].astype(int)): tuple(c[:3])
        for v, c in zip(mesh.vertices, np.asarray(mesh.visual.vertex_colors))
    }
    assert colors == {(0, 0): BLUE, (1, 0): YELLOW, (1, 1): GREEN, (0, 1): RED}
    assert isinstance(mesh.visual, trimesh.visual.ColorVisuals)


def test_bake_subdivides_up_to_target(tmp_path):
    obj = _quad_obj(tmp_path)
    for target, faces in [(0, 2), (2, 2), (3, 8), (8, 8), (9, 32)]:
        mesh = trimesh.load(
            bake_vertex_colors(obj, tmp_path / f"t{target}.ply", target_faces=target),
            process=False,
        )
        assert len(mesh.faces) == faces, target
    colors = np.asarray(trimesh.load(tmp_path / "t9.ply", process=False).visual.vertex_colors)[:, :3]
    assert {tuple(c) for c in colors} <= {RED, GREEN, BLUE, YELLOW}
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q -k bake 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'bake_vertex_colors'` (after adding it to the import line)

- [ ] **Step 3: Implement (verbatim port, logging instead of `log=print`)**

Append to `collab_splats/mesh/texture.py`:
```python
########################################################################################
# Bake-back: textured OBJ → subdivided vertex-colour PLY
# Ported from BasisResearch/collab-data @ d059661
# collab_data/track_reprojection/bake.py (bake_vertex_colors, _geometries, _to_color);
# `log=print` replaced by module logger, otherwise unchanged so both repos bake identically.
########################################################################################


def _geometries(loaded) -> list[trimesh.Trimesh]:
    """
    Triangle meshes of a trimesh load result (Scene or single mesh).
    """
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("scene contains no triangle meshes")
        return geoms
    return [loaded]


def _to_color(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Geometry + vertex colours only, no material.
    """
    colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.uint8)
    out = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=colors)
    return out


def bake_vertex_colors(mesh_path, out_path=None, target_faces: int = 500_000) -> Path:
    """
    Flatten a textured mesh to a vertex-coloured PLY; returns the path written.

    - Subdivides (each pass ×4 faces, UVs interpolated) until ``target_faces`` is reached;
      ``target_faces=0`` bakes at the source tessellation.
    - ``out_path`` defaults to ``<stem>_baked.ply`` beside the input.
    """
    mesh_path = Path(mesh_path)
    out_path = Path(out_path) if out_path else mesh_path.with_name(f"{mesh_path.stem}_baked.ply")
    parts = _geometries(trimesh.load(mesh_path, process=False))
    n_faces = sum(len(p.faces) for p in parts)
    logger.info(
        "loaded %s: %d verts / %d faces in %d geometries",
        mesh_path, sum(len(p.vertices) for p in parts), n_faces, len(parts),
    )

    # Subdivide until the face budget is met so vertex colours carry the texture detail
    n_sub = 0
    while target_faces and n_faces < target_faces:
        parts = [p.subdivide() for p in parts]
        n_faces = sum(len(p.faces) for p in parts)
        n_sub += 1
        logger.info("  subdivide %d: %d faces", n_sub, n_faces)

    baked = [_to_color(p) for p in parts]
    mesh = baked[0] if len(baked) == 1 else trimesh.util.concatenate(baked)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    logger.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q -k bake 2>&1 | tail -3`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
black collab_splats/mesh/texture.py tests/mesh/test_texture.py && isort collab_splats/mesh/texture.py tests/mesh/test_texture.py
git add collab_splats/mesh/texture.py tests/mesh/test_texture.py
git commit --only collab_splats/mesh/texture.py tests/mesh/test_texture.py -m "feat(mesh): port bake_vertex_colors from collab-data d059661"
```

---

### Task 5: `texture_mesh` orchestrator + report

**Files:**
- Modify: `collab_splats/mesh/texture.py`
- Modify: `tests/mesh/test_texture.py`

- [ ] **Step 1: Add failing tests**

Append:
```python
########################################################################################
# texture_mesh (full chain)
########################################################################################


def _write_box_ply(tmp_path) -> Path:
    box = o3d.geometry.TriangleMesh.create_box().translate([-0.5, -0.5, -0.5])
    box = box.subdivide_midpoint(3)  # 768 faces so decimation has something to remove
    p = tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(p), box)
    return p


def _cfg(**over):
    cfg = dict(
        enabled=True,
        source="splats",
        decimate_max_error=0.25,
        smooth_iterations=0,
        tex_size=64,
        bake_target_faces=2000,
    )
    cfg.update(over)
    return cfg


def test_texture_mesh_full_chain(tmp_path):
    mesh_path = _write_box_ply(tmp_path)
    _, c2w, K = _box_and_camera()
    img = np.zeros((1, 256, 256, 3), np.uint8)
    img[..., 0] = 255
    nrm = np.zeros((1, 256, 256, 3), np.float32)
    nrm[..., 2] = 1.0
    out = texture_mesh(mesh_path, tmp_path / "texture" / "splats", img, nrm, c2w, K, voxel_size=0.04, cfg=_cfg())
    d = tmp_path / "texture" / "splats"
    assert out == d / "mesh_baked.ply"
    for name in ["mesh.obj", "mesh.mtl", "albedo.png", "normal_world.png", "mesh_baked.ply", "texture_report.json"]:
        assert (d / name).exists(), name
    rep = json.loads((d / "texture_report.json").read_text())
    assert rep["faces"]["input"] == 768
    assert rep["faces"]["decimated"] < 768
    assert rep["decimate"]["max_error_voxels"] == 0.25
    assert rep["decimate"]["max_error_scene"] == pytest.approx(0.01)
    assert rep["deviation"]["p99"] <= 0.015
    assert 0 < rep["texture"]["texel_fill_fraction"] < 1
    assert rep["faces"]["baked"] >= 2000
    assert set(rep["wall_time_s"]) >= {"geometry", "project", "export", "bake"}
    # mesh.ply is untouched
    assert len(o3d.io.read_triangle_mesh(str(mesh_path)).triangles) == 768
    baked = trimesh.load(d / "mesh_baked.ply", process=False)
    assert isinstance(baked.visual, trimesh.visual.ColorVisuals)


def test_texture_mesh_no_decimation(tmp_path):
    mesh_path = _write_box_ply(tmp_path)
    _, c2w, K = _box_and_camera()
    img = np.full((1, 256, 256, 3), 200, np.uint8)
    texture_mesh(mesh_path, tmp_path / "t", img, None, c2w, K, voxel_size=0.04, cfg=_cfg(decimate_max_error=None))
    rep = json.loads((tmp_path / "t" / "texture_report.json").read_text())
    assert rep["faces"]["decimated"] == rep["faces"]["input"]
    assert rep["decimate"] is None and rep["deviation"] is None
    assert not (tmp_path / "t" / "normal_world.png").exists()


def test_texture_mesh_smoothing_changes_vertices(tmp_path):
    mesh_path = _write_box_ply(tmp_path)
    _, c2w, K = _box_and_camera()
    img = np.full((1, 256, 256, 3), 200, np.uint8)
    texture_mesh(mesh_path, tmp_path / "t", img, None, c2w, K, voxel_size=0.04, cfg=_cfg(decimate_max_error=None, smooth_iterations=5))
    v0 = np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices)
    v1 = trimesh.load(tmp_path / "t" / "mesh.obj", process=False, force="mesh").vertices
    # smoothing pulls box corners inward: smoothed extent < original extent
    assert np.ptp(v1, axis=0).max() < np.ptp(v0, axis=0).max()
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q -k texture_mesh 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'texture_mesh'` (after adding it to the import line)

- [ ] **Step 3: Implement**

Append to `collab_splats/mesh/texture.py`:
```python
########################################################################################
# Orchestrator
########################################################################################


def texture_mesh(
    mesh_path: Path,
    out_dir: Path,
    images: np.ndarray,
    normals: np.ndarray | None,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    voxel_size: float,
    cfg: dict,
) -> Path:
    """
    mesh.ply → cleanup → [Taubin] → [error-bounded decimation] → UV atlas → albedo/normal
    projection → mesh.obj → mesh_baked.ply, with texture_report.json. Returns mesh_baked.ply.

    - ``cfg`` keys: decimate_max_error (voxel units, None = off), smooth_iterations,
      tex_size, bake_target_faces.
    - Never rewrites ``mesh_path``.
    """
    mesh_path, out_dir = Path(mesh_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}
    report: dict = {"source_mesh": str(mesh_path), "faces": {}, "vertices": {}}

    # Geometry pass: Open3D legacy cleanup so the atlas + decimator see a sane manifold
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    report["faces"]["input"] = len(mesh.triangles)
    report["vertices"]["input"] = len(mesh.vertices)
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    report["faces"]["cleaned"] = len(mesh.triangles)
    report["vertices"]["cleaned"] = len(mesh.vertices)

    # Optional Taubin smoothing (volume-preserving, unlike Laplacian)
    smooth = int(cfg.get("smooth_iterations") or 0)
    if smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth)
    reference = o3d.geometry.TriangleMesh(mesh)

    # Decimation to an absolute bound expressed in voxels, then measured against the input
    max_err_vox = cfg.get("decimate_max_error")
    if max_err_vox is not None:
        max_err = float(max_err_vox) * float(voxel_size)
        mesh, result_error = decimate(mesh, max_err)
        report["decimate"] = {
            "max_error_voxels": float(max_err_vox),
            "max_error_scene": max_err,
            "result_error": result_error,
        }
        report["deviation"] = measure_deviation(reference, mesh, voxel_size)
    else:
        report["decimate"] = None
        report["deviation"] = None
    report["faces"]["decimated"] = len(mesh.triangles)
    report["vertices"]["decimated"] = len(mesh.vertices)
    mesh.compute_vertex_normals()
    times["geometry"] = time.time() - t0
    logger.info(
        "texture geometry: %d → %d faces (%.1fs)", report["faces"]["input"], len(mesh.triangles), times["geometry"]
    )

    # Texture pass: UV atlas + projection of every view (no subsampling)
    t0 = time.time()
    tex_size = int(cfg["tex_size"])
    _, uv, albedo, normal = project_textures(mesh, images, normals, c2w, intrinsics, tex_size)
    times["project"] = time.time() - t0
    report["texture"] = {
        "tex_size": tex_size,
        "n_views": int(len(images)),
        "texel_fill_fraction": float(albedo.reshape(-1, 3).any(axis=1).mean()),
        "normal_map": normal is not None,
    }
    logger.info("texture projection: %d views @ %d² (%.1fs)", len(images), tex_size, times["project"])

    # Export + bake-back
    t0 = time.time()
    obj = export_obj(mesh, uv, albedo, normal, out_dir)
    times["export"] = time.time() - t0
    t0 = time.time()
    baked = bake_vertex_colors(obj, out_dir / "mesh_baked.ply", target_faces=int(cfg["bake_target_faces"]))
    times["bake"] = time.time() - t0
    report["faces"]["baked"] = len(trimesh.load(baked, process=False).faces)
    report["wall_time_s"] = times
    (out_dir / "texture_report.json").write_text(json.dumps(report, indent=2))
    return baked
```

- [ ] **Step 4: Run whole test file**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/mesh/texture.py tests/mesh/test_texture.py && isort collab_splats/mesh/texture.py tests/mesh/test_texture.py
git add collab_splats/mesh/texture.py tests/mesh/test_texture.py
git commit --only collab_splats/mesh/texture.py tests/mesh/test_texture.py -m "feat(mesh): texture_mesh orchestrator with texture_report.json"
```

---

### Task 6: `_texture_images` (image/pose source from zarr)

**Files:**
- Modify: `collab_splats/mesh/utils.py` (insert after `_splats_to_tsdf_inputs`, ~line 711-815)
- Modify: `tests/mesh/test_texture.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/mesh/test_texture.py` (add `import zarr` and `from collab_splats.mesh.utils import _texture_images` to the top imports):
```python
########################################################################################
# _texture_images
########################################################################################


def _fake_splats_zarr(path, n=3, h=8, w=6, with_normal=True, image_ids=None):
    g = zarr.open(str(path), mode="w")
    g["rgb"] = np.full((n, h, w, 3), 10, np.uint8)
    g["c2w"] = np.tile(np.eye(4), (n, 1, 1))
    g["K"] = np.tile(np.eye(3), (n, 1, 1))
    if with_normal:
        g["normal"] = np.zeros((n, h, w, 3), np.float32)
    g.attrs["image_ids"] = list(range(n)) if image_ids is None else image_ids
    return path


def _fake_frames_zarr(path, n=5, h=8, w=6):
    g = zarr.open(str(path), mode="w")
    g["images"] = np.full((n, h, w, 3), 200, np.uint8)
    g["frame_idx"] = np.arange(n) * 10
    return path


def test_texture_images_splats(tmp_path):
    sz = _fake_splats_zarr(tmp_path / "splats.zarr")
    images, normals, c2w, K = _texture_images(sz, None, "splats")
    assert images.shape == (3, 8, 6, 3) and (images == 10).all()
    assert normals.shape == (3, 8, 6, 3)
    assert c2w.shape == (3, 4, 4) and K.shape == (3, 3, 3)


def test_texture_images_splats_no_normal(tmp_path):
    sz = _fake_splats_zarr(tmp_path / "splats.zarr", with_normal=False)
    _, normals, _, _ = _texture_images(sz, None, "splats")
    assert normals is None


def test_texture_images_frames(tmp_path):
    sz = _fake_splats_zarr(tmp_path / "splats.zarr", image_ids=[0, 2, 4])
    fz = _fake_frames_zarr(tmp_path / "frames.zarr")
    images, _, _, _ = _texture_images(sz, fz, "frames")
    assert images.shape == (3, 8, 6, 3) and (images == 200).all()


def test_texture_images_frames_shape_mismatch(tmp_path):
    sz = _fake_splats_zarr(tmp_path / "splats.zarr", h=4, w=4)
    fz = _fake_frames_zarr(tmp_path / "frames.zarr")
    with pytest.raises(ValueError, match="source: splats"):
        _texture_images(sz, fz, "frames")


def test_texture_images_bad_source(tmp_path):
    sz = _fake_splats_zarr(tmp_path / "splats.zarr")
    with pytest.raises(ValueError):
        _texture_images(sz, None, "renders")
```

`FrameStore.open` reads `images` and `frame_idx` (`collab_splats/preproc/frame_store.py:33`); the fixture writes exactly those two arrays.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -x -q -k texture_images 2>&1 | tail -3`
Expected: `ImportError: cannot import name '_texture_images'`

- [ ] **Step 3: Implement**

In `collab_splats/mesh/utils.py`, add `from collab_splats.preproc.frame_store import FrameStore` to the top imports (check for a circular import: `preproc` must not import `mesh`; `grep -rn "collab_splats.mesh" collab_splats/preproc/` must be empty). Insert directly after `_splats_to_tsdf_inputs`:
```python
def _texture_images(
    splats_zarr: Path, frames_zarr: Path | None, source: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """
    (images, normals|None, c2w, K) for the texture pass; every view, no subsampling.

    - ``source='splats'``: rendered RGB from splats.zarr.
    - ``source='frames'``: frames.zarr rows for splats.zarr ``attrs['image_ids']``; must match
      the render resolution exactly (poses/K are the rendered ones).
    - Normals are splats.zarr ``normal`` (world frame) when present.
    """
    store = zarr.open(str(splats_zarr), mode="r")
    c2w = np.asarray(store["c2w"])
    intrinsics = np.asarray(store["K"])
    normals = np.asarray(store["normal"]) if "normal" in store else None
    rgb_shape = tuple(store["rgb"].shape)

    if source == "splats":
        images = np.asarray(store["rgb"])
    elif source == "frames":
        if frames_zarr is None or not Path(frames_zarr).exists():
            raise FileNotFoundError(f"mesh.texture.source: frames needs frames.zarr (looked at {frames_zarr})")
        fs = FrameStore.open(frames_zarr)
        images = np.stack([fs.image(int(i)) for i in store.attrs["image_ids"]])
        if images.shape != rgb_shape:
            raise ValueError(
                f"frames.zarr images {images.shape} do not match splats renders {rgb_shape}; "
                "poses/K are the rendered ones — use mesh.texture.source: splats"
            )
    else:
        raise ValueError(f"mesh.texture.source must be 'splats' or 'frames', got {source!r}")
    return images, normals, c2w, intrinsics
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -q 2>&1 | tail -3`
Expected: all PASS. Also `/opt/venv/reconstruction/bin/python -m pytest tests/mesh -q 2>&1 | tail -3` — no regressions.

- [ ] **Step 5: Commit**

```bash
black collab_splats/mesh/utils.py tests/mesh/test_texture.py && isort collab_splats/mesh/utils.py tests/mesh/test_texture.py
git add collab_splats/mesh/utils.py tests/mesh/test_texture.py
git commit --only collab_splats/mesh/utils.py tests/mesh/test_texture.py -m "feat(mesh): _texture_images — splats or frames source for the texture pass"
```

---

### Task 7: Wire into `_run_tsdf_mesh` + `Reconstructor.mesh()` + config

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:599-708` and `:1616-1634`
- Modify: `configs/base.yaml:145-165`
- Modify: `configs/README.md:366-370`
- Test: `tests/wrapper/test_reconstructor.py` (find the existing `_run_tsdf_mesh` splats test with `grep -n "_run_tsdf_mesh\|splats_zarr" tests/wrapper/test_reconstructor.py`)

- [ ] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor.py` (mirror the setup of the nearest existing `_run_tsdf_mesh` test for `result`/monkeypatching; the point is that `texture_mesh` is called with the right arguments after the mesh is written, and not called when disabled):
```python
def test_run_tsdf_mesh_calls_texture_mesh(tmp_path, monkeypatch):
    from collab_splats.wrapper import reconstructor as R

    calls = []
    fake_mesh = tmp_path / "mesh.ply"

    def fake_inputs(*a, **k):
        return np.zeros((2, 4, 4)), np.zeros((2, 4, 4, 3), np.uint8), np.tile(np.eye(4), (2, 1, 1)), np.tile(np.eye(3), (2, 1, 1))

    def fake_mesh_from(*a, **k):
        fake_mesh.write_bytes(b"")
        return SimpleNamespace(mesh_path=fake_mesh)

    def fake_tex_images(splats_zarr, frames_zarr, source):
        return "IMG", "NRM", "C2W", "K"

    def fake_texture_mesh(mesh_path, out_dir, images, normals, c2w, K, voxel_size, cfg):
        calls.append((mesh_path, out_dir, images, normals, voxel_size, cfg["source"]))
        return out_dir / "mesh_baked.ply"

    monkeypatch.setattr("collab_splats.mesh.utils._splats_to_tsdf_inputs", fake_inputs)
    monkeypatch.setattr("collab_splats.mesh.utils.mesh_from_tsdf_inputs", fake_mesh_from)
    monkeypatch.setattr("collab_splats.mesh.utils._texture_images", fake_tex_images)
    monkeypatch.setattr("collab_splats.mesh.texture.texture_mesh", fake_texture_mesh)
    result = SimpleNamespace(extrinsics=np.zeros((2, 3, 4)))
    tex = dict(enabled=True, source="frames", decimate_max_error=0.25, smooth_iterations=0, tex_size=64, bake_target_faces=10)

    out = R._run_tsdf_mesh(result, tmp_path / "pc.zarr", tmp_path, 0.01, 0.04, 1.5, source="splats",
                           splats_zarr=tmp_path / "splats.zarr", frames_zarr=tmp_path / "frames.zarr", texture=tex)
    assert out == fake_mesh
    assert calls == [(fake_mesh, tmp_path / "texture" / "frames", "IMG", "NRM", 0.01, "frames")]

    calls.clear()
    R._run_tsdf_mesh(result, tmp_path / "pc.zarr", tmp_path, 0.01, 0.04, 1.5, source="splats",
                     splats_zarr=tmp_path / "splats.zarr", texture=dict(tex, enabled=False))
    assert calls == []
```
Add `from types import SimpleNamespace` and `import numpy as np` to that test file's imports if missing.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -x -q -k texture 2>&1 | tail -3`
Expected: `TypeError: _run_tsdf_mesh() got an unexpected keyword argument 'texture'`

- [ ] **Step 3: Implement the wrapper hook**

In `collab_splats/wrapper/reconstructor.py`, `_run_tsdf_mesh`:

1. Add the parameter after `splat_max_depth_grad`:
```python
    splat_max_depth_grad: float | None = None,
    texture: dict | None = None,
) -> Path:
```
2. Replace both `return mesh_result.mesh_path` lines (splats branch ~line 661, feedforward branch ~line 708) with `mesh_path = mesh_result.mesh_path` and add one shared tail at the end of the function. Restructure so the feedforward branch is no longer under an implicit `else` — the simplest edit: keep the `if source == "splats": ...` block ending in `mesh_path = mesh_result.mesh_path`, wrap the existing feedforward code in `else:` (indent it one level), then append:
```python
    # Texture pass (opt-in): decimate + UV + albedo/normal + bake-back beside mesh.ply.
    # Poses/K/normals always come from splats.zarr, so it is required for either mesh.source.
    if texture and texture.get("enabled"):
        from collab_splats.mesh.texture import texture_mesh
        from collab_splats.mesh.utils import _texture_images

        if splats_zarr is None or not splats_zarr.exists():
            raise FileNotFoundError(
                f"mesh.texture needs splats.zarr renders (looked at {splats_zarr}) — run the splats stage first"
            )
        images, normals, tex_c2w, tex_k = _texture_images(splats_zarr, frames_zarr, texture["source"])
        texture_mesh(
            mesh_path,
            output_dir / "texture" / texture["source"],
            images,
            normals,
            tex_c2w,
            tex_k,
            voxel_size=voxel_size,
            cfg=texture,
        )
    return mesh_path
```
3. In `Reconstructor.mesh()` (~line 1616) add `texture=mesh_cfg.get("texture"),` to the `_run_tsdf_mesh(...)` call, after `splat_max_depth_grad=...`.
4. In `Reconstructor.mesh()`, where `splats_zarr` is resolved for `source == "splats"` (~line 1590), make sure `splats_zarr` is also set (not `None`) when `source == "feedforward"` so the texture pass can find it: read the surrounding lines and, if `splats_zarr` is only assigned inside the splats branch, hoist `splats_zarr = self.backend_dir / "splats" / "splats.zarr"` above the branch (existence checks stay where they are).

- [ ] **Step 4: Config**

`configs/base.yaml`, append inside the `mesh:` block after `splat_max_depth_grad: null`:
```yaml
  # Texture pass — final mesh step, opt-in. Writes texture/<source>/{mesh.obj, mesh.mtl,
  # albedo.png, normal_world.png, mesh_baked.ply, texture_report.json}; mesh.ply untouched.
  # Needs splats.zarr renders (poses, K, normals) for either mesh.source.
  texture:
    enabled: false
    source: splats            # splats (rendered RGB) | frames (raw frames.zarr at render resolution)
    decimate_max_error: 0.25  # max surface deviation in voxel_size units; null = no decimation
    smooth_iterations: 0      # Taubin smoothing iterations before decimation (0 = off)
    tex_size: 8192            # UV atlas / texture side in texels
    bake_target_faces: 500000 # subdivide until ≥ this many faces before baking vertex colours
```

`configs/README.md`, after the `mesh.sdf_trunc` row (line ~370) add:
```markdown
| `mesh.texture.enabled` | bool | `false` | Texture pass after fusion: decimate, UV atlas, albedo + world-normal maps, `mesh.obj`, `mesh_baked.ply` under `texture/<source>/`; `mesh.ply` untouched |
| `mesh.texture.source` | str | `splats` | `splats` (rendered RGB) or `frames` (frames.zarr, must equal render resolution) |
| `mesh.texture.decimate_max_error` | float\|null | `0.25` | Absolute surface-deviation bound in `voxel_size` units; `null` = no decimation |
| `mesh.texture.smooth_iterations` | int | `0` | Taubin smoothing iterations before decimation |
| `mesh.texture.tex_size` | int | `8192` | Texture side in texels |
| `mesh.texture.bake_target_faces` | int | `500000` | Subdivision target for `mesh_baked.ply` |
```

- [ ] **Step 5: Run tests**

Run:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -q -k "texture or tsdf" 2>&1 | tail -3
/opt/venv/reconstruction/bin/python -c "import yaml; c=yaml.safe_load(open('configs/base.yaml')); print(c['mesh']['texture'])"
```
Expected: PASS; dict printed with the six keys.

- [ ] **Step 6: Commit**

```bash
black collab_splats/wrapper/reconstructor.py && isort collab_splats/wrapper/reconstructor.py
git add collab_splats/wrapper/reconstructor.py configs/base.yaml configs/README.md tests/wrapper/test_reconstructor.py
git commit --only collab_splats/wrapper/reconstructor.py configs/base.yaml configs/README.md tests/wrapper/test_reconstructor.py -m "feat(mesh): wire mesh.texture pass into the mesh stage"
```
(`tests/wrapper/test_reconstructor.py` has foreign uncommitted edits — inspect `git diff tests/wrapper/test_reconstructor.py` first; if the foreign hunk is unrelated, stage only yours with `git add -p` and commit `--only` the staged file, otherwise leave that file uncommitted and note it.)

---

### Task 8: Standalone example script

**Files:**
- Create: `docs/examples/texture_mesh.py`

- [ ] **Step 1: Write it**

```python
"""
Texture an existing mesh.ply from splats.zarr renders without re-running the mesh stage.

    /opt/venv/reconstruction/bin/python docs/examples/texture_mesh.py \
        --mesh /path/mesh.ply --splats /path/splats.zarr --voxel-size 0.2 \
        [--frames /path/frames.zarr --source frames] [--out /path/texture] \
        [--decimate-max-error 0.25] [--smooth-iterations 0] [--tex-size 8192] [--bake-target-faces 500000]

Writes <out>/<source>/{mesh.obj, mesh.mtl, albedo.png, normal_world.png, mesh_baked.ply, texture_report.json}.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from collab_splats.mesh.texture import texture_mesh
from collab_splats.mesh.utils import _texture_images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--splats", type=Path, required=True, help="splats.zarr (poses, K, rgb, normal)")
    p.add_argument("--frames", type=Path, default=None, help="frames.zarr, needed for --source frames")
    p.add_argument("--voxel-size", type=float, required=True, help="TSDF voxel size the mesh was fused at")
    p.add_argument("--source", choices=["splats", "frames"], default="splats")
    p.add_argument("--out", type=Path, default=None, help="default: <mesh dir>/texture")
    p.add_argument("--decimate-max-error", type=float, default=0.25, help="voxel units; <=0 disables")
    p.add_argument("--smooth-iterations", type=int, default=0)
    p.add_argument("--tex-size", type=int, default=8192)
    p.add_argument("--bake-target-faces", type=int, default=500_000)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Same cfg dict the pipeline passes; out dir mirrors backend_dir/texture/<source>
    cfg = dict(
        enabled=True,
        source=a.source,
        decimate_max_error=a.decimate_max_error if a.decimate_max_error > 0 else None,
        smooth_iterations=a.smooth_iterations,
        tex_size=a.tex_size,
        bake_target_faces=a.bake_target_faces,
    )
    out = (a.out or a.mesh.parent / "texture") / a.source
    images, normals, c2w, K = _texture_images(a.splats, a.frames, a.source)
    print(texture_mesh(a.mesh, out, images, normals, c2w, K, voxel_size=a.voxel_size, cfg=cfg))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke it on the test fixture data**

Run:
```bash
/opt/venv/reconstruction/bin/python docs/examples/texture_mesh.py --help | head -3
```
Expected: usage line printed, no import error.

- [ ] **Step 3: Commit**

```bash
black docs/examples/texture_mesh.py && isort docs/examples/texture_mesh.py
git add docs/examples/texture_mesh.py
git commit --only docs/examples/texture_mesh.py -m "docs(examples): standalone texture_mesh.py entry point"
```

---

### Task 9: Measurement on the scaffold-2dgs scene

**Files:**
- Modify: `docs/superpowers/specs/2026-08-29-mesh-texture-bake-design.md` (append `## Measurements`)

Prerequisite: locate the splats.zarr that produced `/workspace/outputs/scaffold2dgs_500f_50k_mesh/mesh_clean.ply` (`find /workspace/outputs -name splats.zarr -maxdepth 4`; check `attrs` / frame count = 500 and that `normal` exists). If it is not on disk, stop, report, and skip this task — do not fabricate numbers.

- [ ] **Step 1: Both sources at defaults, in tmux (memory: 500×H×W×3 uint8 ≈ 3 GB/array; cgroup cap 46.6 GB)**

```bash
tmux new -d -s tex "/opt/venv/reconstruction/bin/python docs/examples/texture_mesh.py --mesh /workspace/outputs/scaffold2dgs_500f_50k_mesh/mesh_clean.ply --splats <SPLATS_ZARR> --voxel-size 0.2 --source splats 2>&1 | tee /workspace/outputs/scaffold2dgs_500f_50k_mesh/texture_splats.log; /usr/bin/time -v /opt/venv/reconstruction/bin/python docs/examples/texture_mesh.py --mesh /workspace/outputs/scaffold2dgs_500f_50k_mesh/mesh_clean.ply --splats <SPLATS_ZARR> --frames <FRAMES_ZARR> --voxel-size 0.2 --source frames 2>&1 | tee /workspace/outputs/scaffold2dgs_500f_50k_mesh/texture_frames.log"
```
Record from each `texture_report.json`: faces input/decimated/baked, deviation p50/p90/p99/max, frac_over_voxel, texel fill, wall-time per step; peak RSS from `/usr/bin/time -v` ("Maximum resident set size").

- [ ] **Step 2: Render comparisons (5 scene cameras: wall / pole / text crops)**

Write a short scratchpad script (one `OffscreenRenderer` per process — Open3D rule) rendering `mesh_clean.ply` (vertex colour), `texture/splats/mesh.obj`, `texture/frames/mesh.obj` and both `mesh_baked.ply` from 5 `c2w` rows of splats.zarr, saved as PNGs under `/workspace/outputs/scaffold2dgs_500f_50k_mesh/texture/compare/`. Look at them (Read tool on the PNGs) and note wall/pole/text legibility.

- [ ] **Step 3: Sweeps**

`--decimate-max-error 0.25 / 0.5 / 1.0` (pole crops, faces, p99) and `--smooth-iterations 0 / 5 / 10` at 0.25 (pole crops), `--tex-size 8192` fixed, output to `--out .../texture_sweep_<val>`.

- [ ] **Step 4: Append `## Measurements` to the spec with the numbers, image paths, and a one-paragraph verdict (which source, which bound, whether smoothing stays 0). Commit.**

```bash
git add -f docs/superpowers/specs/2026-08-29-mesh-texture-bake-design.md
git commit --only docs/superpowers/specs/2026-08-29-mesh-texture-bake-design.md -m "docs(specs): mesh texture bake measurements on scaffold2dgs_500f_50k"
```

---

### Task 10: Docs, changelog, graph

**Files:**
- Modify: `docs/superpowers/CHANGELOG.md`, `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md in-flight entry** — add under `## In-Flight Work` (CLAUDE.md has foreign uncommitted edits; add only this line):
```markdown
- **mesh-texture** — decimate + UV/albedo/normal texture + bake-back as final mesh step ([spec](docs/superpowers/specs/2026-08-29-mesh-texture-bake-design.md) · [plan](docs/superpowers/plans/2026-08-29-mesh-texture-bake.md))
```
When Task 9 is done and verdict written, move the line to `docs/superpowers/CHANGELOG.md` as a dated entry (2026-08-29) summarising: config, outputs, measured numbers, meshoptimizer pin, pymeshlab exclusion.

- [ ] **Step 2: Update graph**

Run: `graphify update .`

- [ ] **Step 3: Full test run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh tests/wrapper -q 2>&1 | tail -3`
Expected: PASS except entries already in `docs/known-test-failures.md`.

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/CHANGELOG.md
git commit --only docs/superpowers/CHANGELOG.md CLAUDE.md -m "docs: record mesh texture pass"
```
(CLAUDE.md: same foreign-hunk caution as Task 7 — `git add -p CLAUDE.md`, stage only the in-flight line.)
