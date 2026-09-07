# Mesh Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `collab_splats/mesh/` (base / poisson / tsdf / utils plus the `feat/mesh-texture-bake` branch) with five array-in modules — `io`, `tsdf`, `clean`, `texture`, `features` — rewire every caller, drop meshlib, and cut the base.yaml mesh block to six keys.

**Architecture:** `mesh/` takes plain arrays (`depths (N,H,W) float32`, `rgbs (N,H,W,3) uint8`, `c2w (N,4,4)`, `K (N,3,3)`) and mesh paths; callers (Reconstructor, dashboard, evals, notebooks) compose the arrays at the call site. The stage is `fuse_tsdf` → `clean_repair_mesh` (Open3D tensor `fill_holes`) → optional `texture_mesh` (meshoptimizer decimation, Open3D UV atlas, NVIDIA Warp projection with BVH occlusion, plyfile textured PLY). No classes, no registry, no module constants — every knob is a keyword argument with a default.

**Tech Stack:** Open3D 0.19 (legacy + tensor API), NVIDIA Warp 1.14.0, meshoptimizer 0.2.30a0, plyfile, scipy, torch, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-mesh-cleanup-design.md` (commit `8a20c4ac`).

---

## Execution precondition — read before Task 1

**Do not start Task 1 until the user says the branch rebase has landed.** The user's constraint, verbatim: "we will wait to execute until we have clean/mesh ready and based against the same starting point". A concurrent agent is reorganising `refactor/cu121-uv-migration`; every `clean/*` branch is rebased onto that result in the order **preproc → splats → mesh**. This plan is written against the tree `clean/mesh` will have AFTER that rebase, which means it assumes:

- `collab_splats/preproc/frames.py` exists (from `clean/preproc`) with `read_frames(images_dir, idxs) -> (N,H,W,3) uint8`, `write_frames(images_dir, frames, records, provenance)`, `frame_idx_from_path(path) -> int`, `frame_paths(images_dir)`; keyframes live in `<scene>/images/` + `frames.json`; `Reconstructor.images_dir` is `Path(config["output_path"]) / "images"`.
- `collab_splats/splats/rendering.py` exists (from `clean/splats`) with `load_checkpoint(path: Path, device: str) -> (model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width))` and `render_views(model, camera_opt, cam_to_world, intrinsics, height, width)` — a generator yielding one dict per camera with torch tensors `rgb (1,H,W,3)` in [0,1], `depth`, `alpha`, `normal`, and `median_depth` for 2dgs only. **Two assumptions to confirm with the `clean/splats` owner before Task 4:** `image_ids` are SOURCE frame indices (the `frame_idx` in `frames.json`), and the returned `cam_to_world` are the pose-opt-adjusted poses the splats were trained against.
- `clean/splats` has dropped its plan Task 17 (`mesh/utils.py` edits) and the mesh-notebook half of its Task 18 — this plan owns those files.

If the rebased tree differs from this in any of those names, fix the plan first (one commit), then execute.

### Verified against the real tree, 2026-09-06 — execution started BEFORE the rebase

The user elected to execute in parallel without waiting for the rebase, so the three assumptions
above were checked against what is actually on disk and on the sibling branches. Read the sibling
branches directly (`git show clean/preproc:<path>`) rather than assuming — the branches exist even
though their content is not in this fork's base.

| Assumption | Reality on `clean/mesh` (base `6f060dfa`) | Consequence |
|---|---|---|
| `preproc/frames.py` with `read_frames` / `write_frames` / `frame_idx_from_path` / `frame_paths` | **Absent here.** Present on `clean/preproc` (tip `d46b6fdd`) with exactly those names plus `read_manifest`. `collab_splats/preproc/__init__.py` here exports `FrameStore`, qa, sampling, undistort, video — no `frames`. | Any `from collab_splats.preproc.frames import ...` raises `ImportError`. Blocks the parts of Tasks 4, 7, 8 that read frames, and the Task 15 gates as written. |
| `splats/rendering.py` with `load_checkpoint` / `render_views` | **Wrong on both branches.** `rendering.py` exists here and on `clean/splats`, but exports `gaussian_normals_in_camera_frame`, `activate_vanilla`, `render_gaussians`, `render_view`. There is no `load_checkpoint` anywhere under `collab_splats/splats/` on either branch. The nearest is `outputs.py::render_all_views(cfg, gaussians, refine, images, cam_to_world, intrinsics, store, anchor_field=None)`, which streams into a zarr group. | `render_tsdf_inputs(ckpt_path, ...)` cannot be built as specified. Production's splats→mesh path reads the resulting `splats.zarr`, not a `ckpt.pt`, so re-rendering from a checkpoint is also a **new capability**, not a port. Deferred. |
| `Reconstructor.images_dir` | **Absent.** No `images_dir` attribute on `reconstructor.py` here. | Task 7 must define it or read the frame store. |

Consequently the execution order below is **not** the plan's task order.

**Builds against the shared base today:** Tasks 1, 2, 3, the `upsample_depths` and
`write_textured_ply` halves of Task 4, Tasks 5, 6, 9, 14, and the
`tests/integration/test_pipeline_cu121.py` half of Task 10.

**Waits on `clean/preproc` merging in:** `render_tsdf_inputs`, Task 7, Task 8, Task 11, the
`evals/scripts/analyze_splats.py` half of Task 10, the frame-reading half of the Task 15 gates,
and Task 15's gate 3 (which fuses `render_tsdf_inputs` output).

**Task 12 is blocked transitively, not by a missing import of its own.** It deletes
`mesh/utils.py`, `mesh/base.py` and `mesh/poisson.py`, but `wrapper/reconstructor.py:618,623`
still imports `pointcloud_to_mesh` and friends from `mesh.utils` — and porting that file is
Task 7, which waits on `clean/preproc`. `evals/scripts/analyze_splats.py:28` and
`tests/wrapper/test_splats_stage.py:187,288` are blocked the same way. So the deletion order is
**7, 8, and the deferred half of 10 first, then 12** — deleting earlier breaks the tree at import.
Task 13 (documentation) is deferred with them: it documents `read_frames`, `images_dir` and
`render_tsdf_inputs`, none of which exist yet, and documenting an API that is not there is worse
than documenting nothing.

The gates do **not** hold Task 12 back. Task 15's old-code halves all run inside the detached
baseline worktree pinned at `2dd22904`, and the fusion/lift/cleaning baselines are already
captured under `/tmp/claude-0/-workspace-collab-splats/mesh-gates/` — so the comparison survives
the deletion either way.

Do not paper over a missing dependency by inventing a substitute API — defer the piece and say so
in the commit body.

## Conventions used in every task

- **Worktree and interpreter.** Every command runs from `/workspace/collab-splats/.worktrees/clean-mesh` with `/opt/venv/reconstruction/bin/python`. The venv's editable finder hard-codes `/workspace/collab-splats`, so a bare `pytest` in the worktree tests the MAIN tree and reports a false green. The only safe form is:

  ```bash
  cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -m pytest <paths> -v
  ```

  The first line must print `/workspace/collab-splats/.worktrees/clean-mesh/collab_splats/__init__.py`. If it prints anything else, stop — the run is not testing this tree. Below, `PYTEST` abbreviates that whole line; substitute the paths.
- **Commits.** The git index is shared with other sessions' worktrees: always `git add <exact paths>` then `git commit --only <exact paths> -m ...`. Never bare `git stash`. `docs/superpowers/` is gitignored — `git add -f` for files under it. Every commit message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **Docstrings** (user rule, applies to every new or touched function): a one-line summary, then `Args:` listing each input and what it is (shape, dtype, units), then `Returns:` — never a paragraph.

  ```python
  """
  What it does, one line.

  Args:
      name: what it is (shape, dtype, units).
  Returns:
      what it is (shape, dtype).
  """
  ```
- **Style** (CLAUDE.md): imports at top (exception: a heavy optional dep that must not load with the module, imported inside the function with a comment saying why), `########` section dividers, block comments, `logging` not `print`, no module constants.
- **Green between tasks.** The old modules (`mesh/base.py`, `mesh/poisson.py`, `mesh/utils.py`, the `Open3DTSDFFusion` class) stay importable until Task 12; new modules are added beside them, and old tests are deleted only once their replacement exists.
- **Installs need user approval** (shared venv). Only Task 5 (meshoptimizer) and Task 14 (`uv lock`) install anything.

## File map

Create:
- `collab_splats/mesh/clean.py` — `get_scene_scale`, `remove_floaters`, `fill_holes`, `clean_repair_mesh`
- `collab_splats/mesh/features.py` — `features2vertex`, `mesh_clustering`
- `collab_splats/mesh/io.py` — `upsample_depths`, `render_tsdf_inputs`, `write_textured_ply`
- `collab_splats/mesh/texture.py` — `decimate_mesh`, `unwrap_mesh_uvs`, `project_images_to_texture`, `texture_mesh` (+ private `_make_manifold`)
- `tests/mesh/test_clean.py`, `tests/mesh/test_features.py`, `tests/mesh/test_io.py`, `tests/mesh/test_texture.py`
- `docs/mesh.md`

Modify:
- `collab_splats/mesh/tsdf.py` — add `fuse_tsdf` (Task 3), delete the class (Task 12)
- `collab_splats/mesh/__init__.py` — final re-exports (Task 12)
- `collab_splats/geometry/metrics.py`, `tests/geometry/test_metrics.py` (Task 6)
- `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml`, `tests/wrapper/_stubs.py`, `tests/wrapper/test_reconstructor.py` (Task 7)
- `tests/wrapper/test_splats_stage.py`, `tests/wrapper/test_reconstructor_preprocess.py`, `tests/mesh/test_absent_confidence.py` → `tests/wrapper/test_absent_confidence.py` (Task 8)
- `collab_splats/dashboard/{pipeline,config,app,viewer}.py`, `tests/dashboard/{test_config,test_pipeline}.py` (Task 9)
- `evals/scripts/analyze_splats.py`, `tests/integration/test_pipeline_cu121.py` (Task 10)
- `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`, `docs/source/tutorials/06_mesh/splats_mesh.ipynb` (Task 11)
- `tests/test_cu121_migration.py` (Task 12)
- `docs/source/api/mesh.rst`, `docs/source/conf.py`, `configs/README.md`, `docs/README.md`, `CLAUDE.md`, `docs/superpowers/CHANGELOG.md`, `docs/known-test-failures.md` (Task 13)
- `pyproject.toml`, `uv.lock` (Task 14)

Delete (Task 12): `collab_splats/mesh/base.py`, `collab_splats/mesh/poisson.py`, `collab_splats/mesh/utils.py`, `tests/mesh/test_utils.py`, `tests/mesh/test_utils_ground_plane.py`, `tests/mesh/test_splats_adapter.py`, `tests/mesh/test_registry.py`, `tests/mesh/test_feature_transfer.py`, `tests/mesh/test_adapter.py`.

---

### Task 1: `clean.py` — scene scale, floater removal, hole filling

**Files:**
- Create: `collab_splats/mesh/clean.py`
- Create: `tests/mesh/test_clean.py`

Reference for the port: `collab_splats/mesh/utils.py` — `_scene_scale` (p1/p99 AABB diagonal), `_select_components` (L242-286) and its caller in `clean_repair_mesh` (L335-338). The meshlib fill section (L340-412: subdivide, smooth, colour reattach) is not ported — Open3D's tensor `fill_holes` replaces all of it.

Calibration facts that fix the test values (probed 2026-09-06 on the fixture below, radius 1, resolution 20, last 12 triangles removed → one hole with 14 boundary edges): `o3d.t.geometry.TriangleMesh.fill_holes(hole_size=...)` fills that hole at an absolute `hole_size ≳ 0.17` and leaves it below. **Convert to a fraction against the whole fixture, not the sphere.** The sphere alone scores `get_scene_scale` = 3.393, but the fixture also carries the far stray at `x = 9r`, which sets p99 and lifts the real scene scale to **10.379** — so the fill boundary sits between `frac` **0.016 (leaves)** and **0.017 (fills)**, not between 0.05 and 0.06. Test fractions: `0.2` fills, `0.005` leaves (3× margin under the boundary; `0.02` is above it and *fills*). Filling gives 0 boundary edges, 1508 → 1520 triangles, vertex count unchanged, vertex colours preserved exactly. After filling, `is_edge_manifold(allow_boundary_edges=False)` is True but `is_watertight()` is False because the fan triangulation self-intersects — so the tests assert boundary edges, never `is_watertight()`.

- [ ] **Step 1: Write the failing tests**

`tests/mesh/test_clean.py`:

```python
import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.clean import clean_repair_mesh, fill_holes, get_scene_scale, remove_floaters


def _holed_sphere_with_strays(path, radius=1.0, resolution=20, color=None):
    """Sphere with one hole (last 12 triangles removed) plus a near and a far stray sphere."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    tris = np.asarray(sphere.triangles)
    sphere.triangles = o3d.utility.Vector3iVector(tris[:-12])  # punch a hole
    sphere.remove_unreferenced_vertices()
    near = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 0.1, resolution=6)
    near.translate((radius * 1.15, 0.0, 0.0))
    far = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 0.1, resolution=6)
    far.translate((radius * 9, 0.0, 0.0))
    combined = sphere + near + far
    if color is not None:
        combined.paint_uniform_color(color)
    o3d.io.write_triangle_mesh(str(path), combined)
    return path


def _n_components(mesh):
    _, sizes, _ = mesh.cluster_connected_triangles()
    return len(sizes)


def _n_boundary_edges(mesh):
    return len(mesh.get_non_manifold_edges(allow_boundary_edges=False))


def test_get_scene_scale_ignores_outliers():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.random((1000, 3)), [[100.0, 100.0, 100.0]]])
    scale = get_scene_scale(pts)
    assert 1.6 < scale < 1.8  # ~sqrt(3) for the unit cube; the outlier would make it ~170


def test_remove_floaters_drops_far_component_keeps_near(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    assert _n_components(mesh) == 3
    out = remove_floaters(mesh, max_gap_frac=0.1)
    assert out is mesh and _n_components(mesh) == 2


def test_remove_floaters_area_floor_drops_small_components(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    remove_floaters(mesh, min_area_frac=0.1, max_gap_frac=0.1)
    assert _n_components(mesh) == 1


def test_remove_floaters_empty_mesh_is_a_no_op():
    mesh = o3d.geometry.TriangleMesh()
    assert remove_floaters(mesh) is mesh


def test_fill_holes_closes_small_hole(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    assert _n_boundary_edges(mesh) == 14
    filled = fill_holes(mesh, max_hole_frac=0.2)
    assert _n_boundary_edges(filled) == 0
    assert filled.is_edge_manifold(allow_boundary_edges=False)
    assert len(filled.triangles) > len(mesh.triangles)
    # Guards the Open3D tensor round-trip: a freed vertex buffer blows the extent up
    orig_extent = mesh.get_axis_aligned_bounding_box().get_extent()
    assert np.allclose(filled.get_axis_aligned_bounding_box().get_extent(), orig_extent, atol=1e-5)


def test_fill_holes_leaves_large_holes_alone(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    # 0.005 × the fixture's real scene scale (10.379, set by the far stray) is well under the
    # 0.017 fill boundary. 0.02 would FILL — the sphere-only scale of 3.393 is the wrong divisor.
    filled = fill_holes(mesh, max_hole_frac=0.005)
    assert _n_boundary_edges(filled) == 14


def test_clean_repair_mesh_writes_in_place(tmp_path):
    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    out = clean_repair_mesh(mesh_path, max_gap_frac=0.1, max_hole_frac=0.2)
    assert out == mesh_path
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert _n_components(after) == 2
    assert _n_boundary_edges(after) == 0


@pytest.mark.parametrize("radius", [1.0, 10.0])
def test_clean_repair_thresholds_follow_mesh_scale(tmp_path, radius):
    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply", radius=radius)
    clean_repair_mesh(mesh_path, max_gap_frac=0.1, max_hole_frac=0.2)
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert (_n_components(after), _n_boundary_edges(after)) == (2, 0)


def test_clean_repair_preserves_vertex_colors(tmp_path):
    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply", color=(0.2, 0.6, 0.9))
    clean_repair_mesh(mesh_path, max_gap_frac=0.1, max_hole_frac=0.2)
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert after.has_vertex_colors()
    colors = np.asarray(after.vertex_colors)
    assert np.allclose(colors, [0.2, 0.6, 0.9], atol=0.02)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tests/mesh/test_clean.py`
Expected: `ImportError` / `ModuleNotFoundError: No module named 'collab_splats.mesh.clean'` at collection.

- [ ] **Step 3: Write `collab_splats/mesh/clean.py`**

```python
"""
Mesh cleaning (floater removal) and repair (hole filling), thresholds relative to scene scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


######## Scale


def get_scene_scale(vertices):
    """
    Robust scene extent: diagonal of the 1st-99th percentile bounding box.

    Args:
        vertices: (V, 3) float positions.
    Returns:
        float, world units.
    """
    lo, hi = np.percentile(np.asarray(vertices), [1, 99], axis=0)
    return float(np.linalg.norm(hi - lo))


######## Cleaning


def remove_floaters(mesh, min_area_frac=6e-6, max_gap_frac=0.01, gap_kdtree_points=200_000):
    """
    Drop connected components that are tiny or far from the largest one; edits in place.

    Args:
        mesh: open3d.geometry.TriangleMesh, edited in place.
        min_area_frac: keep components whose area is at least this × scene_scale².
        max_gap_frac: keep components whose centroid lies within this × scene_scale of the main body.
        gap_kdtree_points: cap on main-body vertices indexed for the gap query (stride-subsampled).
    Returns:
        the same mesh.
    """
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    if len(cluster_sizes) == 0:
        return mesh
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n_comp = len(cluster_sizes)

    # Scale comes from the largest component alone so strays cannot inflate it
    largest = int(cluster_sizes.argmax())
    main_xyz = verts[np.unique(tris[cluster_ids == largest])]
    scale = get_scene_scale(main_xyz)

    # Per-component surface area and centroid
    tri_pts = verts[tris]
    tri_area = 0.5 * np.linalg.norm(np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1)
    comp_area = np.zeros(n_comp)
    np.add.at(comp_area, cluster_ids, tri_area)
    comp_centroid_sum = np.zeros((n_comp, 3))
    np.add.at(comp_centroid_sum, cluster_ids, tri_pts.mean(axis=1))
    comp_centroid = comp_centroid_sum / cluster_sizes[:, None]

    # Gap = distance from each centroid to the (subsampled) main body
    stride = max(1, len(main_xyz) // gap_kdtree_points)
    comp_gap, _ = cKDTree(main_xyz[::stride]).query(comp_centroid, k=1)

    # Keep large-enough, close-enough components; the main body always stays
    keep = (comp_area >= min_area_frac * scale**2) & (comp_gap <= max_gap_frac * scale)
    keep[largest] = True
    mesh.remove_triangles_by_mask(~keep[cluster_ids])
    mesh.remove_unreferenced_vertices()
    logger.info(
        "remove_floaters: kept %d of %d components (removed %d) at scene_scale=%.3f",
        int(keep.sum()), n_comp, int((~keep).sum()), scale,
    )
    return mesh


######## Repair


def fill_holes(mesh, max_hole_frac=0.0045):
    """
    Fill boundary loops up to a size proportional to the scene; returns a new mesh.

    Args:
        mesh: open3d.geometry.TriangleMesh; vertex colours survive.
        max_hole_frac: fills holes whose Open3D `hole_size` (diameter-like: a hole is filled iff
            hole_size is at least ~2 × its radius) is at most this × scene_scale.
    Returns:
        open3d.geometry.TriangleMesh with the small holes triangulated.
    """
    scale = get_scene_scale(np.asarray(mesh.vertices))
    hole_size = max_hole_frac * scale
    # Open3D 0.19's fill_holes returns a mesh whose vertex tensors are VIEWS into the
    # from_legacy source, so that source must stay bound until the result is read. Chaining
    # from_legacy(...).fill_holes(...) frees it early and yields garbage (measured: positions
    # off by 1.0, colours 3.7e19) with no error.
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tmesh.fill_holes(hole_size=hole_size).to_legacy()
    logger.info(
        "fill_holes: hole_size=%.4f (%.4f × scene_scale %.3f), triangles %d -> %d",
        hole_size, max_hole_frac, scale, len(mesh.triangles), len(filled.triangles),
    )
    return filled


def clean_repair_mesh(mesh_path, min_area_frac=6e-6, max_gap_frac=0.01, max_hole_frac=0.0045):
    """
    Remove floaters, fill small holes, and overwrite the mesh file.

    Args:
        mesh_path: PLY to clean; rewritten in place.
        min_area_frac: see remove_floaters.
        max_gap_frac: see remove_floaters.
        max_hole_frac: see fill_holes.
    Returns:
        Path to mesh_path.
    """
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    remove_floaters(mesh, min_area_frac=min_area_frac, max_gap_frac=max_gap_frac)
    mesh = fill_holes(mesh, max_hole_frac=max_hole_frac)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh_path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST tests/mesh/test_clean.py`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/mesh/clean.py tests/mesh/test_clean.py && git commit --only collab_splats/mesh/clean.py tests/mesh/test_clean.py -m "feat(mesh): clean.py — floater removal + Open3D fill_holes, thresholds relative to scene scale

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `features.py` — vertex features and clustering

**Files:**
- Create: `collab_splats/mesh/features.py`
- Create: `tests/mesh/test_features.py`

Reference: `collab_splats/mesh/utils.py` `features2vertex` (L109-171, body ported unchanged) and `mesh_clustering` (L523-572, replaced: the dense `(n_valid, n_valid)` bool adjacency, the Python loop and the tqdm bar go; `cKDTree.query_pairs` builds the sparse adjacency). Behaviour change to note in the commit: the old cluster floor was the inline literal `len(c) > 10`; the new `min_cluster_size=10` keeps clusters with **at least** 10 members.

- [ ] **Step 1: Write the failing tests**

`tests/mesh/test_features.py`:

```python
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from collab_splats.mesh.features import features2vertex, mesh_clustering


######## features2vertex


def _features2vertex_numpy_reference(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """CPU reference: same kernel as the torch path, written with np.add.at."""
    vertices = np.asarray(mesh_vertices)
    M, D = len(vertices), features.shape[1]
    distances, indices = cKDTree(vertices).query(points, k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)
    distances, indices, feats = distances[valid_mask], indices[valid_mask], features[valid_mask]
    sigma = np.mean(distances)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((M, D), dtype=np.float64)
    wsum = np.zeros((M, 1), dtype=np.float64)
    for j in range(k):
        np.add.at(out, indices[:, j], feats * weights[:, j : j + 1])
        np.add.at(wsum, indices[:, j], weights[:, j : j + 1])
    nz = wsum[:, 0] > 0
    out[nz] /= wsum[nz]
    return out.astype(features.dtype)


def test_features2vertex_output_shape():
    rng = np.random.default_rng(0)
    verts = rng.random((50, 3))
    pts = rng.random((200, 3))
    feats = rng.random((200, 16)).astype(np.float32)
    assert features2vertex(verts, pts, feats, k=5).shape == (50, 16)


def test_features2vertex_matches_numpy_reference():
    rng = np.random.default_rng(0)
    verts = rng.random((200, 3))
    pts = rng.random((1000, 3))
    feats = rng.random((1000, 8)).astype(np.float32)
    out = features2vertex(verts, pts, feats, k=5, sdf_trunc=0.1)
    ref = _features2vertex_numpy_reference(verts, pts, feats, k=5, sdf_trunc=0.1)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_features2vertex_all_far_returns_zeros():
    verts = np.zeros((10, 3))
    pts = np.full((20, 3), 100.0)
    feats = np.ones((20, 4), dtype=np.float32)
    out = features2vertex(verts, pts, feats, k=3)
    assert out.shape == (10, 4) and not out.any()


def test_features2vertex_dtype_preserved():
    rng = np.random.default_rng(1)
    verts = rng.random((50, 3)).astype(np.float32)
    pts = rng.random((100, 3)).astype(np.float32)
    feats = rng.random((100, 6)).astype(np.float32)
    out = features2vertex(verts, pts, feats, k=4, sdf_trunc=0.2)
    assert out.dtype == np.float32 and out.shape == (50, 6)


######## mesh_clustering


def _two_blobs_mesh():
    """60 vertices in two tight blobs 1 unit apart plus 10 scattered low-similarity vertices."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.005, (30, 3))
    b = rng.normal(0.0, 0.005, (30, 3)) + [1.0, 0.0, 0.0]
    stray = rng.random((10, 3)) * [0.5, 1.0, 1.0] + [0.25, 0.0, 0.0]
    verts = np.vstack([a, b, stray])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(np.zeros((0, 3), np.int32)))
    similarity = np.r_[np.ones(60), np.zeros(10)]
    return mesh, similarity


def test_mesh_clustering_groups_nearby_high_similarity_vertices():
    mesh, similarity = _two_blobs_mesh()
    clusters = mesh_clustering(mesh, similarity, similarity_threshold=0.5, spatial_radius=0.05, min_cluster_size=10)
    assert len(clusters) == 2
    assert {frozenset(c.tolist()) for c in clusters} == {frozenset(range(30)), frozenset(range(30, 60))}


def test_mesh_clustering_min_cluster_size_drops_small_clusters():
    mesh, similarity = _two_blobs_mesh()
    assert mesh_clustering(mesh, similarity, similarity_threshold=0.5, spatial_radius=0.05, min_cluster_size=31) == []


def test_mesh_clustering_no_valid_vertices_returns_empty_list():
    mesh, similarity = _two_blobs_mesh()
    assert mesh_clustering(mesh, np.zeros_like(similarity)) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tests/mesh/test_features.py`
Expected: `ModuleNotFoundError: No module named 'collab_splats.mesh.features'`.

- [ ] **Step 3: Write `collab_splats/mesh/features.py`**

```python
"""
Per-vertex features from point features, and spatial clustering of high-similarity vertices.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


######## Feature transfer


def features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """
    Gaussian-weighted k-NN scatter of point features onto mesh vertices.

    Args:
        mesh_vertices: (M, 3) float vertex positions.
        points: (P, 3) float point positions in the same frame.
        features: (P, D) per-point features.
        k: nearest vertices each point contributes to.
        sdf_trunc: points whose nearest vertex is farther than this (world units) are ignored.
    Returns:
        (M, D) array in features.dtype; vertices no point reached are zero.
    """
    vertices = np.asarray(mesh_vertices)
    M = len(vertices)
    D = features.shape[1]

    # Nearest-vertex query for every point; workers=-1 uses all cores.
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k, workers=-1)
    # k=1 collapses the neighbour axis; restore it so the kernel below is uniform.
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Drop points whose closest vertex is beyond the truncation band.
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    feats = features[valid_mask]

    # Move the aggregation to the GPU (float32); one .cpu() at the end.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.as_tensor(np.ascontiguousarray(distances), dtype=torch.float32, device=device)
    idx = torch.as_tensor(np.ascontiguousarray(indices), dtype=torch.long, device=device)
    f = torch.as_tensor(np.ascontiguousarray(feats), dtype=torch.float32, device=device)

    # Gaussian kernel over neighbour distances; normalize weights per point (over k).
    sigma = d.mean()
    w = torch.exp(-(d**2) / (2 * sigma**2))
    w = w / w.sum(dim=1, keepdim=True)

    # Scatter weighted features to vertices; accumulate weights for normalization.
    acc = torch.zeros((M, D), dtype=torch.float32, device=device)
    wsum = torch.zeros((M, 1), dtype=torch.float32, device=device)
    for j in range(k):
        acc.index_add_(0, idx[:, j], f * w[:, j : j + 1])
        wsum.index_add_(0, idx[:, j], w[:, j : j + 1])

    # Normalize aggregated features by summed weights (untouched vertices stay zero).
    nz = wsum.squeeze(1) > 0
    acc[nz] /= wsum[nz]
    return acc.cpu().numpy().astype(features.dtype)


######## Clustering


def mesh_clustering(mesh, similarity_values, similarity_threshold=0.8, spatial_radius=0.03, min_cluster_size=10):
    """
    Group spatially connected vertices whose similarity exceeds a threshold.

    Args:
        mesh: open3d.geometry.TriangleMesh; only its vertices are read.
        similarity_values: (V,) float per-vertex similarity.
        similarity_threshold: vertices with similarity above this are candidates.
        spatial_radius: candidates within this distance (world units) are connected.
        min_cluster_size: clusters with fewer vertices are dropped.
    Returns:
        list of (n_i,) int arrays of vertex indices into the mesh, one per cluster.
    """
    similarity_values = np.asarray(similarity_values)
    valid = np.flatnonzero(similarity_values > similarity_threshold)
    if len(valid) == 0:
        return []

    # Sparse adjacency from all candidate pairs within the radius
    xyz = np.asarray(mesh.vertices)[valid]
    pairs = cKDTree(xyz).query_pairs(spatial_radius, output_type="ndarray")
    n = len(valid)
    adjacency = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)

    # Map component labels back to original vertex indices; drop small clusters
    clusters = []
    for label in np.unique(labels):
        members = valid[labels == label]
        if len(members) >= min_cluster_size:
            clusters.append(members)
    return clusters
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST tests/mesh/test_features.py`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/mesh/features.py tests/mesh/test_features.py && git commit --only collab_splats/mesh/features.py tests/mesh/test_features.py -m "feat(mesh): features.py — features2vertex + sparse-adjacency mesh_clustering (min_cluster_size kwarg replaces the > 10 literal)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `fuse_tsdf` — array-in TSDF fusion beside the old class

**Files:**
- Modify: `collab_splats/mesh/tsdf.py` (insert the function after `logger = ...` at L15, ABOVE `@dataclass class Open3DTSDFFusion` at L18; the class and its imports stay until Task 12)
- Rewrite: `tests/mesh/test_tsdf.py`

`fuse_tsdf` is the one fusion path for every caller. Rules from the spec: `rgbs` must be uint8 (no `/255` guessing), one view count everywhere, principal point inside the depth grid (catches the model-res-depth × original-res-K regression of 2026-08-11), `sdf_trunc = 4 × voxel_size` unless given, output always `out_dir/mesh.ply`. `ScalableTSDFVolume` is looked up as `o3d.pipelines.integration.ScalableTSDFVolume` at call time so a test can monkeypatch it.

- [ ] **Step 1: Write the failing tests**

Replace `tests/mesh/test_tsdf.py` wholesale:

```python
import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.tsdf import fuse_tsdf


def _views(n=3, h=32, w=32):
    """n cameras looking down +z at a plane 1 unit away, shifted 0.05 along x per view."""
    depths = np.ones((n, h, w), dtype=np.float32)
    rgbs = np.full((n, h, w, 3), 128, dtype=np.uint8)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, 0, 3] = 0.05 * np.arange(n)
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64), (n, 1, 1))
    return depths, rgbs, c2w, K


def test_fuse_tsdf_writes_mesh_ply(tmp_path):
    depths, rgbs, c2w, K = _views()
    out = fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "a" / "b", voxel_size=0.02, depth_trunc=2.0)
    assert out == tmp_path / "a" / "b" / "mesh.ply" and out.exists()
    mesh = o3d.io.read_triangle_mesh(str(out))
    assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()
    assert np.allclose(np.asarray(mesh.vertex_colors), 128 / 255, atol=0.05)


def test_fuse_tsdf_rejects_float_rgb(tmp_path):
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="uint8"):
        fuse_tsdf(depths, rgbs.astype(np.float32) / 255, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_rejects_principal_point_outside_grid(tmp_path):
    depths, rgbs, c2w, K = _views()
    K[:, 0, 2] = 64  # cx beyond the 32-wide grid: K is at a different resolution than depth
    with pytest.raises(ValueError, match="Principal point"):
        fuse_tsdf(depths, rgbs, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_rejects_frame_count_mismatch(tmp_path):
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="views"):
        fuse_tsdf(depths, rgbs[:2], c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_sdf_trunc_defaults_to_four_voxels(tmp_path, monkeypatch):
    seen = []

    class _Recorder:
        def __init__(self, voxel_length, sdf_trunc, color_type):
            seen.append((voxel_length, sdf_trunc, color_type))

        def integrate(self, *args, **kwargs):
            pass

        def extract_triangle_mesh(self):
            return o3d.geometry.TriangleMesh.create_sphere(0.1)

    monkeypatch.setattr(o3d.pipelines.integration, "ScalableTSDFVolume", _Recorder)
    depths, rgbs, c2w, K = _views(n=1)
    fuse_tsdf(depths, rgbs, c2w, K, tmp_path, voxel_size=0.01, depth_trunc=2.0)
    assert len(seen) == 1
    assert seen[0][0] == 0.01 and seen[0][1] == pytest.approx(0.04)
    assert seen[0][2] == o3d.pipelines.integration.TSDFVolumeColorType.RGB8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tests/mesh/test_tsdf.py`
Expected: `ImportError: cannot import name 'fuse_tsdf' from 'collab_splats.mesh.tsdf'`.

- [ ] **Step 3: Add `fuse_tsdf` to `collab_splats/mesh/tsdf.py`**

Insert after `logger = logging.getLogger(__name__)` (L15) and before `@dataclass` (L18). The existing imports (`numpy`, `open3d`, `tqdm`, `extract_intrinsics`, `invert_poses`, `Path`) already cover it; add nothing to the import block.

```python
######## Fusion


def fuse_tsdf(depths, rgbs, c2w, K, out_dir, voxel_size, depth_trunc, sdf_trunc=None):
    """
    Integrate depth + RGB views into a ScalableTSDFVolume and write the extracted mesh.

    Args:
        depths: (N, H, W) float depth in world units, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 RGB at the same resolution as depths.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at the depth resolution.
        out_dir: directory to create; the mesh is written to out_dir/mesh.ply.
        voxel_size: TSDF voxel edge, world units.
        depth_trunc: depth beyond this (world units) is ignored.
        sdf_trunc: truncation band, world units; None = 4 × voxel_size.
    Returns:
        Path to out_dir/mesh.ply.
    """
    depths = np.asarray(depths)
    rgbs = np.asarray(rgbs)
    c2w = np.asarray(c2w)
    K = np.asarray(K)

    # Input contract: uint8 colour, one view count, K at the depth resolution
    if rgbs.dtype != np.uint8:
        raise ValueError(f"rgbs must be uint8 in [0, 255], got {rgbs.dtype}")
    n, h, w = depths.shape
    if rgbs.shape != (n, h, w, 3) or c2w.shape != (n, 4, 4) or K.shape != (n, 3, 3):
        raise ValueError(f"views disagree: depths {depths.shape}, rgbs {rgbs.shape}, c2w {c2w.shape}, K {K.shape}")
    cx, cy = K[:, 0, 2], K[:, 1, 2]
    if cx.min() < 0 or cx.max() > w or cy.min() < 0 or cy.max() > h:
        raise ValueError(
            f"Principal point outside the {w}x{h} depth grid (cx range [{cx.min():.1f}, {cx.max():.1f}], "
            f"cy range [{cy.min():.1f}, {cy.max():.1f}]) — intrinsics and depth are at different resolutions."
        )
    if sdf_trunc is None:
        sdf_trunc = 4 * voxel_size

    # Integrate every view; Open3D wants world-to-camera extrinsics
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    w2c = invert_poses(c2w)
    for i in tqdm(range(n), desc="TSDF integration"):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgbs[i])),
            o3d.geometry.Image(np.ascontiguousarray(depths[i], dtype=np.float32)),
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        fx, fy, cx_i, cy_i = extract_intrinsics(K[i])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx_i, cy_i)
        volume.integrate(rgbd, intrinsic, np.asarray(w2c[i], dtype=np.float64))

    # Extract and write
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = volume.extract_triangle_mesh()
    mesh_path = out_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    logger.info("fuse_tsdf: %d views, voxel=%.4f sdf_trunc=%.4f -> %s (%d vertices)", n, voxel_size, sdf_trunc, mesh_path, len(mesh.vertices))
    return mesh_path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST tests/mesh/test_tsdf.py`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py && git commit --only collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py -m "feat(mesh): fuse_tsdf — array-in TSDF fusion with uint8/shape/principal-point guards

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `io.py` — depth upsampling, splat rendering, textured PLY

**Files:**
- Create: `collab_splats/mesh/io.py`
- Create: `tests/mesh/test_io.py`

`upsample_depths` ports `mesh/utils.py::guided_upsample_depth` (L413-476) behind a batch signature; `out_hw` comes from the RGB canvas instead of a parameter. `render_tsdf_inputs` is the `mesh.source: splats` input path and imports `collab_splats.splats.rendering` INSIDE the function (gsplat needs CUDA at import — the documented exception in CLAUDE.md); the test injects a fake module into `sys.modules` under that name. `write_textured_ply` splits vertices per face corner (a vertex has one `s t` in PLY) and stores the texture name as `comment TextureFile albedo.png`, which trimesh's loader honours; vertex colours are sampled from the atlas so viewers without texture support still show colour. Texel convention is Open3D's bake: texel `[r, c]` sits at `u=(c+0.5)/S, v=(S-r-0.5)/S`, so a `(u, v)` samples row `(1-v)·H`, col `u·W`.

- [ ] **Step 1: Write the failing tests**

`tests/mesh/test_io.py`:

```python
import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
import torch
import trimesh
from plyfile import PlyData

from collab_splats.mesh.io import render_tsdf_inputs, upsample_depths, write_textured_ply
from collab_splats.preproc.frames import write_frames


######## upsample_depths


def _step_scene(factor=4):
    """Model-res depth with a vertical step edge + RGB guide whose edge aligns with it."""
    h, w = 32, 32
    depth = np.full((h, w), 1.0, dtype=np.float32)
    depth[:, w // 2 :] = 2.0
    H, W = h * factor, w * factor
    rgb = np.full((H, W, 3), 40, dtype=np.uint8)
    rgb[:, W // 2 :] = 200
    return depth, rgb


def test_upsample_depths_places_crop():
    depth, rgb = _step_scene(factor=2)
    canvas_hw = (100, 120)
    full_rgb = np.zeros((*canvas_hw, 3), dtype=np.uint8)
    full_rgb[10:74, 20:84] = rgb
    out = upsample_depths(depth[None], full_rgb[None], [[20, 10, 84, 74]])
    assert out.shape == (1, *canvas_hw) and out.dtype == np.float32
    out = out[0]
    assert np.all(out[:10] == 0) and np.all(out[74:] == 0)
    assert np.all(out[:, :20] == 0) and np.all(out[:, 84:] == 0)
    assert (out[10:74, 20:84] > 0).mean() > 0.99


def test_upsample_depths_masked_pixels_stay_zero():
    depth, rgb = _step_scene(factor=4)
    depth[8:16, 8:16] = 0.0
    H, W = rgb.shape[:2]
    out = upsample_depths(depth[None], rgb[None], [[0, 0, W, H]])[0]
    assert np.all(out[32:64, 32:64] == 0)
    valid = out[out > 0]
    assert valid.min() >= 1.0 - 1e-3 and valid.max() <= 2.0 + 1e-3


def test_upsample_depths_step_edge_stays_sharp():
    depth, rgb = _step_scene(factor=4)
    H, W = rgb.shape[:2]
    out = upsample_depths(depth[None], rgb[None], [[0, 0, W, H]])[0]
    interior = out[:, np.r_[0 : W // 2 - 8, W // 2 + 8 : W]]
    fabricated = (interior > 1.1) & (interior < 1.9)
    assert fabricated.mean() < 0.01


def test_upsample_depths_rejects_count_mismatch():
    depth, rgb = _step_scene(factor=2)
    with pytest.raises(ValueError, match="crop boxes"):
        upsample_depths(depth[None], rgb[None], [[0, 0, 64, 64], [0, 0, 64, 64]])


def test_upsample_depths_rejects_box_outside_canvas():
    depth, rgb = _step_scene(factor=2)
    with pytest.raises(ValueError, match="outside"):
        upsample_depths(depth[None], rgb[None], [[0, 0, 65, 64]])


######## render_tsdf_inputs


def _fake_rendering(views, image_ids, hw):
    """Stand-in for collab_splats.splats.rendering: fixed poses, canned views."""
    n = len(views)

    def load_checkpoint(path, device):
        return "model", "camera_opt", torch.eye(4).repeat(n, 1, 1), torch.eye(3).repeat(n, 1, 1), list(image_ids), hw

    def render_views(model, camera_opt, cam_to_world, intrinsics, height, width):
        yield from views

    return SimpleNamespace(load_checkpoint=load_checkpoint, render_views=render_views)


def _view(h, w, depth, rgb=0.5, alpha=1.0, median_depth=None):
    view = {
        "rgb": torch.full((1, h, w, 3), rgb),
        "depth": torch.full((1, h, w, 1), depth),
        "alpha": torch.full((1, h, w, 1), alpha),
    }
    if median_depth is not None:
        view["median_depth"] = torch.full((1, h, w, 1), median_depth)
    return view


def test_render_tsdf_inputs_stacks_views_and_zeroes_empty_pixels(tmp_path, monkeypatch):
    h, w = 4, 6
    v0 = _view(h, w, depth=2.0)
    v0["alpha"][0, 0, 0, 0] = 0.0
    monkeypatch.setitem(sys.modules, "collab_splats.splats.rendering", _fake_rendering([v0, _view(h, w, depth=3.0)], [0, 1], (h, w)))
    depths, rgbs, c2w, K = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
    assert depths.shape == (2, h, w) and depths.dtype == np.float32
    assert depths[0, 0, 0] == 0.0 and depths[0, 1, 1] == 2.0 and np.all(depths[1] == 3.0)
    assert rgbs.shape == (2, h, w, 3) and rgbs.dtype == np.uint8 and np.all(rgbs == 127)
    assert c2w.shape == (2, 4, 4) and K.shape == (2, 3, 3)
    assert c2w.dtype == np.float32 and K.dtype == np.float32


def test_render_tsdf_inputs_prefers_median_depth(tmp_path, monkeypatch):
    h, w = 4, 6
    monkeypatch.setitem(sys.modules, "collab_splats.splats.rendering", _fake_rendering([_view(h, w, depth=9.0, median_depth=5.0)], [0], (h, w)))
    depths, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
    assert np.all(depths == 5.0)


def test_render_tsdf_inputs_swaps_in_source_frames_by_image_id(tmp_path, monkeypatch):
    h, w = 8, 8
    images_dir = tmp_path / "images"
    frames = [np.full((h, w, 3), idx * 10, dtype=np.uint8) for idx in (0, 5, 7)]
    write_frames(images_dir, frames, [{"frame_idx": idx} for idx in (0, 5, 7)], {})
    monkeypatch.setitem(sys.modules, "collab_splats.splats.rendering", _fake_rendering([_view(h, w, 1.0), _view(h, w, 1.0)], [7, 0], (h, w)))
    _, rgbs, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")
    assert np.all(rgbs[0] == 70) and np.all(rgbs[1] == 0)


def test_render_tsdf_inputs_rejects_frame_size_mismatch(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    write_frames(images_dir, [np.zeros((8, 8, 3), np.uint8)], [{"frame_idx": 0}], {})
    monkeypatch.setitem(sys.modules, "collab_splats.splats.rendering", _fake_rendering([_view(4, 6, 1.0)], [0], (4, 6)))
    with pytest.raises(ValueError, match="checkpoint renders"):
        render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")


######## write_textured_ply


def _unit_square_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    uv = verts[faces][..., :2]  # (F, 3, 2): u = x, v = y
    return mesh, uv


def test_write_textured_ply_writes_uv_and_texture_comment(tmp_path):
    mesh, uv = _unit_square_mesh()
    albedo = np.zeros((8, 8, 3), np.uint8)
    albedo[..., 0] = np.arange(8)[None, :] * 32  # red grows with column (u)
    albedo[..., 1] = np.arange(8)[:, None] * 32  # green grows with row
    out = write_textured_ply(mesh, uv, albedo, tmp_path)
    assert out == tmp_path / "mesh.ply" and (tmp_path / "albedo.png").exists()
    header = out.read_bytes()[:600].decode("ascii", "ignore")
    assert "property float s" in header and "property float t" in header
    assert "comment TextureFile albedo.png" in header

    # Six corners (two triangles), colour sampled from the atlas at each corner's uv
    vertex = PlyData.read(str(out))["vertex"]
    assert len(vertex) == 6
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)
    origin = np.flatnonzero((xyz == [0, 0, 0]).all(axis=1))[0]
    assert tuple(rgb[origin]) == tuple(albedo[7, 0]) == (0, 224, 0)  # v=0 → bottom row, u=0 → col 0
    right = np.flatnonzero((xyz == [1, 0, 0]).all(axis=1))[0]
    assert tuple(rgb[right]) == tuple(albedo[7, 7]) == (224, 224, 0)


def test_write_textured_ply_round_trips_through_trimesh_and_open3d(tmp_path):
    mesh, uv = _unit_square_mesh()
    albedo = np.full((8, 8, 3), 200, np.uint8)
    out = write_textured_ply(mesh, uv, albedo, tmp_path)
    tm = trimesh.load(str(out), process=False)
    assert tm.visual.uv.shape == (6, 2)
    assert tm.visual.material.image.size == (8, 8)
    o3 = o3d.io.read_triangle_mesh(str(out))
    assert o3.has_vertex_colors() and len(o3.triangles) == 2
    assert np.allclose(np.asarray(o3.vertex_colors), 200 / 255, atol=1 / 255)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tests/mesh/test_io.py`
Expected: `ModuleNotFoundError: No module named 'collab_splats.mesh.io'`.

- [ ] **Step 3: Write `collab_splats/mesh/io.py`**

```python
"""
Array inputs for TSDF fusion (depth upsampling, splat rendering) and textured-PLY output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from collab_splats.preproc.frames import read_frames

logger = logging.getLogger(__name__)


######## Depth upsampling


def _box(x, radius):
    """
    Normalized box filter, the O(1) primitive of the guided filter.

    Args:
        x: (H, W) float32 image.
        radius: half window; the kernel is 2 × radius + 1.
    Returns:
        (H, W) float32 local mean.
    """
    k = 2 * radius + 1
    return cv2.boxFilter(x, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide, src, radius, eps):
    """
    He et al. gray-guide guided filter: edge-preserving smoothing of src steered by guide.

    Args:
        guide: (H, W) float32 in [0, 1].
        src: (H, W) float32 signal to smooth.
        radius: box half window.
        eps: regulariser on the guide variance.
    Returns:
        (H, W) float32 filtered src.
    """
    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g
    cov_gs = _box(guide * src, radius) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


def _guided_upsample_depth(depth, rgb_full, crop_box, radius=None, eps=1e-3):
    """
    Upsample one model-res depth map into its crop region of the original-res RGB canvas.

    Args:
        depth: (h, w) float32 model-res depth, 0 = no observation.
        rgb_full: (H, W, 3) uint8 original-res frame, the guide.
        crop_box: (tl_x, tl_y, cr_x, cr_y) model crop in original pixels.
        radius: guided-filter half window; None = ~2 × the upsample factor.
        eps: guided-filter regulariser.
    Returns:
        (H, W) float32 depth; masked pixels stay 0, canvas outside the crop is 0.
    """
    H, W = rgb_full.shape[:2]
    tl_x, tl_y, cr_x, cr_y = (int(round(v)) for v in crop_box)
    cw, ch = cr_x - tl_x, cr_y - tl_y
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Degenerate crop box {crop_box} — original_coords are corrupt")
    if tl_x < 0 or tl_y < 0 or cr_x > W or cr_y > H:
        raise ValueError(f"Crop box {crop_box} lies outside the {H}x{W} canvas")

    # Nearest resize of depth and validity to crop size — blocky but never invents values
    depth_nn = cv2.resize(depth, (cw, ch), interpolation=cv2.INTER_NEAREST)
    valid_nn = (depth_nn > 0).astype(np.float32)

    # Gray guide in [0, 1] from the original-res crop; radius spans ~2x the upsample factor
    guide = cv2.cvtColor(rgb_full[tl_y:cr_y, tl_x:cr_x], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if radius is None:
        radius = max(1, int(np.ceil(2 * cw / depth.shape[1])))

    # Validity-weighted filtering: masked pixels contribute nothing to their neighbours
    num = _guided_filter(guide, depth_nn * valid_nn, radius, eps)
    den = _guided_filter(guide, valid_nn, radius, eps)
    filtered = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)

    # The guide must never resurrect deleted depth, and depth must stay non-negative
    filtered[valid_nn == 0] = 0.0
    np.maximum(filtered, 0.0, out=filtered)

    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[tl_y:cr_y, tl_x:cr_x] = filtered
    return canvas


def upsample_depths(depths, rgbs, crop_boxes):
    """
    Guided-filter upsample model-res depth maps onto their original-res RGB frames.

    Args:
        depths: (N, h, w) float model-res depth, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 original-res frames; H, W set the output size.
        crop_boxes: (N, 4) [tl_x, tl_y, cr_x, cr_y] model crops in original pixels (original_coords[:, :4]).
    Returns:
        (N, H, W) float32 depth at frame resolution.
    """
    depths = np.asarray(depths)
    rgbs = np.asarray(rgbs)
    crop_boxes = np.asarray(crop_boxes)
    if not (len(depths) == len(rgbs) == len(crop_boxes)):
        raise ValueError(f"{len(depths)} depths, {len(rgbs)} rgbs, {len(crop_boxes)} crop boxes")

    # One guided upsample per frame into a preallocated stack
    n, H, W = len(depths), rgbs.shape[1], rgbs.shape[2]
    out = np.zeros((n, H, W), dtype=np.float32)
    for i in range(n):
        out[i] = _guided_upsample_depth(np.asarray(depths[i], dtype=np.float32), rgbs[i], crop_boxes[i])
    return out


######## Splat rendering


def _to_numpy(t):
    """
    Torch tensor or array to a host numpy array.

    Args:
        t: torch.Tensor on any device, or array-like.
    Returns:
        numpy array.
    """
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def render_tsdf_inputs(ckpt_path, images_dir=None, device="cuda"):
    """
    Render depth (+ RGB) from a trained splat checkpoint at its training cameras.

    Args:
        ckpt_path: splats/ckpt.pt written by the splats stage.
        images_dir: keyframe directory (images/ + frames.json); when given, RGB comes from the
            source frames matched by image id instead of the render.
        device: torch device for rendering.
    Returns:
        depths (N, H, W) float32 (median depth for 2dgs, 0 where alpha is 0),
        rgbs (N, H, W, 3) uint8, c2w (N, 4, 4) float32, K (N, 3, 3) float32.
    """
    # Imported here: gsplat needs CUDA at import, and this module must load without it.
    from collab_splats.splats.rendering import load_checkpoint, render_views

    model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width) = load_checkpoint(Path(ckpt_path), device)

    # Source frames replace rendered RGB when a keyframe directory is given
    rgbs = None
    if images_dir is not None:
        rgbs = read_frames(images_dir, [int(i) for i in image_ids])
        if rgbs.shape[1:3] != (height, width):
            raise ValueError(f"{images_dir} frames are {rgbs.shape[1:3]} but the checkpoint renders {(height, width)}")

    # Render every camera; 2dgs exposes median_depth, which is the sharper surface estimate
    depths, rendered = [], []
    for view in render_views(model, camera_opt, cam_to_world, intrinsics, height, width):
        depth = _to_numpy(view["median_depth"] if "median_depth" in view else view["depth"]).reshape(height, width, -1)[..., 0]
        alpha = _to_numpy(view["alpha"]).reshape(height, width, -1)[..., 0]
        depths.append(np.where(alpha > 0, depth, 0.0).astype(np.float32))
        if rgbs is None:
            rgb = _to_numpy(view["rgb"]).reshape(height, width, -1)[..., :3]
            rendered.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    if rgbs is None:
        rgbs = np.stack(rendered)

    return np.stack(depths), rgbs, _to_numpy(cam_to_world).astype(np.float32), _to_numpy(intrinsics).astype(np.float32)


######## Textured PLY


def write_textured_ply(mesh, uv, albedo, out_dir):
    """
    Write a mesh with per-corner UVs and its albedo atlas as mesh.ply + albedo.png.

    Args:
        mesh: open3d.geometry.TriangleMesh (legacy); vertices are split per face corner on write.
        uv: (F, 3, 2) float texture coordinates per face corner, Open3D bake convention (v up).
        albedo: (S, S, 3) uint8 RGB atlas.
        out_dir: directory to create; receives mesh.ply and albedo.png.
    Returns:
        Path to out_dir/mesh.ply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One vertex per face corner so each carries a single (s, t)
    faces = np.asarray(mesh.triangles)
    xyz = np.asarray(mesh.vertices)[faces].reshape(-1, 3)
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)

    # Vertex colour sampled from the atlas at each corner (viewers without texture support)
    h, w = albedo.shape[:2]
    row = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
    col = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    rgb = albedo[row, col]

    # Structured arrays for plyfile: x y z red green blue s t + vertex_indices
    vertex = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("s", "f4"), ("t", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertex["s"], vertex["t"] = uv[:, 0], uv[:, 1]
    face = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = np.arange(len(xyz), dtype=np.int32).reshape(-1, 3)

    mesh_path = out_dir / "mesh.ply"
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")], comments=["TextureFile albedo.png"]).write(str(mesh_path))
    cv2.imwrite(str(out_dir / "albedo.png"), np.ascontiguousarray(albedo[..., ::-1]))
    logger.info("write_textured_ply: %d faces, %dx%d atlas -> %s", len(faces), w, h, mesh_path)
    return mesh_path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST tests/mesh/test_io.py`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/mesh/io.py tests/mesh/test_io.py && git commit --only collab_splats/mesh/io.py tests/mesh/test_io.py -m "feat(mesh): io.py — upsample_depths, render_tsdf_inputs (ckpt + source frames), write_textured_ply

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `texture.py` — decimate, repair, unwrap, project, write

**Files:**
- Create: `collab_splats/mesh/texture.py`
- Create: `tests/mesh/test_texture.py`

Ported verbatim from `feat/mesh-texture-bake`'s `collab_splats/mesh/texture.py` (worktree `.worktrees/mesh-texture-bake`): `decimate` → `decimate_mesh`, `split_non_manifold_vertices` → `_split_non_manifold_vertices`, `_find`, `_drop_duplicate_and_fold_over_faces`, `_clean` → `_make_manifold` (copies its input instead of mutating; no `stats` dict). Dropped from that branch: `measure_deviation`, `project_normals`, `export_obj`, `bake_vertex_colors`, the Open3D `project_images_to_albedo` projector (OOM: 15.9 GB for 2 views at 8192²) and the `_MIN_FACES_PER_PARTITION` module constant (now a kwarg). The projector is an NVIDIA Warp kernel: one thread per texel, cos-weighted average over views, occlusion by `wp.mesh_query_ray` against a BVH of the mesh. Kernels must be module-level (Warp compiles them per module).

Test geometry: camera at (0, 0, 2) looking down −z (`c2w[:3, :3] = diag(1, −1, −1)`), K f=100 c=64 on 128×128 → the unit square at z=0 projects to pixels (64,64)…(114,14), all inside the image. With an occluder square at z=0.5 covering x<0.5, a ray to a floor point (x, y, 0) crosses z=0.5 at (0.75x, 0.75y) → floor points with x<0.667 are hidden. Floor UVs `u=x/2` put those at atlas columns <21 of 64, so columns 0–15 must be black and columns 23–31 coloured. The gutter test compares filled-texel counts on one row between `gutter_px=0` and `gutter_px=4`: +8 exactly, independent of Open3D's bake margin.

- [ ] **Step 1: Install meshoptimizer (needs user approval — shared venv)**

```bash
uv pip install --python /opt/venv/reconstruction/bin/python meshoptimizer==0.2.30a0
/opt/venv/reconstruction/bin/python -c "import importlib.metadata as m; print(m.version('meshoptimizer'), m.version('warp-lang'))"
```

Expected: `0.2.30a0 1.14.0`.

- [ ] **Step 2: Write the failing tests**

`tests/mesh/test_texture.py`:

```python
import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest
from plyfile import PlyData

pytest.importorskip("warp")
pytest.importorskip("meshoptimizer")

from collab_splats.mesh.texture import (  # noqa: E402
    _make_manifold,
    _split_non_manifold_vertices,
    decimate_mesh,
    project_images_to_texture,
    texture_mesh,
    unwrap_mesh_uvs,
)


######## Fixtures


def _dense_plane(n=60, noise=0.0, seed=0):
    """Unit plane in z=0 tessellated n×n, optional gaussian z-noise; normals face +z."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([xs.ravel(), ys.ravel(), rng.normal(0, noise, n * n)], axis=1)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))


def _sphere(res=40):
    return o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=res)


def _bowtie():
    """Two triangles sharing only vertex 0 (a non-manifold vertex, no non-manifold edge)."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
    f = np.array([[0, 1, 2], [0, 3, 4]])
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))


def _deviation_p99(reference, decimated):
    """99th-percentile distance from reference vertices to the decimated surface."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(decimated))
    d = scene.compute_distance(o3c.Tensor(np.asarray(reference.vertices), dtype=o3c.float32)).numpy()
    return float(np.percentile(d, 99))


def _camera():
    """One camera 2 units up +z looking down at the origin; K f=100 c=64 on a 128² image."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = [0.0, 0.0, 2.0]
    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
    return c2w[None], K[None]


def _squares_tm(squares):
    """Tensor mesh of unit squares; each entry is (z, uv_offset, uv_scale, flip_winding)."""
    verts, faces, uvs = [], [], []
    for z, uv_offset, uv_scale, flip in squares:
        v = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3]], np.int32) + 4 * len(verts)
        if flip:
            f = f[:, ::-1].copy()
        verts.append(v)
        faces.append(f)
        uvs.append((np.asarray(uv_offset) + np.asarray(uv_scale) * v[f - 4 * (len(verts) - 1)][..., :2]).astype(np.float32))
    tm = o3d.t.geometry.TriangleMesh(o3c.Tensor(np.vstack(verts)), o3c.Tensor(np.vstack(faces)))
    tm.triangle.texture_uvs = o3c.Tensor(np.concatenate(uvs))
    return tm


def _constant_image(rgb=(51, 128, 204)):
    return np.full((1, 128, 128, 3), rgb, dtype=np.uint8)


######## _split_non_manifold_vertices / _make_manifold


def test_split_non_manifold_vertices_bowtie():
    m = _bowtie()
    assert len(m.get_non_manifold_vertices()) == 1
    out, n_split = _split_non_manifold_vertices(m)
    assert n_split == 1
    assert len(out.get_non_manifold_vertices()) == 0
    assert len(out.vertices) == 6 and len(out.triangles) == 2
    v0, v1 = np.asarray(m.vertices), np.asarray(out.vertices)
    f0, f1 = np.asarray(m.triangles), np.asarray(out.triangles)
    assert np.allclose(v0[f0], v1[f1])  # every corner keeps its position
    assert np.allclose(v1[5], v0[0])  # the copy sits exactly on vertex 0
    assert len(m.vertices) == 5  # input not mutated


def test_split_non_manifold_vertices_keeps_colours():
    m = _bowtie()
    m.vertex_colors = o3d.utility.Vector3dVector(np.linspace(0, 1, 15).reshape(5, 3))
    out, _ = _split_non_manifold_vertices(m)
    c = np.asarray(out.vertex_colors)
    assert c.shape == (6, 3) and np.allclose(c[5], c[0])


def test_split_non_manifold_vertices_noop_on_manifold():
    m = _sphere(10)
    out, n_split = _split_non_manifold_vertices(m)
    assert n_split == 0 and len(out.vertices) == len(m.vertices)


def test_make_manifold_drops_opposite_winding_duplicates_and_fold_overs():
    # One face duplicated in reverse winding and one folded back over a neighbour's directed
    # edge: Open3D calls both manifold, UVAtlas calls both non-manifold
    m = _dense_plane(n=4)
    f = np.asarray(m.triangles)
    dup = f[0][[0, 2, 1]]
    fold = np.array([f[3][0], f[3][1], 99])
    v = np.vstack([np.asarray(m.vertices), [[5.0, 5.0, 5.0]] * 84])
    f = np.vstack([f, dup[None], fold[None]])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    assert not m.is_orientable()
    n_faces_in = len(m.triangles)
    out = _make_manifold(m)
    assert len(m.triangles) == n_faces_in  # input not mutated
    assert out.is_orientable()
    assert len(out.get_non_manifold_edges()) == 0 and len(out.get_non_manifold_vertices()) == 0
    fo = np.asarray(out.triangles)
    de = np.concatenate([fo[:, [0, 1]], fo[:, [1, 2]], fo[:, [2, 0]]])
    assert np.unique(de, axis=0).shape[0] == len(de)  # every directed edge used once
    assert len(fo) == 18  # dup dropped, fold dropped, f3 (first owner) kept
    o3d.t.geometry.TriangleMesh.from_legacy(out).compute_uvatlas(size=64)


def test_make_manifold_removes_degenerate_before_fold_over_check():
    # Degenerate [0,1,1] carries directed edge (0,1) shared with valid [0,1,2]; only the
    # degenerate face may go
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 1]])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    fo = np.asarray(_make_manifold(m).triangles)
    assert len(fo) == 2 and [0, 1, 2] in fo.tolist()


def test_make_manifold_after_decimate_yields_uvatlas_ready_mesh():
    m = _make_manifold(_dense_plane(n=80, noise=0.003))
    out, _ = decimate_mesh(m, 0.01)
    out = _make_manifold(out)
    assert len(out.get_non_manifold_edges()) == 0
    assert len(out.get_non_manifold_vertices()) == 0
    tm = o3d.t.geometry.TriangleMesh.from_legacy(out)
    tm.compute_uvatlas(size=256)
    assert tm.triangle.texture_uvs.shape[0] == len(out.triangles)


######## decimate_mesh


def test_decimate_mesh_respects_absolute_bound():
    m = _dense_plane(noise=0.002)
    bound = 0.01
    out, result_error = decimate_mesh(m, bound)
    assert len(out.triangles) < len(m.triangles)
    assert _deviation_p99(m, out) <= 1.5 * bound
    assert result_error <= bound + 1e-6


def test_decimate_mesh_keeps_curvature_relative_to_planes():
    plane, sphere = _dense_plane(n=60), _sphere(res=40)
    p, _ = decimate_mesh(plane, 0.01)
    s, _ = decimate_mesh(sphere, 0.01)
    assert len(p.triangles) < 0.05 * len(plane.triangles)  # a plane collapses to a handful
    assert len(s.triangles) > len(p.triangles)  # a sphere keeps many to stay in bound


def test_decimate_mesh_never_moves_vertices():
    m = _dense_plane(n=20, noise=0.001)
    out, _ = decimate_mesh(m, 0.01)
    v_in = {tuple(np.round(x, 9)) for x in np.asarray(m.vertices)}
    assert all(tuple(np.round(x, 9)) in v_in for x in np.asarray(out.vertices))


######## unwrap_mesh_uvs


def test_unwrap_mesh_uvs_gives_per_corner_uvs_in_unit_square():
    mesh = _sphere(10)
    tm = unwrap_mesh_uvs(mesh, tex_size=64)
    uv = tm.triangle.texture_uvs.numpy()
    assert uv.shape == (len(mesh.triangles), 3, 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0


######## project_images_to_texture


def test_project_images_to_texture_constant_view_gives_constant_albedo():
    tm = _squares_tm([(0.0, (0, 0), (1, 1), False)])
    c2w, K = _camera()
    albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=0)
    assert albedo.shape == (64, 64, 3) and albedo.dtype == np.uint8
    filled = albedo.any(axis=-1)
    assert filled.mean() >= 0.95
    assert np.abs(albedo[filled].astype(int) - [51, 128, 204]).max() <= 2


def test_project_images_to_texture_occluded_texels_stay_black():
    # Floor square (uv u=x/2 → left half of the atlas) under an occluder at z=0.5 covering
    # x<0.5 (uv u=0.5+x → right half). By perspective the occluder hides floor x<0.667.
    tm = _squares_tm([(0.0, (0, 0), (0.5, 1), False), (0.5, (0.5, 0), (1, 1), False)])
    c2w, K = _camera()
    albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=0)
    assert albedo[:, :16].max() == 0  # floor x<0.5: hidden
    assert albedo[:, 23:32].any(axis=-1).mean() > 0.95  # floor x>0.72: seen
    assert albedo[:, 32:].any(axis=-1).mean() > 0.95  # the occluder itself


def test_project_images_to_texture_back_faces_get_nothing():
    tm = _squares_tm([(0.0, (0, 0), (1, 1), True)])  # winding flipped: normal points away
    c2w, K = _camera()
    albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=0)
    assert albedo.max() == 0


def test_project_images_to_texture_gutter_grows_filled_region():
    tm = _squares_tm([(0.0, (0.25, 0.25), (0.5, 0.5), False)])  # chart in the atlas centre
    c2w, K = _camera()
    filled = {}
    for gutter in (0, 4):
        albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=gutter)
        filled[gutter] = albedo.any(axis=-1)
    assert filled[4][32].sum() - filled[0][32].sum() == 8
    assert filled[4][filled[0]].all()  # dilation never clears a filled texel


######## texture_mesh


def test_texture_mesh_writes_textured_ply(tmp_path):
    mesh_path = tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), _dense_plane(30))
    c2w, K = _camera()
    out = texture_mesh(mesh_path, tmp_path / "texture", _constant_image(), c2w, K, voxel_size=0.01, tex_size=64)
    assert out == tmp_path / "texture" / "mesh.ply" and out.exists()
    assert (tmp_path / "texture" / "albedo.png").exists()
    vertex = PlyData.read(str(out))["vertex"]
    assert "s" in vertex.data.dtype.names and "t" in vertex.data.dtype.names
    albedo = cv2.imread(str(tmp_path / "texture" / "albedo.png"))
    assert albedo.shape == (64, 64, 3) and albedo.max() > 0
    assert mesh_path.exists()  # the fused mesh is never modified
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTEST tests/mesh/test_texture.py`
Expected: `ModuleNotFoundError: No module named 'collab_splats.mesh.texture'`.

- [ ] **Step 4: Write `collab_splats/mesh/texture.py`**

```python
"""
Texture a fused mesh: error-bounded decimation, manifold repair, UV atlas, image projection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import meshoptimizer as mo
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
import warp as wp

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses
from collab_splats.mesh.io import write_textured_ply

logger = logging.getLogger(__name__)


######## Decimation


def decimate_mesh(mesh, max_error):
    """
    QEM decimation to an absolute surface-deviation bound; vertices are removed, never moved.

    Args:
        mesh: open3d.geometry.TriangleMesh.
        max_error: largest allowed surface deviation, world units.
    Returns:
        (decimated TriangleMesh, meshoptimizer's result error in world units).
    """
    # meshoptimizer simplify with an absolute error bound; target_index_count=3 = as few as allowed
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


######## Manifold repair — UVAtlas rejects what Open3D calls manifold


def _find(parent, x):
    """
    Union-find root of x with path halving.

    Args:
        parent: dict node -> parent node.
        x: node.
    Returns:
        root node of x.
    """
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _split_non_manifold_vertices(mesh):
    """
    Duplicate every bowtie vertex once per extra edge-connected triangle fan.

    Args:
        mesh: open3d.geometry.TriangleMesh; not modified.
    Returns:
        (new TriangleMesh, number of vertices split).
    """
    v = np.asarray(mesh.vertices).copy()
    f = np.asarray(mesh.triangles).copy()
    colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
    nm = np.asarray(mesh.get_non_manifold_vertices(), dtype=np.int64)
    if len(nm) == 0:
        return o3d.geometry.TriangleMesh(mesh), 0

    # Vertex → incident-triangle index built once from the flattened corner array
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(len(v) + 1))
    new_v, new_c = [], []
    n_split = 0
    for vid in nm:
        tris = order[starts[vid] : starts[vid + 1]] // 3

        # Union-find over incident triangles: same fan iff they share a vertex besides vid
        parent = {int(t): int(t) for t in tris}
        other = {}
        for t in tris:
            for w in f[t]:
                if w != vid:
                    other.setdefault(int(w), []).append(int(t))
        for group in other.values():
            for t in group[1:]:
                parent[_find(parent, t)] = _find(parent, group[0])
        fans = {}
        for t in tris:
            fans.setdefault(_find(parent, int(t)), []).append(int(t))

        # First fan keeps vid; every further fan is rewired to a fresh copy
        for fan in list(fans.values())[1:]:
            nid = len(v) + len(new_v)
            new_v.append(v[vid])
            if colors is not None:
                new_c.append(colors[vid])
            for t in fan:
                f[t][f[t] == vid] = nid
            n_split += 1

    if new_v:
        v = np.vstack([v, np.asarray(new_v)])
        if colors is not None:
            colors = np.vstack([colors, np.asarray(new_c)])
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    if colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(colors)
    return out, n_split


def _drop_duplicate_and_fold_over_faces(mesh):
    """
    Drop duplicate faces in any winding and fold-overs (a directed edge owned by two faces).

    Args:
        mesh: open3d.geometry.TriangleMesh with degenerate faces already removed.
    Returns:
        (new TriangleMesh, duplicates dropped, fold-overs dropped).
    """
    f = np.asarray(mesh.triangles)
    n_in = len(f)
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    n_dup = n_in - len(f)

    # Directed-edge multiplicity: a manifold, orientable surface uses each direction once.
    # The first face owning a direction keeps it; any face owning a non-first copy goes.
    de = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, first_edge = np.unique(de, axis=0, return_index=True)
    non_first = np.ones(len(de), dtype=bool)
    non_first[first_edge] = False
    fold = non_first.reshape(3, -1).any(axis=0)
    f = f[~fold]
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)), o3d.utility.Vector3iVector(f))
    if mesh.has_vertex_colors():
        out.vertex_colors = mesh.vertex_colors
    return out, n_dup, int(fold.sum())


def _make_manifold(mesh):
    """
    Repair a mesh to a UVAtlas-ready one; the input is left untouched.

    Args:
        mesh: open3d.geometry.TriangleMesh.
    Returns:
        new TriangleMesh without degenerate, duplicate or fold-over faces, non-manifold
        edges or vertices, or orphan vertices.
    """
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_degenerate_triangles()
    mesh, n_dup, n_fold = _drop_duplicate_and_fold_over_faces(mesh)
    mesh.remove_non_manifold_edges()
    mesh, n_split = _split_non_manifold_vertices(mesh)
    mesh.remove_unreferenced_vertices()
    logger.info("_make_manifold: dropped %d duplicate + %d fold-over faces, split %d bowtie vertices", n_dup, n_fold, n_split)
    return mesh


######## UV atlas


def unwrap_mesh_uvs(mesh, tex_size, parallel_partitions=16, min_faces_per_partition=1000):
    """
    Compute a UV atlas with Open3D's UVAtlas.

    Args:
        mesh: manifold open3d.geometry.TriangleMesh (see _make_manifold).
        tex_size: atlas edge in texels.
        parallel_partitions: UVAtlas partitions run in parallel (1 = single-threaded, 20+ min at 500k faces).
        min_faces_per_partition: floor that clamps the partition count (Open3D's PCA partition raises on an empty one).
    Returns:
        o3d.t.geometry.TriangleMesh with triangle.texture_uvs (F, 3, 2).
    """
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    partitions = max(1, min(int(parallel_partitions), len(mesh.triangles) // min_faces_per_partition))
    tm.compute_uvatlas(size=tex_size, parallel_partitions=partitions)
    return tm

def bake_atlas_attributes(tm, tex_size):
    """
    Rasterize a UV atlas into per-texel world position and normal.

    Args:
        tm: o3d.t.geometry.TriangleMesh with triangle.texture_uvs (from unwrap_mesh_uvs).
        tex_size: atlas edge in texels.
    Returns:
        (positions, normals): two (tex_size, tex_size, 3) float32 arrays, zero on texels no triangle covers.
    """
    tm.compute_vertex_normals()
    verts = tm.vertex.positions.numpy()
    faces = tm.triangle.indices.numpy()
    vert_normals = tm.vertex.normals.numpy()

    # One vertex per triangle corner: a vertex on a chart seam carries a different UV in each
    # triangle sharing it, so a shared-vertex buffer cannot express the atlas.
    corner_pos = np.ascontiguousarray(verts[faces].reshape(-1, 3), dtype=np.float32)
    corner_nrm = np.ascontiguousarray(vert_normals[faces].reshape(-1, 3), dtype=np.float32)
    corner_tri = np.arange(len(corner_pos), dtype=np.int32).reshape(-1, 3)

    # UV to clip space. nvdiffrast's first output row is the top of the atlas, so v flips
    # (measured: unflipped disagrees with Open3D's bake by 47.8 world units, flipped by 2e-5).
    uv = tm.triangle.texture_uvs.numpy().reshape(-1, 2)
    zeros = np.zeros(len(uv), dtype=np.float32)
    clip = np.stack([uv[:, 0] * 2.0 - 1.0, (1.0 - uv[:, 1]) * 2.0 - 1.0, zeros, zeros + 1.0], axis=-1)

    # Rasterize once, interpolate both attributes off the same fragment buffer
    tri = torch.as_tensor(corner_tri, device="cuda")
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(
        ctx, torch.as_tensor(clip.astype(np.float32), device="cuda")[None], tri, resolution=[tex_size, tex_size]
    )
    positions, _ = dr.interpolate(torch.as_tensor(corner_pos, device="cuda")[None], rast, tri)
    normals, _ = dr.interpolate(torch.as_tensor(corner_nrm, device="cuda")[None], rast, tri)

    # Uncovered texels keep a zero normal, which is what the projection kernel rejects on
    covered = (rast[0, ..., 3] > 0).unsqueeze(-1)
    return (positions[0] * covered).cpu().numpy(), (normals[0] * covered).cpu().numpy()


######## Projection — NVIDIA Warp, one thread per texel


@wp.kernel
def _project_kernel(
    positions: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    image: wp.array2d(dtype=wp.vec3),
    w2c: wp.mat44,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam: wp.vec3,
    mesh_id: wp.uint64,
    eps: float,
    rgb_acc: wp.array2d(dtype=wp.vec3),
    w_acc: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    p = positions[i, j]
    n = normals[i, j]
    if wp.length(n) < 0.5:
        return
    n = wp.normalize(n)

    # Back-face and grazing rejection: weight is the cosine to the camera
    to_cam = cam - p
    dist = wp.length(to_cam)
    d = to_cam / dist
    cos = wp.dot(n, d)
    if cos <= 0.0:
        return

    # Project into the image; skip texels behind the camera or outside the frame
    pc = wp.transform_point(w2c, p)
    if pc[2] <= 0.0:
        return
    u = fx * pc[0] / pc[2] + cx
    v = fy * pc[1] / pc[2] + cy
    H = image.shape[0]
    W = image.shape[1]
    if u < 0.0 or v < 0.0 or u > float(W - 1) or v > float(H - 1):
        return

    # Occlusion: anything the ray from the camera hits before the texel hides it
    q = wp.mesh_query_ray(mesh_id, cam, -d, dist - eps)
    if q.result:
        return

    # Bilinear sample, cos-weighted accumulate
    x0 = int(wp.floor(u))
    y0 = int(wp.floor(v))
    x1 = wp.min(x0 + 1, W - 1)
    y1 = wp.min(y0 + 1, H - 1)
    ax = u - float(x0)
    ay = v - float(y0)
    c = (image[y0, x0] * (1.0 - ax) + image[y0, x1] * ax) * (1.0 - ay) + (image[y1, x0] * (1.0 - ax) + image[y1, x1] * ax) * ay
    rgb_acc[i, j] = rgb_acc[i, j] + c * cos
    w_acc[i, j] = w_acc[i, j] + cos


def _dilate_texels(albedo, filled, gutter_px):
    """
    Grow filled texels into unfilled neighbours so bilinear sampling never reads black seams.

    Args:
        albedo: (S, S, 3) float atlas.
        filled: (S, S) bool, texels some view coloured.
        gutter_px: dilation radius in texels.
    Returns:
        (S, S, 3) float32 atlas with the gutter filled by the mean of filled neighbours.
    """
    out = albedo.astype(np.float32)
    mask = filled.astype(np.float32)
    for _ in range(gutter_px):
        num = cv2.boxFilter(out * mask[..., None], -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        den = cv2.boxFilter(mask, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        grow = (den > 0) & (mask == 0)
        out[grow] = num[grow] / den[grow][:, None]
        mask[grow] = 1.0
    return out


def project_images_to_texture(tm, rgbs, c2w, K, tex_size, occlusion_eps, gutter_px=4):
    """
    Visibility-weighted projection of images into a mesh's UV atlas.

    Args:
        tm: o3d.t.geometry.TriangleMesh with triangle.texture_uvs (from unwrap_mesh_uvs).
        rgbs: (N, H, W, 3) uint8 images.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        tex_size: atlas edge in texels.
        occlusion_eps: ray-test tolerance, world units (≈ voxel size) so a surface never occludes itself.
        gutter_px: texels of colour dilation around every chart.
    Returns:
        (tex_size, tex_size, 3) uint8 albedo; texels no view saw are 0.
    """
    wp.init()

    # Per-texel world position and normal from the atlas
    positions, normals = bake_atlas_attributes(tm, tex_size)
    posw = wp.array(positions, dtype=wp.vec3)
    nrmw = wp.array(normals, dtype=wp.vec3)

    # BVH over the mesh for the occlusion ray test
    verts = tm.vertex.positions.numpy().astype(np.float32)
    faces = tm.triangle.indices.numpy().astype(np.int32)
    wmesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3), indices=wp.array(faces.ravel(), dtype=wp.int32))

    # Accumulate cos-weighted colour over every view
    rgb_acc = wp.zeros((tex_size, tex_size), dtype=wp.vec3)
    w_acc = wp.zeros((tex_size, tex_size), dtype=float)
    c2w = np.asarray(c2w, dtype=np.float64)
    w2c = invert_poses(c2w)
    for i in range(len(rgbs)):
        image = wp.array(np.ascontiguousarray(rgbs[i], dtype=np.float32) / 255.0, dtype=wp.vec3)
        fx, fy, cx, cy = extract_intrinsics(K[i])
        wp.launch(
            _project_kernel,
            dim=(tex_size, tex_size),
            inputs=[posw, nrmw, image, wp.mat44(w2c[i].astype(np.float32)), fx, fy, cx, cy, wp.vec3(c2w[i, :3, 3].astype(np.float32)), wmesh.id, float(occlusion_eps), rgb_acc, w_acc],
        )

    # Normalise, dilate the gutter, quantise
    rgb = rgb_acc.numpy()
    w = w_acc.numpy()
    albedo = np.where(w[..., None] > 0, rgb / np.maximum(w[..., None], 1e-12), 0.0)
    albedo = _dilate_texels(albedo, w > 0, gutter_px)
    return (np.clip(albedo, 0, 1) * 255).astype(np.uint8)


######## Entry point


def texture_mesh(mesh_path, out_dir, rgbs, c2w, K, *, voxel_size, decimate_max_error=0.25, tex_size=8192):
    """
    Decimate, repair, unwrap and texture a fused mesh; the input file is never modified.

    Args:
        mesh_path: fused mesh.ply (from fuse_tsdf + clean_repair_mesh).
        out_dir: directory to create; receives mesh.ply (with UVs) and albedo.png.
        rgbs: (N, H, W, 3) uint8 views that were fused.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        voxel_size: TSDF voxel the mesh was fused at, world units; sets the decimation bound and the occlusion tolerance.
        decimate_max_error: decimation bound as a multiple of voxel_size.
        tex_size: atlas edge in texels.
    Returns:
        Path to out_dir/mesh.ply.
    """
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    decimated, err = decimate_mesh(mesh, decimate_max_error * voxel_size)
    manifold = _make_manifold(decimated)
    t1 = time.perf_counter()
    tm = unwrap_mesh_uvs(manifold, tex_size)
    t2 = time.perf_counter()
    albedo = project_images_to_texture(tm, rgbs, c2w, K, tex_size, occlusion_eps=voxel_size)
    t3 = time.perf_counter()
    out = write_textured_ply(tm.to_legacy(), tm.triangle.texture_uvs.numpy(), albedo, out_dir)
    logger.info(
        "texture_mesh: %d -> %d faces (error %.4f) in %.1fs, uvatlas %.1fs, projection %.1fs over %d views -> %s",
        len(mesh.triangles), len(manifold.triangles), err, t1 - t0, t2 - t1, t3 - t2, len(rgbs), out,
    )
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTEST tests/mesh/test_texture.py`
Expected: `15 passed`. First run compiles the Warp kernel (~20 s); a `Module collab_splats.mesh.texture load on device 'cuda:0'` line is normal.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/mesh/texture.py tests/mesh/test_texture.py && git commit --only collab_splats/mesh/texture.py tests/mesh/test_texture.py -m "feat(mesh): texture.py — decimate + manifold repair + UV atlas + Warp projection + textured PLY (ports feat/mesh-texture-bake)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: point `geometry/metrics.py` at `mesh.io.upsample_depths`

**Files:**
- Modify: `collab_splats/geometry/metrics.py:328-363`
- Modify: `tests/geometry/test_metrics.py:677-716`

`compute_photometric_ncc` lifts model-resolution depth onto the original frame grid before pairing it with original-resolution K. It calls `guided_upsample_depth` per frame from the deleted `mesh/utils.py`; `upsample_depths` does the same loop over the whole stack, so the per-frame loop collapses to one call and only the K rescale stays a loop. The import stays inline — `collab_splats.mesh` now pulls Warp and meshoptimizer through `texture.py`, and `geometry/__init__.py` pulls bae/vggt/pypose, neither of which a depth metric should pay for.

- [ ] **Step 1: Rewrite the failing test**

Replace `test_the_upsample_guide_is_normalised_whatever_the_backbones_image_scale` (currently `tests/geometry/test_metrics.py:677-716`) with:

```python
def test_the_upsample_guide_is_normalised_whatever_the_backbones_image_scale(monkeypatch):
    """upsample_depths documents a uint8 guide and divides it by 255 internally.

    FeedforwardResult.images is [0, 255] on VGGT-X and [0, 1] on MapAnything, so an uncoerced
    guide is ~255x too flat on one backbone — measured by the reviewer at 0.398 max / 0.013
    mean depth shift on depths of 1-5 — and a float64 guide raises in OpenCV outright. The two
    scales must therefore lift the SAME depth.

    The fixture depth carries an EDGE, not the constant plane every other upsample test uses:
    the guided filter only consults the guide where depth varies, so a constant map returns the
    same answer under any guide at all and could not see this.
    """
    from collab_splats.mesh import io as mesh_io

    real = mesh_io.upsample_depths
    lifted = []

    def spy(depths, rgbs, crop_boxes):
        out = real(depths, rgbs, crop_boxes)
        lifted.append(out)
        return out

    monkeypatch.setattr(mesh_io, "upsample_depths", spy)
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    model_d = np.stack([np.concatenate(
        [np.full((16, 8), 3.0, np.float32), np.full((16, 8), 5.0, np.float32)], axis=1)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))

    compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords, max_separation=1)
    compute_photometric_ncc(img / 255.0, model_d, model_K, e, original_coords=coords,
                            max_separation=1)
    # One call per compute, each lifting the whole stack
    assert len(lifted) == 2
    assert lifted[0].shape == (2, 64, 64) and lifted[1].shape == (2, 64, 64)
    np.testing.assert_array_equal(lifted[0], lifted[1])
    # Anchor: the guide really is load-bearing on this fixture, so the equality above is not
    # two runs of a filter that ignores its guide.
    flat_guide = real(model_d[:1], np.zeros((1, 64, 64, 3), np.uint8), coords[:1, :4])[0]
    assert not np.array_equal(lifted[0][0], flat_guide)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTEST tests/geometry/test_metrics.py::test_the_upsample_guide_is_normalised_whatever_the_backbones_image_scale`
Expected: `ModuleNotFoundError: No module named 'collab_splats.mesh.io'` (the module exists from Task 4 but `collab_splats/mesh/__init__.py` still re-exports the old names — if instead the failure is `AttributeError: module 'collab_splats.mesh.io' has no attribute 'upsample_depths'`, Task 4 was skipped).

- [ ] **Step 3: Rewrite the lift block**

In `collab_splats/geometry/metrics.py`, replace lines 328-363 (from the `# Imported here, not at module top:` comment through `depth, intrinsics = np.stack(lifted_d), np.stack(lifted_K)`) with:

```python
        # Imported here, not at module top: collab_splats.mesh reaches Warp and meshoptimizer
        # through texture.py, and collab_splats.geometry's own __init__ pulls bae/vggt/pypose.
        # A depth metric must not pay either import cost.
        from collab_splats.geometry.bundle_adjustment import _scale_intrinsics_to_original
        from collab_splats.mesh.io import upsample_depths

        model_h, model_w = depth.shape[1:]

        # The guide is documented uint8 and upsample_depths divides it by 255 internally.
        # FeedforwardResult.images is [0, 255] on VGGT-X but [0, 1] on MapAnything, so an
        # uncoerced guide is ~255x too flat on one backbone (measured: 0.398 max depth shift)
        # and float64 raises in OpenCV outright. That [0, 255] vs [0, 1] split is a property of
        # the BACKBONE, not of a frame, so the scale is decided ONCE off the whole array. Deciding
        # it per frame lets a nearly-black frame in a [0, 255] scene — a dark room, a tunnel, a
        # lens-capped shot, every pixel under 1.0 — read as [0, 1] and get amplified 255x:
        # measured 117.78/255 mean absolute guide error on such a frame, black turned near-white.
        rgb_scale = 255.0 if images.max() <= 1.0 else 1.0
        guides = np.clip(np.asarray(images) * rgb_scale, 0, 255).astype(np.uint8)
        lifted_d = upsample_depths(depth, guides, original_coords[:, :4])

        # The CROP was resized to the model grid, so the scale is model/crop, not model/canvas,
        # and the crop origin comes back onto the principal point. The K arithmetic that undoes
        # both is the forward's inverse, shared via _scale_intrinsics_to_original; the scale
        # itself is re-derived here because the forward's principal-point guard returns sx = 1.0
        # on the model-res K we pass.
        lifted_K = []
        for k in range(N):
            tlx, tly, crx, cry = (float(v) for v in original_coords[k][:4])
            sx, sy = model_w / (crx - tlx), model_h / (cry - tly)
            lifted_K.append(_scale_intrinsics_to_original(intrinsics[k], sx, sy, tlx, tly))
        depth, intrinsics = lifted_d, np.stack(lifted_K)
```

Then fix the stale cross-reference three blocks above — `collab_splats/geometry/metrics.py:321` reads `# Same check, same refusal, as the native-resolution mesh path (mesh/utils.py).` Change `mesh/utils.py` to `mesh/io.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST tests/geometry/test_metrics.py`
Expected: all pass, same count as before the change.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/geometry/metrics.py tests/geometry/test_metrics.py && git commit --only collab_splats/geometry/metrics.py tests/geometry/test_metrics.py -m "refactor(geometry): metrics lifts depth through mesh.io.upsample_depths

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: reconstructor — hoist imports, six-key config, new `_run_tsdf_mesh`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (imports; `_run_tsdf_mesh` at 580-691; `mesh()` at 1340-1416; inline imports at 544, 599-604, 724, 1111, 1546-1547)
- Modify: `configs/base.yaml` (mesh block)
- Modify: `tests/wrapper/_stubs.py:28-38`
- Modify: `tests/wrapper/test_reconstructor.py`

`_run_tsdf_mesh` shrinks from sixteen parameters to ten. The old body branched on `native_resolution` (now always on), `color_map_iterations` (deleted — Open3D's rigid colour-map optimiser is dominated by the texture path) and three splat depth-cut knobs (measured no-ops: `clean_repair` already removed everything they removed). The splats branch reads a checkpoint through `render_tsdf_inputs` instead of `splats.zarr`, so `mesh.source: splats` no longer needs the zarr artifact at all.

The `TYPE_CHECKING` guard keeps `Viewer` (a real cycle: the viewer imports the reconstructor) but loses `FeedforwardResult`, which becomes a top-level import along with `invert_poses`, `confidence_mask` and the four mesh entry points. `collab_splats.mesh` reaches vggt through `geometry/__init__.py`, which `reconstructor.py` already pays for via `pointcloud.sfm`.

- [ ] **Step 1: Replace the base.yaml mesh block**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python - <<'PY'
from pathlib import Path
p = Path("configs/base.yaml")
text = p.read_text()
start = text.index("mesh:\n")
end = text.index("# Gaussian splats trained")
block = """mesh:
  enabled: true
  source: feedforward      # feedforward | splats
  voxel_size: 0.0025       # TSDF voxel, world units; sdf_trunc = 4 x voxel_size
  depth_trunc: 1.5         # ignore depth beyond this, world units
  conf_percentile: 20      # drop depth below this confidence percentile (null = off); feedforward only
  texture: false           # decimate + UV atlas + project the fused views; writes mesh/texture/

"""
p.write_text(text[:start] + block + text[end:])
PY
```

Verify: `awk '/^mesh:/,/^$/' configs/base.yaml` prints exactly the six keys above.

- [ ] **Step 2: Update the wrapper test stubs**

In `tests/wrapper/_stubs.py:28-38`, replace the `"mesh"` dict with:

```python
        "mesh": {
            "enabled": True,
            "source": "feedforward",
            "voxel_size": 0.01,
            "depth_trunc": 1.0,
            "conf_percentile": 20,
            "texture": False,
        },
```

In `tests/wrapper/test_reconstructor_preprocess.py:50`, replace the mesh line with:

```python
        "mesh": {"enabled": False, "voxel_size": 0.01, "depth_trunc": 1.0},
```

- [ ] **Step 3: Rewrite the reconstructor tests**

Six edits to `tests/wrapper/test_reconstructor.py`, in this order (later line numbers first so earlier ones do not shift):

1. `tests/wrapper/test_reconstructor.py:1384-1389` — `test_base_yaml_mesh_has_fidelity_keys` becomes an exact-set assertion:

```python
def test_base_yaml_mesh_has_fidelity_keys():
    """The mesh block is six keys and nothing else — every knob a user can reach."""
    cfg = yaml.safe_load((Path(__file__).parents[2] / "configs" / "base.yaml").read_text())
    assert set(cfg["mesh"]) == {
        "enabled", "source", "voxel_size", "depth_trunc", "conf_percentile", "texture",
    }
    assert cfg["mesh"]["source"] == "feedforward"
    assert cfg["mesh"]["texture"] is False
```

2. `tests/wrapper/test_reconstructor.py:760-780` — DELETE `test_run_tsdf_mesh_passes_clean_repair_to_the_fusion`. Cleaning is unconditional now; there is no flag to forward.

3. `tests/wrapper/test_reconstructor.py:695-717` — replace the `_feedforward_to_tsdf_inputs` intrinsics test with:

```python
def test_run_tsdf_mesh_fuses_colmap_intrinsics(tmp_path, monkeypatch):
    """Model-res depth is lifted onto the frame grid, so COLMAP's original-res K is the right one.

    Pairing the model grid's depth with the original grid's K is the 2026-08-11 collapse bug
    (5.06M -> 75k vertices); this pins the pairing that fixed it.
    """
    result, ff = _tsdf_mesh_doubles(n=2, h=16, w=16)
    monkeypatch.setattr(FeedforwardResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(
        reconstructor.frames, "read_frames",
        lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8),
    )
    monkeypatch.setattr(reconstructor, "upsample_depths", lambda d, r, b: np.ones((2, 32, 32), np.float32))
    fuse = MagicMock(return_value=tmp_path / "mesh.ply")
    monkeypatch.setattr(reconstructor, "fuse_tsdf", fuse)
    monkeypatch.setattr(reconstructor, "clean_repair_mesh", MagicMock())

    reconstructor._run_tsdf_mesh(
        result=result,
        pointcloud_zarr=tmp_path / "pointcloud.zarr",
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        voxel_size=0.01,
        depth_trunc=2.0,
    )
    np.testing.assert_array_equal(fuse.call_args.args[3], result.intrinsics)
```

`_tsdf_mesh_doubles` is the existing helper; give its `original_coords` rows `[0, 0, 2 * w, 2 * h, 2 * w, 2 * h]` so the crop boxes match the 32x32 frames `read_frames` returns.

4. `tests/wrapper/test_reconstructor.py:631-659` — DELETE `test_mesh_forwards_clean_repair_from_config` and `test_mesh_clean_repair_defaults_off`, and add in their place:

```python
def test_run_tsdf_mesh_masks_depth_by_confidence(tmp_path, monkeypatch):
    """conf_percentile zeroes the depth under the percentile before it reaches the fusion."""
    result, ff = _tsdf_mesh_doubles(n=2, h=16, w=16)
    ff.confidence = np.tile(np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16), (2, 1, 1))
    monkeypatch.setattr(FeedforwardResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(
        reconstructor.frames, "read_frames",
        lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8),
    )
    seen = {}

    def spy_upsample(depths, rgbs, boxes):
        seen["depths"] = depths
        return np.ones((2, 32, 32), np.float32)

    monkeypatch.setattr(reconstructor, "upsample_depths", spy_upsample)
    monkeypatch.setattr(reconstructor, "fuse_tsdf", MagicMock(return_value=tmp_path / "mesh.ply"))
    monkeypatch.setattr(reconstructor, "clean_repair_mesh", MagicMock())

    reconstructor._run_tsdf_mesh(
        result=result,
        pointcloud_zarr=tmp_path / "pointcloud.zarr",
        output_dir=tmp_path,
        images_dir=tmp_path / "images",
        voxel_size=0.01,
        depth_trunc=2.0,
        conf_percentile=20,
    )
    # The bottom 20% of a linear ramp is zeroed, the rest is untouched
    masked = seen["depths"]
    assert (masked == 0).mean() == pytest.approx(0.2, abs=0.02)
    assert masked.max() == ff.depth.max()
```

5. `tests/wrapper/test_reconstructor.py:583-628` — the fusion call in the two remaining fusion tests becomes:

```python
    fuse_tsdf(
        depths,
        np.full((2, 32, 32, 3), 128, np.uint8),
        c2w,
        intrinsics,
        rec.backend_dir,
        voxel_size=0.05,
        depth_trunc=2.0,
    )
```

and the config dict at 613-628 becomes `{"enabled": True, "voxel_size": 0.01}`. Delete the `mesher` key at 569-580 — there is one mesher.

6. `tests/wrapper/test_reconstructor.py:14` — `from collab_splats.mesh.tsdf import Open3DTSDFFusion` becomes `from collab_splats.mesh import fuse_tsdf`; drop the `sdf_trunc` kwarg wherever it appears (lines 38 and 113).

Leave `test_run_tsdf_mesh_uses_colmap_poses` (720-741) and the frame-count-mismatch test (744-757) as they are apart from the new signature — the first still asserts `np.linalg.inv(result.extrinsics)` reaches the fusion within `atol=1e-5`, the second still matches on `"pointcloud.zarr"`.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `PYTEST tests/wrapper/test_reconstructor.py`
Expected: `ImportError: cannot import name 'fuse_tsdf' from 'collab_splats.mesh'` at collection (the package `__init__.py` is rewritten in Task 12) — or, once that lands, `AttributeError: <module 'collab_splats.wrapper.reconstructor'> does not have the attribute 'upsample_depths'`.

- [ ] **Step 5: Hoist the imports**

In `collab_splats/wrapper/reconstructor.py`, after the `from collab_splats.pointcloud.base import PointcloudResult` line (25), add:

```python
from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import clean_repair_mesh, fuse_tsdf, texture_mesh
from collab_splats.mesh.io import render_tsdf_inputs, upsample_depths
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
```

Extend the `from collab_splats.pointcloud.utils import clean_pointcloud` line (35) to `from collab_splats.pointcloud.utils import clean_pointcloud, confidence_mask`.

Reduce the `TYPE_CHECKING` guard (53-55) to:

```python
if TYPE_CHECKING:
    from collab_splats.viewer import Viewer
```

`Viewer` stays deferred because `collab_splats/viewer.py` imports the reconstructor — a real cycle. `FeedforwardResult` has no such cycle; it was deferred only for import cost, which this module already pays through `pointcloud.sfm`.

Delete the now-shadowed inline imports:
- line 544 `from collab_splats.pointcloud.feedforward.base import FeedforwardResult` (in `_lift_and_save`)
- line 724 the same line in `_build_localization_db`
- line 1111 the same line in `_sfm_result_from_reconstruction`
- lines 1546-1547 `FeedforwardResult` + `confidence_mask` in `splats()`

Then run `cd /workspace/collab-splats/.worktrees/clean-mesh && isort collab_splats/wrapper/reconstructor.py && black collab_splats/wrapper/reconstructor.py`.

- [ ] **Step 6: Rewrite `_run_tsdf_mesh`**

Replace `collab_splats/wrapper/reconstructor.py:580-691` (the whole old function) with:

```python
def _run_tsdf_mesh(
    result: "PointcloudResult",
    pointcloud_zarr: Path,
    output_dir: Path,
    images_dir: Path,
    voxel_size: float,
    depth_trunc: float,
    conf_percentile: float | None = None,
    source: str = "feedforward",
    splats_ckpt: Path | None = None,
    texture: bool = False,
) -> Path:
    """
    Fuse depth and RGB into a TSDF mesh, clean it, and optionally texture it.

    Args:
        result: PointcloudResult; supplies COLMAP poses and original-res K on the feedforward path.
        pointcloud_zarr: the scene's pointcloud.zarr; read on the feedforward path only.
        output_dir: receives mesh.ply, and texture/ when texture is set.
        images_dir: the scene's images/ directory of original-resolution keyframes.
        voxel_size: TSDF voxel edge, world units; sdf_trunc = 4 x voxel_size.
        depth_trunc: ignore depth beyond this, world units.
        conf_percentile: drop depth below this confidence percentile (None = off); feedforward only.
        source: "feedforward" (zarr depth lifted to frame resolution) or "splats" (checkpoint renders).
        splats_ckpt: the splats stage's ckpt.pt; required when source is "splats".
        texture: also decimate, unwrap and project the fused views into output_dir/texture/.
    Returns:
        Path to output_dir/mesh.ply.
    """
    # Splats source: renders come out at frame resolution carrying the poses they were rendered
    # with, pose-opt deltas included, so nothing here has to be lifted or re-posed.
    if source == "splats":
        depths, rgbs, c2w, intrinsics = render_tsdf_inputs(splats_ckpt, images_dir)
    else:
        ff = FeedforwardResult.load_zarr(
            pointcloud_zarr, load_images=False, load_world_points=False
        )
        if ff.depth is None:
            raise ValueError(f"{pointcloud_zarr} has no depth — cannot mesh.")
        if result.extrinsics.shape[0] != ff.depth.shape[0]:
            raise ValueError(
                f"Frame-count mismatch: COLMAP reconstruction has {result.extrinsics.shape[0]} "
                f"images but {pointcloud_zarr} has {ff.depth.shape[0]}. They are from different "
                "runs — re-run the pointcloud stage, or point --stages mesh at the matching scene."
            )

        # Confidence masking first, on the model grid the confidence was predicted on.
        # Absent confidence is a property of the method (sfm, and any backend that ships none),
        # not an error — fuse unmasked and say so.
        depth = np.asarray(ff.depth)
        if conf_percentile is not None:
            if ff.confidence is None:
                logger.info(
                    "mesh.conf_percentile=%s but %s has no confidence array — fusing unmasked",
                    conf_percentile, pointcloud_zarr,
                )
            else:
                keep = confidence_mask(np.asarray(ff.confidence), conf_percentile)
                depth = np.where(keep, depth, 0.0)

        # Lift model-res depth onto the original frame grid so COLMAP's original-res K is the
        # right one to fuse with. Pairing one grid's depth with the other grid's K is the
        # 2026-08-11 collapse bug (5.06M -> 75k vertices).
        rgbs = frames.read_frames(images_dir)
        depths = upsample_depths(depth, rgbs, np.asarray(ff.original_coords)[:, :4])
        c2w = invert_poses(result.extrinsics)
        intrinsics = result.intrinsics

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = fuse_tsdf(
        depths, rgbs, c2w, intrinsics, output_dir,
        voxel_size=voxel_size, depth_trunc=depth_trunc,
    )
    clean_repair_mesh(mesh_path)
    if texture:
        texture_mesh(mesh_path, output_dir / "texture", rgbs, c2w, intrinsics, voxel_size=voxel_size)
    return mesh_path
```

- [ ] **Step 7: Rewrite the `mesh()` config plumbing**

In `collab_splats/wrapper/reconstructor.py`, inside `mesh()`, replace the `splats_zarr` block and the `_run_tsdf_mesh(...)` call. The `splats_zarr` lines become:

```python
        splats_ckpt = None
        if source == "splats":
            splats_ckpt = self.backend_dir / "splats" / "ckpt.pt"
            if not splats_ckpt.exists():
                raise ValueError(
                    f"mesh.source: splats needs {splats_ckpt} — run the splats stage first "
                    "(it is never auto-run)"
                )
```

and the call becomes:

```python
        out = _run_tsdf_mesh(
            result=result,
            pointcloud_zarr=pointcloud_zarr,
            output_dir=self.backend_dir,
            images_dir=self.images_dir,
            voxel_size=mesh_cfg["voxel_size"],
            depth_trunc=mesh_cfg["depth_trunc"],
            conf_percentile=mesh_cfg["conf_percentile"],
            source=source,
            splats_ckpt=splats_ckpt,
            texture=mesh_cfg["texture"],
        )
```

Leave the `depth_scale` attr check for sfm scenes exactly as it is — it is the guard that catches a legacy VDA-metric store fused against COLMAP poses.

Update the docstring's second paragraph to say the splats path fuses the checkpoint's renders rather than `splats.zarr`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `PYTEST tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py`
Expected: all pass. `test_reconstructor.py` should report two fewer tests than before (the two deleted clean_repair tests) plus one new confidence test — net one fewer.

- [ ] **Step 9: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/_stubs.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py && git commit --only collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/_stubs.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py -m "refactor(wrapper): six-key mesh config, hoisted imports, _run_tsdf_mesh over the new mesh API

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: splats-stage and absent-confidence tests follow the new seam

**Files:**
- Modify: `tests/wrapper/test_splats_stage.py`
- Move: `tests/mesh/test_absent_confidence.py` → `tests/wrapper/test_absent_confidence.py`

`mesh.source: splats` now reads `ckpt.pt` through `render_tsdf_inputs`, so `splats.zarr` stops being a mesh input and `_write_minimal_splats_zarr` has no consumer. The two `splat_depth` pass-through tests go with it — `render_tsdf_inputs` picks `median_depth` itself when the checkpoint renders it, which `tests/mesh/test_io.py::test_render_tsdf_inputs_prefers_median_depth` already pins.

`tests/mesh/test_absent_confidence.py` tests three seams and none of them is in `mesh/` any more: the masking decision moved into `_run_tsdf_mesh`, `lift_features` lives in `pointcloud/utils.py`, and the third is `Reconstructor.splats()`. The file moves to `tests/wrapper/`.

- [ ] **Step 1: Edit `tests/wrapper/test_splats_stage.py`**

Delete `_write_minimal_splats_zarr` (149-161), `test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter` and `test_mesh_stage_forwards_splat_depth_from_the_config` (the last two functions in the file). Keep `zarr` imported — the sfm `depth_scale` tests still use it.

Rename `test_mesh_source_splats_without_zarr_raises` and rewrite the fuses-from test:

```python
def test_mesh_source_splats_without_ckpt_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    with pytest.raises(ValueError, match="mesh.source: splats"):
        recon.mesh()


def test_mesh_source_splats_fuses_from_ckpt(tmp_path):
    """The splats path renders the checkpoint; splats.zarr is not an input any more."""
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    splats_dir = recon.backend_dir / "splats"
    splats_dir.mkdir(parents=True)
    (splats_dir / "ckpt.pt").touch()
    rendered = (
        np.ones((3, 4, 5), np.float32),
        np.full((3, 4, 5, 3), 7, np.uint8),
        np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
    )
    with (
        patch("collab_splats.wrapper.reconstructor.render_tsdf_inputs", return_value=rendered) as render,
        patch("collab_splats.wrapper.reconstructor.fuse_tsdf", return_value=recon.backend_dir / "mesh.ply") as fuse,
        patch("collab_splats.wrapper.reconstructor.clean_repair_mesh"),
    ):
        out = recon.mesh()

    assert render.call_args.args == (splats_dir / "ckpt.pt", recon.images_dir)
    assert fuse.call_args.args[0].shape[0] == 3
    assert out == recon.backend_dir / "mesh.ply"
```

Leave `test_mesh_source_unknown_raises`, `test_splats_sfm_aligned_zarr_uses_zarr_depth`, `test_splats_sfm_legacy_zarr_refused`, `test_mesh_sfm_legacy_zarr_refused` and `test_mesh_sfm_aligned_zarr_fuses` untouched.

- [ ] **Step 2: Move and rewrite the absent-confidence test**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git mv tests/mesh/test_absent_confidence.py tests/wrapper/test_absent_confidence.py
```

Then in `tests/wrapper/test_absent_confidence.py`: change the module docstring's `_feedforward_to_tsdf_inputs` to `_run_tsdf_mesh`, replace the import at line 18 with the reconstructor pair, and rewrite the first test.

Imports (lines 18-22) become:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import lift_features
from collab_splats.preproc import frames as fr
from collab_splats.wrapper.reconstructor import Reconstructor, _run_tsdf_mesh
```

`_result_no_confidence()` stays exactly as it is. Replace `test_tsdf_inputs_skip_masking_when_confidence_absent` with:

```python
def test_tsdf_inputs_skip_masking_when_confidence_absent(tmp_path, caplog):
    """conf_percentile set + no confidence -> fuse unmasked with a log, not ValueError."""
    result = _result_no_confidence()
    fused = {}

    def spy_fuse(depths, rgbs, c2w, K, out_dir, **kwargs):
        fused["depths"] = depths
        return tmp_path / "mesh.ply"

    with (
        patch.object(FeedforwardResult, "load_zarr", staticmethod(lambda *a, **k: result)),
        patch("collab_splats.wrapper.reconstructor.frames.read_frames",
              return_value=np.zeros((2, 8, 8, 3), np.uint8)),
        patch("collab_splats.wrapper.reconstructor.upsample_depths", side_effect=lambda d, r, b: d),
        patch("collab_splats.wrapper.reconstructor.fuse_tsdf", side_effect=spy_fuse),
        patch("collab_splats.wrapper.reconstructor.clean_repair_mesh"),
        caplog.at_level("INFO"),
    ):
        _run_tsdf_mesh(
            result=SimpleNamespace(
                extrinsics=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
                intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (2, 1, 1)),
            ),
            pointcloud_zarr=tmp_path / "pointcloud.zarr",
            output_dir=tmp_path,
            images_dir=tmp_path / "images",
            voxel_size=0.01,
            depth_trunc=2.0,
            conf_percentile=20,
        )

    np.testing.assert_array_equal(fused["depths"], result.depth)  # unmasked
    assert any("no confidence" in r.message for r in caplog.records)
```

`test_lift_features_uniform_weights_when_confidence_absent` and `test_splats_depth_targets_skip_masking_when_confidence_absent` are unchanged.

- [ ] **Step 3: Run the tests to verify they pass**

Run: `PYTEST tests/wrapper/test_splats_stage.py tests/wrapper/test_absent_confidence.py`
Expected: all pass. `test_splats_stage.py` reports two fewer tests.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add tests/wrapper/test_splats_stage.py tests/wrapper/test_absent_confidence.py tests/mesh/test_absent_confidence.py && git commit --only tests/wrapper/test_splats_stage.py tests/wrapper/test_absent_confidence.py tests/mesh/test_absent_confidence.py -m "test(wrapper): mesh.source splats renders from ckpt.pt; move absent-confidence tests to tests/wrapper

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: dashboard follows the new mesh API

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py` (imports at 21; `_transfer_mesh_features` at 238-255; mesh block at 395-407)
- Modify: `collab_splats/dashboard/config.py:36-39`
- Modify: `collab_splats/dashboard/app.py` (191, 193, 237, 239, 262, 447, 449)
- Modify: `collab_splats/dashboard/viewer.py:287-288`
- Modify: `tests/dashboard/test_config.py:17,19`
- Modify: `tests/dashboard/test_pipeline.py:48-100`

The dashboard is the only other caller of `pointcloud_to_mesh` and `persist_mesh_vertex_features`. It composes arrays at the call site now, which also removes the `FeedforwardResult` coupling from `mesh/`. Two config knobs go: `mesh_sdf_trunc` (derived, `4 x voxel_size`) and `mesh_clean_repair` (always on). The viewer's lazy `features2vertex` import becomes a top-level one — `collab_splats.dashboard.pipeline` already imports `collab_splats.mesh` at module load, so the viewer pays nothing new.

- [ ] **Step 1: Update the dashboard tests**

`tests/dashboard/test_config.py` — delete the two assertions at 17 and 19:

```python
    assert cfg.mesh_sdf_trunc == 0.02
```
```python
    assert cfg.mesh_clean_repair is False
```

`tests/dashboard/test_pipeline.py:48-100` — `fake_result` is a `MagicMock`, and the new mesh block does real numpy work on `result.images`, so give it real arrays. Insert after `fake_result.points = list(range(10))`:

```python
    # Real arrays: the mesh block reads images.max() and transposes, which a MagicMock
    # would answer with another MagicMock and fail on the comparison.
    fake_result.depth = np.ones((3, 4, 4), np.float32)
    fake_result.images = np.zeros((3, 3, 4, 4), np.float32)
    fake_result.extrinsics = np.tile(np.eye(4), (3, 1, 1))
    fake_result.intrinsics = np.tile(np.eye(3), (3, 1, 1))
```

and swap the mesh patch for the two new names:

```python
        patch.object(pl, "fuse_tsdf") as mesh,
        patch.object(pl, "clean_repair_mesh"),
```

Add `import numpy as np` at the top of the file if it is not already there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tests/dashboard/test_pipeline.py tests/dashboard/test_config.py`
Expected: `AttributeError: <module 'collab_splats.dashboard.pipeline'> does not have the attribute 'fuse_tsdf'`.

- [ ] **Step 3: Rewrite `pipeline.py`**

Replace the import at line 21:

```python
from collab_splats.mesh import clean_repair_mesh, features2vertex, fuse_tsdf
```

and add `import open3d as o3d` plus `from collab_splats.geometry.transforms import invert_poses` to the import block (isort will place them).

Replace the last line of `_transfer_mesh_features` (`persist_mesh_vertex_features(...)`) with the three lines it used to wrap:

```python
    point_features = load_point_features(sem_dir)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertex_features = features2vertex(
        np.asarray(mesh.vertices), result.points, point_features, k=k, sdf_trunc=sdf_trunc
    )
    np.save(Path(out_dir) / "vertex_features.npy", vertex_features)
```

Replace the mesh block (395-407) with:

```python
            # Mesh from TSDF depth fusion. The dashboard fuses at model resolution: depth,
            # RGB and K all come off the same FeedforwardResult grid.
            op_log.update_progress(60, "mesh: tsdf fusion")
            t = time.perf_counter()
            depths = np.ascontiguousarray(result.depth, dtype=np.float32)
            images = np.asarray(result.images)
            # images is [0, 255] on VGGT-X but [0, 1] on MapAnything — decide the scale once
            # off the whole array, never per frame (a dark frame reads as [0, 1] and blows up).
            rgb_scale = 255.0 if images.max() <= 1.0 else 1.0
            rgbs = (images.transpose(0, 2, 3, 1) * rgb_scale).clip(0, 255).astype(np.uint8)
            mesh_path = fuse_tsdf(
                depths,
                rgbs,
                invert_poses(result.extrinsics),
                result.intrinsics,
                out_dir,
                voxel_size=config.mesh_voxel_size,
                depth_trunc=config.mesh_depth_trunc,
            )
            clean_repair_mesh(mesh_path)
            op_log.append_line(f"mesh: tsdf in {time.perf_counter() - t:.1f}s")
```

- [ ] **Step 4: Drop the two config knobs**

`collab_splats/dashboard/config.py` — delete lines 37 and 39, leaving:

```python
    # Mesh (TSDF) params
    mesh_voxel_size: float = 0.005
    mesh_depth_trunc: float = 1.0
```

`collab_splats/dashboard/app.py` — delete the `mesh_sdf` and `mesh_clean` widgets (191, 193), their settings entries (237, 239) and their `RunConfig` kwargs (447, 449). The Card at 261-263 becomes:

```python
            pn.Card(self.mesh_voxel, self.mesh_depth, title="Mesh params", collapsed=True),
```

`collab_splats/dashboard/viewer.py` — delete the lazy-import comment and its import at 287-288, and add `from collab_splats.mesh import features2vertex` to the module's import block.

- [ ] **Step 5: Run the tests and the dashboard smoke gate**

Run: `PYTEST tests/dashboard`
Expected: all pass.

Then the mandatory dashboard gate:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the run ends with `SMOKE PASS`. A dashboard change is not committed without it.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add collab_splats/dashboard tests/dashboard && git commit --only collab_splats/dashboard tests/dashboard -m "refactor(dashboard): compose TSDF arrays at the call site; drop mesh_sdf_trunc and mesh_clean_repair

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 10: eval script and the cu121 integration smoke test move to the array API

`evals/scripts/analyze_splats.py` is the only non-test caller of `_feedforward_to_tsdf_inputs`,
`_splats_to_tsdf_inputs` and `mesh_from_tsdf_inputs` outside `mesh/`. It fuses at MODEL
resolution (its `--zarr` is a `FeedforwardResult` zarr and it never sees an `images/` dir), so
the feedforward branch composes arrays straight off the forward pass — no lift, no COLMAP K.

**Files:**
- Modify: `tests/integration/test_pipeline_cu121.py:226-247`
- Modify: `evals/scripts/analyze_splats.py:28-47` (imports + `MESH_KWARGS`), `:185-205` (`build_mesh`), `:277-291` (`main`)

- [ ] **Step 1: Rewrite the integration smoke test**

In `tests/integration/test_pipeline_cu121.py`, replace `test_tsdf_mesh_synthetic` (lines
226-247) with:

```python
def test_tsdf_mesh_synthetic(tmp_path):
    """
    fuse_tsdf over synthetic depth + RGB frames.

    rgbs must be uint8 [0, 255] — fuse_tsdf rejects float colour outright.
    """
    from collab_splats.mesh import fuse_tsdf

    n, h, w = 4, 64, 64
    rng = np.random.default_rng(1)
    depths = np.full((n, h, w), 2.0, dtype=np.float32)
    rgbs = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    c2w = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)
    for i in range(n):
        c2w[i, 2, 3] = i * 0.05
    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (n, 1, 1))

    mesh_path = fuse_tsdf(depths, rgbs, c2w, intrinsics, tmp_path, voxel_size=0.05, depth_trunc=5.0)
    assert isinstance(mesh_path, Path)
    assert mesh_path.exists()
```

The per-test inline import is this file's own convention — every test in
`test_pipeline_cu121.py` imports inside the function body so one broken module cannot take the
whole migration smoke suite down at collection. Keep it.

Note the voxel size: the old test fused a 2 m constant-depth wall at the class default voxel
(0.005 m), which is 400 voxels across the truncation band and slow. `voxel_size=0.05` /
`depth_trunc=5.0` covers the same wall in seconds.

- [ ] **Step 2: Run the test**

Run: `PYTEST tests/integration/test_pipeline_cu121.py::test_tsdf_mesh_synthetic`
Expected: `1 passed`. This is a migration step, not a new behaviour — Task 3 already shipped
`fuse_tsdf`, and this proves the integration harness reaches it. If it fails with
`ImportError: cannot import name 'fuse_tsdf'`, Task 3 was not committed.

- [ ] **Step 3: Rewrite the eval script's imports and `MESH_KWARGS`**

In `evals/scripts/analyze_splats.py`, replace lines 28-47 (the `from collab_splats.mesh.utils`
import through `NORM_RANGE`) with:

```python
from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import clean_repair_mesh, fuse_tsdf
from collab_splats.mesh.io import render_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import confidence_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mirrors Reconstructor._run_tsdf_mesh; pinned so both sources fuse identically.
# sdf_trunc is derived (4 x voxel_size) and clean_repair always runs — neither is a knob.
MESH_KWARGS = {"voxel_size": 0.0025, "depth_trunc": 1.5}
ALPHA_MIN = 0.5
NORM_RANGE = (0.9, 1.1)
```

`render_tsdf_inputs` is imported from `collab_splats.mesh.io` because `mesh/__init__.py`
re-exports only the five pipeline entry points (`fuse_tsdf`, `clean_repair_mesh`,
`texture_mesh`, `features2vertex`, `mesh_clustering`) — see Task 12 Step 5.

- [ ] **Step 4: Rewrite `build_mesh`**

Replace `build_mesh` (lines 185-205) with:

```python
def build_mesh(name: str, inputs: tuple, results_dir: Path) -> dict:
    """
    Fuse and clean one (depths, rgbs, c2w, K) tuple, then flatten the PLY to mesh_<name>.ply.

    Args:
        name: source label — both the row's "source" field and the PLY's filename suffix.
        inputs: (depths, rgbs, c2w, K) exactly as fuse_tsdf takes them.
        results_dir: directory that receives mesh_<name>.ply.
    Returns:
        Stats dict: vertices, triangles, components, largest_component_fraction, source,
        seconds, path.
    """
    depths, rgbs, c2w, intrinsics = inputs
    work_dir = results_dir / f"_mesh_{name}"

    # Time fusion AND cleaning — clean_repair is not optional in the pipeline, so a
    # fuse-only number would understate what the mesh actually costs
    start = time.perf_counter()
    mesh_path = fuse_tsdf(depths, rgbs, c2w, intrinsics, work_dir, **MESH_KWARGS)
    clean_repair_mesh(mesh_path)
    seconds = time.perf_counter() - start

    # Move the PLY to its flat name; the work dir is only ever the mesher's scratch
    out_path = results_dir / f"mesh_{name}.ply"
    shutil.move(str(mesh_path), str(out_path))
    shutil.rmtree(work_dir, ignore_errors=True)

    stats = mesh_stats(out_path)
    stats["source"] = name
    stats["seconds"] = seconds
    stats["path"] = str(out_path)
    logger.info("%s mesh: %s", name, stats)
    return stats
```

The `work_dir.mkdir(parents=True, exist_ok=True)` line is gone: `fuse_tsdf` creates `out_dir`
itself.

- [ ] **Step 5: Rewrite the two input compositions in `main`**

Replace the `# Table 2` block (lines 277-291, from `feedforward = FeedforwardResult.load_zarr`
through the `for primitive, splats_zarr in available:` loop that appends to `mesh_rows`) with:

```python
    # Table 2: feedforward first, then each primitive's renders.
    # Model-resolution fusion — depth, images and K all come off the same forward pass, so
    # nothing is lifted here and no COLMAP camera is involved.
    ff = FeedforwardResult.load_zarr(args.zarr, load_images=True)
    depths = np.ascontiguousarray(ff.depth, dtype=np.float32)

    # Confidence gate before fusion; a reconstruction without confidence (sfm) fuses unmasked
    if args.conf_percentile is not None and ff.confidence is not None:
        keep = confidence_mask(np.asarray(ff.confidence), args.conf_percentile)
        depths = np.where(keep, depths, 0.0)

    # images is (N, 3, H, W) float in [0, 1]; fuse_tsdf takes (N, H, W, 3) uint8
    rgbs = np.asarray(ff.images).transpose(0, 2, 3, 1)
    rgbs = np.ascontiguousarray((np.clip(rgbs, 0.0, 1.0) * 255).round().astype(np.uint8))

    mesh_rows = [
        build_mesh("feedforward", (depths, rgbs, invert_poses(ff.extrinsics), ff.intrinsics), args.results)
    ]

    # Splat renders leave the checkpoint at frame resolution carrying their own poses
    for primitive, splats_zarr in available:
        mesh_rows.append(build_mesh(primitive, render_tsdf_inputs(splats_zarr.parent / "ckpt.pt"), args.results))
```

`available` still gates on `splats.zarr` — table 1 (normals) reads the zarr's stored normals,
which the checkpoint does not carry. Only the mesh row switches to `ckpt.pt`.

- [ ] **Step 6: Verify the script imports and its CLI still parses**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python evals/scripts/analyze_splats.py --help
```

Expected: the proof line points inside `.worktrees/clean-mesh`, then argparse prints the usage
block ending with the `--conf-percentile` option. No `ImportError`, no `NameError`.

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add evals/scripts/analyze_splats.py tests/integration/test_pipeline_cu121.py && git commit --only evals/scripts/analyze_splats.py tests/integration/test_pipeline_cu121.py -m "refactor(evals): analyze_splats and the cu121 smoke test use fuse_tsdf + render_tsdf_inputs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 11: tutorial notebooks fuse arrays

Two notebooks call the retired API. Both are edited by script, not by hand: a notebook is
JSON, and a hand edit that breaks the JSON is not caught until someone opens it.

Both notebooks fuse at MODEL resolution (neither has an `images/` directory in scope), so the
feedforward branch composes arrays straight off the `FeedforwardResult` — the same shape as
the eval script in Task 10.

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb` (cells 2, 6, 8)
- Modify: `docs/source/tutorials/06_mesh/splats_mesh.ipynb` (cells 2, 4, 6, 8, 10)

- [x] **Step 1: Rewrite `02_pointcloud/feedforward_mesh.ipynb`**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python - <<'PY'
import json
from pathlib import Path

NB = Path("docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb")


def replace_cell(cells, anchor, new_src):
    # Match exactly one code cell by a substring unique to it, then swap its source wholesale
    hits = [c for c in cells if c["cell_type"] == "code" and anchor in "".join(c["source"])]
    assert len(hits) == 1, f"{anchor!r} matched {len(hits)} cells, expected 1"
    hits[0]["source"] = new_src.strip("\n").splitlines(keepends=True)


nb = json.loads(NB.read_text())
cells = nb["cells"]

replace_cell(cells, "from collab_splats.mesh import pointcloud_to_mesh", '''
import os
from pathlib import Path

import numpy as np
import open3d as o3d

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import clean_repair_mesh, fuse_tsdf
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
''')

replace_cell(cells, "MESH_DIR = Path(", '''
MESH_DIR = Path("/tmp/feedforward_mesh_demo/mesh")

# Model-resolution arrays straight off the forward pass. images is (N, 3, H, W) float in
# [0, 1]; fuse_tsdf takes (N, H, W, 3) uint8 and camera-to-world poses.
rgbs = np.asarray(result.images).transpose(0, 2, 3, 1)
rgbs = np.ascontiguousarray((np.clip(rgbs, 0.0, 1.0) * 255).round().astype(np.uint8))

mesh_path = fuse_tsdf(
    np.asarray(result.depth),
    rgbs,
    invert_poses(result.extrinsics),
    result.intrinsics,
    MESH_DIR,
    voxel_size=0.005,
    depth_trunc=1.0,
)

# Cleaning is a separate call: fusion decides geometry, cleaning decides what to keep.
# Thresholds are fractions of the mesh's own extent, so there is nothing to tune per scene.
clean_repair_mesh(mesh_path)
print(f"Mesh saved → {mesh_path}")
''')

replace_cell(cells, "# Inspect mesh statistics", '''
# Inspect mesh statistics
mesh = o3d.io.read_triangle_mesh(str(mesh_path))
print(f"Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")
''')

NB.write_text(json.dumps(nb, indent=1) + "\n")
print(f"rewrote {NB}")
PY
```

`json.dumps(nb, indent=1)` plus a trailing newline is exactly what `nbformat.write` emits, so
this produces no spurious whitespace diff.

Expected output: `rewrote docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`. An
`AssertionError: ... matched 0 cells` means the notebook drifted — re-read it and re-anchor
before continuing.

- [x] **Step 2: Rewrite `06_mesh/splats_mesh.ipynb`**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python - <<'PY'
import json
from pathlib import Path

NB = Path("docs/source/tutorials/06_mesh/splats_mesh.ipynb")


def replace_cell(cells, anchor, new_src):
    hits = [c for c in cells if c["cell_type"] == "code" and anchor in "".join(c["source"])]
    assert len(hits) == 1, f"{anchor!r} matched {len(hits)} cells, expected 1"
    hits[0]["source"] = new_src.strip("\n").splitlines(keepends=True)


nb = json.loads(NB.read_text())
cells = nb["cells"]

replace_cell(cells, "from collab_splats.mesh.utils import", '''
import os
from pathlib import Path

import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec
import open3d as o3d
import pyvista as pv
from PIL import Image
from tqdm.auto import tqdm
import matplotlib

matplotlib.use("Agg") if os.environ.get("PYVISTA_OFF_SCREEN") else None
%matplotlib inline

%run ../notebook_utils.py
set_notebook_backend()

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh import clean_repair_mesh, features2vertex, fuse_tsdf, mesh_clustering
from collab_splats.mesh.io import render_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import confidence_mask, lift_features
from collab_splats.preproc import frames as fr
from collab_splats.semantics.features import BaseQueryableExtractor
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import ae_path
from collab_splats.utils.visualization import VIZ_KWARGS
''')

replace_cell(cells, "VOXEL_SIZE = 0.005", '''
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────
VOXEL_SIZE = 0.005  # base.yaml mesh.voxel_size — sdf_trunc is derived (4 × voxel_size)
DEPTH_TRUNC = 1.0  # base.yaml mesh.depth_trunc (scene units — feedforward depth is not metric)
CONF_PERCENTILE = 20.0

SPLATS_CKPT = OUTPUT_DIR / "splats" / "ckpt.pt"  # written by 03_splats/train_splats.ipynb
MESH_FF_DIR = TUTORIAL_CACHE / "mesh_feedforward"
MESH_SPLATS_DIR = TUTORIAL_CACHE / "mesh_splats"

# Semantic query — the tutorial video (C0043) is an outdoor walk; same queries as nb 05
LATENT_DIM = 13
EXTRACTOR = "maskclip"
QUERY_POSITIVE = ["tree"]
QUERY_NEGATIVE = ["ground"]
QUERY_THRESHOLD = 0.6  # vertices scoring above this seed the clusters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# nb 05 cache layout: compressed per-point codes + the AE that decodes them
_lifted = TUTORIAL_CACHE / "lifted"
AE_MASKCLIP = ae_path(_lifted / "semantics", EXTRACTOR)
LIFTED_MASKCLIP = _lifted / f"lifted_{EXTRACTOR}.zarr"
_lifted.mkdir(parents=True, exist_ok=True)

assert RECON.exists(), f"missing {RECON} — run 02_pointcloud/feedforward_methods.ipynb first"
print(f"RECON:       {RECON}")
print(f"SPLATS_CKPT: {SPLATS_CKPT}  (exists: {SPLATS_CKPT.exists()})")
print(f"Device:      {DEVICE}")
''')

replace_cell(cells, "mesh_ff = pointcloud_to_mesh(", '''
# Depth + RGB come from the zarr; poses are the zarr's extrinsics (the pipeline uses COLMAP's)
ff = FeedforwardResult.load_zarr(RECON, load_images=True)
print(f"Loaded: {ff.points.shape[0]:,} pts, {ff.extrinsics.shape[0]} frames, depth {tuple(ff.depth.shape)}")

# Confidence gate first — never fuse depth the model itself is unsure about
depths = np.asarray(ff.depth)
depths = np.where(confidence_mask(np.asarray(ff.confidence), CONF_PERCENTILE), depths, 0.0)

# images is (N, 3, H, W) float in [0, 1]; fuse_tsdf takes (N, H, W, 3) uint8
rgbs = np.asarray(ff.images).transpose(0, 2, 3, 1)
rgbs = np.ascontiguousarray((np.clip(rgbs, 0.0, 1.0) * 255).round().astype(np.uint8))

mesh_ff_path = fuse_tsdf(
    depths,
    rgbs,
    invert_poses(ff.extrinsics),
    ff.intrinsics,
    MESH_FF_DIR,
    voxel_size=VOXEL_SIZE,
    depth_trunc=DEPTH_TRUNC,
)
print(f"Feedforward mesh → {mesh_ff_path}")
''')

replace_cell(cells, "depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(", '''
assert SPLATS_CKPT.exists(), f"missing {SPLATS_CKPT} — run 03_splats/train_splats.ipynb first"

# Rendered depth / RGB / poses straight out of the checkpoint. The renderer already applies
# the alpha gate and returns frame-resolution uint8 colour with the poses it rendered from,
# pose-opt deltas included — nothing to lift, nothing to re-pose.
depths_sp, rgbs_sp, c2w_sp, K_sp = render_tsdf_inputs(SPLATS_CKPT)
print(f"depths {depths_sp.shape}  rgbs {rgbs_sp.shape} {rgbs_sp.dtype}  c2w {c2w_sp.shape}  K {K_sp.shape}")

mesh_sp_path = fuse_tsdf(
    depths_sp,
    rgbs_sp,
    c2w_sp,
    K_sp,
    MESH_SPLATS_DIR,
    voxel_size=VOXEL_SIZE,
    depth_trunc=DEPTH_TRUNC,
)
print(f"Splats mesh → {mesh_sp_path}")
''')

# Cell 10 only needs its two read_triangle_mesh paths repointed
target = next(c for c in cells if "meshes = {" in "".join(c["source"]))
src = "".join(target["source"]).replace("mesh_ff.mesh_path", "mesh_ff_path").replace(
    "mesh_sp.mesh_path", "mesh_sp_path"
)
assert "mesh_ff_path" in src and "mesh_sp_path" in src
target["source"] = src.splitlines(keepends=True)

NB.write_text(json.dumps(nb, indent=1) + "\n")
print(f"rewrote {NB}")
PY
```

Expected output: `rewrote docs/source/tutorials/06_mesh/splats_mesh.ipynb`.

`MESH_FF_DIR` / `MESH_SPLATS_DIR` no longer need an explicit `mkdir` — `fuse_tsdf` creates
its `out_dir`. Cells 15 and 21 (`features2vertex`, `mesh_clustering`) are untouched: both
signatures survive the refactor unchanged, only their import moved from `mesh.utils` to
`mesh`.

- [x] **Step 3: Verify both notebooks are valid JSON and free of the retired API**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python - <<'PY'
import json
from pathlib import Path

RETIRED = [
    "pointcloud_to_mesh",
    "mesh_from_tsdf_inputs",
    "_splats_to_tsdf_inputs",
    "_feedforward_to_tsdf_inputs",
    "collab_splats.mesh.utils",
    "clean_repair=",
    "SDF_TRUNC",
    "SPLATS_ZARR",
    "color_map_iterations",
]
for nb_path in Path("docs/source/tutorials").rglob("*.ipynb"):
    nb = json.loads(nb_path.read_text())
    body = "".join("".join(c["source"]) for c in nb["cells"])
    hits = [name for name in RETIRED if name in body]
    assert not hits, f"{nb_path}: still references {hits}"
print("notebooks clean")
PY
```

Expected: `notebooks clean`. `sdf_trunc=` is deliberately NOT on the banned list —
`features2vertex(..., sdf_trunc=0.03)` keeps that keyword and is unrelated to TSDF fusion.

- [x] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add docs/source/tutorials && git commit --only docs/source/tutorials -m "docs(tutorials): mesh notebooks fuse arrays via fuse_tsdf and render_tsdf_inputs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 12: delete the retired modules and tests

Everything that reads the old API has now moved (Tasks 6-11). This task removes the old API
itself. **Step 1 must run before anything is deleted** — it is the synthetic half of Task 15's
fusion parity (the real-data half, gate 1a, needs the baseline worktree Task 15 Step 1 sets up),
and it
needs both the old class and the new function alive in the same interpreter.

**Files:**
- Delete: `collab_splats/mesh/base.py`, `collab_splats/mesh/poisson.py`, `collab_splats/mesh/utils.py`
- Delete: `tests/mesh/test_utils.py`, `tests/mesh/test_utils_ground_plane.py`, `tests/mesh/test_splats_adapter.py`, `tests/mesh/test_registry.py`, `tests/mesh/test_feature_transfer.py`, `tests/mesh/test_adapter.py`
- Modify: `collab_splats/mesh/tsdf.py` (drop `Open3DTSDFFusion`, shrink the header)
- Modify: `collab_splats/mesh/__init__.py` (re-exports only)
- Modify: `tests/test_cu121_migration.py:119-123`

- [x] **Step 1: Synthetic fusion parity — old class vs new function**

Both fuse the same synthetic views with matched parameters. The new `fuse_tsdf` must produce a
bit-identical mesh, otherwise the refactor moved geometry and every downstream comparison in
this repo silently shifts.

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import open3d as o3d

import collab_splats
print("PROOF:", collab_splats.__file__)
from collab_splats.mesh.tsdf import Open3DTSDFFusion, fuse_tsdf

# A slanted, rippled surface — a constant-depth wall would fill one voxel plane and hide any
# per-view integration difference
rng = np.random.default_rng(0)
n, h, w = 6, 96, 96
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
depths = np.stack([1.5 + 0.4 * (xx / w) + 0.15 * np.sin(6.0 * yy / h)] * n).astype(np.float32)
rgbs = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
c2w[:, 0, 3] = np.linspace(-0.1, 0.1, n)
K = np.tile(np.array([[80, 0, 48], [0, 80, 48], [0, 0, 1]], dtype=np.float32), (n, 1, 1))

out = Path("/tmp/claude-0/-workspace-collab-splats/mesh-parity")
old = Open3DTSDFFusion(
    output_dir=out / "old", voxel_size=0.01, sdf_trunc=0.04, depth_trunc=5.0, clean_repair=False
).create(depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=K)
new = fuse_tsdf(depths, rgbs, c2w, K, out / "new", voxel_size=0.01, depth_trunc=5.0, sdf_trunc=0.04)

a = o3d.io.read_triangle_mesh(str(old.mesh_path))
b = o3d.io.read_triangle_mesh(str(new))
assert np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices)), "vertex positions differ"
assert np.array_equal(np.asarray(a.triangles), np.asarray(b.triangles)), "triangles differ"
assert np.array_equal(np.asarray(a.vertex_colors), np.asarray(b.vertex_colors)), "colours differ"
print(f"SYNTHETIC FUSION PARITY PASS: {len(a.vertices)} vertices, {len(a.triangles)} triangles")
PY
```

Expected: a `PROOF:` line inside `.worktrees/clean-mesh`, then `SYNTHETIC FUSION PARITY PASS:
<N> vertices, <M> triangles`. Record N and M in the commit message. uint8 colour goes to both
sides on purpose: the old class passes uint8 through untouched but rescales float, so a float
input would compare a rounding difference rather than the integration path.

**If this fails, stop.** Do not delete anything. The divergence is in Task 3's `fuse_tsdf` and
must be fixed there first.

- [x] **Step 2: Delete the three retired modules and six obsolete test files**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git rm collab_splats/mesh/base.py collab_splats/mesh/poisson.py collab_splats/mesh/utils.py tests/mesh/test_utils.py tests/mesh/test_utils_ground_plane.py tests/mesh/test_splats_adapter.py tests/mesh/test_registry.py tests/mesh/test_feature_transfer.py tests/mesh/test_adapter.py
```

What each deleted test covered, and where that coverage now lives:

| Deleted | Covered | Now covered by |
| --- | --- | --- |
| `test_utils.py` | `clean_repair_mesh`, `features2vertex`, `mesh_clustering`, `guided_upsample_depth` | Task 1 `tests/mesh/test_clean.py`, Task 2 `tests/mesh/test_features.py`, Task 4 `tests/mesh/test_io.py` |
| `test_utils_ground_plane.py` | ground-plane removal inside the old `clean_repair_mesh` | nothing — the ground-plane branch is deleted, not moved (see the spec's "removed behaviour") |
| `test_splats_adapter.py` | `_splats_to_tsdf_inputs` reading `splats.zarr` | Task 4 `test_render_tsdf_inputs_*` |
| `test_registry.py` | `get_mesh_creator` / `REGISTRY` | nothing — the registry is deleted; there is one mesher |
| `test_feature_transfer.py` | `transfer_features_to_mesh`, `persist_mesh_vertex_features` | nothing — both are deleted; the dashboard calls `features2vertex` directly (Task 9) |
| `test_adapter.py` | `pointcloud_to_mesh` / `mesh_from_tsdf_inputs` dispatch | nothing — callers compose arrays and call `fuse_tsdf` (Tasks 7, 9, 10, 11) |

- [x] **Step 3: Strip `Open3DTSDFFusion` out of `tsdf.py`**

Keep only the shrunken header and `fuse_tsdf`:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python - <<'PY'
from pathlib import Path

HEADER = '''from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses

logger = logging.getLogger(__name__)


'''

p = Path("collab_splats/mesh/tsdf.py")
src = p.read_text()
i = src.index("def fuse_tsdf(")
p.write_text(HEADER + src[i:])
print(f"tsdf.py is now {len(p.read_text().splitlines())} lines")
PY
```

Dropped from the header: `dataclass` (no class left), `collab_splats.mesh.base` and
`collab_splats.mesh.utils` (both deleted this task). `extract_intrinsics` and `invert_poses`
stay — `fuse_tsdf` uses both.

- [x] **Step 4: Write the final `collab_splats/mesh/__init__.py`**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && cat > collab_splats/mesh/__init__.py <<'PY'
from __future__ import annotations

from collab_splats.mesh.clean import clean_repair_mesh
from collab_splats.mesh.features import features2vertex, mesh_clustering
from collab_splats.mesh.texture import texture_mesh
from collab_splats.mesh.tsdf import fuse_tsdf

__all__ = [
    "clean_repair_mesh",
    "features2vertex",
    "fuse_tsdf",
    "mesh_clustering",
    "texture_mesh",
]
PY
```

Re-exports only — no registry, no factory, no `Path` import. `io.py` is deliberately absent
from `__all__`: `upsample_depths`, `render_tsdf_inputs` and `write_textured_ply` are input
plumbing for callers that already know which path they are on, and they are imported as
`from collab_splats.mesh.io import ...`.

Importing `collab_splats.mesh` now imports `texture.py`, which imports `warp` and
`meshoptimizer` at module level. That is intended — this repo takes hard imports and lets a
missing dependency raise at import time rather than at the end of a long run (see CLAUDE.md
"Hard imports — no stub backends").

- [x] **Step 5: Update the cu121 migration module list**

In `tests/test_cu121_migration.py`, replace lines 119-123 with:

```python
        "collab_splats.mesh",
        "collab_splats.mesh.io",
        "collab_splats.mesh.tsdf",
        "collab_splats.mesh.clean",
        "collab_splats.mesh.texture",
        "collab_splats.mesh.features",
```

- [x] **Step 6: Confirm nothing still imports the deleted names**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && grep -rn --include=*.py --include=*.ipynb --include=*.rst --include=*.md -e 'mesh\.utils' -e 'mesh\.base' -e 'mesh\.poisson' -e 'Open3DTSDFFusion' -e 'pointcloud_to_mesh' -e 'mesh_from_tsdf_inputs' -e 'get_mesh_creator' -e 'MeshResult' -e 'BaseMeshCreator' -e 'transfer_features_to_mesh' -e 'persist_mesh_vertex_features' . | grep -v '^\./docs/superpowers/' | grep -v '^\./\.git/'
```

Expected: no output at all (exit 1 from grep). `docs/superpowers/` is excluded because the
spec and this plan both name the retired symbols on purpose. Any other hit is a caller Tasks
6-11 missed — fix it before committing.

- [x] **Step 7: Run the mesh suite and the migration test**

Run: `PYTEST tests/mesh tests/test_cu121_migration.py`
Expected: all pass. `tests/mesh/` now holds exactly `__init__.py`, `test_clean.py`,
`test_features.py`, `test_io.py`, `test_texture.py`, `test_tsdf.py` — `test_absent_confidence.py`
moved to `tests/wrapper/` in Task 8.

- [x] **Step 8: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add -A collab_splats/mesh tests/mesh tests/test_cu121_migration.py && git commit --only collab_splats/mesh tests/mesh tests/test_cu121_migration.py -m "refactor(mesh)!: delete base, poisson, utils, the mesher registry and Open3DTSDFFusion

Fusion parity verified against the deleted class on synthetic views: identical
vertices, triangles and colours.

BREAKING CHANGE: pointcloud_to_mesh, mesh_from_tsdf_inputs, get_mesh_creator,
MeshResult, BaseMeshCreator, transfer_features_to_mesh and persist_mesh_vertex_features
are removed. Compose arrays and call fuse_tsdf / clean_repair_mesh instead.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 13: documentation

Six documents describe the mesh module and all six are now wrong. None of this is optional
polish: `configs/README.md` is the config contract, and `docs/known-test-failures.md` carries
three red tests this refactor is the owner of.

**Files:**
- Modify: `docs/source/api/mesh.rst`
- Modify: `docs/source/conf.py:42`
- Modify: `configs/README.md:368-373`
- Create: `docs/mesh.md`
- Modify: `docs/README.md:15-18`
- Modify: `CLAUDE.md:103`
- Modify: `docs/superpowers/CHANGELOG.md` (new entry above the 2026-09-05 one)
- Modify: `docs/known-test-failures.md:113-135`, `:397`

- [x] **Step 1: Rewrite the API page**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && cat > docs/source/api/mesh.rst <<'RST'
Mesh
====

TSDF meshing from depth + RGB arrays, plus cleaning, texturing and vertex-feature transfer.

.. automodule:: collab_splats.mesh.io
   :members:
   :show-inheritance:

.. automodule:: collab_splats.mesh.tsdf
   :members:
   :show-inheritance:

.. automodule:: collab_splats.mesh.clean
   :members:
   :show-inheritance:

.. automodule:: collab_splats.mesh.texture
   :members:
   :show-inheritance:

.. automodule:: collab_splats.mesh.features
   :members:
   :show-inheritance:
RST
```

- [x] **Step 2: Swap the mocked dependency in `conf.py`**

In `docs/source/conf.py`, replace line 42 (`    "meshlib",`) with (`nvdiffrast` included — it imports torch and CUDA at doc-build time otherwise):

```python
    "warp",
    "meshoptimizer",
    "nvdiffrast",
```

Autodoc runs without CUDA in CI, and `texture.py` imports both at module level. `plyfile` is
NOT mocked — it is pure Python with no build step, so autodoc can import it for real.

- [x] **Step 3: Rewrite the config table rows**

In `configs/README.md`, replace lines 368-373 (the six `mesh.*` rows, `mesh.enabled` through
`mesh.splat_depth`) with:

```markdown
| `mesh.enabled` | bool | `true` | Fuse a TSDF mesh after the pointcloud stage, writing `mesh/mesh.ply` |
| `mesh.source` | str | `feedforward` | `feedforward` fuses `pointcloud.zarr` depth lifted onto the original frames; `splats` fuses depth and colour rendered from the splats stage's `ckpt.pt` (needs the splats stage, which is never auto-run) |
| `mesh.voxel_size` | float | `0.0025` | TSDF voxel edge, world units. `sdf_trunc` is derived as `4 × voxel_size` and is not separately settable — halving this buys finer geometry for roughly 8× the memory |
| `mesh.depth_trunc` | float | `1.5` | Ignore depth beyond this, world units. Feedforward depth is not metric, so this is in the reconstruction's own scale, not metres |
| `mesh.conf_percentile` | float\|null | `20` | Drop depth below this global confidence percentile before fusing (`null` = off). `source: feedforward` only; a reconstruction that carries no confidence (sfm) fuses unmasked and logs that it did |
| `mesh.texture` | bool | `false` | Also decimate, UV-unwrap and project the fused views into `mesh/texture/` (`albedo.png` beside a UV-carrying `mesh.ply`). Needs a GPU |
```

Six rows replace six rows, but they are not the same six: `mesh.mesher`, `mesh.sdf_trunc` and
`mesh.splat_depth` are gone, and `mesh.depth_trunc`, `mesh.conf_percentile` and `mesh.texture`
are new to the table. If `configs/README.md` documents `native_resolution` or
`color_map_iterations` anywhere else in the file, delete those mentions too:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && grep -n -e native_resolution -e color_map_iterations -e splat_max_depth -e 'mesh\.mesher' -e 'mesh\.sdf_trunc' -e clean_repair configs/README.md
```

Expected after the edit: no output.

- [x] **Step 4: Write `docs/mesh.md`**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && cat > docs/mesh.md <<'MD'
# Mesh Module

`collab_splats.mesh` turns posed depth + RGB into a triangle mesh. Every entry point takes
**plain arrays**, never a `FeedforwardResult` or a `PointcloudResult` — the caller composes
them, because only the caller knows which resolution grid it is on.

| File | Responsibility |
| --- | --- |
| `io.py` | Getting arrays in and out: `upsample_depths`, `render_tsdf_inputs`, `write_textured_ply` |
| `tsdf.py` | `fuse_tsdf` — integrate views into a TSDF volume, write `mesh.ply` |
| `clean.py` | `get_scene_scale`, `remove_floaters`, `fill_holes`, `clean_repair_mesh` |
| `texture.py` | `decimate_mesh`, `unwrap_mesh_uvs`, `project_images_to_texture`, `texture_mesh` |
| `features.py` | `features2vertex`, `mesh_clustering` |

---

## Quickstart

The pipeline runs this stage for you (`mesh:` in the yaml, `--stages mesh`). Direct use:

```python
from pathlib import Path

from collab_splats.mesh import clean_repair_mesh, fuse_tsdf

mesh_path = fuse_tsdf(
    depths,       # (n, h, w) float32, 0 = no observation
    rgbs,         # (n, h, w, 3) uint8
    c2w,          # (n, 4, 4) camera-to-world
    K,            # (n, 3, 3) at the depth resolution
    Path("scene/mesh"),
    voxel_size=0.0025,
    depth_trunc=1.5,
)
clean_repair_mesh(mesh_path)   # rewrites mesh.ply in place
```

`voxel_size` and `depth_trunc` are in **world units, and the world has no fixed scale**. The
values above are `base.yaml`'s, tuned for feedforward backbones whose depth is normalised to
roughly unit scale. A COLMAP-scale reconstruction (`pointcloud.method: sfm`) is typically one
to two orders of magnitude larger — on GH010229 the median depth is 16.5 and the camera
trajectory spans 145, so `depth_trunc=1.5` truncates every sample and fusion returns an empty
mesh with only an `[Open3D WARNING] Write PLY failed: mesh has 0 vertices.` on stderr. Measure
first: `np.percentile(depths[depths > 0], [50, 95])`, put `depth_trunc` near p95, and scale
`voxel_size` by the same ratio.

`fuse_tsdf` raises rather than fusing quietly wrong: float RGB is rejected outright, and a
principal point outside the depth grid raises, because pairing one grid's depth with the other
grid's `K` collapses the mesh instead of failing (the 2026-08-11 regression). An out-of-scale
`depth_trunc` is NOT in that set — Open3D treats it as a legitimately empty volume.

---

## Composing the inputs

**From a feedforward reconstruction, at frame resolution.** Model-resolution depth is
guided-upsampled into the original frames so the COLMAP camera is the right `K` to fuse with:

```python
import numpy as np

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh.io import upsample_depths
from collab_splats.preproc import frames
from collab_splats.pointcloud.utils import confidence_mask

depth = np.asarray(ff.depth)
depth = np.where(confidence_mask(np.asarray(ff.confidence), 20), depth, 0.0)
rgbs = frames.read_frames(images_dir)
depths = upsample_depths(depth, rgbs, np.asarray(ff.original_coords)[:, :4])
c2w = invert_poses(result.extrinsics)
```

**From a trained splat.** Renders come out at frame resolution carrying the poses they were
rendered with, pose-opt deltas included — nothing to lift, nothing to re-pose:

```python
from collab_splats.mesh.io import render_tsdf_inputs

depths, rgbs, c2w, K = render_tsdf_inputs(Path("scene/splats/ckpt.pt"))
```

**At model resolution** (notebooks, evals): use `ff.depth`, `ff.images` transposed to
`(n, h, w, 3)` and scaled to uint8, `invert_poses(ff.extrinsics)` and `ff.intrinsics`. No lift,
no COLMAP camera.

---

## Cleaning

`clean_repair_mesh(mesh_path)` rewrites `mesh.ply` in place: drop floating components, then
fill small holes. **Every threshold is a fraction of the mesh's own extent**, never a world
distance, so one default works on a metric scan and on a scale-free feedforward reconstruction
alike. `get_scene_scale` is that extent: the 1st-to-99th-percentile diagonal of the vertex
cloud, which ignores the stray far component that would otherwise set the scale.

- `min_area_frac` (`6e-6`) — components smaller than this fraction of total surface area go
- `max_gap_frac` (`0.01`) — a component further than this fraction of scene scale from the main
  body goes, however large it is
- `max_hole_frac` (`0.0045`) — holes whose boundary is shorter than this fraction of scene
  scale are filled; anything bigger is a real opening (a doorway, the missing back of the
  scene) and is left alone

Cleaning is not optional in the pipeline and has no config key. `remove_floaters` and
`fill_holes` are public for callers who want one without the other.

---

## Texturing

`texture.py` bakes per-view colour into a single albedo atlas, opt-in via `mesh.texture: true`:

1. `decimate_mesh` — `meshoptimizer` simplification to an error bound expressed in voxels, so
   the budget follows the fusion resolution rather than a triangle count
2. `_make_manifold` — split non-manifold vertices, drop degenerate, duplicate and fold-over
   faces. UVAtlas rejects a mesh that fails any of these
3. `unwrap_mesh_uvs` — Open3D UVAtlas, partitioned for parallelism
4. `project_images_to_texture` — an NVIDIA Warp kernel per texel: ray-cast for occlusion
   (`wp.mesh_query_ray`), accumulate cosine-weighted bilinear samples from every view that sees
   it, then dilate into the gutter
5. `write_textured_ply` — Open3D's PLY writer cannot emit UVs, so this writes the PLY with
   `plyfile` and a `comment TextureFile albedo.png` line

Output is `mesh/texture/mesh.ply` + `mesh/texture/albedo.png`.

---

## Vertex features

`features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03)` transfers per-point
semantic features onto mesh vertices by inverse-distance-weighted k-NN, zeroing vertices with
no neighbour inside `sdf_trunc`. `mesh_clustering` then groups high-scoring vertices into
spatially connected clusters. Both are used by the dashboard's semantic query view and by
`06_mesh/splats_mesh.ipynb`.
MD
```

Then add the module to `docs/README.md`, replacing lines 15-18 with:

```markdown
## Module Notebooks

- [pointcloud/](pointcloud/) — pointcloud + bundle adjustment + ground-truth evals
- [semantics/](semantics/) — feature extraction, MaskCLIP, Talk2DINO
- [mesh.md](mesh.md) — TSDF fusion, cleaning, texturing, vertex features
- [splats.md](splats.md) — Gaussian-splat training on upstream gsplat
```

`splats.md` is listed too — it exists on disk and was never linked.

- [x] **Step 5: Fix the architecture line in `CLAUDE.md`**

Replace `CLAUDE.md:103` with:

```
  mesh/                    # TSDF meshing from arrays: io, tsdf, clean, texture, features
```

- [x] **Step 6: Add the CHANGELOG entry**

Insert above the existing `Recently completed (2026-09-05): **preproc-centralization**` line
in `docs/superpowers/CHANGELOG.md`:

```markdown
Recently completed (2026-09-06): **mesh-cleanup** — `collab_splats/mesh` rebuilt around one rule: **the module takes arrays, the caller composes them** ([spec](specs/2026-09-06-mesh-cleanup-design.md) · [plan](plans/2026-09-06-mesh-cleanup.md)).

`base.py`, `poisson.py`, `utils.py`, the mesher registry and the `Open3DTSDFFusion` dataclass are deleted. Five files replace them: `io.py` (`upsample_depths`, `render_tsdf_inputs`, `write_textured_ply`), `tsdf.py` (`fuse_tsdf`), `clean.py` (`get_scene_scale`, `remove_floaters`, `fill_holes`, `clean_repair_mesh`), `texture.py` (`decimate_mesh`, `unwrap_mesh_uvs`, `project_images_to_texture`, `texture_mesh`) and `features.py` (`features2vertex`, `mesh_clustering`). Fusion parity with the deleted class was verified on synthetic views: identical vertices, triangles and colours.

**Config: 14 keys to 6.** `mesh:` is now `enabled, source, voxel_size, depth_trunc, conf_percentile, texture`. `sdf_trunc` is derived (`4 × voxel_size`); `clean_repair` and `native_resolution` are always on; `mesher`, `color_map_iterations`, `splat_depth`, `splat_max_depth_frac` and `splat_max_depth_grad` are gone — the depth-cut knobs were measured no-ops that `clean_repair` already covered.

**`mesh.source: splats` reads `ckpt.pt`, not `splats.zarr`.** Rendering from the checkpoint gets frame-resolution depth and colour with the poses actually trained (pose-opt deltas included), which makes `splats.zarr` a non-input for meshing.

**meshlib is gone.** Hole filling is Open3D's tensor `fill_holes`, sized proportionally (`0.0045 × scene scale`) like every other cleaning threshold. **Texturing is new:** `meshoptimizer` decimation, Open3D UVAtlas unwrap, and an NVIDIA Warp kernel that ray-casts occlusion per texel and accumulates cosine-weighted samples from every view. Open3D's PLY writer cannot emit UVs, so `write_textured_ply` writes them with `plyfile` plus a `TextureFile` comment.
```

- [x] **Step 7: Close out the two known-test-failures this refactor owns**

In `docs/known-test-failures.md`, retitle the 2026-08-21 entry and add a resolution note.
Replace line 113 with:

```markdown
## 2026-08-21 — 2 of 3 reconstructor/base.yaml failures RESOLVED 2026-09-06
```

and insert immediately after line 135 (`not a value edit. Owed to the mesh owner.`):

```markdown
**2026-09-06 resolution.** The mesh owner took the two that are mesh's.
`test_mesh_clean_repair_defaults_off` is
deleted — cleaning is unconditional, so there is no default to assert.
`test_base_yaml_mesh_has_fidelity_keys` now asserts the exact six-key set the shipped block
carries. `test_init_fills_defaults_from_base_yaml` is preproc's (`fps: 2.0`), still red, and is
untouched here — the entry stays open for it.
```

Then fix the moved path at line 397:

```markdown
- `tests/wrapper/test_absent_confidence.py::test_splats_depth_targets_skip_masking_when_confidence_absent`
```

The file moved from `tests/mesh/` to `tests/wrapper/` in Task 8; the failure itself is the
gsplat-import one and is unrelated to this refactor.

- [x] **Step 8: Verify the docs build and the config table matches the shipped yaml**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python - <<'PY'
import re
from pathlib import Path

import yaml

# Every mesh key in base.yaml has a row in configs/README.md, and vice versa
keys = set(yaml.safe_load(Path("configs/base.yaml").read_text())["mesh"])
rows = set(re.findall(r"^\| `mesh\.([a-z_]+)`", Path("configs/README.md").read_text(), re.M))
assert keys == rows, f"yaml {sorted(keys)} != README {sorted(rows)}"
print(f"config table matches: {sorted(keys)}")
PY
```

Expected: `config table matches: ['conf_percentile', 'depth_trunc', 'enabled', 'source',
'texture', 'voxel_size']`.

- [x] **Step 9: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add -f docs CLAUDE.md configs/README.md && git commit --only docs CLAUDE.md configs/README.md -m "docs(mesh): document the array API, the six-key config and the texture path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

`git add -f` is required: `docs/superpowers/` is gitignored in this repo, and the CHANGELOG
lives under it.

---
### Task 14: dependencies

`meshlib` was pulled in for one thing — hole filling — and Task 1 replaced it with Open3D's
tensor `fill_holes`. It goes. `warp-lang`, `meshoptimizer` and `nvdiffrast` come in for the
texture path, and `plyfile` moves from the `feedforward` extra into the core dependencies because
`mesh/io.py` needs it and nothing in `feedforward` ever did.

**`nvdiffrast` publishes no wheel and no PyPI release.** It installs from a pinned git commit and
builds its CUDA extension at first use, so it needs a real `nvcc` on `PATH` — the pip
`nvidia-cuda-nvcc-cu12` wheels are ptxas-only and will not do. The apt `cuda-nvcc-12-1` that
`setup.sh` already installs for gsplat is what satisfies it (verified: `nvcc 12.1.105` against
`torch 2.5.1+cu121`, context built in 0.4 s, rasterizes 8192² at 2.48 GB peak on the A40).
Install form that works in this uv-managed venv, which has no `pip` binary:

```bash
uv pip install --python /opt/venv/reconstruction/bin/python --no-deps --no-build-isolation \
  "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae"
```

**⚠️ `uv.lock` is shared with the main checkout and with every other `clean/*` worktree, and
another session has it modified.** Step 4 needs the user's go-ahead before it runs.

**Files:**
- Modify: `pyproject.toml:50` (drop meshlib), `:56` area (add deps), `:118` (drop plyfile from the extra), `:211` (addopts)
- Modify: `tests/test_cu121_migration.py:159`
- Modify: `uv.lock` (regenerated)

- [x] **Step 1: Edit `pyproject.toml`**

Delete line 50 (`    "meshlib>=3.1",`) and add four entries in its place:

```toml
    "plyfile",
    # Texture bake: nvdiffrast rasterizes the UV atlas, Warp ray-casts occlusion and runs the
    # per-texel projection kernel, meshoptimizer decimates.
    # meshoptimizer's only published wheel is a pre-release; the explicit ==a0 pin is what
    # makes uv accept it without a global --prerelease flag.
    # nvdiffrast ships no wheel; it builds its CUDA extension at first use and so needs a real
    # nvcc on PATH (apt cuda-nvcc-12-1, installed by setup.sh — the pip wheels are ptxas-only).
    "warp-lang==1.14.0",
    "meshoptimizer==0.2.30a0",
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae",
```

Delete line 118 (`    "plyfile",`) from the `feedforward` extra — it moved to core.

Replace line 211 with:

```toml
addopts = "--import-mode=importlib"
```

`tests/test_meshlib.py` does not exist and has not for some time; the `--ignore` was dead
weight pointing at the dependency this task removes.

- [x] **Step 2: Drop the meshlib import probe from the migration test**

In `tests/test_cu121_migration.py`, delete line 159:

```python
        "meshlib": "import meshlib.mrmeshpy as mrmeshpy",
```

`test_flagged_package_imports` checks numpy-2.x compatibility of flagged packages. meshlib is
no longer a dependency, so probing it would assert on a package that happens to still be in
the shared venv.

- [x] **Step 3: Confirm no source, test or doc still names meshlib**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && grep -rn --include=*.py --include=*.toml --include=*.md --include=*.sh --include=*.yaml --include=*.rst meshlib . | grep -v '^\./docs/superpowers/' | grep -v '^\./uv\.lock'
```

Expected: no output.

- [ ] **Step 4: Regenerate the lock — ASK THE USER FIRST**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python -m uv lock
```

Expected: uv resolves and rewrites `uv.lock`, reporting `meshlib` removed and `warp-lang` +
`meshoptimizer` added.

**Do not run `uv sync`.** A plain sync prunes every package `setup.sh` installed with
`--no-deps` (instantsfm, pyceres, VDA, vggt, mapanything, gsplat) and takes the whole
reconstruction environment down with it — see the `feedforward` extra's own comment at
`pyproject.toml:107-109`. The venv already has `warp-lang==1.14.0`, and `meshoptimizer` is
installed by Task 5 Step 1; `uv lock` here only records what the environment already runs.

- [x] **Step 5: Verify the environment still imports what the module needs**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -c "
import collab_splats; print('PROOF:', collab_splats.__file__)
import collab_splats.mesh as m; print('mesh exports:', sorted(m.__all__))
import warp, meshoptimizer, plyfile, open3d
print('warp', warp.config.version, '| open3d', open3d.__version__)
"
```

Expected: the `PROOF:` line points inside `.worktrees/clean-mesh`, then
`mesh exports: ['clean_repair_mesh', 'features2vertex', 'fuse_tsdf', 'mesh_clustering',
'texture_mesh']`, then `warp 1.14.0 | open3d 0.19.0`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && git add pyproject.toml uv.lock tests/test_cu121_migration.py && git commit --only pyproject.toml uv.lock tests/test_cu121_migration.py -m "build(deps): drop meshlib, add warp-lang and meshoptimizer, move plyfile to core

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

If `uv.lock` still carries another session's unrelated changes, commit `pyproject.toml` and
`tests/test_cu121_migration.py` alone and leave the lock to whoever owns it — `git commit
--only <paths>` exists exactly for this repo's shared-index situation.

---
### Task 15: verification gates

The per-task tests prove each function does what it says. These four gates prove the *module*
still produces the same mesh from real data. They run on GH010229:

```
/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f/instantsfm/
  images/            500 frames
  pointcloud.zarr/   depth, extrinsics, intrinsics, original_coords (no confidence — instantsfm)
  splats_scaffold2dgs_500f_50k/ckpt.pt
```

This scene is sfm-derived and carries **no** `confidence` array, so `conf_percentile` is a
no-op on both sides of every comparison — the gates measure geometry, not masking. Its
`images/` are `.jpg` (it predates the `.png` store); `frames.read_frames` reads them fine.

**Step 1 must run before Task 1.** Gates 1 and 2 compare against the code Task 12 deletes.

**Files:** none — this task only reads and measures.

- [x] **Step 1: Capture the pre-refactor baseline — RUN THIS BEFORE TASK 1**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && B=/tmp/claude-0/-workspace-collab-splats/mesh-baseline && git worktree add --detach "$B" HEAD && for d in LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc vggt-omega vggt_spark xfeat; do ln -sfn "/workspace/collab-splats/third_party/$d" "$B/third_party/$d"; done && ln -sfn /workspace/collab-splats/third_party/.vda_fetch_done "$B/third_party/.vda_fetch_done" && echo BASELINE_AT=$(git -C "$B" rev-parse HEAD) && ls "$B/third_party/"
```

The `third_party` links are not optional: the vendored clones are gitignored, so a fresh
worktree has none, and the guarded tests **silently SKIP instead of failing** — a control run
without them reports a better failure set than the real one and invalidates gate 4.

Link the clones **individually**, not the parent. `third_party/` itself is a tracked directory
(it holds a committed `README.md`), so it already exists in the new worktree and
`ln -s .../third_party "$B/third_party"` drops the link *inside* it instead of replacing it —
silently, with the same zero exit code. The `ls` at the end is the check: nine symlinks plus a
real `README.md`.

Then fuse the reference mesh with the old code:

```bash
cd /tmp/claude-0/-workspace-collab-splats/mesh-baseline && PYTHONPATH=/tmp/claude-0/-workspace-collab-splats/mesh-baseline /opt/venv/reconstruction/bin/python -u - <<'PY'
from pathlib import Path

import numpy as np

import collab_splats
print("PROOF:", collab_splats.__file__)
from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs, mesh_from_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

SCENE = Path("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f/instantsfm")
OUT = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates/baseline")

# Model resolution, unmasked, no cleaning — the pure integration loop is what gate 1 measures
ff = FeedforwardResult.load_zarr(SCENE / "pointcloud.zarr", load_images=True)
depths, rgbs, c2w, K = _feedforward_to_tsdf_inputs(ff, conf_percentile=None)

# The adapter returns float RGB in [0, 1] and Open3DTSDFFusion truncates it per view with
# (x * 255).astype(uint8). Do that once here instead: the new fuse_tsdf REJECTS float RGB,
# so this is the only way both sides integrate the same bytes and colours stay comparable.
rgbs = np.ascontiguousarray((rgbs * 255).astype(np.uint8))

res = mesh_from_tsdf_inputs(
    depths, rgbs, c2w, K, OUT,
    method="open3d_tsdf", color_map_iterations=0,
    voxel_size=0.2, sdf_trunc=0.8, depth_trunc=100.0, clean_repair=False,
)
print("BASELINE MESH:", res.mesh_path)
PY
```

Expected: a `PROOF:` line inside `mesh-baseline`, the TSDF progress bar over 500 views at
roughly 4 it/s (~2 min), then `BASELINE MESH: .../mesh-gates/baseline/mesh.ply` — a 234 MB
file holding **5,368,265 vertices / 7,731,160 triangles**, bbox extent `[219, 126, 233]`.
Keep the worktree until Step 6.

**The fusion parameters are scene-scale, and getting them wrong fails silently.** This scene is
`pointcloud.method: sfm`, so its depth is in COLMAP world units — measured median 16.5, p95
79.4, camera trajectory span 145 — roughly 80× `base.yaml`'s feedforward-tuned defaults. At the
shipped `voxel_size=0.0025 / depth_trunc=1.5` every sample is past the truncation distance, and
Open3D **returns a mesh path anyway** after printing one line to stderr:

```
[Open3D WARNING] Write PLY failed: mesh has 0 vertices.
```

Gate 1a would then compare an empty baseline against an empty new mesh and pass. The tell is
the integration rate: a no-op run streams ~1090 it/s, a real one runs ~4 it/s. Always check the
vertex count above before trusting a gate result.

`sdf_trunc` still follows the `4 × voxel_size` rule the six-key config encodes; only the scale
changed.

- [x] **Step 2: Gate 1 — fusion parity on real data**

Two halves, because the shipped path is a lift followed by an integration, and each half has
its own reference.

**1a — the integration loop.** Same arrays into the old wrapper and the new function:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -u - <<'PY'
from pathlib import Path

import numpy as np
import open3d as o3d

import collab_splats
print("PROOF:", collab_splats.__file__)
from collab_splats.mesh.tsdf import fuse_tsdf
from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

SCENE = Path("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f/instantsfm")
GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")

# Recompose byte-for-byte what the old adapter built, then apply the same truncation
ff = FeedforwardResult.load_zarr(SCENE / "pointcloud.zarr", load_images=True)
depths = np.ascontiguousarray(ff.depth, dtype=np.float32)
rgbs = np.ascontiguousarray(np.asarray(ff.images).transpose(0, 2, 3, 1), dtype=np.float32)
rgbs = np.ascontiguousarray((rgbs * 255).astype(np.uint8))
c2w = invert_poses(ff.extrinsics).astype(np.float32)

new_path = fuse_tsdf(depths, rgbs, c2w, ff.intrinsics, GATES / "new", voxel_size=0.2, depth_trunc=100.0)

a = o3d.io.read_triangle_mesh(str(GATES / "baseline" / "mesh.ply"))
b = o3d.io.read_triangle_mesh(str(new_path))
va, vb = np.asarray(a.vertices), np.asarray(b.vertices)
print(f"baseline {len(va):,} verts / {len(a.triangles):,} tris")
print(f"new      {len(vb):,} verts / {len(b.triangles):,} tris")
assert va.shape == vb.shape, "vertex count differs"
assert np.abs(va - vb).max() < 1e-6, f"max vertex delta {np.abs(va - vb).max():.3g}"
assert np.array_equal(np.asarray(a.triangles), np.asarray(b.triangles)), "triangles differ"

# Both sides integrated identical uint8, so colour is an equality, not a tolerance
ca, cb = np.asarray(a.vertex_colors), np.asarray(b.vertex_colors)
assert ca.shape == cb.shape and np.abs(ca - cb).max() < 1e-9, "vertex colours differ"
print("GATE 1a PASS")
PY
```

**RESULT — gate 1a RUN AND PASSED 2026-09-06**, after Task 3 landed (`d03cb88e`) and before any
deletion. `baseline 5,368,265 verts / 7,731,160 tris` vs `new 5,368,265 verts / 7,731,160 tris`;
max vertex delta under `1e-6`, triangle index arrays `array_equal`, vertex colours equal to
`1e-9`. Integration ran at 2-12 it/s (the real-work range — not the ~1090 it/s no-op tell), so
the meshes compared are populated. `sdf_trunc` was left at `None`, which `fuse_tsdf` resolves to
`4 × voxel_size = 0.8` — exactly the baseline's explicit value.

The truncation lives in `Open3DTSDFFusion.create` (`(rgbs[i] * 255).astype(np.uint8)`, no
clip and no rounding), not in the adapter — the adapter hands back float in `[0, 1]`. Doing it
once in Step 1 rather than per view inside the volume is what makes uint8-only `fuse_tsdf`
comparable at all, and it is why 1a can assert colour equality instead of a tolerance.

TSDF geometry does not depend on colour, so a colour mismatch here means the arrays diverged
upstream — check `ff.images` dtype before touching the fusion code.

**1b — the native lift.** `upsample_depths` must reproduce the old per-frame
`guided_upsample_depth` loop bit for bit. Run in the baseline worktree first.

Both halves read the images with `cv2` directly rather than through `preproc.frames.read_frames`,
for two reasons. `preproc/frames.py` is not on this branch yet (see the precondition table), and
`read_frames(dir, idxs=...)` selects by **source frame index, not row position** — on this scene
the 500 filenames are `frame_000004 … frame_013111`, so `idxs=range(8)` raises `KeyError` and, if
it did not, would pair image `frame_000004` against whatever zarr row happens to be 4th. Zarr rows
are in filename order, so rows `0..7` pair with the first 8 sorted paths. Measured baseline:
`(8, 1078, 1918) float32`, every pixel non-zero, median depth 20.6.

```bash
cd /tmp/claude-0/-workspace-collab-splats/mesh-baseline && PYTHONPATH=/tmp/claude-0/-workspace-collab-splats/mesh-baseline /opt/venv/reconstruction/bin/python -u - <<'PY'
from pathlib import Path

import numpy as np

import cv2

from collab_splats.mesh.utils import guided_upsample_depth
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

SCENE = Path("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f/instantsfm")
GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")
GATES.mkdir(parents=True, exist_ok=True)

# 8 frames is enough — the loop is per-frame and independent. Rows 0..7 of the zarr pair with the
# first 8 images in FILENAME order; this scene's source frame_idx are 4, 24, 59, ... not 0..7.
ff = FeedforwardResult.load_zarr(SCENE / "pointcloud.zarr", load_images=False)
rows = list(range(8))
paths = sorted(p for p in (SCENE / "images").iterdir() if p.suffix.lower() == ".jpg")[:8]
rgbs = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])
out_hw = tuple(rgbs.shape[1:3])
depth = np.asarray(ff.depth)
lifted = np.stack([
    guided_upsample_depth(
        depth[i], rgbs[k], crop_box=tuple(ff.original_coords[i, :4]), out_hw=out_hw
    )
    for k, i in enumerate(rows)
])
np.save(GATES / "lift_baseline.npy", lifted)
print("saved", lifted.shape, lifted.dtype)
PY
```

Then compare from `clean/mesh`:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -u - <<'PY'
from pathlib import Path

import numpy as np

import cv2

from collab_splats.mesh.io import upsample_depths
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

SCENE = Path("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f/instantsfm")
GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")

ff = FeedforwardResult.load_zarr(SCENE / "pointcloud.zarr", load_images=False)
rows = list(range(8))
paths = sorted(p for p in (SCENE / "images").iterdir() if p.suffix.lower() == ".jpg")[:8]
rgbs = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])
new = upsample_depths(np.asarray(ff.depth)[rows], rgbs, np.asarray(ff.original_coords)[rows, :4])
old = np.load(GATES / "lift_baseline.npy")
delta = np.abs(new - old).max()
print(f"shapes {new.shape} vs {old.shape}, max |delta| = {delta:.3g}")
assert new.shape == old.shape and delta < 1e-6, "guided upsample diverged"
print("GATE 1b PASS")
PY
```

**RESULT — gate 1b RUN AND PASSED 2026-09-06.** `shapes (8, 1078, 1918) vs (8, 1078, 1918),
max |delta| = 0` — bit-identical, not merely under tolerance. `upsample_depths` is the old
per-frame `guided_upsample_depth` loop with the loop moved inside.

**The gate snippets import concrete modules, not the package.** Gate 1a takes `fuse_tsdf` from
`collab_splats.mesh.tsdf` and gate 2 takes `clean_repair_mesh` from `collab_splats.mesh.clean`,
because `mesh/__init__.py` still re-exports only the pre-cleanup surface
(`get_mesh_creator, pointcloud_to_mesh, BaseMeshCreator, MeshResult, Open3DTSDFFusion,
DepthNormalPoisson, GaussiansPoisson, REGISTRY`) and the new re-export block is Task 12 Step 5 —
which is blocked behind Tasks 7, 8 and the deferred half of 10. Written against the package,
every gate raises `ImportError: cannot import name ...`, and Task 10 Step 2 tells the reader to
diagnose that exact error as "Task 3 was not committed" — a wrong diagnosis. The submodule paths
are correct both before and after Task 12.

- [x] **Step 3: Gate 2 — cleaning parity, Open3D `fill_holes` vs meshlib**

Clean the *same* fused mesh with both implementations and compare what survives. Run the
baseline half first:

```bash
cd /tmp/claude-0/-workspace-collab-splats/mesh-baseline && PYTHONPATH=/tmp/claude-0/-workspace-collab-splats/mesh-baseline /opt/venv/reconstruction/bin/python -u - <<'PY'
import json
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from collab_splats.mesh.utils import clean_repair_mesh

GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")


def stats(path):
    # Boundary edges are the hole signal: an edge used by exactly one triangle
    m = o3d.io.read_triangle_mesh(str(path))
    tri = np.asarray(m.triangles)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    _, sizes, _ = m.cluster_connected_triangles()
    return {
        "vertices": len(m.vertices),
        "triangles": len(tri),
        "components": len(sizes),
        "boundary_edges": int((counts == 1).sum()),
    }


work = GATES / "clean_old" / "mesh.ply"
work.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(GATES / "baseline" / "mesh.ply", work)
before = stats(work)

# 0.014 is the OLD max_hole_frac: a fraction of BOUNDARY LENGTH. The new default 0.0045 is
# the same size expressed as a fraction of hole DIAMETER (0.014 / pi), which is the unit
# Open3D's fill_holes takes.
start = time.perf_counter()
clean_repair_mesh(work, min_area_frac=6e-6, max_gap_frac=0.01, max_hole_frac=0.014)
seconds = time.perf_counter() - start

out = {"before": before, "after": stats(work), "seconds": seconds}
(GATES / "clean_baseline.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
```

Then the new half:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -u - <<'PY'
import json
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from collab_splats.mesh.clean import clean_repair_mesh, fill_holes, remove_floaters

GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")

# Both implementations run the SAME floater code, measured identical on both trees:
# 547,768 verts / 1,016,318 tris / 1,477 components.
COMPONENTS_AFTER_FLOATERS = 1477


def stats(path):
    m = o3d.io.read_triangle_mesh(str(path))
    tri = np.asarray(m.triangles)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    _, sizes, _ = m.cluster_connected_triangles()
    return {
        "vertices": len(m.vertices),
        "triangles": len(tri),
        "components": len(sizes),
        "boundary_edges": int((counts == 1).sum()),
        "extent": [round(float(x), 4) for x in m.get_axis_aligned_bounding_box().get_extent()],
    }


work = GATES / "clean_new" / "mesh.ply"
work.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(GATES / "baseline" / "mesh.ply", work)
start = time.perf_counter()
clean_repair_mesh(work)
seconds = time.perf_counter() - start
new = {"after": stats(work), "seconds": seconds}
old = json.loads((GATES / "clean_baseline.json").read_text())

# Time the filler on its own. The shared floater step is ~63 s of either run, so an end-to-end
# wall-clock comparison measures machine load, not the two implementations.
m = o3d.io.read_triangle_mesh(str(GATES / "baseline" / "mesh.ply"))
remove_floaters(m)
t0 = time.perf_counter()
fill_holes(m)
fill_seconds = time.perf_counter() - t0

print(json.dumps({"old": old, "new": new, "fill_seconds": round(fill_seconds, 2)}, indent=2))

# 1. Floater removal is shared code and must land exactly where it lands on both sides.
#    meshlib's own post-clean count is 1476: its fill/subdivide/pack round-trip welds one
#    component pair. That is a meshlib artifact, not a rule difference — proven by running
#    floater removal alone on both trees and getting byte-identical counts.
assert new["after"]["components"] == COMPONENTS_AFTER_FLOATERS, (
    f"floater rule changed: {new['after']['components']} components, expected {COMPONENTS_AFTER_FLOATERS}"
)

# 2. One-sided: Open3D must not leave MORE holes open than meshlib did. Closing more is the
#    outcome we want, so a two-sided band fails on success.
old_residual = old["after"]["boundary_edges"]
new_residual = new["after"]["boundary_edges"]
print(f"boundary edges left open: meshlib {old_residual:,}, open3d {new_residual:,}")
assert new_residual <= old_residual, f"open3d left more holes open: {new_residual:,} vs {old_residual:,}"

# 3. Geometry sanity — the only assertion here that catches the o3d.t view-lifetime corruption.
#    Every topological count above is blind to it.
extent = new["after"]["extent"]
assert all(5.0 < x < 1000.0 for x in extent), f"implausible bbox extent {extent} — check fill_holes source lifetime"

# 4. The filler itself must not be the bottleneck.
assert fill_seconds <= 10.0, f"fill_holes too slow: {fill_seconds:.1f}s"
print("GATE 2 PASS")
PY
```

**RESULT — gate 2 RUN 2026-09-06. It FAILED as written, on all three asserts, and every
one of the three was the gate's fault, not the code's.** The meshlib → Open3D `fill_holes` swap
holds; the asserts did not measure it.

| | meshlib (old) | Open3D (new) |
|---|---|---|
| after floater removal | 547,768 v / 1,016,318 t / 1,477 c | **identical** |
| after full clean | 563,264 v / 1,066,659 t / 1,476 c | 547,768 v / 1,044,884 t / 1,477 c |
| boundary edges closed | 32,127 | **41,367** |
| boundary edges left open | 63,879 | **54,639** |
| filler wall clock | — | **1.45 s** |
| end-to-end wall clock | 53.1 s | 72.9 s |

**1. `components` — the gate said "floater rule changed". It did not change.** Running floater
removal *alone* on both trees gives byte-identical output (547,768 / 1,016,318 / 1,477), and the
`_GAP_KDTREE_POINTS` constant is 200,000 on both sides. The 1476 is meshlib's: its
`meshFromFacesVerts` → `fillHoles` → `subdivideMesh` → `pack` round-trip welds one component
pair. Open3D adds **zero** vertices — it triangulates existing boundary loops and nothing else —
so it cannot weld. The gate now asserts the shared floater result (1,477) directly, which is what
the assert was reaching for.

**2. `boundary_edges` — the ±5% band failed in the favourable direction.** Of the 96,006 boundary
edges present after floater removal, Open3D closes 41,367 and meshlib 32,127: the new filler
closes **28.8% more**. A two-sided band treats that as a regression. Now one-sided — Open3D may
not leave more holes open than meshlib did.

**3. `seconds` — it was measuring machine load.** Split timing: `read 1.7 s / floaters 63.0 s /
fill 1.7 s`. The floater step is *the same code on both sides* and is ~87% of either run, so the
53.1 s baseline (captured on an idle box) versus 72.9 s (captured with three subagents running)
compares schedulers. The filler itself is **1.45 s**. Now timed in isolation against a fixed
bound.

**The one real difference, and it is a quality tradeoff, not a defect:** meshlib subdivides and
smooths each patch to the local edge length (+15,496 vertices, +50,341 triangles); Open3D emits
flat caps over the existing boundary ring (+0 vertices, +28,566 triangles). Open3D's fills are
therefore coarser than the surrounding surface where a hole is large. It closes more of them,
36× faster, and drops a dependency — the swap stands.

Re-run with the corrected asserts: **GATE 2 PASS**, bbox extent `[116.7174, 56.8737, 158.4]`
(the corruption tripwire from Task 1).

**Measured baseline (2026-09-06), captured before Task 1:**

```json
{"before": {"vertices": 5368265, "triangles": 7731160, "components": 258275, "boundary_edges": 2497100},
 "after":  {"vertices":  563264, "triangles": 1066659, "components":   1476, "boundary_edges":   63879},
 "seconds": 53.14}
```

Cleaning drops 99.4% of the components and 89.5% of the vertices, so the mesh this gate compares
is dominated by floater removal, not by hole filling.

That is exactly why the comparison is on the **residual**. An earlier version of this gate divided
both sides by the same `before - after` total and asserted the ratio within ±5%. With these numbers
that assert is inert: the common denominator is 2,433,221 closed edges, so moving the ratio 5%
needs a 121,661-edge disagreement, while the entire post-clean residual is only 63,879. A filler
that closed **zero** holes would score 0.9737 and pass. Since the component assert already pins
floater removal exactly, the residual is the only quantity the two fillers can move, and ±5% of
63,879 (≈3,194 edges) is a real tolerance.

Open3D's `fill_holes` at TSDF hole counts is unmeasured — the design probe was a single sphere.
**This gate is where it gets measured, and it is allowed to fail.** If the residual lands outside
±5%, do not widen the tolerance: retune `max_hole_frac` until the residuals match, and record the
value that did it in the CHANGELOG entry.

- [x] **Step 4: Gate 3 — texture fidelity, absolute**

**This gate was rewritten three times, and the bar it started with was never reachable.**

The original asserted the two implementations agree with each other (`overlap >= 0.95`,
`PSNR >= 30`). They never can, and the reasons are worth knowing before touching this code:

- Open3D's `bake_vertex_attr_textures` defaults to `margin=2.0`, which extrapolates positions
  **outside every triangle**. Measured at 2048: it reports 0.891 of the atlas valid where the
  exact triangle coverage is 0.535, and 1,517,007 of those texels lie outside any triangle. That
  margin is a gutter, not geometry. At `margin=0.0` it drops to 0.526 and agrees with an exact
  rasterization to **p99 1e-5 world units**.
- Its visibility test is a depth buffer at **image** resolution. At 291x518 against a 332,855-face
  mesh it is badly undersampled, so it paints occluded texels; our BVH ray cast rejects them.
  On texels that do have geometry we reject 37% backfacing and 32% occluded, which is the
  correct answer for two near-identical views of an outdoor scene, not a bug.
- The two textures disagree with **each other** at 11-16 dB while both reproduce the source
  images at 23 dB. Almost all of the atlas is surface these two views never see, and that is
  where the disagreement lives.

**And parity with Open3D was the wrong bar regardless.** `project_images_to_albedo` is CPU-only
and OOMs past two views at 8192 (15.9 GB for two, ~24 s/view). It cannot run this pipeline at any
scale this repo uses, so it is not a reference implementation — it is the largest run that exists.
The gate reports it and asserts on it nowhere. The bar is absolute: render the atlas back through
one raycast of the same mesh and score it against the images it was built from.

**What this gate now actually gates: the nvdiffrast bake swap.** `bake_atlas_attributes` replaced
Open3D's `bake_vertex_attr_textures`, so it is checked against that function directly, with the
gutter turned off (`margin=0.0`). An orientation error is the failure mode that matters here —
it still fills the atlas, just with the wrong surface, and every downstream metric stays
plausible. Measured: v unflipped disagrees by **47.8 world units**, v flipped by **2e-5**.
nvdiffrast's first output row is the top of the atlas, so `v` flips.

Four further defects in the original snippet, each of which stops the script dead:

1. `decimate_max_error=0.25 * 0.0025` and `occlusion_eps=2 * 0.0025` are **base.yaml
   feedforward-scale constants**; this scene fuses at `voxel_size=0.2`. At the plan's bound
   decimation is a no-op — the same class of error that produced 0-vertex meshes in Wave 0.
   At `0.25 * 0.2` it is a real 3.1x reduction (1,044,884 -> 332,855 faces, error 0.0496).
2. `decimate_mesh` returns **`(mesh, error)`**, a tuple, which the snippet feeds straight into
   `unwrap_mesh_uvs`.
3. It omits the `_make_manifold` repair that `texture_mesh` performs between those two calls.
4. With `update_material=False`, `project_images_to_albedo` returns **the albedo Image**, not the
   mesh — `ref_tm.material.texture_maps["albedo"]` raises `AttributeError`.

`render_tsdf_inputs` is deferred (it needs a `splats.rendering.load_checkpoint` that exists on no
branch), so views come from `pointcloud.zarr`. Both sides get byte-identical inputs and the gate
measures projection, not the reader.

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -u - <<'PY'
"""
Gate 3: does the nvdiffrast + Warp texture pass reproduce the source images?

The bar is absolute, not parity with Open3D. Open3D's `project_images_to_albedo` cannot run
this pipeline at all — it is CPU-only and OOMs past two views at 8192 — so it is reported for
the record and asserted on nowhere. The texture is rendered back through one raycast and scored
against the images it was built from, which is what a texture is for.

The nvdiffrast atlas bake replaced Open3D's `bake_vertex_attr_textures`, so that swap is gated
directly: against Open3D's own bake at margin=0 (its default margin=2 is a gutter, not geometry).
"""
import time
from pathlib import Path

import numpy as np
import open3d as o3d

import collab_splats

print("PROOF:", collab_splats.__file__, flush=True)
from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh.texture import (
    _make_manifold,
    bake_atlas_attributes,
    decimate_mesh,
    project_images_to_texture,
    unwrap_mesh_uvs,
)
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

SCENE = Path("/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f/instantsfm")
GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")
TEX = 2048
VOXEL = 0.2

ff = FeedforwardResult.load_zarr(SCENE / "pointcloud.zarr", load_images=True)
rgbs = np.ascontiguousarray((np.asarray(ff.images).transpose(0, 2, 3, 1)[:2] * 255).astype(np.uint8))
c2w = invert_poses(ff.extrinsics).astype(np.float32)[:2]
K = np.asarray(ff.intrinsics)[:2]

mesh = o3d.io.read_triangle_mesh(str(GATES / "clean_new" / "mesh.ply"))
decimated, err = decimate_mesh(mesh, 0.25 * VOXEL)
tm = unwrap_mesh_uvs(_make_manifold(decimated), tex_size=TEX)

# The bake swap, gated against Open3D's bake with its gutter turned off
t0 = time.perf_counter()
nvdr_pos, nvdr_nrm = bake_atlas_attributes(tm, TEX)
bake_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
o3d_baked = tm.bake_vertex_attr_textures(TEX, {"positions"}, margin=0.0, fill=0.0, update_material=False)
o3d_bake_seconds = time.perf_counter() - t0
o3d_pos = o3d_baked["positions"].numpy()
nvdr_valid = np.linalg.norm(nvdr_nrm, axis=-1) > 0.5
both = nvdr_valid & (np.linalg.norm(o3d_pos, axis=-1) > 0)
bake_err = float(np.percentile(np.linalg.norm(nvdr_pos - o3d_pos, axis=-1)[both], 99))
print(
    f"bake  nvdr {bake_seconds:.3f}s cover {nvdr_valid.mean():.4f}   "
    f"open3d(margin=0) {o3d_bake_seconds:.3f}s   agree on {both.mean():.4f} of atlas, pos p99 {bake_err:.6f}",
    flush=True,
)

t0 = time.perf_counter()
ours = project_images_to_texture(tm, rgbs, c2w, K, tex_size=TEX, occlusion_eps=VOXEL)
ours_seconds = time.perf_counter() - t0

ref_tm = tm.clone()
ref_tm.material.set_default_properties()
t0 = time.perf_counter()
ref = np.asarray(
    ref_tm.project_images_to_albedo(
        [o3d.t.geometry.Image(np.ascontiguousarray(r)) for r in rgbs],
        [o3d.core.Tensor(k.astype(np.float64)) for k in K],
        [o3d.core.Tensor(np.linalg.inv(p).astype(np.float64)) for p in c2w],
        TEX,
        update_material=False,
    ).to_legacy()
)[..., :3]
ref_seconds = time.perf_counter() - t0

# Render both textures back through one raycast of the same mesh
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tm)
tri_uvs = tm.triangle.texture_uvs.numpy()


def sample(texture, tri_uv, flip_v):
    u = tri_uv[..., 0]
    v = 1.0 - tri_uv[..., 1] if flip_v else tri_uv[..., 1]
    col = np.clip((u * TEX).astype(np.int32), 0, TEX - 1)
    row = np.clip((v * TEX).astype(np.int32), 0, TEX - 1)
    return texture[row, col]


print(f"decimate {len(mesh.triangles):,} -> {len(tm.triangle.indices):,} faces (error {err:.4f})", flush=True)
# Wall clock varies 8x run to run on identical work (Warp module load, BVH build, JIT cache
# state), so this is a log line and never a claim.
print(f"seconds  ours {ours_seconds:.2f}  ref {ref_seconds:.2f}", flush=True)

scores = {}
for i in range(len(rgbs)):
    H, W = rgbs[i].shape[:2]
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K[i].astype(np.float64)),
        extrinsic_matrix=o3d.core.Tensor(np.linalg.inv(c2w[i]).astype(np.float64)),
        width_px=W,
        height_px=H,
    )
    ans = scene.cast_rays(rays)
    hit = np.isfinite(ans["t_hit"].numpy())
    pid = ans["primitive_ids"].numpy().copy()
    pid[~hit] = 0
    b = ans["primitive_uvs"].numpy()
    tri = tri_uvs[pid]
    tri_uv = tri[:, :, 0] * (1 - b[..., :1] - b[..., 1:2]) + tri[:, :, 1] * b[..., :1] + tri[:, :, 2] * b[..., 1:2]

    for name, texture in (("ours", ours), ("ref", ref)):
        # Resolve each texture's own v convention rather than assuming they share one
        best = None
        for flip in (False, True):
            got = sample(texture, tri_uv, flip)
            covered = hit & (got.sum(2) > 0)
            if covered.sum() < 1000:
                continue
            mse = ((got[covered].astype(np.float64) - rgbs[i][covered].astype(np.float64)) ** 2).mean()
            cand = (10 * np.log10(255.0**2 / max(mse, 1e-9)), covered.sum() / max(hit.sum(), 1), flip)
            if best is None or cand[0] > best[0]:
                best = cand
        scores.setdefault(name, []).append(best)
        print(f"view {i} {name:4s}: PSNR vs source {best[0]:6.2f} dB   covers {best[1]:.3f} of hit pixels   flip_v={best[2]}", flush=True)

ours_psnr = float(np.mean([s[0] for s in scores["ours"]]))
ref_psnr = float(np.mean([s[0] for s in scores["ref"]]))
ours_cov = float(np.mean([s[1] for s in scores["ours"]]))
ref_cov = float(np.mean([s[1] for s in scores["ref"]]))
print(f"\nmean over views: ours {ours_psnr:.2f} dB @ {ours_cov:.3f} coverage   ref {ref_psnr:.2f} dB @ {ref_cov:.3f}", flush=True)

# The nvdiffrast bake must reproduce the reference bake's geometry, not merely look plausible:
# an orientation error still fills the atlas, it just fills it with the wrong surface (measured
# 47.8 world units when v is not flipped, 2e-5 when it is).
assert bake_err < 1e-3, f"nvdiffrast bake disagrees with Open3D's: p99 {bake_err:.6f} world units"

# Absolute bars, not parity. Measured on this scene at TEX=2048: 23.03 dB @ 0.997 coverage, so
# these sit roughly 3 dB and 5 points below what the pipeline actually delivers.
assert ours_psnr >= 20.0, f"texture PSNR against source images fell to {ours_psnr:.2f} dB"
assert ours_cov >= 0.95, f"texture covers only {ours_cov:.3f} of visible pixels"
extent = tm.to_legacy().get_axis_aligned_bounding_box().get_extent()
assert all(5.0 < x < 1000.0 for x in extent), f"mesh extent {extent} — o3d.t view corruption"
print("GATE 3 PASS", flush=True)
PY
```

**MEASURED 2026-09-06 — GATE 3 PASS, with the nvdiffrast bake in place.**

```
bake  nvdr 0.095s cover 0.5307   open3d(margin=0) 1.385s   agree on 0.5257 of atlas, pos p99 0.000011
decimate 1,044,884 -> 332,855 faces (error 0.0496)
view 0 ours: PSNR vs source  23.42 dB   covers 0.996 of hit pixels   flip_v=True
view 0 ref : PSNR vs source  23.36 dB   covers 0.973 of hit pixels   flip_v=True
view 1 ours: PSNR vs source  22.63 dB   covers 0.997 of hit pixels   flip_v=True
view 1 ref : PSNR vs source  22.57 dB   covers 0.977 of hit pixels   flip_v=True
mean over views: ours 23.02 dB @ 0.996 coverage   ref 22.97 dB @ 0.975
```

The bake swap is **output-neutral and correctness-positive**: 23.02 dB @ 0.996 against 23.03 dB
@ 0.997 with the Open3D bake, a difference inside UVAtlas's own run-to-run drift (measured 0.6%
in the filled-texel count, 2,274,987 vs 2,288,435 on two runs of one script). It changed which
code produces the per-texel geometry, not what the texture looks like — which is the point: the
Open3D bake was the last CPU stage in the projection path, at 1.385 s against 0.095 s here, and
it scales with texel count, so at 8192 it is 16x more work on the wrong processor.

The 0.001 of coverage given up is the `margin=2.0` gutter that no longer comes free from the
bake. `_dilate_texels(gutter_px=4)` after projection already serves that purpose, and doing it
there is better: the margin band had **real interpolated positions extrapolated outside the
triangle**, so the Warp kernel projected photograph colour onto geometry that does not exist.
Now those texels are invalid and get filled from their filled neighbours instead.

**Do not quote a speedup for the projection.** Wall clock for our projection measured 1.72 s,
2.24 s, 13.61 s and 16.06 s across four runs of identical work — it carries Warp module load and
BVH build, and the 13.61 s run made Open3D look 3x faster. The bake numbers above are separate
and do isolate one call. The case for this stack is memory and scale, not milliseconds.

**Coverage claims must set `gutter_px=0`.** At the production `gutter_px=4` our filled-texel
count is 0.546 of the atlas against a true core of **0.170** — the 4-texel dilation inflates it
3x, and comparing a dilated mask against Open3D's undilated one reads as a 1.4x coverage win
that does not exist.

- [x] **Step 5: Gate 4 — the suite, diffed against a control**

Control first, from the baseline worktree, in the background (it takes minutes, and a
foreground call that exceeds the Bash timeout is backgrounded anyway — without the log):

```bash
cd /tmp/claude-0/-workspace-collab-splats/mesh-baseline && PYTHONPATH=/tmp/claude-0/-workspace-collab-splats/mesh-baseline /opt/venv/reconstruction/bin/python -u -m pytest tests/mesh tests/wrapper tests/dashboard -q > /tmp/claude-0/-workspace-collab-splats/mesh-gates/control.txt 2>&1; echo "PYTEST_RC=$?" >> /tmp/claude-0/-workspace-collab-splats/mesh-gates/control.txt
```

Run with `run_in_background: true`. The trailing `echo` is required: a background task's
notification reports the exit code of the **last** command, not pytest's. Poll by watching
`control.txt` grow — `ps` and `stat` are both unreliable on this filesystem, and an
`until ! pgrep -f "pytest ..."` loop matches its own polling shell and hangs forever.

Then the same three directories from `clean/mesh`:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && PYTHONPATH=/workspace/collab-splats/.worktrees/clean-mesh /opt/venv/reconstruction/bin/python -u -m pytest tests/mesh tests/wrapper tests/dashboard -q > /tmp/claude-0/-workspace-collab-splats/mesh-gates/after.txt 2>&1; echo "PYTEST_RC=$?" >> /tmp/claude-0/-workspace-collab-splats/mesh-gates/after.txt
```

Compare:

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && /opt/venv/reconstruction/bin/python - <<'PY'
import re
from pathlib import Path

GATES = Path("/tmp/claude-0/-workspace-collab-splats/mesh-gates")


def summary(path):
    text = path.read_text()
    names = set(re.findall(r"^(FAILED|ERROR) (\S+)", text, re.M))
    tail = [l for l in text.splitlines() if " passed" in l or " failed" in l or "PYTEST_RC" in l]
    return names, tail


ctrl, ctrl_tail = summary(GATES / "control.txt")
after, after_tail = summary(GATES / "after.txt")
print("control:", *ctrl_tail, sep="\n  ")
print("after:  ", *after_tail, sep="\n  ")
new_failures = {n for n in after if n[1] not in {c[1] for c in ctrl}}
print(f"\nNEW failures ({len(new_failures)}):")
for kind, name in sorted(new_failures):
    print(f"  {kind} {name}")
PY
```

The gate is **`NEW failures (0)`**, not a green suite: this branch inherits the environment's
own failures (gsplat 1.4.0 in the shared venv, and whatever the base commit carries). Also
compare SKIP counts in the two tail lines — a lower skip count in the control means the
baseline worktree lost its `third_party` symlink and the control is invalid.

Expected in `after.txt` and NOT in `control.txt`: nothing. Two of the three stale base.yaml
asserts (`test_mesh_clean_repair_defaults_off`, `test_base_yaml_mesh_has_fidelity_keys`) appear
in the control and are **gone** from the after run — Task 7 fixed them and Task 13 Step 7
recorded it. The third, `test_init_fills_defaults_from_base_yaml` (`preproc.fps`), is preproc's
and stays red in BOTH runs; a green one there means the control is not the right base commit.

**MEASURED 2026-09-06 — GATE 4 PASS.**

```
control (baseline worktree @ 2dd22904):  24 failed, 487 passed  in 61.42s
after   (clean/mesh):                    24 failed, 523 passed  in 209.14s
NEW failures (0)      FIXED (0)      skips 0 on both sides
```

The two failure sets are **identical, name for name** — 16 `tests/wrapper/test_vda_context.py`
`test_run_sfm_*`, 3 `tests/wrapper/test_splats_stage.py` mesh cases, and
`test_base_yaml_mesh_has_fidelity_keys` / `test_mesh_clean_repair_defaults_off` /
`test_init_fills_defaults_from_base_yaml`. All three base.yaml asserts are red in both runs,
which is the **corrected** expectation: the paragraph above predicts two of them disappear from
the after run because Task 7 fixed them, and Task 7 is deferred behind `clean/preproc`. When
Task 7 lands, those two must move to green — until then, red-in-both is the passing state.

The +36 passing tests are this branch's new tests; nothing regressed. Skip counts are **0 on
both sides**, so the baseline worktree kept its nine `third_party` symlinks and the control is
a real control (a silently higher skip count there is the failure mode this check exists for).

**RE-MEASURED 2026-09-07 after Tasks 7, 8, 10, 11 and 12 landed — GATE 4 PASS.**

The 2026-09-06 control is **void**: it was captured at `2dd22904`, before `clean/mesh` was
rebased onto `clean/final`. Sixteen of its twenty-four failures are
`tests/wrapper/test_vda_context.py` cases, and that file does not exist on the rebased tree —
diffing against it credits the refactor with sixteen fixes it never made. A control captured at
the wrong base is the same false green as no control at all. Rebuilt at the real merge-base,
`git merge-base clean/mesh clean/final` = `0f1da19e`, with the nine `third_party` symlinks plus
`.vda_fetch_done`:

```
control (fresh worktree @ 0f1da19e):   3 failed, 486 passed  in 39.66s
after   (clean/mesh @ b65dee17):       0 failed, 454 passed  in 49.23s
NEW failures (0)      FIXED (3)      skips 0 on both sides
```

The three FIXED are exactly this branch's inherited base.yaml debt —
`test_base_yaml_mesh_has_fidelity_keys` and `test_mesh_clean_repair_defaults_off` (Task 7's
six-key config; the second is deleted along with the `clean_repair` flag it asserted) and
`test_init_fills_defaults_from_base_yaml` (a stale `preproc.fps == 1.0` assert against
base.yaml's `2.0`, corrected in Task 7). The suite is now green end to end on these three
directories, which the 2026-09-06 note predicted as the passing state once Task 7 landed.

The count drops 489 -> 454 because Task 12 deletes six test files; every line of their coverage
either moved to `test_clean.py` / `test_features.py` / `test_io.py` or died with the behaviour it
tested, per Task 12 Step 2's table.

- [x] **Step 6: Remove the baseline worktree**

```bash
cd /workspace/collab-splats/.worktrees/clean-mesh && cd /workspace/collab-splats/.worktrees/clean-mesh && rm -f /tmp/claude-0/-workspace-collab-splats/mesh-baseline/third_party/* /tmp/claude-0/-workspace-collab-splats/mesh-baseline/third_party/.[!.]* && git worktree remove --force /tmp/claude-0/-workspace-collab-splats/mesh-baseline && git worktree list
```

**DONE 2026-09-06.** The `.[!.]*` term is a correction: `third_party/*` skips dotfiles, and the
tenth link in that directory is `.vda_fetch_done`. Verified afterwards that all nine vendored
clones under `/workspace/collab-splats/third_party/` still have their `.git` — deleting a symlink
never descends, but this is cheap to confirm and expensive to get wrong.

Remove the symlinks before the worktree, not after — `git worktree remove` follows into
them otherwise and would delete the real vendored clones. `rm -f .../third_party/*` unlinks the
symlinks themselves (no `-r`, so it cannot descend into a link target). Never remove a worktree while a run is still using it: the run dies
with `FileNotFoundError` on chdir, which does not look like a test failure.

---

## Branch coordination

Execution is blocked until `clean/mesh` is rebased onto the reorganised trunk together with
the other `clean/*` branches. Before starting:

1. **Confirm the rebase landed.** `clean/mesh`, `clean/preproc`, `clean/splats` and
   `clean/pointcloud` must share one base commit. Every line number in this plan was read
   against `clean/preproc`'s post-merge tree; re-read any file whose task cites a line range
   before editing it.
2. **`clean/splats` must drop its Task 17 and the mesh half of its Task 18.** Task 17 edits
   `mesh/utils.py`, which this plan deletes, and Task 18 edits
   `docs/source/tutorials/06_mesh/splats_mesh.ipynb`, which Task 11 rewrites. `clean/mesh`
   owns `mesh/*` and `06_mesh/*`.
3. **Merge order: `clean/preproc`, then `clean/splats`, then `clean/mesh`.** `clean/preproc`
   owns `preproc.frames`, which Tasks 4 and 7 read through; `clean/splats` owns the checkpoint
   loader and renderer that `render_tsdf_inputs` is written against. If the branches are
   executed in parallel rather than serially, `render_tsdf_inputs` (Task 4) and the splats
   half of Task 8 are the only pieces that wait on `clean/splats` — everything else builds
   against the shared base.

## Out of scope

Named here so nobody adds them mid-flight:

- Scale-relative `voxel_size` (a fraction of the camera-trajectory extent, the way the cleaning
  thresholds already work). It needs a measured comparison across omega-world and sfm-world
  scenes, which is its own piece of work.
- Dashboard display of the texture atlas. viser renders a textured trimesh and trimesh loads
  the PLY this plan writes, so the pieces exist; the dashboard still shows vertex colours.
