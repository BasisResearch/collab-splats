# clean_repair Scalability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `clean_repair_mesh` internals so it runs in ~66 s / 4.5 GB on meshes that currently require ~214 GB (SIGKILL), and stops stripping vertex colors.

**Architecture:** open3d does load/cluster/filter/save (native C++ clustering, colors preserved); meshlib does only hole filling, reached via `mrmeshnumpy` arrays instead of file I/O, with ONE batched `fillHoles` + one subdivide + one smooth. Colors reattach by index (original-vertex prefix is stable through meshlib) with a cKDTree nearest-neighbour fallback.

**Tech Stack:** open3d, numpy, scipy `cKDTree`, meshlib `mrmeshpy`/`mrmeshnumpy` — all already imported by `collab_splats/mesh/utils.py`.

**Spec:** `docs/superpowers/specs/2026-08-18-clean-repair-scalability-design.md`
**Validated prototype:** scratchpad `proto_clean_v2.py`, measured on `2026_07_15-Goprosplat-GH010229` (7.1 M faces, 240,371 comps, 176,162 holes).

---

### Task 1: Failing color-preservation test

**Files:**
- Modify: `tests/mesh/test_utils.py` (fixture `_holed_sphere_with_strays` + one new test)

- [x] **Step 1: Extend the fixture with an optional paint color**

In `tests/mesh/test_utils.py`, change `_holed_sphere_with_strays` to:

```python
def _holed_sphere_with_strays(path, radius=1.0, resolution=20, color=None):
    """Sphere missing a cap, plus one stray blob inside its bbox and one far outside."""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    tris = np.asarray(sphere.triangles)
    sphere.triangles = o3d.utility.Vector3iVector(tris[:-12])  # punch a hole
    sphere.remove_unreferenced_vertices()

    inside = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=6)
    inside.translate((0.2, 0.0, 0.0))
    outside = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=6)
    outside.translate((radius * 9, 0.0, 0.0))

    combined = sphere + inside + outside
    if color is not None:  # colored variant for the color-preservation test
        combined.paint_uniform_color(color)
    o3d.io.write_triangle_mesh(str(path), combined)
    return path
```

- [x] **Step 2: Add the failing test at the end of the clean_repair section**

```python
def test_clean_repair_mesh_preserves_vertex_colors(tmp_path):
    """Vertex colors survive the rewrite — the meshlib file round-trip used to strip them."""
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply", color=(0.2, 0.6, 0.9))
    clean_repair_mesh(mesh_path, max_hole_size=3.0)

    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert after.has_vertex_colors()
    # Every vertex — original and hole-patch alike — carries the painted color
    # (atol covers the uint8 PLY quantisation).
    assert np.allclose(np.asarray(after.vertex_colors), (0.2, 0.6, 0.9), atol=0.02)
```

- [x] **Step 3: Run the new test, verify it FAILS on color loss**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_clean_repair_mesh_preserves_vertex_colors -v`
Expected: FAIL at `assert after.has_vertex_colors()` (current meshlib save strips colors).

- [x] **Step 4: Verify the existing three clean_repair tests still pass (baseline)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -v -k clean_repair`
Expected: 3 pass, 1 fail (the new one).

---

### Task 2: Rewrite `clean_repair_mesh`

**Files:**
- Modify: `collab_splats/mesh/utils.py` (import block ~line 21, and the whole `clean_repair_mesh` body)
- Test: `tests/mesh/test_utils.py`

- [x] **Step 1: Add `mrmeshnumpy` to the guarded meshlib import**

```python
try:
    import meshlib.mrmeshpy as mm
    import meshlib.mrmeshnumpy as mn

    _MM_AVAILABLE = True
except ImportError:
    _MM_AVAILABLE = False
```

- [x] **Step 2: Replace the entire body of `clean_repair_mesh`**

```python
def clean_repair_mesh(
    mesh_path: str | Path,
    max_hole_size: float = 3.0,
    max_edge_splits: int = 1_000_000,
    use_largest: bool = False,  # if True, selects only the largest
) -> Path:
    """Drop stray components and fill small holes in a mesh on disk, rewriting it in place.

    Memory-flat by construction: components come from open3d's native clustering (one int
    per face) and holes are filled in a single batched meshlib call — never one bitset or
    temp mesh per component/hole. The previous meshlib getAllComponents path allocated a
    dense per-component FaceBitSet (240k components x 7.1M faces ≈ 214 GB on a TSDF scene)
    and was OOM-killed.

    Args:
        mesh_path: Mesh to clean. Overwritten with the result.
        max_hole_size: Fill holes whose perimeter is below this; larger ones are real openings
            (an unscanned wall, the open side of a room) and get left alone.
        max_edge_splits: Global subdivision budget shared by all hole patches, so patch
            refinement cannot explode the triangle count.
        use_largest: Keep only the biggest component. Off by default — that also throws away
            legitimate detached geometry (furniture, objects) that sits inside the scene.
    Returns:
        The path written (same as mesh_path).
    """
    if not _MM_AVAILABLE:
        raise ImportError("meshlib is required for clean_repair_mesh. Install it with: pip install meshlib")

    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    has_colors = mesh.has_vertex_colors()

    # Connected components via native clustering: one component id per triangle. For a TSDF
    # scene the largest component is the room itself; everything else is scene content or noise.
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    n_comp = len(cluster_sizes)
    largest = int(cluster_sizes.argmax())

    if use_largest:
        keep = np.zeros(n_comp, dtype=bool)
    else:
        # Per-component AABBs in one vectorized pass, then the cheap separator between scene
        # content and the floating specks TSDF leaves outside the room from stray depth:
        # keep every component whose bounding box sits inside the main one.
        tri_pts = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        comp_min = np.full((n_comp, 3), np.inf)
        comp_max = np.full((n_comp, 3), -np.inf)
        np.minimum.at(comp_min, cluster_ids, tri_pts.min(axis=1))
        np.maximum.at(comp_max, cluster_ids, tri_pts.max(axis=1))
        keep = np.all(comp_min >= comp_min[largest], axis=1) & np.all(
            comp_max <= comp_max[largest], axis=1
        )
    keep[largest] = True
    mesh.remove_triangles_by_mask(~keep[cluster_ids])
    mesh.remove_unreferenced_vertices()
    logger.info(
        "Kept %d of %d components (removed %d)", int(keep.sum()), n_comp, n_comp - int(keep.sum())
    )

    # Hand off to meshlib for hole filling — via arrays, not disk: meshlib's PLY round-trip
    # drops vertex colors, so colors stay behind in numpy and are reattached after.
    faces = np.asarray(mesh.triangles).astype(np.int32)
    verts = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors) if has_colors else None
    mmesh = mn.meshFromFacesVerts(faces, verts)

    # Patch size follows the mesh's own resolution, so a fill matches the surface around it.
    avg_edge_length = mmesh.averageEdgeLength()

    # Perimeter gate in Python (cheap: ~2 s for 176k holes), then ONE native batch fill —
    # a per-hole fill/subdivide/smooth loop does not finish at TSDF hole counts.
    hole_ids = mmesh.topology.findHoleRepresentiveEdges()
    small = mm.std_vector_Id_EdgeTag()
    for he in tqdm(hole_ids, desc=f"Measuring holes ({len(hole_ids)})"):
        if mmesh.holePerimeter(he) < max_hole_size:
            small.append(he)

    new_faces = mm.FaceBitSet()
    fill_params = mm.FillHoleParams()
    fill_params.outNewFaces = new_faces
    mm.fillHoles(mmesh, small, fill_params)

    # One subdivide + smooth over every patch at once, so fills are not flat caps.
    new_verts = mm.VertBitSet()
    subdiv_settings = mm.SubdivideSettings()
    subdiv_settings.maxEdgeLen = avg_edge_length
    subdiv_settings.maxEdgeSplits = max_edge_splits
    subdiv_settings.region = new_faces
    subdiv_settings.newVerts = new_verts
    mm.subdivideMesh(mmesh, subdiv_settings)
    mm.positionVertsSmoothly(mmesh, new_verts)
    logger.info("Filled %d of %d holes (max_hole_size=%s)", len(small), len(hole_ids), max_hole_size)

    # Back to numpy. Original vertices keep their indices through fill/subdivide/pack, so
    # colors copy straight through and only patch vertices need a nearest-neighbour lookup;
    # if meshlib ever reorders, fall back to a full NN transfer.
    mmesh.pack()
    out_verts = mn.getNumpyVerts(mmesh).astype(np.float64)
    out_faces = mn.getNumpyFaces(mmesh.topology)

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(out_verts), o3d.utility.Vector3iVector(out_faces)
    )
    if has_colors:
        out_colors = np.empty((len(out_verts), 3))
        n_orig = len(verts)
        prefix_stable = len(out_verts) >= n_orig and np.allclose(
            out_verts[:n_orig], verts, atol=1e-5
        )
        if prefix_stable:
            out_colors[:n_orig] = colors
            new_idx = np.arange(n_orig, len(out_verts))
        else:
            new_idx = np.arange(len(out_verts))
        if len(new_idx):
            _, nn = cKDTree(verts).query(out_verts[new_idx], k=1)
            out_colors[new_idx] = colors[nn]
        out.vertex_colors = o3d.utility.Vector3dVector(out_colors)

    o3d.io.write_triangle_mesh(str(mesh_path), out)
    return mesh_path
```

- [x] **Step 3: Run all four clean_repair tests, verify PASS**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -v -k clean_repair`
Expected: 4 pass (3 contract tests + the new color test).

- [x] **Step 4: Run the whole mesh test package**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ -v`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add collab_splats/mesh/utils.py tests/mesh/test_utils.py
git commit -m "fix(mesh): memory-flat clean_repair — batch hole fill, colors preserved

getAllComponents allocated a dense FaceBitSet per component
(240,371 comps x 7.1M faces = 214 GB on a TSDF scene -> SIGKILL).
Components now come from o3d cluster_connected_triangles (one int per
face) with vectorized per-component AABBs; holes are filled in a single
batched mm.fillHoles + one subdivide/smooth. Colors travel around
meshlib as numpy and reattach by stable vertex prefix (cKDTree NN for
patch verts) — the old file round-trip silently stripped them.
Measured on the failing mesh: 66 s / 4.5 GB peak."
```

---

### Task 3: Global `max_edge_splits` default on the TSDF creator

**Files:**
- Modify: `collab_splats/mesh/tsdf.py:35` (`clean_max_edge_splits` field)
- Test: `tests/mesh/test_tsdf.py`

- [x] **Step 1: Update the field default and its meaning**

In `Open3DTSDFFusion`, change:

```python
    clean_max_edge_splits: int = 10000
```

to:

```python
    clean_max_edge_splits: int = 1_000_000  # global subdivision budget across all hole patches
```

- [x] **Step 2: Run the tsdf tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_tsdf.py -v`
Expected: all pass (no test pins the old default).

- [x] **Step 3: Commit**

```bash
git add collab_splats/mesh/tsdf.py
git commit -m "fix(mesh): clean_max_edge_splits is a global subdivision budget (1M default)

Per-hole ceilings are meaningless under batched fillHoles; the budget
now caps total patch refinement. maxEdgeLen remains the real control."
```

---

### Task 4: Verify on the real failing mesh

**Files:**
- No repo changes — verification only, on a scratch copy.

- [x] **Step 1: Run the shipped function against a copy of the killed-run mesh**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/a5e0ab39-0895-4524-9574-6fbd037e8da0/scratchpad
cp /workspace/outputs/2026_07_15-Goprosplat-GH010229/vggt_omega/mesh.ply "$SCRATCH/verify_mesh.ply"
/opt/venv/reconstruction/bin/python - <<'EOF'
import resource, time
from collab_splats.mesh.utils import clean_repair_mesh

t0 = time.time()
clean_repair_mesh(
    "/tmp/claude-0/-workspace-collab-splats/a5e0ab39-0895-4524-9574-6fbd037e8da0/scratchpad/verify_mesh.ply"
)
print(f"time={time.time()-t0:.0f}s peak-rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
EOF
```

Expected: completes, `time` ≈ 60–120 s, `peak-rss` < 6 GB.

- [x] **Step 2: Confirm the rewritten PLY still carries colors**

```bash
head -c 400 "$SCRATCH/verify_mesh.ply" | strings | grep -E "element|property uchar"
```

Expected: `property uchar red/green/blue` present; vertex/face counts ≈ 4.9 M / 8.9 M.

- [x] **Step 3: Report the measured numbers in the final summary** (no commit — verification only).
