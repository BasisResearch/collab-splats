# OCR-lens viewer + unified mesh stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One prepared `mesh.ply` from the mesh stage, and a click viewer that shows OCR-lens words on it.

**Architecture:** The fill/decimate/manifold lines move out of `create_texture_mesh` into
`prepare_mesh` (`mesh/clean.py`); `Reconstructor.mesh()` writes its output as `mesh.ply` and hands the
unfilled cleaned mesh to texturing as the occluder. The viewer is three generic `Viewer` methods
plus an `ae=` argument on `verbalize`, glued by one example script in `docs/examples/`.

**Tech Stack:** open3d, meshlib, viser 1.0.29 (`add_mesh_trimesh`, `handle.on_click`), trimesh,
torch, LLaVA-1.6 decoder via `load_decoder`.

**Spec:** [2026-09-30-ocr-lens-viewer-design.md](../specs/2026-09-30-ocr-lens-viewer-design.md)

---

## Ground rules (every task)

- Worktree: `/workspace/collab-splats/.worktrees/ocr-lens`. `cd` into it in EVERY command; absolute paths.
- Python: `/opt/venv/reconstruction/bin/python` (py3.11). `export HF_HOME=/workspace/models HF_HUB_OFFLINE=1`.
- Style (CLAUDE.md + standing feedback):
  - block comments: ONE plain line saying what the code does (no header+bullet runs)
  - imports at top, absolute, isort groups; US spelling; no nested calls (`f(g(x))` → two lines)
  - tunables are keyword defaults, never module constants; blank line above/below every block
  - public functions in `collab_splats/{mesh,semantics,...}`: docstring contract (`"""` on own
    lines, one-line summary, `- ` bullets, `Args:`/`Returns:`); private `_` defs: summary only
  - `collab_splats/viewer.py` is NOT under the contract: match its one-line docstrings
- Format touched files only: `black --target-version py311 <files> && isort <files>`. Never repo-wide.
  If black reflows untouched code in `tests/reconstructor/test_reconstructor.py`, revert that hunk.
- Git: `git add <new files>` then `git commit --only <files> -m ...`. NEVER stash/amend/rebase/reset.
  Trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Cleaned modules (`collab_splats/mesh/`, `collab_splats/reconstructor.py`,
  `collab_splats/semantics/`): STOP before commit and hand the diff to the controller.** The
  controller shows it to the user and commits only after approval.
- No new functions beyond the ones named here. No new files beyond the ones named here.

---

## File map

| file | change |
|---|---|
| `collab_splats/mesh/clean.py` | receives `create_texture_mesh`'s five prepare lines verbatim as `prepare_mesh` (MOVE ONLY, no new mesh code) |
| `collab_splats/mesh/texture.py` | `create_texture_mesh(mesh, occluder, out_dir, ...)`: unwrap + project only |
| `collab_splats/mesh/__init__.py` | export `prepare_mesh` |
| `collab_splats/reconstructor.py` | `Reconstructor.mesh()`: read cleaned → `prepare_mesh` → write `mesh.ply` → optional texture |
| `configs/base.yaml` | `mesh.texture` comment |
| `docs/mesh.md` | usage block, clean section, texturing section |
| `collab_splats/semantics/features/ocr_lens.py` | `verbalize(..., ae=None)`; `_load_processor` → `load_processor` |
| `collab_splats/viewer.py` | `add_mesh`, `add_label_list`, `on_click` (+ private `_upload_mesh`, `_pick_vertex`, `_dispatch_click`, `_highlight`) |
| `docs/examples/ocr_lens_viewer.py` | NEW: scene glue |
| tests | `tests/mesh/test_clean.py`, `tests/mesh/test_texture.py`, `tests/reconstructor/test_reconstructor.py`, `tests/semantics/features/test_ocr_lens.py`, `tests/test_viewer.py` |

---

## Part A — unified mesh stage

### Task 1: move the prepare lines out of `create_texture_mesh` (MOVE ONLY)

**Rule: zero new mesh code.** Functions and their order are decided. This task is a cut/paste:
the five prepare lines already in `create_texture_mesh` (texture.py:74-81) move, byte-for-byte and
in the same order, into a def in `clean.py` next to the functions they call. Nothing else in
`collab_splats/mesh/` changes except what the move forces (signature, imports, docstring bullets
that describe the moved lines, the existing log line split where its fields now live). No new
logic, no new log lines, no new docstring claims, no new mesh tests.

**Files:**
- Modify: `collab_splats/mesh/clean.py` (receives the moved lines, after `make_manifold`)
- Modify: `collab_splats/mesh/texture.py` (loses them; signature takes the mesh + occluder)
- Modify: `collab_splats/mesh/__init__.py` (export)
- Test: `tests/mesh/test_texture.py` (existing test re-pointed at the new signature; no new tests)

- [ ] **Step 1: Re-point the existing texture test** — `test_create_texture_mesh_writes_obj_mtl_and_albedo`
  currently writes a plane PLY and passes its path. Change only the call: read nothing from disk,
  run the moved chain first, then texture:

```python
    prepared = prepare_mesh(plane, voxel_size=0.01)
    out = create_texture_mesh(
        prepared, plane, tmp_path / "texture", _constant_image(), c2w, K, voxel_size=0.01, tex_size=64
    )
```

  Keep every existing assertion unchanged. Import `prepare_mesh` from `collab_splats.mesh.clean`.

- [ ] **Step 2: Run, expect fail** (`ImportError: cannot import name 'prepare_mesh'`)

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_texture.py -q`

- [ ] **Step 3: Move** — in `clean.py` after `make_manifold`. The body is texture.py:76-81 verbatim;
  the Args/bullets are the ones cut from `create_texture_mesh`'s docstring:

```python
def prepare_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    voxel_size: float,
    max_hole_perimeter_ratio: float = 3.9,
    decimate_max_error: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """
    Fill, decimate and repair a cleaned mesh into one UVAtlas accepts; the input is not modified.

    - fill at full density, decimate, make_manifold; lid the pinholes that opens, repair again
    - outer rims stay open whatever max_hole_perimeter_ratio (see fill_holes)

    Args:
        mesh: cleaned mesh (clean_repair_mesh output).
        voxel_size: TSDF voxel the mesh was fused at; sets the decimation bound.
        max_hole_perimeter_ratio: patch holes with a perimeter under this × scene_scale.
        decimate_max_error: decimation bound as a multiple of voxel_size.

    Returns:
        The filled, decimated, manifold mesh.
    """
    filled = fill_holes(mesh, max_hole_perimeter_ratio=max_hole_perimeter_ratio)
    decimated, err = decimate_mesh(filled, max_error=decimate_max_error * voxel_size)
    manifold = make_manifold(decimated)

    # Decimation and repair open pinholes; flat lids close them, then repair what the lids fold
    lidded = fill_holes(manifold, max_hole_perimeter_ratio=max_hole_perimeter_ratio, subdivide_fill=False)
    manifold = make_manifold(lidded)
    logger.info("prepare_mesh: %d -> %d faces (decimation error %.4f)", len(mesh.triangles), len(manifold.triangles), err)
    return manifold
```

  (The log line is the existing `create_texture_mesh` log's `faces`/`err` fields, which can only be
  read here now; `create_texture_mesh`'s log keeps its face count + path.)

  `texture.py` after the move:

```python
def create_texture_mesh(
    mesh: o3d.geometry.TriangleMesh,
    occluder: o3d.geometry.TriangleMesh,
    out_dir: Path | str,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    *,
    voxel_size: float,
    tex_size: int = 8192,
) -> Path:
    """
    Unwrap and texture a prepared mesh.

    - the unfilled occluder hides surfaces, so invented patches never hide a real surface
    - voxel_size is the occlusion tolerance
    - writes out_dir/mesh.obj + mesh.mtl + albedo.png via write_textured_obj

    Args:
        mesh: prepare_mesh output.
        occluder: the cleaned mesh before prepare_mesh.
        out_dir: directory to create.
        rgbs: (N, H, W, 3) uint8 views that were fused.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        voxel_size: TSDF voxel the mesh was fused at, world units.
        tex_size: atlas edge in texels.

    Returns:
        Path to out_dir/mesh.obj.
    """
    rgbs, c2w, K, _ = _validate_views(rgbs, c2w, K)
    tm = unwrap_mesh_uvs(mesh, tex_size)
    albedo = project_images_to_texture(tm, rgbs, c2w, K, tex_size, occlusion_eps=voxel_size, occluder=occluder)

    # Smooth per-vertex normals for the OBJ; without them viewers shade split corners flat
    tm.compute_vertex_normals()
    out = write_textured_obj(
        out_dir,
        tm.vertex.positions.numpy(),
        tm.triangle.indices.numpy(),
        tm.vertex.normals.numpy(),
        tm.triangle.texture_uvs.numpy(),
        albedo,
    )
    logger.info("create_texture_mesh: %d faces -> %s", len(tm.triangle.indices), out)
    return out
```

  Imports: texture.py drops `from collab_splats.mesh.clean import decimate_mesh, fill_holes, make_manifold`.
  Module docstrings: the texture.py bullet `fill_holes, decimate_mesh, make_manifold (clean.py): a mesh
  UVAtlas accepts` moves to clean.py as `- prepare_mesh: fill_holes, decimate_mesh, make_manifold into a
  mesh UVAtlas accepts`; texture.py gets `- input: prepare_mesh output (clean.py)` in its place.
  `__init__.py`: `from collab_splats.mesh.clean import clean_repair_mesh, prepare_mesh`, add
  `"prepare_mesh"` to `__all__`, `clean:` bullet → `- clean: drop floaters, fill small holes; prepare_mesh for mesh.ply`.

- [ ] **Step 4: Move proof** — the moved statements must be identical. Print both and compare by eye
  + hash (RTK `diff` lies: use `sha256sum`):

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git show HEAD:collab_splats/mesh/texture.py | sed -n 76,81p | sed 's/^ *//' | sha256sum && rtk proxy grep -A8 "^    filled = fill_holes" collab_splats/mesh/clean.py | head -7 | rtk proxy grep -v "^$" | sed 's/^ *//' | sha256sum
```
Expected: same hash. `git diff --stat -- collab_splats/mesh` shows only clean.py / texture.py / __init__.py.

- [ ] **Step 5: Run, expect pass**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/mesh tests/test_docstring_contract.py tests/test_import_style.py -q`

- [ ] **Step 6: Format** touched files (black `--target-version py311` + isort).

- [ ] **Step 7: STOP — report `git diff -- collab_splats/mesh tests/mesh` to the user.** Commit after approval:

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git commit --only collab_splats/mesh/clean.py collab_splats/mesh/texture.py collab_splats/mesh/__init__.py tests/mesh/test_texture.py -m "refactor(mesh): move the prepare lines out of create_texture_mesh into prepare_mesh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: `Reconstructor.mesh()` writes the prepared `mesh.ply`

Layout after the rebase onto `clean/reconstructor-release`: the mesh stage is the method
`Reconstructor.mesh()` in `collab_splats/reconstructor.py` (no `_run_tsdf_mesh`, no `wrapper/`);
it reads `cfg = self.config["mesh"]` and writes into `self.backend_dir`. Tests live in
`tests/reconstructor/test_reconstructor.py` (`from collab_splats import reconstructor as R`).

**Files:**
- Modify: `collab_splats/reconstructor.py` (imports; tail of `Reconstructor.mesh()`; its docstring summary)
- Modify: `configs/base.yaml` (`mesh.texture` comment, line ~120)
- Test: `tests/reconstructor/test_reconstructor.py`

- [ ] **Step 1: Keep existing mesh tests green** — two sites stub `clean_repair_mesh` and would now
reach the real `prepare_mesh` on a mesh that was never written:
  - `_mesh_fuse` (after `monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())`) gets:

```python
    monkeypatch.setattr(R, "prepare_mesh", lambda mesh, **kw: mesh)
```

  - `test_mesh_reads_frames_and_poses_from_the_zarr_on_disk` gets one more context manager after
    `patch.object(R, "clean_repair_mesh"),`:

```python
        patch.object(R, "prepare_mesh", side_effect=lambda mesh, **kw: mesh),
```

- [ ] **Step 2: Write the failing tests** — after `test_mesh_masks_depth_by_confidence`:

```python
def _run_prepared_mesh(tmp_path, monkeypatch, texture):
    """
    Run rec.mesh() with fusion writing a real mesh; return (rec, cleaned, prepared, prepare, texture mocks).
    """
    mesh_cfg = {"enabled": True, "source": "feedforward", "voxel_size": 0.01, "texture": texture}
    rec = Reconstructor(_make_config(tmp_path, {"mesh": mesh_cfg}))
    ff = _tsdf_mesh_ff(model_hw=(16, 16))
    monkeypatch.setattr(PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: ff))
    monkeypatch.setattr(R.frames, "read_frames", lambda *a, **k: np.full((2, 32, 32, 3), 128, np.uint8))
    monkeypatch.setattr(pointcloud_utils, "upsample_depths", _unit_upsample)

    # Fusion writes a real sphere to mesh.ply; cleaning is a no-op on it
    cleaned = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)
    mesh_path = rec.backend_dir / "mesh.ply"

    def fuse(*args, **kwargs):
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_path), cleaned)
        return mesh_path

    monkeypatch.setattr(R, "create_tsdf_mesh", fuse)
    monkeypatch.setattr(R, "clean_repair_mesh", MagicMock())

    # prepare_mesh returns a box, so mesh.ply's triangle count tells which mesh was written
    prepared = o3d.geometry.TriangleMesh.create_box()
    prepare = MagicMock(return_value=prepared)
    monkeypatch.setattr(R, "prepare_mesh", prepare)
    texture_mesh = MagicMock()
    monkeypatch.setattr(R, "create_texture_mesh", texture_mesh)

    rec.mesh()

    return rec, cleaned, prepared, prepare, texture_mesh


def test_mesh_writes_the_prepared_mesh_without_texture(tmp_path, monkeypatch):
    """mesh.ply is prepare_mesh's output even with texture off, prepared from the cleaned mesh."""
    rec, cleaned, prepared, prepare, texture_mesh = _run_prepared_mesh(tmp_path, monkeypatch, texture=False)

    assert len(prepare.call_args.args[0].triangles) == len(cleaned.triangles)
    assert prepare.call_args.kwargs == {"voxel_size": 0.01}
    written = o3d.io.read_triangle_mesh(str(rec.backend_dir / "mesh.ply"))
    assert len(written.triangles) == len(prepared.triangles)
    texture_mesh.assert_not_called()


def test_mesh_textures_the_prepared_mesh_behind_the_cleaned_occluder(tmp_path, monkeypatch):
    """Texturing unwraps the same mesh.ply geometry; the unfilled cleaned mesh is the occluder."""
    rec, cleaned, prepared, _, texture_mesh = _run_prepared_mesh(tmp_path, monkeypatch, texture=True)

    mesh, occluder, out_dir = texture_mesh.call_args.args[:3]
    assert mesh is prepared
    assert len(occluder.triangles) == len(cleaned.triangles)
    assert out_dir == rec.backend_dir / "texture"
    assert texture_mesh.call_args.kwargs == {"voxel_size": 0.01}
```

Add `import open3d as o3d` to the test file's third-party import group.

- [ ] **Step 3: Run, expect fail**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor/test_reconstructor.py -k mesh -q`
Expected: the two new tests FAIL (`AttributeError: ... has no attribute 'prepare_mesh'` from monkeypatch, raising=True).

- [ ] **Step 4: Implement** — imports in `collab_splats/reconstructor.py`: `import open3d as o3d`
(third-party group, after `numpy`), and

```python
from collab_splats.mesh import (
    clean_repair_mesh,
    create_texture_mesh,
    create_tsdf_mesh,
    prepare_mesh,
)
```

Tail of `Reconstructor.mesh()` (replaces `clean_repair_mesh(...)` through the texture block; the
`create_tsdf_mesh(...)` call and the final `logger.info("Mesh saved to %s", mesh_path)` stay):

```python
        clean_repair_mesh(mesh_path, use_convex_hull=cfg["use_convex_hull"])

        # Prepare the cleaned mesh into mesh.ply; the unfilled cleaned mesh stays as texturing's occluder
        cleaned = o3d.io.read_triangle_mesh(str(mesh_path))
        prepared = prepare_mesh(cleaned, voxel_size=cfg["voxel_size"])
        o3d.io.write_triangle_mesh(str(mesh_path), prepared)

        if cfg["texture"]:
            texture_dir = self.backend_dir / "texture"
            create_texture_mesh(prepared, cleaned, texture_dir, rgbs, c2w, intrinsics, voxel_size=cfg["voxel_size"])
```

The block comment above `create_tsdf_mesh(...)` changes from `# Fuse, clean, then optionally texture`
to `# Fuse, then clean in place`. Docstring summary:
`Fuse depth and RGB into a TSDF mesh.ply, clean and prepare it, and optionally texture it.`
Add a bullet: `- mesh.ply is the prepared mesh (filled, decimated, manifold); texture/ is its UV bake`.

`configs/base.yaml`, the `mesh.texture` line:
```yaml
  texture: false           # UV atlas + project the fused views onto mesh.ply; writes texture/
```

- [ ] **Step 5: Run, expect pass**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor tests/mesh tests/test_docstring_contract.py tests/test_import_style.py -q --ignore=tests/reconstructor/test_cli.py`
Expected: all pass. (`test_cli.py` cannot collect in this venv: installed `collab_data` lacks
`STATS_ARGS`, which `collab_splats/remote.py` imports — env drift, not this task.)

- [ ] **Step 6: Format** touched files (black `--target-version py311` + isort on `collab_splats/reconstructor.py`
and `tests/reconstructor/test_reconstructor.py`); revert any hunk black makes outside the edited lines.

- [ ] **Step 7: STOP — report diff** (`git diff -- collab_splats/reconstructor.py configs tests/reconstructor`). After approval:

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git commit --only collab_splats/reconstructor.py configs/base.yaml tests/reconstructor/test_reconstructor.py -m "feat(mesh): mesh stage always writes the prepared mesh.ply; texture is its UV bake

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: `docs/mesh.md`

**Files:** Modify `docs/mesh.md` (usage block ~L20-35, clean section ~L142, texturing ~L194-219, file table L10-11)

- [ ] **Step 1: Usage block** — import and call `prepare_mesh`:

```python
from pathlib import Path

import open3d as o3d

from collab_splats.mesh import clean_repair_mesh, create_tsdf_mesh, prepare_mesh

mesh_path = create_tsdf_mesh(
    depths,       # (n, h, w) float32, 0 = no observation
    rgbs,         # (n, h, w, 3) uint8
    c2w,          # (n, 4, 4) camera-to-world
    K,            # (n, 3, 3) at the depth resolution
    Path("scene/mesh"),
    voxel_size=0.0025,
    depth_trunc=1.5,
)
clean_repair_mesh(mesh_path)   # rewrites mesh.ply in place
cleaned = o3d.io.read_triangle_mesh(str(mesh_path))
prepared = prepare_mesh(cleaned, voxel_size=0.0025)   # the mesh.ply the stage ships
```

- [ ] **Step 2: Texturing section** — steps 1-3 (fill / decimate / make_manifold) move under a new
subsection `## Preparing mesh.ply` placed before `## Texturing`, opening with: "The mesh stage always
ships the prepared mesh: `prepare_mesh` fills, decimates and repairs the cleaned mesh, whether or not
`mesh.texture` is on. With `use_convex_hull: true` the ground is patched out to the hull and rim necks
are bridged first, so more inlets become interior holes this fill closes; the outermost edge is never
lidded." `## Texturing` renumbers from unwrap (1. `unwrap_mesh_uvs` ...) and its last paragraph
becomes: "`create_texture_mesh(mesh, occluder, out_dir, ...)` unwraps the prepared mesh as is and
writes `out_dir/mesh.obj` + `mesh.mtl` + `albedo.png`; `occluder` is the cleaned mesh before the
fill. The OBJ's vertex arrays differ from `mesh.ply` (UV seams duplicate vertices); the surface is
the same." File table `clean.py` row gains `prepare_mesh`.

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git commit --only docs/mesh.md -m "docs(mesh): mesh.ply is the prepared mesh; texturing bakes onto it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Part B — viewer

### Task 4: `verbalize(..., ae=)` and public `load_processor`

**Files:**
- Modify: `collab_splats/semantics/features/ocr_lens.py:183-250` (`verbalize`), `:260` (`_load_processor` → `load_processor`) and its two callers (`OCRLensExtractor.__init__`, `score_ocr_heads`)
- Test: `tests/semantics/features/test_ocr_lens.py`

- [ ] **Step 1: Write the failing test** — after `test_verbalize_zero_vocab_mass_is_not_nan`:

```python
def test_verbalize_decodes_ae_codes_like_their_decoded_states():
    torch.manual_seed(0)
    ae = FeatureAutoencoder(input_dim=5, latent_dim=3)
    codes = torch.randn(7, 3)
    with torch.no_grad():
        states = ae.per_point_decode(codes)

    direct = verbalize(states, _identity_decoder(5), _vocab(), k=2, chunk=3)
    via_ae = verbalize(codes, _identity_decoder(5), _vocab(), k=2, chunk=3, ae=ae)

    assert direct[0] == via_ae[0]
    np.testing.assert_allclose(direct[1], via_ae[1], rtol=1e-5)
    np.testing.assert_allclose(direct[2], via_ae[2], rtol=1e-5)
```

Import `from collab_splats.semantics.compression import FeatureAutoencoder` in the test file.

- [ ] **Step 2: Run, expect fail**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_ocr_lens.py -k verbalize -q`
Expected: FAIL `TypeError: verbalize() got an unexpected keyword argument 'ae'`.

- [ ] **Step 3: Implement** — import `from collab_splats.semantics.compression import FeatureAutoencoder`
(own-code group). Signature:

```python
def verbalize(
    features: torch.Tensor,
    decoder: nn.Module,
    vocab: WordVocab,
    *,
    k: int = 10,
    chunk: int = 256,
    ae: Optional[FeatureAutoencoder] = None,
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
```

Docstring: add bullet `- ae: codes are decoded one chunk at a time, so (P, D) states never sit in memory`;
`features:` arg → `(P, D) lens states, (P, latent) codes when ae is given, or a (D, H, W) map.`;
new arg `ae: autoencoder that decodes point codes to lens states, or None when features are states.`

In the loop, between `x = features[start : start + chunk]` and the cast to the decoder:

```python
        x = features[start : start + chunk]

        # Codes decode to lens states on the AE's device first
        if ae is not None:
            ae_param = next(ae.parameters())
            x = x.to(device=ae_param.device, dtype=ae_param.dtype)
            x = ae.per_point_decode(x)

        x = x.to(device=param.device, dtype=param.dtype)
```

(keep the existing `# Decode one block of states ...` comment on the block that follows.)

Rename `_load_processor` → `load_processor` (def + 2 callers) with docstring:

```python
def load_processor(model_id: str) -> LlavaNextProcessor:
    """
    LLaVA-1.6 processor with the fast tokenizer.

    - its `.tokenizer` feeds `word_vocabulary`; `AutoTokenizer` needs sentencepiece for this checkpoint

    Args:
        model_id: hub id or local directory.

    Returns:
        The checkpoint's processor.
    """
```

Check no other caller: `rtk proxy grep -rn "_load_processor" --include=*.py collab_splats tests docs scratch` → empty.

- [ ] **Step 4: Run, expect pass**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_ocr_lens.py tests/test_docstring_contract.py tests/test_import_style.py -q`
Expected: pass.

- [ ] **Step 5: Format; STOP — report diff; commit after approval**

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git commit --only collab_splats/semantics/features/ocr_lens.py tests/semantics/features/test_ocr_lens.py -m "feat(semantics): verbalize decodes AE codes per chunk; public load_processor

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: `Viewer.add_mesh`, `add_label_list`, `on_click`

**Files:**
- Modify: `collab_splats/viewer.py` (imports, `__init__` state, new methods in the "Scene nodes" section, private helpers in the handlers section)
- Test: `tests/test_viewer.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_viewer.py`:

```python
def _quad():
    """Unit square in z=0 as two triangles; vertex 3 is the (1, 1) corner."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    colors = np.full((4, 3), 200, np.uint8)
    return vertices, faces, colors


def test_add_mesh_registers_node(viewer):
    viewer.add_mesh("quad", *_quad())
    assert "quad" in viewer.meshes


def test_pick_vertex_returns_the_hit_triangles_nearest_vertex(viewer):
    viewer.add_mesh("quad", *_quad())
    assert viewer._pick_vertex("quad", (0.9, 0.9, 1.0), (0.0, 0.0, -1.0)) == 3
    assert viewer._pick_vertex("quad", (5.0, 5.0, 1.0), (0.0, 0.0, -1.0)) is None


def test_on_click_hands_the_picked_vertex_to_the_callback(viewer):
    viewer.add_mesh("quad", *_quad())
    picked = []
    viewer.on_click("quad", picked.append)
    event = SimpleNamespace(ray_origin=(0.1, 0.1, 1.0), ray_direction=(0.0, 0.0, -1.0))
    viewer._dispatch_click("quad", event)
    assert picked == [0]


def test_highlight_tints_the_label_and_dims_the_rest(viewer, monkeypatch):
    vertices, faces, colors = _quad()
    viewer.add_mesh("quad", vertices, faces, colors)
    sent = {}
    monkeypatch.setattr(viewer, "_upload_mesh", lambda name, shown: sent.update(shown=shown))
    labels = np.array(["tree", "tree", "rock", ""], dtype=object)

    viewer._highlight("quad", labels, "tree", (255, 80, 0))
    assert sent["shown"][:2].tolist() == [[255, 80, 0]] * 2
    assert sent["shown"][2:].tolist() == [[66, 66, 66]] * 2

    viewer._highlight("quad", labels, None, (255, 80, 0))
    assert (sent["shown"] == colors).all()


def test_add_label_list_counts_labels_most_common_first_and_skips_empty(viewer):
    viewer.add_mesh("quad", *_quad())
    labels = np.array(["rock", "tree", "tree", ""], dtype=object)

    viewer.add_label_list("quad", labels, top_n=5)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == ["Clear", "tree (2)", "rock (1)"]

    # Re-adding replaces the list rather than stacking a second one
    viewer.add_label_list("quad", labels, top_n=1)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == ["Clear", "tree (2)"]
```

Add `from types import SimpleNamespace` to the stdlib imports.

- [ ] **Step 2: Run, expect fail**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -q`
Expected: new tests FAIL (`AttributeError: 'Viewer' object has no attribute 'add_mesh'`).

- [ ] **Step 3: Implement** — `collab_splats/viewer.py`:

Imports (stdlib: `from functools import partial`, `from typing import Callable, Optional`;
third-party: `import open3d as o3d`, `import trimesh`).

Module docstring: add a line `Meshes take per-vertex colors, a clickable vertex pick and a label list.`

`__init__`, next to `self.frustums`:

```python
        # Meshes: name -> (vertices, faces, colors, raycasting scene); click callbacks and label lists
        self.meshes: dict = {}
        self.mesh_clicks: dict = {}
        self.label_lists: dict = {}
```

Scene-node methods (after `add_lines`):

```python
    def add_mesh(self, name: str, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
        """Upsert a named mesh; vertices (V, 3) float, faces (F, 3) int, colors (V, 3) uint8."""
        rays = o3d.t.geometry.RaycastingScene()
        rays.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
        self.meshes[name] = (vertices, faces, colors, rays)
        self._upload_mesh(name, colors)

    def add_label_list(
        self, name: str, labels: np.ndarray, top_n: int = 20, color: tuple = (255, 80, 0)
    ) -> None:
        """Buttons for a mesh's top_n most common vertex labels; clicking one tints its vertices."""
        values, counts = np.unique(labels[labels != ""], return_counts=True)
        order = np.argsort(-counts, kind="stable")[:top_n]

        # Replace an earlier list for this mesh
        if name in self.label_lists:
            self.label_lists[name][0].remove()

        folder = self.server.gui.add_folder(f"Labels: {name}")
        with folder:
            clear = self.server.gui.add_button("Clear")
            clear.on_click(lambda _: self._highlight(name, labels, None, color))
            buttons = [clear]

            for i in order:
                label = str(values[i])
                button = self.server.gui.add_button(f"{label} ({counts[i]})")
                button.on_click(lambda _, label=label: self._highlight(name, labels, label, color))
                buttons.append(button)

        self.label_lists[name] = (folder, buttons)

    def on_click(self, name: str, callback: Callable[[int], None]) -> None:
        """Call callback(vertex index) for the vertex nearest where a click first hits the mesh."""
        self.mesh_clicks[name] = callback
        self._upload_mesh(name, self.meshes[name][2])
```

Private helpers (in the handlers section):

```python
    def _upload_mesh(self, name: str, shown: np.ndarray) -> None:
        """Re-send a mesh with the given vertex colors (no in-place recolor in viser); re-bind its click."""
        vertices, faces, _, _ = self.meshes[name]
        mesh = trimesh.Trimesh(vertices, faces, vertex_colors=shown, process=False)
        handle = self.server.scene.add_mesh_trimesh(name, mesh)

        if name in self.mesh_clicks:
            handle.on_click(partial(self._dispatch_click, name))

    def _dispatch_click(self, name: str, event) -> None:
        """Pick the clicked vertex and hand it to the mesh's callback; misses are ignored."""
        vertex = self._pick_vertex(name, event.ray_origin, event.ray_direction)

        if vertex is not None:
            self.mesh_clicks[name](vertex)

    def _pick_vertex(self, name: str, origin: tuple, direction: tuple) -> Optional[int]:
        """Vertex of the first-hit triangle nearest the hit point, or None when the ray misses."""
        vertices, faces, _, rays = self.meshes[name]
        ray = o3d.core.Tensor([[*origin, *direction]], dtype=o3d.core.float32)
        hit = rays.cast_rays(ray)
        t = float(hit["t_hit"][0].item())

        if not np.isfinite(t):
            return None

        point = np.asarray(origin) + t * np.asarray(direction)
        triangle = faces[int(hit["primitive_ids"][0].item())]
        distances = np.linalg.norm(vertices[triangle] - point, axis=1)
        return int(triangle[np.argmin(distances)])

    def _highlight(self, name: str, labels: np.ndarray, label: Optional[str], color: tuple) -> None:
        """Re-send a mesh with label's vertices in color and the rest dimmed; None restores it."""
        colors = self.meshes[name][2]
        shown = colors

        if label is not None:
            shown = colors // 3
            shown[labels == label] = color

        self._upload_mesh(name, shown)
```

- [ ] **Step 4: Run, expect pass**

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py tests/test_import_style.py -q`
Expected: pass.

- [ ] **Step 5: Format + commit** (viewer.py is not a cleaned module)

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git commit --only collab_splats/viewer.py tests/test_viewer.py -m "feat(viewer): meshes with vertex colors, click-to-vertex, label list highlight

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: `docs/examples/ocr_lens_viewer.py`

**Files:** Create `docs/examples/ocr_lens_viewer.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Click a scene's mesh to read the OCR lens's words there.

- needs mesh.ply, pointcloud.zarr and semantics/ocr_lens_lifted.zarr + ocr_lens_ae.pt under the backend dir
- vertices no lifted point reaches within --max_dist are unobserved: grey, unlabeled
- the smoothing slider averages vertex codes over k neighbors before decoding
- labels list: each vertex's top-1 word, most common first; click one to tint where it is top-1

Usage:
    HF_HOME=/workspace/models HF_HUB_OFFLINE=1 python docs/examples/ocr_lens_viewer.py \\
        /workspace/outputs/<scene>/<backend> --port 8080
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import zarr

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.ocr_lens import load_decoder, load_processor, verbalize, word_vocabulary
from collab_splats.semantics.lifting import transfer_features
from collab_splats.semantics.utils import ae_path, lifted_store_path
from collab_splats.utils.torch_utils import get_device
from collab_splats.viewer import Viewer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("backend_dir", type=Path)
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--max_dist", type=float, default=0.03, help="unobserved beyond this, world units")
parser.add_argument("--model_id", default="llava-hf/llava-v1.6-vicuna-7b-hf")
args = parser.parse_args()

# Mesh with its vertex colors
mesh = o3d.io.read_triangle_mesh(str(args.backend_dir / "mesh.ply"))
vertices = np.asarray(mesh.vertices, dtype=np.float32)
faces = np.asarray(mesh.triangles)
colors = np.asarray(mesh.vertex_colors) * 255
colors = colors.astype(np.uint8)

# Lifted codes and the pointcloud points they index
sem_dir = args.backend_dir / "semantics"
store = zarr.open(str(lifted_store_path(sem_dir, "ocr_lens")), mode="r")
codes = np.asarray(store["features"], dtype=np.float32)
cloud = PointcloudResult.load_zarr(
    args.backend_dir / "pointcloud.zarr",
    load_depth=False,
    load_world_points=False,
    load_confidence=False,
    load_pixel_indices=False,
)
assert len(cloud.points) == len(codes), "lifted codes do not index this pointcloud; re-run semantics"

# Codes onto vertices; vertices no point reaches are unobserved and grey
vertex_codes = transfer_features(vertices, cloud.points, codes, max_dist=args.max_dist)
observed = vertex_codes.any(axis=1)
colors[~observed] = 128

# Lens decoder, word vocabulary and the AE that decodes the codes
ae = FeatureAutoencoder.load(ae_path(sem_dir, "ocr_lens"))
ae.to(get_device())
decoder = load_decoder(args.model_id)
tokenizer = load_processor(args.model_id).tokenizer
vocab = word_vocabulary(tokenizer)

# Scene, word panel and smoothing slider
viewer = Viewer(port=args.port)
viewer.add_mesh("mesh", vertices, faces, colors)
panel = viewer.server.gui.add_markdown("Click the mesh")
smoothing = viewer.server.gui.add_slider("Smoothing k", min=1, max=64, step=1, initial_value=1)
state = {"codes": vertex_codes}


def relabel(_=None) -> None:
    """Smooth observed vertex codes over k neighbors, then list each vertex's top-1 word."""
    smoothed = vertex_codes.copy()

    if smoothing.value > 1:
        seen = vertices[observed]
        smoothed[observed] = transfer_features(seen, seen, vertex_codes[observed], k=smoothing.value)

    state["codes"] = smoothed
    words, _, _ = verbalize(torch.from_numpy(smoothed[observed]), decoder, vocab, k=1, ae=ae)
    labels = np.full(len(vertices), "", dtype=object)
    labels[observed] = [row[0] for row in words]
    viewer.add_label_list("mesh", labels)


def show_words(vertex: int) -> None:
    """Top-10 words at the clicked vertex as a markdown table."""
    if not observed[vertex]:
        panel.content = f"**vertex {vertex}**: unobserved"
        return

    row = torch.from_numpy(state["codes"][vertex : vertex + 1])
    words, probs, mass = verbalize(row, decoder, vocab, k=10, ae=ae)
    lines = [f"**vertex {vertex}**, vocab mass {mass[0]:.2f}", "", "| word | p |", "|---|---|"]
    lines += [f"| {word} | {p:.3f} |" for word, p in zip(words[0], probs[0])]
    panel.content = "\n".join(lines)


smoothing.on_update(relabel)
viewer.on_click("mesh", show_words)
relabel()
viewer.serve_forever()
```

- [ ] **Step 2: Import-check** (no scene needed):

Run: `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python docs/examples/ocr_lens_viewer.py --help`
Expected: usage text, exit 0.

- [ ] **Step 3: Format** (`black --target-version py311` + isort on the file) and commit:

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && git add docs/examples/ocr_lens_viewer.py && git commit --only docs/examples/ocr_lens_viewer.py -m "docs(examples): ocr_lens_viewer — click the mesh for lens words, top-1 word list

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Real-scene run (controller, tmux, one heavy job at a time)

- [ ] **Step 1: Build a scene** with ocr_lens semantics and the mesh stage on the tutorial video.
  `reconstruct local` (`collab_splats/__main__.py`) is the entry point, but it cannot import in this
  venv (`collab_splats.remote` needs `collab_data` `STATS_ARGS`), so drive `Reconstructor` directly —
  the same object the CLI builds, output at `<output-root>/<stem>`:

```bash
cd /workspace/collab-splats/.worktrees/ocr-lens && cat > /tmp/claude-0/-workspace-collab-splats/13a2056b-e3bb-4c00-a5c2-5f7ef258c72a/scratchpad/ocr_viewer_run.py <<'EOF'
import logging

from collab_splats.reconstructor import Reconstructor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
config = {
    "input_path": "/workspace/collab-splats/data/tutorial/tutorial_example-video.mp4",
    "output_path": "/workspace/outputs/ocr_viewer/tutorial_example-video",
    "semantics": {"enabled": True, "extractor": "ocr_lens", "n_components": 128},
    "mesh": {"enabled": True},
}
Reconstructor(config).run()
EOF
tmux new -d -s ocrview "cd /workspace/collab-splats/.worktrees/ocr-lens && HF_HOME=/workspace/models HF_HUB_OFFLINE=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/13a2056b-e3bb-4c00-a5c2-5f7ef258c72a/scratchpad/ocr_viewer_run.py 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/13a2056b-e3bb-4c00-a5c2-5f7ef258c72a/scratchpad/ocr_viewer.log"
```

Record from the log: `prepare_mesh: A -> B faces` and the mesh-stage wall time (report to user —
the spec owes the cost of always preparing).

- [ ] **Step 2: Launch the viewer** in tmux on the produced backend dir; confirm in the log that
`relabel` finished and report: vertex count, observed fraction, top-10 labels with counts. The user
opens the port and experiments.

### Task 8: Gates + graph

- [ ] **Step 1:** `cd /workspace/collab-splats/.worktrees/ocr-lens && /opt/venv/reconstruction/bin/python -m pytest tests/mesh tests/reconstructor tests/semantics tests/test_viewer.py tests/test_docstring_contract.py tests/test_import_style.py -q -p no:cacheprovider > <scratchpad>/gate.log 2>&1; echo exit=$?` — read the log's summary line and exit code (never `| tail`).
- [ ] **Step 2:** `graphify update .`
- [ ] **Step 3:** Remaining branch work stays in the parent plan's T12 (decision 021, CHANGELOG,
CLAUDE.md in-flight entry, spec/plan updates) and the final full suite + branch review.
