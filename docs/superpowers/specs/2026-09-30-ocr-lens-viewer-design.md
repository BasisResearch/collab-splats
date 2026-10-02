# OCR-lens viewer + unified mesh stage — design / handoff

Status: APPROVED (2026-09-30). Branch `feat/ocr-lens`, worktree
`.worktrees/ocr-lens`. Parent spec: [2026-09-29-ocr-lens-design.md](2026-09-29-ocr-lens-design.md).

## Why

- Goal: experiment with the OCR lens on a real scene BEFORE the reconstructor/dashboard refactor.
- Need: click the mesh, read the word distribution there; list the most common words, click
  one, see where it is top-1.
- Blocker: the mesh stage writes two different meshes today, so there is no single mesh to
  put words on.

## Where the branch is (commits on top of `main`)

- Lens: `word_vocabulary`, `verbalize`, `OCRLensExtractor` (LLaVA-1.6, layer configurable),
  `load_decoder`, `score_ocr_heads`.
- Compress-before-lift: fp16 2D cache keyed on `extractor_kwargs`; AE fit on `load_features`.
- `ceb96da4`: `extract_feature_cache(..., overwrite=)` really re-extracts.
- `ce4abee0`: AE stored at `semantics/<extractor>.zarr/autoencoder.pt`, reused when
  `latent_dim == n_components`; re-extraction wipes it.
- `4056b045`: `mesh.features.features2vertex` → `semantics.lifting.transfer_features`
  (`targets, points, features, k, max_dist`); targets beyond `max_dist` get zero features.
- Measured (A40, 3,512 words, AE 64→4096 + `verbalize(k=1)`): 100k vertices 0.8 s / 1.8 GiB,
  500k vertices 3.3 s / 3.1 GiB. Top-1 over the whole mesh is interactive-cheap.

## Part A — unified mesh stage

### Today

| file | made by | what it is |
|---|---|---|
| `mesh.ply` | `create_tsdf_mesh` → `clean_repair_mesh` | full density; floaters cut, holes < 0.014·scale filled; vertex colors |
| `texture/mesh.obj` | `create_texture_mesh` (only if `mesh.texture`) | fill 3.9 → decimate 0.5·voxel → manifold → lid → manifold → UV unwrap → project |

The prepared (filled/decimated/manifold) mesh exists only inside `create_texture_mesh`.

### Proposed

- ONE `mesh.ply` = the prepared mesh, written whether or not `mesh.texture` is on.
- `texture/` = optional UV unwrap + projection of THAT mesh; nothing else.
- Flow in `Reconstructor.mesh()` (`collab_splats/reconstructor.py`, after the rebase onto reconstructor-release):
  1. `create_tsdf_mesh` → raw `mesh.ply`
  2. `clean_repair_mesh` (unchanged: floaters, optional hull, small fill) — result kept in memory as the **occluder**
  3. prepare: fill 3.9 → decimate → manifold → lid → manifold → overwrite `mesh.ply`
  4. if `texture`: unwrap + project prepared mesh, occluder = step-2 mesh
- Code move, no new logic:
  - the prepare lines leave `create_texture_mesh` for `mesh/clean.py`, as one function
    `prepare_mesh(mesh, *, voxel_size, max_hole_perimeter_ratio=3.9, decimate_max_error=0.5)`
    (net zero functions: `create_texture_mesh` shrinks by the same lines)
  - `create_texture_mesh(mesh, occluder, out_dir, rgbs, c2w, K, *, voxel_size, tex_size)`:
    unwrap + project + write only
- Unobserved vertices: NOT stored. The viewer derives them — invented fill/lid patches have
  no lifted point within `max_dist`, so `transfer_features` gives zeros →
  `observed = codes.any(1)` → grey. No new mesh attribute, no new file.

### Consequences to accept

- `mesh.ply` gains invented surface: interior holes < 3.9·scale filled.
  - `use_convex_hull: false` (default): no bridging; inlets touching the outer rim stay open
  - `use_convex_hull: true`: ground patched out to the hull + rim necks bridged, so inlets
    become interior holes and get filled — much more invented area
  - either way the outermost boundary (the rim / hull edge) is never lidded; that lid is the
    GH010229 watertight seal that doubled surface area
  - viewer greys all invented area; splats/dashboard consumers see it as mesh
- `mesh.ply` becomes decimated (fewer vertices) → faster viewer, coarser geometry.
- Prepare now always runs → extra mesh-stage time even with texture off. Measure once on the
  tutorial scene; report before commit.
- Vertex colors through `fill_holes`/`decimate_mesh`/`make_manifold`: must check they survive;
  if invented vertices have no color, they are grey anyway in the viewer.
- `texture/mesh.obj` vertex arrays ≠ `mesh.ply` (UV seams duplicate vertices). Same surface,
  different indexing. Viewer uses `mesh.ply` only.
- Dashboard `pipeline.py:407` calls `clean_repair_mesh(mesh_path)` and skips prepare → dashboard
  mesh stays old-style. Leave (dashboard refactor pending) — note in follow-ups.

### Files touched (cleaned modules — diff shown to user before commit)

- `collab_splats/mesh/clean.py`: + `prepare_mesh` (moved lines)
- `collab_splats/mesh/texture.py`: `create_texture_mesh` signature + body shrink
- `collab_splats/reconstructor.py`: `Reconstructor.mesh()` flow (steps 2-4), docstring
- `configs/base.yaml`: `mesh.texture` comment ("unwrap + project onto mesh.ply")
- `docs/mesh.md`: file table + pipeline section
- tests: `tests/mesh/test_texture.py` new signature; `tests/reconstructor/test_reconstructor.py`
  mesh stage asserts `mesh.ply` is the prepared mesh with texture off

## Part B — viewer

### Shape

- Library changes: small, generic. Scene glue: one example script in `docs/examples/`.
- No CLI, no reconstructor stage, no dashboard wiring (both being refactored).

### Library changes

- `ocr_lens.verbalize(..., ae: FeatureAutoencoder | None = None)`: when given, each chunk is
  decoded with `ae.per_point_decode` before the decoder. Keeps 500k × 4096 fp32 (8 GB) off
  memory; one chunk at a time. No new function.
- `ocr_lens._load_processor` → public `load_processor` (example needs the tokenizer for
  `word_vocabulary`; `AutoTokenizer` fails without sentencepiece).
- `collab_splats/viewer.py` `Viewer`, generic methods:
  - `add_mesh(name, vertices, faces, colors)` — upsert, same pattern as `add_points`
  - `add_label_list(name, labels, top_n=20)` — `labels` (V,) str per vertex; GUI buttons
    "word (count)" for the `top_n` most common; click tints that word's vertices, greys the
    rest; "clear" restores colors
  - `on_click(name, callback)` — scene click ray → nearest vertex index → `callback(i)`
  - `pick(origin, direction)` — first mesh hit of a ray → (name, vertex)
  - `highlight(name, labels, label, color)` — tint `label`'s vertices, dim the rest; None restores

### Example script `docs/examples/ocr_lens_viewer.py` (~30-40 lines)

1. args: scene backend dir, `--port`, `--chunk`
2. load `mesh.ply` (vertices, faces, colors), `pointcloud.zarr` (cameras, depth, confidence), the
   scene's lens cache `<scene>/semantics/ocr_lens.zarr`, rows via `reconstructor.store_rows`
3. per frame: patch states → `word_probabilities` (full vocabulary) → (n_words, H_p, W_p) map
4. `lift_features` straight onto the vertices, chunked, `pixel_indices=None` (decision 021);
   `observed = probs.any(1)`; unobserved vertex colors → grey, no inpainting
5. scene terms = union of observed vertices' top-10; smoothing slider `k` →
   `transfer_features(seen, seen, probs, k=k)` over observed vertices
6. beta slider (default 0.25): label score `p(w|v) / p_scene(w)^beta`, renormalized; groups by
   correlation tree; confidence cut → `viewer.add_label_list`
7. query box (comma-separated words) + top-% slider: summed raw probability, top percent of
   observed vertices → `viewer.highlight`
8. probe: clicked or screen-center vertex → top-10 bar chart of the beta-adjusted terms

### Tests

- `verbalize(codes, ae=ae)` == `verbalize(ae.per_point_decode(codes))`; chunk invariance with ae
- `Viewer.add_label_list` color arrays (tint/grey/clear) without a running server — factor the
  color logic so it is testable, or patch `viser`
- `on_click` nearest-vertex pick on a toy mesh
- no test for the example script (manual run on tutorial scene is the check)

## Decisions (user review 2026-09-30)

1. `prepare_mesh` is a moved function in `mesh/clean.py`; `clean_repair_mesh` unchanged.
2. Occluder = post-`clean_repair_mesh` mesh, kept in memory (today's behavior).
3. Dashboard `mesh.ply` staying old-style is irrelevant: dashboard unused; testing is in the viewer.
4. Smoothing runs on the codes, before decoding.
5. Example script lives in `docs/examples/`.

## Out of scope

- heatmaps, clustering, text-query search, CLI, reconstructor/dashboard viewer integration
- dashboard `_get_extractor` passing `extractor_kwargs`; stale tutorial 05/06 imports

## Remaining branch work after this

- plan (writing-plans) → build A, then B
- T12 docs: decision 021, CHANGELOG, CLAUDE.md in-flight entry, parent spec/plan updates
  (load_features redesign, inline AE reuse, overwrite fix, transfer_features move)
- full suite, whole-branch review, finish branch
