# Scene Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the semantics stage stores mesh-vertex semantics in the lifted store; `python -m collab_splats.viewer <scene>/<backend>` only reads them and launches in seconds.

**Architecture:** `write_point_features` / `read_point_features` gain vertex arrays, attrs and a `name`. `Reconstructor.semantics()` runs after mesh and, when `mesh.ply` exists, lifts onto its vertices inline: words (ocr_lens) decode then lift, codes (queryable) lift. Stores with vertex arrays record `mesh_sha256`; `done("semantics")` and the viewer both compare it to the current `mesh.ply` (no hash: done). The viewer's `_build` picks word or text mode per store.

**Tech Stack:** zarr 3, torch, open3d (stage mesh read), trimesh + viser 1.0.29 (viewer); existing `lift_features`, `word_probabilities`, `_load_frame`, `store_rows`.

Spec: `docs/superpowers/specs/2026-10-07-scene-viewer-design.md` · Decision: `docs/superpowers/decisions/023-store-vertex-semantics.md`

Replaces the phase-1 plan (`48c41343`): phase-1 commits `493ff1d0`, `667733f7`, `3d16c9ab` read `*_vertices.zarr` stores that never shipped; Tasks 6-7 rework them.

---

## Rules for every task

- Run tests from the worktree only: `cd /workspace/collab-splats/.worktrees/scene-viewer && PYTHONPATH=/workspace/collab-splats/.worktrees/scene-viewer /opt/venv/reconstruction/bin/python -m pytest ...`
- Never pipe pytest into `tail`/`head` (eats the exit code).
- Commit with `git commit --only <paths>`; message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Block comments are one plain line. Blank line around every block. No new functions beyond those named here.

## File map

| file | change |
|---|---|
| `collab_splats/semantics/store.py` | `write_point_features(..., vertex_arrays=None, attrs=None)`; `read_point_features(..., name="features")` |
| `collab_splats/reconstructor.py` | `STAGES` order; `done("semantics")` hash check; `semantics()` attrs + vertex lift inline |
| `collab_splats/viewer.py` | `_find_stores(backend_dir)`; `_load_mesh` inlined into `_build`; `_text_mode` on lifted stores; new `_word_mode`, `_chart` (moved); default selection |
| `docs/examples/ocr_lens_viewer.py` | deleted |
| `docs/mesh.md`, `configs/README.md` | viewer pointer, lifted layout, stage order |
| `tests/semantics/test_store.py`, `tests/reconstructor/test_reconstructor.py`, `tests/test_viewer.py` | below |

---

### Task 1: Lifted store holds vertex arrays and attrs

**Files:**
- Modify: `collab_splats/semantics/store.py:1-7` (module docstring), `:144-179` (`write_point_features`), `:182-217` (`read_point_features`)
- Test: `tests/semantics/test_store.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/semantics/test_store.py`:

```python
def test_write_point_features_adds_vertex_arrays_and_attrs_in_one_write(tmp_path):
    store_path = tmp_path / "ocr_lens_lifted.zarr"
    ids = np.arange(8, dtype=np.int16).reshape(4, 2)
    probs = np.full((4, 2), 0.5, np.float16)
    attrs = {"extractor": "ocr_lens", "words": ["a", "b"], "mesh_sha256": "abc"}
    vertex_arrays = {"vertex_word_ids": ids, "vertex_word_probs": probs}
    write_point_features(store_path, np.ones((3, 2), np.float32), None, vertex_arrays=vertex_arrays, attrs=attrs)

    store = zarr.open(str(store_path), mode="r")
    assert store["vertex_word_ids"].dtype == np.int16
    assert store["vertex_word_probs"].dtype == np.float16
    np.testing.assert_array_equal(store["vertex_word_ids"][:], ids)
    assert dict(store.attrs) == {"input_dim": 2, "latent_dim": 2, **attrs}
    assert not (tmp_path / "ocr_lens_lifted.zarr.tmp").exists()


def test_read_point_features_decodes_a_named_array(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    vertex = np.random.default_rng(0).random((5, 3), dtype=np.float32)
    vertex_arrays = {"vertex_features": vertex.astype(np.float16)}
    write_point_features(store_path, np.ones((2, 3), np.float32), None, vertex_arrays=vertex_arrays)

    out = read_point_features(store_path, name="vertex_features")
    assert out.shape == (5, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out, vertex / np.linalg.norm(vertex, axis=1, keepdims=True), rtol=1e-3)
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/semantics/test_store.py -k "vertex_arrays or named_array" -v`
Expected: FAIL, `TypeError: write_point_features() got an unexpected keyword argument 'vertex_arrays'`

- [ ] **Step 3: Implement** — in `collab_splats/semantics/store.py`:

Module docstring line 5 becomes:

```python
- `<extractor>_lifted.zarr`: `features` (P, latent), plus `autoencoder.pt` when compressed, plus mesh-vertex
  arrays (`vertex_word_ids` / `vertex_word_probs` or `vertex_features`) when the mesh existed
```

`write_point_features` becomes:

```python
def write_point_features(
    store_path: Path,
    codes: np.ndarray,
    ae: Optional[FeatureAutoencoder],
    vertex_arrays: Optional[dict[str, np.ndarray]] = None,
    attrs: Optional[dict[str, Any]] = None,
) -> None:
    """
    Write the lifted store atomically via a `.tmp` dir renamed into place.

    - codes stored fp16; read_point_features returns them float32
    - vertex arrays stored in the dtype they arrive in (caller picks fp16 / int16)

    Args:
        store_path: the lifted store path.
        codes: (P, latent) codes, or (P, D) when `ae` is None.
        ae: autoencoder that decodes `codes`, or None.
        vertex_arrays: mesh-vertex arrays by name, e.g. `vertex_features`; None for none.
        attrs: extra store attrs, e.g. `extractor`, `mesh_sha256`; None for none.
    """
    store_path = Path(store_path)
    tmp = store_path.with_name(f"{store_path.name}.tmp")
    codes = np.asarray(codes)
    width = int(codes.shape[1])

    store_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # Codes, vertex arrays, weights, then attrs into the tmp dir; a failure removes it
    try:
        store = zarr.open(str(tmp), mode="w")
        store["features"] = codes.astype(np.float16)

        for key, array in (vertex_arrays or {}).items():
            store[key] = array

        if ae is not None:
            ae.save(tmp / "autoencoder.pt")

        input_dim = int(ae.input_dim) if ae is not None else width
        store.attrs.update({"input_dim": input_dim, "latent_dim": width, **to_json_safe(attrs or {})})
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Swap the finished store in under its real name
    shutil.rmtree(store_path, ignore_errors=True)
    tmp.rename(store_path)
```

`read_point_features`: signature `def read_point_features(store_path: Path, batch_size: int = 65_536, name: str = "features") -> np.ndarray:`; add to `Args:` `name: array to read, `features` (points) or `vertex_features`.`; change the read line to `codes = np.asarray(store[name], dtype=np.float32)`. Nothing else changes.

- [ ] **Step 4: Run the store tests**

Run: `... -m pytest tests/semantics/test_store.py -v`
Expected: all PASS (old tests unchanged: positional callers keep working)

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/semantics/store.py tests/semantics/test_store.py -m "feat(semantics): lifted store holds vertex arrays and extra attrs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Semantics runs after mesh

**Files:**
- Modify: `collab_splats/reconstructor.py:4` (docstring), `:87-96` (`STAGES`)
- Test: `tests/reconstructor/test_reconstructor.py:913-924`

- [ ] **Step 1: Update the order test** — `test_run_calls_stages_in_order` last line becomes:

```python
    assert calls == ["preproc", "pointcloud", "mesh", "semantics"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py::test_run_calls_stages_in_order -v`
Expected: FAIL, `['preproc', 'pointcloud', 'semantics', 'mesh'] != [...]`

- [ ] **Step 3: Implement** — `STAGES` becomes:

```python
# Stage -> stages it needs; dict order is run order; semantics after mesh lifts onto its vertices
STAGES: dict[str, tuple[str, ...]] = {
    "preproc": (),
    "pointcloud": ("preproc",),
    "refine": ("pointcloud",),
    "splats": ("pointcloud",),
    "mesh": ("pointcloud",),
    "semantics": ("pointcloud",),
    "localize": ("pointcloud",),
    "reconstruction_quality_report": ("pointcloud",),
}
```

Module docstring line 4: `- stages: preproc, pointcloud, refine, splats, mesh, semantics, localize, quality report`

- [ ] **Step 4: Run the reconstructor tests**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py -q`
Expected: all PASS (`LEAF_STAGES` test recomputes from the graph; deps unchanged)

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py -m "feat(reconstructor): run semantics after mesh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `done("semantics")` is stale once `mesh.ply` changes

**Files:**
- Modify: `collab_splats/reconstructor.py` imports (`import hashlib` in stdlib group), `done()` at `:448-466`
- Test: `tests/reconstructor/test_reconstructor.py` (after `test_done_semantics_is_per_extractor`); imports add `import hashlib` and `write_point_features` to the `collab_splats.semantics.store` import

- [ ] **Step 1: Write the failing tests**

```python
def _seed_mesh(rec):
    """A two-triangle quad as the backend's mesh.ply; returns its vertices."""
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(np.array([[0, 1, 2], [1, 3, 2]]))
    )
    rec.backend_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(rec.outputs["mesh"]), mesh)
    return vertices.astype(np.float32)


def test_done_semantics_is_stale_once_mesh_ply_changes(tmp_path):
    rec = Reconstructor(_make_config(tmp_path, {"semantics": {"extractor": "dinov2"}}))
    _seed_mesh(rec)
    sha = hashlib.sha256(rec.outputs["mesh"].read_bytes()).hexdigest()
    write_point_features(rec.outputs["semantics"], np.zeros((1, 2), np.float32), None, attrs={"mesh_sha256": sha})
    assert rec.done("semantics") is True

    rec.outputs["mesh"].write_bytes(rec.outputs["mesh"].read_bytes() + b"\n")
    assert rec.done("semantics") is False


def test_done_semantics_without_a_recorded_mesh_stays_done(tmp_path):
    # Points-only store (dinov2, or lifted before any mesh): nothing on the mesh to go stale
    rec = Reconstructor(_make_config(tmp_path, {"semantics": {"extractor": "dinov2"}}))
    write_point_features(rec.outputs["semantics"], np.zeros((1, 2), np.float32), None)
    _seed_mesh(rec)
    assert rec.done("semantics") is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py -k "done_semantics" -v`
Expected: `test_done_semantics_is_stale_once_mesh_ply_changes` FAILS (`True is not False`); the no-hash test and `test_done_semantics_is_per_extractor` PASS (guards)

- [ ] **Step 3: Implement** — in `done()`, after the `pointcloud` branch, before the final `return`:

```python
        # Semantics with vertex arrays is stale when mesh.ply changed since their lift
        if stage == "semantics" and self.outputs["semantics"].exists() and self.outputs["mesh"].exists():
            recorded = zarr.open(str(self.outputs["semantics"]), mode="r").attrs.get("mesh_sha256")
            return recorded is None or recorded == hashlib.sha256(self.outputs["mesh"].read_bytes()).hexdigest()
```

Docstring `Returns:` becomes `True when the marker exists; pointcloud also needs the COLMAP model; semantics with a recorded mesh hash also needs mesh.ply to match it.`

- [ ] **Step 4: Run**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py -m "feat(reconstructor): semantics is stale once mesh.ply changes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Stage writes attrs, `mesh_sha256` and queryable vertex codes

**Files:**
- Modify: `collab_splats/reconstructor.py` imports (`BaseQueryableExtractor` into the `collab_splats.semantics.features` import), `semantics()` at `:843-911`
- Test: `tests/reconstructor/test_reconstructor.py:550-561` (attrs assertion) + new tests

- [ ] **Step 1: Write the failing tests**

`test_semantics_writes_weights_inside_the_lifted_store` attrs assertion becomes:

```python
    assert dict(store.attrs) == {"input_dim": 32, "latent_dim": 8, "extractor": "dinov2", "extractor_kwargs": {}}
```

New, after `_seed_mesh`:

```python
def _mesh_semantics_rec(tmp_path, extractor, extractor_kwargs=None, dim=4):
    """Uncompressed codes seeded (no extraction), a one-point reconstruction and a quad mesh.ply."""
    semantics = {"enabled": True, "extractor": extractor, "extractor_kwargs": extractor_kwargs or {}, "n_components": None}
    rec = _semantics_rec(tmp_path, semantics)
    _seed_codes(rec, extractor, [np.ones((dim, 2, 2), np.float16)] * 2, extractor_kwargs=extractor_kwargs)
    return rec, _seed_mesh(rec)


def _arange_lift(calls):
    """lift_features stub: records (result, num_classes), returns arange rows of the lifted width."""

    def lift(frame_features, result, num_classes=None):
        calls.append((result, num_classes))
        width = num_classes or frame_features(0).shape[0]
        return torch.arange(len(result.points) * width, dtype=torch.float32).reshape(-1, width)

    return lift


def test_semantics_lifts_queryable_codes_onto_mesh_vertices(tmp_path):
    rec, vertices = _mesh_semantics_rec(tmp_path, "talk2dino")
    calls = []

    with patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    # Second lift is the vertices, over the same cameras, no source pixel
    vertex_cloud, _ = calls[1]
    np.testing.assert_array_equal(vertex_cloud.points, vertices)
    assert vertex_cloud.pixel_indices is None

    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert store["vertex_features"].dtype == np.float16
    np.testing.assert_array_equal(store["vertex_features"][:], np.arange(16, dtype=np.float16).reshape(4, 4))
    assert store.attrs["mesh_sha256"] == hashlib.sha256(rec.outputs["mesh"].read_bytes()).hexdigest()
    assert rec.done("semantics")


def test_semantics_without_vertex_mode_writes_points_only(tmp_path):
    rec, _ = _mesh_semantics_rec(tmp_path, "dinov2")
    calls = []

    with patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert len(calls) == 1
    assert "vertex_features" not in store and "vertex_word_ids" not in store
    assert "mesh_sha256" not in store.attrs
    assert rec.done("semantics")


def test_semantics_without_mesh_writes_points_only(tmp_path):
    rec = _semantics_rec(tmp_path, {"extractor": "talk2dino", "n_components": None})
    _seed_codes(rec, "talk2dino", [np.ones((4, 2, 2), np.float16)] * 2)
    calls = []

    with patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert len(calls) == 1 and "vertex_features" not in store and "mesh_sha256" not in store.attrs
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py -k "semantics" -v`
Expected: the attrs test and the three new tests FAIL (no attrs, no vertex lift)

- [ ] **Step 3: Implement** — replace the last block of `semantics()` (from `# Write the per-point codes` to the end) with:

```python
        # Store attrs; the viewer rebuilds a queryable extractor from them
        attrs = {"extractor": name, "extractor_kwargs": extractor_kwargs}
        vertex_arrays = {}
        mesh_path = self.outputs["mesh"]

        # Mesh vertices as points over the same cameras; no source pixel, unseen vertices stay zero
        if mesh_path.exists():
            vertices = np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices, dtype=np.float32)
            colors = np.zeros((len(vertices), 3), dtype=np.uint8)
            vertex_cloud = dataclasses.replace(pointcloud, points=vertices, colors=colors, pixel_indices=None)

            # Queryable: codes lifted like the points, decoded at read
            if issubclass(BaseFeatureExtractor.get(name), BaseQueryableExtractor):
                vertex_codes = lift_features(partial(_load_frame, codes, rows, None), vertex_cloud)
                vertex_arrays["vertex_features"] = to_numpy(vertex_codes).astype(np.float16)

            # Record the mesh the vertex arrays index into
            if vertex_arrays:
                attrs["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()

        # Write points, vertex arrays and attrs together; the lifted store is the stage's done marker
        write_point_features(self.outputs["semantics"], to_numpy(lifted), ae, vertex_arrays=vertex_arrays, attrs=attrs)
```

Docstring bullets of `semantics()` gain:

```python
        - with mesh.ply: queryable extractors also store `vertex_features` (codes on vertices) and the mesh's sha256
```

- [ ] **Step 4: Run**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py tests/semantics -q`
Expected: all PASS (the `_stub_extraction` tests have no mesh.ply, so `BaseFeatureExtractor.get` stays unmocked-safe)

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py -m "feat(semantics): lift queryable codes onto mesh vertices with their mesh_sha256

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Stage stores ocr_lens vertex words (inline decode + one lift)

**Files:**
- Modify: `collab_splats/reconstructor.py` imports, `semantics()` vertex block from Task 4
- Test: `tests/reconstructor/test_reconstructor.py` (new)

- [ ] **Step 1: Write the failing test**

```python
@contextlib.contextmanager
def _stub_lens(n_words=70):
    """Processor, vocabulary, decoder and word_probabilities stubbed; every patch is uniform over n_words."""
    vocab = MagicMock()
    vocab.words = [f"w{i}" for i in range(n_words)]

    def probabilities(states, decoder, vocab, ae=None):
        yield torch.full((len(states), n_words), 1.0 / n_words), None

    with (
        patch.object(R, "load_processor") as processor,
        patch.object(R, "word_vocabulary", return_value=vocab),
        patch.object(R, "load_decoder") as decoder,
        patch.object(R, "word_probabilities", side_effect=probabilities) as decode,
    ):
        yield processor, decoder, decode


def test_semantics_stores_each_vertex_top_64_words(tmp_path):
    rec, _ = _mesh_semantics_rec(tmp_path, "ocr_lens", extractor_kwargs={"model_id": "local/llava"})
    calls = []

    with _stub_lens() as (processor, decoder, decode), patch.object(R, "lift_features", side_effect=_arange_lift(calls)):
        rec.semantics()

    # The extracting checkpoint's lens; one decode per frame; one word lift over the vocabulary
    processor.assert_called_once_with("local/llava")
    decoder.assert_called_once_with("local/llava")
    assert decode.call_count == 2
    assert calls[1][1] == 70

    # Each vertex keeps the top-64 of its lifted row, descending
    expected = torch.arange(4 * 70, dtype=torch.float32).reshape(4, 70).topk(64, dim=1)
    store = zarr.open(str(rec.outputs["semantics"]), mode="r")
    assert store["vertex_word_ids"].dtype == np.int16 and store["vertex_word_probs"].dtype == np.float16
    np.testing.assert_array_equal(store["vertex_word_ids"][:], expected.indices.numpy())
    np.testing.assert_array_equal(store["vertex_word_probs"][:], expected.values.numpy().astype(np.float16))
    assert store.attrs["words"] == [f"w{i}" for i in range(70)]
    assert "vertex_features" not in store
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py::test_semantics_stores_each_vertex_top_64_words -v`
Expected: FAIL, `AttributeError: <module 'collab_splats.reconstructor'> does not have the attribute 'load_processor'`

- [ ] **Step 3: Implement**

`reconstructor.py` imports, after the `collab_splats.semantics.features` import:

```python
from collab_splats.semantics.features.ocr_lens import (
    load_decoder,
    load_processor,
    word_probabilities,
    word_vocabulary,
)
```

In `semantics()`, the `# Queryable:` `if` from Task 4 becomes an `elif` under a new first branch:

```python
            # ocr_lens: decode each frame to its top-64 words per patch, lift the indexed maps, keep each vertex's top-64
            if name == "ocr_lens":
                model_id = extractor_kwargs.get("model_id", "llava-hf/llava-v1.6-vicuna-7b-hf")
                vocab = word_vocabulary(load_processor(model_id).tokenizer)
                decoder = load_decoder(model_id)

                if ae is not None:
                    ae.to(get_device())

                maps = []

                for i in range(len(rows)):
                    # Patch codes as rows, then word probabilities per patch, top-64 kept as an indexed map
                    fmap = _load_frame(codes, rows, None, i)
                    channels, height, width = fmap.shape
                    states = fmap.reshape(channels, -1).T
                    probs = torch.cat([p for p, _ in word_probabilities(states, decoder, vocab, ae=ae)])
                    top = probs.topk(64, dim=1)
                    maps.append((top.indices.T.reshape(64, height, width), top.values.T.reshape(64, height, width)))

                del decoder
                pytorch_gc()

                # One lift, no chunking: ~17 KB per vertex, CUDA OOM past ~2.4M vertices on an A40
                lifted_words = lift_features(maps.__getitem__, vertex_cloud, num_classes=len(vocab.words))
                top = lifted_words.to(get_device()).topk(64, dim=1)
                vertex_arrays["vertex_word_ids"] = to_numpy(top.indices).astype(np.int16)
                vertex_arrays["vertex_word_probs"] = to_numpy(top.values).astype(np.float16)
                attrs["words"] = vocab.words
                del maps, lifted_words

            # Queryable: codes lifted like the points, decoded at read
            elif issubclass(BaseFeatureExtractor.get(name), BaseQueryableExtractor):
```

Docstring bullet from Task 4 becomes:

```python
        - with mesh.ply: ocr_lens also stores each vertex's top-64 words (decode, then lift), queryable
          extractors `vertex_features` (codes on vertices); both record the mesh's sha256
```

- [ ] **Step 4: Run**

Run: `... -m pytest tests/reconstructor/test_reconstructor.py tests/semantics -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_reconstructor.py -m "feat(semantics): store each mesh vertex's top-64 ocr_lens words

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Viewer reads lifted stores; `_load_mesh` inlined; default selection

**Files:**
- Modify: `collab_splats/viewer.py` imports (add `hashlib`; drop `BaseQueryableExtractor`), module docstring line 6, `_load_mesh` (deleted, body into `_build`), `_find_stores`, `_text_mode`, `_build`
- Test: `tests/test_viewer.py:17-22` (imports), `:303-431` (store helpers + mode tests)

- [ ] **Step 1: Rewrite the tests**

Imports: add `import hashlib`; `from collab_splats.viewer import Viewer, _build, _find_stores` (drop `_load_mesh`). Replace `_vertex_store`, `test_load_mesh_...` and the three `_find_stores` tests with:

```python
def _sha(backend):
    return hashlib.sha256((backend / "mesh.ply").read_bytes()).hexdigest()


def _vertex_store(backend, extractor, codes, kwargs=None, mesh_sha256=None):
    """Lifted store with full-width vertex codes, lifted onto the backend's mesh.ply unless told otherwise."""
    path = backend / "semantics" / f"{extractor}_lifted.zarr"
    attrs = {"extractor": extractor, "extractor_kwargs": kwargs or {}, "mesh_sha256": mesh_sha256 or _sha(backend)}
    vertex_arrays = {"vertex_features": np.asarray(codes, np.float16)}
    write_point_features(path, np.ones((1, 2), np.float32), None, vertex_arrays=vertex_arrays, attrs=attrs)
    return path


def _word_store(backend, word_ids, word_probs, words):
    """ocr_lens lifted store with per-vertex word ids + probs."""
    path = backend / "semantics" / "ocr_lens_lifted.zarr"
    attrs = {"extractor": "ocr_lens", "extractor_kwargs": {}, "mesh_sha256": _sha(backend), "words": words}
    vertex_arrays = {
        "vertex_word_ids": np.asarray(word_ids, np.int16),
        "vertex_word_probs": np.asarray(word_probs, np.float16),
    }
    write_point_features(path, np.ones((1, 2), np.float32), None, vertex_arrays=vertex_arrays, attrs=attrs)
    return path


def test_find_stores_without_semantics_is_empty(tmp_path):
    assert _find_stores(_backend(tmp_path)) == {}


def test_find_stores_lists_vertex_stores_only(tmp_path):
    backend = _backend(tmp_path)
    text = _vertex_store(backend, "stub_text", np.eye(4, 2))
    words = _word_store(backend, [[0], [0], [0], [0]], [[1.0]] * 4, ["a"])
    write_point_features(backend / "semantics" / "dinov2_lifted.zarr", np.ones((1, 2), np.float32), None)

    assert _find_stores(backend) == {"ocr_lens": words, "stub_text": text}


def test_find_stores_skips_a_store_lifted_onto_another_mesh(tmp_path, caplog):
    backend = _backend(tmp_path)
    _vertex_store(backend, "stub_text", np.eye(4, 2), mesh_sha256="old")

    with caplog.at_level(logging.WARNING):
        assert _find_stores(backend) == {}

    assert "re-run semantics" in caplog.text
```

After `_record_heat`, add:

```python
def test_build_greys_a_mesh_without_vertex_colors(tmp_path, fresh_viewer, monkeypatch):
    added = []
    monkeypatch.setattr(fresh_viewer, "add_mesh", lambda name, v, f, c, textured=None: added.append((v, f, c, textured)))
    _build(fresh_viewer, _backend(tmp_path), textured=False, texture_size=64)

    vertices, faces, colors, textured = added[0]
    assert vertices.shape == (4, 3) and faces.shape == (2, 3)
    assert (colors == 200).all() and textured is None
```

Existing text-mode tests: the only store is now selected by default, so delete each `dropdown.value = "stub_text"` line; in `test_text_mode_heat_...` the expected line reads `read_point_features(path, name="vertex_features")`; in `test_switching_to_none_...` replace `dropdown.value = "stub_text"` with `assert dropdown.value == "stub_text"`.

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_viewer.py -v`
Expected: FAIL, `_find_stores() missing 1 required positional argument` / lists `*_vertices.zarr` only

- [ ] **Step 3: Implement** — in `collab_splats/viewer.py`:

Module docstring line 6: ``- `python -m collab_splats.viewer <scene>/<backend>`: the mesh, plus a Semantics dropdown over its lifted stores' vertex arrays``

Delete `_load_mesh`. `_find_stores` becomes:

```python
def _find_stores(backend_dir: Path) -> dict[str, Path]:
    """
    Lifted stores holding mesh-vertex arrays, extractor name -> store path.

    - listed: `<backend>/semantics/*_lifted.zarr` with `vertex_word_ids` or `vertex_features`
    - a store lifted onto another mesh.ply (`mesh_sha256` differs) is stale and skipped
    """
    mesh_sha256 = hashlib.sha256((backend_dir / "mesh.ply").read_bytes()).hexdigest()
    stores = {}

    for path in sorted((backend_dir / "semantics").glob("*_lifted.zarr")):
        store = zarr.open(str(path), mode="r")

        # Points-only stores (dinov2) have nothing on the mesh
        if "vertex_word_ids" not in store and "vertex_features" not in store:
            continue

        # Stale against the mesh: skip, a semantics re-run rewrites it
        if store.attrs.get("mesh_sha256") != mesh_sha256:
            logger.warning("%s was lifted onto another mesh.ply; re-run semantics with that extractor", path)
            continue

        stores[store.attrs["extractor"]] = path

    return stores
```

`_text_mode`: docstring summary `Text query GUI over a queryable extractor's lifted store; returns its handles.`; the two reads become:

```python
    observed = np.asarray(store["vertex_features"]).any(axis=1)
    features = torch.from_numpy(read_point_features(store_path, name="vertex_features"))
```

`_build` becomes:

```python
def _build(viewer: Viewer, backend_dir: Path, textured: bool, texture_size: int) -> Optional[viser.GuiDropdownHandle]:
    """
    Add the mesh and, when vertex stores exist, the Semantics dropdown that switches modes.

    - default selection: ocr_lens when listed, else the first by name; `none` clears
    - switching clears heat and probe, removes the previous mode's GUI and handlers, builds the new one

    Args:
        viewer: viewer to populate.
        backend_dir: `<scene>/<backend>` holding mesh.ply.
        textured: show texture/mesh.obj; picks stay on mesh.ply.
        texture_size: displayed texture edge, pixels.

    Returns:
        The Semantics dropdown, or None when no store has vertex arrays.
    """
    # mesh.ply with its vertex colors, light grey when it has none
    mesh = trimesh.load(backend_dir / "mesh.ply", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    if mesh.visual.kind == "vertex":
        colors = np.array(mesh.visual.vertex_colors[:, :3])
    else:
        colors = np.full((len(vertices), 3), 200, dtype=np.uint8)

    # Corner-split OBJ of the same surface, when asked; texture downscaled for display
    shown = None

    if textured:
        shown = trimesh.load(backend_dir / "texture" / "mesh.obj", process=False)
        material = shown.visual.material
        material.image = material.image.resize((texture_size, texture_size), Image.LANCZOS)

    viewer.add_mesh("mesh", vertices, faces, colors, textured=shown)
    stores = _find_stores(backend_dir)

    if not stores:
        return None

    initial = "ocr_lens" if "ocr_lens" in stores else next(iter(stores))
    dropdown = viewer.server.gui.add_dropdown("Semantics", options=("none", *stores), initial_value=initial)
    handles = []

    def switch(_=None) -> None:
        """
        Tear down the current mode, then build the selected one.
        """
        # Heat, probe, label list and click handler of the previous mode
        viewer.show_heat("mesh", None, 0.0)
        viewer.server.scene.remove_by_name("/probe")
        viewer.mesh_clicks.pop("mesh", None)

        if "mesh" in viewer.label_lists:
            viewer.label_lists.pop("mesh")[0].remove()

        for handle in handles:
            handle.remove()

        handles.clear()
        pytorch_gc()

        if dropdown.value == "none":
            return

        handles.extend(_text_mode(viewer, stores[dropdown.value]))

    dropdown.on_update(switch)
    switch()
    return dropdown
```

Task 6 dispatches to `_text_mode` only; Task 7 adds the word-mode dispatch. Task 6 `_build` tests use text stores only.

- [ ] **Step 4: Run**

Run: `... -m pytest tests/test_viewer.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/viewer.py tests/test_viewer.py -m "feat(viewer): read vertex arrays from lifted stores; default selection; mesh load inline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Word mode — query heat, label list, click probe

**Files:**
- Modify: `collab_splats/viewer.py` (new `_chart` under Helpers, moved verbatim from `docs/examples/ocr_lens_viewer.py:111-148`; new `_word_mode` above `_text_mode`)
- Test: `tests/test_viewer.py` (new; import `import collab_splats.viewer as viewer_module`)

- [ ] **Step 1: Write the failing tests**

```python
WORDS = ["apple", "tree", "sky"]


def _word_backend(tmp_path):
    """Quad backend with an ocr_lens store; vertex 3 unobserved."""
    backend = _backend(tmp_path)
    ids = [[0, 1], [1, 2], [2, 0], [0, 1]]
    probs = [[0.75, 0.25], [0.5, 0.5], [1.0, 0.0], [0.0, 0.0]]
    _word_store(backend, ids, probs, WORDS)
    return backend


def test_build_selects_ocr_lens_by_default(tmp_path, fresh_viewer):
    backend = _word_backend(tmp_path)
    _vertex_store(backend, "stub_text", np.eye(4, 2))
    dropdown = _build(fresh_viewer, backend, textured=False, texture_size=64)

    assert dropdown.value == "ocr_lens"
    assert set(dropdown.options) == {"none", "ocr_lens", "stub_text"}
    assert "Query min p" in _inputs(fresh_viewer)


def test_word_mode_heat_is_the_summed_probability_of_the_query_words(tmp_path, fresh_viewer, monkeypatch):
    calls = _record_heat(fresh_viewer, monkeypatch)
    _build(fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64)
    inputs = _inputs(fresh_viewer)
    inputs["Query"].value = "apple, Sky, nope"
    inputs["Search"]._impl.update_cb[0](None)

    name, scores, floor = calls[-1]
    assert name == "mesh" and floor == inputs["Query min p"].value
    np.testing.assert_allclose(scores, [0.75, 0.5, 1.0, 0.0])


def test_word_mode_label_list_ranks_words_by_probability_mass(tmp_path, fresh_viewer):
    _build(fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64)

    _, buttons = fresh_viewer.label_lists["mesh"]
    assert [b.label for b in buttons] == ["Clear", "sky (~2)", "apple (~1)", "tree (~1)"]


def test_word_mode_probe_charts_scene_terms_and_flags_unobserved(tmp_path, fresh_viewer, monkeypatch):
    charts = []
    monkeypatch.setattr(viewer_module, "_chart", lambda title, words, probs: charts.append((title, list(words), probs)) or "")
    _build(fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64)

    fresh_viewer.mesh_clicks["mesh"](1)
    fresh_viewer.mesh_clicks["mesh"](3)

    assert charts[0][:2] == ("vertex 1", ["tree", "sky"])
    np.testing.assert_allclose(charts[0][2], [0.5, 0.5])
    assert charts[1][0] == "vertex 3: unobserved"


def test_switching_away_from_word_mode_removes_labels_clicks_and_probe(tmp_path, fresh_viewer):
    dropdown = _build(fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64)
    fresh_viewer.mesh_clicks["mesh"](0)
    dropdown.value = "none"

    assert "mesh" not in fresh_viewer.label_lists and "mesh" not in fresh_viewer.mesh_clicks
    assert "/probe" not in fresh_viewer.server.scene._handle_from_node_name
    assert not {"Query", "Query min p", "Search"} & set(_inputs(fresh_viewer))
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_viewer.py -k "word or ocr_lens" -v`
Expected: FAIL, `NameError: name '_word_mode' is not defined`

- [ ] **Step 3: Implement** — in `_build`'s `switch`, the dispatch line becomes:

```python
        # Word arrays select word mode, codes text mode
        path = stores[dropdown.value]
        mode = _word_mode if "vertex_word_ids" in zarr.open(str(path), mode="r") else _text_mode
        handles.extend(mode(viewer, path))
```

Move `_chart` verbatim from the example into the Helpers section of `viewer.py` (below `_compact_count`). Add above `_text_mode`:

```python
def _word_mode(viewer: Viewer, store_path: Path) -> list:
    """
    Word query, mass-ranked label list and click probe over an ocr_lens lifted store; returns its GUI handles.

    - heat: per vertex, summed probability of the query words within its stored top-k
    - unobserved vertices (top probability 0) score 0 and probe as unobserved
    - probe: the clicked vertex's top-10 scene terms (words in some observed vertex's top-10), renormalized
    """
    store = zarr.open(str(store_path), mode="r")
    word_ids = np.asarray(store["vertex_word_ids"])
    word_probs = np.asarray(store["vertex_word_probs"], dtype=np.float32)
    words = list(store.attrs["words"])
    row_of = {word: row for row, word in enumerate(words)}
    observed = word_probs[:, 0] > 0

    # Scene terms: every word in some observed vertex's top-10
    is_term = np.zeros(len(words), dtype=bool)
    is_term[np.unique(word_ids[observed, :10])] = True

    # Probe chart, query box, floor, Search and the unknown-word note
    panel = viewer.server.gui.add_html("")
    query = viewer.server.gui.add_text("Query", initial_value="")
    floor = viewer.server.gui.add_slider("Query min p", min=0.0, max=1.0, step=0.01, initial_value=0.3)
    search_button = viewer.server.gui.add_button("Search")
    note = viewer.server.gui.add_markdown("")
    vertices = viewer.meshes["mesh"][0]
    marker_radius = 0.005 * float(np.linalg.norm(np.ptp(vertices, axis=0)))

    def search(_=None) -> None:
        """
        Draw the summed probability of the query words as heat; faces under the floor hidden.
        """
        typed = [word.lower() for word in _split(query.value)]
        known = [word for word in typed if word in row_of]
        unknown = [word for word in typed if word not in row_of]
        note.content = f"not in vocabulary: {', '.join(unknown)}" if unknown else ""

        # Empty query clears; a word outside a vertex's top-k reads as 0 there
        scores = None

        if known:
            hit = np.isin(word_ids, [row_of[word] for word in known])
            scores = (word_probs * hit).sum(axis=1)

        viewer.show_heat("mesh", scores, floor.value)

    def select(word: Optional[str]) -> None:
        """
        Put a label-list word in the query box (Clear empties it) and search.
        """
        query.value = word or ""
        search()

    def show(vertex: int) -> None:
        """
        Mark the clicked vertex and chart its top-10 scene terms in the top left.
        """
        viewer.server.scene.add_icosphere(
            "/probe", radius=marker_radius, color=(255, 0, 255), position=vertices[vertex]
        )

        if not observed[vertex]:
            panel.content = _chart(f"vertex {vertex}: unobserved", [], np.zeros(0))
            return

        # Scene terms among its stored words, already sorted, renormalized
        ids = word_ids[vertex]
        probs = word_probs[vertex]
        keep = is_term[ids] & (probs > 0)
        ids = ids[keep]
        probs = probs[keep] / probs[keep].sum()
        panel.content = _chart(f"vertex {vertex}", [words[j] for j in ids[:10]], probs[:10])

    # Words ranked by probability mass (expected vertex count); a click on the mesh probes
    mass = np.bincount(word_ids.ravel(), weights=word_probs.ravel(), minlength=len(words))
    viewer.add_label_list("mesh", words, mass, select)
    search_button.on_click(search)
    viewer.on_click("mesh", show)
    return [panel, query, floor, search_button, note]
```

- [ ] **Step 4: Run**

Run: `... -m pytest tests/test_viewer.py -v`
Expected: all PASS, including `test_viewer_module_never_imports_the_reconstructor`

- [ ] **Step 5: Commit**

```bash
git commit --only collab_splats/viewer.py tests/test_viewer.py -m "feat(viewer): word mode over stored ocr_lens vertex words

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Retire the example; docs

**Files:**
- Delete: `docs/examples/ocr_lens_viewer.py`
- Modify: `docs/mesh.md:281`, `configs/README.md:42-46` (stage paragraph), `:70-71` and `:750-751` (layout blocks), `:768` (outputs table)

- [ ] **Step 1: Check nothing else imports the example**

Run: `grep -rn "ocr_lens_viewer" --include=*.py --include=*.md --include=*.ipynb . | grep -v docs/superpowers`
Expected: only `docs/mesh.md:281`

- [ ] **Step 2: Edit**

`git rm docs/examples/ocr_lens_viewer.py`

`docs/mesh.md:281` becomes:

```markdown
The semantics stage (after mesh) stores this lift in `<extractor>_lifted.zarr`;
`python -m collab_splats.viewer <scene>/<backend>` reads it.
```

`configs/README.md:70-71` (first layout block) — keep both lines, add a third:

```
      <extractor>_lifted.zarr  ← lifted 3D features (N_points × latent_dim)
                                 (+ autoencoder.pt inside if semantics.n_components set)
                                 (+ mesh-vertex arrays + mesh_sha256 attr, if mesh.ply existed)
```

`configs/README.md:750-751` (second layout block) — same third line under its two.

`configs/README.md:768` — new row under it:

```markdown
| `<backend>/semantics/<extractor>_lifted.zarr` vertex arrays | per-mesh-vertex: ocr_lens `vertex_word_ids` + `vertex_word_probs` (top-64), maskclip/talk2dino `vertex_features` codes; read by `python -m collab_splats.viewer` |
```

`configs/README.md`, end of the stage paragraph that starts at `:42` (after its last line, before the blank line), add:

```markdown
`semantics` runs after `mesh` and also lifts onto `mesh.ply`'s vertices; after a mesh rebuild
it re-runs from the cached codes (the store's `mesh_sha256` no longer matches).
```

- [ ] **Step 3: Commit**

```bash
git commit --only docs/examples/ocr_lens_viewer.py docs/mesh.md configs/README.md -m "docs: scene viewer replaces the ocr_lens example; lifted store vertex arrays

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Gate and GH010229 check

- [ ] **Step 1: Full gate**

Run: `cd /workspace/collab-splats/.worktrees/scene-viewer && PYTHONPATH=/workspace/collab-splats/.worktrees/scene-viewer /opt/venv/reconstruction/bin/python -m pytest tests/ -q` (in tmux; check `gsplat.__version__` first)
Expected: no failures beyond `docs/known-test-failures.md`; compare against `clean/final` run of the same command

- [ ] **Step 2: GH010229 re-run (tmux)** — semantics leaf with ocr_lens, then maskclip, after mesh; record stage time and store sizes (sum file bytes, not `du`)

- [ ] **Step 3: Browser check** — `python -m collab_splats.viewer <scene>/<backend>`: launch in seconds, ocr_lens selected, word query + label list + probe; switch to maskclip, text query; a backend without semantics shows the mesh only

- [ ] **Step 4: Report** results to the user before any merge or CHANGELOG entry
