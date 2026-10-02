# Semantics Store Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store IO moves to `semantics/store.py` with explicit paths, the lifted store holds its own `autoencoder.pt` and is written atomically, and `mesh_clustering` becomes `semantics.utils.cluster_points`.

**Architecture:** `Reconstructor.outputs["semantics"]` is the one spelling of the lifted store path; store functions take paths and never build or glob them. `utils.py` keeps feature math only.

**Tech Stack:** zarr 3, torch, scipy (cKDTree, csgraph), pytest.

Spec: [2026-10-01-semantics-store-cleanup-design.md](../specs/2026-10-01-semantics-store-cleanup-design.md).
Rules: decision 017; `tests/test_docstring_contract.py`; `tests/test_import_style.py`; one-line block
comments; no nested calls; blank lines around blocks; flat tests; US spelling.

**Environment:** worktree `/workspace/collab-splats/.worktrees/ocr-lens`, every command with
`PYTHONPATH=<worktree> HF_HOME=/workspace/models HF_HUB_OFFLINE=1 /opt/venv/reconstruction/bin/python`.

**Gate** (per task, no `-x`, compare pass/skip counts to the baseline):

```bash
python -c "import collab_splats; print(collab_splats.__file__)"   # must print the worktree path
python -m pytest tests/semantics tests/reconstructor tests/mesh tests/test_docstring_contract.py \
  tests/test_import_style.py tests/test_cu121_migration.py --ignore=tests/reconstructor/test_cli.py -q -p no:cacheprovider
```

Baseline (2026-10-01, `99a6983d`): 1616 passed, 80 xpassed, 0 skipped. `test_cli.py` is ignored: it fails
collection on a venv issue (`collab_data` has no `STATS_ARGS`), unrelated to this work.

**Format:** `black --target-version py311 -l 120` + `isort` on touched `.py` files only.
After each task: `graphify update .`. Show the diff to the user before each commit.

---

## File map

| File | Change |
|---|---|
| `collab_splats/semantics/utils.py` | keep `compute_semantic_contrast`, `_tokens_to_feature_map`; add `cluster_points`; drop all store code |
| `collab_splats/semantics/store.py` | NEW: `valid_feature_cache`, `extract_feature_cache` (moved unchanged), `write_point_features`, `load_point_features` (changed) |
| `collab_splats/mesh/features.py` | DELETE |
| `collab_splats/mesh/__init__.py`, `collab_splats/semantics/__init__.py` | exports |
| `collab_splats/reconstructor.py` | `outputs["semantics"]` spelled inline; `semantics()` passes it to the writer |
| `tests/mesh/test_features.py` | DELETE; tests move to `tests/semantics/test_semantics_utils.py` |
| `tests/semantics/test_store.py` | NEW: store tests from `test_semantics_utils.py` + `test_artifact_layout.py` |
| `tests/semantics/test_artifact_layout.py` | DELETE |
| `tests/semantics/features/test_extract_from_zarr.py` | import from `store` |
| `tests/reconstructor/test_reconstructor.py` | AE inside lifted store; explicit paths in `test_done_semantics_is_per_extractor` |
| `tests/test_cu121_migration.py` | module list: `-mesh.features`, `+semantics.store` |
| `docs/mesh.md`, `docs/source/api/mesh.rst`, `configs/README.md` | layout prose |
| `docs/examples/ocr_lens_viewer.py` | explicit `<scene>/semantics/ocr_lens.zarr` |
| `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`, `06_mesh/splats_mesh.ipynb` | `ae_path` → `<store>/autoencoder.pt`; `cluster_points` |

---

### Task 1: `cluster_points` moves to `semantics/utils.py`

**Files:**
- Modify: `collab_splats/semantics/utils.py`, `collab_splats/semantics/__init__.py`, `collab_splats/mesh/__init__.py`
- Delete: `collab_splats/mesh/features.py`, `tests/mesh/test_features.py`
- Modify: `tests/semantics/test_semantics_utils.py`, `tests/test_cu121_migration.py`
- Modify: `docs/mesh.md`, `docs/source/api/mesh.rst`, `docs/source/tutorials/06_mesh/splats_mesh.ipynb`

- [ ] **Step 1: Move the tests (failing)**

Append to `tests/semantics/test_semantics_utils.py`, add `cluster_points` to its `collab_splats.semantics.utils` import, delete `tests/mesh/test_features.py`:

```python
########################################################################
# cluster_points
########################################################################


def _two_blobs():
    """60 points in two tight blobs 1 unit apart plus 10 scattered low-similarity points."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.005, (30, 3))
    b = rng.normal(0.0, 0.005, (30, 3)) + [1.0, 0.0, 0.0]
    stray = rng.random((10, 3)) * [0.5, 1.0, 1.0] + [0.25, 0.0, 0.0]
    points = np.vstack([a, b, stray])
    similarity = np.r_[np.ones(60), np.zeros(10)]
    return points, similarity


def test_cluster_points_groups_nearby_high_similarity_points():
    points, similarity = _two_blobs()
    clusters = cluster_points(points, similarity, similarity_threshold=0.5, spatial_radius=0.05, min_cluster_size=10)
    assert len(clusters) == 2
    assert {frozenset(c.tolist()) for c in clusters} == {frozenset(range(30)), frozenset(range(30, 60))}


def test_cluster_points_min_cluster_size_drops_small_clusters():
    points, similarity = _two_blobs()
    assert cluster_points(points, similarity, similarity_threshold=0.5, spatial_radius=0.05, min_cluster_size=31) == []


def test_cluster_points_no_valid_points_returns_empty_list():
    points, similarity = _two_blobs()
    assert cluster_points(points, np.zeros_like(similarity)) == []
```

- [ ] **Step 2: Run, expect ImportError**

`python -m pytest tests/semantics/test_semantics_utils.py -q -k cluster_points` → `ImportError: cannot import name 'cluster_points'`.

- [ ] **Step 3: Implement**

In `collab_splats/semantics/utils.py`: add imports `from scipy.sparse import csr_matrix`,
`from scipy.sparse.csgraph import connected_components`, `from scipy.spatial import cKDTree`;
add `"cluster_points"` to `__all__`; add after `compute_semantic_contrast`:

```python
def cluster_points(
    points: np.ndarray,
    similarity_values: np.ndarray,
    similarity_threshold: float = 0.8,
    spatial_radius: float = 0.03,
    min_cluster_size: int = 10,
) -> list[np.ndarray]:
    """
    Group spatially connected points whose similarity exceeds a threshold.

    - any point set: pointcloud points or mesh vertices (`np.asarray(mesh.vertices)`)

    Args:
        points: (N, 3) positions.
        similarity_values: (N,) per-point similarity.
        similarity_threshold: points with similarity above this are candidates.
        spatial_radius: candidates within this distance (world units) are connected.
        min_cluster_size: clusters with fewer points are dropped.

    Returns:
        (n_i,) int arrays of point indices, one per cluster.
    """
    similarity_values = np.asarray(similarity_values)
    valid = np.flatnonzero(similarity_values > similarity_threshold)

    if len(valid) == 0:
        return []

    # Sparse adjacency from all candidate pairs within the radius
    xyz = np.asarray(points)[valid]
    pairs = cKDTree(xyz).query_pairs(spatial_radius, output_type="ndarray")
    n = len(valid)
    adjacency = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)

    # Map component labels back to original point indices; drop small clusters
    clusters = []

    for label in np.unique(labels):
        members = valid[labels == label]

        if len(members) >= min_cluster_size:
            clusters.append(members)

    return clusters
```

Delete `collab_splats/mesh/features.py`. In `mesh/__init__.py` drop the `features` docstring
bullet, the import and the `__all__` entry. In `semantics/__init__.py` import and export
`cluster_points`.

Callers:
- `tests/test_cu121_migration.py`: delete `"collab_splats.mesh.features",`.
- `docs/source/api/mesh.rst`: delete the `collab_splats.mesh.features` automodule block.
- `docs/mesh.md:12`: delete the `features.py` row; `:257`: `` `mesh_clustering` `` →
  `` `collab_splats.semantics.utils.cluster_points` (on the vertex positions) ``.
- `06_mesh/splats_mesh.ipynb`: drop `mesh_clustering` from the `collab_splats.mesh` import;
  `from collab_splats.semantics.utils import ae_path` → `... import ae_path, cluster_points`
  (Task 2 drops `ae_path`); `mesh_clustering(query_mesh_src, scores, ...)` →
  `cluster_points(np.asarray(query_mesh_src.vertices), scores, ...)`; markdown
  `` `mesh_clustering` `` → `` `cluster_points` ``.

- [ ] **Step 4: Gate** — expect baseline counts (3 tests moved, none lost).

- [ ] **Step 5: Format, graphify, show diff, commit**

```bash
git rm -q collab_splats/mesh/features.py tests/mesh/test_features.py
git commit --only <touched files> -m "refactor(semantics): mesh_clustering -> semantics.utils.cluster_points over any point set"
```

---

### Task 2: `semantics/store.py`

**Files:**
- Create: `collab_splats/semantics/store.py`, `tests/semantics/test_store.py`
- Modify: `collab_splats/semantics/utils.py`, `collab_splats/semantics/__init__.py`, `collab_splats/reconstructor.py`
- Delete: `tests/semantics/test_artifact_layout.py`
- Modify: `tests/semantics/test_semantics_utils.py`, `tests/semantics/features/test_extract_from_zarr.py`,
  `tests/reconstructor/test_reconstructor.py`, `tests/test_cu121_migration.py`
- Modify: `docs/examples/ocr_lens_viewer.py`, `configs/README.md`, tutorials 05 + 06

- [ ] **Step 1: Write `tests/semantics/test_store.py` (failing)**

Moved from `test_semantics_utils.py` with `su.` → `store.`: every `test_extract_feature_cache_*`,
`test_valid_feature_cache_*`, plus helpers `_one_frame_scene`, `_fake_extractor`.
`test_extract_feature_cache_propagates_unexpected_errors` patches `store.zarr.open`.
Dropped with their code: `test_cache_store_path_*`, `test_load_feature_maps_*`,
`test_point_features_cached_*`, `_write_both_stores`, `_force_lifted_first`,
`test_load_point_features_rejects_legacy_codes_without_weights` (no legacy checks).
From `test_artifact_layout.py`: the orphan test is kept (repointed), the rest are dropped.

Point-feature tests (new or repointed):

```python
def _write_lifted(store_path, n_points=32, latent=8, input_dim=32):
    """
    Write a compressed lifted store.

    - latent != input_dim: equal widths would hide a reader that skips the decode
    """
    torch.manual_seed(0)
    codes = np.random.default_rng(0).random((n_points, latent), dtype=np.float32)
    write_point_features(store_path, codes, FeatureAutoencoder(input_dim=input_dim, latent_dim=latent))


def test_write_point_features_puts_the_autoencoder_inside_the_store(tmp_path):
    store_path = tmp_path / "dinov2_lifted.zarr"
    _write_lifted(store_path)

    assert (store_path / "autoencoder.pt").is_file()
    assert dict(zarr.open(str(store_path), mode="r").attrs) == {"input_dim": 32, "latent_dim": 8}


def test_load_point_features_full_dim_store_needs_no_weights(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    feats = np.random.default_rng(0).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert not (store_path / "autoencoder.pt").exists()
    out = load_point_features(store_path)
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-6)


def test_write_point_features_replaces_the_store_whole(tmp_path):
    """An uncompressed rewrite leaves no earlier autoencoder.pt to decode full-dim codes."""
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=6, latent=2, input_dim=5)
    feats = np.random.default_rng(1).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert not (store_path / "autoencoder.pt").exists()
    out = load_point_features(store_path)
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-6)


def test_write_point_features_killed_before_rename_leaves_no_store(tmp_path, monkeypatch):
    """A write that dies before the rename leaves only the tmp dir, so the stage is not done."""
    store_path = tmp_path / "talk2dino_lifted.zarr"

    def killed(self, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "rename", killed)

    with pytest.raises(KeyboardInterrupt):
        _write_lifted(store_path)

    assert not store_path.exists()
    assert (tmp_path / "talk2dino_lifted.zarr.tmp").exists()


def test_write_point_features_leaves_no_store_when_weights_fail(tmp_path, monkeypatch):
    """A failed weight save removes the tmp dir and never touches the store path."""
    store_path = tmp_path / "talk2dino_lifted.zarr"

    def boom(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(FeatureAutoencoder, "save", boom)

    with pytest.raises(OSError):
        _write_lifted(store_path)

    assert not store_path.exists()
    assert not (tmp_path / "talk2dino_lifted.zarr.tmp").exists()


def test_load_point_features_rejects_latent_codes_without_weights(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=4, latent=2, input_dim=5)
    (store_path / "autoencoder.pt").unlink()

    with pytest.raises(FileNotFoundError, match="autoencoder.pt"):
        load_point_features(store_path)


def test_load_point_features_decodes_in_batches_matching_the_unbatched_result(tmp_path, monkeypatch):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=37, latent=8, input_dim=32)

    # Reference: decode every code in one call through the same weights
    codes = torch.from_numpy(np.asarray(zarr.open(str(store_path), mode="r")["features"]))
    ae = FeatureAutoencoder.load(store_path / "autoencoder.pt")

    with torch.no_grad():
        expected = torch.nn.functional.normalize(ae.per_point_decode(codes), dim=1).numpy()

    # Shrink the batch so 37 points span several calls, and count them
    sizes = []
    real_decode = FeatureAutoencoder.per_point_decode

    def counting_decode(self, x):
        sizes.append(len(x))
        return real_decode(self, x)

    monkeypatch.setattr(FeatureAutoencoder, "per_point_decode", counting_decode)
    out = load_point_features(store_path, batch_size=5)

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert len(sizes) == 8 and max(sizes) <= 5
```

`test_package_re_exports_every_utils_public_name` sweeps both modules:
`for module in (su, store)` over `module.__all__`.

- [ ] **Step 2: Run, expect ImportError** — `python -m pytest tests/semantics/test_store.py -q` →
`ModuleNotFoundError: No module named 'collab_splats.semantics.store'`.

- [ ] **Step 3: Write `collab_splats/semantics/store.py`**

```python
"""
On-disk semantics stores: the 2D patch cache and the lifted per-point codes.

- 2D cache `<scene>/semantics/<extractor>.zarr`: `features` (N, D, H_p, W_p) float16, one chunk
  per frame; attrs `extractor`, `patch_size`, `n_frames`, `extractor_kwargs`; the stage keeps
  its `autoencoder.pt` inside
- lifted store `<backend>/semantics/<extractor>_lifted.zarr`: `features` (P, latent); attrs
  `input_dim`, `latent_dim`; `autoencoder.pt` inside when compressed
- paths come from `Reconstructor`; nothing here builds or globs one
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from collab_splats.preproc.frames import IMAGE_EXTS, frame_paths
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.utils.io import open_valid, read_image, to_json_safe
from collab_splats.utils.torch_utils import batch_iterator

logger = logging.getLogger(__name__)

__all__ = [
    "extract_feature_cache",
    "load_point_features",
    "valid_feature_cache",
    "write_point_features",
]


########################################################
########## 2D patch cache (<extractor>.zarr) ###########
########################################################

# valid_feature_cache + extract_feature_cache: moved verbatim from utils.py


########################################################
########## Lifted store (<extractor>_lifted.zarr) ######
########################################################


def write_point_features(store_path: Path, codes: np.ndarray, ae: Optional[FeatureAutoencoder]) -> None:
    """
    Write the lifted store atomically: codes, `autoencoder.pt` when compressed, width attrs.

    - written to `<name>.tmp`, then renamed into place; a killed write leaves no store
    - an old store is removed whole, so no stale `autoencoder.pt` survives an uncompressed rewrite

    Args:
        store_path: the lifted store, `<backend>/semantics/<extractor>_lifted.zarr`.
        codes: (P, latent) per-point codes, or (P, D) when `ae` is None.
        ae: the autoencoder that decodes `codes`, or None for full-dim codes.
    """
    store_path = Path(store_path)
    tmp = store_path.with_name(f"{store_path.name}.tmp")
    codes = np.asarray(codes)
    width = int(codes.shape[1])

    store_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # Codes, weights, then attrs into the tmp dir; a failure removes it
    try:
        store = zarr.open(str(tmp), mode="w")
        store["features"] = codes

        if ae is not None:
            ae.save(tmp / "autoencoder.pt")

        input_dim = int(ae.input_dim) if ae is not None else width
        store.attrs.update({"input_dim": input_dim, "latent_dim": width})
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # Swap the finished store in under its real name
    shutil.rmtree(store_path, ignore_errors=True)
    tmp.rename(store_path)


def load_point_features(store_path: Path, batch_size: int = 65_536) -> np.ndarray:
    """
    Read the lifted store, decoding latent codes back to full dim.

    - weights present: always decode, even at equal widths (the codes are still encoded)

    Args:
        store_path: the lifted store, `<backend>/semantics/<extractor>_lifted.zarr`.
        batch_size: points per decode chunk; bounds peak memory.

    Returns:
        (P, D) float32, L2-normalized per row.

    Raises:
        FileNotFoundError: when latent codes have no `autoencoder.pt` beside them.
    """
    store_path = Path(store_path)
    store = zarr.open(str(store_path), mode="r")
    codes = torch.from_numpy(np.asarray(store["features"]))
    weights = store_path / "autoencoder.pt"

    # No weights: full-dim codes are returned normalized; latent codes cannot be read
    if not weights.exists():
        if store.attrs["latent_dim"] < store.attrs["input_dim"]:
            raise FileNotFoundError(f"{store_path} holds latent codes but no autoencoder.pt; re-run semantics")

        return F.normalize(codes, dim=1).numpy()

    # Decode in chunks into one preallocated array; row-wise ops, so chunking is exact
    ae = FeatureAutoencoder.load(weights)
    decoded = torch.empty((codes.shape[0], ae.input_dim), dtype=torch.float32)

    with torch.no_grad():
        start = 0

        for (chunk,) in batch_iterator(batch_size, codes):
            end = start + len(chunk)
            decoded[start:end] = F.normalize(ae.per_point_decode(chunk), dim=1)
            start = end

    return decoded.numpy()
```

Note: `latent_dim` attr is the stored code width (`codes.shape[1]`), which equals
`ae.latent_dim` when compressed — same values the old writer stored.

`BaseFeatureExtractor` is a runtime import: `features/base.py` imports `semantics.utils`, not
`store`, so there is no cycle and the `TYPE_CHECKING` guard goes.

- [ ] **Step 4: Strip `utils.py`**

Delete from `utils.py`: the `Artifact paths` section (`lifted_store_path`, `ae_path`,
`find_lifted_extractor`), `cache_store_path`, `valid_feature_cache`, `extract_feature_cache`,
`load_feature_maps`, the per-point section; drop now-unused imports (`json`, `shutil`, `Path`,
`TYPE_CHECKING`, `Any`, `Optional`, `zarr`, `frame_paths`, `IMAGE_EXTS`, `FeatureAutoencoder`,
`collab_splats.utils.io`, `batch_iterator`, `logging`/`logger` if unused).
`__all__ = ["cluster_points", "compute_semantic_contrast"]`. Module docstring:

```python
"""
Semantic feature math: contrastive scoring, patch-token reshape, point clustering.

- store IO lives in `collab_splats.semantics.store`
"""
```

`semantics/__init__.py`: utils exports `cluster_points`, `compute_semantic_contrast`; store
exports the four store functions; docstring bullet `- store: the 2D cache and the lifted
per-point store`, `- utils: contrastive scoring and point clustering`.

- [ ] **Step 5: Callers**

`collab_splats/reconstructor.py`:
- import block → `from collab_splats.semantics.store import extract_feature_cache, valid_feature_cache, write_point_features`
- `outputs`:
  ```python
  extractor = self.config["semantics"]["extractor"]
  return {
      ...
      "semantics": self.backend_dir / "semantics" / f"{extractor}_lifted.zarr",
  ```
- `semantics()` last line → `write_point_features(self.outputs["semantics"], codes, ae)`;
  docstring bullet `- lifted store written atomically with its own autoencoder.pt (pushed; the 2D cache is not)`.

`tests/reconstructor/test_reconstructor.py`:
- `test_semantics_writes_weights_beside_codes` → `test_semantics_writes_weights_inside_the_lifted_store`:
  `(out_dir / "dinov2_lifted.zarr" / "autoencoder.pt").is_file()`.
- `test_semantics_uncompressed_writes_full_dim_and_no_weights`: `not (out_dir / "dinov2_lifted.zarr" / "autoencoder.pt").exists()`.
- `test_done_semantics_is_per_extractor`: drop the inline import; `(sem_dir / "talk2dino_lifted.zarr").mkdir()` /
  `(sem_dir / "dinov2_lifted.zarr").mkdir()`.

Other tests:
- `tests/semantics/features/test_extract_from_zarr.py`: import from `collab_splats.semantics.store`; docstring `semantics.store.extract_feature_cache`.
- `tests/test_cu121_migration.py`: add `"collab_splats.semantics.store",` after `semantics.segmentation`.
- delete `tests/semantics/test_artifact_layout.py`.
- `tests/semantics/test_semantics_utils.py`: drop moved/deleted tests and imports; module docstring
  `contrastive scoring, point clustering, package surface`.

Docs:
- `docs/examples/ocr_lens_viewer.py`: drop `cache_store_path` import; `cache = scene_dir / "semantics" / "ocr_lens.zarr"`.
- `configs/README.md`: the two layout trees lose the `<extractor>_ae.pt` line and the lifted
  line reads `<extractor>_lifted.zarr  ← lifted 3D features (+ autoencoder.pt inside if n_components set)`;
  the processed-scene table row `<backend>/semantics/<extractor>_ae.pt` →
  `<backend>/semantics/<extractor>_lifted.zarr/autoencoder.pt`; the prose "lifted pair is
  `..._lifted.zarr` + `_ae.pt`" → "lifted store is `<backend>/semantics/<extractor>_lifted.zarr`
  with `autoencoder.pt` inside"; drop the "flat layout" sentence about the dashboard.
- `05_lifting/semantic_lifting.ipynb`: drop the `ae_path` import and `SEMANTICS_DIR`; after the
  `LIFTED_*` lines define `AE_MASKCLIP = LIFTED_MASKCLIP / "autoencoder.pt"` and
  `AE_TALK2DINO = LIFTED_TALK2DINO / "autoencoder.pt"` (the AE is saved after the zarr write, so
  `mode="w"` does not wipe it).
- `06_mesh/splats_mesh.ipynb`: `AE_MASKCLIP = LIFTED_MASKCLIP / "autoencoder.pt"` after
  `LIFTED_MASKCLIP`; markdown sentence about the moved AE path → "The AE lives inside the
  lifted store as `autoencoder.pt`; a cache warmed before that change is refit."

- [ ] **Step 6: Gate** — store tests pass; reconstructor semantics tests pass; deleted-test
  delta matches the list in Step 1; docstring + import-style contracts green for `store.py`.

- [ ] **Step 7: Format, graphify, show diff, commit**

```bash
git rm -q tests/semantics/test_artifact_layout.py
git commit --only <touched files> -m "refactor(semantics): store.py takes explicit paths; lifted store holds its AE, written atomically"
```

---

### Task 3: end-to-end check

- [ ] One `reconstruct local` run (tmux, one at a time) on the tutorial clip with
  `semantics.enabled: true`, `n_components: 13`: `<backend>/semantics/<extractor>_lifted.zarr/autoencoder.pt` exists,
  no `.tmp` dir left, `load_point_features` on it returns (P, D).
- [ ] Second run with `--stages semantics --overwrite`: log shows `autoencoder cache hit`.
- [ ] CHANGELOG entry + CLAUDE.md in-flight line removed when done.
