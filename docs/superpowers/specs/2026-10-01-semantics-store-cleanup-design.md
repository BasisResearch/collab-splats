# Semantics store cleanup — design

Date: 2026-10-01 · Branch: `feat/ocr-lens` · Status: approved in brainstorm

Rules: [017 — release cleanup rules](../decisions/017-release-cleanup-rules.md). This spec
applies them and does not restate them.

## Goal

Shrink the semantics store to what `Reconstructor` writes and reads: explicit paths, no
globs, no path helpers, one directory per artifact. Store IO moves out of
`semantics/utils.py` into `semantics/store.py`; `utils.py` keeps the feature math and gains
`cluster_points`.

## What the store is for

| Store | Path (from `Reconstructor`) | Written by | Read by |
|---|---|---|---|
| 2D patch cache + its AE | `rec.semantics_cache_dir / <extractor>.zarr` (`autoencoder.pt` inside) | `Reconstructor.semantics` via `extract_feature_cache`; AE saved by the stage | `Reconstructor.semantics` (cache hit skips the model load; AE reused when `latent_dim` matches) |
| Lifted codes + AE | `rec.outputs["semantics"]` = `<scene>/<backend>/semantics/<extractor>_lifted.zarr` | `Reconstructor.semantics` via `write_point_features` | queries, via `load_point_features` |

The 2D cache depends only on the frames, so every backend shares it and `PUSH_EXCLUDES`
keeps it local. The lifted store depends on the pointcloud, so it is per backend and pushed.

## Decisions (from brainstorm)

- **The layout is `Reconstructor`'s.** Path properties and `outputs` are the one spelling of
  the layout (reconstructor-release spec). Store functions take the path; none build one.
- **The extractor comes from config, never a glob.** `semantics.extractor` names the store;
  `cache_store_path` and `find_lifted_extractor` go.
- **The autoencoder is never refit needlessly.** Unchanged: `Reconstructor.semantics` reuses
  `<extractor>.zarr/autoencoder.pt` when its `latent_dim` matches `n_components`, and
  re-extraction (`zarr.open(mode="w")`) wipes it.
- **The lifted pair becomes one directory.** The AE moves inside the lifted store as
  `<extractor>_lifted.zarr/autoencoder.pt`, matching the 2D cache. One path names the pair,
  so `ae_path` and `lifted_store_path` go, and so does the stale-`_ae.pt` cleanup.
- **The lifted store keeps its own copy of the AE.** `PUSH_EXCLUDES` drops the scene-root
  `/semantics/**`, so the 2D cache's `autoencoder.pt` never reaches the bucket; a pulled
  scene decodes from the copy (~1 MB).
- **The lifted store is written atomically.** `rec.done("semantics")` is a bare `exists()`,
  so a write killed mid-way would mark the scene done and unreadable. The store is written to
  `<name>.tmp` and renamed into place; this replaces the `point_features_cached` check.
- **`cluster_points` moves to semantics.** `mesh_clustering` reads only vertex positions, so
  it serves any point set.
- **The dashboard is not adapted here.** Its imports of deleted names break; it is fixed in
  its own cleanup afterwards. `tests/dashboard` is out of this gate.

## Target API

### `collab_splats/semantics/store.py`

```python
def valid_feature_cache(cache_dir, name, images_dir, extractor_kwargs) -> Path | None
def extract_feature_cache(extractor, images_dir, cache_dir, extractor_kwargs=None, overwrite=False) -> Path
def write_point_features(store_path: Path, codes: np.ndarray, ae: FeatureAutoencoder | None) -> None
def load_point_features(store_path: Path, batch_size: int = 65_536) -> np.ndarray
```

- `write_point_features`: writes `<name>.tmp` (codes, `autoencoder.pt`, attrs last), removes
  any old store, renames into place; on failure the tmp dir is removed; `ae=None` writes
  full-dim codes and no weights file.
- `load_point_features`: full-dim codes are normalized directly; otherwise decoded per chunk
  with the store's `autoencoder.pt`; a store missing its weights raises `FileNotFoundError`.

### `collab_splats/semantics/utils.py`

`compute_semantic_contrast`, `_tokens_to_feature_map`, `cluster_points`.

## Verdict table

### `semantics/utils.py`

| name | verdict | why |
|---|---|---|
| `lifted_store_path` | delete | `rec.outputs["semantics"]` spells it |
| `ae_path` | delete | AE lives inside the lifted store |
| `find_lifted_extractor` | delete | extractor comes from config |
| `cache_store_path` | delete | extractor comes from config |
| `valid_feature_cache`, `extract_feature_cache` | move | to `store.py`, unchanged |
| `load_feature_maps` | delete | no package caller; `Reconstructor.semantics` reads the zarr lazily |
| `write_point_features` | move + change | to `store.py`; takes `store_path`; AE inside; atomic |
| `point_features_cached` | delete | `rec.done("semantics")` + atomic write |
| `load_point_features` | move + change | to `store.py`; takes `store_path` |
| `compute_semantic_contrast`, `_tokens_to_feature_map` | keep | feature math |

### `mesh/features.py`

| name | verdict | why |
|---|---|---|
| `mesh_clustering` | move + rename | `semantics/utils.cluster_points(points, similarity_values, ...)`; file deleted |

## Commits

1. `cluster_points`: move, delete `mesh/features.py`; `mesh/__init__`, `semantics/__init__`,
   `docs/mesh.md`, `06_mesh/splats_mesh.ipynb` updated; its 3 tests move to
   `tests/semantics/test_semantics_utils.py`.
2. `semantics/store.py`: move the four store functions; `write_point_features` /
   `load_point_features` take `store_path`; AE inside the lifted store; atomic write. Delete
   the globs, path helpers, `point_features_cached`, `load_feature_maps`. Callers in the same
   commit: `Reconstructor.outputs` / `semantics()`, `semantics/__init__`,
   `docs/examples/ocr_lens_viewer.py` (explicit `<scene>/semantics/ocr_lens.zarr`), tutorials
   `05_lifting/semantic_lifting.ipynb` and `06_mesh/splats_mesh.ipynb` (`ae_path` →
   `<store>/autoencoder.pt`), `tests/reconstructor/test_sfm_stage.py` patch target, store
   tests moved to `tests/semantics/test_store.py`.

## Testing

- Gate per commit, in the worktree, printing `collab_splats.__file__`:
  `tests/semantics tests/reconstructor tests/mesh tests/test_docstring_contract.py tests/test_import_style.py`,
  SKIP count compared to the baseline.
- Kept and repointed: AE reuse (`assert not fit.called`), refit on `n_components` change,
  no orphan store when the weights fail, full-dim store needs no weights.
- Deleted with their code: `test_find_lifted_extractor_*`, `test_cache_store_path_*`,
  `test_point_features_cached_*`, `load_feature_maps` tests.
- New: a write killed before the rename leaves `done("semantics")` False (tmp dir only); a
  re-write replaces the old store whole (no stale `autoencoder.pt` after `ae=None`).
- One `reconstruct local` run on a short tutorial clip with semantics on: the lifted store
  holds `autoencoder.pt`; a second run with `--overwrite` logs `autoencoder cache hit`.

## Risks

- The dashboard does not import until its own cleanup lands.
- `reconstructor.py` (`outputs`, `semantics()`) also carries the cherry-picked BA commits on
  this branch; keep the semantics edits in their own hunks.

## Follow-up (not this spec)

Dashboard cleanup: the dashboard holds a `Reconstructor`, runs semantics through
`rec.run(["semantics"])`, reads every pipeline path from `rec`, and deletes its own copy of
the stage (`_extract_semantics`, `AutoencoderPolicy`, `_lift_and_compress`,
`resolve_semantics_dir`, the legacy lift and self-upgrade, the dense-member fetch). Open
there: the COLMAP model `rec.done("pointcloud")` needs, and queries on scenes without
semantics.
