# semantics cleanup — design

**Date:** 2026-09-05
**Scope:** `collab_splats/semantics/` and its callers (`wrapper/reconstructor.py`,
`dashboard/pipeline.py`, `dashboard/viewer.py`, `dashboard/app.py`,
`localization/retrieval.py`), their tests, `docs/semantics.md`, tutorial
notebooks 05 and 06
**Status:** design, awaiting approval

First per-module cleanup pass over the semantics package. The brief: fewer
abstractions, no dead code, docstrings that state what a function does, what it
takes, and what it returns. Behavior of the pipeline is unchanged; the on-disk
artifacts are unchanged except for two unread zarr attrs.

---

## 1. Motivation

Problems found by reading every file in the package and grepping every caller.

**The on-disk contract is split across three files.** `compression.py` owns
the write side and the path helpers (`LIFTED_SUFFIX`, `ae_path`,
`write_point_features`). `dashboard/pipeline.py` owns the read side
(`cache_store_path`, `load_feature_maps`, `load_point_features`,
`point_features_cached`, `_is_full_dim` — 130 lines). `features/base.py` owns
the 2D-cache writer. `wrapper/reconstructor.py` bypasses the reader and calls
`zarr.open` itself. A reader of any one file cannot see the layout.

**Two 2D-cache writers.** `BaseFeatureExtractor.extract_and_cache` (path-list
input) has zero production callers; `extract_and_cache_from_zarr` is the live
one. Both are ~60-90 lines of near-identical zarr setup.

**The autoencoder carries three unused features.** `regularization_kwargs` /
`reg_head` / `reg_target`, the `lr_scheduler` hook, and an exposed `hidden_dim`
have zero production call sites. Only their own tests exercise them.

**`preprocess()` is copied verbatim three times** (dino, maskclip, talk2dino):
open, RGB, square-crop-or-longest-edge resize, round to patch multiple,
normalize. ~25 lines each.

**`semantics/utils.py` is mostly a shim.** Six names re-exported from
`utils/torch_utils` (a May back-compat layer); four modules still import through
it. pyflakes reports 13 unused imports across `features/base.py` and `utils.py`.

**Rationale essays where docstrings should be.** `cache_store_path` has a
six-line paragraph; `write_point_features` and the `_lifted` suffix each get one;
the talk2dino class docstring is fifteen lines; a `warnings.filterwarnings`
call has a seven-line comment. `configs/base.yaml` already carries the
autoencoder rationale.

**talk2dino resolves `patch_size` through a triple nested `try/except`** whose
middle branch exists for a test fixture.

**`insid3.py` inline-imports `sklearn`** (a hard dependency, `pyproject.toml`
line 62) and `DINOFeatureExtractor`.

**`_tokens_to_feature_map` is marked private** and imported by three modules and
a test file.

**ImageNet and CLIP normalization constants** are defined in `dino.py`,
`maskclip.py`, and inline again at `localization/retrieval.py:76`.

**sam3** mentions HuggingFace gating only inside the `ImportError` text. The
`device` constructor argument and the `confidence_threshold` argument on
`segment_with_text` are both unused.

**`docs/semantics.md` documents code that does not exist:** `samclip`,
`clip-vit`, `compute_semantic_heatmap`, `protocols.SupportsTextQuery`, a Gradio
dashboard.

---

## 2. Layout

Decision: fold everything flat into `semantics/utils.py`. No new module.

```
collab_splats/semantics/
  __init__.py            re-exports (trimmed, see §11)
  utils.py               tensor helpers + on-disk contract: paths, 2D cache, lifted pair
  compression.py         FeatureAutoencoder only
  features/
    base.py              BaseFeatureExtractor (registry, preprocess, forward, debias), BaseQueryableExtractor
    dino.py              model load + forward
    maskclip.py          model load + forward + encode_text
    talk2dino.py         model load + forward + encode_text
  segmentation/          base.py, insid3.py, mobile_sam.py, sam3.py — import and parameter fixes only
collab_splats/utils/image.py   gains IMAGENET_MEAN/STD, CLIP_MEAN/STD
```

Import graph, acyclic:

- `compression` imports torch only.
- `utils` imports `compression` (for `write_point_features` /
  `load_point_features`), `preproc.frame_store`, zarr.
- `features/base` imports `utils` (`compute_semantic_contrast`,
  `tokens_to_feature_map`). `extract_feature_cache` takes the extractor as an
  argument, so `utils` never imports `features`.
- `wrapper` and `dashboard` import from `collab_splats.semantics.utils` and
  `collab_splats.semantics.compression`.

---

## 3. `semantics/utils.py`

Section dividers in this order: constants, tensor helpers, paths, 2D cache,
lifted pair.

### 3.1 Tensor helpers

- `compute_semantic_contrast(raw_similarities, num_positive, temperature=0.05, reduction="max")`
  — unchanged.
- `tokens_to_feature_map(tokens, input_h, input_w, patch_size) -> Tensor` —
  renamed from `_tokens_to_feature_map`, body unchanged. `(N, D)` patch tokens
  to a `(D, H_p, W_p)` map, L2-normalized over `D`. The name is accurate: the
  input is tokens, the output is a feature map, and "feature map" is the type
  name used across the package. The underscore was the defect — a private
  marker on a function imported by three modules and a test. `reshape_patch_tokens`
  was rejected because it drops "feature map". The docstring states the
  normalize.
- `interpolate_to_patch_size` deleted. Only tests call it.

The shim is deleted: `get_device`, `pytorch_gc`, `infer_batch_size`,
`batch_iterator`, `load_hf_weights`, `load_torchhub_model` are no longer
re-exported. Callers import from `collab_splats.utils.torch_utils`.

### 3.2 Paths

Moved with bodies unchanged:

| name | from |
|---|---|
| `LIFTED_SUFFIX = "_lifted.zarr"`, `AE_SUFFIX = "_ae.pt"` | `compression.py` |
| `lifted_store_path(out_dir, extractor) -> Path` | `compression.py` |
| `ae_path(out_dir, extractor) -> Path` | `compression.py` |
| `find_lifted_extractor(out_dir) -> str \| None` | `compression.py` |
| `cache_store_path(semantics_dir) -> Path` | `dashboard/pipeline.py` |
| `cache_extractor_name(semantics_dir) -> str` | `dashboard/pipeline.py` |

`cache_store_path` returns the first `*.zarr` whose name does not end in
`LIFTED_SUFFIX`, or raises `FileNotFoundError`. Its six-line docstring becomes
one line; the layout rationale lives in the module docstring (§3.5).

### 3.3 2D cache

```python
def extract_feature_cache(extractor, frames_zarr: Path, cache_dir: Path, *, skip_existing: bool = True) -> Path
```

Moved from `BaseFeatureExtractor.extract_and_cache_from_zarr` and made a
function. Writes `cache_dir/<extractor.name>.zarr` with array `features`
`(N, D, H_p, W_p)` float32, chunks `(1, D, H_p, W_p)`, attrs `extractor`,
`patch_size`, `n_frames`. Returns the store path. Reads one frame at a time from
`FrameStore` and calls `extractor.forward([img])`.

Changes from the method:

- `batch_size` parameter dropped. Both callers use the default of 1; `forward`
  already takes a list, so batching is the extractor's concern.
- Attrs `created_at` and `feature_dim` dropped. No reader exists.
- `features/base.py` stops importing zarr, `FrameStore`, and `datetime`.

```python
def load_feature_maps(store_path: Path) -> list[torch.Tensor]
```

Moved from `dashboard/pipeline.py`. Takes the store path, not the directory.
The reconstructor already holds the path and stops calling `zarr.open` itself;
dashboard callers pass `cache_store_path(semantics_dir)`. Returns one
`(D, H_p, W_p)` CPU tensor per frame.

### 3.4 Lifted pair

Moved from `compression.py` and `dashboard/pipeline.py`, bodies unchanged except
the autoencoder save/load calls:

- `write_point_features(out_dir, extractor, codes, ae=None) -> Path` — writes
  `<extractor>_lifted.zarr` (`features` `(P, latent)`, attrs `input_dim`,
  `latent_dim`), then `ae.save(ae_path(out_dir, extractor))` when `ae` is given.
  Removes the store on failure.
- `point_features_cached(semantics_dir) -> bool`
- `load_point_features(semantics_dir, *, decode=True) -> np.ndarray` — calls
  `FeatureAutoencoder.load(ae_path(sem_dir, extractor))`; streamed decode in
  `_DECODE_BATCH_SIZE = 65_536` rows.
- `_is_full_dim(attrs) -> bool`

### 3.5 Module docstring

Replaces the four paragraph essays currently spread over `compression.py` and
`dashboard/pipeline.py`:

```python
"""
Semantic feature helpers and the on-disk layout of semantic artifacts.

Layout inside a scene's semantics dir:
- `<extractor>.zarr`: 2D patch cache, `features` (N, D, H_p, W_p) float32, one chunk per frame;
  attrs `extractor`, `patch_size`, `n_frames`.
- `<extractor>_lifted.zarr` + `<extractor>_ae.pt`: per-point codes `features` (P, latent) and the
  autoencoder that decodes them; attrs `input_dim`, `latent_dim`.
- `latent_dim == input_dim` with no `_ae.pt` is full-dim by design (`n_components: null`);
  `latent_dim < input_dim` with no `_ae.pt` is an interrupted write and raises on read.
- The `_lifted` suffix is the only thing separating the two stores in one flat dir.
"""
```

---

## 4. `compression.py` — `FeatureAutoencoder`

Kept: encoder/decoder MLPs, spatial `encode`/`decode`, `per_point_encode` /
`per_point_decode`, `fit` with `target_cosine` early stop and the `on_epoch`
callback, metrics `recon_cosine` / `recon_mse` / `epochs_run` (the dashboard
reads them).

Removed, all with zero production callers:

- `regularization_kwargs`, `reg_head`, `_reg_dim`, `_reg_weight`, and the
  `reg_target` argument to `fit`.
- The `lr_scheduler` argument to `fit`.
- The `hidden_dim` constructor argument. Every caller uses the default; the rule
  `hidden_dim = max(64, 2 * latent_dim)` stays as an internal line.

Signatures after:

```python
class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None
    def encode(self, feature_map: Tensor) -> Tensor        # (D, H, W) -> (latent, H, W)
    def decode(self, latent_map: Tensor) -> Tensor         # (latent, H, W) -> (D, H, W)
    def per_point_encode(self, feats: Tensor) -> Tensor    # (P, D) -> (P, latent)
    def per_point_decode(self, codes: Tensor) -> Tensor    # (P, latent) -> (P, D)
    def fit(self, features, epochs=10, batch_size=1024, lr=1e-3, on_epoch=None, target_cosine=None) -> dict
    def save(self, path: Path) -> None                     # weights file path
    @classmethod
    def load(cls, path: Path) -> "FeatureAutoencoder"      # weights file path
```

`on_epoch` is called as `on_epoch(epoch, epochs, loss)`, matching the
dashboard's existing lambda.

`save`/`load` take the weights file directly. Path composition (`ae_path`) lives
in `utils`, and `utils` imports `compression`; a `compression -> utils` import
would be a cycle. Callers: `write_point_features` and `load_point_features` in
`utils`, seven test sites, notebooks 05 and 06.

Checkpoint payload: `input_dim`, `latent_dim`, `state_dict`, `recon_cosine`,
`recon_mse`, `epochs_run`. `load` reconstructs from `input_dim`/`latent_dim`.
The `hidden_dim` and `regularization_kwargs` keys are gone; a checkpoint
carrying `reg_head.*` weights fails strict `load_state_dict`. Metrics missing
from the payload default to 0.0 (one line, kept). No production checkpoint
exists (§14).

The eight-line comment on why metrics are persisted becomes one line.

---

## 5. `features/base.py`

Kept, behavior unchanged, docstrings rewritten to §10: the registry, `name`,
the debias stack (`_build_positional_basis`, `_apply_debias`, `debias`,
`get_bias_visualization`, `_pos_basis_cache`, `_zero_feats_cache`,
`svd_components=500`, `_DEBIAS_VALIDATED`), `features_to_rgb`,
`BaseQueryableExtractor` (`encode_text`, `compute_similarity`,
`score_queries`), `TORCH_HOME`.

Deleted:

- `extract_and_cache` — path-list writer, zero callers.
- `extract_and_cache_from_zarr` — becomes `utils.extract_feature_cache` (§3.3).
- `_FALLBACK_MEM_GB` — unused.
- The duplicate `@abstractmethod forward` on `BaseQueryableExtractor`.
- Unused imports per pyflakes. Re-run pyflakes after the hoist below: `T`,
  `open_image`, `resize_image` become used again.

Hoisted from the three extractors:

```python
def __init__(self, resize_mode: str, image_resolution: int, svd_components: int = 500) -> None
def preprocess(self, image) -> Tensor   # (C, H, W) float32 CPU; H and W multiples of self.patch_size
```

`preprocess` is the body the three copies share: open, RGB, square center-crop
and resize or longest-edge resize, round `H`/`W` to a patch multiple,
`self._normalize(T.ToTensor()(img))`. Subclasses set `self.patch_size: int` and
`self._normalize: T.Normalize` in their own `__init__`. `resize_mode` and
`image_resolution` have no base default; each subclass declares its own
(dino 800, maskclip 1024, talk2dino 512) and passes it up. The `**kwargs`
pass-through to `nn.Module` is dropped; nothing supplies any.

---

## 6. Extractors

Common to dino, maskclip, talk2dino:

- `preprocess` removed (base owns it).
- `forward` = stack preprocessed frames, run the model, one
  `tokens_to_feature_map` per frame. Two block comments.
- Imports: `get_device` from `collab_splats.utils.torch_utils`,
  `tokens_to_feature_map` from `collab_splats.semantics.utils`, constants from
  `collab_splats.utils.image`.

**dino**: `self._normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)`.

**maskclip**: `T.Normalize(CLIP_MEAN, CLIP_STD)`. `import maskclip_onnx` stays
inside `__init__` (CLAUDE.md exception for heavy optional deps; the package
imports `pkg_resources` at import time). Its comment becomes one line.

**talk2dino**:

- Class docstring: one line plus two bullets — the two model IDs, and
  `forward_features()[:, 5:]` drops CLS plus four register tokens.
- The `warnings.catch_warnings` block stays; its comment becomes two lines
  (meta-tensor no-op copy spam from the unused CLIP visual tower).
- `_normalize` still comes from `_loaded.image_transforms.transforms[-1]`
  (varies by backbone).
- `self.patch_size = _loaded.model.patch_embed.proj.stride[0]`. The triple
  `try/except` is deleted. The test fixture sets
  `mock_model.model.patch_embed.proj.stride = (p, p)`.

---

## 7. Segmentation

**sam3**

- The module-level `try/except ImportError` stays: it is already at the top of
  the file and `test_segmentation.py` patches those two module names.
- `ImportError` text: facebook/sam3 is gated on HuggingFace — request access at
  `https://huggingface.co/facebook/sam3`, wait for approval, run
  `huggingface-cli login`, then install from
  `https://github.com/facebookresearch/sam3`.
- The class docstring carries the same note as a bullet.
- `device` constructor argument dropped (unused; `Sam3Processor` places itself).
- `confidence_threshold` dropped from `segment_with_text` here and on
  `BaseSegmentation.segment_with_text` (sam3 is the only override; the
  threshold is set at construction).
- The working tree holds a whitespace-only edit to this file (trailing newline)
  from another session; the rewrite absorbs it.

**insid3**: `from sklearn.cluster import AgglomerativeClustering` and
`from collab_splats.semantics.features.dino import DINOFeatureExtractor` move to
the top. `features` never imports `segmentation`, so no cycle. The path comment
on line 1 is removed.

**mobile_sam**: `batch_iterator`, `load_torchhub_model` from
`collab_splats.utils.torch_utils`.

**segmentation/base.py**: `segment_with_text` signature change only. The mask
utilities' docstrings are already in Args/Returns form; trim to §10 where over.

---

## 8. Constants → `collab_splats/utils/image.py`

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
```

`image.py` stays "pure PIL, no torch": these are plain lists. Users: `dino.py`,
`maskclip.py`, `localization/retrieval.py` (replacing its inline literals).

---

## 9. Verdict: zarr stays

The 2D cache for 300 frames at 800 px and 768-D is roughly 3 GB of float32.
Per-frame chunks let the lift read one frame at a time. Attrs carry `extractor`
and `patch_size` for cache validation. Every other artifact in the repo is zarr
(`frames.zarr`, `pointcloud.zarr`, `splats.zarr`). A `.npy` memmap needs a
sidecar for the attrs; `.pt` loads the whole array. The overengineering was two
duplicate writers and a reader/writer split across three files, not the format.

The preproc-centralization spec replaces `frames.zarr` (semantics' input); it
does not touch the semantics caches (semantics' output). Coupling in §14.

---

## 10. Docstrings and comments

Convention, from CLAUDE.md, tightened for this module:

- `"""` on its own line. One-line summary. Blank line. `Args:` bullets with
  type or shape. `Returns:` bullets with shape. Nothing else in a docstring.
- Rationale goes in a one-line block comment at the site, or in the `utils`
  module docstring for the disk contract. Paragraph essays are deleted, not
  moved.
- One block comment per logical block, at most two lines.

Example, `FeatureAutoencoder.fit`:

```python
"""
Train on features with cosine reconstruction loss.

Args:
    features: (P, input_dim) float tensor, any device.
    epochs: max passes over features.
    batch_size: rows per optimizer step.
    lr: Adam learning rate.
    on_epoch: optional callback(epoch, epochs, loss) after each epoch.
    target_cosine: stop early once mean reconstruction cosine >= this.

Returns:
    dict with recon_cosine, recon_mse, epochs_run; also stored on self.
"""
```

Applies to every function and class in the package, including the debias
methods in `features/base.py` ("docstrings only" for that stack).

---

## 11. Call-site changes

| file | change |
|---|---|
| `wrapper/reconstructor.py` | `_extract_2d_features` calls `extract_feature_cache(extractor, frames_zarr, cache_dir)`; `_lift_and_save` calls `load_feature_maps(zarr_path)` in place of the raw `zarr.open` block; imports from `semantics.utils` |
| `dashboard/pipeline.py` | deletes `cache_store_path`, `cache_extractor_name`, `load_feature_maps`, `_is_full_dim`, `point_features_cached`, `load_point_features`, `_DECODE_BATCH_SIZE` (~130 lines) and imports them from `semantics.utils`; `_extract_semantics` calls `extract_feature_cache`; `_lift_and_compress` calls `load_feature_maps(cache_store_path(semantics_dir))` |
| `dashboard/viewer.py` | drops the inline `from collab_splats.dashboard.pipeline import load_feature_maps`; top-level import from `semantics.utils`; call wraps `cache_store_path` |
| `dashboard/app.py` | import path for `point_features_cached` / `load_point_features` |
| `localization/retrieval.py` | `IMAGENET_MEAN` / `IMAGENET_STD` from `utils.image` |
| `semantics/__init__.py`, `features/__init__.py`, `segmentation/__init__.py` | `__all__` drops the six torch_utils names, `_DEBIAS_VALIDATED`, `TORCH_HOME`; gains `tokens_to_feature_map`, `extract_feature_cache`, `load_feature_maps`, `write_point_features`, `load_point_features`, `point_features_cached`, `cache_store_path`, `cache_extractor_name`, `lifted_store_path`, `ae_path`, `find_lifted_extractor` |
| `CLAUDE.md` architecture tree | semantics block gains a `utils.py` line; the `compression.py` line drops "per-point encode/decode + recon_cosine" for "FeatureAutoencoder" |

Line tally, approximate:

| module | now | after |
|---|---|---|
| `compression.py` | 391 | ~190 |
| `utils.py` | 126 | ~250 |
| `features/base.py` | 503 | ~330 |
| `dino.py` + `maskclip.py` + `talk2dino.py` | 431 | ~250 |
| `sam3.py` | 81 | ~65 |
| `dashboard/pipeline.py` | | −130 |
| **semantics package** | **2429** | **~1750** |

---

## 12. Tests

| file | change |
|---|---|
| `tests/semantics/test_compression.py` | delete the five reg-head tests; `save(path)` / `load(path)` |
| `tests/semantics/test_compression_target.py` | path API; the legacy-payload test builds a payload without `hidden_dim` / `regularization_kwargs` and without metrics |
| `tests/semantics/test_artifact_layout.py` | imports from `semantics.utils`; `load(ae_path(...))` |
| `tests/semantics/features/test_extract_from_zarr.py` | `extract_feature_cache(extractor, ...)`; `features_to_rgb` tests unchanged |
| `tests/semantics/test_positional_debiasing.py` | `_DEBIAS_VALIDATED` from `features.base`; `_FakeExtractor` passes `resize_mode`, `image_resolution` |
| `tests/semantics/test_extractor_preprocessing.py` | `tokens_to_feature_map`; talk2dino fixture sets `mock_model.model.patch_embed.proj.stride`; preprocess tests exercise the base method |
| `tests/semantics/test_semantics_utils.py` | torch_utils tests move to a new `tests/utils/test_torch_utils.py`; interpolate tests deleted; gains the four store-selection tests from `tests/dashboard/test_semantics_store_selection.py`, retargeted at `cache_store_path`; that file is deleted |
| `tests/semantics/test_features_guards.py` | `pytorch_gc` from `utils.torch_utils` |
| `tests/semantics/test_segmentation.py` | sam3 constructed without `device`; `segment_with_text(image, prompt)` |
| `tests/dashboard/test_pipeline.py` | `patch.object(pl, "load_feature_maps")` still works (name bound in `pipeline`); `FeatureAutoencoder.load(ae_path(...))` |
| `tests/dashboard/test_viewer.py`, `test_viewer_lift.py` | viewer patches target `collab_splats.dashboard.viewer.load_feature_maps`; `save(ae_path(...))` |
| `tests/wrapper/test_reconstructor.py` | mock target `collab_splats.wrapper.reconstructor.extract_feature_cache` |
| `tests/test_cu121_migration.py` | unchanged; `collab_splats.semantics.utils` stays importable |

Run after each phase:

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/test_semantics_logging.py tests/test_cu121_migration.py
```

---

## 13. Docs and notebooks

- `docs/semantics.md` rewritten to the code: the three extractors and registry
  usage, `extract_feature_cache` → lift → `write_point_features` /
  `load_point_features`, the autoencoder, the three segmentation backends, and
  the sam3 gating steps. Every stale name removed.
- `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` already calls
  `FeatureAutoencoder.load(AE_MASKCLIP)` and `ae_mc.save(AE_MASKCLIP)` with a
  file path — broken today, correct after §4. Verify the `AE_TALK2DINO` cell the
  same way.
- `docs/source/tutorials/06_mesh/splats_mesh.ipynb`: the two-argument
  `FeatureAutoencoder.load(AE_MASKCLIP, EXTRACTOR)` becomes
  `load(ae_path(<semantics dir>, EXTRACTOR))`, `ae_path` imported from
  `semantics.utils`.
- `docs/superpowers/CHANGELOG.md` entry on completion.

---

## 14. Risks

- **Foreign uncommitted edits** sit in `CLAUDE.md`,
  `tests/wrapper/test_reconstructor.py`, and `sam3.py` (whitespace only). Every
  commit uses `git commit --only <paths>`. Before touching those three files,
  re-check `git status`; if still dirty, stage only our hunks via `git apply
  --cached` of a patch. Never `git add -A`.
- **Preproc-centralization spec** (same day) deletes `FrameStore`; its call-site
  table names `semantics/features/base.py` as the frame-read site. After this
  spec that site is `utils.extract_feature_cache`: one function, one loop.
  Whichever lands second retargets that one site — the parameter becomes
  `images_dir` and the loop calls `read_frames`. Neither spec blocks the other.
- **Checkpoint format.** Strict `load_state_dict` rejects a checkpoint with
  `reg_head.*` keys. No production checkpoint exists: every run config has set
  `semantics.enabled: false` since 2026-08-24. No fallback is added.
- **`load_feature_maps` signature** changes from directory to store path. Two
  production callers, both updated in §11.
- **Concurrent sessions on the branch.** Commits land per phase (§15);
  `graphify update .` runs once at the end.

---

## 15. Phasing

Five commits, each leaving the suite green.

1. `refactor(semantics): fold artifact I/O into semantics.utils` — move the
   path, read, and write helpers plus `extract_feature_cache`; update
   reconstructor, dashboard pipeline/viewer/app, and their tests.
   Behavior-preserving.
2. `refactor(semantics): FeatureAutoencoder — drop reg head, lr_scheduler,
   hidden_dim; path-based save/load` — plus tests and notebooks 05/06.
3. `refactor(semantics): hoist preprocess into BaseFeatureExtractor; constants
   to utils.image` — dino, maskclip, talk2dino, retrieval, tests.
4. `refactor(semantics): drop dead code and shim; imports to top;
   tokens_to_feature_map` — base.py deletions, utils shim, insid3 / mobile_sam /
   sam3 imports and parameters, the rename.
5. `docs(semantics): docstrings to Args/Returns; rewrite docs/semantics.md;
   sam3 gating note` — the docstring pass over every file, CLAUDE.md tree,
   CHANGELOG.

---

## 16. Out of scope

- The two lift orders. The reconstructor lifts full-D then trains the
  autoencoder on points; the dashboard trains on 2D patches then lifts latent
  maps. Same artifacts, different cost. A later wrapper/dashboard pass.
- `features_to_rgb` overlaps `utils/visualization.pca_to_rgb`. Kept per the
  "keep the debias stack, docstrings only" decision.
- Debias algorithm, INSID3 internals, mobile_sam internals.
- Adding sam3 to `pyproject.toml`: the gated install cannot be automated.
