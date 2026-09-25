# Semantics release cleanup — design

Date: 2026-09-24 · Branch: `clean/semantics-release` off `clean/final` · Status: approved in brainstorm

Rules: [017-release-cleanup-rules.md](../decisions/017-release-cleanup-rules.md) — binding,
not restated here. Reference run: [preproc](2026-09-24-preproc-release-cleanup-design.md).

## Goal

Make `collab_splats/semantics/` release-ready: brief docs, no dead code, tunables as
kwargs, loud failures. A naive user reads a docstring and knows how to call the function.

## Decisions (from brainstorm)

- **Caller signatures are frozen.** Every external caller lives in a hands-off path
  (`wrapper/`, `dashboard/`, `evals/`, notebooks). A change that would break one is
  either shaped to stay compatible or listed under Deferred. New kwargs get defaults
  equal to today's behavior.
- **Scope:** `collab_splats/semantics/**`, `tests/semantics/**`, and adding `semantics`
  to `RELEASED` in `tests/test_docstring_contract.py`. Nothing else.
- **Contract checks:** cherry-picked from `clean/preproc-release` (`88d4f1ec`,
  `a4121020`); a trivial conflict with preproc on merge is expected.
- **Mask-grouping helpers stay.** `create_patch_mask`, `mask_id_to_binary_mask`,
  `convert_matched_mask` have no caller on `clean/final`, but `utils/grouping.py` on
  `main` uses them and may return.
- **Extractor `forward` moves up to the base class**; backends supply `_patch_tokens`.
- **Sky masks:** full cleanup, including the cache format.
- **Isolation:** worktree `.worktrees/semantics-release`; every gate runs as
  `cd <wt> && PYTHONPATH=<wt> python ...` and prints `collab_splats.__file__`.

## Starting point

Contract release checks on the untouched fork (semantics xfail, not yet released):
4 real violations, 66 xpass.

- `compression.py` banned word (`measured` x2)
- `segmentation/sky.py` banned word (`measured`)
- `utils.py` comment-cap and silent-fallback (`except Exception` at 262, 360, 393)

Everything else in this spec comes from reading the files, not from the linter.

## Round 1 — prose only

No code change. Proof: AST equal after deleting every docstring statement on both sides
and stripping comments, plus one sanity mutation showing the check can fail.

### Per file

- `__init__.py` (package + `features/`, `segmentation/`) — module docstring to 2-4 bullets
  naming what each subpackage holds.
- `utils.py` — `load_point_features` drops the 2.3 GB / 46.6 GB figures (one line: "reads
  in `batch_size` chunks to bound memory"); `point_features_cached` drops the
  `dashboard/app.py` reference; comment runs over 4 lines split to header + bullets;
  `compute_semantic_contrast` docstring to contract (shapes, `reduction` values).
- `compression.py` — `measured` wording out of the fit-quality comment and the
  `target_cosine` Arg; trailing prose after `fit` Args folded into a bullet;
  `decoder_hidden` / `decoder_out` split gets a one-line why (state-dict key compat).
- `features/base.py` — line-by-line comments in `_build_positional_basis` /
  `_apply_debias` to block comments; class docstrings to summary + bullets.
- `features/dino.py` — class docstring summary moves off the `"""` line.
- `features/talk2dino.py` — `low_cpu_mem_usage` / warnings why to one line.
- `features/maskclip.py` — drop the comment about mocks.
- `segmentation/base.py` — docstrings to contract for the five helpers; state
  `create_composite_mask` overlap rule in one bullet.
- `segmentation/mobile_sam.py`, `sam3.py`, `insid3.py` — docstrings to contract.
- `segmentation/sky.py` — drop "measured as a brightness detector", "~0.2 s/frame",
  "853 frames ~10 GB", "this container's onnxruntime"; keep the model source citation.

## Round 2 — code, one commit per logical change

Each row: `name | verdict | exact action`.

### utils.py

| name | verdict | action |
|---|---|---|
| `lifted_store_path`, `ae_path`, `find_lifted_extractor`, `load_feature_maps`, `load_point_features`, `compute_semantic_contrast` | keep | — |
| `write_point_features` | raise | `ae=None` deletes a stale `_ae.pt` beside the store, so a later load cannot pair new features with an old autoencoder (`900958b6`; added after review, user-approved 2026-09-24) |
| `tokens_to_feature_map` | make-private | rename `_tokens_to_feature_map` (only in-package + tests call it); `assert` -> `ValueError` |
| `cache_store_path` | raise | `ValueError` naming the stores when more than one `<extractor>.zarr` exists (today `next(...)` picks one arbitrarily) |
| `extract_feature_cache` | raise | narrow `except Exception` to the zarr/key errors a stale cache raises; one `logger.debug` per frame replaces the `i % 10` log |
| `point_features_cached` | raise | narrow `except Exception` to the same errors; still returns `False` for a missing/stale store |

### compression.py

| name | verdict | action |
|---|---|---|
| `FeatureAutoencoder`, `encode`, `per_point_encode`, `per_point_decode`, `fit`, `save` | keep | — |
| `FeatureAutoencoder.load` | raise | read `recon_cosine`, `recon_mse`, `epochs_run` with `payload[...]`; a legacy checkpoint without them raises `KeyError` (no `.get(..., 0.0)`) |

`hidden = max(64, 2 * latent)` stays: it fixes the checkpoint shape, not a tunable.

### features/base.py, dino.py, talk2dino.py, maskclip.py

| name | verdict | action |
|---|---|---|
| `_DEBIAS_VALIDATED` | merge | becomes class attribute `debias_validated: bool = False`, set `True` on validated backends; warning reads it |
| backend `forward` x3 | merge | base `forward` calls abstract `_patch_tokens(batch)`; backends declare `n_prefix_tokens` (dinov2 1, talk2dino 5, maskclip 0), removing `[:, 1:]` / `[:, 5:]`; any other token count raises `ValueError` (post-review: deriving `N - H_p * W_p` hid a wrong prefix) |
| maskclip `_patch_tokens` | inline | drop its `F.normalize`; `_tokens_to_feature_map` already L2-normalizes (post-review) |
| backend `device` property x2 | merge | one property on the base class |
| `get_bias_visualization` | inline | call `features_to_rgb` instead of repeating its PCA body |
| `score_queries` | keep | apply the `["object"]` negative default before the debug log |
| `DINOFeatureExtractor.model_name` | delete | never read |
| `maskclip_onnx` import | raise | stays inside `__init__` (optional dep) with an `ImportError` naming the install command |
| `preprocess`, `name`, `features_to_rgb`, `encode_text`, `compute_similarity`, `patch_size` | keep | — |

### segmentation/base.py

| name | verdict | action |
|---|---|---|
| `BaseSegmentation`, `aggregate_masked_features`, `create_patch_mask`, `mask_id_to_binary_mask` | keep | — |
| `convert_matched_mask` | raise | `assert` -> `ValueError` |
| `create_composite_mask` | make-kwarg | `min_visible_frac=0.1` replaces the literal; empty `results` raises `ValueError` instead of returning `zeros((0, 0))` |

### segmentation/mobile_sam.py

| name | verdict | action |
|---|---|---|
| `load_mobile_sam` | make-private | `_load_mobile_sam`; one caller; drop from `__init__` exports |
| `MobileSAMSegmentation.__init__` | raise | validate `strategy` here (`ValueError`), not in `segment` |
| `_segment_object` | make-kwarg | `box_batch_size=320` on `__init__`, threaded down |
| `MobileSAMSegmentation.segment` | raise | nothing detected returns an empty `(0, H, W)` mask stack, not `None` |

### segmentation/sam3.py

| name | verdict | action |
|---|---|---|
| module-level `try/except ImportError` stub | delete | import inside `__init__` with an `ImportError` naming the install command |

### segmentation/insid3.py

| name | verdict | action |
|---|---|---|
| `INSID3Segmentation.__init__` | make-kwarg | `device` passed straight to `DINOFeatureExtractor`, which resolves `None` with `get_device()` (post-review: no duplicate default); `fallback_quantile=0.9` threaded to `_locate_candidates` |
| `set_context` | raise | empty mask raises `ValueError` (today it warns and keeps the previous context), checked on the input mask before the backbone runs (post-review); drop the re-normalize of already-normalized features |
| `segment_with_mask` | keep | `try/finally` so the temporary context is always cleared |
| `_cluster_prototypes` empty-cluster branch | delete | unreachable (labels come from the same points) |
| `_seed_and_aggregate` `>= 0` guards + `else` | delete | unreachable |
| `_downsample_mask` fallback cascade | keep | each step covers a real shape case; one-line why |
| `_agglomerative_clustering`, `_upsample_mask`, `_tensor_to_pil`, `segment`, `clear_context` | keep | — |

### segmentation/sky.py

| name | verdict | action |
|---|---|---|
| `SkyWaterSegmentation` | keep | — |
| `sky_masks` | make-kwarg | cache the sky probability as an 8-bit PNG and threshold on read; `threshold=0.5` kwarg, so a changed threshold no longer serves a stale cache; a threshold outside `[0, 1)` raises `ValueError` (post-review). Old 0/255 PNGs still read correctly (they threshold to the same mask). |

### `__init__.py`

- Export `INSID3Segmentation`, `SkyWaterSegmentation`, `sky_masks`; drop `load_mobile_sam`.

## Caller sweep

Checked with `grep -rn` over `collab_splats configs scripts tests evals docs/source`.

| changed | outside callers | effect |
|---|---|---|
| `cache_store_path` raise on >1 | `dashboard/pipeline.py`, `dashboard/viewer.py` | only on an ambiguous store dir, which is already a bug |
| `extract_feature_cache`, `point_features_cached` narrow except | `wrapper/reconstructor.py`, `dashboard/*` | none for missing/stale stores |
| `FeatureAutoencoder.load` KeyError | `dashboard/viewer.py`, tutorials 05/06 | legacy AE checkpoints must be refit |
| MobileSAM empty instead of `None` | tutorial 04 `segmentation.ipynb` | `is None` check becomes dead — deferred |
| `sky_masks(threshold=)` | `wrapper/reconstructor.py`, `evals/scripts/eval_sky_mask.py` | none (default = today) |
| `load_mobile_sam` private | none outside semantics | — |
| `tokens_to_feature_map` private | none outside semantics | — |

## Deferred (hands-off paths)

- `configs/base.yaml` semantics block: `target_cosine` carries 8 lines of measurement lore.
- Tutorial 04 `segmentation.ipynb`: drop the `None` check after MobileSAM `segment`.
- `tests/test_semantics_logging.py` lives outside `tests/semantics/`; move it.
- Grouping revival: `grouping.py` on `main` calls `segment()` then
  `create_composite_mask(results)` — must handle the empty-result and `ValueError` changes.

## Testing

- Baseline gate on the untouched fork before any edit:
  `tests/semantics tests/utils tests/test_docstring_contract.py`, plus canaries
  `tests/test_semantics_logging.py tests/dashboard/test_pipeline.py
  tests/dashboard/test_viewer_lift.py tests/dashboard/test_semantics_layout.py
  tests/wrapper/test_mask_sky.py`. Record pass/fail/skip.
- Gate per commit: same set, `__file__` proof line, SKIP count compared to baseline.
  No full-suite runs (concurrent session, 46.6 GB cgroup).
- Each behavior change gets a test: the new raises, `min_visible_frac`, `box_batch_size`,
  `fallback_quantile`, MobileSAM empty result, INSID3 empty-mask raise + cleared context
  after a failing `segment_with_mask`, sky cache re-thresholding + old-format read,
  hoisted `forward` token count per backend.
- Tests of deleted code are deleted.
- nvdiffrast missing from the shared venv: stub it via a pytest plugin in the scratchpad
  if any gate import pulls it; never committed.
- End: add `semantics` to `RELEASED` in `tests/test_docstring_contract.py`; it must pass.

## Out of scope

- Readability rewrite of `tests/semantics/`.
- Any other package, config, notebook or doc outside this spec and its plan.
