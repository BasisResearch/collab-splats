# Pointcloud release cleanup — design

Date: 2026-09-26 · Branch: `clean/pointcloud-release` off `clean/final` (`1d1fdf20`), sfm
backends squashed in at `a29efaa0` · Status: approved 2026-09-26

Rules: [017-release-cleanup-rules.md](../decisions/017-release-cleanup-rules.md) — binding,
not restated here. SfM design: [018-sfm-backends.md](../decisions/018-sfm-backends.md) — not
re-litigated. Reference runs: [preproc](2026-09-24-preproc-release-cleanup-design.md),
[semantics](2026-09-24-semantics-release-cleanup-design.md),
[geometry](2026-09-25-geometry-release-cleanup-design.md).

## Goal

Make `collab_splats/pointcloud/` release-ready: less code, brief docs, no dead or duplicated
code, no silent fallbacks, every helper sourced from one module — **without changing what the
pointcloud stage writes**, except in the named bug-fix commits.

## Decisions (from brainstorm)

- **Base:** `clean/final` (Q1 a).
- **SfM backends in scope** (user 2026-09-26, lifts Q2 a): colmap + hloc landed as
  `a29efaa0`; `sfm/*`, `sift_db.py` and `_run_sfm` get the same Round 1 + Round 2 treatment.
  Still **no shared SfM base class** — the three creators are single-method dataclasses with
  different fields; shared steps become functions in a new `sfm/common.py`.
- **Real-scene gate: pointcloud stage only** (Q3 A), BA / LC / semantics off,
  `pointcloud.zarr` byte-compared.
- **Output frozen outside section C.** Every Round 2 commit keeps the parity fixtures
  `np.array_equal`. Section C holds the known numeric bugs, one `fix(pointcloud):` commit
  each, fixture re-baselined explicitly in that commit.
- **One source per helper.** Where pointcloud re-implements a helper that exists in
  `utils/torch_utils`, `geometry.transforms` or pointcloud itself, it calls that one.
  Duplicates *outside* pointcloud are listed under Follow-ups, not fixed here.
- **API breaks allowed** (017); callers, configs and tests change in the same commit.
- **Tests:** prune like preproc — delete tests of deleted code and history guards, merge
  duplicates, no readability rewrite.
- **Isolation:** worktree `.worktrees/pointcloud-release`, `third_party/{LoGeR,
  Video-Depth-Anything,hloc}` symlinked in; every gate runs as
  `cd <wt> && PYTHONPATH=<wt> python ...` and prints `collab_splats.__file__`.
- **Hands-off:** notebooks on `clean/final` and `.worktrees/tutorial-rework`
  (read-only grep). Impact listed under Tutorial impact.

## Starting point

Contract release checks (pointcloud in `PACKAGES`, not `RELEASED`): 21 xfailed at `a29efaa0` —
silent fallbacks, `UPPER_CASE` numeric constants, banned words, comment-cap runs across
`feedforward/*`, `utils.py`, `vda.py`, `depth_align.py`, `sfm/*`.

Shared venv regressed (site-packages dated 2026-06-02; no install allowed):

- `vismatch` missing → 17 failures, `tests/localization/test_local_matcher.py`
- `nvdiffrast` missing → 16 collection errors (all of `tests/wrapper`,
  `tests/geometry/test_metrics.py`, `tests/pointcloud/test_loger_creator.py`) via the hard
  import in `mesh/texture.py:18`
- gsplat 1.4.0, no `gsplat.losses` → collection errors in splats tests
- `hloc` not installed → hloc unit tests mock it; no hloc real-scene run

**G′ gate:** runs with a collection-only `nvdiffrast` stub on `PYTHONPATH` (scratchpad
`stubs/`, raises `ImportError` on any attribute), so `tests/wrapper` collects. Baseline and
tip use the identical stub. Gate paths: `tests/pointcloud tests/geometry tests/wrapper
tests/evals tests/localization tests/remote tests/test_docstring_contract.py`.

G′ baseline at `a29efaa0`: **18 failed / 1596 passed / 10 skipped / 21 xfailed / 59 xpassed /
3 errors** (197 s), all environment:

- 17 failed: `vismatch` missing
- 1 failed (`test_absent_confidence.py::test_splats_depth_targets_skip_masking_when_confidence_absent`)
  + 3 errors (`test_splats_stage.py`, `test_analyze_splats.py`, `test_eval_splats.py`):
  `gsplat.losses` missing

(Pre-squash `1d1fdf20`, without `tests/remote`: 18 / 1351 / 10 / 20 / 55 / 3 — same 21.)

## Round 1 — prose only

No code change. Proof: AST equal after deleting every docstring statement on both sides and
stripping comments (geometry's `astcmp.py`), plus one sanity mutation showing it can fail.

- Every `comment-cap` and `banned-word` hit, every missing/malformed docstring (rows 24-33,
  48-49 prose parts), US spelling.
- Path-comment headers at `sfm/__init__.py:1`, `depth_align.py:1` (row 3).
- False docstrings fixed: `_raw_to_world_points`, `depth_align` names, `utils` module
  docstring, `base.py:95` track prose, geometry `bundle_adjustment.py:96,218`
  (`creator.reproject` → `result.reproject()`).
- SfM prose (row 71):
  - `instantsfm.py:2` "the one backend `pointcloud.method: sfm` dispatches to" — false
  - `instantsfm.py:42` scene id, `:290` "measured"; class docstring 9 bullets → ≤6
  - `instantsfm.py` comment runs `:275-281, :285-292, :336-341, :351-360, :368-372, :393-397`
  - `sift_db.py:5` names only instantsfm + colmap (hloc uses it too); `:34` dated URL check;
    `build_sift_database` docstring (`measured`, timings, host thread count, history);
    `sift_database_valid` docstring shape; history comments `:164-166`, `:182`
  - `reconstructor.py` `_run_sfm` docstring (Returns section); comment runs `:1212-1216`,
    `:1218-1222`, `:1228-1236` (numbers, "left out of this commit")
  - `depth_align.py` docstring "2026-08-11 mesh-regression" history
- `configs/base.yaml`: `:78-89` block to header + bullets; every hloc key gets a comment;
  colmap `num_threads` "spawns 96" → host-independent wording.
- `docs/source/api/pointcloud.rst`: add `sfm.sift_db` (plus `sfm.common` once row 90 lands).
- Delete the ROADMAP comment (row 21); cite upstream for 518 (row 19) and for `d3e599e`
  (row 69).

## Round 2 — code, one commit per logical change

Each row: `# | name | verdict | action`. Rows 1-85 are the approved verdict table (updated to
`a29efaa0`); rows 86+ are the SfM review.

### pointcloud/__init__.py, feedforward/__init__.py

| # | name | verdict | action |
|---|---|---|---|
| 1, 85 | omega / loger `try/except ImportError` guards | delete | hard import in both `__init__`s (`pointcloud/__init__.py:14-26`) |
| 2 | `ff/__init__` layout | keep | `########` dividers |
| 36 | `unproject_and_filter_points` re-export | delete | callers import from its home |
| 37 | `_raw_to_world_points` re-export | delete | private |
| 86 | `ColmapCreator` / `HlocCreator` re-export (`:12, :77-78`) | delete | zero callers; `InstantSfMCreator` isn't re-exported; import from `.sfm` |

### base.py

| # | name | verdict | action |
|---|---|---|---|
| 35, 72 | `_REGISTRY` dict (feedforward-only since `a29efaa0`) | merge | `BaseFeedforwardCreator(RegistryMixin)`; reconstructor's hand-built `creator_map` (`reconstructor.py:525-530`) → `get_creator`; `_FEEDFORWARD_BACKENDS` (`:78`) derived from the registry like `_SFM_BACKENDS = set(SFM_CREATORS)`. `RegistryMixin.get` raises `ValueError` "Unknown '…'": `test_registry.py:14-17, :41-43, :53-57` and `tests/geometry/loop_closure/test_wrapper.py:163-167` updated. `SFM_CREATORS` stays a plain dict |
| 38 | `PointcloudResult` K / extrinsics built by hand (`:141-144`, `:167-171`) | merge | `calibration_matrix()` / `extrinsics_to_homogeneous`; also reached by colmap/hloc `SIMPLE_RADIAL` |
| 39 | `features` field, `load_features` (`ff/base.py:82,242,259,279`) | delete | + 3 dashboard call sites |
| 5 | `"conf"` key fallback in `ff/base.py` `load_zarr` (`:283-285`) | delete | scanned 2026-09-26: none of the 3 stored `pointcloud.zarr` under `/workspace/outputs` (none under `data/`) has a `conf` key |
| 75 | bf16→fp32 / detach / cpu in `save_zarr` (`ff/base.py:201-207`, `:224-230`) and `lift_features` (`utils.py:293-297`) | merge | new `torch_utils.to_numpy` |

### feedforward/base.py

| # | name | verdict | action |
|---|---|---|---|
| 4 | `FeedforwardResult.save` / `.load` | delete | `tests/integration/test_pipeline_cu121.py:100-120` → `load_zarr` |
| 6 | `_raw_to_world_points` silent path | raise | + docstring fix; mapanything `_lc_collate_outputs` raise, same commit |
| 7 | `intrinsics_downsampled` alias | merge | one key; drop `vggtx.py:337` `.get` |
| 8 | SPARK-era classvars | raise | default `None`, raise when unset |
| 9 | `_verify_loop_candidate` | raise | delete `layer_index`; ratio required; missing poses raise; delete LC wrapper guard `wrapper.py:375-384` same commit |
| 10 | `reproject` / `_reproject` on creators | delete | `FeedforwardResult.reproject()` is the one; `test_wrapper.py:274-290` + stubs |
| 11 | `_source_paths` | delete | + its test |
| 12, 45 | `_rescale_reconstruction_to_original_dimensions` | delete | dead kwargs (`shared_camera`, `shift_point2d_to_original_res`, `verbose`), `image_paths` param, dead `rescale_camera` local, `deepcopy`; non-pinhole model raises |
| 13 | `console` prints | keep | → `logger`; `tests/test_feedforward_logging.py` capsys → caplog |
| 14, 44 | `build_pycolmap_reconstruction` | raise | unknown camera model raises; drop 3x4/4x4 guessing (one shape); non-uint8 images raise |
| 15 | multiview CPU fallback | raise | — |
| 16, 77 | vggtx / omega / loger / mapanything `_postprocess` bodies | merge | base helper; drop dead `extrinsic_global_4x4`, `_raw_list`, `ndim` guesses; multiview-confidence block shared by all four. The double unprojection stays here (B6 removes it) |
| 41 | `**kwargs` on template steps | delete | — |
| 42 | `_lc_collate_outputs` base default | delete | abstract where LC-capable |
| 43 | `extract_intermediate_features` | raise | `layer_index` required; base raises `NotImplementedError`; loger stub deleted |
| 46 | `_raw_to_world_points` own unprojection | merge | onto the existing unprojection; drop its subsample |
| 47 | verify `.get(...)` defaults | raise | index directly |
| 50 | q/k attention hook, pose decode, `_forward` tails ×3 | merge | base helpers |
| 51, 81 | `frame_{idx:06d}` built inline (ff base `setup_inference`, `reconstructor.py:1195`, `preproc/frames.py:133`) | merge | new `preproc.frames.frame_name(idx)` beside its inverse `frame_idx_from_path`; one-function touch of released preproc |
| 55, 83 | checkpoint resolve ×3 + VDA | merge | one helper |
| 73 | device picking at `ff/base.py:1173`, `utils.py:277`, `sift_db.py:159` | merge | `get_device()` (`use_gpu = get_device() == "cuda"` for the colmap CLI) |
| 76 | `unproject_and_filter_points` in `vggtx.py:91` | merge | moves to `ff/base`; mapanything `:462-467` reuses the tail via `_mask_to_points` |
| 82 | manual `empty_cache` / `gc` | merge | `pytorch_gc` in `run_inference`, `vda` |
| 48 | annotations, style | — | contract |

### feedforward/vggtx.py, vggt_omega.py, loger.py, mapanything.py

| # | name | verdict | action |
|---|---|---|---|
| 52 | vggtx autocast dtype | inline | one expression |
| 53 | vggtx unused `console` import, false comments | delete | — |
| 20 | `VGGT_OMEGA_DEFAULT_RESOLUTION` | delete | unused |
| 54 | omega `enable_text_alignment`, `model_repo`, `model_filename` | delete | never set |
| 19 | `target_size` param (518) | delete | inline 518, upstream citation |
| 18 | `LOGER_CONF_THRESHOLD` | make-kwarg | `_PATCH` → private |
| 56 | loger `model_repo`, bare `assert` | delete / raise | — |
| 74 | `sys.path` insert in `loger.py:257-283`, `vda.py:41-47` | merge | new `torch_utils.vendored_path(root, hint)` context manager; vda stops leaking `sys.path` |
| 78 | full-frame `original_coords` in `loger.py:349-351`, `depth_align.py:304` | merge | `full_frame_coords(W, H, N)` in `ff/base` |
| 17, 57 | mapanything | raise | silent paths (incl. `:586-593`); dead `Tensor` branch; device loop ×4; `kwargs.get`; `pts3d` guards |

### utils.py

| # | name | verdict | action |
|---|---|---|---|
| 23 | `fit_dominant_plane` | delete | + tests; no caller |
| 22, 59, 60 | `lift_features` (`:301-302`), `reproject_pixels` (`:379-380`) shape guessing | raise | one input shape each (`reproject_pixels` takes 4x4 only); torch-only confidence; one `grid_sample` path |
| 58 | `subsample_points(conf=, conf_percentile=)` | delete | + `README.md:171`, 3 tests |
| 61, 84 | `cross_frame_attention_ratio` | raise | dead branch goes; split into raw ratio + `mean_top_quarter`; `evals/scripts/eval_similarity_calibration.py:61,181` imports both |
| 62 | module docstring | — | Round 1 |
| D2 | `confidence_mask` all-True fallback | keep | + `logger.warning` |
| D1 | `confidence_percentile` name | keep | — |

### vda.py

| # | name | verdict | action |
|---|---|---|---|
| 63 | fp32 flag | make-kwarg | drop redundant `astype` (`:147`) |
| 80 | depth `.npy` path built twice (`:100`, `:141`) | merge | one private helper |

### depth_align.py

| # | name | verdict | action |
|---|---|---|---|
| 64 | `np.vstack` 4x4 (`:249-251`) | merge | `extrinsics_to_homogeneous` |
| 65 | unreachable `missing` check (`:86-88`), `_tracked_point3d_ids` (`:26`, `:286`), `n_fallback` (`:177`, `:280`) | delete / inline | row-count check moves up. The partial-registration check `:219-225` is **live** (colmap/hloc subset contract) and stays |
| 79 | world→cam by hand (`:102-104`) | merge | `Rigid3d * xyz`; dropped if parity fails |

### sfm/common.py (new) and sfm/sift_db.py

Shared SfM steps, today copied ×3 (`colmap.py:65-74,99-104`, `hloc.py:106-112,172-176`,
`instantsfm.py:342-365,374,425-428`).

| # | name | verdict | action |
|---|---|---|---|
| 87 | image dir, `colmap_dir` mkdir, `rmtree(sparse)`, `names` | merge | `prepare_sfm_dirs(data_dir, images_dir) -> (image_dir, colmap_dir, names)` in `sfm/common.py` |
| 88 | `sparse/0` mkdir + stem rename + write + log | merge | `write_sfm_model(recon, colmap_dir, label, n_frames)` in `sfm/common.py`; returns the model each creator returns today (instantsfm's re-read stays in instantsfm) — the colmap/hloc re-read is B9, not this row |
| 89 | SIFT DB validity + rebuild (`colmap.py:72-88`, `instantsfm.py:373-377`) | merge | `ensure_sift_database(image_dir, db_path, names, *, pairing, overlap, num_retrieved, vocab_tree, num_threads)` in `sift_db.py`; absorbs `database_params_path`, `matching_params`, `write_database_params`, `sift_database_valid` (its `params=None` path goes). instantsfm now writes the params sidecar too: an existing instantsfm DB without one rebuilds once (cache miss, same SIFT) |
| 90 | `sfm_image_dir`, `rename_images_to_stems` | merge | move from `sift_db.py` to `sfm/common.py` (output-contract helpers, not SIFT; 3 callers each) |
| 91 | `largest_model` (`sift_db.py:355`) | inline | into `colmap.py`, its one caller. hloc's warning (`hloc.py:168-170`) stays: hloc already picked its model, it only counts `models/` dirs |
| 92 | `.get(pairing, "sequential_matcher")` (`sift_db.py:183`) | raise | explicit map of all 4 pairings, indexed |
| 93 | `colmap_cli_version` → `"unknown"` (`sift_db.py:326-330`) | raise | missing binary / empty output raises |
| 94 | `fetch_vocab_tree(cache_dir=None)` | keep | test seam |
| 95 | `PAIRINGS`, `VOCAB_TREE_*` | keep | string/Path constants; `PAIRINGS` read by the reconstructor |

### sfm/colmap.py, sfm/hloc.py, sfm/instantsfm.py, sfm/__init__.py

| # | name | verdict | action |
|---|---|---|---|
| 96 | `ColmapCreator(images_dir=None)` | make-kwarg | required; every caller passes it |
| 97 | `HLOC_PIN`, `sequential_pairs`, lazy hloc import + `ImportError` | keep | pin checked against `setup/hloc.sh`; optional heavy dep |
| 98 | provenance: `HLOC_PIN` / `colmap_cli_version` pulled into `_run_sfm` if/elif | merge | each creator gets `provenance() -> dict`; reconstructor calls it; same keys in the same order (zarr attrs byte gate) |
| 67 | `instantsfm.py:176` `hasattr(..., "filenames")` → `f"{idx}.jpg"` | raise | read the attribute directly |
| 68 | `instantsfm.py` redundant casts, `Path` round-trip (`:283,295,342-377,416-417`) | delete | `rename_images_to_stems` kept (row 90) |
| 30 | `sfm_image_dir` | keep | 3 callers now; tests already in `test_sift_db.py:225-241` (move with row 90) |
| 99 | `getattr(..., "_collab_splats_*", False)` markers | keep | idempotency guards |
| — | `SFM_CREATORS` | keep | dispatch table (018) |

### wrapper/reconstructor.py (lane C)

| # | name | verdict | action |
|---|---|---|---|
| 35 | `creator_map`, `_FEEDFORWARD_BACKENDS` | merge | see base table |
| 51, 81 | `_load_pointcloud_from_disk` naming (`:1195`) | merge | base helper |
| 100 | `_SFM_BLOCK_KEYS` (`:82-95`) | merge | derived from `dataclasses.fields(SFM_CREATORS[b])` + `min_registered_frac` |
| 101 | instantsfm special-case constructor in `_run_sfm` (`:1244-1250`) | merge | generic `SFM_CREATORS[backend](**kwargs)` (`:1253`); the instantsfm block's 3 keys equal its dataclass fields (bar `use_depths`), so the same kwargs reach it; instantsfm block gains the unknown-key check (B10) |
| 102 | `_validate_sfm_block` `pc.get(backend) or {}` (`:1022`); instantsfm `pc.get("instantsfm", {})` + `.get` (`:980-994`) | raise | index directly; `base.yaml` defines every block |
| 116 | `torch.cuda.synchronize` + `empty_cache` after the mapper (`:1256-1258`) | merge | `pytorch_gc()` |
| 103 | `names = [p.name for p in frame_paths(...)]` in `_run_sfm` | delete | each creator computes it (`prepare`) |
| 104 | `keyframes` reads every frame then subsets `[rows]` | merge | `read_frames(images_dir, idxs)` |
| 105 | `_registered_rows` (`:140`) | keep | names a step |
| B2 | refine `write_colmap` | fix | section C |

### Tests (SfM)

| # | test | verdict | action |
|---|---|---|---|
| 106 | `test_registry.py::test_sfm_backends_left_the_feedforward_registry`, `::test_old_keys_removed`, `::test_sfm_creators_maps_every_sfm_backend` | delete | history guards / literal restatement (`test_sfm_config.py:28` covers it) |
| 107 | `test_sfm_config.py` `test_sfm_rejects_bundle_adjustment` `:45`, `test_sfm_rejects_loop_closure` `:138`, `test_colmap_and_hloc_refuse_ba_and_lc` `:162` | merge | one test, 3 backends × 2 keys |
| 108 | `test_sfm_instantsfm_config_validates` `:35` + `test_colmap_and_hloc_validate_at_config_load` `:151` | merge | one test over 3 backends |
| 109 | `test_base_yaml_has_instantsfm_block` `:57`, `test_base_yaml_has_colmap_and_hloc_blocks` `:196` | delete | config-literal snapshots |
| 110 | `test_sfm_stage.py::test_run_sfm_instantsfm_attrs_are_unchanged`, `::test_run_sfm_random_seed_defaults_to_none`; `test_instantsfm.py:189`, `:211` `*_defaults_to_none` | delete | history guard / dataclass defaults |
| 111 | `test_sfm_stage.py` random_seed / min_views validation tests | merge | into `test_sfm_config.py` |
| 112 | `test_sift_db.py:225 test_sfm_points_at_the_scene_images_dir_and_stages_nothing`, `::test_params_none_ignores_the_sidecar` | delete | history guard; `params=None` gone (row 89) |
| 113 | `test_sift_db.py::test_exhaustive_argv_matches_the_pre_move_snapshot` | keep | rename `test_exhaustive_argv`, drop history comment |
| 114 | `_recon(names)` (`test_colmap:19`, `test_hloc:34`, `test_sift_db:190`), `_scene` (`test_colmap:37`, `test_hloc:135`) | merge | `tests/pointcloud/sfm/conftest.py` |
| 115 | partial registration through real pycolmap | add | model with an unregistered image → `result_from_reconstruction` unmocked (B9) |

### Kept (D)

D1 `confidence_percentile` name · D2 `confidence_mask` fallback + warning · D3 result fields ·
D4 one-caller helpers that name a step.

### Left alone

- vggtx percentile filter (`>=`, percentile-or-raw) vs `utils.confidence_mask` (`>`, all-True
  fallback): different semantics, merging changes output.
- hloc `pairs_path` equal to `retrieval_path` under `retrieval` pairing: harmless self-rewrite,
  one comment line.

## C — bug fixes (outside the parity gate)

One `fix(pointcloud):` commit each, after Round 2. Each re-baselines the affected fixture in
the same commit and states the numeric delta.

| # | bug | fix |
|---|---|---|
| B1 | mapanything `original_coords` stores a model-pixel box (`mapanything.py:224-229`) | original-pixel crop box, per the contract |
| B2 | refine stage rebuilds COLMAP with default `PINHOLE` while vggtx is `SIMPLE_PINHOLE` (`reconstructor.py:1341-1356`; feedforward only) | shared `write_colmap(result, out, camera_model)` |
| B3 | omega crop rounding differs from upstream | mirror upstream `int(round())` and `//2` |
| B4 | K rescale ignores crop offset (`_rescale_…`, `eval_splats.py:101-105`) | promote crop-aware `geometry/metrics.py:239 _scale_intrinsics_to_original` to `geometry.transforms`; real for vggtx portrait / omega aspect crops |
| B5 | `depth_align` pixel lookup uses `rint` (`:109-110`), contract uses floor | floor (+ `eval_verification.py:205`); changes instantsfm **and** colmap/hloc zarrs |
| B6 | dense unprojection runs twice per `_postprocess` (float64 vs float32 c2w) | compute once |
| B7 | `torch.linalg.inv` at `ff/base.py:612` | `invert_poses` |
| B8 | unseeded `randomly_limit_trues` (`vggtx.py:132-145`, `mapanything.py:464`) | seeded generator; real-scene baseline run twice to confirm determinism first |
| B9 | colmap/hloc return the in-memory mapper model; `depth_align.py:221` compares `len(reconstruction.images)` to the stems. A pycolmap 4.0.4 probe keeps deregistered images in `.images` in memory (3 images, 2 registered) and drops them on write + re-read | test row 115 first; if it fails, `write_sfm_model` returns the re-read `sparse/0` for colmap/hloc — the partial-registration subset then works. If it passes, no fix |
| B10 | instantsfm sub-block has no unknown-key check: `retriangulate: true` (typo) is ignored | row 101's generic dispatch validates against the dataclass fields; config now raises |

## Caller sweep

Checked with `git grep` over `collab_splats configs scripts tests evals docs/source README.md`.

| changed | outside callers | effect |
|---|---|---|
| registry → `RegistryMixin` | `wrapper/reconstructor.py`, `tests/pointcloud/test_registry.py`, `tests/geometry/loop_closure/test_wrapper.py:163-167` | `ValueError` on unknown name. Unchanged `get_creator` users: `evals/scripts/eval.py:63,261,735`, `ba_start_at_gt.py:39,96`, `loop_closure/wrapper.py:110`, `docs/source/api/geometry.rst:29-32` |
| `FeedforwardResult.save/load` delete | `tests/integration/test_pipeline_cu121.py` | `load_zarr` |
| creator `reproject` delete | `tests/wrapper/test_wrapper.py`, geometry BA docstrings | `result.reproject()` |
| `features` / `load_features` delete | `collab_splats/dashboard` ×3 | removed |
| `subsample_points` conf args | `README.md:171` | removed |
| `cross_frame_attention_ratio` split | `evals/scripts/eval_similarity_calibration.py` | both imported |
| `_verify_loop_candidate` raise | `geometry/loop_closure/wrapper.py:375-384` | guard deleted |
| `_scale_intrinsics_to_original` promote (B4) | `geometry/metrics.py`, `evals/scripts/eval_splats.py`, `tests/geometry/test_metrics.py:17,1268` | import path |
| `sfm_image_dir`, `rename_images_to_stems` move | `sfm/{colmap,hloc,instantsfm}.py`, `tests/pointcloud/sfm/test_sift_db.py` | import path |
| colmap/hloc re-export delete | none | — |
| `_SFM_BLOCK_KEYS` derived, instantsfm key check | `configs/base.yaml`, `evals/configs/*.yaml` | no key removed; a typo'd key now raises |

## Tutorial impact (not edited here)

Grepped read-only on `clean/final` notebooks and `.worktrees/tutorial-rework`; the plan lists
each hit. Known: creator `.reproject(...)` → `result.reproject()`; `FeedforwardResult.load` →
`load_zarr`; `subsample_points(conf=...)`; `fit_dominant_plane`. No notebook names the SfM
creators or `sift_db` helpers.

## Follow-ups (outside pointcloud — separate effort, user's call)

Same helper written in several packages. Recommended as one later "dedup" effort across
released packages:

- `get_device`: ~10 sites (geometry BA 330/560/629, localization, mesh, dashboard, evals)
- `pytorch_gc`: `reconstructor.py:578-583` (redundant inline `import torch as _torch`),
  LC wrapper 292/566, evals (`_run_sfm`'s copy is row 116)
- frame-dir decode vs preproc `read_frame`/`iter_frames`: 6 sites
- `reconstructor.py:172 _store_rows` duplicates the frame_idx→row map in
  `preproc/frames.py:172-176` → a `frames.frame_rows(dir, idxs)` in preproc
- `to_numpy`: `localization/extractors.py:99`, `mesh/io.py:145`
- Blosc lz4 compressor constant; DINO-SALAD loader in the eval script
- camera centers: 7 copies → one `geometry.transforms` function
- dashboard `_rotation_align` vs `rotation_align_vectors`
- `torch.linalg.inv` on poses → `invert_poses`: ~20 sites incl. splats
- manual `Rigid3d` poses: `localizer.py:1107`, `eval.py:372`, `eval_verification.py:61`
- Camera↔K: `localizer.py:1048`; `eval_verification.py:62` wrong for `SIMPLE_PINHOLE`
- `graph.py:628` meshgrid mirror, `analyze_splats.py:64`, `metrics.py:352` multiview loop

## Deferred

- SfM caches (SIFT DB, hloc h5 with `overwrite=False`, VDA depth) are keyed on image names
  only. A preproc re-run with the same frame indices but new pixels (e.g. undistort toggled)
  reuses stale keypoints in hloc; colmap/instantsfm catch it only if image dims change.
  Pre-existing across all three backends; needs a content/preproc hash in the key — its own
  change, user's call.
- `FeedforwardResult` rename (it names a method-agnostic contract).

## Testing

- **Baseline** before any code edit: G′ at `a29efaa0` (Starting point).
- **Gate per commit:** G′ run; passed count drops only by deleted tests, each named. No
  `| tail`, no `--tb=no`, no full suite.
- **Round 1:** astcmp proof + sanity mutation.
- **Parity (bit-exact):** scratchpad `parity.py --save/--check` — seeded, CPU, fp32 —
  on vggtx, omega, loger, mapanything `postprocess()` and
  `depth_align.result_from_reconstruction`. Saved at `1d1fdf20`; re-checked PASS at
  `a29efaa0`; determinism checked twice; a 1-ulp mutation fails 10 fields across all 5 cases.
  `np.array_equal` after every commit touching those paths; a failure reverts the commit.
  Section C commits re-save and report the delta.
- **Real scene:** `Reconstructor` pointcloud stage in tmux, `vggt_omega`, `instantsfm` and
  `colmap` (system COLMAP 3.10 present; hloc not installed, so no hloc run), BA/LC/semantics
  off, baseline vs tip; `pointcloud.zarr` byte-identical or every diff explained (expected
  only from C).
  - baseline runs **twice** first: GPU SIFT, InstantSfM and the unseeded subsample (B8) may
    not repeat; a backend that differs from itself is compared by metric (point count,
    registered frames, depth scale), not bytes, and the report says so
  - each backend runs in a fresh output dir, so no SIFT DB or VDA cache is reused
- Each new raise gets a test. Tests of deleted code are deleted.
- **End:** add `"pointcloud"` to `RELEASED` in `tests/test_docstring_contract.py`; it must
  pass. `graphify update .`.

## Lanes

Disjoint file sets, each in its own worktree, cherry-picked back in order A → B → C:

- **A:** `utils.py`, `vda.py`, `depth_align.py`, `sfm/*` (incl. new `common.py`) + tests
  (`tests/pointcloud/sfm/*`, `test_depth_align.py`, `test_utils*`)
- **B:** `feedforward/*`, `pointcloud/__init__.py`, `base.py` + tests + LC wrapper touch
- **C (after A and B):** `wrapper/reconstructor.py` rows 35, 51/81, 98, 100-104, B2 +
  `tests/wrapper/test_sfm_*.py`, `test_registry.py`

Cross-lane helpers land first, one commit each, on the branch so both lanes base on them:
`torch_utils.to_numpy`, `torch_utils.vendored_path`, `preproc.frames.frame_name`,
`full_frame_coords` (ff/base, used by `depth_align` in lane A).

Touches outside pointcloud ride with their row's lane: row 9 LC wrapper and row 39 dashboard
→ B; row 84 `evals/` → A. Section C commits run serially after C, on the branch.

## Out of scope

- Any notebook; readability rewrite of `tests/pointcloud/`.
- Outside-pointcloud dedup (Follow-ups); SfM cache keying (Deferred).
- 018's design decisions (backends, pairing modes, subset contract).
