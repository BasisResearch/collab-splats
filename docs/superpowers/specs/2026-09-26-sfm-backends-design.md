# SfM backends — wire ColmapCreator + HlocCreator into `pointcloud.method: sfm`

**Date:** 2026-09-26
**Status:** approved design, revised 2026-09-26 after a consistency pass (pre-plan)
**Branch:** `feat/sfm-backends` in `.worktrees/sfm-backends`, off `clean/final` @ `ac93e96b`
**Sequencing:** lands BEFORE the pointcloud release cleanup; that cleanup runs on the result

## Goal

`pointcloud: {method: sfm, backend: colmap | hloc}` runs end to end through `Reconstructor`,
exactly as `instantsfm` does today, producing the same downstream artifacts
(`<backend>/colmap/sparse/0`, `pointcloud.zarr`, stem image names, original-res K).
`instantsfm` output stays byte-identical.

## Base branch

- `clean/pointcloud-release` does not exist
- `clean/geometry-release` is already squashed onto `clean/final` (`e52117da`) and deleted
- so: `clean/final` @ `ac93e96b`

## Decisions (settled in brainstorming)

| # | Question | Decision |
|---|---|---|
| 1 | hloc dependency | editable uv **path** source on the `third_party/hloc` clone, pinned `c13273b`, optional extra `hloc` |
| 2 | matching | one shared `pairing` key, same value set for both backends |
| 3 | mapper | incremental only (`pycolmap.incremental_mapping`); no `mapper` key |
| 4 | intrinsics | one shared `SIMPLE_RADIAL` camera per scene, refined by the mapper — same as instantsfm |
| 5 | config | per-backend blocks beside `instantsfm:`; only the block matching `backend` is read |
| 6 | partial registration | colmap/hloc subset to registered frames above a floor; instantsfm stays strict |

Rationale worth keeping:

- 1: a non-editable git install ships only the `hloc` package, and hloc's `superpoint` /
  `superglue` modules `sys.path`-append `../../third_party` for the SuperGlue submodule
  (`hloc/extractors/superpoint.py:8`, `hloc/matchers/superglue.py:6` @ `c13273b`); editable
  keeps SuperPoint available. vismatch was rejected: not installed in the venv and its
  COLMAP export is mid-rework under `vismatch-fork`.
- 1: SuperGlue is superseded by LightGlue (same group; equal or better accuracy, faster), so
  the default matcher is `superpoint+lightglue`. SuperPoint/SuperGlue weights are Magic Leap
  **non-commercial** — recorded in the decision record, same footing as InstantSfM/VDA.
- 3: the system `colmap` is 3.10-dev (Dockerfile:81, `colmap/colmap:20240213.23`) with no
  `global_mapper`; instantsfm already covers global SfM. A COLMAP 4.x image is a follow-up.
- 4: identical camera freedom is what makes the instantsfm A/B fair.

## Config

Two new blocks in `base.yaml`, beside `instantsfm:`. The defaults above them do not change
(`method: feedforward`, `backend: vggt_omega`); only the `backend` comment gains the sfm names.
A run selects one with `pointcloud: {method: sfm, backend: colmap}`.

```yaml
pointcloud:
  # ...existing keys and instantsfm: block unchanged...
  colmap:
    pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
    overlap: 10              # sequential: pair each frame with the next N
    num_retrieved: 20        # retrieval: vocab-tree neighbors per image
    num_threads: 8           # CPU SIFT/mapper thread cap; colmap default spawns 96 and OOMs
    min_registered_frac: 0.5 # fail below this share of frames registered
  hloc:
    pairing: sequential+retrieval
    overlap: 10
    num_retrieved: 20        # retrieval: top-k neighbors per image
    retrieval_conf: netvlad  # hloc.extract_features.confs key
    feature_conf: superpoint_max
    matcher_conf: superpoint+lightglue  # hloc.match_features.confs key
    num_threads: 8
    min_registered_frac: 0.5
```

`pairing` values, per backend:

| value | colmap (CLI, GPU when available) | hloc |
|---|---|---|
| `sequential` | `sequential_matcher --SequentialMatching.overlap N --SequentialMatching.quadratic_overlap 0` | in-repo pairs: frame i with i+1..i+N |
| `retrieval` | `vocab_tree_matcher --VocabTreeMatching.num_images num_retrieved` | `pairs_from_retrieval`, top `num_retrieved` |
| `sequential+retrieval` | `sequential_matcher` as above + `--SequentialMatching.loop_detection 1` | union of both pair sets, deduplicated |
| `exhaustive` | `exhaustive_matcher` | `pairs_from_exhaustive` |

- flag names checked against the installed binary (`colmap sequential_matcher -h`, 3.10-dev @ 879a296a)
- `quadratic_overlap 0`: colmap's default 1 also pairs i with i+2^k, which hloc's generator does
  not; off, both backends mean the same thing by `sequential`
- `sequential+retrieval` is NOT the same pair set across backends: colmap's `loop_detection` fires
  every `loop_detection_period` (10) frames, hloc retrieves for every frame. So colmap vs hloc
  compares pipelines, not matchers. Acceptable; stated in the report.
- so colmap also takes `num_retrieved` (default 20, unused unless `pairing` retrieves)

Validation in `Reconstructor.validate_config`, the `random_seed` / `min_num_view_per_track` way:

- `pairing` in the four-value set
- `overlap`, `num_retrieved`, `num_threads`: int >= 1 (bool rejected)
- `min_registered_frac`: number in (0, 1]
- hloc conf keys: non-empty strings — NOT checked against `hloc.*.confs`, which would import
  the optional extra at config load
- BA / LC refusals kept for every sfm backend; messages become backend-neutral

## Components

```
collab_splats/pointcloud/sfm/
  sift_db.py     NEW, not re-exported — moved out of instantsfm.py, not InstantSfM-specific
                   build_sift_database(image_dir, db_path, *, pairing="exhaustive", overlap,
                                       num_retrieved, vocab_tree, num_threads=8)
                   sift_database_valid(db_path, names, *, params=None)
                   rename_images_to_stems(recon, sparse_dir)
                   sfm_image_dir(images_dir)
                   largest_model(recons)                   by num_reg_images
  instantsfm.py  imports the above; defaults reproduce today's argv exactly, params=None
  colmap.py      ColmapCreator(pairing, overlap, num_retrieved, num_threads)
  hloc.py        HlocCreator(pairing, overlap, num_retrieved, retrieval_conf,
                             feature_conf, matcher_conf, num_threads)
  __init__.py    SFM_CREATORS = {"instantsfm": ..., "colmap": ..., "hloc": ...}
```

### Shared creator contract

All three sfm creators share instantsfm's shape, not `BasePointcloudCreator`:

- `reconstruct(data_dir, images_dir) -> pycolmap.Reconstruction`
- model written to `data_dir/colmap/sparse/0` (stale `sparse/` removed first)
- image names renamed to filename stems (`frame_NNNNNN`)
- `colmap` / `hloc` leave `pointcloud/__init__.py` `_REGISTRY`, which becomes feedforward only;
  its callers (`geometry/loop_closure/wrapper.py`, `evals/scripts/eval.py`,
  `evals/scripts/ba_start_at_gt.py`) are all feedforward. `tests/pointcloud/test_registry.py`
  `test_get_creator_colmap` / `_hloc` flip to asserting a `KeyError`.
- `BasePointcloudCreator` is then feedforward's base only; `base.py` docstrings ("every backend
  implements", "resolve through the registry") and `pointcloud/__init__.py`'s ("sfm backends
  (colmap, hloc) all resolve through get_creator") are corrected. The class stays: moving it into
  `feedforward/` is cleanup scope.
- moved helpers lose their InstantSfM wording ("InstantSfM expects an image directory",
  "InstantSfM: running colmap ..."); log / error text only

### ColmapCreator

1. `build_sift_database` into `colmap/colmap.db` (reused when valid, see Caching)
2. `pycolmap.incremental_mapping(db, image_dir, colmap/mapper/, options={"num_threads": num_threads})`
   — writes one `mapper/<idx>/` per model, so it cannot target `sparse/` directly
3. `largest_model` — incremental mapping gives no size ordering; warn when > 1
4. rename to stems, `write_binary` to `sparse/0`, remove `mapper/`

Measured 2026-09-26: a colmap 3.10 CLI database (0 rigs, 0 frames in pycolmap 4.0.4's view)
maps fine under `pycolmap.incremental_mapping` — 12/12 registered, 11,236 points on 12 GH010229
frames. Image names come back with `.png`, so the stem rename is needed here too.

### HlocCreator

- `hloc` imported inside `reconstruct`; `ImportError` names `setup/hloc.sh` + the user-run sync
- intermediates under `colmap/hloc/`; `sfm_dir = colmap/hloc/sfm` (its `database.db` never
  collides with verify's `colmap/database.db`)
- pairs file per `pairing` (table above); sequential generator is in-repo (~10 lines, hloc has none)
- `extract_features.main` / `match_features.main` with the configured confs
- `hloc.reconstruction.main(camera_mode=SINGLE, image_options={"camera_model": "SIMPLE_RADIAL"},
  mapper_options={"num_threads": num_threads})` (`hloc/reconstruction.py:142-189` @ `c13273b`)
  - keeps the largest model itself (`:117-139`) but only logs it; the warning on > 1 model counts
    `sfm_dir/models/*`
  - returns `None` (logs an error, does not raise) when nothing reconstructs (`:112-114`) — we raise
  - `mapper_options` merges into the pipeline options with `num_threads` default
    `min(cpu_count, 16)` (`:105`), so the cap is effective
- rename the returned model to stems, `write_binary` to `sparse/0`
- runtime downloads: netvlad and LightGlue weights on first use — prefetched in `setup/hloc.sh`,
  as `setup.sh` already does for vismatch

### `_run_sfm`

1. VDA depth — unchanged
2. build the creator from `SFM_CREATORS[backend]` with the block minus `min_registered_frac`
   (a `_run_sfm` concern, not a creator field); instantsfm's construction stays as today;
   then `.reconstruct(backend_dir, images_dir=images_dir)`
3. colmap / hloc only: filter `names`, `depths`, `keyframes` to registered stems; raise
   `RuntimeError` with N/M below `min_registered_frac`, else warn
4. `result_from_reconstruction` — unchanged logic
5. zarr attrs, beside `method` / `backend` / `align_attrs`:
   - instantsfm: `instantsfm_version`, unchanged
   - colmap: `pycolmap_version` (mapper) + `colmap_cli_version` (SIFT; first line of `colmap -h`)
     — features and mapping come from different COLMAPs (3.10 CLI vs pycolmap 4.0.4)
   - hloc: `pycolmap_version` + `hloc_commit` from a `HLOC_PIN = "c13273b"` constant in
     `sfm/hloc.py`; `hloc.__version__` is `"1.5"` at the pin and identifies nothing. A unit
     test asserts the constant equals the pin in `setup/hloc.sh`.
   - colmap / hloc: `registered_frames`, `total_frames` (no clash with `align_attrs` keys)

The `NotImplementedError` guard in `_run_sfm` goes; `_SFM_BACKENDS = set(SFM_CREATORS)`.

## Caching

- colmap: DB valid only when its image set AND its matching params (`pairing`, `overlap`,
  `num_retrieved`) match; params live in a `colmap.db.json` sidecar written after a successful
  build. Without it a `pairing` change silently reuses stale matches.
- `sift_database_valid(..., params=None)` skips the sidecar, so instantsfm's `instantsfm.db`
  gate is byte-for-byte today's behavior
- hloc: `reconstruction.main` rebuilds its DB every run (`create_empty_db` deletes it); the
  expensive h5 features / matches are reused by hloc's own `overwrite=False` skip
- vocab tree: needed by `retrieval` AND `sequential+retrieval` — i.e. the colmap DEFAULT pulls a
  15.2 MB download on first use
  - `vocab_tree_flickr100K_words32K.bin`, the pre-3.12 format a 3.10 binary reads
  - HEAD-checked 2026-09-26: `github.com/colmap/colmap/releases/download/3.11.1/...` and
    `demuc.de/colmap/...` both 200, both 15,229,678 bytes
  - cached under `~/.cache/collab_splats/`, sha256 pinned in the plan after a load test
    against the 3.10 binary

## Errors

| Failure | Behavior |
|---|---|
| `colmap` binary missing / crashed | `RuntimeError`, partial DB unlinked (existing) |
| hloc not importable | `ImportError` naming `setup/hloc.sh` and the sync |
| mapper returns no model (hloc: `None`) | `RuntimeError` |
| several models | warning, largest kept |
| registered < `min_registered_frac` | `RuntimeError` with N/M |
| vocab-tree download fails | `RuntimeError` with URL and cache path |

## Wording cleanup (messages / docs only, no logic)

- `validate_config` BA/LC refusals: "InstantSfM runs its own global bundle adjustment" -> neutral
- `_run_sfm` docstring; `_SFM_BACKENDS` comment (`reconstructor.py:75`)
- `depth_align.result_from_reconstruction` errors ("InstantSfM registered N/M") and docstring
- `sfm/__init__.py`, `colmap.py`, `hloc.py` module docstrings ("unwired"); `pointcloud/__init__.py`
  and `base.py` docstrings (see Shared creator contract)
- `base.yaml` backend comment; CLAUDE.md architecture line
- `configs/README.md` artifact tree + excludes paragraph (`:533-543`, `:649-650`)

## Dependency

- `pyproject.toml`: optional extra `hloc = ["hloc"]`;
  `[tool.uv.sources] hloc = { path = "third_party/hloc", editable = true }`
  - hloc's own requirements at the pin (`requirements.txt`) are all already satisfied or
    source-pinned here: `pycolmap>=3.13` (4.0.4), `kornia`, `h5py`, `gdown`, `opencv-python`,
    `lightglue` (already a `[tool.uv.sources]` git pin)
- `setup.sh` runs `uv sync --locked --all-extras` (`:89`, `:94`), so the new extra is picked up
  with no flag change, but `--locked` means **`uv.lock` must be regenerated** (`uv lock`)
- a path source must EXIST before any resolve, exactly like collab-data (`setup.sh:70-80`):
  - `setup.sh` calls `setup/hloc.sh` BEFORE the `SETUP_DEPS_ONLY` early exit
  - `Dockerfile:72-73` runs that deps-only pass with only `pyproject.toml uv.lock README.md
    LICENSE setup.sh` copied — it must also get `setup/hloc.sh` (the clone needs network, as
    LoGeR/VDA do)
  - the runtime stage copies only the venv (`Dockerfile:109`); the editable finder points at
    `/workspace/collab-splats/third_party/hloc`, so that tree must be in the runtime image
  - **Dockerfile is touched**; the rebuild is the user's (Mac-only builds)
- `setup/hloc.sh`: path fixed to repo-root `third_party/hloc`; `clone --recursive`, re-pin to
  `c13273b` every run (LoGeR/VDA policy); bare `pip install` removed; weight prefetch added
- **The user runs `uv lock` + the sync.** No `pip install` / `uv lock` / `uv sync` from this work.
  Until then, hloc unit tests mock hloc and the hloc integration run is blocked.

## Testing

Baseline before any edit, in the worktree, with the `collab_splats.__file__` proof line:

```
cd .worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest \
  tests/pointcloud tests/wrapper tests/geometry tests/remote tests/test_docstring_contract.py \
  -q -p no:cacheprovider --continue-on-collection-errors
```

`third_party/{LoGeR,Video-Depth-Anything,hloc}` symlinked in first; SKIP count compared against
the main tree. No new failures; every change in the passed count explained.

Unit (mocks or synthetic):

- `build_sift_database` argv per `pairing`; `exhaustive` argv equal to a snapshot taken before the move
- sequential pair generation; sequential+retrieval union dedups
- `sift_database_valid(params=None)` ignores the sidecar; with params, a mismatch rebuilds
- hloc `reconstruction.main` returning `None` raises; `HLOC_PIN` equals `setup/hloc.sh`'s pin
- `PUSH_EXCLUDES` covers `colmap.db` and `colmap/hloc/`
- `largest_model` picks by registered count, not key
- registered-subset filter: keeps order, floor raises with N/M
- every validation key accepted and rejected; bool rejected where int expected
- ColmapCreator / HlocCreator through mocked pycolmap / hloc: `sparse/0` written, stems
- hloc missing -> `ImportError` text
- `tests/wrapper/test_sfm_config.py` refusal tests flip to acceptance
- docstring contract covers every new / changed public name

instantsfm byte-identical, baseline vs tip:

- InstantSfM is unseeded by default, so the baseline runs **twice** with `random_seed` set
- if the two agree: tip must match on `sparse/0/*.bin` and `pointcloud.zarr` arrays + attrs
- if the baseline cannot reproduce itself (GPU nondeterminism): gate downgrades to identical
  `instantsfm.db` hash + identical argv, and the downgrade is reported, not hidden

Integration, tutorial video, tmux, nothing else running, same frames and budget,
`semantics: {enabled: false}`, per backend (instantsfm / colmap / hloc):

- registered N/M, point count, mean reprojection error, runtime
- downstream splat PSNR vs instantsfm
- also checked: splats / mesh / localize tolerate `images/` holding frames the model omits

## Deliverables

- `colmap`, `hloc` in `_SFM_BACKENDS`; `base.yaml` blocks documented
- `docs/superpowers/decisions/018-sfm-backends.md` (dependency choice, licenses, no GLOMAP)
- `third_party/README.md` hloc row; `configs/README.md` sfm blocks; a short sfm-backends section
  in the docs-site pointcloud page (added if absent); CLAUDE.md in-flight entry

## Scope guard

Files touched beyond the creators, their tests, `setup/hloc.sh`, `base.yaml` sfm block and
`_run_sfm` + validation — allowed because no pointcloud cleanup is running:

- `sfm/instantsfm.py` (move only), `sfm/__init__.py`, `pointcloud/__init__.py`, `pointcloud/base.py`
  (docstrings), `depth_align.py` (messages), `remote/sources.py` (`PUSH_EXCLUDES`),
  `pyproject.toml`, `uv.lock` (user-regenerated), `setup.sh`, `Dockerfile`, docs listed above
- tests: `tests/pointcloud/sfm/*`, `tests/pointcloud/test_registry.py`,
  `tests/wrapper/test_sfm_config.py`, `tests/remote/test_sources.py`

Not touched: notebooks on `clean/final`, `.worktrees/tutorial-rework`.

## Follow-ups (not this spec)

- **Unified COLMAP I/O** — own brainstorm, after `vismatch-fork`'s COLMAP export lands
  - model half: one `write_model(recon, sparse_dir)` owning stem names + original-res K +
    `sparse/0`, replacing feedforward `build_pycolmap_reconstruction` /
    `_rescale_reconstruction_to_original_dimensions` (`feedforward/base.py:798,879`),
    `refine_poses` (`reconstructor.py:1200`) and the sfm rename; `PointcloudResult.from_colmap`
    is its reader
  - DB half: features/matches over `pycolmap.Database`, absorbing `sfm/sift_db.py`,
    `geometry/verification._write_frames` and hloc's writer
  - gates: feedforward + instantsfm byte-identical; an intrinsics-resolution assert
    (`project_mesh_intrinsics_regression`: nothing asserts it today)
- **COLMAP 4.x Docker image** — CLI `global_mapper`, one colmap version for CLI + pycolmap;
  deliberately re-baselines instantsfm (new SIFT); adds `mapper: global` then
- **instantsfm keeps cluster 0, not the largest** — possibly wrong; changing it breaks this
  spec's byte-identical gate, so it is recorded, not fixed
- **megaloc retrieval** A/B against netvlad (torch.hub at runtime; unverified at the pin)
