# SfM backends — wire ColmapCreator + HlocCreator into `pointcloud.method: sfm`

**Date:** 2026-09-26
**Status:** approved design, pre-plan
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

```yaml
pointcloud:
  method: sfm
  backend: instantsfm        # sfm: instantsfm | colmap | hloc
  instantsfm: {...}          # unchanged
  colmap:
    pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
    overlap: 10              # sequential: pair each frame with the next N
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
| `sequential` | `sequential_matcher --SequentialMatching.overlap N` | in-repo pairs: frame i with i+1..i+N |
| `retrieval` | `vocab_tree_matcher` | `pairs_from_retrieval`, top `num_retrieved` |
| `sequential+retrieval` | `sequential_matcher` + `loop_detection` (vocab tree) | union of both pair sets, deduplicated |
| `exhaustive` | `exhaustive_matcher` | `pairs_from_exhaustive` |

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
  sift_db.py     NEW, private to sfm/ — moved out of instantsfm.py, not InstantSfM-specific
                   build_sift_database(image_dir, db_path, *, pairing, overlap, num_threads)
                   sift_database_valid(db_path, names, *, pairing, overlap)
                   rename_images_to_stems(recon, sparse_dir)
                   sfm_image_dir(images_dir)
                   largest_model(recons)                   by num_reg_images
  instantsfm.py  imports the above; build_sift_database(pairing="exhaustive") -> same argv
  colmap.py      ColmapCreator(pairing, overlap, num_threads)
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
  its only callers (`geometry/loop_closure/wrapper.py`, `evals/scripts/eval.py`) are feedforward

### ColmapCreator

1. `build_sift_database` into `colmap/colmap.db` (reused when valid, see Caching)
2. `pycolmap.incremental_mapping(..., options={"num_threads": num_threads})`
3. `largest_model` — incremental mapping gives no size ordering
4. write `sparse/0`, rename to stems

### HlocCreator

- `hloc` imported inside `reconstruct`; `ImportError` says `uv sync --extra hloc`
- intermediates under `colmap/hloc/`
- pairs file per `pairing` (table above); sequential generator is in-repo (~10 lines, hloc has none)
- `extract_features.main` / `match_features.main` with the configured confs
- `hloc.reconstruction.main(camera_mode=SINGLE, image_options={"camera_model": "SIMPLE_RADIAL"},
  mapper_options={"num_threads": num_threads})` — at the pin it already keeps the largest
  model (`hloc/reconstruction.py:117-139` @ `c13273b`)
- copy that model to `sparse/0`, rename to stems

### `_run_sfm`

1. VDA depth — unchanged
2. `SFM_CREATORS[backend](**pc_cfg[backend]).reconstruct(backend_dir, images_dir)`
   (instantsfm keeps its explicit kwargs; `min_registered_frac` is not a creator field)
3. colmap / hloc only: filter `names`, `depths`, `keyframes` to registered stems; raise
   `RuntimeError` with N/M below `min_registered_frac`, else warn
4. `result_from_reconstruction` — unchanged logic
5. zarr attrs: `method`, `backend`, `<backend>_version` (`instantsfm`, `pycolmap`, or the hloc
   pin); colmap / hloc also stamp `registered_frames` / `total_frames`. instantsfm's attr set
   is unchanged.

The `NotImplementedError` guard in `_run_sfm` goes; `_SFM_BACKENDS = set(SFM_CREATORS)`.

## Caching

- DB valid only when its image set, `pairing` and `overlap` all match — the latter two via a
  `colmap.db.json` sidecar. Without it a `pairing` change silently reuses stale matches.
- instantsfm keeps `instantsfm.db` and its current gate (exhaustive only, no sidecar needed);
  gate logic change there is limited to the moved function taking the new keyword args
- vocab tree for colmap `retrieval` pairing: downloaded once to a cache dir. The exact file for
  colmap 3.10's (pre-3.12, FLANN) format is pinned in the plan after a check against that
  binary.

## Errors

| Failure | Behavior |
|---|---|
| `colmap` binary missing / crashed | `RuntimeError`, partial DB unlinked (existing) |
| hloc not importable | `ImportError` naming `uv sync --extra hloc` |
| mapper returns no model | `RuntimeError` |
| several models | warning, largest kept |
| registered < `min_registered_frac` | `RuntimeError` with N/M |
| vocab-tree download fails | `RuntimeError` with URL and cache path |

## Wording cleanup (messages / docs only, no logic)

- `validate_config` BA/LC refusals: "InstantSfM runs its own global bundle adjustment" -> neutral
- `_run_sfm` docstring; `_SFM_BACKENDS` comment (`reconstructor.py:75`)
- `depth_align.result_from_reconstruction` errors ("InstantSfM registered N/M") and docstring
- `sfm/__init__.py` and `pointcloud/__init__.py` docstrings ("unwired")
- `base.yaml` backend comment; CLAUDE.md architecture line

## Dependency

- `pyproject.toml`: optional extra `hloc = ["hloc"]`;
  `[tool.uv.sources] hloc = { path = "third_party/hloc", editable = true }`
- `setup/hloc.sh`: path fixed to repo-root `third_party/hloc`; `clone --recursive`, re-pin to
  `c13273b` every run (LoGeR/VDA policy); bare `pip install` removed
- `setup.sh`: calls `setup/hloc.sh`; its `uv sync` gains `--extra hloc` (plain sync prunes extras)
- **The user runs the sync.** No `pip install` / `uv sync` from this work. Until the sync, hloc
  unit tests mock hloc and the hloc integration run is blocked.

## Testing

Baseline before any edit, in the worktree, with the `collab_splats.__file__` proof line:

```
cd .worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest \
  tests/pointcloud tests/wrapper tests/geometry tests/test_docstring_contract.py \
  -q -p no:cacheprovider --continue-on-collection-errors
```

`third_party/{LoGeR,Video-Depth-Anything,hloc}` symlinked in first; SKIP count compared against
the main tree. No new failures; every change in the passed count explained.

Unit (mocks or synthetic):

- `build_sift_database` argv per `pairing`; `exhaustive` argv equal to a snapshot taken before the move
- sequential pair generation; sequential+retrieval union dedups
- `largest_model` picks by registered count, not key
- sidecar invalidation on `pairing` / `overlap` change
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

- `sfm/instantsfm.py` (move only), `sfm/__init__.py`, `pointcloud/__init__.py`,
  `depth_align.py` (messages), `pyproject.toml`, `setup.sh`, docs listed above

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
