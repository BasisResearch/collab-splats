# Reconstructor release cleanup — design

Date: 2026-09-29 · Branch: `clean/reconstructor-release` off `clean/final` · Status: approved in brainstorm

Rules: [017 — release cleanup rules](../decisions/017-release-cleanup-rules.md). This spec
applies them and does not restate them. Reference run:
[preproc release cleanup](2026-09-24-preproc-release-cleanup-design.md).

Spec 1 of 2. Spec 2 (`dashboard-release`) follows and owns `collab_splats/dashboard/`.

## Goal

Replace `collab_splats/wrapper/` and `collab_splats/remote/` with three files and no
repeated logic.

- `collab_splats/reconstructor.py` — the stage table and `Reconstructor`. Every stage
  composes existing package functions directly.
- `collab_splats/__main__.py` — the one CLI, for local and remote runs.
- `collab_splats/remote.py` — scene knowledge: buckets, scene ids, push excludes. The
  generic rclone transport moves to collab-data's `RcloneClient`.

2562 lines (wrapper + remote) plus 514 lines (`docs/examples` drivers) become about
500 (reconstructor) + 250 (CLI) + 200 (remote).

## Decisions (from brainstorm)

- **One module, not a package.** `wrapper/` held one real file; `reconstructor.py` sits
  one level up, next to the packages it composes. `wrapper/` is deleted, with no shim.
- **Stages call the package directly.** No private helper wraps a package function. A
  capability that appears in more than one place goes into the package that owns it
  (two added: `frames.frame_paths(dir, idxs)` and `pointcloud.utils.frame_depths`).
- **Stage table replaces four structures.** `_STAGE_ORDER`, `_STAGE_DEPS`, the
  `_stage_output_exists` if-chain and the `run_pipeline` if-chain become one `STAGES`
  dict, an `outputs` property and `getattr` dispatch. An unknown stage raises.
- **Stage methods are named after their stages and take no arguments.** Each one always
  rebuilds its own output. `run()` alone decides skip, refuse or run.
- **Transport lives in collab-data.** `RcloneClient` on branch `tlb-3d-tools` of
  `/workspace/collab-data` gains the generic calls. collab-splats pins a collab-data
  commit hash in place of `file:///workspace/collab-data`. Pushing collab-data needs the
  user's explicit OK.
- **CLI takes explicit paths.** Local runs take video files or frame directories only, no
  directory globbing for videos. The output dir is `<output-root>/<stem>`, or
  `<output-root>/<name>` with `--name`. The date-parent derivation (`_DATE_RE`) goes.
- **Legacy input checks go.** The `preprocessing` rename check, the
  `geometric_verification` refusal and the `verify` stage refusal are deleted (no legacy
  file checks).
- **Localization is untouched,** except one line: `_build_localization_db` calls
  `frames.frame_paths(images_dir, idxs)` in place of the deleted `_store_rows`.

## Target layout — `collab_splats/reconstructor.py`

```python
STAGES: dict[str, tuple[str, ...]] = {          # stage -> stages it needs; dict order is run order
    "preproc": (),
    "pointcloud": ("preproc",),
    "refine": ("pointcloud",),
    "semantics": ("pointcloud",),
    "splats": ("pointcloud",),
    "mesh": ("pointcloud",),
    "localize": ("pointcloud",),
    "reconstruction_quality_report": ("pointcloud",),
}
LEAF_STAGES = frozenset(s for s in STAGES if not any(s in deps for deps in STAGES.values()))

def _localization_db_exists(...) -> bool: ...    # untouched
def _build_localization_db(...) -> Path: ...     # one line changed

class Reconstructor:
    def __init__(self, config: dict, base_config: Path | None = None) -> None
    def validate_config(cls, config: dict) -> dict          # classmethod
    # paths: backend_dir, images_dir, pointcloud_zarr, colmap_model_dir, semantics_cache_dir
    outputs: dict[str, Path]                                # property: stage -> marker path
    def done(self, stage: str) -> bool
    result: PointcloudResult                                # property: lazy-loads pointcloud.zarr
    def preproc(self) -> None
    def pointcloud(self) -> None
    def refine(self) -> None
    def semantics(self) -> None
    def splats(self) -> None
    def mesh(self) -> None
    def localize(self) -> None
    def reconstruction_quality_report(self) -> None
    def run(self, stages: list[str] | None = None, overwrite: bool = False) -> None
```

`run()`: `stages=None` takes every enabled stage from config. An unknown stage raises. An
unmet dependency (not in this run, not on disk) raises. A named leaf that is done without
`overwrite` raises. A done non-leaf is skipped. Otherwise `getattr(self, stage)()`.

Stage bodies:

| Stage | Composes |
|---|---|
| `preproc` | video: `load_video_quality` → `sample_fps` / `sample_uniform` / `sample_optical_flow` → `write_frames` → optional `calibrate_camera` + `undistort_frames` + `write_frames` → `plot_photometric`, `plot_motion`. Dir: `read_frames(dir)` + records from `frame_idx_from_path`. Provenance `{"input_path", **preproc_cfg}`, plus `camera.todict()` when undistorted |
| `pointcloud` | `get_creator(b)` or `SFM_CREATORS[b]` → optional `LoopClosure` → `save_zarr` → `dataclasses.replace` drops dense fields → `to_colmap` + `write_ply` |
| `refine` | `BundleAdjustment.refine` → `clean_pointcloud` → `save_zarr(extra_attrs=<existing>)` → `to_colmap` |
| `semantics` | `extract_feature_cache(BaseFeatureExtractor.get(name)(), ...)` → `load_feature_maps` → row pick → `lift_features` → optional `FeatureAutoencoder` → `write_point_features` |
| `splats` | `read_frames(images_dir, idxs)` + `frame_depths` → `train` |
| `mesh` | `render_tsdf_inputs`, or `read_frames` + `frame_depths` + `invert_poses` → optional `sky_masks` → `create_tsdf_mesh` → `clean_repair_mesh` → `create_texture_mesh` |
| `localize` | `_build_localization_db` |
| `reconstruction_quality_report` | `read_frames(images_dir, idxs)` → `compute_reconstruction_quality` |

## Verdict table

### `wrapper/reconstructor.py`

| name | verdict | why |
|---|---|---|
| `DEFAULT_CONFIG_DIR` | delete | `Reconstructor(base_config=None)` resolves `configs/base.yaml` once in `__init__` |
| `_FEEDFORWARD_BACKENDS`, `_SFM_BACKENDS`, `_VALID_METHODS` | delete | `validate_config` checks against the registries directly |
| `_SFM_BLOCK_KEYS`, `_SFM_PAIRINGS` import | delete | each sfm creator's `__init__` checks its own args (`sift_db.PAIRINGS` read there) |
| `_DENSE_FIELDS` | delete | inline `dataclasses.replace(result, images=None, ...)` |
| `_VERIFY_REMOVED` | delete | legacy check |
| `_STAGE_ORDER`, `_STAGE_DEPS` | merge | into `STAGES` |
| `LEAF_STAGES` | keep | derived from `STAGES`; CLI re-run routing reads it |
| `_store_rows` | delete | `frames.frame_paths(dir, idxs)` |
| `_camera_provenance` | delete | `to_json_safe` writes an Enum as its name; stamp `camera.todict()` |
| `_apply_undistortion`, `_frames_from_dir`, `_frames_from_video`, `extract_frames` | inline | into `Reconstructor.preproc` |
| `_run_feedforward` | inline | into `Reconstructor.pointcloud`; LC bool/dict normalized once in `validate_config`; LoGeR `max_frames` warning and reserved-kwarg clash check deleted |
| `_get_extractor`, `_extract_2d_features` | inline | `extract_feature_cache` already skips a valid cache |
| `_lift_and_save` | inline | into `Reconstructor.semantics` |
| `_run_tsdf_mesh` | inline | into `Reconstructor.mesh` |
| `_localization_db_exists`, `_build_localization_db` | keep | localization is out of scope |
| `_scene_frames` | delete | `read_frames(images_dir, idxs)` per stage |
| `Reconstructor.__init__` | keep | `config_dir` → `base_config: Path \| None` |
| `validate_config` | keep | required fields, method/backend, BA×LC, sfm×BA, sfm×LC; LC normalized to a dict |
| `_validate_sfm_block` | delete | moves into the sfm creators |
| `sdf_trunc_mult` check | delete | `create_tsdf_mesh` raises when `sdf_trunc < voxel_size` |
| path properties | keep | the one place the layout is spelled |
| `preprocess`, `build_pointcloud`, `refine_poses`, `extract_semantics`, `build_localization_db` | merge | renamed to their stage names; `getattr` dispatch |
| `splats`, `mesh`, `reconstruction_quality_report` | keep | bodies rewritten per the stage table; `result=` args dropped |
| `_load_pointcloud_from_disk`, `_resolve_result` | merge | into the lazy `result` property |
| `_run_sfm` | inline | same path as feedforward in `pointcloud` |
| `_stage_output_exists` | merge | `outputs` + `done` |
| `run_pipeline` | merge | renamed `run` |
| `launch_dashboard` | delete | zero callers |

### `wrapper/config.py`, `wrapper/batch.py`, `remote/rerun.py`

| name | verdict | why |
|---|---|---|
| `ConfigLoader` | delete | `configs/datasets/` no longer exists; only a test calls it |
| `parse_cli_overrides` | make-private | `__main__._parse_overrides`; one caller |
| `VIDEO_EXTS`, `collect_videos` | delete | CLI takes explicit paths |
| `_DATE_RE`, `scene_output_dir` | delete | `<output-root>/<stem>` or `--name` |
| `build_scene_config`, `run_scene`, `run_all` | merge | into `__main__` |
| `_is_rerun`, `discover_scenes`, `prepare_scene` | merge | into `__main__`; `prepare_scene` keeps its raises |

### `remote/sources.py` → `remote.py`

| name | verdict | why |
|---|---|---|
| `CURATED_BUCKET`, `PROCESSED_BUCKET` | make-kwarg | `SceneSource(curated=..., processed=...)` |
| `_VIDEO_EXTS` | make-kwarg | `SceneSource(video_exts=...)` |
| `SCENE_ID_RE` | keep | path-safety fact; one comment line |
| `PULL_EXCLUDES` | delete | the dashboard's default, moved to `dashboard/pipeline.py` verbatim (spec 2 cleans it); `pull_processed(exclude=())` |
| `PUSH_EXCLUDES` | keep | trimmed: drop legacy `/*/colmap/database.db`; one video pattern |
| `_RCLONE_PCT_RE`, `parse_rclone_percent` | delete | collab-data stats parser (JSON log) |
| `_CHECK_MISMATCH_MARKER` | delete | `RcloneClient.check` reads `check --combined -` |
| `SceneSource` | keep | public methods unchanged; bodies call `RcloneClient` |
| `_run_streaming`, `_lsjson` | delete | `RcloneClient.run_streaming` / `run` |

### collab-data `RcloneClient` (branch `tlb-3d-tools`)

| name | verdict | why |
|---|---|---|
| `run(*args) -> str` | add | public; raises `RuntimeError` on non-zero exit |
| `run_streaming(*args, on_line)` | add | line callback; raises on non-zero exit |
| `copy_dir(src, dst, exclude, on_line)` | add | directory copy with excludes |
| `check(src, dst, exclude) -> bool` | add | `check --combined -`; False on a mismatch, raises on transport failure |
| stats parser | add | reads `--use-json-log` stats lines (rclone 1.53.3 has both flags) |
| `copy_file`, `copy_local_to_remote` | keep | collab-data's own dashboard uses their bool return |

## Package changes (outside the two modules)

- `preproc/frames.py`: `frame_paths(dir, idxs=None)` selects by source frame_idx in the
  requested order and raises `KeyError` on a missing one. `read_frames` and
  `semantics/segmentation/sky.py::sky_masks` call it in place of their own `by_idx` maps.
  `write_frames`'s empty-records message names the cause (no frames selected).
- `utils/io.py`: `to_json_safe` writes an `Enum` as its `.name`.
- `pointcloud/utils.py`: `frame_depths(result, rgbs, conf_percentile=None)` — confidence
  mask on the model grid, then `upsample_depths` onto the frame grid. Callers: the mesh
  and splats stages. `geometry/metrics.py` keeps its own path (different guards, no
  confidence mask).
- `mesh`: `create_tsdf_mesh` raises when `sdf_trunc < voxel_size`.
- `pointcloud/sfm/`: each creator's `__init__` validates the args `_validate_sfm_block`
  checked. A typo now raises when the pointcloud stage builds the creator, after preproc.

## Round 1 — move and prose

No behavior change.

- `git mv collab_splats/wrapper/reconstructor.py collab_splats/reconstructor.py` and
  `git mv collab_splats/remote/sources.py collab_splats/remote.py`; import lines updated
  in every caller. Proof: each moved file is byte-identical to its source except import
  lines.
- Prose on the code that survives Round 2 (path properties, `validate_config`,
  `SceneSource`, `SCENE_ID_RE`, `PUSH_EXCLUDES`): docstrings per 017, block comments one
  plain line. Proof: AST equal after deleting every docstring on both sides, plus one
  sanity mutation that fails it.
- Localization helpers are not edited.

## Round 2 — code, one commit per change

1. `frames.frame_paths(dir, idxs)`; `read_frames`, `sky_masks` use it.
2. `to_json_safe` Enum; `write_frames` message.
3. `frame_depths` in `pointcloud/utils.py`.
4. `create_tsdf_mesh` sdf raise; sfm creators validate their args.
5. collab-data: `RcloneClient` additions on `tlb-3d-tools`, with tests there. Push only
   after the user's OK; then bump the collab-splats pin to that commit.
6. `remote.py` on `RcloneClient`; bucket and video-ext kwargs; `PULL_EXCLUDES` moved to
   the dashboard; `parse_rclone_percent` import in `dashboard/operation_log.py` switched
   to collab-data.
7. `STAGES`, `outputs`, `done`, `result`, `run` with `getattr` dispatch.
8. One commit per stage body: `preproc`, `pointcloud`, `refine`, `semantics`, `mesh`,
   `splats`, `reconstruction_quality_report`, `localize` (the one-line `frame_paths` change).
9. `validate_config` trim; legacy checks and `launch_dashboard` deleted.
10. `collab_splats/__main__.py`: `local` and `remote` subcommands sharing `--config`,
    `--base-config`, `--stages`, `--overwrite`, `--set key=value`; `local` adds `--name`
    and `--keep-viewer`; `remote` adds `--all` and `--keep-local`. Exit codes kept.
    Console script `reconstruct` (`reconstruct local ...`, `reconstruct remote ...`),
    same as `python -m collab_splats`. `wrapper/` and `docs/examples/` deleted.
11. Callers: `evals/eval.py` (`collab_splats.reconstructor`, `recon.run()`),
    `configs/README.md` run instructions, `docs/known-test-failures.md` paths.
12. `tests/test_docstring_contract.py`: `MODULES` gains `reconstructor.py`, `remote.py`,
    `__main__.py`.

## Caller sweep

| Caller | Change |
|---|---|
| `evals/eval.py:33,254` | import path; `run_pipeline()` → `run()` |
| `collab_splats/dashboard/{app,pipeline,localize,shell}.py` | `collab_splats.remote` import still valid; `PULL_EXCLUDES` local to `pipeline.py` |
| `collab_splats/dashboard/operation_log.py` | `parse_rclone_percent` from collab-data |
| `docs/examples/{reconstruct,run_pipeline,run_pipeline_remote}.py` | deleted; replaced by `python -m collab_splats` |
| `configs/README.md` | run instructions and `sources.py` path references |
| `docs/source/tutorials/03_splats/train_splats.ipynb`, `06_mesh/splats_mesh.ipynb` | prose names `Reconstructor.splats()` / `.mesh()`, still valid |
| `tests/geometry/test_metrics.py` | `_STAGE_DEPS` / `_STAGE_ORDER` → `STAGES` |
| `tests/preproc/test_undistort.py` | `extract_frames` / `_camera_provenance` → `Reconstructor.preproc` |
| `tests/pointcloud/test_loger_creator.py` | `_FEEDFORWARD_BACKENDS` → the registry |

## Testing

- Gate per commit, in the worktree, printing `collab_splats.__file__`:
  `tests/reconstructor tests/remote tests/preproc tests/pointcloud tests/mesh tests/geometry tests/dashboard tests/test_docstring_contract.py`,
  SKIP count compared to the baseline.
- `tests/wrapper/`, `tests/examples/`, `tests/scripts/test_reconstruct.py` and
  `tests/remote/test_rerun.py` move to `tests/reconstructor/` (stages, `run`, CLI) and
  `tests/remote/` (`SceneSource`). Tests of deleted code are deleted: `ConfigLoader`,
  `verify` stage, legacy config keys, `collect_videos`, `scene_output_dir` date parsing.
- New tests: unknown stage raises; `frame_paths(idxs)` order + `KeyError`; `frame_depths`;
  `to_json_safe` Enum; sfm creator arg validation; `create_tsdf_mesh` sdf raise; CLI
  `local` / `remote` argument handling with a stubbed `Reconstructor`.
- One end-to-end local run on a short tutorial clip with semantics disabled
  (`semantics: {enabled: false}`), compared against a `clean/final` run of the same clip:
  same frame count, same point count, mesh vertex count within the hull's run-to-run noise.
- collab-data: `RcloneClient` tests against a local-filesystem remote.

## Out of scope

- `collab_splats/localization/` and the localization helpers (beyond the one line).
- `collab_splats/dashboard/` (spec 2), including its own `run_pipeline` and `PULL_EXCLUDES`.
- Tutorial notebooks beyond the caller sweep.
- A shared GCS layout layer between collab-data and collab-splats.
