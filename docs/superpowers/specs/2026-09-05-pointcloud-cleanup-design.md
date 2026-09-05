# Pointcloud module cleanup — design

Date: 2026-09-05
Status: draft, awaiting user review (brainstormed autonomously — see "Confirm before planning")

Scope: `collab_splats/pointcloud/` excluding `feedforward/` (separate pass). Consumers are
touched only where the module's API changes: `wrapper/reconstructor.py`, two lines in
`feedforward/base.py`, `evals/scripts/eval.py`, configs, tests, docs.

## Problem

The module carries structure that nothing uses and rationale that belongs in git history:

1. **Dead coordinate-frame machinery.** `CoordinateFrame.NERFSTUDIO` and
   `PointcloudResult.world_transform` are constructed nowhere in `collab_splats/`; every
   creator passes `frame=CoordinateFrame.COLMAP`. `PointcloudResult.confidence` is never set.
   Only `tests/pointcloud/test_base.py` exercises them. `utils.py`'s header still says poses
   come out in `CoordinateFrame.NERFSTUDIO`.
2. **Function-local imports** in `base.py` (`_write_ply`, to dodge the
   `export -> utils -> base` cycle), `sfm.py` (hloc, VDA, instantsfm, pypose, bae, scipy),
   `utils.py` (open3d in six functions, `rotation_align_vectors`), `__init__.py`
   (`LoopClosure`), and the Reconstructor's imports of pointcloud symbols.
3. **Two PLY writers for one file.** `export.write_pointcloud_ply` (114 lines: dtype,
   header string, shape validation, thinning) is wrapped by
   `BasePointcloudCreator._write_ply`, and `sparse_pc.ply` is written twice per run (creator,
   then Reconstructor after clean). `pycolmap.Reconstruction.export_PLY` (pycolmap 4.0.4,
   probed) writes the identical binary little-endian layout: float x/y/z + uchar r/g/b.
4. **`sfm.py` is 1289 lines** holding three backends plus VDA depth generation plus
   scale/affine depth alignment plus four upstream monkeypatches.
5. **Module-level tuning constants** stand in for keyword defaults: `MIN_ALIGN_OBS`,
   `MIN_AFFINE_OBS`, `AFFINE_REJECT_ROUNDS`, `AFFINE_MIN_FAR_DISPARITY_FRAC`,
   `_SIFT_NUM_THREADS`, `_VDA_MODEL_CONFIGS`, `_DEFAULT_{DOWNSAMPLE,OUTLIER,DISTANCE}_KWARGS`
   + the `_UNSET` sentinel, `_DEFAULT_GREY`.
6. **Overengineering / dead code**: parameters no caller passes (`InstantSfMCreator.features`
   and `.single_camera`, `generate_vda_depth(encoder=, input_size=)`,
   `make_creator(use_lc=, lc_config=)`), a "Legacy cleaning utilities" section used by one
   test, a second (numpy) voxel downsampler used by one notebook, `compute_obb_from_points`
   / `get_points_in_mask` exported to no consumer, a bbox mode + `n_points` + `reference`
   on `filter_distance` used only by tests, a `BaseFeatureExtractor` try-import used by
   nothing, and paragraph docstrings that enumerate every hloc config key upstream offers.

## Approaches considered

- **A. Package split + pycolmap export + trim (chosen).** `sfm.py` becomes a package with
  one file per backend and two support files; `export.py` and `CoordinateFrame` go; every
  constant above becomes a keyword default; docstrings follow one contract. Satisfies all
  six items; the most edits, but each is mechanical and test-covered.
- **B. Same, but delete `ColmapCreator`/`HlocCreator`.** The Reconstructor raises
  `NotImplementedError` for both, `configs/base.yaml` calls them "not implemented", and
  `hloc` is not importable in the venv. Smaller package, but contradicts the brief's
  "separate colmap, hloc and instantsfm as backends". Kept as an option to confirm.
- **C. Minimal: items 1, 2, 4, 6 only, single `sfm.py`.** Leaves a 1100-line file and no
  backend separation. Rejected.

## Decisions

### 1. `base.py` — `PointcloudResult` is a reconstruction plus its image order

`CoordinateFrame`, `PointcloudResult.frame`, `.world_transform`, `.confidence`, and
`BasePointcloudCreator._write_ply` are deleted. Everything the pipeline emits is a
`pycolmap.Reconstruction` in COLMAP's frame; the OpenGL conversion lives in
`geometry/transforms.py` and takes plain arrays.

```python
@dataclass
class PointcloudResult:
    """
    A COLMAP reconstruction plus the image order downstream stages index by.

    - reconstruction: pycolmap.Reconstruction (cameras, images, points3D), COLMAP frame.
    - image_paths: registered images in store order; names key `reconstruction.images`.
    """

    reconstruction: pycolmap.Reconstruction
    image_paths: list[Path]

    # properties unchanged: points (P,3) f32, colors (P,3) u8, extrinsics (N,4,4) w2c,
    # intrinsics (N,3,3) — all in image_paths order


class BasePointcloudCreator(ABC):
    @abstractmethod
    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult: ...
```

### 2. Imports at the top, one sanctioned exception

All imports move to module top. The exception (CLAUDE.md: optional heavy deps that would
break the module on a missing install) is applied to exactly these, each import guarded by
a clear `ImportError` message at the call site:

| import | where | why it stays local |
|---|---|---|
| `hloc.*` | `sfm/hloc.py::HlocCreator.reconstruct` | third_party clone, not installed here |
| `video_depth_anything.video_depth` | `sfm/vda.py::_load_vda_model` | third_party clone on `sys.path`, not site-packages |
| `instantsfm.*`, `pypose.optim.optimizer`, `bae.utils.pysolvers` | `sfm/instantsfm.py` (`_patch_*`, `_build_config`, `reconstruct`) | instantsfm + CUDA extensions, not installed here; `test_import_all_modules` must keep passing |

Everything else is hoisted: `scipy.spatial.transform.Rotation` (hard dep), `open3d` in
`utils.py` (hard dep; `mesh/` and tests import it at top), `rotation_align_vectors`
(`geometry/transforms.py` imports only numpy — no cycle), `PointcloudResult`,
`confidence_mask`, `lift_features` in the Reconstructor.

The `__init__.py -> geometry.loop_closure -> pointcloud.base` cycle is removed rather than
worked around: `make_creator` loses `use_lc`/`lc_config` (only
`tests/geometry/loop_closure/test_wrapper.py:159` used them; the Reconstructor and the
tutorials wrap `LoopClosure(base, config=...)` themselves). `export.py` is deleted, which
removes the `base -> export -> utils -> base` cycle.

### 3. PLY export is `pycolmap.Reconstruction.export_PLY`

`export.py` is deleted. Both writers become one call:

- `feedforward/base.py:1163`: `recon.export_PLY(Path(output_dir) / "sparse_pc.ply")`
  (the only edit to `feedforward/` besides dropping `frame=`).
- `Reconstructor._export_pointcloud_ply`: `result.reconstruction.export_PLY(path)` after
  `backend_dir.mkdir(...)` (pycolmap does not create parent dirs — probed).

`pointcloud.export_max_points` is dropped from `configs/base.yaml`, the Reconstructor, and
its test. It defaults to `null` (every point), no config sets it, and the only reason for a
hand-rolled writer was that thinning hook. Behaviour deltas: colourless points come out
black (pycolmap default) instead of grey 128 — every shipped creator sets colours — and
vertex order follows pycolmap's `points3D` iteration; no consumer depends on order (the
splat trainer reads the file as a set).

### 4. `sfm/` package

```
collab_splats/pointcloud/sfm/
  __init__.py       re-exports: ColmapCreator, HlocCreator, InstantSfMCreator,
                    generate_vda_depth, vda_depth_complete, apply_depth_alignment,
                    DEPTH_ALIGN_MODELS, tracked_point3d_ids, pixel_indices_from_reconstruction
  colmap.py         ColmapCreator — pycolmap SIFT + exhaustive + incremental mapping
  hloc.py           HlocCreator — hloc retrieval/features/matching -> COLMAP mapper
  instantsfm.py     InstantSfMCreator + SIFT DB build + upstream compat patches
  vda.py            Video-Depth-Anything metric depth -> depth_vda/images/npy layout
  depth_align.py    VDA depth -> COLMAP world (scale | affine) + track->pixel helpers
```

Existing import lines (`from collab_splats.pointcloud.sfm import X`) keep working through
the re-exports, so the Reconstructor, `eval.py`, and `pointcloud/__init__.py` change
imports only where a symbol was renamed. VDA and depth alignment are not backends; they
are InstantSfM-only support and get their own files because folding them into
`instantsfm.py` would recreate a 900-line module. `sfm/__init__.py` imports all five
eagerly — safe, because every optional dep stays function-local.

**`colmap.py`** — `ColmapCreator(camera_model="SIMPLE_RADIAL", single_camera=False)`; body
unchanged except `frame=` removed; docstring cut to the contract.

**`hloc.py`** — `HlocCreator(retrieval_conf="netvlad", feature_conf="superpoint_aachen",
matcher_conf="superglue")`; body unchanged except `frame=` removed and the hloc import
wrapped with an `ImportError` pointing at `setup/hloc.sh`; the 60-line option catalogue
is replaced by one line per field ("hloc config key; see `hloc.extract_features.confs`").

**`instantsfm.py`** — sections in this order: SIFT database (`_nudge_edge_keypoints`,
`_sift_database_valid`, `_generate_sift_database(image_path, database_path, *,
num_threads: int = 8)`), upstream compat patches (four `_patch_*`, unchanged logic, one
short attribution comment each), `InstantSfMCreator`.

```python
@dataclass
class InstantSfMCreator:
    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None

    def reconstruct(self, data_dir: Path) -> pycolmap.Reconstruction: ...
```

`features` is dropped (upstream 0.3.0 supports only `"colmap"`; the Reconstructor
validated the value and never dispatched on it) and `single_camera` is dropped (never
passed; the `depth_vda/images/npy` layout the creator writes is upstream's single-camera
branch, so `False` was never a working path). `_build_config` calls `Config("colmap")`;
`_generate_sift_database` passes `--ImageReader.single_camera 1`. The creator stays
outside `BasePointcloudCreator` on purpose: its contract is a staged scene dir in, a
`pycolmap.Reconstruction` out, and the Reconstructor wraps it — one docstring bullet says so.

**`vda.py`** — `VDA_ROOT` and `VDA_CHECKPOINT` stay as module path constants (install
locations, not tuning defaults; tests monkeypatch `VDA_ROOT`). `_VDA_MODEL_CONFIGS` and the
`encoder` parameter go (only `vitl` exists; the config dict is inlined in
`_load_vda_model(device)`). `input_size` goes (never passed; 518 inlined).

```python
def vda_depth_complete(out_dir: Path, names: list[str]) -> bool: ...
def generate_vda_depth(
    frames: np.ndarray, fps: float, out_dir: Path, names: list[str], *,
    depth_width: int = 518, device: str = "cuda", keep_rows: Sequence[int] | None = None,
) -> Path: ...
```

**`depth_align.py`** — public: `DEPTH_ALIGN_MODELS`, `align_depth_to_reconstruction`,
`align_depth_affine`, `apply_depth_alignment(result, reconstruction, model="scale") -> dict`,
plus `tracked_point3d_ids(recon)` and `pixel_indices_from_reconstruction(...)` (renamed
public: the Reconstructor imports them). Constants become keyword-only defaults on the
functions that read them; positional signatures are unchanged:

| constant | becomes |
|---|---|
| `MIN_ALIGN_OBS = 20` | `min_obs: int = 20` on `align_depth_to_reconstruction`, `align_depth_affine` |
| `MIN_AFFINE_OBS = 50` | `min_affine_obs: int = 50` on `align_depth_affine`; passed through to `_fit_affine_disparity` and `_solve_disparity` |
| `AFFINE_REJECT_ROUNDS = 2` | `reject_rounds: int = 2` on `align_depth_affine`; passed through to `_fit_affine_disparity` |
| `AFFINE_MIN_FAR_DISPARITY_FRAC = 0.5` | `min_far_disparity_frac: float = 0.5` on `align_depth_affine` |

`apply_depth_alignment` keeps only `model`; the Reconstructor passes the config value and
nothing else. `sfm/depth_align.py` still type-hints `FeedforwardResult` under
`TYPE_CHECKING` only.

### 5. `utils.py`

| symbol | action |
|---|---|
| header docstring | rewritten: what the module holds; no frame claims |
| `BaseFeatureExtractor` try-import, `_UNSET`, `_DEFAULT_*_KWARGS` | deleted |
| `clean_pcd`, `remove_far_points`, `density_filter` ("Legacy cleaning utilities") | deleted; only `test_base.py` used them |
| `voxel_downsample_point_cloud` | deleted; second adaptive voxel downsampler. `slam_loop_closure.ipynb` (two cells) switches to `subsample_points(pts, colors, max_points=...)` |
| `compute_obb_from_points`, `get_points_in_mask` | deleted; exported to no consumer |
| `_bbox_mask`, `_radial_mask` | deleted; radial logic inlined |
| `filter_distance` | `filter_distance(pcd, max_distance: float, *, return_mask=False)` — radial from centroid. `method="bbox"`, `n_points`, `reference`, `percentile_range`, `max_extent` were test-only |
| `clean_pointcloud` | `clean_pointcloud(pcd, *, voxel_size: float \| None = 0.015, adaptive: bool = True, nb_neighbors: int \| None = 20, std_ratio: float = 2.0, max_distance: float \| None = 50.0) -> tuple[PointCloud, np.ndarray]`; `None` on `voxel_size` / `nb_neighbors` / `max_distance` skips that step. No dict merging |
| `voxel_downsample`, `confidence_mask`, `subsample_points`, `fit_dominant_plane`, `_grid_sample_at_pixels`, `_sample_at_source_pixels`, `lift_features`, `reproject_pixels`, `cross_frame_attention_ratio` | kept; imports hoisted; docstrings to contract |

### 6. `__init__.py`

Drops `CoordinateFrame`, `compute_obb_from_points`, `get_points_in_mask` from imports and
`__all__`; `make_creator(name, **kwargs)` only. The `_REGISTRY` (colmap, hloc, mapanything,
vggtx + optional feedforward entries) is unchanged.

### 7. Docstring contract

Every public function and class in the module (and every private one that keeps a
docstring) follows one shape — `"""` on their own lines, one-line summary, blank line,
one bullet per input, one `returns` bullet, nothing else:

```python
def generate_vda_depth(frames, fps, out_dir, names, *, depth_width=518, device="cuda", keep_rows=None):
    """
    Run Video-Depth-Anything metric depth over a frame stream and write InstantSfM's depth layout.

    - frames: (N, H, W, 3) uint8 RGB stream.
    - fps: stream frame rate; informational (upstream resamples nothing).
    - out_dir: scene dir; maps land at out_dir/depth_vda/images/npy/<stem>.npy.
    - names: staged image filenames, one per written map, in order.
    - depth_width: written map width; height keeps aspect (nearest resize).
    - device: torch device for inference.
    - keep_rows: rows of `frames` to write when it is a context stream; None = all rows.
    - returns: out_dir/depth_vda.
    """
```

Rationale that a maintainer still needs (upstream bug being patched, the OOM measurement
behind `num_threads=8`, layout contracts with instantsfm's reader) moves to a block comment
of at most four lines at the site, keeping repo + commit + file:line attributions
(vendored-code rule). Rationale that restates git history or a memory entry is dropped.

## Consumer edits

**`wrapper/reconstructor.py`**
- Top imports: `PointcloudResult`, `confidence_mask`, `lift_features`,
  `tracked_point3d_ids`, `pixel_indices_from_reconstruction`; drop `write_pointcloud_ply`,
  `CoordinateFrame`, the underscore names, and the four function-local pointcloud imports
  (`_lift_and_save` -> `lift_features`; `_load_pointcloud_from_disk` and `_run_sfm` ->
  `CoordinateFrame, PointcloudResult`; `splats` -> `confidence_mask`). The lazy
  `FeedforwardResult`, `pycolmap`, `open3d`, and gsplat imports there are the wrapper pass's.
- `PointcloudResult(reconstruction=..., image_paths=...)` at both construction sites.
- `_export_pointcloud_ply`: `export_PLY`; delete the `export_max_points` read and the
  "Density is opt-in via pointcloud.export_max_points" comment above `_clean_pointcloud`.
- `_run_sfm`: `InstantSfMCreator(retriangulation=..., random_seed=...)`.
- Validation: delete `_INSTANTSFM_FEATURES` and its check; keep `depth_align` and
  `random_seed` checks, `_SFM_BACKENDS`, and the `NotImplementedError` for
  colmap/hloc (wiring them is a feature, not cleanup).

**`feedforward/base.py`** — two lines: drop `frame=`, replace `self._write_ply(...)` with
`recon.export_PLY(...)`.

**`evals/scripts/eval.py`** — unchanged (`InstantSfMCreator(use_depths=...)`,
`generate_vda_depth(frames, fps=, out_dir=, names=)` are still valid).

**`configs/base.yaml`** — delete `pointcloud.export_max_points` and
`pointcloud.instantsfm.features`; `backend` comment reads
"sfm: instantsfm (colmap | hloc exist as library creators; not wired here)".
**`configs/README.md`** — drop the `pointcloud.instantsfm.features` row and the
"`instantsfm.features` other than `colmap`" sentence in the backend section; the
`pointcloud.backend` row points at `pointcloud/sfm/`.

**Comments/docstrings pointing at old paths** — `geometry/transforms.py:5`
("`CoordinateFrame.NERFSTUDIO`" -> "OpenGL"), `preproc/undistort.py:83`
(`pointcloud/sfm.py::_SIFT_NUM_THREADS` -> `pointcloud/sfm/instantsfm.py::_generate_sift_database`),
`remote/sources.py:71`, `setup.sh:74`.

**Docs** — `docs/source/api/pointcloud.rst`: replace the `collab_splats.pointcloud.sfm`
automodule with the five submodules. `CLAUDE.md` architecture tree: `sfm.py` -> `sfm/`
lines, `export.py` line removed. `docs/superpowers/CHANGELOG.md` entry on completion.
`slam_loop_closure.ipynb` two cells as above.

## Behaviour changes (complete list)

- `PointcloudResult(frame=, world_transform=, confidence=)` and `CoordinateFrame` no longer exist.
- `sparse_pc.ply` is written by pycolmap: colourless points black, vertex order per pycolmap.
- `pointcloud.export_max_points` and `pointcloud.instantsfm.features` config keys removed.
- `InstantSfMCreator(features=, single_camera=)`, `generate_vda_depth(encoder=, input_size=)`,
  `make_creator(use_lc=, lc_config=)` removed.
- `filter_distance` and `clean_pointcloud` signatures as in section 5; six utils functions deleted.
- No numeric behaviour changes: every default keeps its current value.

## Testing

`tests/` mirrors the package: `tests/pointcloud/sfm/{test_colmap,test_hloc,test_instantsfm,
test_vda,test_depth_align}.py`, split from today's `test_sfm_creator.py`,
`test_instantsfm.py`, `test_depth_align.py` with patch targets moved to the module that now
owns the symbol (`...sfm.colmap.pycolmap.extract_features`, `...sfm.vda.VDA_ROOT`,
`...sfm.vda._load_vda_model`).

| test file | change |
|---|---|
| `tests/pointcloud/test_base.py` | drop NERFSTUDIO / `world_transform` / enum / `clean_pcd` tests; construct without `frame=` |
| `tests/pointcloud/test_export.py`, `test_export_wiring.py` | deleted (pycolmap's writer is not ours to test) |
| `tests/pointcloud/test_pointcloud_utils.py` | rewrite `clean_pointcloud` / `filter_distance` cases to the new signatures; delete bbox, `n_points`, `reference`, OBB, mask tests |
| `tests/pointcloud/test_registry.py` | unchanged |
| `tests/test_cu121_migration.py` | add the five `collab_splats.pointcloud.sfm.*` modules |
| `tests/wrapper/test_reconstructor_export.py` | drop `export_max_points` cases; keep "writes `sparse_pc.ply` readable by open3d" |
| `tests/wrapper/test_vda_context.py`, `tests/evals/test_eval_instantsfm.py` | patch the Reconstructor / eval namespaces; no `frame=` / `features=` usage — unchanged |
| `tests/geometry/loop_closure/test_wrapper.py:159` | `LoopClosure(make_creator("vggtx"))` |
| `tests/integration/test_pipeline_cu121.py`, `tests/pointcloud/test_vggtx_creator.py`, `tests/pointcloud/feedforward/test_mapanything_creator.py` | drop `frame=` / `CoordinateFrame` imports and `result.frame` asserts |
| `tests/pointcloud/sfm/test_instantsfm.py` | `InstantSfMCreator()` instead of `InstantSfMCreator(features="colmap")`; `generate_vda_depth` calls unchanged |
| `tests/wrapper/test_reconstructor.py:402` | comment no longer names `write_pointcloud_ply` |

Verification, in order:

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.loop_closure; import collab_splats.pointcloud"
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud; import collab_splats.geometry.loop_closure"
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/geometry/loop_closure tests/evals tests/preproc/test_undistort.py tests/test_cu121_migration.py -x -q
black <touched files> && isort <touched files>      # never repo-wide
graphify update .
```

`instantsfm`, `hloc`, and the VDA clone are not importable in this venv; their tests mock
the modules today and keep doing so. A live `method: sfm` run is not part of this change.

## Out of scope (follow-ups for the named passes)

- `feedforward/`: its own inline imports and docstrings; whether the creator should write
  `sparse_pc.ply` at all when the Reconstructor rewrites it after clean.
- `wrapper/`: `open3d` / `pycolmap` function-local imports in the Reconstructor; the stale
  `clean.voxel_size: null # adaptive` comment (the Reconstructor skips voxelisation on null).
- `preproc/`: `undistort.py`'s duplicate `_SIFT_NUM_THREADS`.
- Wiring `colmap` / `hloc` into `Reconstructor._run_sfm`.

## Confirm before planning

Defaults applied in this spec; flip any and the spec is updated before the plan is written.

1. **Keep `ColmapCreator` / `HlocCreator`** as `sfm/colmap.py`, `sfm/hloc.py` (per the brief).
   Alternative B deletes both: neither is reachable from the Reconstructor and hloc is not
   installed.
2. **Delete `compute_obb_from_points`, `get_points_in_mask`** — exported, never consumed.
3. **Delete `voxel_downsample_point_cloud`**; the loop-closure tutorial uses `subsample_points`.
4. **`filter_distance` is radial-from-centroid only.**
5. **Drop `export_max_points`** rather than reimplement thinning on top of `export_PLY`.
6. **Drop `InstantSfMCreator.features` / `.single_camera`** and the `instantsfm.features` config key.
7. **`VDA_ROOT` / `VDA_CHECKPOINT` stay module constants** (paths, not tuning defaults).
8. **`InstantSfMCreator` stays outside `BasePointcloudCreator`.**
