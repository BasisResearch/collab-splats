# Pointcloud module cleanup — design

Date: 2026-09-05
Status: draft, revision 2 (after the overengineering re-audit), awaiting user review

Scope: `collab_splats/pointcloud/` excluding `feedforward/` (separate pass). Consumers are
touched only where the module's API changes or where a deleted feature's plumbing lives:
`wrapper/reconstructor.py`, `preproc/video.py` (one function), two lines in
`feedforward/base.py`, `evals/scripts/eval.py`, `setup.sh`, configs, tests, docs.

## Problem

The module carries structure that nothing uses, features that were measured and lost, and
rationale that belongs in git history:

1. **Dead coordinate-frame machinery.** `CoordinateFrame.NERFSTUDIO` and
   `PointcloudResult.world_transform` are constructed nowhere in `collab_splats/`; every
   creator passes `frame=CoordinateFrame.COLMAP`. `PointcloudResult.confidence` is never set.
   Only `tests/pointcloud/test_base.py` exercises them. `utils.py`'s header still says poses
   come out in `CoordinateFrame.NERFSTUDIO`.
2. **Function-local imports** in `base.py` (`_write_ply`, to dodge the
   `export -> utils -> base` cycle), `sfm.py` (hloc, VDA, instantsfm, pypose, bae, scipy),
   `utils.py` (open3d in six functions, `rotation_align_vectors`), `__init__.py`
   (`LoopClosure`), and the Reconstructor's imports of pointcloud symbols.
3. **Two PLY writers for one file.** `export.write_pointcloud_ply` (114 lines) is wrapped by
   `BasePointcloudCreator._write_ply`, and `sparse_pc.ply` is written twice per run.
   `pycolmap.Reconstruction.export_PLY` (pycolmap 4.0.4, probed) writes the identical binary
   little-endian layout: float x/y/z + uchar r/g/b.
4. **`sfm.py` is 1289 lines** holding three backends, VDA depth generation, scale + affine
   depth alignment, a SIFT database builder with a CPU fallback, and four upstream
   monkeypatches. Two of the three backends (`ColmapCreator`, `HlocCreator`) are reachable
   from no pipeline path: the Reconstructor raises `NotImplementedError` for them, no
   tutorial imports them, and `hloc` is not installed. Each is a thin wrapper over three or
   four calls into pycolmap / hloc.
5. **Measured-and-lost features still shipped as options.**
   - Affine depth alignment (`align_depth_affine`, `_solve_disparity`,
     `_fit_affine_disparity`, `_apply_affine_depth`, three constants, the
     `pointcloud.instantsfm.depth_align` key, ~250 lines + tests). The 2026-08-26 grid
     (`docs/superpowers/specs/2026-08-26-depth-align-grid-results.md`) verdict: "ship
     median normals, drop affine" (−0.076 dB alone, −0.032 dB stacked).
   - The VDA context stream (`preproc.vda_context_fps`, `generate_vda_depth(keep_rows=)`,
     `Reconstructor._ensure_vda_depth` with its `depth_vda/inputs.json` sidecar,
     `_context_keep_rows`, `_video_unchanged`, `preproc.video.decode_context`,
     `tests/wrapper/test_vda_context.py` at 617 lines). Refuted 2026-08-26 for the metric
     path (contiguity −0.7 % CV, residual −0.9 %; `metric=True` disables upstream's
     scale-and-shift stitching, so contiguous input changes nothing).
6. **Module-level tuning constants** stand in for keyword defaults: `MIN_ALIGN_OBS`,
   `MIN_AFFINE_OBS`, `AFFINE_REJECT_ROUNDS`, `AFFINE_MIN_FAR_DISPARITY_FRAC`,
   `_SIFT_NUM_THREADS`, `_VDA_MODEL_CONFIGS`, `_DEFAULT_{DOWNSAMPLE,OUTLIER,DISTANCE}_KWARGS`
   + the `_UNSET` sentinel, `_DEFAULT_GREY`.
7. **Overengineering / dead code**: parameters no caller passes (`InstantSfMCreator.features`
   and `.single_camera`, `generate_vda_depth(fps=, encoder=, input_size=)`,
   `make_creator(use_lc=, lc_config=)`); a CPU SIFT fallback on a pipeline that needs CUDA
   for VDA and InstantSfM anyway; a `VDA_CHECKPOINT` path + `wget` block for a file
   `huggingface_hub` already knows how to fetch and cache; two voxel downsamplers plus a
   third in the Reconstructor whose result is discarded (`_clean_pointcloud`:
   `pcd = pcd.voxel_down_sample(...)` then `pcd` is never read, so
   `pointcloud.clean.voxel_size` is a no-op, and `clean.confidence_threshold` is
   "NOT YET READ"); `utils.clean_pointcloud` / `filter_distance` / `voxel_downsample`
   used by one tutorial cell while the Reconstructor re-implements outlier removal inline;
   `compute_obb_from_points` / `get_points_in_mask` exported to no consumer; a
   `BaseFeatureExtractor` try-import used by nothing; `density_filter` with no return
   statement; paragraph docstrings that enumerate every hloc config key.

Facts checked for this revision (agent + venv probes, 2026-09-05):

- `instantsfm` is **not installed** in `/opt/venv/reconstruction` (`ModuleNotFoundError`),
  nor are `pyceres`, `scikit-sparse`, `easydict` — the `setup.sh` optional block has not
  been run since the last `uv sync` pruned it. The pin
  `cre185/InstantSfM@d3e599e` (2026-06-22) is upstream `master`'s tip and tag `0.3.0`
  (released 2026-08-08); no PyPI package. The five upstream files our patches touch are
  byte-identical at the pin and at `master`: both bugs (packed int64 track ids into
  `np.int32`; `_write_images_binary` / `_write_points3d_binary` index mismatch) are still
  present upstream. Upstream `feature_handler.GenerateDatabase` also shells out to the
  `colmap` CLI but forces `--SiftExtraction.use_gpu 0` and swallows errors, so it is not a
  substitute for ours.
- `pypose` 0.7.5 and `bae` 0.2.4 are installed. pypose ≥ 0.9.0 (2026-04) fixes both the
  `RobustModel.forward(target)` default and the `CG` column-shape squeeze that
  `_patch_pypose_robustmodel_target` / `_patch_bae_pcg_column_shape` work around; bae has
  not changed `pysolvers.py` through its `release` head (2026-09-04). pypose's import-time
  bae guard reads `>=0.2.1,<0.3` on `main` (would accept 0.2.4), contradicting the
  `pypose<0.9` comment in `pyproject.toml` — not verified at the 0.9.x tags.
- Video-Depth-Anything: upstream head is our clone's `4f5ae23` (2025-10-07). No
  `pyproject.toml` / `setup.py`, no PyPI name, no `transformers` model class (4.57.6 ships
  `depth_anything`, `depth_pro`, `prompt_depth_anything`, `zoedepth` only). Its package
  imports a top-level `utils` from the clone root, so the clone root must be first on
  `sys.path` while it loads. The metric checkpoint is public on the Hub
  (`depth-anything/Metric-Video-Depth-Anything-Large`, ungated); `huggingface_hub` 0.36.2
  is installed.
- `colmap` CLI is a CUDA build (3.10-dev); the `pycolmap` 4.0.4 wheel is CPU-only. SIFT on
  100 × 1080p frames: GPU 7 s extract / 55 s match vs CPU 90 s / ~816 s.

## Approaches considered

- **A. Package split, keep all three backends, trim.** `sfm/{colmap,hloc,instantsfm}.py`.
  Honours the brief's wording ("separate colmap, hloc and instantsfm as backends") but
  keeps two wrappers with no pipeline consumer, one of them over an uninstalled library.
- **B. Package split with InstantSfM as the only backend, VDA and depth alignment as
  top-level pointcloud modules, delete the measured-and-lost features (chosen).**
  `ColmapCreator` / `HlocCreator` go (a tutorial that wants incremental COLMAP calls
  pycolmap's three functions directly). Satisfies every item in the brief except the
  literal three-backend layout; `sfm/` stays a package so a backend can be added beside
  `instantsfm.py` later. Largest deletion, every deletion backed by a measurement or a
  "no consumer" grep.
- **C. Minimal: items 1, 2, 3, 6 only, single `sfm.py`.** Leaves a 1000-line file and
  ships options that lost. Rejected.

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
break the module on a missing install) applies to exactly these, each guarded by a clear
`ImportError` message at the call site:

| import | where | why it stays local |
|---|---|---|
| `video_depth_anything.video_depth` | `pointcloud/vda.py::_load_vda_model` | third_party clone that must sit first on `sys.path` while it imports (it imports a top-level `utils`); not site-packages |
| `instantsfm.*`, `pypose.optim.optimizer`, `bae.utils.pysolvers` | `pointcloud/sfm/instantsfm.py` (`_patch_*`, `_build_config`, `reconstruct`) | optional backend installed outside the lock; `test_import_all_modules` must keep passing |

Everything else is hoisted: `scipy.spatial.transform.Rotation`, `huggingface_hub`,
`open3d` in `utils.py` (hard dep; `mesh/` and tests import it at top),
`rotation_align_vectors` (`geometry/transforms.py` imports only numpy — no cycle),
`PointcloudResult`, `confidence_mask`, `lift_features` in the Reconstructor.

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
its test. It defaults to `null`, no config sets it, and the thinning hook was the only
reason for a hand-rolled writer. Behaviour deltas: colourless points come out black
(pycolmap default) instead of grey 128 — every shipped creator sets colours — and vertex
order follows pycolmap's `points3D` iteration; no consumer depends on order.

### 4. Layout — `sfm/` holds backends; VDA and depth alignment are pointcloud modules

```
collab_splats/pointcloud/
  __init__.py       registry: mapanything, vggtx (+ optional feedforward entries)
  base.py           BasePointcloudCreator, PointcloudResult
  utils.py          lift_features, reproject_pixels, clean_pointcloud, subsample_points, ...
  vda.py            generate_vda_depth — Video-Depth-Anything metric depth -> depth_vda/images/npy
  depth_align.py    apply_depth_alignment (per-frame scale) + tracked_point3d_ids,
                    pixel_indices_from_reconstruction
  sfm/
    __init__.py     re-exports InstantSfMCreator
    instantsfm.py   SIFT database, upstream compat patches, InstantSfMCreator
  feedforward/      unchanged (own pass)
```

Why not under `sfm/`: VDA is a generic monocular video depth model — nothing about it is
InstantSfM's except the on-disk layout its reader expects, which is one line in
`generate_vda_depth`. Depth alignment takes any `pycolmap.Reconstruction` plus a depth
stack; it is not tied to a backend. `sfm/` stays a package rather than a flat
`sfm.py` so a second backend slots in as a sibling file without another split.

**`sfm/instantsfm.py`** — sections in this order, `########` dividers between them:

1. *SIFT database.* `_nudge_edge_keypoints(features, width, height)` (upstream
   `sample_depth_at_pixel` raises `IndexError` at `x == width`; still unfixed at `master`),
   `_sift_database_valid(database_path)` rewritten on `pycolmap.Database`
   (`num_keypoints() > 0 and num_verified_image_pairs() > 0`; the `sqlite3` import goes),
   `_generate_sift_database(image_path, database_path)` — `colmap feature_extractor` +
   `exhaustive_matcher` via `subprocess`, `--ImageReader.camera_model SIMPLE_RADIAL
   --ImageReader.single_camera 1`, GPU always. The CPU fallback branch, the
   `torch.cuda.is_available()` switch, `CUDA_VISIBLE_DEVICES=""`, and `_SIFT_NUM_THREADS`
   are deleted: VDA and InstantSfM already require CUDA, so the fallback could not complete
   a run. Partial DB is still unlinked on failure. A four-line block comment records why the
   CLI and not pycolmap (CUDA build vs CPU wheel, 13× / 15×) and why not upstream's
   `GenerateDatabase` (forced CPU, errors swallowed).
2. *Upstream compat patches.* The four `_patch_*` functions, logic unchanged, one
   attribution comment each (repo + commit + file:line, per the vendored-code rule). All
   four are still needed: upstream `master` == our pin for the two instantsfm bugs; the
   pypose/bae pair is fixed only in pypose ≥ 0.9, which `pyproject.toml` pins out
   (`pypose<0.9`) and which `geometry/` BA also depends on — bumping it is a dependency
   pass, not this one (see Out of scope).
3. *Creator.*

```python
@dataclass
class InstantSfMCreator:
    """
    Global SfM via InstantSfM's python API on a staged scene directory.

    - use_depths: feed depth_vda/ maps into the solve as depth priors.
    - retriangulation: GLOMAP-style retriangulate + re-BA after the global solve.
    - random_seed: seeds InstantSfM's RUNTIME_OPTIONS; None = unseeded.
    - reconstruct(data_dir): data_dir/images/ (+ depth_vda/) -> pycolmap.Reconstruction,
      model written to data_dir/colmap/sparse/0, SIFT DB at data_dir/colmap/instantsfm.db.
    """

    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None
```

`features` is dropped (upstream 0.3.0 supports only `"colmap"`; the Reconstructor
validated the value and never dispatched on it) and `single_camera` is dropped (never
passed; the `depth_vda/images/npy` layout is upstream's single-camera branch, so `False`
was never a working path). `_build_config` calls `Config("colmap")`. The creator stays
outside `BasePointcloudCreator` on purpose: its contract is a staged scene dir in, a
`pycolmap.Reconstruction` out, and the Reconstructor wraps it — one docstring bullet says so.
`reconstruct` body is otherwise unchanged (patches applied, stale `colmap/sparse` removed,
SIFT DB reused when valid, `ReadDepthsIntoFeatures` when `use_depths`,
`SolveGlobalMapper`, `WriteGlomapReconstruction`, cluster checks, read back via pycolmap).

**`vda.py`**

```python
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"
VDA_HF_REPO = "depth-anything/Metric-Video-Depth-Anything-Large"
VDA_HF_FILE = "metric_video_depth_anything_vitl.pth"


def _load_vda_model(device: str):
    """
    Build the metric ViT-L Video-Depth-Anything model with Hub-cached weights.

    - device: torch device the model is moved to.
    - returns: eval-mode VideoDepthAnything.
    """
    # Clone root first on sys.path only while importing: the package imports a top-level
    # `utils` from its own root. Same pattern as feedforward/loger.py and vggt_spark_creator.py.
    ...
    checkpoint = hf_hub_download(repo_id=VDA_HF_REPO, filename=VDA_HF_FILE)


def generate_vda_depth(
    frames: np.ndarray, out_dir: Path, names: list[str], *, depth_width: int = 518, device: str = "cuda",
) -> Path:
    """
    Run Video-Depth-Anything metric depth over keyframes and write InstantSfM's depth layout.

    - frames: (N, H, W, 3) uint8 RGB, one per name.
    - out_dir: scene dir; maps land at out_dir/depth_vda/images/npy/<stem>.npy.
    - names: staged image filenames, one per written map, in order.
    - depth_width: written map width; height keeps aspect (nearest resize).
    - device: torch device for inference.
    - returns: out_dir/depth_vda. No-op when every stem already has a map.
    """
```

Deleted: `VDA_CHECKPOINT` and the `setup.sh` `wget` block (L88–95; `hf_hub_download`
caches under `HF_HOME` and is the standard route for this file), `_VDA_MODEL_CONFIGS` and
the `encoder` parameter (only `vitl` exists; the config dict is inlined), `input_size`
(never passed; 518 inlined), `fps` (upstream's `infer_video_depth` returns `target_fps`
unchanged and resamples nothing; a literal is passed), `keep_rows` and the one-to-one
validation (context stream, item 5 above), and `vda_depth_complete` (callers call
`generate_vda_depth`, which already returns early on a complete stem set). `VDA_ROOT`
stays: the model code cannot be a standard install without a fork carrying build
metadata, and the clone root must be on `sys.path` while the package imports.

**`depth_align.py`** — the scale model only:

```python
def tracked_point3d_ids(reconstruction) -> np.ndarray: ...
def pixel_indices_from_reconstruction(reconstruction, point3d_ids, name_to_row, scale_x, scale_y, depth_hw): ...
def _depth_correspondences(reconstruction, image_names, depth) -> list[tuple[np.ndarray, np.ndarray]]: ...
def align_depth_to_reconstruction(reconstruction, image_names, depth, *, min_obs: int = 20) -> tuple[np.ndarray, list[int]]: ...
def apply_depth_alignment(result: "FeedforwardResult", reconstruction, *, min_obs: int = 20) -> dict: ...
```

`align_depth_to_reconstruction` returns `(scales, fallback_frames)` — the per-frame median
ratio with the global-median fallback, exactly today's scale branch. The stats dict shrinks
to what is logged: global scale, ratio p10/p50/p90 before and after, fallback count.
`apply_depth_alignment` rescales `result.depth` in place, recomputes `world_points`, and
returns `{"depth_scale": "colmap", "depth_scales": [...], "depth_scale_fallback_frames":
[...]}` (the `depth_align_model` attr goes with the option; `depth_scale` is the key the
Reconstructor checks when it loads a store). Deleted: `DepthAlignModel`,
`DEPTH_ALIGN_MODELS`, `align_depth_affine`, `_solve_disparity`, `_fit_affine_disparity`,
`_apply_affine_depth`, `MIN_ALIGN_OBS` (→ `min_obs=20`), `MIN_AFFINE_OBS`,
`AFFINE_REJECT_ROUNDS`, `AFFINE_MIN_FAR_DISPARITY_FRAC`. `FeedforwardResult` stays a
`TYPE_CHECKING`-only hint (no `pointcloud.feedforward` import at load).

### 5. `utils.py`

| symbol | action |
|---|---|
| header docstring | rewritten: what the module holds; no frame claims |
| `BaseFeatureExtractor` try-import, `_UNSET`, `_DEFAULT_*_KWARGS` | deleted |
| `clean_pcd`, `remove_far_points`, `density_filter` ("Legacy cleaning utilities") | deleted; only `test_base.py` used them; `density_filter` returns nothing |
| `voxel_downsample`, `filter_distance`, `_radial_mask`, `_bbox_mask` | deleted; `clean_pointcloud` was their only consumer |
| `voxel_downsample_point_cloud` | deleted; `slam_loop_closure.ipynb` (one cell, two calls) switches to `subsample_points(pts, colors, max_points=...)` |
| `compute_obb_from_points`, `get_points_in_mask` | deleted; exported to no consumer |
| `clean_pointcloud` | `clean_pointcloud(points: np.ndarray, *, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray` — open3d statistical outlier removal, returns a bool keep mask over `points`. The one cleaning step both consumers run (Reconstructor `_clean_pointcloud`, `feedforward_methods.ipynb`) |
| `confidence_mask`, `subsample_points`, `fit_dominant_plane`, `_grid_sample_at_pixels`, `_sample_at_source_pixels`, `lift_features`, `reproject_pixels`, `cross_frame_attention_ratio` | kept; imports hoisted; docstrings to contract |

### 6. `__init__.py`

Drops `CoordinateFrame`, `ColmapCreator`, `HlocCreator`, `compute_obb_from_points`,
`get_points_in_mask` from imports and `__all__`; `make_creator(name, **kwargs)` only.
`_REGISTRY` holds `mapanything`, `vggtx` plus the optional feedforward entries.

### 7. Docstring contract

Every public function and class in the module (and every private one that keeps a
docstring) follows one shape — `"""` on their own lines, one-line summary, blank line,
one bullet per input, one `returns` bullet, nothing else (see `generate_vda_depth` and
`InstantSfMCreator` above). Rationale a maintainer still needs (upstream bug being
patched, the GPU/CPU measurement, layout contracts with instantsfm's reader) moves to a
block comment of at most four lines at the site, keeping repo + commit + file:line
attributions. Rationale that restates git history or a memory entry is dropped.

## Consumer edits

**`wrapper/reconstructor.py`**
- Top imports: `PointcloudResult`, `confidence_mask`, `lift_features`, `clean_pointcloud`,
  `generate_vda_depth`, `apply_depth_alignment`, `tracked_point3d_ids`,
  `pixel_indices_from_reconstruction`; drop `write_pointcloud_ply`, `CoordinateFrame`,
  `DEPTH_ALIGN_MODELS`, `vda_depth_complete`, `context_indices`, `decode_context`, the
  underscore names, and the four function-local pointcloud imports (`_lift_and_save` ->
  `lift_features`; `_load_pointcloud_from_disk` and `_run_sfm` -> `CoordinateFrame,
  PointcloudResult`; `splats` -> `confidence_mask`). The lazy `FeedforwardResult`,
  `pycolmap`, and gsplat imports there are the wrapper pass's.
- `PointcloudResult(reconstruction=..., image_paths=...)` at both construction sites.
- `_export_pointcloud_ply`: `export_PLY`; delete the `export_max_points` read and the
  "Density is opt-in via pointcloud.export_max_points" comment above `_clean_pointcloud`.
- `_clean_pointcloud(result)`: `keep = clean_pointcloud(result.points)`; delete the
  point3D ids where `keep` is false; log the count. The open3d import, the colour
  conversion, the `outlier_removal` branch, and the discarded `voxel_down_sample` go.
  Call site: `if pc_cfg["clean"]["enabled"]`.
- `_run_sfm`: `generate_vda_depth(store frames, backend_dir, names)` inline (three lines
  replace `_ensure_vda_depth` and the `colmap/instantsfm.db` restage note stays);
  `InstantSfMCreator(use_depths=True, retriangulation=..., random_seed=...)`;
  `apply_depth_alignment(outputs, recon)`; the `depth_align=%s` clause in the splats log
  line (L1834–1836) goes.
- Delete `_ensure_vda_depth`, `_context_keep_rows`, `_video_unchanged`.
- `preprocess()`: delete the `vda_context_fps` parameter, its docstring bullet, the
  image-dir warning and `optical_flow` rejection, the context-grid candidate branch, and
  the provenance key (L194–322 pieces).
- Validation: delete `_INSTANTSFM_FEATURES` and its check and the `depth_align` check; keep
  `random_seed`, `_SFM_BACKENDS`, and the `NotImplementedError` for `colmap` / `hloc`
  (the config values stay reserved; wiring a backend is a feature).

**`preproc/video.py`** — delete `decode_context` (its only consumer was the context
stream); `context_indices` stays (`preproc/sampling.py` uses it).

**`feedforward/base.py`** — two lines: drop `frame=`, replace `self._write_ply(...)` with
`recon.export_PLY(...)`.

**`evals/scripts/eval.py`** — `generate_vda_depth(frames, output_dir, names)`; drop the
`vda_depth_complete` import and guard (L358–360).

**`setup.sh`** — delete the checkpoint `wget` block (L88–95) and the `VDA_CKPT` variable;
the clone + pin block stays; the comment at L74 names `pointcloud/vda.py`.

**`configs/base.yaml`** — delete `preproc.vda_context_fps`, `pointcloud.export_max_points`,
`pointcloud.instantsfm.features`, `pointcloud.instantsfm.depth_align`,
`pointcloud.clean.outlier_removal`, `.voxel_size`, `.confidence_threshold`
(`clean: {enabled: true}` remains); `backend` comment reads "sfm: instantsfm".
**`configs/README.md`** — drop the `preproc.vda_context_fps`, `pointcloud.instantsfm.features`
and `.depth_align` rows and the matching sentences in the instantsfm backend section
(L458–515); the `pointcloud.backend` row points at `pointcloud/sfm/`; `clean` rows reduce
to `enabled`.

**Comments/docstrings pointing at old paths** — `geometry/transforms.py:5`
("`CoordinateFrame.NERFSTUDIO`" -> "OpenGL"), `preproc/undistort.py:83` (the
"Mirrors pointcloud/sfm.py::_SIFT_NUM_THREADS" comment goes; its own constant is the
preproc pass's), `remote/sources.py:71`, `setup.sh:74`.

**Docs** — `docs/source/api/pointcloud.rst`: replace the `collab_splats.pointcloud.sfm`
automodule with `sfm.instantsfm`, `vda`, `depth_align`. `CLAUDE.md` architecture tree:
`sfm.py` -> `sfm/` + `vda.py` + `depth_align.py`, `export.py` line removed, `utils.py`
line lists what it holds. `docs/superpowers/CHANGELOG.md` entry on completion.
`slam_loop_closure.ipynb` one cell; `feedforward_methods.ipynb` one cell
(`keep = clean_pointcloud(np.asarray(pcd.points)); pcd = pcd.select_by_index(np.flatnonzero(keep))`).

## Behaviour changes (complete list)

- `PointcloudResult(frame=, world_transform=, confidence=)` and `CoordinateFrame` no longer exist.
- `ColmapCreator`, `HlocCreator` no longer exist; `make_creator("colmap"|"hloc")` raises `KeyError`.
- `sparse_pc.ply` is written by pycolmap: colourless points black, vertex order per pycolmap.
- Affine depth alignment is gone; alignment is always the per-frame scale model.
- The VDA context stream is gone; VDA always runs over the keyframes.
- The VDA checkpoint is fetched by `huggingface_hub` on first use (cached under `HF_HOME`)
  instead of by `setup.sh`.
- SIFT database generation requires a CUDA `colmap`; no CPU fallback.
- Config keys removed: `preproc.vda_context_fps`, `pointcloud.export_max_points`,
  `pointcloud.instantsfm.features`, `pointcloud.instantsfm.depth_align`,
  `pointcloud.clean.outlier_removal` / `.voxel_size` / `.confidence_threshold`.
- Zarr attrs: `depth_align_model` no longer written; `depth_scale`, `depth_scales`,
  `depth_scale_fallback_frames` unchanged.
- Signatures: `InstantSfMCreator(features=, single_camera=)`, `generate_vda_depth(fps,
  encoder=, input_size=, keep_rows=)`, `vda_depth_complete`, `make_creator(use_lc=,
  lc_config=)`, `decode_context` removed; `clean_pointcloud` as in section 5; ten utils
  functions deleted.
- No numeric behaviour change on the shipped path: every surviving default keeps its value.

## Testing

`tests/` mirrors the package: `tests/pointcloud/sfm/test_instantsfm.py`,
`tests/pointcloud/test_vda.py`, `tests/pointcloud/test_depth_align.py`, split from today's
`test_instantsfm.py` / `test_depth_align.py`, with patch targets moved to the module that
now owns the symbol (`collab_splats.pointcloud.vda.VDA_ROOT`, `...vda._load_vda_model`,
`...vda.hf_hub_download`).

| test file | change |
|---|---|
| `tests/pointcloud/test_base.py` | drop NERFSTUDIO / `world_transform` / enum / `clean_pcd` tests; construct without `frame=` |
| `tests/pointcloud/test_export.py`, `test_export_wiring.py`, `test_sfm_creator.py`, `tests/wrapper/test_vda_context.py` | deleted |
| `tests/pointcloud/test_pointcloud_utils.py` | keep `subsample_points`, `confidence_mask`, `fit_dominant_plane`, lift/reproject cases; one `clean_pointcloud` mask case; delete voxel / distance / bbox / OBB / mask cases |
| `tests/pointcloud/test_registry.py` | drop `colmap` / `hloc` entries |
| `tests/pointcloud/test_depth_align.py` | keep scale, correspondence, fallback, track/pixel-index cases; delete affine cases and the `model=` dispatch test |
| `tests/pointcloud/test_vda.py` | drop `keep_rows` / `fps` / context cases; `hf_hub_download` monkeypatched to a temp file |
| `tests/pointcloud/sfm/test_instantsfm.py` | `InstantSfMCreator()`; drop CPU-fallback cases; `_sift_database_valid` cases on a pycolmap-written DB |
| `tests/test_cu121_migration.py` | add `collab_splats.pointcloud.sfm.instantsfm`, `.vda`, `.depth_align` |
| `tests/wrapper/test_reconstructor_export.py` | drop `export_max_points` cases; keep "writes `sparse_pc.ply` readable by open3d" |
| `tests/wrapper/test_sfm_config.py` | drop the `features` allowlist and `depth_align` cases |
| `tests/wrapper/test_reconstructor.py` | `clean` cases use `{enabled}` only; `:402` comment no longer names `write_pointcloud_ply` |
| `tests/preproc/test_video.py` | drop `decode_context` cases |
| `tests/evals/test_eval_instantsfm.py` | patch `generate_vda_depth` only |
| `tests/geometry/loop_closure/test_wrapper.py:159` | `LoopClosure(make_creator("vggtx"))` |
| `tests/integration/test_pipeline_cu121.py`, `tests/pointcloud/test_vggtx_creator.py`, `tests/pointcloud/feedforward/test_mapanything_creator.py` | drop `frame=` / `CoordinateFrame` imports and `result.frame` asserts |

Verification, in order:

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.loop_closure; import collab_splats.pointcloud"
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud; import collab_splats.geometry.loop_closure"
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/geometry/loop_closure tests/evals tests/preproc tests/test_cu121_migration.py -x -q
black <touched files> && isort <touched files>      # never repo-wide
graphify update .
```

`instantsfm` and the VDA clone's model code are not importable in this venv; their tests
mock the modules today and keep doing so. A live `method: sfm` run is not part of this
change (it needs the `setup.sh` InstantSfM block re-run first).

## Out of scope (follow-ups for the named passes)

- `feedforward/`: its own inline imports and docstrings; whether the creator should write
  `sparse_pc.ply` at all when the Reconstructor rewrites it after clean.
- `wrapper/`: `open3d` / `pycolmap` function-local imports in the Reconstructor;
  `_sfm_result_from_reconstruction` (140 lines that build a `FeedforwardResult` from a
  reconstruction + depth maps) arguably belongs beside `depth_align.py`.
- `preproc/`: `undistort.py`'s `_SIFT_NUM_THREADS` and its pycolmap CPU SIFT.
- Dependencies: bump `pypose` to ≥ 0.9 — deletes `_patch_pypose_robustmodel_target` and
  `_patch_bae_pcg_column_shape`, but `geometry/` BA runs on the same pypose, so it needs a
  BA parity run and a check of the bae version guard at the 0.9.x tags.
- Depth model: Depth-Anything-V2-Metric ships in `transformers` (installed) and would delete
  the clone, `VDA_ROOT`, the `sys.path` dance and the `setup.sh` block. Unmeasured against
  the 19.11 / 0.604 InstantSfM+VDA result and split into indoor/outdoor checkpoints — an
  experiment, not a cleanup.
- `InstantSfMCreator.retriangulation` defaults to `False` although the 2026-08-24
  measurement was a win (+0.7 dB, +92 s); flipping it is a behaviour decision.

## Confirm before planning

Defaults applied in this spec; flip any and the spec is updated before the plan is written.

1. **Delete `ColmapCreator` / `HlocCreator`** (approach B). Flipping this keeps approach A:
   `sfm/colmap.py` (`ColmapCreator(camera_model="SIMPLE_RADIAL", single_camera=False)`,
   body unchanged minus `frame=`) and `sfm/hloc.py` (`HlocCreator(retrieval_conf="netvlad",
   feature_conf="superpoint_aachen", matcher_conf="superglue")`, hloc import behind an
   `ImportError` naming `setup/hloc.sh`, option catalogue cut to one line per field), both
   re-exported and registered, `tests/pointcloud/sfm/{test_colmap,test_hloc}.py` split from
   `test_sfm_creator.py`.
2. **Delete affine depth alignment** and the `depth_align` config key (measured loser).
3. **Delete the VDA context stream** — `preproc.vda_context_fps`, `keep_rows`,
   `_ensure_vda_depth` + sidecar, `decode_context`, `test_vda_context.py` (refuted for the
   metric path). Touches `preproc/video.py` and the Reconstructor's `preprocess()`.
4. **VDA checkpoint via `hf_hub_download`**, clone stays with `VDA_ROOT` as the one path
   constant (upstream ships no package; the Depth-Anything-V2 swap is a follow-up).
5. **`pointcloud.clean` reduces to `enabled`**; `utils` keeps one `clean_pointcloud` mask
   function that both the Reconstructor and the tutorial call.
6. **SIFT database is GPU-only**; CPU fallback and `_SIFT_NUM_THREADS` deleted.
7. **Delete `compute_obb_from_points`, `get_points_in_mask`, `voxel_downsample_point_cloud`,
   the legacy trio, `voxel_downsample`, `filter_distance`.**
8. **Drop `export_max_points`** rather than reimplement thinning on top of `export_PLY`.
9. **Drop `InstantSfMCreator.features` / `.single_camera`**; `InstantSfMCreator` stays
   outside `BasePointcloudCreator`.
