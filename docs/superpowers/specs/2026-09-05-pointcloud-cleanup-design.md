# Pointcloud module cleanup — design

Date: 2026-09-05
Status: draft, revision 4 (simplicity pass: one VDA function, one depth-align function,
no HF constants), awaiting user review

Scope: `collab_splats/pointcloud/` excluding `feedforward/` (separate pass). Consumers are
touched only where the module's API changes or where pointcloud logic was living inline:
`wrapper/reconstructor.py`, `preproc/video.py` (one function), two lines in
`feedforward/base.py`, `evals/scripts/eval.py`, `setup.sh`, configs, tests, docs.

## Problem

The module carries structure that nothing uses, features that were measured and lost, and
logic that its main consumer re-implements inline instead of importing:

1. **Dead coordinate-frame machinery.** `CoordinateFrame.NERFSTUDIO` and
   `PointcloudResult.world_transform` are constructed nowhere in `collab_splats/`; every
   creator passes `frame=CoordinateFrame.COLMAP`. `PointcloudResult.confidence` is never set.
   Only `tests/pointcloud/test_base.py` exercises them. `utils.py`'s header still says poses
   come out in `CoordinateFrame.NERFSTUDIO`. The Reconstructor also writes a
   nerfstudio-format `transforms.json` that no code in the repo reads (its own docstring
   says a stock dataparser cannot load it).
2. **Function-local imports** in `base.py` (`_write_ply`, to dodge the
   `export -> utils -> base` cycle), `sfm.py` (hloc, VDA, instantsfm, pypose, bae, scipy),
   `utils.py` (open3d in six functions, `rotation_align_vectors`), `__init__.py`
   (`LoopClosure`), and the Reconstructor's imports of pointcloud symbols.
3. **Two PLY writers for one file.** `export.write_pointcloud_ply` (114 lines) is wrapped by
   `BasePointcloudCreator._write_ply`, and `sparse_pc.ply` is written twice per run.
   `pycolmap.Reconstruction.export_PLY` (pycolmap 4.0.4, probed) writes the identical binary
   little-endian layout: float x/y/z + uchar r/g/b.
4. **`sfm.py` is 1289 lines** holding three backends, VDA depth generation, scale + affine
   depth alignment, a SIFT database builder, and four upstream monkeypatches.
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
7. **The Reconstructor re-implements pointcloud logic inline** instead of importing it:
   outlier removal (`_clean_pointcloud`, open3d by hand, with a `voxel_down_sample` whose
   result is discarded so `pointcloud.clean.voxel_size` is a no-op and
   `clean.confidence_threshold` "NOT YET READ"), PLY export, model load plus registration
   check (`_load_pointcloud_from_disk`), COLMAP image renaming (`_rename_images_to_stems`),
   VDA orchestration (`_ensure_vda_depth`), and 140 lines assembling a `FeedforwardResult`
   from a model plus depth maps (`_sfm_result_from_reconstruction`).
8. **Overengineering / dead code**: parameters no caller passes (`InstantSfMCreator.features`
   and `.single_camera`, `generate_vda_depth(fps=, encoder=, input_size=)`,
   `make_creator(use_lc=, lc_config=)`); a `VDA_CHECKPOINT` path + `wget` block for a file
   `huggingface_hub` already knows how to fetch and cache; two voxel downsamplers;
   `utils.clean_pointcloud` / `filter_distance` / `voxel_downsample` used by one tutorial
   cell; `compute_obb_from_points` / `get_points_in_mask` exported to no consumer; a
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
- `transforms.json` readers: none in `collab_splats/`, `evals/`, `docs/` (only the writer
  and two docstring mentions).

## Approaches considered

- **A. Package split with all three backends, VDA and depth alignment as top-level
  pointcloud modules, Reconstructor delegates, delete the measured-and-lost features
  (chosen).** `sfm/{colmap,hloc,instantsfm}.py`; `vda.py`, `depth_align.py` beside
  `base.py`; every inline pointcloud routine in the Reconstructor becomes an import.
- **B. Same, but delete `ColmapCreator` / `HlocCreator`.** Rejected by the user: the
  creators stay as library backends (the Reconstructor still raises `NotImplementedError`
  for them; wiring is a follow-up).
- **C. Minimal: items 1, 2, 3, 6 only, single `sfm.py`.** Leaves a 1000-line file, ships
  options that lost, leaves the Reconstructor's copies in place. Rejected.

## Decisions

### 1. `base.py` — `PointcloudResult` is a reconstruction plus its image order, and knows how to load and save itself

`CoordinateFrame`, `PointcloudResult.frame`, `.world_transform`, `.confidence`, and
`BasePointcloudCreator._write_ply` are deleted. Everything the pipeline emits is a
`pycolmap.Reconstruction` in COLMAP's frame; the OpenGL conversion lives in
`geometry/transforms.py` and takes plain arrays. I/O lives on the result, mirroring
`FeedforwardResult.save_zarr` / `load_zarr`:

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

    @classmethod
    def from_colmap(cls, colmap_dir: Path, image_paths: list[Path]) -> "PointcloudResult":
        """
        Read a binary COLMAP model and bind it to the caller's image order.

        - colmap_dir: directory holding cameras/images/points3D.bin.
        - image_paths: expected registered names in store order.
        - returns: PointcloudResult; raises ValueError naming the first path not registered.
        """

    def write_ply(self, path: Path) -> None:
        """
        Write the sparse points as a binary little-endian PLY (float xyz, uchar rgb).

        - path: output file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.reconstruction.export_PLY(str(path))


class BasePointcloudCreator(ABC):
    @abstractmethod
    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult: ...
```

`write_ply` keeps the repo's `write_*` / `save_*` verb on the data object; pycolmap's
`export_PLY` is the implementation, not the API consumers see. `from_colmap` is today's
`Reconstructor._load_pointcloud_from_disk` minus the frame-store lookup (the caller passes
`image_paths`), including the registration check and its error message.

### 2. Imports at the top, one sanctioned exception

All imports move to module top. The exception (CLAUDE.md: optional heavy deps that would
break the module on a missing install) applies to exactly these, each guarded by a clear
`ImportError` message at the call site:

| import | where | why it stays local |
|---|---|---|
| `hloc.*` | `sfm/hloc.py::HlocCreator.reconstruct` | third_party clone (`setup/hloc.sh`), not installed here |
| `video_depth_anything.video_depth` | `pointcloud/vda.py::_load_vda_model` | third_party clone that must sit first on `sys.path` while it imports (it imports a top-level `utils`); not site-packages |
| `instantsfm.*`, `pypose.optim.optimizer`, `bae.utils.pysolvers` | `pointcloud/sfm/instantsfm.py` (`_patch_*`, `_build_config`, `reconstruct`) | optional backend installed outside the lock; `test_import_all_modules` must keep passing |

Everything else is hoisted: `scipy.spatial.transform.Rotation`, `huggingface_hub`,
`open3d` in `utils.py` (hard dep; `mesh/` and tests import it at top),
`rotation_align_vectors` (`geometry/transforms.py` imports only numpy — no cycle),
`FeedforwardResult` in `depth_align.py` (`feedforward/base.py` imports `pointcloud.base`
and `pointcloud.utils` only — no cycle), and in the Reconstructor every pointcloud symbol
it uses (section 8).

The `__init__.py -> geometry.loop_closure -> pointcloud.base` cycle is removed rather than
worked around: `make_creator` loses `use_lc`/`lc_config` (only
`tests/geometry/loop_closure/test_wrapper.py:159` used them; the Reconstructor and the
tutorials wrap `LoopClosure(base, config=...)` themselves). `export.py` is deleted, which
removes the `base -> export -> utils -> base` cycle.

### 3. PLY export is `PointcloudResult.write_ply`

`export.py` is deleted. Both writers become one call:

- `feedforward/base.py:1163`: `result.write_ply(Path(output_dir) / "sparse_pc.ply")`
  (the only edit to `feedforward/` besides dropping `frame=`).
- Reconstructor: `result.write_ply(self.backend_dir / "sparse_pc.ply")` after clean.

`pointcloud.export_max_points` is dropped from `configs/base.yaml`, the Reconstructor, and
its test. It defaults to `null`, no config sets it, and the thinning hook was the only
reason for a hand-rolled writer. Behaviour deltas: colourless points come out black
(pycolmap default) instead of grey 128 — every shipped creator sets colours — and vertex
order follows pycolmap's `points3D` iteration; no consumer depends on order.

### 4. Layout — `sfm/` holds backends; VDA and depth alignment are pointcloud modules

```
collab_splats/pointcloud/
  __init__.py       registry: colmap, hloc, mapanything, vggtx (+ optional feedforward entries)
  base.py           BasePointcloudCreator, PointcloudResult (.from_colmap, .write_ply)
  utils.py          lift_features, reproject_pixels, clean_pointcloud, subsample_points, ...
  vda.py            generate_vda_depth — Video-Depth-Anything metric depth, written to
                    depth_vda/images/npy and returned as a stack
  depth_align.py    result_from_reconstruction — model + depth stack -> FeedforwardResult
                    in the model's scale (per-frame scale fit); private helpers
  sfm/
    __init__.py     re-exports ColmapCreator, HlocCreator, InstantSfMCreator
    colmap.py       ColmapCreator — pycolmap SIFT + exhaustive + incremental mapping
    hloc.py         HlocCreator — hloc retrieval/features/matching -> COLMAP mapper
    instantsfm.py   SIFT database, upstream compat patches, InstantSfMCreator
  feedforward/      unchanged (own pass)
```

Why VDA and alignment are not under `sfm/`: VDA is a generic monocular video depth model —
nothing about it is InstantSfM's except the on-disk layout its reader expects, which
`vda.py` owns in one place. Depth alignment and result assembly take any
`pycolmap.Reconstruction` plus a depth stack; they are not tied to a backend.

**`sfm/colmap.py`** — `ColmapCreator(camera_model="SIMPLE_RADIAL", single_camera=False)`;
body unchanged except `frame=` removed and `self._write_ply(...)` -> `result.write_ply(...)`;
docstring cut to the contract.

**`sfm/hloc.py`** — `HlocCreator(retrieval_conf="netvlad", feature_conf="superpoint_aachen",
matcher_conf="superglue")`; body unchanged except the same two edits and the hloc import
wrapped with an `ImportError` pointing at `setup/hloc.sh`; the 60-line option catalogue
is replaced by one line per field ("hloc config key; see `hloc.extract_features.confs`").

**`sfm/instantsfm.py`** — sections in this order, `########` dividers between them:

1. *SIFT database.* `_nudge_edge_keypoints(features, width, height)` (upstream
   `sample_depth_at_pixel` raises `IndexError` at `x == width`; still unfixed at `master`),
   `_sift_database_valid(database_path)` rewritten on `pycolmap.Database`
   (`num_keypoints() > 0 and num_verified_image_pairs() > 0`; the `sqlite3` import goes),
   `_generate_sift_database(image_path, database_path, *, num_threads: int = 8)` —
   `colmap feature_extractor` + `exhaustive_matcher` via `subprocess`,
   `--ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1`, GPU when
   `torch.cuda.is_available()`, else `CUDA_VISIBLE_DEVICES=""` with `num_threads` (the
   `_SIFT_NUM_THREADS` constant becomes this keyword default). Partial DB unlinked on
   failure. A four-line block comment records why the CLI and not pycolmap (CUDA build vs
   CPU wheel, 13× / 15×) and why not upstream's `GenerateDatabase` (forced CPU, errors
   swallowed).
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
    - reconstruct(data_dir): data_dir/images/ (+ depth_vda/) -> pycolmap.Reconstruction
      whose image names are filename stems (frame_NNNNNN); model written to
      data_dir/colmap/sparse/0, SIFT DB at data_dir/colmap/instantsfm.db.
    """

    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None
```

`features` is dropped (upstream 0.3.0 supports only `"colmap"`; the Reconstructor
validated the value and never dispatched on it) and `single_camera` is dropped (never
passed; the `depth_vda/images/npy` layout is upstream's single-camera branch, so `False`
was never a working path). `_build_config` calls `Config("colmap")`. `reconstruct` gains
the stem rename as its last step (today's `Reconstructor._rename_images_to_stems`: set
`im.name = Path(im.name).stem` on every image, `write_binary` the model in place) so the
creator returns the pipeline's naming contract instead of leaving it to the caller. The
creator stays outside `BasePointcloudCreator` on purpose: its contract is a staged scene
dir in, a `pycolmap.Reconstruction` out, and the Reconstructor wraps it — one docstring
bullet says so. The rest of the body is unchanged (patches applied, stale `colmap/sparse`
removed, SIFT DB reused when valid, `ReadDepthsIntoFeatures` when `use_depths`,
`SolveGlobalMapper`, `WriteGlomapReconstruction`, cluster checks, read back via pycolmap).

**`vda.py`**

```python
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"


def _load_vda_model(device: str):
    """
    Build the metric ViT-L Video-Depth-Anything model with Hub-cached weights.

    - device: torch device the model is moved to.
    - returns: eval-mode VideoDepthAnything.
    """
    # Clone root first on sys.path only while importing: the package imports a top-level
    # `utils` from its own root. Same pattern as feedforward/loger.py and vggt_spark_creator.py.
    ...
    checkpoint = hf_hub_download(
        repo_id="depth-anything/Metric-Video-Depth-Anything-Large",
        filename="metric_video_depth_anything_vitl.pth",
    )


def generate_vda_depth(
    frames: np.ndarray, out_dir: Path, names: list[str], *, depth_width: int = 518, device: str = "cuda",
) -> np.ndarray:
    """
    Run Video-Depth-Anything metric depth over keyframes and write InstantSfM's depth layout.

    - frames: (N, H, W, 3) uint8 RGB, one per name.
    - out_dir: scene dir; maps land at out_dir/depth_vda/images/npy/<stem>.npy.
    - names: staged image filenames, one per written map, in order.
    - depth_width: written map width; height keeps aspect (nearest resize).
    - device: torch device for inference.
    - returns: (N, h, w) float32 metric depth in `names` order. When every stem already
      has a map, inference is skipped and the maps are read back.
    """
```

One public function: the maps are written for InstantSfM's reader and returned for result
assembly, so no separate loader exists and the wrapper never touches the layout.

Deleted: `VDA_CHECKPOINT` and the `setup.sh` `wget` block (L88–95; `hf_hub_download`
caches under `HF_HOME` and is the standard route for this file), `_VDA_MODEL_CONFIGS` and
the `encoder` parameter (only `vitl` exists; the config dict is inlined), `input_size`
(never passed; 518 inlined), `fps` (upstream's `infer_video_depth` returns `target_fps`
unchanged and resamples nothing; a literal is passed), `keep_rows` and the one-to-one
validation (context stream, item 5 above), and `vda_depth_complete` (callers call
`generate_vda_depth`, which skips inference on a complete stem set). The Hub repo and
filename are literals in `_load_vda_model` (tests monkeypatch `hf_hub_download`, not the
names). `VDA_ROOT` stays: the model code cannot be a standard install without a fork
carrying build metadata, the clone root must be on `sys.path` while the package imports,
and tests point it at fixtures.

**`depth_align.py`** — one public function; dense depth goes in, a result in the model's
scale comes out:

```python
def result_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    depths: np.ndarray,
    images: np.ndarray,
    names: list[str],
    *,
    min_obs: int = 20,
) -> tuple[FeedforwardResult, dict]:
    """
    Assemble a FeedforwardResult from a COLMAP model and dense depth scaled into the model's frame.

    - reconstruction: fully registered model; image names are the stems of `names`.
    - depths: (N, h, w) float32 metric depth, one per name; defines the model-resolution grid.
    - images: (N, H, W, 3) uint8 RGB at the COLMAP camera resolution, same order.
    - names: staged image filenames in store order.
    - min_obs: tracked observations a frame needs for its own scale; fewer -> global median.
    - returns: (result, attrs). result: K, pixel_indices and depth at (h, w), world_points
      unprojected from the scaled depth, no confidence. attrs: {"depth_scale": "colmap",
      "depth_scales": [...], "depth_scale_fallback_frames": [...]} for the zarr.
      Raises RuntimeError on partial registration, ValueError on a name or camera-resolution
      mismatch.
    """


def _tracked_point3d_ids(reconstruction) -> list[int]: ...
def _pixel_indices_from_reconstruction(reconstruction, point3d_ids, name_to_row, scale_x, scale_y, depth_hw) -> np.ndarray: ...
def _depth_correspondences(reconstruction, image_names, depth) -> list[tuple[np.ndarray, np.ndarray]]: ...
def _fit_depth_scales(reconstruction, image_names, depth, *, min_obs) -> tuple[np.ndarray, list[int]]: ...
```

Today this is two steps in the Reconstructor: `_sfm_result_from_reconstruction` builds a
result (unprojecting `world_points` from unscaled depth), then `apply_depth_alignment`
rescales `depth` in place and unprojects again. The un-aligned result is never consumed,
and with the affine option gone the scale fit is unconditional, so the two collapse into
one function: check, fit scales (`_fit_depth_scales` is today's
`align_depth_to_reconstruction`, per-frame median ratio with the global-median fallback),
scale the depth stack, build the result once. The three checks and their error texts move
with it, naming the arguments instead of `self.frames_zarr`. The stats dict shrinks to
what is logged: global scale, ratio p10/p50/p90 before and after, fallback count.
`depth_scale` stays the attr key the Reconstructor checks when it loads a store.
The helpers keep their underscore names: nothing outside the module calls them once the
Reconstructor stops assembling results itself.

Deleted: `apply_depth_alignment`, `align_depth_to_reconstruction` (folded in),
`DepthAlignModel`, `DEPTH_ALIGN_MODELS`, `align_depth_affine`, `_solve_disparity`,
`_fit_affine_disparity`, `_apply_affine_depth`, `MIN_ALIGN_OBS` (→ `min_obs=20`),
`MIN_AFFINE_OBS`, `AFFINE_REJECT_ROUNDS`, `AFFINE_MIN_FAR_DISPARITY_FRAC`, and the
`depth_align_model` attr.

### 5. `utils.py`

| symbol | action |
|---|---|
| header docstring | rewritten: what the module holds; no frame claims |
| `BaseFeatureExtractor` try-import, `_UNSET`, `_DEFAULT_*_KWARGS` | deleted |
| `clean_pcd`, `remove_far_points`, `density_filter` ("Legacy cleaning utilities") | deleted; only `test_base.py` used them; `density_filter` returns nothing |
| `voxel_downsample`, `filter_distance`, `_radial_mask`, `_bbox_mask` | deleted; `clean_pointcloud` was their only consumer |
| `voxel_downsample_point_cloud` | deleted; `slam_loop_closure.ipynb` (one cell, two calls) switches to `subsample_points(pts, colors, max_points=...)` |
| `compute_obb_from_points`, `get_points_in_mask` | deleted; exported to no consumer |
| `clean_pointcloud` | `clean_pointcloud(points: np.ndarray, *, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray` — open3d statistical outlier removal, returns a bool keep mask over `points`. The one cleaning step both consumers run (Reconstructor, `feedforward_methods.ipynb`) |
| `confidence_mask`, `subsample_points`, `fit_dominant_plane`, `_grid_sample_at_pixels`, `_sample_at_source_pixels`, `lift_features`, `reproject_pixels`, `cross_frame_attention_ratio` | kept; imports hoisted; docstrings to contract |

### 6. `__init__.py`

Drops `CoordinateFrame`, `compute_obb_from_points`, `get_points_in_mask` from imports and
`__all__`; `make_creator(name, **kwargs)` only. `_REGISTRY` (colmap, hloc, mapanything,
vggtx + optional feedforward entries) is unchanged.

### 7. Docstring contract

Every public function and class in the module (and every private one that keeps a
docstring) follows one shape — `"""` on their own lines, one-line summary, blank line,
one bullet per input, one `returns` bullet, nothing else (see `generate_vda_depth` and
`InstantSfMCreator` above). Rationale a maintainer still needs (upstream bug being
patched, the GPU/CPU measurement, layout contracts with instantsfm's reader) moves to a
block comment of at most four lines at the site, keeping repo + commit + file:line
attributions. Rationale that restates git history or a memory entry is dropped.

### 8. The Reconstructor orchestrates; the module owns the logic

Every pointcloud routine the Reconstructor implements inline moves to the module and is
imported at the top of `reconstructor.py`:

| Reconstructor today | becomes |
|---|---|
| `_clean_pointcloud` (open3d by hand, discarded voxel step) | `keep = clean_pointcloud(result.points)`; delete the point3D ids where `keep` is false; log the count |
| `_export_pointcloud_ply` + `export.write_pointcloud_ply` | `result.write_ply(self.backend_dir / "sparse_pc.ply")` |
| `_load_pointcloud_from_disk` | `PointcloudResult.from_colmap(self.backend_dir / "colmap/sparse/0", image_paths)` with `image_paths` built from the store's frame indices |
| `_rename_images_to_stems` | inside `InstantSfMCreator.reconstruct` |
| `_ensure_vda_depth` + `_context_keep_rows` + `_video_unchanged` + sidecar | `depths = generate_vda_depth(frames, self.backend_dir, names)` |
| `_sfm_result_from_reconstruction` + `apply_depth_alignment` | `outputs, align_attrs = result_from_reconstruction(recon, depths, frames, names)` |
| `_write_transforms_json` | deleted — nerfstudio-format artifact with no reader (confirm item 5) |

`_run_sfm` after the change, in full:

```python
def _run_sfm(self) -> PointcloudResult:
    pc_cfg = self.config["pointcloud"]
    if pc_cfg["backend"] != "instantsfm":
        raise NotImplementedError(f"sfm backend {pc_cfg['backend']!r} is not implemented — only 'instantsfm' is")
    store = FrameStore.open(self.frames_zarr)
    names = [f"frame_{int(fi):06d}.jpg" for fi in store.frame_indices()]

    # Stage keyframes as jpgs via store.export; a different staged set also drops the SIFT
    # DB keyed on it (today's inline block, unchanged; shown collapsed here)
    ...

    # VDA depth (cached per stem), global SfM, dense result scaled into the COLMAP world
    frames = store.images()
    depths = generate_vda_depth(frames, self.backend_dir, names)
    recon = InstantSfMCreator(
        retriangulation=pc_cfg["instantsfm"]["retriangulation"],
        random_seed=pc_cfg["instantsfm"]["random_seed"],
    ).reconstruct(self.backend_dir)
    outputs, align_attrs = result_from_reconstruction(recon, depths, frames, names)

    outputs.save_zarr(
        self.backend_dir / "pointcloud.zarr",
        extra_attrs={"method": "sfm", "backend": "instantsfm",
                     "instantsfm_version": importlib.metadata.version("instantsfm"), **align_attrs},
    )
    return PointcloudResult(reconstruction=recon, image_paths=outputs.image_paths)
```

The `torch.cuda.empty_cache()` calls between steps stay where they are today. Staging
files on disk, config reads, and zarr provenance are the wrapper's job and stay.

## Consumer edits

**`wrapper/reconstructor.py`**
- Top imports: `PointcloudResult`, `confidence_mask`, `lift_features`, `clean_pointcloud`,
  `generate_vda_depth`, `result_from_reconstruction`, `InstantSfMCreator`; drop
  `write_pointcloud_ply`, `apply_depth_alignment`,
  `CoordinateFrame`, `DEPTH_ALIGN_MODELS`, `vda_depth_complete`, `context_indices`,
  `decode_context`, `_tracked_point3d_ids`, `_pixel_indices_from_reconstruction`,
  `unproject_depth_map_to_point_map` (if `refine_poses` is its only other user it stays
  for that), and the four function-local pointcloud imports (`_lift_and_save` ->
  `lift_features`; `_load_pointcloud_from_disk` and `_run_sfm` -> `CoordinateFrame,
  PointcloudResult`; `splats` -> `confidence_mask`). The lazy `FeedforwardResult`,
  `pycolmap`, and gsplat imports elsewhere in the file are the wrapper pass's.
- Section 8 table: seven methods replaced or deleted; `_run_sfm` as shown.
- `preprocess()`: delete the `vda_context_fps` parameter, its docstring bullet, the
  image-dir warning and `optical_flow` rejection, the context-grid candidate branch, and
  the provenance key (L194–322 pieces).
- The `depth_align=%s` clause in the splats log line (L1834–1836) goes; the
  "Write the pose+intrinsics transforms.json" call and comment (L1013–1014) go.
- Validation: delete `_INSTANTSFM_FEATURES` and its check and the `depth_align` check; keep
  `random_seed`, `_SFM_BACKENDS`, and the `NotImplementedError` for `colmap` / `hloc`.

**`preproc/video.py`** — delete `decode_context` (its only consumer was the context
stream); `context_indices` stays (`preproc/sampling.py` uses it).

**`feedforward/base.py`** — drop `frame=`; `self._write_ply(result, Path(output_dir))` ->
`result.write_ply(Path(output_dir) / "sparse_pc.ply")`; the `transforms.json` mention at
L769 goes.

**`evals/scripts/eval.py`** — `generate_vda_depth(frames, output_dir, names)` (return
ignored; InstantSfM reads the maps from disk); drop the `vda_depth_complete` import and
guard (L358–360). `InstantSfMCreator(use_depths=...)` is unchanged — the eval's two
conditions are the one caller of that flag.

**`setup.sh`** — delete the checkpoint `wget` block (L88–95) and the `VDA_CKPT` variable;
the clone + pin block stays; the comment at L74 names `pointcloud/vda.py`.

**`configs/base.yaml`** — delete `preproc.vda_context_fps`, `pointcloud.export_max_points`,
`pointcloud.instantsfm.features`, `pointcloud.instantsfm.depth_align`,
`pointcloud.clean.outlier_removal`, `.voxel_size`, `.confidence_threshold`
(`clean: {enabled: true}` remains); `backend` comment reads "sfm: instantsfm (colmap |
hloc are library creators, not wired here)".
**`configs/README.md`** — drop the `preproc.vda_context_fps`, `pointcloud.instantsfm.features`
and `.depth_align` rows and the matching sentences in the instantsfm backend section
(L458–515); the `pointcloud.backend` row points at `pointcloud/sfm/`; `clean` rows reduce
to `enabled`; any mention of `transforms.json` as an output goes.

**Comments/docstrings pointing at old paths** — `geometry/transforms.py:5`
("`CoordinateFrame.NERFSTUDIO`" -> "OpenGL"), `preproc/undistort.py:83`
(`pointcloud/sfm.py::_SIFT_NUM_THREADS` -> `pointcloud/sfm/instantsfm.py::_generate_sift_database`),
`remote/sources.py:71`, `setup.sh:74`, `pointcloud/base.py:108`.

**Docs** — `docs/source/api/pointcloud.rst`: replace the `collab_splats.pointcloud.sfm`
automodule with `sfm.colmap`, `sfm.hloc`, `sfm.instantsfm`, `vda`, `depth_align`.
`CLAUDE.md` architecture tree: `sfm.py` -> `sfm/` + `vda.py` + `depth_align.py`,
`export.py` line removed, `utils.py` line lists what it holds.
`docs/superpowers/CHANGELOG.md` entry on completion. `slam_loop_closure.ipynb` one cell;
`feedforward_methods.ipynb` one cell
(`keep = clean_pointcloud(np.asarray(pcd.points)); pcd = pcd.select_by_index(np.flatnonzero(keep))`).

## Behaviour changes (complete list)

- `PointcloudResult(frame=, world_transform=, confidence=)` and `CoordinateFrame` no longer exist.
- `sparse_pc.ply` is written by pycolmap: colourless points black, vertex order per pycolmap.
- `transforms.json` is no longer written (confirm item 5).
- Affine depth alignment is gone; alignment is always the per-frame scale model.
- The VDA context stream is gone; VDA always runs over the keyframes.
- The VDA checkpoint is fetched by `huggingface_hub` on first use (cached under `HF_HOME`)
  instead of by `setup.sh`.
- `InstantSfMCreator.reconstruct` returns and writes stem-named images.
- Config keys removed: `preproc.vda_context_fps`, `pointcloud.export_max_points`,
  `pointcloud.instantsfm.features`, `pointcloud.instantsfm.depth_align`,
  `pointcloud.clean.outlier_removal` / `.voxel_size` / `.confidence_threshold`.
- Zarr attrs: `depth_align_model` no longer written; `depth_scale`, `depth_scales`,
  `depth_scale_fallback_frames` unchanged.
- `generate_vda_depth` returns the `(N, h, w)` depth stack instead of a path.
- `apply_depth_alignment` and `align_depth_to_reconstruction` are folded into
  `result_from_reconstruction`; world points are unprojected once, from scaled depth
  (same values as today's second unprojection).
- Signatures: `InstantSfMCreator(features=, single_camera=)`, `generate_vda_depth(fps,
  encoder=, input_size=, keep_rows=)`, `vda_depth_complete`, `make_creator(use_lc=,
  lc_config=)`, `decode_context` removed; `clean_pointcloud` as in section 5; ten utils
  functions deleted.
- No numeric behaviour change on the shipped path: every surviving default keeps its value.

## Testing

`tests/` mirrors the package: `tests/pointcloud/sfm/{test_colmap,test_hloc,test_instantsfm}.py`,
`tests/pointcloud/test_vda.py`, `tests/pointcloud/test_depth_align.py`, split from today's
`test_sfm_creator.py` / `test_instantsfm.py` / `test_depth_align.py` /
`tests/wrapper/test_sfm_result.py`, with patch targets moved to the module that now owns
the symbol (`collab_splats.pointcloud.sfm.colmap.pycolmap.extract_features`,
`...vda.VDA_ROOT`, `...vda._load_vda_model`, `...vda.hf_hub_download`).

| test file | change |
|---|---|
| `tests/pointcloud/test_base.py` | drop NERFSTUDIO / `world_transform` / enum / `clean_pcd` tests; construct without `frame=`; add `from_colmap` (round-trip + missing-name ValueError) and `write_ply` (open3d reads it back) |
| `tests/pointcloud/test_export.py`, `test_export_wiring.py`, `tests/wrapper/test_vda_context.py`, `tests/wrapper/test_transforms_json.py` | deleted |
| `tests/wrapper/test_sfm_result.py` | cases move: rename -> `tests/pointcloud/sfm/test_instantsfm.py`, result assembly + checks -> `tests/pointcloud/test_depth_align.py` (`result_from_reconstruction` on arrays) |
| `tests/pointcloud/test_pointcloud_utils.py` | keep `subsample_points`, `confidence_mask`, `fit_dominant_plane`, lift/reproject cases; one `clean_pointcloud` mask case; delete voxel / distance / bbox / OBB / mask cases |
| `tests/pointcloud/test_registry.py` | unchanged |
| `tests/pointcloud/test_depth_align.py` | keep correspondence, scale-fit, fallback, track/pixel-index cases on the private helpers; the two surviving `apply_depth_alignment` cases (attrs stamped; world points from scaled depth) become `result_from_reconstruction` cases; delete affine cases and the `model=` dispatch test |
| `tests/pointcloud/test_vda.py` | drop `keep_rows` / `fps` / context cases; assert the returned stack, and that a complete stem set is read back without inference; `hf_hub_download` monkeypatched to a temp file |
| `tests/pointcloud/sfm/test_instantsfm.py` | `InstantSfMCreator()`; CPU-fallback cases keep passing `num_threads`; `_sift_database_valid` cases on a pycolmap-written DB; stem rename asserted on the returned model |
| `tests/pointcloud/sfm/test_colmap.py`, `test_hloc.py` | split from `test_sfm_creator.py`; drop `frame` asserts |
| `tests/test_cu121_migration.py` | add `collab_splats.pointcloud.sfm.colmap`, `.sfm.hloc`, `.sfm.instantsfm`, `.vda`, `.depth_align` |
| `tests/wrapper/test_reconstructor_export.py` | drop `export_max_points` cases; keep "writes `sparse_pc.ply` readable by open3d" |
| `tests/wrapper/test_sfm_config.py` | drop the `features` allowlist and `depth_align` cases |
| `tests/wrapper/test_reconstructor.py` | `clean` cases use `{enabled}` only; `_load_pointcloud_from_disk` cases assert through `from_colmap`; `:402` comment no longer names `write_pointcloud_ply` |
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

`instantsfm`, `hloc`, and the VDA clone's model code are not importable in this venv;
their tests mock the modules today and keep doing so. A live `method: sfm` run is not part
of this change (it needs the `setup.sh` InstantSfM block re-run first).

## Out of scope (follow-ups for the named passes)

- `feedforward/`: its own inline imports and docstrings; whether the creator should write
  `sparse_pc.ply` at all when the Reconstructor rewrites it after clean.
- `wrapper/`: `open3d` / `pycolmap` / gsplat function-local imports in the Reconstructor;
  wiring `colmap` / `hloc` into `_run_sfm` (today `NotImplementedError`).
- `preproc/`: `undistort.py`'s own `_SIFT_NUM_THREADS` and its pycolmap CPU SIFT.
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
Settled by the user on 2026-09-05: keep `ColmapCreator` / `HlocCreator`; keep the CPU SIFT
fallback; the Reconstructor imports from the module instead of re-implementing.

1. **Delete affine depth alignment** and the `depth_align` config key (measured loser).
2. **Delete the VDA context stream** — `preproc.vda_context_fps`, `keep_rows`,
   `_ensure_vda_depth` + sidecar, `decode_context`, `test_vda_context.py` (refuted for the
   metric path). Touches `preproc/video.py` and the Reconstructor's `preprocess()`.
3. **VDA checkpoint via `hf_hub_download`**, clone stays with `VDA_ROOT` as the one path
   constant (upstream ships no package; the Depth-Anything-V2 swap is a follow-up).
4. **`pointcloud.clean` reduces to `enabled`**; `utils` keeps one `clean_pointcloud` mask
   function that both the Reconstructor and the tutorial call.
5. **Delete `_write_transforms_json`** and stop writing `transforms.json` (nerfstudio
   format, no reader in the repo, not loadable by nerfstudio either per its docstring).
   Flip this and it becomes `PointcloudResult.write_transforms_json(path)`.
6. **Result assembly and scale alignment are one function**, `result_from_reconstruction`
   in `depth_align.py`, taking arrays and returning `(result, attrs)`;
   `generate_vda_depth` returns the depth stack so no loader exists. Alternative: keep
   today's two-step assemble-then-align shape.
7. **Delete `compute_obb_from_points`, `get_points_in_mask`, `voxel_downsample_point_cloud`,
   the legacy trio, `voxel_downsample`, `filter_distance`.**
8. **Drop `export_max_points`** rather than reimplement thinning on top of `write_ply`.
9. **Drop `InstantSfMCreator.features` / `.single_camera`**; `InstantSfMCreator` stays
   outside `BasePointcloudCreator`.
