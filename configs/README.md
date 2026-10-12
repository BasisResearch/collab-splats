# Reconstruction Pipeline Configs

All pipelines start from default template `base.yaml` — this file holds the defaults for each stage. If a separate file is
specified (e.g., `MYFILE.yaml`), the file only overrides what differs from `base.yaml`. Running the pipeline and its outputs are covered in the [main README](../README.md#quickstart).

- [Overriding default arguments](#overriding-default-arguments)
- [Stages](#stages)
- Options by stage: [`preproc`](#preproc) · [`pointcloud`](#pointcloud) ([SfM](#sfm-backends) · [`bundle_adjustment`](#bundle_adjustment) · [`loop_closure`](#loop_closure)) · [`mesh`](#mesh) · [`semantics`](#semantics) · [`splats`](#splats) · [`localization`](#localization) · [`reconstruction_quality_report`](#reconstruction_quality_report)
- [Remote scenes](#remote-scenes)

---

## Overriding default arguments

We override options in three ways, applied in this order (later wins):

1. `base.yaml`
2. `--config MYFILE.yaml`: merged key by key over `base.yaml`
3. `--set key=value`: one dotted key, repeatable

An override file holds only what differs:

```yaml
# configs/my_run.yaml: indoor scene, denser frames, no semantics
preproc:
  fps: 3.0
mesh:
  use_convex_hull: false
semantics:
  enabled: false
```

```bash
reconstruct local --config configs/my_run.yaml --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4

# One key from the command line
reconstruct local --set splats.enabled=true --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4
```

### Re-running after a config change

We re-run only the stages a change affects with `--stages` and `--overwrite`:

```bash
# New mesh options on an existing scene
reconstruct local --config configs/remesh.yaml --stages mesh --overwrite \
  --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4
```

- a named stage whose output exists is a `ValueError` without `--overwrite` (`preproc` / `pointcloud` just skip)
- stages turned on by `enabled` flags skip when their output exists, so an interrupted run resumes
- `localize` re-runs on its own after `refine`; re-run anything else that depends on the change (see [Stages](#stages))

### Reproducing a run

Every backend folder records its full merged config in `run_config.yaml`, rewritten on every run (so it holds the last command's config). We pass it back as `--config`:

```bash
reconstruct local --config /PATH/TO/OUTPUT/<name>/vggt_omega/run_config.yaml \
  --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4

# Same run with another backend (writes its own <name>/vggtx/ folder)
reconstruct local --config /PATH/TO/OUTPUT/<name>/vggt_omega/run_config.yaml \
  --set pointcloud.backend=vggtx \
  --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4
```

---

## Stages

| Stage | Default | Requires |
|---|---|---|
| `preproc` | always | |
| `pointcloud` | always | `preproc` |
| `refine` | on (feedforward only) | `pointcloud` |
| `splats` | off | `pointcloud` |
| `mesh` | on | `pointcloud` (`splats` too with `mesh.source: splats`) |
| `semantics` | on | `pointcloud` |
| `localize` | on | `pointcloud` |
| `reconstruction_quality_report` | always | `pointcloud` |

- `on` / `off` stages switch with their `enabled` key (`refine`: `pointcloud.bundle_adjustment.enabled`; `localize`: `localization.enabled`)
- naming a stage in `--stages` runs it even when its flag is off
- stages run in table order
- `semantics` also lifts onto the mesh when `mesh.ply` exists, and re-lifts from its cached 2D codes when the mesh changes

---

## `preproc`

Preproc measures the video once, then selects keyframes from that measurement.

- **measure**: `video_quality_report.json` (per-frame blur, exposure, clipping; per-pair motion); reused if present, delete it to re-measure
- **select**: a frame is eligible if it passes `preproc.quality`; changing the sampler never re-measures

Each sampler has one density knob; `max_frames` is the count for `uniform` and a ceiling for the others:

| `frame_selection` | Knob | Holds constant |
|---|---|---|
| `fps` | `fps` | time between frames, so the baseline is fixed; count floats |
| `uniform` | `max_frames` | frame count; spacing floats |
| `optical_flow` | `max_frames` (keeps the first N) | inter-frame motion; count and spacing float |

We use `fps` for reconstruction: registration depends on the baseline between frames, which a count-based knob lets vary with video length.

- `[min_frames, max_frames]` bounds the `fps` count; outside it, frames are re-spread over the eligible frames and the effective fps is logged
- `max_frames: 300` binds past ~2.5 min at `fps: 2.0`; it is VGGT-Omega's GPU limit
- keys for other samplers are ignored (e.g. `fps` under `uniform`)

```yaml
# Fixed 150 frames, evenly spaced
preproc:
  frame_selection: uniform
  max_frames: 150
```

```yaml
# Undistort a wide-angle lens (existing scene: --stages preproc --overwrite)
preproc:
  undistort: true
```

| Key | Type | Default | Description |
|---|---|---|---|
| `preproc.frame_selection` | str | `fps` | `fps`, `uniform` or `optical_flow` |
| `preproc.fps` | float | `2.0` | `fps` only: samples per second |
| `preproc.min_frames` | int\|null | `null` | `fps` only: floor on the count |
| `preproc.max_frames` | int\|null | `300` | Count for `uniform`, ceiling otherwise; `null` = no ceiling (not for `uniform`) |
| `preproc.on_empty_slot` | str | `rescue` | `fps` only: a slot with no eligible frame keeps its sharpest (`rescue`) or is skipped (`drop`) |
| `preproc.quality.sharpness_k` | float | `2.0` | MAD z-score cut on `log(laplacian)`; larger keeps more |
| `preproc.quality.max_clipped_frac` | float | `0.25` | Ceiling on clipped-pixel fraction; larger keeps more |
| `preproc.undistort` | bool | `false` | Self-calibrate one OPENCV camera (pycolmap) and rewrite `images/` undistorted; frame size changes |
| `preproc.n_workers` | int | `16` | Parallel decode ranges; never changes the selection. Set `1` during GPU runs |

---

## `pointcloud`

We pick a method (`feedforward` or `sfm`) and a backend. Each backend's kwargs go under its own name and apply only when it is selected; the blocks are documented in [pointcloud.rst](../docs/source/api/pointcloud.rst#backends).

```yaml
# Another feedforward model
pointcloud:
  backend: mapanything
```

| Key | Type | Default | Description |
|---|---|---|---|
| `pointcloud.method` | str | `feedforward` | `feedforward` or `sfm` |
| `pointcloud.backend` | str | `vggt_omega` | feedforward: `vggt_omega`, `vggtx`, `mapanything`, `loger`; sfm: `instantsfm`, `colmap`, `hloc` |
| `pointcloud.<backend>` | dict | `{}` | Kwargs for the selected backend only; unknown keys are a `TypeError` |
| `pointcloud.max_points` | int | `500000` | Final point cap after cleaning; lower it on small machines |
| `pointcloud.<backend>.min_views` | int | `0` | Feedforward only: cross-view depth filter; `0` = off |
| `pointcloud.<backend>.mv_rel_thresh` | float | `0.01` | Feedforward only: cross-view agreement tolerance, as a fraction of depth |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points (sfm also drops them from the COLMAP export) |

### SfM backends

```yaml
# Classic SfM (refine is skipped; loop closure is feedforward only)
pointcloud:
  method: sfm
  backend: colmap
```

| Key | Type | Default | Description |
|---|---|---|---|
| `pointcloud.colmap.pairing` | str | `sequential+retrieval` | `sequential`, `retrieval`, `sequential+retrieval` or `exhaustive` |
| `pointcloud.colmap.overlap` | int | `10` | `sequential*`: pair each frame with the next N |
| `pointcloud.colmap.num_retrieved` | int | `20` | `*retrieval`: vocab-tree neighbors per image |
| `pointcloud.colmap.num_threads` | int | `8` | SIFT + mapper threads (colmap's default OOMs) |
| `pointcloud.colmap.min_registered_frac` | float | `0.5` | Below this share registered: `RuntimeError`; above: keep the registered frames |
| `pointcloud.hloc.pairing` | str | `sequential+retrieval` | As `colmap.pairing` |
| `pointcloud.hloc.overlap` | int | `10` | As `colmap.overlap` |
| `pointcloud.hloc.num_retrieved` | int | `20` | `*retrieval`: global-descriptor neighbors per image |
| `pointcloud.hloc.retrieval_conf` | str | `netvlad` | hloc global-descriptor conf |
| `pointcloud.hloc.feature_conf` | str | `superpoint_max` | hloc local-feature conf |
| `pointcloud.hloc.matcher_conf` | str | `superpoint+lightglue` | hloc matcher conf |
| `pointcloud.hloc.num_threads` | int | `8` | Mapper threads |
| `pointcloud.hloc.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.instantsfm.pairing` | str | `exhaustive` | As `colmap.pairing` |
| `pointcloud.instantsfm.overlap` | int | `10` | As `colmap.overlap` |
| `pointcloud.instantsfm.num_retrieved` | int | `20` | As `colmap.num_retrieved` |
| `pointcloud.instantsfm.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.instantsfm.num_threads` | int | `8` | CPU SIFT threads |
| `pointcloud.instantsfm.random_seed` | int\|null | `null` | Seed InstantSfM's random init; `null` = upstream (unseeded) |

### `bundle_adjustment`

The `refine` stage refines the feedforward poses against feature tracks, rewrites them in place (COLMAP, `sparse_pc.ply`, `pointcloud.zarr`), then re-cleans the cloud, so the point count can change.

- after `--stages refine`, re-run `semantics` with `--overwrite` (its rows index the old points), and `mesh` / `localize` / the report if they matter
- `<backend>/colmap/refine.json` holds the last BA config and loss history

```yaml
# Reprojection + depth only, no photometric term
pointcloud:
  bundle_adjustment:
    use_photometric: false
```

| Key | Type | Default | Description |
|---|---|---|---|
| `pointcloud.bundle_adjustment.enabled` | bool | `true` | Run `refine`; skipped for `method: sfm` |
| `pointcloud.bundle_adjustment.strategy` | str | `global` | With loop closure only: `global`, `window` (inside each LC window) or `window+global` |
| `pointcloud.bundle_adjustment.track_source` | str | `xfeat` | `xfeat` or `loma` matcher tracks on full-res frames, or `vggsfm` (set `track_kwargs: {}`) |
| `pointcloud.bundle_adjustment.track_kwargs` | dict | `{pairing: exhaustive, seed_fraction: 1.0}` | Passed to `extract_tracks`: `pairing` (`exhaustive` / `sequential+retrieval`), `seed_fraction`; with `sequential+retrieval` also `window`, `retrieval`, `retrieval_k` |
| `pointcloud.bundle_adjustment.vis_thresh` | float | `0.2` | Min track score per observation |
| `pointcloud.bundle_adjustment.max_reproj_error` | float\|null | `4.0` | Pre-solve reprojection gate in px; `null` = off |
| `pointcloud.bundle_adjustment.min_inliers_per_frame` | int | `64` | Frames below this sit out the solve |
| `pointcloud.bundle_adjustment.solver` | str | `schur` | `schur` (camera-only, less GPU) or `lm` (joint); `schur` + `refine_focal` + `shared_camera` is a `ValueError` |
| `pointcloud.bundle_adjustment.dtype` | str | `float64` | `float32` or `float64` |
| `pointcloud.bundle_adjustment.lm_steps` | int | `40` | Max LM steps (photometric runs its own fixed schedule) |
| `pointcloud.bundle_adjustment.lm_tol` | float | `1.0e-4` | Relative loss drop that counts as stalled |
| `pointcloud.bundle_adjustment.lm_patience` | int | `2` | Stalled steps that end the solve |
| `pointcloud.bundle_adjustment.increment_size` | int | `0` | `0` = one global solve; `N` = frames per incremental solve |
| `pointcloud.bundle_adjustment.shared_camera` | bool | `true` | One focal per scene; `false` = per frame |
| `pointcloud.bundle_adjustment.refine_focal` | bool | `false` | Solve for focal |
| `pointcloud.bundle_adjustment.use_depth` | bool | `true` | Track camera z against feedforward depth |
| `pointcloud.bundle_adjustment.depth_sigma` | float | `0.01` | Relative depth error weighted like 1 px |
| `pointcloud.bundle_adjustment.use_photometric` | bool | `true` | Brightness matching between overlapping frames; `ValueError` with `increment_size > 0` |
| `pointcloud.bundle_adjustment.device` | str\|null | `null` | CUDA device; `null` = auto |

### `loop_closure`

Loop closure splits a long video into windows, reconstructs each, and joins them through a pose graph with loop edges between revisited places. `true` turns it on with the defaults below; a dict sets keys.

```yaml
# Long video: windows of 50 frames, no loop retrieval
pointcloud:
  loop_closure:
    submap_size: 50
    lc_retrieval_threshold: 0.0
```

| Key | Type | Default | Description |
|---|---|---|---|
| `pointcloud.loop_closure` | bool\|dict | `false` | Run loop closure (`ValueError` with `method: sfm` or `backend: loger`) |
| `pointcloud.loop_closure.retrieval` | str | `dino-salad` | `dino-salad` or `megaloc` (`megaloc` needs its own threshold) |
| `pointcloud.loop_closure.submap_size` | int | `20` | Window stride; shorter scenes skip windowing |
| `pointcloud.loop_closure.submap_overlap` | int | `1` | Frames shared by consecutive windows |
| `pointcloud.loop_closure.lc_retrieval_threshold` | float | `0.95` | Max descriptor L2 in [0, 2] (tuned for dino-salad); `<= 0` skips retrieval |
| `pointcloud.loop_closure.max_loops_per_submap` | int | `5` | Loop candidates per submap, before NMS |
| `pointcloud.loop_closure.verify_match_ratio` | float\|null | `null` | Loop verification threshold; `null` = the creator's default |
| `pointcloud.loop_closure.nms_frame_distance` | int | `25` | Suppression radius in frames |
| `pointcloud.loop_closure.min_submap_gap` | int | `1` | Newest submaps left out of retrieval |
| `pointcloud.loop_closure.conf_percentile` | float | `25.0` | Confidence percentile gating scale points |
| `pointcloud.loop_closure.loop_edge_timing` | str | `deferred` | `deferred` (after the window loop) or `live`; `viz.enabled` forces `live` |
| `pointcloud.loop_closure.viz_max_points` | int | `50000` | Viewer point cap per submap |
| `pointcloud.viz.enabled` | bool | `false` | Live viser viewer during loop closure |
| `pointcloud.viz.port` | int | `8080` | Viewer port |

---

## `mesh`

We fuse depth into a TSDF volume, clean and smooth the mesh, then texture it. The voxel size follows the scene's depth, so the defaults hold across scales; the reasoning behind each knob is in [mesh.rst](../docs/source/api/mesh.rst).

```yaml
# Finer surface
mesh:
  voxel_depth_px: 2.0
```

```yaml
# Indoor scene or object: no ground hull
mesh:
  use_convex_hull: false
```

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh.enabled` | bool | `true` | Fuse a TSDF mesh into `<backend>/mesh.ply` |
| `mesh.source` | str | `feedforward` | Fuse `pointcloud.zarr` depth, or `splats` rendered depth (needs the splats stage) |
| `mesh.voxel_depth_px` | float | `4.0` | Voxel edge in depth pixels at the reference depth; smaller is finer |
| `mesh.voxel_ref_percentile` | float | `50` | Depth percentile the voxel is sized at |
| `mesh.sdf_trunc_mult` | float | `6.0` | Truncation band in voxels; must be `>= 1.0` |
| `mesh.depth_trunc_percentile` | float | `99` | Zero depth beyond this percentile; required, in (0, 99] |
| `mesh.conf_percentile` | float\|null | `10` | Drop depth below this confidence percentile; `null` = off |
| `mesh.mask_sky` | bool | `true` | Zero sky depth before fusing |
| `mesh.max_faces` | int\|null | `1500000` | Face budget before texturing; `null` = no cap |
| `mesh.smooth_iterations` | int | `30` | Volume-keeping smoothing passes; `0` = off |
| `mesh.use_convex_hull` | bool | `true` | Patch the ground out to a convex hull; outdoor scenes only |
| `mesh.texture` | bool | `true` | UV-unwrap and texture into `<backend>/texture/`; needs a GPU, set `false` without one |

---

## `semantics`

We extract 2D features per frame, compress them with an autoencoder, and lift them onto the points (and the mesh vertices when a mesh exists). Extractors run in order and each writes its own store, so they coexist. Details in [semantics.rst](../docs/source/api/semantics.rst).

```yaml
# Talk2DINO only, uncompressed
semantics:
  extractors: [talk2dino]
  n_components: null
```

| Key | Type | Default | Description |
|---|---|---|---|
| `semantics.enabled` | bool | `true` | Extract and lift features |
| `semantics.extractors` | list[str] | `[talk2dino, ocr_lens]` | `talk2dino`, `dinov2`, `maskclip`, `ocr_lens` |
| `semantics.n_components` | int\|null | `128` | Autoencoder latent dim; `null` = no compression |
| `semantics.target_cosine` | float\|null | `null` | Stop training at this mean reconstruction cosine; `null` = run `max_epochs` |
| `semantics.max_epochs` | int | `30` | Autoencoder epochs |

---

## `splats`

`3dgs` keeps thin structure and trains faster; `2dgs` (surface-aligned) gives better surfaces.

```yaml
# 2DGS as the mesh source
splats:
  enabled: true
  primitive: 2dgs
mesh:
  source: splats
```

| Key | Type | Default | Description |
|---|---|---|---|
| `splats.enabled` | bool | `false` | Train Gaussian splats |
| `splats.primitive` | str | `3dgs` | `3dgs` or `2dgs` |
| `splats.representation` | str | `vanilla` | `vanilla` or `scaffold` (set `losses.opacity_reg` weight 0) |
| `splats.scaffold` | dict | absent | Scaffold overrides: `n_offsets`, `feat_dim`, `voxel_multiplier`, `update_from`, `update_until`, `refine_every`, `grad_threshold`, `min_opacity`, `update_init_factor`, `success_threshold`, `appearance_dim`, `mlp_bf16` |
| `splats.max_steps` | int | `30000` | Training iterations |
| `splats.pose_opt` | bool | `true` | Refine camera poses jointly |
| `splats.appearance_opt` | bool | `false` | Per-image affine color (train views only) |
| `splats.sh_degree` | int | `3` | Max spherical-harmonics degree |
| `splats.sh_degree_interval` | int | `1000` | Steps between SH-degree increments |
| `splats.init_opacity` | float | `0.1` | Initial opacity |
| `splats.cap_max` | int | `1000000` | `3dgs` + `vanilla` only: Gaussian budget |
| `splats.grow_grad2d` | float | `2.0e-4` | `2dgs` + `vanilla` only: densification threshold |
| `splats.num_downscales` | int | `2` | Coarse-to-fine: start at `1/2^n` resolution; `0` = off |
| `splats.resolution_schedule` | int | `3000` | Steps per resolution doubling |
| `splats.normalize_scene` | bool | `false` | Train in a normalized frame; outputs map back to world units |
| `splats.log_every` | int | `500` | Steps between loss logs |

### Learning rates

| Key | Type | Default | Description |
|---|---|---|---|
| `splats.means_lr` | float | `1.6e-4` | Means (× scene scale) |
| `splats.scales_lr` | float | `5.0e-3` | Scales |
| `splats.quats_lr` | float | `1.0e-3` | Rotations |
| `splats.opacities_lr` | float | `5.0e-2` | Opacities |
| `splats.sh0_lr` | float | `2.5e-3` | DC color |
| `splats.shN_lr` | float | `1.25e-4` | Higher-order SH |
| `splats.pose_lr` | float | `1.0e-5` | Poses (× scene scale) |
| `splats.appearance_lr` | float | `1.0e-3` | Appearance |

### `losses`

Without a `losses` block, each primitive gets its preset (`default_losses` in `splats/losses.py`): `depth`, `normal_consistency`, `appearance_reg`, plus `opacity_reg` + `scale_reg` for `3dgs` or `distortion` for `2dgs`. The preset starts `normal_consistency` at step 7000 and `distortion` at 3000. A `losses` block replaces the preset: list every loss you want, each with a `weight`.

```yaml
# 3DGS with a stronger depth loss (list every loss you want)
splats:
  enabled: true
  losses:
    depth: {weight: 0.05}
    normal_consistency: {weight: 0.05, start: 7000}
    appearance_reg: {weight: 1.0e-3}
    opacity_reg: {weight: 0.01}
    scale_reg: {weight: 0.01}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `splats.losses.<name>.weight` | float | preset | `depth`, `normal_consistency`, `distortion`, `opacity_reg`, `scale_reg`, `appearance_reg` |
| `splats.losses.<name>.start` | int | `0` | Step the loss switches on |
| `splats.losses.appearance_reg.weight` | float | `1.0e-3` | Regularizer on `appearance_opt` |
| `splats.losses.normal_consistency.depth_ratio` | float | `0.0` | `2dgs` only: median-depth normal blend, in [0, 1] |

---

## `localization`

The `localize` stage caches per-frame features so query images can be located in the scene later. The retrieval model is stored with the cache, so queries always match it.

```yaml
# LoMa matcher instead of XFeat
localization:
  matcher: loma
```

| Key | Type | Default | Description |
|---|---|---|---|
| `localization.enabled` | bool | `true` | Build the localization database |
| `localization.matcher` | str | `xfeat` | A batch-capable vismatch model, e.g. `xfeat` or `loma` |
| `localization.retrieval` | str | `megaloc` | `megaloc` or `dino-salad` |

---

## `reconstruction_quality_report`

`reconstruction_quality_report.json` scores the reconstruction without ground truth. It is report-only: nothing in it feeds back into the pipeline. Tables and columns are in [geometry.rst](../docs/source/api/geometry.rst).

| Key | Type | Default | Description |
|---|---|---|---|
| `reconstruction_quality_report.min_pair_overlap` | float | `0.0` | Skip pairs whose frustum overlap is below this; `0` = every pair |

---

## Remote scenes

`reconstruct remote` takes the same options as `local` (except `--keep-viewer`), plus `--all` and `--keep-local`. Commands are in the [main README](../README.md#remote-video-processing).

- scene ids: any flat path-safe name (`YYYY_MM_DD-PARENTFOLDER-VIDEONAME` by convention)
- a failed scene keeps its local dir and the batch goes on; the push only copies, so nothing remote is deleted
- after any failure we re-probe rclone; if it is unreachable we stop and report the rest `SKIPPED`
- credentials come from the `collab-data` rclone remote only

| Exit | Meaning |
|---|---|
| 0 | every scene succeeded |
| 1 | one or more scenes failed; the rest still ran |
| 2 | nothing to do (`--all` on an empty bucket), or a usage error |
| 3 | rclone unreachable (bucket listing or mid-batch); later scenes never ran (outranks 1) |

### Changing options on processed scenes

Naming only *leaf* stages in `--stages` (`refine`, `mesh`, `semantics`, `splats`, `localize`, `reconstruction_quality_report`) pulls the scene back from `environments-processed` instead of rebuilding it.

- config = the pulled `<backend>/run_config.yaml`; the re-run stage's section (e.g. `mesh`) resets to `base.yaml`, then `--config` / `--set` apply on top
- any set with `preproc` or `pointcloud` rebuilds from the curated video
- `--all` lists `environments-processed` for a leaf re-run, `environments-curated` otherwise

| Situation | Outcome |
|---|---|
| scene has no processed outputs | `FileNotFoundError` |
| no `pointcloud.backend` override and not exactly one `<backend>/run_config.yaml` | `ValueError` listing the backends |
| named backend has no `run_config.yaml` | `FileNotFoundError` |
| named leaf stage's output exists | `ValueError`: pass `--overwrite` |

### What we push

Everything in the scene except videos, temporary full-width features (`semantics/*_features.zarr`) and the SfM match databases (`PUSH_EXCLUDES` in `remote.py`).
