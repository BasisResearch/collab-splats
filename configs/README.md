# Reconstruction Pipeline Configs

All pipelines start from default template `base.yaml` — this file holds the defaults for each stage. If a separate file is
specified (e.g., `MYFILE.yaml`), the file only overrides what differs from `base.yaml`. Running the pipeline and its outputs are covered in the [main README](https://github.com/BasisResearch/collab-splats#quickstart).

- [Overriding defaults](#overriding-defaults)
- [Stages](#stages)
- Options by stage: [`preproc`](#preproc) · [`pointcloud`](#pointcloud) · [`mesh`](#mesh) · [`semantics`](#semantics) · [`splats`](#splats) · [`localization`](#localization) · [`reconstruction_quality_report`](#reconstruction-quality-report)

---

## Overriding defaults

Three layers, applied in order (later wins):

1. `base.yaml`
2. `--config MYFILE.yaml`: merged key by key over `base.yaml`
3. `--set key=value`: one dotted key, repeatable

```yaml
# configs/my_run.yaml: indoor scene, denser frames, no semantics
preproc:
  fps: 3.0
mesh:
  use_convex_hull: false
semantics:
  enabled: false
```

- every backend folder saves its merged config as `<backend>/run_config.yaml`; pass it to `--config` to reproduce a run
- after changing a config, re-run only the affected stages with `--stages <stage> --overwrite`

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
- stages run in table order; `--stages` runs a stage even when its flag is off
- `enabled` stages skip when their output exists, so an interrupted run resumes; a stage named in `--stages` needs `--overwrite`
- `semantics` re-lifts onto a new mesh from its cached 2D features

---

## `preproc`

Preproc measures every frame once (`video_quality_report.json`, reused; delete it to re-measure), then picks keyframes that pass `preproc.quality`.

- `fps` (default): fixed time between frames, so the baseline between frames stays the same for any video length
- `uniform`: exactly `max_frames`, evenly spaced
- `optical_flow`: spaced by motion, up to `max_frames`
- `max_frames: 300` is VGGT-Omega's GPU limit; it caps `fps` past ~2.5 min at `fps: 2.0`

```yaml
# Fixed 150 frames, evenly spaced
preproc:
  frame_selection: uniform
  max_frames: 150
```

| Key | Type | Default | Description |
|---|---|---|---|
| `preproc.frame_selection` | str | `fps` | `fps`, `uniform` or `optical_flow` |
| `preproc.fps` | float | `2.0` | `fps` only: samples per second |
| `preproc.max_frames` | int\|null | `300` | Count for `uniform`, ceiling otherwise; `null` = no ceiling (not for `uniform`) |
| `preproc.undistort` | bool | `false` | Self-calibrate one OPENCV camera (pycolmap) and rewrite `images/` undistorted; frame size changes |
| `preproc.n_workers` | int | `16` | Parallel decode ranges; never changes the selection. Set `1` during GPU runs |

<details>
<summary><code>fps</code> sampling</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `preproc.min_frames` | int\|null | `null` | `fps` only: floor on the count |
| `preproc.on_empty_slot` | str | `rescue` | `fps` only: a slot with no eligible frame keeps its sharpest (`rescue`) or is skipped (`drop`) |

</details>

<details>
<summary><code>preproc.quality</code>: keyframe gate</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `preproc.quality.sharpness_k` | float | `2.0` | MAD z-score cut on `log(laplacian)`; larger keeps more |
| `preproc.quality.max_clipped_frac` | float | `0.25` | Ceiling on clipped-pixel fraction; larger keeps more |

</details>

---

## `pointcloud`

We pick a method (`feedforward` or `sfm`) and a backend. Each backend's kwargs go under its own name and apply only when it is selected; backend details are in the [pointcloud API docs](https://basisresearch.github.io/collab-splats/api/pointcloud.html).

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

SfM backends, refinement and loop closure each have their own options:

<details>
<summary><code>pointcloud.colmap</code>: classic incremental SfM</summary>

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

</details>

<details>
<summary><code>pointcloud.hloc</code>: incremental SfM with learned features</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `pointcloud.hloc.pairing` | str | `sequential+retrieval` | As `colmap.pairing` |
| `pointcloud.hloc.overlap` | int | `10` | As `colmap.overlap` |
| `pointcloud.hloc.num_retrieved` | int | `20` | `*retrieval`: global-descriptor neighbors per image |
| `pointcloud.hloc.retrieval_conf` | str | `netvlad` | hloc global-descriptor conf |
| `pointcloud.hloc.feature_conf` | str | `superpoint_max` | hloc local-feature conf |
| `pointcloud.hloc.matcher_conf` | str | `superpoint+lightglue` | hloc matcher conf |
| `pointcloud.hloc.num_threads` | int | `8` | Mapper threads |
| `pointcloud.hloc.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |

</details>

<details>
<summary><code>pointcloud.instantsfm</code>: global SfM</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `pointcloud.instantsfm.pairing` | str | `exhaustive` | As `colmap.pairing` |
| `pointcloud.instantsfm.overlap` | int | `10` | As `colmap.overlap` |
| `pointcloud.instantsfm.num_retrieved` | int | `20` | As `colmap.num_retrieved` |
| `pointcloud.instantsfm.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.instantsfm.num_threads` | int | `8` | CPU SIFT threads |
| `pointcloud.instantsfm.random_seed` | int\|null | `null` | Seed InstantSfM's random init; `null` = upstream (unseeded) |

</details>

<details>
<summary><code>pointcloud.bundle_adjustment</code>: the <code>refine</code> stage</summary>

The `refine` stage refines feedforward poses against feature tracks and rewrites the pointcloud in place, so the point count can change. After `--stages refine`, re-run `semantics` with `--overwrite`.

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

</details>

<details>
<summary><code>pointcloud.loop_closure</code>: long videos</summary>

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

</details>

---

## `mesh`

We fuse depth into a TSDF volume, clean and smooth the mesh, then texture it. The voxel size follows the scene's depth, so the defaults hold across scales; the reasoning behind each knob is in the [mesh API docs](https://basisresearch.github.io/collab-splats/api/mesh.html).

```yaml
# Indoor scene or object: no ground hull
mesh:
  use_convex_hull: false
```

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh.enabled` | bool | `true` | Fuse a TSDF mesh into `<backend>/mesh.ply` |
| `mesh.source` | str | `feedforward` | Fuse `pointcloud.zarr` depth, or `splats` rendered depth (needs the splats stage) |
| `mesh.texture` | bool | `true` | UV-unwrap and texture into `<backend>/texture/`; needs a GPU, set `false` without one |

<details>
<summary>Depth filtering</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh.depth_trunc_percentile` | float | `99` | Zero depth beyond this percentile; required, in (0, 99] |
| `mesh.conf_percentile` | float\|null | `10` | Drop depth below this confidence percentile; `null` = off |
| `mesh.mask_sky` | bool | `true` | Zero sky depth before fusing |

</details>

<details>
<summary>TSDF fusion</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh.voxel_depth_px` | float | `4.0` | Voxel edge in depth pixels at the reference depth; smaller is finer |
| `mesh.voxel_ref_percentile` | float | `50` | Depth percentile the voxel is sized at |
| `mesh.sdf_trunc_mult` | float | `6.0` | Truncation band in voxels; must be `>= 1.0` |

</details>

<details>
<summary>Cleanup</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `mesh.use_convex_hull` | bool | `true` | Patch the ground out to a convex hull; outdoor scenes only |
| `mesh.smooth_iterations` | int | `30` | Volume-keeping smoothing passes; `0` = off |
| `mesh.max_faces` | int\|null | `1500000` | Face budget before texturing; `null` = no cap |

</details>

---

## `semantics`

We extract 2D features per frame, compress them with an autoencoder, and lift them onto the points (and the mesh vertices when a mesh exists). Extractors run in order and each writes its own store, so they coexist. Details in the [semantics API docs](https://basisresearch.github.io/collab-splats/api/semantics.html).

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

`3dgs` keeps thin structure and trains faster; `2dgs` (surface-aligned) gives better surfaces. For the best surfaces use `scaffold` + `2dgs` as the mesh source.

```yaml
# Best surfaces: scaffold 2DGS as the mesh source
splats:
  enabled: true
  representation: scaffold
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
| `splats.normalize_scene` | bool | `false` | Train in a normalized frame; outputs map back to world units |

<details>
<summary>Schedule and densification</summary>

| Key | Type | Default | Description |
|---|---|---|---|
| `splats.init_opacity` | float | `0.1` | Initial opacity |
| `splats.sh_degree` | int | `3` | Max spherical-harmonics degree |
| `splats.sh_degree_interval` | int | `1000` | Steps between SH-degree increments |
| `splats.cap_max` | int | `1000000` | `3dgs` + `vanilla` only: Gaussian budget |
| `splats.grow_grad2d` | float | `2.0e-4` | `2dgs` + `vanilla` only: densification threshold |
| `splats.num_downscales` | int | `2` | Coarse-to-fine: start at `1/2^n` resolution; `0` = off |
| `splats.resolution_schedule` | int | `3000` | Steps per resolution doubling |
| `splats.log_every` | int | `500` | Steps between loss logs |

</details>

<details>
<summary>Learning rates</summary>

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

</details>

<details>
<summary><code>splats.losses</code></summary>

- no `losses` block: each primitive gets its preset: `depth`, `normal_consistency` (from step 7000), `appearance_reg`, plus `opacity_reg` + `scale_reg` (`3dgs`) or `distortion` (`2dgs`, from step 3000)
- a `losses` block replaces the preset, so list every loss you want, each with a `weight`

| Key | Type | Default | Description |
|---|---|---|---|
| `splats.losses.<name>.weight` | float | preset | `depth`, `normal_consistency`, `distortion`, `opacity_reg`, `scale_reg`, `appearance_reg` |
| `splats.losses.<name>.start` | int | `0` | Step the loss switches on |
| `splats.losses.appearance_reg.weight` | float | `1.0e-3` | Regularizer on `appearance_opt` |
| `splats.losses.normal_consistency.depth_ratio` | float | `0.0` | `2dgs` only: median-depth normal blend, in [0, 1] |

</details>

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

## Reconstruction quality report

`reconstruction_quality_report.json` scores the reconstruction without ground truth. It is report-only: nothing in it feeds back into the pipeline. Tables and columns are in the [geometry API docs](https://basisresearch.github.io/collab-splats/api/geometry.html#report-columns).

| Key | Type | Default | Description |
|---|---|---|---|
| `reconstruction_quality_report.min_pair_overlap` | float | `0.0` | Skip pairs whose frustum overlap is below this; `0` = every pair |
