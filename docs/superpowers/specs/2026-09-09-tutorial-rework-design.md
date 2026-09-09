# Tutorial notebook rework — self-contained pages on the clean/final API

**Status:** design approved, plan pending
**Branch:** `clean/final`
**Date:** 2026-09-09

## Problem

The 14 tutorial notebooks are a *chain*, not a set. `tutorial_config.py` hands every
notebook a shared `data/outputs/` directory, and each page asserts its way into the
previous page's artifacts:

```python
assert RECON.exists(), f"missing {RECON} — run 02_pointcloud/feedforward_methods.ipynb first"
```

Three consequences, all of them now biting:

- **No page runs on its own.** A reader opening 05 gets an `AssertionError`.
- **Stale artifacts are indistinguishable from fresh ones.** `feedforward_methods`
  carries an explicit "cache built from N frames; delete the zarr to re-run" branch —
  a cache-invalidation problem the tutorial should never have had.
- **Drift is invisible.** Because pages read artifacts instead of building them,
  several have rotted against `clean/final` without anything failing until run:

  | Notebook | Break |
  |---|---|
  | `01_preprocessing` | `preproc.sample_frames`, `preproc.score_frames` — neither exists |
  | `02/slam_loop_closure` | `geometry.loop_closure.closure` — module gone (now `graph.py`) |
  | `02/feedforward_mesh` | reads `OUTPUT_DIR/mapanything/reconstruction.zarr`, a layout nothing writes |
  | `03/train_splats` | `train()` called without the required `image_ids=` kwarg |
  | `03/train_splats` | imports `evals.scripts.eval_splats` behind a `sys.path.insert` |
  | `evals/ground_truth_evals` | runs `evals/eval_gt.py` — path does not exist |
  | `02/colmap_sfm` | 4-cell empty stub, `nerfstudio` kernel, predates `pointcloud/sfm/` |

## Goals

1. Every notebook runs **top to bottom from a clean checkout**, building its own inputs.
2. Every notebook writes **only** into its own temporary directory.
3. The set is the **minimum** that explains the package — one clear subject per page.
4. Every call matches the `clean/final` API.

## Non-goals

- Changing any `collab_splats` API. This is a docs change; the one exception is
  `notebook_utils.py`, which is tutorial-owned.
- Teaching every backend. VGGT-Omega is the feedforward backend shown; the others get
  a sentence.
- Keeping the evaluation notebook. Ground-truth evals are CLI work over a gitignored
  results tree and cannot be made self-contained.

## Notebook set

Nine pages, renumbered `01`–`06`:

```
docs/source/tutorials/
  tutorial_config.py                      # committed inputs only
  notebook_utils.py                       # backend selection + bootstrap helpers
  index.rst
  01_preprocessing/keyframe_extraction.ipynb
  02_pointcloud/reconstruction.ipynb
  02_pointcloud/refinement.ipynb
  03_splats/train_splats.ipynb
  04_semantics/feature_extraction.ipynb
  04_semantics/segmentation.ipynb
  04_semantics/lifting_and_query.ipynb
  05_mesh/tsdf_mesh.ipynb
  06_localization/localization.ipynb
```

### Disposition of the current 14

| Current | Fate |
|---|---|
| `01_preprocessing/keyframe_extraction` | rewritten in place |
| `02_pointcloud/feedforward_methods` | → `02_pointcloud/reconstruction` |
| `02_pointcloud/colmap_sfm` | **deleted** (stub; SfM now lives in `reconstruction`) |
| `02_pointcloud/bundle_adjustment` | → `02_pointcloud/refinement` §BA |
| `02_pointcloud/slam_loop_closure` | → `02_pointcloud/refinement` §LC |
| `02_pointcloud/feedforward_mesh` | **deleted** (duplicate of the mesh page's §1) |
| `03_splats/train_splats` | rewritten in place |
| `04_semantics/feature_extraction` | rewritten; absorbs the comparison page |
| `04_semantics/maskclip_vs_talk2dino` | **deleted** (already §4 of `feature_extraction`) |
| `04_semantics/segmentation` | rewritten in place |
| `05_lifting/semantic_lifting` | → `04_semantics/lifting_and_query` |
| `06_mesh/splats_mesh` | → `05_mesh/tsdf_mesh` |
| `07_localization/localization` | → `06_localization/localization` |
| `evals/ground_truth_evals` | **deleted** |

Directories `05_lifting/`, `06_mesh/`, `07_localization/`, `evals/` are removed;
`index.rst` is rewritten to the nine pages above.

## Architecture

### Isolation contract

Every notebook owns one temporary directory and writes nothing outside it.

```python
WORK = work_dir("reconstruction")     # mkdtemp, printed, left for inspection
```

`mkdtemp` honours `$TMPDIR`. The path is printed so a reader can open the artifacts
afterwards; nothing deletes it, and the OS reaps it. Inside, every notebook uses the
same layout the pipeline writes, so what a reader learns transfers:

```
$WORK/
  images/                    frame_NNNNNN.png + frames.json
  video_quality.json
  pointcloud.zarr
  splats/ckpt.pt
  mesh/
```

`data/outputs/` disappears entirely. `tutorial_config.py` keeps only the committed
inputs — `REPO_ROOT`, `VIDEO_PATH`, `QUERY_IMAGE`, and the missing-video check. Its
`OUTPUT_DIR` / `IMAGES_DIR` / `RECON` / `TUTORIAL_CACHE` / `MAX_FRAMES` constants are
deleted; frame counts vary per page and now live in each notebook's `§0`.

### Bootstrap helpers

Pages 02–06 need inputs that page 01 used to leave behind. Rather than repeat the
build in six notebooks, `notebook_utils.py` grows four functions. Each is a thin
composition of public API, prints what it built, and returns a path:

```python
def work_dir(name: str) -> Path
def bootstrap_keyframes(work: Path, *, n_frames: int) -> Path        # -> images/
def bootstrap_reconstruction(work: Path, images_dir: Path) -> Path   # -> pointcloud.zarr
def bootstrap_splats(work: Path, zarr: Path, images_dir: Path, *, max_steps: int) -> Path
```

The rule for what belongs in a helper vs. the notebook body: **a notebook writes out
in full the API it is teaching, and calls a helper for everything upstream of it.**
The reconstruction page calls `bootstrap_keyframes` and then writes the creator calls
out longhand; the mesh page calls all three helpers and writes out only the fusion.

`bootstrap_splats` carries the trainer glue, mirroring `Reconstructor.splats()`
(`wrapper/reconstructor.py:1457-1561`) rather than the evals convenience wrapper:
native frames from `images/` reordered onto `result.image_paths`, `result.extrinsics`
and `result.intrinsics` used as-is (a `PointcloudResult`'s K is already native-res —
no rescale), depth targets read from the zarr and masked with `confidence_mask`, then
`train(..., image_ids=frame_indices)`.

That glue is deliberately written twice: as `bootstrap_splats`, and longhand in nb03,
which is the page that *teaches* it. The rule above makes this the expected outcome for
every helper whose subject is also a page — the helper is the upstream convenience, the
notebook is the lesson. `bootstrap_keyframes` / `bootstrap_reconstruction` duplicate 01
and 02 the same way. Only `bootstrap_splats` has a single caller (05); it earns its place
by keeping the mesh page about fusion rather than about training.

> `evals/scripts/eval_splats.py::inputs_from_pointcloud_zarr` is **not** the production
> path — verified against `Reconstructor.splats()`, which never calls it. It exists for
> evals, where inputs come from a zarr with no sibling `images/`. The tutorial follows
> the pipeline; the `sys.path.insert(REPO_ROOT)` hack in the current nb03 goes away.

### Cost profile

Each page pays its own compute. `§0` of every notebook opens with the knobs, and a
comment giving the production value:

```python
MAX_FRAMES = 16      # pipeline runs 100s of frames; 16 keeps this page a few minutes
MAX_STEPS  = 1000    # base.yaml splats.max_steps is 30000
```

Two pages need more than the default, for reasons that are properties of the code:

- **`refinement`: `MAX_FRAMES = 24`, `submap_size = 8`.** `LoopClosureConfig.submap_size`
  defaults to 20 and `LoopClosure._enough_frames()` (`wrapper.py:207-210`) falls back to
  plain inference below it — *silently*. At 16 frames the page would render a loop-closure
  section that never ran loop closure. 24 frames at `submap_size=8` gives three submaps.
- **`05_mesh`: pays a splat train**, because the page compares both fusion sources.

## Page-by-page

### 01 · Preprocessing — `keyframe_extraction`

Subject: measure the video, then select from it.

1. `get_video_info`, preview grid.
2. **One QA pass over the whole clip:** `compute_video_quality(VIDEO_PATH,
   output_path=WORK/"video_quality.json")`. Explicitly flagged as the expensive cell.
3. Photometry and motion from the report — `plot_photometric`, `plot_motion`,
   `plot_frame_extremes` (blur and exposure extremes), the correlation plots.
4. `filter_frame_quality(report)` — the mask, and what `sharpness_k` /
   `max_clipped_frac` do. The report carries measurements; **this** is where a verdict
   is made.
5. All three samplers over the same report: `sample_uniform` (count is the contract),
   `sample_fps` (spacing is the contract, sharpest frame per slot), `sample_optical_flow`
   (motion + coverage). `plot_selection` overlays them on one timeline.
6. `frames.write_frames(WORK/"images", ...)` — the canonical store, `frame_NNNNNN.png`
   plus `frames.json`.

Replaces the dead `sample_frames` / `score_frames` imports and the JSON score cache
with the real two-step `qa` → `sampling` API.

### 02 · Pointcloud — `reconstruction`

Subject: keyframes in, sparse reconstruction out, by both routes.

- `bootstrap_keyframes`.
- **Feedforward:** `make_creator("vggt_omega")`, `creator.run(images_dir, device)`,
  the `FeedforwardResult` fields, `clean_pointcloud`, PyVista cloud + frustums,
  `save_zarr` to `WORK/pointcloud.zarr`. Prose names the other backends
  (`vggtx`, `mapanything`, `loger`) and says they are the same call.
- **SfM:** `generate_vda_depth(frames, WORK, names)` → `depth_vda/`, then
  `InstantSfMCreator().reconstruct(WORK)`, then `result_from_reconstruction` to get a
  COLMAP-scale `FeedforwardResult`. Prose covers why depth priors are the default
  (`use_depths=True` — it raises without `depth_vda/`) and notes `retriangulation=True`
  as the GLOMAP-style post-solve knob, shown but not run.
- Short comparison of the two clouds.

### 02 · Pointcloud — `refinement`

Subject: improving a first pass. Both passes here are feedforward-only — SfM refuses them.

- `bootstrap_keyframes(n_frames=24)`, then a VGGT-Omega run kept in memory.
- **Bundle adjustment:** `BundleAdjustment(BundleAdjustmentConfig(capture_loss_history=True))`,
  `.refine(result)`, then `result.reproject()`. Camera-centre trajectory before/after,
  LM loss curve from `_last_loss_history`. Keeps the existing honest empty-history branch:
  on a small-baseline scene the reprojection filter can admit no frames, and the notebook
  says so rather than plotting nothing.
- **Loop closure:** `LoopClosure(creator, LoopClosureConfig(submap_size=8))`, `.run(images_dir)`.
  Submap structure, accepted matches, camera-centre gap per accepted pair before/after.
  Imports come from `collab_splats.geometry.loop_closure` (the package `__getattr__`), never
  the deleted `closure` module.

### 03 · Splats — `train_splats`

Subject: Scaffold-2DGS, the configuration in production use.

- `bootstrap_keyframes` + `bootstrap_reconstruction`.
- Trainer inputs assembled inline (this page teaches that glue).
- One train:

  ```python
  cfg = SplatsConfig(
      representation="scaffold",   # anchors + MLP decode
      primitive="2dgs",            # surface-aligned kernel
      scaffold={...},              # base.yaml defaults
      max_steps=MAX_STEPS,
      losses={..., "opacity_reg": {"weight": 0.0}},
  )
  ```

  Prose separates the two axes — `representation` picks the model class, `primitive` the
  rasterizer — and states the two rules the trainer enforces: `sh_degree` /
  `sh_degree_interval` are vanilla-only (scaffold decodes RGB from `mlp_color`), and
  `opacity_reg` must be 0 under scaffold, because opacity is decoded and its sign is the
  offset visibility mask, so regularizing it shuts offsets.
- **A vanilla run, shown not executed:** a markdown cell giving the
  `representation="vanilla", primitive="3dgs"` config and what changes with it.
- Outputs: `splats.ply` / `ckpt.pt` / `splats_quality_report.json`, per-frame PSNR bar,
  then a render gallery via `load_checkpoint` + `render_views`.

### 04 · Semantics — three pages

- **`feature_extraction`** — MaskCLIP and Talk2DINO over one keyframe: `forward`,
  `score_queries`, PCA-to-RGB, heatmaps, and the side-by-side per-query comparison
  absorbed from the deleted `maskclip_vs_talk2dino`.
- **`segmentation`** — `MobileSAMSegmentation` in both strategies, mask overlays,
  then `aggregate_masked_features` into per-object features.
- **`lifting_and_query`** — 2D → 3D. `bootstrap_keyframes` + `bootstrap_reconstruction`,
  then extract per-frame features, train a `FeatureAutoencoder`, `lift_features` onto
  the points, and score text queries per point. All the cache-hit / cache-miss branching
  in the current notebook is deleted — one path, always computed.

### 05 · Mesh — `tsdf_mesh`

Subject: processed data in, mesh out — and what TSDF is actually doing.

- `bootstrap_keyframes` + `bootstrap_reconstruction` + `bootstrap_splats`.
- **Prose on the method**, before any call: TSDF fuses each view's depth into a voxel
  grid of *truncated signed distances* — per voxel, the signed distance to the nearest
  surface along the ray, clamped to ±`sdf_trunc`. Averaging those fields across views is
  what cancels per-view noise; the mesh is the zero level set. Hence `sdf_trunc`
  (derived as 4 × `voxel_size`), not `voxel_size`, sets the thinnest structure that can
  survive, and cancellation needs *both* surfaces of a thin object to be seen.
- **Source A — feedforward depth:** confidence-gate with `confidence_mask`, convert
  `(N,3,H,W)` float images to `(N,H,W,3)` uint8, `invert_poses(ff.extrinsics)` for
  camera-to-world, `fuse_tsdf`.
- **Source B — splat renders:** `render_tsdf_inputs(ckpt)` returns depth, uint8 RGB,
  camera-to-world and K with pose-opt deltas already applied — nothing to lift or re-pose.
  `fuse_tsdf` on those.
- Side-by-side: vertex/triangle counts, connected components, largest-component fraction,
  two renders. Then `clean_repair_mesh` on both, and what its scale-relative thresholds mean.

### 06 · Localization — `localization`

Subject: one query image → a pose in a known map.

- `bootstrap_keyframes` + `bootstrap_reconstruction`.
- `CameraLocalizer.from_feedforward(result, images=..., ids=..., extractor=LocalMatcher("loma"))`,
  then `.localize(query_image)` on `QUERY_IMAGE` — a frame from a *different* video.
- Correspondence plot for the top-ranked reference frame, 3D scene view with the
  localized pose in red. Prose notes the query K is seeded from image proportions, and
  names the other matchers.

## Verification

The deliverable is nine notebooks that run. The proof is running them:

1. Each notebook executed end-to-end, from a clean tempdir, in dependency-free order
   (any page, any order — that is the point).
2. Outputs committed. `nbsphinx_execute` stays `"never"`, so the committed outputs are
   what the docs site renders.
3. A grep gate: no `data/outputs`, no `TUTORIAL_CACHE`, no `sys.path.insert`, no
   `assert .* run .* first` anywhere under `docs/source/tutorials/`.
4. An import gate: every `collab_splats.*` / `evals.*` symbol imported by any notebook
   resolves against the branch. (This is the check that caught the seven breaks above;
   it becomes a small script so it can be re-run.)

Notebooks are executed one at a time — a full A40 is needed for the splat trains, and
concurrent runs risk OOM against the 46.6 GB cgroup cap.

## Risks

- **Wall-clock.** Nine pages, each paying its own feedforward run, plus two splat trains.
  The fast profile is what keeps this tractable; if a page still runs long, the knob to
  turn is `MAX_FRAMES`, never a reintroduced cache.
- **Under-converged figures.** At `MAX_STEPS=1000` the splat renders are visibly softer
  than production. Every such page states the production value next to the knob.
- **Notebook size.** Committing outputs for nine executed pages grows the repo. The
  current tutorials already carry 32 MB of committed outputs; this should land at or
  below that, since four pages are being deleted.
