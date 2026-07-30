# GCS remote pipeline — design

**Date:** 2026-07-29
**Status:** approved, ready for planning
**Branch:** `refactor/cu121-uv-migration`

## Problem

Processing a video from GCS is dashboard-only today. `collab_splats/dashboard/sources.py::SessionSource`
lists and transfers scenes; `dashboard/pipeline.py::_push_async` uploads results. A headless run has no
path to the bucket at all — `docs/examples/run_pipeline.py` takes local file paths and writes local
output, nothing more.

Four things need to change:

1. The curated bucket moved. Source is now `environments-curated`, laid out as flat
   `YYYY_MM_DD-PARENTFOLDER-VIDEONAME/` directories at the bucket root (the naming mirrors the Google
   Drive folder the files came from). One video per directory. Outputs go to `environments-processed`.
2. A headless driver must be able to process remote scenes end to end: pull a video, reconstruct it,
   push results, reclaim local disk.
3. Downstream consumers need `.ply` for both the pointcloud and the mesh, under one name each, at full
   density. The mesh already writes a PLY but under a name half the codebase disagrees with; the
   pointcloud exists only inside `feedforward.zarr`.
4. Storage format needs deciding — which artifacts ship to `environments-processed`, in what form.

## Findings that shaped the design

Measured against a real scene (`outputs/2023_11_05/PXL_20231105_154956078/`), and against the code as of
`92f2e4a`.

### Artifact footprint

| artifact | size | role |
|---|---|---|
| `semantics/lifted_normed.npy` | 1.5 G | per-point features, **raw 768-D** (dashboard path only) |
| `frames.zarr` | 1.2 G | sole persistent keyframe store; localization + BA read it |
| `mesh/vertex_features.npy` | 315 M | per-vertex features, **raw 768-D** |
| `features/<extractor>/<extractor>.zarr` | 735 M | raw 768-D dense 2D feature maps |
| `colmap/` | 31 M | SfM interop |
| `feedforward.zarr` | 28 M | points, colors, extrinsics, intrinsics, depth, world_points, confidence, pixel_indices, `local_features/` |
| `frames/` (jpgs) | 48 M | regenerable decode cache |
| `mesh/mesh_tsdf.ply` | 4.9 M | mesh, already PLY (renamed to `mesh/mesh.ply` here) |
| `transforms.json` | 119 K | nerfstudio camera poses |

**`feedforward.zarr` is 28 MB.** The "zarr is a bad object-storage format" concern does not apply to the
intermediates that matter — every array needed to re-refine a scene fits in 28 MB. The bulk is the
feature artifacts and `frames.zarr`. So the format question is *which artifacts ship*, not *zarr vs PLY*.

### Semantics is compressed on one path and raw on the other

`_lift_and_save` (`wrapper/reconstructor.py:247-291`) with `configs/base.yaml semantics.n_components: 64`
lifts to (P, 768), trains `FeatureAutoencoder(input_dim=768, latent_dim=64)`, calls `per_point_encode`,
and writes `features.zarr` at (P, 64) plus `compressor.pt`. Compression is conditional on
`n_components is not None`.

The dashboard path does not compress. It writes `lifted_normed.npy` at (500000, **768**) and
`vertex_features.npy` at (107232, **768**). Two divergent layouts for the same data, one of them 12×
larger than intended.

### AE decode is lossy; exact recovery needs the raw 2D maps

`FeatureAutoencoder.save()` persists `input_dim`, `latent_dim`, and the full `state_dict` — encoder plus
`decoder_hidden` and `decoder_out`. `load()` rebuilds it and `per_point_decode` maps (N, 64) → (N, 768).
So `compressor.pt` alone recovers full dimensionality.

It recovers an *approximation*. `fit` runs 10 epochs of Adam at lr 1e-3 against
`MSE(recon, x) + (1 - cos(recon, x))` through a 12× bottleneck. Two further constraints: the AE is
trained per scene on that scene's own points, so codes are not comparable across scenes and decoding
requires that scene's `compressor.pt`; and reconstruction error is never measured or recorded.

Exact 768-D is recoverable only from the raw 2D maps in `features/`, by re-lifting against
`feedforward.zarr`. **This design accepts lossy features and drops the raw maps** — see Decisions.

### Cross-scene comparability, and what epochs actually buy

Codes from two scenes are partly comparable, not incomparable. Both AEs compress samples of the *same*
pretrained backbone manifold (talk2dino/DINOv2 768-D), so scenes with overlapping content present
overlapping input distributions and the learned bases partly align. What is not guaranteed is the
alignment itself: two independent fits are free to differ by rotation, permutation, and sign in the
latent basis, and nothing in the objective pins them. So raw 64-D distances across scenes are
suggestive, not trustworthy.

That splits the epochs question in two:

- **Comparing raw 64-D codes across scenes** — more epochs does not help and can mildly hurt. Extra
  training tightens each AE onto its own scene's manifold; it does not pull two bases into a shared
  frame, because no term in the loss references the other scene.
- **Comparing decoded 768-D features** — more epochs helps directly. Decode each scene's codes through
  that scene's own `compressor.pt` and compare in the shared backbone space. The basis question
  disappears by construction, and the only thing standing between the codes and a correct comparison is
  reconstruction fidelity — exactly what training longer improves.

So: decode-then-compare is the supported cross-scene operation, and raising training does improve it.

**Epochs or threshold: threshold, with epochs as a ceiling.** Ten fixed epochs is an arbitrary number
that reports nothing about whether the fit is good. A target says what we actually want. `fit` already
computes mean cosine similarity every step as part of its `MSE + (1 - cosine)` objective, so the stop
signal costs nothing to read — no extra metric pass, no held-out split.

```
epochs: int = 100          # ceiling, was a fixed 10
target_cosine: float = 0.95  # stop as soon as the epoch mean clears this
```

Stop at the target; warn if the ceiling is reached without it, so an underfit scene is visible rather
than silent. Cost is small: 500k points at batch 1024 is ~490 steps/epoch, and well-conditioned scenes
converge long before the ceiling. The achieved `recon_cosine` / `recon_mse` are written into
`compressor.pt`, so every scene ships its own measured fidelity and a consumer can decide whether to
trust a decode.

Making raw 64-D codes directly comparable needs a *shared* AE — trained once over a corpus, versioned,
and shipped as a repo asset instead of per scene. That is a different project (corpus, training driver,
weight versioning) and stays out of scope; it is recorded as a follow-up.

### `frames/` jpgs are dead weight

They exist because two call sites once required an on-disk image directory. Both now accept zarr:

- `setup_inference(source: FrameStore | Path)` — `_decode_source`
  (`pointcloud/feedforward/base.py:846`) already handles `FrameStore`, `*.zarr`, and legacy dirs.
  `Reconstructor` already passes `FrameStore.open(frames_zarr)`.
- semantics — `extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)` already exists
  (`reconstructor.py:244`). The dashboard's jpg-globbing `_extract_semantics` duplicates it.

### A pointcloud PLY already exists — as a side effect, in ASCII

`sparse_pc.ply` is **not** a COLMAP convention. COLMAP names its outputs `points3D.bin` / `.txt` for the
sparse model and `dense/fused.ply` / `meshed-poisson.ply` for dense products; it has no `sparse_pc.ply`,
and its PLY export (`model_converter`) takes the filename from the caller.

It is a **nerfstudio** convention. `colmap_to_json` declares `ply_filename="sparse_pc.ply"` as a default
parameter (`nerfstudio/process_data/colmap_utils.py:396`), calls `create_ply_from_colmap` to write it,
and records `out["ply_file_path"] = ply_filename` in `transforms.json` (`:489`). Splatfacto reads that
key to seed a 3DGS run.

This repo calls it on **both** reconstruction paths — `BasePointcloudCreator._write_transforms`
(`pointcloud/base.py:118-120`) is invoked from `sfm.py:77,197` and from `feedforward/base.py:901`. So
the file is produced today. Verified on disk at
`outputs/2023_11_05/PXL_20231105_154956078/sparse_pc.ply`:

```
element vertex 500000     # == max_points, with red/green/blue
format ascii 1.0          # 19.7 MB  ->  39.4 bytes/point
transforms.json: ply_file_path = "sparse_pc.ply", applied_transform present
```

Full density, colored, correctly cross-referenced. Two things are wrong with it anyway:

1. **ASCII.** Open3D's binary PLY writes `property double x/y/z` + `uchar red/green/blue` = 27.0
   bytes/point (measured), so the same 500k cloud is 13.5 MB against 19.7 MB — ~6 MB/scene, 1.46×. Size
   is the smallest of the three wins. Read speed is the real one: ASCII runs every coordinate through
   `strtod`, binary is one `np.frombuffer`. And ASCII is *lossy* — a measured 200k roundtrip gives
   `max_xyz_err = 5.0e-06` from fixed-digit decimal rounding, where binary returns `0.0` exactly.
   Colors survive both.

   Nerfstudio compatibility verified, not assumed: every dataparser that consumes `ply_file_path` reads
   through `o3d.io.read_point_cloud` (`nerfstudio_dataparser.py:435`, `blender_dataparser.py:120`,
   `scannet_dataparser.py:209`), which detects the encoding from the header;
   `_load_3D_points` then does `np.asarray(pcd.points, dtype=np.float32)` and applies the transform,
   with no dependence on how the file was encoded.
2. **Incidental.** It is a side effect of generating `transforms.json`, built by round-tripping through
   `points3D.bin` rather than from `result.points`. Nothing declares it as an output, nothing tests it,
   and it disappears silently if `colmap_to_json` ever stops being called — including via the
   `_write_transforms_json` conflict below.

### Two transforms.json writers disagree

`Reconstructor._write_transforms_json` (`reconstructor.py:604`, called at `:543`) hand-rolls a
`{camera_model, frames}` document keyed on `frame_idx`, with **no `ply_file_path` and no
`applied_transform`**. `creator.reconstruct(store, output_dir)` (`:200`) has already written
nerfstudio's richer version via `_write_transforms`. If `output_dir == backend_dir`, the hand-rolled one
overwrites it and the scene loses its pointer to its own PLY — which would ship a `transforms.json` to
`environments-processed` that no splat trainer can seed from. The measured scene still has
`ply_file_path`, so this either predates the second writer or the two paths differ. **Must be confirmed
before the push set is finalized.**

### Mesh names: four names, one live bug

| written | expected by | reality |
|---|---|---|
| `mesh/mesh_tsdf.ply` (`mesh/tsdf.py:95`) | `dashboard/{app,localize,pipeline}.py` | the file that actually exists |
| `mesh/mesh.ply` (`reconstructor.py:743`) | `Reconstructor.mesh()` skip-check, `visualize.py:51` | **never exists** |
| `mesh/mesh_tsdf_clean.ply` (`splatter.py:480`) | splatter candidate list | unreachable — `clean_repair` raises |

`Reconstructor.mesh()` guards on `mesh.ply` existing but `_run_tsdf_mesh` writes and returns
`mesh_tsdf.ply`, so the guard never fires: `overwrite=False` is a no-op and the mesh stage recomputes on
every run. `visualize.py:51` reads the same non-existent path and always reports no mesh. Both are fixed
by picking one name, not by adding a lookup.

### The point cap is a memory guard, not a preference

`max_points` (`500_000`, `feedforward/base.py:794`) is applied *during inference, to the confidence
mask* — `randomly_limit_trues(conf_mask, max_points)` at `vggtx.py:144`, `mapanything.py:454`, threaded
from `vggtx.py:356,423` and `vggt_omega.py:267,322`. The full cloud is therefore never materialized.
That is load-bearing: removing the cap is what produced a 5.1 GB `points3D.bin` and a 50 GB SIGKILL in
the loop-closure OOM incident, and `loop_closure/wrapper.py:613` re-applies `self.base.max_points` so LC
and non-LC agree.

Two separate concerns are currently conflated under one number:

- the **memory guard** that keeps inference from exploding — must stay inside the pipeline;
- the **delivery density** a consumer wants in a PLY — belongs at export.

Moving the guard to a postproc step would reintroduce the OOM. Downsampling again at postproc would thin
an already-thinned cloud, and still could not recover what the first cap dropped. The fix is to expose
the guard in config, and give export its own optional knob.

The trap for the exporter: `subsample_points` (`pointcloud/utils.py:299`) defaults to
`max_points=50_000` *and* confidence-filters at the 20th percentile. It serves viewers and submap
balancing. An export path that reaches for it inherits both defaults and silently ships a 10×-sparser,
confidence-clipped cloud.

### Deleting local data is recoverable

The source video is never removed from `environments-curated`; the pipeline only reads it. Worst case
after a failed push is re-fetch and reprocess — GPU hours, not data loss. The verification gate exists
for a different reason: a partial push leaves `environments-processed/<scene>/` half-written, and
`has_processed(scene)` reports True for it, so a broken scene reads as complete and silently poisons
downstream consumers.

## Decisions

| # | Decision |
|---|---|
| 1 | `environments-curated` / `environments-processed` **replace** `fieldwork_curated` / `fieldwork_processed` everywhere, dashboard included. No coexistence. |
| 2 | Scene id = the curated directory name, `YYYY_MM_DD-PARENTFOLDER-VIDEONAME`. Flat, one level, one video per directory, no `reconstruction/` prefix. |
| 3 | `SessionSource` is retargeted, renamed `SceneSource`, and relocated out of `dashboard/` into `collab_splats/remote/`. Both the dashboard and the remote driver import it. One transfer layer. |
| 4 | The per-scene core moves into the package. Two thin drivers, one local and one remote. |
| 5 | rclone throughout, via the existing `RcloneClient`. Credentials stay in the rclone remote config; no key handling in this repo. |
| 6 | **We write the pointcloud PLY ourselves**, in `collab_splats/pointcloud/export.py`, called from the pointcloud stage — binary, straight from `result.points`, replacing nerfstudio's incidental ASCII round-trip. |
| 6a | **The name stays `sparse_pc.ply`.** It is nerfstudio's convention, `transforms.json["ply_file_path"]` already points at it, and every nerfstudio-derived tool expects it. "sparse" is inaccurate for a 500k cloud, but that is cosmetic and the count is in the header. Cheap to revisit — `colmap_to_json` takes `ply_filename` as a parameter. |
| 6b | **Mesh unifies on `mesh/mesh.ply`.** `mesh_tsdf.ply` and `mesh_tsdf_clean.ply` retired; TSDF writes `mesh.ply` directly, which repairs the dead skip-check and the webapp read in the same move. |
| 6c | **The point cap splits in two.** The in-pipeline mask cap stays exactly where it is as a memory guard, exposed as `pointcloud.max_points` in `configs/base.yaml` (default `500_000`, unchanged) so it stops being an invisible hardcode. Export density is a separate optional param on `write_pointcloud_ply`, defaulting to no thinning so the PLY matches the zarr. |
| 7 | Push excludes `features/**` (raw 768-D 2D maps) and `frames/**` (jpgs, deleted outright). Compressed `semantics/` ships. |
| 8 | Semantics converges on the compressed layout: `features.zarr` (64-D) + `compressor.pt`. `lifted_normed.npy` retired; `vertex_features.npy` stored at 64-D. |
| 8a | AE training becomes target-driven: `target_cosine=0.95` with `epochs=100` as a ceiling, replacing the fixed 10. Achieved `recon_cosine` / `recon_mse` ship inside `compressor.pt`. Decode-then-compare in 768-D is the supported cross-scene operation. |
| 9 | Local scene data and the fetched video are deleted after a verified push. `--keep-local` skips it. |
| 10 | No `--list`, no `--fetch-only`, no `--pull-processed`. Raw `rclone ls` / `rclone copy` covers ad-hoc browsing; `--all` removes the need to name scenes. |

### Rejected

- **Coexisting bucket pairs.** Two GCS layouts to maintain and a third listing scheme in `SceneSource`.
- **Dropping zarr for PLY.** PLY cannot carry `depth`, `world_points`, `confidence`, `pixel_indices`, or
  `local_features/`. At 28 MB, `feedforward.zarr` costs nothing to ship.
- **Keeping the raw `features/` 2D maps.** Would guarantee exact 768-D recovery at 735 MB/scene. The
  compressed codes plus `compressor.pt` are accepted as sufficient.
- **A single driver with `--remote`.** A flag that changes what positional arguments *mean* — file paths
  vs scene ids — is a smell. The two input models also disagree on output-dir derivation.
- **`--pull-processed` in this round.** `pull_processed`'s `PULL_EXCLUDES` is display-shaped: it strips
  exactly the dense arrays a re-refinement pull would need. Shipping it headless would hand the compute
  path the wrong exclude set. The bytes are in the bucket (inside `feedforward.zarr`), so nothing is
  stranded — re-refinement gets its own spec when it's needed.
- **A `remote:` config block.** Hides a network fetch and an `rm -rf` behind a YAML file.

## Architecture

```
environments-curated/<scene>/<video>
  → fetch to <output-root>/<scene>/
  → run_scene: preproc → pointcloud → semantics → mesh → localize
  → sparse_pc.ply (binary) + mesh/mesh.ply
  → push to environments-processed/<scene>/    [excludes: features/**, frames/**]
  → rclone check --one-way                     [hard gate]
  → rmtree scene dir + fetched video
```

### Components

| unit | responsibility | depends on |
|---|---|---|
| `collab_splats/remote/sources.py` | `SceneSource` — list/fetch curated, push/pull/verify processed | `RcloneClient` |
| `collab_splats/wrapper/batch.py` | `build_scene_config`, `run_scene`, `run_all` — per-scene config + execution + failure isolation | `Reconstructor` |
| `collab_splats/pointcloud/export.py` | `write_pointcloud_ply(points, colors, path)` — binary PLY, full density | open3d |
| `docs/examples/run_pipeline.py` | local driver: video paths and directories | `wrapper.batch` |
| `docs/examples/run_pipeline_remote.py` | remote driver: scene ids or `--all`, plus the transfer lifecycle | `wrapper.batch`, `remote.sources` |

### `SceneSource`

```
CURATED_BUCKET  = "environments-curated"
PROCESSED_BUCKET = "environments-processed"
_SCENE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-.+-.+$")
```

Retained, all reduced to a single `scene` key: `fetch_video`, `has_processed`, `pull_processed`,
`pull_zarr_members`, `push_outputs`, `list_localization_dbs`, plus `_cached` / `invalidate` /
`_run_streaming` / `parse_rclone_percent` unchanged.

Renamed: `list_sessions` → `list_scenes()` (lists curated scene dirs matching `_SCENE_RE`);
`list_processed_stems(session)` → `list_processed()` (flat list of processed scene ids).

New: `verify_push(local_dir, scene, excludes) -> None`, wrapping `rclone check --one-way` with the same
excludes used for the push. Raises on any mismatch.

Deleted: `ROOT`, `_FIELD_SESSION_RE`, `list_videos`, `list_field_sessions`, `list_rgb_cameras`,
`list_camera_videos`, `fetch_field_video`.

`fetch_video(scene, dest_dir)` resolves the scene's single video itself — lists the scene directory,
filters on `_VIDEO_EXTS` (`.mp4`, `.mov`), and errors naming the scene on zero or multiple matches.
Callers pass a scene id, never a filename; the layout guarantees exactly one video. This absorbs what
`list_videos` was for.

Every remote path collapses from `(session, stem)` to a single `scene`. This is the dashboard's largest
churn: `app.py` and `localize.py` thread the two-level key through listings, cache keys, and output
paths.

### `wrapper/batch.py`

`build_scene_config`, `run_scene`, and `run_all` move here verbatim from `docs/examples/run_pipeline.py`,
minus `scene_output_dir`'s scan-parents-for-a-date-directory logic, which stays with the local driver —
remote scene ids already begin with `YYYY_MM_DD`. `run_all` keeps its per-scene try/except, its FAIL
summary, and its non-zero exit on any failure; both drivers inherit that behavior unchanged.

### `pointcloud/export.py`

One function:

```python
def write_pointcloud_ply(
    points: np.ndarray,
    colors: np.ndarray | None,
    path: Path,
    max_points: int | None = None,
) -> Path:
    """Write a (P, 3) cloud + optional (P, 3) uint8 colors to path as a binary PLY."""
```

Takes arrays, not a result object, so it serves `FeedforwardResult` and `PointcloudResult` alike — both
already expose `.points` / `.colors` with the same shapes and dtypes. `FeedforwardResult` stays a pure
dataclass and open3d stays out of `feedforward/base.py`.

**Binary, not ASCII.** `o3d.io.write_point_cloud(..., write_ascii=False)` — the open3d default, so this
is a parameter we simply do not override. 13.5 MB instead of 19.7 MB, exact coordinates instead of
5e-06 decimal rounding, and a `np.frombuffer`-speed read for consumers instead of 1.5M `strtod` calls.
Nerfstudio reads it through `o3d.io.read_point_cloud` either way — verified, see findings.

**Straight from `result.points`.** The current file is rebuilt from `points3D.bin` as a side effect of
generating `transforms.json`. Writing from the result array removes the round-trip and makes the PLY a
declared output of the pointcloud stage — something a test can assert on and a push can require.

**Density defaults to no thinning.** `max_points=None` means the PLY is exactly the stored cloud, so
zarr and PLY never disagree. A caller who explicitly wants a lighter file passes a number and gets an
even `np.linspace` decimation. The function does **not** call `subsample_points` — that helper's
`50_000` default and 20th-percentile confidence filter serve viewers and LC submap balancing, and
inheriting them here would silently ship a 10×-sparser, confidence-clipped cloud.

Nerfstudio's own PLY write is suppressed so the two do not both produce a file: `_write_transforms`
passes `colmap_to_json(..., ply_filename="sparse_pc.ply")` unchanged, so `transforms.json` keeps
pointing at the same name our writer produces, and our binary version overwrites the ASCII one written
moments earlier in the same stage. Ordering is asserted by test.

**Upstream cap moves into config**, unchanged in value:

```yaml
pointcloud:
  max_points: 500000   # inference-time memory guard (see feedforward/base.py:794)
```

The `BaseFeedforwardCreator` field keeps `500_000` as its default for direct instantiation; the config
becomes the declared knob. Unrelated caps stay unrelated, stated here to stop them creeping in: the
dashboard's `max_display_points` (`app.py:192`) is viewer decimation and never touches stored arrays.

### Naming

| artifact | name | change |
|---|---|---|
| pointcloud | `{backend_dir}/sparse_pc.ply` | **unchanged.** Already correct on both paths and referenced by `transforms.json["ply_file_path"]`. Only the writer and the encoding change. |
| mesh | `{backend_dir}/mesh/mesh.ply` | renamed from `mesh_tsdf.ply`. Touches `mesh/tsdf.py:95`, `wrapper/splatter.py:480`, `dashboard/{app.py:650, localize.py:661, pipeline.py:151,491}`, `visualize.py:51,69` |

The pointcloud name is left alone deliberately. It is nerfstudio's convention, not COLMAP's, and
nerfstudio *consumes* it — `ply_file_path` is what seeds splatfacto. Renaming is a one-line
`ply_filename=` change if the downstream consumer ever finds "sparse" confusing for a 500k cloud, so
nothing is locked in.

Renaming the TSDF output to `mesh.ply` makes `Reconstructor.mesh()`'s existing skip-check live for the
first time — no new logic, the guard was already written for this name. Delete the `mesh_tsdf_clean.ply`
candidate branch in `splatter.py:480` and the `app.py:649` comment documenting the mismatch; both exist
only to work around the split.

No compatibility shims and no fallback path lists — one name, every reader updated.

### Remote driver CLI

```
python docs/examples/run_pipeline_remote.py --output-root DIR [--all | SCENE_ID...]
                                            [--config Y] [--config-dir D]
                                            [--stages S,...] [--overwrite] [--keep-local]
```

`--all` processes every scene in `list_scenes()` for which `has_processed()` is False. Explicit scene
ids remain supported for re-runs and one-offs. `--all` and explicit ids are mutually exclusive.
`--config`, `--config-dir`, `--stages`, and `--overwrite` behave exactly as in the local driver.

## Folded-in cleanups

This work forces these; leaving them would mean pushing 12× oversized features and a dead decode path.

1. **Delete the jpg path.** Remove `_write_frames_jpegs` and `_extract_semantics` from
   `dashboard/pipeline.py`; point the dashboard at `FrameStore.open(frames_zarr)` and
   `extract_and_cache_from_zarr`. Remove `_decode_source`'s legacy-image-dir branch and
   `_decode_dir_to_frames`, now unreachable.
2. **Converge semantics.** Dashboard `_lift_and_compress` calls `_lift_and_save`. Retire
   `lifted_normed.npy`; `viewer.load_lifted_normed` and the localize read paths move to `features.zarr`.
   `persist_mesh_vertex_features` receives encoded 64-D features, so `vertex_features.npy` lands at
   (M, 64).
3. **Make AE fidelity a target, not a guess.** `fit` gains `target_cosine=0.95` and an `epochs=100`
   ceiling, tracking the per-epoch mean of the cosine term it already computes; it stops on target and
   warns if the ceiling is hit first. `save()` writes the achieved `recon_cosine` / `recon_mse` into the
   `compressor.pt` payload alongside `input_dim` / `latent_dim` / `state_dict`, and `load()` tolerates
   their absence so existing checkpoints still open.

4. **Unify the mesh name** (see Naming): `mesh/mesh.ply` everywhere, `mesh_tsdf.ply` and
   `mesh_tsdf_clean.ply` deleted, all readers updated in the same change. The pointcloud name is
   already consistent across both backends and does not move.

5. **Resolve the two transforms.json writers.** Confirm whether
   `Reconstructor._write_transforms_json` overwrites nerfstudio's output. If it does, the pushed
   `transforms.json` loses `ply_file_path` and `applied_transform` and no splat trainer can seed from the
   scene — so the hand-rolled writer either stops running or learns to preserve both keys. This is a
   verification task first and a fix second; the fix is small either way. **Blocks finalizing the push
   set**, since `transforms.json` is a shipped artifact.

## Error handling

Per scene, in order — any failure marks the scene FAIL, leaves local data intact, skips the remaining
steps for that scene, and continues to the next:

| stage | on failure |
|---|---|
| fetch | FAIL. Nothing written. |
| run | FAIL. Partial local output kept for debugging. No push. |
| push | FAIL. Local kept. Remote may be partial — the next `--all` run retries it because verification never marked it complete. |
| verify | FAIL, logged loudly with the `rclone check` output. Local kept. **No delete.** |
| delete | Runs only after verify passes. Skipped entirely under `--keep-local`. |

The process exits non-zero if any scene failed. `has_processed()` is the `--all` filter, so an unverified
scene is retried rather than skipped.

> **Amended as shipped, 2026-07-30.** Two things above did not survive implementation; the plan's
> carry-forward items 18 and 25 carry the detail.
>
> - **Exit codes are four, not two.** `0` ok · `1` scene(s) failed on their own merits · `2` nothing
>   to do · `3` aborted early because rclone became unreachable. The driver re-probes the remote
>   after any scene failure (`SceneSource.check_available()`, an uncached `lsjson`) and stops rather
>   than marching the rest of the list into the same fault, because no exception type or exit code
>   distinguishes a dead remote from a reconstruction error. Un-attempted scenes are reported
>   `SKIPPED`, not `FAIL`. Contract table: `configs/README.md`.
> - **`has_processed()` is not wired to `--all`.** It exists and is tested, but the driver never
>   calls it, so `--all` reconstructs every curated scene including already-published ones. The
>   skip-if-processed behaviour described above is unimplemented; adding it needs a `--skip-processed`
>   flag and a decision on what counts as "already done" (a scene whose push half-completed must
>   still be retried). Deliberately out of scope for the 12 tasks.

## Testing

- `tests/remote/test_sources.py` — fake `RcloneClient` following the existing
  `tests/dashboard/test_sources_field.py::_FakeClient` pattern. Covers scene-id regex, bucket path
  construction, `fetch_video` single-video resolution (including the zero-match and multi-match errors),
  and push excludes.
- `tests/remote/test_verify_gate.py` — `rclone check` returning non-zero must leave the scene directory
  on disk. Asserts `rmtree` is not called. This is the test that protects the destructive step.
- `tests/pointcloud/test_export.py` — PLY roundtrip: write, read back, assert point count and colors
  survive. Plus four guards: the header says `format binary_little_endian`, not `ascii`; the file reads
  back through `o3d.io.read_point_cloud` — nerfstudio's own reader — with bit-exact coordinates; with
  `max_points=None` and more points than `subsample_points`' 50 000 default the PLY count equals
  `len(points)` exactly (catches a future edit reaching for `subsample_points`); and an explicit
  `max_points` thins to exactly that count.
- `tests/pointcloud/test_max_points_config.py` — `pointcloud.max_points` from config reaches the
  creator, and the stored cloud honours it.
- Write-ordering: after the pointcloud stage, `sparse_pc.ply` is the binary file our writer produced,
  not the ASCII one `colmap_to_json` writes earlier in the same stage, and
  `transforms.json["ply_file_path"]` still names it.
- `tests/semantics/test_compression.py` — extend: `fit` stops early once `target_cosine` is met, hits
  the epoch ceiling and warns when it is not, and `save`/`load` roundtrip `recon_cosine` / `recon_mse`
  while still loading a checkpoint written without them.
- Naming: assert the mesh stage writes `mesh/mesh.ply` and that `Reconstructor.mesh(overwrite=False)`
  now actually skips when that file exists — the regression test for the dead guard.
- `transforms.json` survives the full pointcloud stage with `ply_file_path` and `applied_transform`
  intact (the two-writer conflict).
- `tests/wrapper/test_batch.py` — the relocated core keeps its failure isolation: one failing scene does
  not abort the batch, and the summary reports it.
- `tests/scripts/test_run_pipeline_remote.py` — monkeypatched `SceneSource` and `run_scene`; asserts the
  fetch → run → push → verify → delete ordering, that `--all` filters on `has_processed`, that
  `--keep-local` skips the delete, and that a failure at each stage skips the delete.
- Dashboard smoke gate: `python -m collab_splats.dashboard --smoke` must print `SMOKE PASS` before any
  dashboard change is committed (per CLAUDE.md).
- Full suite: `/opt/venv/reconstruction/bin/python -m pytest tests/`.

## Implementation principles

Binding on every task in the plan.

**Reuse before writing.** Each of these already does the job and must be called, not reimplemented:

| need | existing function |
|---|---|
| rclone transport, streaming progress | `RcloneClient`, `_run_streaming`, `parse_rclone_percent` |
| listing memoization | `_cached` / `invalidate` |
| per-scene failure isolation + summary | `run_all` (relocated verbatim) |
| 2D features from zarr | `extract_and_cache_from_zarr` |
| lift + AE-compress + persist | `_lift_and_save` |
| mesh vertex features | `features2vertex`, `persist_mesh_vertex_features` |
| keyframe access | `FrameStore.open` |
| remote/local equality | `rclone check --one-way` (do not hand-roll checksums) |

Genuinely new code is three things: `write_pointcloud_ply`, `verify_push`, and the remote driver's
fetch → run → push → verify → delete loop. Everything else in this spec is a move, a rename, or a
deletion. If a task's diff adds a function that overlaps the table above, the task is wrong.

**Reusable shapes.** `write_pointcloud_ply` takes arrays, not a result type, so both result classes can
use it. `verify_push` takes the same exclude tuple the push used, so the two can never disagree.
`wrapper/batch.py` is backend- and transport-agnostic — it knows nothing about GCS, which is what lets
both drivers share it.

**Retire what this obsoletes.** No shims, no deprecation window, no fallback path lists — this branch
has a single consumer. Deleted: `fieldwork_*` constants, `ROOT`, `_FIELD_SESSION_RE`, `list_videos`,
`list_field_sessions`, `list_rgb_cameras`, `list_camera_videos`, `fetch_field_video`,
`_write_frames_jpegs`, dashboard `_extract_semantics`, `_decode_dir_to_frames`, the legacy image-dir
branch of `_decode_source`, `lifted_normed.npy`, `mesh_tsdf.ply`, the `mesh_tsdf_clean.ply` branch, and
the `app.py:649` comment documenting the name mismatch. A task that adds a reader for an old name
instead of updating it is wrong. `sparse_pc.ply` is **not** on this list — the name stays, only its
writer and encoding change.

**Minimal surface.** No config-driven push manifest — the exclude set is a module constant until a
second caller needs a different one. No `--remote` flag on the local driver. No parameter that is always
passed its default; specifically `write_pointcloud_ply` takes no `max_points`, because there is no
correct caller that would thin the export.

**Inline block comments, per CLAUDE.md.** Every logical block gets a short comment saying what it does —
block level, not per line. New code carries them from the first commit rather than a later pass. The
non-obvious ones are load-bearing and must be present: why the export does *not* subsample, why delete
is gated on verify, why `features/**` is excluded but `semantics/` is not, and why the AE stops on
cosine rather than epoch count. Public functions get a one-line docstring; `########` dividers separate
sections in the new modules.

## Open follow-ups

- **Re-refinement pull.** Fetching a processed scene's intermediates for further processing needs a
  compute-shaped exclude set (the inverse of `PULL_EXCLUDES`) plus stage-dependency checks against a
  partial tree. Own spec.
- **Shared cross-scene AE.** Decode-then-compare in 768-D works today. Direct 64-D comparison needs one
  AE trained over a corpus and shipped as a versioned repo asset — which would also drop `compressor.pt`
  from every scene's payload. Needs a corpus, a training driver, and weight versioning. Own spec.
- **Measured decode fidelity across backbones.** Once `recon_cosine` ships in `compressor.pt`, collect
  it across scenes and backbones to check whether `target_cosine=0.95` and `latent_dim=64` are the right
  operating point, or whether the bottleneck should move.
