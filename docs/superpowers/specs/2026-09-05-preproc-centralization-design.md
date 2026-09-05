# preproc centralization — design

**Date:** 2026-09-05
**Scope:** `collab_splats/preproc/` and its callers
**Status:** design, awaiting approval

Second per-module cleanup pass, following
[2026-08-22-preproc-cleanup-design.md](2026-08-22-preproc-cleanup-design.md).
That pass split measurement from selection. This one changes what preproc
*stores*, makes the quality filter actually filter, deletes the machinery that
existed only to work around it, and replaces three hand-rolled subsystems with
library calls.

---

## 1. Motivation

Nine problems, measured on real scenes rather than asserted.

**The keyframe store is repo-specific and larger than the format everyone else
uses.** `frames.zarr` holds 300 GH010229 frames in 1.50 GB (blosc-lz4 reaches
only 1.25x on photographic uint8 — `shuffle: bitshuffle` at `typesize: 1`
actively hurts). The same 300 frames already sit beside it as
`instantsfm/images/*.jpg` at 0.247 GB, because `_run_sfm` re-stages them. The
feedforward path does the same thing transiently through `FrameStore.export()`.
The pipeline converts zarr to an image directory twice and stores both.

**The quality filter filters nothing.** `filter_frame_quality`'s
`laplacian_min=50` sits 40-170x below the minimum Laplacian variance of any real
video:

| video | n | lap min | lap p50 | frames cut by `lap < 50` |
|---|---|---|---|---|
| GH010229 | 13115 | 1968 | 8514 | **0** |
| data/tutorial | 2388 | 844 | 4681 | **0** |

The exposure half is equally dead: the band is 20-235 against a real
`exposure_mean` range of 56-148, and the contrast floor is 10 against a real
minimum `exposure_std` of 30.6. `filter_frame_quality` returns all-True on
every video the repo has ever processed. Its only surviving effect is as a
constant term in a `max()` key, where it changes nothing.

**Undistortion carries a hand-rolled parameter type and hand-rolled output
framing, both of which pycolmap already provides.** `estimate_camera_distortion`
receives a `pycolmap.Camera` from the mapper and immediately unpacks its eight
params into a frozen dataclass with `to_dict`, `from_dict`, `K` and
`dist_coeffs` — 55 lines to re-implement what the Camera already does, while
discarding the model name. It then stages <= 60 frames to a `TemporaryDirectory`
as re-encoded JPEGs purely to give pycolmap an `image_path`. The output side
hand-rolls the target camera out of `getOptimalNewCameraMatrix`, an even-ROI
crop and a `K` shifted by the crop offset.

**`video.py` reimplements a decoder over ffmpeg subprocesses.** 345 lines of
rawvideo pipes, `select=eq(n\,i)` filter-graph string building, input-seek
guards, fixed-size stdin reads and terminate/wait cleanup — to do what `av`
(already installed, transitively via nerfstudio) does with a for-loop.

**`video.py` also mixes decode primitives with pipeline stages.**
`extract_frame` rebuilds the same ffmpeg command `iter_frames` builds.
`context_indices` is selection-grid arithmetic living in a decode module.
`decode_context` is 85 lines of undistort-plus-downscale-plus-chunk with a
VDA-specific `out_short_side=518`, a single consumer in `_run_sfm`, and a lazy
inline import of `undistort` that exists only because the function is in the
wrong file.

**`viz.py` carries 85 lines that no caller reaches, one of which is broken.**
`plot_disparity_sensitivity` and `plot_quality_examples` have no pipeline
caller. `plot_quality_examples` raises in the shipped tutorial notebook — the
notebook hands it a `_VideoFrameLookup` and the function wants a `FrameStore`;
the traceback is committed in the `.ipynb`. It also hardcodes `lap < 50.0`, a
third copy of the dead threshold, so both of its "Rejected" rows render empty
on every real video even when it does run.

**`qa.py` splits one operation across three public functions.**
`match_descriptors`, `compute_translation` and `compute_parallax` each have
exactly one caller, and that caller runs all three in sequence on the same pair.

**The image-extension listing is written twice.**
`wrapper/reconstructor.py:215` and `pointcloud/feedforward/base.py:968` both
carry `sorted(...)` over `{".png", ".jpg", ".jpeg"}`.

**`search_radius` has drifted.** `configs/base.yaml` says 7; the code default is
3. Nothing reconciles them.

**Docstrings are essays.** `compute_parallax` runs 20 lines before the
signature is explained; `qa.py`'s module docstring is 15. Most functions never
say what their arguments are.

---

## 2. Storage: a COLMAP-style image directory

### 2.1 Layout

```
<scene>/
  images/frame_000000.png        # canonical frames, PNG compression=1
  frames.json                    # selection records + provenance
  video_quality_report.json      # unchanged
  <backend>/...
```

`images/` at scene root is the COLMAP convention and the directory every tool in
the ecosystem consumes — COLMAP, hloc, glomap, nerfstudio, 3DGS/gsplat all take
`image_path` plus an optional `sparse/` model. The repo already writes
`sparse/0/{cameras,images,points3D}.bin` from three places; this makes the other
half of that contract native instead of staged.

`frame_idx` stays encoded in the filename. It is already the join key
everywhere — `FrameStore.frame_idx_from_path`, `geometry/metrics.py`'s
source-frame join, and the sfm contract that COLMAP image names must be the
`frame_NNNNNN` stems.

### 2.2 Format: PNG, compression level 1

Measured on three real GH010229 frames (1920x1080, raw 6.22 MB/frame):

| format | ms/frame | MB/frame | 300 frames | PSNR vs source |
|---|---|---|---|---|
| jpg q95 | 168 | 0.80 | 0.24 GB | 44.1 dB |
| jpg q100 | 150 | 1.46 | 0.44 GB | 50.2 dB |
| **png c=1** | **174** | **2.67** | **0.80 GB, 52 s** | lossless |
| png c=3 | 240 | 2.53 | 0.76 GB, 72 s | lossless |
| png c=6 | 527 | 2.47 | 0.74 GB, 158 s | lossless |
| png c=9 | 1832 | 2.38 | 0.71 GB, 550 s | lossless |
| zarr today | — | 5.00 | 1.50 GB | lossless |

PNG at compression 1 — OpenCV's default. Lossless, so every measured number in
the repo (splats PSNR, mesh, ATE) stays comparable against its published value.
Level 9 costs 10x the time for 11% of the size. Read-back is 52 ms/frame.

Still 1.9x smaller than `frames.zarr`, and it removes the second copy: the sfm
path's 0.247 GB of staged JPEGs disappears, so a GH010229 sfm scene goes from
1.75 GB of frames to 0.80 GB.

JPEG was considered and rejected: 3.3x smaller again, and what the ecosystem
trains on, but lossy at 44 dB, which would shift every published splats number
and require a re-measure this design is otherwise careful to avoid.

### 2.3 `frames.json`

COLMAP has no slot for selection records or capture provenance, and they cannot
fold into `video_quality_report.json` — that file is bound report-only (no
thresholds, no verdicts) and a selection record is a verdict.

```json
{
  "schema_version": 2,
  "provenance": {
    "video_path": "...", "video_mtime": 1786519758.0,
    "method": "fps", "fps": 2.0, "max_frames": 300,
    "vda_context_fps": null,
    "undistort": {
      "camera":             {"model": "OPENCV",  "width": 1920, "height": 1080, "params": [...]},
      "undistorted_camera": {"model": "PINHOLE", "width": 2349, "height": 1139, "params": [...]}
    }
  },
  "frames": [{"frame_idx": 0, "blur_score": 8123.4}, ...]
}
```

Row-oriented `frames`, not the columnar zarr arrays: the list is at most a few
hundred entries and a reader wants one frame's record, not one column.

Both cameras are the four-key `pycolmap.Camera` JSON dict, which round-trips
exactly (verified). There is no `roi` key — section 5 removes the crop it
described.

### 2.4 API: `preproc/frames.py`, no class

`FrameStore` is deleted. Four module functions:

| function | replaces |
|---|---|
| `write_frames(dir, frames, records, provenance)` | `FrameStore.create` |
| `frame_paths(dir) -> list[Path]` | `frame_indices`, `export` |
| `read_frames(dir, idxs=None) -> np.ndarray` | `images`, `image`, `image_by_frame_idx` |
| `read_manifest(dir) -> dict` | `record`, `provenance` |

All four take the scene's `images/` directory as `dir`. Frames are RGB uint8
`(H, W, 3)` at both boundaries — `write_frames` converts to BGR for
`cv2.imwrite` and `read_frames` converts back, so no caller handles BGR.
`read_frames(dir)` returns `(N, H, W, 3)` in filename order; `idxs` selects by
`frame_idx`, not by position, and raises on an index the directory does not
hold.

`frame_paths` is also the repo's single image-extension listing: the duplicated
`sorted(...)` over `{".png", ".jpg", ".jpeg"}` in `wrapper/reconstructor.py:215`
and `pointcloud/feedforward/base.py:968` both become calls to it.

`frame_idx_from_path` moves here unchanged. `has_frame_idx` is dropped — its two
callers become a membership test against `frame_paths`.

`export()` has no replacement because the directory *is* the export. Two
round-trips die with it:

- `_run_sfm`'s `frames.zarr -> <scene>/instantsfm/images/frame_NNNNNN.jpg`
  staging step, which becomes `image_path=<scene>/images`. VDA's depth naming
  (`depth_vda/images/npy/<stem>.npy`) is stem-keyed and unaffected.
- `BaseFeedforwardCreator`'s transient `export(tmp)` for path-locked model
  preprocessing, which becomes the same path.

Consumers that only need paths — instantsfm, splats, hloc, verification — take
`images_dir: Path` and never touch the manifest. `_LazyFrames` becomes an
`lru_cache` over `cv2.imread`.

### 2.5 What was checked for reuse and rejected

Nothing in pycolmap, imageio or PIL writes an image directory from in-memory
arrays; it is a two-line `cv2.imwrite` loop. `pycolmap.Bitmap.write` exists and
is 3.4x slower than `cv2.imwrite` on the same PNG (598 ms vs 174 ms), so the
dependency hop buys nothing.

---

## 3. Quality filter: robust, adaptive, and it fires

```python
def filter_frame_quality(report, *, sharpness_k=2.0, max_clipped_frac=0.25) -> np.ndarray
```

Returns a boolean mask over the report's rows, `True` = keep, indexed by report
row (which is source frame order). Both rules must pass.

**Sharpness — robust z-score on log(laplacian).** Cut a frame when
`log(lap) < median(log lap) - k * MAD`. Laplacian variance has no absolute
meaning: it scales with resolution, texture density and content, which is
exactly why a fixed 50 could never work. A robust z-score is scale-free, so one
default transfers across cameras and scenes.

The MAD comes from `scipy.stats.median_abs_deviation(x, scale="normal")`, not a
hand-rolled `1.4826 * median(|x - median|)`. scipy is already a hard dependency;
`scale="normal"` is the same 1.4826 constant, named.

Measured, with the independent Crete-Roffet `blur` column as a control:

| rule | GH010229 cut | tutorial cut | GH blur cut / kept |
|---|---|---|---|
| `lap < 50` (today) | 0.0% | 0.0% | — |
| percentile bottom 10% | 10.0% | 10.0% | 0.266 / 0.228 |
| **robust z < -2 (MAD)** | **5.1%** | **14.5%** | **0.271 / 0.229** |
| local: `lap < 0.6x` nbhd median | 0.8% | 1.6% | 0.260 / 0.231 |

Every rule cuts frames whose independently-measured perceptual blur is higher
than the kept set, so all three select real blur rather than low texture. The
robust rule is chosen because it is the only one that *responds* to the
footage — 5.1% against 14.5% tracks the genuine quality gap between the two
videos, where a percentile cuts its quota from flawless footage and takes only
its quota from bad footage.

**Clipping — absolute, because destroyed pixels are absolute.**
`clipped_low_frac + clipped_high_frac <= 0.25`. Unlike brightness, a pixel at 0
or 255 recorded nothing recoverable, so a fixed ceiling is meaningful. Measured
range: GH010229 crushes shadows up to 0.211, tutorial blows highlights up to
0.056. At 0.25 neither video loses a frame, which is the correct answer — neither
has a destroyed frame.

**Deleted:** `laplacian_min`, `exposure_mean_range`, `exposure_min_std` (all
three measurably never fire) and `blur_max` (off by default, and its own
docstring documents it as a trap — Crete-Roffet saturates at 1.0 on any
low-detail frame, so thresholding it discards sharp frames of plain surfaces).

`qa.py` still measures everything and still writes no verdict.

---

## 4. Sampling: filter first, then sample

The mask and the VDA `candidates` grid have always been the same mechanism — a
restriction on which source indices may be selected — expressed twice, one as a
boolean array consulted inside a window argmax and the other as a grid the
window is cut from. They become one array.

```python
eligible = np.flatnonzero(filter_frame_quality(report, **(quality or {})))
if candidates is not None:
    eligible = np.intersect1d(eligible, candidates)
```

Then:

- **`sample_uniform(max_frames=N)`** — N evenly spaced picks from `eligible`.
  The COUNT stays the contract; a pool smaller than N returns the whole pool
  and logs the shortfall.
- **`sample_fps(fps=f)`** — constant-time targets from `context_indices`, each
  snapped to the nearest eligible index. The SPACING stays the contract, and
  the `[min_frames, max_frames]` re-spread band is unchanged.
- **`sample_optical_flow`** — streams as today and skips ineligible frames, so
  a bad frame never becomes the reference the next frames are scored against.
  Unlike today, the gate now actually rejects.

**Deleted:** `_sample_by_quality` in full — window construction, radius capping,
the grid-steps-versus-source-frames spacing branch, the sharpest-in-window
argmax, the dedup pass and its collapse warning. `preproc.search_radius` leaves
`base.yaml` (where it says 7) and the code (where it defaults to 3); the drift
is evidence that nothing depended on either value.

Sampling from a filtered pool means uniform spacing is even in pool index rather
than in time, so selections bunch either side of an excised blurry stretch. That
is the intended behaviour: the alternative is spending frame budget inside
footage the filter just condemned.

`context_indices` moves from `video.py` to `sampling.py` — it computes a
selection grid, which is this module's job.

---

## 5. Undistortion: pycolmap picks the camera, cv2 moves the pixels

Two module functions. No class, no `DistortionProfile`, no `TemporaryDirectory`.

This is a deliberate deviation from the brief's "single function": calibration
and warping have different costs, different inputs and different call sites, and
calibration's result is cached in provenance for `decode_context` to reuse. The
brief's actual target — the class and its 55 lines of parameter marshalling — is
still deleted.

```python
def calibrate_camera(images_dir: Path, *, max_frames: int = 60) -> pycolmap.Camera

def undistort_frames(
    frames: np.ndarray,
    camera: pycolmap.Camera,
) -> tuple[np.ndarray, pycolmap.Camera]
```

### 5.1 Calibration reads `images/` directly

`pycolmap.extract_features` takes an `image_names=[...]` subset argument
(confirmed present in the 4.0.4 signature). Calibration therefore points at the
scene's own `images/` directory and names the <= 60 stems it wants. The staging
`TemporaryDirectory`, its two `mkdir`s and its 60 JPEG re-encodes are deleted.
SIFT thread pinning (`_SIFT_NUM_THREADS = 8`, against the cgroup cap) and the
`>= 0.6 * len(idxs)` registration floor are unchanged.

This forces an ordering change in `extract_frames`: `images/` must be written
before calibration runs, so an `undistort: true` run encodes PNGs twice — once
distorted for the calibrator, once undistorted as the final output. Only that
path pays it, the default is `undistort: false`, and such runs already pay a
multi-minute pycolmap SfM.

### 5.2 `pycolmap.Camera` replaces `DistortionProfile`

It is what `estimate_camera_distortion` already receives and currently discards:
`.calibration_matrix()` is `.K`, `.params` is the coefficient vector,
`.model_name` survives, `.width`/`.height` are the dims, and a four-key JSON dict
(`model`, `width`, `height`, `params`) round-trips exactly — verified. It is also
the type COLMAP tooling accepts directly, so a downstream consumer needs no
conversion.

### 5.3 The output camera comes from `pycolmap.undistort_camera`

```python
new_cam = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), camera)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new,
                                         (new_cam.width, new_cam.height), cv2.CV_32FC1)
out = np.stack([cv2.remap(f, map1, map2, cv2.INTER_LINEAR) for f in frames])
```

`undistort_camera` is camera-only — no pixels, **1.7 ms** measured. It deletes
the block that exists purely to answer "what camera did I just produce":
`getOptimalNewCameraMatrix`, `cv2.undistort`, the even-ROI crop, the `K` shifted
by the crop offset, and `roi` in the return signature. That crop-offset
arithmetic is a classic source of silent principal-point error; removing it
removes the bug class.

**This changes the output framing, deliberately.** The two libraries answer the
same question with opposite knobs, measured on a GoPro-like OPENCV camera
`[0.62w, 0.62w, w/2-8, h/2+5, -0.25, 0.06, 0.0008, -0.0004]` at 1920x1080:

| | COLMAP (`blank_pixels` 0.0 / 0.5 / 1.0) | cv2 (`alpha` 0.0 / 0.5 / 1.0) |
|---|---|---|
| canvas | 2349x1139 / 2440x1268 / 2539x1429 | 1920x1080 / 1857x973 / 1796x866 |
| focal | **1190.4 at every setting** | (966.9, 1125.7) / (935.9, 1014.8) / (904.9, 903.8) |

COLMAP fixes the focal and grows the canvas; cv2 fixes the canvas and shrinks
the focal. Barrel distortion (`k1 = -0.25`) pushes content outward when
undistorted, so the valid region is wider than the sensor rectangle — COLMAP
grows the canvas to hold it at native focal, cv2 instead drops the focal ~19% to
fit the canvas it already has.

For this camera cv2 at `alpha=0` crops nothing (`roi = (0,0,1919,1079)`); it
*downsamples*, anisotropically, to 0.81x in x and 0.95x in y. COLMAP's framing
keeps the image centre at 1:1 with the sensor. That is a real detail gain, and
it is the reason for taking pycolmap's camera. The periphery is interpolated
either way — the larger canvas invents no detail out there.

### 5.4 The pixel path stays cv2

| | cv2 `remap` | `pycolmap.undistort_image` |
|---|---|---|
| per frame, warm | **27.6 ms** | **677 ms (24.5x)** |
| 300 frames | 8 s | 203 s |

Earlier measurement on a more strongly distorted camera put the same ratio at
60x; the honest figure is **24-60x depending on distortion strength**. Either
end rules it out for a 300-frame path.

There is no accuracy argument on the other side. Both libraries build the
destination-to-source map from the closed-form *forward* distortion polynomial —
no iteration in either. `cv2.initUndistortRectifyMap` was checked against an
independent closed-form projection and agreed to max `|dx|` 6.06e-05 px, `|dy|`
6.05e-05 px, which is float32 map quantization. Same model, same math.

pycolmap *is* far more accurate at the iterative *point* inverse
(`cam_from_img`: max 5.4e-08 px, against `cv2.undistortPoints`' default max
0.281 px), but the repo makes no `cv2.undistortPoints` call anywhere — grepped
across `collab_splats/` and `evals/`. The trap does not apply here.

`pycolmap.undistort_images` — the directory form, equal to
`colmap image_undistorter`, which nerfstudio and 3DGS run — would delete the most
code of all, but it takes a sparse reconstruction as `input_path` and undistorts
only the images that reconstruction contains. At preproc time the only
reconstruction is calibration's, over <= 60 frames. Wrong stage order.

---

## 6. `video.py`: PyAV for pixels

`av` 17.0.1 is already installed, transitively via nerfstudio; this makes it a
direct dependency in `pyproject.toml`. Every ffmpeg subprocess that decodes
pixels is deleted — the rawvideo pipes, the `select=eq(n\,i)+...` filter-graph
string building, the `-ss (start-0.5)/fps` seek guard, the fixed-size stdin
reads and the terminate/wait cleanup, roughly 200 lines.

Measured, **pixel-identical at every point tested (maxdiff 0)**:

| operation | ffmpeg pipe | PyAV | speedup |
|---|---|---|---|
| sequential decode, 400 frames (GoPro) | 6.21 s | 2.84 s | 2.19x |
| sequential decode (data/tutorial) | — | — | 4.9x |
| scattered indices, 13115 frames (GH010229) | 89.2 s | 76.2 s (scan) | 1.17x |
| single random frame | — | — | 10x |

`iter_frames` becomes a decode loop with an optional index set. **Seek-per-index
is not used for scattered reads** — measured at 97.4 s on GH010229, slower than
both the pipe and a linear scan, because every seek lands on a keyframe and
re-decodes forward. Seek is used only for `extract_frame`, where one frame is
wanted and the scan would be the whole file.

`extract_frame` becomes seek plus one decode plus a BGR-to-RGB convert, deleting
the duplicated ffmpeg command build and the second copy of the PTS-rounding
guard. Its consumers (dashboard query frames, `viz.plot_frame_extremes`) are
unaffected.

**`get_video_info` keeps its `ffprobe` call, for rotation only.** PyAV 17
exposes `Type.DISPLAYMATRIX` in its enum but has no stream-level side-data
accessor — verified: `stream.side_data` raises `AttributeError`, and frame
`side_data` carries only per-frame SEI. PyAV therefore cannot see container
rotation, and unlike the ffmpeg CLI it does not auto-rotate. Dropping `ffprobe`
would silently stop rotating phone footage.

The cost is one 3.5 s call per video (measured; it is process startup and
container parse, not frame counting — `-count_frames` and a rotation-only
`-show_entries` both cost the same 3.5 s). Against preproc's 98 s cold path that
is 3.5%, paid once. PyAV's own metadata is 110 ms and exact
(`stream.frames = 2388`, matching `nb_frames`), so if a rotated-footage fixture
later shows our sources never carry a display matrix, the call is deletable then.
PyAV would make probing 18-23x faster, but only by dropping rotation support;
that speedup is deliberately not taken here.

Moves out:

- `context_indices` -> `sampling.py` (section 4)
- `decode_context` -> `pointcloud/sfm.py`, its only consumer. It is a VDA stage
  (`out_short_side=518` is VDA's native grid), not a decode primitive, and the
  move deletes the lazy `from ... import undistort_frames` that only exists
  because the function sits in the wrong file.

The BGR/RGB convention is stated once in the module docstring rather than
re-derived per function: `iter_frames` yields BGR because cv2 consumes it;
everything that returns a frame to a caller returns RGB.

---

## 7. `qa.py`: one pair-motion function

`match_descriptors`, `compute_translation` and `compute_parallax` collapse into:

```python
def compute_pair_motion(feat_a, feat_b) -> dict
```

Each of the three has exactly one caller, and that caller runs all three in
sequence on the same pair; the intermediate point arrays are never used for
anything else. The returned dict carries the same keys the report already
writes, so `video_quality_report.json` is byte-compatible.

`detect_orb` stays separate. That split is load-bearing and measured: detecting
once per frame and reusing the descriptors across both pair comparisons is 1.34x
faster than detecting per pair.

Everything else in `qa.py` is unchanged. In particular the worker thread pin
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `cv2.setNumThreads(1)`) stays,
with its rationale moved from a shouting comment to a plain one: unpinned, the
process pool measured 0.67x — slower than serial.

---

## 8. `viz.py`: delete the two dead plots

`plot_disparity_sensitivity` and `plot_quality_examples` are deleted, 85 lines.

Neither has a pipeline caller. `plot_quality_examples` does not currently work:
the shipped tutorial notebook passes it a `_VideoFrameLookup` where it expects a
`FrameStore`, and the resulting traceback is committed in the `.ipynb`. It also
hardcodes `lap < 50.0` — the third copy of the threshold section 3 deletes — so
even when reached, both of its "Rejected" rows render empty on every real video.

The remaining nine plotting functions keep their `FrameStore` arguments replaced
by `images_dir: Path` per section 2.4.

---

## 9. Docstrings

Every public function and class:

```
"""
One-line imperative summary. No restating the name.

Args:
    frames: RGB uint8 (N, H, W, 3) frames, all the same size.
    camera: distorted camera the frames were captured with.

Returns:
    (frames, undistorted_camera) — RGB frames on the undistorted canvas, and
    the pinhole camera they are now in.
"""
```

Module docstrings are one to three lines. Rationale that is not part of the API
contract — why bitshuffle hurts, why the thread pin is load-bearing, why
Crete-Roffet saturates — moves to block comments at the code it explains, or to
this spec. Nothing measured is lost; it stops being the first thing a reader
must scroll past to find the signature.

The `Args:`/`Returns:` requirement is added to `CLAUDE.md` Code Style, so it
binds the modules cleaned after this one.

---

## 10. Call-site changes

| file | change |
|---|---|
| `wrapper/reconstructor.py` | `extract_frames` splits into `_frames_from_dir` / `_frames_from_video`; the `method = frame_selection` no-op alias is dropped; `images/` + `frames.json` are written before calibration; `_apply_undistortion` folds into `calibrate_camera` + `undistort_frames`; `_run_sfm` drops the JPEG staging step; the extension listing at L215 becomes `frames.frame_paths`; `_LazyFrames` becomes `lru_cache(cv2.imread)` |
| `pointcloud/feedforward/base.py` | `_decode_source` takes `images_dir`; transient `export(tmp)` deleted; the extension listing at L968 becomes `frames.frame_paths` |
| `pointcloud/sfm.py` | gains `decode_context`; `image_path` points at `<scene>/images` |
| `semantics/features/base.py` | `read_frames` / `frame_paths` in place of `FrameStore.open(...).image(i)` |
| `geometry/metrics.py` | `read_frames(dir)[:n]`; `frame_idx_from_path` import moves |
| `geometry/loop_closure/wrapper.py` | `FrameStore \| Path` union collapses to `Path` |
| `dashboard/pipeline.py`, `dashboard/localize.py` | same, plus the `_local_ref_paths` thumbnail dir now redundant with `images/` |
| `evals/datasets.py`, `evals/scripts/eval_splats.py` | `read_frames` |
| `remote/sources.py` | `PUSH_EXCLUDES`/`PULL_EXCLUDES` and their comment block rewritten for `images/**` |
| `configs/base.yaml` | `search_radius` removed; `quality` block gains `sharpness_k`, `max_clipped_frac` |
| `pyproject.toml` | `av` promoted from transitive to direct dependency |
| `docs/.../tutorial notebook` | `plot_quality_examples` cell removed with the function |

`extract_frames` currently takes ten parameters and branches on directory versus
video throughout its body. The split gives each branch its own function; the
public entry point keeps its signature so no caller changes.

### Line tally

| module | now | after |
|---|---|---|
| `frame_store.py` -> `frames.py` | 146 | ~90 |
| `sampling.py` | 521 | ~300 |
| `undistort.py` | 222 | ~110 |
| `video.py` | 345 | ~145 |
| `qa.py` | 510 | ~415 |
| `viz.py`, `__init__.py` | 524 | ~425 |
| **preproc total** | **2268** | **~1485** |

Plus the call-site deletions above — the JPEG staging loop, the transient
export, the duplicated extension listings, the `FrameStore | Path` unions and the
`_LazyFrames` class.

### Migration

Existing scenes hold `frames.zarr` and no `images/`. Re-running preproc
re-decodes the source video (98 s cold per scene) and the processed bucket holds
several, so `scripts/migrate_frames_zarr.py` reads a store and writes `images/` +
`frames.json` with no decode. The store format itself is a hard rename with no
fallback and no resolver, matching the `feedforward.zarr -> pointcloud.zarr`
precedent: a reader that finds only `frames.zarr` raises and names the script.

---

## 11. Risks

**Selection changes on every scene.** Every measured number in the repo — splats
PSNR, mesh component counts, ATE — was produced with a no-op quality filter and
window substitution. Sections 3 and 4 change which frames are selected.

*Gate: a GH010229 A/B (old selection vs new, everything else fixed) must show no
regression in splats PSNR before the new sampling is the default.* This blocks
the phase, and is the reason the plan is phased at all.

**Undistorted output reframes.** Section 5.3 changes undistorted frames from
1920x1080 at reduced focal to 2349x1139 at native focal — 1.29x the pixels.
`undistort` defaults to `false`, so the published splats/mesh/ATE numbers are
untouched, but any previous `undistort: true` run is no longer comparable, and
the larger canvas costs splat VRAM. The feedforward path resizes to the model
grid anyway, which may consume the sharpness gain before it reaches a splat;
that is worth measuring but does not block, since the framing is strictly more
information than the alternative.

**No rotated-video fixture exists.** `data/tutorial` is natively portrait
(1080x1920) with no `tags.rotate` and no display-matrix side data, and no other
video in the repo carries rotation either. Section 6 keeps `ffprobe` precisely
because PyAV cannot see rotation, but that path is currently untested against
real rotated footage. *A synthesized fixture (`ffmpeg -display_rotation 90`) is
required before the PyAV phase lands.*

**Double PNG encode on `undistort: true`.** Section 5.1's ordering writes
`images/` twice on that path. Bounded, opt-in, and small against the pycolmap SfM
those runs already pay.

**PNG encode adds ~52 s per 300-frame scene** on top of preproc's measured 98 s
cold / 29 s warm. Acceptable; parallelizable over the existing `n_workers`
process pool if it proves to matter.

**Breadth.** 76 `FrameStore` call sites plus config, remote excludes, dashboard,
evals and tests. Mitigated by phasing (section 12) and by the dashboard `--smoke`
gate, which is mandatory before any commit touching it.

---

## 12. Phasing

Each phase leaves the suite green and the dashboard smoke-passing.

- **A — storage.** `preproc/frames.py`, `images/` + `frames.json`, all 76 call
  sites, the shared `frame_paths`, migration script, remote excludes. Pure
  refactor: byte-identical frame selection, so any behavioural difference is a
  bug in this phase.
- **B — quality and sampling.** Robust filter, filter-first pool,
  `_sample_by_quality` deleted, `search_radius` removed. **Gated on the
  GH010229 A/B.**
- **C — undistort.** `pycolmap.Camera`, `calibrate_camera` over `images/` via
  `image_names`, `undistort_camera` + `cv2.remap`, `extract_frames` split.
- **D — video.** PyAV decode, `context_indices` and `decode_context` relocated,
  `extract_frame` collapsed, `av` promoted to a direct dependency. **Gated on
  the rotated-video fixture.**
- **E — dead code and docstrings.** The two `viz.py` plots and the notebook cell,
  the `qa.py` pair-motion collapse, then the mechanical docstring pass and the
  `CLAUDE.md` rule.

---

## 13. Testing

`tests/preproc/` mirrors the new module shape: `test_frame_store.py` becomes
`test_frames.py` against the directory API.

`test_undistort.py` asserts the `pycolmap.Camera` JSON round-trip, that
`calibrate_camera` never creates a temporary directory, and — as a golden
test — the undistorted camera's exact dims and params for a fixed input camera.
Section 5.3 is an intentional output change, so the old "crop and `K` unchanged"
assertion is replaced rather than kept.

`test_video.py` gains a rotated fixture (synthesized, per section 11) asserting
that a 90-degree display matrix still produces upright frames, and a
PyAV-versus-ffmpeg pixel-equality test on the tutorial clip pinned at maxdiff 0.

`test_sampling_parity.py` is rewritten rather than preserved: its current target
is the window-substitution behaviour that section 4 deliberately removes. The new
version asserts the pool model — that no ineligible frame is ever selected, that
`uniform` returns exactly N when the pool allows, and that `fps` preserves its
spacing contract.

`test_qa.py` asserts `compute_pair_motion` returns the same keys and values the
three collapsed functions produced on a fixed pair.

Tests covering `plot_disparity_sensitivity` and `plot_quality_examples` are
deleted with the functions.

`test_public_api_surface`'s exact-set assertion is updated to the new `__all__`.
Today's list is 14 names; it loses `DistortionProfile`, `FrameStore` and
`estimate_camera_distortion` (the last folded into `calibrate_camera`) and gains
the six `frames.py` functions plus `calibrate_camera`:

```python
__all__ = [
    "analysis_gray",
    "calibrate_camera",
    "compute_video_quality",
    "extract_frame",
    "filter_frame_quality",
    "frame_idx_from_path",
    "frame_paths",
    "get_video_info",
    "iter_frames",
    "load_video_quality",
    "read_frames",
    "read_manifest",
    "sample_fps",
    "sample_optical_flow",
    "sample_uniform",
    "undistort_frames",
    "write_frames",
]
```

`compute_pair_motion`, `detect_orb`, `context_indices` and `decode_context` are
module-public but not package-exported, matching how the three functions section
7 collapses are treated today.

New: a test that `filter_frame_quality` actually cuts frames on a fixture with a
soft tail — the regression this whole section exists to prevent.
