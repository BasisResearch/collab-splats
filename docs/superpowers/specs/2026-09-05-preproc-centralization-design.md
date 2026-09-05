# preproc centralization — design

**Date:** 2026-09-05
**Scope:** `collab_splats/preproc/` and its callers
**Status:** design, awaiting approval

Second per-module cleanup pass, following
[2026-08-22-preproc-cleanup-design.md](2026-08-22-preproc-cleanup-design.md).
That pass split measurement from selection. This one changes what preproc
*stores*, makes the quality filter actually filter, and deletes the machinery
that existed only to work around it.

---

## 1. Motivation

Five problems, four of them measured on real scenes rather than asserted.

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

**Undistortion carries a hand-rolled parameter type that pycolmap already
provides.** `estimate_camera_distortion` receives a `pycolmap.Camera` from the
mapper and immediately unpacks its eight params into a frozen dataclass with
`to_dict`, `from_dict`, `K` and `dist_coeffs` — 55 lines to re-implement what
the Camera already does, while discarding the model name.

**`video.py` mixes decode primitives with pipeline stages.** `extract_frame`
rebuilds the same ffmpeg command `iter_frames` builds. `context_indices` is
selection-grid arithmetic living in a decode module. `decode_context` is 85
lines of undistort-plus-downscale-plus-chunk with a VDA-specific
`out_short_side=518`, a single consumer in `_run_sfm`, and a lazy inline import
of `undistort` that exists only because the function is in the wrong file.

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
      "camera":              {"model": "OPENCV",  "width": 1920, "height": 1080, "params": [...]},
      "undistorted_camera":  {"model": "PINHOLE", "width": 1918, "height": 1078, "params": [...]},
      "roi": [0, 0, 1918, 1078]
    }
  },
  "frames": [{"frame_idx": 0, "blur_score": 8123.4}, ...]
}
```

Row-oriented `frames`, not the columnar zarr arrays: the list is at most a few
hundred entries and a reader wants one frame's record, not one column.

### 2.4 API: `preproc/frames.py`, no class

`FrameStore` is deleted. Four module functions:

| function | replaces |
|---|---|
| `write_frames(dir, frames, records, provenance)` | `FrameStore.create` |
| `frame_paths(dir) -> list[Path]` | `frame_indices`, `export` |
| `read_frames(dir, idxs=None) -> np.ndarray` | `images`, `image`, `image_by_frame_idx` |
| `read_manifest(dir) -> dict` | `record`, `provenance` |

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

**Sharpness — robust z-score on log(laplacian).** Cut a frame when
`log(lap) < median(log lap) - k * MAD`, MAD scaled by 1.4826. Laplacian variance
has no absolute meaning: it scales with resolution, texture density and content,
which is exactly why a fixed 50 could never work. A robust z-score is
scale-free, so one default transfers across cameras and scenes.

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

`qa.py` is unchanged. It still measures everything, still writes no verdict.

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
  The COUNT stays the contract.
- **`sample_fps(fps=f)`** — constant-time targets from `context_indices`, each
  snapped to the nearest eligible index. The SPACING stays the contract, and
  the `[min_frames, max_frames]` re-spread band is unchanged.
- **`sample_optical_flow`** — streams as today and skips ineligible frames, so
  a bad frame never becomes the reference the next frames are scored against.
  Unlike today, the gate now actually rejects.

**Deleted:** `_sample_by_quality` in full — window construction, radius capping,
the grid-steps-versus-source-frames spacing branch, the sharpest-in-window
argmax, the dedup pass and its collapse warning. `preproc.search_radius` leaves
`base.yaml`.

Sampling from a filtered pool means uniform spacing is even in pool index rather
than in time, so selections bunch either side of an excised blurry stretch. That
is the intended behaviour: the alternative is spending frame budget inside
footage the filter just condemned.

`context_indices` moves from `video.py` to `sampling.py` — it computes a
selection grid, which is this module's job.

---

## 5. Undistortion: one function over `pycolmap.Camera`

```python
def undistort_frames(
    frames: list[np.ndarray],
    camera: pycolmap.Camera | None = None,
    *,
    max_calib_frames: int = 60,
) -> tuple[list[np.ndarray], pycolmap.Camera, pycolmap.Camera]:
```

`camera=None` self-calibrates from the frames (pycolmap SIFT, exhaustive
matching, incremental mapping on <= `max_calib_frames`, `_SIFT_NUM_THREADS=8`
against the cgroup cap — all unchanged). Returns the frames, the distorted
camera and the undistorted camera.

`DistortionProfile` is deleted. `pycolmap.Camera` is what
`estimate_camera_distortion` already receives and currently discards:
`.calibration_matrix()` is `.K`, `.params` is the coefficient vector,
`.model_name` survives, `.width`/`.height` are the dims, and a four-key JSON dict
(`model`, `width`, `height`, `params`) round-trips exactly — verified. It is also
the type COLMAP tooling accepts directly, so a downstream consumer needs no
conversion.

The two-callable split survives internally (calibration is expensive and its
result is cached in provenance for `decode_context` to reuse), but the public
surface is one function.

**The pixel path stays cv2.** `pycolmap.undistort_image` was measured as a
replacement and rejected:

| | ours (cv2) | pycolmap |
|---|---|---|
| per frame, warm | **17 ms** | **1029 ms (60x)** |
| 300 frames | 5 s | 309 s |
| output dims | 1078x1918 | 1071x1907 |
| principal point | preserved | force-recentered to (w-1)/2 |
| focal length | rescaled to the valid region | kept at input |
| agreement | — | 17.1 dB, a real framing shift |

60x slower, and the reframing would invalidate the splats numbers that
losslessness was chosen to protect. `pycolmap.undistort_images` (the directory
form, equal to `colmap image_undistorter`, which nerfstudio and 3DGS run)
requires an existing `sparse/` model as input; undistortion here runs at preproc,
before any reconstruction exists. Wrong stage order.

---

## 6. `video.py`: three decode primitives

Keeps `get_video_info`, `iter_frames` and `extract_frame`. `extract_frame` is
rewritten as a thin `iter_frames(video, start=i, count=1)` wrapper plus a BGR to
RGB convert, deleting a duplicated ffmpeg command build and a second copy of the
PTS-rounding seek guard. Its three real consumers (dashboard query frames,
`viz.plot_frame_extremes`) decode arbitrary frames from arbitrary videos and are
unaffected.

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

## 7. Docstrings

Every public function and class:

```
"""
One-line imperative summary. No restating the name.

Args:
    frames: RGB uint8 (H, W, 3) frames, all the same size.
    camera: distorted camera, or None to self-calibrate from `frames`.

Returns:
    (frames, camera, undistorted_camera) — cropped RGB frames, the camera they
    were undistorted from, and the pinhole camera they are now in.
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

## 8. Call-site changes

| file | change |
|---|---|
| `wrapper/reconstructor.py` | `extract_frames` writes `images/` + `frames.json`; `_apply_undistortion` folds into the one `undistort_frames` call; `_run_sfm` drops the JPEG staging step; `_LazyFrames` becomes `lru_cache(cv2.imread)` |
| `pointcloud/feedforward/base.py` | `_decode_source` takes `images_dir`; transient `export(tmp)` deleted |
| `pointcloud/sfm.py` | gains `decode_context`; `image_path` points at `<scene>/images` |
| `semantics/features/base.py` | `read_frames` / `frame_paths` in place of `FrameStore.open(...).image(i)` |
| `geometry/metrics.py` | `read_frames(dir)[:n]`; `frame_idx_from_path` import moves |
| `geometry/loop_closure/wrapper.py` | `FrameStore \| Path` union collapses to `Path` |
| `dashboard/pipeline.py`, `dashboard/localize.py` | same, plus the `_local_ref_paths` thumbnail dir now redundant with `images/` |
| `evals/datasets.py`, `evals/scripts/eval_splats.py` | `read_frames` |
| `remote/sources.py` | `PUSH_EXCLUDES`/`PULL_EXCLUDES` and their comment block rewritten for `images/**` |
| `configs/base.yaml` | `search_radius` removed; `quality` block gains `sharpness_k`, `max_clipped_frac` |

### Line tally

| module | now | after |
|---|---|---|
| `frame_store.py` -> `frames.py` | 146 | ~90 |
| `sampling.py` | 521 | ~300 |
| `undistort.py` | 222 | ~140 |
| `video.py` | 345 | ~250 |
| `qa.py` (docstrings only) | 510 | ~450 |
| `viz.py`, `__init__.py` | 524 | ~510 |
| **preproc total** | **2268** | **~1740** |

Plus the call-site deletions above — the JPEG staging loop, the transient
export, the `FrameStore | Path` unions and the `_LazyFrames` class.

### Migration

Existing scenes hold `frames.zarr` and no `images/`. Re-running preproc
re-decodes the source video (98 s cold per scene) and the processed bucket holds
several, so `scripts/migrate_frames_zarr.py` reads a store and writes `images/` +
`frames.json` with no decode. The store format itself is a hard rename with no
fallback and no resolver, matching the `feedforward.zarr -> pointcloud.zarr`
precedent: a reader that finds only `frames.zarr` raises and names the script.

---

## 9. Risks

**Selection changes on every scene.** Every measured number in the repo — splats
PSNR, mesh component counts, ATE — was produced with a no-op quality filter and
window substitution. Sections 3 and 4 change which frames are selected.

*Gate: a GH010229 A/B (old selection vs new, everything else fixed) must show no
regression in splats PSNR before the new sampling is the default.* This blocks
the phase, and is the reason the plan is phased at all.

**PNG encode adds ~52 s per 300-frame scene** on top of preproc's measured 98 s
cold / 29 s warm. Acceptable; parallelizable over the existing `n_workers`
process pool if it proves to matter.

**Breadth.** 76 `FrameStore` call sites plus config, remote excludes, dashboard,
evals and tests. Mitigated by phasing (section 10) and by the dashboard `--smoke`
gate, which is mandatory before any commit touching it.

---

## 10. Phasing

Each phase leaves the suite green and the dashboard smoke-passing.

- **A — storage.** `preproc/frames.py`, `images/` + `frames.json`, all 76 call
  sites, migration script, remote excludes. Pure refactor: byte-identical frame
  selection, so any behavioural difference is a bug in this phase.
- **B — quality and sampling.** Robust filter, filter-first pool,
  `_sample_by_quality` deleted, `search_radius` removed. **Gated on the
  GH010229 A/B.**
- **C — undistort and video.** `pycolmap.Camera`, one `undistort_frames`,
  `context_indices` and `decode_context` relocated, `extract_frame` collapsed.
- **D — docstrings.** Mechanical pass over the module plus the `CLAUDE.md` rule.

---

## 11. Testing

`tests/preproc/` mirrors the new module shape: `test_frame_store.py` becomes
`test_frames.py` against the directory API; `test_undistort.py` asserts the
`pycolmap.Camera` round-trip and that the cv2 crop and `K` are unchanged from
today (this is a type change, not a numerical one, and the test must prove it).

`test_sampling_parity.py` is rewritten rather than preserved: its current target
is the window-substitution behaviour that section 4 deliberately removes. The new
version asserts the pool model — that no ineligible frame is ever selected, that
`uniform` returns exactly N when the pool allows, and that `fps` preserves its
spacing contract.

`test_public_api_surface`'s exact-set assertion is updated to the new `__all__`.

New: a test that `filter_frame_quality` actually cuts frames on a fixture with a
soft tail — the regression this whole section exists to prevent.
