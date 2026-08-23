# Video quality plots — PNGs from `video_quality_report.json`

**Date:** 2026-08-23
**Status:** approved, awaiting implementation plan
**Follows:** `2026-08-20-video-quality-report-design.md` (the report), `2026-08-22-preproc-cleanup-design.md` (measure-then-select)

## Goal

Render the per-frame and per-pair columns of `video_quality_report.json` to static PNGs beside the
report, so the data-dashboard (`collab-data/data_dashboard`, which already serves `.png` through
`file_viewers/image.py`) can show capture quality without running any code. No interactivity.
Notebooks reuse the same functions.

## Implementation principles

- **Reuse, don't add.** Five plotters join `collab_splats/preproc/viz.py` under a new `########`
  section. No new module, no plotting abstraction layer, no shared "figure builder" helper — each
  plotter is a standalone `plt.subplots` block a reader can understand on its own.
- **Raw columns only.** Plots draw the report's columns as shipped. No thresholds, verdicts, smoothing,
  or derived statistics beyond `np.cumsum(translation_px)` — the report is report-only and the plots
  inherit that rule.
- **Existing packages:** matplotlib (already a dependency, already imported by `viz.py`). Nothing new.
- **Headless.** Plotters save and `plt.close`; they never call `plt.show()`. The four existing notebook
  plotters in `viz.py` are untouched.

## Plotters

All five live in `collab_splats/preproc/viz.py`, same signature, same return:

```python
def plot_photometric_blur(report: dict, out_dir: str | Path, *, selected=None) -> Path
def plot_photometric_exposure(report: dict, out_dir: str | Path, *, selected=None) -> Path
def plot_motion_translation(report: dict, out_dir: str | Path, *, selected=None) -> Path
def plot_motion_parallax(report: dict, out_dir: str | Path, *, selected=None) -> Path
def plot_motion_matches(report: dict, out_dir: str | Path, *, selected=None) -> Path
```

- `report`: the dict from `qa.compute_video_quality` / `qa.load_video_quality`. No availability
  check: an unavailable report has no `frames` key and `report["frames"]` raises `KeyError` on its
  own. `extract_frames` never reaches the plotters with one — it refuses an empty selection first.
- `out_dir`: created with `mkdir(parents=True, exist_ok=True)`. The file name is fixed per plotter
  (table below) and the written `Path` is returned.
- `selected`: iterable of source frame indices that made it into `frames.zarr`, or `None`. When given,
  every panel marks each selected frame with a faint full-height line —
  `ax.vlines(t, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)` —
  a sampler-audit overlay showing which frames the reconstruction actually sees. Lines, not
  `axvspan`: a span one frame wide is under a pixel on a long video (13k frames at 12 in × 90 dpi ≈
  0.08 px) and vanishes, while a 1 px line always renders — dense selections read as shading, sparse
  ones as ticks. `None` draws no overlay. **The plots always draw every frame and every pair in the
  report; `selected` only marks which of them were kept.**

x-axis on every panel is wall-clock seconds: `frame_idx / report["video"]["fps"]` (pairs use
`frame_idx_a`). Panels within a file share x (`sharex=True`). `figsize=(12, 2.8 * n_panels)`,
`dpi=90` (matches `collab-data/track_reprojection/report.py`), `fig.tight_layout()`. Every figure
carries `fig.suptitle(f"{Path(video.path).name} — {n_frames} frames @ {fps:.2f} fps, stride {stride}")`
so a PNG viewed alone in the dashboard identifies its source (`n_frames` from the `frames` columns,
`stride` from `report["params"]["motion_stride"]`).

| File | Panels (top → bottom) | Notes |
|---|---|---|
| `photometric-blur.png` | `blur` (y in [0, 1], label "blur (↑ blurrier)") / `laplacian` (label "laplacian var (↑ sharper)") | Opposite directions on purpose; both shown so a saturated `blur` reads against `laplacian` (qa.py `compute_blur` docstring). |
| `photometric-exposure.png` | `exposure_mean` line with `±exposure_std` band (`fill_between`, alpha 0.2) and `exposure_median` dashed / `clipped_low_frac` + `clipped_high_frac` stacked (`stackplot`) | y of the top panel fixed to [0, 255]. |
| `motion-translation.png` | `translation_px` per pair / `np.cumsum` of `translation_px` (nan→0 for the cumsum only; label "path-length proxy (px)") | Failed pairs are `None` in JSON → `np.nan` for plotting; matplotlib leaves a gap. |
| `motion-parallax.png` | `parallax` per pair, with every `None` pair drawn as a red `\|` marker at y=0 | Nulls are the worst pairs — they must stay visible, never dropped (report spec trap 13). |
| `motion-matches.png` | `n_matches` per pair | Single panel. |

Each plotter body is one commented block sequence: `# Unpack columns`, `# Time axis`, `# Panels`,
`# Selected-frame overlay`, `# Save and close` — title, overlay and save are inlined in every plotter
(a few duplicated lines beat a helper layer). Column reads go through `np.asarray(..., dtype=float)`
so `None` becomes `nan` in one place.

## Wiring

`extract_frames` in `collab_splats/wrapper/reconstructor.py` (the video branch only, directly after
`FrameStore.create`) calls all five with `out_dir=frames_zarr.parent` and
`selected=[r["frame_idx"] for r in records]`. Five explicit calls — no registry tuple, no loop. The image-directory branch has no report and writes no
plots.

Import: `from collab_splats.preproc import viz as preproc_viz` at the top of `reconstructor.py`.
This puts matplotlib on `Reconstructor`'s import path; that path already carries torch and pyvista,
and `collab_splats/remote/__init__.py` deliberately does not import `reconstructor` (only
`remote/rerun.py`, a CLI driver, does), so the dashboard fast-bind path stays matplotlib-free. The dashboard's own
`pipeline.py` and `splatter.py` are **not** wired — they keep matplotlib off the serving process.

Output contract: the five PNGs land next to `video_quality_report.json` and `frames.zarr`, ride to
`environments-processed/<scene>/` with the existing push (no exclude change), and are listed in
`configs/README.md` under the processed-scene contract beside `video_quality_report.json`. Re-runs:
the plots are written whenever `extract_frames` runs (it already skips entirely when `frames.zarr`
exists), so they are regenerated exactly when frames are, and never go stale against `frames.zarr`.

## Error handling

- Unavailable report → natural `KeyError` (above); nothing to add.
- Empty `pairs` columns (video shorter than the stride) → the motion plotters still write a PNG with
  empty axes; they do not raise. A one-panel empty plot is a true statement about the report.
- Missing `out_dir` → created.

## Testing

`tests/preproc/test_viz.py` (existing file, new section), flat functions, `matplotlib.use("Agg")`
already set there:

- A synthetic report fixture: 20 frames, fps 10, 10 pairs, one pair with `translation_px` and
  `parallax` `None`.
- For each of the five plotters: returns the expected path, file exists, starts with the PNG magic
  `\x89PNG`, and runs both with `selected=None` and `selected=[3, 7]`.
- Empty `pairs` columns: the three motion plotters still write a file.

`tests/wrapper/test_reconstructor.py`: extend the existing `extract_frames` dispatch test so the
video branch produces all five PNGs beside `frames.zarr`, and the image-dir branch produces none.

## Out of scope

- **Backfill for already-processed scenes.** `extract_frames` skips when `frames.zarr` exists, so
  scenes processed before this change get no PNGs until their frames are re-extracted
  (`preprocess(overwrite=True)`). Deliberate: no existence-probing of five files, no extra stage.

- Interactive or dashboard-native (Panel/Bokeh) plots.
- Reworking the four existing notebook plotters in `viz.py` (still the deferred follow-on from the
  report spec).
- Wiring plots into the dashboard's `pipeline.py` / `splatter.py` preproc paths.
- Any derived statistic (histograms, ECDFs, correlations) — one numpy call away for a reader with the
  JSON, per the report spec.
