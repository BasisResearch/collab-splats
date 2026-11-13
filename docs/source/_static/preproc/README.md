# Preproc quality-report plots

Rendered from the tutorial video's report — `data/tutorial/video_quality/video_quality_report.json`,
2388 frames @ 23.98 fps — by the plotters in `collab_splats/preproc/viz.py`.

| file | plotter |
|---|---|
| `photometric.png` | `plot_photometric` — blur, laplacian, exposure, clipping over time |
| `motion.png` | `plot_motion` — per-pair translation, parallax and match count |
| `extremes-blur.png` | `plot_frame_extremes(column="blur", n=4)` |
| `extremes-exposure_mean.png` | `plot_frame_extremes(column="exposure_mean", n=4)` |
| `correlation-blur-translation_px.png` | `plot_correlation("blur", "translation_px")` |
| `correlation-laplacian-n_matches.png` | `plot_correlation("laplacian", "n_matches")` |

Regenerate after `compute_video_quality` changes:

```python
from collab_splats.preproc import load_video_quality, viz

video = "data/tutorial/tutorial_example-video.mp4"
report = load_video_quality(video, "data/tutorial/video_quality/video_quality_report.json")
out = "docs/source/_static/preproc"
viz.plot_photometric(report, out)
viz.plot_motion(report, out)
# montages ship in-repo at dpi=70; the default 150 costs ~9 MB each
viz.plot_frame_extremes(report, video, out, column="blur", n=4, dpi=70)
viz.plot_frame_extremes(report, video, out, column="exposure_mean", n=4, dpi=70)
viz.plot_correlation(report, "blur", "translation_px", out)
viz.plot_correlation(report, "laplacian", "n_matches", out)
```

The montages are re-encoded through a 256-color palette before committing.
