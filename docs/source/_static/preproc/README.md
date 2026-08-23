# Preproc quality-report plots

Rendered from the tutorial video's report — `data/tutorial/video_quality/video_quality_report.json`,
2388 frames @ 23.98 fps — by the plotters in `collab_splats/preproc/viz.py`.

| file | plotter |
|---|---|
| `photometric.png` | `plot_photometric` — blur, laplacian, exposure, clipping over time |
| `motion.png` | `plot_motion` — per-pair translation, rotation and match count |
| `extremes-blur.png` | `plot_frame_extremes(column="blur", n=4)` |
| `extremes-exposure_mean.png` | `plot_frame_extremes(column="exposure_mean", n=4)` |
| `correlation-blur-translation_px.png` | `plot_correlation("blur", "translation_px")` |
| `correlation-laplacian-n_matches.png` | `plot_correlation("laplacian", "n_matches")` |

Regenerate after `compute_video_quality` changes:

```python
from collab_splats.preproc import load_video_quality, viz

viz._THUMB_DPI = 70  # montages ship in-repo; 150 costs ~9 MB each
report = load_video_quality("data/tutorial/tutorial_example-video.mp4",
                            "data/tutorial/video_quality/video_quality_report.json")
viz.plot_photometric(report, "docs/source/_static/preproc")
```

The montages are re-encoded through a 256-colour palette before committing.
