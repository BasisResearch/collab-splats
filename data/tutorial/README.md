# Tutorial example data

Assets consumed by the notebooks in `docs/source/tutorials/`. Both are committed so
a fresh clone can run the full tutorial with no gcloud access.

## `tutorial_example-video.mp4`
Reconstruction input (nb 01–06). Re-encode of session `2024_02_06` video `C0043`
(1080p, ~100 s), bitrate-reduced to fit GitHub's 100 MB/file limit:

    ffmpeg -i C0043.MP4 -c:v libx264 -b:v 6M -maxrate 6M -bufsize 12M -an \
           tutorial_example-video.mp4

## `tutorial_example-frame.jpg`
External localization query (nb 07). Frame 0 of GoPro video `GX010119` — a *different*
video from the reconstruction — localized against the C0043 map.

## Outputs
Notebooks write regenerated artifacts to `data/outputs/` (gitignored).
