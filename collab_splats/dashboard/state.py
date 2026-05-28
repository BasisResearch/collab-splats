from __future__ import annotations

from pathlib import Path

import param


class AppState(param.Parameterized):
    """Shared data bus passed between all dashboard panes.

    Panes observe fields via param.watch — downstream panes auto-enable
    when upstream data arrives (e.g. output_dir set by PreprocessPane).
    """

    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames_zarr_path = param.Parameter(default=None)   # Path to output_dir/frames.zarr
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
