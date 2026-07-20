"""Video preprocessing: frame sampling, quality gating, frame I/O.

Plots live in collab_splats.preproc.viz and are deliberately not re-exported
(keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.sampling import (
    check_frame_quality,
    compute_blur_score,
    extract_frame,
    get_video_info,
    sample_frames,
    score_frames,
)

__all__ = [
    "FrameStore",
    "sample_frames",
    "score_frames",
    "get_video_info",
    "extract_frame",
    "compute_blur_score",
    "check_frame_quality",
]
