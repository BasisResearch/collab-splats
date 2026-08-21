"""Video preprocessing: decode (video), capture quality (qa), frame selection (sampling).

Plots live in collab_splats.preproc.viz and are deliberately not re-exported
(keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.qa import (
    check_frame_quality,
    compute_blur_score,
    compute_video_quality,
)
from collab_splats.preproc.sampling import sample_frames, score_frames
from collab_splats.preproc.video import extract_frame, get_video_info

__all__ = [
    "FrameStore",
    "sample_frames",
    "score_frames",
    "get_video_info",
    "extract_frame",
    "compute_blur_score",
    "check_frame_quality",
    "compute_video_quality",
]
