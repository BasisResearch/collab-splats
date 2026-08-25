"""
Video preprocessing: decode (video), capture quality (qa), frame selection (sampling).

Two steps: qa.compute_video_quality measures the whole video into a report, then
the samplers select from it. Plots live in collab_splats.preproc.viz and are
deliberately not re-exported (keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.qa import (
    analysis_gray,
    compute_video_quality,
    load_video_quality,
)
from collab_splats.preproc.sampling import (
    filter_frame_quality,
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.undistort import (
    DistortionProfile,
    estimate_camera_distortion,
    undistort_frames,
)
from collab_splats.preproc.video import extract_frame, get_video_info, iter_frames

__all__ = [
    "DistortionProfile",
    "FrameStore",
    "analysis_gray",
    "compute_video_quality",
    "estimate_camera_distortion",
    "extract_frame",
    "filter_frame_quality",
    "get_video_info",
    "iter_frames",
    "load_video_quality",
    "sample_fps",
    "sample_optical_flow",
    "sample_uniform",
    "undistort_frames",
]
