"""
Video preprocessing: probe and decode, measure quality, select and store keyframes.

- measure then select: qa.load_video_quality measures and caches a report, the samplers pick from it
- frames.write_frames stores picks as images/frame_NNNNNN.png plus frames.json
- plots live in preproc.viz, not re-exported, so pipeline imports skip matplotlib
"""

from collab_splats.preproc.frames import (
    frame_idx_from_path,
    frame_paths,
    read_frames,
    read_manifest,
    write_frames,
)
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
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
from collab_splats.preproc.video import extract_frame, get_video_info, iter_frames

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
