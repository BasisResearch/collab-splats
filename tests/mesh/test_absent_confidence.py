"""
Absent-confidence seams.

A pointcloud.zarr may carry no confidence array (absent, never zeros — e.g. one
written by an SfM backend). Three seams must tolerate that: mesh fusion
(_feedforward_to_tsdf_inputs), feature lifting (lift_features), and the splats
depth-targets block in Reconstructor.splats() (sfm scenes use the same zarr
path once depth is aligned (depth_scale attr)).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import lift_features
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor


def _result_no_confidence():
    """Minimal FeedforwardResult with confidence=None: 2 frames, 8x8 model res."""
    n, h, w = 2, 8, 8
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32),
        model_width=w,
        model_height=h,
        images=np.zeros((n, 3, h, w), dtype=np.float32),
        depth=np.ones((n, h, w), dtype=np.float32),
    )


def test_tsdf_inputs_skip_masking_when_confidence_absent(caplog):
    """conf_percentile set + no confidence -> proceed unmasked with a log, not ValueError."""
    result = _result_no_confidence()
    with caplog.at_level("INFO"):
        out = _feedforward_to_tsdf_inputs(result, conf_percentile=20)
    assert out is not None
    depths, _, _, _ = out
    np.testing.assert_array_equal(depths, result.depth)  # unmasked
    assert any("no confidence" in r.message for r in caplog.records)


def test_lift_features_uniform_weights_when_confidence_absent():
    """Absent confidence must fall back to uniform per-frame weights, not assert or crash.

    Point 0 sits at (0, 0, 1) in world/camera space (identity extrinsics), which
    K = [[8,0,4],[0,8,4],[0,0,1]] projects to pixel (4, 4) exactly — the grid_sample
    knot point, so no bilinear blending. depth=1 everywhere matches that projection,
    so visibility passes in both frames. With confidence absent both frames get
    weight 1.0, so the lifted feature must equal the plain mean of the two per-frame
    samples at that pixel, not a confidence-weighted average.
    """
    result = _result_no_confidence()
    result.points = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    result.pixel_indices = np.zeros((5, 3), dtype=np.int32)

    # Distinguishable, per-frame values at pixel (row=4, col=4); the other 4 points
    # (at the origin, z=0) get zero visibility weight and fall back to a source sample.
    fmaps = [torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)]
    fmaps[0][:, 4, 4] = 1.0
    fmaps[1][:, 4, 4] = 3.0

    feats = lift_features(fmaps, result)
    assert feats.shape == (5, 4)
    expected = torch.full((4,), 2.0)  # uniform-weight mean of 1.0 and 3.0
    torch.testing.assert_close(feats[0], expected, atol=1e-4, rtol=0)


def _stub_reconstructor_for_splats(tmp_path, n_views=2, height=4, width=4):
    """Minimal Reconstructor stub for exercising the splats() depth-targets block.

    Same lightweight pattern as tests/wrapper/test_splats_stage.py's
    _stub_reconstructor: patch train() + FeedforwardResult.load_zarr, no real
    training or zarr reconstruction needed.
    """
    recon = Reconstructor.__new__(Reconstructor)
    recon.config = {
        "output_path": str(tmp_path),
        "pointcloud": {"method": "feedforward", "backend": "vggtx"},
        "mesh": {"conf_percentile": 20},
        "splats": {"enabled": True, "max_steps": 1, "losses": {"depth": {"weight": 0.1}}},
    }
    recon._stage_output_exists = lambda stage: False

    frames = np.stack([np.full((height, width, 3), view * 10, np.uint8) for view in range(n_views)])
    records = [{"frame_idx": view} for view in range(n_views)]
    FrameStore.create(recon.frames_zarr, frames, records, provenance={"video_path": "v"})
    image_paths = [Path(f"frame_{view:06d}.jpg") for view in range(n_views)]
    recon._resolve_result = lambda: SimpleNamespace(
        image_paths=image_paths,
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1)),
        points=np.zeros((10, 3), np.float32),
        colors=np.zeros((10, 3), np.uint8),
    )
    return recon


def test_splats_depth_targets_skip_masking_when_confidence_absent(tmp_path, caplog):
    """pointcloud.zarr without confidence; conf_percentile set -> log and skip masking."""
    recon = _stub_reconstructor_for_splats(tmp_path)
    feedforward = SimpleNamespace(
        image_paths=[Path(f"frame_{view:06d}.jpg") for view in range(2)],
        depth=np.ones((2, 4, 4), dtype=np.float32),
        confidence=None,
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)

    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
        caplog.at_level("INFO"),
    ):
        recon.splats()

    depth_targets = train.call_args.kwargs["depth_targets"]
    np.testing.assert_array_equal(depth_targets, feedforward.depth)  # unmasked
    assert any("no confidence" in r.message for r in caplog.records)
