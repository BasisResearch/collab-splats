"""
mesh.mask_sky wiring in the mesh stage.

- the mask is applied after the two source arms converge, so both are covered
- the splats arm orders frames by checkpoint image_ids, not filename, and the mask stack
  must be requested in that same order or every mask lands on the wrong frame
"""

from unittest.mock import patch

import numpy as np
import pytest

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.reconstructor import Reconstructor
from tests.reconstructor._stubs import minimal_feedforward_result, stub_mesh_cleanup


def _mesh_reconstructor(tmp_path, **mesh_overrides):
    """
    Reconstructor whose mesh block is base.yaml plus the given overrides.
    """
    mesh_cfg = {"enabled": True, "texture": False, **mesh_overrides}
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "mesh": mesh_cfg,
    }

    return Reconstructor(config)


def _run_feedforward_mesh(tmp_path, fused, sky_return, **mesh_overrides):
    """
    Drive the mesh stage's feedforward arm with everything below create_tsdf_mesh stubbed out.
    """
    rec = _mesh_reconstructor(tmp_path, source="feedforward", **mesh_overrides)
    result = minimal_feedforward_result()

    def spy_fuse(depths, rgbs, c2w, K, **kw):
        fused["depths"] = depths

    with (
        patch.object(
            PointcloudResult, "load_zarr", staticmethod(lambda *a, **k: result)
        ),
        patch(
            "collab_splats.reconstructor.frames.read_frames",
            return_value=np.zeros((2, 8, 8, 3), np.uint8),
        ),
        patch(
            "collab_splats.pointcloud.utils.upsample_depths",
            side_effect=lambda d, r, b: d,
        ),
        patch("collab_splats.reconstructor.compute_tsdf_voxel_size", return_value=0.01),
        patch("collab_splats.reconstructor.create_tsdf_mesh", side_effect=spy_fuse),
        stub_mesh_cleanup(),
        patch("collab_splats.reconstructor.sky_masks", return_value=sky_return) as sky,
    ):
        rec.mesh()

    return sky


def test_mask_sky_zeroes_feedforward_depth_where_sky_and_nowhere_else(tmp_path):
    # Bottom two rows sky; every other pixel must survive bit-identically
    mask = np.zeros((2, 8, 8), bool)
    mask[:, 6:] = True
    fused = {}

    sky = _run_feedforward_mesh(tmp_path, fused, mask, mask_sky=True)

    assert np.all(fused["depths"][:, 6:] == 0.0)
    assert np.all(fused["depths"][:, :6] == 1.0)
    # Feedforward masks the zarr's own frames (minimal_feedforward_result: frames 0 and 1)
    assert sky.call_args.kwargs["idxs"] == [0, 1]


def test_mask_sky_off_never_loads_the_model(tmp_path):
    fused = {}

    sky = _run_feedforward_mesh(tmp_path, fused, None, mask_sky=False)

    assert sky.call_count == 0
    assert np.all(fused["depths"] == 1.0)


def test_mask_sky_rejects_a_mask_that_does_not_match_the_depth_grid(tmp_path):
    with pytest.raises(ValueError, match="Sky masks are"):
        _run_feedforward_mesh(tmp_path, {}, np.zeros((2, 4, 4), bool), mask_sky=True)


def test_mask_sky_asks_for_splats_frames_in_checkpoint_order(tmp_path):
    # The checkpoint's rgbs rows are in image_ids order (7, 0), NOT filename order (0, 7).
    # Masking row 0 must therefore mask frame 7, which only holds if sky_masks was asked
    # for [7, 0] rather than being left to its own filename-order default.
    rendered = (
        np.ones((2, 8, 8), np.float32),
        np.zeros((2, 8, 8, 3), np.uint8),
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        [7, 0],
    )
    # Sky truth keyed by SOURCE frame_idx: frame 7 is sky, frame 0 is not
    # - the stub honours idxs, so the depth assertions below are live
    # - a fixed return_value would ignore idxs, and then only the call_args assertion could
    #   tell filename order from checkpoint order
    per_frame = {0: np.zeros((8, 8), bool), 7: np.ones((8, 8), bool)}

    def fake_sky_masks(images_dir, idxs=None, **kw):
        wanted = sorted(per_frame) if idxs is None else list(idxs)
        return np.stack([per_frame[int(i)] for i in wanted])

    fused = {}

    def spy_fuse(depths, rgbs, c2w, K, **kw):
        fused["depths"] = depths

    # The stage refuses a missing checkpoint before rendering, so one must exist on disk
    rec = _mesh_reconstructor(tmp_path, source="splats", mask_sky=True)
    ckpt = rec.backend_dir / "splats" / "ckpt.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()

    with (
        patch(
            "collab_splats.splats.checkpoint.render_tsdf_inputs", return_value=rendered
        ),
        patch("collab_splats.reconstructor.compute_tsdf_voxel_size", return_value=0.01),
        patch("collab_splats.reconstructor.create_tsdf_mesh", side_effect=spy_fuse),
        stub_mesh_cleanup(),
        patch(
            "collab_splats.reconstructor.sky_masks", side_effect=fake_sky_masks
        ) as sky,
    ):
        rec.mesh()

    assert sky.call_args.kwargs["idxs"] == [7, 0]
    assert np.all(fused["depths"][0] == 0.0)
    assert np.all(fused["depths"][1] == 1.0)
