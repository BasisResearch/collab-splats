"""
Splats-stage wiring: leaf registration, base.yaml default, and the arrays handed to train().

- Depth targets come from pointcloud.zarr depth for every method — sfm scenes use the
  dense COLMAP-scale-aligned zarr depth through the same path as feedforward backends.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
import zarr

from collab_splats.reconstructor import LEAF_STAGES, STAGES
from collab_splats.splats.trainer import SplatsConfig
from collab_splats.splats.utils import prepare_target
from collab_splats.utils.image import upsample_depths
from tests.reconstructor._stubs import _stub_reconstructor, minimal_feedforward_result

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_splats_is_a_leaf_stage():
    assert "splats" in STAGES
    assert list(STAGES).index("splats") < list(STAGES).index("mesh")
    assert STAGES["splats"] == ("pointcloud",)
    assert "splats" in LEAF_STAGES


def test_base_yaml_defaults():
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())["splats"]
    assert cfg["enabled"] is False and cfg["primitive"] == "3dgs" and cfg["cap_max"] == 1_000_000
    assert set(cfg["losses"]) == {"depth", "normal_consistency", "opacity_reg", "scale_reg", "appearance_reg"}

    # The block must round-trip through from_dict (enabled stripped) and equal dataclass defaults
    parsed = SplatsConfig.from_dict(cfg)
    defaults = SplatsConfig()
    for field in SplatsConfig.__dataclass_fields__:
        assert getattr(parsed, field) == getattr(defaults, field), field
        assert type(getattr(parsed, field)) is type(getattr(defaults, field)), field


def test_splats_stage_assembles_arrays_in_image_path_order(tmp_path):
    recon = _stub_reconstructor(tmp_path)

    # Zarr rows in the result's order (frames 2, 1, 0), per-frame distinguishable depth = frame_idx + 1
    depth = np.stack([np.full((4, 4), view + 1, np.float32) for view in (2, 1, 0)])
    feedforward = SimpleNamespace(
        image_paths=recon.result.image_paths,
        depth=depth,
        confidence=torch.ones(3, 4, 4),
        original_coords=np.tile(np.array([0, 0, 8, 8, 8, 8], np.float32), (3, 1)),  # full-frame box
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.base.PointcloudResult.load_zarr", return_value=feedforward),
    ):
        recon.splats()

    cfg, images, world_to_cam, intrinsics, points, colors, out_dir = train.call_args.args
    depth_targets = train.call_args.kwargs["depth_targets"]
    assert cfg.max_steps == 1
    assert images[0, 0, 0, 0] == 20 and images[2, 0, 0, 0] == 0  # follows image_paths (reversed), not store order
    assert world_to_cam.shape == (3, 4, 4) and intrinsics.shape == (3, 3, 3) and points.shape == (200, 3)
    assert out_dir == recon.backend_dir / "splats"
    # Model-res depth is lifted onto the 8x8 frames through each row's crop box
    assert depth_targets.shape == (3, 8, 8)
    # Depth rows pair with the frames read for them: row 0 is frame 2, row 2 is frame 0
    assert depth_targets[0, 0, 0] == pytest.approx(3, rel=1e-5) and depth_targets[2, 0, 0] == pytest.approx(1, rel=1e-5)


def test_splats_stage_requires_pointcloud_zarr_for_depth_loss(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    with patch("collab_splats.splats.trainer.train"), pytest.raises(FileNotFoundError, match="pointcloud.zarr"):
        recon.splats()


def test_splats_stage_skips_depth_targets_when_depth_loss_off(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["splats"]["losses"] = {}
    with patch("collab_splats.splats.trainer.train") as train:
        recon.splats()
    assert train.call_args.kwargs["depth_targets"] is None


def test_splats_stage_refuses_when_output_exists(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.done = lambda stage: stage in ("preproc", "pointcloud", "splats")
    with patch("collab_splats.splats.trainer.train") as train, pytest.raises(ValueError, match="already exists"):
        recon.run(["splats"])
    train.assert_not_called()


def test_splats_stage_output_marker_is_the_checkpoint(tmp_path):
    """
    The stage marker is read by run() for its skip and refuse rules.
    """
    recon = _stub_reconstructor(tmp_path)
    splats_dir = recon.backend_dir / "splats"
    splats_dir.mkdir(parents=True)

    assert not recon.done("splats")
    (splats_dir / "ckpt.pt").write_bytes(b"")
    assert recon.done("splats")


def test_mesh_source_splats_without_a_checkpoint_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    with pytest.raises(FileNotFoundError, match="mesh.source: splats"):
        recon.mesh()


def test_mesh_source_splats_fuses_from_the_checkpoint(tmp_path):
    """
    The splats path renders the checkpoint; splats.zarr is not an input any more.
    """
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    splats_dir = recon.backend_dir / "splats"
    splats_dir.mkdir(parents=True)
    (splats_dir / "ckpt.pt").touch()

    # Per-view depths: a shape-only assertion also passes on an all-zero stack, which is what
    # a checkpoint rendered from the wrong poses returns
    depths = np.stack([np.full((4, 5), view + 1.0, np.float32) for view in range(3)])
    rendered = (
        depths,
        np.full((3, 4, 5, 3), 7, np.uint8),
        np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
        [0, 1, 2],
    )
    with (
        patch("collab_splats.splats.checkpoint.render_tsdf_inputs", return_value=rendered) as render,
        patch(
            "collab_splats.reconstructor.create_tsdf_mesh", return_value=recon.backend_dir / "mesh.ply"
        ) as fuse,
        patch("collab_splats.reconstructor.clean_repair_mesh") as clean,
        patch("collab_splats.reconstructor.prepare_mesh", side_effect=lambda mesh, **kw: mesh),
    ):
        recon.mesh()

    assert render.call_args.args == (splats_dir / "ckpt.pt", recon.images_dir)
    fused = fuse.call_args.args[0]
    assert [float(fused[view].min()) for view in range(3)] == [1.0, 2.0, 3.0]
    assert clean.call_args.args == (recon.backend_dir / "mesh.ply",)


def test_mesh_source_unknown_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "nerf"
    with pytest.raises(ValueError, match="mesh.source"):
        recon.mesh()


def test_splats_sfm_uses_zarr_depth(tmp_path):
    # sfm scene reads depth targets from the zarr like feedforward
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}

    # Zarr rows in the result's order (frames 2, 1, 0), depth = frame_idx + 1
    depth = np.stack([np.full((4, 4), view + 1, np.float32) for view in (2, 1, 0)])
    feedforward = SimpleNamespace(
        image_paths=recon.result.image_paths,
        depth=depth,
        confidence=None,  # sfm scenes carry no confidence — unmasked targets
        original_coords=np.tile(np.array([0, 0, 8, 8, 8, 8], np.float32), (3, 1)),  # full-frame box
    )
    group = zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    group.attrs["depth_scale"] = "colmap"
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.base.PointcloudResult.load_zarr", return_value=feedforward),
    ):
        recon.splats()

    depth_targets = train.call_args.kwargs["depth_targets"]
    assert depth_targets.shape == (3, 8, 8)
    # Row 0 is frame 2, row 2 is frame 0
    assert depth_targets[0, 0, 0] == pytest.approx(3, rel=1e-5) and depth_targets[2, 0, 0] == pytest.approx(1, rel=1e-5)


def test_mesh_sfm_zarr_fuses(tmp_path):
    # sfm zarr reaches TSDF fusion
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    sfm = minimal_feedforward_result(n=3)
    sfm.image_paths = [Path(f"frame_{view:06d}.jpg") for view in range(3)]

    with (
        patch("collab_splats.pointcloud.base.PointcloudResult.load_zarr", return_value=sfm),
        patch("collab_splats.reconstructor.create_tsdf_mesh", return_value=recon.backend_dir / "mesh.ply") as fuse,
        patch("collab_splats.reconstructor.clean_repair_mesh"),
        patch("collab_splats.reconstructor.prepare_mesh", side_effect=lambda mesh, **kw: mesh),
    ):
        recon.mesh()

    # Unmasked unit depth: sfm carries no confidence, so conf_percentile cannot drop anything
    depths = fuse.call_args.args[0]
    assert depths.shape == (3, 8, 8)
    assert np.all(depths == 1.0)


def test_splats_depth_targets_lift_through_each_frames_crop_box(tmp_path):
    """
    Cropped backends: model-res depth fills only its crop box of the frame, never the whole frame.

    - model 8x4 into a 20x10 frame through a 16x8 crop at a per-frame offset
    - column-ramp depth, so a plain stretch and a crop-aware lift disagree on position
    - boxes differ per store row, so a target paired with the wrong row's box is caught
    """
    recon = _stub_reconstructor(tmp_path, height=10, width=20)

    # Store rows 0..2; each frame's crop sits at a different offset inside the 20x10 frame
    ramp = np.tile(np.arange(1, 9, dtype=np.float32), (4, 1))
    depth = np.stack([ramp * (view + 1) for view in range(3)])
    boxes = np.array([[2 + view, 1, 18 + view, 9, 20, 10] for view in range(3)], np.float32)

    # The zarr holds those rows in the result's order: image_paths is reversed store order
    order = [2, 1, 0]
    feedforward = SimpleNamespace(
        image_paths=recon.result.image_paths,
        depth=depth[order],
        confidence=None,
        original_coords=boxes[order],
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.base.PointcloudResult.load_zarr", return_value=feedforward),
    ):
        recon.splats()

    # What the trainer supervises with: each view's target after train()'s own per-view resize
    images = train.call_args.args[1]
    depth_targets = train.call_args.kwargs["depth_targets"]
    seen = np.stack(
        [prepare_target(images[row], depth_targets[row], "cpu")["depth"][0, ..., 0].numpy() for row in range(3)]
    )

    # Outside each crop box there is no target; a full-frame stretch fills it
    for row, store_row in enumerate(order):
        tl_x, tl_y, cr_x, cr_y = boxes[store_row, :4].astype(int)
        inside = np.zeros((10, 20), bool)
        inside[tl_y:cr_y, tl_x:cr_x] = True
        assert np.all(seen[row][~inside] == 0)
        assert np.all(seen[row][inside] > 0)

    # Inside, the values are the mesh stage's lift of the matching row through its own box
    np.testing.assert_allclose(seen, upsample_depths(depth[order], images, boxes[order, :4]))
