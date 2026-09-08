"""
Splats-stage wiring: leaf registration, base.yaml default, and the arrays handed to train().

- Depth targets come from pointcloud.zarr depth for every method — sfm scenes use the
  dense COLMAP-scale-aligned zarr depth through the same path as feedforward backends.
- A legacy sfm zarr without the depth_scale attr is VDA-metric and raises (re-run the
  pointcloud stage to align).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
import zarr

from collab_splats.splats.trainer import SplatsConfig
from collab_splats.wrapper.reconstructor import _STAGE_DEPS, _STAGE_ORDER, LEAF_STAGES
from tests.wrapper._stubs import _stub_reconstructor

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_splats_is_a_leaf_stage():
    assert "splats" in _STAGE_ORDER
    assert _STAGE_ORDER.index("splats") < _STAGE_ORDER.index("mesh")
    assert _STAGE_DEPS["splats"] == ["pointcloud"]
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
    # Feedforward rows in store order (0, 1, 2) with per-frame distinguishable depth = frame_idx + 1
    depth = np.stack([np.full((4, 4), view + 1, np.float32) for view in range(3)])
    feedforward = SimpleNamespace(
        image_paths=[Path(f"frame_{view:06d}.jpg") for view in range(3)],
        depth=depth,
        confidence=torch.ones(3, 4, 4),
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
    ):
        out = recon.splats()

    assert out == recon.backend_dir / "splats" / "ckpt.pt"
    cfg, images, world_to_cam, intrinsics, points, colors, out_dir = train.call_args.args
    depth_targets = train.call_args.kwargs["depth_targets"]
    assert cfg.max_steps == 1
    assert images[0, 0, 0, 0] == 20 and images[2, 0, 0, 0] == 0  # follows image_paths (reversed), not store order
    assert world_to_cam.shape == (3, 4, 4) and intrinsics.shape == (3, 3, 3) and points.shape == (200, 3)
    assert out_dir == recon.backend_dir / "splats"
    # Model-res depth is handed over as-is; train() resizes per view (prepare_target)
    assert depth_targets.shape == (3, 4, 4)
    # Depth rows reordered to image_paths (reversed): row 0 is frame 2, row 2 is frame 0
    assert depth_targets[0, 0, 0] == 3 and depth_targets[2, 0, 0] == 1


def test_splats_stage_rejects_frames_missing_from_feedforward(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    feedforward = SimpleNamespace(
        image_paths=[Path("frame_000000.jpg"), Path("frame_000001.jpg")],
        depth=np.ones((2, 4, 4), np.float32),
        confidence=None,
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
        pytest.raises(ValueError, match="no depth"),
    ):
        recon.splats()
    train.assert_not_called()


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


def test_splats_stage_skips_when_output_exists(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon._stage_output_exists = lambda stage: stage == "splats"
    with patch("collab_splats.splats.trainer.train") as train:
        recon.splats()
    train.assert_not_called()


def test_splats_stage_skip_path_returns_the_checkpoint_path(tmp_path):
    """
    The early return is a second return statement — the trained path's assertion cannot reach it.
    """
    recon = _stub_reconstructor(tmp_path)
    recon._stage_output_exists = lambda stage: stage == "splats"
    with patch("collab_splats.splats.trainer.train"):
        out = recon.splats()
    assert out == recon.backend_dir / "splats" / "ckpt.pt"


def test_splats_stage_output_marker_is_the_checkpoint(tmp_path):
    """
    The stage marker is a third site, read by run_pipeline and by the skip-check above.
    """
    recon = _stub_reconstructor(tmp_path)
    del recon._stage_output_exists  # _stub_reconstructor stubs it out; this test wants the real one
    splats_dir = recon.backend_dir / "splats"
    splats_dir.mkdir(parents=True)

    assert not recon._stage_output_exists("splats")
    (splats_dir / "ckpt.pt").write_bytes(b"")
    assert recon._stage_output_exists("splats")


def test_splats_conf_percentile_log_reports_the_zero_target_fraction(tmp_path, caplog):
    """
    An SfM zarr has no confidence channel; the log must say so AND report the share of
    targets that are already zero, rather than dropping the setting silently.
    """
    recon = _stub_reconstructor(tmp_path)

    # Quarter of the target pixels zeroed, exactly what VDA writes where it has no depth
    depth = np.ones((3, 4, 4), np.float32)
    depth[:, 0, :] = 0.0
    feedforward = SimpleNamespace(
        image_paths=[Path(f"frame_{view:06d}.jpg") for view in range(3)],
        depth=depth,
        confidence=None,
    )
    (recon.backend_dir / "pointcloud.zarr").mkdir(parents=True)

    with (
        patch("collab_splats.splats.trainer.train"),
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
        caplog.at_level("INFO"),
    ):
        recon.splats()

    assert "conf_percentile=20 not applied (no confidence channel)" in caplog.text
    assert "25.00% of target pixels are zero" in caplog.text


def test_mesh_source_splats_without_a_checkpoint_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    with pytest.raises(ValueError, match="mesh.source: splats"):
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
        patch("collab_splats.wrapper.reconstructor.render_tsdf_inputs", return_value=rendered) as render,
        patch("collab_splats.wrapper.reconstructor.fuse_tsdf", return_value=recon.backend_dir / "mesh.ply") as fuse,
        patch("collab_splats.wrapper.reconstructor.clean_repair_mesh"),
    ):
        out = recon.mesh()

    assert render.call_args.args == (splats_dir / "ckpt.pt", recon.images_dir)
    fused = fuse.call_args.args[0]
    assert [float(fused[view].min()) for view in range(3)] == [1.0, 2.0, 3.0]
    assert out == recon.backend_dir / "mesh.ply"


def test_mesh_source_unknown_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "nerf"
    with pytest.raises(ValueError, match="mesh.source"):
        recon.mesh()


def test_splats_sfm_aligned_zarr_uses_zarr_depth(tmp_path):
    # sfm scene whose zarr carries depth_scale: falls through to the zarr-depth path
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    depth = np.stack([np.full((4, 4), view + 1, np.float32) for view in range(3)])
    feedforward = SimpleNamespace(
        image_paths=[Path(f"frame_{view:06d}.jpg") for view in range(3)],
        depth=depth,
        confidence=None,  # sfm scenes carry no confidence — unmasked targets
    )
    group = zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    group.attrs["depth_scale"] = "colmap"
    with (
        patch("collab_splats.splats.trainer.train") as train,
        patch("collab_splats.pointcloud.feedforward.base.FeedforwardResult.load_zarr", return_value=feedforward),
    ):
        recon.splats()

    depth_targets = train.call_args.kwargs["depth_targets"]
    assert depth_targets.shape == (3, 4, 4)
    # Rows reordered to image_paths (reversed): row 0 is frame 2, row 2 is frame 0
    assert depth_targets[0, 0, 0] == 3 and depth_targets[2, 0, 0] == 1


def test_splats_sfm_legacy_zarr_refused(tmp_path):
    # sfm zarr without depth_scale is VDA-metric — feeding it as targets collapsed
    # training once (PSNR 6.15); must refuse with a re-run pointer
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    with patch("collab_splats.splats.trainer.train") as train, pytest.raises(ValueError, match="depth_scale"):
        recon.splats()
    train.assert_not_called()


def test_mesh_sfm_legacy_zarr_refused(tmp_path):
    # Legacy VDA-metric zarr against COLMAP poses fused geometry at the wrong scale —
    # keep refusing scenes without the depth_scale attr
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    with pytest.raises(ValueError, match="depth_scale"):
        recon.mesh()


def test_mesh_sfm_aligned_zarr_fuses(tmp_path):
    # Aligned sfm zarr (depth_scale attr) passes the guard and reaches TSDF fusion
    recon = _stub_reconstructor(tmp_path)
    recon.config["pointcloud"] = {"method": "sfm", "backend": "instantsfm"}
    group = zarr.open_group(recon.backend_dir / "pointcloud.zarr", mode="w")
    group.attrs["depth_scale"] = "colmap"
    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as fuse:
        fuse.return_value = recon.backend_dir / "mesh" / "mesh.ply"
        out = recon.mesh()
    fuse.assert_called_once()
    assert out == recon.backend_dir / "mesh" / "mesh.ply"
