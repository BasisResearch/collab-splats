"""
Splats-stage wiring: leaf registration, base.yaml default, and the arrays handed to train().

- Depth targets come from pointcloud.zarr depth for every method — sfm scenes use the
  dense COLMAP-scale-aligned zarr depth through the same path as feedforward backends.
- A legacy sfm zarr without the depth_scale attr is VDA-metric and raises (re-run the
  pointcloud stage to align).
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
import zarr

from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.trainer import SplatsConfig
from collab_splats.wrapper.reconstructor import (
    _STAGE_DEPS,
    _STAGE_ORDER,
    LEAF_STAGES,
    _run_tsdf_mesh,
)
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


def _write_ckpt(path, primitive="3dgs", n_views=3, height=4, width=5):
    """
    Minimal trained-model ckpt.pt at `path` — the splats stage's only artifact of record.

    - Deliberately a local copy of the fixture in tests/mesh/test_splats_adapter.py: that file
      belongs to the mesh adapter's suite, and importing across test packages couples them.
    - `primitive` decides whether the adapter will accept `splat_depth: median` (2dgs only).
    """
    rng = np.random.default_rng(0)
    cfg = SplatsConfig.from_dict({"primitive": primitive, "max_steps": 10, "losses": {}})
    model = Gaussians(
        cfg,
        rng.uniform(-0.5, 0.5, (200, 3)).astype(np.float32),
        rng.integers(0, 255, (200, 3)).astype(np.uint8),
        scene_scale=1.0,
        n_views=n_views,
        device="cpu",
    )

    ckpt = model.checkpoint()
    ckpt["config"] = asdict(cfg)
    ckpt["cam_to_world"] = torch.eye(4)[None].repeat(n_views, 1, 1)
    ckpt["intrinsics"] = torch.tensor([[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]])[None].repeat(
        n_views, 1, 1
    )
    ckpt["image_ids"] = list(range(n_views))
    ckpt["image_size"] = (height, width)
    ckpt["appearance"] = None

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def _fake_render_views(depths, medians=None):
    """
    Stand-in rasterizer yielding caller-chosen pixels, shaped as `render_views` does.

    - `rgb` is (1, H, W, 3) in [0, 1]; `depth`, `median_depth` and `alpha` are (1, H, W, 1).
    - Alpha is all-ones, which the adapter's percentile gate keeps in full, so the depths a
      test writes are the depths that reach fusion.
    """
    depths = np.asarray(depths, dtype=np.float32)
    medians = depths if medians is None else np.asarray(medians, dtype=np.float32)
    rgbs = np.zeros(depths.shape + (3,), dtype=np.float32)
    alphas = np.ones_like(depths)

    def render_views(model, camera_opt, cam_to_world, intrinsics, height, width):
        for view in range(len(depths)):
            yield {
                "rgb": torch.from_numpy(rgbs[view])[None],
                "depth": torch.from_numpy(depths[view])[None, ..., None],
                "median_depth": torch.from_numpy(medians[view])[None, ..., None],
                "alpha": torch.from_numpy(alphas[view])[None, ..., None],
            }

    return render_views


def _patch_renders(monkeypatch, render_views):
    """Swap the rasterizer the adapter imports at call time for a scripted one."""
    monkeypatch.setattr("collab_splats.splats.rendering.render_views", render_views)


def test_mesh_source_splats_without_a_checkpoint_raises(tmp_path):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    with pytest.raises(ValueError, match="mesh.source: splats"):
        recon.mesh()


def test_mesh_source_splats_fuses_from_the_checkpoint(tmp_path, monkeypatch):
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    recon.config["mesh"]["native_resolution"] = True  # ignored on this path, must not raise
    _write_ckpt(recon.backend_dir / "splats" / "ckpt.pt", n_views=3)

    # Per-view depths: a shape-only assertion also passes on an all-zero stack, which is what
    # a checkpoint rendered from the wrong poses returns
    depths = np.stack([np.full((4, 5), view + 1.0, np.float32) for view in range(3)])
    _patch_renders(monkeypatch, _fake_render_views(depths))
    with patch("collab_splats.mesh.utils.mesh_from_tsdf_inputs") as fuse:
        fuse.return_value = SimpleNamespace(mesh_path=recon.backend_dir / "mesh.ply")
        out = recon.mesh()

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


def test_mesh_stage_forwards_splat_depth_from_the_config(tmp_path):
    """
    splat_depth has to survive the first hop too — the config dict -> mesh() -> _run_tsdf_mesh.
    """
    recon = _stub_reconstructor(tmp_path)
    recon.config["mesh"]["source"] = "splats"
    recon.config["mesh"]["splat_depth"] = "median"
    splats_dir = recon.backend_dir / "splats"
    splats_dir.mkdir(parents=True)
    (splats_dir / "ckpt.pt").write_bytes(b"")  # _run_tsdf_mesh is patched; only the precondition runs

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as fuse:
        recon.mesh()

    # "median" is not the base.yaml default, so a dropped pass-through cannot fake this
    assert fuse.call_args.kwargs["splat_depth"] == "median"


def test_run_tsdf_mesh_forwards_splat_depth_to_the_adapter(tmp_path, monkeypatch):
    """
    splat_depth has to survive the last hop too — mesh() → _run_tsdf_mesh → the splats adapter.
    """
    # A 2dgs checkpoint (the only kind the adapter renders median depth from), scripted to
    # return BOTH depths, so a dropped pass-through fuses 1.0 instead of raising
    ckpt_path = _write_ckpt(tmp_path / "ckpt.pt", primitive="2dgs", n_views=2)
    _patch_renders(
        monkeypatch,
        _fake_render_views(np.ones((2, 4, 5), np.float32), medians=np.full((2, 4, 5), 3.0, np.float32)),
    )

    # result=None on purpose: the splats branch no longer reads it. The frame-count cross-check
    # that did is gone — a checkpoint carries the poses it was rendered with, so a COLMAP model
    # with a different image count is not something this fusion can be wrong about
    with patch("collab_splats.mesh.utils.mesh_from_tsdf_inputs") as fuse:
        fuse.return_value = SimpleNamespace(mesh_path=tmp_path / "mesh.ply")
        _run_tsdf_mesh(
            result=None,
            pointcloud_zarr=tmp_path / "pointcloud.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            source="splats",
            splats_ckpt=ckpt_path,
            splat_depth="median",
        )

    # Assert on the fused values, not a mock kwarg — this exercises the adapter end of the hop
    assert fuse.call_args.args[0][0, 0, 0] == 3.0  # the median array, not the expected one


def _splats_adapter_kwargs(tmp_path, **overrides):
    """
    Drive _run_tsdf_mesh's splats branch with the adapter mocked and hand back its call kwargs.
    """
    depths = np.ones((2, 4, 5), np.float32)
    rgbs = np.zeros((2, 4, 5, 3), np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)

    with (
        patch("collab_splats.mesh.utils._splats_to_tsdf_inputs") as adapter,
        patch("collab_splats.mesh.utils.mesh_from_tsdf_inputs") as fuse,
    ):
        adapter.return_value = (depths, rgbs, c2w, intrinsics)
        fuse.return_value = SimpleNamespace(mesh_path=tmp_path / "mesh.ply")
        _run_tsdf_mesh(
            result=None,
            pointcloud_zarr=tmp_path / "pointcloud.zarr",
            output_dir=tmp_path,
            voxel_size=0.01,
            sdf_trunc=0.04,
            depth_trunc=2.0,
            source="splats",
            splats_ckpt=tmp_path / "ckpt.pt",
            **overrides,
        )
    return adapter.call_args.kwargs


# The splats branch hands the adapter four tunables. One test each, setting only its own knob:
# a single test asserting all four would die for any of them and so could not say which broke.
def test_run_tsdf_mesh_forwards_conf_percentile_to_the_adapter(tmp_path):
    """
    conf_percentile is the alpha gate; dropped, every low-confidence pixel fuses.
    """
    assert _splats_adapter_kwargs(tmp_path, conf_percentile=20.0)["conf_percentile"] == 20.0


def test_run_tsdf_mesh_forwards_max_depth_frac_to_the_adapter(tmp_path):
    """
    splat_max_depth_frac is the far cut; dropped, blown-out background fuses.
    """
    assert _splats_adapter_kwargs(tmp_path, splat_max_depth_frac=0.5)["max_depth_frac"] == 0.5


def test_run_tsdf_mesh_forwards_max_depth_grad_to_the_adapter(tmp_path):
    """
    splat_max_depth_grad is the discontinuity cut; dropped, depth straddling an edge fuses.
    """
    assert _splats_adapter_kwargs(tmp_path, splat_max_depth_grad=0.05)["max_depth_grad"] == 0.05
