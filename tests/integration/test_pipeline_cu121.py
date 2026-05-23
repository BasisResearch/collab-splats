"""Phase 5 pipeline integration tests for cu121 migration.

Tests pipeline components end-to-end with synthetic data. _forward is mocked
so no model downloads are required. Tests confirm that numpy 2.x + torch 2.4 +
pycolmap 4.0.4 work through the full data-flow path.

Run:
    /opt/conda/envs/nerfstudio/bin/python -m pytest tests/integration/test_pipeline_cu121.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

N_FRAMES = 4
H, W = 224, 224


def _synthetic_extrinsics(n: int) -> np.ndarray:
    """(n, 3, 4) identity extrinsics with small translation offsets."""
    ext = np.tile(np.eye(3, 4), (n, 1, 1)).astype(np.float32)
    for i in range(n):
        ext[i, 2, 3] = i * 0.1  # translate along Z
    return ext


def _synthetic_intrinsics(n: int, w: int = W, h: int = H) -> np.ndarray:
    """(n, 3, 3) pinhole intrinsics."""
    K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    return np.tile(K, (n, 1, 1))


def _synthetic_vggtx_raw(n: int = N_FRAMES, h: int = H, w: int = W) -> dict:
    """Minimal raw_outputs dict matching VGGTXCreator._forward output schema.

    depth is (N, H, W, 1) — shape used by _postprocess for model_h/model_w.
    images is (N, 3, H, W) tensor — used by _raw_to_world_points.
    intrinsics key (not intrinsics_downsampled) — _postprocess falls back to
    raw_outputs.get("intrinsics_downsampled", raw_outputs.get("intrinsics")).
    """
    return {
        "images": torch.rand(n, 3, h, w),
        "extrinsic": _synthetic_extrinsics(n),
        "intrinsics": _synthetic_intrinsics(n, w, h),
        "intrinsics_downsampled": _synthetic_intrinsics(n, w, h),
        "depth": np.ones((n, h, w, 1), dtype=np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32) * 0.9,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_build_pycolmap_reconstruction_roundtrip():
    """build_pycolmap_reconstruction → colmap_reconstruction_to_result.

    Exercises the full pycolmap 4.0.4 path:
      add_camera_with_trivial_rig + add_image_with_trivial_frame + add_point3D.
    """
    from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction
    from collab_splats.pointcloud.utils import colmap_reconstruction_to_result

    P = 30
    rng = np.random.default_rng(42)
    pts3d = rng.standard_normal((P, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (P, 3), dtype=np.uint8)

    recon = build_pycolmap_reconstruction(
        pts3d=pts3d,
        colors=colors,
        extrinsics=_synthetic_extrinsics(N_FRAMES),
        intrinsics=_synthetic_intrinsics(N_FRAMES),
        image_width=W,
        image_height=H,
        image_names=[f"frame_{i:04d}.jpg" for i in range(N_FRAMES)],
    )

    assert len(recon.images) == N_FRAMES, f"Expected {N_FRAMES} images, got {len(recon.images)}"
    assert len(recon.cameras) == N_FRAMES, f"Expected {N_FRAMES} cameras, got {len(recon.cameras)}"
    assert len(recon.points3D) == P, f"Expected {P} points, got {len(recon.points3D)}"

    result = colmap_reconstruction_to_result(recon)
    assert result.points.shape == (P, 3), f"Expected points shape ({P}, 3), got {result.points.shape}"
    assert result.camera_poses is not None
    assert result.camera_poses.shape[0] == N_FRAMES
    assert result.camera_poses.shape[1:] == (4, 4)
    assert result.camera_intrinsics is not None
    assert result.camera_intrinsics.shape == (N_FRAMES, 3, 3)


def test_feedforward_result_save_load(tmp_path):
    """FeedforwardResult .save() / .load() round-trip under numpy 2.x npz."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    P, N = 30, N_FRAMES
    rng = np.random.default_rng(0)
    original = FeedforwardResult(
        pts3d=rng.standard_normal((P, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=_synthetic_intrinsics(N),
        image_paths=[Path(f"frame_{i}.jpg") for i in range(N)],
        original_coords=np.zeros((N, 6), dtype=np.float32),
        model_width=W,
        model_height=H,
    )
    save_path = tmp_path / "result.npz"
    original.save(save_path)
    loaded = FeedforwardResult.load(save_path)

    np.testing.assert_array_equal(original.pts3d, loaded.pts3d)
    np.testing.assert_array_equal(original.colors, loaded.colors)
    np.testing.assert_array_equal(original.extrinsics, loaded.extrinsics)
    assert loaded.model_width == W
    assert loaded.model_height == H
    assert len(loaded.image_paths) == N


def test_vggtx_postprocess_pipeline(tmp_path):
    """VGGTXCreator._postprocess with synthetic raw_outputs → FeedforwardResult.

    Mocks unproject_and_filter_points so no VGGT model download is needed.
    unproject_and_filter_points returns (pts3d, colors, pixel_indices) — 3-tuple.
    """
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    raw = _synthetic_vggtx_raw()
    creator = VGGTXCreator(use_global_alignment=False)
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]
    creator.original_coords = np.zeros((N_FRAMES, 6), dtype=np.float32)
    for i in range(N_FRAMES):
        creator.original_coords[i] = [0, 0, W, H, W, H]

    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.unproject_and_filter_points",
        return_value=(
            np.random.randn(20, 3).astype(np.float32),
            np.random.randint(0, 255, (20, 3)).astype(np.uint8),
            np.zeros((20, 3), dtype=np.int32),  # pixel_indices (frame_id, row, col)
        ),
    ):
        result = creator._postprocess(raw)

    assert result is not None
    assert result.pts3d.shape[1] == 3
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)
    assert result.intrinsics.shape == (N_FRAMES, 3, 3)
    assert result.model_width == W
    assert result.model_height == H


def test_mapanything_postprocess_pipeline(tmp_path):
    """MapAnythingCreator._postprocess with synthetic raw_outputs → FeedforwardResult.

    Mocks collect_pts3d_from_outputs (which consumes the list-of-dicts raw_outputs)
    and voxel_downsample so no MapAnything model download is needed.
    Also confirms _patch_mapanything_torch_compat fires without RuntimeError.
    """
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    N = N_FRAMES
    # MapAnythingCreator._postprocess accesses self.views[i]["img"] for model_h/model_w
    # Build minimal synthetic views list
    synthetic_views = [
        {
            "img": torch.rand(1, 3, H, W),
            "img_no_norm": torch.rand(1, H, W, 3),
            "pts3d": torch.rand(1, H, W, 3),
            "mask": torch.ones(1, H, W, 1),
            "depth_z": torch.ones(1, H, W, 1),
            "intrinsics": torch.eye(3).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
            "conf": None,
        }
        for _ in range(N)
    ]

    creator = MapAnythingCreator()
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(N)]
    creator.original_coords = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        creator.original_coords[i] = [0, 0, W, H, W, H]
    creator.views = synthetic_views

    mock_pts3d = np.random.randn(20, 3).astype(np.float32)
    mock_colors = np.random.randint(0, 255, (20, 3)).astype(np.uint8)
    mock_extrinsics = _synthetic_extrinsics(N)
    mock_intrinsics = _synthetic_intrinsics(N)

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.collect_pts3d_from_outputs",
        return_value=(mock_pts3d, mock_colors, mock_extrinsics, mock_intrinsics),
    ), patch(
        "collab_splats.pointcloud.feedforward.mapanything.voxel_downsample",
        side_effect=lambda pcd, **kw: (pcd, None),
    ):
        result = creator._postprocess(synthetic_views)

    assert result is not None
    assert result.pts3d.shape[1] == 3
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)


def test_tsdf_mesh_synthetic(tmp_path):
    """Open3DTSDFFusion.create with synthetic depth + RGB frames.

    rgbs must be float32 [0, 1] per Open3DTSDFFusion.create docstring.
    """
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    n, h, w = 4, 64, 64
    rng = np.random.default_rng(1)
    depths = np.full((n, h, w), 2.0, dtype=np.float32)
    # create() multiplies by 255 internally: pass float32 [0,1]
    rgbs = rng.random((n, h, w, 3)).astype(np.float32)
    c2w = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)
    for i in range(n):
        c2w[i, 2, 3] = i * 0.05
    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (n, 1, 1))

    fusion = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    mesh_result = fusion.create(
        depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=intrinsics
    )
    assert mesh_result is not None
    assert isinstance(mesh_result.mesh_path, Path)


def test_nerfstudio_method_registry():
    """rade-gs and rade-features appear in nerfstudio method registry after import."""
    import collab_splats.nerfstudio.method_configs.rade_gs  # noqa: F401
    import collab_splats.nerfstudio.method_configs.rade_features  # noqa: F401
    from nerfstudio.configs.method_configs import all_methods
    assert "rade-gs" in all_methods, (
        f"rade-gs not registered. Available: {sorted(all_methods)}"
    )
    assert "rade-features" in all_methods, (
        f"rade-features not registered. Available: {sorted(all_methods)}"
    )
