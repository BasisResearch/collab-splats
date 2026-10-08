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
    images is (N, 3, H, W) tensor — the per-pixel colors.
    raw outputs carry one `intrinsics` key.
    """
    return {
        "images": torch.rand(n, 3, h, w),
        "extrinsic": _synthetic_extrinsics(n),
        "intrinsics": _synthetic_intrinsics(n, w, h),
        "depth": np.ones((n, h, w, 1), dtype=np.float32),
        "depth_conf": np.ones((n, h, w), dtype=np.float32) * 0.9,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_to_colmap_roundtrip():
    """PointcloudResult.to_colmap under pycolmap 4.x: one camera + image per frame, every point."""
    from collab_splats.pointcloud.base import PointcloudResult

    P = 30
    rng = np.random.default_rng(42)
    ext = np.tile(np.eye(4, dtype=np.float32), (N_FRAMES, 1, 1))
    ext[:, :3, :] = _synthetic_extrinsics(N_FRAMES)
    result = PointcloudResult(
        points=rng.standard_normal((P, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (P, 3), dtype=np.uint8),
        extrinsics=ext,
        intrinsics=None,
        model_intrinsics=_synthetic_intrinsics(N_FRAMES),
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(N_FRAMES)],
        original_coords=np.tile(
            np.array([0, 0, W, H, W, H], dtype=np.float32), (N_FRAMES, 1)
        ),
        model_width=W,
        model_height=H,
    )

    recon = result.to_colmap()
    assert len(recon.images) == N_FRAMES, (
        f"Expected {N_FRAMES} images, got {len(recon.images)}"
    )
    assert len(recon.cameras) == N_FRAMES, (
        f"Expected {N_FRAMES} cameras, got {len(recon.cameras)}"
    )
    assert len(recon.points3D) == P, f"Expected {P} points, got {len(recon.points3D)}"


def test_pointcloud_result_save_load(tmp_path):
    """PointcloudResult .save_zarr() / .load_zarr() round-trip under numpy 2.x."""
    from collab_splats.pointcloud.base import PointcloudResult

    P, N = 30, N_FRAMES
    rng = np.random.default_rng(0)
    original = PointcloudResult(
        points=rng.standard_normal((P, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=None,
        model_intrinsics=_synthetic_intrinsics(N),
        image_paths=[Path(f"frame_{i}.jpg") for i in range(N)],
        original_coords=np.tile(
            np.array([0, 0, W, H, W, H], dtype=np.float32), (N, 1)
        ),  # full-frame box
        model_width=W,
        model_height=H,
    )
    save_path = tmp_path / "result.zarr"
    original.save_zarr(save_path)
    loaded = PointcloudResult.load_zarr(save_path)

    np.testing.assert_array_equal(original.points, loaded.points)
    np.testing.assert_array_equal(original.colors, loaded.colors)
    np.testing.assert_array_equal(original.extrinsics, loaded.extrinsics)
    assert loaded.model_width == W
    assert loaded.model_height == H
    assert len(loaded.image_paths) == N


def test_vggtx_postprocess_pipeline(tmp_path):
    """VGGTXCreator._postprocess with synthetic raw_outputs → PointcloudResult, no model download."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    raw = _synthetic_vggtx_raw()
    creator = VGGTXCreator()
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]
    creator.original_coords = np.tile(
        np.array([0, 0, W, H, W, H], dtype=np.float32), (N_FRAMES, 1)
    )  # full-frame box

    result = creator._postprocess(raw)

    assert result is not None
    assert result.points.shape[1] == 3
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)
    assert result.intrinsics.shape == (N_FRAMES, 3, 3)
    assert result.model_width == W
    assert result.model_height == H


def test_mapanything_postprocess_pipeline(tmp_path):
    """
    MapAnything _stack_predictions then the base _postprocess → PointcloudResult.

    - upstream postprocess mocked to return ready-to-consume pred dicts
    - mv-conf stubbed, so no model download runs
    """
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    N = N_FRAMES
    # Minimal synthetic views; each also carries every upstream pred key
    synthetic_views = [
        {
            "img": torch.rand(1, 3, H, W),
            "img_no_norm": torch.rand(1, H, W, 3),
            "pts3d": torch.rand(1, H, W, 3),
            "pts3d_cam": torch.rand(1, H, W, 3),
            "mask": torch.ones(1, H, W, 1),
            "depth_z": torch.ones(1, H, W, 1),
            "intrinsics": torch.eye(3).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
            "conf": 1.0 + torch.rand(1, H, W),  # upstream conf is (B, H, W), >= 1
        }
        for _ in range(N)
    ]

    creator = MapAnythingCreator(min_views=1)
    creator.image_paths = [tmp_path / f"frame_{i:04d}.jpg" for i in range(N)]
    creator.original_coords = np.tile(
        np.array([0, 0, W, H, W, H], dtype=np.float32), (N, 1)
    )  # full-frame box

    # Upstream postprocess returns per-frame pred dicts; the synthetic views carry every key
    mock_processed = synthetic_views

    with (
        patch(
            "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
            return_value=mock_processed,
        ),
        patch(
            "collab_splats.pointcloud.feedforward.base.multiview_depth_confidence",
            side_effect=lambda depth, *a, **kw: (
                np.ones_like(depth, np.int32),
                np.ones_like(depth, np.int32),
            ),
        ),
    ):
        raw = creator._stack_predictions(synthetic_views, synthetic_views, masked=True)
        result = creator._postprocess(raw)

    assert result is not None
    assert result.points.shape[1] == 3
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)


def test_tsdf_mesh_synthetic():
    """
    create_tsdf_mesh over synthetic depth + RGB frames.

    rgbs must be uint8 [0, 255] — create_tsdf_mesh rejects float color outright.
    """
    from collab_splats.mesh import create_tsdf_mesh

    n, h, w = 4, 64, 64
    rng = np.random.default_rng(1)
    depths = np.full((n, h, w), 2.0, dtype=np.float32)
    rgbs = rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)
    c2w = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)
    for i in range(n):
        c2w[i, 2, 3] = i * 0.05
    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (n, 1, 1))

    mesh = create_tsdf_mesh(
        depths, rgbs, c2w, intrinsics, voxel_size=0.05, depth_trunc=5.0
    )
    assert len(mesh.triangles) > 0
