import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_run_mapanything_returns_8tuple(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 2, 20

    mock_views = [{"img": np.zeros((3, 64, 64))} for _ in range(n)]
    mock_image_paths = [image_dir / f"frame_{i:04d}.jpg" for i in range(n)]
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)
    mock_extrinsics = np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32)
    mock_intrinsics = np.eye(3)[None].repeat(n, axis=0).astype(np.float32)
    mock_original_coords = np.zeros((n, 6), dtype=np.float32)

    with patch("collab_splats.pointcloud._mapanything._load_mapanything_model", return_value=MagicMock()), \
         patch("collab_splats.pointcloud._mapanything._load_and_preprocess_images",
               return_value=(mock_views, mock_image_paths, mock_original_coords)), \
         patch("collab_splats.pointcloud._mapanything._run_mapanything_inference",
               return_value=[{}] * n), \
         patch("collab_splats.pointcloud._mapanything._collect_pts3d_from_outputs",
               return_value=(mock_pts3d, mock_colors, mock_extrinsics, mock_intrinsics)), \
         patch("collab_splats.pointcloud._mapanything.voxel_downsample_point_cloud",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._mapanything import run_mapanything
        result = run_mapanything(image_dir, "facebook/map-anything")

    assert len(result) == 8
    pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = result
    assert pts3d.shape[1] == 3
    assert colors.shape[1] == 3
    assert extrinsics.shape == (n, 3, 4)
    assert intrinsics.shape == (n, 3, 3)
    assert len(image_paths) == n
    assert original_coords.shape == (n, 6)
    assert isinstance(model_w, int) and isinstance(model_h, int)


def test_run_mapanything_passes_inference_kwargs(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 2, 5

    mock_views = [{"img": np.zeros((3, 64, 64))} for _ in range(n)]
    mock_image_paths = [image_dir / f"frame_{i:04d}.jpg" for i in range(n)]
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)
    mock_extrinsics = np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32)
    mock_intrinsics = np.eye(3)[None].repeat(n, axis=0).astype(np.float32)
    mock_original_coords = np.zeros((n, 6), dtype=np.float32)

    with patch("collab_splats.pointcloud._mapanything._load_mapanything_model", return_value=MagicMock()), \
         patch("collab_splats.pointcloud._mapanything._load_and_preprocess_images",
               return_value=(mock_views, mock_image_paths, mock_original_coords)), \
         patch("collab_splats.pointcloud._mapanything._run_mapanything_inference",
               return_value=[{}] * n) as mock_infer, \
         patch("collab_splats.pointcloud._mapanything._collect_pts3d_from_outputs",
               return_value=(mock_pts3d, mock_colors, mock_extrinsics, mock_intrinsics)), \
         patch("collab_splats.pointcloud._mapanything.voxel_downsample_point_cloud",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._mapanything import run_mapanything
        run_mapanything(image_dir, "facebook/map-anything",
                        confidence_percentile=60.0, minibatch_size=4)

        _, kwargs = mock_infer.call_args
        assert kwargs["confidence_percentile"] == 60.0
        assert kwargs["minibatch_size"] == 4
