import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


def _mock_vggt_data(image_dir, n=3):
    return {
        "extrinsic": np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32),
        "instrinsics_downsampled": np.eye(3)[None].repeat(n, axis=0).astype(np.float32),
        "depth": np.zeros((n, 64, 64), dtype=np.float32),
        "depth_conf": np.ones((n, 64, 64), dtype=np.float32),
        "images": np.zeros((n, 3, 64, 64), dtype=np.float32),
        "image_paths": [image_dir / f"frame_{i:04d}.jpg" for i in range(n)],
        "original_coords": np.zeros((n, 6), dtype=np.float32),
    }


def test_run_vggt_returns_8tuple(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 3, 30
    data = _mock_vggt_data(image_dir, n)
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)
    extrinsics = data["extrinsic"]
    intrinsics = data["instrinsics_downsampled"]

    with patch("collab_splats.pointcloud._vggt._is_vggt_available", return_value=True), \
         patch("collab_splats.pointcloud._vggt._run_vggt_inference", return_value=data), \
         patch("collab_splats.pointcloud._vggt._maybe_run_global_alignment",
               return_value=(extrinsics, intrinsics)), \
         patch("collab_splats.pointcloud._vggt._unproject_and_filter_points",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._vggt import run_vggt
        result = run_vggt(image_dir, tmp_path / "colmap", "facebook/vggt")

    assert len(result) == 8
    pts3d, colors, ext, intr, image_paths, orig, model_w, model_h = result
    assert pts3d.shape[1] == 3
    assert len(image_paths) == n
    assert isinstance(model_w, int) and isinstance(model_h, int)


def test_run_vggt_raises_if_not_installed(tmp_path):
    with patch("collab_splats.pointcloud._vggt._is_vggt_available", return_value=False):
        from collab_splats.pointcloud._vggt import run_vggt
        with pytest.raises(RuntimeError, match="VGGT-X not installed"):
            run_vggt(tmp_path / "imgs", tmp_path / "colmap", "facebook/vggt")


def test_global_alignment_flag_passed(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 2, 5
    data = _mock_vggt_data(image_dir, n)
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)

    with patch("collab_splats.pointcloud._vggt._is_vggt_available", return_value=True), \
         patch("collab_splats.pointcloud._vggt._run_vggt_inference", return_value=data), \
         patch("collab_splats.pointcloud._vggt._maybe_run_global_alignment",
               return_value=(data["extrinsic"], data["instrinsics_downsampled"])) as mock_align, \
         patch("collab_splats.pointcloud._vggt._unproject_and_filter_points",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._vggt import run_vggt
        run_vggt(image_dir, tmp_path / "colmap", "facebook/vggt", use_global_alignment=True)

        _, kwargs = mock_align.call_args
        assert kwargs["use_global_alignment"] is True
