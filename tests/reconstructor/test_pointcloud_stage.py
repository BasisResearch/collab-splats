"""
The pointcloud stage: sparse_pc.ply and cleaned-set writes, and config forwarded to the creator.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pycolmap

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
from collab_splats.reconstructor import Reconstructor
from tests.pointcloud.conftest import _frame_files, _frames
from tests.reconstructor._stubs import stub_creator_cls

########################################################################
########## sparse_pc.ply and the cleaned set ###########################
########################################################################


def _pointcloud_result(xyz):
    """
    A real PointcloudResult over one full-frame 8x6 camera and the given points.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    colors = (np.arange(len(xyz))[:, None] * np.array([1, 2, 3])).astype(np.uint8)
    return PointcloudResult(
        points=xyz,
        colors=colors,
        extrinsics=np.eye(4, dtype=np.float32)[None],
        intrinsics=None,
        model_intrinsics=np.array(
            [[[4.0, 0.0, 4.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]]], dtype=np.float32
        ),
        image_paths=[Path("frame_000000")],
        original_coords=np.array([[0, 0, 8, 6, 8, 6]], dtype=np.float32),
        model_width=8,
        model_height=6,
    )


def test_pointcloud_stage_writes_sparse_pc_ply(tmp_path):
    """
    The pointcloud stage lands a binary PLY of the final result at backend_dir/sparse_pc.ply.
    """
    # clean disabled so the result reaching the writer is the one the backend returned.
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "clean": {"enabled": False},
        },
    }
    rec = Reconstructor(config)
    result = _pointcloud_result([[float(i), 0.0, 1.0] for i in range(3)])

    with patch(
        "collab_splats.reconstructor.get_creator", return_value=stub_creator_cls(result)
    ):
        rec.pointcloud()

    # The path is the contract: downstream stages and the remote sync both look here by name
    out = rec.backend_dir / "sparse_pc.ply"
    assert out.exists()

    # Binary little-endian, and carrying THIS result's points — not an empty or stub file.
    # xyz/rgb fidelity is PointcloudResult.write_ply's own contract (tests/pointcloud/test_base.py).
    header = out.read_bytes()[:200]
    assert header.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 3\n" in header


@dataclass
class _ClusterCreator(BaseFeedforwardCreator):
    """
    Feedforward creator whose inference yields 60 clustered points plus one far outlier.
    """

    def _load_model(self, device: str) -> Any:
        return None

    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        return None, np.array([[0, 0, 8, 6, 8, 6]], dtype=np.float32)

    def _forward(self, model: Any, views: Any) -> Any:
        return None

    def _postprocess(self, raw_outputs: Any) -> PointcloudResult:
        # 60 cluster points clear outlier_mask's nb_neighbors=20 guard; the outlier is far out
        cluster = np.random.default_rng(0).normal(scale=0.01, size=(60, 3))
        result = _pointcloud_result(np.vstack([cluster, [[50.0, 50.0, 50.0]]]))
        rows = np.arange(len(result.points), dtype=np.int32)
        return replace(
            result,
            pixel_indices=np.stack([np.zeros_like(rows), rows % 6, rows % 8], axis=-1),
        )


def test_pointcloud_stage_zarr_colmap_and_ply_hold_one_cleaned_set(tmp_path):
    """
    The creator cleans before any write: fresh result, zarr reload, COLMAP model and PLY all agree.
    """
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "clean": {"enabled": True},
        },
    }
    rec = Reconstructor(config)
    _frame_files(_frames([(8, 6)]), rec.images_dir)

    with patch("collab_splats.reconstructor.get_creator", return_value=_ClusterCreator):
        rec.pointcloud()

    # The outlier went before the zarr save; colors and pixel_indices travel with the points
    stored = PointcloudResult.load_zarr(
        rec.pointcloud_zarr, load_depth=False, load_world_points=False
    )
    assert len(stored.points) == len(stored.colors) == len(stored.pixel_indices) == 60
    assert np.abs(stored.points).max() < 1.0

    # A second Reconstructor reloads the same points and colors from disk
    reloaded = Reconstructor(config).result
    np.testing.assert_array_equal(reloaded.points, stored.points)
    np.testing.assert_array_equal(reloaded.colors, stored.colors)

    # The COLMAP export and the PLY carry the same 60 points
    assert pycolmap.Reconstruction(str(rec.colmap_model_dir)).num_points3D() == 60
    header = (rec.backend_dir / "sparse_pc.ply").read_bytes()[:200]
    assert b"element vertex 60\n" in header


def test_pointcloud_stage_clean_off_keeps_every_point_in_zarr_colmap_and_ply(tmp_path):
    """
    pointcloud.clean.enabled False reaches the creator: the outlier survives in every artifact.
    """
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggtx",
            "clean": {"enabled": False},
        },
    }
    rec = Reconstructor(config)
    _frame_files(_frames([(8, 6)]), rec.images_dir)

    with patch("collab_splats.reconstructor.get_creator", return_value=_ClusterCreator):
        rec.pointcloud()

    # The zarr, the COLMAP export and the PLY all hold the uncleaned 61, outlier included
    stored = PointcloudResult.load_zarr(
        rec.pointcloud_zarr, load_depth=False, load_world_points=False
    )
    assert len(stored.points) == 61
    assert np.abs(stored.points).max() == 50.0
    assert pycolmap.Reconstruction(str(rec.colmap_model_dir)).num_points3D() == 61
    header = (rec.backend_dir / "sparse_pc.ply").read_bytes()[:200]
    assert b"element vertex 61\n" in header


########################################################################
########## Multiview knobs reach the creator ###########################
########################################################################


def _run(tmp_path, **pointcloud):
    """Run the pointcloud stage with the creator class patched out; returns that class."""
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"backend": "vggt_omega", **pointcloud},
    }
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(MagicMock())

    with patch("collab_splats.reconstructor.get_creator", return_value=creator_cls):
        rec.pointcloud()

    return creator_cls


def test_backend_block_reaches_the_creator_as_kwargs(tmp_path):
    """The pointcloud stage forwards pointcloud.<backend> verbatim, alongside max_points."""
    block = {"min_views": 2, "mv_rel_thresh": 0.03}
    creator_cls = _run(tmp_path, max_points=1234, vggt_omega=block)
    kwargs = creator_cls.call_args.kwargs
    assert {key: kwargs[key] for key in block} == block
    assert kwargs["max_points"] == 1234
