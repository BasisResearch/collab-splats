"""_write_transforms_json owns transforms.json outright: stale files are replaced, not merged."""

import json

import numpy as np

from collab_splats.wrapper.reconstructor import Reconstructor


class _Result:
    """Two-frame stand-in: only extrinsics/intrinsics/image_paths are read."""

    points = np.zeros((3, 3), dtype=np.float32)
    colors = None
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)
    intrinsics = np.stack([np.eye(3, dtype=np.float32)] * 2)
    image_paths = ["frame_000000.jpg", "frame_000001.jpg"]


def _reconstructor(tmp_path):
    return Reconstructor(
        {
            "input_path": str(tmp_path / "in.mp4"),
            "output_path": str(tmp_path / "out"),
            "pointcloud": {"method": "feedforward", "backend": "vggt_omega"},
        }
    )


def test_replaces_stale_file(tmp_path):
    r = _reconstructor(tmp_path)
    r.backend_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        "camera_model": "PINHOLE",
        "ply_file_path": "sparse_pc.ply",
        "applied_transform": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]],
        "frames": [{"file_path": "stale.jpg"}],
    }
    (r.backend_dir / "transforms.json").write_text(json.dumps(existing))

    r._write_transforms_json(_Result())

    out = json.loads((r.backend_dir / "transforms.json").read_text())
    # Keys from the old merged-over file do not survive
    assert set(out) == {"camera_model", "frames"}
    # frames are ours, not the stale ones (real frame dicts key on frame_idx, not file_path)
    assert len(out["frames"]) == 2
    assert out["frames"][0] != existing["frames"][0]
    assert "frame_idx" in out["frames"][0]


def test_works_with_no_existing_file(tmp_path):
    r = _reconstructor(tmp_path)
    r.backend_dir.mkdir(parents=True, exist_ok=True)
    r._write_transforms_json(_Result())
    out = json.loads((r.backend_dir / "transforms.json").read_text())
    assert out["camera_model"] == "PINHOLE"
    assert len(out["frames"]) == 2
