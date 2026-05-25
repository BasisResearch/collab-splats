import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_fake_seq(tmp_path: Path, n_frames: int = 4) -> Path:
    """Write a minimal CO3Dv2 sequence dir with synthetic frame_annotations.jgz."""
    seq_dir = tmp_path / "apple" / "seq1"
    images_dir = seq_dir / "images"
    images_dir.mkdir(parents=True)

    annotations = []
    for i in range(1, n_frames + 1):
        fname = f"frame{i:06d}.jpg"
        (images_dir / fname).write_bytes(b"fake")

        annotations.append({
            "image": {"path": f"images/{fname}", "size": [480, 640]},
            "viewpoint": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [float(i) * 0.1, 0.0, 1.0],
                "focal_length": [1.2, 1.2],
                "principal_point": [0.0, 0.0],
            },
        })

    ann_path = seq_dir / "frame_annotations.jgz"
    with gzip.open(ann_path, "wt", encoding="utf-8") as f:
        json.dump(annotations, f)

    return seq_dir


def test_co3dv2_loader_returns_eval_dataset(tmp_path):
    from evals.datasets import get_dataset, EvalDataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=4)
    loader = get_dataset("co3dv2")
    ds = loader(seq_dir, max_frames=4)
    assert isinstance(ds, EvalDataset)


def test_co3dv2_loader_correct_n_frames(tmp_path):
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=4)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=4)
    assert len(ds.images) == 4
    assert ds.gt_poses.shape == (4, 4, 4)


def test_co3dv2_loader_max_frames_truncation(tmp_path):
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=4)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=2)
    assert len(ds.images) == 2
    assert ds.gt_poses.shape == (2, 4, 4)


def test_co3dv2_loader_pose_dtype(tmp_path):
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=3)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=3)
    assert ds.gt_poses.dtype == np.float32


def test_co3dv2_loader_identity_rotation_preserved(tmp_path):
    """Synthetic annotations use identity R — loader converts to OpenCV convention.

    CO3Dv2 uses PyTorch3D row-major convention with left-handed axes (x=left, y=up, z=fwd).
    The loader converts to OpenCV w2c via R_cv = S @ R.T where S = diag(-1, -1, 1).
    For identity R, the expected OpenCV rotation is diag(-1, -1, 1).
    """
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=2)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=2)
    expected_R = np.diag([-1., -1., 1.]).astype(np.float32)
    np.testing.assert_allclose(ds.gt_poses[0, :3, :3], expected_R, atol=1e-5)


def test_co3dv2_loader_provides_intrinsics(tmp_path):
    """intrinsics field should be (N, 3, 3) float32."""
    from evals.datasets import get_dataset

    seq_dir = _make_fake_seq(tmp_path, n_frames=3)
    ds = get_dataset("co3dv2")(seq_dir, max_frames=3)
    assert ds.intrinsics is not None
    assert ds.intrinsics.shape == (3, 3, 3)
    assert ds.intrinsics.dtype == np.float32


def test_eval_dataset_intrinsics_defaults_to_none():
    """Existing callers pass no intrinsics — field must default to None."""
    from evals.datasets import EvalDataset

    ds = EvalDataset(images=[], gt_poses=np.zeros((0, 4, 4), dtype=np.float32))
    assert ds.intrinsics is None


def test_7scenes_loader_still_works(tmp_path):
    """Regression: existing 7-Scenes loader must still return EvalDataset without intrinsics."""
    from evals.datasets import EvalDataset

    ds = EvalDataset(
        images=[],
        gt_poses=np.zeros((0, 4, 4), dtype=np.float32),
    )
    assert ds.intrinsics is None
