"""In-memory _preprocess is bit-equivalent to the old path-based upstream loaders.

This is the ATE guard for P5.1a: _preprocess now takes decoded (N, H, W, 3) uint8
frames instead of an image dir. The produced model-input tensor MUST be byte-for-byte
identical to the legacy path-based ``load_and_preprocess_images`` output, otherwise
every backbone's poses (and ATE) would shift.
"""

import tempfile

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator


def _write_pngs(tmp_path, sizes):
    """Write synthetic RGB PNGs; return (paths, decoded frames, frame_idxs)."""
    rng = np.random.default_rng(0)
    paths, frames = [], []
    for i, (w, h) in enumerate(sizes):
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        p = tmp_path / f"frame_{i:06d}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
        # frames = the exact decoded pixels the images/ store holds
        frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
    return paths, frames, list(range(len(sizes)))


def test_frames_decode_matches_file_open(tmp_path):
    """Sanity: Image.fromarray(frame) reproduces Image.open(path).convert('RGB')."""
    paths, frames, _ = _write_pngs(tmp_path, [(160, 120)])
    from_file = np.asarray(Image.open(paths[0]).convert("RGB"))
    from_mem = np.asarray(Image.fromarray(frames[0]))
    assert np.array_equal(from_file, from_mem)


def test_vggtx_preprocess_bit_equivalent(tmp_path):
    """VGGTX in-memory _preprocess == upstream load_and_preprocess_images(mode='crop')."""
    from vggt.utils.load_fn import load_and_preprocess_images

    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    paths, frames, frame_idxs = _write_pngs(tmp_path, [(160, 120), (100, 200), (300, 90)])
    old = load_and_preprocess_images([str(p) for p in paths], mode="crop")

    views, image_paths, coords = VGGTXCreator()._preprocess(frames, frame_idxs)

    # Bit-equivalence anchor — the tensor the model sees must be identical
    assert torch.equal(views, old)
    # Interface contract: (views (N,3,H,W), labels, original_coords (N,6))
    assert views.shape[0] == 3 and views.shape[1] == 3
    assert [p.name for p in image_paths] == [f"frame_{i:06d}" for i in frame_idxs]
    assert coords.shape == (3, 6) and coords.dtype == np.float32


def test_vggt_omega_preprocess_bit_equivalent(tmp_path):
    """VGGT-Omega in-memory _preprocess == upstream load_and_preprocess_images(mode='balanced')."""
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator

    paths, frames, frame_idxs = _write_pngs(tmp_path, [(160, 120), (100, 200)])
    creator = VGGTOmegaCreator(resolution=512, resize_mode="balanced")
    old = load_and_preprocess_images([str(p) for p in paths], image_resolution=512, mode="balanced")

    views, image_paths, coords = creator._preprocess(frames, frame_idxs)

    assert torch.equal(views, old)
    assert [p.name for p in image_paths] == [f"frame_{i:06d}" for i in frame_idxs]
    assert coords.shape == (2, 6) and coords.dtype == np.float32


def test_mapanything_preprocess_bit_equivalent(tmp_path):
    """MapAnything in-memory _preprocess view tensors == upstream load_images()."""
    from mapanything.utils.image import load_images

    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    paths, frames, frame_idxs = _write_pngs(tmp_path, [(160, 120), (100, 200)])
    old = load_images([str(p) for p in paths], resize_mode="fixed_mapping", resolution_set=518)

    views, image_paths, coords = MapAnythingCreator()._preprocess(frames, frame_idxs)

    assert len(views) == len(old)
    for o, n in zip(old, views):
        assert torch.equal(o["img"], n["img"])
    assert [p.name for p in image_paths] == [f"frame_{i:06d}" for i in frame_idxs]
    assert coords.shape == (2, 6) and coords.dtype == np.float32


def test_decode_source_takes_an_images_dir_and_makes_no_tempdir(tmp_path, monkeypatch):
    """
    Path-locked model preprocessing reads the scene's images/ directly.
    """
    images = tmp_path / "images"
    images.mkdir()
    for i in (0, 5):
        cv2.imwrite(str(images / f"frame_{i:06d}.png"), np.full((8, 12, 3), i + 5, np.uint8))

    # Any TemporaryDirectory here means a copy is still being staged
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda *a, **k: pytest.fail("staged a temporary copy"))

    paths = BaseFeedforwardCreator._source_paths(images)
    assert [p.name for p in paths] == ["frame_000000.png", "frame_000005.png"]
