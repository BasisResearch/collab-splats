"""
Torch-free IO helpers: JSON reports, images, zarr stores.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import pytest
import trimesh
import zarr
from PIL import Image

from collab_splats.utils.io import (
    LZ4,
    open_valid,
    read_image,
    to_json_safe,
    to_uint8_hwc,
    write_json,
    write_textured_obj,
)


def test_to_json_safe_nulls_every_non_finite_float():
    """json.dumps writes a bare NaN/Infinity, which no strict parser accepts."""
    out = to_json_safe({"rho": float("nan"), "nested": [float("inf"), -np.inf, 1.0], "f32": np.float32("nan")})
    assert out == {"rho": None, "nested": [None, None, 1.0], "f32": None}


def test_to_json_safe_writes_a_pybind11_enum_as_its_name():
    camera = pycolmap.Camera(model="OPENCV", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0])

    safe = to_json_safe(camera.todict())

    assert safe["model"] == "OPENCV"
    json.dumps(safe)


def test_to_json_safe_turns_numpy_into_python():
    out = to_json_safe({"i": np.int64(3), "b": np.bool_(True), "f": np.float32(0.125), "a": np.array([[1, 2]])})
    assert out == {"i": 3, "b": True, "f": 0.125, "a": [[1, 2]]}
    assert type(out["i"]) is int and type(out["b"]) is bool and type(out["f"]) is float


def test_to_json_safe_walks_tuples_and_nan_inside_arrays():
    assert to_json_safe((1, np.array([np.nan, 2.0]))) == [1, [None, 2.0]]


def test_write_json_matches_json_dumps_bytes_and_leaves_no_tmp(tmp_path):
    payload = {"a": [1, 2.5, "x"], "b": {"c": None}}
    path = write_json(tmp_path / "r.json", payload)
    assert path.read_text() == json.dumps(payload, indent=2)
    assert not (tmp_path / "r.json.tmp").exists()
    assert "NaN" not in write_json(tmp_path / "n.json", {"x": float("nan")}).read_text()


########################################################################
# Images
########################################################################


def test_read_image_returns_rgb_not_bgr(tmp_path):
    """cv2 decodes BGR; a red pixel must come back red."""
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    cv2.imwrite(str(tmp_path / "r.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    out = read_image(tmp_path / "r.png")

    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    np.testing.assert_array_equal(out, rgb)


def test_read_image_ignores_exif_orientation_like_pil(tmp_path):
    """Stored-pixel order, as PIL and the feedforward loaders read it: cv2 alone would rotate."""
    # 20x40 JPEG tagged EXIF orientation 6: an EXIF-following decode would give 40x20
    jpg = tmp_path / "rotated.jpg"
    pixels = np.zeros((20, 40, 3), np.uint8)
    pixels[:, :10] = 255
    im = Image.fromarray(pixels)
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(jpg, exif=exif, quality=95)

    assert read_image(jpg).shape == np.asarray(Image.open(jpg).convert("RGB")).shape == (20, 40, 3)


def test_read_image_names_a_missing_file(tmp_path):
    """cv2.imread returns None; the helper raises by name instead of a cvtColor assert."""
    with pytest.raises(FileNotFoundError, match="nope.png"):
        read_image(tmp_path / "nope.png")


def test_to_uint8_hwc_rounds_instead_of_truncating():
    """0.6/255 must land on 1, not 0: truncation biases every channel down half a level."""
    x = np.full((1, 3, 1, 2), 0.6 / 255.0, dtype=np.float32)
    x[0, :, 0, 1] = 1.0

    out = to_uint8_hwc(x, channels_first=True)

    assert out.shape == (1, 1, 2, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out[0, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(out[0, 0, 1], [255, 255, 255])


def test_to_uint8_hwc_channels_last_and_clip():
    x = np.array([[[-0.1, 0.5, 1.2]]], dtype=np.float32)  # (1, 1, 3) HWC

    out = to_uint8_hwc(x, channels_first=False)

    np.testing.assert_array_equal(out, [[[0, 128, 255]]])


def test_to_uint8_hwc_rejects_a_0_255_array():
    """A [0, 255] input is a stale-zarr contract break, not a scale to guess."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_uint8_hwc(np.full((1, 3, 2, 2), 200.0, dtype=np.float32), channels_first=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_to_uint8_hwc_rejects_non_finite(bad):
    """A NaN max compares False against the [0, 255] guard, so it must be caught on its own."""
    x = np.full((1, 1, 3), 200.0, dtype=np.float32)
    x[0, 0, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        to_uint8_hwc(x, channels_first=False)


def test_to_uint8_hwc_rejects_a_misdeclared_channel_axis():
    """channels_first=True on an HWC array would move the width axis to the end."""
    with pytest.raises(ValueError, match="3 channels"):
        to_uint8_hwc(np.zeros((4, 5, 3), dtype=np.float32), channels_first=True)


def test_to_uint8_hwc_rejects_integer_input():
    """A uint8 mask or dark image has max <= 1 and would be scaled to 255."""
    with pytest.raises(ValueError, match="float"):
        to_uint8_hwc(np.ones((2, 2, 3), dtype=np.uint8), channels_first=False)


########################################################################
# Zarr
########################################################################


def _store(path: Path, **attrs) -> Path:
    """
    A one-array zarr store at `path` carrying `attrs`.
    """
    root = zarr.open(str(path), mode="w")
    root.create_array("x", shape=(2,), dtype="float32", compressors=LZ4)
    root.attrs.update(attrs)
    return path


def test_open_valid_returns_the_store_when_attrs_match(tmp_path):
    path = _store(tmp_path / "s.zarr", extractor="dino", n_frames=3)

    store = open_valid(path, {"extractor": "dino", "n_frames": 3})

    assert store is not None and store.attrs["n_frames"] == 3


def test_open_valid_rejects_a_missing_or_mismatched_store(tmp_path):
    path = _store(tmp_path / "s.zarr", extractor="dino", n_frames=3)

    assert open_valid(tmp_path / "absent.zarr", {"n_frames": 3}) is None
    assert open_valid(path, {"extractor": "dino", "n_frames": 4}) is None
    assert open_valid(path, {"never_written": 1}) is None


def test_open_valid_matches_a_tuple_against_its_stored_list(tmp_path):
    """zarr attrs round-trip through JSON, so a stamped tuple reads back as a list."""
    path = _store(tmp_path / "s.zarr", shape=(1, 2))

    assert open_valid(path, {"shape": (1, 2)}) is not None


def test_open_valid_treats_corrupt_metadata_as_absent(tmp_path, caplog):
    path = tmp_path / "bad.zarr"
    path.mkdir()
    (path / "zarr.json").write_text("{not json")

    with caplog.at_level(logging.WARNING):
        assert open_valid(path, {"n_frames": 1}) is None
    assert "unreadable" in caplog.text


def test_open_valid_propagates_a_bug(tmp_path, monkeypatch):
    """Only UNREADABLE_STORE means stale; any other error is a bug and must surface."""
    path = _store(tmp_path / "s.zarr", n_frames=1)

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(zarr, "open", boom)
    with pytest.raises(RuntimeError, match="not a store error"):
        open_valid(path, {"n_frames": 1})


def test_io_imports_without_torch():
    """io is the torch-free layer: cv2, numpy, zarr only."""
    root = Path(__file__).resolve().parents[2]
    code = "import sys; sys.modules['torch'] = None; import collab_splats.utils.io"
    env = {**os.environ, "PYTHONPATH": str(root)}
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_write_textured_obj_splits_corners_with_white_kd_and_normals(tmp_path):
    """A shared vertex becomes one corner per face, so each face keeps its own UV."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    uvs = np.random.default_rng(0).random((2, 3, 2)).astype(np.float32)
    albedo = np.full((8, 8, 3), 200, dtype=np.uint8)

    out = write_textured_obj(tmp_path / "tex", vertices, faces, normals, uvs, albedo)

    assert sorted(p.name for p in out.parent.iterdir()) == ["albedo.png", "mesh.mtl", "mesh.obj"]
    assert "Kd 1.00000000 1.00000000 1.00000000" in (out.parent / "mesh.mtl").read_text()
    assert "\nvn " in out.read_text()
    loaded = trimesh.load(out, process=False)
    assert len(loaded.vertices) == 6
    np.testing.assert_allclose(loaded.visual.uv, uvs.reshape(-1, 2), atol=1e-6)
