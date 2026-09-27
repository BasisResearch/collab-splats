"""
One atomic, nan-safe JSON writer for every report.
"""

import json

import numpy as np

from collab_splats.utils.io import to_json_safe, write_json


def test_to_json_safe_nulls_every_non_finite_float():
    """json.dumps writes a bare NaN/Infinity, which no strict parser accepts."""
    out = to_json_safe({"rho": float("nan"), "nested": [float("inf"), -np.inf, 1.0], "f32": np.float32("nan")})
    assert out == {"rho": None, "nested": [None, None, 1.0], "f32": None}


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
