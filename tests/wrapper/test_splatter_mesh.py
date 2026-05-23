import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import pytest
import torch
import numpy as np


def _bare_splatter(tmp_path):
    from collab_splats.wrapper.splatter import Splatter

    s = object.__new__(Splatter)
    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
    }
    return s


def test_export_gaussian_splats_calls_ns_export(tmp_path):
    """_export_gaussian_splats runs ns-export and returns path to splats.ply."""
    s = _bare_splatter(tmp_path)
    mesh_dir = tmp_path / "rade-features" / "mesh"
    mesh_dir.mkdir(parents=True)

    def fake_run(cmd, check, **kwargs):
        (mesh_dir / "splats.ply").write_bytes(b"ply")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        result = s._export_gaussian_splats(mesh_dir, overwrite=False)

    assert result == mesh_dir / "splats.ply"
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "ns-export" in call_args
    assert "gaussian-splat" in call_args
    assert "--load-config" in call_args
    assert str(tmp_path / "config.yml") in call_args
    assert "--output-dir" in call_args
    assert "--output-filename" in call_args


def test_export_gaussian_splats_skips_if_exists(tmp_path):
    """_export_gaussian_splats does not call ns-export when splats.ply exists."""
    s = _bare_splatter(tmp_path)
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    existing = mesh_dir / "splats.ply"
    existing.write_bytes(b"ply")

    with patch("subprocess.run") as mock_run:
        result = s._export_gaussian_splats(mesh_dir, overwrite=False)

    mock_run.assert_not_called()
    assert result == existing


def test_export_gaussian_splats_overwrite_reruns(tmp_path):
    """overwrite=True re-runs ns-export even when splats.ply exists."""
    s = _bare_splatter(tmp_path)
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    (mesh_dir / "splats.ply").write_bytes(b"old")

    def fake_run(cmd, check, **kwargs):
        (mesh_dir / "splats.ply").write_bytes(b"new")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        s._export_gaussian_splats(mesh_dir, overwrite=True)

    mock_run.assert_called_once()


def test_mesh_else_branch_sets_splats_key(tmp_path):
    """mesh(overwrite=False) sets mesh_info['splats'] when splats.ply exists."""
    from collab_splats.wrapper.splatter import Splatter

    s = object.__new__(Splatter)
    mesh_dir = tmp_path / "rade-features" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh_tsdf_clean.ply").write_bytes(b"ply")
    (mesh_dir / "splats.ply").write_bytes(b"ply")
    (mesh_dir / "mesh_features.pt").write_bytes(b"pt")

    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
    }

    with patch.object(Splatter, "_select_run"):
        s.mesh(overwrite=False)

    assert "splats" in s.config["mesh_info"]
    assert s.config["mesh_info"]["splats"] == mesh_dir / "splats.ply"


def test_mesh_else_branch_no_splats_key_when_missing(tmp_path):
    """mesh(overwrite=False) does not set mesh_info['splats'] when splats.ply absent."""
    from collab_splats.wrapper.splatter import Splatter

    s = object.__new__(Splatter)
    mesh_dir = tmp_path / "rade-features" / "mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "mesh_tsdf_clean.ply").write_bytes(b"ply")

    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
    }

    with patch.object(Splatter, "_select_run"):
        s.mesh(overwrite=False)

    assert "splats" not in s.config["mesh_info"]


def test_extract_mesh_features_saves_decoder(tmp_path):
    """_extract_mesh_features must write mesh_decoder.pt next to mesh_features.pt."""
    from collab_splats.wrapper.splatter import Splatter

    fake_decoder = MagicMock()
    fake_decoder.state_dict.return_value = {"hidden_conv.weight": torch.zeros(64, 13, 1, 1)}

    fake_model = SimpleNamespace(
        decoder=fake_decoder,
        main_features_name="maskclip",
        device="cpu",
        means=torch.zeros(10, 3),
        gauss_params={"distill_features": torch.zeros(10, 13)},
    )

    mesh_path = tmp_path / "mesh_tsdf_clean.ply"
    mesh_path.touch()

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy",
        "mesh_info": {"mesh": mesh_path, "features": tmp_path / "mesh_features.pt"},
    }
    splatter.model = fake_model

    vertex_features = np.zeros((5, 13), dtype=np.float32)

    with patch("collab_splats.mesh.utils.features2vertex", return_value=vertex_features), \
         patch("collab_splats.wrapper.splatter.o3d") as mock_o3d:
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.zeros((5, 3))
        mock_o3d.io.read_triangle_mesh.return_value = mock_mesh
        splatter._extract_mesh_features(features_name="distill_features")

    assert (tmp_path / "mesh_decoder.pt").exists(), "mesh_decoder.pt not written"
    saved = torch.load(tmp_path / "mesh_decoder.pt", map_location="cpu", weights_only=True)
    assert "hidden_conv.weight" in saved
