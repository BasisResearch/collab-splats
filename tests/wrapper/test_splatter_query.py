import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from collab_splats.wrapper.splatter import Splatter


def _make_splatter_with_model(tmp_path, n_verts=50):
    """Return a Splatter instance with a fake model wired in — no disk I/O."""
    from collab_splats.wrapper.splatter import Splatter

    mesh_pt = tmp_path / "mesh_features.pt"
    mesh_ply = tmp_path / "mesh_clean.ply"
    mesh_ply.touch()

    s = object.__new__(Splatter)
    s.config = {
        "output_path": tmp_path,
        "method": "rade-features",
        "model_config_path": str(tmp_path / "config.yml"),
        "mesh_info": {
            "mesh": mesh_ply,
            "features": mesh_pt,
        },
    }

    mock_model = MagicMock()
    mock_model.main_features_name = "distill_features"
    mock_model.device = torch.device("cpu")

    fake_features_pt = torch.zeros(n_verts, 64)
    decoded = {"distill_features": torch.zeros(n_verts, 64)}
    mock_model.decoder.per_gaussian_forward.return_value = decoded
    mock_model.similarity_fx.return_value = torch.zeros(n_verts, 1)

    s.model = mock_model
    return s, mock_model, fake_features_pt


def test_query_mesh_returns_ndarray(tmp_path):
    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    with patch("torch.load", return_value=fake_features):
        result = s.query_mesh(positive_queries=["feeder"], negative_queries=["ground"])
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"


def test_query_mesh_temperature_forwarded(tmp_path):
    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return torch.zeros(50, 1)

    mock_model.similarity_fx.side_effect = capture

    with patch("torch.load", return_value=fake_features):
        s.query_mesh(positive_queries=["feeder"], negative_queries=["ground"], temperature=0.01)

    assert "temperature" in captured, "temperature was not forwarded to similarity_fx"
    assert captured["temperature"] == pytest.approx(0.01)


def test_query_mesh_default_temperature_is_005(tmp_path):
    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return torch.zeros(50, 1)

    mock_model.similarity_fx.side_effect = capture

    with patch("torch.load", return_value=fake_features):
        s.query_mesh(positive_queries=["feeder"])

    assert captured.get("temperature") == pytest.approx(0.05)


def test_query_mesh_output_fn_writes_ply(tmp_path):
    """When output_fn is given, a PLY is written AND np.ndarray is still returned."""
    o3d = pytest.importorskip("open3d")

    mesh_ply = tmp_path / "mesh_clean.ply"
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.zeros((50, 3)))
    m.triangles = o3d.utility.Vector3iVector(np.zeros((1, 3), dtype=np.int32))
    o3d.io.write_triangle_mesh(str(mesh_ply), m)

    s, mock_model, fake_features = _make_splatter_with_model(tmp_path)
    s.config["mesh_info"]["mesh"] = mesh_ply

    with patch("torch.load", return_value=fake_features):
        result = s.query_mesh(
            positive_queries=["feeder"],
            negative_queries=["ground"],
            output_fn="query-feeder.ply",
        )

    assert isinstance(result, np.ndarray)
    assert (tmp_path / "query-feeder.ply").exists()


def _make_decoder_state(input_dim=13, hidden_dim=4, output_dim=8, feature_type="maskclip"):
    """Build a minimal decoder state dict matching TwoLayerMLP structure."""
    import torch
    return {
        "hidden_conv.weight": torch.randn(hidden_dim, input_dim, 1, 1),
        "hidden_conv.bias":   torch.zeros(hidden_dim),
        f"feature_branch_dict.{feature_type}.weight": torch.randn(output_dim, hidden_dim, 1, 1),
        f"feature_branch_dict.{feature_type}.bias":   torch.zeros(output_dim),
    }


def test_query_mesh_fast_path_skips_eval_setup(tmp_path):
    """When mesh_decoder.pt exists, query_mesh must not call eval_setup."""
    import torch
    from unittest.mock import MagicMock, patch

    mesh_path = tmp_path / "mesh_clean.ply"
    mesh_path.touch()
    features = torch.zeros(5, 13)
    torch.save(features, tmp_path / "mesh_features.pt")
    torch.save(_make_decoder_state(), tmp_path / "mesh_decoder.pt")

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy",
        "mesh_info": {
            "mesh": mesh_path,
            "features": tmp_path / "mesh_features.pt",
        },
    }

    fake_scores = torch.zeros(5)
    mock_extractor = MagicMock()
    mock_extractor.score_queries.return_value = fake_scores

    with patch("collab_splats.wrapper.splatter.eval_setup") as mock_eval, \
         patch("collab_splats.wrapper.splatter.BaseFeatureExtractor") as mock_bfe:
        mock_bfe.get.return_value = lambda **kw: mock_extractor
        result = splatter.query_mesh(positive_queries=["feeder"])

    mock_eval.assert_not_called()
    assert result.shape == (5, 3)


def test_query_mesh_fast_path_decoder_reconstructed_from_state(tmp_path):
    """Decoder output dim is correctly inferred from weight shape."""
    import torch
    from unittest.mock import MagicMock, patch

    input_dim, hidden_dim, output_dim = 13, 4, 8
    mesh_path = tmp_path / "mesh_clean.ply"
    mesh_path.touch()
    features = torch.zeros(5, input_dim)
    torch.save(features, tmp_path / "mesh_features.pt")
    torch.save(_make_decoder_state(input_dim, hidden_dim, output_dim), tmp_path / "mesh_decoder.pt")

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy",
        "mesh_info": {
            "mesh": mesh_path,
            "features": tmp_path / "mesh_features.pt",
        },
    }

    fake_scores = torch.zeros(5)
    mock_extractor = MagicMock()
    mock_extractor.score_queries.return_value = fake_scores

    with patch("collab_splats.wrapper.splatter.eval_setup"), \
         patch("collab_splats.wrapper.splatter.BaseFeatureExtractor") as mock_bfe:
        mock_bfe.get.return_value = lambda **kw: mock_extractor
        splatter.query_mesh(positive_queries=["feeder"])

    call_args = mock_extractor.score_queries.call_args
    _feat = call_args[1].get("features")
    feat_arg = _feat if _feat is not None else call_args[0][0]
    assert feat_arg.shape[0] == output_dim


def test_query_mesh_falls_back_to_eval_setup_when_no_decoder(tmp_path):
    """When mesh_decoder.pt is absent, query_mesh falls back to eval_setup."""
    import torch
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace

    mesh_path = tmp_path / "mesh_clean.ply"
    mesh_path.touch()
    features = torch.zeros(5, 13)
    torch.save(features, tmp_path / "mesh_features.pt")
    # No mesh_decoder.pt written

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy/config.yml",
        "mesh_info": {
            "mesh": mesh_path,
            "features": tmp_path / "mesh_features.pt",
        },
    }

    fake_scores = torch.zeros(5)
    mock_model = MagicMock()
    mock_model.main_features_name = "maskclip"
    mock_model.device = "cpu"
    mock_model.decoder.per_gaussian_forward.return_value = {"maskclip": torch.zeros(5, 8)}
    mock_model.similarity_fx.return_value = fake_scores

    mock_pipeline = MagicMock()
    mock_pipeline.model = mock_model

    with patch("collab_splats.wrapper.splatter.eval_setup", return_value=(None, mock_pipeline, None, None)) as mock_eval:
        result = splatter.query_mesh(positive_queries=["feeder"])

    mock_eval.assert_called_once()
    assert result.shape == (5, 3)
