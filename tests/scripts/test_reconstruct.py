"""Tests for scripts/reconstruct.py CLI."""

import importlib.util
import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "reconstruct.py"
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "reconstruction"


def _load_main():
    """Load the reconstruct module and return its main() function."""
    spec = importlib.util.spec_from_file_location("reconstruct", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def _make_mock_reconstructor(tmp_path):
    """Return (mock_cls, mock_instance) with config.output_path set to tmp_path."""
    mock_r = MagicMock()
    mock_r.config = {
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggt_omega"},
    }
    mock_r.backend_dir = tmp_path / "vggt_omega"
    mock_cls = MagicMock()
    mock_cls.from_config_file.return_value = mock_r
    mock_cls.return_value = mock_r  # direct Reconstructor(config) path
    return mock_cls, mock_r


def test_dataset_arg_calls_from_config_file(monkeypatch, tmp_path):
    """--dataset NAME calls Reconstructor.from_config_file with correct args."""
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--dataset", "birds_c0043"])
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_cls.from_config_file.assert_called_once_with(
        dataset="birds_c0043",
        config_dir=DEFAULT_CONFIG_DIR,
        overrides=None,
    )


def test_stages_parsed_to_list(monkeypatch, tmp_path):
    """--stages preprocess,pointcloud passes list to run_pipeline."""
    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--dataset", "birds_c0043", "--stages", "preprocess,pointcloud"],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_r.run_pipeline.assert_called_once_with(
        stages=["preprocess", "pointcloud"], overwrite=False
    )


def test_overwrite_flag(monkeypatch, tmp_path):
    """--overwrite passes overwrite=True to run_pipeline."""
    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--dataset", "birds_c0043", "--overwrite"],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_r.run_pipeline.assert_called_once_with(stages=None, overwrite=True)


def test_key_value_overrides_parsed(monkeypatch, tmp_path):
    """KEY=VALUE positional args are parsed and passed as overrides dict."""
    monkeypatch.setattr(
        sys, "argv",
        [
            "reconstruct.py", "--dataset", "birds_c0043",
            "pointcloud.backend=vggtx",
            "semantics.enabled=true",
        ],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_cls.from_config_file.assert_called_once_with(
        dataset="birds_c0043",
        config_dir=DEFAULT_CONFIG_DIR,
        overrides={"pointcloud": {"backend": "vggtx"}, "semantics": {"enabled": True}},
    )


def test_direct_config_path_loads_yaml(monkeypatch, tmp_path):
    """--config /path/to/config.yaml loads YAML directly, skips dataset hierarchy."""
    config_yaml = tmp_path / "myconfig.yaml"
    cfg_data = {
        "input_path": "/data/video.mp4",
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggtx"},
        "semantics": {"enabled": True, "extractor": "dinov2", "n_components": 64},
    }
    config_yaml.write_text(yaml.dump(cfg_data))

    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--config", str(config_yaml)],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path / "out")
    mock_r.config = {**cfg_data, "output_path": str(tmp_path / "out")}
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    # Should call Reconstructor(config) directly, not from_config_file
    mock_cls.assert_called_once()
    mock_cls.from_config_file.assert_not_called()


def test_run_config_yaml_written_to_output_dir(monkeypatch, tmp_path):
    """main() writes run_config.yaml into output_path before running pipeline."""
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--dataset", "birds_c0043"])
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    mock_r.config = {
        "input_path": "/data/video.mp4",
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggt_omega"},
    }
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    run_cfg = tmp_path / "run_config.yaml"
    assert run_cfg.exists(), "run_config.yaml was not written"
    loaded = yaml.safe_load(run_cfg.read_text())
    assert loaded["pointcloud"]["backend"] == "vggt_omega"


def test_dataset_and_config_are_mutually_exclusive(monkeypatch, tmp_path, capsys):
    """--dataset and --config together cause argparse error (SystemExit)."""
    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text("{}")
    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--dataset", "birds_c0043", "--config", str(config_yaml)],
    )
    main = _load_main()
    with pytest.raises(SystemExit):
        main()
