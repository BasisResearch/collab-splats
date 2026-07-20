"""Tests for docs/examples/reconstruct.py CLI (reproduce-from-saved-config)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

SCRIPT_PATH = Path(__file__).parent.parent.parent / "docs" / "examples" / "reconstruct.py"


def _load_main():
    """Load the reconstruct module and return its main() function."""
    spec = importlib.util.spec_from_file_location("reconstruct", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def _make_mock_reconstructor(tmp_path, config=None):
    """Return (mock_cls, mock_instance) with config.output_path set to tmp_path."""
    mock_r = MagicMock()
    mock_r.config = config or {
        "input_path": "/data/video.mp4",
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggt_omega"},
    }
    mock_r.backend_dir = tmp_path / "vggt_omega"
    mock_cls = MagicMock()
    mock_cls.return_value = mock_r
    return mock_cls, mock_r


def _write_config(tmp_path, **extra):
    """Write a minimal saved config.yaml and return its path + data."""
    cfg = {
        "input_path": "/data/video.mp4",
        "output_path": str(tmp_path),
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega"},
        **extra,
    }
    path = tmp_path / "run_config.yaml"
    path.write_text(yaml.dump(cfg))
    return path, cfg


def test_config_required(monkeypatch, tmp_path):
    """--config is required — omitting it exits."""
    monkeypatch.setattr(sys, "argv", ["reconstruct.py"])
    main = _load_main()
    with pytest.raises(SystemExit):
        main()


def test_config_loads_yaml_and_constructs_reconstructor(monkeypatch, tmp_path):
    """--config PATH loads the YAML and builds Reconstructor(config)."""
    path, cfg = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--config", str(path)])
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path, config=cfg)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_cls.assert_called_once_with(cfg)


def test_stages_parsed_to_list(monkeypatch, tmp_path):
    """--stages preprocess,pointcloud passes a list to run_pipeline."""
    path, cfg = _write_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reconstruct.py", "--config", str(path), "--stages", "preprocess,pointcloud"],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path, config=cfg)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_r.run_pipeline.assert_called_once_with(stages=["preprocess", "pointcloud"], overwrite=False)


def test_overwrite_flag(monkeypatch, tmp_path):
    """--overwrite passes overwrite=True to run_pipeline."""
    path, cfg = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--config", str(path), "--overwrite"])
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path, config=cfg)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_r.run_pipeline.assert_called_once_with(stages=None, overwrite=True)


def test_key_value_overrides_merged_into_config(monkeypatch, tmp_path):
    """KEY=VALUE positional args are merged over the loaded config."""
    path, cfg = _write_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reconstruct.py", "--config", str(path), "pointcloud.backend=vggtx", "semantics.enabled=true"],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path, config=cfg)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    # Config passed to Reconstructor has overrides applied
    (called_config,), _ = mock_cls.call_args
    assert called_config["pointcloud"]["backend"] == "vggtx"
    assert called_config["semantics"]["enabled"] is True


def test_run_config_yaml_written_to_output_dir(monkeypatch, tmp_path):
    """main() writes run_config.yaml into output_path before running the pipeline."""
    out = tmp_path / "out"
    path, cfg = _write_config(tmp_path)
    cfg = {**cfg, "output_path": str(out)}
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--config", str(path)])
    mock_cls, mock_r = _make_mock_reconstructor(out, config=cfg)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    run_cfg = out / "run_config.yaml"
    assert run_cfg.exists(), "run_config.yaml was not written"
    loaded = yaml.safe_load(run_cfg.read_text())
    assert loaded["pointcloud"]["backend"] == "vggt_omega"
