"""run_pipeline.py: CLI surface only.

The per-scene helpers this script used to define now live in
collab_splats.wrapper.batch and are tested in tests/wrapper/test_batch.py. What is
left here is the script's own responsibility: argparse wiring into run_all.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

# Load run_pipeline.py by path (docs/examples is not a package)
_MODULE_PATH = Path(__file__).parent.parent.parent / "docs" / "examples" / "run_pipeline.py"
_spec = importlib.util.spec_from_file_location("run_pipeline", _MODULE_PATH)
run_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["run_pipeline"] = run_pipeline
_spec.loader.exec_module(run_pipeline)


def _write_configs(tmp_path):
    """Create a minimal configs/ dir with base.yaml."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    base = {
        "input_path": None,
        "output_path": None,
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega"},
        "semantics": {"enabled": False},
        "localization": {"enabled": False, "matcher": "loma"},
    }
    (cfg_dir / "base.yaml").write_text(yaml.dump(base))
    return cfg_dir


def _capture_run_all(monkeypatch):
    """Swap run_all for a recorder so main() never reconstructs anything."""
    captured = {}

    def fake_run_all(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run_pipeline, "run_all", fake_run_all)
    return captured


########################################
# main() — argparse wiring into run_all
########################################


def test_main_wires_keep_viewer_flag_through_to_run_all(tmp_path, monkeypatch):
    """--keep-viewer on the CLI reaches run_all's keep_viewer kwarg (end-to-end argparse check)."""
    cfg_dir = _write_configs(tmp_path)
    v = tmp_path / "a.mp4"
    v.touch()

    captured = _capture_run_all(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--output-root",
            str(tmp_path / "out"),
            "--config-dir",
            str(cfg_dir),
            "--keep-viewer",
            str(v),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()
    assert exc.value.code == 0
    assert captured["keep_viewer"] is True


def test_main_defaults_keep_viewer_false(tmp_path, monkeypatch):
    """Without --keep-viewer, run_all is called with keep_viewer=False."""
    cfg_dir = _write_configs(tmp_path)
    v = tmp_path / "a.mp4"
    v.touch()

    captured = _capture_run_all(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--output-root", str(tmp_path / "out"), "--config-dir", str(cfg_dir), str(v)],
    )
    with pytest.raises(SystemExit):
        run_pipeline.main()
    assert captured["keep_viewer"] is False


def test_main_parses_stages_and_override_config(tmp_path, monkeypatch):
    """--stages splits on commas; --config is loaded as the shared override dict."""
    cfg_dir = _write_configs(tmp_path)
    v = tmp_path / "a.mp4"
    v.touch()
    override = tmp_path / "override.yaml"
    override.write_text(yaml.dump({"localization": {"enabled": True}}))

    captured = _capture_run_all(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--output-root",
            str(tmp_path / "out"),
            "--config-dir",
            str(cfg_dir),
            "--config",
            str(override),
            "--stages",
            "preproc, pointcloud",
            str(v),
        ],
    )
    with pytest.raises(SystemExit):
        run_pipeline.main()
    assert captured["stages"] == ["preproc", "pointcloud"]
    assert captured["override_config"] == {"localization": {"enabled": True}}
    assert captured["videos"] == [v]


def test_main_exits_two_when_no_videos_found(tmp_path, monkeypatch):
    """An empty directory yields no videos: exit code 2, run_all never called."""
    cfg_dir = _write_configs(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()

    captured = _capture_run_all(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--output-root", str(tmp_path / "out"), "--config-dir", str(cfg_dir), str(empty)],
    )
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()
    assert exc.value.code == 2
    assert captured == {}
