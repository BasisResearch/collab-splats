"""Unit tests for evals/data/download_datasets.py (7-Scenes portion) — no network required."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure evals/data/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evals" / "data"))

from download_datasets import SCENES, _scene_already_downloaded, main


# ---------------------------------------------------------------------------
# _scene_already_downloaded
# ---------------------------------------------------------------------------


def test_already_downloaded_true(tmp_path):
    """Seq dir exists and contains at least one *.color.png → True."""
    seq_dir = tmp_path / "seq-01"
    seq_dir.mkdir()
    (seq_dir / "frame-000000.color.png").touch()

    assert _scene_already_downloaded(tmp_path, "seq-01") is True


def test_already_downloaded_false_missing_dir(tmp_path):
    """Seq dir does not exist → False."""
    assert _scene_already_downloaded(tmp_path, "seq-01") is False


def test_already_downloaded_false_empty_dir(tmp_path):
    """Seq dir exists but no *.color.png files → False."""
    seq_dir = tmp_path / "seq-01"
    seq_dir.mkdir()
    # No color png files
    (seq_dir / "frame-000000.pose.txt").touch()

    assert _scene_already_downloaded(tmp_path, "seq-01") is False


# ---------------------------------------------------------------------------
# SCENES dict
# ---------------------------------------------------------------------------


def test_scenes_dict_has_required():
    """SCENES must include the three benchmarked scenes at minimum."""
    assert "chess" in SCENES
    assert "fire" in SCENES
    assert "office" in SCENES


def test_scenes_dict_has_all_seven():
    """SCENES should cover all 7 scenes in the dataset."""
    expected = {"chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"}
    assert expected.issubset(set(SCENES.keys()))


def test_scenes_dict_urls_are_microsoft_cdn():
    """All URLs should point to the Microsoft Research CDN."""
    cdn_base = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
    for scene, url in SCENES.items():
        assert url.startswith(cdn_base), f"{scene} URL does not start with CDN base"
        assert url.endswith(".zip"), f"{scene} URL does not end with .zip"


# ---------------------------------------------------------------------------
# main() — --list mode
# ---------------------------------------------------------------------------


def test_main_list_mode(capsys):
    """--list prints scene names and URLs then exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["7scenes", "--list"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    output = captured.out
    for scene in SCENES:
        assert scene in output, f"'{scene}' missing from --list output"
    # URLs should also appear
    assert "https://download.microsoft.com" in output


# ---------------------------------------------------------------------------
# download_scene — mocked network
# ---------------------------------------------------------------------------


def test_download_scene_skips_if_already_present(tmp_path):
    """download_scene returns immediately if seq dir already has color.png files."""
    from download_datasets import download_scene

    scene = "fire"
    seq = "seq-01"
    # REPO_ROOT / "data" / "7scenes" / scene / seq
    seq_dir = tmp_path / "data" / "7scenes" / scene / seq
    seq_dir.mkdir(parents=True)
    (seq_dir / "frame-000000.color.png").touch()

    # Patch REPO_ROOT so data lands in tmp_path
    with patch("download_datasets.REPO_ROOT", tmp_path):
        result = download_scene(scene, seq=seq, force=False)

    assert result == seq_dir


def test_download_scene_raises_on_bad_scene():
    """download_scene raises ValueError for unknown scene names."""
    from download_datasets import download_scene

    with pytest.raises(ValueError):
        download_scene("nonexistent_scene")
