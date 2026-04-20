"""Tests for video filesystem discovery."""
from pathlib import Path
import pytest
from collab_splats.dashboard.video_discovery import discover_videos, yaml_name_for_video


def test_discover_videos_single_video(tmp_path):
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()
    result = discover_videos(tmp_path)
    assert list(result.keys()) == ["birds"]
    assert list(result["birds"].keys()) == ["2024-02-06"]
    assert result["birds"]["2024-02-06"] == [video]


def test_discover_videos_multiple_species(tmp_path):
    for species, date, vid in [
        ("birds", "2024-02-06", "C0043.MP4"),
        ("ants", "2025-11-16", "GH010210.MP4"),
    ]:
        p = tmp_path / species / date / "SplatsSD" / vid
        p.parent.mkdir(parents=True)
        p.touch()
    result = discover_videos(tmp_path)
    assert set(result.keys()) == {"birds", "ants"}


def test_discover_videos_multiple_dates(tmp_path):
    for date, vid in [("2024-02-06", "C0043.MP4"), ("2024-05-27", "GH010097.MP4")]:
        p = tmp_path / "birds" / date / "SplatsSD" / vid
        p.parent.mkdir(parents=True)
        p.touch()
    result = discover_videos(tmp_path)
    assert set(result["birds"].keys()) == {"2024-02-06", "2024-05-27"}


def test_discover_videos_empty_base(tmp_path):
    assert discover_videos(tmp_path) == {}


def test_discover_videos_missing_base():
    assert discover_videos(Path("/nonexistent/path/xyz")) == {}


def test_discover_videos_ignores_dir_without_splatssd(tmp_path):
    (tmp_path / "birds" / "2024-02-06").mkdir(parents=True)
    assert discover_videos(tmp_path) == {}


def test_discover_videos_ignores_empty_splatssd(tmp_path):
    (tmp_path / "birds" / "2024-02-06" / "SplatsSD").mkdir(parents=True)
    assert discover_videos(tmp_path) == {}


def test_discover_videos_lowercase_extension(tmp_path):
    video = tmp_path / "rats" / "2024-07-11" / "SplatsSD" / "C0119.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    result = discover_videos(tmp_path)
    assert result["rats"]["2024-07-11"] == [video]


def test_yaml_name_for_video_basic():
    assert yaml_name_for_video("birds", "2024-02-06", "C0043") == "birds_date-02062024_video-C0043"


def test_yaml_name_for_video_ants():
    assert yaml_name_for_video("ants", "2025-11-16", "GH010210") == "ants_date-11162025_video-GH010210"


def test_yaml_name_for_video_rats():
    assert yaml_name_for_video("rats", "2024-07-11", "C0119") == "rats_date-07112024_video-C0119"
