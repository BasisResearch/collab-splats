from pathlib import Path
from unittest.mock import MagicMock

import pytest

from collab_splats.dashboard.sources import SessionSource


def _client():
    c = MagicMock()
    c.remote_name = "collab-data"
    c._cmd = lambda *a: ["rclone", *a]
    return c


def test_list_sessions_returns_dir_names_sorted():
    client = _client()
    client.list_directory.return_value = [
        {"Name": "2026_05_08", "IsDir": True},
        {"Name": "2026_05_07", "IsDir": True},
        {"Name": "notes.txt", "IsDir": False},
    ]
    src = SessionSource(client)
    assert src.list_sessions() == ["2026_05_07", "2026_05_08"]
    client.list_directory.assert_called_with("fieldwork_curated", "reconstruction")


def test_list_videos_filters_mp4():
    client = _client()
    client.list_directory.return_value = [
        {"Name": "clip_01.mp4", "IsDir": False},
        {"Name": "clip_02.MP4", "IsDir": False},
        {"Name": "meta.json", "IsDir": False},
        {"Name": "sub", "IsDir": True},
    ]
    src = SessionSource(client)
    assert src.list_videos("2026_05_07") == ["clip_01.mp4", "clip_02.MP4"]
    client.list_directory.assert_called_with(
        "fieldwork_curated", "reconstruction/2026_05_07"
    )


def test_fetch_video_invokes_rclone_copyto(monkeypatch, tmp_path):
    client = _client()
    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("collab_splats.dashboard.sources.subprocess.run", fake_run)
    src = SessionSource(client)
    local = src.fetch_video("2026_05_07", "clip_03.mp4", tmp_path)
    assert local == tmp_path / "clip_03.mp4"
    assert calls["cmd"] == [
        "rclone", "copyto",
        "collab-data:fieldwork_curated/reconstruction/2026_05_07/clip_03.mp4",
        str(tmp_path / "clip_03.mp4"),
    ]


def test_has_processed_true_when_listing_nonempty():
    client = _client()
    client.list_directory.return_value = [{"Name": "feedforward.zarr", "IsDir": True}]
    src = SessionSource(client)
    assert src.has_processed("2026_05_07", "clip_03") is True
    client.list_directory.assert_called_with(
        "fieldwork_processed", "reconstruction/2026_05_07/clip_03"
    )


def test_has_processed_false_on_error():
    client = _client()
    client.list_directory.side_effect = RuntimeError("not found")
    src = SessionSource(client)
    assert src.has_processed("2026_05_07", "clip_03") is False


def test_push_outputs_calls_copy_local_to_remote(tmp_path):
    client = _client()
    client.copy_local_to_remote.return_value = True
    src = SessionSource(client)
    src.push_outputs(tmp_path, "2026_05_07", "clip_03")
    client.copy_local_to_remote.assert_called_with(
        str(tmp_path), "fieldwork_processed", "reconstruction/2026_05_07/clip_03"
    )
