from pathlib import Path
from unittest.mock import MagicMock

import pytest

from collab_splats.dashboard.sources import SessionSource, parse_rclone_percent


def _client():
    c = MagicMock()
    c.remote_name = "collab-data"
    c._cmd = lambda *a: ["rclone", *a]
    return c


class _FakeProc:
    """Context-manager stub for subprocess.Popen: streams `lines`, exits with `code`."""

    def __init__(self, lines=(), code=0):
        self.stdout = iter(lines)
        self._code = code

    def wait(self):
        return self._code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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
    client.list_directory.assert_called_with("fieldwork_curated", "reconstruction/2026_05_07")


def test_fetch_video_invokes_rclone_copyto(monkeypatch, tmp_path):
    client = _client()
    calls = {}

    def fake_popen(cmd, stdout, stderr, text):
        calls["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr("collab_splats.dashboard.sources.subprocess.Popen", fake_popen)
    src = SessionSource(client)
    local = src.fetch_video("2026_05_07", "clip_03.mp4", tmp_path)
    assert local == tmp_path / "clip_03.mp4"
    assert calls["cmd"][:2] == ["rclone", "copyto"]
    assert "collab-data:fieldwork_curated/reconstruction/2026_05_07/clip_03.mp4" in calls["cmd"]
    assert str(tmp_path / "clip_03.mp4") in calls["cmd"]


def test_list_sessions_is_memoized():
    client = _client()
    client.list_directory.return_value = [{"Name": "2026_05_07", "IsDir": True}]
    src = SessionSource(client=client)

    src.list_sessions()
    src.list_sessions()
    assert client.list_directory.call_count == 1  # second call served from cache


def test_invalidate_clears_memoized_listing():
    client = _client()
    client.list_directory.return_value = [{"Name": "2026_05_07", "IsDir": True}]
    src = SessionSource(client=client)

    src.list_sessions()
    src.invalidate()
    src.list_sessions()
    assert client.list_directory.call_count == 2


def test_push_outputs_invalidates_scene_listings(monkeypatch, tmp_path):
    client = _client()
    client.list_directory.return_value = [{"Name": "feedforward.zarr", "IsDir": True}]
    monkeypatch.setattr("collab_splats.dashboard.sources.subprocess.Popen", lambda *a, **k: _FakeProc())
    src = SessionSource(client)

    # Memoize both scene listers, push, then verify both re-hit rclone (cache dropped).
    src.has_processed("2026_05_07", "clip_03")
    src.list_localization_dbs("2026_05_07", "clip_03")
    before = client.list_directory.call_count
    src.push_outputs(tmp_path, "2026_05_07", "clip_03")
    src.has_processed("2026_05_07", "clip_03")
    src.list_localization_dbs("2026_05_07", "clip_03")
    assert client.list_directory.call_count == before + 2


def test_ttl_expiry_and_keyed_invalidate():
    client = _client()
    client.list_directory.return_value = [{"Name": "2026_05_07", "IsDir": True}]
    src = SessionSource(client)

    # TTL of zero -> every call refetches.
    src._listing_ttl = 0
    src.list_sessions()
    src.list_sessions()
    assert client.list_directory.call_count == 2

    # Keyed invalidate drops only the named key; other entries stay cached.
    src._listing_ttl = 60.0
    src.list_sessions()
    src.list_videos("s")
    count = client.list_directory.call_count
    src.invalidate(("list_videos", "s"))
    src.list_sessions()  # still cached, no new call
    assert client.list_directory.call_count == count
    src.list_videos("s")  # dropped, refetches
    assert client.list_directory.call_count == count + 1


def test_has_processed_true_when_listing_nonempty():
    client = _client()
    client.list_directory.return_value = [{"Name": "feedforward.zarr", "IsDir": True}]
    src = SessionSource(client)
    assert src.has_processed("2026_05_07", "clip_03") is True
    client.list_directory.assert_called_with("fieldwork_processed", "reconstruction/2026_05_07/clip_03")


def test_has_processed_false_on_error():
    client = _client()
    client.list_directory.side_effect = RuntimeError("not found")
    src = SessionSource(client)
    assert src.has_processed("2026_05_07", "clip_03") is False


def test_pull_processed_invokes_rclone_copy(monkeypatch, tmp_path):
    client = _client()
    calls = {}

    def fake_popen(cmd, stdout, stderr, text):
        calls["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr("collab_splats.dashboard.sources.subprocess.Popen", fake_popen)
    src = SessionSource(client)
    out = src.pull_processed("2026_05_07", "clip_03", tmp_path)
    assert out == tmp_path
    assert calls["cmd"][:2] == ["rclone", "copy"]
    assert "collab-data:fieldwork_processed/reconstruction/2026_05_07/clip_03" in calls["cmd"]
    assert str(tmp_path) in calls["cmd"]


def test_parse_rclone_percent_extracts_percentage():
    line = "Transferred:   1.234 GiB / 5.678 GiB, 21%, 45.6 MiB/s, ETA 1m30s"
    assert parse_rclone_percent(line) == 21
    assert parse_rclone_percent("no percent here") is None
    assert parse_rclone_percent("Transferred: 0 / 0 Bytes, 100%, 0/s") == 100


def test_pull_processed_streams_stats_to_on_line(monkeypatch, tmp_path):
    lines_seen = []

    def fake_popen(cmd, stdout, stderr, text):
        assert "--stats-one-line" in cmd
        return _FakeProc(lines=["Transferred: 1 GiB / 2 GiB, 50%, 10 MiB/s\n"])

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    src = SessionSource(client=_client())
    src.pull_processed("2026_05_07", "clip_03", tmp_path, on_line=lines_seen.append)
    assert any("50%" in ln for ln in lines_seen)


def test_push_outputs_streams_rclone_copy(monkeypatch, tmp_path):
    client = _client()
    calls = {}
    lines = []

    def fake_popen(cmd, stdout, stderr, text):
        calls["cmd"] = cmd
        return _FakeProc(lines=["Transferred: 1.2 MiB / 1.2 MiB\n", "\n"])

    monkeypatch.setattr("collab_splats.dashboard.sources.subprocess.Popen", fake_popen)
    src = SessionSource(client)
    src.push_outputs(tmp_path, "2026_05_07", "clip_03", on_line=lines.append)

    # Recursive idempotent `copy` verb, not file-only `copyto`; native retries + stats.
    assert calls["cmd"][:2] == ["rclone", "copy"]
    assert "copyto" not in calls["cmd"]
    for flag in ("--transfers", "--retries", "--stats-one-line"):
        assert flag in calls["cmd"]
    assert calls["cmd"][-2:] == [
        str(tmp_path),
        "collab-data:fieldwork_processed/reconstruction/2026_05_07/clip_03",
    ]
    assert lines == ["Transferred: 1.2 MiB / 1.2 MiB"]


def test_push_outputs_raises_on_nonzero_exit(monkeypatch, tmp_path):
    client = _client()

    monkeypatch.setattr(
        "collab_splats.dashboard.sources.subprocess.Popen", lambda *a, **k: _FakeProc(code=1)
    )
    src = SessionSource(client)
    with pytest.raises(RuntimeError):
        src.push_outputs(tmp_path, "2026_05_07", "clip_03")
