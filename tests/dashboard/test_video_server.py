# tests/dashboard/test_video_server.py
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from collab_splats.dashboard.video_server import VideoFileServer


def _free_port() -> int:
    """Find a free localhost port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """Fake video file — 9000 bytes of repeating pattern."""
    f = tmp_path / "test.mp4"
    f.write_bytes(b"FAKEVIDEO" * 1000)
    return f


@pytest.fixture
def server(video_file: Path):
    port = _free_port()
    srv = VideoFileServer(port=port)
    srv.register(video_file)
    yield srv
    srv.shutdown()


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, b"", {}


def test_registered_file_200(server, video_file):
    url = f"http://127.0.0.1:{server.port}{video_file}"
    status, body, headers = _get(url)
    assert status == 200
    assert body == video_file.read_bytes()
    assert headers.get("Accept-Ranges") == "bytes"


def test_unregistered_path_403(server, tmp_path):
    other = tmp_path / "other.mp4"
    other.write_bytes(b"OTHER")
    url = f"http://127.0.0.1:{server.port}{other}"
    status, _, _ = _get(url)
    assert status == 403


def test_path_traversal_blocked(server):
    url = f"http://127.0.0.1:{server.port}/tmp/../etc/passwd"
    status, _, _ = _get(url)
    assert status == 403


def test_range_request_206(server, video_file):
    url = f"http://127.0.0.1:{server.port}{video_file}"
    status, body, headers = _get(url, headers={"Range": "bytes=0-8"})
    assert status == 206
    assert body == b"FAKEVIDEO"
    assert "Content-Range" in headers


def test_range_mid_file(server, video_file):
    url = f"http://127.0.0.1:{server.port}{video_file}"
    status, body, headers = _get(url, headers={"Range": "bytes=9-17"})
    assert status == 206
    assert body == b"FAKEVIDEO"
    assert headers.get("Content-Range") == "bytes 9-17/9000"


def test_unregister_revokes_access(server, video_file):
    url = f"http://127.0.0.1:{server.port}{video_file}"
    status, _, _ = _get(url)
    assert status == 200
    server.unregister(video_file)
    status, _, _ = _get(url)
    assert status == 403


def test_suffix_range_last_bytes(server, video_file):
    """bytes=-9 should return the last 9 bytes (one FAKEVIDEO repetition)."""
    url = f"http://127.0.0.1:{server.port}{video_file}"
    status, body, headers = _get(url, headers={"Range": "bytes=-9"})
    assert status == 206
    assert body == b"FAKEVIDEO"
    assert "Content-Range" in headers


def test_invalid_range_returns_416(server, video_file):
    """Malformed Range header should return 416."""
    url = f"http://127.0.0.1:{server.port}{video_file}"
    # start > end is invalid
    status, _, _ = _get(url, headers={"Range": "bytes=9000-100"})
    assert status == 416
