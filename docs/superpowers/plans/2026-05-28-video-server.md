# Video Playback Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken local-path video display in `PreprocessPane` with an allowlist-gated HTTP server that supports range requests, enabling real browser video playback.

**Architecture:** A `ThreadingHTTPServer` daemon (stdlib) listens on `127.0.0.1:7863`. A custom handler checks every incoming request against an explicit `Set[Path]` allowlist before serving — anything not registered returns 403. `App.__init__` starts the server and passes it to `PreprocessPane`, which registers the video path and feeds a `http://localhost:7863/...` URL to `pn.pane.Video`.

**Tech Stack:** Python stdlib (`http.server`, `threading`, `mimetypes`, `urllib.parse`), Panel `pn.pane.Video`, pytest.

---

## File Map

| File | Action |
|---|---|
| `collab_splats/dashboard/video_server.py` | **Create** — `VideoFileServer`, `_VideoRequestHandler`, `start_video_server()` |
| `tests/dashboard/test_video_server.py` | **Create** — unit tests for server + handler |
| `collab_splats/dashboard/app.py` | **Modify** — start server in `__init__`, pass to `PreprocessPane` |
| `collab_splats/dashboard/panes/preprocess.py` | **Modify** — accept `video_server` arg, use URL in `_load_video`, drop cv2 |
| `tests/dashboard/test_preprocess.py` | **Modify** — add mock `video_server` to `PreprocessPane` instantiations |

---

## Task 1: `VideoFileServer` — write failing tests

**Files:**
- Create: `tests/dashboard/test_video_server.py`

- [ ] **Step 1: Write the test file**

```python
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
    status, body, _ = _get(url, headers={"Range": "bytes=9-17"})
    assert status == 206
    assert body == b"FAKEVIDEO"


def test_unregister_revokes_access(server, video_file):
    url = f"http://127.0.0.1:{server.port}{video_file}"
    status, _, _ = _get(url)
    assert status == 200
    server.unregister(video_file)
    status, _, _ = _get(url)
    assert status == 403
```

- [ ] **Step 2: Run tests — expect import error (module doesn't exist yet)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_video_server.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'collab_splats.dashboard.video_server'`

---

## Task 2: Implement `VideoFileServer`

**Files:**
- Create: `collab_splats/dashboard/video_server.py`

- [ ] **Step 1: Write the module**

```python
# collab_splats/dashboard/video_server.py
from __future__ import annotations

import logging
import mimetypes
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoFileServer:
    """Range-request HTTP server that serves only explicitly registered video files."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.allowlist: set[Path] = set()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _VideoRequestHandler)
        # Share the allowlist reference with the handler via the server object
        self._server.allowlist = self.allowlist  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("VideoFileServer started on 127.0.0.1:%d", port)

    def register(self, path: Path) -> None:
        """Add resolved absolute path to the serve allowlist."""
        self.allowlist.add(path.resolve())
        logger.debug("VideoFileServer registered: %s", path)

    def unregister(self, path: Path) -> None:
        """Remove resolved absolute path from the serve allowlist."""
        self.allowlist.discard(path.resolve())
        logger.debug("VideoFileServer unregistered: %s", path)

    def shutdown(self) -> None:
        """Stop the HTTP server (used in tests; daemon thread exits automatically in prod)."""
        self._server.shutdown()


class _VideoRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler: serves only allowlisted files with range-request support."""

    def do_GET(self) -> None:  # noqa: N802
        # Decode URL path and resolve to canonical absolute path
        url_path = urllib.parse.unquote(self.path.split("?")[0])
        requested = Path(url_path).resolve()

        # Allowlist gate — blocks path traversal, symlink tricks, and unregistered files
        if requested not in self.server.allowlist:  # type: ignore[attr-defined]
            self.send_error(403, "Forbidden")
            return

        if not requested.is_file():
            self.send_error(404, "Not Found")
            return

        file_size = requested.stat().st_size
        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"

        # Parse Range header if present
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1

        if range_header:
            range_spec = range_header.replace("bytes=", "").strip()
            parts = range_spec.split("-")
            try:
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            except (ValueError, IndexError):
                self.send_error(416, "Range Not Satisfiable")
                return
            end = min(end, file_size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            length = file_size
            self.send_response(200)

        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        # Stream file in 64 KB chunks to avoid loading large videos into memory
        with open(requested, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("VideoServer: " + format, *args)


def start_video_server(port: int = 7863) -> VideoFileServer:
    """Start the video file server on the given port and return the instance.

    Raises OSError if the port is already in use.
    """
    return VideoFileServer(port=port)
```

- [ ] **Step 2: Run tests — expect all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_video_server.py -v
```

Expected: 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/dashboard/video_server.py tests/dashboard/test_video_server.py
git commit -m "feat(dashboard): add allowlist-gated HTTP video server with range-request support"
```

---

## Task 3: Integrate server into `app.py`

**Files:**
- Modify: `collab_splats/dashboard/app.py`

- [ ] **Step 1: Add import at top of `app.py` (after existing imports)**

Add after line 16 (`logger = logging.getLogger(__name__)`):

```python
from collab_splats.dashboard.video_server import start_video_server
```

- [ ] **Step 2: Start server in `App.__init__` and pass to `PreprocessPane`**

In `App.__init__`, replace:
```python
        self._tabs: pn.Tabs | None = None

        self._preprocess = PreprocessPane(state=self._state, op_log=self._op_log)
```

With:
```python
        self._tabs: pn.Tabs | None = None
        self._video_server = start_video_server(port=7863)

        self._preprocess = PreprocessPane(
            state=self._state, op_log=self._op_log, video_server=self._video_server
        )
```

- [ ] **Step 3: Verify existing app tests still import cleanly**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_app.py -v 2>&1 | tail -15
```

Expected: tests may fail on `PreprocessPane` constructor mismatch — that's fixed in Task 5.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/app.py
git commit -m "feat(dashboard): start video server in App.__init__, pass to PreprocessPane"
```

---

## Task 4: Update `preprocess.py` — use server URL, drop cv2

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`

- [ ] **Step 1: Add import at top of `preprocess.py`**

After the existing imports block (after line 32, `logger = logging.getLogger(__name__)`), add:

```python
from collab_splats.dashboard.video_server import VideoFileServer
```

- [ ] **Step 2: Update `PreprocessPane.__init__` signature**

Replace:
```python
    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
```

With:
```python
    def __init__(self, state: AppState, op_log: OperationLog, video_server: VideoFileServer, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._video_server = video_server
```

- [ ] **Step 3: Replace `_load_video` — drop cv2, use HTTP URL**

Replace the entire `_load_video` method:
```python
    def _load_video(self, video_path: Path) -> None:
        """Extract first frame as thumbnail + fetch metadata in background."""
        import cv2  # optional heavy dep — imported here intentionally
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(frame_rgb).save(buf, format="PNG")
            self._video_pane.object = buf.getvalue()
            self._video_pane.visible = True
        self._video_info_html.object = (
            f"<p style='font-size:11px;color:#aaa'>{video_path.name} · loading info…</p>"
        )
        # Fetch full metadata (frame count, fps) without blocking IOLoop
        threading.Thread(target=self._fetch_video_info, args=(video_path,), daemon=True).start()
```

With:
```python
    def _load_video(self, video_path: Path) -> None:
        """Register video with side server and begin playback; fetch metadata in background."""
        self._video_server.register(video_path)
        url = f"http://localhost:{self._video_server.port}{video_path}"
        self._video_pane.object = url
        self._video_pane.visible = True
        self._video_info_html.object = (
            f"<p style='font-size:11px;color:#aaa'>{video_path.name} · loading info…</p>"
        )
        # Fetch full metadata (frame count, fps) without blocking IOLoop
        threading.Thread(target=self._fetch_video_info, args=(video_path,), daemon=True).start()
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py
git commit -m "fix(preprocess): serve video via range-request HTTP server instead of local path"
```

---

## Task 5: Fix `test_preprocess.py` — add mock server

**Files:**
- Modify: `tests/dashboard/test_preprocess.py`

The existing tests instantiate `PreprocessPane(state=state, op_log=OperationLog())` without `video_server`. They also have a stale test (`test_video_path_change_uses_pn_state_execute`) that asserts `pn.state.execute` is called — but the current implementation calls `_load_video` directly. Both issues must be fixed.

- [ ] **Step 1: Add a `mock_video_server` fixture**

Add after the existing imports in `test_preprocess.py`:

```python
import unittest.mock as mock
```

(already imported — confirm it's there on line 2)

Add this fixture after the imports:

```python
@pytest.fixture
def mock_video_server():
    """Minimal stand-in for VideoFileServer — avoids binding a real port in unit tests."""
    srv = mock.MagicMock()
    srv.port = 17863
    return srv
```

- [ ] **Step 2: Replace stale `test_video_path_change_uses_pn_state_execute` with updated test**

The old test patched `pn.state.execute` expecting it to be called — that behavior was removed in a prior refactor. Replace the entire function:

```python
def test_video_path_change_loads_video(tmp_path, mock_video_server):
    """_on_video_path_change must call _load_video when video path is valid."""
    state = AppState()
    pane = PreprocessPane(state=state, op_log=OperationLog(), video_server=mock_video_server)

    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake")

    class FakeEvent:
        new = str(video_file)

    with mock.patch.object(pane, "_load_video") as mock_load:
        pane._on_video_path_change(FakeEvent())

    mock_load.assert_called_once_with(video_file)
```

- [ ] **Step 3: Run all dashboard tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v 2>&1 | tail -30
```

Expected: all tests pass (including the 6 new video server tests).

- [ ] **Step 4: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_preprocess.py
git commit -m "test(preprocess): update PreprocessPane fixture to supply mock video_server"
```
