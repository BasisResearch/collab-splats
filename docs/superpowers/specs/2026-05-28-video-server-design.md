# Video Playback Server — Design Spec

**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** `collab_splats/dashboard/` — new `video_server.py`, changes to `app.py` and `panes/preprocess.py`

---

## Problem

`pn.pane.Video` passed a local file path streams the entire file through Bokeh's WebSocket
server, which does not support HTTP range requests. The browser hangs waiting for the full
file (often 700 MB+) before playback can start.

## Solution

Start a dedicated `ThreadingHTTPServer` (stdlib) at app startup that serves registered video
files with full HTTP range-request support. Only explicitly registered paths are served;
everything else returns 403.

---

## Architecture

Three files change:

| File | Change |
|---|---|
| `collab_splats/dashboard/video_server.py` | **New** — `VideoFileServer` + handler |
| `collab_splats/dashboard/app.py` | Start server in `__init__`; pass to `PreprocessPane` |
| `collab_splats/dashboard/panes/preprocess.py` | Accept server arg; register path; set video URL |

---

## `VideoFileServer`

### Class interface

```python
class VideoFileServer:
    port: int
    allowlist: set[Path]          # resolved absolute paths

    def register(self, path: Path) -> None: ...
    def unregister(self, path: Path) -> None: ...
```

`start_video_server(port: int = 7863) -> VideoFileServer` — module-level factory that
binds the server and starts it in a daemon thread. Raises `OSError` on bind failure.

### Request handler

`_VideoRequestHandler(BaseHTTPRequestHandler)`:

1. URL-decode `self.path` → construct absolute `Path`
2. Call `Path.resolve()` to canonicalize (eliminates `../` and symlink traversal)
3. If resolved path **not in** `server.allowlist` → `403 Forbidden`, return
4. Parse `Range:` header if present → `bytes=start[-end]`
5. Open file, seek to `start`
6. Respond `206 Partial Content` (range request) or `200 OK` (full), stream in 64 KB chunks
7. Headers: `Accept-Ranges: bytes`, `Content-Length`, `Content-Range` (206 only),
   `Content-Type` inferred via `mimetypes.guess_type(path)` (covers `.mp4`, `.MP4`, `.mov`, etc.)

### Security model

- Allowlist stores `Path(video_path).resolve()` — resolved at registration time
- Handler resolves the requested path before lookup — blocks symlink traversal and path traversal
- No directory listing is ever served
- Server binds `localhost` only (`127.0.0.1`) — not exposed on the container's external interface

### Port

Default `7863`. Configurable via `start_video_server(port=...)`. If port is taken at startup,
`OSError` is raised immediately with a clear message (fail-fast; don't silently pick a random
port, as the URL embedded in `pn.pane.Video` would be wrong).

---

## Integration

### `App.__init__`

```python
from collab_splats.dashboard.video_server import start_video_server

self._video_server = start_video_server(port=7863)
self._preprocess = PreprocessPane(
    state=self._state, op_log=self._op_log, video_server=self._video_server
)
```

### `PreprocessPane.__init__`

Accept `video_server: VideoFileServer` as a new required arg. Store as `self._video_server`.

### `PreprocessPane._load_video(video_path: Path)`

Replace current cv2 + PNG-bytes approach with:

1. `self._video_server.register(video_path)`
2. `url = f"http://localhost:{self._video_server.port}{video_path}"`
   — e.g. `http://localhost:7863/workspace/fieldwork-data/seq01/video.MP4`
3. `self._video_pane.object = url`
4. `self._video_pane.visible = True`
5. Keep background thread for metadata fetch (frame count / fps / duration display unchanged)

Drop: cv2 import, first-frame PNG extraction, `io.BytesIO` buffer in `_load_video`.
The cv2 import at line 229 was already guarded as an optional dep; it can be removed from
`_load_video` entirely (cv2 is still used in `frame_sampling.py` upstream, not here).

### `_seek_to_frame`

Unchanged — `self._video_pane.time = frame_idx / fps` still works once the video is
properly loaded via range-request-aware URL.

---

## What does NOT change

- Frame extraction pipeline (`_run_extraction`, zarr write, thumbnail strip)
- Metrics panel (Bokeh figures, TapTool)
- `_fetch_video_info` background thread
- `_seek_to_frame` logic
- All other panes

---

## Testing notes

- Unit-test `VideoFileServer` directly: register a file, GET it, GET a range, GET an
  unregistered path (expect 403), attempt path traversal (expect 403).
- Manual test: load a 700 MB video in the dashboard — playback should start within ~2s
  without buffering the full file.
