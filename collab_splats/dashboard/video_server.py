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
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _VideoRequestHandler)
        # Share the allowlist reference and lock with the handler via the server object
        self._server.allowlist = self.allowlist  # type: ignore[attr-defined]
        self._server._lock = self._lock  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("VideoFileServer started on 127.0.0.1:%d", port)

    def register(self, path: Path) -> None:
        """Add resolved absolute path to the serve allowlist."""
        with self._lock:
            self.allowlist.add(path.resolve())
        logger.debug("VideoFileServer registered: %s", path)

    def unregister(self, path: Path) -> None:
        """Remove resolved absolute path from the serve allowlist."""
        with self._lock:
            self.allowlist.discard(path.resolve())
        logger.debug("VideoFileServer unregistered: %s", path)

    def shutdown(self) -> None:
        """Stop the HTTP server (used in tests; daemon thread exits automatically in prod)."""
        self._server.shutdown()


class _VideoRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler: serves only allowlisted files with range-request support."""

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET request: serve an allowlisted file with optional Range support."""
        # Decode URL path and resolve to canonical absolute path
        url_path = urllib.parse.unquote(self.path.split("?")[0])
        requested = Path(url_path).resolve()

        # Allowlist gate — blocks path traversal, symlink tricks, and unregistered files
        with self.server._lock:  # type: ignore[attr-defined]
            allowed = requested in self.server.allowlist  # type: ignore[attr-defined]
        if not allowed:
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
                if not parts[0]:
                    # Suffix range: bytes=-N means last N bytes
                    suffix = int(parts[1])
                    start = max(0, file_size - suffix)
                    end = file_size - 1
                else:
                    start = int(parts[0])
                    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            except (ValueError, IndexError):
                self.send_error(416, "Range Not Satisfiable")
                return
            end = min(end, file_size - 1)
            if start < 0 or start > end:
                self.send_error(416, "Range Not Satisfiable")
                return
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
        try:
            with open(requested, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("VideoServer: client disconnected during transfer")

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("VideoServer: " + format, *args)


def start_video_server(port: int = 7863) -> VideoFileServer:
    """Start the video file server on the given port and return the instance.

    Raises OSError if the port is already in use.
    """
    return VideoFileServer(port=port)
