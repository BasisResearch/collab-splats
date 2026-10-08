"""
Shared progress and log bus for dashboard background operations.

- one OperationLog per server, read by every session's poll
"""

from __future__ import annotations

import contextlib
import html
import logging
import threading
import time
from typing import Callable, Iterator

from collab_data.data_dashboard.rclone_client import parse_percent

########
# Logging bridge
########


class _OpLogHandler(logging.Handler):
    """
    Forward log records emitted during a run into an OperationLog's log_lines.
    """

    def __init__(self, op_log: OperationLog) -> None:
        """
        Bind the handler to one OperationLog at INFO level.
        """
        super().__init__(level=logging.INFO)
        self._op_log = op_log

    def emit(self, record: logging.LogRecord) -> None:
        """
        Append the record's message through op_log's thread-safe line buffer.
        """
        try:
            self._op_log.append_line(record.getMessage())
        except (TypeError, ValueError):  # bad format args; never let logging crash the run
            self.handleError(record)


########
# OperationLog
########


class OperationLog:
    """
    Progress state and capped log lines shared by every session.

    - background threads call start_op / update_progress / finish_op
    - thread-safe: a lock guards every mutation
    """

    def __init__(self, max_lines: int = 100) -> None:
        """
        Start idle with an empty log.

        Args:
            max_lines: log lines kept; older lines drop off.
        """
        self.current_op = ""
        self.progress = 0
        self.is_running = False
        self.log_lines: list[str] = []
        self._max_lines = max_lines
        self._lock = threading.Lock()
        self._version = 0  # bumped on every visible mutation; UI polls compare-and-skip

    @property
    def version(self) -> int:
        """
        Monotonic change counter; pollers re-render only when it moves.

        Returns:
            The current counter value.
        """
        return self._version

    @contextlib.contextmanager
    def step(self, label: str) -> Iterator[None]:
        """
        Log a step's start, then its duration or failure on exit.

        - entry line `<label>…`; exit line `<label> done (Xs)` or `<label> FAILED (Xs): <error>`
        - thread-safe and exception-safe; re-raises so callers still see failures

        Args:
            label: step name shown in the log.

        Yields:
            Nothing; the body of the `with` block is the timed step.
        """
        self.append_line(f"{label}…")
        t0 = time.perf_counter()

        try:
            yield
        except Exception as exc:
            self.append_line(f"{label} FAILED ({time.perf_counter() - t0:.1f}s): {exc}")
            raise
        else:
            self.append_line(f"{label} done ({time.perf_counter() - t0:.1f}s)")

    def start_op(self, name: str) -> None:
        """
        Begin a named operation and reset progress to 0.

        Args:
            name: operation label shown in the status line.
        """
        with self._lock:
            self.current_op = name
            self.progress = 0
            self.is_running = True
            self._version += 1

    def update_progress(self, pct: int, message: str = "", log: bool = True) -> None:
        """
        Set the progress percentage and status label, optionally logging the step.

        - log=False is for high-frequency pings that would flood the scrolling log

        Args:
            pct: progress percentage, clamped to [0, 100].
            message: new status label; empty keeps the current one.
            log: also append the message to the log lines.
        """
        with self._lock:
            self.progress = min(100, max(0, pct))

            # Surface the current stage in the status label, not just the bar
            if message:
                self.current_op = message

            self._version += 1

        if message and log:
            self.append_line(message)

    def rclone_progress(self, label: str) -> Callable[[str], None]:
        """
        Build an rclone on_line callback that forwards its --stats percent to the status label.

        Args:
            label: status label shown while the transfer runs.

        Returns:
            Callback taking one rclone output line.
        """

        def _on_line(line: str) -> None:
            pct = parse_percent(line)

            if pct is not None:
                self.update_progress(pct, label, log=False)

        return _on_line

    def append_line(self, message: str) -> None:
        """
        Append one log line; thread-safe, capped, consecutive duplicates collapsed.

        Args:
            message: the line to append.
        """
        with self._lock:
            lines = list(self.log_lines)

            # Collapse repeated progress pings
            if lines and lines[-1] == message:
                return

            lines.append(message)

            if len(lines) > self._max_lines:
                lines = lines[-self._max_lines :]

            self.log_lines = lines
            self._version += 1

    @contextlib.contextmanager
    def attach_logging(self) -> Iterator[None]:
        """
        Bridge the collab_splats logger into log_lines for the duration of a run.

        - INFO step records (e.g. the creators' frame counters) stream into the dashboard log

        Yields:
            Nothing; the handler is detached and the logger level restored on exit.
        """
        handler = _OpLogHandler(self)
        target = logging.getLogger("collab_splats")
        prev_level = target.level
        target.addHandler(handler)

        # Let INFO records reach the handler without raising the root level
        if prev_level == logging.NOTSET or prev_level > logging.INFO:
            target.setLevel(logging.INFO)

        try:
            yield
        finally:
            target.removeHandler(handler)
            target.setLevel(prev_level)

    def finish_op(self) -> None:
        """
        Mark the current operation complete.
        """
        with self._lock:
            self.progress = 100
            self.is_running = False
            self._version += 1

    def error_op(self, message: str) -> None:
        """
        Mark the current operation as failed with an error message.

        Args:
            message: error text, logged with an ERROR prefix.
        """
        self.append_line(f"ERROR: {message}")

        with self._lock:
            self.is_running = False
            self._version += 1

    def render_html(self) -> str:
        """
        Current state as one HTML string, from a locked snapshot.

        - read by each session's IOLoop poll, not param binding: worker-thread doc pushes glitch
        - any session, including one opened mid-run, shows live flicker-free progress

        Returns:
            Status line, progress bar and the last 40 log lines.
        """
        with self._lock:
            current_op, progress = self.current_op, self.progress
            is_running, lines = self.is_running, list(self.log_lines)

        err = bool(lines and lines[-1].startswith("ERROR"))
        status_color = "#50c050" if is_running else ("#e05050" if err else "#666")
        status_label = html.escape(current_op) if current_op else "Idle"
        bar_color = "#2596be" if is_running else ("#e05050" if err else "#50c050")

        # No max-height on the pre: the log fills the user-resizable console
        log_text = html.escape("\n".join(lines[-40:]))
        return (
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:4px 0'>"
            f"<span style='color:{status_color};font-size:11px;font-weight:700'>{status_label}</span>"
            f"<span style='color:#666;font-size:10px'>{progress}%</span></div>"
            f"<div style='background:#222;border-radius:3px;height:6px;overflow:hidden'>"
            f"<div style='background:{bar_color};width:{progress}%;height:6px'></div></div>"
            f"<pre style='font-size:11px;color:#aaa;background:#0d1117;padding:6px;border-radius:3px;"
            f"margin:6px 0 0 0'>{log_text}</pre>"
        )
