from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Callable

import panel as pn
import param

from collab_splats.dashboard.sources import parse_rclone_percent


class _OpLogHandler(logging.Handler):
    """Forward log records emitted during a run into an OperationLog's log_lines."""

    def __init__(self, op_log: "OperationLog") -> None:
        super().__init__(level=logging.INFO)
        self._op_log = op_log

    def emit(self, record: logging.LogRecord) -> None:
        # Append the formatted message; reuse op_log's thread-safe line buffer.
        try:
            self._op_log.append_line(record.getMessage())
        except Exception:  # never let logging crash the run
            pass


class OperationLog(param.Parameterized):
    """Shared progress and log bus for all dashboard background operations.

    Background threads call start_op / update_progress / finish_op.
    The Panel widget (panel()) reactively reflects current state.
    Thread-safe: uses a lock for log_lines mutations.
    """

    current_op = param.String(default="")
    progress = param.Integer(default=0, bounds=(0, 100))
    is_running = param.Boolean(default=False)
    log_lines = param.List(default=[])

    _MAX_LINES = 100

    def __init__(self, **params: Any):
        super().__init__(**params)
        self._lock = threading.Lock()

    def start_op(self, name: str) -> None:
        """Begin a named operation; resets progress to 0."""
        with self._lock:
            self.current_op = name
            self.progress = 0
            self.is_running = True

    def update_progress(self, pct: int, message: str = "", log: bool = True) -> None:
        """Update progress percentage; set the status label and (optionally) log the step.

        log=False updates the live status label only — use for high-frequency pings (e.g. the
        per-frame sampling counter) that would otherwise flood the scrolling log.
        """
        with self._lock:
            self.progress = min(100, max(0, pct))
            # Surface the current stage in the status label, not just the bar.
            if message:
                self.current_op = message
        if message and log:
            self.append_line(message)

    def rclone_progress(self, label: str) -> Callable[[str], None]:
        """Return an on_line callback that forwards rclone --stats percent to the status label."""

        def _on_line(line: str) -> None:
            pct = parse_rclone_percent(line)
            if pct is not None:
                self.update_progress(pct, label, log=False)

        return _on_line

    def append_line(self, message: str) -> None:
        """Append a single log line (thread-safe, capped, consecutive dupes collapsed)."""
        with self._lock:
            lines = list(self.log_lines)
            # Collapse repeated progress pings (e.g. 'sampling frames' per frame).
            if lines and lines[-1] == message:
                return
            lines.append(message)
            if len(lines) > self._MAX_LINES:
                lines = lines[-self._MAX_LINES :]
            self.log_lines = lines

    @contextlib.contextmanager
    def attach_logging(self, *logger_names: str, level: int = logging.INFO):
        """Bridge module loggers into log_lines for the duration of a run.

        Attaches a handler to each named logger (default: the top-level
        'collab_splats' logger) so step-level INFO records — e.g. the creators'
        '%d/%d frames' lines — stream into the dashboard log while a run is active.
        """
        names = logger_names or ("collab_splats",)
        handler = _OpLogHandler(self)
        targets = [logging.getLogger(n) for n in names]
        prev_levels = []
        for lg in targets:
            lg.addHandler(handler)
            # Ensure INFO records propagate to our handler without globally raising root.
            prev_levels.append(lg.level)
            if lg.level == logging.NOTSET or lg.level > level:
                lg.setLevel(level)
        try:
            yield
        finally:
            for lg, prev in zip(targets, prev_levels):
                lg.removeHandler(handler)
                lg.setLevel(prev)

    def finish_op(self) -> None:
        """Mark the current operation complete."""
        with self._lock:
            self.progress = 100
            self.is_running = False

    def error_op(self, message: str) -> None:
        """Mark the current operation as failed with an error message."""
        with self._lock:
            lines = list(self.log_lines)
            lines.append(f"ERROR: {message}")
            if len(lines) > self._MAX_LINES:
                lines = lines[-self._MAX_LINES :]
            self.log_lines = lines
            self.is_running = False

    def render_html(self) -> str:
        """Render the current state as a single HTML string (thread-safe snapshot).

        Used by a per-session periodic poll on the IOLoop, instead of reactive param binding:
        op_log state is mutated from the GpuWorker thread, and pushing Bokeh doc updates from a
        non-IOLoop thread glitches. Polling reads a locked snapshot here and updates the pane on
        the session's own IOLoop, so any session (incl. one opened after a run started) shows live,
        flicker-free progress.
        """
        with self._lock:
            current_op, progress = self.current_op, self.progress
            is_running, lines = self.is_running, list(self.log_lines)
        err = bool(lines and lines[-1].startswith("ERROR"))
        status_color = "#50c050" if is_running else ("#e05050" if err else "#666")
        status_label = current_op if current_op else "Idle"
        bar_color = "#2596be" if is_running else ("#e05050" if err else "#50c050")
        log_text = "\n".join(lines[-20:]) if lines else ""
        return (
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:4px 0'>"
            f"<span style='color:{status_color};font-size:11px;font-weight:700'>{status_label}</span>"
            f"<span style='color:#666;font-size:10px'>{progress}%</span></div>"
            f"<div style='background:#222;border-radius:3px;height:6px;overflow:hidden'>"
            f"<div style='background:{bar_color};width:{progress}%;height:6px'></div></div>"
            f"<pre style='font-size:11px;color:#aaa;background:#0d1117;padding:6px;border-radius:3px;"
            f"margin:6px 0 0 0;overflow-y:auto;max-height:80px'>{log_text}</pre>"
        )

    @param.depends("current_op", "progress", "is_running", "log_lines")
    def _render(self) -> pn.Column:
        status_color = (
            "#50c050"
            if self.is_running
            else ("#e05050" if (self.log_lines and self.log_lines[-1].startswith("ERROR")) else "#666")
        )
        status_label = self.current_op if self.current_op else "Idle"

        progress_bar = pn.widgets.Progress(
            value=self.progress,
            max=100,
            bar_color="info" if self.is_running else "success",
            sizing_mode="stretch_width",
            height=6,
        )

        log_text = "\n".join(self.log_lines[-20:]) if self.log_lines else ""
        log_area = pn.pane.HTML(
            f"<pre style='font-size:11px;color:#aaa;background:#0d1117;padding:6px;"
            f"border-radius:3px;margin:0;overflow-y:auto;max-height:80px'>{log_text}</pre>",
            sizing_mode="stretch_width",
        )

        header = pn.pane.HTML(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:4px 0'>"
            f"<span style='color:{status_color};font-size:11px;font-weight:700'>{status_label}</span>"
            f"<span style='color:#666;font-size:10px'>{self.progress}%</span>"
            f"</div>",
            sizing_mode="stretch_width",
        )

        return pn.Column(header, progress_bar, log_area, sizing_mode="stretch_width")

    def panel(self) -> pn.Card:
        """Return a collapsible Panel widget for the global progress strip."""
        return pn.Card(
            pn.panel(self._render),
            title="Operations",
            collapsed=True,
            sizing_mode="stretch_width",
            header_background="#1a1a2e",
        )
