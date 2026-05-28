from __future__ import annotations

import threading
from typing import Any

import panel as pn
import param


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

    def update_progress(self, pct: int, message: str = "") -> None:
        """Update progress percentage and optionally append a log line."""
        with self._lock:
            self.progress = min(100, max(0, pct))
            if message:
                lines = list(self.log_lines)
                lines.append(message)
                if len(lines) > self._MAX_LINES:
                    lines = lines[-self._MAX_LINES:]
                self.log_lines = lines

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
                lines = lines[-self._MAX_LINES:]
            self.log_lines = lines
            self.is_running = False

    @param.depends("current_op", "progress", "is_running", "log_lines")
    def _render(self) -> pn.Column:
        status_color = "#50c050" if self.is_running else (
            "#e05050" if (self.log_lines and self.log_lines[-1].startswith("ERROR")) else "#666"
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
