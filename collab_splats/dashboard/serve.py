# collab_splats/dashboard/serve.py
"""Fast-binding dashboard server: page up in seconds, heavy imports stream to the page.

The CLI binds the HTTP server BEFORE the heavy reconstruction stack (torch models,
creators, pipeline) is imported. Until the stack is warm, every session gets a light
loading page whose progress strip polls the shared OperationLog — the same status UI
the dashboard uses for run progress — and the page reloads itself into the real
dashboard when the stack is ready.

This module must stay light: no torch / pyvista / dashboard.app imports at module
level (the package __init__ is lazy for the same reason).
"""

from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path

import panel as pn

from collab_splats.dashboard.operation_log import OperationLog

logger = logging.getLogger(__name__)

# Heavy modules the warm thread imports, in order, with user-facing labels. torch is
# listed first so the biggest single import gets its own progress line.
_WARM_MODULES = (
    ("torch", "torch runtime"),
    ("collab_splats.dashboard.app", "dashboard core (viewer + models)"),
    ("collab_splats.dashboard.pipeline", "reconstruction pipeline"),
    ("collab_splats.semantics.features.base", "semantic extractors"),
    ("collab_splats.localization.localizer", "localization stack"),
)


class ServerState:
    """Warm/ready state shared between the warm thread and per-session factories."""

    def __init__(self) -> None:
        self.ready = False
        self.failed = False
        self.gpu_worker = None  # constructed post-warm: GpuWorker's import pulls torch


def _finalize(state: ServerState) -> None:
    """Post-import setup: headless display + the shared GPU worker (warm thread)."""
    from collab_splats.dashboard.app import _ensure_display
    from collab_splats.dashboard.gpu_worker import GpuWorker

    _ensure_display()  # Xvfb must exist before the first real session builds VTK panes
    state.gpu_worker = GpuWorker()


def warm(state: ServerState, op_log: OperationLog, modules=_WARM_MODULES, finalize=_finalize) -> None:
    """Import the heavy stack, streaming per-module progress into the op log."""
    op_log.start_op("starting dashboard")
    total = len(modules) + 1  # +1 for the display/worker finalize step
    core_ok = True
    for i, (name, label) in enumerate(modules):
        op_log.update_progress(int(100 * i / total), f"importing {label}")
        try:
            with op_log.step(f"import {label}"):
                importlib.import_module(name)
        except Exception:
            logger.exception("warm import failed: %s", name)
            # Only dashboard core is load-bearing; optional stacks may be absent.
            if name.endswith("dashboard.app"):
                core_ok = False
    op_log.update_progress(int(100 * len(modules) / total), "preparing display + GPU worker")
    if not core_ok:
        op_log.error_op("dashboard startup failed: core import error (see server log)")
        state.failed = True
        return
    try:
        finalize(state)
    except Exception as exc:
        logger.exception("dashboard warm finalize failed")
        op_log.error_op(f"dashboard startup failed: {exc}")
        state.failed = True
        return
    state.ready = True
    op_log.finish_op()


def _loading_page(state: ServerState, op_log: OperationLog) -> pn.template.MaterialTemplate:
    """Session shown while the stack warms: live import progress, then self-reload."""
    progress = pn.pane.HTML(op_log.render_html(), sizing_mode="stretch_width")

    def _tick() -> None:
        progress.object = op_log.render_html()
        if state.ready:
            # Stack is warm: reload this session; the factory now serves the real shell.
            pn.state.location.reload = True

    try:
        pn.state.add_periodic_callback(_tick, period=300, start=True)
    except Exception:
        logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)

    body = pn.Column(
        pn.indicators.LoadingSpinner(value=True, size=48),
        pn.pane.HTML(
            "<h3>Starting dashboard…</h3><p>The reconstruction stack is importing — "
            "this page reloads automatically when it is ready.</p>"
        ),
        progress,
        sizing_mode="stretch_width",
        max_width=700,
    )
    return pn.template.MaterialTemplate(title="splats", main=[body], header_background="#2596be")


def make_factory(base_dir, state: ServerState, op_log: OperationLog):
    """Return the per-session page factory: loading page until warm, then the real shell."""

    def factory() -> pn.template.MaterialTemplate:
        if not state.ready:
            return _loading_page(state, op_log)
        # Import stays local: shell pulls the heavy app stack, guaranteed warm here.
        from collab_splats.dashboard.shell import DashboardShell

        return DashboardShell(base_dir=Path(base_dir), gpu_worker=state.gpu_worker, op_log=op_log).view()

    return factory


def run_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    base_dir: str = "/workspace/outputs",
    websocket_origin: "str | list[str] | None" = None,
) -> None:
    """Serve the dashboard, binding BEFORE the heavy stack imports.

    websocket_origin=None restricts connections to host:port + localhost:port. Pass an
    explicit list (or "*") to allow remote-IP / SSH-tunnel access.
    """
    # inline=True serves all JS/CSS from this server (headless hosts can't reach CDNs);
    # registering the vtk extension is JS-side only and does not import python vtk.
    pn.extension("vtk", inline=True)

    op_log = OperationLog()
    state = ServerState()
    threading.Thread(target=warm, args=(state, op_log), name="warm", daemon=True).start()

    if websocket_origin is None:
        origin: "str | list[str]" = [f"{host}:{port}", f"localhost:{port}"]
    else:
        origin = websocket_origin

    print(
        f"dashboard listening on http://{host}:{port} — open it now; import progress shows on the page",
        flush=True,
    )
    pn.serve(
        make_factory(base_dir, state, op_log),
        address=host,
        port=port,
        show=False,
        title="splats",
        websocket_origin=origin,
        session_token_expiration=1800,
    )
