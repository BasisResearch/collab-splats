"""
Fast-binding dashboard server: page up in seconds, heavy imports stream to the page.

- binds the HTTP server BEFORE the heavy reconstruction stack (torch, creators, pipeline) imports
- until warm, every session gets a loading page polling the shared OperationLog
- the loading page reloads itself into the real dashboard once the stack is ready
- must stay light: no torch / pyvista / dashboard.app imports at module level
"""

from __future__ import annotations

import atexit
import importlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import panel as pn

from collab_splats.dashboard.operation_log import OperationLog

# Type-only: GpuWorker imports torch, which this module must not load at import
if TYPE_CHECKING:
    from collab_splats.dashboard.gpu_worker import GpuWorker

logger = logging.getLogger(__name__)

# Heavy modules the warm thread imports, in order, with user-facing labels; torch first
_WARM_MODULES = (
    ("torch", "torch runtime"),
    ("collab_splats.dashboard.app", "dashboard core (viewer + models)"),
)


########
# Warm-up
########


class ServerState:
    """
    Warm/ready state shared between the warm thread and per-session factories.

    - gpu_worker is built post-warm: GpuWorker's import pulls torch
    """

    def __init__(self) -> None:
        """
        Start cold, with no GPU worker.
        """
        self.ready = False
        self.gpu_worker: GpuWorker | None = None


def _ensure_display() -> None:
    """
    Start a headless Xvfb display if none is set, so VTK gets an OpenGL context.

    - pn.pane.VTK builds a vtkXOpenGLRenderWindow per document; with no DISPLAY it blocks in C
    """
    if os.environ.get("DISPLAY"):
        return

    if not shutil.which("Xvfb"):
        logger.warning(
            "no DISPLAY and Xvfb not installed; VTK rendering will fail on a headless host"
        )
        return

    # Software GL via Mesa; containers rarely expose GLX on the GPU
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    display = ":99"
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    os.environ["DISPLAY"] = display
    time.sleep(1.0)  # let Xvfb come up before VTK probes the display
    logger.info("started Xvfb on %s for headless VTK rendering", display)


def _finalize(state: ServerState) -> None:
    """
    Post-import setup on the warm thread: headless display and the shared GPU worker.
    """
    # GpuWorker's module imports torch, so it loads only once the stack is warm
    from collab_splats.dashboard.gpu_worker import GpuWorker

    _ensure_display()  # Xvfb must exist before the first real session builds VTK panes
    state.gpu_worker = GpuWorker()


def warm(
    state: ServerState,
    op_log: OperationLog,
    modules: tuple[tuple[str, str], ...] = _WARM_MODULES,
    finalize: Callable[[ServerState], None] = _finalize,
) -> None:
    """
    Import the heavy stack, streaming per-module progress into the op log.

    - any failure stops startup: the op log shows the error and state.ready stays False

    Args:
        state: shared server state; ready flips True on success.
        op_log: shared log the loading page polls.
        modules: (module name, user-facing label) pairs, imported in order.
        finalize: post-import setup step, given the state.
    """
    op_log.start_op("starting dashboard")
    total = len(modules) + 1  # +1 for the display/worker finalize step

    try:
        for i, (name, label) in enumerate(modules):
            op_log.update_progress(int(100 * i / total), f"importing {label}")

            with op_log.step(f"import {label}"):
                importlib.import_module(name)

        op_log.update_progress(
            int(100 * len(modules) / total), "preparing display + GPU worker"
        )
        finalize(state)
    except Exception as exc:
        logger.exception("dashboard warm-up failed")
        failure = f"dashboard startup failed: {exc}"
        op_log.error_op(failure)
        return

    state.ready = True
    op_log.finish_op()


########
# Pages
########


def _loading_page(
    state: ServerState, op_log: OperationLog
) -> pn.template.MaterialTemplate:
    """
    Session shown while the stack warms: live import progress, then self-reload.
    """
    progress = pn.pane.HTML(op_log.render_html(), sizing_mode="stretch_width")

    def _tick() -> None:
        progress.object = op_log.render_html()

        # Stack is warm: reload this session; the factory now serves the real page
        if state.ready:
            pn.state.location.reload = True

    try:
        pn.state.add_periodic_callback(_tick, period=300, start=True)
    except RuntimeError:
        logger.debug(
            "no periodic callback (no server doc); progress is static", exc_info=True
        )

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
    return pn.template.MaterialTemplate(
        title="splats", main=[body], header_background="#2596be"
    )


def make_factory(
    base_dir: str | Path, state: ServerState, op_log: OperationLog
) -> Callable[[], pn.template.MaterialTemplate]:
    """
    Per-session page factory: the loading page until warm, then the real dashboard.

    Args:
        base_dir: local outputs root, one directory per scene.
        state: shared server state the factory checks for readiness.
        op_log: shared log handed to every page.

    Returns:
        Zero-argument callable building one session's page.
    """

    def factory() -> pn.template.MaterialTemplate:
        if not state.ready:
            return _loading_page(state, op_log)

        # Import stays local: app pulls the heavy stack, guaranteed warm here
        from collab_splats.dashboard.app import SplatsApp

        return SplatsApp(
            base_dir=Path(base_dir),
            gpu_worker=cast("GpuWorker", state.gpu_worker),
            op_log=op_log,
        ).view()

    return factory


########
# Server
########


def run_app(
    base_dir: str,
    host: str = "0.0.0.0",
    port: int = 7860,
    websocket_origin: str | list[str] | None = None,
) -> None:
    """
    Serve the dashboard, binding BEFORE the heavy stack imports.

    Args:
        base_dir: local outputs root, one directory per scene.
        host: bind address.
        port: bind port.
        websocket_origin: allowed Origin host:port list, or "*"; None allows host:port and localhost:port.
    """
    # The run job's preproc stage draws QA figures on the worker thread; force the thread-safe Agg backend
    import matplotlib

    matplotlib.use("Agg")

    # inline=True serves all JS/CSS locally; the vtk extension is JS-side only
    pn.extension("vtk", inline=True)

    op_log = OperationLog()
    state = ServerState()
    threading.Thread(
        target=warm, args=(state, op_log), name="warm", daemon=True
    ).start()

    if websocket_origin is None:
        origin: str | list[str] = [f"{host}:{port}", f"localhost:{port}"]
    else:
        origin = websocket_origin

    logger.info(
        "dashboard listening on http://%s:%s — open it now; import progress shows on the page",
        host,
        port,
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
