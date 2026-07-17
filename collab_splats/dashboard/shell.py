"""Tabbed shell: splats + localize pages in one Panel session (no-reload switching)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import panel as pn

from collab_splats.dashboard.app import SplatsApp
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.localize import LocalizePage, SceneCache
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource

logger = logging.getLogger(__name__)


class DashboardShell:
    """One MaterialTemplate hosting both pages under dynamic tabs; sidebar follows the tab."""

    def __init__(
        self,
        base_dir: Path,
        source: SessionSource | None = None,
        gpu_worker: GpuWorker | None = None,
        op_log: OperationLog | None = None,
    ) -> None:
        # Shared session-wide collaborators: one source/worker/log/cache across both pages.
        source = source if source is not None else SessionSource()
        gpu_worker = gpu_worker if gpu_worker is not None else GpuWorker()
        op_log = op_log if op_log is not None else OperationLog()
        self._cache = SceneCache()
        self._splats = SplatsApp(
            base_dir=Path(base_dir), source=source, gpu_worker=gpu_worker, op_log=op_log, cache=self._cache
        )
        self._localize = LocalizePage(
            base_dir=Path(base_dir), source=source, gpu_worker=gpu_worker, op_log=op_log, cache=self._cache
        )
        self._gpu = gpu_worker
        self._op_log = op_log
        self._tabs: pn.Tabs | None = None
        self._sidebar_holder: pn.Column | None = None

    def _on_tab(self, event) -> None:
        """Swap sidebar to the active tab; build the localize view lazily; free GPU on leave."""
        if event.new == 1 and not self._localize_built:
            self._localize_built = True
            # Paint a spinner NOW; defer the heavy main() (pyvista/VTK) one tick so the
            # browser renders feedback before the build blocks the loop.
            self._localize_holder[:] = [
                pn.Column(
                    pn.indicators.LoadingSpinner(value=True, size=40),
                    pn.pane.HTML("<i>Building Localize page…</i>"),
                )
            ]

            def build() -> None:
                t0 = time.perf_counter()
                try:
                    self._localize_holder[:] = [self._localize.main()]
                except Exception as exc:  # e.g. VTK/offscreen GL failure
                    logger.warning("localize page build failed", exc_info=True)
                    self._localize_holder[:] = [
                        pn.pane.HTML(f"<b style='color:#e05050'>Localize page failed to build: {exc}</b>")
                    ]
                    self._op_log.error_op(f"localize page build failed: {exc}")
                    return
                self._op_log.append_line(f"localize page built ({time.perf_counter() - t0:.1f}s)")

            doc = pn.state.curdoc
            doc.add_next_tick_callback(build) if doc is not None else build()
        page = self._splats if event.new == 0 else self._localize
        self._sidebar_holder[:] = [page.sidebar()]
        if event.old == 1:
            # pytorch_gc CUDA-syncs — run it on the worker, not the tab-switch watcher.
            self._gpu.submit(self._localize.release_gpu, lambda _res: None, pn.state.curdoc)

    def view(self) -> pn.template.MaterialTemplate:
        """Assemble tabs + swapping sidebar. The Localize tab holds an empty placeholder
        until first activation — its main() (and the pyvista/VTK plotter behind it) is
        built lazily in _on_tab, so an untouched Localize tab costs nothing."""
        self._localize_built = False
        self._localize_holder = pn.Column(sizing_mode="stretch_both")
        self._tabs = pn.Tabs(
            ("Splats", self._splats.main()),
            ("Localize", self._localize_holder),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        self._sidebar_holder = pn.Column(self._splats.sidebar(), sizing_mode="stretch_width")
        self._tabs.param.watch(self._on_tab, "active")
        # One shared operations console pinned at the BOTTOM of the page: tabs and console
        # share a single flex column (tabs stretch, console keeps its height), so the log
        # persists across tab switches and during the localize build. The console's height
        # is user-adjustable via the browser-native resize handle (drag its bottom edge).
        self._progress = pn.pane.HTML(self._op_log.render_html(), sizing_mode="stretch_both")
        self._console = pn.Column(
            self._progress,
            sizing_mode="stretch_width",
            height=190,
            styles={
                "resize": "vertical",
                "overflow": "auto",
                "min-height": "70px",
                "border-top": "2px solid #2596be",
                "background": "#0d1117",
                "padding": "4px 8px",
            },
        )
        self._seen_log_version = -1
        try:
            pn.state.add_periodic_callback(self._on_progress_tick, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)
        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self._sidebar_holder],
            main=[pn.Column(self._tabs, self._console, sizing_mode="stretch_both")],
            header_background="#2596be",
            sidebar_width=340,
        )

    def _on_progress_tick(self) -> None:
        """Refresh the shared console when the op log changed (300 ms poll, version-gated)."""
        if self._op_log.version != self._seen_log_version:
            self._seen_log_version = self._op_log.version
            self._progress.object = self._op_log.render_html()
