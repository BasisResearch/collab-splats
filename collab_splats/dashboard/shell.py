"""Tabbed shell: splats + localize pages in one Panel session (no-reload switching)."""

from __future__ import annotations

import logging
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
        self._tabs: pn.Tabs | None = None
        self._sidebar_holder: pn.Column | None = None

    def _on_tab(self, event) -> None:
        """Swap sidebar contents to match the active tab; free GPU when leaving localize."""
        page = self._splats if event.new == 0 else self._localize
        self._sidebar_holder[:] = [page.sidebar()]
        if event.old == 1:
            self._localize.release_gpu()

    def view(self) -> pn.template.MaterialTemplate:
        """Assemble tabs + swapping sidebar. dynamic=True defers frontend rendering of
        the inactive tab only — both pages (and their VTK plotters) are built eagerly
        in __init__; lazy plotter construction is a possible follow-up."""
        self._tabs = pn.Tabs(
            ("Splats", self._splats.main()),
            ("Localize", self._localize.main()),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        self._sidebar_holder = pn.Column(self._splats.sidebar(), sizing_mode="stretch_width")
        self._tabs.param.watch(self._on_tab, "active")
        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self._sidebar_holder],
            main=[self._tabs],
            header_background="#2596be",
            sidebar_width=340,
        )
