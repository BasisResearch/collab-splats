"""ReconstructPane — Tab 3 of the unified dashboard.

Runs feedforward reconstruction (VGGTXCreator / MapAnythingCreator / VGGTOmegaCreator)
in a background thread against frames already written by PreprocessPane.
"""
from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Any

import panel as pn
import param

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState

logger = logging.getLogger(__name__)

########################################################################
# ReconstructPane
########################################################################


class ReconstructPane(param.Parameterized):
    """Feedforward reconstruction pane: pick backend, run pipeline, stream logs."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._recon_thread: threading.Thread | None = None
        self._log_lines: list[str] = []
        self._log_lock = threading.Lock()

        # Config widgets
        self._backend_dd = pn.widgets.Select(
            name="Backend",
            options=["vggtx", "mapanything", "vggt_omega"],
            value="vggtx",
            width=260,
        )
        self._conf_slider = pn.widgets.FloatSlider(
            name="Conf threshold", start=0.0, end=100.0, step=1.0, value=35.0, width=260
        )

        # BA stub (disabled — not yet implemented)
        self._ba_toggle = pn.widgets.Toggle(
            name="Bundle Adjustment", disabled=True, width=260
        )
        self._ba_stub_html = pn.pane.HTML(
            "<span style='color:#e0a020;font-size:12px'>⚠ Not yet implemented</span>",
            width=260,
        )

        # LC stub (disabled — not yet implemented)
        self._lc_toggle = pn.widgets.Toggle(
            name="Loop Closure", disabled=True, width=260
        )
        self._lc_stub_html = pn.pane.HTML(
            "<span style='color:#e0a020;font-size:12px'>⚠ Not yet implemented</span>",
            width=260,
        )

        # Run button + status
        self._run_btn = pn.widgets.Button(
            name="Run Reconstruction", button_type="primary", disabled=True, width=260
        )
        self._status_html = pn.pane.HTML("", width=260)

        # Log area (readonly via disabled=True)
        self._log_area = pn.widgets.TextAreaInput(
            value="",
            disabled=True,
            height=500,
            width=500,
            placeholder="Reconstruction log will appear here…",
        )

        # Periodic callback guard — registered once in panel()
        self._cb_registered = False

        # Wire callbacks
        self._run_btn.on_click(self._on_run)
        self._state.param.watch(self._on_output_dir_changed, ["output_dir"])

        # Initialize gate state
        self._on_output_dir_changed(None)

    def _on_output_dir_changed(self, event: Any) -> None:
        """Enable Run button when output_dir is set."""
        self._run_btn.disabled = self._state.output_dir is None

    def _on_run(self, event: Any) -> None:
        """Spawn background reconstruction thread on button click."""
        if self._recon_thread and self._recon_thread.is_alive():
            return
        # Snapshot widget values on main thread before passing to worker
        backend = self._backend_dd.value
        conf = self._conf_slider.value
        self._run_btn.disabled = True
        self._status_html.object = "<span style='color:#2596be'>⏳ Running…</span>"
        self._log_area.value = ""
        self._recon_thread = threading.Thread(
            target=self._run_reconstruction, args=(backend, conf), daemon=True
        )
        self._recon_thread.start()

    def _run_reconstruction(self, backend: str, conf: float) -> None:
        """Background thread: validate, run creator, save zarr, set state."""
        try:
            output_dir = Path(self._state.output_dir)
            images_dir = output_dir / "frames"

            # Validate frames directory written by PreprocessPane
            if not images_dir.exists():
                msg = f"frames dir not found: {images_dir}. Run Preprocess first."
                self._op_log.error_op(msg)
                self._append_log(f"ERROR: {msg}")
                self._finish(success=False)
                return

            self._op_log.start_op("Running reconstruction")
            self._append_log(f"Backend: {backend}  conf_threshold: {conf}")

            creator = self._build_creator(backend, conf)
            backend_dir = output_dir / backend

            self._append_log(f"Starting {backend} inference…")
            creator.reconstruct(images_dir, backend_dir)

            ff = creator.outputs
            if ff is None:
                raise RuntimeError("Creator produced no outputs after reconstruct()")

            zarr_path = backend_dir / "feedforward.zarr"
            self._append_log(f"Saving {zarr_path}")
            ff.save_zarr(zarr_path)

            self._state.feedforward_result = ff
            self._op_log.finish_op()
            self._append_log(f"Done. {len(ff.points):,} points.")
            self._finish(success=True)

        except Exception as exc:
            tb = traceback.format_exc()
            self._op_log.error_op(str(exc))
            self._append_log(f"ERROR: {exc}\n{tb}")
            self._finish(success=False)

    def _build_creator(self, backend: str, conf: float) -> Any:
        """Instantiate the appropriate feedforward creator for the selected backend.

        All imports are deferred to avoid pulling in heavy optional deps (xfeat,
        VGGT-Omega submodule) at module load time.
        """
        if backend == "vggtx":
            from collab_splats.pointcloud.feedforward import VGGTXCreator
            return VGGTXCreator(conf_threshold=conf)
        if backend == "mapanything":
            from collab_splats.pointcloud.feedforward import MapAnythingCreator
            # confidence_percentile expects 0–100 (percentile), matching the slider range
            return MapAnythingCreator(confidence_percentile=conf)
        if backend == "vggt_omega":
            # Optional submodule dep — import lazily so missing install doesn't break the module
            try:
                from collab_splats.pointcloud.feedforward import VGGTOmegaCreator
            except ImportError as exc:
                raise ImportError(
                    "vggt_omega requires the VGGT-Omega submodule. "
                    "Run setup_feedforward.sh with the submodule initialized."
                ) from exc
            return VGGTOmegaCreator(conf_threshold=conf)
        raise ValueError(f"Unknown backend: {backend!r}")

    def _append_log(self, line: str) -> None:
        """Thread-safe: queue a log line for drain into _log_area."""
        with self._log_lock:
            self._log_lines.append(line)

    def _drain_log(self) -> None:
        """Periodic callback (500ms): flush queued log lines into _log_area."""
        with self._log_lock:
            lines = self._log_lines[:]
            self._log_lines.clear()
        if lines:
            self._log_area.value = (self._log_area.value or "") + "\n".join(lines) + "\n"

    def _finish(self, *, success: bool) -> None:
        """Update status HTML and re-enable Run button."""
        if success:
            self._status_html.object = "<span style='color:#50e050'>✓ Complete</span>"
        else:
            self._status_html.object = "<span style='color:#e05050'>✗ Failed</span>"
        self._run_btn.disabled = False

    def panel(self) -> pn.viewable.Viewable:
        """Return the full ReconstructPane Panel layout."""
        if not self._cb_registered:
            try:
                pn.state.add_periodic_callback(self._drain_log, period=500)
                self._cb_registered = True
            except Exception:
                pass  # outside live server context (e.g. tests)

        config_col = pn.Column(
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 6px 0'>Reconstruction Config</h4>"),
            self._backend_dd,
            self._conf_slider,
            pn.layout.Divider(),
            self._ba_toggle,
            self._ba_stub_html,
            pn.layout.Divider(),
            self._lc_toggle,
            self._lc_stub_html,
            pn.layout.Divider(),
            self._run_btn,
            self._status_html,
            width=380,
        )

        log_col = pn.Column(
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 6px 0'>Log</h4>"),
            self._log_area,
        )

        return pn.Row(config_col, log_col, sizing_mode="stretch_width")
