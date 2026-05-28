"""SemanticsPane — Tab 2: 2D feature extraction and comparison."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import panel as pn
import param
import torch

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.semantics.features.base import BaseFeatureExtractor, BaseQueryableExtractor

logger = logging.getLogger(__name__)

########################################################################
# Pure helpers — testable without Panel
########################################################################


def _score_to_rgb(score: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float score map to a viridis RGB image (H, W, 3) uint8."""
    import matplotlib.cm as cm

    score = score.astype(np.float32)
    lo, hi = score.min(), score.max()
    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score = np.zeros_like(score)
    rgba = cm.viridis(score)  # (H, W, 4) float64 in [0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def _load_frame_rgb(frames_zarr_path: Path, idx: int) -> np.ndarray:
    """Load frame idx from frames.zarr and return (H, W, 3) uint8 array."""
    import zarr

    z = zarr.open(str(frames_zarr_path), mode="r")
    return z["frames"][idx]


########################################################################
# ExtractorColumn — per-column state and widgets
########################################################################

_IDLE = "idle"
_RUNNING = "running"
_DONE = "done"
_ERROR = "error"


class ExtractorColumn(param.Parameterized):
    """One extractor column: dropdown, run button, status badge, image display."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._extractor: BaseFeatureExtractor | None = None
        self._feature_zarr_path: Path | None = None
        self._thread: threading.Thread | None = None
        self._current_frame_idx: int = 0

        # Build dropdown from registered extractors
        registry_keys = list(BaseFeatureExtractor._registry.keys())
        self._method_dd = pn.widgets.Select(
            name="Extractor",
            options=registry_keys,
            value=registry_keys[0] if registry_keys else None,
            width=200,
        )
        self._run_btn = pn.widgets.Button(
            name="▶ Run", button_type="primary", width=100, disabled=True
        )
        self._status_html = pn.pane.HTML(
            "<span style='color:#888;font-size:11px'>idle</span>", width=200
        )
        self._image_pane = pn.pane.PNG(None, width=320, height=240)

        # Wire callbacks
        self._method_dd.param.watch(self._on_method_change, "value")
        self._run_btn.on_click(self._on_run)
        state.param.watch(self._on_frames_ready, "frames_zarr_path")

        # Enable immediately if frames already available (e.g. loaded session)
        if state.frames_zarr_path is not None:
            self._run_btn.disabled = False

    def _on_frames_ready(self, event: Any) -> None:
        """Enable Run button when frames zarr becomes available."""
        if event.new is not None:
            self._run_btn.disabled = False

    def _on_method_change(self, event: Any) -> None:
        """Unload current extractor from memory and reset column when method changes."""
        if self._extractor is not None:
            del self._extractor
            self._extractor = None
            torch.cuda.empty_cache()
        self._feature_zarr_path = None
        self._image_pane.object = None
        self._set_status(_IDLE)

    def _on_run(self, event: Any) -> None:
        """Launch background extraction thread."""
        if self._thread and self._thread.is_alive():
            return
        if self._state.frames_zarr_path is None:
            return
        # Unload any previous extractor before loading a new one
        if self._extractor is not None:
            del self._extractor
            self._extractor = None
            torch.cuda.empty_cache()
        self._feature_zarr_path = None
        self._image_pane.object = None
        self._set_status(_RUNNING)
        self._run_btn.disabled = True
        method = self._method_dd.value
        self._thread = threading.Thread(
            target=self._run_extraction, args=(method,), daemon=True
        )
        self._thread.start()

    def _run_extraction(self, method: str) -> None:
        """Background: instantiate extractor, extract all frames, write zarr, update display."""
        try:
            self._op_log.start_op(f"Extracting {method} features")
            extractor_cls = BaseFeatureExtractor.get(method)
            self._extractor = extractor_cls()
            output_dir = Path(self._state.output_dir)
            cache_dir = output_dir / "features" / method
            zarr_path = self._extractor.extract_and_cache_from_zarr(
                frames_zarr_path=Path(self._state.frames_zarr_path),
                cache_dir=cache_dir,
            )
            self._feature_zarr_path = zarr_path
            self._state.feature_maps_path = zarr_path
            self._set_status(_DONE)
            self._op_log.finish_op()
            self._refresh_display()
        except Exception as exc:
            logger.exception("Extraction failed for %s", method)
            self._set_status(f"{_ERROR}: {exc}")
            self._op_log.error_op(str(exc))
        finally:
            self._run_btn.disabled = False

    def refresh(self, frame_idx: int) -> None:
        """Called by SemanticsPane when frame slider changes."""
        self._current_frame_idx = frame_idx
        if self._feature_zarr_path is not None and self._extractor is not None:
            self._refresh_display()

    def _refresh_display(self) -> None:
        """Load features for current frame and render PCA RGB overlay."""
        from io import BytesIO

        import zarr
        from PIL import Image as PILImage

        if self._feature_zarr_path is None:
            return
        idx = self._current_frame_idx
        z = zarr.open(str(self._feature_zarr_path), mode="r")
        feat = torch.from_numpy(np.array(z["features"][idx]))  # (D, H_p, W_p)
        rgb = BaseFeatureExtractor.features_to_rgb(feat)
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        self._image_pane.object = buf.getvalue()

    def render_query(self, frame_idx: int, query_text: str) -> None:
        """Render similarity heatmap for query_text on frame_idx."""
        from io import BytesIO

        import zarr
        from PIL import Image as PILImage

        if self._feature_zarr_path is None or not isinstance(self._extractor, BaseQueryableExtractor):
            return
        z = zarr.open(str(self._feature_zarr_path), mode="r")
        feat = torch.from_numpy(np.array(z["features"][frame_idx]).astype(np.float32))
        score = self._extractor.score_queries(feat, positive=[query_text])  # (H_p, W_p)
        rgb = _score_to_rgb(score.cpu().numpy())
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        self._image_pane.object = buf.getvalue()

    def is_queryable_and_ready(self) -> bool:
        """True when column has a completed queryable extractor."""
        return (
            self._feature_zarr_path is not None
            and isinstance(self._extractor, BaseQueryableExtractor)
        )

    def panel(self) -> pn.Column:
        """Return Panel layout for this column."""
        return pn.Column(
            pn.Row(self._method_dd, self._run_btn),
            self._status_html,
            self._image_pane,
        )

    def _set_status(self, status: str) -> None:
        """Update the status badge color and text."""
        self._status = status
        color_map = {_IDLE: "#888", _RUNNING: "#f0c040", _DONE: "#50c050", _ERROR: "#e05050"}
        color = color_map.get(status.split(":")[0], "#888")
        self._status_html.object = (
            f"<span style='color:{color};font-size:11px'>{status}</span>"
        )


########################################################################
# SemanticsPane
########################################################################


class SemanticsPane(param.Parameterized):
    """Tab 2 — 2D feature extraction and comparison across up to 3 extractors."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._current_frame_idx = 0
        self._query_thread: threading.Thread | None = None

        # Frame selector widgets
        self._frame_slider = pn.widgets.IntSlider(
            name="Frame", value=0, start=0, end=0, step=1, width=400, disabled=True
        )
        self._prev_btn = pn.widgets.Button(name="◀", width=50, disabled=True)
        self._next_btn = pn.widgets.Button(name="▶", width=50, disabled=True)
        self._frame_count_html = pn.pane.HTML("", width=200)

        # Original frame display (column 0)
        self._original_pane = pn.pane.PNG(None, width=320, height=240)

        # Three independent extractor columns
        self._columns: list[ExtractorColumn] = [
            ExtractorColumn(state=state, op_log=op_log) for _ in range(3)
        ]
        self._col_panels = [c.panel() for c in self._columns]

        # Text query bar — hidden until a queryable column is ready
        self._query_input = pn.widgets.TextInput(
            placeholder="e.g. chair, table, floor ...", width=400
        )
        self._query_btn = pn.widgets.Button(
            name="Query", button_type="success", width=100, disabled=True
        )
        self._query_bar = pn.Row(self._query_input, self._query_btn, visible=False)

        # Wire callbacks
        self._frame_slider.param.watch(self._on_slider_change, "value")
        self._prev_btn.on_click(self._on_prev)
        self._next_btn.on_click(self._on_next)
        self._query_btn.on_click(self._on_query)
        state.param.watch(self._on_frames_zarr_ready, "frames_zarr_path")

        # Activate immediately if state already populated (loaded session)
        if state.frames_zarr_path is not None:
            self._activate(Path(state.frames_zarr_path))

    def _on_frames_zarr_ready(self, event: Any) -> None:
        """Enable frame selector and load first frame when zarr path is set."""
        if event.new is not None:
            self._activate(Path(event.new))

    def _activate(self, frames_zarr_path: Path) -> None:
        """Enable controls, set slider range, display first frame."""
        import zarr

        z = zarr.open(str(frames_zarr_path), mode="r")
        n = int(z.attrs["n_frames"])
        self._frame_slider.end = max(0, n - 1)
        self._frame_slider.disabled = False
        self._prev_btn.disabled = False
        self._next_btn.disabled = False
        self._frame_count_html.object = (
            f"<span style='font-size:11px;color:#aaa'>{n} frames</span>"
        )
        self._load_original_frame(0)

    def _on_slider_change(self, event: Any) -> None:
        """Refresh original frame and all extractor columns on slider change."""
        idx = int(event.new)
        self._current_frame_idx = idx
        self._load_original_frame(idx)
        for col in self._columns:
            col.refresh(idx)
        self._update_query_bar_visibility()

    def _on_prev(self, event: Any) -> None:
        """Step slider back one frame."""
        if self._frame_slider.value > 0:
            self._frame_slider.value -= 1

    def _on_next(self, event: Any) -> None:
        """Step slider forward one frame."""
        if self._frame_slider.value < self._frame_slider.end:
            self._frame_slider.value += 1

    def _load_original_frame(self, idx: int) -> None:
        """Load frame from zarr and display in the original-frame pane."""
        from io import BytesIO

        from PIL import Image as PILImage

        if self._state.frames_zarr_path is None:
            return
        frame = _load_frame_rgb(Path(self._state.frames_zarr_path), idx)
        buf = BytesIO()
        PILImage.fromarray(frame).save(buf, format="PNG")
        self._original_pane.object = buf.getvalue()

    def _update_query_bar_visibility(self) -> None:
        """Show query bar when any column has a ready queryable extractor."""
        any_queryable = any(c.is_queryable_and_ready() for c in self._columns)
        self._query_bar.visible = any_queryable
        self._query_btn.disabled = not any_queryable

    def _on_query(self, event: Any) -> None:
        """Launch background query thread across all ready queryable columns."""
        if self._query_thread and self._query_thread.is_alive():
            return
        query_text = self._query_input.value.strip()
        if not query_text:
            return
        self._query_btn.disabled = True
        frame_idx = self._current_frame_idx
        self._query_thread = threading.Thread(
            target=self._run_query, args=(frame_idx, query_text), daemon=True
        )
        self._query_thread.start()

    def _run_query(self, frame_idx: int, query_text: str) -> None:
        """Background: call render_query on each ready queryable column."""
        try:
            for col in self._columns:
                if col.is_queryable_and_ready():
                    col.render_query(frame_idx, query_text)
        except Exception as exc:
            logger.exception("Query failed")
            self._op_log.error_op(f"Query error: {exc}")
        finally:
            self._update_query_bar_visibility()

    def panel(self) -> pn.Column:
        """Return Panel layout for Tab 2."""
        frame_selector = pn.Row(
            self._prev_btn,
            self._frame_slider,
            self._next_btn,
            self._frame_count_html,
        )
        grid = pn.Row(
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>Original</b>"),
                self._original_pane,
            ),
            *self._col_panels,
        )
        return pn.Column(
            pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Semantics</h3>"),
            frame_selector,
            grid,
            self._query_bar,
            sizing_mode="stretch_width",
        )
