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
import zarr
from io import BytesIO
from PIL import Image as PILImage

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
    z = zarr.open(str(frames_zarr_path), mode="r")
    return z["frames"][idx]


########################################################################
# SemanticsPane
########################################################################


class SemanticsPane(param.Parameterized):
    """Feature extraction pane: single extractor, three-panel display."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._extractor: BaseFeatureExtractor | None = None
        self._feature_zarr_path: Path | None = None
        self._current_frame_idx: int = 0
        self._run_thread: threading.Thread | None = None
        self._query_thread: threading.Thread | None = None

        # Frame selector
        self._frame_slider = pn.widgets.IntSlider(
            name="Frame", start=0, end=0, value=0, width=300
        )
        self._prev_btn = pn.widgets.Button(name="◄", width=40)
        self._next_btn = pn.widgets.Button(name="►", width=40)
        self._frame_count_html = pn.pane.HTML("", width=100)

        # Single extractor selector
        self._method_dd = pn.widgets.Select(
            name="Extractor",
            options=list(BaseFeatureExtractor._registry.keys()),
            width=200,
        )
        self._run_btn = pn.widgets.Button(name="▶ Run", button_type="primary", width=90)
        self._status_html = pn.pane.HTML("", width=400)

        # Three image panels: ground truth | PCA features | cosine similarity
        self._original_pane = pn.pane.PNG(None, width=320, height=240)
        self._pca_pane = pn.pane.PNG(None, width=320, height=240)
        self._sim_pane = pn.pane.PNG(None, width=320, height=240)

        # Query bar
        self._query_input = pn.widgets.TextInput(
            placeholder="Enter text query…", width=300
        )
        self._query_btn = pn.widgets.Button(
            name="Query", button_type="success", width=80, disabled=True
        )

        # Wire callbacks
        self._frame_slider.param.watch(self._on_frame_slider, "value")
        self._prev_btn.on_click(lambda e: self._step_frame(-1))
        self._next_btn.on_click(lambda e: self._step_frame(1))
        self._run_btn.on_click(self._on_run)
        self._query_btn.on_click(self._on_query)
        self._state.param.watch(self._on_frames_zarr_change, "frames_zarr_path")
        self._state.param.watch(self._on_frames_zarr_change, "selected_indices")

    def _on_frames_zarr_change(self, event: Any) -> None:
        """Update frame slider range and load first frame when zarr is ready."""
        if self._state.frames_zarr_path is None:
            return
        z = zarr.open(str(self._state.frames_zarr_path), mode="r")
        n = int(z["frames"].shape[0])
        self._frame_slider.end = max(0, n - 1)
        self._frame_count_html.object = f"<small>/ {n}</small>"
        self._load_original(0)

    def _on_frame_slider(self, event: Any) -> None:
        """Update displays when frame slider moves."""
        idx = event.new
        self._current_frame_idx = idx
        self._load_original(idx)
        if self._feature_zarr_path is not None:
            self._refresh_pca(idx)

    def _step_frame(self, delta: int) -> None:
        """Move frame slider by delta, clamped to valid range."""
        new_val = max(
            self._frame_slider.start,
            min(self._frame_slider.end, self._frame_slider.value + delta),
        )
        self._frame_slider.value = new_val

    def _load_original(self, idx: int) -> None:
        """Load and display the ground-truth frame for idx."""
        if self._state.frames_zarr_path is None:
            return
        frame = _load_frame_rgb(Path(self._state.frames_zarr_path), idx)
        buf = BytesIO()
        PILImage.fromarray(frame).save(buf, format="PNG")
        self._original_pane.object = buf.getvalue()

    def _refresh_pca(self, idx: int) -> None:
        """Render PCA RGB overlay for features at frame idx."""
        if self._feature_zarr_path is None:
            return
        z = zarr.open(str(self._feature_zarr_path), mode="r")
        feat = torch.from_numpy(np.array(z["features"][idx]))  # (D, H_p, W_p)
        rgb = BaseFeatureExtractor.features_to_rgb(feat)
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        self._pca_pane.object = buf.getvalue()

    def _on_run(self, event: Any) -> None:
        """Start background extraction when Run button clicked."""
        if self._run_thread and self._run_thread.is_alive():
            return
        if self._state.frames_zarr_path is None or self._state.output_dir is None:
            self._status_html.object = (
                "<small style='color:#e05050'>Load frames first (Preprocess tab)</small>"
            )
            return
        method = self._method_dd.value
        self._run_btn.disabled = True
        self._status_html.object = f"<small style='color:#aaa'>Running {method}…</small>"
        self._run_thread = threading.Thread(
            target=self._run_extraction, args=(method,), daemon=True
        )
        self._run_thread.start()

    def _run_extraction(self, method: str) -> None:
        """Background: instantiate extractor, extract all frames, render PCA for current frame."""
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
            self._op_log.finish_op()
            self._refresh_pca(self._current_frame_idx)
            self._update_query_btn()
            self._status_html.object = f"<small style='color:#50c050'>{method} ready</small>"
        except Exception as exc:
            logger.exception("Extraction failed for %s", method)
            self._status_html.object = f"<small style='color:#e05050'>Error: {exc}</small>"
            self._op_log.error_op(str(exc))
        finally:
            self._run_btn.disabled = False

    def _update_query_btn(self) -> None:
        """Enable query button only for queryable extractors with loaded features."""
        queryable = (
            self._feature_zarr_path is not None
            and isinstance(self._extractor, BaseQueryableExtractor)
        )
        self._query_btn.disabled = not queryable
        if self._feature_zarr_path is not None and not queryable:
            self._status_html.object = (
                "<br><small style='color:#888'>"
                "Run a queryable extractor (e.g. DINOv2 SAM) to enable text queries"
                "</small>"
            )

    def _on_query(self, event: Any) -> None:
        """Start background query thread."""
        if self._query_thread and self._query_thread.is_alive():
            return
        query_text = self._query_input.value.strip()
        if not query_text:
            return
        self._query_btn.disabled = True
        self._query_thread = threading.Thread(
            target=self._run_query,
            args=(self._current_frame_idx, query_text),
            daemon=True,
        )
        self._query_thread.start()

    def _run_query(self, frame_idx: int, query_text: str) -> None:
        """Background: compute cosine similarity heatmap and display in sim pane."""
        try:
            z = zarr.open(str(self._feature_zarr_path), mode="r")
            feat = torch.from_numpy(
                np.array(z["features"][frame_idx]).astype(np.float32)
            )
            score = self._extractor.score_queries(
                feat, positive=[query_text]
            )  # (H_p, W_p)
            rgb = _score_to_rgb(score.cpu().numpy())
            buf = BytesIO()
            PILImage.fromarray(rgb).save(buf, format="PNG")
            self._sim_pane.object = buf.getvalue()
        except Exception as exc:
            logger.exception("Query failed")
            self._op_log.error_op(f"Query error: {exc}")
        finally:
            self._update_query_btn()

    def panel(self) -> pn.Column:
        """Return Panel layout for Tab 2."""
        top_row = pn.Row(
            self._prev_btn,
            self._frame_slider,
            self._next_btn,
            self._frame_count_html,
            pn.HSpacer(),
            pn.pane.HTML("<b style='align-self:center'>Extractor:</b>"),
            self._method_dd,
            self._run_btn,
        )
        image_row = pn.Row(
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>Ground truth</b>"),
                self._original_pane,
            ),
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>PCA features</b>"),
                self._pca_pane,
            ),
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>Cosine similarity</b>"),
                self._sim_pane,
            ),
            sizing_mode="stretch_width",
        )
        query_row = pn.Row(
            pn.pane.HTML("<b style='align-self:center'>Query:</b>"),
            self._query_input,
            self._query_btn,
        )
        return pn.Column(
            pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Semantics</h3>"),
            top_row,
            self._status_html,
            image_row,
            query_row,
            sizing_mode="stretch_width",
        )
