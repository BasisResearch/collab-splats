from __future__ import annotations

########################################################################
# Imports
########################################################################

import base64
import io
import logging
import threading
from pathlib import Path
from typing import Any

from bokeh.models import ColumnDataSource, Span, TapTool
from bokeh.plotting import figure as bokeh_figure
import numpy as np
import panel as pn
import param
import zarr
from PIL import Image
from zarr.codecs import BloscCodec

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.utils.frame_sampling import (
    get_video_info,
    load_video_frames,
    sample_frames_fps,
    sample_frames_optical_flow,
    score_all_frames,
)

logger = logging.getLogger(__name__)


########################################################################
# Pure helpers — testable without Panel
########################################################################

def _window_frame_indices(
    total_frames: int,
    window_start: float,
    window_end: float,
) -> list[int]:
    """Return sorted list of frame indices within [window_start, window_end] fraction."""
    start = max(0, int(window_start * total_frames))
    end = min(total_frames, int(window_end * total_frames))
    return list(range(start, end))


########################################################################
# Metrics constants
########################################################################

_METRIC_STYLE: dict[str, tuple[str, str]] = {
    "disparity": ("Optical Flow / Disparity", "#7ec8e3"),
    "rotation": ("Rotation (°)", "#f0a500"),
    "hist_similarity": ("Hist. Similarity", "#d090e0"),
}


def _build_metrics_sources(
    frame_scores: dict[str, list[float]],
) -> dict[str, ColumnDataSource]:
    """Build Bokeh ColumnDataSources for each non-empty metric series."""
    sources = {}
    for key in _METRIC_STYLE:
        vals = frame_scores.get(key, [])
        if not vals:
            continue
        sources[key] = ColumnDataSource(
            data={"x": list(range(len(vals))), "y": vals}
        )
    return sources


def _frames_to_thumbnails(
    frames: list[np.ndarray],
    max_size: tuple[int, int] = (160, 120),
) -> list[bytes]:
    """Convert RGB numpy frame arrays to PNG thumbnail bytes."""
    thumbnails = []
    for frame in frames:
        img = Image.fromarray(frame)
        img.thumbnail(max_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        thumbnails.append(buf.getvalue())
    return thumbnails


def _write_frames_zarr(frames: list[np.ndarray], path: Path) -> None:
    """Write frame list to a zarr store at path, one chunk per frame."""
    if not frames:
        raise ValueError("frames list is empty — nothing to write")
    arr = np.stack(frames)  # (N, H, W, 3) uint8
    store = zarr.open_group(str(path), mode="w")
    store.create_array(
        "frames",
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=(1, *arr.shape[1:]),
        compressors=[BloscCodec(cname="lz4", clevel=3)],
    )
    store["frames"][:] = arr


def _build_frame_strip_html(thumbnails: list[bytes], active_idx: int) -> str:
    """Return HTML string for horizontal frame strip with base64 thumbnails.

    Each thumbnail has id="frame-{i}" for scrollIntoView targeting.
    Active frame gets green border; others get dark border.
    """
    if not thumbnails:
        return "<div style='color:#666;font-size:11px;padding:8px'>No frames extracted yet</div>"

    # Clamp active_idx to valid range — silently out-of-range is confusing
    active_idx = max(0, min(active_idx, len(thumbnails) - 1)) if thumbnails else 0

    imgs = []
    for i, png_bytes in enumerate(thumbnails):
        b64 = base64.b64encode(png_bytes).decode()
        border_color = "#50c050" if i == active_idx else "#333"
        imgs.append(
            f'<img id="frame-{i}" src="data:image/png;base64,{b64}" '
            f'style="width:120px;height:90px;cursor:pointer;margin:2px;'
            f'border:2px solid {border_color};border-radius:3px;flex-shrink:0;" />'
        )

    inner = "".join(imgs)
    return (
        f'<div id="frame-strip-container" '
        f'style="display:flex;flex-direction:row;overflow-x:auto;'
        f'padding:4px;background:#0d1117;border-radius:4px;">'
        f"{inner}</div>"
    )


########################################################################
# PreprocessPane
########################################################################

class PreprocessPane(param.Parameterized):
    """Video preprocessing pane: extract keyframes, view quality metrics, confirm frame set."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._frame_scores: dict[str, list[float]] = {}
        self._selected_frames: list[np.ndarray] = []
        self._selected_indices: list[int] = []
        self._extraction_thread: threading.Thread | None = None
        self._cached_thumbnails: list[bytes] = []

        # Video first-frame thumbnail (faster than streaming 700MB via Bokeh server)
        self._video_pane = pn.pane.PNG(None, width=560, height=360, visible=False)
        self._video_info_html = pn.pane.HTML("", width=560)

        # Frame selection controls
        self._method_dd = pn.widgets.Select(
            name="Method", options=["fps", "optical_flow"], value="fps", width=200
        )
        self._n_frames_slider = pn.widgets.IntSlider(
            name="Target frames", value=200, start=20, end=2000, step=10, width=260
        )
        self._fps_slider = pn.widgets.FloatSlider(
            name="FPS", value=2.0, start=0.5, end=30.0, step=0.5, width=260
        )
        self._window_start_slider = pn.widgets.FloatSlider(
            name="Window start (%)", value=0.0, start=0.0, end=1.0, step=0.01, width=260
        )
        self._window_end_slider = pn.widgets.FloatSlider(
            name="Window end (%)", value=1.0, start=0.0, end=1.0, step=0.01, width=260
        )
        self._min_disparity_slider = pn.widgets.FloatSlider(
            name="Min disparity", value=50.0, start=10.0, end=200.0, step=5.0,
            width=260, visible=False,
        )
        self._extract_btn = pn.widgets.Button(
            name="▶ Extract Frames", button_type="primary", width=260
        )
        self._frame_count_html = pn.pane.HTML("", width=260)

        # Controls wrapped in collapsible Card — hidden until video is loaded
        self._controls_card = pn.Card(
            self._method_dd,
            self._fps_slider,
            self._n_frames_slider,
            self._window_start_slider,
            self._window_end_slider,
            self._min_disparity_slider,
            pn.layout.Divider(),
            self._extract_btn,
            self._frame_count_html,
            title="Frame Extraction",
            collapsed=False,
            visible=False,
            sizing_mode="stretch_width",
        )

        # Metrics (Bokeh figures, built lazily after extraction)
        self._metrics_col = pn.Column(sizing_mode="stretch_width", visible=False)

        # Frame strip (HTML for scrollIntoView support)
        self._frame_strip_pane = pn.pane.HTML(
            _build_frame_strip_html([], active_idx=0),
            sizing_mode="stretch_width",
            height=120,
        )
        self._scroll_script = pn.pane.HTML("", width=0, height=0)
        self._active_frame_idx: int = 0

        # Wire callbacks
        self._method_dd.param.watch(self._on_method_change, "value")
        self._extract_btn.on_click(self._on_extract)
        self._state.param.watch(self._on_video_path_change, "video_path")

    def _on_video_path_change(self, event: Any) -> None:
        """Auto-load video display and reveal controls when AppState.video_path is set."""
        if event.new and Path(event.new).exists():
            self._load_video(Path(event.new))
            self._controls_card.visible = True

    def _load_video(self, video_path: Path) -> None:
        """Extract first frame as thumbnail + fetch metadata in background."""
        import cv2  # optional heavy dep — imported here intentionally
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(frame_rgb).save(buf, format="PNG")
            self._video_pane.object = buf.getvalue()
            self._video_pane.visible = True
        self._video_info_html.object = (
            f"<p style='font-size:11px;color:#aaa'>{video_path.name} · loading info…</p>"
        )
        # Fetch full metadata (frame count, fps) without blocking IOLoop
        threading.Thread(target=self._fetch_video_info, args=(video_path,), daemon=True).start()

    def _fetch_video_info(self, video_path: Path) -> None:
        """Background: read video metadata and update info HTML."""
        info = get_video_info(str(video_path))
        self._video_info_html.object = (
            f"<p style='font-size:11px;color:#aaa'>"
            f"{video_path.name} · {info.get('total_frames', '?')} frames · "
            f"{info.get('fps', 0.0):.1f} fps · "
            f"{info.get('duration_s', 0) / 60:.1f} min</p>"
        )

    def _on_method_change(self, event: Any) -> None:
        """Toggle visibility of method-specific controls."""
        is_of = event.new == "optical_flow"
        self._min_disparity_slider.visible = is_of
        self._fps_slider.visible = not is_of

    def _on_extract(self, event: Any) -> None:
        """Start background extraction thread when Extract button is clicked."""
        video_path = self._state.video_path
        if video_path is None or not Path(video_path).exists():
            self._frame_count_html.object = (
                "<p style='color:#e05050'>Set video path in sidebar first</p>"
            )
            return
        if self._extraction_thread and self._extraction_thread.is_alive():
            return
        self._extract_btn.disabled = True
        self._extraction_thread = threading.Thread(
            target=self._run_extraction, args=(Path(video_path),), daemon=True
        )
        self._extraction_thread.start()

    def _run_extraction(self, video_path: Path) -> None:
        """Background thread: score frames, extract keyframes, update UI state."""
        try:
            # Score all frames for the metrics chart
            self._op_log.start_op("Computing frame scores")
            frame_scores = score_all_frames(
                str(video_path),
                on_progress=lambda pct, msg="": self._op_log.update_progress(pct // 2, msg),
                verbose=False,
            )
            self._frame_scores = frame_scores

            # Extract frames using selected method
            self._op_log.start_op("Extracting frames")
            method = self._method_dd.value
            if method == "fps":
                frames = sample_frames_fps(
                    str(video_path),
                    fps=self._fps_slider.value,
                    max_frames=self._n_frames_slider.value,
                    on_progress=lambda pct, msg="": self._op_log.update_progress(50 + pct // 2, msg),
                    verbose=False,
                )
            else:
                frames = sample_frames_optical_flow(
                    str(video_path),
                    min_disparity=self._min_disparity_slider.value,
                    max_frames=self._n_frames_slider.value,
                    on_progress=lambda pct, msg="": self._op_log.update_progress(50 + pct // 2, msg),
                    verbose=False,
                )

            # Apply window filter proportionally to extracted set
            if self._window_start_slider.value > 0.0 or self._window_end_slider.value < 1.0:
                n_total = len(frames)
                start_i = int(self._window_start_slider.value * n_total)
                end_i = max(start_i + 1, int(self._window_end_slider.value * n_total))
                frames = frames[start_i:end_i]

            self._selected_frames = frames
            self._selected_indices = list(range(len(frames)))

            # Write frames to zarr; set state (no in-memory frame list)
            if self._state.output_dir is None and self._state.video_path is not None:
                self._state.output_dir = Path("/workspace/outputs") / Path(self._state.video_path).stem
            zarr_path = Path(self._state.output_dir) / "frames.zarr"
            _write_frames_zarr(frames, zarr_path)
            self._state.frames_zarr_path = zarr_path
            self._state.selected_indices = self._selected_indices
            self._selected_frames = []  # clear after zarr write — frames now on disk

            # Rebuild Bokeh metrics panel with new data
            self._metrics_col.objects = [self._make_metrics_panel()]
            self._metrics_col.visible = True

            # Generate initial thumbnails and render HTML strip
            thumbnails = _frames_to_thumbnails(frames[:100])
            self._cached_thumbnails = thumbnails  # cache for tap callbacks
            self._active_frame_idx = 0
            self._frame_strip_pane.object = _build_frame_strip_html(thumbnails, active_idx=0)
            self._frame_count_html.object = (
                f"<p style='font-size:11px;color:#50c050'>✓ {len(frames)} frames extracted</p>"
            )

            # Auto-collapse controls card now that extraction is done
            self._controls_card.collapsed = True
            self._op_log.finish_op()
        except Exception as exc:
            logger.exception("Frame extraction failed")
            self._op_log.error_op(str(exc))
            self._frame_count_html.object = (
                f"<p style='color:#e05050'>Extraction failed: {exc}</p>"
            )
        finally:
            self._extract_btn.disabled = False

    def _seek_to_frame(self, frame_idx: int) -> None:
        """Seek video to frame_idx; update active thumbnail in strip; scroll strip."""
        self._active_frame_idx = frame_idx

        # Seek video player to timestamp
        if self._state.video_path:
            try:
                info = get_video_info(str(self._state.video_path))
                fps = info.get("fps", 25.0)
                self._video_pane.time = frame_idx / fps
            except Exception as exc:
                logger.warning("Could not seek video to frame %d: %s", frame_idx, exc)

        # Rebuild strip HTML using cached thumbnails (avoid re-decoding zarr on each tap)
        self._frame_strip_pane.object = _build_frame_strip_html(
            self._cached_thumbnails, active_idx=frame_idx
        )

        # Inject scroll script to jump strip to the active thumbnail
        self._scroll_script.object = (
            f"<script>var el=document.getElementById('frame-{frame_idx}');"
            f"if(el)el.scrollIntoView({{behavior:'smooth',inline:'center'}});</script>"
        )

    def _make_metrics_panel(self) -> pn.Column:
        """Build stacked Bokeh metric figures with TapTool; return as pn.Column."""
        sources = _build_metrics_sources(self._frame_scores)
        if not sources:
            return pn.Column(
                pn.pane.HTML(
                    "<p style='color:#666;font-size:11px'>No metrics — run extraction first</p>"
                )
            )

        figs = []
        for key, source in sources.items():
            label, color = _METRIC_STYLE[key]
            p = bokeh_figure(
                height=110,
                sizing_mode="stretch_width",
                toolbar_location=None,
                x_range=(0, max(source.data["x"]) + 1),
            )
            p.background_fill_color = "#111827"
            p.border_fill_color = "#0d1117"
            p.outline_line_color = "#333"
            p.grid.grid_line_color = "#333"
            p.yaxis.axis_label = label
            p.yaxis.axis_label_text_color = color
            p.yaxis.axis_label_text_font_size = "10px"
            p.xaxis.major_label_text_color = "#555"
            p.yaxis.major_label_text_color = "#555"

            # Line trace
            p.line("x", "y", source=source, color=color, line_width=1.2, alpha=0.9)

            # Invisible circles for tap selection
            circles = p.circle("x", "y", source=source, size=6, alpha=0, color=color)

            # Green span at each selected frame index
            for idx in self._selected_indices:
                p.add_layout(
                    Span(location=idx, dimension="height",
                         line_color="#50c050", line_alpha=0.5, line_width=1.0)
                )

            # TapTool fires Python callback via source selection
            tap = TapTool(renderers=[circles])
            p.add_tools(tap)

            def _on_tap(attr, old, new, src=source):  # noqa: ANN001
                if new:
                    x_val = src.data["x"][new[0]]
                    # Snap to nearest selected frame index
                    if self._selected_indices:
                        nearest = min(self._selected_indices, key=lambda i: abs(i - x_val))
                        self._seek_to_frame(nearest)

            source.selected.on_change("indices", _on_tap)
            figs.append(pn.pane.Bokeh(p, sizing_mode="stretch_width"))

        return pn.Column(*figs, sizing_mode="stretch_width")

    def panel(self) -> pn.Row:
        """Return the full PreprocessPane Panel layout."""
        # Left column: video player + info
        video_col = pn.Column(
            self._video_pane,
            self._video_info_html,
            sizing_mode="stretch_height",
            min_width=400,
        )

        # Right column: controls card → metrics → frame strip → scroll script
        right_col = pn.Column(
            self._controls_card,
            pn.layout.Divider(),
            pn.pane.HTML(
                "<h4 style='color:#7ec8e3;margin:4px 0'>Frame Quality Metrics</h4>",
                sizing_mode="stretch_width",
            ),
            self._metrics_col,
            pn.layout.Divider(),
            pn.pane.HTML(
                "<h4 style='color:#7ec8e3;margin:4px 0'>Selected Frames</h4>",
                sizing_mode="stretch_width",
            ),
            self._frame_strip_pane,
            self._scroll_script,
            sizing_mode="stretch_both",
        )

        return pn.Row(video_col, right_col, sizing_mode="stretch_width")
