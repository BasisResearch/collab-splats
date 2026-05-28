from __future__ import annotations

########################################################################
# Imports
########################################################################

import io
import logging
import threading
from pathlib import Path
from typing import Any

import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
from PIL import Image

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


def _render_metrics_figure(
    frame_scores: dict[str, list[float]],
    selected_indices: list[int],
    total_frames: int,
) -> bytes:
    """Render stacked per-frame metrics time-series as PNG bytes.

    Uses score_all_frames() output keys: disparity / rotation / hist_similarity.
    Selected frame indices shown as green vertical bands.
    """
    keys = ["disparity", "rotation", "hist_similarity"]
    labels = ["Optical Flow / Disparity", "Rotation (°)", "Hist. Similarity"]
    colors = ["#7ec8e3", "#f0a500", "#d090e0"]

    n_panels = sum(1 for k in keys if frame_scores.get(k))
    if n_panels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 1))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No metrics — run extraction first",
            ha="center", va="center", color="#666", transform=ax.transAxes,
        )
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, n_panels * 1.2), sharex=True)
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0d1117")
        fig.subplots_adjust(hspace=0.15)

        panel_idx = 0
        for key, label, color in zip(keys, labels, colors):
            vals = frame_scores.get(key, [])
            if not vals:
                continue
            ax = axes[panel_idx]
            ax.set_facecolor("#111827")
            ax.plot(vals, color=color, linewidth=0.9, alpha=0.9)
            ax.set_ylabel(label, color=color, fontsize=7, labelpad=2)
            ax.tick_params(colors="#555", labelsize=6)
            for spine in ax.spines.values():
                spine.set_color("#333")
            for sel_idx in selected_indices:
                if 0 <= sel_idx < len(vals):
                    ax.axvline(x=sel_idx, color="#50c050", alpha=0.5, linewidth=0.7)
            panel_idx += 1

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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


def _write_frames_zarr(frames: list[np.ndarray], zarr_path: Path) -> Path:
    """Write extracted frames to a Blosc-compressed zarr store.

    Layout: frames (N, H, W, 3) uint8, chunks=(1, H, W, 3) — one chunk per frame.
    Returns the zarr store path.
    """
    import zarr
    from zarr.codecs import BloscCodec

    N = len(frames)
    H, W = frames[0].shape[:2]
    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({"n_frames": N, "height": H, "width": W})
    arr = store.create_array(
        "frames",
        shape=(N, H, W, 3),
        chunks=(1, H, W, 3),
        dtype="uint8",
        fill_value=0,
        compressors=[BloscCodec(cname="lz4", clevel=5)],
    )
    for i, frame in enumerate(frames):
        arr[i] = frame
    return zarr_path


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

        # Video player
        self._video_pane = pn.pane.Video(None, width=560, height=360, loop=False, visible=False)
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

        # Metrics (matplotlib PNG)
        self._metrics_pane = pn.pane.PNG(None, width=700, height=200, visible=False)

        # Frame strip
        self._frame_strip_row = pn.Row(scroll=True, height=150, sizing_mode="stretch_width")
        self._frame_strip_label = pn.pane.HTML("", sizing_mode="stretch_width")

        # Wire callbacks
        self._method_dd.param.watch(self._on_method_change, "value")
        self._extract_btn.on_click(self._on_extract)
        self._state.param.watch(self._on_video_path_change, "video_path")

    def _on_video_path_change(self, event: Any) -> None:
        """Auto-load video display when AppState.video_path is set."""
        if event.new and Path(event.new).exists():
            self._load_video(Path(event.new))

    def _load_video(self, video_path: Path) -> None:
        """Update video player and info text for the given path."""
        info = get_video_info(str(video_path))
        self._video_pane.object = str(video_path)
        self._video_pane.visible = True
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

            # Write output_dir if not already set (session started from video path)
            if self._state.output_dir is None and self._state.video_path is not None:
                self._state.output_dir = Path("/workspace/outputs") / Path(self._state.video_path).stem

            # Write frames to zarr and update shared state
            output_dir = Path(self._state.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            frames_zarr_path = _write_frames_zarr(frames, output_dir / "frames.zarr")
            self._state.frames_zarr_path = frames_zarr_path

            # Update metrics display with selected frame markers
            info = get_video_info(str(video_path))
            total = info.get("total_frames", len(frames))
            metrics_png = _render_metrics_figure(frame_scores, self._selected_indices, total)
            self._metrics_pane.object = metrics_png
            self._metrics_pane.visible = True

            # Update frame strip (cap at 100 thumbnails for performance)
            thumbnails = _frames_to_thumbnails(frames[:100])
            self._frame_strip_row.objects = [
                pn.pane.PNG(t, width=120, height=90) for t in thumbnails
            ]
            self._frame_strip_label.object = (
                f"<p style='font-size:11px;color:#aaa'>{len(frames)} frames selected</p>"
            )
            self._frame_count_html.object = (
                f"<p style='font-size:11px;color:#50c050'>✓ {len(frames)} frames extracted</p>"
            )
            self._op_log.finish_op()
        except Exception as exc:
            logger.exception("Frame extraction failed")
            self._op_log.error_op(str(exc))
            self._frame_count_html.object = (
                f"<p style='color:#e05050'>Extraction failed: {exc}</p>"
            )
        finally:
            self._extract_btn.disabled = False

    def panel(self) -> pn.Column:
        """Return the full PreprocessPane Panel layout."""
        controls = pn.Column(
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 6px 0'>Frame Selection</h4>"),
            self._method_dd,
            self._fps_slider,
            self._n_frames_slider,
            self._window_start_slider,
            self._window_end_slider,
            self._min_disparity_slider,
            pn.layout.Divider(),
            self._extract_btn,
            self._frame_count_html,
            width=300,
        )

        video_col = pn.Column(self._video_pane, self._video_info_html)
        top_row = pn.Row(video_col, controls, sizing_mode="stretch_width")

        return pn.Column(
            top_row,
            pn.layout.Divider(),
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 4px 0'>Frame Quality Metrics</h4>"),
            self._metrics_pane,
            pn.layout.Divider(),
            self._frame_strip_label,
            self._frame_strip_row,
            sizing_mode="stretch_width",
        )
