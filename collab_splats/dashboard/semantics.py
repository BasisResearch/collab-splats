"""
Semantics + training pipeline dashboard.

Launch with:
    collab-dashboard semantics --base-dir /workspace/fieldwork-data
"""

from __future__ import annotations

import io
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import torch
import numpy as np
import panel as pn
import param
from PIL import Image

from collab_splats.dashboard.config_panel import ConfigPanel
from collab_splats.dashboard.video_discovery import discover_videos, yaml_name_for_video
from collab_splats.utils.frame_sampling import sample_frames_fps, sample_frames_optical_flow

CONFIGS_DIR = Path(__file__).parents[2] / "docs" / "splats" / "configs"

_DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

_TAB_CSS = """
.bk-tab.bk-active {
    border-bottom: 4px solid #2596be !important;
    font-weight: 700 !important;
    color: #2596be !important;
}
.bk-tab {
    color: #aaa;
    font-weight: 400;
}
"""


def _mode_badge(label: str) -> "pn.pane.HTML":
    style = (
        "display:inline-flex;align-items:center;gap:6px;"
        "background:#e8f4fd;border:1px solid #2596be;"
        "border-radius:20px;padding:4px 14px;margin-bottom:12px"
    )
    dot = "<span style='width:8px;height:8px;border-radius:50%;background:#2596be;display:inline-block'></span>"
    text_style = "font-size:12px;font-weight:700;color:#2596be;text-transform:uppercase;letter-spacing:0.5px"
    return pn.pane.HTML(f"<div style='{style}'>{dot}<span style='{text_style}'>{label}</span></div>")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------



def _pca_to_rgb(features_chw: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    C, H, W = features_chw.shape
    flat = features_chw.reshape(C, -1).T
    rgb_flat = PCA(n_components=3).fit_transform(flat)
    lo, hi = rgb_flat.min(0), rgb_flat.max(0)
    rgb_flat = (rgb_flat - lo) / (hi - lo + 1e-8)
    return (rgb_flat.reshape(H, W, 3) * 255).astype(np.uint8)


def _overlay_pca(frame_rgb: np.ndarray, pca_rgb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    frame_pil = Image.fromarray(frame_rgb).resize((pca_rgb.shape[1], pca_rgb.shape[0]))
    blended = np.array(frame_pil).astype(float) * (1 - alpha) + pca_rgb.astype(float) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def _colorize_masks(frame_rgb: np.ndarray, masks_tensor: Any) -> np.ndarray:
    COLORS = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 255, 80),
        (80, 255, 255),
        (255, 80, 255),
        (255, 160, 80),
        (160, 80, 255),
    ]
    vis = frame_rgb.copy().astype(np.float32)
    for i, mask in enumerate(masks_tensor):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        m = mask.numpy().astype(bool)
        vis[m] = vis[m] * 0.45 + color * 0.55
    return vis.clip(0, 255).astype(np.uint8)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------


class SemanticsDashboard(param.Parameterized):
    """Panel dashboard for semantic exploration and Splatter training."""

    def __init__(self, base_dir: str = "/workspace/fieldwork-data", **params: Any):
        super().__init__(**params)

        self._base_dir = base_dir
        self._video_tree: dict = {}
        self._frames: list[np.ndarray] = []
        self._current_frame: np.ndarray | None = None
        self._training_process: subprocess.Popen | None = None
        self._poll_cb: Any = None
        self._tmp_cfg_path: str | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._config_panel = ConfigPanel(configs_dir=CONFIGS_DIR)

        try:
            from collab_splats.semantics.features import BaseFeatureExtractor

            extractor_names = list(BaseFeatureExtractor._registry.keys()) or ["(none)"]
        except Exception:
            extractor_names = ["(none)"]

        # Sidebar widgets
        self.base_dir_input = pn.widgets.TextInput(name="Base directory", value=base_dir, width=350)
        self.species_select = pn.widgets.Select(name="Species", options=[], width=350)
        self.date_select = pn.widgets.Select(name="Date", options=[], width=350)
        self.video_select = pn.widgets.Select(name="Video", options=[], width=350)
        self.save_yaml_btn = pn.widgets.Button(name="Save YAML", button_type="success", width=160)
        self.reset_yaml_btn = pn.widgets.Button(name="Reset to base", button_type="light", width=160)

        # Status / loading
        self.status_pane = pn.pane.HTML("<p>Ready</p>", width=800, height=30)
        self.loading_modal = pn.pane.HTML("", visible=False, sizing_mode="stretch_both")

        # Progress bar for frame extraction
        self.progress_bar = pn.widgets.Progress(
            name="Extracting frames",
            value=0,
            max=100,
            bar_color="info",
            sizing_mode="stretch_width",
            visible=False,
        )
        self.progress_label = pn.pane.HTML("", visible=False)

        # Extraction thread state
        self._extraction_progress: tuple[int, int] = (0, 1)
        self._extraction_lock = threading.Lock()
        self._extraction_done = False
        self._extraction_error: str | None = None
        self._extraction_cb: Any = None

        # Training tab widgets
        self.output_path_input = pn.widgets.TextInput(name="Output path", value="/workspace/outputs", width=700)
        self.launch_btn = pn.widgets.Button(name="Launch Splatter", button_type="primary", width=160)
        self.stop_btn = pn.widgets.Button(name="Stop", button_type="danger", width=100, disabled=True)
        self.training_log = pn.widgets.TextAreaInput(name="Training log", value="", rows=20, width=700, disabled=True)

        # Explore tab widgets
        self.fps_slider = pn.widgets.IntSlider(name="FPS to extract", value=5, start=1, end=30, width=350)
        self.max_frames_slider = pn.widgets.IntSlider(name="Max frames", value=200, start=10, end=2000, step=10, width=350)
        self.sampling_mode_dd = pn.widgets.Select(
            name="Sampling mode",
            options=["FPS", "Optical Flow"],
            value="FPS",
            width=200,
            disabled=False,
        )
        self.min_disparity_slider = pn.widgets.FloatSlider(
            name="Min Disparity (px)", value=50.0, start=10.0, end=200.0, step=5.0,
            visible=False,
        )
        self.motion_weight_slider = pn.widgets.FloatSlider(
            name="Motion Weight", value=0.6, start=0.0, end=1.0, step=0.05,
        )
        self.coverage_weight_slider = pn.widgets.FloatSlider(
            name="Coverage Weight", value=0.4, start=0.0, end=1.0, step=0.05,
        )
        self.advanced_accordion = pn.Card(
            self.motion_weight_slider,
            self.coverage_weight_slider,
            title="Advanced",
            collapsed=True,
            visible=False,
        )
        self.extract_frames_btn = pn.widgets.Button(name="Extract Frames", button_type="primary", width=160)
        self.frame_count_txt = pn.pane.HTML("")
        self.frame_slider = pn.widgets.IntSlider(
            name="Frame index", value=0, start=0, end=1, sizing_mode="stretch_width"
        )
        self.current_frame_pane = pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
        self.extractor_dd = pn.widgets.Select(
            name="Extractor", options=extractor_names, width=200
        )
        self.device_dd = pn.widgets.Select(
            name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100
        )
        self.extract_features_btn = pn.widgets.Button(
            name="Extract Features", button_type="primary", width=160
        )
        self.feature_overlay_pane = pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
        self.seg_strategy_dd = pn.widgets.Select(
            name="Strategy", options=["object", "auto"], width=200
        )
        self.seg_device_dd = pn.widgets.Select(
            name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100
        )
        self.seg_btn = pn.widgets.Button(
            name="Segment", button_type="primary", width=100
        )
        self.seg_output_pane = pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
        self.seg_count_txt = pn.pane.HTML("")
        self.hf_model_dd = pn.widgets.Select(
            name="Talk2DINO model",
            options=["lorebianchi98/Talk2DINOv3-ViTB", "lorebianchi98/Talk2DINO-ViTB"],
            width=300,
        )
        self.query_device_dd = pn.widgets.Select(
            name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100
        )
        self.text_pairs_input = pn.widgets.TextAreaInput(
            name="Text pairs (JSON)",
            value='{"object": [["object", "thing"], ["background", "empty"]]}',
            rows=5,
            width=700,
        )
        self.method_dd = pn.widgets.Select(name="Method", options=["standard", "pairwise"], width=150)
        self.temp_slider = pn.widgets.FloatSlider(
            name="Softmax temperature",
            value=0.05,
            start=0.001,
            end=0.1,
            step=0.001,
            width=350,
        )
        self.query_btn = pn.widgets.Button(name="Generate Heatmaps", button_type="primary", width=160)
        self.query_gallery = pn.GridBox(ncols=3)

        # Wire callbacks
        self.base_dir_input.param.watch(self._on_base_dir_change, "value")
        self.species_select.param.watch(self._on_species_change, "value")
        self.date_select.param.watch(self._on_date_change, "value")
        self.video_select.param.watch(self._on_video_change, "value")
        self.save_yaml_btn.on_click(self._save_yaml)
        self.reset_yaml_btn.on_click(self._reset_yaml)
        self.launch_btn.on_click(self._launch_training)
        self.stop_btn.on_click(self._stop_training)
        self.extract_frames_btn.on_click(self._extract_frames)
        self.frame_slider.param.watch(self._on_frame_slider_change, "value")
        self.sampling_mode_dd.param.watch(self._on_sampling_mode_change, "value")
        self.extract_features_btn.on_click(self._extract_features)
        self.seg_btn.on_click(self._run_segmentation)
        self.query_btn.on_click(self._run_semantic_query)

        self._training_badge = _mode_badge("Training Mode")
        self._explore_badge = _mode_badge("Explore Mode")

        self._refresh_species()

    def _datasets_dir(self) -> Path:
        d = Path(self._base_dir) / ".splats-configs" / "datasets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Watch chain
    # ------------------------------------------------------------------

    def _on_base_dir_change(self, event: Any) -> None:
        self._base_dir = self.base_dir_input.value
        self._refresh_species()

    def _refresh_species(self) -> None:
        try:
            self._video_tree = discover_videos(self._base_dir)
            species = sorted(self._video_tree.keys())
            self.species_select.options = species
            self.date_select.options = []
            self.video_select.options = []
            if species:
                # Force the watch chain even if value doesn't change.
                self.species_select.value = species[0]
                self._on_species_change(None)
            self._update_status(f"Found {len(species)} species in {self._base_dir}")
        except Exception as e:
            self._update_status(f"Error scanning {self._base_dir}: {e}", error=True)

    def _on_species_change(self, event: Any) -> None:
        species = self.species_select.value
        if not species:
            self.date_select.options = []
            self.video_select.options = []
            return
        dates = sorted(self._video_tree.get(species, {}).keys())
        self.date_select.options = dates
        self.video_select.options = []
        if dates:
            self.date_select.value = dates[0]
            self._on_date_change(None)

    def _on_date_change(self, event: Any) -> None:
        species = self.species_select.value
        date = self.date_select.value
        if not species or not date:
            self.video_select.options = []
            return
        videos = self._video_tree.get(species, {}).get(date, [])
        stems = [v.stem for v in videos]
        self.video_select.options = stems
        if stems:
            self.video_select.value = stems[0]

    def _on_video_change(self, event: Any) -> None:
        try:
            self._load_video()
        except Exception as e:
            self._update_status(f"Error loading video: {e}", error=True)

    def _load_video(self) -> None:
        species = self.species_select.value
        date = self.date_select.value
        video_stem = self.video_select.value
        if not all([species, date, video_stem]):
            return
        yaml_name = yaml_name_for_video(species, date, video_stem)
        yaml_path = self._datasets_dir() / f"{yaml_name}.yaml"
        self._config_panel.load_from_yaml(yaml_path if yaml_path.exists() else None)
        self._update_status(f"Loaded {'existing' if yaml_path.exists() else 'base'} config for {video_stem}")

    # ------------------------------------------------------------------
    # YAML save / reset
    # ------------------------------------------------------------------

    def _save_yaml(self, event: Any) -> None:
        try:
            species = self.species_select.value
            date = self.date_select.value
            video_stem = self.video_select.value
            if not all([species, date, video_stem]):
                self._update_status("Select a video before saving.", error=True)
                return
            yaml_name = yaml_name_for_video(species, date, video_stem)
            yaml_path = self._datasets_dir() / f"{yaml_name}.yaml"
            self._config_panel.save_to_yaml(yaml_path)
            self._update_status(f"Saved config to {yaml_path.name}")
        except Exception as e:
            self._update_status(f"Save failed: {e}", error=True)

    def _reset_yaml(self, event: Any) -> None:
        self._config_panel.load_from_yaml(None)
        self._update_status("Reset to base defaults")

    # ------------------------------------------------------------------
    # Training tab
    # ------------------------------------------------------------------

    def _resolve_video_path(self) -> Path | None:
        species = self.species_select.value
        date = self.date_select.value
        video_stem = self.video_select.value
        if not all([species, date, video_stem]):
            return None
        videos = self._video_tree.get(species, {}).get(date, [])
        for v in videos:
            if v.stem == video_stem:
                return v
        return None

    def _launch_training(self, event: Any) -> None:
        video_path = self._resolve_video_path()
        if video_path is None:
            self._update_status("Select a video before launching.", error=True)
            return
        if self._training_process and self._training_process.poll() is None:
            self._update_status("Training already running.", error=True)
            return

        output_path = Path(self.output_path_input.value)
        config = self._config_panel.to_splatter_config(video_path, output_path)

        import tempfile
        import yaml as _yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(dict(config), f)
            self._tmp_cfg_path = f.name

        self.training_log.value = f"Launching Splatter for {video_path.name}...\n"
        self.launch_btn.disabled = True
        self.stop_btn.disabled = False

        self._log_queue = queue.Queue()
        self._training_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"from collab_splats.wrapper.splatter import Splatter; "
                f"import yaml; cfg = yaml.safe_load(open('{self._tmp_cfg_path}')); "
                f"Splatter(cfg).run()",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(
            target=self._stdout_reader,
            args=(self._training_process.stdout,),
            daemon=True,
        ).start()
        self._poll_cb = pn.state.add_periodic_callback(self._poll_training_log, period=500)
        self._update_status(f"Training started (PID {self._training_process.pid})")

    def _stdout_reader(self, stream: Any) -> None:
        for line in stream:
            self._log_queue.put(line)

    def _poll_training_log(self) -> None:
        if self._training_process is None:
            return
        while True:
            try:
                self.training_log.value += self._log_queue.get_nowait()
            except queue.Empty:
                break
        if self._training_process.poll() is not None:
            # Drain any lines the reader thread finished after poll
            while True:
                try:
                    self.training_log.value += self._log_queue.get_nowait()
                except queue.Empty:
                    break
            rc = self._training_process.returncode
            if rc == 0:
                self._update_status("Training complete")
            else:
                self._update_status(f"Training failed (exit {rc})", error=True)
            if self._tmp_cfg_path:
                Path(self._tmp_cfg_path).unlink(missing_ok=True)
                self._tmp_cfg_path = None
            self._poll_cb.stop()
            self._poll_cb = None
            self.launch_btn.disabled = False
            self.stop_btn.disabled = True

    def _stop_training(self, event: Any) -> None:
        if self._training_process and self._training_process.poll() is None:
            self._training_process.terminate()
            try:
                self._training_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._training_process.kill()
            self._update_status("Training stopped by user")
        if self._poll_cb:
            self._poll_cb.stop()
            self._poll_cb = None
        if self._tmp_cfg_path:
            Path(self._tmp_cfg_path).unlink(missing_ok=True)
            self._tmp_cfg_path = None
        self.launch_btn.disabled = False
        self.stop_btn.disabled = True

    # ------------------------------------------------------------------
    # Explore tab
    # ------------------------------------------------------------------

    def _extract_frames(self, event: Any) -> None:
        video_path = self._resolve_video_path()
        if video_path is None:
            self._update_status("Select a video first.", error=True)
            return

        # Stop any in-flight extraction before starting a new one
        if self._extraction_cb is not None:
            self._extraction_cb.stop()
            self._extraction_cb = None

        # Reset state
        self._extraction_error = None
        with self._extraction_lock:
            self._extraction_progress = (0, 1)
            self._extraction_done = False

        self.progress_bar.value = 0
        self.progress_bar.visible = True
        self.progress_label.object = "<small>Starting…</small>"
        self.progress_label.visible = True
        self._update_status(f"Extracting frames from {video_path.name}…")

        def on_progress(current: int, total: int) -> None:
            with self._extraction_lock:
                self._extraction_progress = (current, max(1, total))

        sampling_mode = self.sampling_mode_dd.value
        min_disparity = self.min_disparity_slider.value
        motion_weight = self.motion_weight_slider.value
        coverage_weight = self.coverage_weight_slider.value
        fps = self.fps_slider.value
        max_frames = self.max_frames_slider.value

        def run_extraction() -> None:
            try:
                if sampling_mode == "Optical Flow":
                    result = sample_frames_optical_flow(
                        str(video_path),
                        min_disparity=min_disparity,
                        max_frames=max_frames,
                        motion_weight=motion_weight,
                        coverage_weight=coverage_weight,
                        on_progress=on_progress,
                    )
                else:
                    result = sample_frames_fps(
                        str(video_path),
                        fps,
                        on_progress=on_progress,
                        max_frames=max_frames,
                    )
                self._frames = result
            except Exception as e:
                self._frames = []
                self._extraction_error = str(e)
            finally:
                with self._extraction_lock:
                    self._extraction_done = True

        threading.Thread(target=run_extraction, daemon=True).start()
        self._extraction_cb = pn.state.add_periodic_callback(
            self._poll_extraction_progress, period=200
        )

    def _poll_extraction_progress(self) -> None:
        try:
            with self._extraction_lock:
                current, total = self._extraction_progress
                done = self._extraction_done

            pct = int(current / total * 100)
            self.progress_bar.value = min(pct, 100)
            self.progress_label.object = f"<small>{pct}%</small>"

            if not done:
                return

            # Extraction finished — tear down
            self.progress_bar.visible = False
            self.progress_label.visible = False
            if self._extraction_cb is not None:
                self._extraction_cb.stop()
                self._extraction_cb = None

            if self._extraction_error:
                self._update_status(f"Frame extraction failed: {self._extraction_error}", error=True)
                return

            n = len(self._frames)
            self.frame_slider.end = max(1, n - 1)
            self.frame_slider.value = 0
            self.frame_count_txt.object = f"<p>{n} frames extracted</p>"
            if self._frames:
                self._current_frame = self._frames[0]
                self.current_frame_pane.object = _to_png_bytes(self._frames[0])
            self._update_status(f"Extracted {n} frames")
        except Exception:
            # Ensure teardown even if polling raises
            self.progress_bar.visible = False
            self.progress_label.visible = False
            if self._extraction_cb is not None:
                self._extraction_cb.stop()
                self._extraction_cb = None

    def _on_frame_slider_change(self, event: Any) -> None:
        idx = int(self.frame_slider.value)
        if self._frames and idx < len(self._frames):
            self._current_frame = self._frames[idx]
            self.current_frame_pane.object = _to_png_bytes(self._frames[idx])

    def _extract_features(self, event: Any) -> None:
        if self._current_frame is None:
            self._update_status("Extract frames first.", error=True)
            return
        try:
            from collab_splats.semantics.features import BaseFeatureExtractor

            extractor_name = self.extractor_dd.value
            device = self.device_dd.value
            cls = BaseFeatureExtractor.get(extractor_name)
            extractor = cls(device=device)
            pil = Image.fromarray(self._current_frame)

            if extractor_name == "talk2dino":
                pil_pre = extractor.preprocess(pil)
                feats = extractor.forward(pil_pre)
                n = feats.shape[0]
                g = int(n**0.5)
                feat_np = feats.cpu().float().numpy().reshape(g, g, -1).transpose(2, 0, 1)
            elif extractor_name == "dinov2":
                tensor, H, W = extractor.preprocess(pil)
                feats = extractor.forward(tensor)
                feat_np = extractor.reshape(feats, H, W).numpy()
            else:
                tensor = extractor.preprocess(pil).unsqueeze(0)
                feat_np = extractor.forward(tensor)[0].cpu().float().numpy()

            pca_rgb = _pca_to_rgb(feat_np)
            overlay = _overlay_pca(self._current_frame, pca_rgb)
            self.feature_overlay_pane.object = _to_png_bytes(overlay)
            self._update_status("Feature extraction complete")
        except Exception as e:
            self._update_status(f"Feature extraction failed: {e}", error=True)

    def _run_segmentation(self, event: Any) -> None:
        if self._current_frame is None:
            self._update_status("Extract frames first.", error=True)
            return
        try:
            from collab_splats.semantics.segmentation import Segmentation

            seg = Segmentation(
                backend="mobilesamv2",
                strategy=self.seg_strategy_dd.value,
                device=self.seg_device_dd.value,
            )
            result = seg.segment(self._current_frame)
            if result is None:
                self.seg_output_pane.object = _to_png_bytes(self._current_frame)
                self.seg_count_txt.object = "<p>0 masks found</p>"
            else:
                masks, _ = result
                vis = _colorize_masks(self._current_frame, masks)
                self.seg_output_pane.object = _to_png_bytes(vis)
                self.seg_count_txt.object = f"<p>{len(masks)} masks found</p>"
            self._update_status("Segmentation complete")
        except Exception as e:
            self._update_status(f"Segmentation failed: {e}", error=True)

    def _run_semantic_query(self, event: Any) -> None:
        if self._current_frame is None:
            self._update_status("Extract frames first.", error=True)
            return
        try:
            text_pairs_raw = json.loads(self.text_pairs_input.value)
            text_pairs = {k: (list(v[0]), list(v[1])) for k, v in text_pairs_raw.items()}
        except json.JSONDecodeError as e:
            self._update_status(f"Invalid JSON: {e}", error=True)
            return
        try:
            from collab_splats.semantics.features import Talk2DinoExtractor

            extractor = Talk2DinoExtractor(hf_model_id=self.hf_model_dd.value, device=self.query_device_dd.value)
            heatmaps = extractor.compute_semantic_heatmap(
                Image.fromarray(self._current_frame),
                text_pairs,
                self.temp_slider.value,
                self.method_dd.value,
            )
            items = []
            for label, masked_img in heatmaps.items():
                img_u8 = (masked_img * 255).clip(0, 255).astype(np.uint8)
                items.append(
                    pn.Column(
                        pn.pane.PNG(_to_png_bytes(img_u8), width=220),
                        pn.pane.HTML(f"<p><b>{label}</b></p>"),
                    )
                )
            self.query_gallery.objects = items
            self._update_status(f"Generated {len(items)} heatmaps")
        except Exception as e:
            self._update_status(f"Semantic query failed: {e}", error=True)

    # ------------------------------------------------------------------
    # Loading overlay
    # ------------------------------------------------------------------

    def _show_loading(self, message: str = "Loading...") -> None:
        self.loading_modal.object = f"""
        <div style='position:fixed;top:0;left:0;width:100%;height:100%;
                    background:rgba(0,0,0,0.5);z-index:10000;
                    display:flex;align-items:center;justify-content:center;'>
          <div style='background:white;padding:40px;border-radius:16px;
                      box-shadow:0 12px 40px rgba(0,0,0,0.15);text-align:center;'>
            <p style='color:#555;font-size:16px;'>{message}</p>
          </div>
        </div>"""
        self.loading_modal.visible = True
        self._update_status(f"Loading: {message}")

    def _on_sampling_mode_change(self, event: Any) -> None:
        is_flow = event.new == "Optical Flow"
        self.min_disparity_slider.visible = is_flow
        self.advanced_accordion.visible = is_flow

    def _hide_loading(self) -> None:
        self.loading_modal.object = ""
        self.loading_modal.visible = False

    def _update_status(self, msg: str, error: bool = False) -> None:
        color = "red" if error else "inherit"
        self.status_pane.object = f"<p style='color:{color}'>{msg}</p>"

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def create_layout(self) -> pn.template.MaterialTemplate:
        sidebar = pn.Column(
            "## Video Browser",
            self.base_dir_input,
            self.species_select,
            self.date_select,
            self.video_select,
            pn.layout.Divider(),
            "## Config",
            self._config_panel.panel(),
            pn.Row(self.save_yaml_btn, self.reset_yaml_btn),
        )

        training_tab = pn.Column(
            self._training_badge,
            self.output_path_input,
            pn.Row(self.launch_btn, self.stop_btn),
            self.training_log,
        )

        frames_controls = pn.Column(
            self.progress_bar,
            self.progress_label,
            self.fps_slider,
            self.max_frames_slider,
            self.sampling_mode_dd,
            self.min_disparity_slider,
            self.advanced_accordion,
            self.extract_frames_btn,
            self.frame_count_txt,
            width=300,
        )

        features_controls = pn.Column(
            self.extractor_dd,
            self.device_dd,
            self.extract_features_btn,
            width=300,
        )

        seg_controls = pn.Column(
            self.seg_strategy_dd,
            self.seg_device_dd,
            self.seg_btn,
            self.seg_count_txt,
            width=300,
        )

        query_controls = pn.Column(
            self.hf_model_dd,
            self.query_device_dd,
            self.text_pairs_input,
            self.method_dd,
            self.temp_slider,
            self.query_btn,
            width=300,
        )

        explore_inner_tabs = pn.Tabs(
            ("① Frames", pn.Column(
                pn.Row(frames_controls, self.current_frame_pane),
                self.frame_slider,
            )),
            ("② Features", pn.Row(features_controls, self.feature_overlay_pane)),
            ("③ Segmentation", pn.Row(seg_controls, self.seg_output_pane)),
            ("④ Query", pn.Row(query_controls, self.query_gallery)),
            dynamic=True,
        )

        explore_tab = pn.Column(self._explore_badge, explore_inner_tabs)

        return pn.template.MaterialTemplate(
            title="collab-splats Semantic Explorer",
            sidebar=[sidebar],
            main=[
                self.loading_modal,
                self.status_pane,
                pn.Tabs(
                    ("Explore", explore_tab),
                    ("Training", training_tab),
                ),
            ],
            header_background="#2596be",
            sidebar_width=400,
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def build_app(base_dir: str = "/workspace/fieldwork-data") -> pn.template.MaterialTemplate:
    pn.extension(raw_css=[_TAB_CSS])
    dashboard = SemanticsDashboard(base_dir=base_dir)
    return dashboard.create_layout()


def run_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    base_dir: str = "/workspace/fieldwork-data",
) -> None:
    app = build_app(base_dir=base_dir)
    pn.serve(app, address=host, port=port, show=False)
