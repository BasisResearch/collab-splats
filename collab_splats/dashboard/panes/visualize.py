from __future__ import annotations

########################################################################
# Imports
########################################################################

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import panel as pn
import param
import pyvista as pv
import zarr

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.utils.visualization import (
    create_camera_frustum_pyvista,
    pointcloud_to_polydata,
)

if TYPE_CHECKING:
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.semantics.features.base import BaseQueryableExtractor

logger = logging.getLogger(__name__)

########################################################################
# Discovery helpers
########################################################################


def _scan_datasets(base_dir: Path) -> list[Path]:
    """Return subdirectories of base_dir that contain run_config.yaml."""
    if not base_dir or not base_dir.is_dir():
        return []
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and (p / "run_config.yaml").exists()
    )


def _scan_backends(dataset_dir: Path) -> list[str]:
    """Return backend names (subdirs containing feedforward.zarr)."""
    if not dataset_dir or not dataset_dir.is_dir():
        return []
    return sorted(
        p.name for p in dataset_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    )


def _scan_extractors(dataset_dir: Path, backend: str) -> list[str]:
    """Return queryable extractor names with a features.zarr under semantics/."""
    semantics_dir = dataset_dir / backend / "semantics"
    if not semantics_dir.is_dir():
        return []
    from collab_splats.semantics.features.base import BaseQueryableExtractor  # noqa: PLC0415
    queryable = set(BaseQueryableExtractor._registry.keys())
    return sorted(
        p.name for p in semantics_dir.iterdir()
        if p.is_dir() and (p / "features.zarr").exists() and p.name in queryable
    )


########################################################################
# Viridis colormap helper
########################################################################


def _apply_viridis(sims: np.ndarray) -> np.ndarray:
    """Map similarity scores (P,) to viridis RGB uint8 (P, 3)."""
    s_min, s_max = sims.min(), sims.max()
    if s_max > s_min:
        normalized = (sims - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(sims)
    rgba = cm.viridis(normalized)
    return (rgba[:, :3] * 255).astype(np.uint8)


########################################################################
# Lifted features helper
########################################################################


def _load_lifted_features(features_zarr_path: Path) -> np.ndarray:
    """Load and L2-normalise per-point features from zarr. Returns (P, D) float32."""
    store = zarr.open(str(features_zarr_path), mode="r")
    feats = store["features"][:]
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-8)


########################################################################
# ScenePanel
########################################################################


class ScenePanel(param.Parameterized):
    """3D viewer panel for a single reconstruction scene.

    Manages dataset/backend selection, mode switching (PCD/Mesh/Similarity),
    and a PyVista plotter embedded via pn.pane.VTK.
    """

    mode = param.String(default="Mesh")

    def __init__(
        self,
        base_dir: Path,
        state: AppState,
        op_log: OperationLog,
        _off_screen: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir) if base_dir else Path("/workspace/outputs")
        self._state = state
        self._op_log = op_log

        # Internal data
        self._result: FeedforwardResult | None = None
        self._lifted_normed: np.ndarray | None = None
        self._compressor: Any | None = None
        self._extractor_cache: dict[str, Any] = {}
        self._available_modes: set[str] = set()
        self._available_extractors: list[str] = []
        self._load_thread: threading.Thread | None = None
        self._current_dataset_dir: Path | None = None
        self._current_backend: str | None = None

        # PyVista plotter — one per scene, never recreated
        self._plotter = pv.Plotter(off_screen=_off_screen)
        self._vtk_pane = pn.pane.VTK(
            self._plotter.ren_win,
            sizing_mode="stretch_both",
            min_height=500,
        )

        # Dataset / backend discovery
        datasets = _scan_datasets(self._base_dir)
        dataset_names = [p.name for p in datasets]
        self._dataset_dd = pn.widgets.Select(
            name="Dataset", options=dataset_names, width=200
        )
        self._backend_dd = pn.widgets.Select(
            name="Backend", options=[], width=150
        )
        self._load_btn = pn.widgets.Button(
            name="Load", button_type="primary", width=80
        )

        # Mode selector — RadioButtonGroup replaces three separate buttons
        self._mode_selector = pn.widgets.RadioButtonGroup(
            options=["Points", "Mesh", "Similarity"],
            value="Mesh",
            button_type="success",
            width=380,
            disabled=True,
        )

        # Viewer controls — frustum checkbox + point size inside a contextual row
        self._frustum_check = pn.widgets.Checkbox(name="Show frustums", value=False)
        self._point_size_slider = pn.widgets.IntSlider(
            name="Point size", value=2, start=1, end=10, width=180
        )
        self._points_options_row = pn.Row(
            self._frustum_check,
            self._point_size_slider,
            visible=False,
        )

        # Extractor selector — always visible, disabled until extractors are found
        self._extractor_dd = pn.widgets.Select(
            name="Extractor", options=[], width=180, disabled=True
        )

        # Per-scene similarity query row (positive + optional negative)
        self._sim_pos_input = pn.widgets.TextInput(
            placeholder="Enter positive query…", width=200
        )
        self._sim_neg_input = pn.widgets.TextInput(
            placeholder="background, wall, floor…",
            width=200,
            styles={"color": "#888"},
        )
        self._sim_query_btn = pn.widgets.Button(
            name="Query", button_type="success", width=80
        )
        self._sim_query_row = pn.Row(
            pn.pane.HTML("<b style='color:#6f6'>+</b>"),
            self._sim_pos_input,
            pn.pane.HTML("<b style='color:#f66'>−</b>"),
            self._sim_neg_input,
            self._sim_query_btn,
            visible=False,
        )
        self._reset_btn = pn.widgets.Button(name="Reset camera", width=130)
        self._snapshot_btn = pn.widgets.Button(name="Snapshot", width=100)
        self._status_html = pn.pane.HTML("", width=400)

        # Wire callbacks
        self._dataset_dd.param.watch(self._on_dataset_change, "value")
        self._backend_dd.param.watch(self._on_backend_change, "value")
        self._extractor_dd.param.watch(self._on_extractor_change, "value")
        self._load_btn.on_click(self._on_load)
        self._mode_selector.param.watch(
            lambda e: self._on_mode_change(e.new), "value"
        )
        self._frustum_check.param.watch(self._on_frustum_toggle, "value")
        self._point_size_slider.param.watch(self._on_point_size_change, "value")
        self._reset_btn.on_click(lambda e: self._plotter.reset_camera() or self._vtk_pane.synchronize())
        self._snapshot_btn.on_click(self._on_snapshot)
        self._sim_query_btn.on_click(self._on_sim_query_click)

        # Watch AppState for auto-suggest and rescan
        state.param.watch(self._on_feedforward_result, "feedforward_result")
        state.param.watch(self._on_output_dir_change, "output_dir")
        state.param.watch(self._on_lifted_features_path, "lifted_features_path")

        # Populate backend dropdown for initial dataset selection
        if dataset_names:
            self._on_dataset_change(None)

    ####################################################################
    # Discovery
    ####################################################################

    def _on_dataset_change(self, event: Any) -> None:
        """Repopulate backend dropdown when dataset changes."""
        name = self._dataset_dd.value
        if not name:
            self._backend_dd.options = []
            return
        ds_dir = self._base_dir / name
        backends = _scan_backends(ds_dir)
        self._backend_dd.options = backends
        if backends:
            self._backend_dd.value = backends[0]

    def _on_backend_change(self, event: Any) -> None:
        """Clear loaded result when backend changes; repopulate extractor dropdown."""
        self._result = None
        self._lifted_normed = None
        self._compressor = None
        self._available_modes = set()
        self._update_mode_buttons()

        # Repopulate extractor dropdown for the new backend
        ds_name = self._dataset_dd.value
        backend = self._backend_dd.value
        if ds_name and backend:
            ds_dir = self._base_dir / ds_name
            extractors = _scan_extractors(ds_dir, backend)
            self._extractor_dd.options = extractors
            self._extractor_dd.value = extractors[0] if extractors else None
            self._extractor_dd.disabled = not bool(extractors)
        else:
            self._extractor_dd.options = []
            self._extractor_dd.disabled = True

    def _on_extractor_change(self, event: Any) -> None:
        """Invalidate feature cache when extractor selection changes."""
        self._lifted_normed = None
        self._compressor = None
        if self.mode == "Similarity":
            threading.Thread(
                target=self._load_lifted_features_for_current_extractor, daemon=True
            ).start()

    def _scan_available_modes(self) -> None:
        """Check disk for available modes; update button states."""
        ds_name = self._dataset_dd.value
        backend = self._backend_dd.value
        if not ds_name or not backend:
            self._available_modes = set()
            self._available_extractors = []
            self._update_mode_buttons()
            return

        ds_dir = self._base_dir / ds_name
        modes: set[str] = set()

        # PCD mode requires a loaded result
        if self._result is not None:
            modes.add("PCD")

        # Mesh mode requires mesh.ply on disk
        mesh_path = ds_dir / backend / "mesh" / "mesh.ply"
        if mesh_path.exists():
            modes.add("Mesh")

        # Similarity mode requires at least one extractor with features.zarr
        extractors = _scan_extractors(ds_dir, backend)
        if extractors:
            modes.add("Similarity")
        self._available_extractors = extractors

        self._available_modes = modes
        self._current_dataset_dir = ds_dir
        self._current_backend = backend
        self._update_mode_buttons()

    def _update_mode_buttons(self) -> None:
        """Enable/disable the RadioButtonGroup based on available modes."""
        # Disable selector when no modes are available; enable once data is loaded
        self._mode_selector.disabled = not bool(self._available_modes)

        # Map internal mode names to selector labels
        label_map = {"PCD": "Points", "Mesh": "Mesh", "Similarity": "Similarity"}

        # Fall back to first available mode if current mode unavailable
        if self.mode not in self._available_modes and self._available_modes:
            first_available = next(iter(sorted(self._available_modes)))
            self._on_mode_change(first_available)
        elif self.mode in self._available_modes:
            # Sync selector value to current mode without re-triggering render
            label = label_map.get(self.mode, self.mode)
            if self._mode_selector.value != label:
                self._mode_selector.value = label

    ####################################################################
    # AppState watchers
    ####################################################################

    def _on_feedforward_result(self, event: Any) -> None:
        """Auto-suggest dataset/backend from AppState (no auto-load)."""
        result = event.new
        if result is None or self._state.output_dir is None:
            return
        self._suggest_dataset_from_output_dir()

    def _on_output_dir_change(self, event: Any) -> None:
        """Auto-suggest dataset when an existing session is loaded (no active reconstruction)."""
        if self._state.output_dir is None:
            return
        self._suggest_dataset_from_output_dir()

    def _suggest_dataset_from_output_dir(self) -> None:
        """Select dataset dropdown entry matching current output_dir name if present."""
        ds_name = Path(self._state.output_dir).name
        if ds_name in (self._dataset_dd.options or []):
            self._dataset_dd.value = ds_name

    def _on_lifted_features_path(self, event: Any) -> None:
        """Re-scan available modes when semantics pipeline writes new features."""
        self._lifted_normed = None
        self._scan_available_modes()

    def wire_tabs(self, tabs: pn.Tabs, tab_index: int) -> None:
        """Connect tab activation signal so modes rescan when this tab becomes active."""
        def _on_tab_change(event: Any) -> None:
            if event.new == tab_index:
                self.rescan()
        tabs.param.watch(_on_tab_change, "active")

    def rescan(self) -> None:
        """Re-scan available modes (called on tab activation)."""
        self._scan_available_modes()

    ####################################################################
    # Load
    ####################################################################

    def _on_load(self, event: Any) -> None:
        """Start background load thread."""
        if self._load_thread and self._load_thread.is_alive():
            return
        self._load_btn.disabled = True
        self._status_html.object = "<em>Loading…</em>"
        self._load_thread = threading.Thread(target=self._do_load, daemon=True)
        self._load_thread.start()

    def _do_load(self) -> None:
        """Background: load FeedforwardResult from zarr."""
        try:
            # Lazy import: avoids pulling in the heavy pointcloud chain at module load time
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult  # noqa: PLC0415

            ds_name = self._dataset_dd.value
            backend = self._backend_dd.value
            if not ds_name or not backend:
                self._set_status("No dataset/backend selected.")
                return

            zarr_path = self._base_dir / ds_name / backend / "feedforward.zarr"
            self._result = FeedforwardResult.load_zarr(zarr_path)
            self._lifted_normed = None
            self._scan_available_modes()

            self._plotter.reset_camera()
            if "PCD" in self._available_modes:
                self._on_mode_change("Points")

            n_pts = len(self._result.points)
            if n_pts > 500_000:
                self._op_log.log(
                    f"Large PCD ({n_pts:,} points) — may be slow to render"
                )
            self._set_status(f"Loaded {n_pts:,} points.")
        except Exception as exc:
            logger.exception("ScenePanel load failed")
            self._set_status(f"Load failed: {exc}")
        finally:
            self._load_btn.disabled = False

    def _set_status(self, msg: str) -> None:
        """Update status HTML pane."""
        self._status_html.object = f"<small>{msg}</small>"

    ####################################################################
    # Mode switching
    ####################################################################

    def _on_mode_change(self, new_display_mode: str) -> None:
        """Switch viewer mode; update contextual controls visibility."""
        # Map selector labels ("Points") to internal mode names ("PCD")
        mode_map = {"Points": "PCD", "Mesh": "Mesh", "Similarity": "Similarity"}
        new_mode = mode_map.get(new_display_mode, new_display_mode)

        if new_mode not in self._available_modes and self._result is not None:
            return
        self.mode = new_mode

        # Show/hide contextual rows based on active mode
        self._points_options_row.visible = (new_mode == "PCD")
        self._sim_query_row.visible = (new_mode == "Similarity")

        self._plotter.clear()
        if new_mode == "PCD":
            self._rebuild_pcd_viewer()
        elif new_mode == "Mesh":
            self._rebuild_mesh_viewer()
        elif new_mode == "Similarity":
            if self._lifted_normed is None:
                self._load_lifted_features_for_current_extractor()
            self._rebuild_sim_viewer(colors=None)

        self._vtk_pane.synchronize()

    ####################################################################
    # PCD viewer
    ####################################################################

    def _rebuild_pcd_viewer(self) -> None:
        """Render points + optional frustums."""
        if self._result is None:
            return
        point_size = self._point_size_slider.value
        cloud = pointcloud_to_polydata(self._result.points, RGB=self._result.colors)
        self._plotter.add_mesh(
            cloud, scalars="RGB", rgb=True, point_size=point_size, render_points_as_spheres=False
        )
        if self._frustum_check.value:
            self._add_frustums()

    def _add_frustums(self) -> None:
        """Add camera frustum actors for all extrinsics."""
        if self._result is None:
            return
        for ext in self._result.extrinsics:
            frustum = create_camera_frustum_pyvista(ext)
            self._plotter.add_mesh(frustum, color="cornflowerblue", line_width=1)

    def _on_frustum_toggle(self, event: Any) -> None:
        """Rebuild PCD view when frustum toggle changes."""
        if self.mode != "PCD" or self._result is None:
            return
        self._plotter.clear()
        self._rebuild_pcd_viewer()
        self._vtk_pane.synchronize()

    def _on_point_size_change(self, event: Any) -> None:
        """Rebuild PCD view when point size slider changes."""
        if self.mode != "PCD" or self._result is None:
            return
        self._plotter.clear()
        self._rebuild_pcd_viewer()
        self._vtk_pane.synchronize()

    ####################################################################
    # Mesh viewer
    ####################################################################

    def _rebuild_mesh_viewer(self) -> None:
        """Load and render mesh.ply."""
        if self._current_dataset_dir is None or self._current_backend is None:
            return
        mesh_path = self._current_dataset_dir / self._current_backend / "mesh" / "mesh.ply"
        if not mesh_path.exists():
            self._set_status("mesh.ply not found.")
            return
        mesh = pv.read(str(mesh_path))
        self._plotter.add_mesh(mesh, rgb=True)

    ####################################################################
    # Similarity viewer
    ####################################################################

    def _current_extractor_name(self) -> str | None:
        """Return the currently selected extractor name from the per-scene dropdown."""
        if self._extractor_dd.options and self._extractor_dd.value:
            return self._extractor_dd.value
        if self._available_extractors:
            return self._available_extractors[0]
        return None

    def _load_lifted_features_for_current_extractor(self) -> None:
        """Load and L2-normalise lifted features + compressor for the current extractor."""
        name = self._current_extractor_name()
        if not name or self._current_dataset_dir is None or self._current_backend is None:
            return
        feat_dir = self._current_dataset_dir / self._current_backend / "semantics" / name
        feat_path = feat_dir / "features.zarr"
        if not feat_path.exists():
            self._set_status(f"features.zarr not found for {name}")
            return
        try:
            self._lifted_normed = _load_lifted_features(feat_path)
            # Load sibling compressor so text queries can be projected into latent space
            compressor_dir = feat_dir / "compressor.pt"
            if compressor_dir.is_dir():
                from collab_splats.semantics.compression import FeatureAutoencoder  # noqa: PLC0415
                self._compressor = FeatureAutoencoder.load(compressor_dir)
                self._compressor.eval()
            else:
                self._compressor = None
        except Exception as exc:
            self._set_status(f"Feature load failed: {exc}")

    def _rebuild_sim_viewer(self, colors: np.ndarray | None) -> None:
        """Render PCD with viridis similarity colours, or original RGB before first query."""
        if self._result is None:
            return
        if colors is None:
            rgb = self._result.colors
            self._set_status("Enter a query to colour by similarity.")
        else:
            rgb = colors
        cloud = pointcloud_to_polydata(self._result.points, RGB=rgb)
        self._plotter.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)

    def _encode_query(self, extractor: Any, text: str) -> np.ndarray:
        """Encode text to a unit vector in the stored feature space.

        If a compressor is loaded, projects the raw text embedding through
        the AE encoder so dims match the stored 64-dim lifted features.
        """
        import torch
        import torch.nn.functional as F

        raw = extractor.encode_text([text])[0].detach().cpu()  # (D,)
        if self._compressor is not None:
            with torch.no_grad():
                latent = self._compressor.per_point_encode(raw.unsqueeze(0))  # (1, latent_dim)
                raw = F.normalize(latent, dim=-1).squeeze(0)                  # (latent_dim,)
        vec = raw.numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec

    def do_query(self, pos_text: str, neg_text: str, extractor_name: str) -> None:
        """Run similarity query; called from _on_sim_query_click in a background thread."""
        # Lazy-load features on first query or after extractor change
        if self._lifted_normed is None:
            self._load_lifted_features_for_current_extractor()
        if self._lifted_normed is None:
            return

        # Lazy-instantiate extractor; cache after first use
        if extractor_name not in self._extractor_cache:
            try:
                from collab_splats.semantics.features.base import BaseQueryableExtractor  # noqa: PLC0415
                extractor_cls = BaseQueryableExtractor.get(extractor_name)
                self._extractor_cache[extractor_name] = extractor_cls()
            except Exception as exc:
                self._set_status(f"Extractor load failed: {extractor_name} — {exc}")
                return

        extractor = self._extractor_cache[extractor_name]
        try:
            pos_vec = self._encode_query(extractor, pos_text)
            sims = self._lifted_normed @ pos_vec

            if neg_text:
                neg_vec = self._encode_query(extractor, neg_text)
                sims = sims - self._lifted_normed @ neg_vec

            colors = _apply_viridis(sims)
        except Exception as exc:
            self._set_status(f"Query failed: {exc}")
            return

        # Re-render with similarity colours and sync VTK pane
        self._plotter.clear()
        self._rebuild_sim_viewer(colors=colors)
        self._vtk_pane.synchronize()
        label = f'"{pos_text}"'
        if neg_text:
            label += f' − "{neg_text}"'
        self._set_status(f"Query: {label} via {extractor_name}")

    def _on_sim_query_click(self, event: Any) -> None:
        """Fire similarity query from this scene's per-scene query inputs."""
        pos_text = self._sim_pos_input.value.strip()
        neg_text = self._sim_neg_input.value.strip()
        extractor_name = self._extractor_dd.value
        if not pos_text or not extractor_name:
            return
        threading.Thread(
            target=self.do_query, args=(pos_text, neg_text, extractor_name), daemon=True
        ).start()

    ####################################################################
    # Snapshot
    ####################################################################

    def _on_snapshot(self, event: Any) -> None:
        """Save plotter screenshot to disk."""
        path = Path("scene_snapshot.png")
        self._plotter.screenshot(str(path))
        self._set_status(f"Saved {path.name}")

    ####################################################################
    # Layout
    ####################################################################

    def panel(self) -> pn.Column:
        """Return the full scene panel layout."""
        controls_row = pn.Row(self._dataset_dd, self._backend_dd, self._load_btn, align="end")
        extractor_row = pn.Row(self._extractor_dd)
        action_row = pn.Row(self._reset_btn, self._snapshot_btn)
        return pn.Column(
            "### Scene",
            controls_row,
            extractor_row,
            self._mode_selector,
            self._vtk_pane,
            self._points_options_row,
            self._sim_query_row,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )
