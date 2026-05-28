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
    """Return extractor names with a features.zarr under semantics/."""
    semantics_dir = dataset_dir / backend / "semantics"
    if not semantics_dir.is_dir():
        return []
    return sorted(
        p.name for p in semantics_dir.iterdir()
        if p.is_dir() and (p / "features.zarr").exists()
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
    """Self-contained per-scene 3D viewer panel.

    Manages dataset/backend selection, mode switching (PCD/Mesh/Similarity),
    and a PyVista plotter embedded via pn.pane.VTK.
    """

    # Observed by VisualizePane to toggle shared query bar
    mode = param.String(default="PCD")

    def __init__(
        self,
        scene_id: str,
        base_dir: Path,
        state: AppState,
        op_log: OperationLog,
        _off_screen: bool = False,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._scene_id = scene_id
        self._base_dir = Path(base_dir) if base_dir else Path("/workspace/outputs")
        self._state = state
        self._op_log = op_log

        # Internal data
        self._result: FeedforwardResult | None = None
        self._lifted_normed: np.ndarray | None = None
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

        # Mode buttons
        self._pcd_btn = pn.widgets.Button(
            name="PCD", button_type="primary", width=80, disabled=True
        )
        self._mesh_btn = pn.widgets.Button(
            name="Mesh", width=120, disabled=True
        )
        self._sim_btn = pn.widgets.Button(
            name="Similarity", width=140, disabled=True
        )

        # Viewer controls
        self._frustum_toggle = pn.widgets.Toggle(
            name="Show frustums", value=False, width=130
        )
        self._point_size_slider = pn.widgets.IntSlider(
            name="Point size", value=2, start=1, end=10, width=180
        )
        self._reset_btn = pn.widgets.Button(name="Reset camera", width=130)
        self._snapshot_btn = pn.widgets.Button(name="Snapshot", width=100)
        self._status_html = pn.pane.HTML("", width=400)

        # Wire callbacks
        self._dataset_dd.param.watch(self._on_dataset_change, "value")
        self._backend_dd.param.watch(self._on_backend_change, "value")
        self._load_btn.on_click(self._on_load)
        self._pcd_btn.on_click(lambda e: self._on_mode_change("PCD"))
        self._mesh_btn.on_click(lambda e: self._on_mode_change("Mesh"))
        self._sim_btn.on_click(lambda e: self._on_mode_change("Similarity"))
        self._frustum_toggle.param.watch(self._on_frustum_toggle, "value")
        self._point_size_slider.param.watch(self._on_point_size_change, "value")
        self._reset_btn.on_click(lambda e: self._plotter.reset_camera() or self._vtk_pane.synchronize())
        self._snapshot_btn.on_click(self._on_snapshot)

        # Scene A: watch AppState for auto-suggest and rescan
        if scene_id == "A":
            state.param.watch(self._on_feedforward_result, "feedforward_result")
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
        """Clear loaded result when backend changes."""
        self._result = None
        self._lifted_normed = None
        self._available_modes = set()
        self._update_mode_buttons()

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
        """Enable/disable mode buttons based on available modes."""
        has_pcd = "PCD" in self._available_modes
        has_mesh = "Mesh" in self._available_modes
        has_sim = "Similarity" in self._available_modes

        self._pcd_btn.disabled = not has_pcd
        self._mesh_btn.disabled = not has_mesh
        self._mesh_btn.name = "Mesh" if has_mesh else "Mesh (no mesh.ply)"
        self._sim_btn.disabled = not has_sim
        self._sim_btn.name = "Similarity" if has_sim else "Similarity (no features)"

        # Fall back to first available mode if current mode unavailable
        if self.mode not in self._available_modes and self._available_modes:
            first_available = next(iter(sorted(self._available_modes)))
            self._on_mode_change(first_available)

    ####################################################################
    # AppState watchers (Scene A only)
    ####################################################################

    def _on_feedforward_result(self, event: Any) -> None:
        """Auto-suggest dataset/backend from AppState (no auto-load)."""
        result = event.new
        if result is None or self._state.output_dir is None:
            return
        output_dir = Path(self._state.output_dir)
        ds_name = output_dir.name
        if ds_name in (self._dataset_dd.options or []):
            self._dataset_dd.value = ds_name

    def _on_lifted_features_path(self, event: Any) -> None:
        """Re-scan available modes when semantics pipeline writes new features."""
        self._lifted_normed = None
        self._scan_available_modes()

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
                self._on_mode_change("PCD")

            n_pts = len(self._result.points)
            if n_pts > 500_000:
                self._op_log.log(
                    f"Scene {self._scene_id}: large PCD ({n_pts:,} points) — may be slow to render"
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

    def _on_mode_change(self, new_mode: str) -> None:
        """Switch viewer mode; update button highlight."""
        if new_mode not in self._available_modes and self._result is not None:
            return
        self.mode = new_mode

        # Update button highlight to reflect active mode
        for btn, name in (
            (self._pcd_btn, "PCD"),
            (self._mesh_btn, "Mesh"),
            (self._sim_btn, "Similarity"),
        ):
            btn.button_type = "primary" if name == new_mode else "default"

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
        if self._frustum_toggle.value:
            self._add_frustums()

    def _add_frustums(self) -> None:
        """Add camera frustum actors for all extrinsics."""
        if self._result is None:
            return
        for ext in self._result.extrinsics:
            frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
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
        """Return currently selected extractor name."""
        return getattr(self, "_selected_extractor", None) or (
            self._available_extractors[0] if self._available_extractors else None
        )

    def _load_lifted_features_for_current_extractor(self) -> None:
        """Load and L2-normalise lifted features for the current extractor."""
        name = self._current_extractor_name()
        if not name or self._current_dataset_dir is None or self._current_backend is None:
            return
        feat_path = (
            self._current_dataset_dir / self._current_backend / "semantics" / name / "features.zarr"
        )
        if not feat_path.exists():
            self._set_status(f"features.zarr not found for {name}")
            return
        try:
            self._lifted_normed = _load_lifted_features(feat_path)
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

    def do_query(self, text: str, extractor_name: str) -> None:
        """Run similarity query; called from VisualizePane in a background thread."""
        self._selected_extractor = extractor_name

        # Lazy-load features on first query or after extractor change
        if self._lifted_normed is None:
            self._load_lifted_features_for_current_extractor()
        if self._lifted_normed is None:
            return

        # Lazy-instantiate extractor; cache after first use
        if extractor_name not in self._extractor_cache:
            try:
                # Lazy import: avoids pulling in heavy semantics chain at module load time
                from collab_splats.semantics.features.base import BaseQueryableExtractor  # noqa: PLC0415
                extractor_cls = BaseQueryableExtractor.create(extractor_name)
                self._extractor_cache[extractor_name] = extractor_cls()
            except Exception as exc:
                self._set_status(f"Extractor load failed: {extractor_name} — {exc}")
                return

        extractor = self._extractor_cache[extractor_name]
        try:
            query_tensor = extractor.encode_text([text])
            query_vec = query_tensor[0].detach().cpu().numpy()
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 1e-8:
                query_vec = query_vec / q_norm

            sims = self._lifted_normed @ query_vec
            colors = _apply_viridis(sims)
        except Exception as exc:
            self._set_status(f"Query failed: {exc}")
            return

        # Re-render with similarity colours and sync VTK pane
        self._plotter.clear()
        self._rebuild_sim_viewer(colors=colors)
        self._vtk_pane.synchronize()
        self._set_status(f'Query: "{text}" via {extractor_name}')

    ####################################################################
    # Snapshot
    ####################################################################

    def _on_snapshot(self, event: Any) -> None:
        """Save plotter screenshot to disk."""
        path = Path(f"scene_{self._scene_id}_snapshot.png")
        self._plotter.screenshot(str(path))
        self._set_status(f"Saved {path.name}")

    ####################################################################
    # Layout
    ####################################################################

    def panel(self) -> pn.Column:
        """Return the full scene panel layout."""
        controls_row = pn.Row(self._dataset_dd, self._backend_dd, self._load_btn)
        mode_row = pn.Row(
            self._pcd_btn, self._mesh_btn, self._sim_btn,
            self._frustum_toggle, self._point_size_slider,
        )
        action_row = pn.Row(self._reset_btn, self._snapshot_btn)
        return pn.Column(
            f"### Scene {self._scene_id}",
            controls_row,
            mode_row,
            self._vtk_pane,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )


########################################################################
# VisualizePane
########################################################################


class VisualizePane(param.Parameterized):
    """Compositor: two ScenePanels + shared semantic query bar."""

    def __init__(
        self,
        state: AppState,
        op_log: "OperationLog",
        base_dir: Path,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._state = state
        self._op_log = op_log

        self._scene_a = ScenePanel("A", base_dir, state, op_log)
        self._scene_b = ScenePanel("B", base_dir, state, op_log)

        # Shared query bar — visible when ≥1 scene is in Similarity mode
        self._query_input = pn.widgets.TextInput(
            placeholder="Enter text query…", width=300
        )
        self._a_extractor_dd = pn.widgets.Select(
            name="Scene A extractor", options=[], width=160
        )
        self._b_extractor_dd = pn.widgets.Select(
            name="Scene B extractor", options=[], width=160
        )
        self._query_btn = pn.widgets.Button(
            name="Query", button_type="success", width=90
        )
        self._query_bar = pn.Row(
            self._query_input,
            self._a_extractor_dd,
            self._b_extractor_dd,
            self._query_btn,
            visible=False,
        )

        # Watch mode changes on both scenes
        self._scene_a.param.watch(self._on_scene_mode_change, "mode")
        self._scene_b.param.watch(self._on_scene_mode_change, "mode")
        self._scene_a.param.watch(self._update_extractor_dropdowns, "mode")
        self._scene_b.param.watch(self._update_extractor_dropdowns, "mode")

        self._query_btn.on_click(self._on_query_click)

    ####################################################################
    # Query bar visibility
    ####################################################################

    def _on_scene_mode_change(self, event: Any) -> None:
        """Show query bar when ≥1 scene is in Similarity mode."""
        a_sim = self._scene_a.mode == "Similarity"
        b_sim = self._scene_b.mode == "Similarity"
        self._query_bar.visible = a_sim or b_sim
        self._a_extractor_dd.visible = a_sim
        self._b_extractor_dd.visible = b_sim
        self._update_extractor_dropdowns(None)

    def _update_extractor_dropdowns(self, event: Any) -> None:
        """Sync extractor dropdown options from each scene's available extractors."""
        self._a_extractor_dd.options = self._scene_a._available_extractors or []
        self._b_extractor_dd.options = self._scene_b._available_extractors or []

    ####################################################################
    # Query dispatch
    ####################################################################

    def _on_query_click(self, event: Any) -> None:
        """Fire similarity query on all active Similarity scenes in parallel threads."""
        text = self._query_input.value
        if not text:
            return
        targets = []
        if self._scene_a.mode == "Similarity" and self._a_extractor_dd.value:
            targets.append((self._scene_a, self._a_extractor_dd.value))
        if self._scene_b.mode == "Similarity" and self._b_extractor_dd.value:
            targets.append((self._scene_b, self._b_extractor_dd.value))
        for scene, extractor in targets:
            t = threading.Thread(
                target=scene.do_query, args=(text, extractor), daemon=True
            )
            t.start()

    ####################################################################
    # Tab activation rescan
    ####################################################################

    def _rescan_both_scenes(self) -> None:
        """Re-scan available modes on both scenes (called on tab activation)."""
        self._scene_a.rescan()
        self._scene_b.rescan()
        self._update_extractor_dropdowns(None)

    def wire_tabs(self, tabs: pn.Tabs, tab_index: int) -> None:
        """Connect tab activation signal so modes rescan when this tab becomes active."""
        def _on_tab_change(event: Any) -> None:
            if event.new == tab_index:
                self._rescan_both_scenes()
        tabs.param.watch(_on_tab_change, "active")

    ####################################################################
    # Layout
    ####################################################################

    def panel(self) -> pn.Column:
        """Return the full VisualizePane layout."""
        scenes_row = pn.Row(
            self._scene_a.panel(),
            self._scene_b.panel(),
            sizing_mode="stretch_both",
        )
        return pn.Column(
            self._query_bar,
            scenes_row,
            sizing_mode="stretch_both",
        )
