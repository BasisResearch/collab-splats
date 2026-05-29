from __future__ import annotations

########################################################################
# Imports
########################################################################

import logging
import multiprocessing
import queue as _queue_mod
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
# Mesh subprocess worker
########################################################################


def _mesh_subprocess_worker(
    zarr_path: str,
    mesh_dir: str,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clean_repair: bool,
    progress_queue: multiprocessing.Queue,
) -> None:
    """Run mesh generation in an isolated subprocess.

    Loaded as a separate process so OOM kills only this process,
    not the dashboard server.
    """
    # Patch tqdm BEFORE importing mesh modules.  Each module does
    # `from tqdm.auto import tqdm` at import time, so patching the module
    # attribute here means those `from … import` bindings pick up the custom
    # class automatically.
    import tqdm as _tqdm_mod
    import tqdm.auto as _tqdm_auto
    _q = progress_queue
    _orig_tqdm = _tqdm_mod.tqdm

    class _QueueTqdm(_orig_tqdm):
        def update(self, n: int = 1) -> None:
            super().update(n)
            total = self.total or 0
            # Throttle: emit every ~1 % of total or on completion.
            if total > 0 and (self.n % max(1, total // 100) == 0 or self.n >= total):
                try:
                    _q.put_nowait({
                        "desc": self.desc or "",
                        "n": int(self.n),
                        "total": int(total),
                    })
                except Exception:
                    pass

    _tqdm_mod.tqdm = _QueueTqdm
    _tqdm_auto.tqdm = _QueueTqdm
    _tqdm_mod.trange = lambda *a, **kw: _QueueTqdm(range(*a), **kw)
    _tqdm_auto.trange = _tqdm_mod.trange

    import shutil as _shutil  # noqa: PLC0415
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult  # noqa: PLC0415
    from collab_splats.mesh.utils import pointcloud_to_mesh  # noqa: PLC0415

    result = FeedforwardResult.load_zarr(zarr_path)
    mesh_result = pointcloud_to_mesh(
        result,
        mesh_dir,
        method="open3d_tsdf",
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        clean_repair=clean_repair,
    )
    # Copy to canonical mesh.ply so the dashboard's scan/render can find it.
    canonical = Path(mesh_dir) / "mesh.ply"
    if mesh_result.mesh_path.resolve() != canonical.resolve() and mesh_result.mesh_path.exists():
        _shutil.copy2(str(mesh_result.mesh_path), str(canonical))

    # Persist params so the dashboard can detect stale caches.
    import json as _json  # noqa: PLC0415
    params_path = Path(mesh_dir) / "mesh_params.json"
    params_path.write_text(_json.dumps({
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "clean_repair": clean_repair,
    }))


########################################################################
# ScenePanel
########################################################################


class ScenePanel(param.Parameterized):
    """3D viewer panel for a single reconstruction scene.

    Receives data via AppState.feedforward_result; mode switching (Points/Mesh/Similarity)
    and a PyVista plotter embedded via pn.pane.VTK.
    """

    mode = param.String(default="PCD")

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
        self._current_dataset_dir: Path | None = None
        self._current_backend: str | None = None

        # Actor cache and display-decimated result
        self._pcd_actor = None
        self._mesh_actor = None
        self._sim_actor = None
        self._sim_cloud = None
        self._display_result = None
        self._state_frustum_enabled = False

        # PyVista plotter — one per scene, never recreated
        self._plotter = pv.Plotter(off_screen=_off_screen)
        self._vtk_pane = pn.pane.VTK(
            self._plotter.ren_win,
            sizing_mode="stretch_both",
            min_height=500,
        )

        # Dataset / backend discovery (kept for _scan_available_modes and mesh worker)
        datasets = _scan_datasets(self._base_dir)
        dataset_names = [p.name for p in datasets]
        self._dataset_dd = pn.widgets.Select(
            name="Dataset", options=dataset_names, width=200
        )
        self._backend_dd = pn.widgets.Select(
            name="Backend", options=[], width=150
        )

        # Mode selector — RadioButtonGroup replaces three separate buttons
        self._mode_selector = pn.widgets.RadioButtonGroup(
            options=["Points", "Mesh", "Similarity"],
            value="Points",
            button_type="success",
            width=380,
            disabled=True,
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
        # Mesh generation params — set via run_mesh() from sidebar
        self._mesh_voxel: float = 0.01
        self._mesh_sdf: float = 0.04
        self._mesh_depth: float = 10.0
        self._mesh_clean: bool = True
        self._mesh_on_done: Any = None
        self._mesh_on_progress: Any = None
        self._mesh_thread: threading.Thread | None = None
        self._mesh_proc: multiprocessing.Process | None = None
        self._reset_btn = pn.widgets.Button(name="Reset camera", width=130)
        self._snapshot_btn = pn.widgets.Button(name="Snapshot", width=100)
        self._status_html = pn.pane.HTML("", width=400)

        # Wire callbacks
        self._dataset_dd.param.watch(self._on_dataset_change, "value")
        self._backend_dd.param.watch(self._on_backend_change, "value")
        self._extractor_dd.param.watch(self._on_extractor_change, "value")
        self._mode_selector.param.watch(
            lambda e: self._on_mode_change(e.new), "value"
        )
        self._reset_btn.on_click(self._on_reset_camera)
        self._snapshot_btn.on_click(self._on_snapshot)
        self._sim_query_btn.on_click(self._on_sim_query_click)

        # Watch AppState for result, output_dir, lifted features
        state.param.watch(self._on_feedforward_result_change, "feedforward_result")
        state.param.watch(self._on_output_dir_change, "output_dir")
        state.param.watch(self._on_lifted_features_path, "lifted_features_path")
        state.param.watch(self._on_ground_plane_change, ["ground_plane_enabled", "ground_plane_R"])

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
            preferred = self._state.pointcloud_backend
            self._backend_dd.value = preferred if preferred in backends else backends[0]

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
            if extractors:
                preferred = self._state.semantic_extractor
                self._extractor_dd.value = preferred if preferred in extractors else extractors[0]
            else:
                self._extractor_dd.value = None
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
        """Enable/disable the RadioButtonGroup based on available modes.

        Only updates widget state — never triggers VTK rendering directly.
        Rendering happens via user interaction (mode selector click) on the IOLoop thread.
        """
        self._mode_selector.disabled = not bool(self._available_modes)
        # Update internal mode to first available if current mode is gone
        if self.mode not in self._available_modes and self._available_modes:
            self.mode = next(iter(sorted(self._available_modes)))

    ####################################################################
    # AppState watchers
    ####################################################################

    def _on_feedforward_result_change(self, event: Any) -> None:
        """Called when state.feedforward_result is set — prepare display result."""
        result = event.new
        if result is None:
            return
        self._result = result
        self._display_result = self._prepare_display_result(result)
        self._pcd_actor = None
        self._mesh_actor = None
        self._sim_actor = None
        self._sim_cloud = None
        self._lifted_normed = None
        self._compressor = None
        self._current_dataset_dir = Path(str(self._state.output_dir)) if self._state.output_dir else None
        self._current_backend = self._state.pointcloud_backend or ""
        # Sync dataset/backend dropdowns from state
        if self._state.output_dir is not None:
            self._suggest_dataset_from_output_dir()
        self._scan_available_modes()
        n_pts = len(result.points)

        # Wrap render on the IOLoop thread — feedforward_result may be set from a
        # background thread (e.g. ReconstructPane), and synchronize() is a no-op
        # unless we're on the Tornado IOLoop.
        def _render() -> None:
            self._plotter.clear()
            start_mode = "Mesh" if "Mesh" in self._available_modes else "Points"
            self._on_mode_change(start_mode)
            self._mode_selector.value = start_mode
            self._vtk_pane.reset_camera()
            self._set_status(f"Loaded {n_pts:,} pts")

        pn.io.state.execute(_render)

    def _prepare_display_result(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Return result unchanged — all points rendered for full fidelity."""
        return result

    def _apply_ground_plane(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Apply or invert ground plane transform (R, t) to points and extrinsics.

        When state.ground_plane_enabled=True: applies R and t.
        When False: applies inverse (R.T, -R.T @ t).
        Returns new FeedforwardResult via dataclasses.replace — does not mutate input.
        """
        import dataclasses
        R = self._state.ground_plane_R
        t = self._state.ground_plane_t
        if R is None or t is None:
            return result
        if self._state.ground_plane_enabled:
            R_use, t_use = R, t
        else:
            R_use = R.T
            t_use = -(R.T @ t)
        pts_new = (R_use @ result.points.T).T + t_use
        # Transform extrinsics (N,4,4) world-to-camera: E_new = E @ T_use_inv
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R_use.T
        T_inv[:3, 3] = -(R_use.T @ t_use)
        extrinsics_new = result.extrinsics.astype(np.float64) @ T_inv
        return dataclasses.replace(
            result,
            points=pts_new.astype(result.points.dtype),
            extrinsics=extrinsics_new,
        )

    def _get_render_result(self) -> "FeedforwardResult":
        """Return display result with ground plane transform applied if enabled."""
        base = self._display_result if self._display_result is not None else self._result
        return self._apply_ground_plane(base)

    def _on_ground_plane_change(self, event: Any) -> None:
        """Re-render when ground plane toggle or R changes."""
        if self._display_result is None:
            return
        self._plotter.clear()
        self._pcd_actor = None
        self._mesh_actor = None
        self._sim_actor = None
        self._sim_cloud = None
        mode_display = {"PCD": "Points", "Mesh": "Mesh", "Similarity": "Similarity"}
        if self.mode in mode_display:
            self._on_mode_change(mode_display[self.mode])
        self._vtk_pane.reset_camera()

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
        # Re-render the current mode so the VTK pane syncs with any result that
        # arrived while this tab was inactive (synchronize is a no-op off-screen).
        if self._display_result is not None:
            self._on_mode_change(self._mode_selector.value)

    ####################################################################
    # Status helper
    ####################################################################

    def _set_status(self, msg: str) -> None:
        """Update status HTML pane."""
        self._status_html.object = f"<small>{msg}</small>"

    ####################################################################
    # Mode switching
    ####################################################################

    def _on_mode_change(self, new_display_mode: str) -> None:
        """Switch viewer mode; use cached actors where available."""
        # Map selector labels ("Points") to internal mode names ("PCD")
        mode_map = {"Points": "PCD", "Mesh": "Mesh", "Similarity": "Similarity"}
        new_mode = mode_map.get(new_display_mode, new_display_mode)

        if new_mode not in self._available_modes and self._result is not None:
            return
        self.mode = new_mode

        # Hide all cached actors
        if self._pcd_actor is not None:
            self._pcd_actor.VisibilityOff()
        if self._mesh_actor is not None:
            self._mesh_actor.VisibilityOff()
        if self._sim_actor is not None:
            self._sim_actor.VisibilityOff()

        self._sim_query_row.visible = (new_mode == "Similarity")

        if new_mode == "PCD":
            if self._pcd_actor is None:
                self._plotter.clear()
                self._sim_actor = None
                self._sim_cloud = None
                self._rebuild_pcd_viewer()
            else:
                self._pcd_actor.VisibilityOn()
        elif new_mode == "Mesh":
            if self._mesh_actor is None:
                self._plotter.clear()
                self._sim_actor = None
                self._sim_cloud = None
                self._rebuild_mesh_viewer()
            else:
                self._mesh_actor.VisibilityOn()
        elif new_mode == "Similarity":
            if self._lifted_normed is None:
                self._load_lifted_features_for_current_extractor()
            if self._sim_actor is None:
                self._rebuild_sim_viewer(colors=None)
            else:
                self._sim_actor.VisibilityOn()

        self._vtk_pane.synchronize()

    ####################################################################
    # PCD viewer
    ####################################################################

    def _rebuild_pcd_viewer(self) -> None:
        """Render points + optional frustums; cache actor."""
        if self._display_result is None and self._result is None:
            return
        result_to_use = self._get_render_result()
        cloud = pointcloud_to_polydata(result_to_use.points, RGB=result_to_use.colors)
        self._pcd_actor = self._plotter.add_mesh(
            cloud, scalars="RGB", rgb=True, point_size=2, render_points_as_spheres=False
        )
        if self._state_frustum_enabled:
            self._add_frustums()

    def _add_frustums(self) -> None:
        """Add camera frustum actors coloured by frame order (plasma colormap)."""
        if self._display_result is None and self._result is None:
            return
        result_to_use = self._get_render_result()
        extrinsics = result_to_use.extrinsics
        n = len(extrinsics)
        colors = cm.plasma(np.linspace(0, 1, max(n, 1)))
        for i, ext in enumerate(extrinsics):
            frustum = create_camera_frustum_pyvista(ext)
            r, g, b = colors[i, :3]
            self._plotter.add_mesh(frustum, color=(r, g, b), line_width=2)



    def _on_frustum_toggle_from_sidebar(self, enabled: bool) -> None:
        """Called from App sidebar frustum checkbox."""
        self._state_frustum_enabled = enabled
        if self.mode != "PCD" or self._result is None:
            return
        self._plotter.clear()
        self._pcd_actor = None
        self._sim_actor = None
        self._sim_cloud = None
        self._rebuild_pcd_viewer()
        self._vtk_pane.synchronize()

    ####################################################################
    # Mesh viewer
    ####################################################################

    def run_mesh(
        self,
        voxel_size: float = 0.01,
        sdf_trunc: float = 0.04,
        depth_trunc: float = 10.0,
        clean_repair: bool = True,
        on_done: Any = None,
        on_progress: Any = None,
    ) -> None:
        """Spawn mesh generation thread with given params; call on_done(ok, msg) when done."""
        if self._mesh_thread and self._mesh_thread.is_alive():
            return
        self._mesh_voxel = voxel_size
        self._mesh_sdf = sdf_trunc
        self._mesh_depth = depth_trunc
        self._mesh_clean = clean_repair
        self._mesh_on_done = on_done
        self._mesh_on_progress = on_progress
        self._set_status("Running mesh generation…")
        self._mesh_thread = threading.Thread(target=self._run_mesh_worker, daemon=True)
        self._mesh_thread.start()

    def stop_mesh(self) -> None:
        """Terminate the running mesh subprocess."""
        proc = self._mesh_proc
        if proc is not None and proc.is_alive():
            proc.terminate()

    def _run_mesh_worker(self) -> None:
        """Background thread: spawn mesh generation subprocess → report result.

        Runs in an isolated subprocess so OOM only kills that process,
        not the dashboard server.  Progress messages flow back via a
        multiprocessing.Queue polled here in the background thread.
        """
        _ok = False
        _msg = "No dataset selected."
        _cb = self._mesh_on_done
        _on_progress = self._mesh_on_progress
        if self._current_dataset_dir is None or self._current_backend is None:
            pn.io.state.execute(lambda: self._set_status(_msg))
            if _cb is not None:
                pn.io.state.execute(lambda: _cb(_ok, _msg))
            return
        try:
            zarr_path = self._current_dataset_dir / self._current_backend / "feedforward.zarr"
            mesh_dir = self._current_dataset_dir / self._current_backend / "mesh"
            progress_queue: multiprocessing.Queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_mesh_subprocess_worker,
                args=(
                    str(zarr_path),
                    str(mesh_dir),
                    self._mesh_voxel,
                    self._mesh_sdf,
                    self._mesh_depth,
                    self._mesh_clean,
                    progress_queue,
                ),
                daemon=True,
            )
            proc.start()
            self._mesh_proc = proc

            # Poll progress queue while subprocess is alive
            while proc.is_alive():
                try:
                    msg = progress_queue.get(timeout=0.2)
                    if _on_progress is not None:
                        desc = msg.get("desc", "")
                        n = msg.get("n", 0)
                        total = msg.get("total", 1) or 1
                        pct = min(99, int(n / total * 100))
                        pn.io.state.execute(lambda d=desc, p=pct: _on_progress(d, p))
                except _queue_mod.Empty:
                    pass

            proc.join()
            self._mesh_proc = None

            if proc.exitcode == 0:
                mesh_path = mesh_dir / "mesh.ply"
                _msg = f"Mesh done — {mesh_path.name}"
                _ok = True
                pn.io.state.execute(lambda: self._refresh_after_mesh(_msg))
            elif proc.exitcode == -15:
                # SIGTERM from stop_mesh() — not an error, just cancelled
                _msg = "Mesh cancelled."
                pn.io.state.execute(lambda: self._set_status(_msg))
            else:
                _msg = f"Mesh failed (exit {proc.exitcode})"
                pn.io.state.execute(lambda: self._set_status(_msg))
        except Exception as exc:
            logger.exception("Mesh generation failed")
            _err = str(exc)
            _msg = f"Mesh failed: {_err}"
            pn.io.state.execute(lambda: self._set_status(_msg))
        finally:
            self._mesh_proc = None
            ok_val, msg_val = _ok, _msg
            if _cb is not None:
                pn.io.state.execute(lambda: _cb(ok_val, msg_val))

    def _refresh_after_mesh(self, status_msg: str) -> None:
        """IOLoop-thread: redraw mesh viewer after successful generation."""
        self._plotter.clear()
        self._pcd_actor = None
        self._mesh_actor = None
        self._sim_actor = None
        self._sim_cloud = None
        self._set_status(status_msg)
        self._available_modes.add("Mesh")
        self._update_mode_buttons()
        # If the selector is already "Mesh" the param watcher won't fire on
        # assignment, so _on_mode_change (which rebuilds the mesh and calls
        # synchronize()) would be skipped — nothing visible would update.
        # Call it explicitly in that case; otherwise change the value and let
        # the watcher handle it.
        if self._mode_selector.value == "Mesh":
            self._on_mode_change("Mesh")
        else:
            self._mode_selector.value = "Mesh"

    def _rebuild_mesh_viewer(self) -> None:
        """Load and render mesh.ply; cache actor."""
        if self._current_dataset_dir is None or self._current_backend is None:
            return
        mesh_path = self._current_dataset_dir / self._current_backend / "mesh" / "mesh.ply"
        if not mesh_path.exists():
            self._set_status("mesh.ply not found.")
            return
        mesh = pv.read(str(mesh_path))
        self._mesh_actor = self._plotter.add_mesh(mesh, rgb=True)

    def _auto_display_mesh(self) -> None:
        """Render mesh.ply immediately — called from scan, no Load required."""
        self._plotter.clear()
        self._mesh_actor = None
        self._sim_actor = None
        self._sim_cloud = None
        self._rebuild_mesh_viewer()
        self._vtk_pane.synchronize()
        # Switch selector to Mesh so the UI reflects the active view
        self.mode = "Mesh"
        self._mode_selector.value = "Mesh"

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
        if self._display_result is None and self._result is None:
            return
        result_to_use = self._get_render_result()
        if colors is None:
            rgb = result_to_use.colors
            self._set_status("Enter a query to colour by similarity.")
        else:
            rgb = colors
        if self._sim_actor is None:
            self._sim_cloud = pointcloud_to_polydata(result_to_use.points, RGB=rgb)
            self._sim_actor = self._plotter.add_mesh(
                self._sim_cloud, scalars="RGB", rgb=True, point_size=2
            )
        else:
            # Update colors in-place — skips full mesh rebuild and re-serialization
            self._sim_cloud["RGB"] = rgb

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

        # Update sim cloud colors in-place and sync; no full scene rebuild
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
    # Reset camera
    ####################################################################

    def _on_reset_camera(self, event: Any) -> None:
        """Reset camera to fit current scene bounds."""
        self._vtk_pane.reset_camera()

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

    def _noop(self) -> None:
        """Periodic no-op — keeps the Bokeh/Tornado WebSocket alive."""

    def panel(self) -> pn.Column:
        """Return the full scene panel layout."""
        pn.state.add_periodic_callback(self._noop, period=25_000)
        extractor_row = pn.Row(self._extractor_dd)
        action_row = pn.Row(self._reset_btn, self._snapshot_btn)
        return pn.Column(
            "### Scene",
            extractor_row,
            self._mode_selector,
            self._sim_query_row,
            self._vtk_pane,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )
