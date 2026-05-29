"""LocalizePane — Tab 5 of the unified dashboard.

Camera localization in a known reconstruction: single-image + batch modes.
"""
from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import panel as pn
import param
import pyvista as pv

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.localization import (
    CameraLocalizer,
    DiskExtractor,
    XFeatExtractor,
    plot_correspondences,
)
from collab_splats.utils.visualization import (
    create_camera_frustum_pyvista,
    pointcloud_to_polydata,
)

logger = logging.getLogger(__name__)

########################################################################
# Extractor registry

# Map dropdown label → extractor class
_EXTRACTOR_CLASSES: dict[str, type] = {
    "DISK+LightGlue": DiskExtractor,
    "XFeat+MNN": XFeatExtractor,
}

########################################################################
# Colours

_COLOR_DEFAULT = "cornflowerblue"
_COLOR_QUERY   = "tomato"
_COLOR_REF     = "gold"


def _rgb(name: str) -> tuple[float, float, float]:
    """Convert matplotlib colour name to (r, g, b) 0–1 floats for VTK."""
    return mc.to_rgb(name)


_RGB_DEFAULT = _rgb(_COLOR_DEFAULT)
_RGB_QUERY   = _rgb(_COLOR_QUERY)
_RGB_REF     = _rgb(_COLOR_REF)

########################################################################


class LocalizeScenePanel(param.Parameterized):
    """Minimal PyVista 3D viewer for the Localize tab.

    Renders point cloud + camera frustums. Highlights a query/ref pair
    after localization via highlight(); reset() clears the highlight.
    """

    def __init__(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        image_paths: list,
        _off_screen: bool = False,
        **params: Any,
    ):
        """Build scene from pts3d (P,3), extrinsics (N,4,4), image_paths length-N."""
        super().__init__(**params)
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._image_paths = image_paths
        self._highlighted_query_idx: int | None = None
        self._highlighted_ref_idx: int | None = None
        self._query_actor = None
        self._connector_actor = None

        self._plotter = pv.Plotter(off_screen=_off_screen, notebook=False)
        try:
            self._vtk_pane = pn.pane.VTK(
                self._plotter.ren_win,
                sizing_mode="stretch_both",
                min_height=300,
            )
        except Exception as e:
            # ren_win may be unavailable (e.g. in tests with a mocked Plotter)
            logger.debug("VTK pane init failed (headless?): %s", e)
            self._vtk_pane = None
        self._build_scene()

    def _build_scene(self) -> None:
        """Render point cloud + all camera frustums at default colour."""
        self._plotter.clear()

        # Point cloud
        if len(self._pts3d) > 0:
            cloud = pointcloud_to_polydata(self._pts3d)
            self._plotter.add_mesh(
                cloud, color="lightgray", point_size=2, render_points_as_spheres=True
            )

        # Camera frustums — store actors so we can recolour on highlight
        self._frustum_actors: list[Any] = []
        for ext in self._extrinsics:
            frustum = create_camera_frustum_pyvista(ext)
            actor = self._plotter.add_mesh(frustum, color=_RGB_DEFAULT, line_width=1)
            self._frustum_actors.append(actor)

        self._plotter.reset_camera()

    def highlight(
        self,
        query_ext: np.ndarray,
        ref_ext: np.ndarray,
        query_idx: int,
        ref_idx: int,
    ) -> None:
        """Highlight query camera (red) and ref camera (yellow); grey out others.

        When query_idx == -1, query_ext is not in self._extrinsics — add it
        as a temporary actor instead of recolouring an existing one.
        """
        self._highlighted_query_idx = query_idx
        self._highlighted_ref_idx = ref_idx

        # Remove previous temporary query actor if any
        if self._query_actor is not None:
            self._plotter.remove_actor(self._query_actor)
            self._query_actor = None

        # Remove previous connector if any
        if self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None

        # Recolour reference frustums
        for i, actor in enumerate(self._frustum_actors):
            if i == ref_idx:
                actor.GetProperty().SetColor(*_RGB_REF)
                actor.GetProperty().SetLineWidth(3)
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetColor(*_RGB_DEFAULT)
                actor.GetProperty().SetOpacity(0.25)
                actor.GetProperty().SetLineWidth(1)

        # Add query camera as temporary red frustum (query_idx == -1 means not in array)
        if query_idx == -1:
            frustum = create_camera_frustum_pyvista(query_ext)
            self._query_actor = self._plotter.add_mesh(
                frustum, color=_COLOR_QUERY, line_width=3
            )
        else:
            self._frustum_actors[query_idx].GetProperty().SetColor(*_RGB_QUERY)
            self._frustum_actors[query_idx].GetProperty().SetLineWidth(3)
            self._frustum_actors[query_idx].GetProperty().SetOpacity(1.0)

        # Draw connector between query and ref camera centres
        q_pos = np.linalg.inv(query_ext)[:3, 3]
        r_pos = np.linalg.inv(ref_ext)[:3, 3]
        line = pv.Line(q_pos.tolist(), r_pos.tolist())
        self._connector_actor = self._plotter.add_mesh(
            line, color=_COLOR_REF, line_width=2
        )

        if self._vtk_pane is not None:
            self._vtk_pane.param.trigger("object")

    def reset(self) -> None:
        """Clear highlight; restore all cameras to default colour."""
        self._highlighted_query_idx = None
        self._highlighted_ref_idx = None

        if self._query_actor is not None:
            self._plotter.remove_actor(self._query_actor)
            self._query_actor = None

        if self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None

        for actor in self._frustum_actors:
            actor.GetProperty().SetColor(*_RGB_DEFAULT)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetLineWidth(1)

        if self._vtk_pane is not None:
            self._vtk_pane.param.trigger("object")

    def panel(self) -> pn.pane.VTK | None:
        """Return the VTK Panel pane."""
        return self._vtk_pane


########################################################################


def _scan_recon_methods(output_dir: Path) -> list[str]:
    """Return method names whose feedforward.zarr exists under output_dir."""
    if not output_dir or not output_dir.is_dir():
        return []
    return sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    )


def _empty_batch_df():
    """Return empty DataFrame with batch result columns."""
    return pd.DataFrame(columns=["image", "inliers", "status", "t-err (m)", "pose t"])


def _render_correspondences_to_png(
    loc: Any,
    query_img: np.ndarray,
    image_paths: list,
    warp_corners: bool,
) -> bytes | None:
    """Call plot_correspondences() and capture matplotlib output as PNG bytes."""
    buf = io.BytesIO()
    try:
        fig = plt.figure(figsize=(10, 4))
        plot_correspondences(loc, query_img, image_paths, warp_corners=warp_corners)
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        plt.close("all")
        logger.exception("plot_correspondences failed")
        return None


########################################################################


class LocalizePane(param.Parameterized):
    """Camera localization tab — single-image + batch modes.

    Left panel: plot_correspondences() matplotlib PNG.
    Right panel: LocalizeScenePanel PyVista 3D viewer.
    """

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._localizer = None
        self._ff_result = None
        self._scene_panel: LocalizeScenePanel | None = None
        self._loc_thread: threading.Thread | None = None
        self._batch_thread: threading.Thread | None = None

        # Controls
        self._query_input = pn.widgets.TextInput(
            placeholder="Path to query image…",
            width=320,
        )
        self._method_dd = pn.widgets.Select(
            name="Recon",
            options=[],
            width=120,
        )
        self._extractor_dd = pn.widgets.Select(
            name="Extractor",
            options=["DISK+LightGlue", "XFeat+MNN"],
            value="DISK+LightGlue",
            width=140,
        )
        self._run_btn = pn.widgets.Button(
            name="▶ Localize",
            button_type="success",
            disabled=True,
            width=100,
        )
        self._warp_cb = pn.widgets.Checkbox(name="warp corners", value=True)
        self._status_html = pn.pane.HTML("", width=400)

        # Correspondence display (left panel)
        self._corr_png = pn.pane.PNG(
            object=None,
            sizing_mode="stretch_both",
            min_height=200,
        )
        self._corr_info = pn.pane.HTML(
            "<span style='color:#666;font-size:12px'>No result yet</span>",
        )

        # Batch mode widgets
        self._batch_folder_input = pn.widgets.TextInput(
            placeholder="Path to query image folder…",
            width=280,
        )
        self._batch_run_btn = pn.widgets.Button(
            name="▶ Run batch",
            button_type="primary",
            disabled=True,
            width=100,
        )
        self._batch_export_btn = pn.widgets.Button(
            name="⬇ Export CSV",
            disabled=True,
            width=110,
        )
        self._batch_table = pn.widgets.Tabulator(
            value=_empty_batch_df(),
            show_index=False,
            sizing_mode="stretch_width",
            height=180,
        )

        # Right-column container for 3D scene panel (initialized here so _run_localize
        # can update it before panel() is called)
        right_placeholder = pn.pane.HTML(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:100%;color:#555;font-size:13px'>Run localization to see 3D scene</div>",
            sizing_mode="stretch_both",
            min_height=300,
        )
        self._right_col = pn.Column(right_placeholder, sizing_mode="stretch_both")

        # Wire callbacks
        self._run_btn.on_click(self._on_run)
        self._batch_run_btn.on_click(self._on_batch_run)
        self._batch_export_btn.on_click(self._on_export_csv)
        self._query_input.param.watch(self._on_query_or_dir_changed, ["value"])
        self._state.param.watch(self._on_output_dir_changed, ["output_dir"])

        # Initial gate state
        self._on_output_dir_changed(None)

    # ------------------------------------------------------------------
    # Gate helpers

    def _on_output_dir_changed(self, event: Any) -> None:
        """Rescan method dropdown; re-evaluate run-button gate."""
        output_dir = self._state.output_dir
        methods = _scan_recon_methods(Path(output_dir)) if output_dir else []
        self._method_dd.options = methods
        if methods:
            self._method_dd.value = methods[0]
        self._batch_run_btn.disabled = not bool(methods)
        self._update_run_btn_gate()

    def _on_query_or_dir_changed(self, event: Any) -> None:
        """Re-evaluate run-button gate when query path changes."""
        self._update_run_btn_gate()

    def _update_run_btn_gate(self) -> None:
        """Enable run only when output_dir set, method available, and query path non-empty."""
        has_dir = self._state.output_dir is not None
        has_method = bool(self._method_dd.options)
        has_query = bool(self._query_input.value and self._query_input.value.strip())
        self._run_btn.disabled = not (has_dir and has_method and has_query)

    # ------------------------------------------------------------------
    # Shared localizer builder

    def _build_localizer(
        self, method: str, extractor_name: str
    ) -> tuple["FeedforwardResult", "CameraLocalizer"]:
        """Load feedforward result from zarr and build CameraLocalizer."""
        output_dir = Path(self._state.output_dir)
        zarr_path = output_dir / method / "feedforward.zarr"
        ff = FeedforwardResult.load_zarr(zarr_path)
        extractor = _EXTRACTOR_CLASSES.get(extractor_name, DiskExtractor)()
        localizer = CameraLocalizer.from_feedforward(ff, extractor=extractor)
        return ff, localizer

    # ------------------------------------------------------------------
    # Single-image localization

    def _on_run(self, event: Any) -> None:
        """Spawn background localization thread on button click."""
        if self._loc_thread and self._loc_thread.is_alive():
            return
        # Snapshot widget values on main thread before passing to worker
        method = self._method_dd.value
        extractor_name = self._extractor_dd.value
        query_path = Path(self._query_input.value.strip())
        warp_corners = self._warp_cb.value
        self._run_btn.disabled = True
        self._status_html.object = "<span style='color:#2596be'>⏳ Localizing…</span>"
        self._loc_thread = threading.Thread(
            target=self._run_localize,
            args=(method, extractor_name, query_path, warp_corners),
            daemon=True,
        )
        self._loc_thread.start()

    def _run_localize(
        self,
        method: str,
        extractor_name: str,
        query_path: Path,
        warp_corners: bool,
    ) -> None:
        """Background thread: load result, build localizer, run, update UI."""
        try:
            self._op_log.start_op(f"Localizing in {method}")

            # Load feedforward result and build localizer
            ff, localizer = self._build_localizer(method, extractor_name)
            self._ff_result = ff
            self._localizer = localizer

            # Build scene panel if not yet built
            if self._scene_panel is None:
                self._scene_panel = LocalizeScenePanel(
                    pts3d=ff.points,
                    extrinsics=ff.extrinsics,
                    image_paths=ff.image_paths,
                )
                self._right_col[:] = [self._scene_panel.panel()]

            # Load query image
            bgr = cv2.imread(str(query_path))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read query image: {query_path}")
            query_img = bgr[..., ::-1].copy()  # BGR → RGB

            # Use mean reference intrinsics as query intrinsics (same camera assumed)
            query_intrinsics = ff.intrinsics.mean(axis=0)

            # Run localization
            loc = localizer.localize(query_img, query_intrinsics)

            self._op_log.finish_op()

            if loc.pose is None:
                n_inliers = int(loc.inlier_mask.sum()) if loc.inlier_mask is not None else 0
                self._corr_info.object = (
                    f"<span style='color:#f85149'>✗ Failed — {n_inliers} inliers</span>"
                )
                self._status_html.object = "<span style='color:#e05050'>✗ Localization failed</span>"
                self._op_log.error_op(f"Localization failed: {n_inliers} inliers")
            else:
                # Inlier stats + best ref frame
                inlier_frames = loc.ref_frame_indices[loc.inlier_mask]
                best_ref_idx = int(np.bincount(inlier_frames.astype(np.intp)).argmax())
                n_inliers = int(loc.inlier_mask.sum())
                n_total = int(loc.inlier_mask.shape[0])
                ref_name = Path(ff.image_paths[best_ref_idx]).name

                self._corr_info.object = (
                    f"<span style='color:#3fb950'>● {n_inliers} inliers</span> &nbsp;"
                    f"<span style='color:#f85149'>● {n_total - n_inliers} outliers</span> &nbsp;| "
                    f"&nbsp;best ref: <span style='color:#e3b341'>{ref_name}</span>"
                )

                # Render correspondence PNG
                buf = _render_correspondences_to_png(
                    loc, query_img, ff.image_paths, warp_corners
                )
                if buf is not None:
                    self._corr_png.object = buf

                # Update pose text
                t = loc.pose[:3, 3]
                self._status_html.object = (
                    f"<span style='color:#3fb950'>✓ pose t=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]</span>"
                )

                # Highlight cameras in 3D
                if self._scene_panel is not None:
                    self._scene_panel.highlight(
                        query_ext=loc.pose,
                        ref_ext=ff.extrinsics[best_ref_idx],
                        query_idx=-1,
                        ref_idx=best_ref_idx,
                    )

        except Exception as exc:
            logger.exception("LocalizePane: localization failed")
            self._op_log.error_op(str(exc))
            self._status_html.object = f"<span style='color:#e05050'>✗ {exc}</span>"
        finally:
            self._run_btn.disabled = False

    # ------------------------------------------------------------------
    # Batch mode (Task 4)

    def _on_batch_run(self, event: Any) -> None:
        """Spawn background batch thread."""
        if self._batch_thread and self._batch_thread.is_alive():
            return
        folder_val = self._batch_folder_input.value.strip()
        if not folder_val:
            return
        method = self._method_dd.value
        extractor_name = self._extractor_dd.value
        folder_path = Path(folder_val)
        self._batch_run_btn.disabled = True
        self._batch_export_btn.disabled = True
        self._batch_table.value = _empty_batch_df()
        self._batch_thread = threading.Thread(
            target=self._run_batch,
            args=(method, extractor_name, folder_path),
            daemon=True,
        )
        self._batch_thread.start()

    def _run_batch(
        self,
        method: str,
        extractor_name: str,
        folder_path: Path,
    ) -> None:
        """Background thread: localize every image in folder_path, stream rows to table."""
        try:
            # Load feedforward result and build localizer
            ff, localizer = self._build_localizer(method, extractor_name)
            query_intrinsics = ff.intrinsics.mean(axis=0)

            # Collect image paths (common image extensions)
            img_exts = {".jpg", ".jpeg", ".png"}
            query_paths = sorted(
                p for p in folder_path.iterdir()
                if p.suffix.lower() in img_exts
            )

            rows: list[dict] = []
            for qp in query_paths:
                bgr = cv2.imread(str(qp))
                if bgr is None:
                    rows.append({
                        "image": qp.name, "inliers": 0, "status": "✗",
                        "t-err (m)": "—", "pose t": "—",
                    })
                    self._batch_table.value = pd.DataFrame(rows)
                    continue

                query_img = bgr[..., ::-1].copy()
                try:
                    loc = localizer.localize(query_img, query_intrinsics)
                except Exception as exc:
                    rows.append({
                        "image": qp.name, "inliers": 0, "status": "✗",
                        "t-err (m)": "—", "pose t": str(exc)[:40],
                    })
                    self._batch_table.value = pd.DataFrame(rows)
                    continue

                if loc.pose is None:
                    n_in = int(loc.inlier_mask.sum()) if loc.inlier_mask is not None else 0
                    rows.append({
                        "image": qp.name, "inliers": n_in, "status": "✗",
                        "t-err (m)": "—", "pose t": "—",
                    })
                else:
                    n_in = int(loc.inlier_mask.sum())
                    t = loc.pose[:3, 3]
                    rows.append({
                        "image": qp.name,
                        "inliers": n_in,
                        "status": "✓",
                        "t-err (m)": "—",
                        "pose t": f"[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}]",
                    })

                self._batch_table.value = pd.DataFrame(rows)

            self._batch_export_btn.disabled = False

        except Exception as exc:
            logger.exception("LocalizePane: batch failed")
            self._op_log.error_op(str(exc))
        finally:
            self._batch_run_btn.disabled = False

    def _on_export_csv(self, event: Any) -> None:
        """Trigger CSV download from Tabulator."""
        self._batch_table.download(filename="localize_batch.csv")

    # ------------------------------------------------------------------
    # Layout

    def panel(self) -> pn.viewable.Viewable:
        """Return the full LocalizePane layout."""
        controls_bar = pn.Row(
            self._query_input,
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.HTML("<b style='color:#8b949e;font-size:12px'>Recon:</b>"),
            self._method_dd,
            self._extractor_dd,
            self._run_btn,
            sizing_mode="stretch_width",
            margin=(4, 0),
        )

        corr_header = pn.Row(
            self._corr_info,
            pn.Spacer(sizing_mode="stretch_width"),
            self._warp_cb,
            margin=(0, 0, 4, 0),
        )

        left_panel = pn.Column(
            corr_header,
            self._corr_png,
            self._status_html,
            sizing_mode="stretch_both",
        )

        split = pn.Row(
            pn.Column(left_panel, sizing_mode="stretch_both", width_policy="max"),
            pn.Column(self._right_col, sizing_mode="stretch_both", width_policy="max"),
            sizing_mode="stretch_width",
            min_height=360,
        )

        batch_section = pn.Card(
            pn.Column(
                pn.Row(
                    self._batch_folder_input,
                    self._batch_run_btn,
                    self._batch_export_btn,
                    margin=(4, 0),
                ),
                self._batch_table,
                sizing_mode="stretch_width",
            ),
            title="Batch mode",
            collapsed=True,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            controls_bar,
            split,
            batch_section,
            sizing_mode="stretch_width",
        )

