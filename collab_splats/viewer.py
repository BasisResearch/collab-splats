"""Generic viser-based 3D scene viewer for live visualization from running jobs.

Domain-neutral: arrays in, named scene nodes out. Nodes are upserted by name —
adding a node with an existing name replaces it, which is how callers refresh a
node (e.g. re-place a submap after pose-graph correction). View in a browser at
http://<host>:<port>. No GL/display needed on the host (websocket only).
"""

from __future__ import annotations

import logging
import threading
import zlib
from typing import Optional

import numpy as np
import viser
import viser.transforms as viser_tf

logger = logging.getLogger(__name__)

# Deterministic palette for the "Color by node" toggle; node -> color via name hash
_PALETTE_SEED = 100
_PALETTE_SIZE = 256

# Frustum image thumbnails are capped to this width before upload (bandwidth)
_MAX_IMAGE_WIDTH = 128


class Viewer:
    """Web-viewable live 3D scene: named point clouds, camera frusta, line sets."""

    def __init__(self, port: int = 8080):
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        logger.info("viser scene viewer on port %d", port)

        # Per-name state so GUI toggles can restyle nodes already in the scene.
        # points maps name -> (handle, points, colors, point_size); frustums -> handle.
        self.points: dict = {}
        self.frustums: dict = {}
        self.palette = np.random.default_rng(_PALETTE_SEED).integers(0, 256, size=(_PALETTE_SIZE, 3), dtype=np.uint8)

        # GUI: camera visibility toggle + flat per-node colors (shows node boundaries)
        self.show_cameras = self.server.gui.add_checkbox("Show cameras", initial_value=True)
        self.show_cameras.on_update(lambda _: self._apply_camera_visibility())
        self.color_by_node = self.server.gui.add_checkbox("Color by node", initial_value=False)
        self.color_by_node.on_update(lambda _: self._apply_point_colors())

    ########################################################
    ########## Scene nodes (upsert by name) ###############
    ########################################################

    def add_points(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 0.01,
    ) -> None:
        """Upsert a named point cloud; points (N, 3) float, colors (N, 3) uint8."""
        shown = self._flat_color(name, len(points)) if self.color_by_node.value else colors
        handle = self.server.scene.add_point_cloud(name, points=points, colors=shown, point_size=point_size)
        self.points[name] = (handle, points, colors, point_size)

    def add_frustum(
        self,
        name: str,
        pose: np.ndarray,
        intrinsic: np.ndarray,
        image: Optional[np.ndarray] = None,
        scale: float = 0.05,
    ) -> None:
        """Upsert one camera frustum; pose (4, 4) world-to-cam, intrinsic (3, 3).

        Callers loop over frames with hierarchical names (e.g. "submap_3/cams/frame_0").
        image is (H, W, 3) uint8, optional; downscaled before upload.
        """
        # Sensor size: from the image when given (strided down to cap upload size),
        # else approximated from the principal point (cx, cy) ~ image center.
        if image is not None:
            h, w = image.shape[:2]
            if w > _MAX_IMAGE_WIDTH:
                stride = int(np.ceil(w / _MAX_IMAGE_WIDTH))
                image = image[::stride, ::stride]
        else:
            w, h = 2.0 * float(intrinsic[0, 2]), 2.0 * float(intrinsic[1, 2])
        fov = 2.0 * float(np.arctan2(h / 2.0, float(intrinsic[1, 1])))

        # Repo poses are world-to-cam; viser frames are cam-to-world.
        T = viser_tf.SE3.from_matrix(np.linalg.inv(np.asarray(pose, dtype=np.float64)))
        handle = self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=w / h,
            scale=scale,
            image=image,
            wxyz=T.rotation().wxyz,
            position=T.translation(),
        )
        handle.visible = self.show_cameras.value
        self.frustums[name] = handle

    def add_lines(
        self,
        name: str,
        segments: np.ndarray,
        color: tuple = (0, 255, 0),
        line_width: float = 2.0,
    ) -> None:
        """Upsert named line segments; segments (M, 2, 3) float."""
        colors = np.tile(np.asarray(color, dtype=np.uint8), (len(segments), 2, 1))
        self.server.scene.add_line_segments(name, points=np.asarray(segments), colors=colors, line_width=line_width)

    ########################################################
    ########## Keep-alive #################################
    ########################################################

    def serve_forever(self, poll: float = 0.5) -> None:
        """Block so the viser server thread survives pipeline end (Ctrl-C / _stop to exit)."""
        if not hasattr(self, "_stop"):
            self._stop = threading.Event()
        try:
            while not self._stop.is_set():
                self._stop.wait(poll)
        except KeyboardInterrupt:
            pass

    ########################################################
    ########## GUI toggle handlers ########################
    ########################################################

    def _apply_camera_visibility(self) -> None:
        """Show/hide every frustum to match the checkbox."""
        for handle in self.frustums.values():
            handle.visible = self.show_cameras.value

    def _apply_point_colors(self) -> None:
        """Re-upload each cloud with flat or original colors (no in-place recolor in viser)."""
        for name, (_, points, colors, point_size) in list(self.points.items()):
            self.add_points(name, points, colors, point_size=point_size)

    def _flat_color(self, name: str, n: int) -> np.ndarray:
        """Deterministic per-name palette color, broadcast to (n, 3)."""
        color = self.palette[zlib.crc32(name.encode()) % _PALETTE_SIZE]
        return np.tile(color, (n, 1))
