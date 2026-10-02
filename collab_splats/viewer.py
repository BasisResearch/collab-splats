"""
Browser 3D scene viewer over viser: arrays in, named scene nodes out.

- nodes upsert by name: re-adding a name replaces the node, which is how callers refresh one
- point clouds, camera frusta, line sets and meshes; meshes add vertex picking and label lists
- served at http://<host>:<port> over a websocket; no display needed on the host
"""

import logging
import threading
import zlib
from typing import Callable, Optional

import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as viser_tf

logger = logging.getLogger(__name__)


class Viewer:
    """
    Live viser scene of named point clouds, camera frusta, line sets and pickable meshes.

    - GUI: camera toggle, flat color per node, point size, up direction, reset view
    - `points` holds name -> (handle, points, colors, point_size) so toggles can re-send clouds

    Args:
        port: port the viser server listens on.
    """

    def __init__(self, port: int = 8080) -> None:
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        logger.info("viser scene viewer on port %d", port)

        # Per-name node state, so GUI toggles can restyle nodes already in the scene
        self.points: dict[str, tuple[viser.PointCloudHandle, np.ndarray, np.ndarray, float]] = {}
        self.frustums: dict[str, viser.CameraFrustumHandle] = {}

        # Meshes: name -> (vertices, faces, raycasting scene); click callbacks and label lists
        self.meshes: dict[str, tuple[np.ndarray, np.ndarray, o3d.t.geometry.RaycastingScene]] = {}
        self.mesh_clicks: dict[str, Callable[[int], None]] = {}
        self.label_lists: dict[str, tuple[viser.GuiFolderHandle, list[viser.GuiButtonHandle]]] = {}
        self._stop = threading.Event()

        # Camera visibility and flat per-node colors (shows node boundaries)
        self.show_cameras = self.server.gui.add_checkbox("Show cameras", initial_value=True)
        self.show_cameras.on_update(lambda _: self._apply_camera_visibility())
        self.color_by_node = self.server.gui.add_checkbox("Color by node", initial_value=False)
        self.color_by_node.on_update(lambda _: self._resend_points())

        # Point size for every cloud; re-sends them on change
        self.point_size = self.server.gui.add_slider(
            "Point size", min=0.0005, max=0.02, step=0.0005, initial_value=0.003
        )
        self.point_size.on_update(lambda _: self._resend_points(self.point_size.value))

        # World up axis and a button that frames the scene
        self.up_direction = self.server.gui.add_dropdown(
            "Up direction", ("-y", "+y", "+z", "-z", "+x", "-x"), initial_value="-y"
        )
        self.up_direction.on_update(lambda _: self.server.scene.set_up_direction(self.up_direction.value))
        self.server.scene.set_up_direction("-y")
        self.server.gui.add_button("Reset view").on_click(lambda _: self._reset_view())

    ########################################################################
    # Scene nodes (upsert by name)
    ########################################################################

    def add_points(self, name: str, points: np.ndarray, colors: np.ndarray, point_size: float = 0.003) -> None:
        """
        Upsert a named point cloud.

        Args:
            name: scene node name.
            points: (N, 3) positions.
            colors: (N, 3) uint8 colors; replaced by the node color while "Color by node" is on.
            point_size: dot size, world units.
        """
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
        max_image_width: int = 128,
    ) -> None:
        """
        Upsert one camera frustum.

        - hierarchical names group a submap's cameras, e.g. "submap_3/cams/frame_0"
        - without an image, the sensor size comes from the principal point (image center)

        Args:
            name: scene node name.
            pose: (4, 4) world-to-camera.
            intrinsic: (3, 3) camera matrix.
            image: optional (H, W, 3) uint8 thumbnail shown in the frustum.
            scale: frustum size, world units.
            max_image_width: thumbnails wider than this are strided down before upload.
        """
        # Sensor size from the image, strided down to cap the upload; else from the principal point
        if image is not None:
            h, w = image.shape[:2]

            if w > max_image_width:
                stride = int(np.ceil(w / max_image_width))
                image = image[::stride, ::stride]

        else:
            w, h = 2.0 * float(intrinsic[0, 2]), 2.0 * float(intrinsic[1, 2])

        fov = 2.0 * float(np.arctan2(h / 2.0, float(intrinsic[1, 1])))

        # Repo poses are world-to-camera; viser frames are camera-to-world
        pose64 = np.asarray(pose, dtype=np.float64)
        cam_to_world = np.linalg.inv(pose64)
        transform = viser_tf.SE3.from_matrix(cam_to_world)
        handle = self.server.scene.add_camera_frustum(
            name,
            fov=fov,
            aspect=w / h,
            scale=scale,
            image=image,
            wxyz=transform.rotation().wxyz,
            position=transform.translation(),
        )
        handle.visible = self.show_cameras.value
        self.frustums[name] = handle

    def add_lines(
        self, name: str, segments: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), line_width: float = 2.0
    ) -> None:
        """
        Upsert named line segments in one color.

        Args:
            name: scene node name.
            segments: (M, 2, 3) segment endpoints.
            color: RGB 0-255.
            line_width: pixels.
        """
        segments = np.asarray(segments)
        rgb = np.asarray(color, dtype=np.uint8)
        colors = np.tile(rgb, (len(segments), 2, 1))
        self.server.scene.add_line_segments(name, points=segments, colors=colors, line_width=line_width)

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        textured: Optional[trimesh.Trimesh] = None,
    ) -> None:
        """
        Upsert a named mesh; picking, labels and highlights index its vertices.

        Args:
            name: scene node name.
            vertices: (V, 3) positions.
            faces: (F, 3) vertex indices.
            colors: (V, 3) uint8 vertex colors.
            textured: optional display mesh of the same surface, shown instead; picks stay on `vertices`.
        """
        # Ray-cast scene over the pick geometry
        rays = o3d.t.geometry.RaycastingScene()
        vertices32 = o3d.core.Tensor(vertices.astype(np.float32))
        faces32 = o3d.core.Tensor(faces.astype(np.uint32))
        rays.add_triangles(vertices32, faces32)
        self.meshes[name] = (vertices, faces, rays)

        # Display the texture when given, else the vertex colors
        shown = textured

        if shown is None:
            shown = trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False)

        self.server.scene.add_mesh_trimesh(name, shown)

    def add_label_list(
        self, name: str, labels: np.ndarray, top_n: int = 20, color: tuple[int, int, int] = (255, 80, 0)
    ) -> None:
        """
        GUI buttons for a mesh's most common labels; each highlights its vertices, "Clear" removes it.

        - re-adding replaces the mesh's earlier list

        Args:
            name: mesh node name, already added with add_mesh.
            labels: (V,) str per vertex; "" is unlabeled.
            top_n: labels listed, most common first.
            color: RGB 0-255 of the highlight.
        """
        # Most common labels first, unlabeled vertices skipped
        named = labels[labels != ""]
        values, counts = np.unique(named, return_counts=True)
        order = np.argsort(-counts, kind="stable")[:top_n]

        # Replace an earlier list for this mesh
        if name in self.label_lists:
            self.label_lists[name][0].remove()

        folder = self.server.gui.add_folder(f"Labels: {name}")

        with folder:
            clear = self.server.gui.add_button("Clear")
            clear.on_click(lambda _: self.highlight(name, None, color))
            buttons = [clear]

            for i in order:
                label = str(values[i])
                button = self.server.gui.add_button(f"{label} ({counts[i]})")
                button.on_click(lambda _, label=label: self.highlight(name, labels == label, color))
                buttons.append(button)

        self.label_lists[name] = (folder, buttons)

    def highlight(self, name: str, mask: Optional[np.ndarray], color: tuple[int, int, int]) -> None:
        """
        Overlay a mesh's masked vertices as a `<name>/highlight` point cloud.

        - the mesh itself is never re-sent: only the masked vertices go over the wire

        Args:
            name: mesh node name, already added with add_mesh.
            mask: (V,) bool, vertices to highlight; None clears the overlay.
            color: RGB 0-255 of the overlay.
        """
        overlay = f"{name}/highlight"

        # Drop the previous overlay
        if overlay in self.points:
            self.points.pop(overlay)[0].remove()

        if mask is None:
            return

        # Masked vertices in one color, at the slider's point size
        hits = self.meshes[name][0][mask]
        rgb = np.asarray(color, dtype=np.uint8)
        colors = np.tile(rgb, (len(hits), 1))
        self.add_points(overlay, hits, colors, point_size=self.point_size.value)

    ########################################################################
    # Picking
    ########################################################################

    def on_click(self, name: str, callback: Callable[[int], None]) -> None:
        """
        Call `callback` with the vertex nearest where a click first hits the mesh.

        - one scene-wide click handler serves every mesh; the nearest hit across meshes wins
        - no per-node handler, so viser draws no hover outline on the mesh

        Args:
            name: mesh node name, already added with add_mesh.
            callback: receives the picked vertex index.
        """
        # Register the scene-wide click handler once, on the first clickable mesh
        if not self.mesh_clicks:
            register = self.server.scene.on_click()
            register(self._dispatch_click)

        self.mesh_clicks[name] = callback

    def pick(self, origin: tuple, direction: tuple) -> Optional[tuple[str, int]]:
        """
        Clickable mesh and vertex a ray hits first.

        - only meshes registered with `on_click` are tested

        Args:
            origin: ray origin, world frame.
            direction: unit ray direction, world frame.

        Returns:
            (mesh name, vertex index) of the hit nearest the origin, or None on a miss.
        """
        best_t, best = np.inf, None

        for name in self.mesh_clicks:
            hit = self._pick_vertex(name, origin, direction)

            if hit is not None and hit[0] < best_t:
                best_t, best = hit[0], (name, hit[1])

        return best

    def _dispatch_click(self, event: viser.SceneClickEvent) -> None:
        """
        Hand the clicked vertex of the nearest-hit clickable mesh to its callback; misses are ignored.
        """
        best = self.pick(event.ray_origin, event.ray_direction)

        if best is not None:
            name, vertex = best
            self.mesh_clicks[name](vertex)

    def _pick_vertex(self, name: str, origin: tuple, direction: tuple) -> Optional[tuple[float, int]]:
        """
        Ray distance and vertex of the first-hit triangle nearest the hit point.

        - None when the ray misses the mesh
        """
        vertices, faces, rays = self.meshes[name]
        ray = o3d.core.Tensor([[*origin, *direction]], dtype=o3d.core.float32)
        hit = rays.cast_rays(ray)
        t = hit["t_hit"][0].item()

        if not np.isfinite(t):
            return None

        # Hit point, then the triangle corner nearest it
        triangle = faces[hit["primitive_ids"][0].item()]
        point = np.asarray(origin) + t * np.asarray(direction)
        distances = np.linalg.norm(vertices[triangle] - point, axis=1)
        nearest = np.argmin(distances)
        return t, int(triangle[nearest])

    ########################################################################
    # Keep-alive
    ########################################################################

    def serve_forever(self, poll: float = 0.5) -> None:
        """
        Block so the viser server thread outlives the caller; Ctrl-C or setting `_stop` exits.

        Args:
            poll: seconds between stop checks.
        """
        try:
            while not self._stop.is_set():
                self._stop.wait(poll)

        except KeyboardInterrupt:
            pass

    ########################################################################
    # GUI handlers
    ########################################################################

    def _apply_camera_visibility(self) -> None:
        """
        Show or hide every frustum to match the checkbox.
        """
        for handle in self.frustums.values():
            handle.visible = self.show_cameras.value

    def _resend_points(self, point_size: Optional[float] = None) -> None:
        """
        Re-send every cloud with the current color mode, at `point_size` or its own size.

        - viser sets colors and point size at add time only
        """
        for name, (_, points, colors, own_size) in list(self.points.items()):
            size = own_size if point_size is None else point_size
            self.add_points(name, points, colors, point_size=size)

    def _reset_view(self) -> None:
        """
        Point every client's camera at the scene centroid from one extent back along -Z.

        - the scene is every point cloud and mesh vertex; up axis comes from set_up_direction
        """
        clouds = [points for (_, points, _, _) in self.points.values()]
        clouds += [vertices for (vertices, _, _) in self.meshes.values()]

        if not clouds:
            return

        # Centroid and bounding-box diagonal of everything shown
        pts = np.concatenate(clouds)
        center = pts.mean(axis=0)
        extent = float(np.linalg.norm(np.ptp(pts, axis=0)))

        # A single point has no extent; back off one unit
        if extent == 0.0:
            extent = 1.0

        for client in self.server.get_clients().values():
            client.camera.look_at = center
            client.camera.position = center + np.array([0.0, 0.0, -extent])

    def _flat_color(self, name: str, n: int) -> np.ndarray:
        """
        Deterministic color from the name's hash, broadcast to (n, 3) uint8.
        """
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        return np.tile(color, (n, 1))
