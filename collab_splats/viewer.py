"""
Browser 3D scene viewer over viser: arrays in, named scene nodes out.

- nodes upsert by name: re-adding a name replaces the node, which is how callers refresh one
- point clouds, camera frusta, line sets and meshes; meshes add vertex picking, label lists and a heat overlay
- served at http://<host>:<port> over a websocket; no display needed on the host
- `python -m collab_splats.viewer <scene>/<backend>`: the mesh, plus a Semantics dropdown
  over its lifted stores' vertex arrays

Usage:
    HF_HOME=/workspace/models HF_HUB_OFFLINE=1 python -m collab_splats.viewer \\
        /workspace/outputs/<scene>/<backend> --port 8080
"""

import argparse
import hashlib
import logging
import threading
import zlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import open3d as o3d
import torch
import trimesh
import viser
import viser.transforms as viser_tf
import zarr
from PIL import Image

from collab_splats.geometry.transforms import invert_poses
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.semantics.store import read_point_features
from collab_splats.utils.torch_utils import pytorch_gc
from collab_splats.utils.visualization import apply_viridis

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
        self.points: dict[
            str, tuple[viser.PointCloudHandle, np.ndarray, np.ndarray, float]
        ] = {}
        self.frustums: dict[str, viser.CameraFrustumHandle] = {}

        # Meshes: name -> (vertices, faces, raycasting scene); click callbacks and label lists
        self.meshes: dict[
            str, tuple[np.ndarray, np.ndarray, o3d.t.geometry.RaycastingScene]
        ] = {}
        self.mesh_clicks: dict[str, Callable[[int], None]] = {}
        self.label_lists: dict[
            str, tuple[viser.GuiFolderHandle, list[viser.GuiButtonHandle]]
        ] = {}
        self.heats: dict[str, viser.GlbHandle] = {}
        self.heat_shifts: dict[str, np.ndarray] = {}
        self._click_registered = False
        self._stop = threading.Event()

        # Camera visibility and flat per-node colors (shows node boundaries)
        self.show_cameras = self.server.gui.add_checkbox(
            "Show cameras", initial_value=True
        )
        self.show_cameras.on_update(lambda _: self._apply_camera_visibility())
        self.color_by_node = self.server.gui.add_checkbox(
            "Color by node", initial_value=False
        )
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
        self.up_direction.on_update(
            lambda _: self.server.scene.set_up_direction(self.up_direction.value)
        )
        self.server.scene.set_up_direction("-y")
        self.server.gui.add_button("Reset view").on_click(lambda _: self._reset_view())

    ########################################################################
    # Scene nodes (upsert by name)
    ########################################################################

    def add_points(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 0.003,
    ) -> None:
        """
        Upsert a named point cloud.

        Args:
            name: scene node name.
            points: (N, 3) positions.
            colors: (N, 3) uint8 colors; replaced by the node color while "Color by node" is on.
            point_size: dot size, world units.
        """
        shown = (
            self._flat_color(name, len(points)) if self.color_by_node.value else colors
        )
        handle = self.server.scene.add_point_cloud(
            name, points=points, colors=shown, point_size=point_size
        )
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
        cam_to_world = invert_poses(pose64)
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
        self,
        name: str,
        segments: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        line_width: float = 2.0,
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
        self.server.scene.add_line_segments(
            name, points=segments, colors=colors, line_width=line_width
        )

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        textured: Optional[trimesh.Trimesh] = None,
    ) -> None:
        """
        Upsert a named mesh; picking, labels and the heat overlay index its vertices.

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
        self.heat_shifts.pop(name, None)

        # Display the texture when given, else the vertex colors
        shown = textured

        if shown is None:
            shown = trimesh.Trimesh(
                vertices, faces, vertex_colors=colors, process=False
            )

        self.server.scene.add_mesh_trimesh(name, shown)

    def add_label_list(
        self,
        name: str,
        words: Sequence[str],
        weights: np.ndarray,
        on_select: Callable[[Optional[str]], None],
        top_n: int = 20,
    ) -> None:
        """
        GUI buttons for a mesh's heaviest words; a click hands the word to `on_select`.

        - "Clear" hands None; re-adding replaces the mesh's earlier list
        - each button shows its weight rounded, e.g. `grass (~28.7k)`

        Args:
            name: mesh node name; titles the folder.
            words: candidate words.
            weights: (len(words),) weight per word, e.g. summed probability over vertices.
            on_select: called with the clicked word, or None from "Clear".
            top_n: words listed, heaviest first.
        """
        # Heaviest words first
        order = np.argsort(-np.asarray(weights), kind="stable")[:top_n]

        # Replace an earlier list for this mesh
        if name in self.label_lists:
            self.label_lists[name][0].remove()

        folder = self.server.gui.add_folder(f"Labels: {name}")

        with folder:
            clear = self.server.gui.add_button("Clear")
            clear.on_click(lambda _: on_select(None))
            buttons = [clear]

            for i in order:
                word = str(words[i])
                button = self.server.gui.add_button(
                    f"{word} (~{_compact_count(float(weights[i]))})"
                )
                button.on_click(lambda _, word=word: on_select(word))
                buttons.append(button)

        self.label_lists[name] = (folder, buttons)

    def show_heat(
        self,
        name: str,
        scores: Optional[np.ndarray],
        floor: float,
        *,
        offset: float = 0.25,
    ) -> None:
        """
        Overlay a per-vertex score on a mesh as a vertex-colored `<name>/heat` sub-mesh.

        - faces with any vertex at or above `floor` are drawn; colors blend across each face
        - faces touching a NaN vertex are skipped, so no heat blends onto vertices without data
        - viridis over the drawn vertices' scores, min-max normalized
        - the mesh itself is never re-sent: only the drawn sub-mesh goes over the wire
        - lift direction: whole-mesh vertex normals times median edge length, computed once per mesh

        Args:
            name: mesh node name, already added with add_mesh.
            scores: (V,) score per mesh vertex, NaN = no data: faces touching it are never drawn; None clears.
            floor: score a face needs on one vertex to be drawn.
            offset: shift along vertex normals, in median edge lengths, against z-fighting.
        """
        # Drop the previous overlay
        if name in self.heats:
            self.heats.pop(name).remove()

        if scores is None:
            return

        # Faces reaching the floor and touching no NaN vertex, remapped onto the vertices they use
        vertices, faces, _ = self.meshes[name]
        face_scores = scores[faces]
        keep = (face_scores >= floor).any(axis=1) & ~np.isnan(face_scores).any(axis=1)

        if not keep.any():
            return

        used, inverse = np.unique(faces[keep], return_inverse=True)
        sub_faces = inverse.reshape(-1, 3)

        # Viridis over the drawn scores
        rgb = apply_viridis(scores[used].astype(np.float64))

        # Whole-mesh lift direction, computed on the first overlay of this mesh
        if name not in self.heat_shifts:
            surface = trimesh.Trimesh(vertices, faces, process=False)
            self.heat_shifts[name] = (
                np.median(surface.edges_unique_length) * surface.vertex_normals
            )

        # Lift off the base surface along vertex normals
        lifted = vertices[used] + offset * self.heat_shifts[name][used]
        heat = trimesh.Trimesh(lifted, sub_faces, vertex_colors=rgb, process=False)

        self.heats[name] = self.server.scene.add_mesh_trimesh(f"{name}/heat", heat)

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
        # Register the scene-wide click handler once; mode switches must not stack it
        if not self._click_registered:
            register = self.server.scene.on_click()
            register(self._dispatch_click)
            self._click_registered = True

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

    def _pick_vertex(
        self, name: str, origin: tuple, direction: tuple
    ) -> Optional[tuple[float, int]]:
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


########################################################################
# Helpers
########################################################################


def _compact_count(value: float) -> str:
    """
    Short count for a button label: 950 -> "950", 28718 -> "28.7k".
    """
    if value < 1000:
        return f"{value:.0f}"

    return f"{value / 1000:.1f}k"


def _chart(title: str, words: list[str], probs: np.ndarray) -> str:
    """
    SVG bar chart of word probabilities, pinned to the viewport's top left.

    - x: words, labels tilted 35 degrees; y: probability on a fixed 0-1 axis
    """
    width, height, left, bottom, top = 420, 230, 34, 80, 24
    plot_h = height - bottom - top
    step = (width - left - 8) / max(len(words), 1)
    parts = [
        f'<text x="{left}" y="15" fill="#ddd" font-size="12" font-weight="bold">{title}</text>'
    ]

    # Gridlines and y ticks every 0.25
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - tick)
        parts.append(
            f'<line x1="{left}" x2="{width - 8}" y1="{y:.1f}" y2="{y:.1f}" stroke="#555" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{left - 4}" y="{y + 3:.1f}" fill="#aaa" font-size="9" text-anchor="end">{tick:.2f}</text>'
        )

    # One bar per word, its label tilted under it
    for i, (word, p) in enumerate(zip(words, probs)):
        x = left + i * step
        bar_h = plot_h * float(p)
        cx = x + step / 2
        base = top + plot_h
        parts.append(
            f'<rect x="{x + 2:.1f}" y="{base - bar_h:.1f}" width="{step - 4:.1f}" height="{bar_h:.1f}" fill="#ff5000"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{base + 10:.1f}" fill="#ddd" font-size="10" text-anchor="end" '
            f'transform="rotate(-35 {cx:.1f} {base + 10:.1f})">{word}</text>'
        )

    svg = f'<svg width="{width}" height="{height}">{"".join(parts)}</svg>'
    return (
        '<div style="position:fixed;top:12px;left:12px;z-index:10;padding:6px;'
        f'background:#1a1b1ee6;border-radius:6px">{svg}</div>'
    )


########################################################################
# Scene: mesh + optional semantics
########################################################################


def _find_stores(backend_dir: Path) -> dict[str, Path]:
    """
    Lifted stores holding mesh-vertex arrays, extractor name -> store path.

    - listed: `<backend>/semantics/*_lifted.zarr` with `vertex_word_ids` or `vertex_features`
    - a store lifted onto another mesh.ply (`mesh_sha256` differs) is stale and skipped
    """
    mesh_sha256 = hashlib.sha256((backend_dir / "mesh.ply").read_bytes()).hexdigest()
    stores = {}

    for path in sorted((backend_dir / "semantics").glob("*_lifted.zarr")):
        store = zarr.open(str(path), mode="r")

        # Points-only stores (dinov2) have nothing on the mesh
        if "vertex_word_ids" not in store and "vertex_features" not in store:
            continue

        # Stale against the mesh: skip, a semantics re-run rewrites it
        if store.attrs.get("mesh_sha256") != mesh_sha256:
            logger.warning(
                "%s was lifted onto another mesh.ply; re-run semantics with that extractor",
                path,
            )
            continue

        stores[store.attrs["extractor"]] = path

    return stores


def _split(text: str) -> list[str]:
    """
    Comma-separated entries, stripped, empties dropped.
    """
    return [part.strip() for part in text.split(",") if part.strip()]


def _word_mode(viewer: Viewer, store_path: Path) -> list:
    """
    Word query, mass-ranked label list and click probe over an ocr_lens store; returns GUI handles.

    - heat: per vertex, summed probability of the query words within its stored top-k
    - unobserved vertices (top probability 0) score NaN and probe as unobserved
    - probe: the clicked vertex's top-10 scene terms (words in some observed vertex's top-10), renormalized
    """
    store = zarr.open(str(store_path), mode="r")
    word_ids = np.asarray(store["vertex_word_ids"])
    word_probs = np.asarray(store["vertex_word_probs"], dtype=np.float32)
    words = list(store.attrs["words"])
    row_of = {word: row for row, word in enumerate(words)}
    observed = word_probs[:, 0] > 0

    # Scene terms: every word in some observed vertex's top-10
    is_term = np.zeros(len(words), dtype=bool)
    is_term[np.unique(word_ids[observed, :10])] = True

    # Probe chart, query box, floor, Search and the unknown-word note
    panel = viewer.server.gui.add_html("")
    query = viewer.server.gui.add_text("Query", initial_value="")
    floor = viewer.server.gui.add_slider(
        "Query min p", min=0.0, max=1.0, step=0.01, initial_value=0.3
    )
    search_button = viewer.server.gui.add_button("Search")
    note = viewer.server.gui.add_markdown("")
    vertices = viewer.meshes["mesh"][0]
    marker_radius = 0.005 * float(np.linalg.norm(np.ptp(vertices, axis=0)))

    def search(_=None) -> None:
        """
        Draw the summed probability of the query words as heat; faces under the floor hidden.
        """
        typed = [word.lower() for word in _split(query.value)]
        known = [word for word in typed if word in row_of]
        unknown = [word for word in typed if word not in row_of]
        note.content = f"not in vocabulary: {', '.join(unknown)}" if unknown else ""

        # Empty query clears; a word outside a vertex's top-k reads as 0 there
        scores = None

        if known:
            hit = np.isin(word_ids, [row_of[word] for word in known])
            scores = (word_probs * hit).sum(axis=1)
            scores[~observed] = np.nan

        viewer.show_heat("mesh", scores, floor.value)

    def select(word: Optional[str]) -> None:
        """
        Put a label-list word in the query box (Clear empties it) and search.
        """
        query.value = word or ""
        search()

    def show(vertex: int) -> None:
        """
        Mark the clicked vertex and chart its top-10 scene terms in the top left.
        """
        viewer.server.scene.add_icosphere(
            "/probe",
            radius=marker_radius,
            color=(255, 0, 255),
            position=vertices[vertex],
        )

        if not observed[vertex]:
            panel.content = _chart(f"vertex {vertex}: unobserved", [], np.zeros(0))
            return

        # Scene terms among its stored words, already sorted, renormalized
        ids = word_ids[vertex]
        probs = word_probs[vertex]
        keep = is_term[ids] & (probs > 0)
        ids = ids[keep]
        probs = probs[keep] / probs[keep].sum()
        panel.content = _chart(
            f"vertex {vertex}", [words[j] for j in ids[:10]], probs[:10]
        )

    # Words ranked by probability mass (expected vertex count); a click on the mesh probes
    mass = np.bincount(
        word_ids.ravel(), weights=word_probs.ravel(), minlength=len(words)
    )
    viewer.add_label_list("mesh", words, mass, select)
    search_button.on_click(search)
    viewer.on_click("mesh", show)
    return [panel, query, floor, search_button, note]


def _text_mode(viewer: Viewer, store_path: Path) -> list:
    """
    Text query GUI over a queryable extractor's lifted store; returns its handles.

    - features decoded once on entry; the extractor is built on the first Search
    - features move to the extractor's device once, so each Search copies nothing
    - unobserved vertices (all-zero raw codes) score NaN: decoded zero codes are not zero
    - both die with the handles' callbacks when the mode is switched away
    """
    store = zarr.open(str(store_path), mode="r")
    name = store.attrs["extractor"]
    kwargs = store.attrs.get("extractor_kwargs", {})
    observed = np.asarray(store["vertex_features"]).any(axis=1)
    features = torch.from_numpy(read_point_features(store_path, name="vertex_features"))
    extractor = None

    # Query, negatives (Talk2DINO's "object" convention), floor, Search
    query = viewer.server.gui.add_text("Query", initial_value="")
    negatives = viewer.server.gui.add_text("Negatives", initial_value="object")
    floor = viewer.server.gui.add_slider(
        "Query min score", min=0.0, max=1.0, step=0.01, initial_value=0.5
    )
    search_button = viewer.server.gui.add_button("Search")

    def search(_=None) -> None:
        """
        Draw the contrastive score of the query against the negatives as heat.
        """
        nonlocal extractor, features
        positives = _split(query.value)

        # Empty query clears
        if not positives:
            viewer.show_heat("mesh", None, floor.value)
            return

        if extractor is None:
            extractor = BaseFeatureExtractor.get(name)(**kwargs)
            features = features.to(extractor.device)

        with torch.no_grad():
            scores = extractor.score_queries(
                features, positives, _split(negatives.value)
            )

        scores = scores.float().cpu().numpy()
        scores[~observed] = np.nan
        viewer.show_heat("mesh", scores, floor.value)

    search_button.on_click(search)
    return [query, negatives, floor, search_button]


def _build(
    viewer: Viewer, backend_dir: Path, textured: bool, texture_size: int
) -> Optional[viser.GuiDropdownHandle]:
    """
    Add the mesh and, when vertex stores exist, the Semantics dropdown that switches modes.

    - default selection: ocr_lens when listed, else the first by name; `none` clears
    - switching clears heat and probe, removes the previous mode's GUI and handlers, builds the new one

    Args:
        viewer: viewer to populate.
        backend_dir: `<scene>/<backend>` holding mesh.ply.
        textured: show texture/mesh.obj; picks stay on mesh.ply.
        texture_size: displayed texture edge, pixels.

    Returns:
        The Semantics dropdown, or None when no store has vertex arrays.
    """
    # mesh.ply with its vertex colors, light grey when it has none
    mesh = trimesh.load(backend_dir / "mesh.ply", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    if mesh.visual.kind == "vertex":
        colors = np.array(mesh.visual.vertex_colors[:, :3])
    else:
        colors = np.full((len(vertices), 3), 200, dtype=np.uint8)

    # Corner-split OBJ of the same surface, when asked; texture downscaled for display
    shown = None

    if textured:
        shown = trimesh.load(backend_dir / "texture" / "mesh.obj", process=False)
        material = shown.visual.material
        material.image = material.image.resize(
            (texture_size, texture_size), Image.LANCZOS
        )

    viewer.add_mesh("mesh", vertices, faces, colors, textured=shown)
    stores = _find_stores(backend_dir)

    if not stores:
        return None

    initial = "ocr_lens" if "ocr_lens" in stores else next(iter(stores))
    dropdown = viewer.server.gui.add_dropdown(
        "Semantics", options=("none", *stores), initial_value=initial
    )
    handles = []

    def switch(_=None) -> None:
        """
        Tear down the current mode, then build the selected one.
        """
        # Heat, probe, label list and click handler of the previous mode
        viewer.show_heat("mesh", None, 0.0)
        viewer.server.scene.remove_by_name("/probe")
        viewer.mesh_clicks.pop("mesh", None)

        if "mesh" in viewer.label_lists:
            viewer.label_lists.pop("mesh")[0].remove()

        for handle in handles:
            handle.remove()

        handles.clear()
        pytorch_gc()

        if dropdown.value == "none":
            return

        # Word arrays select word mode, codes text mode
        path = stores[dropdown.value]
        mode = (
            _word_mode
            if "vertex_word_ids" in zarr.open(str(path), mode="r")
            else _text_mode
        )
        handles.extend(mode(viewer, path))

    dropdown.on_update(switch)
    switch()
    return dropdown


def main() -> None:
    """
    Serve a backend's mesh with its queryable semantics; blocks until interrupted.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("backend_dir", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--textured",
        action="store_true",
        help="show texture/mesh.obj over the same surface",
    )
    parser.add_argument(
        "--texture_size", type=int, default=4096, help="displayed texture edge, pixels"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    viewer = Viewer(port=args.port)
    viewer.server.gui.configure_theme(control_layout="fixed")
    _build(viewer, args.backend_dir, args.textured, args.texture_size)
    viewer.serve_forever()


if __name__ == "__main__":
    main()
