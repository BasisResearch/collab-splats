import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, List

if TYPE_CHECKING:
    import torch
    from matplotlib.axes import Axes
    from collab_splats.semantics.features import BaseFeatureExtractor

# Main visualization code - adaptation of your original
MESH_KWARGS = {
    "scalars": "RGB",
    "rgb": True,
}

PCD_KWARGS = MESH_KWARGS.copy()
PCD_KWARGS.update({
    "render_points_as_spheres": True,
    "point_size": 0.5,
    "ambient": 0.3,
    "diffuse": 0.8,
    "specular": 0.1,
})


VIZ_KWARGS = {
    "position": (2, 2, 1),
    "focal_point": (0, 0, 0),
    "view_up": (0, 0, 1),
    "azimuth": 235,
    "elevation": 15,
    "zoom": 0.9,
    "lighting": [
        {"position": (10, 10, 10), "intensity": 0.8},
        {"position": (-10, -10, 10), "intensity": 0.4},
        {"position": (0, 0, -10), "intensity": 0.2},
    ],
}

CAMERA_KWARGS = {
    "scale": 0.02,
    "aspect_ratio": 1.33,
    "fov": 60,
    "line_width": 1,
    "opacity": 0.6,
    "n_poses": 3,
    "color": "red",
}


# ── Semantic Visualization ──────────────────────────────────────────────────


def compute_heatmap(
    image: np.ndarray,
    sim_map: Union["torch.Tensor", np.ndarray],
    alpha: float = 0.5,
    colormap: str = "viridis",
) -> np.ndarray:
    import torch

    if isinstance(sim_map, torch.Tensor):
        sim_map = sim_map.detach().cpu().numpy()
    sim_map = np.squeeze(sim_map)  # (H,W)

    h, w = image.shape[:2]
    if sim_map.shape != (h, w):
        import cv2

        sim_map = cv2.resize(sim_map, (w, h), interpolation=cv2.INTER_LINEAR)

    eps = 1e-8
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + eps)

    cmap = plt.get_cmap(colormap)
    heatmap_rgb = (cmap(sim_map)[:, :, :3] * 255).astype(np.uint8)

    blended = (1 - alpha) * image.astype(float) + alpha * heatmap_rgb.astype(float)
    return np.clip(blended, 0, 255).astype(np.uint8)


def plot_heatmap(
    heatmap: np.ndarray,
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots()
    ax.imshow(heatmap)
    if title:
        ax.set_title(title)
    ax.axis("off")
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")
    elif created_fig:
        plt.show()


def plot_context_segmentation(
    ref_image,
    ref_mask,
    tgt_image,
    pred_mask,
    alpha: float = 0.45,
):
    """Side-by-side colored overlay for in-context segmentation results.

    Reference + context mask shown in red; target + predicted mask shown in green.
    Returns the Figure — caller decides whether to show or save.

    Args:
        ref_image: Reference image as numpy (H, W, 3) uint8 or PIL Image.
        ref_mask: Binary context mask as numpy bool or torch bool tensor (H, W).
        tgt_image: Target image as numpy (H, W, 3) uint8 or PIL Image.
        pred_mask: Predicted binary mask as numpy bool or torch bool tensor (H, W).
        alpha: Overlay opacity (default 0.45).
    """
    import torch as _torch
    from PIL import Image as _Image

    def _to_np_image(img):
        # Convert PIL or numpy image to uint8 numpy array
        if isinstance(img, _Image.Image):
            return np.array(img.convert("RGB"))
        return np.asarray(img)

    def _to_np_mask(mask, ref_shape):
        # Normalize mask to bool numpy array; resize to match image if needed
        if isinstance(mask, _torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).squeeze().astype(bool)
        if mask.shape != ref_shape[:2]:
            resized = _Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (ref_shape[1], ref_shape[0]), resample=_Image.NEAREST
            )
            mask = np.array(resized) > 0
        return mask

    def _overlay(image_np, mask_np, color, alpha):
        # Blend colored overlay onto masked pixels
        out = image_np.astype(np.float32).copy()
        color_arr = np.array(color, dtype=np.float32) * 255.0
        out[mask_np] = (1.0 - alpha) * out[mask_np] + alpha * color_arr
        return np.clip(out, 0, 255).astype(np.uint8)

    ref_np = _to_np_image(ref_image)
    tgt_np = _to_np_image(tgt_image)
    ref_mask_np = _to_np_mask(ref_mask, ref_np.shape)
    pred_mask_np = _to_np_mask(pred_mask, tgt_np.shape)

    ref_overlay = _overlay(ref_np, ref_mask_np, color=(0.95, 0.25, 0.2), alpha=alpha)
    tgt_overlay = _overlay(tgt_np, pred_mask_np, color=(0.15, 0.8, 0.35), alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(ref_overlay)
    axes[0].set_title("Reference + context mask")
    axes[0].axis("off")
    axes[1].imshow(tgt_overlay)
    axes[1].set_title("Target + prediction")
    axes[1].axis("off")
    return fig


def query_heatmap(
    image: np.ndarray,
    text: str,
    extractor: "BaseFeatureExtractor",
    alpha: float = 0.5,
    colormap: str = "viridis",
) -> np.ndarray:
    from PIL import Image

    pil_image = Image.fromarray(image)
    text_emb = extractor.encode_text([text])
    features = extractor.forward([pil_image])
    sim_map = extractor.compute_similarity(features[0], text_emb)
    return compute_heatmap(image, sim_map, alpha=alpha, colormap=colormap)


def pca_to_rgb(
    features: "torch.Tensor",
    image: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Project (C, pH, pW) feature tensor to RGB via PCA, blend over image.

    Args:
        features: Feature tensor of shape (C, pH, pW).
        image: Original image (H, W, 3) uint8.
        alpha: Blend weight for PCA overlay (0=image only, 1=PCA only).

    Returns:
        Blended image (H, W, 3) uint8.
    """
    import torch
    from sklearn.decomposition import PCA
    import cv2

    if isinstance(features, torch.Tensor):
        feat_np = features.detach().cpu().float().numpy()  # (C, pH, pW)
    else:
        feat_np = features

    C, pH, pW = feat_np.shape
    flat = feat_np.reshape(C, -1).T  # (pH*pW, C)

    pca = PCA(n_components=3)
    projected = pca.fit_transform(flat)  # (pH*pW, 3)

    # Normalize each channel to [0, 1]
    for i in range(3):
        col = projected[:, i]
        col_min, col_max = col.min(), col.max()
        projected[:, i] = (col - col_min) / (col_max - col_min + 1e-8)

    pca_img = projected.reshape(pH, pW, 3)  # (pH, pW, 3)
    pca_img = (pca_img * 255).astype(np.uint8)

    H, W = image.shape[:2]
    pca_resized = cv2.resize(pca_img, (W, H), interpolation=cv2.INTER_LINEAR)

    blended = (1 - alpha) * image.astype(float) + alpha * pca_resized.astype(float)
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_masked_image(
    image: np.ndarray,
    sim_map: Union["torch.Tensor", np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """Mask image to black where similarity is below threshold.

    Args:
        image: Original image (H, W, 3) uint8.
        sim_map: Similarity map (H, W) or (H, W, 1), float in [0, 1].
        threshold: Pixels with sim < threshold are set to black.

    Returns:
        Masked image (H, W, 3) uint8.
    """
    import torch
    import cv2

    if isinstance(sim_map, torch.Tensor):
        sim_map = sim_map.detach().cpu().numpy()
    sim_map = np.squeeze(sim_map)  # (H, W)

    H, W = image.shape[:2]
    if sim_map.shape != (H, W):
        sim_map = cv2.resize(sim_map, (W, H), interpolation=cv2.INTER_LINEAR)

    mask = sim_map >= threshold  # (H, W) bool
    result = image.copy()
    result[~mask] = 0
    return result


def overlay_masks(
    image: np.ndarray,
    masks: "torch.Tensor",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay segmentation masks on image with distinct colors per mask.

    Args:
        image: Original image (H, W, 3) uint8.
        masks: Binary masks (N, H, W) float32, one per detected object.
        alpha: Blend weight for mask colors (0=image only, 1=colors only).

    Returns:
        Blended image (H, W, 3) uint8.
    """
    import torch
    import cv2

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()  # (N, H, W)
    else:
        masks_np = masks

    N, mH, mW = masks_np.shape
    H, W = image.shape[:2]

    cmap = plt.get_cmap("tab20")
    base = image.astype(float)
    overlay = base.copy()

    for i in range(N):
        mask = masks_np[i]  # (mH, mW)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        color = np.array(cmap(i % 20)[:3]) * 255  # RGB in [0,255]
        overlay[mask > 0.5] = (
            (1 - alpha) * base[mask > 0.5] + alpha * color
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _resolve_mesh_kwargs(mesh: pv.PolyData, mesh_kwargs: dict) -> dict:
    """Return mesh_kwargs, auto-detecting RGB pointcloud mode when kwargs are empty."""
    if mesh_kwargs:
        return mesh_kwargs
    if (
        isinstance(mesh, pv.PolyData)
        and "RGB" in mesh.point_data
        and mesh.point_data["RGB"].ndim == 2
        and mesh.point_data["RGB"].shape[1] == 3
    ):
        return PCD_KWARGS
    return {}


# ── 3D Visualization ────────────────────────────────────────────────────────


def visualize_splat(
    mesh: Union[str, pv.PolyData],
    aligned_cameras: Optional[List[np.ndarray]] = None,
    mesh_kwargs: dict = {},
    camera_kwargs: dict = {},
    viz_kwargs: dict = {},
    out_fn: Optional[str] = None,
):
    """
    Visualize point cloud with camera frustums using PyVista

    Args:
        mesh: Path to a PLY file or a PyVista PolyData to visualize.
        aligned_cameras: List of (4,4) world-to-camera matrices (OpenCV convention,
            e.g. FeedforwardResult.extrinsics). Passed directly to
            create_camera_frustum_pyvista, which inverts internally.
    """
    plotter = pv.Plotter()

    # Either a mesh or a point cloud
    if isinstance(mesh, str):
        mesh = pv.read(mesh)

    mesh_kwargs = _resolve_mesh_kwargs(mesh, mesh_kwargs)
    plotter.add_mesh(mesh, **mesh_kwargs)

    # Work on a copy so the caller's dict is not mutated across calls
    cam_kw = dict(camera_kwargs)
    if aligned_cameras is not None:
        n_poses = cam_kw.pop("n_poses", 3)
        scale = cam_kw.pop("scale", 0.02)
        aspect_ratio = cam_kw.pop("aspect_ratio", 1.33)
        fov = cam_kw.pop("fov", 60)
        cmap = plt.get_cmap("viridis")

        for i in range(0, len(aligned_cameras), n_poses):
            pose = aligned_cameras[i]
            frustum = create_camera_frustum_pyvista(pose, scale=scale, aspect_ratio=aspect_ratio, fov=fov)

            # Per-frustum color (copy cam_kw so color override doesn't leak between iterations)
            kw = dict(cam_kw)
            if "color" not in kw:
                kw["color"] = cmap(i / max(1, len(aligned_cameras)))[:3]

            plotter.add_mesh(frustum, **kw)

    # Set a specific camera position:
    # camera_position = [camera_location, focal_point, view_up]
    plotter.camera_position = [
        viz_kwargs.get("position", (2, 2, 1)),  # Camera location
        viz_kwargs.get("focal_point", (0, 0, 0)),  # Look-at point (focal point)
        viz_kwargs.get("view_up", (0, 0, 1)),  # View-up vector
    ]

    # Rotate camera
    plotter.camera.azimuth = viz_kwargs.get(
        "azimuth", 235
    )  # Rotate 45° horizontally around focal point
    plotter.camera.elevation = viz_kwargs.get(
        "elevation", 15
    )  # Rotate 30° vertically around focal point

    # Adjust zoom (zoom > 1 zooms in, < 1 zooms out)
    plotter.camera.Zoom(viz_kwargs.get("zoom", 0.9))  # 1.5x zoom in

    # Enhanced lighting
    for light in viz_kwargs.get("lighting", []):
        plotter.add_light(pv.Light(**light))

    if out_fn is not None:
        plotter.screenshot(
            filename=out_fn,
            window_size=viz_kwargs.get("window_size", [3000, 3000]),
            scale=viz_kwargs.get("scale", 1),
            transparent_background=viz_kwargs.get("transparent_background", True),
            return_img=viz_kwargs.get("return_img", False),
        )

    return plotter


def create_camera_frustum_pyvista(pose, scale=0.02, aspect_ratio=1.33, fov=60):
    """Create a camera frustum wireframe in world space.

    Args:
        pose: (4, 4) float32 world-to-camera matrix (OpenCV convention).
            Matches FeedforwardResult.extrinsics[i] directly — no inversion needed.
        scale: Controls overall frustum size (near = scale*0.1, far = scale*5).
        aspect_ratio: Width / height of the image plane.
        fov: Vertical field of view in degrees.
    """
    fov_rad = np.radians(fov)
    near = scale * 0.1
    far = scale * 5.0

    near_height = 2 * near * np.tan(fov_rad / 2)
    near_width = near_height * aspect_ratio
    far_height = 2 * far * np.tan(fov_rad / 2)
    far_width = far_height * aspect_ratio

    # Frustum in camera space: apex at origin, camera looks in +Z (OpenCV)
    vertices = np.array(
        [
            # Apex (camera centre)
            [0, 0, 0],
            # Near plane corners (+Z)
            [-near_width / 2, -near_height / 2, near],
            [ near_width / 2, -near_height / 2, near],
            [ near_width / 2,  near_height / 2, near],
            [-near_width / 2,  near_height / 2, near],
            # Far plane corners (+Z)
            [-far_width / 2, -far_height / 2, far],
            [ far_width / 2, -far_height / 2, far],
            [ far_width / 2,  far_height / 2, far],
            [-far_width / 2,  far_height / 2, far],
        ],
        dtype=np.float64,
    )

    lines = []
    # Apex → near corners
    for i in range(1, 5):
        lines.extend([2, 0, i])
    # Apex → far corners
    for i in range(5, 9):
        lines.extend([2, 0, i])
    # Near plane rectangle (closed)
    lines.extend([5, 1, 2, 3, 4, 1])
    # Far plane rectangle (closed)
    lines.extend([5, 5, 6, 7, 8, 5])
    # Near → far edges
    for i in range(4):
        lines.extend([2, i + 1, i + 5])

    frustum = pv.PolyData(vertices, lines=lines)

    # Transform camera-space vertices to world space via c2w = inv(w2c)
    c2w = np.linalg.inv(pose)
    pts_h = np.column_stack([frustum.points, np.ones(len(frustum.points))])
    frustum.points = (c2w @ pts_h.T).T[:, :3]

    return frustum


def o3d_mesh_to_polydata(mesh: "o3d.geometry.TriangleMesh") -> pv.PolyData:
    """Convert Open3D TriangleMesh to PyVista PolyData with RGB vertex scalars.

    Compatible with visualize_splat when mesh has vertex colors (MESH_KWARGS applies).
    """
    import open3d as o3d  # optional heavy dep — imported here to avoid top-level dependency
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    pv_faces = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int32), faces]
    ).ravel()
    pd = pv.PolyData(verts, pv_faces)
    if mesh.has_vertex_colors():
        pd["RGB"] = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    return pd


def pointcloud_to_polydata(pts3d: np.ndarray, **point_data) -> pv.PolyData:
    """Convert pts3d + named scalar arrays to a PyVista PolyData.

    Args:
        pts3d: (P, 3) float32 world-space XYZ
        **point_data: named scalar arrays to attach as PyVista point arrays.
            e.g. RGB=colors, features=feat_arr, similarity=scores
    """
    cloud = pv.PolyData(pts3d.copy())
    for k, v in point_data.items():
        cloud[k] = v
    return cloud
