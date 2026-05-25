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
        aligned_cameras: List of 4x4 world-to-camera pose matrices.
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
    """
    Create a camera frustum using PyVista
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov)

    # Calculate frustum dimensions
    near = scale * 0.1
    far = scale * 5.0

    # Near plane dimensions
    near_height = 2 * near * np.tan(fov_rad / 2)
    near_width = near_height * aspect_ratio

    # Far plane dimensions
    far_height = 2 * far * np.tan(fov_rad / 2)
    far_width = far_height * aspect_ratio

    # Define frustum vertices
    vertices = np.array(
        [
            # Camera center (apex)
            [0, 0, 0],
            # Near plane corners
            [-near_width / 2, -near_height / 2, -near],
            [near_width / 2, -near_height / 2, -near],
            [near_width / 2, near_height / 2, -near],
            [-near_width / 2, near_height / 2, -near],
            # Far plane corners
            [-far_width / 2, -far_height / 2, -far],
            [far_width / 2, -far_height / 2, -far],
            [far_width / 2, far_height / 2, -far],
            [-far_width / 2, far_height / 2, -far],
        ]
    )

    # Define lines connecting vertices to form frustum wireframe
    lines = []
    # Lines from camera center to near plane corners
    for i in range(1, 5):
        lines.extend([2, 0, i])

    # Lines from camera center to far plane corners
    for i in range(5, 9):
        lines.extend([2, 0, i])

    # Near plane rectangle
    near_rect = [4, 1, 2, 3, 4]
    lines.extend(near_rect)

    # Far plane rectangle
    far_rect = [4, 5, 6, 7, 8]
    lines.extend(far_rect)

    # Connect near to far plane corners
    for i in range(4):
        lines.extend([2, i + 1, i + 5])

    # Create PyVista polydata for the frustum
    frustum = pv.PolyData(vertices, lines=lines)

    points = frustum.points
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    transformed_points = (pose @ points_homo.T).T
    frustum.points = transformed_points[:, :3]

    return frustum


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
