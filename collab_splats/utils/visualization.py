from typing import Any, List, Optional, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.figure import Figure

from collab_splats.geometry.projection import project
from collab_splats.geometry.transforms import (
    invert_poses,
    rescale_intrinsics,
    shift_intrinsics,
    transform_points,
)

# Main visualization code - adaptation of your original
MESH_KWARGS = {
    "scalars": "RGB",
    "rgb": True,
}

PCD_KWARGS = MESH_KWARGS.copy()
PCD_KWARGS.update(
    {
        "render_points_as_spheres": True,
        "point_size": 0.5,
        "ambient": 0.3,
        "diffuse": 0.8,
        "specular": 0.1,
    }
)


VIZ_KWARGS: dict[str, Any] = {
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


def apply_viridis(sims: np.ndarray) -> np.ndarray:
    """
    Similarity scores mapped to viridis RGB, min-max normalized.

    - NaN scores (unobserved mesh vertices) are left out of the range and drawn grey

    Args:
        sims: per-element scores, (P,).

    Returns:
        RGB colors, (P, 3) uint8.
    """
    finite = np.isfinite(sims)
    s_min, s_max = (
        (sims[finite].min(), sims[finite].max()) if finite.any() else (0.0, 0.0)
    )
    normalized = (
        (sims - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(sims)
    )
    rgba = matplotlib.colormaps["viridis"](normalized)
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    rgb[~finite] = 128
    return rgb


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
    import cv2
    import torch
    from sklearn.decomposition import PCA

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
    import cv2
    import torch

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
        masks: Binary masks (N, H, W) bool or float, one per detected object.
        alpha: Blend weight for mask colors (0=image only, 1=colors only).

    Returns:
        Blended image (H, W, 3) uint8.
    """
    import cv2
    import torch

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
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        color = np.array(cmap(i % 20)[:3]) * 255  # RGB in [0,255]
        overlay[mask > 0.5] = (1 - alpha) * base[mask > 0.5] + alpha * color

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


def apply_view(plotter: pv.Plotter, viz_kwargs: dict) -> None:
    """
    Pin a plotter's camera and add lights from viz_kwargs.

    - missing camera keys fall back to the VIZ_KWARGS defaults; missing `lighting` adds none
    - view_angle is set absolutely (30 / zoom), so re-applying does not compound like Zoom
    - lights accumulate: callers re-applying to one plotter clear its lights first

    Args:
        plotter: plotter whose camera and lights are set.
        viz_kwargs: position, focal_point, view_up, azimuth, elevation, zoom and lighting.
    """
    plotter.camera_position = [
        viz_kwargs.get("position", (2, 2, 1)),
        viz_kwargs.get("focal_point", (0, 0, 0)),
        viz_kwargs.get("view_up", (0, 0, 1)),
    ]
    plotter.camera.azimuth = viz_kwargs.get("azimuth", 235)
    plotter.camera.elevation = viz_kwargs.get("elevation", 15)
    plotter.camera.view_angle = 30.0 / viz_kwargs.get("zoom", 0.9)

    for light in viz_kwargs.get("lighting", []):
        plotter.add_light(pv.Light(**light))


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
            e.g. PointcloudResult.extrinsics). Passed directly to
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
            frustum = create_camera_frustum_pyvista(
                pose, scale=scale, aspect_ratio=aspect_ratio, fov=fov
            )

            # Per-frustum color (copy cam_kw so color override doesn't leak between iterations)
            kw = dict(cam_kw)
            if "color" not in kw:
                kw["color"] = cmap(i / max(1, len(aligned_cameras)))[:3]

            plotter.add_mesh(frustum, **kw)

    # Pin the camera and add the configured lights
    apply_view(plotter, viz_kwargs)

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
            Matches PointcloudResult.extrinsics[i] directly — no inversion needed.
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
            [near_width / 2, -near_height / 2, near],
            [near_width / 2, near_height / 2, near],
            [-near_width / 2, near_height / 2, near],
            # Far plane corners (+Z)
            [-far_width / 2, -far_height / 2, far],
            [far_width / 2, -far_height / 2, far],
            [far_width / 2, far_height / 2, far],
            [-far_width / 2, far_height / 2, far],
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
    c2w = invert_poses(pose)
    frustum.points = transform_points(frustum.points, c2w)

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


# ── Reprojection ───────────────────────────────────────────────────────────


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    hw: tuple[int, int],
    *,
    radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-buffered point splat of a colored pointcloud seen from one pinhole camera.

    - points at or behind the camera (z <= 1e-6) are dropped
    - each point writes a (2 * radius + 1)^2 pixel square at its rounded projection
    - nearest point wins per pixel
    - uint8 colors are read as 0-255, anything else as 0-1

    Args:
        points: (P, 3) world points.
        colors: (P, 3) per-point RGB, uint8 0-255 or float 0-1.
        w2c: (4, 4) or (3, 4) world-to-camera transform, OpenCV convention.
        K: (3, 3) camera matrix at the output resolution.
        hw: output (height, width) in pixels.
        radius: footprint half-width in pixels.

    Returns:
        (rgb (H, W, 3) float in [0, 1] on a white background, depth (H, W) with inf where empty).
    """
    h, w = hw

    # Colors to float 0-1
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        colors = colors / 255.0
    else:
        colors = colors.astype(np.float64)

    # Project, keep points in front, round to pixel centers
    points_t = torch.as_tensor(points, dtype=torch.float64)
    w2c_t = torch.as_tensor(w2c, dtype=torch.float64)
    K_t = torch.as_tensor(K, dtype=torch.float64)
    pixels, points_cam = project(points_t, w2c_t, K_t)
    cam = points_cam.numpy()
    in_front = cam[:, 2] > 1e-6
    pixels = pixels.numpy()[in_front]
    colors = colors[in_front]
    z = cam[in_front, 2]
    pixels = np.round(pixels).astype(np.int64)
    u, v = pixels[:, 0], pixels[:, 1]

    rgb = np.ones((h, w, 3))
    depth = np.full((h, w), np.inf)
    rgb_flat = rgb.reshape(-1, 3)
    depth_flat = depth.reshape(-1)

    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            # Shifted footprint pixels inside the image
            uu = u + du
            vv = v + dv
            inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            flat = vv[inside] * w + uu[inside]
            zz = z[inside]
            cc = colors[inside]

            # Nearest write per pixel: sort by pixel then depth, keep each pixel's first entry
            order = np.lexsort((zz, flat))
            flat = flat[order]
            _, first = np.unique(flat, return_index=True)
            keep = order[first]
            flat = flat[first]

            # Replace the buffer only where the new write is closer
            closer = zz[keep] < depth_flat[flat]
            depth_flat[flat[closer]] = zz[keep][closer]
            rgb_flat[flat[closer]] = cc[keep][closer]

    return rgb, depth


def plot_reprojection(
    image: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    *,
    max_width: int = 1200,
    radius: int = 1,
    title: str | None = None,
) -> Figure:
    """
    Photo, pointcloud render from its solved camera, and a 50/50 blend in one row.

    - photo and K are downscaled together so the width is at most max_width
    - the render panel's title carries its coverage: the fraction of pixels hit by a point

    Args:
        image: (H, W, 3) uint8 RGB photo.
        points: (P, 3) world points.
        colors: (P, 3) per-point RGB, uint8 0-255 or float 0-1.
        w2c: (4, 4) or (3, 4) world-to-camera pose of the photo.
        K: (3, 3) pixel-center camera matrix of the photo at full resolution, as localize returns.
        max_width: widest displayed width in pixels.
        radius: point footprint half-width in pixels.
        title: figure suptitle; none when omitted.

    Returns:
        The three-panel figure.
    """
    h, w = image.shape[:2]

    # Downscale photo and K to max_width
    scale = min(1.0, max_width / w)
    out_w = round(w * scale)
    out_h = round(h * scale)
    photo = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    photo = photo / 255.0

    # Pixel-center K rescales via the corner convention: shift +0.5, rescale, shift -0.5
    K_scaled = shift_intrinsics(K, (0.5, 0.5))
    K_scaled = rescale_intrinsics(K_scaled, (h, w), (out_h, out_w))
    K_scaled = shift_intrinsics(K_scaled, (-0.5, -0.5))

    # Render and blend
    rgb, depth = render_points(
        points, colors, w2c, K_scaled, (out_h, out_w), radius=radius
    )
    coverage = np.isfinite(depth).mean()
    blend = 0.5 * photo + 0.5 * rgb

    # Three panels in one row
    fig, axes = plt.subplots(1, 3, figsize=(18, 18 * out_h / (3 * out_w) + 1))
    panels = [
        (photo, "photo"),
        (rgb, f"render from pose (coverage {coverage:.0%})"),
        (blend, "50/50 blend"),
    ]

    for ax, (panel, panel_title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(panel_title)
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig
