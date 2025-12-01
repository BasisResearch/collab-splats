import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List

# Main visualization code - adaptation of your original
MESH_KWARGS = {
    "scalars": "RGB",
    "rgb": True,
}

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


# Main visualization code - adaptation of your original
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
        ply_path: Path to the ply file
        aligned_poses: List of 4x4 transformation matrices
    """
    # Create PyVista plotter
    plotter = pv.Plotter()

    # Either a mesh or a point cloud
    if isinstance(mesh, str):
        mesh = pv.read(mesh)

    plotter.add_mesh(mesh, **mesh_kwargs)

    # Remove scale from kwargs
    if aligned_cameras is not None:
        n_poses = camera_kwargs.pop("n_poses", 3)

        # Create and add camera frustums for every third pose
        for i in range(0, len(aligned_cameras), n_poses):
            pose = aligned_cameras[i]

            # Make a camera frustum
            frustum = create_camera_frustum_pyvista(
                pose,
                scale=camera_kwargs.get("scale", 0.02),
                aspect_ratio=camera_kwargs.get("aspect_ratio", 1.33),
                fov=camera_kwargs.get("fov", 60),
            )

            if "color" not in camera_kwargs:
                cmap = plt.get_cmap("viridis")
                color = cmap(i / max(1, len(aligned_cameras)))
                camera_kwargs["color"] = color[:3]

            # Remove non-pyvista kwargs
            remove_kwargs = ["scale", "aspect_ratio", "fov", "n_poses"]
            for key in remove_kwargs:
                camera_kwargs.pop(key, None)

            # Add to plotter with different color for each camera
            plotter.add_mesh(frustum, **camera_kwargs)

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
