"""
Synthetic splat-training scene: random colored points in a box, cameras on a ring, analytic depth.
"""

import numpy as np

FOCAL = 60.0


def make_scene(n_views=8, height=64, width=64, n_points=200):
    """
    Build one synthetic scene.

    Returns:
        (images uint8 (n,H,W,3), world_to_cam (n,4,4), intrinsics (n,3,3), points, colors,
        depths (n,H,W)).
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, (n_points, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (n_points, 3)).astype(np.uint8)
    intrinsics = np.array([[FOCAL, 0, width / 2], [0, FOCAL, height / 2], [0, 0, 1]], dtype=np.float32)

    images, depths, world_to_cam = [], [], []
    for view in range(n_views):
        # Camera on a ring of radius 3, looking at the origin (OpenCV: +z forward)
        angle = 2 * np.pi * view / n_views
        position = np.array([3 * np.cos(angle), 0.3, 3 * np.sin(angle)], dtype=np.float32)
        forward = -position / np.linalg.norm(position)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation_c2w = np.stack([right, down, forward], axis=1)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation_c2w.T
        pose[:3, 3] = -rotation_c2w.T @ position
        world_to_cam.append(pose)

        # Paint each point as a 3x3 dot, far to near so nearer points overwrite
        image = np.zeros((height, width, 3), np.uint8)
        depth = np.zeros((height, width), np.float32)
        points_cam = (pose[:3, :3] @ points.T + pose[:3, 3:]).T
        far_to_near = np.argsort(-points_cam[:, 2])
        for idx in far_to_near:
            z = points_cam[idx, 2]
            if z <= 0:
                continue
            pixel = intrinsics[:2, :2] @ (points_cam[idx, :2] / z) + intrinsics[:2, 2]
            u, v = pixel.round().astype(int)
            if 1 <= u < width - 1 and 1 <= v < height - 1:
                image[v - 1 : v + 2, u - 1 : u + 2] = colors[idx]
                depth[v - 1 : v + 2, u - 1 : u + 2] = z
        images.append(image)
        depths.append(depth)

    intrinsics_per_view = np.stack([intrinsics] * n_views)
    return np.stack(images), np.stack(world_to_cam), intrinsics_per_view, points, colors, np.stack(depths)
