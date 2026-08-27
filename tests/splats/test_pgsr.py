"""
PGSR multi-view machinery: forward-backward reprojection noise and plane-homography patch NCC.
"""

import math

import torch

from collab_splats.splats.pgsr import forward_backward_noise, patch_ncc, pixel_rays

# Everything here is analytic geometry on tiny images, so run it on the CPU in double precision:
# a correct round trip is then exact to ~1e-6 pixels and any sign or ordering slip is O(1) pixels.
DTYPE = torch.float64


########################################
# Synthetic scene helpers
########################################


def _intrinsics(fx, fy, cx, cy):
    """
    A (1, 3, 3) pinhole intrinsics matrix.
    """
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=DTYPE)[None]


def _rotation_y(degrees):
    """
    A (3, 3) rotation about the camera's y axis.
    """
    cos, sin = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return torch.tensor([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=DTYPE)


def _pose(rotation, centre):
    """
    A (1, 4, 4) `world_to_cam` from a camera rotation and a camera centre, `t = -R C`.
    """
    pose = torch.eye(4, dtype=DTYPE)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ centre
    return pose[None]


def _plane_depth_map(height, width, intrinsics, world_to_cam, plane_normal, plane_offset):
    """
    Analytic z-depth of the world plane `n . x = c` as one camera sees it -> (1, H, W, 1).

    - With `x_cam = R x + t` and `m = R n`: `n . x = m . x_cam - m . t`, and `x_cam = z * ray`,
      so `z = (c + m . t) / (m . ray)`.
    """
    rays = pixel_rays(height, width, intrinsics)
    rotation = world_to_cam[0, :3, :3]
    translation = world_to_cam[0, :3, 3]
    normal_cam = rotation @ plane_normal
    numerator = plane_offset + normal_cam @ translation
    return numerator / (rays * normal_cam).sum(dim=-1, keepdim=True)


def _plane_pair(near_centre=(0.35, 0.05, -0.05), near_degrees=8.0):
    """
    Two views of one world plane, each with the analytically exact depth map.

    - The plane sits 3 units down the NEIGHBOUR's optical axis, so the neighbour's depth map is
      exactly constant and the round trip's bilinear lookup into it carries no interpolation error.
      A plane depth `c / (n . ray)` is not bilinear in the pixel grid, and interpolating one would
      put ~1e-2 px of its own curvature into the measurement.
    - The same plane is slanted in the reference view, which is the view whose rotation and pixel
      ordering the round trip actually has to get right.
    """
    height, width = 24, 32
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)

    # Reference at the world origin; the neighbour rotated and translated off to the side
    ref_pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    rotation_near = _rotation_y(near_degrees)
    centre_near = torch.tensor(near_centre, dtype=DTYPE)
    near_pose = _pose(rotation_near, centre_near)

    # Plane normal = the neighbour's optical axis in world; offset puts it 3 units in front of it
    normal = rotation_near.transpose(-1, -2) @ torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    offset = normal @ (centre_near + 3.0 * normal)

    ref_depth = _plane_depth_map(height, width, intrinsics, ref_pose, normal, offset)
    near_depth = _plane_depth_map(height, width, intrinsics, near_pose, normal, offset)
    return ref_depth, ref_pose, near_depth, near_pose, intrinsics


def _texture_at(u, v):
    """
    A smooth, non-constant grey texture sampled at arbitrary continuous pixel coordinates.
    """
    return 0.5 + 0.25 * torch.sin(0.45 * u) + 0.2 * torch.cos(0.4 * v) + 0.12 * torch.sin(0.3 * (u - v))


def _pixel_centres(height, width, dtype=DTYPE):
    """
    (H, W) grids of pixel-centre `u` and `v` coordinates, `i + 0.5`.
    """
    u = torch.arange(width, dtype=dtype) + 0.5
    v = torch.arange(height, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return grid_u, grid_v


def _texture_image(height, width):
    """
    The texture rendered onto a pixel grid -> (1, 1, H, W).
    """
    grid_u, grid_v = _pixel_centres(height, width)
    return _texture_at(grid_u, grid_v)[None, None]


def _interior_pixels(height, width, margin, stride=4):
    """
    (M, 2) pixel centres on a coarse grid, kept `margin` pixels clear of every border.
    """
    u = torch.arange(margin, width - margin, stride, dtype=DTYPE) + 0.5
    v = torch.arange(margin, height - margin, stride, dtype=DTYPE) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)


def _homography_scene(distance_scale=1.0):
    """
    A slanted plane seen by a translated neighbour whose image is the exact homography warp.

    - The reference camera is the world frame, so the neighbour pose IS the relative pose.
    - `distance_scale` fakes a wrong accumulated plane distance without changing the images.
    """
    height, width = 48, 64
    intrinsics = _intrinsics(60.0, 60.0, 31.7, 23.3)

    # Reference at the origin; the neighbour translated sideways and rotated slightly inwards
    ref_pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    near_pose = torch.eye(4, dtype=DTYPE)[None]
    near_pose[0, :3, :3] = _rotation_y(4.0)
    near_pose[0, :3, 3] = torch.tensor([0.4, 0.03, 0.0], dtype=DTYPE)

    # PGSR's plane convention: the normal faces the camera, so the plane is `n . x_cam = -d`
    normal = torch.tensor([0.15, -0.1, -1.0], dtype=DTYPE)
    normal = normal / normal.norm()
    distance = 3.0

    # The homography that carries a reference pixel onto its neighbour pixel for that plane
    relative_rotation = near_pose[0, :3, :3]
    relative_translation = near_pose[0, :3, 3]
    plane_term = relative_translation[:, None] @ normal[None, :] / (distance * distance_scale)
    homography = intrinsics[0] @ (relative_rotation - plane_term) @ torch.linalg.inv(intrinsics[0])

    # Build the neighbour image by pulling the texture back through the homography
    grid_u, grid_v = _pixel_centres(height, width)
    pixels_near = torch.stack([grid_u, grid_v, torch.ones_like(grid_u)], dim=-1)
    pixels_ref = pixels_near @ torch.linalg.inv(homography).transpose(-1, -2)
    pixels_ref = pixels_ref[..., :2] / pixels_ref[..., 2:]
    near_gray = _texture_at(pixels_ref[..., 0], pixels_ref[..., 1])[None, None]

    return _texture_image(height, width), near_gray, ref_pose, near_pose, intrinsics, normal, distance


########################################
# forward_backward_noise
########################################


def test_forward_backward_noise_round_trips_a_consistent_plane():
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair()

    noise, valid = forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics)

    # A wrong pixel-grid order or a transposed rotation shows up here as O(10) pixels of noise
    assert noise.shape == (24 * 32,)
    assert valid.float().mean() > 0.3
    # 1e-6 is the epsilon floor under forward_backward_noise's root; a swapped grid order or a
    # transposed rotation lands at O(10) pixels, so the threshold is nowhere near that floor
    assert noise[valid].max() < 1e-5


def test_forward_backward_noise_rejects_points_outside_the_neighbour():
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair(near_centre=(10.0, 0.0, 0.0), near_degrees=0.0)

    _, valid = forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics)

    assert not valid.any()


def test_forward_backward_noise_flows_gradient_to_both_depth_maps():
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair()

    # Bias the neighbour depth so the noise is strictly positive; `norm` has no gradient at zero
    ref_leaf = ref_depth.clone().requires_grad_(True)
    near_leaf = (near_depth * 1.05).requires_grad_(True)

    noise, _ = forward_backward_noise(ref_leaf, ref_pose, intrinsics, near_leaf, near_pose, intrinsics)
    noise.sum().backward()

    assert ref_leaf.grad is not None and torch.isfinite(ref_leaf.grad).all() and ref_leaf.grad.abs().sum() > 0
    assert near_leaf.grad is not None and torch.isfinite(near_leaf.grad).all() and near_leaf.grad.abs().sum() > 0


########################################
# patch_ncc
########################################


def test_patch_ncc_is_zero_for_identical_views():
    height, width = 48, 64
    intrinsics = _intrinsics(60.0, 60.0, 31.7, 23.3)
    pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    gray = _texture_image(height, width)
    pixels = _interior_pixels(height, width, margin=8)

    # Same pose both sides: the homography collapses to K K^-1 = I for any plane
    normal = torch.tensor([0.15, -0.1, -1.0], dtype=DTYPE).expand(len(pixels), 3)
    distance = torch.full((len(pixels),), 3.0, dtype=DTYPE)

    ncc, mask = patch_ncc(gray, gray, pixels, normal, distance, pose, intrinsics, pose, intrinsics)

    # K @ inv(K) is the identity only to float round-off, so the residual NCC is ~1e-7, not 0
    assert ncc.shape == (len(pixels), 1)
    assert ncc.max() < 1e-5
    assert mask.all()


def test_patch_ncc_flows_gradient_to_normal_and_distance():
    ref_gray, near_gray, ref_pose, near_pose, intrinsics, normal, distance = _homography_scene()
    pixels = _interior_pixels(48, 64, margin=16)

    # Start off the true plane so the NCC sits away from its clamped floor
    normal_leaf = normal.expand(len(pixels), 3).clone().requires_grad_(True)
    distance_leaf = torch.full((len(pixels),), distance * 1.3, dtype=DTYPE).requires_grad_(True)

    ncc, _ = patch_ncc(
        ref_gray, near_gray, pixels, normal_leaf, distance_leaf, ref_pose, intrinsics, near_pose, intrinsics
    )
    ncc.sum().backward()

    assert normal_leaf.grad is not None and torch.isfinite(normal_leaf.grad).all()
    assert normal_leaf.grad.abs().sum() > 0
    assert distance_leaf.grad is not None and torch.isfinite(distance_leaf.grad).all()
    assert distance_leaf.grad.abs().sum() > 0


def test_patch_ncc_prefers_the_true_plane_over_a_wrong_one():
    ref_gray, near_gray, ref_pose, near_pose, intrinsics, normal, distance = _homography_scene()
    pixels = _interior_pixels(48, 64, margin=16)
    normal_batch = normal.expand(len(pixels), 3)

    # The correct plane reproduces the warp that built the neighbour image
    correct, correct_mask = patch_ncc(
        ref_gray,
        near_gray,
        pixels,
        normal_batch,
        torch.full((len(pixels),), distance, dtype=DTYPE),
        ref_pose,
        intrinsics,
        near_pose,
        intrinsics,
    )

    # Doubling the plane distance halves the parallax term, so the patches decorrelate
    wrong, _ = patch_ncc(
        ref_gray,
        near_gray,
        pixels,
        normal_batch,
        torch.full((len(pixels),), 2.0 * distance, dtype=DTYPE),
        ref_pose,
        intrinsics,
        near_pose,
        intrinsics,
    )

    # Measured: ~2e-5 for the true plane against ~0.74 for the doubled one
    assert correct.mean() < 1e-3
    assert wrong.mean() > 0.3
    assert correct_mask.all()
