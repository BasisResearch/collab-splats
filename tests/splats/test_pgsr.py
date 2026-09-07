"""
PGSR multi-view machinery: reprojection noise, patch NCC, the neighbor render, and the tunables
taken as keyword arguments rather than module-level constants.
"""

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.splats import pgsr
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.pgsr import (
    forward_backward_noise,
    patch_ncc,
    pixel_rays,
    plane_depth,
    project,
    render_neighbor,
    sample_at_pixels,
    select_near_views,
    unproject,
)
from collab_splats.splats.trainer import SplatsConfig

# gsplat's rasterization kernels are CUDA-only
# - `device="cpu"`: NotImplementedError, "Could not run 'gsplat::projection_ewa_3dgs_fused_fwd'"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")

# CPU float64 for everything but the `render_neighbor` pair
# - analytic geometry on tiny images
# - correct round trip exact to ~1e-6 px; any sign or ordering slip is O(1) px
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


def _pose(rotation, center):
    """
    A (1, 4, 4) `world_to_cam` from a camera rotation and a camera center, `t = -R C`.
    """
    pose = torch.eye(4, dtype=DTYPE)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ center
    return pose[None]


def _plane_depth_map(height, width, intrinsics, world_to_cam, plane_normal, plane_offset):
    """
    Analytic z-depth of the world plane `n . x = c` as one camera sees it -> (1, H, W, 1).

    - `x_cam = R x + t`, `m = R n`: `n . x = m . x_cam - m . t`
    - `x_cam = z * ray`, so `z = (c + m . t) / (m . ray)`
    """
    rays = pixel_rays(height, width, intrinsics)
    rotation = world_to_cam[0, :3, :3]
    translation = world_to_cam[0, :3, 3]
    normal_cam = rotation @ plane_normal
    numerator = plane_offset + normal_cam @ translation
    return numerator / (rays * normal_cam).sum(dim=-1, keepdim=True)


def _plane_pair(near_center=(0.35, 0.05, -0.05), near_degrees=8.0):
    """
    Two views of one world plane, each with the analytically exact depth map.

    - plane 3 units down the NEIGHBOR's optical axis: its depth map is exactly constant, so the
      round trip's bilinear lookup carries no interpolation error
    - `c / (n . ray)` is not bilinear in the pixel grid; interpolating one adds ~1e-2 px of curvature
    - the same plane is slanted in the reference view, whose rotation and pixel ordering the round
      trip has to get right
    """
    height, width = 24, 32
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)

    # Reference at the world origin; the neighbor rotated and translated off to the side
    ref_pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    rotation_near = _rotation_y(near_degrees)
    center_near = torch.tensor(near_center, dtype=DTYPE)
    near_pose = _pose(rotation_near, center_near)

    # Plane normal = the neighbor's optical axis in world; offset puts it 3 units in front of it
    normal = rotation_near.transpose(-1, -2) @ torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    offset = normal @ (center_near + 3.0 * normal)

    ref_depth = _plane_depth_map(height, width, intrinsics, ref_pose, normal, offset)
    near_depth = _plane_depth_map(height, width, intrinsics, near_pose, normal, offset)
    return ref_depth, ref_pose, near_depth, near_pose, intrinsics


def _look_at(center):
    """
    The (3, 3) `world_to_cam` rotation of a camera at `center` looking at the world origin.
    """
    forward = -center / center.norm()
    right = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE), forward)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    return torch.stack([right, down, forward], dim=0)


def _ring_poses(angles_degrees, radius=4.0):
    """
    (N, 4, 4) `world_to_cam` for cameras on a ring in the xz plane, each looking at the origin.

    - two cameras subtend their ring separation at the origin, so a test can name the angle
      `select_near_views` scores on rather than reverse-engineering it
    """
    poses = []
    for degrees in angles_degrees:
        radians = math.radians(degrees)
        center = torch.tensor([radius * math.sin(radians), 0.0, -radius * math.cos(radians)], dtype=DTYPE)
        poses.append(_pose(_look_at(center), center))
    return torch.cat(poses)


def _origin_cluster(n_points=64, spread=0.005):
    """
    (P, 3) points in a tight cube at the world origin, co-visible from every ring camera.
    """
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(n_points, 3, generator=generator, dtype=DTYPE) - 0.5) * (2 * spread)


def _texture_at(u, v):
    """
    A smooth, non-constant gray texture sampled at arbitrary continuous pixel coordinates.
    """
    return 0.5 + 0.25 * torch.sin(0.45 * u) + 0.2 * torch.cos(0.4 * v) + 0.12 * torch.sin(0.3 * (u - v))


def _pixel_centers(height, width, dtype=DTYPE):
    """
    (H, W) grids of pixel-center `u` and `v` coordinates, `i + 0.5`.
    """
    u = torch.arange(width, dtype=dtype) + 0.5
    v = torch.arange(height, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return grid_u, grid_v


def _texture_image(height, width):
    """
    The texture rendered onto a pixel grid -> (1, 1, H, W).
    """
    grid_u, grid_v = _pixel_centers(height, width)
    return _texture_at(grid_u, grid_v)[None, None]


def _interior_pixels(height, width, margin, stride=4):
    """
    (M, 2) pixel centers on a coarse grid, kept `margin` pixels clear of every border.
    """
    u = torch.arange(margin, width - margin, stride, dtype=DTYPE) + 0.5
    v = torch.arange(margin, height - margin, stride, dtype=DTYPE) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)


def _homography_scene(distance_scale=1.0):
    """
    A slanted plane seen by a translated neighbor whose image is the exact homography warp.

    - reference camera is the world frame, so the neighbor pose IS the relative pose
    - `distance_scale`: fakes a wrong accumulated plane distance without changing the images
    """
    height, width = 48, 64
    intrinsics = _intrinsics(60.0, 60.0, 31.7, 23.3)

    # Reference at the origin; the neighbor translated sideways and rotated slightly inwards
    ref_pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    near_pose = torch.eye(4, dtype=DTYPE)[None]
    near_pose[0, :3, :3] = _rotation_y(4.0)
    near_pose[0, :3, 3] = torch.tensor([0.4, 0.03, 0.0], dtype=DTYPE)

    # PGSR's plane convention: the normal faces the camera, so the plane is `n . x_cam = -d`
    normal = torch.tensor([0.15, -0.1, -1.0], dtype=DTYPE)
    normal = normal / normal.norm()
    distance = 3.0

    # The homography that carries a reference pixel onto its neighbor pixel for that plane
    relative_rotation = near_pose[0, :3, :3]
    relative_translation = near_pose[0, :3, 3]
    plane_term = relative_translation[:, None] @ normal[None, :] / (distance * distance_scale)
    homography = intrinsics[0] @ (relative_rotation - plane_term) @ torch.linalg.inv(intrinsics[0])

    # Build the neighbor image by pulling the texture back through the homography
    grid_u, grid_v = _pixel_centers(height, width)
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
    # 1e-6 is the epsilon under the root; a swapped grid order or transposed rotation lands at O(10)
    assert noise[valid].max() < 1e-5


def test_forward_backward_noise_rejects_points_outside_the_neighbor():
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair(near_center=(10.0, 0.0, 0.0), near_degrees=0.0)

    _, valid = forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics)

    assert not valid.any()


def test_forward_backward_noise_rejects_points_inside_the_neighbors_depth_floor():
    """
    A point in frame but nearer than 0.1 is rejected by the depth arm, not by the bounds arms.
    """
    height, width = 24, 32
    intrinsics = _intrinsics(40.0, 42.0, 16.0, 12.0)
    ref_pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))

    # Reference sees a plane at 0.2; the neighbor sits 0.05 short of it, facing the same way
    # - every point is IN FRONT of the neighbor, just inside the floor upstream refuses
    near_pose = _pose(torch.eye(3, dtype=DTYPE), torch.tensor([0.0, 0.0, 0.15], dtype=DTYPE))
    ref_depth = torch.full((1, height, width, 1), 0.2, dtype=DTYPE)
    near_depth = torch.full((1, height, width, 1), 0.05, dtype=DTYPE)

    _, valid = forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics)

    # The bounds arms alone keep a central block, so dropping the depth arm is visible here
    pixels, points_near_cam = project(unproject(ref_depth, ref_pose, intrinsics), near_pose, intrinsics)
    in_frame = (pixels[:, 0] > 0) & (pixels[:, 0] < width) & (pixels[:, 1] > 0) & (pixels[:, 1] < height)
    assert int(in_frame.sum()) > 0
    assert 0.0 < float(points_near_cam[:, 2].max()) < 0.1
    assert not valid.any()


def test_forward_backward_noise_flows_gradient_to_both_depth_maps():
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair()

    # Bias the neighbor depth so the noise is strictly positive; `norm` has no gradient at zero
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

    # The correct plane reproduces the warp that built the neighbor image
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


########################################
# Tunables as keyword arguments
########################################


def _module_level_bindings(source: str) -> list[str]:
    """
    Every name bound by a module-level assignment, sorted.

    - both assignment nodes: a tunable can come back annotated, `MIN_DEPTH: float = 1e-6`
    - every target of a chained `Assign`, not just first or last: `THETA0 = logger = 5.0`
    - every `Name` under a target, so unpacking counts: `THETA0, MIN_DEPTH = 5.0, 1e-6`
    """
    module = ast.parse(source)
    return sorted(
        name.id
        for node in module.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    )


def test_the_module_keeps_no_tunables_as_module_level_constants():
    # Tunables live on the functions that read them, not on the module
    # - checks BINDINGS, not text: a substring search misses a rename and fires on prose
    # - equality is tighter than the rule: ANY new binding trips it, so add it here, don't loosen
    assert _module_level_bindings(Path(pgsr.__file__).read_text()) == ["logger"]


@pytest.mark.parametrize(
    "statement, planted",
    (
        ("THETA0 = 5.0", "THETA0"),
        ("MIN_DEPTH: float = 1e-6", "MIN_DEPTH"),
        ("THETA0, MIN_DEPTH = 5.0, 1e-6", "MIN_DEPTH"),
        ("logger = THETA0 = 5.0", "THETA0"),
        ("THETA0 = logger = 5.0", "THETA0"),
    ),
)
def test_the_module_binding_check_sees_a_planted_constant(statement, planted):
    # One case per node shape the helper walks
    # - `Assign`, `AnnAssign`, unpacking target, and both ends of a chained `Assign`
    # - `node.targets[:1]` kills only `logger = THETA0`; `node.targets[-1:]` only `THETA0 = logger`
    # - positive assertion: `!= ["logger"]` passes under either truncation, so it cannot discriminate
    assert planted in _module_level_bindings(f"logger = 1\n{statement}\n")


def test_the_module_binding_check_returns_its_names_sorted():
    # Pins the sort, which nothing above can see
    # - three names: sorted order is neither source order, its reverse, nor reverse-sorted
    # - two would not do it: reversing a 2-list lands back on sorted order
    # - `logger` ASCII-sorts behind the upper-case names but case-folds ahead: kills `key=str.lower`
    assert _module_level_bindings("logger = 1\nMIN_DEPTH = 1e-6\nTHETA0 = 5.0\n") == ["MIN_DEPTH", "THETA0", "logger"]


def test_select_near_views_takes_its_scoring_shape_as_keyword_only_arguments():
    signature = inspect.signature(select_near_views)

    assert signature.parameters["theta0"].default == 5.0
    assert signature.parameters["sigma_below"].default == 1.0
    assert signature.parameters["sigma_above"].default == 10.0
    for name in ("theta0", "sigma_below", "sigma_above"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_select_near_views_refuses_its_scoring_shape_positionally():
    poses = _ring_poses([0.0, 5.0])
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7).expand(2, 3, 3)

    # An eighth positional argument is theta0; without the bare `*` it would silently bind
    with pytest.raises(TypeError, match="takes from 5 to 7 positional arguments but 8 were given"):
        select_near_views(poses, intrinsics, _origin_cluster(), 24, 32, 1, 1000, 5.0)


def test_plane_depth_and_the_two_projections_take_their_floors_as_keyword_only_arguments():
    floors = {
        plane_depth: ("min_cosine", 1e-4),
        project: ("min_depth", 1e-6),
        forward_backward_noise: ("min_depth", 1e-6),
    }

    for function, (name, default) in floors.items():
        parameter = inspect.signature(function).parameters[name]
        assert parameter.default == default, function.__name__
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function.__name__


def test_plane_depth_refuses_its_floor_positionally():
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE).expand(1, 24, 32, 3)
    distance = torch.full((1, 24, 32, 1), 3.0, dtype=DTYPE)

    with pytest.raises(TypeError, match="takes 3 positional arguments but 4 were given"):
        plane_depth(normal, distance, intrinsics, 1e-4)


def test_project_refuses_its_floor_positionally():
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)
    pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))

    with pytest.raises(TypeError, match="takes 3 positional arguments but 4 were given"):
        project(torch.tensor([[1.0, 2.0, 4.0]], dtype=DTYPE), pose, intrinsics, 1e-6)


def test_forward_backward_noise_refuses_its_floor_positionally():
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair()

    # Six is the whole positional call; a seventh binds `min_depth` without the bare `*`
    with pytest.raises(TypeError, match="takes 6 positional arguments but 7 were given"):
        forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics, 1e-6)


def test_plane_depth_divides_by_the_ray_cosine_and_floors_it_at_min_cosine():
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)

    # A fronto-parallel plane: the normal faces the camera, so -(n . ray) is exactly 1 everywhere
    normal = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE).expand(1, 24, 32, 3)
    distance = torch.full((1, 24, 32, 1), 3.0, dtype=DTYPE)

    depth = plane_depth(normal, distance, intrinsics)
    floored = plane_depth(normal, distance, intrinsics, min_cosine=1e3)

    # The default floor is 1e-4, four orders below the cosine here, so it cannot bite
    assert torch.equal(depth, distance)
    # A floor above every cosine replaces the divisor outright, which a hardcoded 1e-4 would not
    assert torch.allclose(floored, distance / 1e3)


def test_project_divides_by_the_camera_depth_and_floors_it_at_min_depth():
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)
    pose = _pose(torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    points = torch.tensor([[1.0, 2.0, 4.0]], dtype=DTYPE)

    pixels, points_cam = project(points, pose, intrinsics)
    floored, _ = project(points, pose, intrinsics, min_depth=1e3)

    # Identity pose: the camera-frame point is the world point and the divide is by z = 4
    assert torch.equal(points_cam, points)
    assert torch.allclose(pixels, torch.tensor([[40.0 / 4 + 15.3, 2 * 42.0 / 4 + 11.7]], dtype=DTYPE))
    # A floor above z takes over the divide, collapsing the point towards the principal point
    assert torch.allclose(floored, torch.tensor([[40.0 / 1e3 + 15.3, 2 * 42.0 / 1e3 + 11.7]], dtype=DTYPE))


def test_forward_backward_noise_threads_one_min_depth_into_both_projections(monkeypatch):
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair()
    seen = []
    real_project = pgsr.project

    def spy(points, world_to_cam, camera_matrix, **kwargs):
        seen.append(kwargs.get("min_depth"))
        return real_project(points, world_to_cam, camera_matrix, **kwargs)

    monkeypatch.setattr(pgsr, "project", spy)
    forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics, min_depth=7.5e-3)

    # Exactly two projections, each with the caller's floor
    # - forward into the neighbor, backward into the reference
    # - a site keeping its own default records None; the list also pins the count
    assert seen == [7.5e-3, 7.5e-3]


def test_forward_backward_noise_threads_one_min_depth_into_the_ray_rescale(monkeypatch):
    ref_depth, ref_pose, near_depth, near_pose, intrinsics = _plane_pair()
    real_project = pgsr.project

    # Pin both projections to the default floor
    # - the ray-rescale clamp is then the only consumer of `min_depth` left
    # - not a `project` call, so no `project` spy can see it
    def pinned(points, world_to_cam, camera_matrix, **_):
        return real_project(points, world_to_cam, camera_matrix)

    monkeypatch.setattr(pgsr, "project", pinned)
    noise, valid = forward_backward_noise(ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics)
    floored, floored_valid = forward_backward_noise(
        ref_depth, ref_pose, intrinsics, near_depth, near_pose, intrinsics, min_depth=10.0
    )

    # `valid` is decided before the rescale, so raising the floor cannot move it
    assert torch.equal(valid, floored_valid) and valid.any()
    # The default 1e-6 sits far below every depth in this scene, so the round trip still closes
    assert noise[valid].max() < 1e-5
    # Neighbor depths are ~3, so a floor of 10 takes over the divide
    # - measured: 11.63-11.83 px for every valid pixel
    assert floored[floored_valid].min() > 10.0


def test_select_near_views_scores_its_neighbors_around_theta0():
    poses = _ring_poses([0.0, 2.0, 5.0, 8.0, 40.0])
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7).expand(len(poses), 3, 3)
    points = _origin_cluster()

    default = select_near_views(poses, intrinsics, points, 24, 32, num_views=4, max_points=1000)
    wide = select_near_views(poses, intrinsics, points, 24, 32, num_views=4, max_points=1000, theta0=40.0)

    # 5 degrees is the ideal baseline at the default theta0
    # - 8 beats 2: the falloff above theta0 is ten times gentler than the one below
    # - 40 is too wide to be useful
    assert default[0] == [2, 3, 1, 4]
    # Moving the ideal angle onto the widest pair reorders the same five views
    assert wide[0][0] == 4


def test_select_near_views_penalizes_a_near_duplicate_harder_than_a_wide_baseline():
    poses = _ring_poses([0.0, 2.0, 8.0])
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7).expand(len(poses), 3, 3)
    points = _origin_cluster()

    # 2 and 8 sit the same 3 degrees either side of theta0: only the asymmetric sigmas can order them
    default = select_near_views(poses, intrinsics, points, 24, 32, num_views=2, max_points=1000)
    swapped = select_near_views(
        poses, intrinsics, points, 24, 32, num_views=2, max_points=1000, sigma_below=10.0, sigma_above=1.0
    )

    assert default[0] == [2, 1]
    assert swapped[0] == [1, 2]


########################################
# Shared pixel-grid and normalization helpers
########################################


def test_pixel_rays_are_unnormalized_directions_through_the_pixel_centers():
    height, width = 24, 32
    intrinsics = _intrinsics(40.0, 42.0, 15.3, 11.7)

    rays = pixel_rays(height, width, intrinsics)

    u = (torch.arange(width, dtype=DTYPE) + 0.5)[None, None]
    v = (torch.arange(height, dtype=DTYPE) + 0.5)[None, :, None]
    assert rays.shape == (1, height, width, 3)
    assert torch.allclose(rays[..., 0], (u - 15.3) / 40.0)
    assert torch.allclose(rays[..., 1], (v - 11.7) / 42.0)
    assert (rays[..., 2] == 1.0).all()


def test_pixel_rays_builds_its_grid_from_pixel_grid(monkeypatch):
    calls = []
    real_pixel_grid = pgsr.pixel_grid

    def spy(height, width, device, dtype=torch.float32):
        calls.append((height, width, dtype))
        return real_pixel_grid(height, width, device, dtype)

    monkeypatch.setattr(pgsr, "pixel_grid", spy)
    pixel_rays(24, 32, _intrinsics(40.0, 42.0, 15.3, 11.7))

    # A second meshgrid here is a second chance for unproject's and pixel_grid's orderings to drift
    assert calls == [(24, 32, DTYPE)]


def test_sample_at_pixels_reads_the_value_under_the_pixel_center():
    height, width = 24, 32
    grid_u, grid_v = _pixel_centers(height, width)

    # A separable ramp: the value names its pixel, so a u/v swap or half-pixel shift lands O(1) off
    image = (grid_u + 100.0 * grid_v)[None, None]

    sampled = sample_at_pixels(image, torch.tensor([[4.5, 6.5], [20.5, 2.5]], dtype=DTYPE), height, width)

    assert torch.allclose(sampled[:, 0], torch.tensor([654.5, 270.5], dtype=DTYPE))


def test_sample_at_pixels_normalizes_through_the_helper_patch_ncc_uses(monkeypatch):
    calls = []
    real_normalize = pgsr._normalize_pixels

    def spy(pixels, height, width):
        calls.append((height, width))
        return real_normalize(pixels, height, width)

    monkeypatch.setattr(pgsr, "_normalize_pixels", spy)
    sample_at_pixels(torch.zeros(1, 1, 24, 32, dtype=DTYPE), torch.tensor([[4.5, 6.5]], dtype=DTYPE), 24, 32)

    # A private copy of the same two lines is how the sampler and the patch warp drift apart
    assert calls == [(24, 32)]


def test_patch_ncc_normalizes_both_its_patches_through_the_shared_helper(monkeypatch):
    ref_gray, near_gray, ref_pose, near_pose, intrinsics, normal, distance = _homography_scene()
    pixels = _interior_pixels(48, 64, margin=16)

    # Crop the neighbor so the two grays differ in shape
    # - the recorded pairs then pin WHICH image's dimensions reached each call
    # - an equal-sized pair cannot distinguish them
    near_gray = near_gray[..., :40, :56]

    calls = []
    real_normalize = pgsr._normalize_pixels

    def spy(pixels, height, width):
        calls.append((height, width))
        return real_normalize(pixels, height, width)

    monkeypatch.setattr(pgsr, "_normalize_pixels", spy)
    patch_ncc(
        ref_gray,
        near_gray,
        pixels,
        normal.expand(len(pixels), 3),
        torch.full((len(pixels),), distance, dtype=DTYPE),
        ref_pose,
        intrinsics,
        near_pose,
        intrinsics,
    )

    # The other two of `_normalize_pixels`' three call sites, the sampler's being guarded above
    # - reference patch grid, then the warped neighbor grid
    # - list comparison pins the count: inlining either is caught, a per-call loop would not be
    assert calls == [(48, 64), (40, 56)]


########################################
# render_neighbor
########################################


def _neighbor_inputs(height=24, width=40):
    """
    A tiny CUDA `Gaussians` plus one neighbor frame, pose and camera matrix for `render_neighbor`.

    - frame 24x40, deliberately NOT square: a swapped pair is invisible on a square frame
    - `render_neighbor` reads `height, width = image.shape[:2]`, then calls
      `model.render(..., width, height, ...)`
    """
    cfg = SplatsConfig.from_dict({"max_steps": 10, "losses": {}})
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (200, 3)).astype(np.uint8)
    model = Gaussians(cfg, points, colors, scene_scale=1.0, n_views=2, device="cuda")

    # The camera sits back from the point cloud so the whole cluster lands inside the frame
    image = rng.integers(0, 255, (height, width, 3)).astype(np.uint8)
    cam_to_world = torch.eye(4)[None].cuda()
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[[40.0, 0.0, 20.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]]).cuda()
    return model, image, cam_to_world, intrinsics, torch.tensor([1]).cuda()


@cuda
def test_render_neighbor_returns_the_four_keys_the_multiview_loss_reads(monkeypatch):
    model, image, cam_to_world, intrinsics, camera_id = _neighbor_inputs()

    # Keep what the render actually returned
    # - the same dict also carries `depth`, the alpha-weighted one, at the identical shape
    # - only an identity check can tell the two apart
    captured = {}
    real_render = model.render

    def spy(*args, **kwargs):
        render, info = real_render(*args, **kwargs)
        captured.update(render)
        return render, info

    monkeypatch.setattr(model, "render", spy)

    neighbor = render_neighbor(model, image, cam_to_world, intrinsics, camera_id)

    assert set(neighbor) == {"plane_depth", "gray", "world_to_cam", "intrinsics"}
    # (1, H, W, 1) with H=24 and W=40: the width/height pair reached `model.render` in that order
    assert neighbor["plane_depth"].shape == (1, 24, 40, 1)
    # The ray-plane depth, which is the distinction this whole module exists to make
    assert neighbor["plane_depth"] is captured["plane_depth"]
    # to_gray returns (1, 1, H, W) to match torchvision's Grayscale, NOT (1, H, W, 1)
    assert neighbor["gray"].shape == (1, 1, 24, 40)
    # ITU-601 luma of the uint8 frame scaled to [0, 1], as `losses.py` builds the reference gray
    # - unscaled: the neighbor would be 255x the reference
    weights = torch.tensor([0.2989, 0.587, 0.114], device=intrinsics.device)
    frame = torch.from_numpy(image).to(intrinsics.device).float() / 255.0
    assert torch.allclose(neighbor["gray"], (frame * weights).sum(dim=-1)[None, None])
    # world_to_cam, not cam_to_world: the multi-view loss projects INTO this view
    assert torch.allclose(neighbor["world_to_cam"], torch.linalg.inv(cam_to_world))
    assert neighbor["intrinsics"] is intrinsics


@cuda
def test_render_neighbor_keeps_the_gradient_path():
    model, image, cam_to_world, intrinsics, camera_id = _neighbor_inputs()

    neighbor = render_neighbor(model, image, cam_to_world, intrinsics, camera_id)

    # Upstream does not detach the neighbor
    # - the geometric term pulls both plane depths together
    # - a detached render makes it a one-sided fit
    assert neighbor["plane_depth"].requires_grad
