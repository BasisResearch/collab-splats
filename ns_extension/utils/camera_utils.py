#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

## Taken from https://github.com/myuito3/splatfacto-360.
# Take from https://github.com/brian-xu/scaffold-gs-nerfstudio


import math
from typing import List, Optional, Tuple
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras

####################################################
####### Functions taken from scaffold-gs ###########
####################################################

class ColmapCamera:
    def __init__(
        self,
        R,
        T,
        fovx,
        fovy,
        image_width,
        image_height,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        self.R = R
        self.T = T
        self.fovx = fovx
        self.fovy = fovy

        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(get_world2view_transform(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            get_projection_matrix(
                znear=self.znear, zfar=self.zfar, fovx=self.fovx, fovy=self.fovy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def get_world2view_transform(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    c2w = torch.linalg.inv(Rt)
    cam_center = c2w[:3, 3]
    cam_center = (cam_center + translate) * scale
    c2w[:3, 3] = cam_center
    Rt = torch.linalg.inv(c2w)
    return Rt

def get_projection_matrix(znear, zfar, fovx, fovy):
    """
    Returns an OpenGL projection matrix
    """
    tanhalf_fovy = math.tan((fovy / 2))
    tanhalf_fovx = math.tan((fovx / 2))

    top = tanhalf_fovy * znear
    bottom = -top
    right = tanhalf_fovx * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def convert_to_colmap_camera(camera: Cameras):
    # NeRF 'transform_matrix' is a camera-to-world transform
    c2w = torch.eye(4).to(camera.camera_to_worlds)
    c2w[:3, :] = camera.camera_to_worlds[0]
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1

    # get the world-to-camera transform and set R, T
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3].T  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())
    fovx = focal2fov(K[0, 0, 0], W)
    fovy = focal2fov(K[0, 1, 1], H)

    return ColmapCamera(R=R, T=T, fovx=fovx, fovy=fovy, image_height=H, image_width=W)

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

########################################
#### RaDe-GS functions for camera ######
########################################

def depth_double_to_normal(camera: Cameras, depth1: torch.Tensor, depth2: torch.Tensor):
    """Convert two depth maps to normal maps using camera parameters.

    Args:
        camera (Cameras): Camera object containing intrinsics and other parameters
        depth1 (torch.Tensor): First depth map
        depth2 (torch.Tensor): Second depth map

    Returns:
        torch.Tensor: Normal maps derived from the depth maps, shape (2, H, W, 3)
    """
    points1, points2 = _depths_double_to_points(camera, depth1, depth2)
    return _point_double_to_normal(points1, points2)

def _depths_double_to_points(camera: Cameras, depthmap1: torch.Tensor, depthmap2: torch.Tensor):
    """Convert two depth maps to 3D points using camera parameters.

    Args:
        camera (Cameras): Camera object containing intrinsics and other parameters
        depthmap1 (torch.Tensor): First depth map
        depthmap2 (torch.Tensor): Second depth map

    Returns:
        tuple(torch.Tensor, torch.Tensor): Two sets of 3D points in camera space,
            each with shape (3, H_scaled, W_scaled)
    """

    colmap_camera: ColmapCamera = convert_to_colmap_camera(camera)
    W, H = colmap_camera.image_width, colmap_camera.image_height
    fx = W / (2 * math.tan(colmap_camera.fovx / 2.0))
    fy = H / (2 * math.tan(colmap_camera.fovy / 2.0))

    intrinsics_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    
    # Create pixel coordinate grid (adding 0.5 to get center of pixels)
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')

    # Stack coordinates and reshape to proper format
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).permute(1, 2, 0)
    points = points.reshape(3, -1).float().cuda()

    # Calculate ray directions by multiplying inverse intrinsics with pixel coordinates
    rays_d = intrinsics_inv @ points

    # Scale rays by depth to get 3D points
    points1 = depthmap1.reshape(1,-1) * rays_d
    points2 = depthmap2.reshape(1,-1) * rays_d

    # Reshape points to final format (3, H, W)
    points1 = points1.reshape(H,W,3).permute(2,0,1)
    points2 = points2.reshape(H,W,3).permute(2,0,1)
    
    return points1, points2

def _point_double_to_normal(points1: torch.Tensor, points2: torch.Tensor):
    """Calculate normal maps from two sets of 3D points using cross products of spatial derivatives.

    Args:
        points1 (torch.Tensor): First set of 3D points, shape (3, H, W)
        points2 (torch.Tensor): Second set of 3D points, shape (3, H, W)

    Returns:
        torch.Tensor: Normal maps derived from the points, shape (2, H, W, 3)
    """
    # Stack points along first dimension
    points = torch.stack([points1, points2],dim=0)
    output = torch.zeros_like(points)
    
    # Calculate spatial derivatives using central differences
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]  # x direction derivatives
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]  # y direction derivatives
    
    # Calculate normal vectors using cross product and normalize
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    
    # Insert normal vectors into output tensor (excluding borders)
    output[...,1:-1, 1:-1] = normal_map
    
    # Return with dimensions rearranged to (2, H, W, 3)
    return output.permute(0, 2, 3, 1)

########################################
#### Taken from dn-splatter ############
########################################

# opengl to opencv transformation matrix
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# ndc space is x to the right y up. uv space is x to the right, y down.
def pix2ndc_x(x, W):
    x = x.float()
    return (2 * x) / W - 1


def pix2ndc_y(y, H):
    y = y.float()
    return 1 - (2 * y) / H

# ndc is y up and x right. uv is y down and x right
def ndc2pix_x(x, W):
    return (x + 1) * 0.5 * W


def ndc2pix_y(y, H):
    return (1 - y) * 0.5 * H


def euclidean_to_z_depth(
    depths: torch.Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Convert euclidean depths to z_depths given camera intrinsics"""
    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
    image_coords = get_camera_coords(img_size=img_size)
    image_coords = image_coords.to(device)

    z_depth = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    z_depth[:, 0] = (image_coords[:, 0] - cx) / fx  # x
    z_depth[:, 1] = (image_coords[:, 1] - cy) / fy  # y
    z_depth[:, 2] = 1  # z

    z_depth = z_depth / torch.norm(z_depth, dim=-1, keepdim=True)
    z_depth = (z_depth * depths)[:, 2]  # pick only z component

    z_depth = z_depth[..., None]
    z_depth = z_depth.view(img_size[1], img_size[0], 1)

    return z_depth

def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> torch.Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords


def get_means3d_backproj(
    depths: torch.Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: torch.Tensor,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def project_pix(
    p: torch.Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    c2w: torch.Tensor,
    device: torch.device,
    return_z_depths: bool = False,
) -> torch.Tensor:
    """Projects a world 3D point to uv coordinates using intrinsics/extrinsics

    Returns:
        uv coords
    """
    if c2w is None:
        c2w = torch.eye((p.shape[0], 4, 4), device=device)  # type: ignore
    if c2w.device != device:
        c2w = c2w.to(device)

    points_cam = (p.to(device) - c2w[..., :3, 3]) @ c2w[..., :3, :3]
    u = points_cam[:, 0] * fx / points_cam[:, 2] + cx  # x
    v = points_cam[:, 1] * fy / points_cam[:, 2] + cy  # y
    if return_z_depths:
        return torch.stack([u, v, points_cam[:, 2]], dim=-1)
    return torch.stack([u, v], dim=-1)


def get_colored_points_from_depth(
    depths: torch.Tensor,
    rgbs: torch.Tensor,
    c2w: torch.Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

    Returns:
        Tuple of (points, colors)
    """
    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        colors = rgbs.view(-1, 3)[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)


def get_rays_x_y_1(H, W, focal, c2w):
    """Get ray origins and directions in world coordinates.

    Convention here is (x,y,-1) such that depth*rays_d give real z depth values in world coordinates.
    """
    assert c2w.shape == torch.Size([3, 4])
    image_coords = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="ij",
    )
    i, j = image_coords
    # dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim = -1)
    dirs = torch.stack(
        [(pix2ndc_x(i, W)) / focal, pix2ndc_y(j, H) / focal, -torch.ones_like(i)],
        dim=-1,
    )
    dirs = dirs.view(-1, 3)
    rays_d = dirs[..., :] @ c2w[:3, :3]
    rays_o = c2w[:3, -1].expand_as(rays_d)

    # return world coordinate rays_o and rays_d
    return rays_o, rays_d


