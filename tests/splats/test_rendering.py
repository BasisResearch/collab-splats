"""
render_view: output shape contract for both primitives; 3DGS normals face the camera.
"""

import pytest
import torch

from collab_splats.splats.rendering import gaussian_normals_in_camera_frame, render_view

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def _gaussians(n_points=200, device="cuda"):
    gen = torch.Generator().manual_seed(0)
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.rand(n_points, 3, generator=gen) - 0.5),
            "scales": torch.nn.Parameter(torch.full((n_points, 3), -3.0)),
            "quats": torch.nn.Parameter(torch.rand(n_points, 4, generator=gen)),
            "opacities": torch.nn.Parameter(torch.zeros(n_points)),
            "sh0": torch.nn.Parameter(torch.rand(n_points, 1, 3, generator=gen)),
            "shN": torch.nn.Parameter(torch.zeros(n_points, 15, 3)),
        }
    )
    return gaussians.to(device)


def _camera(device="cuda"):
    cam_to_world = torch.eye(4, device=device)[None]
    cam_to_world[0, 2, 3] = -4.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_render_view_shapes(primitive):
    cam_to_world, intrinsics = _camera()
    render, info = render_view(primitive, _gaussians(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert render["alpha"].shape == (1, 64, 64, 1)
    assert render["depth"].shape == (1, 64, 64, 1)
    assert render["normal"].shape == (1, 64, 64, 3)
    assert render["depth_normal"].shape == (1, 64, 64, 3)
    assert ("distortion" in render) == (primitive == "2dgs")
    expected_gradient_key = "means2d" if primitive == "3dgs" else "gradient_2dgs"
    assert expected_gradient_key in info


@cuda
def test_render_view_3dgs_without_normals_omits_normal_keys():
    cam_to_world, intrinsics = _camera()
    render, info = render_view(
        "3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False, render_normals=False
    )
    assert set(render) == {"rgb", "alpha", "depth"}
    assert render["rgb"].shape == (1, 64, 64, 3)
    assert "render_extra_signals" not in info or info["render_extra_signals"] is None
    assert "means2d" in info


@cuda
def test_gaussian_normals_face_the_camera():
    gaussians = _gaussians()
    cam_to_world, _ = _camera()
    world_to_cam = torch.linalg.inv(cam_to_world)[0]
    scales = torch.exp(gaussians["scales"])
    normals = gaussian_normals_in_camera_frame(gaussians["quats"], scales, gaussians["means"], world_to_cam)

    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    means_cam = gaussians["means"] @ rotation_w2c.T + translation_w2c
    normal_lengths = normals.norm(dim=-1)
    facing = (normals * means_cam).sum(-1)
    assert normals.shape == (200, 3)
    assert torch.allclose(normal_lengths, torch.ones(200, device="cuda"), atol=1e-5)
    assert (facing <= 1e-6).all()


def _flat_disc(device="cuda"):
    # One large flat disc at the origin: local z (the 2DGS normal axis, and the thin 3DGS axis) is rotated
    # 90 degrees about y onto world +x, so both primitives agree the normal is world +x; opaque
    gaussians = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(1, 3)),
            "scales": torch.nn.Parameter(torch.tensor([[-1.0, -1.0, -8.0]])),
            "quats": torch.nn.Parameter(torch.tensor([[0.7071068, 0, 0.7071068, 0]])),
            "opacities": torch.nn.Parameter(torch.full((1,), 10.0)),
            "sh0": torch.nn.Parameter(torch.ones(1, 1, 3)),
            "shN": torch.nn.Parameter(torch.zeros(1, 15, 3)),
        }
    )
    return gaussians.to(device)


def _rotated_camera(device="cuda"):
    # Camera on the +x axis at distance 4, rotated 90 degrees about y so OpenCV +z looks at the origin
    cam_to_world = torch.zeros(1, 4, 4, device=device)
    cam_to_world[0, :3, 0] = torch.tensor([0.0, 0, 1])
    cam_to_world[0, :3, 1] = torch.tensor([0.0, 1, 0])
    cam_to_world[0, :3, 2] = torch.tensor([-1.0, 0, 0])
    cam_to_world[0, :3, 3] = torch.tensor([4.0, 0, 0])
    cam_to_world[0, 3, 3] = 1.0
    intrinsics = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)[None]
    return cam_to_world, intrinsics


@cuda
@pytest.mark.parametrize("primitive", ["3dgs", "2dgs"])
def test_normals_are_camera_frame_under_rotated_camera(primitive):
    # The disc normal is world +x, which a 90-degree camera rotation maps to camera -z; a world-frame
    # normal leaking through would read as camera +x instead
    cam_to_world, intrinsics = _rotated_camera()
    render, _ = render_view(primitive, _flat_disc(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    centre_normal = render["normal"][0, 32, 32]
    centre_depth_normal = render["depth_normal"][0, 32, 32]
    assert render["alpha"][0, 32, 32, 0] > 0.5
    unit_normal = torch.nn.functional.normalize(centre_normal, dim=-1)
    unit_depth_normal = torch.nn.functional.normalize(centre_depth_normal, dim=-1)
    expected = torch.tensor([0.0, 0, -1], device="cuda")
    assert unit_normal[2] < -0.9
    assert torch.allclose(unit_normal, expected, atol=0.05)
    assert torch.allclose(unit_depth_normal, expected, atol=0.05)


@cuda
def test_2dgs_render_carries_median_depth_and_its_normal():
    cam_to_world, intrinsics = _camera()
    render, _info = render_view("2dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    assert render["median_depth"].shape == render["depth"].shape
    assert render["depth_normal_median"].shape == render["depth_normal"].shape
    assert render["median_depth"].requires_grad

    # Content, not just plumbing: over a cloud the median depth is not the alpha-weighted
    # expectation, and its finite difference is not a second copy of the expected depth normal
    assert (render["median_depth"] > 0).any()
    assert not torch.equal(render["median_depth"], render["depth"])
    assert not torch.allclose(render["depth_normal_median"], render["depth_normal"])


@cuda
def test_2dgs_median_depth_equals_expected_depth_on_a_single_surface():
    # One opaque disc face-on at distance 4: exactly one Gaussian per ray, so the median depth IS the
    # expected depth and the distortion map is 0 there. Both rasterizer outputs are (C,H,W,1) and both
    # differentiable, so only their values separate median_depth from the distortion map next to it.
    cam_to_world, intrinsics = _rotated_camera()
    render, _info = render_view("2dgs", _flat_disc(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    assert render["alpha"][0, 32, 32, 0] > 0.5
    assert render["depth"][0, 32, 32, 0].item() == pytest.approx(4.0, rel=1e-3)
    assert render["median_depth"][0, 32, 32, 0].item() == pytest.approx(4.0, rel=1e-3)
    assert render["distortion"][0, 32, 32, 0].item() == pytest.approx(0.0, abs=1e-6)


@cuda
def test_2dgs_without_normals_omits_the_finite_differenced_depth_normals():
    # The rasterizer's own normal and median depth come free and stay; only depth_to_normal is skipped
    cam_to_world, intrinsics = _camera()
    render, _info = render_view(
        "2dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False, render_normals=False
    )
    assert set(render) == {"rgb", "alpha", "depth", "median_depth", "normal", "distortion"}


@cuda
def test_3dgs_render_has_no_median_depth():
    cam_to_world, intrinsics = _camera()
    render, _info = render_view("3dgs", _gaussians(), cam_to_world, intrinsics, 64, 64, sh_degree=0, absgrad=False)
    assert "median_depth" not in render
