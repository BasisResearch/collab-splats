"""
Tests for collab_splats.geometry.photometric: sample selection and the brightness residual.
"""

from __future__ import annotations

import pytest
import torch

# Photometric deps: the module imports them at load, so their absence skips the module
pp = pytest.importorskip("pypose")
pytest.importorskip("bae.autograd.function")

from collab_splats.geometry.photometric import photometric_residual, photometric_samples


def _scene(n_frames=2, H=16, W=24):
    """
    Smooth texture seen by identity cameras at depth 2, with a centered f=50 K.
    """
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    gray = (torch.sin(u / 3.0) + torch.cos(v / 4.0)).expand(n_frames, H, W).clone()
    depth = torch.full((n_frames, H, W), 2.0, dtype=torch.float64)
    K = torch.tensor([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]], dtype=torch.float64)
    w2c = torch.eye(4, dtype=torch.float64).expand(n_frames, 4, 4).clone()
    return w2c, gray, depth, K.expand(n_frames, 3, 3).clone()


def _residual(samples, w2c):
    """
    photometric_residual over every sample, at the given world-to-cam poses.
    """
    poses = pp.mat2SE3(w2c).tensor()
    return photometric_residual(
        poses[samples["i_idx"]],
        poses[samples["j_idx"]],
        samples["x_i"],
        samples["K_j"],
        samples["I_i"],
        samples["I_j"],
        samples["gx_j"],
        samples["gy_j"],
        samples["uv_j"],
        samples["weight"],
    )


def test_identity_pose_pair_has_zero_residual():
    """
    Same image under the same pose: every sample keeps its brightness.
    """
    torch.manual_seed(0)
    w2c, gray, depth, K = _scene()
    samples = photometric_samples(w2c, gray, depth, K, n_samples=32)
    assert samples is not None

    residual = _residual(samples, w2c)

    assert residual.shape == (len(samples["i_idx"]), 3)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=0)


def test_shifted_pose_residual_is_the_linearized_brightness_change():
    """
    Moving frame 1 along x gives I_i - (I_j + gx_j * du), du = f * tx / z, signed by pair direction.
    """
    torch.manual_seed(0)
    w2c, gray, depth, K = _scene()
    samples = photometric_samples(w2c, gray, depth, K, n_samples=32)
    assert samples is not None

    # Frame 1 moves tx along x after sampling: f=50 at z=2 shifts its pixels by 25 * tx
    tx = 0.01
    moved = w2c.clone()
    moved[1, 0, 3] = tx
    residual = _residual(samples, moved)

    # Pairs into frame 1 shift +du, pairs out of frame 1 shift -du
    du = 25.0 * tx * (samples["j_idx"] - samples["i_idx"]).to(torch.float64)
    linearized = samples["I_j"] + samples["gx_j"] * du
    expected = (samples["I_i"] - linearized) * samples["weight"]

    assert residual.shape == (len(samples["i_idx"]), 3)
    assert bool(torch.isfinite(residual).all())
    assert float(residual[:, 0].abs().max()) > 0
    torch.testing.assert_close(residual[:, 0], expected, atol=1e-9, rtol=1e-6)


def test_occluded_samples_are_dropped():
    """
    Samples whose target depth disagrees with the carried depth never survive.
    """
    torch.manual_seed(0)
    w2c, gray, depth, K = _scene()
    W = depth.shape[-1]

    # Frame 1 sees a nearer surface over its left half
    depth[1, :, : W // 2] = 1.0
    samples = photometric_samples(w2c, gray, depth, K, n_samples=128)
    assert samples is not None

    assert len(samples["uv_j"]) > 0
    assert bool((samples["uv_j"][:, 0] >= W / 2 - 1).all())


def test_too_few_samples_returns_none():
    """
    Fewer than 8 surviving samples yields None.
    """
    torch.manual_seed(0)
    w2c, gray, depth, K = _scene()

    # Two frames at 3 pixels each give at most 6 samples
    assert photometric_samples(w2c, gray, depth, K, n_samples=3) is None


def test_same_seed_draws_same_samples_and_new_seed_differs():
    """
    The pixel draw repeats under one seed and changes under another, whatever the global RNG.
    """
    w2c, gray, depth, K = _scene()

    # Global RNG moved between calls; only the seed argument may matter
    torch.manual_seed(0)
    first = photometric_samples(w2c, gray, depth, K, n_samples=32, seed=3)
    torch.manual_seed(1)
    again = photometric_samples(w2c, gray, depth, K, n_samples=32, seed=3)
    other = photometric_samples(w2c, gray, depth, K, n_samples=32, seed=4)

    assert first.keys() == again.keys()

    for key in first:
        assert torch.equal(first[key], again[key]), key

    # Another seed draws other pixels
    same_draw = first["x_i"].shape == other["x_i"].shape and torch.equal(first["x_i"], other["x_i"])
    assert not same_draw


def test_samples_read_pixel_centers_at_integer_coordinates():
    """
    Pixel-center K: a drawn pixel's coordinate is its index, so its brightness is the pixel value itself.
    """
    w2c, gray, depth, K = _scene()
    samples = photometric_samples(w2c, gray, depth, K, n_samples=32)
    assert samples is not None

    # Project each source point back through K: lands on an integer pixel index
    x_i, K_i = samples["x_i"], K[samples["i_idx"]]
    uvw = torch.einsum("mij,mj->mi", K_i, x_i)
    uv = uvw[:, :2] / uvw[:, 2:3]
    torch.testing.assert_close(uv, uv.round(), atol=1e-9, rtol=0)

    # Brightness is the raw pixel at that index, not a blend of neighbors
    u, v = uv.round().long().unbind(1)
    torch.testing.assert_close(samples["I_i"], gray[samples["i_idx"], v, u], atol=1e-9, rtol=0)

    # Identity poses carry each pixel onto itself in frame j
    torch.testing.assert_close(samples["uv_j"], uv, atol=1e-9, rtol=0)
    torch.testing.assert_close(samples["I_j"], gray[samples["j_idx"], v, u], atol=1e-9, rtol=0)
