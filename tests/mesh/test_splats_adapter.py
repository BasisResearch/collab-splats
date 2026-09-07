"""
_splats_to_tsdf_inputs: ckpt.pt -> (depths, rgbs, c2w, K) with alpha as the confidence gate.
"""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from collab_splats.mesh.utils import _splats_to_tsdf_inputs
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.rendering import load_checkpoint
from collab_splats.splats.trainer import SplatsConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


def _write_ckpt(tmp_path, *, primitive="3dgs", n_views=3, height=24, width=40, camera_span=1.0):
    """
    A tiny trained-model checkpoint: 200 gaussians over `n_views` cameras looking at the origin.
    """
    rng = np.random.default_rng(0)
    cfg = SplatsConfig.from_dict({"primitive": primitive, "max_steps": 10, "losses": {}})
    model = Gaussians(
        cfg,
        rng.uniform(-0.5, 0.5, (200, 3)).astype(np.float32),
        rng.integers(0, 255, (200, 3)).astype(np.uint8),
        scene_scale=1.0,
        n_views=n_views,
        device="cpu",
    )

    # Cameras travel along x so the far-depth cut has a trajectory to measure against; a zero
    # span leaves every pose identical, which is the degenerate rig that cut must skip
    cam_to_world = torch.eye(4)[None].repeat(n_views, 1, 1)
    cam_to_world[:, 2, 3] = -3.0
    cam_to_world[:, 0, 3] = torch.linspace(0.0, camera_span, n_views)

    # A different focal per view: an adapter that broadcast one camera over the stack would
    # still return the right shape, so the values have to differ view by view
    intrinsics = torch.stack(
        [
            torch.tensor([[20.0 + view, 0.0, width / 2], [0.0, 20.0 + view, height / 2], [0.0, 0.0, 1.0]])
            for view in range(n_views)
        ]
    )

    ckpt = model.checkpoint()
    ckpt["config"] = asdict(cfg)
    ckpt["cam_to_world"] = cam_to_world
    ckpt["intrinsics"] = intrinsics
    ckpt["image_ids"] = list(range(n_views))
    ckpt["image_size"] = (height, width)
    ckpt["appearance"] = None
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    return path


def _fake_render_views(depths, *, alphas=None, rgbs=None, medians=None):
    """
    Stand-in for the rasterizer that yields caller-chosen pixels, shaped as `render_views` does.

    - `rgb` is (1, H, W, 3) in [0, 1]; `depth`, `median_depth` and `alpha` are (1, H, W, 1).
    - The real rasterizer cannot be asked for a specific depth, so the masks downstream of it
      are pinned here and the render plumbing itself is pinned by the `@cuda` tests.
    - Each call appends its arguments to `render_views.calls`, so a test can pin what the
      adapter forwarded and not only what it did with the result. A wrong pose or K renders
      pure background, which is shaped exactly like a correct render of an empty scene.
    """
    depths = np.asarray(depths, dtype=np.float32)
    alphas = np.ones_like(depths) if alphas is None else np.asarray(alphas, dtype=np.float32)
    medians = depths if medians is None else np.asarray(medians, dtype=np.float32)
    rgbs = np.zeros(depths.shape + (3,), dtype=np.float32) if rgbs is None else np.asarray(rgbs, dtype=np.float32)

    def _views():
        for view in range(len(depths)):
            yield {
                "rgb": torch.from_numpy(rgbs[view])[None],
                "depth": torch.from_numpy(depths[view])[None, ..., None],
                "median_depth": torch.from_numpy(medians[view])[None, ..., None],
                "alpha": torch.from_numpy(alphas[view])[None, ..., None],
            }

    # Recorded here rather than inside `_views`: a generator function's body does not run
    # until the caller starts iterating, so the arguments would go unrecorded on a call whose
    # result is never consumed
    def render_views(model, camera_opt, cam_to_world, intrinsics, height, width):
        render_views.calls.append(
            {
                "model": model,
                "camera_opt": camera_opt,
                "cam_to_world": cam_to_world,
                "intrinsics": intrinsics,
                "height": height,
                "width": width,
            }
        )
        return _views()

    render_views.calls = []
    return render_views


def _patch_renders(monkeypatch, render_views):
    """Swap the rasterizer the adapter imports at call time for a scripted one."""
    monkeypatch.setattr("collab_splats.splats.rendering.render_views", render_views)


def _spy_load_checkpoint(monkeypatch):
    """
    Record what `load_checkpoint` handed back, so a test can pin what got forwarded on.

    - The real loader still runs; only its return value is observed. `model` and `camera_opt`
      are opaque objects a test cannot rebuild, so identity is the only thing to assert.
    """
    loaded = {}

    def spy(ckpt_path, device):
        result = load_checkpoint(ckpt_path, device)
        loaded["model"], loaded["camera_opt"] = result[0], result[1]
        return result

    monkeypatch.setattr("collab_splats.splats.rendering.load_checkpoint", spy)
    return loaded


########
# The rendered tuple: shape, order, poses, intrinsics
########


@cuda
def test_splats_adapter_reads_a_checkpoint(tmp_path):
    ckpt_path = _write_ckpt(tmp_path)

    depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(ckpt_path)

    # 24x40, never square: `image_size` is stored (height, width) and the adapter renders
    # (width, height) — on a square checkpoint a swap in either direction passes
    assert depths.shape == (3, 24, 40)
    assert depths.dtype == np.float32
    assert rgbs.shape == (3, 24, 40, 3)
    assert rgbs.dtype == np.uint8
    assert c2w.shape == (3, 4, 4)
    assert c2w.dtype == np.float32
    assert intrinsics.shape == (3, 3, 3)
    assert intrinsics.dtype == np.float32


def test_splats_adapter_returns_the_checkpoint_poses_not_their_inverse(tmp_path, monkeypatch):
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    _patch_renders(monkeypatch, _fake_render_views(np.ones((3, 4, 5), dtype=np.float32)))
    saved = torch.load(ckpt_path, weights_only=False)["cam_to_world"].numpy()

    _depths, _rgbs, c2w, _intrinsics = _splats_to_tsdf_inputs(ckpt_path)

    # The fixture's cameras sit at z = -3, so the world-to-camera convention is not a fixed
    # point of this fixture and an inverted return is visible in the translation alone
    assert np.allclose(c2w, saved, atol=1e-6)
    assert not np.allclose(c2w, np.linalg.inv(saved), atol=1e-3)


def test_splats_adapter_returns_each_views_own_intrinsics(tmp_path, monkeypatch):
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    _patch_renders(monkeypatch, _fake_render_views(np.ones((3, 4, 5), dtype=np.float32)))

    saved = torch.load(ckpt_path, weights_only=False)["intrinsics"].numpy()

    _depths, _rgbs, _c2w, intrinsics = _splats_to_tsdf_inputs(ckpt_path)

    # The whole 3x3 goes to Open3D, so cx/cy and a transposed K have to be caught as well as
    # the focals — the fixture's principal point is (2.5, 2.0), off-centre in both axes
    assert np.allclose(intrinsics, saved, atol=1e-6)

    # Focal 20 + view in the fixture: one camera broadcast over the stack would give 20 thrice.
    # Not redundant with the line above — it pins that the fixture itself still varies by view,
    # without which `allclose` would compare a stack of identical matrices and prove nothing
    assert [float(intrinsics[view, 0, 0]) for view in range(3)] == [20.0, 21.0, 22.0]


def test_splats_adapter_stacks_views_in_render_order(tmp_path, monkeypatch):
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    depths = np.stack([np.full((4, 5), view + 1.0, dtype=np.float32) for view in range(3)])
    rgbs = np.stack([np.full((4, 5, 3), 0.2 * (view + 1), dtype=np.float32) for view in range(3)])
    _patch_renders(monkeypatch, _fake_render_views(depths, rgbs=rgbs))

    out_depths, out_rgbs, _c2w, _intrinsics = _splats_to_tsdf_inputs(ckpt_path)

    # Depth and rgb are filled in separate statements, so a permutation can hit one alone
    assert [float(out_depths[view].min()) for view in range(3)] == [1.0, 2.0, 3.0]
    assert [int(out_rgbs[view].min()) for view in range(3)] == [51, 102, 153]
    assert [int(out_rgbs[view].max()) for view in range(3)] == [51, 102, 153]


def test_splats_adapter_rounds_rgb_to_uint8_rather_than_truncating(tmp_path, monkeypatch):
    """Byte conversion truncates; the adapter rounds, and rounding goes both ways."""
    ckpt_path = _write_ckpt(tmp_path, n_views=7, height=4, width=5)
    # Eighths m/8 for m = 1..7: exact in float32, and 255m/8 never lands on a whole byte. Since
    # 255 mod 8 == 7, the product's fractional part is ((-m) mod 8)/8, which shrinks as m grows
    # and crosses a half at m = 4 — so one run spans both directions of the rule:
    #   m = 1..3 -> 31.875 / 63.75 / 95.625     round UP   -> 32 / 64 / 96
    #   m = 4    -> 127.5                       the tie    -> 128
    #   m = 5..7 -> 159.375 / 191.25 / 223.125  round DOWN -> 159 / 191 / 223
    # Both halves carry weight: truncation shows only on the up half, .ceil() only on the down
    # half. The tie is not a third direction — .ceil() agrees with round() at 127.5, so m = 4
    # only repeats the up half's catch, and it is the sole eighth whose product ties at all.
    # The 0.2/0.4/0.6 the ordering test uses land on 51/102/153 exactly and tell none apart
    rgbs = np.stack([np.full((4, 5, 3), 0.125 * (view + 1), dtype=np.float32) for view in range(7)])
    _patch_renders(monkeypatch, _fake_render_views(np.ones((7, 4, 5), dtype=np.float32), rgbs=rgbs))

    _out_depths, out_rgbs, _c2w, _intrinsics = _splats_to_tsdf_inputs(ckpt_path)

    # min and max together: a per-view fill is constant, so this pins the whole frame, not a pixel
    rounded = [32, 64, 96, 128, 159, 191, 223]
    assert [int(out_rgbs[view].min()) for view in range(7)] == rounded
    assert [int(out_rgbs[view].max()) for view in range(7)] == rounded


########
# What the adapter forwards to the rasterizer
########


def test_splats_adapter_renders_from_the_checkpoint_poses(tmp_path, monkeypatch):
    """The fused geometry has to be rendered from the trained poses, not their inverse."""
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    fake = _fake_render_views(np.ones((3, 4, 5), dtype=np.float32))
    _patch_renders(monkeypatch, fake)
    saved = torch.load(ckpt_path, weights_only=False)["cam_to_world"].numpy()

    _splats_to_tsdf_inputs(ckpt_path)

    # The returned c2w is copied straight out of the checkpoint, so it stays right even when the
    # render was posed wrongly: inverted poses render pure background, which has the same shape
    # and dtype as a correct render. Only the forwarded tensor pins this
    forwarded = fake.calls[0]["cam_to_world"].cpu().numpy()
    assert np.allclose(forwarded, saved, atol=1e-6)
    assert not np.allclose(forwarded, np.linalg.inv(saved), atol=1e-3)


def test_splats_adapter_renders_with_the_checkpoint_intrinsics(tmp_path, monkeypatch):
    """An identity K renders pure background and still returns every right shape and dtype."""
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    fake = _fake_render_views(np.ones((3, 4, 5), dtype=np.float32))
    _patch_renders(monkeypatch, fake)
    saved = torch.load(ckpt_path, weights_only=False)["intrinsics"].numpy()

    _splats_to_tsdf_inputs(ckpt_path)

    # Whole 3x3 and every view: the returned K is a separate copy of the same tensor, so a
    # substitution on the way into the rasterizer is invisible in the returned tuple
    forwarded = fake.calls[0]["intrinsics"].cpu().numpy()
    assert np.allclose(forwarded, saved, atol=1e-6)


def test_splats_adapter_renders_at_the_checkpoint_image_size(tmp_path, monkeypatch):
    """`image_size` is stored (height, width) and must reach the rasterizer that way round."""
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    fake = _fake_render_views(np.ones((3, 4, 5), dtype=np.float32))
    _patch_renders(monkeypatch, fake)

    _splats_to_tsdf_inputs(ckpt_path)

    # Non-square, so a swapped unpack cannot pass in either direction
    assert (fake.calls[0]["height"], fake.calls[0]["width"]) == (4, 5)


def test_splats_adapter_renders_with_the_checkpoint_model(tmp_path, monkeypatch):
    """The trained gaussians must reach the rasterizer, not a freshly built model."""
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    loaded = _spy_load_checkpoint(monkeypatch)
    fake = _fake_render_views(np.ones((3, 4, 5), dtype=np.float32))
    _patch_renders(monkeypatch, fake)

    _splats_to_tsdf_inputs(ckpt_path)

    assert fake.calls[0]["model"] is loaded["model"]


def test_splats_adapter_renders_with_the_checkpoint_camera_opt(tmp_path, monkeypatch):
    """Pose-opt deltas are trained state — dropping them silently re-poses every view."""
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    loaded = _spy_load_checkpoint(monkeypatch)
    fake = _fake_render_views(np.ones((3, 4, 5), dtype=np.float32))
    _patch_renders(monkeypatch, fake)

    _splats_to_tsdf_inputs(ckpt_path)

    # Separate from the model check: forwarded positionally and side by side, so a swap of the
    # two would leave a single combined assertion with nothing that pins which went where
    assert fake.calls[0]["camera_opt"] is loaded["camera_opt"]


########
# Depth source selection
########


def test_splats_adapter_rejects_an_unknown_depth_name_before_any_io(tmp_path):
    # The value check must fire on the config, not on a missing file
    with pytest.raises(ValueError, match="must be 'expected' or 'median'"):
        _splats_to_tsdf_inputs(tmp_path / "does_not_exist.pt", splat_depth="surface")


def test_splats_adapter_raises_on_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the splats stage first"):
        _splats_to_tsdf_inputs(tmp_path / "ckpt.pt")


def test_splats_adapter_refuses_median_depth_on_a_3dgs_checkpoint(tmp_path):
    ckpt_path = _write_ckpt(tmp_path, primitive="3dgs")

    with pytest.raises(ValueError, match="needs a 2dgs"):
        _splats_to_tsdf_inputs(ckpt_path, splat_depth="median")


def test_splat_depth_median_reads_the_median_render(tmp_path, monkeypatch):
    ckpt_path = _write_ckpt(tmp_path, primitive="2dgs", height=4, width=5)
    depths = np.ones((3, 4, 5), dtype=np.float32)
    _patch_renders(monkeypatch, _fake_render_views(depths, medians=depths * 3.0))

    expected, _rgbs, _c2w, _intrinsics = _splats_to_tsdf_inputs(ckpt_path, splat_depth="expected")
    median, _rgbs, _c2w, _intrinsics = _splats_to_tsdf_inputs(ckpt_path, splat_depth="median")
    assert expected[0, 0, 0] == 1.0
    assert median[0, 0, 0] == 3.0

    # Pin the default against a checkpoint that CAN render median — otherwise a flipped default
    # only trips the 2dgs guard, and that catch dies the day the fixture becomes a 2dgs run
    default, _rgbs, _c2w, _intrinsics = _splats_to_tsdf_inputs(ckpt_path)
    assert default[0, 0, 0] == 1.0  # omitted splat_depth means expected, not median


########
# Alpha gate
########


@cuda
def test_splats_adapter_alpha_gate_zeroes_empty_pixels(tmp_path):
    ckpt_path = _write_ckpt(tmp_path)

    depths, _, _, _ = _splats_to_tsdf_inputs(ckpt_path)

    # A 200-gaussian model over a 24x40 frame leaves background: those pixels must fuse as 0.
    # Two-sided, because "> 0.0" alone is satisfied by an all-background render — which is
    # exactly what a wrong pose or K produces, and would otherwise pass here
    empty_frac = float((depths == 0).mean())
    assert 0.0 < empty_frac < 1.0


def test_zero_alpha_pixels_are_always_dropped(tmp_path, monkeypatch):
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    alphas = np.linspace(0.0, 1.0, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    _patch_renders(monkeypatch, _fake_render_views(np.ones((3, 4, 5), dtype=np.float32), alphas=alphas))

    depths, _, _, _ = _splats_to_tsdf_inputs(ckpt_path, conf_percentile=None)

    assert depths.flat[0] == 0.0  # alpha == 0 at the first pixel -> no observation
    assert depths.flat[1] == 1.0  # every alpha > 0 pixel survives


def test_alpha_percentile_zeroes_the_lowest_alpha_depth(tmp_path, monkeypatch):
    ckpt_path = _write_ckpt(tmp_path, height=4, width=5)
    alphas = np.linspace(0.0, 1.0, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    _patch_renders(monkeypatch, _fake_render_views(np.ones((3, 4, 5), dtype=np.float32), alphas=alphas))

    depths, _, _, _ = _splats_to_tsdf_inputs(ckpt_path, conf_percentile=50)

    dropped = depths == 0.0
    assert 0.45 < dropped.mean() < 0.55  # global percentile over all views
    assert np.all(alphas[dropped] <= np.percentile(alphas, 50))


########
# Pre-fusion depth cuts: far-field and discontinuity
########


def test_far_depth_cut_drops_depth_past_the_camera_trajectory(tmp_path, monkeypatch):
    """Rendered depth well beyond where the cameras went is unconstrained by any view."""
    ckpt_path = _write_ckpt(tmp_path, n_views=4, height=2, width=3, camera_span=10.0)
    depths = np.full((4, 2, 3), 1.0, dtype=np.float32)
    depths[:, 0, 0] = 500.0  # blown-out background
    _patch_renders(monkeypatch, _fake_render_views(depths))

    out, _, _, _ = _splats_to_tsdf_inputs(ckpt_path, max_depth_frac=0.75)
    assert np.all(out[:, 0, 0] == 0.0)  # the far pixel went
    assert np.all(out[:, 0, 1] == 1.0)  # everything inside the trajectory stayed


def test_far_depth_cut_skipped_when_the_cameras_never_move(tmp_path, monkeypatch):
    """A zero-extent rig has no trajectory to measure against — cut it and nothing survives."""
    ckpt_path = _write_ckpt(tmp_path, n_views=4, height=2, width=3, camera_span=0.0)
    depths = np.full((4, 2, 3), 1.0, dtype=np.float32)
    _patch_renders(monkeypatch, _fake_render_views(depths))

    out, _, _, _ = _splats_to_tsdf_inputs(ckpt_path, max_depth_frac=0.75)
    assert out.max() == 1.0


def test_depth_gradient_cut_drops_both_sides_of_a_jump(tmp_path, monkeypatch):
    """A silhouette is a jump between two good surfaces; TSDF welds a tendril across it."""
    ckpt_path = _write_ckpt(tmp_path, n_views=2, height=3, width=4, camera_span=10.0)
    depths = np.full((2, 3, 4), 1.0, dtype=np.float32)
    depths[:, :, 2:] = 2.0  # step edge between columns 1 and 2
    _patch_renders(monkeypatch, _fake_render_views(depths))

    out, _, _, _ = _splats_to_tsdf_inputs(ckpt_path, max_depth_grad=0.3)
    assert np.all(out[:, :, 1] == 0.0) and np.all(out[:, :, 2] == 0.0)  # both sides go
    assert np.all(out[:, :, 0] == 1.0) and np.all(out[:, :, 3] == 2.0)  # the surfaces stay


def test_depth_gradient_cut_ignores_already_dropped_pixels(tmp_path, monkeypatch):
    """A zero neighbour is a hole, not a surface — it must not take live pixels with it."""
    ckpt_path = _write_ckpt(tmp_path, n_views=2, height=3, width=4, camera_span=10.0)
    depths = np.full((2, 3, 4), 1.0, dtype=np.float32)
    depths[:, :, 0] = 0.0  # no observation in the first column
    _patch_renders(monkeypatch, _fake_render_views(depths))

    out, _, _, _ = _splats_to_tsdf_inputs(ckpt_path, max_depth_grad=0.3)
    assert np.all(out[:, :, 1:] == 1.0)


def test_depth_cuts_ship_off(tmp_path, monkeypatch):
    """Neither cut runs unless a config asks for it — both delete real surface if mistuned."""
    ckpt_path = _write_ckpt(tmp_path, n_views=2, height=3, width=4, camera_span=10.0)
    depths = np.full((2, 3, 4), 1.0, dtype=np.float32)
    depths[:, :, 2:] = 500.0
    _patch_renders(monkeypatch, _fake_render_views(depths))

    # No cut arguments at all: the 1 -> 500 step would trip either filter if one defaulted on
    out, _, _, _ = _splats_to_tsdf_inputs(ckpt_path)
    assert out.min() == 1.0 and out.max() == 500.0
