"""LoGeR feedforward backend: resize rule and creator contract."""
import sys
import types

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from collab_splats.pointcloud.feedforward import loger as loger_mod
from collab_splats.pointcloud.feedforward.loger import LoGeRCreator, _LOGER_ROOT, _compute_target_size


@pytest.fixture
def stub_pi3(monkeypatch):
    """Seed a fake loger.models.pi3 so _load_model runs without the vendored tree."""
    # _load_model does `from loger.models.pi3 import Pi3` after patching sys.path. Seeding
    # sys.modules short-circuits that import, so these tests neither require
    # third_party/LoGeR nor depend on an earlier test having populated the module cache —
    # the ordering dependency pytest-randomly would otherwise expose.

    class _StubPi3:
        # Mirrors the real Pi3.__init__ parameter names (github.com/Junyi42/LoGeR @ 7685b7a,
        # loger/models/pi3.py:20-35), because _load_model validates config keys against this
        # signature. A stub cannot notice an upstream signature change; that is covered by
        # loading the real checkpoints, not here.
        def __init__(
            self, pos_type=None, decoder_size=None, ttt_insert_after=None, ttt_head_dim=None,
            ttt_inter_multi=None, num_muon_update_steps=None, use_momentum=None,
            ttt_update_steps=None, conf=None, attn_insert_after=None, ttt_pre_norm=None,
            pi3x=None, pi3x_metric=None,
        ):
            self.init_kwargs = {k: v for k, v in locals().items() if k != "self"}

        def load_state_dict(self, state, strict=True):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

    # Seed all three package levels: `from a.b.c import D` still walks the parent packages,
    # so seeding only the leaf leaves the import resolving against a real (absent) `loger`.
    for name in ("loger", "loger.models", "loger.models.pi3"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["loger.models.pi3"].Pi3 = _StubPi3

    # Skip the network and the multi-GB checkpoint; neither is what these tests measure.
    monkeypatch.setattr(loger_mod, "hf_hub_download", lambda **kwargs: "/nonexistent/latest.pt")
    monkeypatch.setattr(loger_mod.torch, "load", lambda *args, **kwargs: {})
    return _StubPi3


@pytest.mark.parametrize(
    "orig_w,orig_h",
    [(1920, 1080), (1080, 1920), (640, 480), (1000, 1000), (3840, 2160), (1920, 1200)],
)
def test_target_size_is_patch_aligned_and_within_budget(orig_w, orig_h):
    # Both invariants are load-bearing: a non-multiple of 14 crashes the ViT
    # patch embedding, and exceeding the budget is what OOMs long sequences.
    w, h = _compute_target_size(orig_w, orig_h, pixel_limit=255_000)
    assert w % 14 == 0 and h % 14 == 0
    assert w >= 14 and h >= 14
    assert w * h <= 255_000


def test_target_size_preserves_orientation():
    # Landscape stays landscape. Independent per-axis rounding perturbs the exact
    # ratio by a few percent, but must never transpose it.
    w, h = _compute_target_size(1920, 1080, pixel_limit=255_000)
    assert w > h


def test_target_size_aspect_error_is_small_but_real():
    # The residual anisotropy is why fx and fy are fitted separately and why
    # camera_model must be PINHOLE. Assert it exists and is bounded — if a future
    # change makes it exactly zero, the separate-focal machinery is still correct
    # but this test documents why it is there.
    orig_w, orig_h = 1920, 1080
    w, h = _compute_target_size(orig_w, orig_h, pixel_limit=255_000)
    ratio_error = abs((w / h) / (orig_w / orig_h) - 1.0)
    assert ratio_error < 0.05


def test_target_size_upscales_small_images_to_the_budget():
    # The rule is an area budget, not a cap: a tiny input is scaled up to fill it.
    w, h = _compute_target_size(64, 48, pixel_limit=255_000)
    assert w * h > 200_000


def test_zero_area_raises_instead_of_silently_returning_a_tile():
    # Deliberate divergence from upstream, which guards this and falls through to a
    # 14x14 image the model would consume without complaint. Our caller passes frame
    # store dimensions, positive by construction, so a zero here means the store is
    # corrupt and a traceback naming the division is more useful than a silent tile.
    # This is the only behaviour in this function that intentionally differs from the
    # vendored loader, so it gets a test rather than a comment.
    with pytest.raises(ZeroDivisionError):
        _compute_target_size(0, 1080, pixel_limit=255_000)


# 1920x1200 is here for a specific reason: it is the only case in either list that takes
# the `patches_h -= 1` branch of the shrink loop. Without it that branch never executes,
# and three separate mutations to _compute_target_size pass all the other cases —
# dropping the tie-break entirely, inverting it to `patches_w > patches_h`, and swapping
# round() for floor(). Do not remove it to trim runtime.
@pytest.mark.skipif(not _LOGER_ROOT.exists(), reason="vendored tree absent (setup/loger.sh)")
@pytest.mark.parametrize("orig_w,orig_h", [(1920, 1080), (1080, 1920), (640, 480), (1000, 1000), (1920, 1200)])
def test_target_size_matches_the_vendored_loader(tmp_path, orig_w, orig_h):
    # This helper is reimplemented rather than copied, so parity with upstream is a
    # thing we MEASURE, not a thing we claim. The model trains on images preprocessed
    # by the vendored loader; if our size arithmetic drifts from it we feed the model
    # out-of-distribution input, and nothing downstream would report that.
    sys.path.insert(0, str(_LOGER_ROOT))
    try:
        from loger.utils.basic import load_images_as_tensor
    finally:
        sys.path.remove(str(_LOGER_ROOT))

    # The loader takes a directory, so give it two frames at the size under test.
    for i in range(2):
        Image.fromarray(
            np.random.default_rng(i).integers(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        ).save(tmp_path / f"{i:04d}.jpg")

    # PIXEL_LIMIT is upstream's casing, not a typo on our side — the vendored signature
    # is `load_images_as_tensor(path, interval, PIXEL_LIMIT, Target_W, Target_H)` at
    # github.com/Junyi42/LoGeR @ 7685b7a, loger/utils/basic.py:11. Passing it explicitly
    # rather than leaning on its default is what makes this a parity test of the SAME
    # budget our own call uses; a silent default drift upstream would otherwise pass.
    upstream = load_images_as_tensor(str(tmp_path), PIXEL_LIMIT=255_000)
    _, _, up_h, up_w = upstream.shape

    assert _compute_target_size(orig_w, orig_h, pixel_limit=255_000) == (up_w, up_h)


def test_creator_is_instantiable():
    # Tasks 6-7 needed a stub subclass to run at all; this asserts that crutch is genuinely
    # gone rather than merely deleted from the call sites. The ABC closed here is
    # BaseFeedforwardCreator, which declares SIX abstract methods — _load_model,
    # _preprocess, _forward, _postprocess (collab_splats/pointcloud/feedforward/base.py:915,
    # :918, :921, :924), extract_intermediate_features (:927) and _reproject (:1022).
    # BasePointcloudCreator (collab_splats/pointcloud/base.py:101) is the ABC further up the
    # chain and contributes only `reconstruct` (:102), which BaseFeedforwardCreator already
    # implements concretely (base.py:805) — so it is not what a subclass must satisfy.
    assert isinstance(LoGeRCreator(), LoGeRCreator)


def test_creator_defaults_match_upstream_effective_values():
    # These are NOT read from the shipped yaml — both original_config.yaml files hold
    # only a model: key, so build_forward_kwargs' fallbacks are what actually run
    # (github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:149-164). window_size and
    # overlap_size are that file's argparse defaults at :47 and :49.
    c = LoGeRCreator()
    assert c.variant == "LoGeR_star"
    assert c.window_size == 32
    assert c.overlap_size == 3
    assert c.reset_every == 0
    assert c.num_iterations == 1
    assert c.pixel_limit == 255_000
    assert c.use_multiview_confidence is False


def test_creator_uses_pinhole_camera_model():
    # Weak by construction and kept deliberately: the base already defaults to PINHOLE
    # (base.py:793), so this passes even without loger.py's redeclaration. It guards the
    # contract, not the local line — vggtx overrides to SIMPLE_PINHOLE, which averages
    # (fx + fy) / 2 at COLMAP export and would silently destroy the anisotropy the
    # separate-focal fit exists to preserve.
    assert LoGeRCreator().camera_model == "PINHOLE"


def test_unknown_variant_rejected_at_construction():
    # Fail at construction, not at _load_model — a typo'd variant should not survive
    # until after a multi-GB checkpoint download.
    with pytest.raises(ValueError, match="variant"):
        LoGeRCreator(variant="LoGeR_turbo")


def test_load_model_rejects_unknown_model_config_key(tmp_path, monkeypatch, stub_pi3):
    # A forward-only key silently dropped is how LoGeR_star would degrade invisibly:
    # se3 is declared under model: but is popped inside forward
    # (github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:589), so a naive
    # "filter to constructor signature" would discard it and run the wrong alignment
    # mode with no error. se3 is therefore routed explicitly, and anything else
    # unrecognised must stop the run rather than be dropped.
    #
    # The stub_pi3 fixture is load-bearing here, not convenience: _LOGER_ROOT is
    # redirected at a tmp dir with no loger/ package, so without it this raises
    # ModuleNotFoundError before reaching the check it asserts — and passes only when an
    # earlier test happened to leave `loger` in sys.modules.
    ckpt_dir = tmp_path / "ckpts" / "LoGeR_star"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "original_config.yaml").write_text(
        yaml.safe_dump({"model": {"se3": True, "some_future_forward_kwarg": 3}})
    )
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)

    with pytest.raises(ValueError, match="some_future_forward_kwarg"):
        LoGeRCreator()._load_model("cpu")


def test_load_model_rejects_an_empty_model_block(tmp_path, monkeypatch, stub_pi3):
    # An empty model: block is not harmless. It builds Pi3 on constructor defaults, which is a
    # different architecture from either shipped config, and the run then dies 278 state_dict
    # keys later — after a 5 GB download, with an error naming neither the file nor the cause.
    ckpt_dir = tmp_path / "ckpts" / "LoGeR_star"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "original_config.yaml").write_text("model:\n")
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)

    with pytest.raises(ValueError, match="no 'model:' block"):
        LoGeRCreator()._load_model("cpu")


def test_load_model_reports_missing_config(tmp_path, monkeypatch):
    # The vendored tree is gitignored, so "file not found" is the single most likely
    # first-run failure. The message must name the script that fixes it.
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="setup/loger.sh"):
        LoGeRCreator()._load_model("cpu")


def test_se3_is_captured_from_the_variant_yaml_not_merely_dropped(tmp_path, monkeypatch, stub_pi3):
    # The point of the whole routing. Replacing the capture with a bare
    # `model_cfg.pop("se3", None)` passes every other test in this file while producing
    # exactly the silent wrong-alignment-mode run the design exists to prevent, so the
    # value has to be asserted per variant rather than inferred from "it loaded".
    #
    # Mirrors the real configs: LoGeR declares no se3, LoGeR_star sets it true
    # (github.com/Junyi42/LoGeR @ 7685b7a, ckpts/*/original_config.yaml).
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)

    for variant, model_block, expected in (
        ("LoGeR", {"decoder_size": "large"}, False),
        ("LoGeR_star", {"decoder_size": "large", "se3": True}, True),
    ):
        cfg_dir = tmp_path / "ckpts" / variant
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "original_config.yaml").write_text(yaml.safe_dump({"model": model_block}))

        creator = LoGeRCreator(variant=variant)
        creator._load_model("cpu")
        assert creator._se3 is expected


def _fake_frames(n: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def test_preprocess_returns_patch_aligned_unit_range_tensor():
    # Non-default pixel_limit, and the exact expected shape rather than a divisibility
    # check: (H, W) both being multiples of 14 is also true of the transposed size, so
    # `% 14 == 0` alone cannot tell a correct resize from an axis swap, and a hardcoded
    # budget from the field being ignored altogether.
    creator = LoGeRCreator(pixel_limit=100_000)
    frames = _fake_frames(4, 480, 640)
    views, image_paths, original_coords = creator._preprocess(frames, [0, 5, 10, 15])

    assert views.shape[0] == 4 and views.shape[1] == 3
    # _compute_target_size returns (w, h); views is (N, 3, H, W) — hence the reversal.
    assert tuple(views.shape[2:]) == _compute_target_size(640, 480, 100_000)[::-1]
    assert views.dtype == torch.float32
    # unproject_and_filter_points reads these as colors; _forward asserts the range.
    assert float(views.min()) >= 0.0 and float(views.max()) <= 1.0
    assert [p.name for p in image_paths] == [
        "frame_000000", "frame_000005", "frame_000010", "frame_000015"
    ]


def test_preprocess_original_coords_is_full_frame():
    # LoGeR resizes and never crops, so every row is the whole image. This is what
    # _rescale_reconstruction_to_original_dimensions consumes.
    frames = _fake_frames(3, 480, 640)
    _, _, original_coords = LoGeRCreator()._preprocess(frames, [0, 1, 2])

    assert original_coords.shape == (3, 6)
    np.testing.assert_allclose(original_coords, np.tile([0, 0, 640, 480, 640, 480], (3, 1)))


def test_preprocess_rejects_non_uniform_frame_sizes():
    # LoGeR sizes from frame 0 alone; refuse rather than silently mis-resize the rest.
    frames = [_fake_frames(1, 480, 640)[0], _fake_frames(1, 240, 320)[0]]
    with pytest.raises(ValueError, match="uniform"):
        LoGeRCreator()._preprocess(frames, [0, 1])


@pytest.mark.parametrize("frame_idxs", [[0, 10, 5], [0, 5, 5]])
def test_preprocess_rejects_out_of_order_frames(frame_idxs):
    # Windows and overlap stitching assume temporal order; out-of-order input
    # degrades quality with no error. No other backend cares, so this is LoGeR's.
    # The duplicate case is the one a `b < a` guard would let through, and duplicates
    # also collide in the frame_{idx:06d} labels — hence "strictly" ascending.
    frames = _fake_frames(3, 480, 640)
    with pytest.raises(ValueError, match="ascending"):
        LoGeRCreator()._preprocess(frames, frame_idxs)


def _loaded_creator(se3: bool = False, **kwargs) -> LoGeRCreator:
    """LoGeRCreator with _se3 set to what _load_model would have read from the variant yaml."""
    # Named parameter, not a hidden default: these tests stub the model, so _load_model never
    # runs and _forward's se3 guard would otherwise fire on every one of them.
    creator = LoGeRCreator(**kwargs)
    creator._se3 = se3
    return creator


def _synthetic_local_points(h: int, w: int, fx: float, fy: float, z: float = 2.0) -> np.ndarray:
    """Exact pinhole camera-frame pointmap, so a K fit over it must recover (fx, fy)."""
    # The principal point must match the one estimate_intrinsics_from_points assumes —
    # cx=(W-1)/2, cy=(H-1)/2 (collab_splats/geometry/transforms.py:159-162), NOT w/2.
    # Off-by-half-a-pixel here biases the recovered focal, and the assertions below would
    # then be pinning the bias rather than the fit.
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )
    # Forward pinhole: X = u_c * Z / fx, Y = v_c * Z / fy, channel 2 IS Z. Constant z is what
    # lets the depth test assert a single number independently of the focals.
    pts = np.stack([uu * z / fx, vv * z / fy, np.full_like(uu, z)], axis=-1)
    return pts[None].astype(np.float32)  # (1,H,W,3)


class _FakeLoGeR(torch.nn.Module):
    """Minimal stand-in for Pi3 that returns a known pinhole scene.

    Emits RAW conf logits deliberately outside [0, 1], so any path that forgets
    torch.sigmoid produces an out-of-range depth_conf the assertions catch.
    """

    def __init__(self, n: int, h: int, w: int, fx: float, fy: float, conf_logit: float = 4.0):
        super().__init__()
        self.register_parameter("_p", torch.nn.Parameter(torch.zeros(1)))
        pts = _synthetic_local_points(h, w, fx, fy)  # (1,H,W,3)
        self.local_points = torch.from_numpy(np.repeat(pts, n, axis=0))[None]  # (1,N,H,W,3)
        # conf_logit is a parameter, not a constant: the default 4.0 (sigmoid ~0.982) clears
        # both 0.02 and the 0.1 library default, so it cannot tell the two apart. The
        # threshold test below drives it down into LoGeR's real measured band.
        self.conf = torch.full((1, n, h, w, 1), conf_logit)
        # Distinct non-identity c2w poses: camera i sits at x = i along the world axis
        poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        poses[:, 0, 3] = np.arange(n, dtype=np.float32)
        self.camera_poses = torch.from_numpy(poses)[None]  # (1,N,4,4)
        self.seen_kwargs: dict = {}
        self.seen_images_shape: tuple = ()

    def forward(self, images, **kwargs):
        # Record the input rather than consuming it: the fixed synthetic scene is what every
        # assertion below is written against, so the OUTPUTS must stay input-independent.
        self.seen_kwargs = kwargs
        self.seen_images_shape = tuple(images.shape)
        # `+ self._p` (a zeros Parameter) leaves the values untouched but makes this output
        # require grad, so dropping torch.no_grad() in _forward turns .numpy() into a raise
        # instead of a silent autograd-tracked run.
        return {
            "local_points": self.local_points + self._p,
            "conf": self.conf,
            "camera_poses": self.camera_poses,
            "points": self.local_points,
        }


# (h, w) = (56, 70) throughout: both (w-1)/2 and (h-1)/2 land on .5, so no pixel has
# x == 0 or y == 0 and none is dropped by estimate_intrinsics_from_points' validity gate
# `(|x| > 1e-6) & (|y| > 1e-6)` (collab_splats/geometry/transforms.py:166). Both are also
# multiples of the patch size 14.


def test_forward_refuses_to_run_before_load_model_sets_se3():
    # _se3 is None until _load_model reads the variant yaml. Both real values are valid, so
    # there is nothing safe to default to — guessing picks an alignment mode silently.
    n, h, w = 2, 56, 70
    with pytest.raises(RuntimeError, match="se3 unset"):
        LoGeRCreator()._forward(_FakeLoGeR(n, h, w, 80.0, 80.0), torch.rand(n, 3, h, w))


def test_forward_applies_sigmoid_to_raw_confidence_logits():
    # LoGeR's conf_head is a bare LinearPts3d with no activation (Junyi42/LoGeR @
    # 7685b7a, loger/models/pi3.py:172); upstream applies sigmoid at the call site
    # (PolyCam/LoGeR @ 5d7c1a7, run_loger.py:481). The K fit's conf gate is a
    # threshold on a probability, so this must run first.
    n, h, w = 3, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    views = torch.rand(n, 3, h, w)

    raw = _loaded_creator()._forward(model, views)

    assert raw["depth_conf"].shape == (n, h, w)
    assert raw["depth_conf"].min() >= 0.0 and raw["depth_conf"].max() <= 1.0
    assert raw["depth_conf"].max() == pytest.approx(1 / (1 + np.exp(-4.0)), rel=1e-4)


def test_forward_inverts_camera_poses_to_world_to_camera():
    # LoGeR returns camera-to-world; FeedforwardResult.extrinsics is world-to-camera.
    # The single easiest thing to get backwards, and silent when wrong.
    n, h, w = 3, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)

    raw = _loaded_creator()._forward(model, torch.rand(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)
    # c2w camera 2 sits at x=+2, so the w2c translation must be -2, not +2.
    assert raw["extrinsic"][2, 0, 3] == pytest.approx(-2.0)


def test_forward_fits_and_broadcasts_intrinsics():
    n, h, w = 3, 56, 70
    fx, fy = 88.0, 80.0
    raw = _loaded_creator()._forward(_FakeLoGeR(n, h, w, fx, fy), torch.rand(n, 3, h, w))

    assert raw["intrinsics"].shape == (n, 3, 3)
    assert raw["intrinsics"][0, 0, 0] == pytest.approx(fx, rel=1e-3)
    assert raw["intrinsics"][0, 1, 1] == pytest.approx(fy, rel=1e-3)
    # np.broadcast_to alone returns a read-only zero-stride view over one 3x3; the .copy()
    # is what makes this N real matrices. Without it any downstream in-place write raises.
    assert raw["intrinsics"].flags["OWNDATA"]
    # _raw_to_world_points hard-requires this key and returns (None, None) without it
    # (collab_splats/pointcloud/feedforward/base.py:335).
    np.testing.assert_allclose(raw["intrinsics_downsampled"], raw["intrinsics"])


def test_forward_passes_the_measured_conf_threshold_not_the_library_default():
    # Found by mutation: dropping the explicit LOGER_CONF_THRESHOLD passed every other
    # test in this file, because they all run at conf logit 4.0 (sigmoid ~0.982) which
    # clears both gates. LoGeR's conf head is uncalibrated — measured logits span
    # -4.257..-2.019, i.e. a post-sigmoid band of [0.0140, 0.1172] — so
    # estimate_intrinsics_from_points' 0.1 default sits at its 92nd percentile.
    #
    # logit -3.0 -> sigmoid 0.0474 is squarely inside that measured band: above
    # LOGER_CONF_THRESHOLD (0.02) and below the 0.1 default. The fit must therefore
    # SUCCEED here, and would raise on the inherited default.
    n, h, w = 2, 56, 70
    assert loger_mod.LOGER_CONF_THRESHOLD < 1 / (1 + np.exp(3.0)) < 0.1
    model = _FakeLoGeR(n, h, w, 80.0, 80.0, conf_logit=-3.0)

    raw = _loaded_creator()._forward(model, torch.rand(n, 3, h, w))

    assert raw["intrinsics"][0, 0, 0] == pytest.approx(80.0, rel=1e-3)


def test_forward_extracts_depth_from_the_third_channel():
    # LoGeR builds local_points as cat([xy * z, z]), so channel 2 IS depth — no
    # reprojection needed to recover it.
    n, h, w = 2, 56, 70
    raw = _loaded_creator()._forward(_FakeLoGeR(n, h, w, 80.0, 80.0), torch.rand(n, 3, h, w))
    assert raw["depth"].shape == (n, h, w, 1)
    np.testing.assert_allclose(raw["depth"], 2.0, rtol=1e-5)


def test_forward_passes_window_knobs_and_se3():
    # All NINE kwargs are asserted, not just the four backed by dataclass fields: _forward's
    # comment claims parity with build_forward_kwargs (github.com/PolyCam/LoGeR @ 5d7c1a7,
    # run_loger.py:149-164) and nothing else enforces that claim — mutating any of the five
    # literals otherwise survives the whole file. Every field-backed knob is set to a
    # NON-default value (defaults are 32/3/0/1, se3 False) so a hardcode dies here too.
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    creator = _loaded_creator(se3=True, window_size=16, overlap_size=4, reset_every=8, num_iterations=3)

    creator._forward(model, torch.rand(n, 3, h, w))

    assert model.seen_kwargs["window_size"] == 16
    assert model.seen_kwargs["overlap_size"] == 4
    assert model.seen_kwargs["reset_every"] == 8
    assert model.seen_kwargs["num_iterations"] == 3
    assert model.seen_kwargs["se3"] is True
    assert model.seen_kwargs["sim3"] is False
    assert model.seen_kwargs["sim3_scale_mode"] == "median"
    assert model.seen_kwargs["turn_off_ttt"] is False
    assert model.seen_kwargs["turn_off_swa"] is False


def test_forward_adds_the_batch_dimension_pi3_requires():
    # _preprocess hands _forward an (N, 3, H, W) stack; Pi3 takes (B, N, 3, H, W). The fake
    # ignores its input by design, so without recording the shape here, dropping the
    # images[None] passes every other assertion in this file while the real model would read
    # N as the batch size and 3 as the frame count.
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)

    _loaded_creator()._forward(model, torch.rand(n, 3, h, w))

    assert len(model.seen_images_shape) == 5
    assert model.seen_images_shape == (1, n, 3, h, w)


def test_forward_runs_under_no_grad():
    # _FakeLoGeR derives local_points from its _p Parameter, so autograd tracking would make
    # the .numpy() calls raise. Assert plain ndarrays out so the reason is stated here rather
    # than left as an unexplained RuntimeError in whichever test happens to run first.
    n, h, w = 2, 56, 70
    raw = _loaded_creator()._forward(_FakeLoGeR(n, h, w, 80.0, 80.0), torch.rand(n, 3, h, w))
    assert isinstance(raw["local_points"], np.ndarray)
    assert isinstance(raw["depth"], np.ndarray)


def test_forward_rejects_rgb_outside_unit_range():
    # Guards the a157421 [0,255] bug class at the source.
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    with pytest.raises(AssertionError, match=r"\[0, 1\]"):
        _loaded_creator()._forward(model, torch.rand(n, 3, h, w) * 255.0)


@pytest.mark.parametrize("fill", [torch.zeros, torch.ones])
def test_forward_accepts_the_rgb_range_boundaries(fill):
    # The bounds are INCLUSIVE, and nothing else here says so: every other test feeds
    # torch.rand, which never emits exactly 0.0 or 1.0, so tightening either `<=` to `<`
    # survives. A pure-black pixel is common in real frames and `<` would reject the frame.
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)

    # The fake's outputs do not depend on its input, so both extremes run to completion.
    raw = _loaded_creator()._forward(model, fill(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)


def _forward_and_postprocess(n=3, fx=88.0, fy=80.0, **creator_kwargs):
    """Drive the real _preprocess -> _forward -> _postprocess chain over the synthetic scene."""
    # pixel_limit 4000 resizes the 112x140 input DOWN to the (56, 70) model resolution the
    # rest of this file is written against — verified: _compute_target_size(140, 112, 4000)
    # == (70, 56). The 255_000 default would UPSCALE it to 560x448 and push 750k points
    # through unproject_and_filter_points, past max_points into random subsampling, for no
    # added signal and a much slower test.
    creator_kwargs.setdefault("pixel_limit", 4_000)
    creator = _loaded_creator(**creator_kwargs)
    views, image_paths, original_coords = creator._preprocess(_fake_frames(n, 112, 140), list(range(n)))
    # setup_inference sets these two on the real path (base.py:843); _postprocess reads them
    # off self, so a helper that skipped this would test a different object than production.
    creator.image_paths, creator.original_coords = image_paths, original_coords
    raw = creator._forward(_FakeLoGeR(n, views.shape[2], views.shape[3], fx, fy), views)
    return creator, raw, creator._postprocess(raw)


def test_postprocess_field_contract():
    n = 3
    _, raw, result = _forward_and_postprocess(n=n)

    assert result.extrinsics.shape == (n, 4, 4)
    assert result.intrinsics.shape == (n, 3, 3)
    # depth is stored (N,H,W), trailing axis squeezed, matching every other backend
    assert result.depth.ndim == 3 and result.depth.shape[0] == n
    assert result.world_points is not None
    assert result.world_points.shape == (n, result.model_height, result.model_width, 3)
    assert result.colors.dtype == np.uint8
    assert result.points.shape[1] == 3 and len(result.points) == len(result.colors)
    # confidence is a torch.Tensor while depth is an np.ndarray — the asymmetry is the
    # dataclass' declared contract (base.py:72 vs :74), not an oversight, and BA consumes
    # it. Dropping it to None survived every other assertion here.
    assert isinstance(result.confidence, torch.Tensor)
    assert tuple(result.confidence.shape) == (n, result.model_height, result.model_width)
    # Forwarded from the creator, where setup_inference put them. build_colmap reads both
    # off the result (base.py:887, :893-894), so losing them exports a COLMAP model with no
    # filenames and no rescale back to original resolution — silent, and not otherwise caught.
    assert [p.name for p in result.image_paths] == [f"frame_{i:06d}" for i in range(n)]
    assert result.original_coords.shape == (n, 6)
    # world_points must be the K-consistent grid from _raw_to_world_points, not LoGeR's own
    # `points`. The fake's cameras sit at world x = 0..n-1 with identity rotation and the
    # scene is a constant z=2 plane, so every frame's grid is offset by exactly its index.
    # A subsample != 1 changes the point count and a transposed reshape changes the shape,
    # but only a value check catches world_points sourced from the wrong array entirely.
    np.testing.assert_allclose(
        result.world_points[2] - result.world_points[0],
        np.tile(np.array([2.0, 0.0, 0.0], dtype=np.float32), (*result.world_points.shape[1:3], 1)),
        atol=1e-4,
    )


def test_postprocess_preserves_anisotropic_focals():
    # camera_model PINHOLE keeps fx and fy; SIMPLE_PINHOLE would average them to
    # (fx + fy) / 2 at COLMAP export (collab_splats/pointcloud/feedforward/base.py:551-554)
    # and silently destroy the aspect correction the separate-focal fit exists to produce.
    creator, _, result = _forward_and_postprocess(fx=88.0, fy=80.0)
    assert creator.camera_model == "PINHOLE"
    assert result.intrinsics[0, 0, 0] != pytest.approx(result.intrinsics[0, 1, 1], rel=1e-3)


def test_reproject_returns_points_and_colors_for_refined_poses():
    creator, raw, _ = _forward_and_postprocess()
    pts, colors = creator._reproject(raw, raw["extrinsic"], raw["intrinsics"])
    assert pts.shape[1] == 3
    assert len(pts) == len(colors)

    # Shapes alone cannot tell a _reproject that USES its extrinsics_3x4 argument from one
    # that quietly re-reads raw_outputs["extrinsic"] — and using the stale poses is exactly
    # the bug this method exists to prevent, since BundleAdjustment calls it precisely
    # because the stored poses are the ones it just refined. So hand it DIFFERENT poses.
    # The fake's rotations are identity, so shifting the w2c translation by +d moves every
    # reconstructed world point by -d (measured, not assumed).
    offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shifted = raw["extrinsic"].copy()
    shifted[:, :, 3] += offset
    moved_pts, moved_colors = creator._reproject(raw, shifted, raw["intrinsics"])

    np.testing.assert_allclose(moved_pts, pts - offset, atol=1e-4)
    # Colors come from the images, not the poses, so they must be untouched by the shift.
    np.testing.assert_array_equal(moved_colors, colors)


def test_reproject_uses_the_intrinsics_it_is_handed():
    # The mirror of the pose test above, on the other argument. _reproject's contract is
    # that BOTH the refined poses and the refined K come from its parameters; passing the
    # stored K on both calls leaves a mutant that reads raw_outputs["intrinsics"] invisible.
    # n=1 because the fake's camera 0 sits at the world origin with identity rotation, so
    # world coordinates ARE camera coordinates and the pinhole relation is exact rather
    # than entangled with the pose.
    creator, raw, _ = _forward_and_postprocess(n=1)
    pts, colors = creator._reproject(raw, raw["extrinsic"], raw["intrinsics"])

    # x_c = (u - cx) * z / fx, y_c = (v - cy) * z / fy, z_c = z. Halving both focals must
    # therefore double the two lateral axes at fixed depth and leave depth untouched — a
    # predicted direction and magnitude, not merely "something changed". Measured exact.
    softer = raw["intrinsics"].copy()
    softer[:, 0, 0] /= 2.0
    softer[:, 1, 1] /= 2.0
    wider_pts, wider_colors = creator._reproject(raw, raw["extrinsic"], softer)

    np.testing.assert_allclose(
        wider_pts, pts * np.array([2.0, 2.0, 1.0], dtype=np.float32), atol=1e-4
    )
    # Guard against the relation being satisfied trivially by a cloud that did not move.
    assert np.abs(wider_pts - pts).max() > 0.1
    # Colors come from the images, not the camera model, so the halved K must not touch them.
    np.testing.assert_array_equal(wider_colors, colors)


def test_max_points_caps_the_returned_cloud():
    # max_points defaults to 500_000 (base.py:794) against an 11,760-point scene, so the cap
    # never engages and a mutant changing it dies only by crashing on None, never because a
    # test noticed the cap. Construct below the scene size so it actually bites. Measured:
    # the cut is EXACT, not approximate, and the three arrays stay index-aligned through it.
    _, _, result = _forward_and_postprocess(max_points=100)

    assert len(result.points) == 100
    assert len(result.colors) == 100
    assert len(result.pixel_indices) == 100


def test_max_points_caps_the_reprojected_cloud():
    # The mirror of the cap test above on the BA path, which had only a crash pin: at the
    # 500_000 default the cap never engages against the 11,760-point scene, so a mutant
    # doubling _reproject's max_points survived the whole suite and only `None` died — as a
    # TypeError inside randomly_limit_trues (vggtx.py:144), which says nothing about whether
    # the cap is applied. Construct below the scene size so it bites here too. The cut is
    # EXACT when it engages: randomly_limit_trues draws size=max_trues without replacement.
    creator, raw, _ = _forward_and_postprocess(max_points=100)
    pts, colors = creator._reproject(raw, raw["extrinsic"], raw["intrinsics"])

    assert len(pts) == 100
    assert len(colors) == 100


def test_multiview_confidence_mask_is_wired_and_off_by_default():
    # Found by mutation: forcing this branch either way — permanently off, or permanently
    # on — passed every other test in this file, so the flag was decorative. Both
    # directions are pinned here.
    n = 3
    _, _, off = _forward_and_postprocess(n=n)
    _, _, on = _forward_and_postprocess(n=n, use_multiview_confidence=True)

    # The fake emits one constant confidence, so the percentile gate keeps every pixel.
    # Fewer than that means a mask was applied when the flag said not to.
    assert len(off.points) == n * off.model_height * off.model_width
    # And with the flag on, the geometric cross-view mask must actually reach
    # unproject_and_filter_points as extra_mask rather than being computed and discarded.
    assert 0 < len(on.points) < len(off.points)


def test_conf_threshold_field_reaches_the_point_filter():
    # Found by mutation: replacing self.conf_threshold with a literal survived everything
    # else. Values <= 1.0 are read as a RAW confidence and > 1.0 as a percentile
    # (vggtx.py:132-136), so this also pins the duality the class docstring warns about —
    # the fake's confidence is sigmoid(4.0) ~= 0.982, which 0.5 admits and 0.999 rejects.
    _, _, kept = _forward_and_postprocess(conf_threshold=0.5)
    _, _, dropped = _forward_and_postprocess(conf_threshold=0.999)

    assert len(kept.points) > 0
    assert len(dropped.points) == 0


def test_extract_intermediate_features_refuses():
    # LC is out of scope for the first cut: thresholds are per-backbone and uncalibrated
    # here. _verify_loop_candidate is concrete on the base class
    # (collab_splats/pointcloud/feedforward/base.py:960) and calls this at :991, so the
    # refusal must be explicit.
    with pytest.raises(NotImplementedError, match="loop closure"):
        LoGeRCreator().extract_intermediate_features(torch.rand(2, 3, 56, 70))
