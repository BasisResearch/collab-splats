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


def _creator(**kwargs) -> LoGeRCreator:
    """LoGeRCreator with only the not-yet-implemented abstract methods stubbed out."""
    # BasePointcloudCreator is an abc.ABC (collab_splats/pointcloud/base.py:101). Stubbing
    # ONLY the unwritten methods keeps every test below pointed at real code as it lands,
    # rather than deferring all signal to the task that happens to close the ABC.
    # Each task deletes the stub it just implemented. Task 8 deletes this helper entirely.
    # Signatures are copied from the abstract declarations
    # (collab_splats/pointcloud/feedforward/base.py:922, :925, :928, :1023) so a later task
    # implementing one against the real contract cannot silently disagree with its stub.
    class _PartialLoGeRCreator(LoGeRCreator):
        def _forward(self, model, views, **kwargs):
            raise NotImplementedError

        def _postprocess(self, raw_outputs, **kwargs):
            raise NotImplementedError

        def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
            raise NotImplementedError

        def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
            raise NotImplementedError

    return _PartialLoGeRCreator(**kwargs)


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


def test_creator_defaults_match_upstream_effective_values():
    # These are NOT read from the shipped yaml — both original_config.yaml files hold
    # only a model: key, so build_forward_kwargs' fallbacks are what actually run
    # (github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:149-164). window_size and
    # overlap_size are that file's argparse defaults at :47 and :49.
    c = _creator()
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
    assert _creator().camera_model == "PINHOLE"


def test_unknown_variant_rejected_at_construction():
    # Fail at construction, not at _load_model — a typo'd variant should not survive
    # until after a multi-GB checkpoint download.
    with pytest.raises(ValueError, match="variant"):
        _creator(variant="LoGeR_turbo")


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
        _creator()._load_model("cpu")


def test_load_model_rejects_an_empty_model_block(tmp_path, monkeypatch, stub_pi3):
    # An empty model: block is not harmless. It builds Pi3 on constructor defaults, which is a
    # different architecture from either shipped config, and the run then dies 278 state_dict
    # keys later — after a 5 GB download, with an error naming neither the file nor the cause.
    ckpt_dir = tmp_path / "ckpts" / "LoGeR_star"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "original_config.yaml").write_text("model:\n")
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)

    with pytest.raises(ValueError, match="no 'model:' block"):
        _creator()._load_model("cpu")


def test_load_model_reports_missing_config(tmp_path, monkeypatch):
    # The vendored tree is gitignored, so "file not found" is the single most likely
    # first-run failure. The message must name the script that fixes it.
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="setup/loger.sh"):
        _creator()._load_model("cpu")


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

        creator = _creator(variant=variant)
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
    creator = _creator(pixel_limit=100_000)
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
    _, _, original_coords = _creator()._preprocess(frames, [0, 1, 2])

    assert original_coords.shape == (3, 6)
    np.testing.assert_allclose(original_coords, np.tile([0, 0, 640, 480, 640, 480], (3, 1)))


def test_preprocess_rejects_non_uniform_frame_sizes():
    # LoGeR sizes from frame 0 alone; refuse rather than silently mis-resize the rest.
    frames = [_fake_frames(1, 480, 640)[0], _fake_frames(1, 240, 320)[0]]
    with pytest.raises(ValueError, match="uniform"):
        _creator()._preprocess(frames, [0, 1])


@pytest.mark.parametrize("frame_idxs", [[0, 10, 5], [0, 5, 5]])
def test_preprocess_rejects_out_of_order_frames(frame_idxs):
    # Windows and overlap stitching assume temporal order; out-of-order input
    # degrades quality with no error. No other backend cares, so this is LoGeR's.
    # The duplicate case is the one a `b < a` guard would let through, and duplicates
    # also collide in the frame_{idx:06d} labels — hence "strictly" ascending.
    frames = _fake_frames(3, 480, 640)
    with pytest.raises(ValueError, match="ascending"):
        _creator()._preprocess(frames, frame_idxs)
