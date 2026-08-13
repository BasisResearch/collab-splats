"""LoGeR feedforward backend: resize rule and creator contract."""
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

from collab_splats.pointcloud.feedforward.loger import LoGeRCreator, _LOGER_ROOT, _compute_target_size

# LoGeRCreator subclasses an ABC (BasePointcloudCreator, collab_splats/pointcloud/base.py:4)
# and Tasks 6-8 own the four remaining abstract methods, so the class cannot be
# instantiated yet. These tests are correct as written and are the reason the fields and
# routing below are shaped the way they are — they run for real the moment Task 8 lands.
#
# strict=True is the point: when the last abstract method arrives these turn XPASS, which
# pytest reports as a FAILURE, forcing this marker to be deleted. A non-strict xfail would
# quietly survive its own cause and hide whatever it was guarding.
_NEEDS_FULL_CREATOR = pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="LoGeRCreator's abstract methods land in Tasks 6-8; remove this marker there",
)


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


@_NEEDS_FULL_CREATOR
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


@_NEEDS_FULL_CREATOR
def test_creator_uses_pinhole_camera_model():
    # SIMPLE_PINHOLE averages (fx + fy) / 2 at COLMAP export, silently discarding the
    # anisotropy the separate-focal fit exists to preserve. vggtx sets SIMPLE_PINHOLE,
    # so inheriting or copying its class body would inherit that.
    assert LoGeRCreator().camera_model == "PINHOLE"


@_NEEDS_FULL_CREATOR
def test_unknown_variant_rejected_at_construction():
    # Fail at construction, not at _load_model — a typo'd variant should not survive
    # until after a multi-GB checkpoint download.
    with pytest.raises(ValueError, match="variant"):
        LoGeRCreator(variant="LoGeR_turbo")


@_NEEDS_FULL_CREATOR
def test_load_model_rejects_unknown_model_config_key(tmp_path, monkeypatch):
    # A forward-only key silently dropped is how LoGeR_star would degrade invisibly:
    # se3 is declared under model: but is popped inside forward
    # (github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:589), so a naive
    # "filter to constructor signature" would discard it and run the wrong alignment
    # mode with no error. se3 is therefore routed explicitly, and anything else
    # unrecognised must stop the run rather than be dropped.
    from collab_splats.pointcloud.feedforward import loger as loger_mod

    ckpt_dir = tmp_path / "ckpts" / "LoGeR_star"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "original_config.yaml").write_text(
        yaml.safe_dump({"model": {"se3": True, "some_future_forward_kwarg": 3}})
    )
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)

    with pytest.raises(ValueError, match="some_future_forward_kwarg"):
        LoGeRCreator()._load_model("cpu")


@_NEEDS_FULL_CREATOR
def test_load_model_reports_missing_config(tmp_path, monkeypatch):
    # The vendored tree is gitignored, so "file not found" is the single most likely
    # first-run failure. The message must name the script that fixes it.
    from collab_splats.pointcloud.feedforward import loger as loger_mod

    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="setup/loger.sh"):
        LoGeRCreator()._load_model("cpu")
