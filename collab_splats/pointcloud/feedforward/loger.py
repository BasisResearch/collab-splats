"""LoGeR feedforward backend: resize rule and creator.

LoGeR is a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window
inference and overlap stitching.  Unlike every other backend we run it predicts no
camera intrinsics, so K is fitted from its camera-frame pointmap by
``collab_splats.geometry.transforms.estimate_intrinsics_from_points``.

Upstream sources.  Two forks are involved and they are NOT interchangeable.  Neither
ships a LICENSE, so nothing here is taken from either as an artifact — the forks are
cited as prior art and as the behavioural reference we match.  One honest caveat:
``_compute_target_size`` is written to a behavioural spec, but a greedy decrement to
an area budget has close to one natural form, so its arithmetic necessarily converges
on upstream's line for line.  That convergence is disclosed rather than disguised, and
it is measured by a parity test instead of asserted.  Citations carry repo, commit,
file, and line because third_party/ is gitignored and cannot be read from this repo
alone:

  * VENDORED — the tree we actually execute against
    (setup/loger.sh clones it into third_party/LoGeR/):
      github.com/Junyi42/LoGeR @ 7685b7a
  * PRIOR ART — read for reference, not vendored, not a dependency, nothing copied:
      github.com/PolyCam/LoGeR @ 5d7c1a7

Provides:
  LOGER_HF_REPO        — HuggingFace repo holding both checkpoints
  LOGER_VARIANTS       — the two shipped variants
  LOGER_CONF_THRESHOLD — confidence floor for the K fit, measured not inherited
  _compute_target_size — patch-aligned resize matching the vendored loader
  LoGeRCreator         — feedforward creator using LoGeR depth + pose (not yet implemented)
"""

from __future__ import annotations

import math
from pathlib import Path

########################################################################
########## Constants ###################################################
########################################################################

LOGER_HF_REPO = "Junyi42/LoGeR"
LOGER_VARIANTS = ("LoGeR", "LoGeR_star")

# Confidence floor for the K fit, passed explicitly to estimate_intrinsics_from_points
# instead of taking its 0.1 default. LoGeR's conf head is uncalibrated: measured logits
# span -4.257..-2.019, so the post-sigmoid band is [0.0140, 0.1172] and 0.1 is its 92nd
# percentile — the default keeps 7.9% of pixels (12,217/155,232 on an 8-frame run) and a
# marginally duller scene keeps none, raising. 0.02 sits just above the band floor, so it
# rejects only what the model calls junk; the confidence WEIGHTING inside the median is
# what actually discriminates. Re-measure this if the checkpoint changes.
LOGER_CONF_THRESHOLD = 0.02

# Vendored tree, populated by setup/loger.sh.  parents[3] resolves
# collab_splats/pointcloud/feedforward/loger.py -> repo root.
_LOGER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "LoGeR"

# LoGeR's ViT patch size; every model-resolution image dimension is a multiple of it.
_PATCH = 14

########################################################################
########## Preprocessing ###############################################
########################################################################


def _compute_target_size(orig_w: int, orig_h: int, pixel_limit: int) -> tuple[int, int]:
    """Scale to an area budget, then align both axes to a whole number of patches.

    Unlike the rest of this module this one is not free to differ from upstream: the
    model is trained on images preprocessed this way, so a different rule would feed
    it out-of-distribution input.  The rule is therefore written to a *behavioural*
    spec — area budget, both axes multiples of 14, shrink whichever axis overshoots
    the target aspect until the budget is met — and that behaviour is pinned by
    a measured parity test against the vendored loader
    (github.com/Junyi42/LoGeR @ 7685b7a, ``loger/utils/basic.py:55-61``, inside
    ``load_images_as_tensor``, whose signature is at ``basic.py:11``), not asserted.
    Lines 62-63 there are a Target_W/Target_H override we deliberately do not
    reimplement — we always compute the size, never accept one.  See
    ``test_target_size_matches_the_vendored_loader``.

    The two axes round **independently**, so exact aspect ratio is not preserved —
    the image is stretched by up to a few percent on one axis.  That is in
    distribution (the model trains with this preprocessing) and is absorbed into K,
    because ``estimate_intrinsics_from_points`` fits fx and fy separately and
    ``LoGeRCreator.camera_model`` is ``"PINHOLE"``.  Cropping instead would discard
    field of view and reintroduce crop arithmetic in ``original_coords``.

    ``pixel_limit`` is a budget, not a hard ceiling, and the difference only shows at
    absurd aspect ratios: past roughly 1300:1 one axis rounds to zero patches, the
    shrink loop never runs because ``0 > pixel_limit`` is false, and the ``max(1, ...)``
    clamp resurrects that axis to a single patch — returning slightly more area than
    asked for, up to 9x at 100000:1.  A ``pixel_limit`` under 196 likewise always
    returns 14x14.  Both are upstream's behaviour and are kept deliberately; the
    ``w * h <= pixel_limit`` assertion in the tests holds for every real image shape,
    not for every input.
    """
    # Area-budget scale factor. Upstream guards this against zero area and falls through
    # to a 14x14 image; we do not carry that over, because the caller this exists for
    # (`_preprocess`, Task 6) passes frame store dimensions, which are positive by
    # construction. A zero here means the store is corrupt, and a ZeroDivisionError
    # naming this line is a more useful failure than a silent 14x14 tensor that the
    # model would happily consume.
    scale = math.sqrt(pixel_limit / (orig_w * orig_h))
    w_target, h_target = orig_w * scale, orig_h * scale

    # Round each axis to a whole number of patches, then shrink whichever axis is
    # furthest above the target aspect until the budget is met.
    patches_w, patches_h = round(w_target / _PATCH), round(h_target / _PATCH)
    while (patches_w * _PATCH) * (patches_h * _PATCH) > pixel_limit:
        if patches_w / patches_h > w_target / h_target:
            patches_w -= 1
        else:
            patches_h -= 1

    return max(1, patches_w) * _PATCH, max(1, patches_h) * _PATCH
