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
  LoGeRCreator         — feedforward creator using LoGeR depth + pose
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download

from .base import BaseFeedforwardCreator

logger = logging.getLogger(__name__)

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


########################################################################
########## Creator #####################################################
########################################################################


@dataclass
class LoGeRCreator(BaseFeedforwardCreator):
    """Pointcloud via LoGeR: Pi3 backbone + TTT memory + sliding-window inference.

    Built for long sequences — the window bounds model memory regardless of sequence
    length, where the set-based VGGT family OOMs past a few hundred frames.

    Unlike every other backend, LoGeR predicts no intrinsics; K is solved from its
    camera-frame pointmap by ``estimate_intrinsics_from_points`` and shared across frames.

    Attributes:
        camera_model:   pycolmap camera model.  ``"PINHOLE"``, not ``"SIMPLE_PINHOLE"``,
                        because the fit produces genuinely distinct fx and fy and
                        SIMPLE_PINHOLE averages them away at COLMAP export.
        variant:        ``"LoGeR"`` or ``"LoGeR_star"``.  Selects a config-plus-weights
                        pair, not merely a weight file — the two ``original_config.yaml``
                        differ (``ttt_pre_norm`` vs ``se3``).
        model_path:     Local checkpoint override.  ``None`` downloads from HuggingFace.
        window_size:    Sliding-window length.  Default from github.com/PolyCam/LoGeR @
                        5d7c1a7, ``run_loger.py:47`` (argparse).
        overlap_size:   Frames shared between adjacent windows.  Same file, ``:49``.
        reset_every:    Hard-reset the TTT fast weights every N frames; ``0`` disables.
        num_iterations: TTT inner-loop iterations per step.
        pixel_limit:    Area budget for the resize.  Same file, ``:117``.
        conf_threshold: Depth-confidence **percentile** cutoff (0-100), matching
                        ``vggt_omega``.  ``unproject_and_filter_points`` reads a value
                        > 1.0 as a percentile and <= 1.0 as a raw confidence, so
                        lowering this to e.g. ``0.5`` switches semantics rather than
                        tightening the cut.
    """

    camera_model: str = "PINHOLE"

    variant: str = "LoGeR_star"
    model_path: str | None = None
    model_repo: str = LOGER_HF_REPO

    # Window knobs. These do NOT come from the shipped yaml: both original_config.yaml
    # files contain only a model: key, so build_forward_kwargs
    # (github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:149-164) always falls through
    # to its own fallbacks, which are these values.
    window_size: int = 32
    overlap_size: int = 3
    reset_every: int = 0
    num_iterations: int = 1

    pixel_limit: int = 255_000
    conf_threshold: float = 50.0
    use_multiview_confidence: bool = False
    mv_conf_threshold: float = 0.0

    # Resolved in _load_model from the variant's yaml. se3 is declared under model:
    # but is a forward kwarg, so it cannot ride along in the constructor kwargs.
    _se3: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Reject at construction rather than at _load_model, so a typo does not survive
        # until after a multi-GB checkpoint download.
        if self.variant not in LOGER_VARIANTS:
            raise ValueError(f"variant must be one of {LOGER_VARIANTS}, got {self.variant!r}")

    def _load_model(self, device: str) -> Any:
        """Build Pi3 from the vendored per-variant yaml and load the HF checkpoint."""
        cfg_path = _LOGER_ROOT / "ckpts" / self.variant / "original_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"LoGeR config not found: {cfg_path}. Run `bash setup/loger.sh` to vendor the tree."
            )

        model_cfg = dict(yaml.safe_load(cfg_path.read_text()).get("model", {}))

        # se3 sits under model: but is not a Pi3.__init__ parameter — it is popped
        # inside forward (github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:589).
        # Route it out before validating the rest, or the check below would reject it.
        self._se3 = bool(model_cfg.pop("se3", False))

        # The vendored tree is not pip-installed, so the module has to be reached via
        # sys.path rather than a top-of-file import. The flag keeps the finally
        # idempotent: without it a nested load would pop a path its caller installed.
        root = str(_LOGER_ROOT)
        _patched = root not in sys.path
        if _patched:
            sys.path.insert(0, root)
        try:
            from loger.models.pi3 import Pi3

            # Every remaining model: key must be a real constructor parameter. Raise
            # rather than drop — a silent drop is how a future forward-only key would
            # degrade the run invisibly, exactly as se3 would have.
            unknown = sorted(set(model_cfg) - set(inspect.signature(Pi3.__init__).parameters))
            if unknown:
                raise ValueError(
                    f"{cfg_path} 'model:' holds keys that are neither Pi3.__init__ "
                    f"parameters nor the known forward kwarg 'se3': {unknown}. "
                    "Upstream changed the config; route them explicitly rather than dropping them."
                )

            # Some checkpoints serialise list fields as strings, e.g. "[4,8]".
            for key in ("ttt_insert_after", "attn_insert_after"):
                if isinstance(model_cfg.get(key), str):
                    model_cfg[key] = ast.literal_eval(model_cfg[key])

            # The yaml overrides Pi3's own defaults, which are wrong for both shipped
            # variants — ttt_inter_multi is 4 in each config and 2 in the constructor.
            model = Pi3(**model_cfg)

            ckpt = self.model_path or hf_hub_download(
                repo_id=self.model_repo, filename=f"{self.variant}/latest.pt"
            )
            state = torch.load(str(ckpt), map_location="cpu")
            state = state.get("model_state_dict", state)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        finally:
            if _patched:
                sys.path.remove(root)

        logger.info("LoGeRCreator: loaded %s (se3=%s) on %s", self.variant, self._se3, device)
        return model.eval().to(device)
