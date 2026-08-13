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

import inspect
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from ...geometry.transforms import estimate_intrinsics_from_points, extrinsics_to_homogeneous, invert_poses
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _raw_to_world_points,
    compute_multiview_depth_confidence,
)
from .vggtx import unproject_and_filter_points

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
    # but is a forward kwarg, so it cannot ride along in the constructor kwargs. None
    # means _load_model has not run yet; False is a valid post-load value, so reusing
    # it as the unset sentinel would let an unset flag silently read as LoGeR mode.
    # Task 7's _forward must treat None as a contract violation, not default it.
    _se3: bool | None = field(default=None, init=False, repr=False)

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

        # `or {}` twice: an empty file parses to None, and `model:` with no body parses to
        # None under the key. Neither can be allowed through — an empty model_cfg builds Pi3
        # on constructor defaults, which is a DIFFERENT architecture (ttt_inter_multi is 4 in
        # both shipped configs and 2 in the constructor), so the run fails 278 state_dict keys
        # later, after a 5 GB download, with an error that names none of this.
        model_cfg = dict((yaml.safe_load(cfg_path.read_text()) or {}).get("model") or {})
        if not model_cfg:
            raise ValueError(f"{cfg_path} has no 'model:' block; the config is empty or truncated")

        # se3 sits under model: but is not a Pi3.__init__ parameter — it is popped
        # inside forward (github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:589).
        # Route it out before validating the rest, or the check below would reject it.
        self._se3 = bool(model_cfg.pop("se3", False))

        # The vendored tree is not pip-installed, so the module has to be reached via
        # sys.path rather than a top-of-file import. The flag keeps the finally
        # idempotent: without it a nested load would pop a path its caller installed.
        # sys.path is released before the download/checkpoint load below, not held across
        # them — the house pattern (vggt_spark_creator.py:124-132) closes it immediately
        # after the import + construction that actually need it.
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

            # The yaml overrides Pi3's own defaults, which are wrong for both shipped
            # variants — ttt_inter_multi is 4 in each config and 2 in the constructor.
            model = Pi3(**model_cfg)
        finally:
            # Guarded so a ValueError raised above can never be chained over by a
            # spurious error removing a path something else already popped.
            if _patched and root in sys.path:
                sys.path.remove(root)

        ckpt = self.model_path or hf_hub_download(repo_id=self.model_repo, filename=f"{self.variant}/latest.pt")
        # weights_only=True: torch 2.5.1 still defaults to False and warns. The
        # checkpoint comes from a third-party HuggingFace repo, so restricting the
        # unpickler is worth one kwarg. Verified against both real checkpoints.
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        logger.info("LoGeRCreator: loaded %s (se3=%s) on %s", self.variant, self._se3, device)
        return model.eval().to(device)

    def _preprocess(self, frames: Any, frame_idxs: list[int]) -> tuple[Any, list[Path], np.ndarray]:
        """Resize decoded frames to LoGeR's patch-aligned budget; return (N,3,H,W) in [0,1]."""
        # Windows and overlap stitching assume temporal order, and equal indices would
        # additionally collide in the frame_{idx:06d} labels below. frames.zarr is ordered
        # by construction today, so this guards an assumption rather than a known bug.
        # Report the offending pair, not frame_idxs itself: at the 300-frame budget that
        # would put a 300-element list in the traceback and bury the one bad index.
        for i, (a, b) in enumerate(zip(frame_idxs, frame_idxs[1:])):
            if b <= a:
                raise ValueError(
                    f"LoGeR requires strictly ascending frame_idxs (sliding-window inference); "
                    f"got frame_idxs[{i}]={a} >= frame_idxs[{i + 1}]={b}; sort before calling"
                )

        # LoGeR derives the target size from frame 0 alone
        # (github.com/Junyi42/LoGeR @ 7685b7a, loger/utils/basic.py:53-54, inside
        # load_images_as_tensor at basic.py:11). Rather than inherit that silent
        # assumption, refuse mixed sizes.
        shapes = {(int(f.shape[0]), int(f.shape[1])) for f in frames}
        if len(shapes) != 1:
            raise ValueError(
                f"LoGeR needs uniform frame sizes; got {sorted(shapes)}. All frames must share "
                "one resolution; re-extract the frame store."
            )

        orig_h, orig_w = shapes.pop()
        target_w, target_h = _compute_target_size(orig_w, orig_h, self.pixel_limit)
        logger.debug("LoGeRCreator: %dx%d -> %dx%d", orig_w, orig_h, target_w, target_h)

        # Resize in memory with PIL directly. No frames_as_pil_source
        # (collab_splats/pointcloud/feedforward/base.py:679): that helper monkeypatches the
        # process-global PIL.Image.open to drive path-based loaders, and LoGeR's
        # load_images_as_tensor enumerates a directory with os.listdir
        # (github.com/Junyi42/LoGeR @ 7685b7a, loger/utils/basic.py:21), which patching
        # Image.open cannot reach.
        resized = np.stack(
            [np.asarray(Image.fromarray(f).resize((target_w, target_h), Image.LANCZOS)) for f in frames]
        )
        # div_ rather than `/ 255.0`: the out-of-place divide would hold two full float32
        # copies at once, and this is the backend built for long sequences — measured at
        # 300 frames of 1080p that second copy is 914 MB. Safe in place because `resized`
        # is uint8 (PIL RGB always decodes to uint8), so .float() always allocates a fresh
        # tensor and never aliases the numpy buffer.
        views = torch.from_numpy(resized).permute(0, 3, 1, 2).float().div_(255.0)

        # Stable synthetic labels — the frame store is the sole IO path, no filenames exist
        image_paths = [Path(f"frame_{idx:06d}") for idx in frame_idxs]

        # Pure resize, no crop, so every row is the full original frame. Layout is
        # [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], consumed by
        # _rescale_reconstruction_to_original_dimensions (base.py:580).
        original_coords = np.tile(
            np.array([0, 0, orig_w, orig_h, orig_w, orig_h], dtype=np.float32), (len(image_paths), 1)
        )

        return views, image_paths, original_coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run windowed LoGeR inference; return raw outputs plus a solved shared K."""
        # _se3 is populated by _load_model from the variant's yaml. None means _forward was
        # reached without it — refuse rather than pick a default, because both values are
        # legitimate (LoGeR is False, LoGeR_star is True) and guessing runs the wrong
        # alignment mode with no error anywhere downstream.
        if self._se3 is None:
            raise RuntimeError("LoGeRCreator._forward requires _load_model to have run (se3 unset)")

        device = next(model.parameters()).device
        images = views.to(device)

        # Guards the a157421 [0,255] bug class at the source rather than at the mesh.
        assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0, (
            f"LoGeR expects RGB in [0, 1]; got [{float(images.min())}, {float(images.max())}]"
        )

        # Pi3 takes (B, N, 3, H, W). Kwargs mirror build_forward_kwargs in
        # github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:149-164, so behaviour matches
        # upstream exactly; each name is popped in Pi3.forward at
        # github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:584-593. sim3 stays False
        # unconditionally: it and se3 are mutually exclusive and raise together (same file,
        # :595-596), so LoGeR_star's se3=True has no valid sim3 counterpart.
        with torch.no_grad():
            preds = model(
                images[None],
                window_size=self.window_size,
                overlap_size=self.overlap_size,
                reset_every=self.reset_every,
                num_iterations=self.num_iterations,
                sim3=False,
                sim3_scale_mode="median",
                se3=self._se3,
                turn_off_ttt=False,
                turn_off_swa=False,
            )

        local_points = preds["local_points"].squeeze(0).cpu().float().numpy()  # (N,H,W,3)

        # conf_head is a bare LinearPts3d with NO output activation —
        # github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:172 — so the model emits
        # logits, and upstream activates at the call site (github.com/PolyCam/LoGeR @
        # 5d7c1a7, run_loger.py:481). This must run before the K fit, whose conf gate is a
        # threshold on a probability. Measured on the Task 1 run, the raw logits span
        # -4.257..-2.019 — entirely negative — so skipping the sigmoid does not merely
        # shift the gate, it admits ZERO pixels and the fit raises.
        depth_conf = torch.sigmoid(preds["conf"]).squeeze(0).cpu().float().numpy()
        if depth_conf.ndim == 4:
            depth_conf = depth_conf.squeeze(-1)  # (N,H,W)

        # LoGeR returns camera-to-world; FeedforwardResult.extrinsics is world-to-camera.
        camera_poses = preds["camera_poses"].squeeze(0).cpu().float().numpy()  # (N,4,4) c2w
        extrinsic = invert_poses(camera_poses)[:, :3, :].astype(np.float32)  # (N,3,4) w2c

        # LoGeR predicts no intrinsics — solve one shared K and broadcast it per frame. The
        # threshold is passed EXPLICITLY rather than inheriting estimate_intrinsics_from_points'
        # 0.1 default, which this uncalibrated conf head cannot clear; see LOGER_CONF_THRESHOLD.
        k = estimate_intrinsics_from_points(local_points, depth_conf, LOGER_CONF_THRESHOLD)
        intrinsic = np.broadcast_to(k, (local_points.shape[0], 3, 3)).copy()

        # Channel 2 IS depth: the model builds local_points as cat([xy * z, z]) at
        # github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:772-775.
        depth = local_points[..., 2:3]  # (N,H,W,1)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,
            "intrinsics_downsampled": intrinsic,  # alias — _raw_to_world_points needs this key
            "depth": depth,
            "depth_conf": depth_conf,
            "local_points": local_points,  # kept for the parity test only
        }

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """Unproject depth with the fitted K and build the FeedforwardResult."""
        extrinsic = raw_outputs["extrinsic"]  # (N,3,4) at model resolution
        intrinsic = raw_outputs["intrinsics"]  # (N,3,3) at model resolution

        # Optional geometric cross-view depth consistency mask
        mv_mask = None
        if self.use_multiview_confidence:
            depth_np = raw_outputs["depth"]
            if depth_np.ndim == 4:
                depth_np = depth_np.squeeze(-1)
            mv_conf = compute_multiview_depth_confidence(
                depth_np,
                intrinsic,
                extrinsics_to_homogeneous(extrinsic),
                abs_thresh=0.0,
                rel_thresh=0.05,
            )
            mv_mask = mv_conf > self.mv_conf_threshold

        # Unproject to filtered world-space points and per-point colors. conf_threshold > 1.0
        # is read as a percentile by this function (vggtx.py:132-136), which is why the
        # default 50.0 is a percentile and not a probability.
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
        )

        # FeedforwardResult.depth is (N,H,W); _forward emits (N,H,W,1) from the pointmap's
        # third channel, so the trailing axis is squeezed to match every other backend.
        depth = raw_outputs["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(-1)  # (N,H,W)
        model_h, model_w = int(depth.shape[1]), int(depth.shape[2])

        # BA fields: dense world-point grid, matching vggtx and vggt_omega. LoGeR's own
        # `points` is NOT used here — it can encode non-pinhole geometry that the fitted
        # K cannot reproduce, so feeding it to BA alongside that K makes BA fight the
        # model. The parity test measures the gap between the two clouds.
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
        world_points = (
            world_pts_flat.reshape(world_pts_flat.shape[0], model_h, model_w, 3)
            if world_pts_flat is not None
            else None
        )

        # confidence is a torch.Tensor and depth an np.ndarray by declaration
        # (base.py:72 vs :74); the asymmetry is the dataclass contract, not an oversight.
        return FeedforwardResult(
            points=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            features=None,
            extrinsics=extrinsics_to_homogeneous(extrinsic),
            intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=raw_outputs["images"],
            confidence=torch.from_numpy(raw_outputs["depth_conf"]),
            world_points=world_points,
            depth=depth,
        )

    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
        # The pose/K arguments, NOT raw_outputs' stored copies: BundleAdjustment calls this
        # precisely because it has just refined them, so reading the raw dict would silently
        # return the pre-BA cloud.
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
        )
        return pts3d, colors  # pixel_indices unused; post-BA uses stored indices

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Not supported — LoGeR carries its own windowed TTT memory across frames."""
        # Satisfying the ABC contract, not a courtesy stub: the class will not instantiate
        # without it, and _verify_loop_candidate (concrete on the base class, base.py:960)
        # calls it at base.py:991. Reaching here means the Reconstructor-level loop closure
        # refusal was bypassed.
        raise NotImplementedError(
            "LoGeR does not support loop closure feature extraction. Its windowed TTT "
            "fast-weight memory already carries state across frames, and LC verification "
            "thresholds are calibrated per backbone (see the spec). Use vggt_omega, vggtx, "
            "or mapanything if loop closure is required."
        )
