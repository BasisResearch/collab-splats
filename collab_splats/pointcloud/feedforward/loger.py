"""
LoGeR feedforward backend: sliding-window Pi3 with a test-time-trained memory.

- predicts no intrinsics; one K is fitted from the camera-frame pointmap
- vendored: github.com/Junyi42/LoGeR @ 7685b7a (setup/loger.sh -> third_party/LoGeR/)
- cites: loger/models/pi3.py:59, 172, 584-596, 772-775; loger/utils/basic.py:21, 53-63
- read, not a dependency: github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:47-164, 481
- neither fork ships a LICENSE; nothing is copied
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from collab_splats.geometry.transforms import (
    estimate_intrinsics_from_points,
    invert_poses,
)
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
from collab_splats.utils.torch_utils import load_hf_weights, vendored_path

logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

LOGER_HF_REPO = "Junyi42/LoGeR"
LOGER_VARIANTS = ("LoGeR", "LoGeR_star")

# Location of the vendored LoGeR code, installed by setup/loger.sh
_LOGER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "LoGeR"

# Model image width and height must be multiples of this patch size
_LOGER_PATCH = 14

########################################################################
# Creator
########################################################################


@BaseFeedforwardCreator.register("loger")
@dataclass
class LoGeRCreator(BaseFeedforwardCreator):
    """
    Pointcloud from LoGeR's windowed inference, for sequences too long for the VGGT family.

    - memory is bounded by the window, not the sequence length
    - one K is fitted from the camera-frame pointmap and shared across frames
    - multiview-filter defaults are the VGGT family's, not tuned for LoGeR

    Attributes:
        variant: "LoGeR" or "LoGeR_star"; each is a config and weights pair.
        model_path: local checkpoint; None downloads from HuggingFace.
        k_fit_conf_threshold: post-sigmoid confidence floor for the shared-K fit.
        window_size: frames per sliding window.
        overlap_size: frames shared between adjacent windows.
        reset_every: reset the memory every N frames; 0 disables.
        num_iterations: memory-update iterations per window.
        pixel_limit: area budget for the resize, in px².
    """

    variant: str = "LoGeR_star"
    model_path: str | None = None

    # Confidence cutoff for fitting the shared intrinsics, kept low because LoGeR's scores are small
    k_fit_conf_threshold: float = 0.02

    # Sliding-window settings, copied from upstream's defaults because the shipped config has none
    window_size: int = 32
    overlap_size: int = 3
    reset_every: int = 0
    num_iterations: int = 1

    pixel_limit: int = 255_000

    # Whether the model runs in se3 mode, read from the config by _load_model (None until then)
    _se3: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Reject an unknown variant before any multi-GB checkpoint download.
        """
        if self.variant not in LOGER_VARIANTS:
            raise ValueError(f"variant must be one of {LOGER_VARIANTS}, got {self.variant!r}")

    def _load_model(self, device: str) -> Any:
        """
        Build Pi3 from the vendored variant yaml and load its checkpoint.

        - FileNotFoundError: the vendored config is missing
        - ValueError: the config's model: block is empty or holds unknown keys
        """
        # Path to this variant's vendored config file
        cfg_path = _LOGER_ROOT / "ckpts" / self.variant / "original_config.yaml"

        if not cfg_path.exists():
            raise FileNotFoundError(
                f"LoGeR config not found: {cfg_path}. Run `bash setup/loger.sh` to vendor the tree."
            )

        # Read the model settings from the config and refuse an empty one
        model_cfg = dict((yaml.safe_load(cfg_path.read_text()) or {}).get("model") or {})

        if not model_cfg:
            raise ValueError(f"{cfg_path} has no 'model:' block; the config is empty or truncated")

        # se3 is passed when running the model, not when building it, so take it out of the settings
        self._se3 = bool(model_cfg.pop("se3", False))

        # Import and build the Pi3 model from the vendored code
        with vendored_path(_LOGER_ROOT, "run `bash setup/loger.sh` to vendor the tree"):
            from loger.models.pi3 import Pi3

            # Refuse unknown settings, since silently dropping them would change the model
            unknown = sorted(set(model_cfg) - set(inspect.signature(Pi3.__init__).parameters))

            if unknown:
                raise ValueError(
                    f"{cfg_path} 'model:' holds keys that are neither Pi3.__init__ "
                    f"parameters nor the known forward kwarg 'se3': {unknown}. "
                    "Upstream changed the config; route them explicitly rather than dropping them."
                )

            # Build the model, with the config overriding Pi3's default settings
            model = Pi3(**model_cfg)

        # Load the checkpoint, with weights_only protecting against unsafe pickled files
        ckpt = self.model_path or load_hf_weights(LOGER_HF_REPO, f"{self.variant}/latest.pt")
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        logger.info("LoGeRCreator: loaded %s (se3=%s) on %s", self.variant, self._se3, device)

        return model.eval().to(device)

    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        """
        Resize frames to LoGeR's patch-aligned area budget.

        - paths in temporal order; all frames must share one size, else ValueError
        - returns (N, 3, H, W) float32 images in [0, 1] and (N, 6) original_coords
        """
        # Refuse frames of different sizes, since LoGeR sizes every frame from the first one
        sizes = {Image.open(p).size for p in paths}

        if len(sizes) != 1:
            raise ValueError(
                f"LoGeR needs uniform frame sizes; got (w, h) {sorted(sizes)}. All frames must share "
                "one resolution; re-extract the frame store."
            )

        orig_w, orig_h = sizes.pop()
        target_w, target_h = _compute_target_size(orig_w, orig_h, self.pixel_limit)
        logger.debug("LoGeRCreator: %dx%d -> %dx%d", orig_w, orig_h, target_w, target_h)

        # Resize with PIL, as LoGeR's own directory loader does
        imgs = [Image.open(p).convert("RGB").resize((target_w, target_h), Image.LANCZOS) for p in paths]
        resized = np.stack([np.asarray(img) for img in imgs])

        # Convert to float and scale to [0, 1] in place to avoid an extra copy
        views = torch.from_numpy(resized).permute(0, 3, 1, 2).float().div_(255.0)

        # No cropping, so every frame's crop box is the whole original frame
        row = np.array([0, 0, orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        original_coords = np.tile(row, (len(paths), 1))

        return views, original_coords

    def _forward_kwargs(self) -> dict:
        """
        Kwargs for one Pi3 forward pass, mirroring upstream's build_forward_kwargs.

        - RuntimeError: _load_model has not run
        """
        # Refuse to run before se3 is known, since guessing would be wrong for one of the variants
        if self._se3 is None:
            raise RuntimeError("LoGeRCreator._forward requires _load_model to have run (se3 unset)")

        # sim3 stays off because Pi3 fails when sim3 and se3 are both on
        return dict(
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

    def _forward(self, model: Any, views: Any) -> dict:
        """
        Windowed inference, plus one shared K fitted from the camera-frame pointmap.

        - views: (N, 3, H, W) images in [0, 1]; ValueError outside that range
        - returns images, extrinsic (N, 3, 4) w2c, intrinsics, depth and depth_conf
        """
        # Build the settings first so a missing se3 fails before any GPU work
        forward_kwargs = self._forward_kwargs()

        # Move the images to the model's device
        device = next(model.parameters()).device
        images = views.to(device)

        # Check that the images are in the [0, 1] range
        lo, hi = float(images.min()), float(images.max())

        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"LoGeR expects RGB in [0, 1]; got [{lo}, {hi}]")

        # Run the model, adding the batch dimension it expects
        with torch.no_grad():
            preds = model(images[None], **forward_kwargs)

        # Pointmap in each camera's own frame
        local_points = preds["local_points"].squeeze(0).cpu().float().numpy()  # (N,H,W,3)

        # Turn the raw confidence scores into probabilities, which the intrinsics fit below expects
        depth_conf = torch.sigmoid(preds["conf"]).squeeze(0).cpu().float().numpy()

        if depth_conf.ndim == 4:
            depth_conf = depth_conf.squeeze(-1)  # (N,H,W)

        # Turn LoGeR's camera-to-world poses into world-to-camera
        camera_poses = preds["camera_poses"].squeeze(0).cpu().float().numpy()  # (N,4,4) c2w
        extrinsic = invert_poses(camera_poses)[:, :3, :].astype(np.float32)  # (N,3,4) w2c

        # Fit one set of intrinsics and copy it for every frame
        k = estimate_intrinsics_from_points(local_points, depth_conf, self.k_fit_conf_threshold)
        intrinsic = np.broadcast_to(k, (local_points.shape[0], 3, 3)).copy()

        # Depth is the z coordinate of the camera-frame pointmap
        depth = local_points[..., 2:3]  # (N,H,W,1)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,
            "depth": depth,
            "depth_conf": depth_conf,
        }


########################################################################
# Preprocessing helpers
########################################################################


def _compute_target_size(orig_w: int, orig_h: int, pixel_limit: int) -> tuple[int, int]:
    """
    Scale to an area budget, then round both axes to whole 14-px patches.

    - matches LoGeR's own loader; pinned by a parity test
    - axes round independently, so aspect may stretch a few percent
    - extreme aspect or a pixel_limit under 196 can exceed the budget, as upstream does
    - sizes in px, pixel_limit in px²; returns (width, height), both multiples of 14
    """
    # Scale factor that fits the image into the pixel budget
    scale = math.sqrt(pixel_limit / (orig_w * orig_h))  # an empty frame raises here on purpose
    w_target, h_target = orig_w * scale, orig_h * scale

    # Round to whole patches, then shrink one side at a time until the image fits the budget
    patches_w, patches_h = round(w_target / _LOGER_PATCH), round(h_target / _LOGER_PATCH)

    while (patches_w * _LOGER_PATCH) * (patches_h * _LOGER_PATCH) > pixel_limit:
        if patches_w / patches_h > w_target / h_target:
            patches_w -= 1
        else:
            patches_h -= 1

    return max(1, patches_w) * _LOGER_PATCH, max(1, patches_h) * _LOGER_PATCH
