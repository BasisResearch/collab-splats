"""Semantics-specific utilities: contrastive scoring and patch alignment.

General torch utilities live in collab_splats.utils.torch_utils.
Re-exported here so existing callers don't need to update import paths.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


########################################################
########## Re-exports from collab_splats.utils.torch_utils
########################################################

# Canonical location: collab_splats.utils.torch_utils
from collab_splats.utils.torch_utils import (
    get_device,
    pytorch_gc,
    infer_batch_size,
    batch_iterator,
    load_hf_weights,
    load_torchhub_model,
)


########################################################
########## Contrastive scoring #########################
########################################################


def compute_semantic_contrast(
    raw_similarities: torch.Tensor,
    num_positive: int,
    temperature: float = 0.05,
    reduction: str = "max",
) -> torch.Tensor:
    """Contrastive scoring: how strongly positive queries match relative to negatives.

    When no negatives are present (num_positive == raw_similarities.shape[0]),
    falls back to raw reduction over positives — contrastive scoring is undefined
    without a negative to push against.

    Args:
        raw_similarities: (N_queries, N) dot-product similarities per patch.
        num_positive: rows [0:num_positive] are positive queries; rest are negative.
        temperature: scaling parameter τ. Lower = sharper. Ignored when no negatives.
        reduction: aggregation over positive queries:
            "max"  — each positive independently scored against all negatives via
                     binary softmax; max over per-positive scores. Use for distinct
                     concepts where any match counts.
            "pool" — positives averaged in similarity space before softmax; one
                     representative competes against all negatives. Use for synonymous
                     concepts that should be treated as one combined query.

    Returns:
        (N,) contrastive scores in [0, 1].
    """
    if reduction not in ("max", "pool"):
        raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")

    pos = raw_similarities[:num_positive]
    neg = raw_similarities[num_positive:]

    if neg.shape[0] == 0:
        return pos.max(dim=0).values if reduction == "max" else pos.mean(dim=0)

    if reduction == "max":
        scores = []
        for p_i in pos:
            stacked = torch.cat([p_i.unsqueeze(0), neg], dim=0)
            scores.append(stacked.div(temperature).softmax(dim=0)[0])
        return torch.stack(scores).max(dim=0).values

    if reduction == "pool":
        avg_pos = pos.mean(dim=0, keepdim=True)
        stacked = torch.cat([avg_pos, neg], dim=0)
        return stacked.div(temperature).softmax(dim=0)[0]

    raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")


########################################################################
# Shared token utilities
########################################################################


def _tokens_to_feature_map(
    tokens: torch.Tensor, input_h: int, input_w: int, patch_size: int
) -> torch.Tensor:
    """Reshape (N, D) patch tokens to (D, H_p, W_p), L2-normalized along channel dim."""
    ph = input_h // patch_size
    pw = input_w // patch_size
    assert tokens.shape[0] == ph * pw, (
        f"Expected {ph * pw} tokens for {input_h}x{input_w} "
        f"(patch_size={patch_size}), got {tokens.shape[0]}"
    )
    feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)  # (D, H_p, W_p)
    return F.normalize(feat, dim=0)


########################################################
########## Patch alignment #############################
########################################################


def interpolate_to_patch_size(
    img_bchw: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, int, int]:
    """Interpolate image tensor so H and W are evenly divisible by patch_size.

    Args:
        img_bchw: Image tensor of shape (B, C, H, W).
        patch_size: Patch dimension to align to.

    Returns:
        Tuple of (resized_tensor, target_H, target_W).
    """
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = F.interpolate(
        img_bchw, size=(target_H, target_W), mode="bilinear", align_corners=False
    )
    return img_bchw, target_H, target_W
