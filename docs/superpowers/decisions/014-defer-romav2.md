# 014 — Defer RoMaV2 dense matcher integration

**Date:** 2026-07-08
**Status:** Accepted

## Context

While integrating LoMa as a localization matcher
(spec: ../specs/2026-07-08-loma-matcher-integration-design.md), we evaluated
RoMaV2 (github.com/Parskatt/RoMaV2) — the current top dense matcher
(Mega-1500 pose AUC@5 62.8, ScanNet-1500 34.0).

## Decision

Do not integrate RoMaV2 now. Reasons:

1. Its required `fused-local-corr` CUDA kernel pins `torch==2.11.0`, and
   `torchvision>=0.23` implies torch>=2.8 — our stack is torch 2.5.1+cu121.
2. DINOv3 backbone weights carry Meta's custom DINOv3 license
   (the 1045 MB release checkpoint embeds DINOv3-derived weights).
3. Dense warp output does not fit the `BaseLocalExtractor` keypoint
   extract/match contract — it would need a separate dense-matcher path.
4. Open stability issues upstream: VRAM leak in the fused kernel (#40),
   OOM (#15, #17), high outlier ratios (#21).
5. LoMa-G already matches or beats RoMa v2 on WxBS and IMC22, with clean
   MIT/Apache licensing and no custom CUDA ops.

## Revisit trigger

Torch stack upgrade to >=2.8 (kernel pin may also relax upstream). Re-check
the DINOv3 license question at that point.
