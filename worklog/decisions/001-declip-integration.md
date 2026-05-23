# ADR 001: DeCLIP Integration

- **Status:** Deferred
- **Date:** 2026-04-28
- **Deciders:** Tommy
- **Re-evaluate when:** Talk2DINOv3 baseline shows quality ceiling on our scenes, OR DeCLIP authors publish a clean HF Hub / PyPI release.

## Context

[DeCLIP](https://github.com/xiaomoguhz/DeCLIP) (CVPR 2025, Apache-2.0) is an unsupervised fine-tuning framework for open-vocabulary dense perception. It decouples CLIP self-attention into "content" and "context" branches, distilling the context branch from vision foundation models (DINOv2, SAM) to improve dense feature quality. Reports SOTA on open-vocab segmentation/detection benchmarks vs MaskCLIP/CLIPSelf/ProxyCLIP.

We currently have three dense extractors registered via `BaseQueryableExtractor` in `collab_splats/semantics/features.py`:

- `MaskCLIPExtractor` — patch-level CLIP via `maskclip_onnx`
- `DINOFeatureExtractor` — DINOv2 patches (regularization only, no text)
- `Talk2DinoExtractor` — DINOv3 + text via HF Hub `lorebianchi98/Talk2DINOv3-ViTB`, recent dense open-vocab method, single-line load

Question raised: replace or complement MaskCLIP with DeCLIP?

## Findings

### Inference API

Clean. ~15 lines:

```python
model = create_model("EVA02-CLIP-L-14-336", "eva", pretrained_hf=True,
                    cache_dir="<declip_weights>").eval().to(device)
features = model.encode_dense(img, normalize=True, keep_shape=True, mode="qq")
```

Returns `(B, C, H, W)` directly. Cleaner than MaskCLIP patch reshape.

### Packaging — fundamentally not portable

- No PyPI / HF Hub release. Monorepo intended for training, not consumption.
- `setup.py` registers `name='open_clip_torch'` — collides with upstream `open_clip_torch` PyPI package. `pip install -e .` would silently hijack `import open_clip` system-wide.
- Pinned `torch==2.0.0`, `xformers==0.0.18`, `timm==0.4.12`, `torchvision==0.15.1`. All would downgrade nerfstudio env if installed via `requirements.txt`.
- xformers verified optional in source (try/except + vanilla PyTorch fallback). timm dual-pathed (old + new layers).

### Single-env integration path

Only viable route is **vendor-and-rename**:

1. Copy `src/open_clip/` → `collab_splats/_vendor/declip_open_clip/`
2. Rename top-level package to kill `import open_clip` collision
3. Search-replace internal imports (`from open_clip.X` → `from collab_splats._vendor.declip_open_clip.X`)
4. Skip their `setup.py` and `requirements.txt` entirely
5. Skip xformers (use vanilla fallback). Use nerfstudio env's existing torch/timm.
6. Wrap as `DeCLIPExtractor(BaseQueryableExtractor)` mirroring `Talk2DinoExtractor` pattern.
7. Weights via `huggingface_hub.snapshot_download` (~1 GB subset of the 65 GB HF dataset).

Estimated effort: ~1 focused day.

## Decision

**Defer integration.** Run Talk2DINOv3 vs MaskCLIP comparison on our scenes first.

## Reasoning

- Talk2DINOv3 is already integrated, also a 2025 dense open-vocab method, also competitive with DeCLIP/ProxyCLIP per Talk2DINO benchmarks. No vendoring, no env risk.
- "SOTA on COCO panoptic" ≠ "noticeably better on 3D feature lifting / point-cloud querying for our pipeline."
- Vendor-and-rename is acceptable design (precedent: pip, requests vendor `urllib3`) but adds ~70 KB frozen fork to the tree with permanent manual-sync debt. First instance of this pattern in our repo; sets convention.
- 1-day spike could show no gain → wasted work + sunk-cost pressure to keep code we don't need.

## Alternatives considered

- **Vendor-and-rename now.** Single env, but ~1 day work + permanent fork maintenance. Rejected absent evidence current extractors are insufficient.
- **Separate conda env.** Rejected — user explicitly wants seamless single-env integration.
- **Wait for upstream HF Hub / PyPI release.** Cheap. DeCLIP TODO list shows active dev. Possible in 2-3 months.
- **Drop DeCLIP entirely, evaluate ProxyCLIP / CLIP-DINOiser.** Possible but same packaging concern likely applies.

## Consequences

- No DeCLIP code in tree. No vendor maintenance.
- Risk: miss any genuine quality gain DeCLIP provides over Talk2DINOv3 on our task. Mitigated by re-evaluation trigger.
- Sets precedent: integrate research forks only when (a) measured baseline gap exists, or (b) upstream packages cleanly.

## Re-evaluation triggers

Open this ADR again when any of:

1. Talk2DINOv3 + MaskCLIP both fail on a target scene (concrete failure case, not anecdote).
2. DeCLIP authors publish to HF Hub or PyPI as a clean library.
3. A downstream task (segmentation panel, query UI) has measurable accuracy gap traceable to extractor quality.
