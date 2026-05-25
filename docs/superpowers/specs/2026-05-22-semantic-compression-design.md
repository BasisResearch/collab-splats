# Semantic Feature Compression for 3D Lifting

**Date:** 2026-05-22  
**Status:** Approved for implementation  
**Branch:** refactor/core-modules

## Context

Lifting semantic patch features (DINOv2, MaskCLIP) into 3D pointclouds via `lift_features()` stores raw D-dimensional vectors (384–1024) per 3D point. At scale this is expensive. We want 8–32 dims per point with decode-at-query-time reconstruction, using a scene-agnostic shallow autoencoder trained standalone and plugged into the existing feedforward pipeline.

This is not a replacement for the Gaussian splatting MLP path — it is an alternative for the feedforward path only.

## Architecture

**New file:** `collab_splats/semantics/compression.py`

**`FeatureAutoencoder(nn.Module)`**

```python
self.encoder = nn.Sequential(Linear(d_in, hidden), ReLU(), Linear(hidden, k))
self.decoder = nn.Sequential(Linear(k, hidden), ReLU(), Linear(hidden, d_in))
```

- `D` = source feature dim (e.g. 384 for DINOv2-small)
- `K` = compressed dim, 8–32
- `H` = `max(64, 2 * K)` (hidden dim)

**Entry points** — flat `(N, D)` ↔ `(N, K)` throughout. Simple, pipeline-agnostic. Caller reshapes at spatial boundaries.

- `encode(x: Tensor) -> Tensor` — `(N, D) → (N, K)`
- `decode(x: Tensor) -> Tensor` — `(N, K) → (N, D)`

## Training Setup

**`FeatureAutoencoder.fit(features, epochs=10, batch_size=1024, lr=1e-3, lr_scheduler=None)`**

- `features`: `(N_patches, D)` tensor — caller collects via existing `extractor.forward()` + `extractor.debias()` + flatten
- Loss: `MSE(decode(encode(x)), x) + (1 - cosine_similarity(decode(encode(x)), x).mean())`
  - MSE preserves magnitude; cosine term preserves direction for downstream cosine-sim queries
- Optimizer: Adam
- Optional `lr_scheduler`: pass any `torch.optim.lr_scheduler` instance, called each epoch

**Persistence** — single checkpoint with hyperparams + full state dict:

```python
ae.save(path: Path)           # writes autoencoder.pt (d_in, k, hidden + state_dict)
FeatureAutoencoder.load(path) # classmethod; reconstructs architecture from saved hyperparams
```

Single file avoids duplicating hyperparams across encoder/decoder halves.

**Standalone training (notebook):**

```python
extractor = DINOFeatureExtractor()
frames = sample_frames_optical_flow(video_path, n_frames=200)
features = torch.cat([
    extractor.debias(extractor.forward(frame)).flatten(1).T   # (D,H,W) → (N_p, D)
    for frame in frames
])

ae = FeatureAutoencoder(d_in=384, k=16)
ae.fit(features, epochs=10, batch_size=1024)
ae.save(Path("checkpoints/dino384_k16"))
```

## Integration Points

No changes to `lift_features()`, `score_queries()`, or `VGGTXCreator`.

**Encode before lift** — caller reshapes to/from spatial for `lift_features()`:
```python
feats = extractor.debias(extractor.forward(frame))          # (D, H_p, W_p)
feats_k = ae.encode(feats.flatten(1).T)                     # (N_p, K)
feats_k = feats_k.T.reshape(ae.k, *feats.shape[1:])        # (K, H_p, W_p)
result = lift_features(feats_k, pixel_indices, ...)
```

**Decode before query** — lifted features are already flat `(N_points, K)`:
```python
scores = extractor.score_queries(ae.decode(result.features), positive=[...], negative=[...])
```

Autoencoder is optional middleware — skip it and both pipelines work as before.

## Verification

1. **Reconstruction cosine sim:** `F.cosine_similarity(ae.decode(ae.encode(x)), x).mean()` — target >0.90
2. **Query rank preservation:** Spearman rank correlation between `score_queries()` on original vs decoded features — target >0.85
3. **PCA visualization:** `pca_to_rgb()` on decoded lifted features vs original — should be visually similar

## Files

| Action | Path |
|--------|------|
| New | `collab_splats/semantics/compression.py` |
| New | `tests/semantics/test_compression.py` |
| Tutorial | `docs/source/tutorials/semantics/semantic_compression.ipynb` |

## Out of Scope

- PyTorch Lightning / advanced schedulers (vanilla Adam sufficient for 2-layer MLP)
- `collect_features()` helper (caller uses existing extractor tools)
- Gaussian splatting integration (feedforward path only)
- Scene-specific fine-tuning (model is scene-agnostic)
