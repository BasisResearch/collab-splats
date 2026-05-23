# Positional Debiasing for BaseFeatureExtractor

**Date:** 2026-05-20
**Status:** Draft
**Reference:** [INSID3 insid3.py](https://github.com/visinf/INSID3/blob/main/models/insid3.py#L253)

## Problem

ViT patch features contain both semantic content and a positional bias introduced by
the learned positional embeddings. This bias causes patch tokens at the same spatial
location to be systematically similar across images, regardless of image content —
degrading nearest-neighbour matching, segmentation, and clustering quality.

## Algorithm (from INSID3)

1. **Build positional basis** — forward a zero-pixel (black) image through the encoder.
   The resulting patch features contain only the positional signal (no semantic content).
   Run SVD on the flattened feature matrix to extract the top-K principal directions =
   positional subspace `U (D, K)`.
2. **Debias** — project real features onto the orthogonal complement of that subspace:
   `P_perp = I_D - U @ U.T`, then `X_deb = P_perp @ X`. Re-normalize L2.
3. **Visualize** — PCA of the zero-image features to 3 components → RGB heatmap
   `(H_p, W_p, 3)` showing the spatial structure of the positional bias.

## Design

### Scope

All changes live in `collab_splats/semantics/features.py`. No other files change.

### Key principle

`forward()` in each subclass reflects how that model's forward pass works — its
signature, parameters, and behavior are untouched. Debiasing is a standalone
post-processing step exposed as a separate method on `BaseFeatureExtractor`.

### No template method — `forward` unchanged in all classes

`BaseFeatureExtractor.forward` stays `@abstractmethod` exactly as today. Each
subclass continues to define `forward` with whatever signature suits the model
(e.g. `MaskCLIPExtractor` keeps `forward(images, resolution=None)`). Python does not
enforce abstract-method signature matching, so subclasses are unconstrained.

### New API on `BaseFeatureExtractor`

```
debias(features)                   → list[Tensor(D,H_p,W_p)]  ← public: apply debiasing
get_bias_visualization(H_p, W_p)  → np.ndarray(H_p,W_p,3)    ← public: PCA heatmap
_build_positional_basis(H_p, W_p) → None                      ← private: SVD, fills caches
_apply_debias(fmap)                → Tensor(D,H_p,W_p)         ← private: P_perp projection
```

### New state on `BaseFeatureExtractor.__init__`

```python
svd_components: int = 500          # matches INSID3 default
_pos_basis_cache: dict             # keyed by (H_p, W_p) → Tensor(D, K)
_zero_feats_cache: dict            # keyed by (H_p, W_p) → Tensor(D, H_p, W_p)
```

Basis is **lazy** — built on first `debias()` call at a given `(H_p, W_p)`, then
cached for the lifetime of the extractor instance.

Concrete class `__init__` methods add `**kwargs` and pass them to `super().__init__()`
so callers can set `svd_components` without subclass boilerplate.

### Validation guard

A module-level set marks extractors where debiasing has been empirically validated:

```python
_DEBIAS_VALIDATED: frozenset = frozenset({"DINOFeatureExtractor", "Talk2DinoExtractor"})
```

`debias()` logs a `WARNING` (does not raise) when `type(self).__name__` is not in
that set. Other extractors still work — the warning signals "untested, proceed with care."

### `_build_positional_basis(H_p, W_p)`

Mirrors INSID3's `_build_positional_basis` exactly:

```python
# No silent fallback — if patch_size missing, raise immediately with a clear message
# rather than silently producing a wrong-sized zero image.
if not hasattr(self, "patch_size"):
    raise AttributeError(
        f"{type(self).__name__} must define `self.patch_size` "
        "(pixel stride per patch token) before calling debias()."
    )
patch_size = self.patch_size              # set in each concrete subclass __init__
H_img = H_p * patch_size                 # image height → H_p patch rows
W_img = W_p * patch_size                 # image width  → W_p patch cols

# Zero-pixel (black) image — matches INSID3's torch.zeros approach in PIL form.
# Black pixels, after each extractor's own normalization, produce a positional-only response.
zero_pil = Image.fromarray(np.zeros((H_img, W_img, 3), dtype=np.uint8))

with torch.no_grad():
    [zero_feat] = self.forward([zero_pil])  # (D, H_p', W_p') — uses subclass forward as-is

# Interpolate if internal preprocessing changed the patch grid — bias is spatially smooth.
if zero_feat.shape[1:] != (H_p, W_p):
    zero_feat = F.interpolate(
        zero_feat.unsqueeze(0), size=(H_p, W_p), mode="bilinear", align_corners=False
    ).squeeze(0)

self._zero_feats_cache[(H_p, W_p)] = zero_feat

D = zero_feat.shape[0]
E = zero_feat.reshape(D, -1)                # (D, H_p*W_p)
E = E - E.mean(dim=1, keepdim=True)         # center per feature channel
U, _, _ = torch.linalg.svd(E, full_matrices=False)
self._pos_basis_cache[(H_p, W_p)] = U[:, :self.svd_components].contiguous()  # (D, K)
```

Note: `_build_positional_basis` calls `self.forward([zero_pil])` — the subclass's own
`forward` — so preprocessing is handled identically to real inference. No extra code path.

### `debias(features)`

```python
def debias(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
    if type(self).__name__ not in _DEBIAS_VALIDATED:
        logger.warning(...)  # warn but do not raise
    _, H_p, W_p = features[0].shape
    if (H_p, W_p) not in self._pos_basis_cache:
        self._build_positional_basis(H_p, W_p)  # lazy build on first call
    return [self._apply_debias(f) for f in features]
```

### `_apply_debias(fmap)`

```python
D, H_p, W_p = fmap.shape
basis = self._pos_basis_cache[(H_p, W_p)].to(fmap.device)  # (D, K)
P_perp = torch.eye(D, device=fmap.device, dtype=fmap.dtype) - basis @ basis.T  # (D, D)
X = fmap.reshape(D, -1)                  # (D, H_p*W_p)
X_deb = P_perp @ X                       # (D, H_p*W_p) — positional component removed
X_deb = F.normalize(X_deb, p=2, dim=0)  # re-normalize for cosine similarity downstream
return X_deb.reshape(D, H_p, W_p)
```

### `get_bias_visualization(H_p, W_p)`

```python
# Requires prior debias() call to populate _zero_feats_cache
zero_feats = self._zero_feats_cache[(H_p, W_p)]  # (D, H_p, W_p)
E = zero_feats.reshape(D, -1).T                   # (N, D) where N = H_p*W_p
E = E - E.mean(dim=0, keepdim=True)               # center
_, _, Vt = torch.linalg.svd(E, full_matrices=False)
rgb = (E @ Vt[:3].T).cpu().numpy()               # (N, 3) — top-3 PCs
rgb -= rgb.min(); rgb /= rgb.max() + 1e-8
return (rgb.reshape(H_p, W_p, 3) * 255).astype(np.uint8)
```

### Concrete class changes

| Class | Change |
|---|---|
| `DINOFeatureExtractor` | Add `**kwargs` to `__init__`, pass to `super().__init__()` |
| `Talk2DinoExtractor` | Add `**kwargs` to `__init__`, pass to `super().__init__()` |
| `MaskCLIPExtractor` | Add `**kwargs` to `__init__`, pass to `super().__init__()` |
| `BaseQueryableExtractor` | No change |

**No `forward` method is renamed or modified in any class.**

## Data Flow

```
features = extractor.forward([pil_img])        ← unchanged, subclass-defined
debiased = extractor.debias(features)          ← new: remove positional bias
viz      = extractor.get_bias_visualization(H_p, W_p)  ← new: PCA heatmap
```

## Not In Scope

- Threading `debias` through `lift_features()` — follow-on task
- Disk-caching the positional basis across sessions
- Automatic selection of `svd_components` per extractor

## Verification

- [ ] All existing `forward()` call sites work unchanged (no `debias` param added)
- [ ] `extractor.debias(features)` returns same shape as input features
- [ ] `extractor.debias(features)` output differs from input (bias removed)
- [ ] `get_bias_visualization` returns `(H_p, W_p, 3)` uint8
- [ ] `MaskCLIPExtractor.debias(features)` logs warning, does not raise
- [ ] Basis cached — second `debias()` call at same resolution does not re-run SVD
