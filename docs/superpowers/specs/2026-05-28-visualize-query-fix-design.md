# Visualize Pane — Query Fix + UI Improvements

**Date:** 2026-05-28  
**Status:** Approved for implementation

## Problem

Three issues with the Similarity mode in `ScenePanel`:

### 1. matmul crash (768 ≠ 64)

The wrapper pipeline (`_lift_and_save` in `reconstructor.py`) compresses per-point features through a `FeatureAutoencoder` (768 → 64) and saves:
- `{backend}/semantics/{extractor}/features.zarr` — (P, 64) compressed features
- `{backend}/semantics/{extractor}/compressor.pt/autoencoder.pt` — AE weights

`do_query` loads the 64-dim features but calls `extractor.encode_text()` → 768-dim, then does `lifted @ query_vec` → crash. The AE encoder is never applied to the text query.

### 2. Stale features on extractor switch

`_extractor_dd` has no `param.watch`. When the user switches extractors, `_lifted_normed` and `_compressor` are not invalidated. The old extractor's 64-dim features remain while the new extractor's text encoder runs, causing a dim mismatch even if both have `compressor.pt`.

### 3. Extractor buried in Similarity-only row

The extractor dropdown only appears when Similarity mode is active. The user can't see or select the extractor until after switching modes. Should be always-visible next to Backend.

---

## Design

### Section 1 — Bug fix: apply AE encoder to text query

**In `_load_lifted_features_for_current_extractor`:**

After loading `features.zarr`, also attempt to load the sibling compressor:

```python
compressor_dir = feat_path.parent / "compressor.pt"
if compressor_dir.is_dir():
    from collab_splats.semantics.compression import FeatureAutoencoder
    self._compressor = FeatureAutoencoder.load(compressor_dir)
    self._compressor.eval()
else:
    self._compressor = None
```

**In `do_query`, after encoding text:**

```python
query_vec = extractor.encode_text([text])[0].detach().cpu().numpy()  # (768,)

if self._compressor is not None:
    import torch, torch.nn.functional as F
    q_t = torch.from_numpy(query_vec).unsqueeze(0)          # (1, 768)
    q_latent = self._compressor.per_point_encode(q_t)        # (1, 64)
    query_vec = F.normalize(q_latent, dim=-1).squeeze(0).numpy()  # (64,)

q_norm = np.linalg.norm(query_vec)
if q_norm > 1e-8:
    query_vec = query_vec / q_norm

sims = self._lifted_normed @ query_vec  # (P,64) @ (64,) — dims match
```

### Section 2 — Cache invalidation on extractor change

Wire in `__init__`:

```python
self._extractor_dd.param.watch(self._on_extractor_change, "value")
```

New handler:

```python
def _on_extractor_change(self, event: Any) -> None:
    self._lifted_normed = None
    self._compressor = None
    if self.mode == "Similarity":
        threading.Thread(
            target=self._load_lifted_features_for_current_extractor, daemon=True
        ).start()
```

Also reset in `_on_backend_change` and `_on_dataset_change`:

```python
self._lifted_normed = None
self._compressor = None
```

### Section 3 — Extractor DD always visible at top

Move `_extractor_dd` out of `_sim_query_row` and into a permanent second row, always visible below `controls_row`. Populate it on `_on_backend_change` / `_on_dataset_change` rather than only in `_on_mode_change`. Disable when no queryable extractors found.

`_scan_extractors` already returns folder names from `{backend}/semantics/`. Filter to names registered in `BaseQueryableExtractor._registry`:

```python
def _scan_extractors(dataset_dir: Path, backend: str) -> list[str]:
    from collab_splats.semantics.features.base import BaseQueryableExtractor
    semantics_dir = dataset_dir / backend / "semantics"
    if not semantics_dir.is_dir():
        return []
    return sorted(
        p.name for p in semantics_dir.iterdir()
        if p.is_dir()
        and (p / "features.zarr").exists()
        and p.name in BaseQueryableExtractor._registry
    )
```

Updated layout (`ScenePanel.panel()`):

```python
controls_row    = Row(dataset_dd, backend_dd, load_btn, align="end")
extractor_row   = Row(extractor_dd)                    # always visible
mode_selector   = RadioButtonGroup(...)
vtk_pane        = VTK(...)
points_opts_row = Row(frustum_check, point_size_slider, visible=False)
sim_query_row   = Row(pos_input, neg_input, query_btn, visible=False)
```

`_extractor_dd` is still referenced in `_on_mode_change("Similarity")` to set options/value, but options are populated earlier (on backend change).

### Section 4 — Positive + negative queries

Replace `_sim_query_input` with two labeled inputs:

```python
self._sim_pos_input = pn.widgets.TextInput(
    placeholder="Enter positive query…", width=200
)
self._sim_neg_input = pn.widgets.TextInput(
    placeholder="background, wall, floor…",
    width=200,
    styles={"color": "#888"},
)
```

`_sim_query_row`:

```python
self._sim_query_row = pn.Row(
    pn.pane.HTML("<b style='color:#6f6'>+</b>"),
    self._sim_pos_input,
    pn.pane.HTML("<b style='color:#f66'>−</b>"),
    self._sim_neg_input,
    self._sim_query_btn,
    visible=False,
)
```

Updated score in `do_query` (after AE encode + normalize of pos_vec):

```python
pos_sims = self._lifted_normed @ pos_vec

neg_text = self._sim_neg_input.value.strip()
if neg_text:
    neg_tensor = extractor.encode_text([neg_text])[0].detach().cpu().numpy()
    if self._compressor is not None:
        neg_t = torch.from_numpy(neg_tensor).unsqueeze(0)
        neg_tensor = F.normalize(
            self._compressor.per_point_encode(neg_t), dim=-1
        ).squeeze(0).numpy()
    neg_norm = np.linalg.norm(neg_tensor)
    if neg_norm > 1e-8:
        neg_tensor = neg_tensor / neg_norm
    sims = pos_sims - self._lifted_normed @ neg_tensor
else:
    sims = pos_sims
```

`_on_sim_query_click` reads `self._sim_pos_input.value.strip()` (was `_sim_query_input`).

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/dashboard/panes/visualize.py` | All changes above — no other files |

## Out of Scope

- Saving `extractor_name` / `n_components` into `feedforward.zarr` attrs (separate cleanup)
- Reading features from `feedforward.zarr` directly (current `semantics/` path is correct — written by wrapper)
- Weighted negative query slider (straight subtraction sufficient for now)
- Pre-loading features in background on extractor select before Similarity mode activates
