# Semantics Module + Dashboard Design

**Date:** 2026-04-16  
**Branch:** `tlb-semantics-refactor` (from `tlb-improve-mesh`)

---

## Problem

Feature extraction (CLIP) and segmentation (SAM) live as ad-hoc utilities in `collab_splats/utils/`. There is no way to test or explore semantic queries interactively before committing to a full splat training run. Talk2DINO (DINO + DINOv3 text-conditioned segmentation) needs to be integrated as a first-class extractor.

---

## Goals

1. Promote `utils/features.py` and `utils/segmentation.py` into a proper `semantics` submodule with clean structure
2. Integrate Talk2DINO (loaded from HuggingFace Hub) as a new registered extractor
3. Build an interactive Gradio dashboard for: load video frames → extract features → segment → semantic query
4. Zero breaking changes to existing pipeline callers

---

## Architecture

### Module structure

```
collab_splats/
├── semantics/
│   ├── __init__.py       # public re-exports
│   ├── features.py       # all extractors: BaseFeatureExtractor, MaskCLIPExtractor,
│   │                     # DINOFeatureExtractor, Talk2DinoExtractor + helper fns
│   ├── segmentation.py   # Segmentation class + all helpers (device bug fixed)
│   └── protocols.py      # SupportsTextQuery Protocol
├── utils/
│   ├── features.py       # shim → re-exports from semantics.features; TwoLayerMLP stays here
│   └── segmentation.py   # shim → re-exports from semantics.segmentation
└── dashboard/
    ├── __init__.py
    ├── __main__.py       # argparse router: `python -m collab_splats.dashboard semantics`
    └── semantics.py      # Gradio app for semantic exploration (4 tabs)
```

**Why flat `semantics/` (not nested `features/` + `segmentation/` subdirs):** 3 extractors + 1 segmenter does not justify 2-level nesting. Flat is easier to navigate and import. Promote to subdirs if/when extractors proliferate past ~8.

---

## Feature Extractors (`semantics/features.py`)

The existing `BaseFeatureExtractor` registry pattern (`@register(name)` / `.get(name)`) is preserved exactly. All three extractors live in one file.

### Existing extractors (moved, no interface change)

- `MaskCLIPExtractor` — uses `maskclip_onnx`; registered as `"samclip"` and `"clip-vit"`
- `DINOFeatureExtractor` — loads DINOv2 via `torch.hub`; registered as `"dinov2"`

### New: `Talk2DinoExtractor`

Loaded from HuggingFace Hub — no local Talk2DINO repo required at runtime.

```python
@BaseFeatureExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        hf_model_id: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: str = "cpu",
    ):
        # AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)
        ...

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Center-crop to square (required by Talk2DINO)"""

    def forward(self, image: Image.Image) -> torch.Tensor:
        """encode_image([img])[0] → patch tokens (N_patches, D)"""

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """encode_text(texts) → (N, D) normalized embeddings"""

    def compute_semantic_heatmap(
        self,
        image: Image.Image,
        text_pairs: Dict[str, Tuple[List[str], List[str]]],
        softmax_temp: float = 0.05,
        method: str = "standard",  # "standard" | "pairwise"
    ) -> Dict[str, np.ndarray]:
        """
        Port of segment_image() + compute_similarity() from hf_demo.ipynb.
        text_pairs: {"label": (positive_queries, negative_queries)}
        Returns: {"label": HxW float array} — per-label similarity heatmaps
        """
```

Available HF models:
- `lorebianchi98/Talk2DINOv3-ViTB` — DINOv3 (default, cleaner interface)
- `lorebianchi98/Talk2DINO-ViTB` — DINOv2 (older)

---

## Capabilities (`semantics/protocols.py`)

`Protocol` is standard Python (PEP 544, `typing` module, Python 3.8+). It defines **structural duck typing**: any class that implements the required methods satisfies the protocol without inheriting from it. Python's stdlib uses this pattern for `SupportsInt`, `SupportsFloat`, etc. `@runtime_checkable` enables `isinstance()` checks at runtime.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SupportsTextQuery(Protocol):
    def encode_text(self, texts: List[str]) -> torch.Tensor: ...
    def compute_semantic_heatmap(
        self,
        image: Image.Image,
        text_pairs: Dict[str, Tuple[List[str], List[str]]],
        softmax_temp: float,
        method: str,
    ) -> Dict[str, np.ndarray]: ...
```

`Talk2DinoExtractor` satisfies this protocol (no inheritance needed). Dashboard uses `isinstance(extractor, SupportsTextQuery)` to show/hide the semantic query tab — no hardcoded extractor-type checks, new extractors declare capability implicitly by implementing the methods.

---

## Segmentation (`semantics/segmentation.py`)

Verbatim move of `utils/segmentation.py` with one bug fix:

**Line 164 device bug:**
```python
# Before (broken on CPU/non-CUDA machines):
transformed_boxes = torch.from_numpy(transformed_boxes).to("cuda")

# After:
transformed_boxes = torch.from_numpy(transformed_boxes).to(
    next(iter(mobile_sam.parameters())).device
)
```

---

## Backward Compatibility

`utils/features.py` and `utils/segmentation.py` become thin shims:

```python
# utils/features.py
"""Backward-compat shim. Code lives in collab_splats.semantics.features."""
from collab_splats.semantics.features import (
    BaseFeatureExtractor, MaskCLIPExtractor, DINOFeatureExtractor,
    load_hf_weights, load_torchhub_model, pytorch_gc,
    resize_image, interpolate_to_patch_size, batch_iterator,
)
# TwoLayerMLP stays here — it's a splatting layer, not a feature extractor
class TwoLayerMLP(nn.Module): ...
```

All existing callers (`feedforward.py`, `features_datamanager.py`, `rade_features_model.py`, `grouping.py`) require zero changes.

---

## Dashboard

**Launch:**
```bash
python -m collab_splats.dashboard semantics   # module invocation
collab-dashboard semantics                     # entry point alias
```

`dashboard/__main__.py` routes subcommands:
```python
DASHBOARDS = {
    "semantics": run_semantics_app,
    # "mesh": run_mesh_app,   # future
}
def main():
    parser = argparse.ArgumentParser(prog="collab-dashboard")
    parser.add_argument("mode", choices=DASHBOARDS.keys())
    DASHBOARDS[parser.parse_args().mode]()
```

Adding a new dashboard = one dict entry + one module. No changes to routing logic.

**`dashboard/semantics.py` — 4-tab Gradio app:**

**Tab 1 — Load Video**
- `gr.Video` → extract frames via `cv2.VideoCapture` at configurable FPS
- `gr.Gallery` preview + `gr.Slider` to select current frame
- Frames stored in `gr.State`

**Tab 2 — Feature Extraction**
- Extractor dropdown: populated from `BaseFeatureExtractor._registry.keys()`
- Extract → PCA-to-RGB (3-component) → overlay on frame
- Lazy model init inside callback to avoid CUDA OOM on startup

**Tab 3 — Segmentation**
- Backend dropdown: `mobilesamv2`
- Segment → colored composite mask overlay
- Calls `Segmentation(...).segment(frame)`

**Tab 4 — Semantic Query**
- Visible only when selected extractor satisfies `SupportsTextQuery`
- UI mirrors `hf_demo.ipynb` `segment_image()` API:
  - Per-label: positive query text, negative query text
  - Method: `standard` | `pairwise`
  - Softmax temperature slider (0.001–0.1)
- Output: per-label masked image overlays (one image per label)

All callbacks wrapped in `try/except` with `gr.Warning` for CPU-only graceful degradation.

---

## pyproject.toml Changes

**Entry point:**
```toml
[project.scripts]
collab-dashboard = "collab_splats.dashboard.__main__:main"
```

**Missing dependencies for Talk2DINO (add to `dependencies`):**
```toml
"transformers>=4.30,<5.0",   # AutoModel.from_pretrained; <5.0 avoids numpy-2 compat risk
"timm>=0.9,<2.0",            # DINOv3 backbone loaded by HF trust_remote_code model
```

`gradio` already present. `opencv-python` needed for video frame extraction — add if not transitively available:
```toml
"opencv-python",
```

**Compatibility audit:**
| Constraint | Risk with new deps |
|---|---|
| `numpy<2.0.0` | None — `transformers<5.0` and `timm<2.0` both numpy-1.x compatible |
| `scipy==1.11.4` | None — no scipy dependency in transformers/timm |
| `meshlib==3.0.9.196` | None — unrelated |
| `pycolmap==3.10` | None — unrelated |
| CLIP (openai) | Likely transitive via `maskclip_onnx`; verify or add `open_clip_torch` |

**`trust_remote_code=True` risk:** HF model code runs locally and could import unlisted packages. Do a dry-run in clean venv before merging: `pip install -e . && python -c "from transformers import AutoModel; AutoModel.from_pretrained('lorebianchi98/Talk2DINOv3-ViTB', trust_remote_code=True)"`

---

## Tests

**`tests/test_semantics_extractors.py`** (CPU-only, mock heavy models):
- Registry round-trip: `"samclip"`, `"dinov2"`, `"talk2dino"` all registered
- `"clip-vit"` backward-compat alias preserved
- Unknown extractor raises `ValueError`
- `resize_image` longest-edge scaling
- `batch_iterator` correct batch sizes
- `Talk2DinoExtractor` registers on init (mocked `AutoModel.from_pretrained`)

**`tests/test_semantics_segmentation.py`**:
- `create_patch_mask` covers all pixels
- `mask_id_to_binary_mask` shape + pixel counts
- Backward-compat: `from collab_splats.utils.segmentation import Segmentation` works
- Backward-compat: `from collab_splats.utils.features import BaseFeatureExtractor` works

---

## Migration Sequence

1. `git checkout -b tlb-semantics-refactor` from `tlb-improve-mesh`
2. Create `semantics/__init__.py`, `dashboard/__init__.py`
3. Write `semantics/features.py` (move + add Talk2DinoExtractor)
4. Write `semantics/segmentation.py` (move + fix device bug)
5. Write `semantics/protocols.py`
6. Wire `semantics/__init__.py`
7. Replace `utils/features.py` body with shim (keep TwoLayerMLP)
8. Replace `utils/segmentation.py` body with shim
9. `make test` — all existing tests pass, zero caller changes
10. Write `dashboard/__main__.py` (router) + `dashboard/semantics.py` (Gradio app)
11. Add `collab-dashboard` entry point + missing deps to `pyproject.toml`
12. Write new tests
13. `make test && mypy collab_splats && ruff check collab_splats`

---

## Verification

```bash
# After shims (step 9): existing tests pass
make test

# Smoke test registry
python -c "
from collab_splats.semantics.features import BaseFeatureExtractor, Talk2DinoExtractor
from collab_splats.semantics.protocols import SupportsTextQuery
assert 'talk2dino' in BaseFeatureExtractor._registry
assert isinstance(Talk2DinoExtractor(device='cpu'), SupportsTextQuery)
print('OK')
"

# Launch dashboard
python -m collab_splats.dashboard semantics
# or: collab-dashboard semantics

# Full suite
make test && mypy collab_splats && ruff check collab_splats
```
