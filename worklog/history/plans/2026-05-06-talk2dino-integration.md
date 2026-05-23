# Talk2DINO Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs blocking Talk2DINO in the semantics notebook, widen `main_features` to accept `"talk2dino"` for training, and add a side-by-side comparison section to the notebook.

**Architecture:** Three focused code changes (env dep, one-line default fix, one-line Literal widen) plus notebook updates. No new files. No new classes. Each task is independently committable.

**Tech Stack:** Python, PyTorch, transformers (HuggingFace), nerfstudio dataclass config, Jupyter notebook (JSON).

**Spec:** `worklog/history/specs/2026-05-06-talk2dino-integration-design.md`

---

## File Map

| File | Change |
|------|--------|
| `setup.sh` | Add `clip` install line |
| `collab_splats/semantics/features.py` | Change `score_queries` default `negative` |
| `collab_splats/nerfstudio/datamanagers/features.py` | Widen `main_features` Literal |
| `tests/semantics/test_query_api.py` | Add bounds test for no-negative call |
| `tests/nerfstudio/test_datamanager_config.py` | New: config acceptance test |
| `docs/semantics/feature_extraction.ipynb` | Remove redundant preprocess, add comparison cells |

---

## Task 1: Install `clip` and update `setup.sh`

**Files:**
- Modify: `setup.sh`

Talk2DINOv3's HuggingFace remote code imports OpenAI `clip` at model load time. Not on PyPI as `clip` — must install from GitHub.

- [ ] **Step 1: Install in nerfstudio env**

```bash
/opt/conda/envs/nerfstudio/bin/pip install git+https://github.com/openai/CLIP.git
```

- [ ] **Step 2: Verify import works**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import clip; print('clip OK:', clip.__version__)"
```

Expected output: `clip OK: 1.0` (or similar — no ImportError).

- [ ] **Step 3: Add to `setup.sh`**

`setup.sh` currently contains:
```bash
#!/bin/bash

# Install dependencies
pip install -e .

# Install collab-data
pip install git+https://github.com/BasisResearch/collab-data.git
```

Add the clip install after the collab-data line:
```bash
#!/bin/bash

# Install dependencies
pip install -e .

# Install collab-data
pip install git+https://github.com/BasisResearch/collab-data.git

# OpenAI CLIP — required by Talk2DINOv3 remote code
pip install git+https://github.com/openai/CLIP.git
```

- [ ] **Step 4: Commit**

```bash
git add setup.sh
git commit -m "fix(deps): add OpenAI clip install required by Talk2DINOv3 remote code"
```

---

## Task 2: Fix `score_queries` default negative

**Files:**
- Modify: `collab_splats/semantics/features.py:131-151`
- Modify: `tests/semantics/test_query_api.py`

**Root cause:** When `negative=None`, `compute_semantic_contrast` falls back to raw dot-product similarities (range ~0.1–0.3). Visualization thresholds expect softmax-normalized scores in [0,1]. Result: all-black heatmaps.

**Fix:** Default `negative` to `["background"]`. This forces the softmax contrastive path always.

Note on mutable default: `score_queries` never mutates `negative` (only concatenates it), so `["background"]` as default is safe in practice. Use tuple `("background",)` if you want to be pedantic — the concat line `positive + (negative or [])` handles both.

- [ ] **Step 1: Add failing test to `tests/semantics/test_query_api.py`**

Open `tests/semantics/test_query_api.py`. Add this test after `test_score_queries_shape_no_negative` (around line 78):

```python
def test_score_queries_no_negative_returns_bounded_scores():
    """score_queries with no explicit negative must return softmax scores in [0, 1].

    The default negative ("background") forces the contrastive softmax path.
    Raw dot-product fallback would return values well outside [0, 1].
    """
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"])
    assert result.min() >= 0.0, "score below 0 — contrastive softmax path not taken"
    assert result.max() <= 1.0 + 1e-6, "score above 1 — contrastive softmax path not taken"
```

- [ ] **Step 2: Run test — verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_query_api.py::test_score_queries_no_negative_returns_bounded_scores -v
```

Expected: FAIL — raw similarities exceed [0,1] or fall below 0.

- [ ] **Step 3: Fix `score_queries` in `collab_splats/semantics/features.py`**

Find `score_queries` at line ~131. Change the `negative` parameter:

```python
# Before (line ~135):
negative: Optional[List[str]] = None,

# After:
negative: List[str] = ["background"],
```

Leave all other lines in `score_queries` unchanged — `negative or []` still works correctly with a list default (empty list is falsy, `["background"]` is truthy).

Also update the docstring for `negative` to document the default:

Find the docstring block inside `score_queries` (or the class-level docstring for `BaseQueryableExtractor`). Add/update the `negative` parameter description to say:

```
negative: Text queries that should score low. Defaults to ["background"] —
    ensures contrastive softmax is always used. Pass [] to skip contrast
    and return raw cosine similarities directly (not recommended for visualization).
```

- [ ] **Step 4: Run test — verify it passes**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_query_api.py -v
```

Expected: all tests pass (including the new one and the existing `test_score_queries_shape_no_negative`).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_query_api.py
git commit -m "fix(semantics): default score_queries negative to [\"background\"] to prevent black heatmaps"
```

---

## Task 3: Widen `main_features` Literal to include `"talk2dino"`

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py:41`
- Create: `tests/nerfstudio/test_datamanager_config.py`

- [ ] **Step 1: Create `tests/nerfstudio/__init__.py`**

`tests/nerfstudio/` exists but has no `__init__.py`. Create an empty one so pytest discovers the new test file:

```bash
touch tests/nerfstudio/__init__.py
```

- [ ] **Step 2: Write failing test**

Create or add to `tests/nerfstudio/test_datamanager_config.py`:

```python
"""Tests for FeatureSplattingDataManagerConfig."""
import pytest


def test_main_features_accepts_maskclip():
    """Default value works."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig()
    assert config.main_features == "maskclip"


def test_main_features_accepts_talk2dino():
    """talk2dino is a valid main_features value."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig(main_features="talk2dino")
    assert config.main_features == "talk2dino"


def test_regularization_features_can_be_none():
    """regularization_features=None is valid (used with talk2dino, no DINOv2 needed)."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig(
        main_features="talk2dino",
        regularization_features=None,
    )
    assert config.regularization_features is None
```

- [ ] **Step 3: Run tests — verify `test_main_features_accepts_talk2dino` fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py -v
```

Expected: `test_main_features_accepts_talk2dino` FAIL with a validation error; other two may pass.

- [ ] **Step 4: Widen the Literal in `collab_splats/nerfstudio/datamanagers/features.py`**

Find line ~41:
```python
# Before:
main_features: Literal["maskclip"] = "maskclip"
"""Type of features to extract - MaskCLIP or CLIP."""

# After:
main_features: Literal["maskclip", "talk2dino"] = "maskclip"
"""Feature extractor for main training features.

"maskclip": patch CLIP features — use with regularization_features="dinov2".
"talk2dino": DINOv3 + CLIP projection — structural grounding is built in;
    set regularization_features=None. Note: images are center-cropped to
    square during extraction (feature spatial coverage ≠ full frame).
"""
```

- [ ] **Step 5: Run tests — verify all three pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py -v
```

Expected: 3 pass.

- [ ] **Step 6: Run full test suite — no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/nerfstudio/test_imports.py -x 2>&1 | tail -20
```

(Skip `test_imports.py` — pre-existing nerfstudio env failures unrelated to this change.)

- [ ] **Step 7: Commit**

```bash
git add collab_splats/nerfstudio/datamanagers/features.py tests/nerfstudio/test_datamanager_config.py
git commit -m "feat(datamanager): support talk2dino as main_features extractor"
```

---

## Task 4: Fix notebook + add comparison section

**Files:**
- Modify: `docs/semantics/feature_extraction.ipynb`

Two changes:
1. **Cell `cell-9`:** Remove the redundant explicit `extractor_t2d.preprocess(pil_frame)` call. Keep `sq_frame` derivation in cell-10 (needed for visualization alignment).
2. **Add comparison section** after cell-10: side-by-side MaskCLIP vs Talk2DINO.

- [ ] **Step 1: Remove redundant preprocess call from cell `cell-9`**

Cell `cell-9` currently:
```python
# preprocess() center-crops to square — forward() re-crops internally.
# We call it explicitly here only to derive sq_frame dimensions for visualization.
preprocessed = extractor_t2d.preprocess(pil_frame)
features_t2d = extractor_t2d.forward([pil_frame])  # pass original — forward re-crops
print("Feature shape:", features_t2d[0].shape)
```

Replace with:
```python
features_t2d = extractor_t2d.forward([pil_frame])  # forward() center-crops internally
print("Feature shape:", features_t2d[0].shape)
```

Use the NotebookEdit tool (or direct JSON edit) to replace the source of cell `cell-9`.

- [ ] **Step 2: Add comparison section after cell `cell-10`**

Add two new cells after `cell-10`:

**New markdown cell:**
```markdown
## MaskCLIP vs Talk2DINO Comparison

Side-by-side feature comparison on the same image.
MaskCLIP uses the full frame; Talk2DINO uses the center-cropped square (center-crop is required by the model).
```

**New code cell:**
```python
# Re-score with the same query for fair comparison
QUERY_POSITIVE = ["bird", "animal"]
QUERY_NEGATIVE = ["background", "sky", "ground"]

sim_clip = extractor_clip.score_queries(
    features_clip[0], positive=QUERY_POSITIVE, negative=QUERY_NEGATIVE
)
sim_t2d_cmp = extractor_t2d.score_queries(
    features_t2d[0], positive=QUERY_POSITIVE, negative=QUERY_NEGATIVE
)
sq_frame = np.array(extractor_t2d.preprocess(pil_frame))  # cropped frame for t2d viz

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].imshow(pca_to_rgb(features_clip[0], frame))
axes[0, 0].set_title("MaskCLIP: PCA → RGB")
axes[0, 1].imshow(compute_heatmap(frame, sim_clip))
axes[0, 1].set_title(f"MaskCLIP: {QUERY_POSITIVE}")
axes[0, 2].imshow(compute_masked_image(frame, sim_clip))
axes[0, 2].set_title("MaskCLIP: Masked")

axes[1, 0].imshow(pca_to_rgb(features_t2d[0], sq_frame))
axes[1, 0].set_title("Talk2DINO: PCA → RGB (center crop)")
axes[1, 1].imshow(compute_heatmap(sq_frame, sim_t2d_cmp))
axes[1, 1].set_title(f"Talk2DINO: {QUERY_POSITIVE}")
axes[1, 2].imshow(compute_masked_image(sq_frame, sim_t2d_cmp))
axes[1, 2].set_title("Talk2DINO: Masked (center crop)")

for ax in axes.flat:
    ax.axis("off")
plt.suptitle("MaskCLIP vs Talk2DINO feature comparison", fontsize=14)
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Run all notebook cells top-to-bottom — verify no errors**

In the nerfstudio kernel, run all cells in order. Expected:
- Cell 1: prints available datasets, shows first frame
- Cells 2–3: MaskCLIP loads, extracts `[768, 73, 41]` features, shows 3-panel plot
- Cells 4–5 (Talk2DINO): loads without `ImportError`, extracts features, shows 3-panel plot
- Comparison cell: shows 2×3 grid with both extractors

- [ ] **Step 4: Commit**

```bash
git add docs/semantics/feature_extraction.ipynb
git commit -m "fix(notebook): remove redundant preprocess call, add MaskCLIP vs Talk2DINO comparison"
```

---

## Verification Checklist

After all tasks:

- [ ] `Talk2DinoExtractor(model_name="lorebianchi98/Talk2DINOv3-ViTB")` loads without error
- [ ] `extractor.score_queries(features, positive=["bird"])` returns heatmap in [0,1], not all-black
- [ ] `FeatureSplattingDataManagerConfig(main_features="talk2dino")` instantiates without error
- [ ] All cells in `feature_extraction.ipynb` run top-to-bottom without error
- [ ] Training smoke test (optional): `ns-train rade-features --pipeline.datamanager.main_features talk2dino --pipeline.datamanager.regularization_features None` starts feature extraction

## Training Usage Reference

```bash
# Existing: maskclip + dinov2 regularization (unchanged default)
ns-train rade-features --pipeline.datamanager.main_features maskclip

# New: talk2dino, no regularization needed
ns-train rade-features \
  --pipeline.datamanager.main_features talk2dino \
  --pipeline.datamanager.regularization_features None
```
