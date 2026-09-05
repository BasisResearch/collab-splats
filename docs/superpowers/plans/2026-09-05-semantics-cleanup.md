# Semantics Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold every semantic-artifact I/O helper into one flat `collab_splats/semantics/utils.py`, trim `FeatureAutoencoder` to the surface actually used, hoist the three extractors' duplicated `preprocess()` into `BaseFeatureExtractor`, and delete the dead code and back-compat shim — cutting the semantics package from 2429 to ~1750 lines with no behaviour change.

**Architecture:** `utils.py` becomes the single module that knows the on-disk layout of a scene's `semantics/` dir (`<extractor>.zarr` 2D patch cache, `<extractor>_lifted.zarr` per-point codes, `<extractor>_ae.pt` weights). Import graph stays acyclic by construction: `compression.py` imports torch only; `utils.py` imports `compression` + `preproc.frame_store` + zarr; `features/base.py` imports `utils`. `extract_feature_cache()` takes the extractor as an *argument*, so `utils` never imports `features`. Every external call site (`wrapper/reconstructor.py`, `dashboard/pipeline.py`, `dashboard/viewer.py`, `dashboard/app.py`) imports these helpers from `semantics.utils` and from nowhere else.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), PyTorch, zarr v3 (`compressors=[BloscCodec(...)]`), numpy, pytest, black/isort, pyflakes.

**Source spec:** [docs/superpowers/specs/2026-09-05-semantics-cleanup-design.md](../specs/2026-09-05-semantics-cleanup-design.md)

---

## Ground Rules — read before Task 1

These apply to **every** step in this plan. They are not optional.

### The test command

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/test_semantics_logging.py tests/test_cu121_migration.py -q
```

Never `python` from the base shell — that is py3.13 and wrong for this project.
Baseline before Task 1: the suite is green. Every task ends green.

### The dashboard smoke gate

Any commit touching `collab_splats/dashboard/**` must first pass:

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the command prints `SMOKE PASS` and exits 0. Anything else blocks the commit.
Tasks 1 and 2 touch the dashboard. Tasks 3, 4 and 5 do not, but the final verification runs it anyway.

### Commits: `--only`, never `-A`

The branch carries **foreign uncommitted work from other sessions**. `git add -A` would sweep it into our commit.

```bash
# CORRECT
git commit --only path/one.py path/two.py -m "refactor(semantics): ..."

# NEVER
git add -A && git commit -m "..."
```

Three files this plan edits already have foreign uncommitted hunks:
`CLAUDE.md`, `tests/wrapper/test_reconstructor.py`, `collab_splats/semantics/segmentation/sam3.py`.
Before editing any of those three, run `git status --short <file>` and `git diff <file>` first, so you know
what was there before you touched it. `git commit --only <file>` commits the *whole* working-tree file —
if the foreign hunk in that file is unrelated whitespace (as it is in `sam3.py` today) that is acceptable
and should be noted in the commit body. If a foreign hunk is substantive, stage only our own hunks with
`git apply --cached` and commit the index instead.

### Formatting

Do NOT run repo-wide `black .` — the venv's black is 26.5.1, newer than what the repo was
formatted with, and it will reformat thousands of unrelated lines. Format only the files
you touched:

```bash
/opt/venv/reconstruction/bin/python -m black <files you edited>
/opt/venv/reconstruction/bin/python -m isort <files you edited>
```

`isort` carries `profile="black"`, which wraps at **88 columns**, not 120. Write long import
lists parenthesized so isort does not rewrite them.

### Docstring convention (used throughout, formalised in Task 5)

- `"""` opens and closes on its own line. Summary starts on the line *after* the opening quotes.
- One-line summary. Blank line. Then `Args:` / `Returns:` bullets, with tensor shapes. Nothing else.
- Rationale belongs in one-line block comments at the code it explains, at most two lines per block.

---

## File Structure

### Created

| path | responsibility |
|---|---|
| `tests/utils/test_torch_utils.py` | The six torch-utils tests, moved out of `tests/semantics/` in Task 4 — they test `collab_splats.utils.torch_utils`, so they belong under `tests/utils/`. |

### Rewritten (substantially)

| path | after |
|---|---|
| `collab_splats/semantics/utils.py` | 126 → ~250 lines. The one module that knows the semantics on-disk layout: contrastive scoring, `tokens_to_feature_map`, path helpers, `extract_feature_cache`, `load_feature_maps`, `write_point_features`, `point_features_cached`, `load_point_features`. |
| `collab_splats/semantics/compression.py` | 391 → ~260 lines. `FeatureAutoencoder` only. Path helpers move out (Task 2), reg head / `lr_scheduler` / `hidden_dim` ctor arg deleted, `save`/`load` become path-based, `fit()` returns `None` (see the deviation blockquote in Task 2 — the metrics dict goes). |
| `collab_splats/semantics/features/base.py` | 503 → ~330 lines. Gains the hoisted `__init__(resize_mode, image_resolution, svd_components)` + `preprocess()`. Loses `extract_and_cache`, `extract_and_cache_from_zarr`, `_FALLBACK_MEM_GB`, the duplicate `forward` abstractmethod on `BaseQueryableExtractor`, and eight unused imports. |
| `docs/semantics.md` | 224 lines of stale API (`samclip`, `compute_semantic_heatmap`, `SupportsTextQuery`, a Gradio app that does not exist) → rewritten against the real surface (Task 5). |

### Modified (targeted)

| path | change |
|---|---|
| `collab_splats/semantics/features/dino.py` | drop `preprocess`, drop `**kwargs`, constants come from `utils.image`, `__init__` delegates to base with `image_resolution=800`. |
| `collab_splats/semantics/features/maskclip.py` | same, `image_resolution=1024`; keeps its lazy `import maskclip_onnx`. |
| `collab_splats/semantics/features/talk2dino.py` | same, `image_resolution=512`; triple try/except for `patch_size` collapses to one line. |
| `collab_splats/utils/image.py` | gains `IMAGENET_MEAN/STD` and `CLIP_MEAN/STD`. |
| `collab_splats/localization/retrieval.py` | uses `IMAGENET_MEAN`/`IMAGENET_STD` instead of inline literals. |
| `collab_splats/semantics/segmentation/sam3.py` | drops `device` ctor arg and `confidence_threshold` from `segment_with_text`; new gating `ImportError` text. |
| `collab_splats/semantics/segmentation/base.py` | `segment_with_text` drops `confidence_threshold`. |
| `collab_splats/semantics/segmentation/insid3.py` | two inline imports hoisted to the top; path comment removed. |
| `collab_splats/semantics/segmentation/mobile_sam.py` | imports `batch_iterator`/`load_torchhub_model` from `collab_splats.utils.torch_utils`. |
| `collab_splats/semantics/__init__.py`, `features/__init__.py` | re-export lists rebuilt. |
| `collab_splats/dashboard/pipeline.py` | −130 lines; imports the artifact helpers from `semantics.utils`. |
| `collab_splats/dashboard/viewer.py`, `dashboard/app.py` | import paths retargeted at `semantics.utils`. |
| `collab_splats/wrapper/reconstructor.py` | `_extract_2d_features` → `extract_feature_cache`; `_lift_and_save` → `load_feature_maps(zarr_path)`. |
| `docs/source/tutorials/05_lifting/semantic_lifting.ipynb` | AE paths become real file paths (currently broken — see Task 2 Step 12). |
| `docs/source/tutorials/06_mesh/splats_mesh.ipynb` | same. |
| `CLAUDE.md`, `docs/superpowers/CHANGELOG.md` | Task 5. |

### Deleted

| path | why |
|---|---|
| `tests/dashboard/test_semantics_store_selection.py` | its three tests are relocated: two to `tests/semantics/test_semantics_utils.py` (the functions moved to `semantics.utils`), one to `tests/dashboard/test_viewer_lift.py`. |

---

## Task 1: Fold artifact I/O into `semantics.utils`

Everything that reads or writes a file inside a scene's `semantics/` dir moves into `utils.py`.
Two functions change shape while moving:

- `BaseFeatureExtractor.extract_and_cache_from_zarr(self, ...)` → `extract_feature_cache(extractor, frames_zarr, cache_dir) -> Path` (a plain function taking the extractor; the dead `batch_size` and `skip_existing` args and the `created_at`/`feature_dim` attrs go away).
- `load_feature_maps(semantics_dir)` → `load_feature_maps(store_path)` — it takes the **store path** now, not the directory. Callers wrap with `cache_store_path(...)`.

`lifted_store_path`, `ae_path` and `find_lifted_extractor` stay physically
in `compression.py` for this task and are re-exported through `utils.py`. `compression.FeatureAutoencoder.save/load`
still call `ae_path` at this point, so moving them now would create a cycle. Task 2 makes `save`/`load`
path-based and *then* moves them. Every external call site imports from `semantics.utils` from this task onward,
so Task 2's physical move touches no call site.

**Files:**
- Modify: `collab_splats/semantics/utils.py` (full rewrite)
- Modify: `collab_splats/semantics/compression.py` (delete `write_point_features`)
- Modify: `collab_splats/semantics/features/base.py:85-155`, `:177-251` (delete both cache methods)
- Modify: `collab_splats/semantics/__init__.py`
- Modify: `collab_splats/dashboard/pipeline.py`
- Modify: `collab_splats/dashboard/viewer.py`
- Modify: `collab_splats/dashboard/app.py`
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/semantics/test_semantics_utils.py`, `tests/semantics/test_artifact_layout.py`,
  `tests/semantics/features/test_extract_from_zarr.py`, `tests/dashboard/test_pipeline.py`,
  `tests/dashboard/test_viewer_lift.py`, `tests/dashboard/test_viewer.py`, `tests/dashboard/test_app.py`,
  `tests/wrapper/test_reconstructor.py`
- Delete: `tests/dashboard/test_semantics_store_selection.py`

---

- [ ] **Step 1: Record the green baseline**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/test_semantics_logging.py tests/test_cu121_migration.py -q 2>&1 | tail -5
```

Expected: a passing summary line, e.g. `271 passed, 3 skipped in 84.12s`. Write the exact count down —
every later task must land on the same or a deliberately-explained different number.

- [ ] **Step 2: Write the failing store-selection tests in their new home**

Append to `tests/semantics/test_semantics_utils.py`. These are the two tests being relocated out of
`tests/dashboard/test_semantics_store_selection.py`, retargeted at `semantics.utils`, plus one new
test that pins `load_feature_maps`'s new store-path signature.

```python
########################################################################
# On-disk layout: the 2D patch cache vs the lifted per-point store
########################################################################


def _write_both_stores(tmp_path):
    """Write a `talk2dino.zarr` 2D cache and a `talk2dino_lifted.zarr` per-point store side by side."""
    import numpy as np
    import zarr

    cache = zarr.open(str(tmp_path / "talk2dino.zarr"), mode="w")
    cache["features"] = np.zeros((2, 4, 3, 3), dtype=np.float32)
    cache.attrs.update({"extractor": "talk2dino", "patch_size": 14, "n_frames": 2})

    lifted = zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="w")
    lifted["features"] = np.zeros((6, 4), dtype=np.float32)
    lifted.attrs.update({"input_dim": 4, "latent_dim": 4})


def _force_lifted_first(monkeypatch, tmp_path):
    """Make `Path.glob` yield the lifted store first, so suffix filtering is what does the work."""
    from pathlib import Path

    real_glob = Path.glob

    def ordered(self, pattern):
        return sorted(real_glob(self, pattern), key=lambda p: "_lifted" not in p.name)

    monkeypatch.setattr(Path, "glob", ordered)


def test_cache_store_path_ignores_the_lifted_store(tmp_path, monkeypatch):
    from collab_splats.semantics.utils import cache_store_path

    _write_both_stores(tmp_path)
    _force_lifted_first(monkeypatch, tmp_path)
    assert cache_store_path(tmp_path).name == "talk2dino.zarr"
    # Callers read the extractor name off `.stem`; "talk2dino_lifted" would be the answer
    # if the lifted store were picked up
    assert cache_store_path(tmp_path).stem == "talk2dino"


def test_cache_store_path_raises_when_there_is_no_2d_cache(tmp_path):
    import pytest as _pytest

    from collab_splats.semantics.utils import cache_store_path

    with _pytest.raises(FileNotFoundError, match="no 2D feature cache"):
        cache_store_path(tmp_path)


def test_load_feature_maps_takes_a_store_path(tmp_path):
    from collab_splats.semantics.utils import cache_store_path, load_feature_maps

    _write_both_stores(tmp_path)
    maps = load_feature_maps(cache_store_path(tmp_path))
    assert len(maps) == 2
    assert maps[0].shape == (4, 3, 3)
```

- [ ] **Step 3: Run the new tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_semantics_utils.py -q -k "cache_store_path or load_feature_maps"
```

Expected: FAIL — `ImportError: cannot import name 'cache_store_path' from 'collab_splats.semantics.utils'`.

- [ ] **Step 4: Rewrite `collab_splats/semantics/utils.py`**

Full replacement content. The module docstring is verbatim from spec §3.5 and is the canonical
description of the on-disk layout — do not paraphrase it.

The re-export block (`ae_path` … `lifted_store_path`) and the `torch_utils` shim block are
both temporary: the first becomes a real definition in Task 2, the second is deleted in Task 4.
`interpolate_to_patch_size` and the `_tokens_to_feature_map` name also survive only until Task 4.

```python
"""
Semantic feature helpers and the on-disk layout of semantic artifacts.

Layout inside a scene's semantics dir:
- `<extractor>.zarr`: 2D patch cache, `features` (N, D, H_p, W_p) float32, one chunk per frame;
  attrs `extractor`, `patch_size`, `n_frames`.
- `<extractor>_lifted.zarr` + `<extractor>_ae.pt`: per-point codes `features` (P, latent) and the
  autoencoder that decodes them; attrs `input_dim`, `latent_dim`.
- `latent_dim == input_dim` with no `_ae.pt` is full-dim by design (`n_components: null`);
  `latent_dim < input_dim` with no `_ae.pt` is an interrupted write and raises on read.
- The `_lifted` suffix is the only thing separating the two stores in one flat dir.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from PIL import Image

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.semantics.compression import (
    FeatureAutoencoder,
    ae_path,
    find_lifted_extractor,
    lifted_store_path,
)
from collab_splats.utils.torch_utils import batch_iterator

logger = logging.getLogger(__name__)

# The three path helpers are re-exported so every caller reaches the layout through one module.
# Task 2 moves the definitions here; they stay in compression.py while save()/load() call ae_path.
__all__ = [
    "ae_path",
    "cache_store_path",
    "compute_semantic_contrast",
    "extract_feature_cache",
    "find_lifted_extractor",
    "lifted_store_path",
    "load_feature_maps",
    "load_point_features",
    "point_features_cached",
    "write_point_features",
]


########################################################
########## Re-exports from collab_splats.utils.torch_utils
########################################################

# Canonical location: collab_splats.utils.torch_utils
from collab_splats.utils.torch_utils import (  # noqa: E402,F401
    get_device,
    infer_batch_size,
    load_hf_weights,
    load_torchhub_model,
    pytorch_gc,
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

    avg_pos = pos.mean(dim=0, keepdim=True)
    stacked = torch.cat([avg_pos, neg], dim=0)
    return stacked.div(temperature).softmax(dim=0)[0]


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


########################################################
########## 2D patch cache (<extractor>.zarr) ###########
########################################################


def cache_store_path(semantics_dir: Path) -> Path:
    """
    Find the 2D patch cache store in semantics_dir — `<extractor>.zarr`.

    Args:
        semantics_dir: the scene's semantics dir.

    Returns:
        Path of the cache store. Its `.stem` is the extractor name.

    Raises:
        FileNotFoundError: when the dir holds no cache store.
    """
    sem_dir = Path(semantics_dir)
    # `*.zarr` also matches the lifted store next door; the suffix is the only thing separating them
    store = next((p for p in sem_dir.glob("*.zarr") if not p.name.endswith("_lifted.zarr")), None)
    if store is None:
        raise FileNotFoundError(
            f"no 2D feature cache (*.zarr) in {sem_dir} — extract this scene's semantics first"
        )
    return store


def extract_feature_cache(extractor, frames_zarr: Path, cache_dir: Path) -> Path:
    """
    Extract patch features from a frames zarr into `cache_dir/<extractor>.zarr`.

    Re-entrant: a cache whose extractor name and frame count both match is returned untouched.

    Args:
        extractor: a BaseFeatureExtractor instance — supplies `.name`, `.patch_size`, `.forward`.
        frames_zarr: path to the scene's canonical frames.zarr.
        cache_dir: directory to write the 2D patch cache into.

    Returns:
        Path of the store; `features` is (N, D, H_p, W_p) float32, one chunk per frame.
    """
    zarr_path = Path(cache_dir) / f"{extractor.name}.zarr"

    # Read through FrameStore, not raw zarr keys: the store's arrays are `images`/`frame_idx`
    # and its attrs are record_keys/provenance/schema_version — no `frames` key, no `n_frames`.
    frames = FrameStore.open(frames_zarr)
    N = len(frames)

    # A cache is valid when the extractor name and frame count both match
    if zarr_path.exists():
        try:
            z = zarr.open(str(zarr_path), mode="r")
            if z.attrs.get("extractor") == extractor.name and z.attrs.get("n_frames") == N:
                logger.info("Feature cache valid, skipping extraction: %s", zarr_path)
                return zarr_path
        except Exception:
            logger.warning("Cache at %s is corrupt or unreadable, re-extracting", zarr_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Probe the first frame to learn (D, H_p, W_p) before allocating the store
    first_frame = Image.fromarray(frames.image(0)).convert("RGB")
    with torch.no_grad():
        [first_feat] = extractor.forward([first_frame])
    D, H_p, W_p = first_feat.shape

    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update(
        {"extractor": extractor.name, "patch_size": extractor.patch_size, "n_frames": N}
    )
    # One chunk per frame: reading frame i loads exactly 1 disk chunk
    arr = store.create_array(
        "features",
        shape=(N, D, H_p, W_p),
        chunks=(1, D, H_p, W_p),
        dtype="float32",
        fill_value=0,
    )
    arr[0] = first_feat.cpu().float().numpy()

    # Iterate the rest lazily — never more than one frame in RAM
    for i in range(1, N):
        pil_img = Image.fromarray(frames.image(i)).convert("RGB")
        with torch.no_grad():
            [feat] = extractor.forward([pil_img])
        arr[i] = feat.cpu().float().numpy()
        if i % 10 == 0:
            logger.info("extract_feature_cache: %d/%d frames written", i + 1, N)

    logger.info("Feature cache written: %s  shape=%s", zarr_path, tuple(arr.shape))
    return zarr_path


def load_feature_maps(store_path: Path) -> list[torch.Tensor]:
    """
    Load per-frame dense feature maps from a 2D patch cache store.

    Args:
        store_path: path of an `<extractor>.zarr` store (see `cache_store_path`).

    Returns:
        One (D, H_p, W_p) CPU tensor per frame.
    """
    arr = zarr.open(str(store_path), mode="r")["features"]  # (N, D, H_p, W_p)
    return [torch.from_numpy(np.asarray(arr[i])) for i in range(arr.shape[0])]


########################################################
########## Per-point pair (<extractor>_lifted.zarr) ####
########################################################


def write_point_features(
    out_dir: Path,
    extractor: str,
    codes: np.ndarray,
    ae: Optional[FeatureAutoencoder] = None,
) -> Path:
    """
    Write the per-point pair: `<extractor>_lifted.zarr` (+ `_ae.pt` when compressed).

    Args:
        out_dir: the scene's semantics dir.
        extractor: extractor name — both halves of the pair carry it.
        codes: (P, latent) per-point codes, or (P, D) when `ae` is None.
        ae: the autoencoder that decodes `codes`, or None for full-dim codes.

    Returns:
        Path of the written lifted store.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(codes)
    lifted_zarr = lifted_store_path(out_dir, extractor)

    # input_dim == latent_dim marks full-dim-by-design codes that need no weights; unequal
    # widths mark codes the weights are REQUIRED to decode. Without the marker a missing
    # _ae.pt is ambiguous — uncompressed vs orphaned by a crash — and a reader must guess.
    # Codes, then attrs, then weights; any failure after the store exists removes it again
    # so no half-pair (unreadable codes) is ever left behind on disk.
    try:
        store = zarr.open(str(lifted_zarr), mode="w")
        store["features"] = codes
        store.attrs.update(
            {
                "input_dim": int(ae.input_dim) if ae is not None else int(codes.shape[1]),
                "latent_dim": int(ae.latent_dim) if ae is not None else int(codes.shape[1]),
            }
        )
        if ae is not None:
            ae.save(out_dir, extractor)
    except Exception:
        shutil.rmtree(lifted_zarr, ignore_errors=True)
        raise
    return lifted_zarr


def point_features_cached(semantics_dir: Path) -> bool:
    """
    Report whether the lifted store exists and is readable.

    Args:
        semantics_dir: the scene's semantics dir.

    Returns:
        True when the codes are present and either the weights are too or the codes are full-dim.
    """
    sem_dir = Path(semantics_dir)
    extractor = find_lifted_extractor(sem_dir)
    if extractor is None:
        return False
    if ae_path(sem_dir, extractor).exists():
        return True
    # No weights: usable only if the codes describe themselves as full-dim. Anything else is a
    # half-written pair (crash between the two writes) — report NOT cached so the caller re-lifts.
    try:
        attrs = zarr.open(str(lifted_store_path(sem_dir, extractor)), mode="r").attrs
        return int(attrs["latent_dim"]) >= int(attrs["input_dim"])
    except Exception:
        return False


def load_point_features(semantics_dir: Path, batch_size: int = 65_536) -> np.ndarray:
    """
    Read the lifted per-point store, decoding latent codes back to full dim.

    Args:
        semantics_dir: the scene's semantics dir.
        batch_size: points per decode chunk — a one-shot decode of a 500k-point scene
            materialises ~2.3 GB of float32 at 768-D, against a 46.6 GB shared container cap.

    Returns:
        (P, D) float32, L2-normalized per row.

    Raises:
        FileNotFoundError: when no lifted store exists, or when latent codes have no weights.
    """
    sem_dir = Path(semantics_dir)
    extractor = find_lifted_extractor(sem_dir)
    if extractor is None:
        raise FileNotFoundError(
            f"no *_lifted.zarr store in {sem_dir} — lift this scene's features first"
        )
    lifted_zarr = lifted_store_path(sem_dir, extractor)
    weights = ae_path(sem_dir, extractor)
    store = zarr.open(str(lifted_zarr), mode="r")
    codes = np.asarray(store["features"])

    # Weights present -> always decode, even at equal widths: an equal-width autoencoder still
    # encodes, so its codes are not full-dim features. Weights absent is the ambiguous case.
    if not weights.exists():
        try:
            full_dim = int(store.attrs["latent_dim"]) >= int(store.attrs["input_dim"])
        except KeyError:
            full_dim = False
        if full_dim:
            return F.normalize(torch.from_numpy(codes), dim=1).cpu().numpy()
        raise FileNotFoundError(
            f"{lifted_zarr} holds {codes.shape[1]}-D per-point codes but the autoencoder that "
            f"decodes them ({weights}) is missing — the pair was written only halfway "
            "(interrupted run). Returning the raw codes would be silent garbage; re-lift this "
            f"scene's semantic features instead (delete {lifted_zarr} and re-run the semantics step)."
        )

    # Task 2 collapses this to `FeatureAutoencoder.load(weights)`; the dir+extractor form is
    # still the API at this commit.
    ae = FeatureAutoencoder.load(sem_dir, extractor)
    # Streamed decode into a preallocated output: peak stays at (result + one chunk). Row-wise
    # normalize and the decoder's linear layers are both row-independent, so chunking is exact.
    codes_t = torch.from_numpy(codes)
    decoded = torch.empty((codes_t.shape[0], ae.input_dim), dtype=torch.float32)
    with torch.no_grad():
        start = 0
        for (chunk,) in batch_iterator(batch_size, codes_t):
            end = start + len(chunk)
            decoded[start:end] = F.normalize(ae.per_point_decode(chunk), dim=1)
            start = end
    return decoded.numpy()
```

Three things that were separate names in `dashboard/pipeline.py` are gone here:

| dropped | why |
|---|---|
| `_is_full_dim(attrs)` | a three-line predicate with two call sites, both in this module — inlined |
| `cache_extractor_name(dir)` | `return cache_store_path(dir).stem` — an alias, not a function |
| `_DECODE_BATCH_SIZE` | module constant reached only by a monkeypatching test — now a parameter |
| `skip_existing=` on `extract_feature_cache`, `decode=` on `load_point_features` | no caller ever passed either; the non-default branch was dead |

- [ ] **Step 5: Delete `write_point_features` from `compression.py`**

Remove lines 360-391 of `collab_splats/semantics/compression.py` — the whole
`def write_point_features(...)` block — and drop the now-unused imports it was the only user of:
`shutil` (line 19), `numpy as np` (line 23) and `zarr` (line 27).

Verify nothing else in the file uses them:

```bash
grep -n "shutil\|np\.\|zarr" collab_splats/semantics/compression.py
```

Expected: no output.

- [ ] **Step 6: Run the new store-selection tests — they should pass now**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_semantics_utils.py -q
```

Expected: PASS (the four new tests plus the existing contrast/interpolate/torch-utils ones).

- [ ] **Step 7: Delete both cache methods from `features/base.py`**

Delete `extract_and_cache` (lines 85-155) and `extract_and_cache_from_zarr` (lines 177-251) from
`collab_splats/semantics/features/base.py`. `features_to_rgb` sits between them and **stays**.

Then drop the imports they were the only users of. Replace the import block at lines 10-32 with:

```python
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    get_device,
    interpolate_to_patch_size,
)
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.torch_utils import RegistryMixin
```

`datetime` and `FrameStore` are gone (only the deleted methods used them). `Tuple`, `T`, `AutoModel`,
`get_device`, `interpolate_to_patch_size`, `open_image` and `resize_image` are flagged unused by
pyflakes *today* but Task 3 makes `T`, `open_image`, `resize_image` and `Image` live again via the
hoisted `preprocess`, and Task 4 deletes the rest. Leaving them here for one task is deliberate.

- [ ] **Step 8: Retarget `tests/semantics/features/test_extract_from_zarr.py`**

Replace the three `extract_and_cache_from_zarr` tests (lines 74-113) with function-call versions.
The `features_to_rgb` tests above them are untouched.

```python
########################################################################
# extract_feature_cache
########################################################################


def test_extract_feature_cache_creates_zarr():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "cache"
        result = extract_feature_cache(extractor, frames_zarr, cache_dir)
        assert result == cache_dir / "_test_extractor.zarr"
        assert result.exists()


def test_extract_feature_cache_feature_shape():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "cache"
        result = extract_feature_cache(extractor, frames_zarr, cache_dir)
        store = zarr.open(str(result), mode="r")
        # The name says feature_shape, so assert the whole shape, not just N
        n, d, h_p, w_p = store["features"].shape
        assert n == 3
        assert d == extractor._D
        assert h_p == extractor._H_p
        assert w_p == extractor._W_p
        assert store.attrs["extractor"] == "_test_extractor"
        assert store.attrs["n_frames"] == 3
        assert store.attrs["patch_size"] == 16
        # created_at / feature_dim were dropped — nothing read them
        assert "created_at" not in store.attrs
        assert "feature_dim" not in store.attrs


def test_extract_feature_cache_reuses_a_matching_cache():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "cache"
        result1 = extract_feature_cache(extractor, frames_zarr, cache_dir)
        mtime = result1.stat().st_mtime_ns
        result2 = extract_feature_cache(extractor, frames_zarr, cache_dir)
        assert result1 == result2
        assert result2.stat().st_mtime_ns == mtime  # untouched -> extraction was skipped
```

Add the import at the top of the file, next to the existing imports:

```python
from collab_splats.semantics.utils import extract_feature_cache
```

and update the module docstring line 1 to:

```python
"""Tests for BaseFeatureExtractor.features_to_rgb and semantics.utils.extract_feature_cache."""
```

Also fix the stale reference inside `_make_frames_zarr`'s docstring at line 38 —
`extract_and_cache_from_zarr` becomes `extract_feature_cache`.

- [ ] **Step 9: Run the extractor-cache tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/features/test_extract_from_zarr.py -q
```

Expected: PASS, 6 tests.

- [ ] **Step 10: Retarget `tests/semantics/test_artifact_layout.py` imports**

Change the import block at lines 8-12 from `collab_splats.semantics.compression` to pull
`find_lifted_extractor` and `write_point_features` from `collab_splats.semantics.utils`,
leaving `FeatureAutoencoder` on `compression`:

```python
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import find_lifted_extractor, write_point_features
```

Add one test at the end of the file pinning the new return type (spec §3.4 declares `-> Path`;
the function returned `None` before this task):

```python
def test_write_point_features_returns_the_lifted_store_path(tmp_path):
    import numpy as np

    from collab_splats.semantics.utils import lifted_store_path

    out = write_point_features(tmp_path, "talk2dino", np.zeros((4, 8), dtype=np.float32))
    assert out == lifted_store_path(tmp_path, "talk2dino")
    assert out.exists()
```

- [ ] **Step 11: Rewrite `dashboard/pipeline.py`'s semantics section**

Replace the `collab_splats.semantics.*` import block (lines 33-42) with:

```python
from collab_splats.semantics.utils import (
    cache_store_path,
    extract_feature_cache,
    load_feature_maps,
    load_point_features,
    point_features_cached,
    write_point_features,
)
```

`FeatureAutoencoder` is still needed by `_lift_and_compress`, so keep one line for it:

```python
from collab_splats.semantics.compression import FeatureAutoencoder
```

`BaseFeatureExtractor` (line 43) stays. Drop `from collab_splats.utils.torch_utils import batch_iterator`
(line 45) — only `load_point_features` used it and that has moved. Drop `from collab_splats.utils.image
import open_image` only if nothing else in the file uses it; verify with
`grep -n "open_image" collab_splats/dashboard/pipeline.py` first and keep the import if there are
other hits.

Rewrite `_extract_semantics` (lines 95-103):

```python
def _extract_semantics(extractor_name: str, frames_zarr: Path, out_dir: Path) -> None:
    """Extract + cache patch features straight from frames.zarr — no JPG export.

    Args:
        extractor_name: registry key of the extractor to run.
        frames_zarr: path to the scene's canonical frames.zarr.
        out_dir: the scene's semantics dir.
    """
    # The returned cache path is deliberately dropped: every consumer resolves the store by
    # glob (cache_store_path), including the viewer's legacy lift, which has no handle to thread.
    extract_feature_cache(BaseFeatureExtractor.get(extractor_name)(), frames_zarr, out_dir)
```

Delete these (lines 106-136 and 210-280). `cache_store_path`, `load_feature_maps`,
`point_features_cached` and `load_point_features` now live in `semantics.utils`; the other three
are gone entirely — `_DECODE_BATCH_SIZE` became `load_point_features`'s `batch_size` parameter,
`_is_full_dim` was inlined into its two call sites, and `cache_extractor_name` was an alias for
`cache_store_path(...).stem`.

**Keep** `AutoencoderPolicy`, `semantics_ae_policy`, `resolve_latent_dim`, `resolve_semantics_dir` —
they are dashboard config policy, not artifact layout.

In `_lift_and_compress` (line 292), change the loader call to pass a store path:

```python
    feature_maps = load_feature_maps(cache_store_path(semantics_dir))  # list of (D, H_p, W_p) on CPU
```

The `write_point_features(...)` call at the end of that function is unchanged — same signature,
new import source.

- [ ] **Step 12: Retarget `dashboard/viewer.py`**

In `lift_point_features` (lines 39-52), drop the inline pipeline import and use the store path:

```python
    # Lazy import: pointcloud.utils pulls the heavy feedforward stack.
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.pointcloud.utils import lift_features
```

and further down:

```python
    feature_maps = load_feature_maps(cache_store_path(semantics_dir))
```

In `_save_point_features` (lines 73-82), the lazy import block becomes:

```python
    # Lazy import: dashboard.pipeline pulls the feedforward stack at module import.
    from collab_splats.dashboard.pipeline import resolve_latent_dim, semantics_ae_policy
    from collab_splats.semantics.compression import FeatureAutoencoder
```

In `ensure_lifted` (lines 246-249), delete the lazy pipeline import of
`load_point_features` / `point_features_cached` entirely — they come from the top-level import below.

Add at the top of `viewer.py`, with the other module-level imports:

```python
from collab_splats.semantics.utils import (
    cache_store_path,
    load_feature_maps,
    load_point_features,
    point_features_cached,
    write_point_features,
)
```

`_save_point_features` (line 94) named the extractor via `cache_extractor_name(semantics_dir)`;
that alias is gone, so the call reads the name off the store path directly:

```python
    write_point_features(
        Path(semantics_dir),
        cache_store_path(semantics_dir).stem,
        codes.detach().cpu().numpy(),
        ae,
    )
```

`dashboard/pipeline.py:320` has the same call and takes the same edit.

- [ ] **Step 13: Measure the dashboard import cost this adds**

`app.py:26` imports `viewer` at module scope, and importing `collab_splats.semantics.utils` executes
`collab_splats/semantics/__init__.py`, which pulls the extractors and SAM. That is exactly the cost
viewer's lazy imports were built to avoid, so measure it rather than assume.

```bash
/opt/venv/reconstruction/bin/python -X importtime -c "import collab_splats.dashboard.app" 2>&1 | tail -1
```

Run this **before** Step 12 too, and compare. If the top-level import adds more than 2 s of
cumulative import time, revert exactly one thing: make the `semantics.utils` import in
`viewer.py` lazy again — same names, same module, moved back inside the three functions that use
them (`lift_point_features`, `_save_point_features`, `ensure_lifted`). The spec's requirement is
that the names come from `semantics.utils`; where the import statement sits is a loadtime decision,
and the dashboard-loadtime work is what makes it one. Record the two numbers in the commit body.

- [ ] **Step 14: Retarget `dashboard/app.py`**

Both lazy import blocks (lines 743-746 and 781-784) split — `point_features_cached` moves to
`semantics.utils`, `resolve_semantics_dir` stays on `pipeline`:

```python
        from collab_splats.dashboard.pipeline import resolve_semantics_dir
        from collab_splats.semantics.utils import point_features_cached
```

Apply the same replacement at both sites. The surrounding comments stay.

- [ ] **Step 15: Retarget `wrapper/reconstructor.py`**

Replace the import block at lines 50-54:

```python
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import (
    extract_feature_cache,
    lifted_store_path,
    load_feature_maps,
    write_point_features,
)
```

Rewrite `_extract_2d_features` (lines 526-539):

```python
def _extract_2d_features(
    extractor_name: str,
    frames_zarr: Path,
    cache_dir: Path,
) -> Path:
    """Extract 2D features for all frames straight from the canonical store.

    Args:
        extractor_name: registry key of the extractor to run.
        frames_zarr: path to the scene's canonical frames.zarr.
        cache_dir: directory to write `<extractor>.zarr` into.

    Returns:
        Path of the written 2D patch cache.
    """
    # extract_feature_cache iterates the frames zarr lazily — no temp JPG export, no full-RAM load.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extract_feature_cache(_get_extractor(extractor_name), frames_zarr, cache_dir)
```

In `_lift_and_save`, replace the raw zarr block (three lines starting `store = zarr.open(str(zarr_path), mode="r")`)
with:

```python
    # Load feature maps from the 2D cache: one (D, H_p, W_p) tensor per frame
    feature_maps = load_feature_maps(zarr_path)
```

Then check whether `zarr` is still used elsewhere in the file:

```bash
grep -n "zarr\." collab_splats/wrapper/reconstructor.py | head
```

Keep the `import zarr` if there are other hits; drop it if not.

- [ ] **Step 16: Retarget the remaining test import sites**

Five files import these helpers from their old homes. Change only the import lines:

`tests/dashboard/test_pipeline.py:15`

```python
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import write_point_features
```

`tests/dashboard/test_viewer_lift.py:125` (inside `test_ensure_lifted_relifts_when_cached_codes_lack_weights`)

```python
    from collab_splats.semantics.compression import FeatureAutoencoder
    from collab_splats.semantics.utils import write_point_features
```

`tests/dashboard/test_app.py:784`

```python
    from collab_splats.semantics.utils import write_point_features
```

(keep whatever `FeatureAutoencoder` import that block already has, pointing at `compression`).

`tests/dashboard/test_viewer.py:480`

```python
    from collab_splats.semantics.utils import load_point_features
```

`tests/wrapper/test_reconstructor.py:1266`

```python
    from collab_splats.semantics.utils import lifted_store_path
```

⚠️ `tests/wrapper/test_reconstructor.py` has foreign uncommitted edits. Run
`git diff tests/wrapper/test_reconstructor.py` before editing and note what was already there.

- [ ] **Step 17: Fix the two `patch.object(pl, ...)` monkeypatch targets in `tests/dashboard/test_pipeline.py`**

`patch.object(pl, "load_feature_maps", ...)` at lines 195 and 297 still works — `pipeline` imports the
name into its own namespace, and `_lift_and_compress` calls it unqualified. But the call is now
`load_feature_maps(cache_store_path(semantics_dir))`, so `cache_store_path` runs for real and needs a
2D cache on disk. Both tests already call `_write_semantics(...)`; confirm it writes `<extractor>.zarr`.
If it only writes the lifted store, add the 2D cache to the fixture:

```python
    cache = zarr.open(str(Path(semantics_dir) / f"{extractor}.zarr"), mode="w")
    cache["features"] = np.zeros((1, input_dim, 2, 2), dtype=np.float32)
    cache.attrs.update({"extractor": extractor, "patch_size": 14, "n_frames": 1})
```

Replace the `_DECODE_BATCH_SIZE` monkeypatch at line ~340 — the constant is gone, the chunk size
is a parameter now, so the test passes it:

```python
    out = pl.load_point_features(sem_dir, batch_size=5)
```

Delete the `monkeypatch.setattr(pl, "_DECODE_BATCH_SIZE", 5)` line above it. If `monkeypatch` is
now unused in that test, drop it from the signature too.

Rewrite `test_extract_semantics_returns_nothing` (line ~370) — it mocked a method that no longer exists:

```python
def test_extract_semantics_returns_nothing(tmp_path, monkeypatch):
    """_extract_semantics drops the cache path — every consumer re-resolves it by glob."""
    calls = []
    monkeypatch.setattr(
        pl, "extract_feature_cache", lambda extractor, frames, out: calls.append((frames, out))
    )
    monkeypatch.setattr(pl.BaseFeatureExtractor, "get", staticmethod(lambda name: lambda: object()))
    assert pl._extract_semantics("talk2dino", tmp_path / "frames.zarr", tmp_path) is None
    assert len(calls) == 1
```

- [ ] **Step 18: Delete `tests/dashboard/test_semantics_store_selection.py`, relocating its third test**

Two of its three tests were rewritten in Step 2. The third,
`test_viewer_lift_routes_through_the_shared_loader`, belongs with the viewer's other lift tests.
Move it into `tests/dashboard/test_viewer_lift.py` with the patch target updated:

```python
def test_viewer_lift_routes_through_the_shared_loader(tmp_path):
    """viewer.lift_point_features must use the ONE loader, not a private re-implementation."""
    from unittest.mock import patch

    from collab_splats.dashboard import viewer as viewer_mod

    with (
        patch("collab_splats.dashboard.viewer.load_feature_maps", return_value=[]) as loader,
        patch("collab_splats.pointcloud.utils.lift_features", return_value=torch.zeros(3, 4)),
        patch("collab_splats.dashboard.viewer.cache_store_path", return_value=tmp_path / "x.zarr"),
    ):
        viewer_mod.lift_point_features(object(), tmp_path)
    assert loader.call_count == 1
```

Then delete the file:

```bash
git rm tests/dashboard/test_semantics_store_selection.py
```

The other two `patch("collab_splats.dashboard.pipeline.load_feature_maps", ...)` sites in
`test_viewer_lift.py` (lines 36 and 87) must also become
`patch("collab_splats.dashboard.viewer.load_feature_maps", ...)` — the viewer holds its own
reference now.

- [ ] **Step 19: Rebuild `collab_splats/semantics/__init__.py`**

```python
"""
collab_splats.semantics — feature extraction, segmentation, and query interfaces.
"""

from .compression import FeatureAutoencoder
from .features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
    DINOFeatureExtractor,
    MaskCLIPExtractor,
    Talk2DinoExtractor,
)
from .segmentation import (
    BaseSegmentation,
    MobileSAMSegmentation,
    SAM3Segmentation,
    aggregate_masked_features,
    convert_matched_mask,
    create_composite_mask,
    create_patch_mask,
    load_mobile_sam,
    mask_id_to_binary_mask,
)
from .utils import (
    ae_path,
    cache_store_path,
    compute_semantic_contrast,
    extract_feature_cache,
    find_lifted_extractor,
    interpolate_to_patch_size,
    lifted_store_path,
    load_feature_maps,
    load_point_features,
    point_features_cached,
    write_point_features,
)

__all__ = [
    # compression
    "FeatureAutoencoder",
    # extractors
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    # semantic helpers + artifact layout
    "compute_semantic_contrast",
    "interpolate_to_patch_size",
    "cache_store_path",
    "extract_feature_cache",
    "load_feature_maps",
    "write_point_features",
    "load_point_features",
    "point_features_cached",
    "lifted_store_path",
    "ae_path",
    "find_lifted_extractor",
    # segmentation
    "BaseSegmentation",
    "MobileSAMSegmentation",
    "SAM3Segmentation",
    "load_mobile_sam",
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
]
```

The six torch_utils names are gone from `__all__` but still importable from `semantics.utils` — the
shim itself dies in Task 4. `interpolate_to_patch_size` is still listed; Task 4 removes it too.

- [ ] **Step 20: Run the full suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/test_semantics_logging.py tests/test_cu121_migration.py -q 2>&1 | tail -20
```

Expected: count = the Step 1 baseline minus 3 (the deleted store-selection file) plus 4
(the three new store-selection tests in `test_semantics_utils.py`, the one new
`write_point_features` return-type test) plus 1 (the relocated viewer test) = **baseline + 2**.

Two caveats the plan's author did not anticipate:

- The suite is **not** all-green at baseline. Add `--continue-on-collection-errors` to the command
  and compare against the measured baseline, not against zero failures. The pre-existing
  non-passing items belong to a concurrent session's in-flight uv migration and are unrelated to
  semantics: 3 × `tests/wrapper/test_reconstructor.py` base.yaml defaults, 3 ×
  `tests/test_cu121_migration.py` gsplat-pin tests, and 2 collection errors
  (`tests/wrapper/test_splats_stage.py`, `tests/wrapper/test_vda_context.py`) from an installed
  gsplat 1.4.0 that lacks `gsplat.losses`.
- Concurrent sessions add and remove tests in `tests/wrapper/` while this plan runs, so the raw
  total can move for reasons that are not yours. Verify the delta per file rather than trusting
  the total.

- [ ] **Step 21: Dashboard smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: prints `SMOKE PASS`, exit 0. This is mandatory — this task edits three dashboard modules.

- [ ] **Step 22: Format the files you touched**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/semantics/utils.py collab_splats/semantics/compression.py collab_splats/semantics/features/base.py collab_splats/semantics/__init__.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/viewer.py collab_splats/dashboard/app.py collab_splats/wrapper/reconstructor.py tests/semantics/test_semantics_utils.py tests/semantics/test_artifact_layout.py tests/semantics/features/test_extract_from_zarr.py tests/dashboard/test_pipeline.py tests/dashboard/test_viewer_lift.py tests/dashboard/test_viewer.py tests/dashboard/test_app.py tests/wrapper/test_reconstructor.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/semantics/utils.py collab_splats/semantics/compression.py collab_splats/semantics/features/base.py collab_splats/semantics/__init__.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/viewer.py collab_splats/dashboard/app.py collab_splats/wrapper/reconstructor.py
```

Re-run Step 20 after formatting. Expected: still PASS.

- [ ] **Step 23: Commit**

```bash
git status --short
git commit --only \
  collab_splats/semantics/utils.py \
  collab_splats/semantics/compression.py \
  collab_splats/semantics/features/base.py \
  collab_splats/semantics/__init__.py \
  collab_splats/dashboard/pipeline.py \
  collab_splats/dashboard/viewer.py \
  collab_splats/dashboard/app.py \
  collab_splats/wrapper/reconstructor.py \
  tests/semantics/test_semantics_utils.py \
  tests/semantics/test_artifact_layout.py \
  tests/semantics/features/test_extract_from_zarr.py \
  tests/dashboard/test_pipeline.py \
  tests/dashboard/test_viewer_lift.py \
  tests/dashboard/test_viewer.py \
  tests/dashboard/test_app.py \
  tests/dashboard/test_semantics_store_selection.py \
  tests/wrapper/test_reconstructor.py \
  -m "refactor(semantics): fold artifact I/O into semantics.utils

Move the on-disk layout of a scene's semantics dir into one module:
cache_store_path, extract_feature_cache, load_feature_maps,
write_point_features, point_features_cached and load_point_features now
all live in semantics/utils.py.

Three names did not survive the move. cache_extractor_name was an alias
for cache_store_path(dir).stem. _is_full_dim was a three-line predicate
with two call sites, both in the same module. _DECODE_BATCH_SIZE was a
module constant reached only by a monkeypatching test and is now
load_point_features' batch_size parameter. The skip_existing and decode
flags are gone too — no caller ever passed either, so both non-default
branches were dead.

extract_and_cache_from_zarr becomes the function extract_feature_cache
(extractor as an argument, so utils never imports features); the dead
batch_size arg and the unread created_at/feature_dim attrs go with it.
extract_and_cache, which nothing called, is deleted.

load_feature_maps takes the store path instead of the directory —
callers wrap with cache_store_path.

Dashboard import time: <before>s -> <after>s.

Suite green: <count> passed. SMOKE PASS."
```

Fill in the three placeholders from Steps 13 and 20 before committing.
`tests/wrapper/test_reconstructor.py` carries foreign uncommitted hunks — check
`git diff` first and mention them in the commit body if any are swept in.

---

## Task 2: Trim `FeatureAutoencoder`, make persistence path-based

Three changes, one commit:

1. Delete the regularization branch (`regularization_kwargs`, `reg_head`, `_reg_dim`, `_reg_weight`,
   `fit(reg_target=...)`), the `lr_scheduler` arg, and the `hidden_dim` constructor arg. Nothing in
   the repo passes any of them. `hidden_dim = max(64, 2 * latent_dim)` stays as an internal local.
2. `save(path)` / `load(path)` take the **weights file path**, not a directory plus an extractor name.
   The old form's `path.mkdir()` is a trap — passing a filename silently created a *directory* of that
   name (`tests/wrapper/test_reconstructor.py:554` documents it). The new form does
   `path.parent.mkdir(parents=True, exist_ok=True)`, which is what nb05 relies on to create its
   semantics dir.
3. The path helpers move physically out of `compression.py` into `utils.py`. With `save`/`load` no
   longer calling `ae_path`, `compression.py` imports torch and nothing else from this package —
   the acyclic import graph the spec specifies.
4. Two more things with no caller go: the spatial `decode()`, and the `LIFTED_SUFFIX` / `AE_SUFFIX`
   module constants (folded into the path helpers as they move).

No call site changes: Task 1 already pointed everything at `semantics.utils`.

> **Deviation from spec §4/§10, deliberate.** The spec has `fit` return a
> `{"recon_cosine", "recon_mse", "epochs_run"}` dict "also stored on self", and keeps the spatial
> `decode()`. Neither has a caller: all three `fit` sites
> (`reconstructor.py:589`, `dashboard/viewer.py:89`, `dashboard/pipeline.py:302`) drop the return
> and read the attributes, and nothing outside `test_decode_spatial_shape` calls `decode` —
> the dashboard encodes patch maps, lifts them, then decodes *per point* on read. A return value
> duplicating three attributes, and a method exercised only by its own test, are the same dead
> weight as the `reg_head` this task deletes. `fit` returns `None`; `decode` goes.

**Files:**
- Modify: `collab_splats/semantics/compression.py` (full rewrite)
- Modify: `collab_splats/semantics/utils.py` (path helpers arrive; `ae.save` / `FeatureAutoencoder.load` calls updated)
- Test: `tests/semantics/test_compression.py`, `tests/semantics/test_compression_target.py`,
  `tests/semantics/test_artifact_layout.py`, `tests/dashboard/test_pipeline.py`, `tests/dashboard/test_viewer.py`
- Modify: `docs/source/tutorials/05_lifting/semantic_lifting.ipynb`, `docs/source/tutorials/06_mesh/splats_mesh.ipynb`

---

- [ ] **Step 1: Write the failing tests for the new surface**

Replace the "Persistence" section of `tests/semantics/test_compression.py` (lines 101-125), delete
the entire "Regularization head" section (lines 128-204), and delete `test_decode_spatial_shape`
(lines 53-57) along with the `decode` half of the module docstring on line 1:

```python
"""Tests for FeatureAutoencoder — encode (spatial + point) and per_point_decode."""
```

Also delete the now-unused constants
`REG_DIM` (line 15) and `REG_KWARGS` (line 18), and the `import pytest` at line 6 if no test left in
the file uses it (`grep -n "pytest" tests/semantics/test_compression.py` — after the deletions there
are no hits, so remove it).

New persistence section:

```python
########################################################################
# Persistence
########################################################################


def test_save_load_roundtrip():
    """Save + load produces identical per_point_encode output."""
    torch.manual_seed(99)
    features = torch.randn(N, INPUT_DIM)

    ae = _make_ae()
    ae.fit(features, epochs=2, batch_size=N)

    x = torch.randn(N, INPUT_DIM)
    with torch.no_grad():
        expected = ae.per_point_encode(x)

    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "talk2dino_ae.pt"
        ae.save(weights)
        ae2 = FeatureAutoencoder.load(weights)

    with torch.no_grad():
        actual = ae2.per_point_encode(x)

    assert torch.allclose(expected, actual, atol=1e-6), "encode output changed after save/load"


def test_save_creates_the_parent_dir_not_a_dir_named_after_the_file():
    """save() takes a FILE path: it mkdirs the parent, never the path itself.

    The old dir+extractor signature mkdir'd whatever it was handed, so passing a filename
    silently produced a DIRECTORY of that name and the weights went inside it.
    """
    ae = _make_ae()
    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "nested" / "talk2dino_ae.pt"
        ae.save(weights)
        assert weights.is_file()
        assert weights.parent.is_dir()


def test_hidden_dim_is_not_a_constructor_arg():
    """Width is derived from latent_dim — an override nothing passed is not a parameter."""
    import inspect

    params = inspect.signature(FeatureAutoencoder.__init__).parameters
    assert list(params) == ["self", "input_dim", "latent_dim"]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_compression.py -q
```

Expected: FAIL — `test_save_load_roundtrip` raises `TypeError: save() missing 1 required positional
argument: 'extractor'`, and `test_hidden_dim_is_not_a_constructor_arg` fails on the extra params.

- [ ] **Step 3: Rewrite `collab_splats/semantics/compression.py`**

Full replacement content:

```python
"""
Lightweight feature autoencoder for compressing patch features before 3D lifting.

Trained once on keyframe features, then used to reduce the dimensionality of the semantic
features stored per point. Compression happens on either shape, decompression only on points:
  encode                  — spatial patch maps (D, H, W) -> (latent_dim, H, W)
  per_point_encode/decode — flat point arrays  (P, D)   <-> (P, latent_dim)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

########################################################################
# FeatureAutoencoder
########################################################################


class FeatureAutoencoder(nn.Module):
    """
    Two-layer MLP autoencoder for compressing semantic patch features.

    Args:
        input_dim: width of the features being compressed.
        latent_dim: width of the codes written to disk.
    """


    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()

        # Hidden width is derived, never configured: twice the latent dim, floor 64
        hidden_dim = max(64, 2 * latent_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # decoder: latent_dim -> hidden_dim -> input_dim, split so both branches share the hidden
        self.decoder_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        # Fit quality from the last fit(), persisted in the checkpoint. Both are measured on the
        # TRAINING set, so gate on epochs_run > 0 before trusting recon_cosine/recon_mse at all.
        self.recon_cosine: float = 0.0
        self.recon_mse: float = 0.0
        self.epochs_run: int = 0

    ####################################################################
    # Image branch — spatial patch maps (D, H, W)
    ####################################################################

    # Compress-only: the dashboard encodes patch maps then lifts the latent maps to points,
    # so decoding a map back to (D, H, W) never happens. per_point_decode is the read path.

    def encode(self, feature_map: Tensor) -> Tensor:
        """
        Encode a spatial patch map.

        Args:
            feature_map: (input_dim, H, W).

        Returns:
            (latent_dim, H, W).
        """
        _, H, W = feature_map.shape
        return self.encoder(feature_map.flatten(1).T).T.reshape(self.latent_dim, H, W)

    ####################################################################
    # Point branch — flat point arrays (P, D)
    ####################################################################

    def per_point_encode(self, feats: Tensor) -> Tensor:
        """
        Encode flat point features.

        Args:
            feats: (P, input_dim).

        Returns:
            (P, latent_dim).
        """
        return self.encoder(feats)

    def per_point_decode(self, codes: Tensor) -> Tensor:
        """
        Decode flat point codes.

        Args:
            codes: (P, latent_dim).

        Returns:
            (P, input_dim).
        """
        return self.decoder_out(self.decoder_hidden(codes))

    ####################################################################
    # Training
    ####################################################################

    def fit(
        self,
        features: Tensor,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        on_epoch: Optional[Callable[[int, int, float], None]] = None,
        target_cosine: Optional[float] = None,
    ) -> None:
        """
        Train the autoencoder in place; loss is MSE(recon, x) + (1 - cosine(recon, x)).

        Args:
            features: (N, input_dim) training features.
            epochs: epoch ceiling.
            batch_size: mini-batch size.
            lr: Adam learning rate.
            on_epoch: callback(epoch, epochs, avg_loss) fired once per epoch — the dashboard
                reads progress through it, since tqdm and logger.debug do not reach the UI.
            target_cosine: stop once mean reconstruction cosine reaches this, with `epochs`
                as the ceiling. Measured on the training set, so optimistic at small N.

        Fit quality lands on `self` as recon_cosine / recon_mse / epochs_run.

        Raises:
            ValueError: if `features` has no samples.
        """
        # Reject empty input up front: zero gradient steps would still record metrics and could
        # satisfy target_cosine, publishing a false "trained" signal downstream.
        if features.shape[0] == 0:
            raise ValueError(
                f"fit() requires at least one sample; got features with shape {tuple(features.shape)}"
            )

        self.to(features.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        N = features.shape[0]

        pbar = tqdm(range(epochs), desc="fit autoencoder", unit="epoch")
        for epoch in pbar:
            # Shuffle patch indices each epoch for unbiased mini-batches
            perm = torch.randperm(N, device=features.device)
            epoch_loss = 0.0
            epoch_cos = 0.0
            epoch_mse = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x = features[idx]

                recon = self.decoder_out(self.decoder_hidden(self.encoder(x)))
                mse = F.mse_loss(recon, x)
                cos = F.cosine_similarity(recon, x).mean()
                loss = mse + (1 - cos)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cos += cos.item()
                epoch_mse += mse.item()
                n_batches += 1

            # N >= 1 is guaranteed above and range(0, N, batch_size) yields >= 1 batch,
            # so n_batches is never 0 here.
            avg_loss = epoch_loss / n_batches
            self.recon_cosine = epoch_cos / n_batches
            self.recon_mse = epoch_mse / n_batches
            self.epochs_run = epoch + 1

            pbar.set_postfix(loss=f"{avg_loss:.6f}", cos=f"{self.recon_cosine:.4f}")
            logger.debug("epoch %d/%d  loss=%.6f  cos=%.4f", epoch + 1, epochs, avg_loss, self.recon_cosine)
            if on_epoch is not None:
                on_epoch(epoch + 1, epochs, avg_loss)

            # Early stop once reconstruction is good enough; epochs is the ceiling
            if target_cosine is not None and self.recon_cosine >= target_cosine:
                logger.info(
                    "target cosine %.4f reached at epoch %d/%d (cos=%.4f) — stopping",
                    target_cosine,
                    epoch + 1,
                    epochs,
                    self.recon_cosine,
                )
                break

        self.eval()

    ####################################################################
    # Persistence
    ####################################################################

    def save(self, path: Path) -> None:
        """
        Write the checkpoint to a weights file path.

        Args:
            path: destination `.pt` file — its parent dir is created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise from CPU so the checkpoint is device-agnostic, then restore the device
        device = next(self.parameters()).device
        self.cpu()
        torch.save(
            {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "state_dict": self.state_dict(),
                "recon_cosine": self.recon_cosine,
                "recon_mse": self.recon_mse,
                "epochs_run": self.epochs_run,
            },
            path,
        )
        self.to(device)
        logger.info("saved autoencoder → %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureAutoencoder":
        """
        Read a checkpoint from a weights file path.

        Args:
            path: the `.pt` file written by `save`.

        Returns:
            An eval-mode FeatureAutoencoder with the checkpoint's weights and fit metrics.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ae = cls(input_dim=payload["input_dim"], latent_dim=payload["latent_dim"])
        ae.load_state_dict(payload["state_dict"])
        # Checkpoints predating the metric keys default to the untrained values rather than failing
        ae.recon_cosine = payload.get("recon_cosine", 0.0)
        ae.recon_mse = payload.get("recon_mse", 0.0)
        ae.epochs_run = payload.get("epochs_run", 0)
        ae.eval()

        logger.info("loaded autoencoder ← %s", path)
        return ae
```

- [ ] **Step 4: Move the path helpers into `utils.py`**

In `collab_splats/semantics/utils.py`, replace the `from collab_splats.semantics.compression import (...)`
block (added in Task 1) with just:

```python
from collab_splats.semantics.compression import FeatureAutoencoder
```

Then add this section immediately after the module's `logger = logging.getLogger(__name__)` and
`__all__`, before "Contrastive scoring":

```python
########################################################
########## Artifact paths ##############################
########################################################

# `_lifted` is load-bearing, not decorative: the flat layout puts the 2D patch cache
# (`<extractor>.zarr`) and the lifted per-point codes in the SAME dir, so the filename is the
# only thing that can tell them apart there.


def lifted_store_path(out_dir: Path, extractor: str) -> Path:
    """
    Path of one extractor's per-point latent store.

    Args:
        out_dir: the scene's semantics dir.
        extractor: extractor name.

    Returns:
        `out_dir/<extractor>_lifted.zarr`.
    """
    return Path(out_dir) / f"{extractor}_lifted.zarr"


def ae_path(out_dir: Path, extractor: str) -> Path:
    """
    Path of the autoencoder that decodes `lifted_store_path`'s codes.

    Args:
        out_dir: the scene's semantics dir.
        extractor: extractor name.

    Returns:
        `out_dir/<extractor>_ae.pt`.
    """
    return Path(out_dir) / f"{extractor}_ae.pt"


def find_lifted_extractor(out_dir: Path) -> Optional[str]:
    """
    Name of the single lifted extractor in a dir, or None when there is none.

    Args:
        out_dir: the scene's semantics dir.

    Returns:
        The extractor name, or None.

    Raises:
        ValueError: when the dir holds more than one lifted store — guessing would pair one
            extractor's codes with another's decoder.
    """
    stems = sorted(p.name[: -len("_lifted.zarr")] for p in Path(out_dir).glob("*_lifted.zarr"))
    if not stems:
        return None
    if len(stems) > 1:
        raise ValueError(f"{out_dir} holds several lifted stores {stems} — pass the extractor explicitly")
    return stems[0]
```

The two module constants that carried these suffixes in `compression.py` (`LIFTED_SUFFIX`,
`AE_SUFFIX`) do not come along. Each was read by exactly the functions above, which are the only
code allowed to know the convention in the first place; a constant read from one place is a second
name for the same fact.

Delete the corresponding block from `compression.py` (`LIFTED_SUFFIX` through `find_lifted_extractor`,
plus its `Optional` import if unused — `fit`'s signature still uses it, so keep it).

Update the two persistence calls in `utils.py` to the new path-based form. In
`write_point_features`:

```python
        if ae is not None:
            ae.save(ae_path(out_dir, extractor))
```

In `load_point_features` — `weights` is already bound to `ae_path(sem_dir, extractor)` a few lines
up, so the two-line comment about the dir+extractor form goes with it:

```python
    ae = FeatureAutoencoder.load(weights)
```

- [ ] **Step 5: Run the compression tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_compression.py -q
```

PASS, 8 tests (3 shape, 2 training, 3 persistence). The plan originally said 10 (4 shape, 4 persistence); that miscounted its own edit — Step 1 deletes `test_decode_spatial_shape` (4 shape → 3) and the persistence block it supplies defines 3 tests, not 4. 12 → 8 is the −4 that Step 10's suite arithmetic depends on.

- [ ] **Step 6: Update `tests/semantics/test_compression_target.py`**

Two edits. `test_metrics_survive_save_load` (lines 77-84):

```python
def test_metrics_survive_save_load(tmp_path):
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.fit(_features(), epochs=2)
    weights = tmp_path / "talk2dino_ae.pt"
    ae.save(weights)
    loaded = FeatureAutoencoder.load(weights)
    assert loaded.recon_cosine == ae.recon_cosine
    assert loaded.recon_mse == ae.recon_mse
    assert loaded.epochs_run == ae.epochs_run
```

`test_load_legacy_checkpoint_without_metrics` (lines 87-104) — the hand-written payload drops
`hidden_dim` and `regularization_kwargs`, which the new `load` no longer reads:

```python
def test_load_legacy_checkpoint_without_metrics(tmp_path):
    """Checkpoints predating the metric keys still load, defaulting to the untrained values."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)

    # Hand-write a pre-metrics payload: the three metric keys are simply absent
    payload = {"input_dim": 32, "latent_dim": 8, "state_dict": ae.state_dict()}
    weights = tmp_path / "talk2dino_ae.pt"
    torch.save(payload, weights)

    loaded = FeatureAutoencoder.load(weights)
    assert loaded.recon_cosine == 0.0
    assert loaded.recon_mse == 0.0
    assert loaded.epochs_run == 0
```

- [ ] **Step 7: Update `tests/semantics/test_artifact_layout.py`**

`test_load_features_and_decode_round_trip` (lines 24 and 31):

```python
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.save(tmp_path / "talk2dino_ae.pt")
```

```python
    decoded = FeatureAutoencoder.load(tmp_path / "talk2dino_ae.pt").per_point_decode(torch.from_numpy(codes))
```

`test_write_point_features_leaves_no_orphan_codes_when_weights_fail` (line 68) — `save` takes one arg now:

```python
    def boom(self, path):
        raise OSError("disk full")
```

- [ ] **Step 8: Update `tests/dashboard/test_pipeline.py`**

The `FeatureAutoencoder.load(sem_dir, "talk2dino")` call becomes path-based. Add the helper import
at the top of the file next to the others:

```python
from collab_splats.semantics.utils import ae_path, write_point_features
```

and at the call site:

```python
    ae = FeatureAutoencoder.load(ae_path(sem_dir, "talk2dino"))
```

Find every remaining two-arg call with:

```bash
grep -rn "\.save(\|FeatureAutoencoder.load(" tests/ collab_splats/ docs/source/tutorials/
```

Expected after this task: no hit passes two arguments to either.

- [ ] **Step 9: Update `tests/dashboard/test_viewer.py:487`**

```python
    FeatureAutoencoder(input_dim=32, latent_dim=8).save(ae_path(sem_dir, "talk2dino"))
```

with `ae_path` imported alongside `load_point_features` in that test's import block:

```python
    from collab_splats.semantics.utils import ae_path, load_point_features
```

- [ ] **Step 10: Run the full suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/test_semantics_logging.py tests/test_cu121_migration.py -q 2>&1 | tail -20
```

Expected: count = Task 1's count, minus 6 (five reg tests plus
`test_decode_spatial_shape`), plus 2 (`test_save_creates_the_parent_dir_...`,
`test_hidden_dim_is_not_a_constructor_arg`) = Task 1 count − 4.

Same two caveats as Task 1 Step 20: pass `--continue-on-collection-errors` and compare against the
measured baseline (8 pre-existing non-passing items, none in semantics), and verify the delta
per file — concurrent sessions move the raw total.

- [ ] **Step 11: Dashboard smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: `SMOKE PASS`, exit 0.

- [ ] **Step 12: Fix nb05's autoencoder paths**

`docs/source/tutorials/05_lifting/semantic_lifting.ipynb` is **broken today**, independently of this
refactor: `AE_MASKCLIP = SEMANTICS_DIR / "maskclip"` is a directory-shaped path with no `.pt`, and
cells 9/15 call `FeatureAutoencoder.load(AE_MASKCLIP)` / `ae_mc.save(AE_MASKCLIP)` with one argument
against the old two-argument signature. The new signature makes those one-argument calls correct —
they just need to point at a real file path.

Use the `NotebookEdit` tool (or a small `json` script) to make three edits:

Cell 1 — add the `ae_path` import beside the existing one:

```python
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import ae_path
```

Cell 3 — replace the two directory-shaped constants:

```python
AE_MASKCLIP = ae_path(SEMANTICS_DIR, "maskclip")
AE_TALK2DINO = ae_path(SEMANTICS_DIR, "talk2dino")
```

Leave `_base.mkdir(parents=True, exist_ok=True)` as-is — `save()` now mkdirs `SEMANTICS_DIR` itself.

Cells 9 and 15 need **no change**: `AE_MASKCLIP.exists()`, `FeatureAutoencoder.load(AE_MASKCLIP)` and
`ae_mc.save(AE_MASKCLIP)` are all correct once the constant is a file path. Same for `AE_TALK2DINO`/`ae_t2d`.

- [ ] **Step 13: Fix nb06's autoencoder paths**

`docs/source/tutorials/06_mesh/splats_mesh.ipynb`, four edits:

Cell 2 (imports) — `ae_path` moved module:

```python
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import ae_path
```

Cell 4 (config):

```python
AE_MASKCLIP = ae_path(_lifted / "semantics", EXTRACTOR)
```

Cell 13, three call sites:

```python
if LIFTED_MASKCLIP.exists() and AE_MASKCLIP.exists():
    # Cache hit: compressed codes + the AE that decodes them back to CLIP space
    ae = FeatureAutoencoder.load(AE_MASKCLIP).to(DEVICE)
```

```python
    ae.save(AE_MASKCLIP)
```

- [ ] **Step 14: Verify both notebooks parse and hold no stale two-arg calls**

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
import json
for nb in ("docs/source/tutorials/05_lifting/semantic_lifting.ipynb",
           "docs/source/tutorials/06_mesh/splats_mesh.ipynb"):
    src = "\n".join("".join(c["source"]) for c in json.load(open(nb))["cells"])
    bad = [ln for ln in src.splitlines()
           if ("ae.save(" in ln or "ae_mc.save(" in ln or "ae_t2d.save(" in ln
               or "FeatureAutoencoder.load(" in ln) and "," in ln.split("(", 1)[1]]
    print(nb, "OK" if not bad else bad)
PY
```

Expected: both lines print `OK`.

- [ ] **Step 15: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/semantics/compression.py collab_splats/semantics/utils.py tests/semantics/test_compression.py tests/semantics/test_compression_target.py tests/semantics/test_artifact_layout.py tests/dashboard/test_pipeline.py tests/dashboard/test_viewer.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/semantics/compression.py collab_splats/semantics/utils.py
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/test_semantics_logging.py tests/test_cu121_migration.py -q 2>&1 | tail -5
```

```bash
git commit --only \
  collab_splats/semantics/compression.py \
  collab_splats/semantics/utils.py \
  tests/semantics/test_compression.py \
  tests/semantics/test_compression_target.py \
  tests/semantics/test_artifact_layout.py \
  tests/dashboard/test_pipeline.py \
  tests/dashboard/test_viewer.py \
  docs/source/tutorials/05_lifting/semantic_lifting.ipynb \
  docs/source/tutorials/06_mesh/splats_mesh.ipynb \
  -m "refactor(semantics): FeatureAutoencoder — drop reg head, lr_scheduler, hidden_dim; path-based save/load

Nothing in the repo passed regularization_kwargs, reg_target, lr_scheduler
or hidden_dim. The reg branch and its five tests go; hidden_dim stays as
the derived max(64, 2 * latent_dim) local it always effectively was.

save()/load() now take the weights FILE path. The old dir+extractor form
mkdir'd whatever it was handed, so passing a filename silently created a
directory of that name; the new form mkdirs the parent instead.

fit() returns None; the metrics dict goes. With ae_path no longer called from
save(), the path helpers move to semantics/utils.py and compression.py
imports torch and nothing else from this package.

nb05's AE paths were already broken (a dir-shaped path, one-arg calls
against a two-arg signature) — both notebooks now use ae_path().

Suite green: <count> passed. SMOKE PASS."
```

---

## Task 3: Hoist `preprocess` into the base; normalization constants to `utils.image`

`preprocess` is byte-identical in all three extractors — same square/max_size branch, same
round-to-patch-multiple, same `self._normalize(T.ToTensor()(img))` tail. Three copies of a
resize routine is three places to fix a resize bug. It moves to `BaseFeatureExtractor`, along
with the `resize_mode` / `image_resolution` / `svd_components` constructor plumbing that feeds it.

The `**kwargs` pass-through goes with it. It existed only to forward `svd_components` up, so
each extractor now names that parameter explicitly — `insid3.py:248` calls
`DINOFeatureExtractor(svd_components=..., device=...)` and must keep working.

`resize_mode` and `image_resolution` are **required** on the base: no base class can guess a
sensible resolution for an arbitrary backbone. Each extractor keeps its own default (dino 800,
maskclip 1024, talk2dino 512) and passes it up.

The four normalization constants move to `collab_splats/utils/image.py` next to `open_image` /
`resize_image`, so `localization/retrieval.py` can stop hand-writing the ImageNet numbers.

**Files:**
- Modify: `collab_splats/utils/image.py` (constants)
- Modify: `collab_splats/semantics/features/base.py` (`__init__` + `preprocess` arrive)
- Modify: `collab_splats/semantics/features/dino.py`, `maskclip.py`, `talk2dino.py`
- Modify: `collab_splats/localization/retrieval.py:73-77`
- Test: `tests/semantics/test_extractor_preprocessing.py`, `tests/semantics/test_positional_debiasing.py`,
  `tests/semantics/test_features.py`, `tests/semantics/test_query_api.py`,
  `tests/test_semantics_logging.py`, `tests/semantics/features/test_extract_from_zarr.py`,
  `tests/utils/test_image.py`

---

- [ ] **Step 1: Write the failing tests**

Append to `tests/semantics/test_extractor_preprocessing.py`:

```python
########################################################################
# Shared preprocess
########################################################################


def test_preprocess_is_defined_once_on_the_base():
    """All three extractors share BaseFeatureExtractor.preprocess — no per-backend copies."""
    from collab_splats.semantics.features.base import BaseFeatureExtractor
    from collab_splats.semantics.features.dino import DINOFeatureExtractor
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    for cls in (DINOFeatureExtractor, MaskCLIPExtractor, Talk2DinoExtractor):
        assert "preprocess" not in vars(cls), f"{cls.__name__} still overrides preprocess"
        assert cls.preprocess is BaseFeatureExtractor.preprocess


def test_each_extractor_keeps_its_own_default_resolution():
    """Hoisting the plumbing must not flatten the per-backbone defaults."""
    import inspect

    from collab_splats.semantics.features.dino import DINOFeatureExtractor
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    defaults = {
        DINOFeatureExtractor: 800,
        MaskCLIPExtractor: 1024,
        Talk2DinoExtractor: 512,
    }
    for cls, expected in defaults.items():
        params = inspect.signature(cls.__init__).parameters
        assert params["image_resolution"].default == expected
        assert params["resize_mode"].default == "max_size"
        assert "kwargs" not in params, f"{cls.__name__} still takes **kwargs"


def test_svd_components_still_reaches_the_base():
    """insid3 constructs DINOFeatureExtractor(svd_components=...) — the path must survive."""
    import inspect

    from collab_splats.semantics.features.dino import DINOFeatureExtractor

    assert inspect.signature(DINOFeatureExtractor.__init__).parameters["svd_components"].default == 500
```

Append to `tests/utils/test_image.py`:

```python
def test_normalization_constants():
    """The stats extractors normalize with live next to the resize helpers that feed them."""
    from collab_splats.utils.image import (
        CLIP_MEAN,
        CLIP_STD,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]
    assert CLIP_MEAN == [0.48145466, 0.4578275, 0.40821073]
    assert CLIP_STD == [0.26862954, 0.26130258, 0.27577711]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_extractor_preprocessing.py tests/utils/test_image.py -q 2>&1 | tail -20
```

Expected: FAIL — `ImportError: cannot import name 'IMAGENET_MEAN'`, and
`AssertionError: DINOFeatureExtractor still overrides preprocess`.

- [ ] **Step 3: Add the constants to `collab_splats/utils/image.py`**

Insert directly under the module docstring, above `open_image`:

```python
########################################################
########## Normalization constants #####################
########################################################

# ImageNet stats — DINOv2/DINOv3 training preprocessing, and DINO-SALAD retrieval
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP stats — from maskclip_onnx/clip.py _transform()
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
```

- [ ] **Step 4: Hoist `__init__` and `preprocess` into `BaseFeatureExtractor`**

In `collab_splats/semantics/features/base.py`, replace the current `__init__` (lines 60-67) with:

```python
    def __init__(self, resize_mode: str, image_resolution: int, svd_components: int = 500) -> None:
        """
        Store the shared preprocessing and debiasing configuration.

        Subclasses must set `self._normalize` (a `T.Normalize`) and `self.patch_size` before
        `preprocess` is called — the stats and stride are backbone-specific.

        Args:
            resize_mode: "max_size" (proportional longest-edge) or "square" (center-crop + resize).
            image_resolution: longest-edge target for "max_size", square side for "square".
            svd_components: top singular vectors kept for the positional subspace. 500 matches
                the INSID3 default (see reference implementation).
        """
        super().__init__()

        self._resize_mode = resize_mode
        self._image_resolution = image_resolution
        self.svd_components = svd_components

        # Caches keyed by (H_p, W_p) so different input resolutions each get their own basis.
        self._pos_basis_cache: dict = {}   # (H_p, W_p) → Tensor(D, K) positional subspace basis
        self._zero_feats_cache: dict = {}  # (H_p, W_p) → Tensor(D, H_p, W_p) zero-image features for viz
```

Then add `preprocess` immediately after the abstract `forward` and before the `name` property:

```python
    def preprocess(self, image) -> torch.Tensor:
        """
        Resize to the configured resolution, round to patch multiples, and normalize.

        Args:
            image: anything `open_image` accepts — path, ndarray, or PIL image.

        Returns:
            (C, H, W) float32 CPU tensor. H and W are multiples of `patch_size`.
        """
        img = open_image(image).convert("RGB")

        if self._resize_mode == "square":
            # Center-crop to square, then resize to target resolution
            w, h = img.size
            crop = min(w, h)
            img = img.crop(
                ((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2)
            )
            img = img.resize((self._image_resolution, self._image_resolution), Image.BILINEAR)
        else:
            # Proportional longest-edge resize
            img = resize_image(img, longest_edge=self._image_resolution)

        # Round H and W to the nearest patch_size multiple
        w, h = img.size
        ph = round(h / self.patch_size) * self.patch_size
        pw = round(w / self.patch_size) * self.patch_size
        if (ph, pw) != (h, w):
            img = img.resize((pw, ph), Image.BILINEAR)

        return self._normalize(T.ToTensor()(img))
```

- [ ] **Step 5: Strip `dino.py`**

Delete the `_IMAGENET_MEAN` / `_IMAGENET_STD` constants block and the whole `preprocess` method.
Replace the import line and the constructor:

```python
from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD
```

```python
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        resize_mode: str = "max_size",
        image_resolution: int = 800,
        device: Optional[str] = None,
        svd_components: int = 500,
    ):
        if device is None:
            device = get_device()
        super().__init__(resize_mode, image_resolution, svd_components)
        self.model_name = model_name

        # Load DINOv2 from HuggingFace and move to device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        # ImageNet normalization — correct stats for DINOv2
        self._normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._device = torch.device(device)
```

`open_image` and `resize_image` are no longer used in this file — drop them from the
`collab_splats.utils.image` import. `Image` (PIL) is also now unused — drop that import line too.

- [ ] **Step 6: Strip `maskclip.py`**

Delete the `_CLIP_MEAN` / `_CLIP_STD` constants block (including its `#####` divider) and the whole
`preprocess` method plus its `# Preprocessing` divider. Update the import and constructor:

```python
from collab_splats.utils.image import CLIP_MEAN, CLIP_STD
```

```python
    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        resize_mode: str = "max_size",
        image_resolution: int = 1024,
        cache_dir: str = TORCH_HOME,
        device: Optional[str] = None,
        svd_components: int = 500,
    ):
        if device is None:
            device = get_device()
        super().__init__(resize_mode, image_resolution, svd_components)

        # Lazy import: maskclip_onnx depends on pkg_resources.packaging, removed in setuptools>=71
        import maskclip_onnx  # noqa: PLC0415

        # Load the MaskCLIP model; discard the library's default preprocess (square crop)
        # since we apply our own transform with correct CLIP normalization stats
        self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
        self._maskclip_onnx = maskclip_onnx

        # Read patch_size before chaining .to().eval() (chained calls return new objects on mocks)
        self.patch_size: int = self.model.visual.patch_size
        self.model = self.model.to(device).eval()
        self._device = torch.device(device)

        # CLIP normalization — matches maskclip_onnx/clip.py _transform() stats
        self._normalize = T.Normalize(CLIP_MEAN, CLIP_STD)
```

Drop the now-unused `open_image`, `resize_image` and `PIL.Image` imports.

- [ ] **Step 7: Strip `talk2dino.py` and collapse `patch_size`**

Replace the 12-line class docstring with the spec's short form:

```python
@BaseQueryableExtractor.register("talk2dino")
class Talk2DinoExtractor(BaseQueryableExtractor):
    """
    Talk2DINO patch features and text-conditioned similarity, from HuggingFace Hub.

    - Supports DINOv3 ("lorebianchi98/Talk2DINOv3-ViTB", default) and DINOv2 ("lorebianchi98/Talk2DINO-ViTB").
    - forward_features is called directly, bypassing encode_image's internal resize — preprocess()
      has already produced patch-aligned tensors.

    Algorithm from Talk2DINO (https://github.com/lorebianchi98/Talk2DINO).
    """
```

Replace the constructor's body from `super().__init__` through the `patch_size` try-stack:

```python
    def __init__(
        self,
        model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: Optional[str] = None,
        resize_mode: str = "max_size",
        image_resolution: int = 512,
        svd_components: int = 500,
    ):
        """
        Args:
            model_name: HuggingFace Hub model ID.
            device: Torch device string ("cpu" or "cuda").
            resize_mode: "max_size" (proportional, longest-edge) or "square" (center-crop + resize).
            image_resolution: Longest-edge target for "max_size"; square side for "square".
            svd_components: top singular vectors kept for positional debiasing.
        """
        if device is None:
            device = get_device()
        super().__init__(resize_mode, image_resolution, svd_components)

        # low_cpu_mem_usage=False avoids meta-tensor init: Talk2DINO's HF code calls
        # load_state_dict() without assign=True, so weight copies would silently no-op.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            _loaded = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False)

        # Normalize transform from the model's own image_transforms — correct stats per backbone
        self._normalize: T.Normalize = _loaded.image_transforms.transforms[-1]

        # patch_size from the backbone conv stride — the config key is absent on some variants
        self.patch_size: int = _loaded.model.patch_embed.proj.stride[0]

        # Move to device after extracting metadata
        self._model = _loaded.to(device).eval()
        self._device = torch.device(device)
```

Delete the whole `preprocess` method. Drop the now-unused `open_image`, `resize_image` and
`PIL.Image` imports.

- [ ] **Step 8: Point `localization/retrieval.py` at the shared constants**

Add to the import block:

```python
from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD
```

and replace lines 73-77:

```python
        # Build input transform once — resize + normalize to ImageNet stats
        self._transform = T.Compose([
            T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
```

- [ ] **Step 9: Update the talk2dino test fixture**

`tests/semantics/test_extractor_preprocessing.py:187` sets the stride on the *top-level* mock. The
collapsed lookup reads `_loaded.model.patch_embed.proj.stride`, so the value must live on
`mock_model.model` — which is `mock_backbone`. Replace that line with:

```python
    mock_backbone.patch_embed.proj.stride = (patch_size, patch_size)
```

Leave `mock_model.model = mock_backbone` as-is; it is what wires the two together.

- [ ] **Step 10: Update the test fakes that construct extractors**

Five files. Each fake now has to supply the two required base arguments.

`tests/semantics/test_positional_debiasing.py:30-31` — pass concrete values, keep `**kwargs` so
`svd_components=` still reaches the base:

```python
    def __init__(self, feature_dim: int = 16, h_p: int = 4, w_p: int = 4, **kwargs):
        super().__init__("max_size", 512, **kwargs)  # resize/resolution unused: forward is synthetic
```

Same file, `_NoPatchSizeExtractor` (line ~193):

```python
    extractor = _NoPatchSizeExtractor("max_size", 512)
```

`tests/semantics/features/test_extract_from_zarr.py:24-25`:

```python
    def __init__(self, **kwargs):
        super().__init__("max_size", 512, **kwargs)
```

`tests/test_semantics_logging.py` — give `_MockQueryable` a no-arg constructor so its three bare
instantiations keep working:

```python
class _MockQueryable(BaseQueryableExtractor):
    """Minimal queryable extractor — no external deps needed."""

    def __init__(self):
        super().__init__("max_size", 512)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return torch.randn(len(texts), 64)

    def forward(self, images: list) -> list:
        return [torch.randn(64, 8, 8) for _ in images]
```

`tests/semantics/test_query_api.py:29` — same treatment for `_ConcreteExtractor`:

```python
    def __init__(self):
        super().__init__("max_size", 512)
```

`tests/semantics/test_features.py:11` — same treatment for `_FakeExtractor`:

```python
    def __init__(self):
        super().__init__("max_size", 512)
```

- [ ] **Step 11: Run the semantics tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/test_semantics_logging.py -q 2>&1 | tail -20
```

Expected: PASS. If `test_talk2dino_*` fails with `TypeError: 'MagicMock' object cannot be
interpreted as an integer`, Step 9 was applied to the wrong mock — the stride belongs on
`mock_backbone`, not `mock_model`.

- [ ] **Step 12: Confirm the three extractors still import and register**

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.semantics.features import BaseFeatureExtractor
print(sorted(BaseFeatureExtractor._registry))
"
```

Expected: a list containing `dinov2`, `maskclip`, `talk2dino`.

- [ ] **Step 13: Run the full suite and the smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/localization tests/test_semantics_logging.py tests/test_cu121_migration.py -q --continue-on-collection-errors 2>&1 | tail -10
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: PASS, and `SMOKE PASS`.

- [ ] **Step 14: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/utils/image.py collab_splats/semantics/features/base.py collab_splats/semantics/features/dino.py collab_splats/semantics/features/maskclip.py collab_splats/semantics/features/talk2dino.py collab_splats/localization/retrieval.py tests/semantics/test_extractor_preprocessing.py tests/semantics/test_positional_debiasing.py tests/semantics/test_features.py tests/semantics/test_query_api.py tests/semantics/features/test_extract_from_zarr.py tests/test_semantics_logging.py tests/utils/test_image.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/utils/image.py collab_splats/semantics/features/base.py collab_splats/semantics/features/dino.py collab_splats/semantics/features/maskclip.py collab_splats/semantics/features/talk2dino.py collab_splats/localization/retrieval.py
```

```bash
git commit --only \
  collab_splats/utils/image.py \
  collab_splats/semantics/features/base.py \
  collab_splats/semantics/features/dino.py \
  collab_splats/semantics/features/maskclip.py \
  collab_splats/semantics/features/talk2dino.py \
  collab_splats/localization/retrieval.py \
  tests/semantics/test_extractor_preprocessing.py \
  tests/semantics/test_positional_debiasing.py \
  tests/semantics/test_features.py \
  tests/semantics/test_query_api.py \
  tests/semantics/features/test_extract_from_zarr.py \
  tests/test_semantics_logging.py \
  tests/utils/test_image.py \
  -m "refactor(semantics): hoist preprocess into BaseFeatureExtractor; constants to utils.image

preprocess() was byte-identical in dino, maskclip and talk2dino — same
square/max_size branch, same round-to-patch-multiple, same normalize
tail. One copy now lives on the base, together with the resize_mode /
image_resolution plumbing that feeds it. Each extractor keeps its own
default resolution (800 / 1024 / 512).

**kwargs pass-through dropped: it only ever forwarded svd_components,
which is now an explicit parameter on all three.

talk2dino's three-deep try/except for patch_size collapses to the one
path that exists on real checkpoints; the test fixture was mocking the
fallback path, not the real one.

ImageNet and CLIP stats move to utils.image, so localization/retrieval.py
stops hand-writing the ImageNet numbers.

Suite green: <count> passed. SMOKE PASS."
```

---

## Task 4: Drop dead code and the shim; imports to the top; `tokens_to_feature_map`

> **Carry-forwards from the Task 1 code-quality review.** Fold these into this task's steps — they
> are the same kind of work it already does, and they were deferred here to keep Task 1's fix round
> from colliding with Task 2's worktree.
>
> - **`__all__` in `semantics/utils.py` is an incomplete surface.** It lists 10 names but omits
>   `interpolate_to_patch_size`, which `semantics/__init__.py:29` imports from this very module, plus
>   the `torch_utils` re-exports. Since this task deletes both of those, finish the job: rebuild
>   `__all__` to match what actually survives, or drop it. A half-list that contradicts the package's
>   own `__init__` is worse than none.
> - **Two function-local imports in tests, neither a heavy optional dep** — this task's "imports to
>   the top" rule covers them: `tests/semantics/test_artifact_layout.py:79-81` (`import numpy as np`
>   shadowing the module-level import at `:3`, and a `from collab_splats.semantics.utils import
>   lifted_store_path` when line 9 already imports from that module), and
>   `tests/dashboard/test_viewer_lift.py:227` (`from collab_splats.dashboard import viewer as
>   viewer_mod` when `lift_point_features` is imported at line 9).
> - **`tests/dashboard/test_pipeline.py:369-375` asserts nothing useful.**
>   `test_extract_semantics_returns_nothing` appends `(frames, out)` into `calls` then asserts only
>   `len(calls) == 1`, so `extract_feature_cache(anything, anything, anything)` passes. Its sibling in
>   `tests/wrapper/test_reconstructor.py` asserts the full tuple — match it.
> - **Four tests of `semantics.utils` live under `tests/dashboard/test_pipeline.py`**
>   (`:207`, `:210-239`, `:338-360`), reached through the `pl.` alias. `tests/` no longer mirrors
>   `collab_splats/` for this code. Relocate them to `tests/semantics/test_semantics_utils.py`. Their
>   bodies are good — real zarr stores, real round-trips — so this is a move, not a rewrite.

The cleanup pass. Nothing here changes behaviour — it removes what Tasks 1-3 orphaned plus a
few things that were already dead:

- The six-name `torch_utils` re-export shim in `semantics/utils.py`. It exists only so old call
  sites keep working; the two real ones (`mobile_sam.py`, the tests) import through it. Point them
  at `collab_splats.utils.torch_utils` and the shim goes.
- `interpolate_to_patch_size` — imported by `features/base.py` and re-exported, called by nothing.
- `_tokens_to_feature_map` → `tokens_to_feature_map`. Three modules outside `utils.py` import it,
  so the underscore is a lie about its visibility. Body unchanged.
- `_FALLBACK_MEM_GB` on `BaseFeatureExtractor` — a class attribute nothing reads.
- The duplicate `@abstractmethod forward` on `BaseQueryableExtractor`, re-declaring what it already
  inherits from `BaseFeatureExtractor`.
- Inline imports in `insid3.py` (two) and the stale path comment on its line 1.
- `SAM3Segmentation`'s unused `device` argument, and `segment_with_text`'s `confidence_threshold`,
  which its own docstring documents as "Unused".

**Files:**
- Modify: `collab_splats/semantics/utils.py`, `collab_splats/semantics/__init__.py`,
  `collab_splats/semantics/features/base.py`, `features/__init__.py`,
  `features/dino.py`, `features/maskclip.py`, `features/talk2dino.py`
- Modify: `collab_splats/semantics/segmentation/insid3.py`, `mobile_sam.py`, `base.py`, `sam3.py`
- Create: `tests/utils/test_torch_utils.py`
- Test: `tests/semantics/test_semantics_utils.py`, `tests/semantics/test_extractor_preprocessing.py`,
  `tests/semantics/test_features_guards.py`, `tests/semantics/test_positional_debiasing.py`,
  `tests/semantics/test_segmentation.py`

---

- [ ] **Step 1: Create `tests/utils/test_torch_utils.py` with the seven moved tests**

The helpers live in `collab_splats/utils/torch_utils.py`; their tests were sitting in
`tests/semantics/test_semantics_utils.py` because that is where the shim was. Full new file:

```python
"""Tests for collab_splats.utils.torch_utils — device, batching, and GC helpers."""

import pytest
import torch

from collab_splats.utils.torch_utils import batch_iterator, infer_batch_size, pytorch_gc


def test_pytorch_gc_no_error():
    pytorch_gc()


def test_infer_batch_size_cpu():
    if not torch.cuda.is_available():
        assert infer_batch_size(3.0) == 1


def test_infer_batch_size_negative_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(-1.0)


def test_infer_batch_size_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(0.0)


def test_batch_iterator_basic():
    items = list(range(10))
    batches = list(batch_iterator(3, items))
    assert len(batches) == 4
    assert batches[0] == [[0, 1, 2]]
    assert batches[-1] == [[9]]


def test_batch_iterator_multiple_args():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    batches = list(batch_iterator(2, a, b))
    assert len(batches) == 2
    assert batches[0] == [[1, 2], [5, 6]]


def test_batch_iterator_mismatched_raises():
    with pytest.raises(AssertionError):
        list(batch_iterator(2, [1, 2], [3]))
```

- [ ] **Step 2: Run it**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/utils/test_torch_utils.py -q
```

Expected: PASS, 7 tests. They pass immediately — the functions already live there, only the tests
were in the wrong place.

- [ ] **Step 3: Write the failing test for the removals**

Append to `tests/semantics/test_semantics_utils.py`:

```python
def test_utils_no_longer_re_exports_torch_helpers():
    """The back-compat shim is gone — torch helpers come from collab_splats.utils.torch_utils."""
    import collab_splats.semantics.utils as su

    for name in (
        "get_device",
        "pytorch_gc",
        "infer_batch_size",
        "load_hf_weights",
        "load_torchhub_model",
        "interpolate_to_patch_size",
    ):
        assert not hasattr(su, name), f"semantics.utils still exposes {name}"

    # batch_iterator cannot use the hasattr check: Task 1's load_point_features imports it at
    # module level for the streamed decode, so the name is necessarily bound. __all__ is the
    # public-surface claim, and that is what must not list it.
    assert "batch_iterator" not in su.__all__


def test_tokens_to_feature_map_is_public():
    """Three modules import it — the leading underscore was a lie about its visibility."""
    from collab_splats.semantics.utils import tokens_to_feature_map

    assert callable(tokens_to_feature_map)
```

- [ ] **Step 4: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_semantics_utils.py -q 2>&1 | tail -10
```

Expected: FAIL — `AssertionError: semantics.utils still exposes get_device` and
`ImportError: cannot import name 'tokens_to_feature_map'`.

- [ ] **Step 5: Clean up `collab_splats/semantics/utils.py`**

Three edits:

1. Delete the whole temporary shim block (the `from collab_splats.utils.torch_utils import (...)`
   re-export added in Task 1 Step 4). Note the parent's `__all__` lists none of the torch
helpers and the shim block itself holds five names, not six — do not go looking for six.
2. Delete `interpolate_to_patch_size` entirely, and its `__all__` entry.
3. Rename `_tokens_to_feature_map` to `tokens_to_feature_map` and add it to `__all__`. The body
   does not change.

`utils.py` still needs `get_device` internally? No — `extract_feature_cache` calls
`extractor.forward` and never touches devices. If pyflakes reports anything unused after this
step, delete it.

- [ ] **Step 6: Delete the two interpolate tests**

In `tests/semantics/test_semantics_utils.py`, delete `test_interpolate_divisible` and
`test_interpolate_non_divisible` (they were the only callers of the function). Also delete the six
torch_utils tests now living in `tests/utils/test_torch_utils.py`, and rewrite the import block:

```python
"""Tests for collab_splats.semantics.utils — contrastive scoring and artifact layout."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from collab_splats.semantics.utils import (
    cache_store_path,
    compute_semantic_contrast,
    load_feature_maps,
)
```

The store-selection tests added in Task 1 use this same top block (they have no imports further down the file), so Path, numpy, zarr,
cache_store_path and load_feature_maps must all survive here — leave them.

- [ ] **Step 7: Retarget `tokens_to_feature_map` in the three extractors**

`dino.py`, `maskclip.py`, `talk2dino.py` — the import and the one call site in each:

```python
from collab_splats.semantics.utils import get_device, tokens_to_feature_map
```

```python
                tokens_to_feature_map(tokens_all[i].cpu(), H, W, self.patch_size)
```

Note `get_device` moves too: it now comes from `collab_splats.utils.torch_utils`. Each extractor's
import block becomes:

```python
from collab_splats.semantics.utils import tokens_to_feature_map
from collab_splats.utils.torch_utils import get_device
```

`talk2dino.py:155` writes the call on one line — same rename, same arguments.

- [ ] **Step 8: Retarget `tests/semantics/test_extractor_preprocessing.py`**

Line 7 and the three call sites in `test_tokens_to_feature_map_shape`,
`test_tokens_to_feature_map_l2_normalized`, `test_tokens_to_feature_map_wrong_count_raises`:

```python
from collab_splats.semantics.utils import tokens_to_feature_map
```

```python
    out = tokens_to_feature_map(tokens, H, W, patch_size)
```

```python
        tokens_to_feature_map(torch.randn(99, 8), 196, 196, 14)
```

- [ ] **Step 9: Retarget `tests/semantics/test_features_guards.py`**

Line 8 currently pulls `pytorch_gc` through the shim:

```python
from collab_splats.semantics.utils import compute_semantic_contrast
from collab_splats.utils.torch_utils import pytorch_gc
```

- [ ] **Step 10: Trim `collab_splats/semantics/features/base.py`**

Final import block — `Tuple`, `AutoModel`, `get_device` and `interpolate_to_patch_size` are all
dead now:

```python
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from collab_splats.semantics.utils import compute_semantic_contrast
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.torch_utils import RegistryMixin
```

`Path` and `Union` must stay: Task 3 hoisted `preprocess` here with the signature
`Union[str, Path, np.ndarray, Image.Image]`, and this module has no
`from __future__ import annotations`, so the annotation is evaluated at def time — dropping either
name is a `NameError` at import, not a lint nit.

Delete the `_FALLBACK_MEM_GB: float = 2.0` class attribute (line 56 today) — nothing reads it.

Delete the duplicate abstract `forward` on `BaseQueryableExtractor` (line 330 today; work from
content, the line numbers in this plan predate Tasks 1-3). The
class already inherits the abstract `forward` from `BaseFeatureExtractor`; re-declaring it adds a
second docstring to keep in sync and nothing else.

- [ ] **Step 11: Update `collab_splats/semantics/__init__.py`**

Remove `interpolate_to_patch_size` from the `.utils` import block and from `__all__`. Everything
else stays as Task 1 left it.

- [ ] **Step 12: Update `collab_splats/semantics/features/__init__.py`**

Two names go. `_DEBIAS_VALIDATED` is private and re-exported through a public package
`__init__` — tests import it from here, so point them at the defining module instead. `TORCH_HOME`
is re-exported and never read through this path: its one consumer, `maskclip.py:12`, imports it
from `.base` directly.

```python
"""collab_splats.semantics.features — feature extractor registry and backends."""

from .base import BaseFeatureExtractor, BaseQueryableExtractor
from .dino import DINOFeatureExtractor
from .maskclip import MaskCLIPExtractor
from .talk2dino import Talk2DinoExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
]
```

Verify nothing else reached `TORCH_HOME` through the package:

```bash
grep -rn "TORCH_HOME" collab_splats tests --include=*.py
```

Expected: only `features/base.py` (the definition and its docstring line) and `maskclip.py`.

Then `tests/semantics/test_positional_debiasing.py:14`:

```python
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.semantics.features.base import _DEBIAS_VALIDATED
```

- [ ] **Step 13: Imports to the top in `insid3.py`**

Delete the line-1 path comment `# collab_splats/semantics/segmentation/insid3.py` — the file knows
where it is. Move both inline imports into the top block:

```python
"""INSID3 in-context segmentation backend.

Provides:
  INSID3Segmentation — training-free in-context segmentation via frozen DINOv2 features
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from collab_splats.semantics.features.dino import DINOFeatureExtractor

from .base import BaseSegmentation
```

Delete the `from sklearn.cluster import AgglomerativeClustering` line inside
`_agglomerative_clustering` (line 35 today) and the `from collab_splats.semantics.features.dino
import DINOFeatureExtractor` line inside `INSID3Segmentation.__init__` (line 247 today).

`features` does not import `segmentation`, so the top-level DINO import creates no cycle.

- [ ] **Step 14: Retarget `mobile_sam.py`**

```python
from collab_splats.utils.torch_utils import batch_iterator, load_torchhub_model
```

- [ ] **Step 15: Trim `SAM3Segmentation`**

`device` was never used — the comment on line 50 says so outright ("Sam3Processor handles device
placement internally"). `segment_with_text`'s `confidence_threshold` is documented as "Unused".
Both go, on the subclass and on `BaseSegmentation`.

Replace lines 31-53 of `collab_splats/semantics/segmentation/sam3.py`:

```python
@BaseSegmentation.register("sam3")
class SAM3Segmentation(BaseSegmentation):
    """
    SAM3 text-prompted segmentation backend.

    - facebook/sam3 is a gated model: request access, wait for approval, then `huggingface-cli login`.
    - Device placement is handled inside Sam3Processor — there is no device argument.

    Args:
        confidence_threshold: minimum score for returned masks.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        if build_sam3_image_model is None or Sam3Processor is None:
            raise ImportError(
                "sam3 is not installed. Request access at https://huggingface.co/facebook/sam3, "
                "wait for approval, run `huggingface-cli login`, then install from "
                "https://github.com/facebookresearch/sam3"
            )
        sam3_model = build_sam3_image_model()
        self._processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold)
```

and `segment_with_text` (lines 62-79):

```python
    def segment_with_text(self, image, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Text-prompted segmentation.

        Args:
            image: PIL Image.
            prompt: text prompt, e.g. "red tractor".

        Returns:
            masks (N, 1, H, W) float32, boxes (N, 4) float32, scores (N,) float32.
        """
```

The two-line body below it is unchanged.

In `collab_splats/semantics/segmentation/base.py:45-58`:

```python
    def segment_with_text(self, image, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Text-prompted segmentation → (masks, boxes, scores).

        Args:
            image: PIL Image.
            prompt: text prompt.

        Raises:
            NotImplementedError: for backends without text prompts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text-prompted segmentation. "
            "Use backend='sam3'."
        )
```

`tests/semantics/test_segmentation.py` needs no change: line 54 calls
`seg.segment_with_text(MagicMock(), "a dog")`, line 92 constructs
`SAM3Segmentation(confidence_threshold=0.5)`, and line 96 calls
`seg.segment_with_text(fake_img, "a cat")` — all two-positional-argument forms that still fit.

> **`sam3.py` carries foreign uncommitted edits.** Before staging, run
> `git diff --quiet collab_splats/semantics/segmentation/sam3.py`. Exit 0 means it is clean —
> include it in the commit below. Non-zero means someone else's work is still sitting in that
> file: make the edit, run the tests, but **leave `sam3.py` out of the `git commit --only` list**
> and say so in the handoff. Committing it would sweep their changes in.

- [ ] **Step 16: pyflakes the touched modules**

```bash
/opt/venv/reconstruction/bin/python -m pyflakes \
  collab_splats/semantics/utils.py \
  collab_splats/semantics/compression.py \
  collab_splats/semantics/__init__.py \
  collab_splats/semantics/features/base.py \
  collab_splats/semantics/features/__init__.py \
  collab_splats/semantics/features/dino.py \
  collab_splats/semantics/features/maskclip.py \
  collab_splats/semantics/features/talk2dino.py \
  collab_splats/semantics/segmentation/insid3.py \
  collab_splats/semantics/segmentation/mobile_sam.py \
  collab_splats/semantics/segmentation/sam3.py \
  collab_splats/semantics/segmentation/base.py
```

Expected: no output. Every line printed is an unused import or name — delete it and re-run.
`maskclip.py`'s `import maskclip_onnx  # noqa: PLC0415` stays: it is assigned to
`self._maskclip_onnx`, so pyflakes will not flag it.

- [ ] **Step 17: Confirm nothing still reaches through the deleted shim**

```bash
grep -rn "from collab_splats.semantics.utils import" collab_splats tests docs/source --include=*.py --include=*.ipynb
```

Expected: every hit imports only from the surviving surface — `compute_semantic_contrast`,
`tokens_to_feature_map`, `cache_store_path`, `extract_feature_cache`,
`load_feature_maps`, `write_point_features`, `load_point_features`, `point_features_cached`,
`lifted_store_path`, `ae_path`, `find_lifted_extractor`. No `get_device`, `pytorch_gc`,
`infer_batch_size`, `batch_iterator`, `load_hf_weights`, `load_torchhub_model` or
`interpolate_to_patch_size`.

- [ ] **Step 18: Run the full suite and the smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils tests/dashboard tests/wrapper tests/localization tests/test_semantics_logging.py tests/test_cu121_migration.py -q 2>&1 | tail -10
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: PASS, and `SMOKE PASS`. Net test count: Task 3's count, −2 (interpolate tests),
+2 (the two new guard tests), **+1** (the carry-forward `point_features_cached` regression test
this task folds in — the original arithmetic here omitted it and predicted net 0); the seven
torch_utils tests moved file but were not deleted, and `tests/utils/test_torch_utils.py` adds none
beyond them. Net **+1**: 383 → 384 on the scoped
`tests/semantics tests/utils tests/dashboard tests/test_semantics_logging.py` run.

- [ ] **Step 19: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/semantics collab_splats/utils tests/semantics tests/utils/test_torch_utils.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/semantics collab_splats/utils tests/semantics tests/utils/test_torch_utils.py
```

Note the paths are scoped to the directories this task touched — never `black .` (the venv's
black 26.5.1 is newer than the one the repo was formatted with and would rewrite unrelated files).

```bash
git commit --only \
  collab_splats/semantics/utils.py \
  collab_splats/semantics/__init__.py \
  collab_splats/semantics/features/base.py \
  collab_splats/semantics/features/__init__.py \
  collab_splats/semantics/features/dino.py \
  collab_splats/semantics/features/maskclip.py \
  collab_splats/semantics/features/talk2dino.py \
  collab_splats/semantics/segmentation/insid3.py \
  collab_splats/semantics/segmentation/mobile_sam.py \
  collab_splats/semantics/segmentation/base.py \
  collab_splats/semantics/segmentation/sam3.py \
  tests/utils/test_torch_utils.py \
  tests/semantics/test_semantics_utils.py \
  tests/semantics/test_extractor_preprocessing.py \
  tests/semantics/test_features_guards.py \
  tests/semantics/test_positional_debiasing.py \
  tests/dashboard/test_pipeline.py \
  tests/dashboard/test_viewer_lift.py \
  tests/semantics/test_artifact_layout.py \
  -m "refactor(semantics): drop dead code and shim; imports to top; tokens_to_feature_map

The torch_utils re-export shim in semantics.utils goes: the two real
callers (mobile_sam, the tests) now import from collab_splats.utils.
torch_utils directly. Its tests move to tests/utils/test_torch_utils.py,
where the functions actually live.

Also deleted, all unreferenced: interpolate_to_patch_size,
_FALLBACK_MEM_GB, and BaseQueryableExtractor's duplicate abstract
forward (already inherited from BaseFeatureExtractor).

_tokens_to_feature_map -> tokens_to_feature_map: three modules outside
utils.py import it, so the underscore was wrong about its visibility.
Body unchanged.

insid3's two inline imports move to the top; features does not import
segmentation, so the DINO import creates no cycle.

SAM3Segmentation drops the device argument its own comment called unused
and segment_with_text's confidence_threshold its own docstring called
unused. The gating ImportError now spells out the access flow.

pyflakes clean. Suite green: <count> passed. SMOKE PASS."
```

If Step 15's `git diff --quiet` said `sam3.py` is dirty with foreign work, drop that one path from
the command above and report it as left uncommitted.

---

## Task 5: Docstrings, `docs/semantics.md`, CHANGELOG

> **Carry-forwards from the Task 1 code-quality review.** Both are exactly this task's job.
>
> - **`semantics/utils.py` is stylistically bimodal.** All six functions Task 1 moved in open their
>   docstring on the line *after* `"""`; the three that already lived there do not —
>   `compute_semantic_contrast`, `_tokens_to_feature_map` (renamed in Task 4), and
>   `interpolate_to_patch_size` (deleted in Task 4, so only the first two need the fix). Bring them to
>   the convention.
> - **`collab_splats/dashboard/viewer.py:39` and `:74` carry a number that contradicts two other
>   records.** The comments say the semantics import is "~20s of the app's import"; Task 1's commit
>   body measures the app at 4.18 s → 3.63 s, and the review measured the import itself at 10.3 s
>   (45.5 s cold). Whatever the true figure, the phrasing asserts something the other records deny.
>   Re-measure and write down what you observe, or drop the number and keep the reason.

The last commit is prose. Tasks 1-4 wrote every new or rewritten function in the spec's docstring
convention already; this task brings the untouched survivors up to it and replaces the module doc,
which describes an API that has not existed for a long time.

**The convention (spec §10):**

- `"""` alone on the opening line, `"""` alone on the closing line. One-line summary on the line
  after the opening quotes. Blank line. `Args:` bullets with type or shape. `Returns:` bullets with
  shape. `Raises:` when the function raises. Nothing else in a docstring.
- Rationale is a one-line block comment at the site, not a paragraph in the docstring.
- Restating the function name is not a summary.

Before / after:

```python
    def features_to_rgb(feat):
        """Project a feature map onto its top-3 PCs to produce an RGB display image.

        Args:
            feat: (D, H_p, W_p) float tensor — output of forward() for one frame.

        Returns:
            np.ndarray of shape (H_p, W_p, 3) dtype uint8.
        """
```

```python
    def features_to_rgb(feat):
        """
        Project a feature map onto its top-3 PCs for display.

        Args:
            feat: (D, H_p, W_p) float tensor — one frame's output from forward().

        Returns:
            (H_p, W_p, 3) uint8 ndarray.
        """
```

**Files:**
- Modify: `collab_splats/semantics/features/base.py`, `collab_splats/semantics/segmentation/base.py`,
  `collab_splats/semantics/segmentation/mobile_sam.py`, `collab_splats/semantics/segmentation/insid3.py`
- Rewrite: `docs/semantics.md`
- Modify: `CLAUDE.md` (architecture tree), `docs/superpowers/CHANGELOG.md`

---

- [ ] **Step 1: Reflow the docstrings that only need reflow**

These already carry correct `Args:` / `Returns:` bullets. Move the summary onto its own line below
the opening `"""` and add the blank line after it. Do not change any wording or any code.

| file | function |
|---|---|
| `semantics/segmentation/base.py` | `create_patch_mask`, `create_composite_mask`, `mask_id_to_binary_mask`, `convert_matched_mask`, `aggregate_masked_features` |
| `semantics/segmentation/insid3.py` | `_agglomerative_clustering`, `_cluster_prototypes`, `_upsample_mask`, `_tokens_to_grid`, `_grid_to_image`, `_locate_candidates`, `_seed_and_aggregate`, `INSID3Segmentation.__init__`, `set_context`, `segment`, `segment_with_mask` |
| `semantics/features/base.py` | `_build_positional_basis`, `_apply_debias`, `get_bias_visualization`, `debias`, `compute_similarity`, `score_queries` |

For the four `features/base.py` debias methods, the spec says "docstrings only" — the block
comments inside those bodies stay exactly as they are.

- [ ] **Step 2: Rewrite the docstrings that are missing `Args` / `Returns`**

`collab_splats/semantics/features/base.py` — class docstring, `forward`, `name`, `features_to_rgb`:

```python
class BaseFeatureExtractor(RegistryMixin, nn.Module, ABC):
    """
    Abstract base for image feature extractors with a name-based registry.

    - Register subclasses via `@BaseFeatureExtractor.register("name")`, retrieve with `.get("name")`.
    - Subclasses set `self._normalize` and `self.patch_size`; `preprocess` is shared.
    """
```

```python
    @abstractmethod
    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Preprocess, run inference, and reshape to patch grids.

        Args:
            images: anything `open_image` accepts, one entry per frame.

        Returns:
            One (D, H_p, W_p) tensor per image, on CPU.
        """
        ...
```

```python
    @property
    def name(self) -> str:
        """
        Registry key this extractor was registered under.

        Returns:
            The key string — also the stem of its `.zarr` cache.

        Raises:
            AttributeError: if the class was never registered.
        """
```

```python
    @staticmethod
    def features_to_rgb(feat: "torch.Tensor") -> "np.ndarray":
        """
        Project a feature map onto its top-3 PCs for display.

        Args:
            feat: (D, H_p, W_p) float tensor — one frame's output from forward().

        Returns:
            (H_p, W_p, 3) uint8 ndarray. All-zero when the features have no variance.
        """
```

`BaseQueryableExtractor` class docstring and `encode_text`:

```python
class BaseQueryableExtractor(BaseFeatureExtractor, ABC):
    """
    Feature extractor that also embeds text, for cosine queries against patch features.

    - Subclasses implement `encode_text` and `forward`; `compute_similarity` and
      `score_queries` are shared.
    - All subclasses take `model_name` as their first constructor parameter.
    """
```

```python
    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Embed text into the same space as the patch features.

        Args:
            texts: query strings.

        Returns:
            (N, D) L2-normalized embeddings.
        """
        ...
```

`collab_splats/semantics/segmentation/base.py` — class docstring and the abstract `segment`:

```python
class BaseSegmentation(RegistryMixin, ABC):
    """
    Abstract base for segmentation backends with a name-based registry.

    Register subclasses via `@BaseSegmentation.register("name")`, retrieve with `.get("name")`.
    """
```

```python
    @abstractmethod
    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """
        Class-agnostic segmentation with no prompt.

        Args:
            image: (H, W, 3) uint8 array or PIL Image, per backend.

        Returns:
            (masks, metadata) — masks is (N, H, W) float32; metadata is backend-specific.
        """
```

`collab_splats/semantics/segmentation/mobile_sam.py` — `load_mobile_sam`, the class docstring, and
`segment`, which has no docstring at all today:

```python
def load_mobile_sam(
    mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2", device: str = "cpu"
):
    """
    Load the MobileSAMV2 model trio from torchhub.

    Args:
        mobilesam_encoder_name: encoder variant published by RogerQi/MobileSAMV2.
        device: torch device string for the SAM model.

    Returns:
        (mobilesamv2, ObjAwareModel, predictor) — SAM model, YOLOv8 detector, SAMPredictor.
    """
```

```python
class MobileSAMSegmentation(BaseSegmentation):
    """
    MobileSAMv2 class-agnostic segmentation.

    Args:
        strategy: "object" prompts SAM with YOLOv8 boxes; "auto" runs SAM's mask generator.
        device: torch device string.
        mobilesam_encoder_name: encoder variant to load from torchhub.
    """
```

```python
    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """
        Segment one frame with the configured strategy.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            (masks, metadata) — masks is (N, H, W) float32. None when nothing is detected.

        Raises:
            ValueError: if `strategy` is neither "object" nor "auto".
        """
```

```python
    def _segment_auto(self, image) -> tuple[torch.Tensor, Any] | None:
        """
        SAM's automatic mask generator, no prompts.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            (masks, raw results), or None when the generator returns nothing.
        """
```

```python
    def _segment_object(self, image, batch_size: int = 320) -> tuple[torch.Tensor, Any] | None:
        """
        YOLOv8 boxes prompt SAM once per detected object.

        Args:
            image: (H, W, 3) uint8 array.
            batch_size: boxes per SAM decoder call.

        Returns:
            (masks, raw results), or None when the detector finds no objects.
        """
```

- [ ] **Step 3: Verify nothing broke**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics tests/utils -q 2>&1 | tail -5
```

Expected: PASS. Docstrings are not executable, so a failure here means an edit slipped into code.

- [ ] **Step 4: Rewrite `docs/semantics.md`**

The current file documents extractors named `samclip` and `clip-vit` that are not registered, a
`preprocess(img, resolution=…)` signature that takes no such argument, a `reshape()` method that
does not exist, `compute_semantic_heatmap` which was deleted, a
`collab_splats.semantics.protocols` module that does not exist, a
`Segmentation(backend=…)` constructor that was replaced by the registry, and a Gradio dashboard
that was replaced by the Panel one. Replace the whole file:

````markdown
# Semantics Module

`collab_splats.semantics` extracts patch-level features from keyframes, compresses them, and
lifts them onto the pointcloud. It also wraps three segmentation backends.

---

## On-disk layout

Everything for one scene lives in its `semantics/` directory, keyed by extractor name:

| file | shape | written by |
|---|---|---|
| `<extractor>.zarr` | `(N, D, H_p, W_p)` float32, one chunk per frame | `extract_feature_cache` |
| `<extractor>_lifted.zarr` | `(P, latent)` float32 | `write_point_features` |
| `<extractor>_ae.pt` | autoencoder checkpoint | `write_point_features` |

The `_lifted` suffix is load-bearing: the 2D cache and the per-point codes share a directory, so
the filename is the only thing distinguishing them.

---

## Feature extractors

Three are registered: `dinov2`, `maskclip`, `talk2dino`.

```python
from collab_splats.semantics.features import BaseFeatureExtractor

extractor = BaseFeatureExtractor.get("dinov2")(device="cuda")

features = extractor.forward([img])      # list of (D, H_p, W_p) CPU tensors
rgb = extractor.features_to_rgb(features[0])   # (H_p, W_p, 3) uint8, PCA to RGB
```

All three share `preprocess` from the base class: `resize_mode="max_size"` does a proportional
longest-edge resize, `"square"` center-crops first; either way the result is rounded to a multiple
of `patch_size`. Defaults are 800 (dinov2), 1024 (maskclip), 512 (talk2dino).

| extractor | backbone | width |
|---|---|---|
| `dinov2` | `facebook/dinov2-small` via transformers | 384 |
| `maskclip` | `ViT-L/14@336px` via `maskclip_onnx` | 768 |
| `talk2dino` | `lorebianchi98/Talk2DINOv3-ViTB` | 768 |

### Text queries

`maskclip` and `talk2dino` are `BaseQueryableExtractor`s — they embed text into the same space:

```python
extractor = BaseFeatureExtractor.get("talk2dino")(device="cuda")
feats = extractor.forward([img])[0]              # (D, H_p, W_p)

scores = extractor.score_queries(
    feats,
    positive=["tree", "trunk", "branch"],
    negative=["sky", "ground"],
    temperature=0.05,     # lower = sharper
    reduction="max",      # "max" or "mean" across the positives
)  # → (H_p, W_p) in [0, 1]
```

`score_queries` also accepts a `(P, D)` point array and returns `(P,)` — the same call works on
lifted features.

### Positional debiasing

DINO-family features carry a positional component that survives into cosine similarity.
`debias` projects it out, estimating the subspace from a black image at the same patch grid:

```python
clean = extractor.debias(features)      # same shapes, L2-renormalized
viz = extractor.get_bias_visualization(H_p, W_p)   # what was removed, as RGB
```

Validated for DINOv2 and Talk2DINO. Other extractors log a warning and proceed.

---

## Extract → lift → store

```python
from collab_splats.semantics.utils import (
    cache_store_path,
    extract_feature_cache,
    load_feature_maps,
    load_point_features,
    write_point_features,
)

# 1. 2D patch cache, one chunk per frame, skipped if it already exists
store = extract_feature_cache(extractor, scene / "frames.zarr", scene / "semantics")

# 2. read it back for lifting
feature_maps = load_feature_maps(store)          # list of (D, H_p, W_p) tensors

# 3. after lifting + compressing, write both halves together
write_point_features(scene / "semantics", "talk2dino", codes, ae)

# 4. read them back, decoded to full width
feats = load_point_features(scene / "semantics")  # (P, D) float32
```

`write_point_features` writes the codes and the autoencoder as a pair and deletes the codes if
the weights fail to save — a lifted store with no decoder is unreadable, so a partial write is
worse than no write.

`cache_store_path(semantics_dir)` finds the 2D cache and ignores `_lifted` stores. Its `.stem` is
the extractor name — that is how `write_point_features` learns which name to write under.

---

## FeatureAutoencoder

Compresses ViT-width features to the latent width stored per point.

```python
from collab_splats.semantics.compression import FeatureAutoencoder

ae = FeatureAutoencoder(input_dim=768, latent_dim=32)
ae.fit(features, epochs=10, target_cosine=0.9)
ae.recon_cosine, ae.recon_mse, ae.epochs_run    # fit quality, also written into the checkpoint

codes = ae.per_point_encode(points)     # (P, 768) → (P, 32)
back = ae.per_point_decode(codes)       # (P, 32)  → (P, 768)

maps = ae.encode(feature_map)           # (768, H_p, W_p) → (32, H_p, W_p)

ae.save(path)                           # a .pt FILE path; parent dirs created
ae = FeatureAutoencoder.load(path)
```

Compression works on patch maps or points; decompression only on points — the pipeline encodes
maps, lifts the latent maps to 3D, and decodes per point on read.

`fit` measures cosine on the training set, so `recon_cosine` is optimistic at small `N`. Check
`epochs_run > 0` before trusting it at all.

---

## Segmentation

Three backends, same registry pattern:

```python
from collab_splats.semantics.segmentation import BaseSegmentation

seg = BaseSegmentation.get("mobilesamv2")(strategy="object", device="cuda")
masks, metadata = seg.segment(frame)     # masks: (N, H, W) float32
```

| name | what it does | notes |
|---|---|---|
| `mobilesamv2` | class-agnostic masks | `strategy="object"` (YOLOv8 boxes → SAM) or `"auto"` |
| `insid3` | in-context segmentation from one annotated reference | training-free, frozen DINOv2 features |
| `sam3` | text-prompted masks | gated model, see below |

### Text-prompted segmentation

```python
seg = BaseSegmentation.get("sam3")(confidence_threshold=0.5)
masks, boxes, scores = seg.segment_with_text(img, "red tractor")
```

Backends without text support raise `NotImplementedError`.

`sam3` is a gated Meta model. To use it:

1. Request access at <https://huggingface.co/facebook/sam3> and wait for approval.
2. `huggingface-cli login`
3. Install from <https://github.com/facebookresearch/sam3>.

Until then, constructing `SAM3Segmentation` raises `ImportError` with those steps.

---

## Dashboard

The semantics tabs live in the Panel dashboard:

```bash
python -m collab_splats.dashboard
```
````

- [ ] **Step 5: Update the architecture tree in `CLAUDE.md`**

Replace the `semantics/` block:

```
  semantics/               # 2D feature extraction + on-disk artifact layout
    utils.py               # semantics/<extractor>.zarr + _lifted.zarr: read, write, discover
    features/              # BaseFeatureExtractor + RegistryMixin (base.py); shared preprocess;
                           #   registered dinov2, maskclip, talk2dino — ViT-width (384-1024D)
    compression.py         # FeatureAutoencoder
    segmentation/          # BaseSegmentation; registered insid3, mobilesamv2, sam3
```

The `compression.py` comment loses "per-point encode/decode + recon_cosine" — that was a method
list, and the tree is for orientation, not API reference.

> **`CLAUDE.md` carries foreign uncommitted edits** (it is dirty in the branch's status). Run
> `git diff --quiet CLAUDE.md` first. Exit 0 → include it in the commit. Non-zero → make the edit,
> but leave `CLAUDE.md` out of the `git commit --only` list and say so in the handoff. The
> `.claude/hooks/claude-md-guard.py` PreToolUse hook also enforces the 40,000-character ceiling;
> this edit adds one line, so it stays well under.

- [ ] **Step 6: Append the CHANGELOG entry**

Add to the top of the entries in `docs/superpowers/CHANGELOG.md`:

```markdown
## 2026-09-05 — semantics cleanup

Artifact I/O folded into `semantics/utils.py`. `extract_and_cache` and
`extract_and_cache_from_zarr` were methods on `BaseFeatureExtractor` that did no
extraction of their own; they are now `extract_feature_cache(extractor, ...)`, so
`utils` never imports `features` and the loader has one implementation instead of
three (pipeline, viewer, reconstructor each had their own).

`FeatureAutoencoder` lost its regularization head, `lr_scheduler` and `hidden_dim`
— nothing in the repo passed any of them. `save`/`load` take a weights file path;
the old dir+extractor form mkdir'd whatever it was handed, so a filename silently
became a directory. `fit` returns its metrics.

`preprocess` was byte-identical in all three extractors and now lives on the base,
along with the `resize_mode` / `image_resolution` plumbing. ImageNet and CLIP
normalization constants moved to `collab_splats/utils/image.py`.

Deleted: the `torch_utils` re-export shim in `semantics.utils`,
`interpolate_to_patch_size`, `_FALLBACK_MEM_GB`, `BaseQueryableExtractor`'s
duplicate abstract `forward`, `SAM3Segmentation`'s unused `device` argument, and
`segment_with_text`'s unused `confidence_threshold`. `_tokens_to_feature_map` is
now public. `docs/semantics.md` rewritten — it described `samclip`,
`compute_semantic_heatmap`, `semantics.protocols` and a Gradio app, none of which
exist.

Spec: `docs/superpowers/specs/2026-09-05-semantics-cleanup-design.md`
Plan: `docs/superpowers/plans/2026-09-05-semantics-cleanup.md`
```

Then remove nothing from `CLAUDE.md`'s In-Flight Work list — this work was never listed there.

- [ ] **Step 7: Refresh the knowledge graph**

```bash
graphify update .
```

Expected: completes without error. AST-only, no API cost.

- [ ] **Step 8: Full suite and smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q --continue-on-collection-errors 2>&1 | tail -15
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the whole suite passes except the entries already listed in
`docs/known-test-failures.md`, and `SMOKE PASS`.

- [ ] **Step 9: Commit**

```bash
git commit --only \
  collab_splats/semantics/features/base.py \
  collab_splats/semantics/segmentation/base.py \
  collab_splats/semantics/segmentation/mobile_sam.py \
  collab_splats/semantics/segmentation/insid3.py \
  docs/semantics.md \
  docs/superpowers/CHANGELOG.md \
  CLAUDE.md \
  -m "docs(semantics): docstrings to Args/Returns; rewrite docs/semantics.md

Every surviving docstring in the package now opens and closes on its own
line with Args/Returns bullets. MobileSAMSegmentation.segment had no
docstring at all.

docs/semantics.md described an API that has not existed for a long time:
extractors named samclip and clip-vit that are not registered, a
preprocess(img, resolution=...) signature, a reshape() method,
compute_semantic_heatmap, a collab_splats.semantics.protocols module,
and the Gradio dashboard. Rewritten to the code: the on-disk layout, the
three extractors, extract -> lift -> store, the autoencoder, the three
segmentation backends, and the sam3 gating steps.

Suite green. SMOKE PASS."
```

If Step 5's `git diff --quiet` said `CLAUDE.md` is dirty with foreign work, drop that path from the
command and report it as left uncommitted.

Note `docs/superpowers/` is gitignored in this repo — if git refuses the CHANGELOG path, re-run
that one with `git add -f docs/superpowers/CHANGELOG.md` before committing.
