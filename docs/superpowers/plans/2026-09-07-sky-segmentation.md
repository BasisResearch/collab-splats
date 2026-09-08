# Sky Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port VGGT's skyseg ONNX model as a registered `BaseSegmentation` backend, consume it in the mesh stage behind `mesh.mask_sky`, and measure whether masking sky improves the fused mesh.

**Architecture:** A `SkySegmentation` backend in `collab_splats/semantics/segmentation/sky.py` does per-image inference (matching every other backend); a module-level `sky_masks()` beside it carries the pipeline concerns — an on-disk PNG cache (632 ms/frame earns it) and a frame-ordering contract that mirrors `frames.read_frames`. `_run_tsdf_mesh` applies the mask at frame resolution after the two source arms converge, which is why `render_tsdf_inputs` must start returning `image_ids`.

**Tech Stack:** onnxruntime 1.26.0 (CPU-only here), OpenCV, NumPy, Open3D, pytest.

Spec: `docs/superpowers/specs/2026-09-07-sky-segmentation-design.md`
Worktree: `/workspace/collab-splats/.worktrees/sky-mask`, branch `feat/sky-segmentation`, forked at `clean/final`.

---

## Before you start

**The worktree PYTHONPATH trap will give you a false green.** The venv's editable finder hardcodes `/workspace/collab-splats`, so a bare `pytest` inside this worktree tests the MAIN tree. `PYTHONPATH` alone is not enough — `sys.path[0]` is cwd and cwd resets between shell calls. Every test command in this plan is written as one chained invocation. Do not shorten them.

Prove it once before Task 1:

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
```

Expected: `/workspace/collab-splats/.worktrees/sky-mask/collab_splats/__init__.py`
If it prints anything else, stop and fix it — every result after this point is meaningless otherwise.

**Two deviations from the spec, decided while mapping the code. Both are deliberate.**

1. **`segment` returns `(H, W)` bool, not `(1, H, W)`.** `BaseSegmentation.segment`'s docstring enumerates per-backend mask ranks, and insid3 — the closest analogue, one semantic mask — already returns `(H, W)` bool. Matching it avoids a squeeze at every call site. Task 3 adds skyseg to that docstring enumeration.
2. **The mesh-wiring tests live in a new `tests/wrapper/test_mask_sky.py`, not in
   `tests/wrapper/test_reconstructor.py` as the spec says.** That file is already ~1400
   lines covering config validation and stage orchestration; a fourth concern in it helps
   nobody, and `tests/wrapper/` is already one-file-per-seam
   (`test_absent_confidence.py`, `test_refine_stage.py`, `test_verify_stage.py`).
3. **No bool validation of `mesh.mask_sky`.** The spec said "bool-validated with the other mesh keys". There are no bool-validated mesh keys — `texture: false` has none, and only `source` is checked (it is a string with two legal values). Adding validation for this one key alone would be the odd thing out. YAGNI.

**Task 5 has a blast radius.** Adding a fifth return to `render_tsdf_inputs` breaks four tests, one eval script and one notebook. All six sites are listed in that task. Do not skip any.

---

### Task 1: Promote `PNG_COMPRESSION` to public

`sky_masks` becomes a second PNG writer and needs the same compression level. The
justification comment ("level 9 costs 10x the time for 11% of the size") lives in
`frames.py`; duplicating the literal would orphan it.

**Files:**
- Modify: `collab_splats/preproc/frames.py:36-37,138`

- [ ] **Step 1: Rename the constant and its one use**

In `collab_splats/preproc/frames.py`, replace lines 36-37:

```python
# OpenCV's default. Level 9 costs 10x the time for 11% of the size (measured, spec 2.2).
_PNG_COMPRESSION = 1
```

with:

```python
# OpenCV's default. Level 9 costs 10x the time for 11% of the size (measured, spec 2.2).
# Public because the sky-mask cache writes PNGs at the same level and must not fork it.
PNG_COMPRESSION = 1
```

Then in `write_frames`, change the single use (line 138):

```python
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
```

- [ ] **Step 2: Verify no other reference survives**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && grep -rn "_PNG_COMPRESSION" collab_splats/ tests/
```

Expected: no output (exit 1 from grep is correct here).

- [ ] **Step 3: Run the preproc frames tests**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q
```

Expected: PASS, no failures.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
git add collab_splats/preproc/frames.py
git commit -m "refactor(preproc): make PNG_COMPRESSION public for the sky-mask cache"
```

---

### Task 2: `SkySegmentation` backend

**Files:**
- Create: `collab_splats/semantics/segmentation/sky.py`
- Test: `tests/semantics/test_sky_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/semantics/test_sky_segmentation.py`:

```python
"""
Sky segmentation backend and mask cache.

- polarity: the model emits HIGH values for sky, inverted from upstream's stored 255-is-not-sky
- the per-image min-max rescale upstream applies is deliberately dropped
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from collab_splats.semantics.segmentation import sky

######## Fakes


class _FakeSession:
    """
    Stand-in for onnxruntime.InferenceSession returning one canned probability map.
    """

    def __init__(self, prob):
        self._prob = prob.astype(np.float32)
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, output_names, feed):
        self.calls += 1
        return [self._prob[None, None]]


def _backend(monkeypatch, prob, threshold=0.5):
    """
    A SkySegmentation wired to a fake session; no download, no ONNX.
    """
    session = _FakeSession(prob)
    monkeypatch.setattr(sky, "hf_hub_download", lambda **kwargs: "skyseg.onnx")
    monkeypatch.setattr(sky.ort, "InferenceSession", lambda path, providers=None: session)
    return sky.SkySegmentation(threshold=threshold), session


######## Polarity


def test_high_probability_means_sky(monkeypatch):
    # Top half of the model grid is sky; upstream's comment claims the opposite of this
    prob = np.zeros((320, 320), np.float32)
    prob[:160] = 0.9
    backend, _ = _backend(monkeypatch, prob)

    mask, meta = backend.segment(np.zeros((64, 96, 3), np.uint8))

    assert mask.shape == (64, 96)
    assert mask.dtype == torch.bool
    assert mask[:30].all()
    assert not mask[34:].any()
    assert meta["raw"].shape == (64, 96)
    assert meta["raw"].dtype == np.float32


def test_threshold_is_a_probability(monkeypatch):
    # 0.125 is upstream's 32/255; a uniform 0.2 map is sky under it and not under 0.5
    prob = np.full((320, 320), 0.2, np.float32)

    loose, _ = _backend(monkeypatch, prob, threshold=0.125)
    assert loose.segment(np.zeros((16, 16, 3), np.uint8))[0].all()

    strict, _ = _backend(monkeypatch, prob, threshold=0.5)
    assert not strict.segment(np.zeros((16, 16, 3), np.uint8))[0].any()


######## The dropped rescale


def test_a_dim_map_yields_no_sky(monkeypatch):
    # Ramp 0.0 -> 0.3. Upstream's per-image min-max rescale would stretch the top row to
    # 1.0 and manufacture sky from nothing; thresholding the raw probability must not.
    prob = np.linspace(0.0, 0.3, 320, dtype=np.float32)[:, None].repeat(320, axis=1)
    backend, _ = _backend(monkeypatch, prob)

    mask, meta = backend.segment(np.zeros((32, 32, 3), np.uint8))

    assert not mask.any()
    assert meta["raw"].max() <= 0.3 + 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_sky_segmentation.py -q
```

Expected: collection error — `ImportError: cannot import name 'sky' from 'collab_splats.semantics.segmentation'`.

- [ ] **Step 3: Write the implementation**

Create `collab_splats/semantics/segmentation/sky.py`:

```python
"""
Sky segmentation backend over VGGT's skyseg ONNX model.

- SkySegmentation: one binary sky mask per frame, True where sky
- ported from facebookresearch/vggt @ a288dd0f14786c93483e45524328726ab7b1b4ce,
  visual_util.py:365-434 (segment_sky, run_skyseg)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD, open_image

from .base import BaseSegmentation

logger = logging.getLogger(__name__)


########################################################
########## Model constants #############################
########################################################

SKYSEG_REPO = "JianyuanWang/skyseg"
SKYSEG_FILE = "skyseg.onnx"

# The network's fixed square input. Upstream ignores aspect ratio here and so do we —
# the mask is resized back to the frame, and matching upstream keeps the port verifiable.
SKYSEG_SIZE = 320

# CUDA first when the wheel provides it; this container's onnxruntime is CPU-only, and
# naming a provider that is not available raises rather than falling back.
PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


########################################################
########## Sky segmentation backend ####################
########################################################


@BaseSegmentation.register("skyseg")
class SkySegmentation(BaseSegmentation):
    """
    Sky mask from VGGT's skyseg ONNX model.

    - one binary mask per image, unlike the object backends' N masks
    - the raw probability map rides along, so a threshold sweep needs no second forward pass
    - the model output is already sigmoid-terminated; upstream's per-image min-max rescale
      is dropped because it stretches a dim frame's maximum to 1.0 and invents sky

    Args:
        threshold: probability above which a pixel is sky; upstream's 32/255 is ~0.125.
        model_path: local skyseg.onnx; None downloads it from the Hub.
    """

    def __init__(self, threshold: float = 0.5, model_path: Path | str | None = None) -> None:
        path = model_path or hf_hub_download(repo_id=SKYSEG_REPO, filename=SKYSEG_FILE)
        providers = [p for p in PREFERRED_PROVIDERS if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._threshold = threshold

    def segment(self, image: np.ndarray | Image.Image) -> tuple[torch.Tensor, dict]:
        """
        Sky mask for one frame.

        - True is sky, inverted from upstream's stored 255-is-not-sky convention

        Args:
            image: the frame to segment, coerced through `utils.image.open_image`.

        Returns:
            (mask, metadata) — mask (H, W) bool at input resolution; metadata carries
            'raw', the (H, W) float32 probability map.
        """
        rgb = np.asarray(open_image(image).convert("RGB"))
        height, width = rgb.shape[:2]

        # Upstream preprocessing verbatim: square resize, /255, ImageNet normalize, NCHW
        square = cv2.resize(rgb, (SKYSEG_SIZE, SKYSEG_SIZE)).astype(np.float32) / 255.0
        square = (square - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
        tensor = square.transpose(2, 0, 1)[None].astype(np.float32)

        # Output is (1, 1, 320, 320) and already sigmoid-terminated, so it needs no rescale
        prob = np.asarray(self._session.run(None, {self._input_name: tensor})[0])[0, 0]

        # Resize the probability map, then threshold — thresholding first would alias the
        # mask boundary onto the 320-grid instead of the frame grid
        raw = cv2.resize(prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        mask = raw > self._threshold

        return torch.from_numpy(mask), {"raw": raw}
```

`segment` returns a `torch.Tensor` because `BaseSegmentation.segment` declares
`tuple[torch.Tensor, Any] | None`, and insid3 — the other single-mask backend — returns a
`(H, W)` `torch.bool` tensor the same way.

**`mask.dtype == bool` is False for a torch tensor.** Measured: `torch.bool == bool`
evaluates to `False`, unlike numpy, where `arr.dtype == bool` is `True`. The test above
asserts `torch.bool` for that reason; `sky_masks` in Task 4 returns numpy and asserts
`bool` there. `mask.shape == (64, 96)` does work — `torch.Size` compares equal to a tuple.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_sky_segmentation.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Run the docstring contract, which now covers this file**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -q
```

Expected: PASS. If it fails on `sky.py`, the message names the exact rule — fix the
docstring, do not add the file to an exclusion list.

- [ ] **Step 6: Format**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && /opt/venv/reconstruction/bin/python -m black \
  collab_splats/semantics/segmentation/sky.py tests/semantics/test_sky_segmentation.py \
  && /opt/venv/reconstruction/bin/python -m isort \
  collab_splats/semantics/segmentation/sky.py tests/semantics/test_sky_segmentation.py
```

Two measured facts about this repo's formatters:

- **Never run repo-wide `black .`.** The venv's black is newer than the one the repo was
  formatted with, so it rewrites unrelated files. Name the two files, as above.
- **The blank line after the module docstring is deliberate.** The venv's black inserts one.
  `segmentation/`'s four existing modules do not have it; `preproc/frames.py` and
  `mesh/clean.py` — the two most recently cleaned packages — do. The sketch above is written
  with the blank line already in, so black is a no-op rather than a surprise diff. Do not
  delete it to match the neighbours; the next formatter run only puts it back.
- **A `########` divider directly after the import block takes ONE blank line, not two.**
  Measured: black collapses two to one there, and keeps two once any statement — a
  `logger = ...`, say — sits between. That is why `sky.py` spaces its dividers with two
  blank lines and the test file spaces its first one with one.

Every code block in this plan was run through `black --config pyproject.toml --check` and
`isort --check-only` before the plan was written, and both are clean — measured, not
assumed. If black proposes a change here, the code was retyped, not copied.

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
git add collab_splats/semantics/segmentation/sky.py tests/semantics/test_sky_segmentation.py
git commit -m "feat(semantics): sky segmentation backend over VGGT's skyseg ONNX model"
```

---

### Task 3: Register the backend and update the docs map

**Files:**
- Modify: `collab_splats/semantics/segmentation/__init__.py`
- Modify: `collab_splats/semantics/segmentation/base.py:50-56`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the failing test**

Append to `tests/semantics/test_sky_segmentation.py`:

```python
######## Registration


def test_skyseg_is_registered_and_exported():
    from collab_splats.semantics.segmentation import BaseSegmentation, SkySegmentation

    assert BaseSegmentation.get("skyseg") is SkySegmentation
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_sky_segmentation.py::test_skyseg_is_registered_and_exported -q
```

Expected: FAIL with `ImportError: cannot import name 'SkySegmentation'`.

- [ ] **Step 3: Add the export**

In `collab_splats/semantics/segmentation/__init__.py`, add to the concrete-backends block
(after the `insid3` line):

```python
from .sky import SkySegmentation
```

and to `__all__`, under the `# backends` comment, after `"INSID3Segmentation",`:

```python
    "SkySegmentation",
```

- [ ] **Step 4: Add skyseg to the base docstring's rank enumeration**

In `collab_splats/semantics/segmentation/base.py`, the abstract `segment` docstring's
`Returns:` block currently reads `... (H, W) bool for insid3, (N, H, W) float32 for
mobilesamv2, ...`. Change that sentence to:

```
            (masks, metadata) — masks rank and dtype are backend-specific: (H, W) bool
            for insid3 and skyseg, (N, H, W) float32 for mobilesamv2, (N, 1, H, W)
            float32 for sam3, and mobilesamv2 returns None outright when nothing is
            detected. metadata is backend-specific.
```

- [ ] **Step 5: Update the architecture map**

In `CLAUDE.md`, find the line:

```
    segmentation/          # BaseSegmentation; registered insid3, mobilesamv2, sam3
```

Replace with:

```
    segmentation/          # BaseSegmentation; registered insid3, mobilesamv2, sam3, skyseg
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/semantics/ tests/test_docstring_contract.py -q
```

Expected: PASS, including the 4 sky tests.

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
git add collab_splats/semantics/segmentation/__init__.py collab_splats/semantics/segmentation/base.py \
        tests/semantics/test_sky_segmentation.py CLAUDE.md
git commit -m "feat(semantics): register skyseg backend and document its mask rank"
```

---

### Task 4: `sky_masks` — cached, order-preserving mask stack

632 ms/frame on CPU is what earns the cache. The ordering contract is what stops a silent
wrong-mask-on-wrong-frame bug on the splats path.

**Files:**
- Modify: `collab_splats/semantics/segmentation/sky.py`
- Modify: `collab_splats/semantics/segmentation/__init__.py`
- Test: `tests/semantics/test_sky_segmentation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/semantics/test_sky_segmentation.py`:

```python
######## Mask cache and ordering


def _scene(tmp_path, monkeypatch, prob, frame_idxs=(0, 5, 7)):
    """
    An images/ dir plus a SkySegmentation whose registry entry is the fake-session one.
    """
    from collab_splats.preproc import frames as fr

    images = [np.full((16, 16, 3), idx, np.uint8) for idx in frame_idxs]
    fr.write_frames(tmp_path / "images", images, [{"frame_idx": i} for i in frame_idxs], {})

    backend, session = _backend(monkeypatch, prob)
    monkeypatch.setattr(sky.BaseSegmentation, "get", classmethod(lambda cls, name: lambda: backend))
    return tmp_path / "images", session


def test_sky_masks_reads_every_frame_in_filename_order(tmp_path, monkeypatch):
    prob = np.zeros((320, 320), np.float32)
    prob[:160] = 0.9
    images_dir, session = _scene(tmp_path, monkeypatch, prob)

    masks = sky.sky_masks(images_dir)

    assert masks.shape == (3, 16, 16)
    assert masks.dtype == bool
    assert masks[:, :6].all() and not masks[:, 10:].any()
    assert session.calls == 3


def test_sky_masks_returns_masks_in_the_order_idxs_names_them(tmp_path, monkeypatch):
    # Per-frame probability so a permutation is detectable: frame_idx 7 is the only sky one
    calls = {"n": 0}

    def _prob_for_call():
        calls["n"] += 1
        return np.full((320, 320), 0.9 if calls["n"] == 3 else 0.0, np.float32)

    images_dir, session = _scene(tmp_path, monkeypatch, np.zeros((320, 320), np.float32))
    monkeypatch.setattr(session, "run", lambda names, feed: [_prob_for_call()[None, None]])

    # Filename order is (0, 5, 7), so the third segmented frame is frame_idx 7
    sky.sky_masks(images_dir)
    permuted = sky.sky_masks(images_dir, idxs=[7, 0, 5])

    assert permuted[0].all()
    assert not permuted[1].any() and not permuted[2].any()


def test_sky_masks_caches_to_disk_with_255_meaning_sky(tmp_path, monkeypatch):
    import cv2

    prob = np.zeros((320, 320), np.float32)
    prob[:160] = 0.9
    images_dir, session = _scene(tmp_path, monkeypatch, prob)

    sky.sky_masks(images_dir)
    cached = cv2.imread(str(tmp_path / "sky" / "frame_000005.png"), cv2.IMREAD_GRAYSCALE)

    assert cached.shape == (16, 16)
    assert cached[0, 0] == 255 and cached[15, 15] == 0


def test_sky_masks_second_call_runs_the_model_zero_times(tmp_path, monkeypatch):
    prob = np.full((320, 320), 0.9, np.float32)
    images_dir, session = _scene(tmp_path, monkeypatch, prob)

    sky.sky_masks(images_dir)
    assert session.calls == 3

    again = sky.sky_masks(images_dir)
    assert session.calls == 3
    assert again.all()


def test_sky_masks_rejects_an_unknown_frame_idx(tmp_path, monkeypatch):
    images_dir, _ = _scene(tmp_path, monkeypatch, np.zeros((320, 320), np.float32))

    with pytest.raises(KeyError, match="99"):
        sky.sky_masks(images_dir, idxs=[0, 99])
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_sky_segmentation.py -q -k sky_masks
```

Expected: FAIL with `AttributeError: module ... has no attribute 'sky_masks'`.

- [ ] **Step 3: Implement `sky_masks`**

In `collab_splats/semantics/segmentation/sky.py`, add to the imports:

```python
from collections.abc import Sequence

from collab_splats.preproc import frames
```

and append this section at the end of the file:

```python
########################################################
########## Cached mask stack ###########################
########################################################


def sky_masks(
    images_dir: Path | str,
    idxs: Sequence[int] | None = None,
    cache_dir: Path | str | None = None,
    backend: str = "skyseg",
) -> np.ndarray:
    """
    Sky masks for a keyframe directory, segmenting only what is not already cached.

    - inference costs ~632 ms/frame on CPU, so the PNG cache is load-bearing, not an
      optimization
    - cached PNGs store 255 for sky, inverted from upstream's 255-is-NOT-sky files

    Args:
        images_dir: keyframe directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices in the order wanted; None takes every frame in
            filename order, exactly as `frames.read_frames` does.
        cache_dir: directory of cached mask PNGs; None uses images_dir's sibling sky/.
        backend: registered BaseSegmentation name to segment with.

    Returns:
        (N, H, W) bool, True where sky, in the order `idxs` names.

    Raises:
        FileNotFoundError: when images_dir holds no frames.
        KeyError: when idxs names a frame_idx the directory does not hold.
    """
    images_dir = Path(images_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else images_dir.parent / "sky"

    paths = frames.frame_paths(images_dir)
    if not paths:
        raise FileNotFoundError(f"sky_masks: no frame images in {images_dir}")

    # Resolve the wanted source indices up front, so the contract holds whether or not the
    # cache is warm — validating only the uncached ones would let a warm cache accept junk
    by_idx = {frames.frame_idx_from_path(p): p for p in paths}
    wanted = [int(i) for i in idxs] if idxs is not None else list(by_idx)
    missing = [i for i in wanted if i not in by_idx]
    if missing:
        raise KeyError(f"sky_masks: frame_idx {missing[:5]} not in {images_dir}")

    # Segment only the cache misses, in one read_frames call rather than one per frame
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in wanted if not (cache_dir / f"frame_{i:06d}.png").exists()]
    if todo:
        model = BaseSegmentation.get(backend)()
        for idx, frame in zip(todo, frames.read_frames(images_dir, todo)):
            mask, _ = model.segment(frame)
            cv2.imwrite(
                str(cache_dir / f"frame_{idx:06d}.png"),
                np.asarray(mask, dtype=np.uint8) * 255,
                [cv2.IMWRITE_PNG_COMPRESSION, frames.PNG_COMPRESSION],
            )
        logger.info("sky_masks: segmented %d of %d frames into %s", len(todo), len(wanted), cache_dir)

    # Read every mask back from the cache, so a hit and a miss return the identical array
    return np.stack([cv2.imread(str(cache_dir / f"frame_{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in wanted])
```

Add `sky_masks` to the module docstring bullets:

```
- sky_masks: cached (N, H, W) mask stack for a keyframe directory, order-preserving
```

- [ ] **Step 4: Export it**

In `collab_splats/semantics/segmentation/__init__.py`, change the sky import to:

```python
from .sky import SkySegmentation, sky_masks
```

and add to `__all__` after `"SkySegmentation",`:

```python
    "sky_masks",
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_sky_segmentation.py tests/test_docstring_contract.py -q
```

Expected: 9 passed in the sky file, contract green.

- [ ] **Step 6: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
/opt/venv/reconstruction/bin/python -m black collab_splats/semantics/segmentation/sky.py tests/semantics/test_sky_segmentation.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/semantics/segmentation/sky.py tests/semantics/test_sky_segmentation.py
git add collab_splats/semantics/segmentation/sky.py collab_splats/semantics/segmentation/__init__.py \
        tests/semantics/test_sky_segmentation.py
git commit -m "feat(semantics): cached, order-preserving sky_masks over a keyframe directory"
```

---

### Task 5: `render_tsdf_inputs` returns `image_ids`

The two mesh source arms order frames differently — feedforward by filename, splats by
checkpoint `image_ids`. One mask stack in filename order would silently misalign on the
splats path. This task exposes the ordering so Task 6 can honour it.

**Files:**
- Modify: `collab_splats/mesh/io.py:147-189`
- Modify: `collab_splats/wrapper/reconstructor.py:617`
- Modify: `evals/scripts/analyze_splats.py:180-192`
- Modify: `tests/mesh/test_io.py` (4 unpack sites)
- Modify: `tests/wrapper/test_splats_stage.py:188-192`
- Modify: `docs/source/tutorials/06_mesh/splats_mesh.ipynb`

- [ ] **Step 1: Write the failing test**

Add to `tests/mesh/test_io.py`, after `test_render_tsdf_inputs_swaps_in_source_frames_by_image_id`:

```python
def test_render_tsdf_inputs_returns_the_image_ids_it_read(tmp_path, monkeypatch):
    h, w = 8, 8
    images_dir = tmp_path / "images"
    frames = [np.full((h, w, 3), idx * 10, dtype=np.uint8) for idx in (0, 5, 7)]
    write_frames(images_dir, frames, [{"frame_idx": idx} for idx in (0, 5, 7)], {})
    monkeypatch.setitem(
        sys.modules,
        "collab_splats.splats.rendering",
        _fake_rendering([_view(h, w, 1.0), _view(h, w, 1.0)], [7, 0], (h, w)),
    )
    _, rgbs, _, _, image_ids = render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")

    # The ids must be the order the RGB rows are actually in, not the directory's order
    assert image_ids == [7, 0]
    assert np.all(rgbs[0] == 70) and np.all(rgbs[1] == 0)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_io.py::test_render_tsdf_inputs_returns_the_image_ids_it_read -q
```

Expected: FAIL with `ValueError: not enough values to unpack (expected 5, got 4)`.

- [ ] **Step 3: Change the return**

In `collab_splats/mesh/io.py`, replace the `Returns:` block of `render_tsdf_inputs`:

```
    Returns:
        depths (N, H, W) float32 (median depth for 2dgs, 0 where alpha is 0),
        rgbs (N, H, W, 3) uint8, c2w (N, 4, 4) float32, K (N, 3, 3) float32,
        image_ids list[int] — the source frame_idx per row, which is the checkpoint's
        order and NOT the images/ directory's filename order.
```

Normalise the ids once, right after `load_checkpoint` — replace:

```python
    # Source frames replace rendered RGB when a keyframe directory is given
    rgbs = None
    if images_dir is not None:
        rgbs = read_frames(images_dir, [int(i) for i in image_ids])
```

with:

```python
    # One int list for both the RGB lookup and the return, so a caller masking per frame
    # cannot pair row order with a different ordering
    image_ids = [int(i) for i in image_ids]

    # Source frames replace rendered RGB when a keyframe directory is given
    rgbs = None
    if images_dir is not None:
        rgbs = read_frames(images_dir, image_ids)
```

and replace the final return:

```python
    return (
        np.stack(depths),
        rgbs,
        _to_numpy(cam_to_world).astype(np.float32),
        _to_numpy(intrinsics).astype(np.float32),
        image_ids,
    )
```

- [ ] **Step 4: Update the four existing unpack sites in `tests/mesh/test_io.py`**

```python
    depths, rgbs, c2w, K, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
```

```python
    depths, _, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
```

```python
    _, rgbs, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")
```

The fourth (`test_render_tsdf_inputs_rejects_frame_size_mismatch`) does not unpack — it
asserts a raise — so it needs no change.

- [ ] **Step 5: Update the reconstructor call site**

In `collab_splats/wrapper/reconstructor.py`, line 617:

```python
        depths, rgbs, c2w, intrinsics, image_ids = render_tsdf_inputs(splats_ckpt, images_dir)
```

- [ ] **Step 6: Update the splats-stage test's mock tuple**

In `tests/wrapper/test_splats_stage.py`, the `rendered` tuple gains a fifth element:

```python
    rendered = (
        depths,
        np.full((3, 4, 5, 3), 7, np.uint8),
        np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
        [0, 1, 2],
    )
```

- [ ] **Step 7: Update the eval script**

In `evals/scripts/analyze_splats.py`, change `build_mesh`'s docstring line and unpack:

```
        inputs: (depths, rgbs, c2w, K, image_ids) as render_tsdf_inputs returns them;
            the ids are unused here because every view is fused.
```

```python
    depths, rgbs, c2w, intrinsics, _ = inputs
```

Also update the summary line above it to match:

```
    Fuse and clean one render_tsdf_inputs tuple, then flatten the PLY to mesh_<name>.ply.
```

- [ ] **Step 8: Update the tutorial notebook**

In `docs/source/tutorials/06_mesh/splats_mesh.ipynb`, the code cell reading:

```python
depths_sp, rgbs_sp, c2w_sp, K_sp = render_tsdf_inputs(SPLATS_CKPT)
```

becomes:

```python
depths_sp, rgbs_sp, c2w_sp, K_sp, ids_sp = render_tsdf_inputs(SPLATS_CKPT)
```

and the markdown cell that says `re-renders ckpt.pt into the (depths, rgbs, c2w, K) arrays
fuse_tsdf takes` becomes `re-renders ckpt.pt into the (depths, rgbs, c2w, K, image_ids)
tuple — the first four are what fuse_tsdf takes`.

- [ ] **Step 9: Run every affected test**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/mesh/ tests/wrapper/ -q
```

Expected: PASS. `tests/wrapper/test_splats_stage.py` may show pre-existing failures
unrelated to this change — compare against the control you captured before Task 1 if any
appear, and do not "fix" an inherited failure here.

- [ ] **Step 10: Confirm no unpack site was missed**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && grep -rn "render_tsdf_inputs" --include=*.py --include=*.ipynb . | grep -v "^./third_party"
```

Expected: every call site either unpacks 5 values or passes the tuple through.

- [ ] **Step 11: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
git add collab_splats/mesh/io.py collab_splats/wrapper/reconstructor.py evals/scripts/analyze_splats.py \
        tests/mesh/test_io.py tests/wrapper/test_splats_stage.py docs/source/tutorials/06_mesh/splats_mesh.ipynb
git commit -m "feat(mesh): render_tsdf_inputs returns image_ids so callers can align per-frame data"
```

---

### Task 6: `mesh.mask_sky` config and mesh-stage wiring

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (imports, `_run_tsdf_mesh`, `mesh()`)
- Modify: `configs/base.yaml`
- Modify: `tests/wrapper/_stubs.py`
- Modify: `tests/wrapper/test_absent_confidence.py:23-37`
- Test: `tests/wrapper/test_mask_sky.py`

**Two traps in this task.**

`tests/wrapper/_stubs.py` hand-writes a `mesh` config dict, because `_stub_reconstructor`
builds its Reconstructor with `__new__` and so never merges base.yaml. The moment `mesh()`
reads `mesh_cfg["mask_sky"]`, all five `recon.mesh()` calls in `test_splats_stage.py`
KeyError. The stub dict must gain the key.

`test_absent_confidence.py` already holds the exact minimal `FeedforwardResult` these tests
need. Do not copy it a second time — move it into `_stubs.py`, which exists to be the shared
stub home, and point both files at it.

- [ ] **Step 1: Move the shared fixture into `_stubs.py`**

Add to `tests/wrapper/_stubs.py`, after the imports:

```python
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
```

and after `_stub_reconstructor`:

```python
def minimal_feedforward_result(n=2, h=8, w=8):
    """
    FeedforwardResult with unit depth and confidence=None; the smallest thing _run_tsdf_mesh accepts.
    """
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.array([[0, 0, w, h, w, h]] * n, dtype=np.float32),
        model_width=w,
        model_height=h,
        images=np.zeros((n, 3, h, w), dtype=np.float32),
        depth=np.ones((n, h, w), dtype=np.float32),
    )


def minimal_pose_result(n=2):
    """
    The PointcloudResult stand-in _run_tsdf_mesh reads poses and original-res K from.
    """
    return SimpleNamespace(
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (n, 1, 1)),
    )
```

Then in `tests/wrapper/test_absent_confidence.py`, delete the whole
`_result_no_confidence()` function (lines 23-37) and replace its two call sites' source with
the import. Add to that file's imports:

```python
from tests.wrapper._stubs import minimal_feedforward_result
```

and replace both `result = _result_no_confidence()` lines with:

```python
    result = minimal_feedforward_result()
```

- [ ] **Step 2: Confirm that refactor is behaviour-neutral before adding anything**

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_absent_confidence.py -q
```

Expected: 3 passed, exactly as before the move. The absolute `tests.wrapper._stubs` form is
what `test_splats_stage.py:22` already uses and `tests/__init__.py` exists, so it resolves.

- [ ] **Step 3: Write the failing tests**

Create `tests/wrapper/test_mask_sky.py`:

```python
"""
mesh.mask_sky wiring in the mesh stage.

- the mask is applied after the two source arms converge, so both are covered
- the splats arm orders frames by checkpoint image_ids, not filename, and the mask stack
  must be requested in that same order or every mask lands on the wrong frame
"""

from unittest.mock import patch

import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.wrapper.reconstructor import _run_tsdf_mesh
from tests.wrapper._stubs import minimal_feedforward_result, minimal_pose_result


def _run_feedforward_mesh(tmp_path, fused, sky_return, **kwargs):
    """
    Drive _run_tsdf_mesh's feedforward arm with everything below fuse_tsdf stubbed out.
    """
    result = minimal_feedforward_result()

    def spy_fuse(depths, rgbs, c2w, K, out_dir, **kw):
        fused["depths"] = depths
        return tmp_path / "mesh.ply"

    with (
        patch.object(FeedforwardResult, "load_zarr", staticmethod(lambda *a, **k: result)),
        patch(
            "collab_splats.wrapper.reconstructor.frames.read_frames",
            return_value=np.zeros((2, 8, 8, 3), np.uint8),
        ),
        patch("collab_splats.wrapper.reconstructor.upsample_depths", side_effect=lambda d, r, b: d),
        patch("collab_splats.wrapper.reconstructor.fuse_tsdf", side_effect=spy_fuse),
        patch("collab_splats.wrapper.reconstructor.clean_repair_mesh"),
        patch("collab_splats.wrapper.reconstructor.sky_masks", return_value=sky_return) as sky,
    ):
        _run_tsdf_mesh(
            result=minimal_pose_result(),
            pointcloud_zarr=tmp_path / "pointcloud.zarr",
            output_dir=tmp_path,
            images_dir=tmp_path / "images",
            voxel_size=0.01,
            depth_trunc=2.0,
            **kwargs,
        )
    return sky


def test_mask_sky_zeroes_feedforward_depth_where_sky_and_nowhere_else(tmp_path):
    # Top two rows sky; every other pixel must survive bit-identically
    mask = np.zeros((2, 8, 8), bool)
    mask[:, :2] = True
    fused = {}

    sky = _run_feedforward_mesh(tmp_path, fused, mask, mask_sky=True)

    assert np.all(fused["depths"][:, :2] == 0.0)
    assert np.all(fused["depths"][:, 2:] == 1.0)
    assert sky.call_args.kwargs["idxs"] is None


def test_mask_sky_off_never_loads_the_model(tmp_path):
    fused = {}

    sky = _run_feedforward_mesh(tmp_path, fused, None, mask_sky=False)

    assert sky.call_count == 0
    assert np.all(fused["depths"] == 1.0)


def test_mask_sky_rejects_a_mask_that_does_not_match_the_depth_grid(tmp_path):
    with pytest.raises(ValueError, match="Sky masks are"):
        _run_feedforward_mesh(tmp_path, {}, np.zeros((2, 4, 4), bool), mask_sky=True)


def test_mask_sky_asks_for_splats_frames_in_checkpoint_order(tmp_path):
    # The checkpoint's rgbs rows are in image_ids order (7, 0), NOT filename order (0, 7).
    # Masking row 0 must therefore mask frame 7, which only holds if sky_masks was asked
    # for [7, 0] rather than being left to its own filename-order default.
    rendered = (
        np.ones((2, 8, 8), np.float32),
        np.zeros((2, 8, 8, 3), np.uint8),
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        [7, 0],
    )
    mask = np.zeros((2, 8, 8), bool)
    mask[0] = True
    fused = {}

    def spy_fuse(depths, rgbs, c2w, K, out_dir, **kw):
        fused["depths"] = depths
        return tmp_path / "mesh.ply"

    with (
        patch("collab_splats.wrapper.reconstructor.render_tsdf_inputs", return_value=rendered),
        patch("collab_splats.wrapper.reconstructor.fuse_tsdf", side_effect=spy_fuse),
        patch("collab_splats.wrapper.reconstructor.clean_repair_mesh"),
        patch("collab_splats.wrapper.reconstructor.sky_masks", return_value=mask) as sky,
    ):
        _run_tsdf_mesh(
            result=minimal_pose_result(),
            pointcloud_zarr=tmp_path / "pointcloud.zarr",
            output_dir=tmp_path,
            images_dir=tmp_path / "images",
            voxel_size=0.01,
            depth_trunc=2.0,
            source="splats",
            splats_ckpt=tmp_path / "ckpt.pt",
            mask_sky=True,
        )

    assert sky.call_args.kwargs["idxs"] == [7, 0]
    assert np.all(fused["depths"][0] == 0.0)
    assert np.all(fused["depths"][1] == 1.0)
```

- [ ] **Step 4: Run them to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_mask_sky.py -q
```

Expected: FAIL — `_run_tsdf_mesh() got an unexpected keyword argument 'mask_sky'`.

- [ ] **Step 5: Import `sky_masks` in the reconstructor**

In `collab_splats/wrapper/reconstructor.py`, beside the other `collab_splats.semantics` imports:

```python
from collab_splats.semantics.segmentation import sky_masks
```

- [ ] **Step 6: Add the parameter and apply the mask**

In `_run_tsdf_mesh`'s signature, after `conf_percentile: float | None = None,`:

```python
    mask_sky: bool = False,
```

In its docstring `Args:`, after the `conf_percentile:` line:

```
        mask_sky: zero out depth where the sky segmenter fires; applies to both sources.
```

Give the feedforward arm an explicit ordering, so both arms leave `image_ids` defined. At
the end of the `else` branch, after `intrinsics = result.intrinsics`:

```python

        # read_frames above took filename order, which sky_masks defaults to
        image_ids = None
```

And in the `if source == "splats":` branch, the unpack from Task 5 already binds
`image_ids`. Then, immediately before `output_dir.mkdir(parents=True, exist_ok=True)`:

```python
    # Sky has no surface but every depth source gives it one
    # - fuses as a backdrop and seeds floaters around it
    # - applied after the arms converge: both are frame-resolution on the rgbs grid
    # - image_ids None on the feedforward path (filename order), the checkpoint's own order
    #   on the splats path, which is the order its rgbs rows are already in
    if mask_sky:
        sky = sky_masks(images_dir, idxs=image_ids)
        if sky.shape != depths.shape:
            raise ValueError(
                f"Sky masks are {sky.shape} but depths are {depths.shape} — "
                f"{images_dir} does not match the depth source."
            )
        dropped = np.count_nonzero(sky & (depths > 0)) / max(depths.size, 1)
        depths = np.where(sky, 0.0, depths)
        logger.info("mesh.mask_sky: dropped %.2f%% of depth pixels as sky", 100 * dropped)
```

- [ ] **Step 7: Thread the config through `mesh()`**

In `Reconstructor.mesh()`'s `_run_tsdf_mesh(...)` call (around line 1297), after
`conf_percentile=mesh_cfg["conf_percentile"],`:

```python
            mask_sky=mesh_cfg["mask_sky"],
```

- [ ] **Step 8: Add the key to the stub config**

`_stub_reconstructor` bypasses `__init__`, so base.yaml never reaches it. In
`tests/wrapper/_stubs.py`'s `"mesh"` dict, after `"conf_percentile": 20,`:

```python
            "mask_sky": False,
```

Without this, all five `recon.mesh()` calls in `test_splats_stage.py` raise `KeyError:
'mask_sky'`.

- [ ] **Step 9: Add the config key**

In `configs/base.yaml`, in the `mesh:` block after the `conf_percentile` line:

```yaml
  mask_sky: false          # zero depth where the sky segmenter fires, before fusion. Sky has no
                           # surface but every depth source assigns it one, so it fuses a backdrop
                           # and seeds floaters. Applies to BOTH mesh sources — note the asymmetry
                           # with conf_percentile, which also masks the splats stage's depth targets
                           # (see splats below); mask_sky does not. Ships off: the segmenter calls
                           # blown-out overexposed surfaces sky (measured 12.1% of one GH010229
                           # frame), and those are real geometry with valid depth.
```

- [ ] **Step 10: Run the tests to verify they pass**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_mask_sky.py -q
```

Expected: 4 passed.

- [ ] **Step 11: Run the full affected suite**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/mesh/ tests/semantics/ tests/preproc/ \
  tests/test_docstring_contract.py -q
```

Expected: PASS apart from any failure that was already failing before Task 1.

- [ ] **Step 12: Format and commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
/opt/venv/reconstruction/bin/python -m black collab_splats/wrapper/reconstructor.py tests/wrapper/
/opt/venv/reconstruction/bin/python -m isort collab_splats/wrapper/reconstructor.py tests/wrapper/
git add collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/test_mask_sky.py \
        tests/wrapper/_stubs.py tests/wrapper/test_absent_confidence.py
git commit -m "feat(mesh): mesh.mask_sky drops sky depth before TSDF fusion on both sources"
```

---

### Task 7: A/B harness

Reuses `mesh_stats` from `analyze_splats.py` rather than re-deriving connected components,
so the numbers stay comparable to the floater-cleanup and banded-fusion runs. Note its
`largest_component_fraction` is a TRIANGLE fraction; the spec says "vertex fraction". The
existing function wins — say "triangle fraction" in the report rather than forking it.

**Files:**
- Create: `evals/scripts/eval_sky_mask.py`

- [ ] **Step 1: Write the script**

```python
"""
A/B one scene's mesh with and without mesh.mask_sky.

- runs the mesh stage twice against the SAME pointcloud.zarr, so the only difference is the flag
- reports component stats from analyze_splats.mesh_stats
- renders both meshes from ONE fixed camera, and writes a mask-overlay contact sheet
- compute only: run from a shell or tmux, never a notebook
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml

from collab_splats.preproc import frames
from collab_splats.utils.visualization import overlay_masks
from collab_splats.wrapper.reconstructor import Reconstructor
from evals.scripts.analyze_splats import mesh_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


########################################
# Arms
########################################


def load_recon(config_path: Path, mask_sky: bool) -> Reconstructor:
    """
    A Reconstructor for one scene at a given mask_sky setting.

    - Reconstructor takes a dict and deep-merges it over base.yaml itself; there is no
      from_yaml classmethod, and a scene's run_config.yaml is already a full config

    Args:
        config_path: scene config YAML; its pointcloud.zarr must already exist.
        mask_sky: the setting under test, forced after the merge.

    Returns:
        A configured Reconstructor.
    """
    config = yaml.safe_load(config_path.read_text())
    recon = Reconstructor(config)
    recon.config["mesh"]["mask_sky"] = mask_sky
    return recon


def run_arm(config_path: Path, mask_sky: bool, results_dir: Path) -> dict:
    """
    Run the mesh stage once at a given mask_sky setting and collect its stats.

    Args:
        config_path: scene config; its pointcloud.zarr must already exist.
        mask_sky: the setting under test.
        results_dir: receives mesh_{on,off}.ply.

    Returns:
        Stats dict from mesh_stats plus 'mask_sky' and 'path'.
    """
    recon = load_recon(config_path, mask_sky)

    mesh_path = recon.mesh(overwrite=True)

    # Move the PLY aside before the other arm overwrites it in place
    arm = "on" if mask_sky else "off"
    out_path = results_dir / f"mesh_{arm}.ply"
    shutil.move(str(mesh_path), str(out_path))

    stats = mesh_stats(out_path)
    stats["mask_sky"] = mask_sky
    stats["path"] = str(out_path)
    logger.info("mask_sky=%s: %s", mask_sky, stats)
    return stats


########################################
# Renders
########################################


def render_meshes(mesh_paths: dict[str, Path], out_path: Path, size: int = 900) -> Path:
    """
    Render every arm's mesh from one shared camera and tile them side by side.

    - ONE OffscreenRenderer for the whole process: Open3D segfaults on a second one
    - the camera is derived from the FIRST mesh and reused, so the arms are comparable;
      a per-mesh camera would silently reframe when masking changes the bounds

    Args:
        mesh_paths: arm label -> PLY path; insertion order sets the camera and the tiling.
        out_path: PNG to write.
        size: square render edge, pixels.

    Returns:
        Path to out_path.
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    eye = center = up = None
    tiles = []
    for label, path in mesh_paths.items():
        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.compute_vertex_normals()

        # Frame on the first mesh only; a 3/4 view off the bounding box diagonal
        if eye is None:
            box = mesh.get_axis_aligned_bounding_box()
            center = box.get_center()
            extent = np.linalg.norm(box.get_extent())
            eye = center + np.array([0.6, -0.9, 0.6]) * max(extent, 1e-6)
            up = np.array([0.0, 0.0, 1.0])

        renderer.scene.add_geometry(label, mesh, material)
        renderer.setup_camera(60.0, center, eye, up)
        tile = np.asarray(renderer.render_to_image()).copy()  # putText writes in place
        renderer.scene.remove_geometry(label)

        # Label in the corner, so a saved PNG is readable without its filename
        cv2.putText(tile, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(tile)

    cv2.imwrite(str(out_path), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    logger.info("mesh renders: %d arms -> %s", len(tiles), out_path)
    return out_path


########################################
# Mask QA
########################################


def write_mask_contact_sheet(images_dir: Path, sky_dir: Path, out_path: Path, stride: int = 10) -> Path:
    """
    Tile every stride-th frame with its sky mask overlaid, so false positives are visible.

    Args:
        images_dir: the scene's keyframe directory.
        sky_dir: the mask cache written by sky_masks.
        out_path: PNG to write.
        stride: take every stride-th frame.

    Returns:
        Path to out_path.
    """
    paths = frames.frame_paths(images_dir)[::stride]
    tiles = []
    for path in paths:
        idx = frames.frame_idx_from_path(path)
        mask_path = sky_dir / f"frame_{idx:06d}.png"
        if not mask_path.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        overlaid = overlay_masks(rgb, torch.from_numpy(mask[None]).float())
        tiles.append(cv2.resize(overlaid, (320, 180)))

    if not tiles:
        raise FileNotFoundError(f"no cached masks in {sky_dir} for the frames in {images_dir}")

    # Four per row; pad the last row so vstack has uniform widths
    rows = [tiles[i : i + 4] for i in range(0, len(tiles), 4)]
    blank = np.zeros_like(tiles[0])
    rows = [row + [blank] * (4 - len(row)) for row in rows]
    sheet = np.vstack([np.hstack(row) for row in rows])

    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    logger.info("mask contact sheet: %d frames -> %s", len(tiles), out_path)
    return out_path


########################################
# Report
########################################


def comparison_table(off: dict, on: dict) -> str:
    """
    Markdown table of the two arms with a delta column.

    Args:
        off: stats from the mask_sky=false arm.
        on: stats from the mask_sky=true arm.

    Returns:
        The table as a string.
    """
    keys = ("vertices", "triangles", "components", "largest_component_fraction")
    lines = [
        "| metric | mask_sky off | mask_sky on | delta |",
        "|---|---|---|---|",
    ]
    for key in keys:
        a, b = off[key], on[key]
        delta = f"{b - a:+.4f}" if isinstance(a, float) else f"{b - a:+d}"
        fmt = "{:.4f}" if isinstance(a, float) else "{}"
        lines.append(f"| {key} | {fmt.format(a)} | {fmt.format(b)} | {delta} |")
    return "\n".join(lines)


def main() -> None:
    """
    Run both arms for one scene and write stats.json, a table and a contact sheet.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="scene config YAML")
    parser.add_argument("--results", type=Path, required=True, help="output directory")
    parser.add_argument("--stride", type=int, default=10, help="contact-sheet frame stride")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    # Off first: it populates nothing, so an on-arm cache miss is genuinely the first run
    off = run_arm(args.config, mask_sky=False, results_dir=args.results)
    on = run_arm(args.config, mask_sky=True, results_dir=args.results)

    # Off first in the dict too: it sets the shared camera both tiles are framed with
    render_meshes(
        {"mask_sky off": Path(off["path"]), "mask_sky on": Path(on["path"])},
        args.results / "mesh_renders.png",
    )

    recon = load_recon(args.config, mask_sky=True)
    write_mask_contact_sheet(
        recon.images_dir,
        recon.images_dir.parent / "sky",
        args.results / "sky_masks.png",
        stride=args.stride,
    )

    (args.results / "stats.json").write_text(json.dumps({"off": off, "on": on}, indent=2))
    table = comparison_table(off, on)
    (args.results / "table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses and its help renders**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python evals/scripts/eval_sky_mask.py --help
```

Expected: the argparse help text, exit 0.

- [ ] **Step 3: Confirm the mesh stage honours the forced flag**

`load_recon` sets `config["mesh"]["mask_sky"]` AFTER `Reconstructor.__init__` has deep-merged
the caller's dict over base.yaml and validated it. That is only safe if `mesh()` reads the
live dict rather than a snapshot taken at construction:

```bash
sed -n '/    def mesh(/,/mesh_cfg = /p' collab_splats/wrapper/reconstructor.py | tail -3
```

Expected: `mesh_cfg = self.config["mesh"]` inside `mesh()`. If it reads a snapshot instead,
put `mask_sky` into the dict before constructing rather than after.

- [ ] **Step 4: Smoke the renderer on a throwaway mesh before trusting it in a long run**

Open3D's offscreen path fails at import time on a headless box rather than at render time,
and only one `OffscreenRenderer` may exist per process. Prove it works now, not after a
40-minute pointcloud rebuild:

```bash
PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -c "
import open3d as o3d
from pathlib import Path
from evals.scripts.eval_sky_mask import render_meshes
out = Path('/tmp/claude-0/-workspace-collab-splats/sky_smoke'); out.mkdir(parents=True, exist_ok=True)
m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
o3d.io.write_triangle_mesh(str(out/'a.ply'), m)
print(render_meshes({'a': out/'a.ply', 'b': out/'a.ply'}, out/'r.png'))
"
```

Expected: prints the PNG path, and the file is 1800x900. If Open3D raises on
`OffscreenRenderer`, this box has no EGL — report that and fall back to writing the two
meshes' stats only, noting the missing renders in the Task 8 report rather than faking them.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
git add evals/scripts/eval_sky_mask.py
git commit -m "feat(evals): A/B harness for mesh.mask_sky over one scene"
```

---

### Task 8: Run the A/B and write the report

**Gate: the tutorial scene must improve AND GH010229 must not regress.** On the frame
evidence in the spec, the control is expected to regress. If it does, the honest outcome is
"ships off, documented, needs a depth-agreement gate before it can be a default" — write
that, do not massage the numbers into a pass.

**Files:**
- Create: `docs/superpowers/specs/2026-09-07-sky-mask-measured-report.md`

- [ ] **Step 1: Build the tutorial scene's pointcloud once**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py --help
```

Read the help, then run preproc + pointcloud for `data/tutorial/tutorial_example-video.mp4`
with `semantics: {enabled: false}` in the overrides (standing instruction — semantics stays
off until lifted). Write the scene config to `/tmp/claude-0/-workspace-collab-splats/*/scratchpad/sky_ab/tutorial.yaml`.

Run it in tmux, not inline — the container caps at 46.6 GB and pointcloud is the heavy stage.

- [ ] **Step 2: Run the tutorial A/B**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python evals/scripts/eval_sky_mask.py \
  --config /tmp/claude-0/-workspace-collab-splats/*/scratchpad/sky_ab/tutorial.yaml \
  --results /tmp/claude-0/-workspace-collab-splats/*/scratchpad/sky_ab/tutorial_results
```

Expected: a printed table plus `stats.json`, `table.md` and `sky_masks.png` in the results dir.

- [ ] **Step 3: Look at the contact sheet before trusting any number**

Read `sky_masks.png`. The tutorial scene's value is sky through bare branches — if the
branches are being swallowed, the component delta is measuring the wrong thing and the
threshold needs revisiting before the report is written.

- [ ] **Step 4: Rebuild GH010229 and run its A/B**

GH010229's `images/` and `pointcloud.zarr` are gone, so this needs a full preproc +
pointcloud rebuild from the local 746 MB mp4. Same overrides, same two commands as
Steps 1-2, its own scene config and results dir. Run in tmux.

- [ ] **Step 5: Write the report**

Create `docs/superpowers/specs/2026-09-07-sky-mask-measured-report.md` with:

- both tables, tutorial and GH010229, verbatim from `table.md`
- the % of depth pixels dropped, read from the `mesh.mask_sky:` log line in each on-arm run
- `mesh_renders.png` and `sky_masks.png` referenced by path, with a sentence on what each
  shows; the mesh renders are the two arms from one shared camera, so any reframing between
  tiles is a bug in `render_meshes`, not a result
- an explicit verdict against the gate, naming which half passed and which failed
- if GH010229 regressed: the follow-up option the spec deliberately left undesigned —
  ANDing the sky mask with a far-depth test, since the false positives are near surfaces

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask
git add -f docs/superpowers/specs/2026-09-07-sky-mask-measured-report.md
git commit -m "docs(specs): measured A/B report for mesh.mask_sky"
```

---

## Final gate

- [ ] **Run the whole suite and compare against the pre-Task-1 control**

```bash
cd /workspace/collab-splats/.worktrees/sky-mask && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" \
  && PYTHONPATH=/workspace/collab-splats/.worktrees/sky-mask \
  /opt/venv/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -30
```

The printed `__file__` must be the worktree's. A failure set identical to the control is
the pass condition — not a green run, which this repo does not have.

- [ ] **Update CLAUDE.md's In-Flight Work list**

Add a `sky-mask` entry pointing at the spec and this plan, and remove it again only when
the work is appended to `docs/superpowers/CHANGELOG.md`.

- [ ] **Refresh the knowledge graph**

```bash
cd /workspace/collab-splats && graphify update .
```
