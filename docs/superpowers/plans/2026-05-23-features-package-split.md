# features/ Package Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `collab_splats/semantics/features.py` into a `features/` package mirroring the structure of `segmentation/`, with no behavior or API change.

**Architecture:** Pure structural refactor. Move `BaseFeatureExtractor` + `BaseQueryableExtractor` + module-level constants into `base.py`; each concrete extractor into its own file; re-export the identical public API from `features/__init__.py`. `collab_splats/semantics/__init__.py` is untouched.

**Tech Stack:** Python, PyTorch, HuggingFace transformers, maskclip_onnx, torchvision

---

## File Map

| Action | Path | Content |
|--------|------|---------|
| CREATE | `collab_splats/semantics/features/__init__.py` | Re-exports all 5 public names |
| CREATE | `collab_splats/semantics/features/base.py` | `BaseFeatureExtractor`, `BaseQueryableExtractor`, `TORCH_HOME`, `_DEBIAS_VALIDATED` |
| CREATE | `collab_splats/semantics/features/maskclip.py` | `MaskCLIPExtractor` (lines 413–525 of old file) |
| CREATE | `collab_splats/semantics/features/dino.py` | `DINOFeatureExtractor` (lines 526–607 of old file) |
| CREATE | `collab_splats/semantics/features/talk2dino.py` | `Talk2DinoExtractor` (lines 608–716 of old file) |
| DELETE | `collab_splats/semantics/features.py` | Replaced by package |
| NO CHANGE | `collab_splats/semantics/__init__.py` | Already imports from `.features` — unchanged |

---

### Task 1: Record baseline test pass count

**Files:** none

- [ ] **Step 1: Run existing semantics tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass. Note the exact count (e.g. "47 passed"). If any tests are already failing, do NOT proceed — fix them first.

---

### Task 2: Create `features/` package — `base.py`

**Files:**
- Create: `collab_splats/semantics/features/base.py`

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p /workspace/collab-splats/collab_splats/semantics/features
```

- [ ] **Step 2: Write `base.py`**

Copy everything from `collab_splats/semantics/features.py` lines 1–412 verbatim (the module docstring, all top-level imports, `TORCH_HOME`, `_DEBIAS_VALIDATED`, `logger`, `BaseFeatureExtractor`, `BaseQueryableExtractor`). Then update the module docstring to reflect the new file's scope:

```python
"""Base classes and shared constants for feature extractors.

Provides:
  BaseFeatureExtractor     — abstract registry-based extractor with caching and debiasing
  BaseQueryableExtractor   — extends base with text-query scoring
  TORCH_HOME               — resolved torch cache directory
  _DEBIAS_VALIDATED        — set of extractor class names with validated debiasing
"""
```

The import block at the top of `base.py` stays identical to the original `features.py` import block (lines 10–30):

```python
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.torch_utils import RegistryMixin
from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    get_device,
    interpolate_to_patch_size,
)
```

After writing, run black + isort to drop any unused imports:

```bash
cd /workspace/collab-splats && \
  /opt/conda/envs/nerfstudio/bin/python -m black collab_splats/semantics/features/base.py && \
  /opt/conda/envs/nerfstudio/bin/python -m isort collab_splats/semantics/features/base.py
```

---

### Task 3: Create `features/maskclip.py`

**Files:**
- Create: `collab_splats/semantics/features/maskclip.py`

- [ ] **Step 1: Write `maskclip.py`**

Copy `MaskCLIPExtractor` class body verbatim from `collab_splats/semantics/features.py` lines 413–525. Use these imports (replace the original top-level imports):

```python
"""MaskCLIP feature extractor backend."""
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device
from .base import BaseQueryableExtractor, TORCH_HOME

logger = logging.getLogger(__name__)
```

Then paste the `MaskCLIPExtractor` class (with `@BaseQueryableExtractor.register("maskclip")` decorator) unchanged below.

Run formatter:

```bash
cd /workspace/collab-splats && \
  /opt/conda/envs/nerfstudio/bin/python -m black collab_splats/semantics/features/maskclip.py && \
  /opt/conda/envs/nerfstudio/bin/python -m isort collab_splats/semantics/features/maskclip.py
```

---

### Task 4: Create `features/dino.py`

**Files:**
- Create: `collab_splats/semantics/features/dino.py`

- [ ] **Step 1: Write `dino.py`**

Copy `DINOFeatureExtractor` class body verbatim from `collab_splats/semantics/features.py` lines 526–607. Use these imports:

```python
"""DINOv2 feature extractor backend."""
import logging
from typing import Optional, Tuple

import torch
import torchvision.transforms as T
from transformers import AutoModel

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, interpolate_to_patch_size
from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)
```

Then paste the `DINOFeatureExtractor` class (with `@BaseFeatureExtractor.register("dinov2")` decorator) unchanged below.

Run formatter:

```bash
cd /workspace/collab-splats && \
  /opt/conda/envs/nerfstudio/bin/python -m black collab_splats/semantics/features/dino.py && \
  /opt/conda/envs/nerfstudio/bin/python -m isort collab_splats/semantics/features/dino.py
```

---

### Task 5: Create `features/talk2dino.py`

**Files:**
- Create: `collab_splats/semantics/features/talk2dino.py`

- [ ] **Step 1: Write `talk2dino.py`**

Copy `Talk2DinoExtractor` class body verbatim from `collab_splats/semantics/features.py` lines 608–716 (end of file). Use these imports — adjust to only what Talk2DinoExtractor actually uses (run black/isort to drop any that aren't referenced):

```python
"""Talk2DINO feature extractor backend."""
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from collab_splats.utils.image import open_image, resize_image
from collab_splats.semantics.utils import get_device, interpolate_to_patch_size
from .base import BaseQueryableExtractor

logger = logging.getLogger(__name__)
```

Then paste the `Talk2DinoExtractor` class (with `@BaseQueryableExtractor.register("talk2dino")` decorator) unchanged below.

Run formatter:

```bash
cd /workspace/collab-splats && \
  /opt/conda/envs/nerfstudio/bin/python -m black collab_splats/semantics/features/talk2dino.py && \
  /opt/conda/envs/nerfstudio/bin/python -m isort collab_splats/semantics/features/talk2dino.py
```

---

### Task 6: Write `features/__init__.py`

**Files:**
- Create: `collab_splats/semantics/features/__init__.py`

- [ ] **Step 1: Write the re-export file**

```python
"""collab_splats.semantics.features — feature extractor registry and backends."""

from .base import BaseFeatureExtractor, BaseQueryableExtractor
from .maskclip import MaskCLIPExtractor
from .dino import DINOFeatureExtractor
from .talk2dino import Talk2DinoExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
]
```

---

### Task 7: Delete `features.py` and verify imports

**Files:**
- Delete: `collab_splats/semantics/features.py`

- [ ] **Step 1: Delete the old flat file**

```bash
rm /workspace/collab-splats/collab_splats/semantics/features.py
```

- [ ] **Step 2: Smoke-test imports**

```bash
cd /workspace/collab-splats && \
  /opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.semantics import (
    BaseFeatureExtractor, BaseQueryableExtractor,
    MaskCLIPExtractor, DINOFeatureExtractor, Talk2DinoExtractor,
)
print('all imports OK')
"
```

Expected output: `all imports OK`

If you see `ModuleNotFoundError` or `ImportError`, the most likely cause is a missing import in one of the new files. Check the traceback — it will name the missing symbol and the file that needs it.

---

### Task 8: Run full semantics test suite

**Files:** none

- [ ] **Step 1: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/ -v --tb=short 2>&1 | tail -20
```

Expected: same pass count as Task 1 baseline. Zero new failures.

If any test fails with `ImportError`, trace the import chain — a concrete extractor file is likely missing an import that was previously provided by the shared top-level block in `features.py`.

---

### Task 9: Commit

- [ ] **Step 1: Stage and commit**

```bash
cd /workspace/collab-splats && \
  git add collab_splats/semantics/features/ && \
  git rm collab_splats/semantics/features.py && \
  git commit -m "$(cat <<'EOF'
refactor(semantics): split features.py into features/ package

Mirrors segmentation/ structure: base.py holds BaseFeatureExtractor +
BaseQueryableExtractor; one file per concrete backend (maskclip, dino,
talk2dino). Public API unchanged — semantics/__init__.py untouched.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
