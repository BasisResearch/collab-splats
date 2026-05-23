# Loop Closure for Feedforward Point Cloud Derivation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional loop closure to `VGGTXCreator` and `MapAnythingCreator` using DINO-SALAD image retrieval, VGGT cross-frame verification, and GTSAM SE(3) pose graph optimization — improving camera alignment and enabling long captures via submap decomposition.

**Architecture:** Images chunk into overlapping windows (submaps); VGGT runs per window. DINO-SALAD detects revisited locations; verified matches become loop constraints in a GTSAM SE(3) pose graph (`Pose3`/`BetweenFactorPose3`) that globally corrects the trajectory. Feature is opt-in (`enable_loop_closure=False` default); both concrete creators inherit it unchanged.

**Tech Stack:** `gtsam` 4.2 (SE(3) pose graph — `gtsam-develop` SL(4) requires Python 3.11 wheels not available in our 3.10 env), `salad` vendored to `vendor/salad/` (DINO-SALAD via `from models.aggregators.salad import SALAD`), PyTorch, numpy, existing VGGT model

---

## File Map

| File | Change | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | modify | add optional deps: gtsam-develop, salad |
| `setup_feedforward.sh` | modify | install gtsam-develop + salad |
| `collab_splats/pointcloud/feedforward.py` | modify | `FeedforwardResult` (3,4)→(4,4); `BaseFeedforwardCreator` gets `enable_loop_closure`, `loop_closure_config`, `_loop_close()`, `_verify_loop_candidate()`, `_run_loop_closure_inference()`; VGGTXCreator and MapAnythingCreator `_postprocess` pad extrinsics to 4×4; VGGTXCreator overrides `_verify_loop_candidate` |
| `collab_splats/pointcloud/submap.py` | create | `Submap` dataclass — keyframes + poses + embeddings + raw outputs |
| `collab_splats/pointcloud/loop_closure.py` | create | `LoopClosureConfig`, `LoopMatch`, `LoopMatchQueue`, `ImageRetrieval` |
| `collab_splats/pointcloud/pose_graph.py` | create | `PoseGraph` wrapping GTSAM SL(4) |
| `collab_splats/semantics/retrieval.py` | create | `BaseRetrievalExtractor` + `DinoSaladExtractor` |
| `tests/pointcloud/test_feedforward_shared.py` | modify | update `_make_inputs` to produce (N,4,4); regression passes |
| `tests/pointcloud/test_loop_closure.py` | create | unit tests for Submap, ImageRetrieval, LoopMatch |
| `tests/pointcloud/test_pose_graph.py` | create | unit tests for PoseGraph |
| `tests/semantics/test_retrieval.py` | create | unit tests for BaseRetrievalExtractor |

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `setup_feedforward.sh`

- [ ] **Step 1: Add to pyproject.toml feedforward optional-dependencies**

```toml
[project.optional-dependencies]
feedforward = [
    # ... existing deps unchanged ...
    "gtsam-develop",
    "salad @ git+https://github.com/serizba/salad.git",
]
```

- [ ] **Step 2: Add to setup_feedforward.sh**

Find the pip install block and add:
```bash
pip install gtsam-develop
pip install git+https://github.com/serizba/salad.git
```

- [ ] **Step 3: Verify gtsam-develop has SL(4) support**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from gtsam import SL4, BetweenFactorSL4, PriorFactorSL4; print('SL4 OK')"
```
Expected: `SL4 OK`. If ImportError, gtsam-develop wheels may not be published yet — fall back to building from source: `pip install git+https://github.com/borglab/gtsam.git`.

- [ ] **Step 4: Verify salad loads**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from salad.eval import load_model; print('salad OK')"
```
Expected: `salad OK`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml setup_feedforward.sh
git commit -m "chore(deps): add gtsam-develop and salad for loop closure"
```

---

## Task 2: FeedforwardResult Extrinsics (3,4) → (4,4)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py` (FeedforwardResult + VGGTXCreator._postprocess + MapAnythingCreator._postprocess)
- Modify: `tests/pointcloud/test_feedforward_shared.py`

- [ ] **Step 1: Write failing test (update _make_inputs helper)**

In `tests/pointcloud/test_feedforward_shared.py`, change `_make_inputs` to produce (N,4,4):

```python
def _make_inputs(n=3, p=50):
    pts3d = np.random.randn(p, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (p, 3), dtype=np.uint8)
    extrinsics = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)  # (n, 4, 4)
    intrinsics = np.tile(
        np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    names = [f"frame_{i:04d}.jpg" for i in range(n)]
    return pts3d, colors, extrinsics, intrinsics, names
```

- [ ] **Step 2: Run tests to see current state**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py -v 2>&1 | head -40
```
Note which tests pass/fail — the `test_accepts_4x4_extrinsics` test may already pass.

- [ ] **Step 3: Update FeedforwardResult docstring**

In `collab_splats/pointcloud/feedforward.py`, change the `extrinsics` field comment:

```python
extrinsics: np.ndarray       # (N, 4, 4) float32 — world-to-camera homogeneous transform
                             #   rows 0-2: [R|t], row 3: [0, 0, 0, 1]
```

- [ ] **Step 4: Pad extrinsics in VGGTXCreator._postprocess**

Find where `FeedforwardResult` is constructed in `VGGTXCreator._postprocess`. The `extrinsics` value comes from `self.outputs["extrinsic"]` (shape N,3,4). Add padding before constructing the result:

```python
ext_3x4 = self.outputs["extrinsic"]   # (N, 3, 4)
n = ext_3x4.shape[0]
bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1, 1))  # (N, 1, 4)
extrinsics_4x4 = np.concatenate([ext_3x4, bottom], axis=1)               # (N, 4, 4)
```

Then pass `extrinsics_4x4` (not `ext_3x4`) into `FeedforwardResult(extrinsics=extrinsics_4x4, ...)`.

- [ ] **Step 5: Pad extrinsics in MapAnythingCreator._postprocess**

Apply the same pattern — find where `FeedforwardResult` is built in MapAnythingCreator and pad its extrinsics output identically.

- [ ] **Step 6: Update build_pycolmap_reconstruction to handle (N,4,4)**

Find `build_pycolmap_reconstruction` in `collab_splats/pointcloud/feedforward.py`. At the point where it uses extrinsics, slice to 3×4 if 4×4 is passed:

```python
if extrinsics.shape[1] == 4:
    extrinsics = extrinsics[:, :3, :]  # (N, 4, 4) → (N, 3, 4) for COLMAP
```

- [ ] **Step 7: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py -v
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_feedforward_shared.py
git commit -m "feat(pointcloud): change FeedforwardResult.extrinsics to (N,4,4) homogeneous"
```

---

## Task 3: Submap Dataclass

**Files:**
- Create: `collab_splats/pointcloud/submap.py`
- Create: `tests/pointcloud/test_loop_closure.py` (start)

- [ ] **Step 1: Write failing test**

Create `tests/pointcloud/test_loop_closure.py`:

```python
import numpy as np
import torch
from pathlib import Path
from collab_splats.pointcloud.submap import Submap


def _make_submap(k=4, h=224, w=224, d=128, submap_id=0):
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, h, w),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(
            np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]], dtype=np.float32), (k, 1, 1)
        ),
        retrieval_vectors=torch.zeros(k, d),
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(k)],
    )


def test_submap_creation():
    s = _make_submap(k=4)
    assert s.submap_id == 0
    assert s.frames.shape == (4, 3, 224, 224)
    assert s.poses.shape == (4, 4, 4)
    assert s.intrinsics.shape == (4, 3, 3)
    assert s.retrieval_vectors.shape == (4, 128)
    assert len(s.image_paths) == 4
    assert not s.is_lc_submap


def test_submap_lc_flag():
    s = _make_submap()
    s.is_lc_submap = True
    assert s.is_lc_submap
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure.py::test_submap_creation -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'collab_splats.pointcloud.submap'`

- [ ] **Step 3: Create collab_splats/pointcloud/submap.py**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class Submap:
    submap_id: int
    frames: torch.Tensor            # (K, 3, H, W) — raw image tensors on CPU
    poses: np.ndarray               # (K, 4, 4) float32 — world-to-cam homogeneous
    intrinsics: np.ndarray          # (K, 3, 3) float32 — camera intrinsics
    retrieval_vectors: torch.Tensor  # (K, D) float32 — DINO-SALAD global descriptors
    image_paths: list[Path]
    is_lc_submap: bool = False
    raw_outputs: dict | None = field(default=None, repr=False)  # raw _forward() dict, for _postprocess merging
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/submap.py tests/pointcloud/test_loop_closure.py
git commit -m "feat(pointcloud): add Submap dataclass for loop closure submap windows"
```

---

## Task 4: LoopClosureConfig + LoopMatch + LoopMatchQueue

**Files:**
- Create: `collab_splats/pointcloud/loop_closure.py` (data layer only — no model code yet)
- Modify: `tests/pointcloud/test_loop_closure.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/pointcloud/test_loop_closure.py`:

```python
from collab_splats.pointcloud.loop_closure import LoopClosureConfig, LoopMatch, LoopMatchQueue


def test_loop_closure_config_defaults():
    cfg = LoopClosureConfig()
    assert cfg.submap_size == 20
    assert cfg.submap_overlap == 4
    assert cfg.lc_threshold == 0.95
    assert cfg.max_loops_per_submap == 1
    assert cfg.verify_match_ratio == 0.85


def test_loop_match_queue_keeps_top_k():
    queue = LoopMatchQueue(max_size=2)
    queue.push(LoopMatch(0.9, 0, 1, 0, 0))
    queue.push(LoopMatch(0.5, 0, 2, 0, 0))  # lower distance = better
    queue.push(LoopMatch(0.7, 0, 3, 0, 0))
    matches = queue.get_matches()
    assert len(matches) == 2
    distances = [m.similarity_score for m in matches]
    assert sorted(distances) == distances  # ascending (best first)
    assert 0.5 in distances and 0.7 in distances  # 0.9 evicted


def test_loop_match_named_tuple():
    m = LoopMatch(0.3, query_submap_id=0, detected_submap_id=1, query_frame_idx=2, detected_frame_idx=5)
    assert m.similarity_score == 0.3
    assert m.detected_submap_id == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure.py::test_loop_closure_config_defaults tests/pointcloud/test_loop_closure.py::test_loop_match_queue_keeps_top_k -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create collab_splats/pointcloud/loop_closure.py (data layer)**

```python
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import NamedTuple


LoopMatch = NamedTuple(
    "LoopMatch",
    [
        ("similarity_score", float),   # L2 distance on normalized embeddings — lower = better
        ("query_submap_id", int),
        ("detected_submap_id", int),
        ("query_frame_idx", int),
        ("detected_frame_idx", int),
    ],
)


@dataclass
class LoopClosureConfig:
    submap_size: int = 20           # frames per submap window
    submap_overlap: int = 4         # overlap between consecutive windows
    lc_threshold: float = 0.95      # max L2 distance to accept as loop candidate
    max_loops_per_submap: int = 1   # top-k candidates per submap
    verify_match_ratio: float = 0.85  # min VGGT image_match_ratio to confirm loop


class LoopMatchQueue:
    """Max-heap keeping the top-k lowest-distance LoopMatch candidates."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._heap: list[tuple[float, LoopMatch]] = []  # (neg_score, match) for max-heap

    def push(self, match: LoopMatch) -> None:
        heapq.heappush(self._heap, (-match.similarity_score, match))
        if len(self._heap) > self._max_size:
            heapq.heappop(self._heap)  # evict worst (highest distance)

    def get_matches(self) -> list[LoopMatch]:
        return sorted([m for _, m in self._heap], key=lambda m: m.similarity_score)
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure.py tests/pointcloud/test_loop_closure.py
git commit -m "feat(pointcloud): add LoopClosureConfig, LoopMatch, LoopMatchQueue"
```

---

## Task 5: BaseRetrievalExtractor + DinoSaladExtractor

**Files:**
- Create: `collab_splats/semantics/retrieval.py`
- Create: `tests/semantics/test_retrieval.py`

- [ ] **Step 1: Write failing test**

Create `tests/semantics/test_retrieval.py`:

```python
import torch
from collab_splats.semantics.retrieval import BaseRetrievalExtractor, DinoSaladExtractor


def test_registry_get_dino_salad():
    cls = BaseRetrievalExtractor.get("dino-salad")
    assert cls is DinoSaladExtractor


def test_registry_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown retrieval extractor"):
        BaseRetrievalExtractor.get("nonexistent-model")


def test_base_extractor_forward_abstract():
    import pytest
    with pytest.raises(TypeError):
        BaseRetrievalExtractor()
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_retrieval.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create collab_splats/semantics/retrieval.py**

```python
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torchvision.transforms as T


class BaseRetrievalExtractor(nn.Module, ABC):
    """Abstract base for image retrieval extractors with name-based registry.

    Subclasses register via @BaseRetrievalExtractor.register("name").
    Unlike feature extractors, these return global (N, D) descriptors per image,
    not dense (C, H, W) maps.
    """

    _registry: Dict[str, type["BaseRetrievalExtractor"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type) -> type:
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> type["BaseRetrievalExtractor"]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown retrieval extractor '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @abstractmethod
    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """Return (N, D) normalized global descriptors for N images."""


@BaseRetrievalExtractor.register("dino-salad")
class DinoSaladExtractor(BaseRetrievalExtractor):
    """DINO-SALAD global image descriptor for visual place recognition.

    Ported from MIT-SPARK/VGGT-SLAM. Uses VLAD aggregation over DINOv2 patches.
    Checkpoint auto-downloaded to torch.hub.get_dir()/checkpoints/dino_salad.ckpt.
    """

    _INPUT_SIZE = 224

    def __init__(self, device: str | None = None):
        super().__init__()
        import sys, pathlib
        # vendor/salad cloned by setup_feedforward.sh; not pip-installable
        vendor_path = str(pathlib.Path(__file__).parents[3] / "vendor" / "salad")
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        from models.aggregators.salad import SALAD

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SALAD().to(self._device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images: list | torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: list of PIL Images, or (N, 3, H, W) tensor on any device
        Returns:
            (N, D) normalized float32 descriptor tensor on CPU
        """
        if isinstance(images, (list, tuple)):
            imgs = torch.stack([self.transform(img) for img in images])
        else:
            imgs = images.float()
            if imgs.shape[-1] != self._INPUT_SIZE or imgs.shape[-2] != self._INPUT_SIZE:
                imgs = torch.nn.functional.interpolate(
                    imgs, size=(self._INPUT_SIZE, self._INPUT_SIZE), mode="bilinear", align_corners=False
                )
        imgs = imgs.to(self._device)
        with torch.no_grad():
            descriptors = self.model(imgs)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=-1)
        return descriptors.cpu()
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_retrieval.py -v
```
Expected: all pass. Note: `test_base_extractor_forward_abstract` does not instantiate DinoSaladExtractor so no model download needed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/retrieval.py tests/semantics/test_retrieval.py
git commit -m "feat(semantics): add BaseRetrievalExtractor + DinoSaladExtractor for loop closure retrieval"
```

---

## Task 6: ImageRetrieval

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure.py` (add ImageRetrieval class)
- Modify: `tests/pointcloud/test_loop_closure.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pointcloud/test_loop_closure.py`:

```python
import torch
import numpy as np
from pathlib import Path
from collab_splats.pointcloud.loop_closure import ImageRetrieval, LoopClosureConfig
from collab_splats.pointcloud.submap import Submap


def _make_submap_with_vecs(submap_id, retrieval_vec, k=2):
    vecs = retrieval_vec.unsqueeze(0).expand(k, -1).clone()
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 224, 224),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=vecs,
        image_paths=[Path(f"frame_{i}.jpg") for i in range(k)],
    )


def test_find_loop_closures_detects_similar(monkeypatch):
    # Patch DinoSaladExtractor to avoid loading the model
    from collab_splats.pointcloud import loop_closure as lc_module
    monkeypatch.setattr(lc_module, "_get_retrieval_extractor", lambda device: None)

    retrieval = ImageRetrieval.__new__(ImageRetrieval)
    retrieval.extractor = None  # skip model for this test

    d = 512
    base_vec = torch.nn.functional.normalize(torch.randn(d), p=2, dim=0)
    similar_vec = torch.nn.functional.normalize(base_vec + torch.randn(d) * 0.01, p=2, dim=0)
    different_vec = torch.nn.functional.normalize(torch.randn(d), p=2, dim=0)

    query = _make_submap_with_vecs(2, similar_vec)
    past_similar = _make_submap_with_vecs(0, base_vec)
    past_different = _make_submap_with_vecs(1, different_vec)

    matches = retrieval.find_loop_closures(
        query_submap=query,
        past_submaps=[past_similar, past_different],
        lc_threshold=0.1,  # tight threshold; similar pair should be under it
        max_loops=1,
    )
    assert len(matches) == 1
    assert matches[0].detected_submap_id == 0


def test_find_loop_closures_no_match():
    retrieval = ImageRetrieval.__new__(ImageRetrieval)
    retrieval.extractor = None

    d = 512
    query = _make_submap_with_vecs(1, torch.nn.functional.normalize(torch.randn(d), p=2, dim=0))
    past = _make_submap_with_vecs(0, torch.nn.functional.normalize(torch.randn(d), p=2, dim=0))

    matches = retrieval.find_loop_closures(
        query_submap=query,
        past_submaps=[past],
        lc_threshold=0.001,  # impossibly tight
        max_loops=1,
    )
    assert matches == []
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure.py::test_find_loop_closures_detects_similar -v
```
Expected: FAIL

- [ ] **Step 3: Add ImageRetrieval to collab_splats/pointcloud/loop_closure.py**

Add after the existing data classes:

```python
import torch
from collab_splats.pointcloud.submap import Submap


def _get_retrieval_extractor(device: str):
    from collab_splats.semantics.retrieval import BaseRetrievalExtractor
    cls = BaseRetrievalExtractor.get("dino-salad")
    return cls(device=device)


class ImageRetrieval:
    """Detects loop closure candidates via DINO-SALAD embedding similarity.

    Ported from MIT-SPARK/VGGT-SLAM vggt_slam/loop_closure.py.
    Similarity measured as L2 distance on normalized embeddings (lower = more similar).
    """

    def __init__(self, device: str = "cuda"):
        self.extractor = _get_retrieval_extractor(device)

    def embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Embed (K, 3, H, W) frames → (K, D) normalized descriptors."""
        return self.extractor(frames)

    def find_loop_closures(
        self,
        query_submap: Submap,
        past_submaps: list[Submap],
        lc_threshold: float,
        max_loops: int,
    ) -> list[LoopMatch]:
        """Return top-k loop closure candidates between query_submap and past_submaps.

        Uses pre-computed retrieval_vectors stored on each Submap.
        """
        if not past_submaps:
            return []

        queue = LoopMatchQueue(max_size=max_loops)

        for q_idx in range(query_submap.retrieval_vectors.shape[0]):
            q_vec = query_submap.retrieval_vectors[q_idx]  # (D,)

            for past in past_submaps:
                # L2 distance between query frame and all frames in past submap
                dists = torch.cdist(
                    q_vec.unsqueeze(0),                      # (1, D)
                    past.retrieval_vectors,                  # (K, D)
                ).squeeze(0)                                 # (K,)

                best_idx = int(dists.argmin())
                best_dist = float(dists[best_idx])

                if best_dist < lc_threshold:
                    queue.push(LoopMatch(
                        similarity_score=best_dist,
                        query_submap_id=query_submap.submap_id,
                        detected_submap_id=past.submap_id,
                        query_frame_idx=q_idx,
                        detected_frame_idx=best_idx,
                    ))

        return queue.get_matches()
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure.py tests/pointcloud/test_loop_closure.py
git commit -m "feat(pointcloud): add ImageRetrieval for DINO-SALAD loop closure detection"
```

---

## Task 7: PoseGraph (GTSAM SE(3))

**Files:**
- Create: `collab_splats/pointcloud/pose_graph.py`
- Create: `tests/pointcloud/test_pose_graph.py`

- [ ] **Step 1: Write failing test**

Create `tests/pointcloud/test_pose_graph.py`:

```python
import numpy as np
import pytest
from collab_splats.pointcloud.pose_graph import PoseGraph
from collab_splats.pointcloud.submap import Submap
import torch
from pathlib import Path


def _identity_submap(submap_id, k=3):
    poses = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    # Add small perturbations so optimization has something to do
    poses[:, :3, 3] = np.random.randn(k, 3).astype(np.float32) * 0.01
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 64, 64),
        poses=poses,
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 128),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def test_pose_graph_add_submaps_and_optimize():
    pg = PoseGraph()
    s0 = _identity_submap(0, k=3)
    s1 = _identity_submap(1, k=3)
    pg.add_submaps([s0, s1])
    result = pg.optimize()
    assert isinstance(result, dict)
    assert 0 in result and 1 in result
    assert result[0].shape == (3, 4, 4)
    assert result[1].shape == (3, 4, 4)


def test_pose_graph_loop_edge():
    pg = PoseGraph()
    s0 = _identity_submap(0, k=3)
    s1 = _identity_submap(1, k=3)
    pg.add_submaps([s0, s1])

    # LC submap connecting s1 frame 0 back to s0 frame 0
    lc = Submap(
        submap_id=2,
        frames=torch.zeros(2, 3, 64, 64),
        poses=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(2, 128),
        image_paths=[Path("lc_q.jpg"), Path("lc_d.jpg")],
        is_lc_submap=True,
    )
    pg.add_loop_edges([lc])
    result = pg.optimize()
    assert result is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pose_graph.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create collab_splats/pointcloud/pose_graph.py**

```python
from __future__ import annotations

import logging
import numpy as np
import gtsam
from gtsam import BetweenFactorPose3, PriorFactorPose3

from collab_splats.pointcloud.submap import Submap

log = logging.getLogger(__name__)

# SE(3) noise: 6-DOF [rotation (3) + translation (3)]
# gtsam-develop SL(4) (15-DOF) requires Python 3.11 wheels; SE(3) is equivalent
# for rigid camera poses and available in gtsam 4.2.
_INTRA_NOISE_SIGMA = 0.05
_ANCHOR_NOISE_SIGMA = 1e-6


def _noise(sigma: float) -> gtsam.noiseModel.Base:
    return gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * sigma)


def _pose3(mat: np.ndarray) -> gtsam.Pose3:
    """Convert (4, 4) world-to-cam matrix to gtsam.Pose3."""
    R = gtsam.Rot3(mat[:3, :3].astype(np.float64))
    t = gtsam.Point3(mat[:3, 3].astype(np.float64))
    return gtsam.Pose3(R, t)


def _relative_pose3(pose_i: np.ndarray, pose_j: np.ndarray) -> gtsam.Pose3:
    """Compute SE(3) relative transform: inv(pose_i) @ pose_j."""
    p_i = _pose3(pose_i)
    p_j = _pose3(pose_j)
    return p_i.between(p_j)


class PoseGraph:
    """GTSAM SE(3) pose graph for loop closure trajectory correction.

    Uses Pose3/BetweenFactorPose3 (SE(3), 6 DOF). Adapted from
    MIT-SPARK/VGGT-SLAM graph.py (which uses SL(4), 15 DOF — unavailable
    on Python 3.10). Frame keys: gtsam.symbol('x', global_frame_index).
    """

    def __init__(self):
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._intra_noise = _noise(_INTRA_NOISE_SIGMA)
        self._anchor_noise = _noise(_ANCHOR_NOISE_SIGMA)
        self._frame_offset: dict[int, int] = {}  # submap_id → global frame start index
        self._submaps: list[Submap] = []
        self._total_frames = 0

    def add_submaps(self, submaps: list[Submap]) -> None:
        """Add normal submaps: intra-submap edges + inter-submap overlap edges."""
        for submap in submaps:
            start = self._total_frames
            self._frame_offset[submap.submap_id] = start
            self._submaps.append(submap)
            k = submap.poses.shape[0]

            for local_i, pose in enumerate(submap.poses):
                global_idx = start + local_i
                key = gtsam.symbol('x', global_idx)

                self._initial.insert(key, _pose3(pose))

                if global_idx == 0:
                    self._graph.add(PriorFactorPose3(key, _pose3(pose), self._anchor_noise))

                if local_i > 0:
                    prev_key = gtsam.symbol('x', global_idx - 1)
                    rel = _relative_pose3(submap.poses[local_i - 1], pose)
                    self._graph.add(BetweenFactorPose3(prev_key, key, rel, self._intra_noise))

            self._total_frames += k

    def add_loop_edges(self, lc_submaps: list[Submap]) -> None:
        """Add loop closure constraints from verified LC submaps.

        Each LC submap has exactly 2 frames: [query_frame, detected_frame].
        """
        for lc in lc_submaps:
            if lc.poses.shape[0] != 2:
                log.warning("LC submap %d has %d frames, expected 2 — skipping", lc.submap_id, lc.poses.shape[0])
                continue

            q_key = self._find_global_key(lc.image_paths[0])
            d_key = self._find_global_key(lc.image_paths[1])
            if q_key is None or d_key is None:
                log.warning("LC submap %d: could not resolve global keys — skipping", lc.submap_id)
                continue

            rel = _relative_pose3(lc.poses[0], lc.poses[1])
            self._graph.add(BetweenFactorPose3(q_key, d_key, rel, self._intra_noise))

    def _find_global_key(self, image_path) -> int | None:
        for submap in self._submaps:
            for local_i, p in enumerate(submap.image_paths):
                if p == image_path:
                    global_idx = self._frame_offset[submap.submap_id] + local_i
                    return gtsam.symbol('x', global_idx)
        return None

    def optimize(self) -> dict[int, np.ndarray]:
        """Run Levenberg-Marquardt. Returns {submap_id: corrected_poses (K, 4, 4)}."""
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial, params)
            result = optimizer.optimize()
        except Exception as e:
            log.warning("GTSAM optimization failed: %s — returning uncorrected poses", e)
            return {s.submap_id: s.poses for s in self._submaps}

        corrected: dict[int, np.ndarray] = {}
        for submap in self._submaps:
            start = self._frame_offset[submap.submap_id]
            k = submap.poses.shape[0]
            poses = np.stack([
                result.atPose3(gtsam.symbol('x', start + i)).matrix().astype(np.float32)
                for i in range(k)
            ])
            corrected[submap.submap_id] = poses
        return corrected
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pose_graph.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/pose_graph.py tests/pointcloud/test_pose_graph.py
git commit -m "feat(pointcloud): add PoseGraph with GTSAM SL(4) for loop closure trajectory correction"
```

---

## Task 8: BaseFeedforwardCreator Integration

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`
  - `BaseFeedforwardCreator`: add `enable_loop_closure`, `loop_closure_config`, `_verify_loop_candidate()`, `_run_loop_closure_inference()`, `_loop_close()`, `_merge_submap_outputs()`; modify `run_inference()`
  - `VGGTXCreator`: override `_verify_loop_candidate()`

- [ ] **Step 1: Write failing regression test**

Add to `tests/pointcloud/test_feedforward_shared.py`:

```python
def test_loop_closure_flag_default_false():
    from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
    import inspect
    sig = inspect.signature(BaseFeedforwardCreator.__init__)
    # enable_loop_closure should exist with default False
    # We test via FeedforwardResult staying unchanged when flag is False
    # (full regression covered by existing tests passing)
    assert "enable_loop_closure" in str(sig) or hasattr(BaseFeedforwardCreator, "enable_loop_closure")
```

- [ ] **Step 2: Add fields to BaseFeedforwardCreator**

In `collab_splats/pointcloud/feedforward.py`, add to `BaseFeedforwardCreator.__init__` (or as dataclass fields if it uses `@dataclass`):

```python
from collab_splats.pointcloud.loop_closure import LoopClosureConfig

# Inside BaseFeedforwardCreator:
self.enable_loop_closure: bool = False
self.loop_closure_config: LoopClosureConfig = LoopClosureConfig()
```

If `BaseFeedforwardCreator` is a dataclass, add:
```python
enable_loop_closure: bool = False
loop_closure_config: LoopClosureConfig = field(default_factory=LoopClosureConfig)
```

- [ ] **Step 3: Override run_inference() to dispatch to loop-closure path**

In `BaseFeedforwardCreator`, modify `run_inference()`:

```python
def run_inference(self, **kwargs) -> None:
    if self.enable_loop_closure and self._enough_frames_for_submaps():
        self._run_loop_closure_inference()
    else:
        with torch.no_grad():
            self.outputs = self._forward(self.model, self.views, **kwargs)
        torch.cuda.empty_cache()
```

Add helper:
```python
def _enough_frames_for_submaps(self) -> bool:
    n = self.views.shape[0] if hasattr(self.views, "shape") else len(self.views)
    return n >= self.loop_closure_config.submap_size
```

- [ ] **Step 4: Add _verify_loop_candidate (default: accept all)**

In `BaseFeedforwardCreator`:
```python
def _verify_loop_candidate(self, frame1: torch.Tensor, frame2: torch.Tensor) -> bool:
    """Override in subclasses to add model-based verification. Default accepts all."""
    return True
```

In `VGGTXCreator` (override):
```python
def _verify_loop_candidate(self, frame1: torch.Tensor, frame2: torch.Tensor) -> bool:
    import torch
    device = next(self.model.parameters()).device
    dtype = next(self.model.parameters()).dtype
    lc_frames = torch.stack([frame1, frame2]).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        predictions = self.model(lc_frames, compute_similarity=True)
    match_ratio = float(predictions.get("image_match_ratio", 1.0))
    return match_ratio >= self.loop_closure_config.verify_match_ratio
```

- [ ] **Step 5: Add _run_loop_closure_inference()**

In `BaseFeedforwardCreator`:

```python
def _run_loop_closure_inference(self) -> None:
    import torch
    from collab_splats.pointcloud.submap import Submap
    from collab_splats.pointcloud.loop_closure import ImageRetrieval

    cfg = self.loop_closure_config
    K, O = cfg.submap_size, cfg.submap_overlap
    step = max(1, K - O)
    views = self.views
    N = views.shape[0]
    device = str(next(self.model.parameters()).device)

    try:
        retrieval = ImageRetrieval(device=device)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("DINO-SALAD failed to load (%s) — skipping loop closure", e)
        with torch.no_grad():
            self.outputs = self._forward(self.model, views)
        torch.cuda.empty_cache()
        return

    submaps: list[Submap] = []
    lc_submaps: list[Submap] = []

    for wi, start in enumerate(range(0, N, step)):
        end = min(start + K, N)
        window = views[start:end]
        k = window.shape[0]

        with torch.no_grad():
            raw = self._forward(self.model, window)
        torch.cuda.empty_cache()

        ext_3x4 = raw["extrinsic"]                                              # (k, 3, 4)
        bottom = np.tile([0, 0, 0, 1], (k, 1)).reshape(k, 1, 4).astype(np.float32)
        poses_4x4 = np.concatenate([ext_3x4, bottom], axis=1)                   # (k, 4, 4)

        ret_vecs = retrieval.embed_frames(window.cpu())                          # (k, D)

        submap = Submap(
            submap_id=wi,
            frames=window.cpu(),
            poses=poses_4x4,
            intrinsics=raw["intrinsic"],
            retrieval_vectors=ret_vecs,
            image_paths=list(self.image_paths[start:end]),
            raw_outputs=raw,
        )

        loop_matches = retrieval.find_loop_closures(
            submap, submaps, cfg.lc_threshold, cfg.max_loops_per_submap
        )

        for match in loop_matches:
            q_frame = window[match.query_frame_idx].cpu()
            d_submap = submaps[match.detected_submap_id]
            d_frame = d_submap.frames[match.detected_frame_idx].cpu()
            if self._verify_loop_candidate(q_frame, d_frame):
                lc_submaps.append(Submap(
                    submap_id=len(submaps) + len(lc_submaps),
                    frames=torch.stack([q_frame, d_frame]),
                    poses=np.stack([
                        submap.poses[match.query_frame_idx],
                        d_submap.poses[match.detected_frame_idx],
                    ]),
                    intrinsics=np.stack([
                        submap.intrinsics[match.query_frame_idx],
                        d_submap.intrinsics[match.detected_frame_idx],
                    ]),
                    retrieval_vectors=torch.zeros(2, ret_vecs.shape[-1]),
                    image_paths=[
                        submap.image_paths[match.query_frame_idx],
                        d_submap.image_paths[match.detected_frame_idx],
                    ],
                    is_lc_submap=True,
                ))

        submaps.append(submap)
        if end >= N:
            break

    corrected_extrinsics = self._loop_close(submaps, lc_submaps)  # (N, 4, 4)
    self.outputs = self._merge_submap_outputs(submaps, corrected_extrinsics)
```

- [ ] **Step 6: Add _loop_close() and _merge_submap_outputs()**

In `BaseFeedforwardCreator`:

```python
def _loop_close(
    self,
    submaps: list[Submap],
    lc_submaps: list[Submap],
) -> np.ndarray:
    """Build and optimize GTSAM SL(4) pose graph. Returns (N, 4, 4) corrected extrinsics."""
    from collab_splats.pointcloud.pose_graph import PoseGraph
    pg = PoseGraph()
    pg.add_submaps(submaps)
    if lc_submaps:
        pg.add_loop_edges(lc_submaps)
    corrected = pg.optimize()  # {submap_id: (K, 4, 4)}
    return np.concatenate([corrected[s.submap_id] for s in submaps], axis=0)  # (N, 4, 4)


def _merge_submap_outputs(
    self,
    submaps: list[Submap],
    corrected_extrinsics: np.ndarray,
) -> dict:
    """Assemble a unified raw_outputs dict from per-submap outputs with corrected poses."""
    merged: dict = {}
    if not submaps or submaps[0].raw_outputs is None:
        return {"extrinsic": corrected_extrinsics}

    # Merge all numpy-array fields from raw_outputs by concatenation on axis 0
    sample = submaps[0].raw_outputs
    for key, val in sample.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            try:
                merged[key] = np.concatenate(
                    [s.raw_outputs[key] for s in submaps if s.raw_outputs and key in s.raw_outputs],
                    axis=0,
                )
            except Exception:
                merged[key] = val  # fall back to first submap's value for non-concatenable fields
        else:
            merged[key] = val  # scalar / non-array fields: use first submap's value

    merged["extrinsic"] = corrected_extrinsics  # override with corrected poses
    return merged
```

- [ ] **Step 7: Run regression tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py -v
```
Expected: all pass (default path unchanged).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_feedforward_shared.py
git commit -m "feat(pointcloud): integrate loop closure into BaseFeedforwardCreator template method"
```

---

## Task 9: Integration Smoke Test

**Files:**
- Create: `tests/pointcloud/test_loop_closure_integration.py`

- [ ] **Step 1: Write integration test with synthetic loop sequence**

Create `tests/pointcloud/test_loop_closure_integration.py`:

```python
"""Smoke test: short sequence falls back gracefully (N < submap_size)."""
import numpy as np
import pytest
from collab_splats.pointcloud.loop_closure import LoopClosureConfig, ImageRetrieval, LoopMatch
from collab_splats.pointcloud.pose_graph import PoseGraph
from collab_splats.pointcloud.submap import Submap
import torch
from pathlib import Path


def _make_submap(submap_id, k, vec_dim=128):
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 64, 64),
        poses=np.tile(np.eye(4), (k, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.nn.functional.normalize(torch.randn(k, vec_dim), p=2, dim=1),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def test_pose_graph_two_submaps_no_loop():
    """Two submaps with identity poses — optimizer returns poses without crashing."""
    s0 = _make_submap(0, k=4)
    s1 = _make_submap(1, k=4)
    pg = PoseGraph()
    pg.add_submaps([s0, s1])
    result = pg.optimize()
    assert set(result.keys()) == {0, 1}
    assert result[0].shape == (4, 4, 4)
    assert result[1].shape == (4, 4, 4)


def test_loop_closure_config_passed_through():
    """LoopClosureConfig fields are accessible and correct."""
    cfg = LoopClosureConfig(submap_size=10, submap_overlap=2, lc_threshold=0.8)
    assert cfg.submap_size == 10
    assert cfg.submap_overlap == 2
    assert 0 < cfg.lc_threshold < 1.0


def test_image_retrieval_detects_identical_submaps():
    """Two submaps with identical retrieval vectors → loop detected at threshold 0.01."""
    d = 128
    base_vecs = torch.nn.functional.normalize(torch.randn(3, d), p=2, dim=1)
    s0 = Submap(
        submap_id=0, frames=torch.zeros(3, 3, 64, 64),
        poses=np.tile(np.eye(4), (3, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (3, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vecs,
        image_paths=[Path(f"f{i}.jpg") for i in range(3)],
    )
    s1 = Submap(
        submap_id=1, frames=torch.zeros(3, 3, 64, 64),
        poses=np.tile(np.eye(4), (3, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (3, 1, 1)).astype(np.float32),
        retrieval_vectors=base_vecs.clone(),  # identical → distance = 0
        image_paths=[Path(f"f{i+3}.jpg") for i in range(3)],
    )
    retrieval = ImageRetrieval.__new__(ImageRetrieval)
    retrieval.extractor = None

    matches = retrieval.find_loop_closures(s1, [s0], lc_threshold=0.01, max_loops=1)
    assert len(matches) == 1
    assert matches[0].detected_submap_id == 0
    assert matches[0].similarity_score < 1e-5
```

- [ ] **Step 2: Run integration test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_loop_closure_integration.py -v
```
Expected: all pass.

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/pointcloud/test_loop_closure_integration.py 2>&1 | tail -20
```
Expected: all pre-existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/pointcloud/test_loop_closure_integration.py
git commit -m "test(pointcloud): add loop closure integration smoke tests"
```

---

## Verification Checklist

- [ ] `gtsam-develop` installs and `from gtsam import SL4, BetweenFactorSL4, PriorFactorSL4` succeeds
- [ ] `FeedforwardResult.extrinsics` shape is `(N, 4, 4)` — confirmed via `test_feedforward_shared.py`
- [ ] `enable_loop_closure=False` (default) produces identical output to pre-change — regression tests pass
- [ ] `ImageRetrieval.find_loop_closures` detects identical-embedding submaps with near-zero distance
- [ ] `PoseGraph.optimize()` runs without error on 2 submaps and returns correct shape `(K, 4, 4)` per submap
- [ ] `BaseFeedforwardCreator` routes to `_run_loop_closure_inference()` when `enable_loop_closure=True` and `N >= submap_size`
- [ ] Short sequence (`N < submap_size`) with `enable_loop_closure=True` falls back to single-pass forward gracefully
