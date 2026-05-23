# Loop Closure for Feedforward Point Cloud Derivation

**Date:** 2026-04-22  
**Status:** Draft  
**Branch:** refactor/core-modules

## Context

Feedforward methods (VGGT, MapAnything) derive camera poses from images in a single forward pass. Two problems:

1. **Drift:** Without global constraints, pose errors accumulate over long sequences — cameras that physically revisit a location end up misaligned.
2. **Scalability:** VGGT attention is O(N²). Long sequences (>30–40 images) hit GPU memory limits and produce noisier poses.

Loop closure addresses both. By detecting when the camera revisits a known location, we can add constraints that globally correct the pose trajectory. Decomposing sequences into overlapping submap windows also unlocks arbitrarily long captures.

This design follows the VGGT-SLAM architecture (MIT-SPARK/VGGT-SLAM) adapted to our feedforward pipeline's batch context.

---

## Architecture

Three new files alongside existing `feedforward.py` — no package restructure:

```
collab_splats/
├── semantics/
│   └── retrieval.py       # NEW — BaseRetrievalExtractor + DinoSaladExtractor
└── pointcloud/
    ├── feedforward.py     # MODIFIED — adds _loop_close() stage + enable_loop_closure flag to BaseFeedforwardCreator
    ├── submap.py          # NEW — Submap dataclass
    ├── loop_closure.py    # NEW — ImageRetrieval, LoopMatch, LoopMatchQueue
    └── pose_graph.py      # NEW — PoseGraph (GTSAM SL(4) wrapper)
```

`BaseFeedforwardCreator` template method gains a gated 5th stage:

```
_load_model → _preprocess → _forward (submap-chunked) → _loop_close → _postprocess → build_colmap
```

`_loop_close()` is a no-op when `enable_loop_closure=False` (default). Both `VGGTXCreator` and `MapAnythingCreator` inherit it without modification.

---

## Data Flow

```
image_dir (N images)
        │
        ▼
_preprocess() → views, image_paths, original_coords
        │
        ▼
_forward() [enable_loop_closure=True]
  ├─ chunk N images → overlapping Submap windows [0:K], [K-O:2K-O], ...
  ├─ for each window:
  │    ├─ VGGT forward → poses (K,4,4), intrinsics (K,3,3), pts3d (K,H,W,3)
  │    ├─ DinoSaladExtractor.forward(frames) → retrieval_vectors (K,D)
  │    └─ store as Submap
  ├─ ImageRetrieval.find_loop_closures(current_submap, past_submaps)
  │    └─ → List[LoopMatch]  (similarity_score < lc_threshold)
  ├─ for each LoopMatch:
  │    ├─ VGGT forward on (query_frame, detected_frame) pair
  │    ├─ check image_match_ratio > verify_match_ratio
  │    └─ if verified → LC Submap (is_lc_submap=True)
  ├─ PoseGraph.add_submaps() → BetweenFactorSL4 intra + inter-submap edges
  ├─ PoseGraph.add_loop_edges() → BetweenFactorSL4 on verified LC submaps
  ├─ PoseGraph.optimize() → Levenberg-Marquardt
  └─ flatten corrected poses → (N, 4, 4)
        │
        ▼
_postprocess() → FeedforwardResult (corrected extrinsics)
        │
        ▼
build_colmap() → PointcloudResult   [slices extrinsics[:, :3, :] at COLMAP boundary]
```

---

## Key Data Structures

```python
# collab_splats/pointcloud/submap.py
@dataclass
class Submap:
    submap_id: int
    frames: torch.Tensor           # (K, 3, H, W)
    poses: np.ndarray              # (K, 4, 4) world-to-cam homogeneous
    intrinsics: np.ndarray         # (K, 3, 3)
    pts3d: np.ndarray              # (K, H, W, 3)
    retrieval_vectors: torch.Tensor  # (K, D)
    image_paths: list[Path]
    is_lc_submap: bool = False

# collab_splats/pointcloud/loop_closure.py
LoopMatch = NamedTuple("LoopMatch", [
    ("similarity_score", float),
    ("query_submap_id", int),
    ("detected_submap_id", int),
    ("query_frame_idx", int),
    ("detected_frame_idx", int),
])

# collab_splats/pointcloud/loop_closure.py
@dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 4
    lc_threshold: float = 0.95
    max_loops_per_submap: int = 1
    verify_match_ratio: float = 0.85
```

### Pose Matrix Convention Change

`FeedforwardResult.extrinsics` changes from `(N, 3, 4)` to `(N, 4, 4)` homogeneous matrices (4th row always `[0, 0, 0, 1]`). This matches VGGT-SLAM's native format and GTSAM SL(4) expectations. COLMAP boundary converts via `extrinsics[:, :3, :]` slice.

---

## Component Details

### `BaseRetrievalExtractor` (`semantics/retrieval.py`)

Parallel to `BaseFeatureExtractor` but produces global descriptors, not dense maps:

```python
class BaseRetrievalExtractor(nn.Module, ABC):
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str): ...

    @classmethod
    def get(cls, name: str): ...

    @abstractmethod
    def forward(self, images: list) -> torch.Tensor:
        """Returns (N, D) normalized global descriptors."""
```

`DinoSaladExtractor` registers as `"dino-salad"`. Wraps the DINO-SALAD model (VLAD aggregation over DINOv2 patches) for visual place recognition.

### `ImageRetrieval` (`pointcloud/loop_closure.py`)

Ported from `vggt_slam/loop_closure.py`:

```python
class ImageRetrieval:
    def __init__(self, retrieval_model: str = "dino-salad", device: str = "cuda")
    def find_loop_closures(
        self,
        query_submap: Submap,
        past_submaps: list[Submap],
        lc_threshold: float,
        max_loops: int,
    ) -> list[LoopMatch]
```

Uses L2 distance on normalized `retrieval_vectors` (equivalent to `sqrt(2 - 2*cosine_similarity)` — lower = more similar). `similarity_score` in `LoopMatch` is this L2 distance; candidates with `similarity_score < lc_threshold` are loop closure candidates. `LoopMatchQueue` is a max-heap keeping only the top-k best (lowest distance) matches per submap.

### `PoseGraph` (`pointcloud/pose_graph.py`)

Ported from `vggt_slam/graph.py`. Wraps GTSAM SL(4):

```python
class PoseGraph:
    def __init__(self)
    def add_submaps(self, submaps: list[Submap]) -> None
    def add_loop_edges(self, lc_submaps: list[Submap]) -> None
    def optimize(self) -> dict[int, np.ndarray]  # submap_id → corrected poses (K, 4, 4)
```

Uses `BetweenFactorSL4` for relative constraints, `PriorFactorSL4` to anchor first frame.

### `BaseFeedforwardCreator` integration

```python
@dataclass
class BaseFeedforwardCreator(ABC):
    enable_loop_closure: bool = False
    loop_closure_config: LoopClosureConfig = field(default_factory=LoopClosureConfig)

    def _loop_close(
        self,
        submaps: list[Submap],
        lc_submaps: list[Submap],
    ) -> np.ndarray:  # (N, 4, 4) corrected poses
        """Optimizes pose graph given verified LC submaps. Detection + verification
        happen upstream in _run_loop_closure_inference."""
```

---

## Dependency

Add to `setup_feedforward.sh` and `pyproject.toml` optional deps:

```
gtsam-develop
```

SL(4) factors (`SL4`, `BetweenFactorSL4`, `PriorFactorSL4`) available in `gtsam-develop` PyPI package as of August 2025.

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| `N < submap_size` | Skip loop closure, single-pass forward |
| DINO-SALAD load fails | Log warning, skip loop closure |
| No loop candidates found | Optimize intra-submap only (normal case) |
| VGGT verification rejects all candidates | Same as above |
| GTSAM fails to converge | Log warning, return uncorrected per-submap poses |
| Only one submap produced | Skip loop closure |

No exceptions propagate out of `_loop_close()`.

---

## Verification

### Regression
- `enable_loop_closure=False` (default) must produce byte-identical output to current pipeline
- Existing `tests/pointcloud/test_feedforward_shared.py` passes unchanged

### Unit Tests
- Submap window chunking: correct indices and overlap for various N, K, O values
- `ImageRetrieval` similarity thresholding: mock embeddings, verify LoopMatch queue behavior  
- `PoseGraph` construction: add submaps + LC edges, verify GTSAM factor count
- `(4,4)` → COLMAP slice: `extrinsics[:, :3, :]` preserves correct `[R|t]`

### Integration Tests
- Synthetic loop sequence: repeated scene at frame 0 and frame N-1 — verify loop detected, final poses converge tighter than without loop closure
- Short sequence (N < submap_size): graceful skip, no crash, correct output

---

## Out of Scope

- Approach C (DINO-SALAD detection + COLMAP bundle adjustment) — bookmarked for future
- Standalone `submap_size` parameter without loop closure (scalability-only mode) — future work
- Streaming/online mode (all inference remains batch)
