# LC/BA Pipeline Refactor Design

**Date:** 2026-05-07
**Branch:** refactor/core-modules

## Problem

`BundleAdjustment` and `LoopClosure` logic is baked into creator constructors via boolean flags (`use_ba`, `enable_loop_closure`). This causes:

- BA logic duplicated across `VGGTXCreator._postprocess()` and `MapAnythingCreator._postprocess()`
- LC logic buried in `BaseFeedforwardCreator._run_loop_closure_inference()` — not independently testable
- Creator constructors polluted with orthogonal concerns
- Adding a new feedforward backend requires copy-pasting BA/LC logic

## Goals

1. Extract BA and LC into standalone composable wrapper classes
2. Creator constructors have zero BA/LC flags
3. BA and LC are independently testable
4. Backward-compat factory `make_creator()` for existing call sites

## Non-Goals

- Formal `Pipeline` class (see Future section)
- BA support for SfM creators (ColmapCreator, HlocCreator)
- Changing `run_bundle_adjustment` or `extract_tracks_vggsfm` signatures

## Architecture

**Pattern: decorator wrappers implementing `BasePointcloudCreator`.**

```
LoopClosure(base: BaseFeedforwardCreator, config: LoopClosureConfig)
BundleAdjustment(base: BasePointcloudCreator, config: BundleAdjustmentConfig)
```

`LoopClosure.base` is `BaseFeedforwardCreator` — needs `_forward`, `setup`, `_verify_loop_candidate`.
`BundleAdjustment.base` is `BasePointcloudCreator` — accepts any creator including `LoopClosure`; enforced at runtime via `result.images is not None` check.

Composition — LC wraps base creator, BA wraps outermost:

```python
# BA only
creator = BundleAdjustment(VGGTXCreator(...))

# LC + BA
creator = BundleAdjustment(
    LoopClosure(VGGTXCreator(...), config=LoopClosureConfig()),
)

# Convenience factory (kwargs forwarded to VGGTXCreator — model_name defaults to "facebook/VGGT-1B")
creator = make_creator("vggtx", use_lc=True, use_ba=True)
```

## Components

### `BundleAdjustmentConfig` (new dataclass)

Promotes existing `run_bundle_adjustment` kwargs into a typed struct:

```python
@dataclass
class BundleAdjustmentConfig:
    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
```

Lives in `collab_splats/pointcloud/bundle_adjustment.py` alongside `run_bundle_adjustment`.

### `FeedforwardResult` extension

BA needs raw images, conf, and world_points from the forward pass. These are added as optional fields:

```python
@dataclass
class FeedforwardResult(PointcloudResult):
    # existing fields unchanged ...
    images: "torch.Tensor | None" = None       # (N, 3, H, W) raw RGB
    conf: "torch.Tensor | None" = None          # (N, H, W) confidence
    world_points: "np.ndarray | None" = None    # (N, H, W, 3)
```

Each feedforward creator always populates these in `_postprocess()` — no flag required. `PointcloudResult` (base) is unchanged; SfM creators do not carry images.

### `LoopClosure` wrapper

Moves `_run_loop_closure_inference()` and all submap orchestration out of `BaseFeedforwardCreator`:

```python
class LoopClosure(BasePointcloudCreator):
    def __init__(
        self,
        base: BaseFeedforwardCreator,
        config: LoopClosureConfig | None = None,
    ):
        self.base = base
        self.config = config or LoopClosureConfig()

    def run(self, image_paths, **kwargs) -> FeedforwardResult:
        # 1. base.setup(image_paths) — load model, build views
        # 2. submap loop: base._forward(submap_views) per window
        # 3. detect loops via ImageRetrieval
        # 4. verify via self.base._verify_loop_candidate(f1, f2)
        # 5. merge with Sim3PoseGraph
        # 6. return merged FeedforwardResult
```

`_verify_loop_candidate` stays as an abstract method on `BaseFeedforwardCreator` — `LoopClosure` calls `self.base._verify_loop_candidate(f1, f2)` directly. No extra protocol needed.

`BaseFeedforwardCreator` loses: `enable_loop_closure`, `loop_closure_config`, `_run_loop_closure_inference()`. Keeps: `_verify_loop_candidate` (abstract), `_forward`, `setup`.

### `BundleAdjustment` wrapper

Moves BA logic out of each creator's `_postprocess()`:

```python
class BundleAdjustment(BasePointcloudCreator):
    def __init__(
        self,
        base: BasePointcloudCreator,
        config: BundleAdjustmentConfig | None = None,
    ):
        self.base = base
        self.config = config or BundleAdjustmentConfig()

    def run(self, image_paths, **kwargs) -> FeedforwardResult:
        result = self.base.run(image_paths, **kwargs)
        # result.images / result.conf / result.world_points populated by base
        tracks, vis_scores, _ = extract_tracks_vggsfm(
            result.images, result.conf, result.world_points
        )
        refined_pts, refined_ext, refined_intr = run_bundle_adjustment(
            ..., **dataclasses.asdict(self.config)
        )
        # reproject and return refined FeedforwardResult
```

Each creator's `_postprocess()` loses the `if self.use_ba:` block entirely. `use_ba` constructor param removed.

**Error handling:** raises `ValueError` if `result.images is None` (base is not a feedforward creator).

### `make_creator` factory

Backward-compat shim for callers using boolean flags:

```python
def make_creator(
    name: str,
    *,
    use_lc: bool = False,
    use_ba: bool = False,
    lc_config: LoopClosureConfig | None = None,
    ba_config: BundleAdjustmentConfig | None = None,
    **kwargs,
) -> BasePointcloudCreator:
    creator = get_creator(name)(**kwargs)
    if use_lc:
        creator = LoopClosure(creator, config=lc_config)
    if use_ba:
        creator = BundleAdjustment(creator, config=ba_config)
    return creator
```

`use_ba` / `enable_loop_closure` are **removed** from creator constructors — no deprecated shims. Direct constructor callers migrate to `make_creator` or explicit wrapping.

## Testing

- `test_bundle_adjustment.py` — unit test `BundleAdjustment` with a synthetic `FeedforwardResult`; mock `extract_tracks_vggsfm` to avoid GPU dependency
- `test_loop_closure.py` — migrate existing tests to `LoopClosure(MockCreator(...))` pattern; remove `enable_loop_closure=True` constructor usage
- `test_sim3_pose_graph.py` — unchanged (pure math)
- Existing BA integration tests stay; update constructor call style only

## Decision: Wrapper (Option B) vs. Pipeline class (Option A)

**Chose wrapper pattern (Option B).**

Option A (formal `Pipeline` class) pros:
- Explicit ordered step list, inspectable and serializable
- Single `.run()` entry point
- Familiar to sklearn users

Option A cons:
- Requires a container class + heterogeneous step protocol: step 1 takes `image_paths → FeedforwardResult`, subsequent steps take `FeedforwardResult → FeedforwardResult`
- 2–3 wrapping levels don't justify the abstraction

**When to migrate to Option A:**
- Steps exceed ~4 (e.g., depth filtering → LC → BA → outlier removal → mesh repair)
- Pipeline configs need YAML serialization / persistence
- Partial execution (fit/transform split) becomes a requirement

Migration path is straightforward: `Pipeline.__call__` chains each wrapper's `.run()` call. Data model (`FeedforwardResult` extension) is unchanged — no migration cost there.
