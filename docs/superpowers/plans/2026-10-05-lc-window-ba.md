# lc-window-ba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run bundle adjustment inside each loop-closure window as a package feature, replacing the scratch hook `lc_inner_light.py`, so the perf-1k gates run on package code.

**Architecture:** `LoopClosure` takes an optional `BundleAdjustmentConfig`. The existing `_forward_window(window, start)` runs the forward, then (when `ba` is set) the window solve inline: no new function or method. The reconstructor stops refusing BA + LC, passes the BA config into `LoopClosure`, skips the `refine` stage under LC, and stores the per-window records in the zarr attrs.

**Tech Stack:** Python 3.11, numpy, torch, bae/pypose BA (`collab_splats.geometry.bundle_adjustment`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-lc-window-ba-design.md` (approved 2026-10-05).

---

## Ground rules for every task

- Worktree: `/workspace/collab-splats/.worktrees/rgbd-ba-cf`, branch `feat/rgbd-ba-cf`. Never `cd` to the main checkout.
- Python: `/opt/venv/reconstruction/bin/python` (aliased `$PY` below).
- Test command shape (bae 0.2.5 lives on PYTHONPATH; never pipe to `tail`):
  ```bash
  cd /workspace/collab-splats/.worktrees/rgbd-ba-cf && S=/tmp/claude-0/-workspace-collab-splats/d91a918d-4149-421d-8686-7bc7e948176f/scratchpad && PYTHONPATH=$S/bae025:$PWD /opt/venv/reconstruction/bin/python -m pytest <tests> -p no:cacheprovider -q; echo EXIT=$?
  ```
- Format only the files you touched: `/opt/venv/reconstruction/bin/python -m black -l 120 --target-version py311 <files>` then `/opt/venv/reconstruction/bin/python -m isort <files>`. Never repo-wide.
- Code style (CLAUDE.md + memory): absolute imports, imports at file top; one-line block comments; blank line above every block comment and around every `if/for/with/try`; no nested calls (`f(g(x))` → two lines); US spelling; `logging`, not `print`; docstrings `"""` on their own lines, one-line summary then `- ` bullets, private defs have no `Args:`/`Returns:`.
- Commit with `git commit --only <paths>`, conventional message, trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Files under `docs/superpowers/` need `git add -f` first. Never amend, reset, stash, or rebase.
- Never edit site-packages, `third_party/`, or anything under `.worktrees/rgbd-ba`.
- Do not run GPU jobs; unit tests only (BA is monkeypatched in them).

## File map

| File | Change |
|---|---|
| `collab_splats/geometry/loop_closure/wrapper.py` | `ba` param, `window_ba` / `_ba_focal` state, window solve inside `_forward_window`, `start` passed at both submits, both fallbacks routed through `_forward_window` |
| `tests/geometry/loop_closure/test_window_ba.py` | new: window-BA unit tests |
| `tests/geometry/loop_closure/test_wrapper.py` | `__new__`-built wrappers get `wrapper.ba = None`; too-few-frames fallback test updated |
| `collab_splats/reconstructor.py` | refusal removed, stage set, `refine()` guard, `pointcloud()` wiring + `window_ba` attrs |
| `tests/reconstructor/test_refine_stage.py`, `tests/reconstructor/test_reconstructor.py` | refusal tests become acceptance tests; new stage / guard / attrs tests; viewer test's `LoopClosure` call assertion gains `ba=None` |
| `docs/superpowers/decisions/022-lc-window-ba.md` | new decision |
| `configs/README.md`, `configs/base.yaml`, `docs/pointcloud.md` | BA-with-LC meaning |

---

### Task 1: LoopClosure window BA

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py` (imports; `__init__` ~113-141; `run_inference` ~189-200; `_forward_window` ~414-421; DINO-SALAD fallback ~566-575; both `pool.submit` calls ~600 and ~612)
- Create: `tests/geometry/loop_closure/test_window_ba.py`
- Modify: `tests/geometry/loop_closure/test_wrapper.py` (6 `LoopClosure.__new__` sites; test at ~188)

Background the implementer needs:

- `LoopClosure.__getattr__` delegates missing attributes to `self.base`. A wrapper built with `LoopClosure.__new__` (as 6 tests in `test_wrapper.py` do) and a `MagicMock` base would see `self.ba` as a truthy `MagicMock` and run BA. Every such test must set `wrapper.ba = None`.
- Raw forward dicts: VGGT-X / VGGT-Omega give `extrinsic (k,3,4)`, `intrinsics (k,3,3)`, `depth (k,H,W,1)`, `depth_conf (k,H,W)`, `images`; MapAnything gives `depth (k,H,W)` and `images (k,3,H,W)` in [0,1], and its window is a list of view dicts, not a tensor.
- `BundleAdjustment(cfg).refine(images, confidence, world_points, extrinsics, intrinsics, image_paths, depth=)` returns `(N,4,4)` extrinsics and `(N,3,3)` K; it raises `ValueError` when too few frames/points survive. After a solve, `ba.alignment_scale` (float | None) and `ba.loss_history` (`list[list[float]]`) are set.
- `unproject_frames(depth (N,H,W), extrinsics (N,3,4)|(N,4,4), intrinsics) -> (N,H,W,3) float32` takes the precision lock re-entrantly (it is an `RLock`), so calling it inside `hold_matmul_precision()` on the same thread is safe.
- `pytorch_gc` lives in `collab_splats.utils.torch_utils`.

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/loop_closure/test_window_ba.py`:

```python
"""
Bundle adjustment inside each loop-closure window (LoopClosure ba=...).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig
from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig

WRAPPER = "collab_splats.geometry.loop_closure.wrapper"


def _raw(k: int, focal: float = 100.0, squeeze: bool = False) -> dict:
    """
    Forward output for k frames on a 4x4 grid: identity poses, constant depth, varied confidence.
    """
    rows = np.arange(4, dtype=np.float32)[:, None]
    cols = np.arange(4, dtype=np.float32)[None, :]
    K = np.array([[focal, 0, 2], [0, focal, 2], [0, 0, 1]], dtype=np.float32)
    depth = np.ones((k, 4, 4), dtype=np.float32) if squeeze else np.ones((k, 4, 4, 1), dtype=np.float32)
    return {
        "extrinsic": np.tile(np.eye(3, 4, dtype=np.float32), (k, 1, 1)),
        "intrinsics": np.tile(K, (k, 1, 1)),
        "depth": depth,
        "depth_conf": np.tile(50.0 + rows + cols, (k, 1, 1)).astype(np.float32),
        "images": np.zeros((k, 3, 4, 4), dtype=np.float32),
    }


def _wrapper(n_frames: int, submap_size: int, ba: BundleAdjustmentConfig | None, forward=None) -> LoopClosure:
    """
    LoopClosure over a MagicMock creator with n_frames tensor views and a stub forward.
    """
    base = MagicMock()
    base.max_points = 500_000
    base.views = torch.zeros(n_frames, 3, 4, 4)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base.original_coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n_frames, 1))
    base._forward = forward or (lambda model, views: _raw(views.shape[0]))
    base.default_verify_match_ratio = 1.0
    cfg = LoopClosureConfig(
        submap_size=submap_size, submap_overlap=1, min_submap_gap=0, max_loops_per_submap=0, lc_retrieval_threshold=999.0
    )
    return LoopClosure(base, config=cfg, ba=ba)


class _FakeBA:
    """
    BundleAdjustment stand-in: records each call, returns poses shifted by +1 in x and focal 77.
    """

    calls: list[dict] = []

    def __init__(self, config: BundleAdjustmentConfig) -> None:
        self.config = config
        self.alignment_scale = 1.0
        self.loss_history = [[3.0, 2.0]]

    def refine(self, images, confidence, world_points, extrinsics, intrinsics, image_paths=None, depth=None):
        type(self).calls.append(
            {
                "config": self.config,
                "images": images,
                "intrinsics": intrinsics.copy(),
                "paths": image_paths,
                "depth": depth,
                "world_points": world_points,
            }
        )
        refined = extrinsics.astype(np.float64).copy()
        refined[:, 0, 3] += 1.0
        K = intrinsics.astype(np.float64).copy()
        K[:, 0, 0] = 77.0
        K[:, 1, 1] = 77.0
        return refined, K


@pytest.fixture
def fake_ba():
    """
    Patch the wrapper's BundleAdjustment with _FakeBA; yields the call list.
    """
    _FakeBA.calls = []

    with patch(f"{WRAPPER}.BundleAdjustment", _FakeBA):
        yield _FakeBA.calls


def _run_loop(wrapper: LoopClosure) -> None:
    """
    run_inference with retrieval and loop detection stubbed out.
    """
    with (
        patch(f"{WRAPPER}.BaseRetrievalExtractor") as retrieval,
        patch(f"{WRAPPER}.find_loop_closures", return_value=[]),
    ):
        retrieval.get.return_value = lambda device: (lambda frames: torch.zeros(frames.shape[0], 128))
        wrapper.run_inference()


def test_ba_runs_once_per_window_with_that_windows_paths(fake_ba, tmp_path):
    # 9 frames, submap 3 + overlap 1: windows start at 0, 3, 6
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(tracks_cache_dir=tmp_path))
    _run_loop(wrapper)

    assert [r["start"] for r in wrapper.window_ba] == [0, 3, 6]
    assert [c["paths"] for c in fake_ba] == [
        ["img_000.png", "img_001.png", "img_002.png", "img_003.png"],
        ["img_003.png", "img_004.png", "img_005.png", "img_006.png"],
        ["img_006.png", "img_007.png", "img_008.png"],
    ]


def test_ba_gets_squeezed_depth_and_window_frames(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())
    _run_loop(wrapper)

    call = fake_ba[0]
    assert call["depth"].shape == (4, 4, 4)
    assert call["images"].shape == (4, 3, 4, 4)
    assert call["world_points"].shape == (4, 4, 4, 3)


def test_mapanything_window_uses_raw_images(fake_ba):
    # View-dict window: frames come from raw["images"], depth is already (k, H, W)
    raw = _raw(3, squeeze=True)
    raw["images"] = np.full((3, 3, 4, 4), 0.5, dtype=np.float32)
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())

    with patch.object(wrapper.base, "_forward", return_value=raw):
        out = wrapper._forward_window([{"img": None}] * 3, 0)

    np.testing.assert_array_equal(fake_ba[0]["images"], raw["images"])
    assert fake_ba[0]["depth"].shape == (3, 4, 4)
    assert out is raw


def test_later_windows_hold_window_zero_focal(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(refine_focal=True))
    _run_loop(wrapper)

    assert fake_ba[0]["config"].refine_focal is True
    assert fake_ba[0]["intrinsics"][0, 0, 0] == 100.0

    for call in fake_ba[1:]:
        assert call["config"].refine_focal is False
        assert call["intrinsics"][0, 0, 0] == 77.0
        assert call["intrinsics"][0, 1, 1] == 77.0


def test_refine_focal_false_never_holds(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(refine_focal=False))
    _run_loop(wrapper)

    for call in fake_ba:
        assert call["intrinsics"][0, 0, 0] == 100.0


def test_failed_window_keeps_feedforward_poses(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    raw = _raw(3)
    expected = raw["extrinsic"].copy()

    with (
        patch.object(_FakeBA, "refine", side_effect=ValueError("too few points")),
        patch.object(wrapper.base, "_forward", return_value=raw),
    ):
        out = wrapper._forward_window(wrapper.base.views, 0)

    np.testing.assert_array_equal(out["extrinsic"], expected)
    assert wrapper.window_ba[0]["ok"] is False
    assert wrapper.window_ba[0]["focal"] == 100.0
    assert wrapper._ba_focal is None


def test_refined_poses_are_anchored_to_frame_zero(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    out = wrapper._forward_window(wrapper.base.views, 0)

    np.testing.assert_allclose(out["extrinsic"][0], np.eye(3, 4), atol=1e-6)
    assert out["extrinsic"].dtype == np.float32
    assert out["intrinsics"].dtype == np.float32
    assert out["intrinsics"][0, 0, 0] == 77.0


def test_window_record_fields(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper._forward_window(wrapper.base.views, 0)

    record = wrapper.window_ba[0]
    assert set(record) == {"start", "n_frames", "ok", "seconds", "focal", "alignment_scale", "loss_final", "gpu_max_mib"}
    assert record["ok"] is True
    assert record["n_frames"] == 3
    assert record["focal"] == 77.0
    assert record["alignment_scale"] == 1.0
    assert record["loss_final"] == 2.0


def test_track_cache_dir_per_window(fake_ba, tmp_path):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig(tracks_cache_dir=tmp_path))
    _run_loop(wrapper)

    dirs = [Path(c["config"].tracks_cache_dir) for c in fake_ba]
    assert dirs == [tmp_path / "w000000", tmp_path / "w000003", tmp_path / "w000006"]


def test_no_cache_dir_stays_none(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper._forward_window(wrapper.base.views, 0)

    assert fake_ba[0]["config"].tracks_cache_dir is None


def test_too_few_frames_fallback_refines_once(fake_ba):
    # 3 frames < submap_size 20: one whole-scene forward, refined as window 0
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper.run_inference()

    assert len(fake_ba) == 1
    assert wrapper.window_ba[0]["start"] == 0
    assert wrapper.base.raw_outputs["intrinsics"][0, 0, 0] == 77.0


def test_dino_salad_failure_fallback_refines_once(fake_ba):
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())

    with patch(f"{WRAPPER}.BaseRetrievalExtractor") as retrieval:
        retrieval.get.side_effect = OSError("no weights")
        wrapper.run_inference()

    assert len(fake_ba) == 1
    assert wrapper.window_ba[0]["n_frames"] == 9
    assert wrapper.outputs is None


def test_verify_pairs_are_never_refined(fake_ba):
    # Loop verification goes through base._verify_loop_candidate, never _forward_window
    wrapper = _wrapper(9, 3, BundleAdjustmentConfig())
    _run_loop(wrapper)

    assert len(fake_ba) == 3
    wrapper.base._verify_loop_candidate.assert_not_called()


def test_run_inference_resets_window_state(fake_ba):
    wrapper = _wrapper(3, 20, BundleAdjustmentConfig())
    wrapper.run_inference()
    wrapper.run_inference()

    assert len(wrapper.window_ba) == 1
    assert fake_ba[1]["config"].refine_focal is True


def test_ba_none_leaves_forward_untouched():
    raw = _raw(3)
    expected = {key: value.copy() for key, value in raw.items()}
    wrapper = _wrapper(3, 20, None)

    with (
        patch.object(wrapper.base, "_forward", return_value=raw),
        patch(f"{WRAPPER}.BundleAdjustment") as ba_cls,
    ):
        out = wrapper._forward_window(wrapper.base.views, 0)

    ba_cls.assert_not_called()

    for key, value in expected.items():
        np.testing.assert_array_equal(out[key], value)

    assert wrapper.window_ba == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: the test command with `tests/geometry/loop_closure/test_window_ba.py`
Expected: FAIL at collection or call with `TypeError: LoopClosure.__init__() got an unexpected keyword argument 'ba'` (or `ImportError` on `BundleAdjustment` patch target).

- [ ] **Step 3: Implement in `wrapper.py`**

3a. Imports (keep isort grouping; add `time`, `BundleAdjustment`, `BundleAdjustmentConfig`, `pytorch_gc`):

```python
import dataclasses
import logging
import math
import time
from collections.abc import Callable
...
from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
from collab_splats.geometry.loop_closure.graph import PoseGraph
...
from collab_splats.utils.torch_utils import hold_matmul_precision, pytorch_gc
```

3b. `__init__` signature and docstring `Args:`; new state after `self.outputs`:

```python
    def __init__(
        self, base: Any, config: LoopClosureConfig | None = None, ba: BundleAdjustmentConfig | None = None
    ) -> None:
        """
        Wrap a creator, resolving the config against its per-model defaults.

        Args:
            base: feedforward creator whose forward pass the LC loop drives.
            config: loop-closure settings; None uses LoopClosureConfig().
            ba: bundle adjustment run inside each window after its forward; None runs none.
        """
        self.base = base
        self.config = config if config is not None else LoopClosureConfig()
        self.ba = ba
        ...  # existing body unchanged

        # Per-window BA records and the focal later windows hold; reset by run_inference
        self.window_ba: list[dict] = []
        self._ba_focal: float | None = None
```

3c. `run_inference`:

```python
    def run_inference(self) -> None:
        """
        Run the LC loop, or fall back to one whole-scene forward.

        - LC loop: sets outputs to the assembled PointcloudResult
        - fewer than submap_size frames: one forward through _forward_window, refined as window 0 when ba is set
        - DINO-SALAD fails to load: sets base.raw_outputs only; outputs is not assembled
        """
        # Fresh per-window BA state for this run
        self.window_ba = []
        self._ba_focal = None

        if self._enough_frames():
            self._run_lc_loop()
        else:
            self.base.raw_outputs = self._forward_window(self.base.views, 0)
            pytorch_gc()
```

3d. `_forward_window` (replaces the 3-line body; the whole solve is inline):

```python
    def _forward_window(self, window: Any, start: int) -> dict:
        """
        One window's forward under no_grad, then its bundle adjustment when ba is set.

        - runs on the pipeline worker thread; no_grad is thread-local, so it is entered here
        - BA enters enable_grad itself; worker order makes window 0's focal known before window 1's solve
        - the first refined window's focal is held fixed in every later window (refine_focal only)
        - a solve raising ValueError keeps the feedforward poses; its record has ok False
        - start: the window's first frame index into base.image_paths
        """
        with hold_matmul_precision(), torch.no_grad():
            raw = self.base._forward(self.base.model, window)

            if self.ba is None:
                return raw

            # Window record; peak GPU memory counted from here
            t0 = time.perf_counter()
            k = raw["extrinsic"].shape[0]
            record = {"start": start, "n_frames": k, "ok": False, "alignment_scale": None, "loss_final": None}

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Model-grid frames in [0, 1]: the window tensor, or MapAnything's raw images
            if isinstance(window, torch.Tensor):
                frames = window.float().cpu().numpy()
            else:
                frames = raw["images"]

            # Held focal: later windows solve with the first refined window's focal fixed
            intrinsics = raw["intrinsics"].copy()
            cfg = self.ba

            if self._ba_focal is not None:
                intrinsics[:, 0, 0] = self._ba_focal
                intrinsics[:, 1, 1] = self._ba_focal
                cfg = dataclasses.replace(cfg, refine_focal=False)

            # One track cache per window
            if cfg.tracks_cache_dir is not None:
                cache_dir = Path(cfg.tracks_cache_dir) / f"w{start:06d}"
                cfg = dataclasses.replace(cfg, tracks_cache_dir=cache_dir)

            # Solve inputs: (k, H, W) depth, world points under the solve's K, 4x4 poses, this window's paths
            depth = raw["depth"]

            if depth.ndim == 4:
                depth = depth[..., 0]

            world_points = unproject_frames(depth, raw["extrinsic"], intrinsics)
            poses = extrinsics_to_homogeneous(raw["extrinsic"])
            confidence = torch.from_numpy(raw["depth_conf"])
            paths = list(self.base.image_paths[start : start + k])

            # Solve; a ValueError keeps the feedforward poses
            ba = BundleAdjustment(cfg)

            try:
                refined, refined_intrinsics = ba.refine(
                    frames, confidence, world_points, poses, intrinsics, paths, depth=depth
                )
            except ValueError as e:
                logger.warning("Window BA at frame %d failed (%s); keeping feedforward poses", start, e)
            else:
                # Back to frame 0's camera, as run_predictions expects
                refined = refined.astype(np.float64)
                frame0_inv = invert_poses(refined[:1])[0]
                local = refined @ frame0_inv
                raw["extrinsic"] = local[:, :3, :].astype(np.float32)
                raw["intrinsics"] = refined_intrinsics.astype(np.float32)
                record["ok"] = True
                record["alignment_scale"] = ba.alignment_scale

                if ba.loss_history and ba.loss_history[-1]:
                    record["loss_final"] = ba.loss_history[-1][-1]

                # First refined window sets the focal later windows hold
                if self._ba_focal is None and self.ba.refine_focal:
                    self._ba_focal = float(refined_intrinsics[0, 0, 0])

            # Finish the record: the focal this window carries, time, peak GPU memory
            record["focal"] = float(raw["intrinsics"][0, 0, 0])
            record["seconds"] = time.perf_counter() - t0
            record["gpu_max_mib"] = torch.cuda.max_memory_allocated() >> 20 if torch.cuda.is_available() else None
            self.window_ba.append(record)
            logger.info(
                "Window BA at frame %d: %d frames, ok %s, focal %.1f, %.0f s",
                start,
                k,
                record["ok"],
                record["focal"],
                record["seconds"],
            )
            return raw
```

3e. DINO-SALAD fallback in `_run_lc_loop`: replace `self.base.raw_outputs = self.base._forward(self.base.model, views)` with:

```python
                self.base.raw_outputs = self._forward_window(views, 0)
```

3f. Both submits pass the window's start:

```python
            pending = pool.submit(self._forward_window, windows[0], bounds[0][0])
            ...
                if wi + 1 < len(windows):
                    pending = pool.submit(self._forward_window, windows[wi + 1], bounds[wi + 1][0])
```

Update the `_run_lc_loop` docstring bullet "DINO-SALAD failing to load sets base.raw_outputs from one full forward pass" to "... from one full forward through _forward_window (refined as window 0 when ba is set)".

- [ ] **Step 4: Fix existing tests for the new attribute and fallback**

In `tests/geometry/loop_closure/test_wrapper.py`:

- after every `wrapper = LoopClosure.__new__(LoopClosure)` (6 sites; find with `rtk proxy grep -n "LoopClosure.__new__" tests/geometry/loop_closure/test_wrapper.py`), add `wrapper.ba = None` next to the existing `wrapper.viz = None` line, so `__getattr__` never hands back the MagicMock base's `ba`
- `test_loop_closure_run_inference_falls_back_to_base_when_too_few_frames` (~line 188): rewrite to the new fallback:

```python
def test_loop_closure_run_inference_falls_back_to_base_when_too_few_frames():
    """When fewer frames than submap_size, one whole-scene forward sets base.raw_outputs."""
    from collab_splats.geometry.loop_closure import LoopClosureConfig
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure

    mock_base = MagicMock()
    mock_base.views = torch.zeros(3, 3, 8, 8)  # 3 frames

    cfg = LoopClosureConfig(submap_size=20)
    lc = LoopClosure(mock_base, config=cfg)
    lc.run_inference()
    mock_base._forward.assert_called_once_with(mock_base.model, mock_base.views)
    assert mock_base.raw_outputs is mock_base._forward.return_value
    mock_base.run_inference.assert_not_called()
```

(`tests/pointcloud/test_base.py:304` also builds via `__new__` but overrides `run_inference`, so `ba` is never read; leave it.)

- [ ] **Step 5: Run the tests**

Run: the test command with `tests/geometry/loop_closure/ tests/pointcloud/test_base.py`
Expected: all pass, `EXIT=0`. Any other failure: compare with `docs/known-test-failures.md` before touching it.

- [ ] **Step 6: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black -l 120 --target-version py311 collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_window_ba.py tests/geometry/loop_closure/test_wrapper.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_window_ba.py tests/geometry/loop_closure/test_wrapper.py
git add tests/geometry/loop_closure/test_window_ba.py
git commit --only collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_window_ba.py tests/geometry/loop_closure/test_wrapper.py -m "feat(geometry): bundle adjustment inside each loop-closure window

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

If `black` reformats unrelated lines in `test_wrapper.py`, revert those hunks so the diff stays the 7 intended edits.

---

### Task 2: Reconstructor wiring

**Files:**
- Modify: `collab_splats/reconstructor.py` (refusal ~332-337; default stage set ~529-536; `pointcloud()` ~690-765; `refine()` ~767-790; imports)
- Modify: `tests/reconstructor/test_refine_stage.py` (~28-42), `tests/reconstructor/test_reconstructor.py` (~1018)

- [ ] **Step 1: Write the failing tests**

In `tests/reconstructor/test_refine_stage.py`, replace `test_validate_config_rejects_ba_with_lc_bool` and `test_validate_config_rejects_ba_with_lc_dict` with:

```python
def test_validate_config_accepts_ba_with_lc_bool(tmp_path):
    """bundle_adjustment + loop_closure=true is per-window BA, accepted at construction."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))
    assert r.config["pointcloud"]["bundle_adjustment"]["enabled"] is True


def test_validate_config_accepts_ba_with_lc_dict(tmp_path):
    """Dict-form loop_closure with BA on is accepted too."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure={"submap_size": 16}))
    assert r.config["pointcloud"]["loop_closure"]["enabled"] is True


def test_run_config_driven_skips_refine_under_lc(tmp_path):
    """BA + LC: BA runs inside the pointcloud stage, so the default stage set drops refine."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))
    calls = []
    with patch.object(Reconstructor, "preproc", side_effect=lambda **kwargs: calls.append("preproc")), \
         patch.object(Reconstructor, "pointcloud", side_effect=lambda: calls.append("pointcloud")), \
         patch.object(Reconstructor, "refine", side_effect=lambda: calls.append("refine")), \
         patch.object(Reconstructor, "semantics", side_effect=lambda: calls.append("semantics")), \
         patch.object(Reconstructor, "mesh", side_effect=lambda: calls.append("mesh")), \
         patch.object(Reconstructor, "localize", side_effect=lambda: calls.append("localize")), \
         patch.object(Reconstructor, "reconstruction_quality_report",
                      side_effect=lambda: calls.append("reconstruction_quality_report")):
        r.run()
    assert "pointcloud" in calls
    assert "refine" not in calls


def test_refine_refuses_under_lc(tmp_path):
    """An explicit refine under LC would re-solve a store BA already ran in."""
    r = Reconstructor(_cfg(tmp_path, bundle_adjustment=True, loop_closure=True))
    with pytest.raises(ValueError, match="refine is not supported with pointcloud.loop_closure"):
        r.refine()
```

In `tests/reconstructor/test_reconstructor.py`:

- `test_pointcloud_stage_attaches_viewer_when_lc_and_viz_enabled`: change `mock_lc_cls.assert_called_once_with(base=creator_cls.return_value, config=None)` to `mock_lc_cls.assert_called_once_with(base=creator_cls.return_value, config=None, ba=None)`
- add, next to `test_pointcloud_stage_builds_lc_config_from_dict`:

```python
def test_pointcloud_stage_passes_window_ba_config_and_attrs(tmp_path):
    """BA + LC: LoopClosure gets a BA config with the window_ba cache dir; the zarr records window_ba."""
    config = _make_config(
        tmp_path,
        {"pointcloud": {"loop_closure": {"submap_size": 32}, "bundle_adjustment": {"enabled": True, "dtype": "float64"}}},
    )
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    result = _make_mock_pointcloud_result(tmp_path)
    mock_lc_instance.create_pointcloud.return_value = result
    mock_lc_instance.window_ba = [{"start": 0, "focal": np.float32(368.5), "loss_final": float("nan")}]

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ):
        rec.pointcloud()

    ba_cfg = mock_lc_cls.call_args.kwargs["ba"]
    assert ba_cfg.dtype == "float64"
    assert ba_cfg.tracks_cache_dir == rec.backend_dir / "window_ba"

    attrs = result.save_zarr.call_args.kwargs["extra_attrs"]
    assert attrs["window_ba"] == [{"start": 0, "focal": 368.5, "loss_final": None}]


def test_pointcloud_stage_lc_without_ba_passes_none(tmp_path):
    """LC alone: LoopClosure gets ba=None and the zarr attrs carry no window_ba."""
    config = _make_config(tmp_path, {"pointcloud": {"loop_closure": True}})
    rec = Reconstructor(config)
    creator_cls = stub_creator_cls(_make_mock_pointcloud_result(tmp_path))
    mock_lc_instance = MagicMock()
    result = _make_mock_pointcloud_result(tmp_path)
    mock_lc_instance.create_pointcloud.return_value = result

    with (
        patch("collab_splats.reconstructor.get_creator", return_value=creator_cls),
        patch("collab_splats.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ):
        rec.pointcloud()

    assert mock_lc_cls.call_args.kwargs["ba"] is None
    assert "window_ba" not in result.save_zarr.call_args.kwargs["extra_attrs"]
```

`_make_mock_pointcloud_result` returns a `MagicMock`, and `pointcloud()` calls `save_zarr` on the object `create_pointcloud` returned, so `result.save_zarr.call_args` holds the attrs.

- [ ] **Step 2: Run them to verify they fail**

Run: the test command with `tests/reconstructor/test_refine_stage.py tests/reconstructor/test_reconstructor.py -k "lc or ba or viewer"`
Expected: the accept tests fail with `ValueError: ... mutually exclusive`; the `ba=` tests fail on the call assertion / `KeyError: 'ba'`.

- [ ] **Step 3: Implement in `reconstructor.py`**

3a. Delete the refusal block (the comment `# Refuse BA with LC: ...` and its `if`/`raise`). Both sfm refusals stay.

3b. Default stage set:

```python
        # Default stage set from the enable flags; BA with LC runs inside pointcloud, not refine
        if stages is None:
            pc = self.config["pointcloud"]
            enabled = {
                "refine": pc["bundle_adjustment"]["enabled"] and not pc["loop_closure"]["enabled"],
                ...
```

3c. `refine()` guard right after the sfm guard, and a `Raises:` line in its docstring (`ValueError: pointcloud.method is sfm, or loop closure is on.`):

```python
        if cfg["loop_closure"]["enabled"]:
            raise ValueError(
                "refine is not supported with pointcloud.loop_closure — BA already ran inside each window"
            )
```

3d. `pointcloud()` LC wrap — build the BA config like `refine()` does and pass it:

```python
            if lc.pop("enabled"):
                lc_config = LoopClosureConfig(**lc) if lc else None

                # BA with LC runs inside each window, with the refine stage's terms
                if cfg["bundle_adjustment"]["enabled"]:
                    terms = {k: v for k, v in cfg["bundle_adjustment"].items() if k != "enabled"}
                    ba_cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir / "window_ba", **terms)

                creator = LoopClosure(base=creator, config=lc_config, ba=ba_cfg)
```

Add `ba_cfg = None` once, with a one-line comment (`# Window BA config; set only for feedforward with LC and BA on`), right before the `if cfg["method"] == "feedforward":` block, so the attrs code below reads it on every path (sfm included).

3e. Provenance attrs:

```python
        # Provenance attrs; an sfm creator supplies its own subset and alignment attrs
        if cfg["method"] == "feedforward":
            attrs = {"method": "feedforward", "backend": backend}
        else:
            attrs = {"backend": backend, **creator.attrs}

        # Per-window BA records; zarr attrs need strict JSON
        if ba_cfg is not None:
            attrs["window_ba"] = to_json_safe(creator.window_ba)
```

Import `to_json_safe` in the existing `from collab_splats.utils.io import read_image, write_json` line.

Update the `pointcloud()` docstring bullets: add `- BA on with LC: bundle adjustment runs inside each window; records land in the zarr attrs as window_ba`.

- [ ] **Step 4: Run the reconstructor suite**

Run: the test command with `tests/reconstructor/`
Expected: `EXIT=0` apart from entries already in `docs/known-test-failures.md`.

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black -l 120 --target-version py311 collab_splats/reconstructor.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/reconstructor.py tests/reconstructor/test_refine_stage.py tests/reconstructor/test_reconstructor.py
git commit --only collab_splats/reconstructor.py tests/reconstructor/test_refine_stage.py tests/reconstructor/test_reconstructor.py -m "feat(reconstructor): BA with loop closure runs inside each window

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(The two test files use backslash-continued `with` blocks black would rewrite; do not black them — isort only.)

---

### Task 3: Decision and docs

**Files:**
- Create: `docs/superpowers/decisions/022-lc-window-ba.md`
- Modify: `configs/README.md` (~387, ~405), `configs/base.yaml` (~58), `docs/pointcloud.md`

- [ ] **Step 1: Write the decision**

```markdown
# 022 — Bundle adjustment inside each loop-closure window

Date: 2026-10-05 · Status: accepted · Branch: `feat/rgbd-ba-cf`

## Context

- since the 2026-08-19 BA wiring, `validate_config` refused BA with LC: LC submaps do not carry the per-frame model tensors the whole-scene `refine` stage needs; no decision recorded the refusal
- perf-1k needs ~1000-frame scenes, which only run windowed (LC), and BA is what makes the 294-frame GH010229 baseline good
- the baseline came from a scratch hook (`lc_inner_light.py`) that solved BA inside each window, on the worker thread, before the submap entered the pose graph

## Decision

- BA on + LC on = BA inside each LC window; no `refine` stage; no new config keys
- the first refined window sets the focal; later windows hold it (`refine_focal: false` for their solve)
- the solve runs in `LoopClosure._forward_window`, after the forward, on the existing worker thread; loop-verify pairs are never refined
- a window whose solve raises `ValueError` keeps its feedforward poses; per-window records land in the zarr attrs as `window_ba`

## Consequences

- an explicit `refine` stage under LC raises; whole-scene BA after LC is out of scope
- loop carriers come from 2-frame forwards with the feedforward focal (known limit, measured at the chess gate)
- spec: `docs/superpowers/specs/2026-10-05-lc-window-ba-design.md`
```

- [ ] **Step 2: Config docs**

`configs/README.md`:
- row `pointcloud.bundle_adjustment.enabled`: `Run LM bundle adjustment after pointcloud (\`ValueError\` with \`method: sfm\`); with \`loop_closure\` on it runs inside each LC window instead (first window sets the focal, later windows hold it; no \`refine\` stage); a bare bool sets this`
- row `pointcloud.loop_closure`: append `; with \`bundle_adjustment.enabled\` BA runs inside each window`
- line ~44 paragraph: after "`refine` ... run only when enabled", add the sentence: `With loop closure on, \`refine\` is not planned: BA runs inside the pointcloud stage, one solve per window.`

`configs/base.yaml` line ~58 comment: replace `exclusive with loop_closure` with `with loop_closure on, runs inside each LC window instead (no refine stage)`.

`docs/pointcloud.md`: append

```markdown
## Bundle adjustment with loop closure

With `pointcloud.loop_closure` and `pointcloud.bundle_adjustment.enabled` both on, BA runs
inside each loop-closure window, right after the window's forward pass and before its submap
enters the pose graph; the `refine` stage is skipped. The first refined window sets the focal
and later windows hold it fixed. A window whose solve fails keeps its feedforward poses. Each
window's outcome (start frame, focal, loss, time, peak GPU memory) is stored in the
`pointcloud.zarr` attrs under `window_ba`.
```

- [ ] **Step 3: Commit**

```bash
git add -f docs/superpowers/decisions/022-lc-window-ba.md
git commit --only docs/superpowers/decisions/022-lc-window-ba.md configs/README.md configs/base.yaml docs/pointcloud.md -m "docs(decisions): 022 bundle adjustment inside each loop-closure window

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Full test gate

- [ ] **Step 1:** Run the test command with `tests/geometry tests/reconstructor tests/pointcloud tests/test_docstring_contract.py tests/test_import_style.py`. Expected `EXIT=0` apart from entries in `docs/known-test-failures.md`. No measurement run may be in flight while this runs.
- [ ] **Step 2:** `graphify update .` in the worktree.

---

### Task 5: Parity vs the scratch hook (controller runs; GPU, tmux, one job at a time)

Both sides run on this branch. Outputs under `$S/parity/` only — never `/workspace/outputs/ocr_viewer/*`. Fresh track-cache dirs on both sides.

**Hook fix first.** The hook records `start` in `run_predictions`, but since `dc72d9e2` the next window's forward is submitted after `run_predictions` of the current one, so on this branch the hook reads the previous window's start. That only touches the hook's cache key and log line (the cache key also hashes world points, so results are unaffected), but parity compares per-window records by start. The parity copies derive start from a forward counter instead:

```python
# Window start from the forward count: windows forward in order on the worker
state["n_fwd"] += 1  # only for k >= 10 forwards
start = (state["n_fwd"] - 1) * SUBMAP_SIZE
```

- [ ] **Step 1: chess, 200 frames, submap 50**
  - hook side: `$S/parity/hook_chess.py` = copy of `lc_inner_light.py` with the counter fix, cell `…ibfz` (float64), grid `$S/parity/chess200.yaml` (`max_frames: 200`, `loop_closure {submap_size: 50, lc_retrieval_threshold: 0.95}`, `output_dir` under `$S/parity/`)
  - package side: `$S/parity/pkg_chess.py` = same driver, same stubs (quality report, lean save, depth metrics), no `_forward` patch; condition adds `bundle_adjustment {enabled: true, query_frame_num: 4, max_query_pts: 2048, fine_tracking: false, dtype: float64}`
- [ ] **Step 2: GH010229, 294 frames, pointcloud stage only**
  - hook side: `$S/parity/hook_gh.py` = copy of `gh_lcba.py` with the counter fix, `stages=["pointcloud"]`
  - package side: `$S/parity/pkg_gh.py` = same config + the BA block above, `stages=["pointcloud"]`
- [ ] **Step 3: Compare** with `$S/parity/compare.py <hook_zarr> <pkg_zarr> <hook_log>`:
  - per-window focal (hook `INNER` lines vs package `window_ba` attrs) within 5e-3 px; same ok/failed windows
  - zarr extrinsics max abs diff ≤ 1e-4
  - chess ATE (eval metrics JSON) equal within 0.1 mm
- [ ] **Step 4: On a miss**, switch the hook copy to `unproject_frames` and re-run that side only, to isolate the one deliberate difference.

Report parity numbers to the user before the perf-1k gates (item 11, chess GT 1000, GH010229 1000), which run from `docs/superpowers/plans/2026-10-03-perf-1k.md` on package config only.
