# LoGeR Feedforward Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `loger` as a fourth `Reconstructor`-reachable feedforward backend, selectable with `pointcloud.backend: loger` and otherwise indistinguishable from `vggtx` / `mapanything` / `vggt_omega`.

**Architecture:** A single new module `collab_splats/pointcloud/feedforward/loger.py` holds `LoGeRCreator` (a `BaseFeedforwardCreator` subclass implementing the six abstract methods) plus three module-level helpers. The LoGeR model itself is vendored into the gitignored `third_party/LoGeR/` by a new setup script and imported behind a `sys.path` insert inside `_load_model`, following VGGT-SPARK. Because LoGeR predicts no intrinsics, K is **solved** from its camera-frame pointmap by a confidence-weighted median pinhole fit. Everything downstream — unprojection, COLMAP export, BA reprojection — reuses existing base-class and VGGT-X functions unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, numpy, PIL, pyyaml, `huggingface_hub`, pytest. Zero new pip dependencies.

**Spec:** `docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md`

**Python:** every command below uses `/opt/venv/reconstruction/bin/python`. The base-shell `python` may be 3.13 and is wrong for this project.

---

## File Structure

| | Path | Responsibility |
|---|---|---|
| new | `setup/loger.sh` | Clone upstream `Junyi42/LoGeR` @ `7685b7a` into `third_party/LoGeR/`. Idempotent. |
| new | `collab_splats/pointcloud/feedforward/loger.py` | `LoGeRCreator` + `_weighted_median` + `_estimate_shared_intrinsics` + `_loger_target_size` |
| new | `tests/pointcloud/test_loger_creator.py` | Fit, resize, forward, postprocess, refusals |
| new | `tests/pointcloud/feedforward/test_loger_load_guard.py` | Absent-tree guarded-import behaviour |
| mod | `third_party/README.md` | Table row + pinned commit + missing-LICENSE note |
| mod | `collab_splats/pointcloud/feedforward/__init__.py` | Guarded `LoGeRCreator` export |
| mod | `collab_splats/pointcloud/__init__.py` | `_LOGER_AVAILABLE` + `_REGISTRY["loger"]` |
| mod | `collab_splats/wrapper/reconstructor.py` | Backend set, `creator_map`, `creator_kwargs` passthrough, LC refusal, `max_frames` warning |
| mod | `configs/base.yaml` | Backend comment + `pointcloud.loger:` block |
| mod | `configs/README.md` | Backend row, `max_frames` row, intrinsics + when-to-choose notes |
| mod | `docs/source/api/pointcloud.rst` | One `automodule` stanza |
| mod | `tests/pointcloud/test_registry.py` | `test_get_creator_loger` (skipif) |
| mod | `tests/pointcloud/test_feedforward_intrinsics.py` | `original_coords` → original-res K round-trip |

### Attribution convention (applies to every task)

**Every line of ported or adapted code carries a citation naming the repository, the pinned commit, the file, and the line range.** A bare filename is not enough — `run_loger.py:167` is ambiguous between the two forks, and `pi3.py:172` is a line number in a tree that is gitignored and therefore unreadable from the repo alone.

Two upstreams are involved and they are not interchangeable:

| Short form used below | Means |
|---|---|
| **PolyCam @ 5d7c1a7** | `github.com/PolyCam/LoGeR` @ `5d7c1a7` — the fork we port the intrinsics estimator *from*. Not vendored. |
| **Junyi42 @ 7685b7a** | `github.com/Junyi42/LoGeR` @ `7685b7a` — the tree we *vendor* into `third_party/LoGeR/` (Task 1). |

Write the long form in the code, not the short form. Every module-level helper docstring and every non-obvious inline comment that reflects upstream behaviour states which of the two it came from. When a comment cites vendored-tree behaviour we depend on but do not copy (the conf head emitting logits, `se3` being popped inside `forward`), that is still a citation and still names repo, commit, file, and line — a reader cannot check it otherwise, because `third_party/` is gitignored.

### One deliberate deviation from the spec

The spec's "Genuinely new" section names **two** new functions. This plan adds a **third**, `_loger_target_size`. Justification, since the spec requires one for every addition: it is a ~10-line ported algorithm with a `while` loop and a citation (`loger/utils/basic.py:51-63`), it is the sole source of the resize anisotropy that three other decisions depend on, and it is the only part of `_preprocess` that can be tested without a model. Inlining it would make the anisotropy untestable in isolation. This is a different case from the rejected `_loger_original_coords`, which would have wrapped `np.tile` of a constant row.

---

## Task 1: Vendor the tree and prove torch 2.5.1 runs it

**This task gates every other task.** LoGeR pins torch 2.6.0; we run 2.5.1+cu121. Nothing else is worth starting until a real forward pass succeeds. This is spec open item 1.

**Files:**
- Create: `setup/loger.sh`
- Modify: `third_party/README.md`

- [ ] **Step 1: Write the setup script**

Create `setup/loger.sh`:

```bash
#!/usr/bin/env bash
# setup/loger.sh — vendor LoGeR into third_party/ for the `loger` feedforward backend.
#
# LoGeR = Pi3 backbone + TTT fast-weight memory + sliding-window inference.
# Used by collab_splats/pointcloud/feedforward/loger.py via a sys.path insert
# inside _load_model (the tree is not pip-installed).
#
# Upstream is pinned: 7685b7a is the commit that adds "load images to cpu to
# reduce VRAM usage", which is load-bearing under our 46.6 GB cgroup cap and is
# absent from the PolyCam fork.
#
# Usage:
#   bash setup/loger.sh
set -e

LOGER_COMMIT="7685b7a"
LOGER_DIR="$(dirname "$0")/../third_party/LoGeR"

if [ ! -d "$LOGER_DIR" ]; then
    echo "=== Cloning LoGeR into $LOGER_DIR ==="
    git clone https://github.com/Junyi42/LoGeR.git "$LOGER_DIR"
else
    echo "=== $LOGER_DIR already exists, skipping clone ==="
fi

echo "=== Pinning to $LOGER_COMMIT ==="
git -C "$LOGER_DIR" fetch --all --quiet
git -C "$LOGER_DIR" checkout --quiet "$LOGER_COMMIT"

echo ""
echo "=== LoGeR vendored at $(git -C "$LOGER_DIR" rev-parse --short HEAD) ==="
echo "Checkpoint weights download on first use from HF Junyi42/LoGeR."
echo "No pip install needed — zero new dependencies."
```

- [ ] **Step 2: Run it and confirm the tree lands**

Run:
```bash
bash setup/loger.sh && ls third_party/LoGeR/ckpts/
```
Expected: clone output, then `LoGeR  LoGeR_star` listed.

- [ ] **Step 3: Confirm the config assumption this plan depends on**

Both shipped configs must contain exactly one top-level key, `model:`. If a future upstream adds `training_settings`, Task 5's hard-coded window defaults become wrong.

Run:
```bash
/opt/venv/reconstruction/bin/python -c "
import yaml
for v in ['LoGeR','LoGeR_star']:
    d = yaml.safe_load(open(f'third_party/LoGeR/ckpts/{v}/original_config.yaml'))
    print(v, sorted(d))
    assert sorted(d) == ['model'], f'{v} gained a top-level key — revisit Task 5 defaults'
print('OK: model-only configs, run_loger.py fallbacks are the real defaults')
"
```
Expected:
```
LoGeR ['model']
LoGeR_star ['model']
OK: model-only configs, run_loger.py fallbacks are the real defaults
```

- [ ] **Step 4: Write the smoke script**

This is a throwaway gate, not a deliverable — it lives in the scratchpad, not the repo.

Create `/tmp/claude-0/-workspace-collab-splats/d78ed8d1-0f5a-4555-8eb3-eecc7f2e302a/scratchpad/loger_smoke.py`:

```python
"""Gate: can torch 2.5.1+cu121 construct and run LoGeR at all?"""
import sys
import time
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path("third_party/LoGeR").resolve()
sys.path.insert(0, str(ROOT))

from loger.models.pi3 import Pi3  # noqa: E402

VARIANT = "LoGeR_star"
cfg = yaml.safe_load((ROOT / "ckpts" / VARIANT / "original_config.yaml").read_text())
model_cfg = dict(cfg["model"])
se3 = bool(model_cfg.pop("se3", False))

model = Pi3(**model_cfg)
ckpt = hf_hub_download(repo_id="Junyi42/LoGeR", filename=f"{VARIANT}/latest.pt")
state = torch.load(ckpt, map_location="cpu")
state = state.get("model_state_dict", state)
state = {k.removeprefix("module."): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"missing={len(missing)} unexpected={len(unexpected)}")
assert not missing and not unexpected, (missing[:5], unexpected[:5])

model = model.eval().to("cuda")
imgs = torch.rand(1, 8, 3, 336, 462, device="cuda")  # 8 frames, both dims % 14 == 0

t0 = time.time()
with torch.no_grad():
    out = model(imgs, window_size=32, overlap_size=3, reset_every=0,
                num_iterations=1, sim3=False, sim3_scale_mode="median",
                se3=se3, turn_off_ttt=False, turn_off_swa=False)
torch.cuda.synchronize()

print(f"forward OK in {time.time() - t0:.1f}s")
print("keys:", sorted(out))
for k in ("local_points", "conf", "camera_poses", "points"):
    print(f"  {k}: {tuple(out[k].shape)} {out[k].dtype}")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("conf range (RAW, pre-sigmoid):", float(out["conf"].min()), float(out["conf"].max()))
print("SMOKE PASS")
```

- [ ] **Step 5: Run the smoke script**

Run:
```bash
/opt/venv/reconstruction/bin/python /tmp/claude-0/-workspace-collab-splats/d78ed8d1-0f5a-4555-8eb3-eecc7f2e302a/scratchpad/loger_smoke.py
```
Expected: `missing=0 unexpected=0`, four printed shapes, and `SMOKE PASS`.

**Record these four facts — later tasks depend on them:**
1. Exact output key names (this plan assumes `local_points`, `conf`, `camera_poses`, `points`). If they differ, fix Task 6's `_forward` accordingly.
2. Whether `conf` is `(1,N,H,W,1)` or `(1,N,H,W)` — decides whether `_forward` squeezes a trailing axis.
3. The raw `conf` range. It must **not** already be in `[0, 1]`; if it is, upstream changed the head and the sigmoid step in Task 6 must be re-derived rather than copied.
4. Peak VRAM for 8 frames — the baseline for Task 13's sweep.

**If the forward pass fails on a torch API gap:** stop and report. Do not patch the vendored tree by hand — `third_party/README.md` policy is that patches live in the setup script. The fix belongs in `setup/loger.sh` as a post-clone patch step.

- [ ] **Step 6: Add the README row**

In `third_party/README.md`, add to the "Current entries" table after the `hloc/` row:

```markdown
| `LoGeR/` | `Junyi42/LoGeR` @ `7685b7a` | `setup/loger.sh` | LoGeR feedforward backend (`collab_splats/pointcloud/feedforward/loger.py`). `sys.path` insert inside `_load_model`. **No LICENSE file upstream** — see the spec's open items. |
```

- [ ] **Step 7: Commit**

```bash
git add setup/loger.sh third_party/README.md
git commit -m "feat(loger): vendor LoGeR via setup script

Pins upstream Junyi42/LoGeR at 7685b7a — the commit adding CPU image loading
to reduce VRAM, which the PolyCam fork lacks and which matters under our
46.6 GB cgroup cap. Clone is idempotent; the tree itself stays gitignored.

Verified a forward pass runs under torch 2.5.1+cu121 despite LoGeR pinning
2.6.0, which gates the rest of the backend work."
```

---

### Task 1 measured results (2026-08-13, commit `3cf792d`) — later tasks are written against these

- **Output keys (10):** `attn_gate_scale`, `avg_gate_scale`, `camera_poses`, `camera_qvec`, `conf`, `local_camera_poses`, `local_camera_qvec`, `local_points`, `metric`, `points`. All four the plan assumes are present, so Task 7's key access is safe.
- **Shapes at 8 frames, 336x462:** `local_points` and `points` `(1,8,336,462,3)`, `conf` `(1,8,336,462,1)`, `camera_poses` `(1,8,4,4)`, all float32.
- **`conf` carries a trailing axis of 1.** Task 7's `squeeze(-1)` after `squeeze(0)` is REQUIRED, not defensive.
- **Raw `conf` range: `-4.257` to `-2.019`** — entirely negative, confirming the head is unactivated and the sigmoid is ours to apply. (Values are low because the smoke input was random noise; sigmoid maps this to ~0.014-0.117.)
- **Peak VRAM 6.77 GB, forward 18.0 s** at 8 frames — Task 14's sweep baseline.
- **The two `model:` configs are NOT key-identical.** Both carry `attn_insert_after: [10,18,26,34]`, `ttt_head_dim: 512`, `ttt_insert_after` (18 even values 0-34), `ttt_inter_multi: 4`. But `ttt_pre_norm: True` is **LoGeR-only** and `se3: True` is **LoGeR_star-only**. Task 5's `model_cfg.pop("se3", False)` two-argument form is therefore load-bearing — neither key can be assumed present.
- **`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`** fires on every run. Functionally fine, but every timing number in Task 14 measures the slow RoPE2D path. Report it as such rather than as LoGeR's achievable speed.
- **No `LICENSE` or `COPYING` at the vendored tree's top level** — verified, not assumed. The licence question below is real.

---

## Task 2: `_weighted_median`

**Files:**
- Create: `collab_splats/pointcloud/feedforward/loger.py`
- Create: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pointcloud/test_loger_creator.py`:

```python
"""LoGeR feedforward backend: intrinsics fit, resize rule, creator contract."""
import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.loger import _weighted_median


def test_weighted_median_equal_weights_matches_plain_median():
    # With uniform weights the weighted median is the ordinary median.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.ones_like(values)
    assert _weighted_median(values, weights) == pytest.approx(3.0)


def test_weighted_median_follows_the_weight_mass():
    # Weight concentrated on the low values pulls the median down, even though
    # the high values are the numerical majority by count.
    values = np.array([1.0, 1.0, 9.0, 9.0, 9.0], dtype=np.float32)
    weights = np.array([50.0, 50.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert _weighted_median(values, weights) == pytest.approx(1.0)


def test_weighted_median_empty_returns_none():
    # Signals "no estimate" to the caller, which raises rather than falling back.
    assert _weighted_median(np.array([]), np.array([])) is None


def test_weighted_median_subsamples_deterministically():
    # Above max_n the seeded RNG must give the same answer every call — the
    # estimator is otherwise non-reproducible at production frame counts.
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 10.0, size=200_000).astype(np.float32)
    weights = np.ones_like(values)
    first = _weighted_median(values, weights, max_n=1000)
    assert first == _weighted_median(values, weights, max_n=1000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.pointcloud.feedforward.loger'`

- [ ] **Step 3: Write the module skeleton and the helper**

Create `collab_splats/pointcloud/feedforward/loger.py`:

```python
"""LoGeR feedforward backend: intrinsics fit, resize rule, and creator.

LoGeR is a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window
inference and overlap stitching.  Unlike every other backend we run, it predicts
no camera intrinsics, so K is solved from its camera-frame pointmap.

Upstream sources.  Two forks are involved and they are NOT interchangeable; every
port and every behavioural claim below cites one of them by repo, commit, file, and
line, because third_party/ is gitignored and cannot be read from this repo alone:

  * VENDORED (setup/loger.sh clones into third_party/LoGeR/):
      github.com/Junyi42/LoGeR @ 7685b7a
  * PORTED FROM (not vendored, not a dependency — code copied out by hand):
      github.com/PolyCam/LoGeR @ 5d7c1a7

Provides:
  LOGER_HF_REPO               — HuggingFace repo holding both checkpoints
  LOGER_VARIANTS              — the two shipped variants
  _weighted_median            — confidence-weighted median with a seeded subsample
  _estimate_shared_intrinsics — fit one pinhole K to LoGeR's camera-frame pointmap
  _loger_target_size          — LoGeR's own resize rule (area budget, multiples of 14)
  LoGeRCreator                — feedforward creator using LoGeR depth + pose
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

########################################################################
########## Constants ###################################################
########################################################################

LOGER_HF_REPO = "Junyi42/LoGeR"
LOGER_VARIANTS = ("LoGeR", "LoGeR_star")

# Vendored tree, populated by setup/loger.sh.  parents[3] resolves
# collab_splats/pointcloud/feedforward/loger.py -> repo root.
_LOGER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "LoGeR"

# LoGeR's ViT patch size; every model-resolution image dimension is a multiple of it.
_PATCH = 14

########################################################################
########## Intrinsics fit ##############################################
########################################################################


def _weighted_median(values: np.ndarray, weights: np.ndarray, max_n: int = 50_000) -> float | None:
    """Confidence-weighted median, subsampled above ``max_n`` with a seeded RNG.

    Ported from github.com/PolyCam/LoGeR @ 5d7c1a7, ``run_loger.py:167``
    (``_weighted_median``).  Returns ``None`` for an empty input so the caller can
    raise rather than invent a value.
    """
    if len(values) == 0:
        return None

    # A weighted median needs a full argsort, and the pooled per-pixel population is
    # H*W*N — 76.5M values at 300 frames, 255M at the 1000-frame sequences this
    # backend exists for.  The cap bounds that; the fixed seed keeps it reproducible.
    if len(values) > max_n:
        idx = np.random.default_rng(42).choice(len(values), max_n, replace=False)
        values, weights = values[idx], weights[idx]

    # Sort by value, then walk the cumulative weight to the halfway mass.
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumw = np.cumsum(weights)
    return float(values[np.searchsorted(cumw, cumw[-1] / 2.0)])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add confidence-weighted median helper

Ported from github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:167. The 50k cap is a memory
guard at the frame counts this backend targets: the pooled per-pixel sample
population is H*W*N, which is 255M values at 1000 frames, and a weighted median
needs a full argsort. The fixed seed keeps the estimate reproducible."
```

---

## Task 3: `_estimate_shared_intrinsics`

The core new algorithm. LoGeR emits no K; this solves one by inverting the pinhole model per pixel and taking a confidence-weighted median. It returns the whole `(3,3)` matrix rather than `(fx, fy)` because the centred pixel grid already fixes `cx`/`cy`.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
from collab_splats.pointcloud.feedforward.loger import _estimate_shared_intrinsics


def _synthetic_local_points(h: int, w: int, fx: float, fy: float, depth: float = 2.0) -> np.ndarray:
    """Exact camera-frame pointmap for a centre-principal pinhole camera, shape (1,H,W,3).

    Built about the same centre the estimator assumes, cx=(W-1)/2 and cy=(H-1)/2,
    so recovery is exact and the principal-point assertion below is meaningful.
    """
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )
    z = np.full((h, w), depth, dtype=np.float32)
    return np.stack([uu * z / fx, vv * z / fy, z], axis=-1)[None].astype(np.float32)


@pytest.mark.parametrize("fx,fy", [(320.0, 320.0), (352.0, 320.0)])
def test_fit_recovers_known_intrinsics(fx, fy):
    # The anisotropic row (fx/fy = 1.10) is what protects the aspect-ratio argument:
    # our resize rounds each axis to a multiple of 14 independently, so a real camera
    # genuinely produces fx != fy at model resolution. Any "simplification" that
    # averages them fails here.
    h, w = 224, 308
    pts = _synthetic_local_points(h, w, fx, fy)
    conf = np.ones((1, h, w), dtype=np.float32)

    k = _estimate_shared_intrinsics(pts, conf)

    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(fx, rel=1e-3)
    assert k[1, 1] == pytest.approx(fy, rel=1e-3)
    # cx/cy are decided by the estimator's own centred grid, not by the caller.
    assert k[0, 2] == pytest.approx((w - 1) / 2.0)
    assert k[1, 2] == pytest.approx((h - 1) / 2.0)
    assert k[2, 2] == pytest.approx(1.0)
    if fx != fy:
        assert k[0, 0] != pytest.approx(k[1, 1], rel=1e-3)


def test_fit_survives_confident_outliers():
    # Corrupt 30% of pixels AND give them full confidence. A weighted median is
    # unmoved; a least-squares fit would be dragged toward the corrupted focal.
    h, w = 112, 154
    fx = fy = 160.0
    pts = _synthetic_local_points(h, w, fx, fy)
    conf = np.ones((1, h, w), dtype=np.float32)

    rng = np.random.default_rng(7)
    bad = rng.random((1, h, w)) < 0.30
    pts[bad, 0] *= 0.5  # halving X doubles the implied fx for those pixels
    pts[bad, 1] *= 0.5

    k = _estimate_shared_intrinsics(pts, conf)

    assert k[0, 0] == pytest.approx(fx, rel=1e-2)
    assert k[1, 1] == pytest.approx(fy, rel=1e-2)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p, c: (p, np.zeros_like(c)), id="empty_conf_mask"),
        pytest.param(lambda p, c: (np.full_like(p, np.nan), c), id="nan_points"),
        pytest.param(lambda p, c: (p * np.array([1, 1, -1], np.float32), c), id="negative_depth"),
    ],
)
def test_degenerate_input_raises_instead_of_falling_back(mutate):
    # No 1.2*max(W,H) fallback focal, by design: a silently-wrong K is exactly the
    # regression class of be24be2, which produced a plausible mesh from a bad camera.
    h, w = 56, 70
    pts, conf = mutate(_synthetic_local_points(h, w, 80.0, 80.0), np.ones((1, h, w), np.float32))
    with pytest.raises(RuntimeError, match="intrinsics fit failed"):
        _estimate_shared_intrinsics(pts, conf)


def test_fit_accepts_trailing_axis_confidence():
    # LoGeR's conf head emits (N,H,W,1); the estimator must not require a squeeze
    # from its caller, since _forward and the tests reach it by different routes.
    h, w = 56, 70
    pts = _synthetic_local_points(h, w, 80.0, 80.0)
    k4 = _estimate_shared_intrinsics(pts, np.ones((1, h, w, 1), np.float32))
    k3 = _estimate_shared_intrinsics(pts, np.ones((1, h, w), np.float32))
    np.testing.assert_allclose(k4, k3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: `ImportError: cannot import name '_estimate_shared_intrinsics'`

- [ ] **Step 3: Implement the estimator**

Append to the "Intrinsics fit" section of `collab_splats/pointcloud/feedforward/loger.py`:

```python
def _estimate_shared_intrinsics(local_points: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Fit one pinhole K to LoGeR's camera-frame pointmap by confidence-weighted median.

    LoGeR has no intrinsics head, so K is solved rather than predicted.  Ported from
    github.com/PolyCam/LoGeR @ 5d7c1a7, ``run_loger.py``: ``estimate_focal_lengths``
    at :206 with ``_focal_from_frame`` at :180 inlined.  ``_snap_square_pixels`` at
    :195 is deliberately NOT ported — it would merge fx and fy at model resolution,
    which the separate-axis rescale downstream then un-merges incorrectly.

    Args:
        local_points: (N, H, W, 3) camera-frame points.  LoGeR builds these as
            ``cat([xy * z, z])`` (Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:772-775),
            so channel 2 is depth and ``xy`` is a free per-pixel ray field — not
            constrained to any pinhole K, which is why this fit is an approximation.
        conf: (N, H, W) or (N, H, W, 1) confidence, **already sigmoid-activated**.
            The ``> 0.1`` gate below is a threshold on a probability; on raw logits it
            would admit roughly half of all pixels instead of a deliberate floor.

    Returns:
        (3, 3) float32 K, shared across frames, centre-principal by construction.

    Raises:
        RuntimeError: if too few pixels survive to fit either focal.  There is no
            fallback focal on purpose.
    """
    n, h, w, _ = local_points.shape
    if conf.ndim == 4:
        conf = conf.squeeze(-1)

    # Centred pixel grid.  This line is why the function returns K and not (fx, fy):
    # every per-pixel focal below is conditioned on cx=(W-1)/2, cy=(H-1)/2, so the
    # principal point is already decided here and must not be re-chosen by a caller.
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )

    # Invert the pinhole model per pixel: u_c = fx * X / Z  =>  fx = u_c * Z / X.
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    valid = (z > 1e-3) & (np.abs(x) > 1e-6) & (np.abs(y) > 1e-6) & (conf > 0.1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_per_pixel = uu * z / x
        fy_per_pixel = vv * z / y

    fx_vals, fy_vals = fx_per_pixel[valid], fy_per_pixel[valid]
    weights = conf[valid]

    # Sanity bounds before the median: a focal outside [0.1, 10] image dimensions is a
    # degenerate inversion near the principal axis, not a plausible camera.
    ok_fx = (fx_vals > w * 0.1) & (fx_vals < w * 10)
    ok_fy = (fy_vals > h * 0.1) & (fy_vals < h * 10)
    fx = _weighted_median(fx_vals[ok_fx], weights[ok_fx])
    fy = _weighted_median(fy_vals[ok_fy], weights[ok_fy])

    # Fail loudly.  Upstream falls back to 1.2 * max(W, H); we do not, because a
    # plausible-but-wrong K fails silently all the way through to the mesh.
    if fx is None or fy is None or not math.isfinite(fx) or not math.isfinite(fy) or fx <= 0 or fy <= 0:
        raise RuntimeError(
            f"LoGeR intrinsics fit failed over {n} frames: "
            f"{int(valid.sum())}/{valid.size} pixels passed the validity mask, "
            f"{int(ok_fx.sum())} survived the fx bounds and {int(ok_fy.sum())} the fy bounds "
            f"(fx={fx}, fy={fy}). No fallback focal is applied by design."
        )

    # fx and fy stay distinct — our resize rounds each axis to a multiple of 14
    # independently, so the anisotropy is real and belongs in K.
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0],
         [0.0, fy, (h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 11 passed (4 from Task 2, 7 here)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): solve shared pinhole K from LoGeR's pointmap

LoGeR has no intrinsics head, so K is fitted rather than predicted: invert the
pinhole model per pixel and take a confidence-weighted median, shared across
frames. Returns the full 3x3 rather than (fx, fy) because the centred pixel
grid already fixes cx=(W-1)/2 — splitting that across caller and callee would
let a caller pick W/2 and introduce a silent half-pixel offset.

fx and fy stay independent: the resize rounds each axis to a multiple of 14
separately, so the anisotropy is real. Upstream's square-pixel snap is not
ported — averaging at model resolution is un-averaged wrongly by the
separate-axis rescale downstream.

No fallback focal on degenerate input; it raises."
```

---

## Task 4: `_loger_target_size`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
from collab_splats.pointcloud.feedforward.loger import _loger_target_size


@pytest.mark.parametrize(
    "orig_w,orig_h",
    [(1920, 1080), (1080, 1920), (640, 480), (1000, 1000), (3840, 2160)],
)
def test_target_size_is_patch_aligned_and_within_budget(orig_w, orig_h):
    # Both invariants are load-bearing: a non-multiple of 14 crashes the ViT
    # patch embedding, and exceeding the budget is what OOMs long sequences.
    w, h = _loger_target_size(orig_w, orig_h, pixel_limit=255_000)
    assert w % 14 == 0 and h % 14 == 0
    assert w >= 14 and h >= 14
    assert w * h <= 255_000


def test_target_size_preserves_orientation():
    # Landscape stays landscape. Independent per-axis rounding perturbs the exact
    # ratio by a few percent, but must never transpose it.
    w, h = _loger_target_size(1920, 1080, pixel_limit=255_000)
    assert w > h


def test_target_size_aspect_error_is_small_but_real():
    # The residual anisotropy is why fx and fy are fitted separately and why
    # camera_model must be PINHOLE. Assert it exists and is bounded — if a future
    # change makes it exactly zero, the separate-focal machinery is still correct
    # but this test documents why it is there.
    orig_w, orig_h = 1920, 1080
    w, h = _loger_target_size(orig_w, orig_h, pixel_limit=255_000)
    ratio_error = abs((w / h) / (orig_w / orig_h) - 1.0)
    assert ratio_error < 0.05


def test_target_size_upscales_small_images_to_the_budget():
    # The rule is an area budget, not a cap: a tiny input is scaled up to fill it.
    w, h = _loger_target_size(64, 48, pixel_limit=255_000)
    assert w * h > 200_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k target_size -v -p no:randomly`
Expected: `ImportError: cannot import name '_loger_target_size'`

- [ ] **Step 3: Implement**

Append a new section to `collab_splats/pointcloud/feedforward/loger.py`:

```python
########################################################################
########## Preprocessing ###############################################
########################################################################


def _loger_target_size(orig_w: int, orig_h: int, pixel_limit: int) -> tuple[int, int]:
    """LoGeR's own resize rule: scale to an area budget, then align both axes to 14.

    Ported from the VENDORED tree — github.com/Junyi42/LoGeR @ 7685b7a,
    ``loger/utils/basic.py:51-63`` (inside ``load_images_as_tensor``) — rather than
    from PolyCam's copy of the same arithmetic, since that is the tree we actually
    run against.

    The two axes round **independently**, so exact aspect ratio is not preserved —
    the image is stretched by up to a few percent on one axis.  That is in
    distribution (the model trains with this preprocessing) and is absorbed into K,
    because ``_estimate_shared_intrinsics`` fits fx and fy separately and
    ``LoGeRCreator.camera_model`` is ``"PINHOLE"``.  Cropping instead would discard
    field of view and reintroduce crop arithmetic in ``original_coords``.
    """
    # Area-budget scale factor
    scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1.0
    w_target, h_target = orig_w * scale, orig_h * scale

    # Round each axis to a whole number of patches, then shrink whichever axis is
    # furthest above the target aspect until the budget is met.
    k, m = round(w_target / _PATCH), round(h_target / _PATCH)
    while (k * _PATCH) * (m * _PATCH) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1

    return max(1, k) * _PATCH, max(1, m) * _PATCH
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): port LoGeR's patch-aligned resize rule

Cited to the vendored loger/utils/basic.py rather than PolyCam's copy of the
same arithmetic. Kept as a named function rather than inlined because it is the
sole source of the resize anisotropy that the separate fx/fy fit and the PINHOLE
camera model both exist to absorb, and it is the only part of _preprocess
testable without a model."
```

---

## Task 5: `LoGeRCreator._load_model`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
import yaml

from collab_splats.pointcloud.feedforward.loger import LoGeRCreator


def test_creator_defaults_match_upstream_effective_values():
    # These are NOT from the shipped yaml — both original_config.yaml files hold
    # only a model: key, so build_forward_kwargs' fallbacks are what actually run.
    # window_size/overlap_size are argparse defaults from PolyCam/LoGeR @ 5d7c1a7,
    # run_loger.py:47 and :49.
    c = LoGeRCreator()
    assert c.variant == "LoGeR_star"
    assert c.window_size == 32
    assert c.overlap_size == 3
    assert c.reset_every == 0
    assert c.num_iterations == 1
    assert c.pixel_limit == 255_000
    assert c.use_multiview_confidence is False


def test_creator_uses_pinhole_camera_model():
    # SIMPLE_PINHOLE averages (fx + fy) / 2 at COLMAP export (base.py:551-554),
    # silently discarding the anisotropy the separate-focal fit works to preserve.
    # vggtx.py sets SIMPLE_PINHOLE, so copying its class body would inherit the bug.
    assert LoGeRCreator().camera_model == "PINHOLE"


def test_unknown_variant_rejected_at_construction():
    with pytest.raises(ValueError, match="variant"):
        LoGeRCreator(variant="LoGeR_turbo")


def test_load_model_rejects_unknown_model_config_key(tmp_path, monkeypatch):
    # A forward-only key silently dropped is how LoGeR* would degrade invisibly:
    # se3 lives under model: but is popped inside forward (Junyi42/LoGeR @ 7685b7a,
    # loger/models/pi3.py:589), so a naive signature filter would discard it.
    # Anything else unknown must stop the run.
    from collab_splats.pointcloud.feedforward import loger as loger_mod

    ckpt_dir = tmp_path / "ckpts" / "LoGeR_star"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "original_config.yaml").write_text(
        yaml.safe_dump({"model": {"se3": True, "some_future_forward_kwarg": 3}})
    )
    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)

    with pytest.raises(ValueError, match="some_future_forward_kwarg"):
        LoGeRCreator()._load_model("cpu")


def test_load_model_reports_missing_config(tmp_path, monkeypatch):
    from collab_splats.pointcloud.feedforward import loger as loger_mod

    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="setup/loger.sh"):
        LoGeRCreator()._load_model("cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k creator -v -p no:randomly`
Expected: `ImportError: cannot import name 'LoGeRCreator'`

- [ ] **Step 3: Add imports, then the class and `_load_model`**

Extend the import block at the top of `collab_splats/pointcloud/feedforward/loger.py`:

```python
import ast
import inspect
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image

from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses

from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _raw_to_world_points,
    compute_multiview_depth_confidence,
)
from .vggtx import unproject_and_filter_points
```

Note two things the spec calls out. `_decode_verify_geometry` is **not** imported — it exists only to serve `_verify_loop_candidate`, and LoGeR refuses loop closure. `frames_as_pil_source` is **not** imported — it monkeypatches the process-global `PIL.Image.open` to drive path-based loaders, and LoGeR's loader takes a directory, so it cannot work here.

Append a new section:

```python
########################################################################
########## Creator #####################################################
########################################################################


@dataclass
class LoGeRCreator(BaseFeedforwardCreator):
    """Pointcloud via LoGeR: Pi3 backbone + TTT memory + sliding-window inference.

    Built for long sequences — the window bounds model memory regardless of sequence
    length, where the set-based VGGT family OOMs past a few hundred frames.

    Unlike every other backend, LoGeR predicts no intrinsics; K is solved from its
    camera-frame pointmap by ``_estimate_shared_intrinsics`` and shared across frames.

    Attributes:
        camera_model:   pycolmap camera model.  ``"PINHOLE"``, not ``"SIMPLE_PINHOLE"``,
                        because the fit produces genuinely distinct fx and fy and
                        SIMPLE_PINHOLE would average them away at COLMAP export.
        variant:        ``"LoGeR"`` or ``"LoGeR_star"``.  Selects a config-plus-weights
                        pair, not merely a weight file — the two ``original_config.yaml``
                        differ (``ttt_pre_norm`` vs ``se3``).
        model_path:     Local checkpoint override.  ``None`` → download from HuggingFace.
        window_size:    Sliding-window length.  Default from PolyCam/LoGeR @ 5d7c1a7,
                        ``run_loger.py:47`` (argparse).
        overlap_size:   Frames shared between adjacent windows.  Same source, :49.
        reset_every:    Hard-reset the TTT fast weights every N frames; ``0`` disables.
        num_iterations: TTT inner-loop iterations per step.
        pixel_limit:    Area budget for the resize.  Same source, :117.
        conf_threshold: Depth-confidence **percentile** cutoff (0–100), matching
                        ``vggt_omega``.  ``unproject_and_filter_points`` reads values
                        > 1.0 as a percentile and <= 1.0 as a raw confidence
                        (``vggtx.py:132-138``), so lowering this to e.g. ``0.5``
                        silently switches semantics rather than tightening the cut.
    """

    camera_model: str = "PINHOLE"

    variant: str = "LoGeR_star"
    model_path: str | None = None
    model_repo: str = LOGER_HF_REPO

    # Window knobs.  These do NOT come from the shipped yaml: both original_config.yaml
    # files contain only a model: key, so build_forward_kwargs (PolyCam/LoGeR @ 5d7c1a7,
    # run_loger.py:149-164) always falls through to these values.
    window_size: int = 32
    overlap_size: int = 3
    reset_every: int = 0
    num_iterations: int = 1

    pixel_limit: int = 255_000
    conf_threshold: float = 50.0
    use_multiview_confidence: bool = False
    mv_conf_threshold: float = 0.0

    # Resolved in _load_model from the variant's yaml; se3 is declared under model:
    # but is a forward kwarg, so it cannot ride along in the constructor kwargs.
    _se3: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.variant not in LOGER_VARIANTS:
            raise ValueError(f"variant must be one of {LOGER_VARIANTS}, got {self.variant!r}")

    def _load_model(self, device: str) -> Any:
        """Build Pi3 from the vendored per-variant yaml and load the HF checkpoint."""
        cfg_path = _LOGER_ROOT / "ckpts" / self.variant / "original_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"LoGeR config not found: {cfg_path}. Run `bash setup/loger.sh` to vendor the tree."
            )

        model_cfg = dict(yaml.safe_load(cfg_path.read_text()).get("model", {}))

        # se3 sits under model: but is not a Pi3.__init__ parameter — it is popped
        # inside forward (Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:589).
        # Route it out before validating the rest.
        self._se3 = bool(model_cfg.pop("se3", False))

        # Vendored tree is not pip-installed. The flag makes the finally idempotent:
        # without it, a nested load would pop a path its caller installed.
        root = str(_LOGER_ROOT)
        _patched = root not in sys.path
        if _patched:
            sys.path.insert(0, root)
        try:
            from loger.models.pi3 import Pi3  # noqa: PLC0415

            # Every remaining model: key must be a real constructor parameter. Raise
            # rather than drop — a silent drop is how a future forward-only key would
            # degrade the run invisibly, exactly as se3 would have.
            unknown = sorted(set(model_cfg) - set(inspect.signature(Pi3.__init__).parameters))
            if unknown:
                raise ValueError(
                    f"{cfg_path} 'model:' holds keys that are neither Pi3.__init__ "
                    f"parameters nor the known forward kwarg 'se3': {unknown}. "
                    "Upstream changed the config; route them explicitly rather than dropping them."
                )

            # Some checkpoints serialise list fields as strings, e.g. "[4,8]"
            for key in ("ttt_insert_after", "attn_insert_after"):
                if isinstance(model_cfg.get(key), str):
                    model_cfg[key] = ast.literal_eval(model_cfg[key])

            # yaml overrides Pi3 defaults, which are wrong for both shipped variants
            # (ttt_inter_multi is 4 in each config and 2 in code).
            model = Pi3(**model_cfg)

            ckpt = self.model_path or hf_hub_download(
                repo_id=self.model_repo, filename=f"{self.variant}/latest.pt"
            )
            state = torch.load(str(ckpt), map_location="cpu")
            state = state.get("model_state_dict", state)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        finally:
            if _patched:
                sys.path.remove(root)

        logger.info("LoGeRCreator: loaded %s (se3=%s) on %s", self.variant, self._se3, device)
        return model.eval().to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 24 passed

- [ ] **Step 5: Verify the real checkpoint loads**

Run:
```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.pointcloud.feedforward.loger import LoGeRCreator
for v in ('LoGeR', 'LoGeR_star'):
    c = LoGeRCreator(variant=v)
    m = c._load_model('cpu')
    print(v, 'se3 =', c._se3, '| params =', sum(p.numel() for p in m.parameters()) / 1e6, 'M')
"
```
Expected: both variants load with no exception; `LoGeR se3 = False`, `LoGeR_star se3 = True`.

If `_se3` is `False` for `LoGeR_star`, the yaml routing is broken — stop and fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add LoGeRCreator._load_model

Reads the vendored per-variant original_config.yaml, which is authoritative:
the two variants differ from each other and both set ttt_inter_multi=4 where
the Pi3 constructor defaults to 2.

se3 is routed out of the model: block explicitly because it is a forward kwarg,
popped inside forward at Junyi42/LoGeR @ 7685b7a loger/models/pi3.py:589, not a
constructor parameter. Any other unrecognised model: key raises rather than
being dropped — a silent drop is precisely how LoGeR* would have run in the
wrong alignment mode.

Window knobs are dataclass defaults, not yaml reads: both shipped configs hold
only a model: key, so the build_forward_kwargs fallbacks at PolyCam/LoGeR @
5d7c1a7 run_loger.py:149-164 are what actually run."
```

---

## Task 6: `_preprocess`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
def _fake_frames(n: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def test_preprocess_returns_patch_aligned_unit_range_tensor():
    frames = _fake_frames(4, 480, 640)
    views, image_paths, original_coords = LoGeRCreator()._preprocess(frames, [0, 5, 10, 15])

    assert views.shape[0] == 4 and views.shape[1] == 3
    assert views.shape[2] % 14 == 0 and views.shape[3] % 14 == 0
    assert views.dtype == torch.float32
    # unproject_and_filter_points reads these as colors; _forward asserts the range.
    assert float(views.min()) >= 0.0 and float(views.max()) <= 1.0
    assert [p.name for p in image_paths] == [
        "frame_000000", "frame_000005", "frame_000010", "frame_000015"
    ]


def test_preprocess_original_coords_is_full_frame():
    # LoGeR resizes and never crops, so every row is the whole image. This is what
    # _rescale_reconstruction_to_original_dimensions consumes.
    frames = _fake_frames(3, 480, 640)
    _, _, original_coords = LoGeRCreator()._preprocess(frames, [0, 1, 2])

    assert original_coords.shape == (3, 6)
    np.testing.assert_allclose(original_coords, np.tile([0, 0, 640, 480, 640, 480], (3, 1)))


def test_preprocess_rejects_non_uniform_frame_sizes():
    # LoGeR sizes from frame 0 alone; refuse rather than silently mis-resize the rest.
    frames = [_fake_frames(1, 480, 640)[0], _fake_frames(1, 240, 320)[0]]
    with pytest.raises(ValueError, match="uniform"):
        LoGeRCreator()._preprocess(frames, [0, 1])


def test_preprocess_rejects_out_of_order_frames():
    # Windows and overlap stitching assume temporal order; out-of-order input
    # degrades quality with no error. No other backend cares, so this is LoGeR's.
    frames = _fake_frames(3, 480, 640)
    with pytest.raises(ValueError, match="ascending"):
        LoGeRCreator()._preprocess(frames, [0, 10, 5])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k preprocess -v -p no:randomly`
Expected: 4 failed, `AttributeError` or `NotImplementedError` from the abstract method

- [ ] **Step 3: Implement**

Add to `LoGeRCreator`, after `_load_model`:

```python
    def _preprocess(self, frames: Any, frame_idxs: list[int]) -> tuple[Any, list[Path], np.ndarray]:
        """Resize decoded frames to LoGeR's patch-aligned budget; return (N,3,H,W) in [0,1]."""
        # Windows and overlap stitching assume temporal order. frames.zarr is ordered
        # by construction today, so this guards an assumption rather than a known bug.
        if any(b <= a for a, b in zip(frame_idxs, frame_idxs[1:])):
            raise ValueError(
                f"LoGeR requires strictly ascending frame_idxs (sliding-window inference); got {frame_idxs}"
            )

        # LoGeR derives the target size from frame 0 alone. Rather than inherit that
        # silent assumption, refuse mixed sizes.
        shapes = {(int(f.shape[0]), int(f.shape[1])) for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"LoGeR needs uniform frame sizes; got {sorted(shapes)}")

        orig_h, orig_w = shapes.pop()
        target_w, target_h = _loger_target_size(orig_w, orig_h, self.pixel_limit)
        logger.debug("LoGeRCreator: %dx%d -> %dx%d", orig_w, orig_h, target_w, target_h)

        # Resize in memory with PIL directly. No frames_as_pil_source monkeypatch:
        # that helper drives path-based loaders, and LoGeR's load_images_as_tensor
        # enumerates a directory with os.listdir, which patching Image.open cannot reach.
        resized = np.stack([
            np.asarray(Image.fromarray(f).resize((target_w, target_h), Image.LANCZOS))
            for f in frames
        ])
        views = torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255.0

        # Stable synthetic labels — the frame store is the sole IO path, no filenames exist
        image_paths = [Path(f"frame_{idx:06d}") for idx in frame_idxs]

        # Pure resize, no crop, so every row is the full original frame
        original_coords = np.tile(
            np.array([0, 0, orig_w, orig_h, orig_w, orig_h], dtype=np.float32), (len(image_paths), 1)
        )

        return views, image_paths, original_coords
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 28 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add LoGeRCreator._preprocess

Resizes decoded frames with PIL directly rather than through
frames_as_pil_source: that helper monkeypatches the process-global
PIL.Image.open to drive path-based loaders, and LoGeR's loader enumerates a
directory with os.listdir, so patching cannot reach it. Six lines of PIL drops
a global-state monkeypatch and its single-threaded caveat.

original_coords is the full frame on every row — LoGeR resizes and never crops.

Two guards: uniform frame sizes (LoGeR sizes from frame 0 alone) and strictly
ascending frame indices (windows assume temporal order and degrade silently
without it)."
```

---

## Task 7: `_forward`

The sigmoid-before-fit ordering lives here and is load-bearing.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
class _FakeLoGeR(torch.nn.Module):
    """Minimal stand-in for Pi3 that returns a known pinhole scene.

    Emits RAW conf logits deliberately outside [0, 1], so any path that forgets
    torch.sigmoid produces an out-of-range depth_conf the assertions catch.
    """

    def __init__(self, n: int, h: int, w: int, fx: float, fy: float):
        super().__init__()
        self.register_parameter("_p", torch.nn.Parameter(torch.zeros(1)))
        pts = _synthetic_local_points(h, w, fx, fy)  # (1,H,W,3)
        self.local_points = torch.from_numpy(np.repeat(pts, n, axis=0))[None]  # (1,N,H,W,3)
        self.conf = torch.full((1, n, h, w, 1), 4.0)  # logit 4.0 -> sigmoid ~0.982
        # Distinct non-identity c2w poses: camera i sits at x = i along the world axis
        poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        poses[:, 0, 3] = np.arange(n, dtype=np.float32)
        self.camera_poses = torch.from_numpy(poses)[None]  # (1,N,4,4)
        self.seen_kwargs: dict = {}

    def forward(self, images, **kwargs):
        self.seen_kwargs = kwargs
        return {
            "local_points": self.local_points,
            "conf": self.conf,
            "camera_poses": self.camera_poses,
            "points": self.local_points,
        }


def test_forward_applies_sigmoid_to_raw_confidence_logits():
    # LoGeR's conf_head is a bare LinearPts3d with no activation (Junyi42/LoGeR @
    # 7685b7a, loger/models/pi3.py:172); upstream applies sigmoid at the call site
    # (PolyCam/LoGeR @ 5d7c1a7, run_loger.py:481). The K fit's conf > 0.1 gate is a
    # threshold on a probability, so this must run first.
    n, h, w = 3, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    views = torch.rand(n, 3, h, w)

    raw = LoGeRCreator()._forward(model, views)

    assert raw["depth_conf"].shape == (n, h, w)
    assert raw["depth_conf"].min() >= 0.0 and raw["depth_conf"].max() <= 1.0
    assert raw["depth_conf"].max() == pytest.approx(1 / (1 + np.exp(-4.0)), rel=1e-4)


def test_forward_inverts_camera_poses_to_world_to_camera():
    # LoGeR returns camera-to-world; FeedforwardResult.extrinsics is world-to-camera.
    # The single easiest thing to get backwards, and silent when wrong.
    n, h, w = 3, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)

    raw = LoGeRCreator()._forward(model, torch.rand(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)
    # c2w camera 2 sits at x=+2, so the w2c translation must be -2, not +2.
    assert raw["extrinsic"][2, 0, 3] == pytest.approx(-2.0)


def test_forward_fits_and_broadcasts_intrinsics():
    n, h, w = 3, 56, 70
    fx, fy = 88.0, 80.0
    raw = LoGeRCreator()._forward(_FakeLoGeR(n, h, w, fx, fy), torch.rand(n, 3, h, w))

    assert raw["intrinsics"].shape == (n, 3, 3)
    assert raw["intrinsics"][0, 0, 0] == pytest.approx(fx, rel=1e-3)
    assert raw["intrinsics"][0, 1, 1] == pytest.approx(fy, rel=1e-3)
    # _raw_to_world_points hard-requires this key (vggtx.py:304) and returns None without it.
    np.testing.assert_allclose(raw["intrinsics_downsampled"], raw["intrinsics"])


def test_forward_extracts_depth_from_the_third_channel():
    # LoGeR builds local_points as cat([xy * z, z]), so channel 2 IS depth — no
    # reprojection needed to recover it.
    n, h, w = 2, 56, 70
    raw = LoGeRCreator()._forward(_FakeLoGeR(n, h, w, 80.0, 80.0), torch.rand(n, 3, h, w))
    assert raw["depth"].shape == (n, h, w, 1)
    np.testing.assert_allclose(raw["depth"], 2.0, rtol=1e-5)


def test_forward_passes_window_knobs_and_se3():
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    creator = LoGeRCreator(window_size=16, overlap_size=4)
    creator._se3 = True

    creator._forward(model, torch.rand(n, 3, h, w))

    assert model.seen_kwargs["window_size"] == 16
    assert model.seen_kwargs["overlap_size"] == 4
    assert model.seen_kwargs["se3"] is True
    assert model.seen_kwargs["sim3"] is False


def test_forward_rejects_rgb_outside_unit_range():
    # Guards the a157421 [0,255] bug class at the source.
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    with pytest.raises(AssertionError, match=r"\[0, 1\]"):
        LoGeRCreator()._forward(model, torch.rand(n, 3, h, w) * 255.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k forward -v -p no:randomly`
Expected: 6 failed

- [ ] **Step 3: Implement**

Add to `LoGeRCreator`, after `_preprocess`:

```python
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run windowed LoGeR inference; return raw outputs plus a solved shared K."""
        device = next(model.parameters()).device
        images = views.to(device)

        # Guards the a157421 [0,255] bug class at the source rather than at the mesh.
        assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0, (
            f"LoGeR expects RGB in [0, 1]; got [{float(images.min())}, {float(images.max())}]"
        )

        # Pi3 takes (B, N, 3, H, W). Kwargs mirror build_forward_kwargs in
        # PolyCam/LoGeR @ 5d7c1a7, run_loger.py:149-164, so behaviour matches
        # upstream exactly.
        with torch.no_grad():
            preds = model(
                images[None],
                window_size=self.window_size,
                overlap_size=self.overlap_size,
                reset_every=self.reset_every,
                num_iterations=self.num_iterations,
                sim3=False,
                sim3_scale_mode="median",
                se3=self._se3,
                turn_off_ttt=False,
                turn_off_swa=False,
            )

        local_points = preds["local_points"].squeeze(0).cpu().float().numpy()  # (N,H,W,3)

        # conf_head is a bare LinearPts3d with NO output activation — Junyi42/LoGeR @
        # 7685b7a, loger/models/pi3.py:172 — so the model emits logits, and upstream
        # activates at the call site (PolyCam/LoGeR @ 5d7c1a7, run_loger.py:481).
        # This must run before the K fit, whose conf > 0.1 gate is a threshold on a
        # probability: on raw logits it would admit roughly half of all pixels.
        depth_conf = torch.sigmoid(preds["conf"]).squeeze(0).cpu().float().numpy()
        if depth_conf.ndim == 4:
            depth_conf = depth_conf.squeeze(-1)  # (N,H,W)

        # LoGeR returns camera-to-world; FeedforwardResult.extrinsics is world-to-camera.
        camera_poses = preds["camera_poses"].squeeze(0).cpu().float().numpy()  # (N,4,4) c2w
        extrinsic = invert_poses(camera_poses)[:, :3, :].astype(np.float32)  # (N,3,4) w2c

        # LoGeR predicts no intrinsics — solve one shared K and broadcast it per frame.
        k = _estimate_shared_intrinsics(local_points, depth_conf)
        intrinsic = np.broadcast_to(k, (local_points.shape[0], 3, 3)).copy()

        # Channel 2 IS depth: the model builds local_points as cat([xy * z, z]) at
        # Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:772-775.
        depth = local_points[..., 2:3]  # (N,H,W,1)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,
            "intrinsics_downsampled": intrinsic,  # alias — _raw_to_world_points needs this key
            "depth": depth,
            "depth_conf": depth_conf,
            "local_points": local_points,  # kept for the parity test only
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 34 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add LoGeRCreator._forward

Three conversions the rest of the pipeline depends on:

- sigmoid on conf. LoGeR's conf_head is a bare LinearPts3d with no activation
  (Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:172); upstream activates at the
  call site (PolyCam/LoGeR @ 5d7c1a7, run_loger.py:481). It runs before the K
  fit, whose conf > 0.1 gate thresholds a probability — on raw logits it would
  admit roughly half of all pixels instead of a 10% floor.
- invert_poses on camera_poses. LoGeR returns c2w, FeedforwardResult wants w2c.
- depth straight off channel 2, since the model builds cat([xy * z, z]).

K is solved and broadcast per frame, aliased to intrinsics_downsampled because
_raw_to_world_points hard-requires that key."
```

---

## Task 8: `_postprocess`, `_reproject`, and the LC refusal

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
def _forward_and_postprocess(n=3, h=56, w=70, fx=88.0, fy=80.0, **creator_kwargs):
    creator = LoGeRCreator(**creator_kwargs)
    views, image_paths, original_coords = creator._preprocess(_fake_frames(n, 112, 140), list(range(n)))
    creator.image_paths, creator.original_coords = image_paths, original_coords
    raw = creator._forward(_FakeLoGeR(n, views.shape[2], views.shape[3], fx, fy), views)
    return creator, raw, creator._postprocess(raw)


def test_postprocess_field_contract():
    n = 3
    _, raw, result = _forward_and_postprocess(n=n)

    assert result.extrinsics.shape == (n, 4, 4)
    assert result.intrinsics.shape == (n, 3, 3)
    # depth is stored (N,H,W), trailing axis squeezed, matching every other backend
    assert result.depth.ndim == 3 and result.depth.shape[0] == n
    assert result.world_points is not None
    assert result.world_points.shape == (n, result.model_height, result.model_width, 3)
    assert result.colors.dtype == np.uint8
    assert result.points.shape[1] == 3 and len(result.points) == len(result.colors)


def test_postprocess_preserves_anisotropic_focals():
    # camera_model PINHOLE keeps fx and fy; SIMPLE_PINHOLE would average them at
    # COLMAP export (base.py:551-554) and silently destroy the aspect correction.
    creator, _, result = _forward_and_postprocess(fx=88.0, fy=80.0)
    assert creator.camera_model == "PINHOLE"
    assert result.intrinsics[0, 0, 0] != pytest.approx(result.intrinsics[0, 1, 1], rel=1e-3)


def test_reproject_returns_points_and_colors_for_refined_poses():
    creator, raw, result = _forward_and_postprocess()
    pts, colors = creator._reproject(raw, raw["extrinsic"], raw["intrinsics"])
    assert pts.shape[1] == 3
    assert len(pts) == len(colors)


def test_extract_intermediate_features_refuses():
    # LC is out of scope for the first cut: thresholds are per-backbone and
    # uncalibrated here. _verify_loop_candidate is concrete on the base class and
    # calls this, so the refusal must be explicit.
    with pytest.raises(NotImplementedError, match="loop closure"):
        LoGeRCreator().extract_intermediate_features(torch.rand(2, 3, 56, 70))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k "postprocess or reproject or intermediate" -v -p no:randomly`
Expected: 4 failed

- [ ] **Step 3: Implement**

Add to `LoGeRCreator`, after `_forward`:

```python
    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """Unproject depth with the fitted K and build the FeedforwardResult."""
        extrinsic = raw_outputs["extrinsic"]  # (N,3,4) at model resolution
        intrinsic = raw_outputs["intrinsics"]  # (N,3,3) at model resolution

        # Optional geometric cross-view depth consistency mask
        mv_mask = None
        if self.use_multiview_confidence:
            depth_np = raw_outputs["depth"]
            if depth_np.ndim == 4:
                depth_np = depth_np.squeeze(-1)
            mv_conf = compute_multiview_depth_confidence(
                depth_np,
                intrinsic,
                extrinsics_to_homogeneous(extrinsic),
                abs_thresh=0.0,
                rel_thresh=0.05,
            )
            mv_mask = mv_conf > self.mv_conf_threshold

        # Unproject to filtered world-space points and per-point colors.
        # conf_threshold > 1.0 is read as a percentile by this function (vggtx.py:132-138).
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
        )

        depth = raw_outputs["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(-1)  # (N,H,W)
        model_h, model_w = int(depth.shape[1]), int(depth.shape[2])

        # BA fields: dense world-point grid, matching vggtx and vggt_omega. LoGeR's own
        # `points` is NOT used here — it can encode non-pinhole geometry that the fitted
        # K cannot reproduce, so feeding it to BA alongside that K makes BA fight the
        # model. The parity test measures the gap between the two clouds.
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
        world_points = (
            world_pts_flat.reshape(world_pts_flat.shape[0], model_h, model_w, 3)
            if world_pts_flat is not None
            else None
        )

        return FeedforwardResult(
            points=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            features=None,
            extrinsics=extrinsics_to_homogeneous(extrinsic),
            intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=raw_outputs["images"],
            confidence=torch.from_numpy(raw_outputs["depth_conf"]),
            world_points=world_points,
            depth=depth,
        )

    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
        )
        return pts3d, colors  # pixel_indices unused; post-BA uses stored indices

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Not supported — LoGeR carries its own windowed TTT memory across frames."""
        # Satisfying the ABC contract, not a courtesy stub: the class will not
        # instantiate without it, and _verify_loop_candidate (concrete on the base
        # class, base.py:960) calls it. Reaching here means the Reconstructor-level
        # loop closure refusal was bypassed.
        raise NotImplementedError(
            "LoGeR does not support loop closure feature extraction. Its windowed TTT "
            "fast-weight memory already carries state across frames, and LC verification "
            "thresholds are calibrated per backbone (see the spec). Use vggt_omega, vggtx, "
            "or mapanything if loop closure is required."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: 38 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add _postprocess, _reproject, and the LC refusal

world_points comes from _raw_to_world_points, matching vggtx and vggt_omega,
rather than from LoGeR's own `points`. The native cloud is free and already
world-space, but it can encode non-pinhole geometry the fitted K cannot
reproduce, so feeding it to BA alongside that K makes BA fight the model. The
parity test measures the gap.

extract_intermediate_features raises: it satisfies the ABC contract (the class
will not instantiate without it) and _verify_loop_candidate is concrete on the
base class, so without an explicit refusal an LC run would do a full forward
pass and then die deep inside the LC loop."
```

---

## Task 9: Registry and package exports

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `tests/pointcloud/test_registry.py`
- Create: `tests/pointcloud/feedforward/test_loger_load_guard.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_registry.py`:

```python
_LOGER_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "LoGeR"


@pytest.mark.skipif(not _LOGER_ROOT.exists(), reason="third_party/LoGeR not vendored")
def test_get_creator_loger():
    from collab_splats.pointcloud.feedforward import LoGeRCreator

    assert get_creator("loger") is LoGeRCreator
```

Add `from pathlib import Path` to that file's imports.

Create `tests/pointcloud/feedforward/test_loger_load_guard.py`:

```python
"""LoGeR fails at _load_model time, not import time — the guard that keeps the
registry importable in a bare checkout with no vendored tree."""
import pytest

from collab_splats.pointcloud.feedforward.loger import _LOGER_ROOT, LoGeRCreator


def test_module_imports_without_the_vendored_tree():
    # loger.py must not import from third_party at module level. If it did, a bare
    # checkout would break `import collab_splats.pointcloud` entirely rather than
    # just omitting the backend. The import at the top of this file IS the assertion.
    assert LoGeRCreator is not None


def test_load_model_names_the_setup_script(tmp_path, monkeypatch):
    from collab_splats.pointcloud.feedforward import loger as loger_mod

    monkeypatch.setattr(loger_mod, "_LOGER_ROOT", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="setup/loger.sh"):
        LoGeRCreator()._load_model("cpu")


def test_root_points_at_third_party():
    assert _LOGER_ROOT.name == "LoGeR"
    assert _LOGER_ROOT.parent.name == "third_party"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_registry.py tests/pointcloud/feedforward/test_loger_load_guard.py -v -p no:randomly`
Expected: `test_get_creator_loger` fails with `ImportError: cannot import name 'LoGeRCreator'`

- [ ] **Step 3: Wire the exports**

In `collab_splats/pointcloud/feedforward/__init__.py`, after the `vggt_omega` try/except block:

```python
# loger is an optional backend — the model tree is vendored by setup/loger.sh into
# the gitignored third_party/. The vendored import is deferred to _load_model, so
# this only guards against the module itself being absent.
try:
    from .loger import LoGeRCreator
except ImportError:
    pass
```

and add `"LoGeRCreator"` to `__all__`.

In `collab_splats/pointcloud/__init__.py`, after the `_SPARK_AVAILABLE` block:

```python
try:
    from .feedforward import LoGeRCreator
    _LOGER_AVAILABLE = True
except ImportError:
    _LOGER_AVAILABLE = False
```

and after `_REGISTRY["vggt_spark"] = ...`:

```python
if _LOGER_AVAILABLE:
    _REGISTRY["loger"] = LoGeRCreator
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ -v -p no:randomly`
Expected: all pass, including `test_get_creator_loger`

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py tests/pointcloud/test_registry.py tests/pointcloud/feedforward/test_loger_load_guard.py
git commit -m "feat(loger): register the loger backend

Guarded export mirroring VGGTOmegaCreator. The vendored tree is imported inside
_load_model behind a sys.path insert, so a bare checkout omits the backend
rather than breaking `import collab_splats.pointcloud`.

The registry test is skipif-guarded on the tree being present, unlike
vggt_omega and vggt_spark which are simply omitted — the guarded test is the
only thing that proves the wiring."
```

---

## Task 10: Reconstructor wiring

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
from collab_splats.wrapper.reconstructor import _FEEDFORWARD_BACKENDS


def test_loger_is_a_recognised_feedforward_backend():
    # vggt_spark is in _REGISTRY but absent here, so it is unreachable from
    # Reconstructor. loger must be in both.
    assert "loger" in _FEEDFORWARD_BACKENDS


def test_loop_closure_with_loger_is_refused(tmp_path):
    # Refuse at the Reconstructor level, before any inference. _verify_loop_candidate
    # is concrete on the base class, so without this an LC run would burn a full
    # forward pass and then raise NotImplementedError deep in the LC loop.
    from collab_splats.wrapper.reconstructor import _run_feedforward

    with pytest.raises(ValueError, match="loop closure"):
        _run_feedforward(
            backend="loger",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure=True,
            viz_enabled=False,
            viz_port=8080,
            max_points=1000,
        )


def test_creator_kwargs_reach_the_constructor():
    # The per-backend config block is the only way to set model knobs from yaml.
    creator = LoGeRCreator(max_points=1234, window_size=64, variant="LoGeR")
    assert creator.max_points == 1234
    assert creator.window_size == 64
    assert creator.variant == "LoGeR"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k "backend or loop_closure or creator_kwargs" -v -p no:randomly`
Expected: 2 failed (`loger` not in the set; no `ValueError` raised)

- [ ] **Step 3: Wire the Reconstructor**

In `collab_splats/wrapper/reconstructor.py`:

1. Line 43 — add `loger`:
```python
_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega", "loger"}
```

2. In `_run_feedforward`'s signature (line 143), add a keyword parameter after `max_points`:
```python
    max_points: int,
    creator_kwargs: dict | None = None,
```

3. In the deferred import block (around line 165), add `LoGeRCreator`:
```python
    from collab_splats.pointcloud.feedforward import (
        LoGeRCreator,
        MapAnythingCreator,
        VGGTOmegaCreator,
        VGGTXCreator,
    )
```

4. Immediately after that import block, before the `loop_closure` normalisation, add the refusal and the warning:
```python
    # LoGeR refuses loop closure in this cut. Refuse here rather than in the creator:
    # _verify_loop_candidate is concrete on BaseFeedforwardCreator, so an LC run would
    # otherwise complete a full forward pass before dying inside the LC loop. LC verify
    # thresholds are calibrated per backbone and none exists for LoGeR.
    if backend == "loger" and loop_closure:
        raise ValueError(
            "pointcloud.loop_closure is not supported with backend 'loger'. LoGeR's windowed "
            "TTT memory already carries state across frames, and LC verification thresholds "
            "are calibrated per backbone. Use vggt_omega, vggtx, or mapanything for LC."
        )
```

5. Replace line 195:
```python
    # max_points caps the confidence mask during inference — a memory guard, not a preference
    creator_map = {
        "vggtx": VGGTXCreator,
        "mapanything": MapAnythingCreator,
        "vggt_omega": VGGTOmegaCreator,
        "loger": LoGeRCreator,
    }
    # Reject max_points from the config block — it is already passed explicitly and a
    # duplicate would surface as an opaque TypeError. Unknown keys are left to the
    # constructor's own TypeError, which names them correctly.
    extra = dict(creator_kwargs or {})
    if "max_points" in extra:
        raise ValueError(
            f"pointcloud.{backend}.max_points is not settable; use pointcloud.max_points"
        )
    creator = creator_map[backend](max_points=max_points, **extra)
```

6. At the call site, line 552, add the passthrough:
```python
            max_points=pc_cfg["max_points"],
            creator_kwargs=pc_cfg.get(pc_cfg["backend"], {}),
```

- [ ] **Step 4: Add the `max_frames` warning**

Immediately after the LC refusal in `_run_feedforward`, add:

```python
    # max_frames: 300 is a VGGT-Omega GPU property that lives in the preproc stage, and
    # preproc has already run by the time we get here. Flipping to loger at defaults
    # therefore processes exactly as many frames as Omega would, and LoGeR appears to
    # buy nothing. Warn rather than change behaviour — the true ceiling is unmeasured.
    if backend == "loger":
        # FrameStore is already imported at reconstructor.py:23; count is __len__.
        n_frames = len(FrameStore.open(frames_zarr)) if Path(frames_zarr).exists() else 0
        if 0 < n_frames <= 300:
            logger.warning(
                "LoGeR is running on %d frames. preprocessing.max_frames defaults to 300, "
                "which is VGGT-Omega's GPU limit, not LoGeR's — LoGeR uses sliding-window "
                "inference and is built for longer sequences. Raise max_frames to use it.",
                n_frames,
            )
```

`FrameStore` is already imported at `reconstructor.py:23`, so no new import is needed. The count is `__len__` (`frame_store.py:65`) — there is no `.count` attribute. The warning is best-effort and must never raise.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ tests/wrapper/ -v -p no:randomly
```
Expected: all pass. The three previously-failing tests now pass, and no existing `_run_feedforward` test regresses — `creator_kwargs` is defaulted, so existing callers are unaffected.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): wire loger into Reconstructor

Adds the backend to _FEEDFORWARD_BACKENDS and creator_map, refuses loop closure
before any inference runs, and warns when LoGeR runs under the default
max_frames of 300 — that cap is VGGT-Omega's GPU limit sitting in the preproc
stage, so flipping the backend at defaults silently buys nothing.

Also adds a generic per-backend creator_kwargs passthrough: every backend now
has a config-file surface for its constructor kwargs, where previously a
creator's tunables were reachable only by constructing it in Python. Nested
under pointcloud.<backend> rather than flattened, because pointcloud: also
holds pipeline-level keys that would TypeError the constructor."
```

---

## Task 11: `original_coords` round-trip to original-resolution K

This belongs in `test_feedforward_intrinsics.py`, which already owns "result.intrinsics at model resolution" across backends.

**Files:**
- Modify: `tests/pointcloud/test_feedforward_intrinsics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/test_feedforward_intrinsics.py`:

```python
def test_loger_original_coords_rescale_recovers_anisotropic_focals():
    """LoGeR's model-res K must rescale to original resolution with fx != fy intact.

    This is the third of the three places that keep fx and fy distinct. Our resize
    rounds each axis to a multiple of 14 independently, so a square-pixel camera
    genuinely produces fx != fy at model resolution; _rescale_... divides by
    scale_x and scale_y separately, which recovers the true focals exactly.
    """
    import numpy as np

    from collab_splats.pointcloud.feedforward.loger import _loger_target_size

    orig_w, orig_h = 1920, 1080
    model_w, model_h = _loger_target_size(orig_w, orig_h, pixel_limit=255_000)

    # A square-pixel physical camera, f = 1600 px at original resolution
    f = 1600.0
    scale_x, scale_y = model_w / orig_w, model_h / orig_h
    k_model = np.array(
        [[f * scale_x, 0.0, (model_w - 1) / 2.0],
         [0.0, f * scale_y, (model_h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    # Independent per-axis rounding makes these genuinely different at model res
    assert k_model[0, 0] != pytest.approx(k_model[1, 1], rel=1e-3)

    # The rescale is exactly this division, per axis (base.py:580)
    fx_orig = k_model[0, 0] / scale_x
    fy_orig = k_model[1, 1] / scale_y
    assert fx_orig == pytest.approx(f, rel=1e-5)
    assert fy_orig == pytest.approx(f, rel=1e-5)

    # And this is why the square-pixel snap is not ported: snapping at model
    # resolution would set both to their mean, which the separate-axis division
    # then un-averages incorrectly.
    snapped = (k_model[0, 0] + k_model[1, 1]) / 2.0
    assert snapped / scale_x != pytest.approx(f, rel=1e-4)
```

- [ ] **Step 2: Run the test**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py -k loger -v -p no:randomly`
Expected: PASS (the test asserts arithmetic that Tasks 3 and 4 already made true)

If the final assertion fails because `scale_x == scale_y` exactly for `1920x1080` at this budget, the resize happened to be isotropic for that input. Substitute an input size where `_loger_target_size` produces different per-axis scales — check with:
```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.pointcloud.feedforward.loger import _loger_target_size
for ow, oh in [(1920,1080),(1280,720),(640,480),(3840,2160),(1440,1080)]:
    w, h = _loger_target_size(ow, oh, 255_000)
    print(ow, oh, '->', w, h, '| sx/sy =', (w/ow)/(h/oh))
"
```
Pick a row where `sx/sy` differs from 1.0 by more than 0.5%.

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_feedforward_intrinsics.py
git commit -m "test(loger): pin the model-res to original-res K round-trip

Lives alongside the existing cross-backend intrinsics-resolution tests rather
than in test_loger_creator.py, since that file already owns this contract.

The final assertion is the one that matters: it shows averaging fx and fy at
model resolution does NOT round-trip, which is the concrete reason upstream's
_snap_square_pixels is not ported."
```

---

## Task 12: Config and docs

**Files:**
- Modify: `configs/base.yaml`
- Modify: `configs/README.md`
- Modify: `docs/source/api/pointcloud.rst`

- [ ] **Step 1: Update `configs/base.yaml`**

**`configs/base.yaml` is being edited concurrently** (a `preprocessing.frame_proportion` removal as of 2026-08-13). Read the file immediately before editing, match on content rather than the line numbers quoted here, and stage with an explicit pathspec at commit time so an unrelated concurrent edit is not swept in.

Replace the `max_frames` and `backend` lines and add the block. `max_frames` becomes:

```yaml
  max_frames: 300             # cap — vggt_omega OOMs above ~300 on 44 GB GPU.
                              # NOT a LoGeR limit: LoGeR is windowed and built for
                              # longer sequences, but preproc runs first, so this
                              # caps it too. See configs/README.md.
```

The `backend` line and the new block become:

```yaml
  backend: vggt_omega         # vggt_omega | vggtx | mapanything | loger  (feedforward only)
  # Per-backend creator kwargs. Only the block matching `backend` is read, so all
  # four can be documented here at once. max_points is NOT settable here — it is a
  # pipeline-level guard above.
  loger:
    variant: LoGeR_star       # LoGeR | LoGeR_star (SE(3)); selects config AND weights
    window_size: 32           # sliding-window length (PolyCam/LoGeR run_loger.py default)
    overlap_size: 3           # frames shared between adjacent windows
    reset_every: 0            # hard-reset TTT fast weights every N frames; 0 = never
    conf_threshold: 50.0      # depth-confidence PERCENTILE (0-100), not a raw value
```

Note the stale "cap at 200" comment beside a value of 300 is corrected as part of this.

- [ ] **Step 2: Update `configs/README.md`**

Same concurrency caution as Step 1 — match on row content, not line numbers.

The `preprocessing.max_frames` row becomes:
```markdown
| `preprocessing.max_frames` | int\|null | `300` | Cap on frames (vggt_omega OOMs above ~300; not a LoGeR limit — see below) |
```

The `pointcloud.backend` row becomes:
```markdown
| `pointcloud.backend` | str | `vggt_omega` | `vggt_omega`, `vggtx`, `mapanything`, or `loger` |
```

Add a new row directly after it:
```markdown
| `pointcloud.<backend>` | dict | `{}` | Per-backend creator kwargs, e.g. `pointcloud.loger.window_size`. Only the block matching `backend` is read. `max_points` is rejected here. |
```

Add a new section after the config table:

```markdown
### The `loger` backend

LoGeR is a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window
inference. Requires `bash setup/loger.sh` once; weights download from HuggingFace on
first use.

**Choose it for:** sequences past the ~300-frame ceiling where VGGT-Omega OOMs, and long
captures where drift accumulates — the TTT memory is designed to carry state across the
sequence.

**Avoid it for:** short sequences (<100 frames), where the set-based VGGT models see every
frame jointly and the windowing buys nothing; anything needing loop closure, which
`loger` refuses (thresholds are calibrated per backbone and none exists yet); captures
where intrinsics genuinely vary, e.g. zoom, which the shared-K fit cannot represent; and
unordered image collections — LoGeR's windows are sequential, whereas the VGGT family is
set-based and has no ordering requirement.

**Intrinsics differ from every other backend.** VGGT-family backends and MapAnything
*predict* K. LoGeR does not: K is *solved* from its predicted pointmap by a
confidence-weighted median pinhole fit, shared across all frames. The failure modes are
inverted — a predicted K can be geometrically invalid (a principal point outside the
image), whereas a fitted K is centre-principal by construction but can be
plausibly-but-globally-wrong. There is no fallback focal; a degenerate fit raises.

**`max_frames` is not tuned for LoGeR.** The default 300 is VGGT-Omega's GPU limit and
lives in the preproc stage, which runs first. Raise it to use LoGeR's windowing. The real
ceiling is `FeedforwardResult`, which holds dense per-frame images, world points, depth,
and confidence — roughly 8 MB/frame at the default pixel budget — against a 46.6 GB
container cap. That limit applies to every backend equally; LoGeR is merely the first one
able to feed it enough frames to matter.
```

- [ ] **Step 3: Update `docs/source/api/pointcloud.rst`**

Append:
```rst
.. automodule:: collab_splats.pointcloud.feedforward.loger
   :members:
   :show-inheritance:
```

No `docs/source/conf.py` change is needed. The vendored import happens inside `_load_model` behind the `sys.path` insert, so autodoc never imports `loger`. (`vggt` and `mapanything` are in `autodoc_mock_imports` only because `vggtx.py` and `mapanything.py` import them at module level.)

- [ ] **Step 4: Verify the docs build**

Run:
```bash
cd docs && /opt/venv/reconstruction/bin/python -m sphinx -b html source _build/html -q 2>&1 | tail -20
```
Expected: no error mentioning `loger`. Pre-existing warnings unrelated to this module are acceptable — compare against `git stash` if unsure.

- [ ] **Step 5: Verify config load**

Run:
```bash
/opt/venv/reconstruction/bin/python -c "
import yaml
cfg = yaml.safe_load(open('configs/base.yaml'))
print(cfg['pointcloud']['loger'])
assert cfg['pointcloud']['loger']['overlap_size'] == 3
assert cfg['preprocessing']['max_frames'] == 300
print('OK')
"
```
Expected: the dict, then `OK`.

- [ ] **Step 6: Commit**

```bash
git add configs/base.yaml configs/README.md docs/source/api/pointcloud.rst
git commit -m "docs(loger): document the loger backend and per-backend kwargs

Adds the pointcloud.loger block to base.yaml, which keeps its role as the single
source of defaults: all four backends' knobs can be documented at once because
only the block matching `backend` is read.

README gains a loger section covering when to choose it, how its solved
intrinsics differ from every other backend's predicted K, and why max_frames is
not tuned for it. Corrects the stale 'cap at 200' comment beside a value of 300."
```

---

## Task 13: Parity integration test and the residual measurement

Spec open item 3. This one assertion proves the focal fit, the pose inversion, the depth extraction, and that reusing `unproject_and_filter_points` is legitimate — because LoGeR itself computes `points = camera_poses @ homogenize(local_points)`, so any of the four being wrong breaks it.

**Files:**
- Modify: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the test**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
@pytest.mark.slow
def test_pinhole_residual_against_logers_native_pointcloud():
    """Unproject depth with the fitted K and compare to LoGeR's own world points.

    LoGeR's `xy` is a free per-pixel ray field, not constrained to any pinhole K, so
    the native cloud can encode lens distortion and per-frame ray variation that
    K-unprojection cannot reproduce. This measures how large that gap actually is.
    """
    if not _LOGER_ROOT.exists():
        pytest.skip("third_party/LoGeR not vendored")

    n, h, w = 8, 336, 462
    creator = LoGeRCreator()
    model = creator._load_model("cuda")

    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, size=(n, 480, 640, 3), dtype=np.uint8)
    views, image_paths, original_coords = creator._preprocess(frames, list(range(n)))
    creator.image_paths, creator.original_coords = image_paths, original_coords

    raw = creator._forward(model, views)

    # LoGeR's own world points, straight off the model
    with torch.no_grad():
        native = model(
            views.to("cuda")[None],
            window_size=creator.window_size, overlap_size=creator.overlap_size,
            reset_every=creator.reset_every, num_iterations=creator.num_iterations,
            sim3=False, sim3_scale_mode="median", se3=creator._se3,
            turn_off_ttt=False, turn_off_swa=False,
        )["points"].squeeze(0).cpu().float().numpy()

    # Ours, via the fitted K and the same reuse path production takes
    ours, _ = _raw_to_world_points(raw, subsample=1)
    ours = ours.reshape(n, h, w, 3)

    # Confident pixels only — the residual is meaningless where the model is unsure
    mask = raw["depth_conf"] > 0.5
    err = np.linalg.norm(ours[mask] - native[mask], axis=-1)
    scene_scale = float(np.percentile(np.linalg.norm(native[mask], axis=-1), 95))
    median_rel = float(np.median(err)) / scene_scale

    # LOG IT — this number is the deliverable, not the pass/fail
    print(f"\nPINHOLE RESIDUAL: median {median_rel * 100:.3f}% of scene scale "
          f"(abs {np.median(err):.4f}, scene scale {scene_scale:.3f}, "
          f"p95 {np.percentile(err, 95) / scene_scale * 100:.3f}%)")

    assert median_rel < 0.02, (
        f"Fitted-K unprojection diverges from LoGeR's native cloud by "
        f"{median_rel * 100:.2f}% of scene scale. The model is meaningfully "
        f"non-pinhole; the shared-K fit is then also lossy for the mesh and BA "
        f"paths, which outranks the world_points decision. See spec open item 3."
    )
```

Add `from collab_splats.pointcloud.feedforward.base import _raw_to_world_points` and `from collab_splats.pointcloud.feedforward.loger import _LOGER_ROOT` to the test file's imports.

- [ ] **Step 2: Run it**

Run:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k residual -v -s -p no:randomly
```
Expected: PASS, with the `PINHOLE RESIDUAL:` line printed.

**If it fails the 2% threshold:** do not raise the threshold to make it pass. That result is the finding — LoGeR is meaningfully non-pinhole, and the spec says it "outranks this decision entirely". Record the number, stop, and report; the `world_points` source and the mesh/BA implications need revisiting before this ships.

Random-noise frames are a weak test scene. If the residual looks implausible either way, re-run against `data/tutorial/`'s real video through `frames.zarr` before trusting it.

- [ ] **Step 3: Record the measured number in the spec**

Edit `docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md`, open item 3, replacing "is unmeasured" with the measured figure and the date.

- [ ] **Step 4: Commit**

```bash
git add tests/pointcloud/test_loger_creator.py docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md
git commit -m "test(loger): measure the fitted-K pinhole residual

One assertion covers the focal fit, the c2w->w2c inversion, the depth
extraction, and the legitimacy of reusing unproject_and_filter_points, because
LoGeR itself computes points = camera_poses @ homogenize(local_points) — any of
the four being wrong breaks it. Zero new production code carries the validation.

Logs the residual as a number rather than only passing or failing: it decides
whether the shared-K approximation is tight, and a large value would affect the
mesh and BA paths, not just world_points. Closes spec open item 3."
```

---

## Task 14: End-to-end run and the `max_frames` sweep

Spec open item 2. Everything before this is unit-level; this is the first time the backend runs through `Reconstructor`.

**Files:** none modified except docs at the end.

- [ ] **Step 1: Run the full suite for regressions**

Run:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly -m "not slow" 2>&1 | tail -20
```
Expected: no new failures against `docs/known-test-failures.md`. Check that file first — some failures are pre-existing and documented.

- [ ] **Step 2: Run the dashboard smoke gate**

Required before committing anything that could touch the dashboard's import path; `collab_splats.pointcloud` is on it.

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`

- [ ] **Step 3: End-to-end reconstruction on the tutorial video**

`run_pipeline.py` takes positional video paths plus `--output-root`, and overrides come from a YAML merged over `base.yaml` — there is **no** `--set` flag. Write the override file first.

Create `/tmp/claude-0/-workspace-collab-splats/d78ed8d1-0f5a-4555-8eb3-eecc7f2e302a/scratchpad/loger_e2e.yaml`:

```yaml
pointcloud:
  backend: loger
semantics:
  enabled: false
mesh:
  enabled: false
localization:
  enabled: false
```

Run in tmux, not a notebook — heavy inference.

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/d78ed8d1-0f5a-4555-8eb3-eecc7f2e302a/scratchpad
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py \
    data/tutorial/tutorial_example-video.mp4 \
    --output-root "$SCRATCH/loger_e2e" \
    --config "$SCRATCH/loger_e2e.yaml"
```

Expected: completes; `loger/sparse_pc.ply`, `loger/colmap/sparse/0/*.bin`, and `loger/feedforward.zarr` all present under the run directory. The run lands in `<output-root>/[<date>/]tutorial_example-video/`, so locate it with `find "$SCRATCH/loger_e2e" -name sparse_pc.ply`.

- [ ] **Step 4: Verify the COLMAP camera kept both focals**

This is the end-to-end proof that `camera_model = "PINHOLE"` survived export.

Run:
```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/d78ed8d1-0f5a-4555-8eb3-eecc7f2e302a/scratchpad
SPARSE=$(find "$SCRATCH/loger_e2e" -type d -path "*/loger/colmap/sparse/0" | head -1)
/opt/venv/reconstruction/bin/python -c "
import pycolmap, sys
r = pycolmap.Reconstruction(sys.argv[1])
cam = list(r.cameras.values())[0]
print(cam.model.name, cam.params, cam.width, cam.height)
assert cam.model.name == 'PINHOLE', 'SIMPLE_PINHOLE would have averaged fx and fy away'
assert len(cam.params) == 4
print('points3D:', r.num_points3D(), '| images:', r.num_images())
" "$SPARSE"
```
Expected: `PINHOLE` with four params, and non-zero point/image counts.

- [ ] **Step 5: Sweep the frame ceiling**

The binding constraint is probably `FeedforwardResult`'s dense per-frame fields (~8 MB/frame), not LoGeR's windowing. Measure rather than guess.

Run in tmux, one frame count at a time, with nothing else running (side shells risk OOM). One override YAML per frame count, since there is no `--set`:

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/d78ed8d1-0f5a-4555-8eb3-eecc7f2e302a/scratchpad
for N in 300 500 750 1000; do
  cat > "$SCRATCH/loger_sweep_$N.yaml" <<EOF
preprocessing:
  min_frames: $N
  max_frames: $N
pointcloud:
  backend: loger
semantics:
  enabled: false
mesh:
  enabled: false
localization:
  enabled: false
EOF
  echo "=== $N frames ==="
  /usr/bin/time -v /opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py \
      data/tutorial/tutorial_example-video.mp4 \
      --output-root "$SCRATCH/loger_sweep_$N" \
      --config "$SCRATCH/loger_sweep_$N.yaml" \
      2>&1 | grep -E "Maximum resident|Elapsed|Error|error"
done
```

`min_frames` and `max_frames` are both pinned to N. Setting `max_frames` alone measures nothing: it is a cap, and the actual extracted count is decided by the sampler, so the run would process whatever the sampler chose rather than N. Pinning the floor too forces the count and is robust to `preprocessing.frame_proportion` being removed — a concurrent change as of 2026-08-13. Read `configs/base.yaml`'s `preprocessing:` block before running this and drop any key that no longer exists.

Confirm the tutorial video has enough source frames to reach 1000 before trusting the top row:
```bash
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
    -of csv=p=0 data/tutorial/tutorial_example-video.mp4
```
If it holds fewer than ~1000 frames, the sweep cannot reach the ceiling with this input — say so in Step 6 rather than reporting an unreached limit as a measured one.

Repeat the highest surviving count with `pointcloud.loger.reset_every: 64` added to that YAML, to see whether resetting the TTT fast weights moves the ceiling.

Record for each: frames, peak RSS, peak VRAM, wall clock, and whether it completed.

- [ ] **Step 6: Write the measured ceiling down**

Update three places with the highest frame count that completed:
- `configs/base.yaml` — the `max_frames` comment
- `configs/README.md` — the `loger` section's `max_frames` paragraph
- the spec's open item 2

State plainly whether the limit was RSS, VRAM, or wall clock, and whether `reset_every` helped. If nothing OOMed even at 1000 frames, say so and give the peak RSS at 1000 — a ceiling that was not reached is a different fact from one that was measured.

- [ ] **Step 7: Commit**

```bash
git add configs/base.yaml configs/README.md docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md
git commit -m "docs(loger): record the measured frame ceiling

Replaces the guessed max_frames guidance with a swept number, including whether
the binding limit was host RSS or VRAM and whether reset_every moved it. Closes
spec open item 2."
```

- [ ] **Step 8: Update CLAUDE.md**

Add to the "In-Flight Work" section's recently-completed list, following the existing entries' format: what shipped, the measured pinhole residual, the measured frame ceiling, the LC and multiview-confidence follow-ups that remain owed, and the unresolved LICENSE question.

```bash
git add CLAUDE.md
git commit -m "docs: record loger backend completion in CLAUDE.md"
```

---

## Owed to the user before this ships

1. **The missing LICENSE.** Neither LoGeR repository ships one — confirmed against the vendored tree in Task 1, not assumed. Spec open item 7. **This gates Task 2, not Task 3** as originally written: `_weighted_median` is itself a port from PolyCam's `run_loger.py:167`, so copied code enters the repo one task earlier than the plan first said. Task 4 is affected too, porting ~10 lines out of the vendored Junyi42 tree. The fallback is reimplementing all three from first principles — a cumulative-weight-to-half median, the pinhole identity `fx = u_c * Z / X`, and an area-budget resize are each textbook — citing the originals as prior art rather than as source.
2. **Loop closure calibration** for the LoGeR backbone. Refused until then; needs its own clean-negative sweep like the other four backbones. Spec open item 4.
3. **Multiview confidence** — owned by `2026-08-12-multiview-confidence-all-models-design.md`; `loger` should be added to its scope. Spec open item 5.
4. **Square-pixel averaging in original-resolution space** — dropped from this cut with a reason, revisit only if Task 13's residual shows the estimator's spread exceeds the ~1% the model-resolution version would cost. Spec open item 8.

## Self-review notes

**Spec coverage.** Every spec section maps to a task: source selection and the torch gate → Task 1; the fit and its three preserved behaviours → Tasks 2–3; the resize rule and anisotropy → Tasks 4, 6, 11; `_load_model` and the `se3` routing → Task 5; `_forward`'s sigmoid ordering and c2w→w2c → Task 7; `_postprocess` / `_reproject` / `world_points` → Task 8; error-handling table → Tasks 3, 5, 6, 7, 8, 10; registry and exports → Task 9; Reconstructor wiring, LC refusal, `max_frames` warning, kwargs passthrough → Task 10; configuration → Task 12; the eight unit tests and the parity integration test → Tasks 2–9, 11, 13; open items 1, 2, 3 → Tasks 1, 14, 13.

**Two spec claims were corrected during planning**, both verified against the vendored tree: the window knobs do not come from a `training_settings` block (there is none — both configs hold only `model:`), and `overlap_size` defaults to 3, not 8. `_run_feedforward` also takes explicit scalars and never sees `pc_cfg`, so the kwargs passthrough needed a parameter plus a call-site change rather than the single line the spec showed. All three are fixed in the spec at `1f06798`.

**Three plan-authoring errors were caught and fixed** by checking against the code rather than assuming: `FrameStore` has no `.count` — the frame count is `__len__` (`frame_store.py:65`), and it is already imported at `reconstructor.py:23`; `run_pipeline.py` has no `--set` flag, taking positional video paths plus `--output-root` and a `--config` override YAML, so Task 14 writes override files instead; and the sweep must pin `preprocessing.min_frames` alongside `max_frames`, since `max_frames` alone is only a cap and the sampler would decide the real count. `min_frames`/`max_frames` was chosen over raising `frame_proportion` because that key is being removed in concurrent work as of 2026-08-13; the floor-and-cap form survives either way.

**Two things this plan cannot pin down in advance**, each with an explicit check step rather than an assumption: LoGeR's exact output key names and `conf` rank (Task 1 Step 5 records them; Task 7 depends on them), and whether `1920x1080` happens to resize isotropically at the default pixel budget (Task 11 Step 2 gives a fallback input).
