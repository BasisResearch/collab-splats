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
| new | `collab_splats/pointcloud/feedforward/loger.py` | `LoGeRCreator` + `_compute_target_size` |
| new | `tests/pointcloud/test_loger_creator.py` | Resize, forward, postprocess, refusals |
| new | `tests/pointcloud/feedforward/test_loger_load_guard.py` | Absent-tree guarded-import behaviour |
| mod | `collab_splats/geometry/transforms.py` | `estimate_intrinsics_from_points` (public) + `_compute_weighted_median` (private). Model-agnostic, sits beside `extract_intrinsics`; LoGeR is only the first caller. |
| mod | `collab_splats/geometry/__init__.py` | Re-export `estimate_intrinsics_from_points` |
| mod | `tests/geometry/test_transforms.py` | Median + intrinsics-fit tests |
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

**Every citation names the repository, the pinned commit, the file, and the line range** — whether it marks code adapted from upstream, an algorithm we reimplemented with upstream as prior art, or a behavioural claim about upstream that our code depends on but does not copy (e.g. "this head emits logits, so we apply the sigmoid ourselves"). A bare filename is not enough — `run_loger.py:167` is ambiguous between the two forks, and `pi3.py:172` is a line number in a tree that is gitignored and therefore unreadable from the repo alone.

Two upstreams are involved and they are not interchangeable:

| Short form used below | Means |
|---|---|
| **PolyCam @ 5d7c1a7** | `github.com/PolyCam/LoGeR` @ `5d7c1a7` — read as prior art for the intrinsics estimator. Not vendored, not a dependency, nothing copied from it. |
| **Junyi42 @ 7685b7a** | `github.com/Junyi42/LoGeR` @ `7685b7a` — the tree we *vendor* into `third_party/LoGeR/` (Task 1). |

Write the long form in the code, not the short form. Every module-level helper docstring and every non-obvious inline comment that reflects upstream behaviour states which of the two it came from. When a comment cites vendored-tree behaviour we depend on but do not copy (the conf head emitting logits, `se3` being popped inside `forward`), that is still a citation and still names repo, commit, file, and line — a reader cannot check it otherwise, because `third_party/` is gitignored.

### Two deliberate deviations from the spec

**1. Placement.** The spec put both new functions in `loger.py`. Two of the three now live in `collab_splats/geometry/transforms.py` instead — `estimate_intrinsics_from_points` (public) and its private helper `_compute_weighted_median`. The maths is model-agnostic: it takes a camera-frame pointmap and per-pixel confidence, which any pointmap backend without an intrinsics head produces. `transforms.py` already describes itself as "Pure-numpy camera geometry utilities shared across the pipeline" and already holds `extract_intrinsics`, and the repo's precedent for a function that *produces* a K is `seed_intrinsics` (`localization/localizer.py:24`) — verb + noun, no backend tag. Leaving them in `loger.py` would put a third intrinsics helper in a fourth location. Cost of the move, stated plainly: one caller today, and the `conf > 0.1` gate becomes a documented keyword default instead of a constant.

**2. A third function.** The spec's "Genuinely new" section names **two**. This plan adds `_compute_target_size`, which does stay in `loger.py`. Justification, since the spec requires one for every addition: it is a ~10-line algorithm with a `while` loop that must stay behaviourally identical to the vendored loader (github.com/Junyi42/LoGeR @ 7685b7a, `loger/utils/basic.py:55-61`) or the model receives out-of-distribution input, it is the sole source of the resize anisotropy that three other decisions depend on, and it is the only part of `_preprocess` that can be tested without a model — including against the vendored loader directly. Inlining it would make the anisotropy untestable in isolation. This is a different case from the rejected `_loger_original_coords`, which would have wrapped `np.tile` of a constant row.

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
- **No `LICENSE` or `COPYING` at the vendored tree's top level** — verified, not assumed. Resolved by the decision recorded at the head of Task 2: reimplement, cite as prior art, copy nothing.

---

## Task 2: `_compute_weighted_median`

**Licence decision (2026-08-13, user):** neither LoGeR fork ships a `LICENSE`, so nothing is copied into tracked source. Tasks 2, 3, and 4 are **written from the underlying maths** — a cumulative-weight-to-half median, the pinhole identity `fx = u_c * Z / X`, and an area-budget resize are each textbook. Both repos are cited as **prior art and as the behavioural reference we match**, never as the source of the lines. Task 4 additionally pins that behavioural match with a measured parity test against the vendored loader, so "we match upstream" is a test result rather than an assertion.

**Placement decision (2026-08-13, user).** The median and the intrinsics fit do **not** live in `loger.py`. They go in `collab_splats/geometry/transforms.py`, whose module docstring already reads "Pure-numpy camera geometry utilities shared across the pipeline" and which already holds `extract_intrinsics`. Reasons: the maths is model-agnostic (any pointmap backend with no intrinsics head needs it), the repo's precedent for "function that produces a K" is `seed_intrinsics` (`localization/localizer.py:24`) — tag-free, verb + noun — and putting a third intrinsics helper in a fourth location would worsen an existing scatter. `transforms.py` imports numpy only, so nothing heavy follows it in.

Consequence for task ordering: **Task 2 no longer creates `loger.py`.** Task 4 does.

**Files:**
- Modify: `collab_splats/geometry/transforms.py`
- Modify: `tests/geometry/test_transforms.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_transforms.py`. Add `_compute_weighted_median` to the existing `from collab_splats.geometry.transforms import (...)` block at the top of that file, then append these four tests:

```python
def test_compute_weighted_median_equal_weights_matches_plain_median():
    # With uniform weights the weighted median is the ordinary median.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.ones_like(values)
    assert _compute_weighted_median(values, weights) == pytest.approx(3.0)


def test_compute_weighted_median_follows_the_weight_mass():
    # Weight concentrated on the low values pulls the median down, even though
    # the high values are the numerical majority by count.
    values = np.array([1.0, 1.0, 9.0, 9.0, 9.0], dtype=np.float32)
    weights = np.array([50.0, 50.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert _compute_weighted_median(values, weights) == pytest.approx(1.0)


def test_compute_weighted_median_empty_returns_none():
    # Signals "no estimate" to the caller, which raises rather than falling back.
    assert _compute_weighted_median(np.array([]), np.array([])) is None


def test_compute_weighted_median_subsamples_deterministically():
    # Above max_n the seeded RNG must give the same answer every call — the
    # estimator is otherwise non-reproducible at production frame counts.
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 10.0, size=200_000).astype(np.float32)
    weights = np.ones_like(values)
    first = _compute_weighted_median(values, weights, max_n=1000)
    assert first == _compute_weighted_median(values, weights, max_n=1000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py -v -p no:randomly`
Expected: collection error, `ImportError: cannot import name '_compute_weighted_median' from 'collab_splats.geometry.transforms'`

- [ ] **Step 3: Write the helper**

Append to `collab_splats/geometry/transforms.py`, immediately after `extract_intrinsics`, under a new section divider matching the file's existing `########`-style:

```python
########################################################################
########## Intrinsics estimation #######################################
########################################################################


def _compute_weighted_median(values: np.ndarray, weights: np.ndarray, max_n: int = 50_000) -> float | None:
    """Confidence-weighted median, subsampled above ``max_n`` with a seeded RNG.

    Textbook definition — sort by value, walk the cumulative weight, return the value
    at half the total mass.  Prior art for using one here: github.com/PolyCam/LoGeR @
    5d7c1a7, ``run_loger.py:167`` reduces its per-pixel focal estimates the same way.
    Returns ``None`` for an empty input so the caller can raise rather than invent a
    value.
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

`transforms.py` imports numpy only and must stay that way — use `np.isfinite`/`np.argsort` rather than adding a `math` import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py -v -p no:randomly`
Expected: all pre-existing tests still pass, plus the 4 new ones.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/transforms.py tests/geometry/test_transforms.py
git commit -m "feat(geometry): add confidence-weighted median helper

Written from the textbook definition, not copied: neither LoGeR fork ships a
LICENSE. github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:167 is cited as prior
art for reducing per-pixel focal estimates this way.

The 50k cap is our own memory guard, sized for the frame counts this backend
targets: the pooled per-pixel sample population is H*W*N, which is 255M values
at 1000 frames, and a weighted median needs a full argsort. The fixed seed keeps
the estimate reproducible."
```

---

## Task 3: `estimate_intrinsics_from_points`

The core new algorithm. LoGeR emits no K; this solves one by inverting the pinhole model per pixel and taking a confidence-weighted median. It returns the whole `(3,3)` matrix rather than `(fx, fy)` because the centred pixel grid already fixes `cx`/`cy`.

**Public, and named for what it does, not for its caller.** It sits beside `extract_intrinsics` in `geometry/transforms.py` and is exported from `collab_splats.geometry`. Nothing in it is LoGeR-specific — it takes a camera-frame pointmap and per-pixel confidence, both of which any pointmap backend without an intrinsics head produces. LoGeR is simply the first caller. The `conf > 0.1` gate is therefore a keyword argument with a documented default rather than a buried constant.

**Files:**
- Modify: `collab_splats/geometry/transforms.py`
- Modify: `collab_splats/geometry/__init__.py`
- Modify: `tests/geometry/test_transforms.py`

- [ ] **Step 1: Write the failing tests**

Add `estimate_intrinsics_from_points` to the existing import block at the top of `tests/geometry/test_transforms.py`, then append:

```python
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

    k = estimate_intrinsics_from_points(pts, conf)

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

    k = estimate_intrinsics_from_points(pts, conf)

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
        estimate_intrinsics_from_points(pts, conf)


def test_fit_accepts_trailing_axis_confidence():
    # LoGeR's conf head emits (N,H,W,1); the estimator must not require a squeeze
    # from its caller, since _forward and the tests reach it by different routes.
    h, w = 56, 70
    pts = _synthetic_local_points(h, w, 80.0, 80.0)
    k4 = estimate_intrinsics_from_points(pts, np.ones((1, h, w, 1), np.float32))
    k3 = estimate_intrinsics_from_points(pts, np.ones((1, h, w), np.float32))
    np.testing.assert_allclose(k4, k3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py -v -p no:randomly`
Expected: `ImportError: cannot import name 'estimate_intrinsics_from_points'`

- [ ] **Step 3: Implement the estimator**

Append to the "Intrinsics estimation" section of `collab_splats/geometry/transforms.py`, after `_compute_weighted_median`:

```python
def estimate_intrinsics_from_points(
    local_points: np.ndarray, conf: np.ndarray, conf_threshold: float = 0.1
) -> np.ndarray:
    """Fit one shared pinhole K to a camera-frame pointmap by confidence-weighted median.

    This is a **fit**, not a readout: the pointmap is not guaranteed to be consistent
    with any single pinhole camera, so the returned K is the best shared pinhole
    explanation of it rather than a recovered ground truth.  Callers that need to know
    how good that explanation is should measure the reprojection residual.

    For backends whose model emits no intrinsics head.  Invert the pinhole model at
    every pixel — ``u_c = fx * X / Z``, so ``fx = u_c * Z / X`` — and reduce the pooled
    per-pixel estimates with a confidence-weighted median.  One K is returned for the
    whole batch, which is correct when every frame comes from the same physical camera
    at the same resolution.

    Prior art for the same approach: github.com/PolyCam/LoGeR @ 5d7c1a7,
    ``run_loger.py``, ``estimate_focal_lengths`` at :206 over ``_focal_from_frame`` at
    :180.  That fork also has ``_snap_square_pixels`` at :195, which we deliberately do
    **not** do — it would merge fx and fy, and a caller that rescales the two axes
    separately then un-merges the average incorrectly.

    Args:
        local_points: (N, H, W, 3) camera-frame points, channel 2 being depth.  Note
            that a pointmap head is free to emit a per-pixel ray field not constrained
            to any pinhole K — LoGeR's, for instance, is built as ``cat([xy * z, z])``
            (github.com/Junyi42/LoGeR @ 7685b7a, ``loger/models/pi3.py:772-775``) —
            which is the reason this is an approximation.
        conf: (N, H, W) or (N, H, W, 1) per-pixel confidence, **already activated into
            [0, 1]**.  ``conf_threshold`` is a probability floor; passing raw logits
            would admit roughly half of all pixels instead.
        conf_threshold: minimum confidence for a pixel to contribute.

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
    valid = (z > 1e-3) & (np.abs(x) > 1e-6) & (np.abs(y) > 1e-6) & (conf > conf_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_per_pixel = uu * z / x
        fy_per_pixel = vv * z / y

    fx_vals, fy_vals = fx_per_pixel[valid], fy_per_pixel[valid]
    weights = conf[valid]

    # Sanity bounds before the median, derived from field of view: f = 0.1 * W is a
    # ~157 degree horizontal FOV and f = 10 * W is ~6 degrees.  Real cameras live well
    # inside that; values outside it are degenerate inversions from pixels near the
    # principal axis, where X or Y is small enough that u_c * Z / X explodes.
    #
    # These bounds are also what enforce _compute_weighted_median's finite precondition,
    # and that is not incidental: the `valid` mask above cannot do it, because inf passes
    # `z > 1e-3`.  A surviving non-finite value would not poison the median visibly, it
    # would skew it — argsort sorts +inf and NaN to the tail (biasing the focal upward)
    # and -inf to the head (biasing it downward), and `u_c * Z / X` produces -inf as
    # readily as +inf as X approaches zero from below.  Either way the result stays
    # finite enough for the np.isfinite check below to wave it through.  BOTH bounds are
    # load-bearing: the upper rejects +inf, the lower rejects -inf, and NaN fails both.
    # Do not loosen either to a one-sided test without adding an explicit isfinite mask.
    ok_fx = (fx_vals > w * 0.1) & (fx_vals < w * 10)
    ok_fy = (fy_vals > h * 0.1) & (fy_vals < h * 10)
    fx = _compute_weighted_median(fx_vals[ok_fx], weights[ok_fx])
    fy = _compute_weighted_median(fy_vals[ok_fy], weights[ok_fy])

    # Fail loudly.  Upstream falls back to 1.2 * max(W, H); we do not, because a
    # plausible-but-wrong K fails silently all the way through to the mesh.
    if fx is None or fy is None or not np.isfinite(fx) or not np.isfinite(fy) or fx <= 0 or fy <= 0:
        raise RuntimeError(
            f"Pinhole intrinsics fit failed over {n} frames: "
            f"{int(valid.sum())}/{valid.size} pixels passed the validity mask, "
            f"{int(ok_fx.sum())} survived the fx bounds and {int(ok_fy.sum())} the fy bounds "
            f"(fx={fx}, fy={fy}). No fallback focal is applied by design."
        )

    # fx and fy stay distinct.  Callers whose preprocessing rounds the two axes
    # independently (LoGeR's does, to multiples of 14) produce genuinely non-square
    # pixels, and that anisotropy belongs in K rather than being averaged away.
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0],
         [0.0, fy, (h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Export it from the package**

`estimate_intrinsics_from_points` is public, so it joins the explicit re-export list in `collab_splats/geometry/__init__.py` — both the `from .transforms import (...)` block and `__all__`, each of which is alphabetically sorted. `_compute_weighted_median` is private and is **not** exported.

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v -p no:randomly`
Expected: all pre-existing geometry tests still pass, plus the 4 from Task 2 and the 7 here.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/transforms.py collab_splats/geometry/__init__.py tests/geometry/test_transforms.py
git commit -m "feat(geometry): fit a shared pinhole K from a camera-frame pointmap

Some backends emit a pointmap but no intrinsics head, so K has to be fitted
rather than read out: invert the pinhole model per pixel and take a
confidence-weighted median, shared across frames. Written from that identity,
not copied — neither LoGeR fork ships a LICENSE. github.com/PolyCam/LoGeR @
5d7c1a7, run_loger.py:180-206 is prior art for the same approach; the validity
bounds here are derived from field of view (f = 0.1*W is ~157 degrees, f = 10*W
is ~6).

Lives in geometry/transforms.py beside extract_intrinsics rather than in the
first caller's module: the maths is model-agnostic, and the repo's precedent for
a function that produces a K is seed_intrinsics (localization/localizer.py:24) —
verb + noun, no backend tag. LoGeR is only the first caller.

Returns the full 3x3 rather than (fx, fy) because the centred pixel grid already
fixes cx=(W-1)/2 — splitting that across caller and callee would let a caller
pick W/2 and introduce a silent half-pixel offset.

fx and fy stay independent. Upstream's square-pixel snap is deliberately not
applied: a caller that rescales the two axes separately then un-averages it
wrongly.

No fallback focal on degenerate input; it raises."
```

---

## Task 4: `_compute_target_size`

This task **creates** `loger.py` — Tasks 2 and 3 put their helpers in `geometry/transforms.py`, so `_compute_target_size` is the first thing that actually belongs in the backend module.

It stays a module-level function rather than becoming a method or property on `LoGeRCreator`. A property does not fit (it takes `orig_w, orig_h`); a method would work and would drop the `pixel_limit` parameter by reading `self.pixel_limit`, but both direct siblings in this package — `_compute_omega_original_coords` (`vggt_omega.py:52`) and `_compute_vggtx_crop_coords` (`vggtx.py:49`) — are module-level functions doing exactly this job for their backend, and the explicit `pixel_limit` argument is what lets the parity test pin 255,000 independently of any instance config. `_preprocess` passes `self.pixel_limit`. The `loger` tag is dropped from the name because the module path already carries it.

**Files:**
- Create: `collab_splats/pointcloud/feedforward/loger.py`
- Create: `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pointcloud/test_loger_creator.py`:

```python
"""LoGeR feedforward backend: resize rule and creator contract."""
import sys

import numpy as np
import pytest
from PIL import Image

from collab_splats.pointcloud.feedforward.loger import _LOGER_ROOT, _compute_target_size


@pytest.mark.parametrize(
    "orig_w,orig_h",
    [(1920, 1080), (1080, 1920), (640, 480), (1000, 1000), (3840, 2160)],
)
def test_target_size_is_patch_aligned_and_within_budget(orig_w, orig_h):
    # Both invariants are load-bearing: a non-multiple of 14 crashes the ViT
    # patch embedding, and exceeding the budget is what OOMs long sequences.
    w, h = _compute_target_size(orig_w, orig_h, pixel_limit=255_000)
    assert w % 14 == 0 and h % 14 == 0
    assert w >= 14 and h >= 14
    assert w * h <= 255_000


def test_target_size_preserves_orientation():
    # Landscape stays landscape. Independent per-axis rounding perturbs the exact
    # ratio by a few percent, but must never transpose it.
    w, h = _compute_target_size(1920, 1080, pixel_limit=255_000)
    assert w > h


def test_target_size_aspect_error_is_small_but_real():
    # The residual anisotropy is why fx and fy are fitted separately and why
    # camera_model must be PINHOLE. Assert it exists and is bounded — if a future
    # change makes it exactly zero, the separate-focal machinery is still correct
    # but this test documents why it is there.
    orig_w, orig_h = 1920, 1080
    w, h = _compute_target_size(orig_w, orig_h, pixel_limit=255_000)
    ratio_error = abs((w / h) / (orig_w / orig_h) - 1.0)
    assert ratio_error < 0.05


def test_target_size_upscales_small_images_to_the_budget():
    # The rule is an area budget, not a cap: a tiny input is scaled up to fill it.
    w, h = _compute_target_size(64, 48, pixel_limit=255_000)
    assert w * h > 200_000


@pytest.mark.skipif(not _LOGER_ROOT.exists(), reason="vendored tree absent (setup/loger.sh)")
@pytest.mark.parametrize("orig_w,orig_h", [(1920, 1080), (1080, 1920), (640, 480), (1000, 1000)])
def test_target_size_matches_the_vendored_loader(tmp_path, orig_w, orig_h):
    # This helper is reimplemented rather than copied, so parity with upstream is a
    # thing we MEASURE, not a thing we claim. The model trains on images preprocessed
    # by the vendored loader; if our size arithmetic drifts from it we feed the model
    # out-of-distribution input, and nothing downstream would report that.
    sys.path.insert(0, str(_LOGER_ROOT))
    try:
        from loger.utils.basic import load_images_as_tensor
    finally:
        sys.path.remove(str(_LOGER_ROOT))

    # The loader takes a directory, so give it two frames at the size under test.
    for i in range(2):
        Image.fromarray(
            np.random.default_rng(i).integers(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        ).save(tmp_path / f"{i:04d}.jpg")

    # PIXEL_LIMIT is upstream's casing, not a typo on our side — the vendored signature
    # is `load_images_as_tensor(path, interval, PIXEL_LIMIT, Target_W, Target_H)` at
    # github.com/Junyi42/LoGeR @ 7685b7a, loger/utils/basic.py:11. Passing it explicitly
    # rather than leaning on its default is what makes this a parity test of the SAME
    # budget our own call uses; a silent default drift upstream would otherwise pass.
    upstream = load_images_as_tensor(str(tmp_path), PIXEL_LIMIT=255_000)
    _, _, up_h, up_w = upstream.shape

    assert _compute_target_size(orig_w, orig_h, pixel_limit=255_000) == (up_w, up_h)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.pointcloud.feedforward.loger'`

- [ ] **Step 3: Implement**

Create `collab_splats/pointcloud/feedforward/loger.py`:

```python
"""LoGeR feedforward backend: resize rule and creator.

LoGeR is a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window
inference and overlap stitching.  Unlike every other backend we run it predicts no
camera intrinsics, so K is fitted from its camera-frame pointmap by
``collab_splats.geometry.transforms.estimate_intrinsics_from_points``.

Upstream sources.  Two forks are involved and they are NOT interchangeable.  Neither
ships a LICENSE, so no code here is copied from either — the maths is written from
first principles and the forks are cited as prior art and as the behavioural reference
we match.  Citations carry repo, commit, file, and line because third_party/ is
gitignored and cannot be read from this repo alone:

  * VENDORED — the tree we actually execute against
    (setup/loger.sh clones it into third_party/LoGeR/):
      github.com/Junyi42/LoGeR @ 7685b7a
  * PRIOR ART — read for reference, not vendored, not a dependency, nothing copied:
      github.com/PolyCam/LoGeR @ 5d7c1a7

Provides:
  LOGER_HF_REPO        — HuggingFace repo holding both checkpoints
  LOGER_VARIANTS       — the two shipped variants
  LOGER_CONF_THRESHOLD — confidence floor for the K fit, measured not inherited
  _compute_target_size — patch-aligned resize matching the vendored loader
  LoGeRCreator         — feedforward creator using LoGeR depth + pose
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

########################################################################
########## Constants ###################################################
########################################################################

LOGER_HF_REPO = "Junyi42/LoGeR"
LOGER_VARIANTS = ("LoGeR", "LoGeR_star")

# Confidence floor for the K fit, passed explicitly to estimate_intrinsics_from_points
# instead of taking its 0.1 default. LoGeR's conf head is uncalibrated: measured logits
# on the Task 1 run span -4.257..-2.019, so the post-sigmoid band is [0.0140, 0.1172] and
# 0.1 is its 92nd percentile — the default keeps 7.9% of pixels (12,217/155,232) and a
# marginally duller scene keeps none, raising. 0.02 sits just above the band floor, so it
# rejects only what the model calls junk; the confidence WEIGHTING inside the median is
# what actually discriminates. Re-measure this if the checkpoint changes.
LOGER_CONF_THRESHOLD = 0.02

# Vendored tree, populated by setup/loger.sh.  parents[3] resolves
# collab_splats/pointcloud/feedforward/loger.py -> repo root.
_LOGER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "LoGeR"

# LoGeR's ViT patch size; every model-resolution image dimension is a multiple of it.
_PATCH = 14

########################################################################
########## Preprocessing ###############################################
########################################################################


def _compute_target_size(orig_w: int, orig_h: int, pixel_limit: int) -> tuple[int, int]:
    """Scale to an area budget, then align both axes to a whole number of patches.

    Unlike the other two helpers this one is not free to differ from upstream: the
    model is trained on images preprocessed this way, so a different rule would feed
    it out-of-distribution input.  The rule is therefore written to a *behavioural*
    spec — area budget, both axes multiples of 14, shrink whichever axis sits furthest
    above the target aspect until the budget is met — and that behaviour is pinned by
    a measured parity test against the vendored loader
    (github.com/Junyi42/LoGeR @ 7685b7a, ``loger/utils/basic.py:55-61``, inside
    ``load_images_as_tensor``, whose signature is at ``basic.py:11``), not asserted.
    Lines 62-63 there are a Target_W/Target_H override we deliberately do not
    reimplement — we always compute the size, never accept one.  See
    ``test_target_size_matches_the_vendored_loader``.

    The two axes round **independently**, so exact aspect ratio is not preserved —
    the image is stretched by up to a few percent on one axis.  That is in
    distribution (the model trains with this preprocessing) and is absorbed into K,
    because ``estimate_intrinsics_from_points`` fits fx and fy separately and
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
Expected: 12 passed. If the 4 parity cases report `skipped`, the vendored tree is missing — re-run `setup/loger.sh` rather than accepting the skip, because those 4 are the only thing standing between us and silently feeding the model out-of-distribution input.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add patch-aligned resize matching the vendored loader

Written to a behavioural spec rather than copied (no LICENSE upstream), and the
match is measured: test_target_size_matches_the_vendored_loader runs
github.com/Junyi42/LoGeR @ 7685b7a loger/utils/basic.py load_images_as_tensor on
real files at four aspect ratios and compares the resulting H,W against ours.
Parity matters more here than in the other helpers because the model trains on
this preprocessing, so drift means out-of-distribution input rather than a
slightly different number.

Kept as a named function rather than inlined because it is the sole source of
the resize anisotropy that the separate fx/fy fit and the PINHOLE camera model
both exist to absorb, and it is the only part of _preprocess testable without a
model."
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

from collab_splats.geometry.transforms import (
    estimate_intrinsics_from_points,
    extrinsics_to_homogeneous,
    invert_poses,
)

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
    camera-frame pointmap by ``estimate_intrinsics_from_points`` and shared across frames.

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
Expected: **15 passed, 7 xfailed** (as shipped — the review rounds added a sixth test pinning the
`_se3` capture and a seventh rejecting an empty `model:` block). Task 6 Step 0 converts all seven
to real passes.

Two corrections to what this plan originally said here. The old figure of "24 passed" was simply
wrong arithmetic. And the five tests added by this task **cannot pass yet**: `LoGeRCreator`
subclasses `BasePointcloudCreator`, an `abc.ABC` (`collab_splats/pointcloud/base.py:101`), and
four abstract methods remain unimplemented until Task 8, so `LoGeRCreator()` raises
`TypeError: Can't instantiate abstract class`. Do NOT stub those methods to get past it — that
would pre-empt Tasks 6-8. Mark the five with the module-level

```python
_NEEDS_FULL_CREATOR = pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="LoGeRCreator's abstract methods land in Tasks 6-8; remove this marker there",
)
```

and let Task 8's Step 0 delete it. `strict=True` is load-bearing: it converts these to failures
the moment they start passing, which is what forces the marker out.

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

> **Step 0 of this task replaces the xfail scheme with a partial subclass.** Task 5 shipped
> its tests behind `_NEEDS_FULL_CREATOR` because `LoGeRCreator` subclasses an `abc.ABC`
> (`collab_splats/pointcloud/base.py:101`) and cannot be instantiated until the last abstract
> method lands. Extending that marker through Tasks 6 and 7 would leave **16 tests carrying no
> signal at all** — `_preprocess` and `_forward` would sit unverified under three commits of
> later work, and a bug in either would surface at Task 8 with the hardest possible debugging
> context. Stub only the methods not yet written instead, so each task's tests exercise the
> real implementation the moment it lands.
>
> **Step 0:** add the `_creator` helper below, replace every `LoGeRCreator(` call in the
> existing tests with `_creator(`, and delete `_NEEDS_FULL_CREATOR` and all seven
> `@_NEEDS_FULL_CREATOR` decorators.
>
> All seven convert, including `test_unknown_variant_rejected_at_construction`: the helper
> forwards `**kwargs` straight to `__init__`, so `_creator(variant="LoGeR_turbo")` still runs
> the real `__post_init__` and still raises the real `ValueError`. The subclass overrides
> nothing that any of these tests assert on.
>
> ```python
> def _creator(**kwargs) -> LoGeRCreator:
>     """LoGeRCreator with only the not-yet-implemented abstract methods stubbed out."""
>     # BasePointcloudCreator is an abc.ABC (collab_splats/pointcloud/base.py:101). Stubbing
>     # ONLY the unwritten methods keeps every test below pointed at real code as it lands,
>     # rather than deferring all signal to the task that happens to close the ABC.
>     # Each task deletes the stub it just implemented. Task 8 deletes this helper entirely.
>     class _PartialLoGeRCreator(LoGeRCreator):
>         def _preprocess(self, frames, frame_idxs):
>             raise NotImplementedError
>
>         def _forward(self, model, views, **kwargs):
>             raise NotImplementedError
>
>         def _postprocess(self, raw_outputs, **kwargs):
>             raise NotImplementedError
>
>         def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
>             raise NotImplementedError
>
>         def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
>             raise NotImplementedError
>
>     return _PartialLoGeRCreator(**kwargs)
> ```
>
> **These signatures were verified against `collab_splats/pointcloud/feedforward/base.py` during
> Task 6** — `_preprocess:919`, `_forward:922`, `_postprocess:925`, `extract_intermediate_features:928`,
> `_reproject:1023`. An earlier draft of this plan had three of the five wrong (`_postprocess`,
> `_reproject`, and `extract_intermediate_features` all took different argument names, and two
> take `**kwargs`). A stub whose signature disagrees with the abstract method **still satisfies
> the ABC** — Python does not check signatures — so a mismatch is invisible until the real method
> lands. Re-read the file rather than trusting this block.
>
> `LoGeRCreator` has **five** abstract methods outstanding after Task 5. Note the reason: an
> earlier draft claimed `_load_model` is concrete on `BaseFeedforwardCreator`, which is **false** —
> it is `@abstractmethod` at `base.py:915-916`. There are six abstract methods in total; Task 5
> implemented `_load_model` on the subclass, leaving five. The stub list was right for the wrong
> reason, so do not reason from the old explanation.
>
> **Each later task deletes the stub for the method it implements**, in the same step that
> implements it, so the tests written that task hit real code rather than the stub. Task 6
> deletes `_preprocess`, Task 7 deletes `_forward`, Task 8 deletes the remaining three along
> with the whole helper.
>
> Run `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -v -p no:randomly`
> after Step 0 alone. Expected: **22 passed** — the 7 previously-xfailed tests now run for real.
> If any of them FAILS rather than passes, stop and report: that means Task 5's code is wrong
> and the xfail was hiding it, which is exactly what this step exists to find out.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
def _fake_frames(n: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def test_preprocess_returns_patch_aligned_unit_range_tensor():
    frames = _fake_frames(4, 480, 640)
    views, image_paths, original_coords = _creator()._preprocess(frames, [0, 5, 10, 15])

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
    _, _, original_coords = _creator()._preprocess(frames, [0, 1, 2])

    assert original_coords.shape == (3, 6)
    np.testing.assert_allclose(original_coords, np.tile([0, 0, 640, 480, 640, 480], (3, 1)))


def test_preprocess_rejects_non_uniform_frame_sizes():
    # LoGeR sizes from frame 0 alone; refuse rather than silently mis-resize the rest.
    frames = [_fake_frames(1, 480, 640)[0], _fake_frames(1, 240, 320)[0]]
    with pytest.raises(ValueError, match="uniform"):
        _creator()._preprocess(frames, [0, 1])


def test_preprocess_rejects_out_of_order_frames():
    # Windows and overlap stitching assume temporal order; out-of-order input
    # degrades quality with no error. No other backend cares, so this is LoGeR's.
    frames = _fake_frames(3, 480, 640)
    with pytest.raises(ValueError, match="ascending"):
        _creator()._preprocess(frames, [0, 10, 5])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k preprocess -v -p no:randomly`
Expected: 4 failed with `NotImplementedError` raised by the `_preprocess` stub in `_creator`.

- [ ] **Step 3: Implement**

**Also delete the `_preprocess` stub from `_creator`'s `_PartialLoGeRCreator`** — otherwise the
stub keeps shadowing the real method and all four tests above go on hitting `NotImplementedError`.

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
        target_w, target_h = _compute_target_size(orig_w, orig_h, self.pixel_limit)
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
Expected: **26 passed** (22 after Step 0, plus the 4 written here). No xfails remain in this file.

Task 6's quality review then raised the out-of-order test to a parametrized pair (out-of-order and
duplicate, since `b <= a` weakened to `b < a` survived otherwise), landing the file at **27** in
commit `225d629`. Later tasks count up from 27, not 26.

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

> **`_se3` is `None` until `_load_model` runs, and `_forward` must refuse it.** Task 5's review
> changed the field to `bool | None` defaulting to `None`, precisely so an unset flag cannot be
> mistaken for "LoGeR mode" — `False` is a *valid* value, so a `False` default would have made
> the unset case indistinguishable from the real LoGeR variant and silently selected the wrong
> alignment mode. That guard has to be honoured here: every test below constructs a creator
> without calling `_load_model`, so `self._se3` is `None` at `_forward` time.
>
> Add this at the top of `_forward`, before anything else:
>
> ```python
>         # _se3 is populated by _load_model from the variant's yaml. None means _forward was
>         # reached without it — refuse rather than pick a default, because both values are
>         # legitimate (LoGeR is False, LoGeR_star is True) and guessing runs the wrong
>         # alignment mode with no error anywhere downstream.
>         if self._se3 is None:
>             raise RuntimeError("LoGeRCreator._forward requires _load_model to have run (se3 unset)")
> ```
>
> Do NOT default `_se3` inside `_creator` — that would restore the exact silent default the
> field change removed. Add a second, explicitly-named helper instead, and use it for the six
> tests that are not about the flag:
>
> ```python
> def _loaded_creator(se3: bool = False, **kwargs) -> LoGeRCreator:
>     """_creator with _se3 set to what _load_model would have read from the variant yaml."""
>     # Named parameter, not a hidden default: these tests stub the model, so _load_model never
>     # runs and _forward's se3 guard would otherwise fire on every one of them.
>     creator = _creator(**kwargs)
>     creator._se3 = se3
>     return creator
> ```
>
> Add one test for the refusal itself, which uses bare `_creator()` so `_se3` stays `None`:
>
> ```python
> def test_forward_refuses_to_run_before_load_model_sets_se3():
>     # _se3 is None until _load_model reads the variant yaml. Both real values are valid, so
>     # there is nothing safe to default to — guessing picks an alignment mode silently.
>     n, h, w = 2, 56, 70
>     with pytest.raises(RuntimeError, match="se3 unset"):
>         _creator()._forward(_FakeLoGeR(n, h, w, 80.0, 80.0), torch.rand(n, 3, h, w))
> ```
>
> That makes **7** tests in this task, not 6.
>
> `_FakeLoGeR` calls `_synthetic_local_points`. **That helper does NOT exist yet** — verified
> against `tests/pointcloud/test_loger_creator.py` at commit `3ad4a4e`, whose only module-level
> helpers are `_creator`, `stub_pi3`, and `_fake_frames`. Write it here, above `_FakeLoGeR`:
>
> ```python
> def _synthetic_local_points(h: int, w: int, fx: float, fy: float, z: float = 2.0) -> np.ndarray:
>     """Exact pinhole camera-frame pointmap, so a K fit over it must recover (fx, fy)."""
>     # The principal point must match the one estimate_intrinsics_from_points assumes —
>     # cx=(W-1)/2, cy=(H-1)/2 (collab_splats/geometry/transforms.py:159-162), NOT w/2.
>     # Off-by-half-a-pixel here biases the recovered focal, and the assertions below would
>     # then be pinning the bias rather than the fit.
>     uu, vv = np.meshgrid(
>         np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
>         np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
>     )
>     # Forward pinhole: X = u_c * Z / fx, Y = v_c * Z / fy, channel 2 IS Z. Constant z is what
>     # lets the depth test assert a single number independently of the focals.
>     pts = np.stack([uu * z / fx, vv * z / fy, np.full_like(uu, z)], axis=-1)
>     return pts[None].astype(np.float32)  # (1,H,W,3)
> ```
>
> The `(h, w) = (56, 70)` used by every test below is deliberate: both `(w-1)/2` and `(h-1)/2`
> land on `.5`, so no pixel has `x == 0` or `y == 0` and none is dropped by that function's
> `abs(x) > 1e-6` validity gate. Both are also multiples of the patch size 14.

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

    raw = _loaded_creator()._forward(model, views)

    assert raw["depth_conf"].shape == (n, h, w)
    assert raw["depth_conf"].min() >= 0.0 and raw["depth_conf"].max() <= 1.0
    assert raw["depth_conf"].max() == pytest.approx(1 / (1 + np.exp(-4.0)), rel=1e-4)


def test_forward_inverts_camera_poses_to_world_to_camera():
    # LoGeR returns camera-to-world; FeedforwardResult.extrinsics is world-to-camera.
    # The single easiest thing to get backwards, and silent when wrong.
    n, h, w = 3, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)

    raw = _loaded_creator()._forward(model, torch.rand(n, 3, h, w))

    assert raw["extrinsic"].shape == (n, 3, 4)
    # c2w camera 2 sits at x=+2, so the w2c translation must be -2, not +2.
    assert raw["extrinsic"][2, 0, 3] == pytest.approx(-2.0)


def test_forward_fits_and_broadcasts_intrinsics():
    n, h, w = 3, 56, 70
    fx, fy = 88.0, 80.0
    raw = _loaded_creator()._forward(_FakeLoGeR(n, h, w, fx, fy), torch.rand(n, 3, h, w))

    assert raw["intrinsics"].shape == (n, 3, 3)
    assert raw["intrinsics"][0, 0, 0] == pytest.approx(fx, rel=1e-3)
    assert raw["intrinsics"][0, 1, 1] == pytest.approx(fy, rel=1e-3)
    # _raw_to_world_points hard-requires this key (vggtx.py:304) and returns None without it.
    np.testing.assert_allclose(raw["intrinsics_downsampled"], raw["intrinsics"])


def test_forward_extracts_depth_from_the_third_channel():
    # LoGeR builds local_points as cat([xy * z, z]), so channel 2 IS depth — no
    # reprojection needed to recover it.
    n, h, w = 2, 56, 70
    raw = _loaded_creator()._forward(_FakeLoGeR(n, h, w, 80.0, 80.0), torch.rand(n, 3, h, w))
    assert raw["depth"].shape == (n, h, w, 1)
    np.testing.assert_allclose(raw["depth"], 2.0, rtol=1e-5)


def test_forward_passes_window_knobs_and_se3():
    n, h, w = 2, 56, 70
    model = _FakeLoGeR(n, h, w, 80.0, 80.0)
    creator = _loaded_creator(se3=True, window_size=16, overlap_size=4)

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
        _loaded_creator()._forward(model, torch.rand(n, 3, h, w) * 255.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k forward -v -p no:randomly`
Expected: 7 failed with `NotImplementedError` raised by the `_forward` stub in `_creator`.

- [ ] **Step 3: Implement**

**Also delete the `_forward` stub from `_creator`'s `_PartialLoGeRCreator`**, so these tests
reach the real method instead of the stub.

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
        # This must run before the K fit, whose conf gate is a threshold on a
        # probability. Measured on the Task 1 run, the raw logits span -4.257..-2.019 —
        # entirely negative — so skipping the sigmoid does not merely shift the gate, it
        # admits ZERO pixels and the fit raises.
        depth_conf = torch.sigmoid(preds["conf"]).squeeze(0).cpu().float().numpy()
        if depth_conf.ndim == 4:
            depth_conf = depth_conf.squeeze(-1)  # (N,H,W)

        # LoGeR returns camera-to-world; FeedforwardResult.extrinsics is world-to-camera.
        camera_poses = preds["camera_poses"].squeeze(0).cpu().float().numpy()  # (N,4,4) c2w
        extrinsic = invert_poses(camera_poses)[:, :3, :].astype(np.float32)  # (N,3,4) w2c

        # LoGeR predicts no intrinsics — solve one shared K and broadcast it per frame.
        #
        # The threshold is passed EXPLICITLY, not inherited. estimate_intrinsics_from_points
        # defaults to 0.1, which is right for a calibrated head but wrong for this one:
        # sigmoid over the measured logit range -4.257..-2.019 gives a confidence band of
        # [0.0140, 0.1172], so 0.1 is its 92nd PERCENTILE. At the default only 7.9% of
        # pixels survive (12,217/155,232 on the Task 1 run), and a slightly duller scene —
        # all logits below -2.2, still inside the measured band — admits none at all and
        # raises. LOGER_CONF_THRESHOLD sits near the bottom of the band so the gate rejects
        # only what the model calls junk, and the weighted median does the rest of the work.
        k = estimate_intrinsics_from_points(local_points, depth_conf, LOGER_CONF_THRESHOLD)
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
Expected: **34 passed** (27 after Task 6, plus the 7 written here).

Landed at **35** in commit `d72f32f`: an eighth test was needed because every test here runs
`_FakeLoGeR` at conf logit 4.0 (sigmoid 0.982), which clears both `LOGER_CONF_THRESHOLD` and the
library's 0.1 default — so omitting the explicit threshold survived mutation, leaving the one
constant this module exists to justify untested. `_FakeLoGeR` gained a `conf_logit` parameter and
a test runs at logit -3.0 (sigmoid 0.0474, inside LoGeR's measured band and below 0.1).

A post-review coverage round (`082e08d`) then took the file to **39**: the source comment claimed
parity with upstream's `build_forward_kwargs` while five of the nine kwargs were mutable with the
suite green, and the RGB assert's bounds were untested because `torch.rand` never emits exactly
0.0 or 1.0 — a pure-black pixel is common in real frames and `<` would reject it. Later tasks
count up from **39**.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/loger.py tests/pointcloud/test_loger_creator.py
git commit -m "feat(loger): add LoGeRCreator._forward

Three conversions the rest of the pipeline depends on:

- sigmoid on conf. LoGeR's conf_head is a bare LinearPts3d with no activation
  (Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:172); upstream activates at the
  call site (PolyCam/LoGeR @ 5d7c1a7, run_loger.py:481). It runs before the K
  fit, whose gate thresholds a probability. The measured logits are entirely
  negative (-4.257..-2.019), so skipping the sigmoid admits ZERO pixels and the
  fit raises — it does not merely shift the gate.
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

> **Task 8 closes the ABC.** `LoGeRCreator` subclasses `BasePointcloudCreator`, an `abc.ABC`
> (`collab_splats/pointcloud/base.py:101`), so it is **uninstantiable** until the last abstract
> methods land here. Tasks 6-7 ran their tests through `_creator`, a helper returning a subclass
> that stubs whichever abstract methods were not yet written.
>
> **Step 0 of this task: delete the `_creator` helper and its `_PartialLoGeRCreator`, and point
> every remaining call at the real `LoGeRCreator`.** After `_postprocess`, `_reproject`, and
> `extract_intermediate_features` land below, nothing is abstract and the stub scaffolding must
> not outlive its cause — a lingering stub would silently shadow a real method that someone
> later breaks.
>
> **`_loaded_creator` survives**, rebased onto the real class:
>
> ```python
> def _loaded_creator(se3: bool = False, **kwargs) -> LoGeRCreator:
>     """LoGeRCreator with _se3 set to what _load_model would have read from the variant yaml."""
>     creator = LoGeRCreator(**kwargs)
>     creator._se3 = se3
>     return creator
> ```
>
> It is still needed: these tests stub the model, so `_load_model` never runs and `_forward`'s
> `se3` guard would fire on every one of them.
>
> Add one test asserting the ABC is genuinely closed, which is the thing the scaffolding was
> standing in for all along:
>
> ```python
> def test_creator_is_instantiable():
>     # The five abstract methods of BasePointcloudCreator (collab_splats/pointcloud/base.py:101)
>     # are all concrete as of this task. Tasks 6-7 needed a stub subclass to run at all; this
>     # asserts that crutch is genuinely gone rather than merely deleted from the call sites.
>     assert isinstance(LoGeRCreator(), LoGeRCreator)
> ```
>
> That makes **5** tests in this task, not 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_loger_creator.py`:

```python
def _forward_and_postprocess(n=3, h=56, w=70, fx=88.0, fy=80.0, **creator_kwargs):
    creator = _loaded_creator(**creator_kwargs)
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
Expected: 4 failed (`test_creator_is_instantiable` is not matched by that `-k` filter; it fails too, with `TypeError: Can't instantiate abstract class`, until Step 3 lands).

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
Expected: **44 passed, 0 xfailed** (39 after Task 7, plus the 5 written here). Zero xfails and
zero stub scaffolding is the real completion signal for this task.

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

> **Plan correction (verified against the tree before dispatch).** The draft below contained a
> self-contradiction and one vacuous test. Both are fixed here; the reasoning is recorded because
> the same mistake is easy to re-introduce.
>
> **1. The `skipif` was backwards and is deleted.** The draft guarded `test_get_creator_loger` on
> `_LOGER_ROOT.exists()` while the *other* new test asserts loger.py imports fine **without** the
> vendored tree. Both cannot be right. The design is the second one: the vendored import is
> deferred into `_load_model`, so `import collab_splats.pointcloud` populates `_REGISTRY["loger"]`
> whether or not `third_party/LoGeR` is present. A `skipif` therefore skips the registry test in
> exactly the environment that most needs it — a bare CI checkout — leaving the wiring unverified
> where it is most likely to break. The draft commit message even claimed the guarded test "is the
> only thing that proves the wiring", which is the argument *against* guarding it.
>
> **2. `test_module_imports_without_the_vendored_tree` was vacuous** and needs a real assertion.
> `third_party/LoGeR` IS vendored in this dev environment, so "the import at the top of this file
> IS the assertion" proves nothing about the bare-checkout case it names. Assert the observable
> consequence instead: after importing `loger`, no vendored module may be in `sys.modules`.
>
> **3. Pre-existing wart, mirror it, do not fix it.** `__all__` in
> `collab_splats/pointcloud/feedforward/__init__.py` already lists `VGGTOmegaCreator` even though
> its import is guarded, so `import *` raises `AttributeError` when that backend is absent. Adding
> `"LoGeRCreator"` alongside reproduces the wart. That is correct for this task — consistency beats
> a one-backend fix — but do not describe it as safe.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_registry.py`:

```python
def test_get_creator_loger():
    # NOT guarded on third_party/LoGeR being present: the vendored import is deferred
    # into _load_model, so the registry entry exists in a bare checkout too. Guarding
    # this would skip it in exactly the environment where the wiring can break.
    from collab_splats.pointcloud.feedforward import LoGeRCreator

    assert get_creator("loger") is LoGeRCreator
```

No new imports are needed in that file.

Create `tests/pointcloud/feedforward/test_loger_load_guard.py`:

```python
"""LoGeR fails at _load_model time, not import time — the guard that keeps the
registry importable in a bare checkout with no vendored tree."""
import sys

import pytest

from collab_splats.pointcloud.feedforward.loger import _LOGER_ROOT, LoGeRCreator


def test_module_import_does_not_touch_the_vendored_tree():
    # loger.py must not import from third_party at module level. If it did, a bare
    # checkout would break `import collab_splats.pointcloud` entirely rather than just
    # omitting the backend. Asserting `LoGeRCreator is not None` would be vacuous here
    # — the tree IS vendored in this environment — so assert the observable
    # consequence: importing the module must not have pulled the vendored package in.
    assert "pi3" not in sys.modules
    assert not any(m.startswith("loger.") for m in sys.modules)


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

> **Verify the module names in `test_module_import_does_not_touch_the_vendored_tree` rather than
> trusting them.** `"pi3"` and `"loger.*"` are inferred from the vendored layout
> (`loger/models/pi3.py` in `github.com/Junyi42/LoGeR @ 7685b7a`) and from how `_load_model`
> inserts `_LOGER_ROOT` on `sys.path` — but which key actually lands in `sys.modules` depends on
> the exact import statement `_load_model` uses. Read `_load_model`, then **prove the assertion
> can fail**: import the vendored package by hand in a throwaway process, confirm the key you are
> asserting on really appears, and only then keep the assertion. An assertion on a module name
> that never appears under any circumstance passes forever and pins nothing — that failure mode
> has already cost this task nine review findings.

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
rather than breaking \`import collab_splats.pointcloud\`.

The registry test is deliberately NOT skipif-guarded on third_party/LoGeR being
present. Because the vendored import is deferred, the registry entry exists in a
bare checkout too, so guarding the test would skip it in exactly the environment
where the wiring is most likely to break."
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
            use_multiview_confidence=False,
        )


class _KwargsRecorded(Exception):
    """Sentinel: the creator was constructed, so stop before any inference runs."""


def test_creator_kwargs_reach_the_constructor(tmp_path, monkeypatch):
    # The per-backend config block is the only way to set model knobs from yaml, so the
    # thing under test is the _run_feedforward passthrough — NOT that LoGeRCreator accepts
    # kwargs, which was already true before this task. Substituting a recording stub that
    # raises as soon as it is constructed pins the plumbing without running inference.
    from collab_splats.pointcloud import feedforward as ff_mod
    from collab_splats.wrapper.reconstructor import _run_feedforward

    seen = {}

    class _Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _KwargsRecorded

    monkeypatch.setattr(ff_mod, "LoGeRCreator", _Recorder)

    with pytest.raises(_KwargsRecorded):
        _run_feedforward(
            backend="loger",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure=False,
            viz_enabled=False,
            viz_port=8080,
            max_points=1234,
            use_multiview_confidence=False,
            creator_kwargs={"window_size": 64, "variant": "LoGeR"},
        )

    # Non-empty first: a monkeypatch that missed its target would let a real creator be
    # constructed, and that could raise for its own reasons that pytest.raises misreads.
    assert seen, "LoGeRCreator was never constructed — the monkeypatch did not take"
    assert seen["window_size"] == 64 and seen["variant"] == "LoGeR"
    assert seen["max_points"] == 1234


@pytest.mark.parametrize("reserved", ["max_points", "use_multiview_confidence"])
def test_creator_kwargs_may_not_redeclare_a_reserved_key(tmp_path, reserved):
    # Both keys are already passed explicitly; a duplicate would surface as an opaque
    # TypeError from the constructor rather than naming the config key at fault.
    from collab_splats.wrapper.reconstructor import _run_feedforward

    with pytest.raises(ValueError, match=reserved):
        _run_feedforward(
            backend="loger",
            frames_zarr=tmp_path / "frames.zarr",
            output_dir=tmp_path / "out",
            loop_closure=False,
            viz_enabled=False,
            viz_port=8080,
            max_points=1000,
            use_multiview_confidence=False,
            creator_kwargs={reserved: 5},
        )
```

> **`use_multiview_confidence=False` is mandatory in every call above.** It is a
> **non-defaulted** parameter of `_run_feedforward` (`:158`) as of the concurrent session's
> `63cd05f`. Omit it and every one of these tests dies on
> `TypeError: _run_feedforward() missing 1 required positional argument` — *before* reaching the
> refusal or the passthrough, so `pytest.raises(ValueError)` fails with a confusing message that
> looks like the feature is broken. **Re-read the signature before writing the tests**; if more
> required parameters have appeared, add those too.

> **Plan correction (verified against the tree before dispatch).**
>
> **1. The drafted `test_creator_kwargs_reach_the_constructor` tested nothing this task adds.** It
> constructed `LoGeRCreator(max_points=1234, window_size=64, variant="LoGeR")` directly — never
> touching `_run_feedforward`, the new `creator_kwargs` parameter, the `pc_cfg.get(...)`
> passthrough, or the duplicate-`max_points` rejection. It passes identically before and after
> Task 10. That matters more than usual here because the commit message itself describes
> `creator_kwargs` as a **generic passthrough for every backend** — scope beyond "add loger" —
> which would have landed with zero coverage. Rewritten above to drive `_run_feedforward`, plus a
> second test for the rejection branch.
>
> One thing to verify rather than assume in the recorder test: it patches `LoGeRCreator` on
> `collab_splats.pointcloud.feedforward`, which works only because `_run_feedforward` imports the
> name from that package *inside the function body* (`:175-179`), so the lookup happens at call
> time. **Confirm the patch actually takes effect** — assert `seen` is non-empty rather than
> trusting `pytest.raises` alone, since a patch that missed would surface as a real construction
> attempt and could plausibly raise something else that the test then mistakes for success.
>
> **2. Line numbers drift constantly — DO NOT trust any number written here.** A concurrent
> session landed eight commits into `reconstructor.py` and `loger.py` while this plan was being
> written, and every line number below moved within an hour of being corrected. **Locate each
> edit site by searching for its text, never by line number.** As of the last check:
> `_FEEDFORWARD_BACKENDS` at `:43`, `FrameStore` imported at `:23`, `def _run_feedforward` at
> `:150`, the deferred `from collab_splats.pointcloud.feedforward import (...)` block at `:176`,
> `creator_map = {` at `:197`, the creator construction at `:205`, and the call site at `:562`.
>
> **3. `pc_cfg.get(...)` is safe**, checked: `pc_cfg = self.config["pointcloud"]` is a plain dict
> merged from yaml by `mergedeep`, not a strict-access wrapper, so `.get` exists and defaults.
>
> **4. CORRECTION TO A CORRECTION — `max_points` is NO LONGER the last parameter.** An earlier
> revision of this plan stated that `_run_feedforward`'s signature ends at `max_points: int`, so a
> defaulted `creator_kwargs` could be appended straight after it. **That is now false.** The
> concurrent session's `63cd05f` inserted `use_multiview_confidence: bool` *after* `max_points`
> (`:157-158`), so putting a defaulted parameter between them is a syntax error —
> `SyntaxError: parameter without a default follows parameter with a default`. **Append
> `creator_kwargs: dict | None = None` at the very end of the signature, after whatever
> non-defaulted parameters exist when you get there**, and re-read the signature first.
>
> **5. The creator construction now passes two explicit kwargs, not one.** It reads
> `creator_map[backend](max_points=max_points, use_multiview_confidence=use_multiview_confidence)`.
> Both are therefore duplicate-collision hazards for `creator_kwargs`, not just `max_points`.
> **Reject both**, with the same reasoning and one message naming whichever key collided:
>
> ```python
> # Both of these are already passed explicitly; a duplicate in the per-backend block
> # would surface as an opaque TypeError that names neither the key nor the config
> # path. Unknown keys are left to the constructor's own TypeError, which names them.
> extra = dict(creator_kwargs or {})
> for reserved in ("max_points", "use_multiview_confidence"):
>     if reserved in extra:
>         raise ValueError(
>             f"pointcloud.{backend}.{reserved} is not settable; use pointcloud.{reserved}"
>         )
> ```
>
> Update `test_creator_kwargs_may_not_redeclare_max_points` to cover **both** reserved keys —
> parametrize rather than duplicating the test body. And when you assert on `seen` in the recorder
> test, remember the expected dict now contains `use_multiview_confidence` too; **measure what the
> recorder actually captures rather than predicting it.**
>
> **6. Owed to Task 12, not here:** nothing writes a `pointcloud.loger:` block into `base.yaml`.
> Until it exists, `pc_cfg.get("loger", {})` returns `{}` and LoGeR runs at its dataclass
> defaults. That is correct behaviour, not a bug — but the config surface is not real until
> Task 12 adds the block.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k "backend or loop_closure or creator_kwargs or max_points" -v -p no:randomly`
Expected: 4 failed — `loger` not in `_FEEDFORWARD_BACKENDS`; no `ValueError` for LC; and both
`creator_kwargs` tests failing on `_run_feedforward() got an unexpected keyword argument`.
**Check what `-k` actually selects before trusting the count** — `backend` and `max_points` are
broad substrings and may match unrelated tests already in the file.

- [ ] **Step 3: Wire the Reconstructor**

In `collab_splats/wrapper/reconstructor.py`:

1. Line 43 — add `loger`:
```python
_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega", "loger"}
```

2. In `_run_feedforward`'s signature, add a defaulted parameter **at the very end**, after the
   last non-defaulted parameter (currently `use_multiview_confidence: bool` — re-read it first,
   it has moved once already):
```python
    max_points: int,
    use_multiview_confidence: bool,
    creator_kwargs: dict | None = None,
```

3. In the deferred import block, add `LoGeRCreator`:
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

5. Add `loger` to `creator_map` and route `creator_kwargs` into the construction. **Keep the two
   existing explicit kwargs and the existing comment above them** — `use_multiview_confidence`
   arrived in `63cd05f` and this task must not drop it:
```python
    # Select creator class by backend name
    creator_map = {
        "vggtx": VGGTXCreator,
        "mapanything": MapAnythingCreator,
        "vggt_omega": VGGTOmegaCreator,
        "loger": LoGeRCreator,
    }
    # max_points caps the confidence mask during inference — a memory guard, not a preference.
    # use_multiview_confidence is the only mv knob exposed: rel_thresh and min_views stay as
    # calibrated creator field defaults so nobody hand-tunes bare floats in YAML.
    # Both are passed explicitly, so a duplicate in the per-backend config block would surface
    # as an opaque TypeError naming neither the key nor its config path. Reject those two by
    # name; unknown keys are left to the constructor's own TypeError, which names them.
    extra = dict(creator_kwargs or {})
    for reserved in ("max_points", "use_multiview_confidence"):
        if reserved in extra:
            raise ValueError(
                f"pointcloud.{backend}.{reserved} is not settable; use pointcloud.{reserved}"
            )
    creator = creator_map[backend](
        max_points=max_points,
        use_multiview_confidence=use_multiview_confidence,
        **extra,
    )
```

6. At the call site (find it by searching for `max_points=pc_cfg["max_points"]`, currently `:562`,
   directly above `use_multiview_confidence=pc_cfg["use_multiview_confidence"]`), add the
   passthrough as a third line — do not disturb the two that are already there:
```python
            max_points=pc_cfg["max_points"],
            use_multiview_confidence=pc_cfg["use_multiview_confidence"],
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

> **Plan correction — the drafted test was a tautology, and its inputs do not work.**
> Measured against the tree before dispatch; all four points below are defects in the original
> draft, not preferences.
>
> **(a) It never called production code.** It built `k_model` itself as `f * scale_x`, then
> asserted `k_model[0,0] / scale_x == f` — algebraically `f` regardless of what the codebase does.
> It would pass with the entire LoGeR feature reverted. The real function is
> `_rescale_reconstruction_to_original_dimensions` and it must actually be invoked.
>
> **(b) `1920x1080` is exactly isotropic at this pixel budget**, so the anisotropy assertion
> `k_model[0,0] != approx(k_model[1,1])` fails immediately — while Step 2 claimed "Expected:
> PASS". Measured: `1920x1080 → 672x378`, `sx = sy = 0.350000`, ratio `1.000000`. Same for
> `1280x720` and `3840x2160`. **Use `640x480 → 574x434`** (`sx=0.896875`, `sy=0.904167`, ratio
> `0.991935`), which is genuinely anisotropic. `1440x1080` and `1600x1200` also work.
>
> **(c) The citation `base.py:580` is wrong twice over.** The rescale is at
> `collab_splats/pointcloud/feedforward/base.py:801-805`, and it **multiplies** by
> `scale = original/model` — the reciprocal of the draft's `scale_x = model/original`. Writing the
> comment as "the rescale is exactly this division" inverts the production convention.
>
> **(d) The "snap to the mean" it argues against does not live in the rescale.** The mean is in
> `build_colmap` at `base.py:719-722`
> (`params = [(K[0,0] + K[1,1]) / 2.0, ...]` for `SIMPLE_PINHOLE`); the rescale's SIMPLE_PINHOLE
> branch then applies `max(scale_x, scale_y)` (`base.py:801-802`), not a mean. The real penalty is
> the **composite of those two stages**, and it is worth measuring rather than asserting.
>
> **Measured composite** for a square-pixel `f = 1600` camera at `640x480`:
>
> | camera_model | recovered focal | error |
> |---|---|---|
> | `PINHOLE` | `fx = fy = 1600.0000` | `~2e-8` relative — exact |
> | `SIMPLE_PINHOLE` | `f = 1606.5041` | **+6.504 px, +0.41%** |
>
> That 6.5 px is the concrete reason `LoGeRCreator.camera_model` is `"PINHOLE"`, and the test
> below pins it by driving both real functions.

Add to the **top-level imports** of the file (house style: no inline imports):

```python
from types import SimpleNamespace

from collab_splats.pointcloud.feedforward.base import (
    _rescale_reconstruction_to_original_dimensions,
)
from collab_splats.pointcloud.feedforward.loger import _compute_target_size
```

Then append:

```python
def _rescaled_camera_params(camera_model, params, model_wh, orig_wh):
    """Run the real rescale over a single camera and return its original-res params.

    _rescale_reconstruction_to_original_dimensions is duck-typed over pycolmap — it
    touches only .images/.cameras, .model.name, .params, .width, .height and .name.
    A SimpleNamespace stands in because constructing a real pycolmap.Reconstruction
    needs a Frame binding (`Check failed: image.HasFrameId()`) that is pure ceremony
    for a camera-only assertion.
    """
    model_w, model_h = model_wh
    orig_w, orig_h = orig_wh
    camera = SimpleNamespace(
        model=SimpleNamespace(name=camera_model),
        params=np.array(params, dtype=np.float64),
        width=model_w,
        height=model_h,
    )
    reconstruction = SimpleNamespace(
        images={1: SimpleNamespace(camera_id=1, name="0.png", points2D=[])},
        cameras={1: camera},
    )
    _rescale_reconstruction_to_original_dimensions(
        reconstruction,
        [Path("0.png")],
        np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]], dtype=np.float32),
        (model_w, model_h),
    )
    return reconstruction.cameras[1].params


def test_loger_pinhole_k_round_trips_to_original_resolution():
    """LoGeR's model-res K must rescale back to original resolution with fx != fy intact.

    _compute_target_size rounds each axis to a multiple of 14 independently, so a
    square-pixel physical camera genuinely produces fx != fy at model resolution.
    PINHOLE carries both focals through and the per-axis rescale recovers the true
    focal exactly on both axes.
    """
    orig_w, orig_h = 640, 480
    model_w, model_h = _compute_target_size(orig_w, orig_h, 255_000)

    # A square-pixel physical camera: one true focal, f = 1600 px at original resolution
    f = 1600.0
    scale_x, scale_y = model_w / orig_w, model_h / orig_h
    fx_model, fy_model = f * scale_x, f * scale_y

    # The anisotropy is real, not a rounding artefact — guard the premise of the test
    assert fx_model != pytest.approx(fy_model, rel=1e-3)

    # build_colmap's PINHOLE branch (base.py:719-720) keeps both focals
    params = _rescaled_camera_params(
        "PINHOLE",
        [fx_model, fy_model, (model_w - 1) / 2.0, (model_h - 1) / 2.0],
        (model_w, model_h),
        (orig_w, orig_h),
    )
    assert params[0] == pytest.approx(f, rel=1e-6)
    assert params[1] == pytest.approx(f, rel=1e-6)


def test_simple_pinhole_would_lose_the_focal_loger_keeps():
    """Why LoGeRCreator.camera_model is PINHOLE: SIMPLE_PINHOLE costs 6.5 px here.

    Two production stages compose. build_colmap averages fx and fy into one param
    for SIMPLE_PINHOLE (base.py:721-722), then the rescale multiplies that single
    param by max(scale_x, scale_y) (base.py:801-802) rather than per axis. Neither
    stage is lossy alone; together they do not round-trip.
    """
    orig_w, orig_h = 640, 480
    model_w, model_h = _compute_target_size(orig_w, orig_h, 255_000)
    f = 1600.0
    fx_model = f * (model_w / orig_w)
    fy_model = f * (model_h / orig_h)

    params = _rescaled_camera_params(
        "SIMPLE_PINHOLE",
        [(fx_model + fy_model) / 2.0, (model_w - 1) / 2.0, (model_h - 1) / 2.0],
        (model_w, model_h),
        (orig_w, orig_h),
    )

    # Measured: 1606.5041 against a true 1600.0 — +6.504 px, +0.41%
    assert params[0] == pytest.approx(1606.5041, abs=1e-3)
    assert params[0] != pytest.approx(f, rel=1e-3)
```

- [ ] **Step 2: Run the tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py -v -p no:randomly`

Expected: PASS. These assert behaviour Tasks 3 and 4 already made true, so they pass on first
run — that is expected here and is **not** a licence to skip Step 2b.

- [ ] **Step 2b: Prove the tests can fail**

A test that passes on first write has demonstrated nothing. Confirm each one is load-bearing:

1. Change `"PINHOLE"` to `"SIMPLE_PINHOLE"` in the first test's `_rescaled_camera_params` call.
   `test_loger_pinhole_k_round_trips_to_original_resolution` must fail **on the assertion**, not
   on an index error. Restore.
2. In `base.py:801`, change `max(scale_x, scale_y)` to `min(scale_x, scale_y)`.
   `test_simple_pinhole_would_lose_the_focal_loger_keeps` must fail. Restore, and verify the
   restore with `md5sum` — **`base.py` is being edited by a concurrent session, so never restore
   it with git.**
3. Change `orig_w, orig_h` to `1920, 1080` in either test. The anisotropy guard must fail,
   demonstrating that the isotropic-input trap is actively detected rather than silently passing.

Report which mutations killed which tests.

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_feedforward_intrinsics.py
git commit -m "test(loger): pin the model-res to original-res K round-trip

Lives alongside the existing cross-backend intrinsics-resolution tests rather
than in test_loger_creator.py, since that file already owns this contract.

Both tests drive the real _rescale_reconstruction_to_original_dimensions over a
duck-typed camera rather than re-deriving its arithmetic, so they fail if the
rescale changes. The second one is the load-bearing half: it measures that
SIMPLE_PINHOLE recovers 1606.50 px against a true 1600.0 — build_colmap averages
fx and fy, then the rescale scales that single param by max(scale_x, scale_y)
instead of per axis. That 6.5 px is why LoGeRCreator.camera_model is PINHOLE.

Uses 640x480 as the source size: 1920x1080 is exactly isotropic at this pixel
budget (sx = sy = 0.35), so it cannot exercise the fx != fy path at all." -- tests/pointcloud/test_feedforward_intrinsics.py
```

---

## Task 12: Config and docs

**Files:**
- Modify: `configs/base.yaml`
- Modify: `configs/README.md`
- Modify: `docs/source/api/pointcloud.rst`
- Modify: `docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md`

> **Step 0: the owed citation sweep.** This task already touches docs, so the deferred citation
> work lands here. Three groups, all in the spec:
>
> 1. **~7 bare `run_loger.py:NNN` references in the prose.** `run_loger.py` exists ONLY in
>    `github.com/PolyCam/LoGeR @ 5d7c1a7` — it is not in the vendored tree — so they are not
>    fork-ambiguous, but they still fail the repo+commit+file+line contract. Add the pin.
> 2. **~9 `feedforward/base.py:NNN` references** (spec lines 284-287, 293, 296, 439, 451, 583).
>    These are first-party, so no repo pin is needed, but **every line number must be re-verified
>    against the current file** — this is `collab_splats/pointcloud/feedforward/base.py`, a
>    different file from `collab_splats/pointcloud/base.py`, and the two have been conflated
>    once already on this task.
> 3. **Anything citing `collab_splats/pointcloud/base.py`.** The class `BasePointcloudCreator`
>    is at **line 101**; line 4 is the `from abc import ABC, abstractmethod` line. Five instances
>    of the wrong `:4` were already corrected in this plan and one in the test file — check
>    nothing else carries it.
>
> 4. **Every `basic.py:NNN` reference must name its enclosing function.** `loger/utils/basic.py`
>    contains **two near-identical loaders** — frame-0 sizing appears at both `:53-54` and
>    `:289-290`, `os.listdir` at both `:21` and `:254`. A bare line number there is genuinely
>    ambiguous between them. Ours all refer to `load_images_as_tensor` (signature at
>    `basic.py:11`); say so at each site.
>
> **Verify, do not remember.** Wrong line ranges have been caught five separate times on this
> task (`basic.py:51-63` → `:55-61`; `base.py:4` → `:101`, plus a reviewer's own off-by-two on
> that same line; `basic.py:16` → `:21`; `basic.py:52-53` → `:53-54`). Open each file and read
> the line before writing the citation.

> **Required deliverable inherited from Task 10 — do not drop it.** Task 10 added the call-site
> passthrough `creator_kwargs=pc_cfg.get(pc_cfg["backend"], {})` in `_run_feedforward`'s caller.
> Mutating that expression to a literal `{}` is currently killed by **no test**, and both the
> implementer and the spec reviewer confirmed it. That was accepted for Task 10 on a specific
> ground: with no `pointcloud.<backend>:` block anywhere in `base.yaml`, `pc_cfg.get(...)` returns
> `{}` for every backend, so the mutant is **semantically identical to the real code** and no test
> could distinguish them.
>
> **Adding the `pointcloud.loger:` block below is exactly what ends that.** The moment it exists,
> the passthrough carries real values and the mutant becomes a genuine silent-no-op bug: LoGeR
> would run at dataclass defaults while the YAML says otherwise, with nothing failing. **This task
> must therefore add a `Reconstructor`-level test** that builds a config containing a
> `pointcloud.loger:` block, drives the path that calls `_run_feedforward`, and asserts a value
> from that block reaches the creator. Substituting a recording stub for `LoGeRCreator` (as
> `test_creator_kwargs_reach_the_constructor` does) avoids running inference. **Verify it by
> mutating the call site to `creator_kwargs={}` and confirming your new test fails** — if it does
> not, the test is not doing its job.

- [ ] **Step 1: Update `configs/base.yaml`**

**`configs/base.yaml` is being edited concurrently** (a `preprocessing.frame_proportion` removal as of 2026-08-13). Read the file immediately before editing, match on content rather than the line numbers quoted here, and stage with an explicit pathspec at commit time so an unrelated concurrent edit is not swept in.

> **CORRECTION (verified against `configs/base.yaml` on 2026-08-13): do not make the `max_frames`
> edit as drafted — it has gone stale and is now a regression.**
>
> The concurrent `fps-frame-sampling` work has already landed. The line now reads:
>
> ```yaml
>   max_frames: 300             # frame budget; vggt_omega OOMs above ~300 on a 44 GB GPU
> ```
>
> Two things follow. First, **the "stale cap at 200" note below is describing a state that no longer
> exists** — that comment is already gone; there is nothing to correct. Second, and more important,
> **the drafted replacement calls `max_frames` a "cap", which actively undoes a deliberate
> terminology fix.** `max_frames` now has ONE meaning across all three samplers — the *frame
> budget*: the target COUNT for `uniform`, a ceiling for `fps` and `optical_flow`. "Cap" is simply
> wrong for `uniform`, where it is the target, not a limit.
>
> **Keep the existing wording and append the LoGeR clause to it**, rather than replacing the line:
>
> ```yaml
>   max_frames: 300             # frame budget; vggt_omega OOMs above ~300 on a 44 GB GPU.
>                               # NOT a LoGeR limit — LoGeR is windowed and built for longer
>                               # sequences — but preproc runs first, so it binds LoGeR too.
>                               # See configs/README.md.
> ```
>
> The same correction applies to Step 2's README row; see the note there.

The `backend` line and the new block become:

```yaml
  backend: vggt_omega         # vggt_omega | vggtx | mapanything | loger  (feedforward only)
  # Per-backend creator kwargs. Only the block matching `backend` is read, so all
  # four can be documented here at once. max_points is NOT settable here — it is a
  # pipeline-level guard above.
  loger:
    variant: LoGeR_star       # LoGeR | LoGeR_star (SE(3)); selects config AND weights
    window_size: 32           # sliding-window length; upstream default, see README
    overlap_size: 3           # frames shared between adjacent windows
    reset_every: 0            # hard-reset TTT fast weights every N frames; 0 = never
    conf_threshold: 50.0      # depth-confidence PERCENTILE (0-100), not a raw value
```

> **These five values were verified against the built class on 2026-08-13** by introspecting
> `LoGeRCreator.__init__` — `variant='LoGeR_star'`, `window_size=32`, `overlap_size=3`,
> `reset_every=0`, `conf_threshold=50.0`. The YAML therefore restates the dataclass defaults and
> changes no behaviour on its own; its job is to make the knobs discoverable and to give the Task 10
> passthrough something real to carry. **Re-introspect rather than trusting this list** — the
> constructor has grown since: `num_iterations`, `pixel_limit`, and the four multiview-confidence
> kwargs (`use_multiview_confidence`, `min_views`, `mv_conf_abs_thresh`, `mv_conf_rel_thresh`, added
> by the concurrent session) also exist. Documenting a curated subset is the right call — the four
> multiview kwargs belong to `2026-08-12-multiview-confidence-all-models-design.md` and should be
> documented there, once, for all backends — but say in the commit that the subset is deliberate.
>
> **`window_size: 32` is a citation and Step 0 governs it.** Write it as
> `github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:<line>` in `configs/README.md` (the YAML comment
> stays short and points at the README), and **read the line before writing the number** — that file
> is not in the vendored tree, so it must be checked against the PolyCam fork, not `third_party/`.
> A bare `run_loger.py` here would be the exact defect Step 0 exists to sweep.

- [ ] **Step 2: Update `configs/README.md`**

Same concurrency caution as Step 1 — match on row content, not line numbers.

> **CORRECTION (verified against `configs/README.md` on 2026-08-13): do not replace the
> `preprocessing.max_frames` row.** The `fps-frame-sampling` work rewrote it and added three
> siblings plus a whole "Choosing a frame sampler" section. It currently reads:
>
> ```markdown
> | `preprocessing.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300) |
> ```
>
> The drafted replacement drops the per-method distinction that row was just given, for the same
> reason set out in Step 1. **Append the LoGeR clause instead of overwriting**, keeping the budget
> wording intact:
>
> ```markdown
> | `preprocessing.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300 — not a LoGeR limit, see below) |
> ```
>
> The "see below" needs somewhere to point. The existing **"`max_frames` still dominates on long
> video"** paragraph in "Choosing a frame sampler" is the natural home: it already says the cap is a
> measured GPU limit that `fps` cannot route around. Add one sentence there noting the limit is
> `vggt_omega`'s rather than the pipeline's, and that `loger` is windowed and expected to run past
> it — with the number Task 14 measures, or an explicit "not yet swept" if Task 14 has not run.

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

    n, orig_h, orig_w = 8, 480, 640
    creator = LoGeRCreator()
    # Derive model resolution rather than hardcoding it — _compute_target_size rounds
    # each axis to a multiple of 14 under creator.pixel_limit, and the reshape below
    # raises on any mismatch.
    w, h = _compute_target_size(orig_w, orig_h, creator.pixel_limit)
    model = creator._load_model("cuda")

    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, size=(n, orig_h, orig_w, 3), dtype=np.uint8)
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

    # Confident pixels only — the residual is meaningless where the model is unsure.
    # raw["depth_conf"] is ALREADY post-sigmoid (loger.py:392) and this head's measured
    # band is [0.0140, 0.1172], so any fixed threshold near 0.5 selects nothing. Gate on
    # LOGER_CONF_THRESHOLD, the same floor the K fit uses (loger.py:79, :403).
    mask = raw["depth_conf"] > LOGER_CONF_THRESHOLD
    # Fail loudly on an empty mask. Without this the medians below are nan, the assert
    # reads as "LoGeR is non-pinhole", and the plan's own instruction not to relax the
    # threshold would send you chasing a measurement that never happened.
    assert mask.sum() > 0, (
        f"confidence mask selected 0 of {mask.size} pixels at "
        f"threshold {LOGER_CONF_THRESHOLD}; conf range "
        f"[{raw['depth_conf'].min():.4f}, {raw['depth_conf'].max():.4f}]"
    )
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

Add `from collab_splats.pointcloud.feedforward.base import _raw_to_world_points` and
`from collab_splats.pointcloud.feedforward.loger import _LOGER_ROOT, LOGER_CONF_THRESHOLD, _compute_target_size`
to the test file's imports (several may already be there — check before adding duplicates).

> **Plan correction — two defects, verified against the tree. One of them fails silently in the
> most misleading way possible.**
>
> **(a) `mask = raw["depth_conf"] > 0.5` selects ZERO pixels.** `_forward` stores
> `depth_conf = torch.sigmoid(preds["conf"])` (`loger.py:392`), so the value is already
> post-sigmoid, and this head's measured logit span `-4.257..-2.019` puts the whole band in
> `[0.0140, 0.1172]`. Nothing ever exceeds `0.5`. The consequence is not a clean failure:
> `np.median` of an empty array is `nan`, `nan < 0.02` is `False`, and the assertion fires with
> its "the model is meaningfully non-pinhole" message. Step 2 then tells you **not** to relax the
> threshold because "that result is the finding" — so the plan as drafted walks you into
> recording a fabricated finding from a measurement that never ran. Fixed above by gating on
> `LOGER_CONF_THRESHOLD` and asserting the mask is non-empty first.
>
> **(b) `n, h, w = 8, 336, 462` is the wrong model resolution.** With `pixel_limit = 255_000`
> (`loger.py:196`) and a `640x480` source, `_compute_target_size` returns `(574, 434)` — measured.
> `ours.reshape(n, 336, 462, 3)` raises `ValueError: cannot reshape array of size ...`. Also note
> the ordering trap: `_compute_target_size` returns `(w, h)`, while the reshape wants
> `(n, h, w, 3)`. Fixed above by deriving both from `creator.pixel_limit`.
>
> Two things that DO check out, so leave them alone: `@pytest.mark.slow` is registered
> (`pyproject.toml:214-216`), and `_raw_to_world_points(raw, subsample=8)` really does read
> `'depth'`, `'extrinsic'`, `'intrinsics_downsampled'` (`feedforward/base.py:333,342-343`) — which
> is what the `"intrinsics_downsampled"` alias in `_forward` exists to satisfy.

> **Plan correction — third defect, plus one thing that looks like a defect and is not.**
>
> **(c) `raw["local_points"]` is dead payload, and the comment justifying it names a consumer
> that does not exist.** `loger.py:419` returns `"local_points": local_points,  # kept for the
> parity test only`. The parity test drafted above never reads it — it takes `native` from a
> second forward and `ours` from `_raw_to_world_points(raw)`. Measured: the ONLY consumer in the
> tree is `tests/pointcloud/test_loger_creator.py:504`, inside `test_forward_runs_under_no_grad`,
> which is a no-grad/type contract test rather than a parity test — and the adjacent
> `assert isinstance(raw["depth"], np.ndarray)` already covers that contract, since `depth` is a
> numpy slice of the same array (`loger.py:410`). So the key carries an `(N,H,W,3)` float32 array
> — ~3.0 MB/frame at the 574x434 model resolution — through every production `_forward` return to
> satisfy nothing. **Do at Task 13:** delete the key and that one assertion, per the standing
> "delete dead code the work obsoletes" constraint. If instead you find a real use, fix the
> comment to name it — do not leave it pointing at this test.
>
> **NOT a defect: the second forward pass.** It is tempting to derive `native` from
> `raw["local_points"]` and `raw["extrinsic"]` and skip the second 18 s / 6.77 GB run. Do not.
> `_forward` obtains `extrinsic` by `invert_poses(camera_poses)` (`loger.py:400`), so a derived
> `native` would apply OUR inversion to both sides, cancelling it — and the preamble's claim that
> this one assertion covers the pose inversion would silently become false. Taking `native` from
> LoGeR's own `preds["points"]` (a real key — `pi3.py:816`) keeps it independent. Checked and
> cleared separately: TTT fast weights are per-call, not module state (`ttt_dict` is created at
> `pi3.py:713` and consumed at `:721`, both inside one call), so the second pass is NOT
> contaminated by the first. **Add a comment in the test saying why it re-runs**, or the next
> reader will collapse it and delete the inversion coverage without noticing.
>
> **(d) Cosmetic.** Plan line 203 calls the 8-frame VRAM figure "the baseline for Task 13's
> sweep"; the sweep is Task 14 (line 237 says so correctly). Task 13 has no sweep.
>
> **Also fix the commit message below:** it claims the test proves "the legitimacy of reusing
> `unproject_and_filter_points`". That function exists (`vggtx.py:94`) but is NOT what this test
> exercises — the test calls `_raw_to_world_points` (`feedforward/base.py:333`), which performs no
> filtering whatsoever: it builds a dense `meshgrid` and returns all `(K, P, 3)` points
> (`:369-391`). Name the function actually under test.

- [ ] **Step 2: Run it**

Run:
```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -k residual -v -s -p no:randomly
```
Expected: PASS, with the `PINHOLE RESIDUAL:` line printed.

**If it fails the 2% threshold:** first confirm the mask assertion passed — a real measurement
must have happened before its result means anything. Then do not raise the threshold to make it
pass. That result is the finding — LoGeR is meaningfully non-pinhole, and the spec says it
"outranks this decision entirely". Record the number, stop, and report; the `world_points` source
and the mesh/BA implications need revisiting before this ships.

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
print('fx, fy =', cam.params[0], cam.params[1], '| ratio', cam.params[0] / cam.params[1])
print('points3D:', r.num_points3D(), '| images:', r.num_images())
" "$SPARSE"
```
Expected: `PINHOLE` with four params, and non-zero point/image counts.

**What this step does and does not prove — read before recording the result.** The tutorial video is
**1920x1080** (measured, see Step 5), and `_compute_target_size` maps that to **672x378** at the
default `pixel_limit=255_000`: `sx = 672/1920 = 0.350000` and `sy = 378/1080 = 0.350000` — *exactly*
isotropic. So on this input the original-resolution rescale applies the same factor to both axes, and
this run **cannot** exercise the resize anisotropy that motivated PINHOLE in the first place.

That does not make the step vacuous, but it does change what it measures. Any `fx != fy` you see here
comes from the **estimator** fitting the two axes independently, not from the resize. Record the
printed ratio as exactly that — the estimator's own anisotropy on a square-pixel source, where the
ground truth is `fx == fy`. It is the same quantity Task 13 measures as its residual, now observed
end-to-end through COLMAP export and rescale, so the two numbers should agree; if they disagree, one
of the two paths is wrong and that is a finding. Do not report this run as evidence that PINHOLE
preserves resize anisotropy — Task 11's unit test is what proves that, on a deliberately anisotropic
`640x480 → 574x434` input.

> **Plan correction — Step 4 states a transposed resolution as "measured". The conclusion
> survives; the numbers in it do not.**
>
> Measured with `ffprobe -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate`:
> `data/tutorial/tutorial_example-video.mp4` is **1080x1920 portrait**, 2388 frames, 24000/1001 fps —
> not `1920x1080`. `_compute_target_size` therefore returns **378x672**, not `672x378`. This is the
> same `(w, h)`-vs-`(h, w)` axis-order trap that Task 13's correction (b) already caught once on this
> feature; `_compute_target_size` returns `(w, h)`.
>
> **The substantive claim is unaffected:** both orientations scale by exactly `0.350000` on both axes
> (`378/1080 = 672/1920 = 0.35`, `378*672 = 254016 <= 255_000`), so the run still cannot exercise
> resize anisotropy, and Step 4's warning not to read it as evidence for PINHOLE still stands.
>
> **What to change when running it:** expect `cam.width, cam.height` to print `1080 1920`. Do not
> treat that as a broken export — the drafted text primes you to expect the transpose, and this is
> exactly the sort of mismatch that gets "fixed" in the wrong place.
>
> **Related measurement, recorded here because it changes what Task 13's number means:** the only real
> footage in this repo (`data/tutorial/`, and `data/outputs/frames.zarr` which shares the portrait
> geometry) is *exactly isotropic* under this mapping. The synthetic `640x480 -> 574x434` fixture used
> by Task 13 and Task 11 is **anisotropic** (`sx = 0.896875`, `sy = 0.904167`) and is the only thing
> in the suite that puts the two axes on different scale factors. So the synthetic fixture is not a
> weaker substitute for real data here — for the anisotropy question it is the *stronger* one.

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

`min_frames` and `max_frames` are both pinned to N. Setting `max_frames` alone measures nothing: it is a cap, and the actual extracted count is decided by the sampler, so the run would process whatever the sampler chose rather than N.

**Both preconditions for this were re-verified against the post-`fps-frame-sampling` sampler on
2026-08-13 — do not re-hedge them, but do re-check if that code moves again:**

1. **`min_frames` is legal here only because `frame_selection` is `fps`.** `sample_frames` raises
   `ValueError` if `min_frames is not None` under any other method (`preproc/sampling.py:424-427`),
   and the floor is `fps`-only by design. `configs/base.yaml` currently defaults
   `frame_selection: fps`, so the sweep YAML above inherits it and the floor is accepted. **If you
   add `frame_selection: uniform` to any sweep YAML you must also drop `min_frames`, or the run dies
   before decoding a single frame.**
2. **Pinning both knobs yields exactly N**, not approximately N. `_sample_fps` clamps
   `max_frames` first and `min_frames` second (`preproc/sampling.py:611-618`), so
   `bounded = max(min(requested, N), min(N, total))` collapses to `N` for any `N <= total`. The
   re-spread then calls `_uniform_targets(total, N)` over the **whole** video rather than truncating.
   The blur gate does not erode the count either — it substitutes the sharpest neighbour inside its
   validation window instead of dropping the frame.

**The video is already measured — do not re-run ffprobe to decide whether the sweep is reachable.**
`data/tutorial/tutorial_example-video.mp4` is **2388 frames, 1920x1080, 24000/1001 = 23.976 fps**
(≈99.6 s). All four rows are therefore reachable: 1000 < 2388. Note that at the base `fps: 1.0` an
unpinned run would sample only ~100 frames, which is why every row here overrides both knobs.

If a row fails, attribute the failure before recording it: a `ValueError` naming `min_frames` is
precondition 1 above, not a memory ceiling.

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

Add to the "In-Flight Work" section's recently-completed list, following the existing entries' format: what shipped, the measured pinhole residual, the measured frame ceiling, the LC and multiview-confidence follow-ups that remain owed, and the licence position (nothing copied — all three helpers written from the maths, both forks cited as prior art; `third_party/LoGeR` is still executed at runtime under no stated licence, which is a deployment question).

```bash
git add CLAUDE.md
git commit -m "docs: record loger backend completion in CLAUDE.md"
```

---

## Owed to the user before this ships

1. ~~**The missing LICENSE.**~~ **RESOLVED 2026-08-13 (user).** Neither LoGeR repository ships one — confirmed against the vendored tree in Task 1, not assumed. Spec open item 7. It gated Task 2 rather than Task 3 as first written, since `_compute_weighted_median` was itself drafted as a port. **Decision: reimplement all three from first principles**, citing both forks as prior art and as the behavioural reference, never as the source of the lines. Task 4 additionally proves the behavioural match by test against the vendored loader instead of asserting it. Nothing is copied into tracked source, so the missing LICENSE no longer blocks anything here. What remains is a *non-blocking* note for the spec's open items: `third_party/LoGeR` is still cloned and executed at runtime under no stated licence, which is a deployment question rather than a source question.
2. **Loop closure calibration** for the LoGeR backbone. Refused until then; needs its own clean-negative sweep like the other four backbones. Spec open item 4.
3. **Multiview confidence** — owned by `2026-08-12-multiview-confidence-all-models-design.md`; `loger` should be added to its scope. Spec open item 5.
4. **Square-pixel averaging in original-resolution space** — dropped from this cut with a reason, revisit only if Task 13's residual shows the estimator's spread exceeds the ~1% the model-resolution version would cost. Spec open item 8.

## Self-review notes

**Spec coverage.** Every spec section maps to a task: source selection and the torch gate → Task 1; the fit and its three preserved behaviours → Tasks 2–3; the resize rule and anisotropy → Tasks 4, 6, 11; `_load_model` and the `se3` routing → Task 5; `_forward`'s sigmoid ordering and c2w→w2c → Task 7; `_postprocess` / `_reproject` / `world_points` → Task 8; error-handling table → Tasks 3, 5, 6, 7, 8, 10; registry and exports → Task 9; Reconstructor wiring, LC refusal, `max_frames` warning, kwargs passthrough → Task 10; configuration → Task 12; the eight unit tests and the parity integration test → Tasks 2–9, 11, 13; open items 1, 2, 3 → Tasks 1, 14, 13.

**Two spec claims were corrected during planning**, both verified against the vendored tree: the window knobs do not come from a `training_settings` block (there is none — both configs hold only `model:`), and `overlap_size` defaults to 3, not 8. `_run_feedforward` also takes explicit scalars and never sees `pc_cfg`, so the kwargs passthrough needed a parameter plus a call-site change rather than the single line the spec showed. All three are fixed in the spec at `1f06798`.

**Three plan-authoring errors were caught and fixed** by checking against the code rather than assuming: `FrameStore` has no `.count` — the frame count is `__len__` (`frame_store.py:65`), and it is already imported at `reconstructor.py:23`; `run_pipeline.py` has no `--set` flag, taking positional video paths plus `--output-root` and a `--config` override YAML, so Task 14 writes override files instead; and the sweep must pin `preprocessing.min_frames` alongside `max_frames`, since `max_frames` alone is only a cap and the sampler would decide the real count. `min_frames`/`max_frames` was chosen over raising `frame_proportion` because that key is being removed in concurrent work as of 2026-08-13; the floor-and-cap form survives either way.

**Two things this plan cannot pin down in advance**, each with an explicit check step rather than an assumption: LoGeR's exact output key names and `conf` rank (Task 1 Step 5 records them; Task 7 depends on them), and whether `1920x1080` happens to resize isotropically at the default pixel budget (Task 11 Step 2 gives a fallback input).
