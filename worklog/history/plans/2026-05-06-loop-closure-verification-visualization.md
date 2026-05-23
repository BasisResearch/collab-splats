# Loop Closure Verification & Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add richer `LoopMatch` dataclass and developer-facing audit sections to the eval notebook.

**Architecture:** Two independent layers — (1) `LoopMatch` converted to dataclass with `accepted` field + heap tiebreaker fix in `LoopMatchQueue` + feedforward gate loop accumulates all post-NMS candidates in `creator._lc_all_matches`; (2) notebook restructured with §4 detection audit and §9 endpoint gap. Each layer is independently testable and committed.

**Tech Stack:** numpy, gtsam, matplotlib, pytest

**Key decisions:**
- No 3D geometric gate — `world_points` are in submap-local frames; KD-tree overlap between distant LC candidates is meaningless. VGGT-Long's RANSAC-on-correspondences is not portable (requires a second VGGT forward pass per candidate). Existing DINO cosine + `translation_jump_check` remain.
- `verify_match_ratio` is NOT dead — gates DINO cosine in `_verify_loop_candidate`. No new config field.
- `LoopMatchQueue` heap breaks on equal scores after NamedTuple→dataclass. Fix with tiebreaker counter.
- Notebook uses `optimized` and `initial_chained` (not `corrected_extrinsics`).

---

## File Map

| Action | File | Change |
|--------|------|--------|
| ✅ Modify | `collab_splats/pointcloud/loop_closure/retrieval.py` | `LoopMatch` → dataclass with `accepted` field, heap tiebreaker counter in `LoopMatchQueue` |
| ✅ Modify | `collab_splats/pointcloud/feedforward.py` | Gate loop accumulates all candidates in `all_loop_candidates`; sets `match.accepted = True`; assigns `self._lc_all_matches`; updates docstring |
| ✅ Modify | `tests/pointcloud/test_loop_closure.py` | `test_loop_match_dataclass` replaces NamedTuple test; tiebreaker test added |
| Modify | `docs/pointcloud/loop_closure_eval.ipynb` | Part I/II headers, §4 detection audit, §9 endpoint gap, renumber §9→§10 |

---

## Task 1: `LoopMatch` dataclass + feedforward gate loop ✅ COMPLETE

**Files:**
- ✅ `collab_splats/pointcloud/loop_closure/retrieval.py`
- ✅ `collab_splats/pointcloud/feedforward.py`
- ✅ `tests/pointcloud/test_loop_closure.py`

Implemented changes:
- `LoopMatch` is now a `@dataclass` with `accepted: bool = False`
- `LoopMatchQueue.__init__` adds `self._counter: int = 0`; `push` uses `(-score, counter, match)` tiebreaker
- `get_matches` unpacks 3-tuple `(_, _, m)`
- Gate loop: `accepted` local var renamed to `verify_ok`; `continue` converted to `else` branch; `match.accepted = True` set on pass; `all_loop_candidates.append(match)` unconditionally at loop end
- `self._lc_all_matches = all_loop_candidates` assigned with other `_lc_*` inspection attrs
- `test_loop_match_dataclass`: verifies dataclass, mutable `accepted`, and default False
- `test_loop_match_queue_equal_score_tiebreak`: verifies no TypeError on equal scores

---

## Task 2: Notebook Restructure

**Files:**
- Modify: `docs/pointcloud/loop_closure_eval.ipynb`

- [ ] **Step 1: Add Part I/II markdown headers and renumber §9→§10**

After the existing §3 cell, insert a markdown cell:
```markdown
## Part I — Detection
```

After the existing §8 cell, insert a markdown cell:
```markdown
## Part II — Correction
```

Change the existing §9 (GT stubs) section header to §10.

- [ ] **Step 2: Run existing notebook cells to confirm no regressions**

```bash
jupyter nbconvert --to notebook --execute docs/pointcloud/loop_closure_eval.ipynb \
  --output /tmp/lc_eval_check.ipynb --ExecutePreprocessor.timeout=120
```
Expected: executes without error up to §8 (GPU cells may be skipped with `# noqa` tags).

- [ ] **Step 3: Add §4 Detection audit cell**

Insert after the Part I header comment / §3 cell. Cell type: code.

```python
# §4  Detection audit — image pair grid for all post-NMS loop candidates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

candidates = getattr(creator, "_lc_all_matches", [])

if not candidates:
    print("No loop closure candidates — run §3 with enable_loop_closure=True")
else:
    n = len(candidates)
    fig = plt.figure(figsize=(10, 2.5 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.5)

    for row, match in enumerate(candidates):
        q_submap = creator._lc_submaps[match.query_submap_id]
        d_submap = creator._lc_submaps[match.detected_submap_id]

        q_img = q_submap.frames[match.query_frame_idx].permute(1, 2, 0).cpu().numpy()
        d_img = d_submap.frames[match.detected_frame_idx].permute(1, 2, 0).cpu().numpy()
        q_img = np.clip(q_img, 0, 1)
        d_img = np.clip(d_img, 0, 1)

        verdict = "✓ accepted" if match.accepted else "✗ rejected"
        color = "green" if match.accepted else "red"

        ax_q = fig.add_subplot(gs[row, 0])
        ax_q.imshow(q_img)
        ax_q.set_title(
            f"Query  submap={match.query_submap_id} frame={match.query_frame_idx}\n"
            f"dist={match.similarity_score:.3f}  {verdict}",
            fontsize=8, color=color,
        )
        ax_q.axis("off")

        ax_d = fig.add_subplot(gs[row, 1])
        ax_d.imshow(d_img)
        ax_d.set_title(
            f"Detected  submap={match.detected_submap_id} frame={match.detected_frame_idx}",
            fontsize=8,
        )
        ax_d.axis("off")

    plt.suptitle(
        f"§4 Detection audit — {len(candidates)} post-NMS candidates  "
        f"({sum(m.accepted for m in candidates)} accepted)",
        fontsize=11,
    )
    plt.show()
```

- [ ] **Step 4: Add §9 Endpoint gap cell**

Insert after the §8 cell (before the Part II → §10 GT stubs cell). Cell type: code.

```python
# §9  Endpoint gap — world-space distance before vs after pose graph correction
import matplotlib.pyplot as plt
import numpy as np

accepted = [m for m in getattr(creator, "_lc_all_matches", []) if m.accepted]

if not accepted:
    print("No accepted loop closures — nothing to plot.")
else:
    labels, gaps_before, gaps_after = [], [], []

    for match in accepted:
        q_id = match.query_submap_id
        d_id = match.detected_submap_id
        q_fi = match.query_frame_idx
        d_fi = match.detected_frame_idx

        # Recover global frame indices from submap metadata
        q_submap = creator._lc_submaps[q_id]
        d_submap = creator._lc_submaps[d_id]
        q_global = q_submap.frame_start + q_fi
        d_global = d_submap.frame_start + d_fi

        pos_q_before = initial_chained[q_global][:3, 3]
        pos_d_before = initial_chained[d_global][:3, 3]
        gap_before = float(np.linalg.norm(pos_q_before - pos_d_before))

        pos_q_after = optimized[q_global][:3, 3]
        pos_d_after = optimized[d_global][:3, 3]
        gap_after = float(np.linalg.norm(pos_q_after - pos_d_after))

        labels.append(f"s{q_id}f{q_fi}↔s{d_id}f{d_fi}")
        gaps_before.append(gap_before)
        gaps_after.append(gap_after)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    ax.bar(x - width / 2, gaps_before, width, label="Before", alpha=0.8)
    ax.bar(x + width / 2, gaps_after, width, label="After", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("World-space endpoint gap (m)")
    ax.set_title("§9 Endpoint gap — accepted loop closure pairs")
    ax.legend()
    plt.tight_layout()
    plt.show()

    n_total = len(getattr(creator, "_lc_all_matches", []))
    n_accepted = len(accepted)
    mean_before = np.mean(gaps_before)
    mean_after = np.mean(gaps_after)
    mean_reduction = mean_before - mean_after
    print(f"Candidates (post-NMS): {n_total}  |  Accepted: {n_accepted} ({100*n_accepted/max(n_total,1):.0f}%)")
    print(f"Mean gap before: {mean_before:.3f} m  |  after: {mean_after:.3f} m  |  reduction: {mean_reduction:.3f} m")
```

- [ ] **Step 5: Commit**

```bash
git add docs/pointcloud/loop_closure_eval.ipynb
git commit -m "feat(notebook): add §4 detection audit and §9 endpoint gap to loop_closure_eval"
```

---

## Verification

1. `retrieval.py` unit validation:
   - `LoopMatch` is a dataclass, `accepted` defaults to `False`, is mutable
   - `LoopMatchQueue` with equal scores doesn't raise `TypeError`
   - NMS and top-k still work

2. Feedforward review:
   - `all_loop_candidates` initialized before submap loop
   - `match.accepted = True` set in the `else` branch (after both gates pass)
   - `all_loop_candidates.append(match)` unconditionally at end of match loop
   - `self._lc_all_matches` assigned next to `_lc_submaps`

3. Notebook:
   - Part I/II headers present
   - §4 renders without error on a real run
   - §9 renders without error and summary table prints
   - §10 (ex-§9 GT stubs) still runs
