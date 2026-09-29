# Pointcloud Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release-clean `collab_splats/pointcloud/` (plus the sfm call site in `wrapper/reconstructor.py`, the sfm block of `configs/base.yaml` and the pointcloud API page). Every commit makes the package smaller, truer or shorter, and none changes output that it does not name.

**Architecture:** There are two rounds on `clean/pointcloud-release`.
- **Round 1 (prose only)** changes docstrings, comments and yaml comments only. `prose_proof.py` proves each commit is AST-identical once docstrings are stripped.
- **Round 2 (code)** runs in three lane worktrees (A: utils/vda/depth_align/sfm, B: feedforward, C: reconstructor). Lanes run **one after another**: each forks from the branch tip after the previous lane is cherry-picked back, gating after every pick.
- Output-affecting ideas do not land here. They go to "Proposed additions (need approval)".

**Tech Stack:**
- Python 3.11 (`/opt/venv/reconstruction`), pytest
- numpy, torch, pycolmap, zarr 3
- `tests/test_docstring_contract.py` (base checks, plus `RELEASE_CHECKS` xfail pairs)
- the scratchpad tools `parity.py`, `prose_proof.py` and `contract_hits.py`

**References:**
- Spec: `docs/superpowers/specs/2026-09-26-pointcloud-release-cleanup-design.md`
- Decisions: `docs/superpowers/decisions/017-release-cleanup-rules.md`, `docs/superpowers/decisions/018-sfm-backends.md`
- Shape mirrors: `docs/superpowers/plans/2026-09-25-geometry-release-cleanup.md`

---

## Conventions for every task

```bash
WT=/workspace/collab-splats/.worktrees/pointcloud-release
PY=/opt/venv/reconstruction/bin/python
SP=/tmp/claude-0/-workspace-collab-splats/e908d41b-1068-4caa-9c92-045ca0ee282c/scratchpad/pc-release
```

- **G′ (full gate).** The first command is the proof line. It must print `$WT/collab_splats/__init__.py`, never `/workspace/collab-splats/collab_splats/...`.
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -c "import collab_splats;print(collab_splats.__file__)" && cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors tests/pointcloud tests/geometry tests/wrapper tests/evals tests/localization tests/remote tests/test_docstring_contract.py
  ```
  - Baseline (`$SP/baseline_a29.txt`): **18 failed / 1596 passed / 10 skipped / 21 xfailed / 59 xpassed / 3 errors**. All failures and errors are environmental.
  - A task passes G′ when `failed`, `errors`, `skipped` and `passed` equal the baseline, plus any count the task states.
  - `xfailed`/`xpassed` must equal the task's stated pair.
  - Compare the FAILED/ERROR node-id list against the baseline too. Counts alone are not enough.
- **Targeted run:**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/<file>::<test>
  ```
- **P (parity).** Run it after any commit touching `feedforward/*`, `depth_align.py`, `utils.py`, `pointcloud/base.py`, `utils/torch_utils.py` or `geometry/transforms.py`:
  ```bash
  cd $WT && PYTHONPATH=$WT PYTHONUTF8=1 $PY $SP/parity.py --check
  ```
  - It must print `SUMMARY PASS`.
  - On failure, run `cd $WT && git revert --no-edit HEAD`, then stop and report.
- **Round 1 proof.** The spec names geometry's `astcmp.py`; `prose_proof.py` is the same check (delete docstring statements, compare AST dumps) and reports per file. Run it before committing, against the pre-edit tip:
  ```bash
  cd $WT && $PY $SP/prose_proof.py HEAD <files...>
  ```
  - It must print `PROSE ONLY`.
  - After the commit, the same check is `prose_proof.py HEAD~1 <files...>`.
  - `prose_proof` deletes every string-constant `Expr` statement and replaces an emptied body with `Pass`. So a docstring added to an `...`-bodied abstract method is prose only **as long as the `...` line stays**. A body left as only a docstring becomes `Pass` and reads as CODE CHANGED.
- **Contract hits:**
  ```bash
  cd $WT && PYTHONPATH=$WT $PY $SP/contract_hits.py
  ```
  - With no args it prints the whole pointcloud package. Pass files to scope it.
  - It prints `TOTAL n`. HEAD 11c5d7c7 = **54**.
- **Commit:**
  ```bash
  cd $WT && git add <paths> && git commit --only <paths> -m "<type>(pointcloud): <subject>

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  - Use `git rm` for deletions.
  - Never use `--amend`, rebase, reset or a bare stash. Other sessions share the index and stash.
- Never pipe pytest through `tail`/`head`, and never use `--tb=no`.
- No notebook edits.
- Use US spelling.
- No inline imports except optional heavy deps (with a clear `ImportError`).
- Line length is 120 (black). isort's width is 88, so wrap long imports in parentheses.
- Every Bash command starts with `cd $WT &&`, because cwd resets between calls.
- `rtk` mangles some grep/awk/diff output. Use `rtk proxy <cmd>` when output looks wrong.

### Task IDs

| ID | Scope |
|---|---|
| T0 | preflight, no commit |
| R1-1 … R1-6 | Round 1, prose only, on `clean/pointcloud-release` directly |
| H1 … H4 | Round 2 shared helpers that the lanes depend on; land on the branch before the lanes fork |
| A-x | lane A: `utils.py`, `vda.py`, `depth_align.py`, `sfm/*` (+ new `sfm/common.py`), their tests, `evals/` row 84 |
| B-x | lane B: `feedforward/*`, `pointcloud/__init__.py`, `base.py`, LC wrapper row 9, dashboard row 39 |
| C-x | lane C: `wrapper/reconstructor.py`, `tests/wrapper/test_sfm_*.py`, `configs/` |
| F-B1 … F-B10 | spec section C bug fixes, serial on the branch after lane C |
| E-x | end tasks: docs, changelog, final gate |

### Lane workflow (Round 2)

**Lanes are sequential, not parallel.** Lane A edits `feedforward/base.py` (A-14, A-15) and `evals/scripts/eval_similarity_calibration.py` (A-14), which lane B also edits; lane B edits one line of `reconstructor.py` (B-11), which lane C rewrites. So: fork A → land A → fork B from the new tip → land B → fork C → land C. Each lane still gets its own worktree, so the main worktree stays clean while a lane is in progress.

```bash
cd $WT && git worktree add .worktrees/pc-lane-a -b pc-lane-a clean/pointcloud-release
cd $WT && for d in LoGeR Video-Depth-Anything hloc; do ln -s $WT/third_party/$d $WT/.worktrees/pc-lane-a/third_party/$d; done
# same for pc-lane-b / pc-lane-c, each created only after the previous lane is landed
```

- **P inside a lane:** `cd <lane-wt> && PARITY_WT=<lane-wt> PYTHONPATH=<lane-wt> PYTHONUTF8=1 $PY $SP/parity.py --check`. `parity.py` asserts the imported tree is `$PARITY_WT`.

- Inside a lane, substitute `WT=$WT/.worktrees/pc-lane-<x>` in every command above, and re-print the proof line.
  - Without the `third_party` symlinks, guarded tests SKIP instead of failing. So a lane gate must show the baseline **10 skipped**.
- Merge back on `clean/pointcloud-release`: `git cherry-pick <lane commits in order>`, one lane at a time.
  - After **each** pick: run G′, plus P when the pick touches a P path.
  - On a conflict, stop and report. Never `-X theirs`.
- After each lane is landed and gated, remove that lane's worktree and branch:
  ```bash
  git worktree remove .worktrees/pc-lane-<x>
  git branch -D pc-lane-<x>
  ```
  - Check with `git cherry clean/pointcloud-release pc-lane-<x>` first. Every line must be `-`.

### Reviews

Every task gets two reviews, in order:
1. **Spec compliance.** Does the diff do exactly the spec rows it names, and nothing else? Are the stated gate counts met?
2. **Code quality.** Does the diff follow CLAUDE.md style: US spelling, comment-run shape, no dead code? Is the result shorter, not just different?

Only after both reviews pass does the next task start.

### Reduction directive (user, binding)

"make sure that we are reducing, cleaning, and making things concise wherever possible."

- Output-neutral trims ride inside the task that touches the file, tagged `(extra)`.
- Anything that could change output goes under **Proposed additions (need approval)** and is not implemented.
- Round 1 does not polish text that Round 2 deletes. That covers:
  - `save_zarr`/`load_zarr` bodies, `features`
  - `reproject`/`_reproject`, `_rescale_*`, `fit_dominant_plane`, `_source_paths`
  - omega's unused fields and `VGGT_OMEGA_DEFAULT_RESOLUTION`
  - console→logger
  - `_tracked_point3d_ids` (row 65)
  - the `_raw_to_world_points` subsample (row 46)

  A hit inside those gets the minimum edit only.

---

### Task 0: Preflight (no commit)

**Files:** none modified.

- [ ] **Step 1: Branch, base and cleanliness.**
  ```bash
  cd $WT && git branch --show-current && git status --short && git merge-base --is-ancestor 11c5d7c7 HEAD && echo BASE_OK && git diff --stat 11c5d7c7 HEAD -- collab_splats tests configs
  ```
  - Expect `clean/pointcloud-release`, an empty status, `BASE_OK`, and an empty diffstat.
  - Docs-only commits after 11c5d7c7 are allowed. Any code, test or config commit means stop and report.
- [ ] **Step 2: third_party symlinks.**
  ```bash
  cd $WT && ls -l third_party/
  ```
  - Expect `LoGeR`, `Video-Depth-Anything` and `hloc` as symlinks, all resolving.
- [ ] **Step 3: Scratchpad tools exist.**
  ```bash
  cd $WT && ls $SP/parity.py $SP/prose_proof.py $SP/contract_hits.py $SP/stubs/nvdiffrast $SP/baseline_a29.txt $SP/parity_baseline/{depth_align,loger,mapanything,vggt_omega,vggtx}.npz
  ```
- [ ] **Step 4: Proof line.**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -c "import collab_splats;print(collab_splats.__file__)"
  ```
  - Expect `/workspace/collab-splats/.worktrees/pointcloud-release/collab_splats/__init__.py`.
- [ ] **Step 5: Parity.** Run P and expect `SUMMARY PASS`.
- [ ] **Step 6: Contract hits.**
  ```bash
  cd $WT && PYTHONPATH=$WT $PY $SP/contract_hits.py
  ```
  - Expect `TOTAL 54`, split as follows:

    | File | Hits |
    |---|---|
    | ff/base | 7 |
    | loger | 20 |
    | mapanything | 4 |
    | vggt_omega | 3 |
    | vggtx | 3 |
    | sfm/instantsfm | 10 |
    | sfm/sift_db | 2 |
    | utils | 1 |
    | vda | 4 |

- [ ] **Step 7: Baseline.**
  ```bash
  cd $WT && tail -1 $SP/baseline_a29.txt
  ```
  - Expect `18 failed, 1596 passed, 10 skipped, 21 xfailed, 59 xpassed, 52 warnings, 3 errors`.
  - Do **not** re-run G′ here. That file is the recorded baseline at this tree.

### Round 1 bookkeeping

Every Round 1 task leaves `passed` at 1596, because xfail is `strict=False`: a fixed pair moves XFAIL→XPASS.

| After | xfailed / xpassed | contract TOTAL |
|---|---|---|
| T0 | 21 / 59 | 54 |
| R1-1 | 18 / 62 | 47 |
| R1-2 | 10 / 70 | 21 |
| R1-3 | 7 / 73 | 16 |
| R1-4 | 3 / 77 | 4 |
| R1-5 | 3 / 77 | 4 |
| R1-6 | 3 / 77 | 4 |

- The residual 4 are all `numeric-constant` hits, which Round 2 owns:
  - loger `LOGER_CONF_THRESHOLD` :80
  - loger `_PATCH` :87
  - omega `VGGT_OMEGA_DEFAULT_RESOLUTION` :47
  - vggtx `VGGTX_IMG_LOAD_RESOLUTION` :42
- They are the 3 remaining xfail pairs.

> Spec deviation: spec rows 3, 21, 24-33, 69 and 71 have no line table in the spec. The per-hit lists below are their tables at HEAD 11c5d7c7.

---

## Round 1 — prose only

### Task R1-1: feedforward/base.py prose  (spec rows: 24-33 and 48-49 prose parts; Round 1 false-docstring bullet)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`

Pure prose, so steps 1-2 (failing test) are skipped.

- [ ] **Step 3: Apply these replacements** (line numbers are HEAD 11c5d7c7; apply bottom-up so the numbers hold).

**:1-10, module docstring (Provides-style) → summary + bullets:**
```python
"""
Shared types, helpers and the abstract template-method pipeline for feedforward creators.

- FeedforwardResult: typed output of every feedforward backend, with zarr save/load
- multiview depth confidence: compute_multiview_depth_confidence + multiview_mask
- build_pycolmap_reconstruction: pycolmap model from points + cameras
- BaseFeedforwardCreator: the five-step pipeline each backend subclasses
"""
```

**:211-214, mv save comment.** Drop the history bullet:
```python
        # Save mv confidence arrays (N, H, W), chunked by frame
        # - absent entirely when mv was not computed, never a zeros array, which a consumer
        #   cannot tell apart from "every pixel disagreed"
```

**:283 — not touched.** Row 5 deletes the whole `"conf"` fallback in Round 2, and with it the only British spelling in scope ("honour").


**:326-327 (extra).** Delete both lines. They restate `reproject`'s docstring.

**:346-360, `_decode_verify_geometry` docstring** (summary moves off the quote line):
```python
    """
    Decode a verify forward's depth tensors into float32 (world_points, conf).

    - shared by the VGGT-family _verify_loop_candidate paths: LC anchor scale reuses the same forward

    Args:
        depth_t: (1, 2, H, W, 1) depth prediction from the pair forward.
        depth_conf_t: (1, 2, H, W) depth confidence from the pair forward.
        extrinsics_3x4: (2, 3, 4) world-to-cam extrinsics, float32.
        intrinsics: (2, 3, 3) camera intrinsics, float32.

    Returns:
        (2, H, W, 3) points in the pair's local world frame and (2, H, W) confidence.
    """
```

**:374-389, `_raw_to_world_points` docstring.**
> Spec deviation: HEAD text is FALSE. It says "Used by BundleAdjustment for track extraction … subsample=8".

Real callers:
- `vggtx.py:373`, `vggt_omega.py:293` and `loger.py:488` pass `subsample=1` (the dense `world_points` BA field).
- `geometry/loop_closure/wrapper.py:319` keeps the default 8.

```python
    """
    Unproject raw depth to per-frame world points on a pixel grid of stride `subsample`.

    - _postprocess passes subsample=1 for the dense world_points field
    - loop closure keeps the default 8 for submap anchor scale

    Args:
        raw: _forward output with 'depth', 'extrinsic', 'intrinsics_downsampled', optional 'depth_conf'.
        subsample: pixel stride of the grid.

    Returns:
        (K, P, 3) world points and (K, P) confidence (None without depth_conf); (None, None) when a key is absent.
    """
```

**:435-440, `_frustum_world_aabbs`:**
```python
    """
    World-space axis-aligned bounds of each view's depth frustum, (N, 2, 3) [min, max].

    - conservative: an AABB over the 8 frustum corners is a superset of the frustum
    - a truly overlapping pair is never gated out; over-inclusion costs only compute
    """
```

**:469, `_aabbs_overlap`:**
```python
    """
    True when two (2, 3) [min, max] boxes intersect on every axis.
    """
```

**:490-512, `multiview_mask`.** This fixes the hits: banned "measured" at :500 and the history "old threshold=0.0" at :502-503 and :508.

The code at :513-514 is `required = np.minimum(min_views, mv.valid_count)`, and it keeps `inlier >= required & valid_count > 0` where judged, otherwise `valid_depth`.

```python
    """
    Keep-mask from multiview confidence: at least min_views other views agree.

    - a count, not a ratio: the ratio is a quantized k/N that moves with sequence length
    - unjudged views (no overlapping partners) keep their valid pixels
    - K is capped per pixel at valid_count: a pixel with fewer than K partners must satisfy all it has
    - a judged pixel with no partner at all is dropped

    Args:
        mv: output of compute_multiview_depth_confidence.
        valid_depth: (N, H, W) bool, pixels eligible before mv filtering.
        min_views: K in "at least K other views agree".

    Returns:
        (N, H, W) bool keep-mask.
    """
```

**:522-526, `_mv_result_fields`:**
```python
    """
    The three FeedforwardResult mv kwargs, empty when mv was not computed.

    - an empty dict leaves the fields None, so save_zarr omits the arrays rather than writing zeros
    """
```

**:557-559, `compute_multiview_depth_confidence` contract bullet.** Drop the dated bug-class tail:
```python
    - depth, intrinsics and extrinsics are pixel-aligned at ONE resolution, OpenCV convention,
      +Z forward
```

**:563-579, same docstring, Args** (column alignment dropped; `collect` condensed, extra):
```python
    Args:
        depth: (N, H, W) float32 Z-depth per frame.
        intrinsics: (N, 3, 3) float32 pinhole intrinsics in pixel units.
        extrinsics: (N, 4, 4) float32 world-to-cam transforms.
        depth_masks: (N, H, W) bool source pixels to include; None = all valid depth.
        abs_thresh: absolute depth tolerance, depth units; 0.0 for non-metric depth.
        rel_thresh: relative depth tolerance as a fraction of expected depth.
        pair_gate: skip view pairs whose depth frusta cannot overlap; conservative, so output is unchanged.
        collect: optional dict filled IN PLACE with "pairs" (list[PairStats]), "rel_depth_error_counts"
            (int64 histogram) and "rel_depth_error_edges"; the return value is unchanged either way.
        device: torch device for computation.
```

**:641-646, comment run (6 lines, HIT):**
```python
        # Bin count from the exact residual count, no pre-pass
        # - the pair loop is ordered: (i, j) and (j, i) both feed the histogram
        # - so n = N*(N-1)*H*W residuals at most
```

**:653-656 (extra):**
```python
    # O(N^2) pair loop, the dominant cost, so it carries a progress bar
    # - unit is the source frame, each checked against every other
```

**:697-703 (7, HIT):**
```python
            # Sample frame-j depth at the projected pixels, nearest-neighbor
            # - matches upstream mapanything/utils/multiview_confidence.py, whose 0.02 tolerances assume it
            # - bilinear across a depth edge yields a depth on no surface
            # - grid[0, r, c] is where source pixel (r, c) projects
```

**:714-719 (6, HIT):**
```python
            # Disagreement means opposite things by direction
            # - sampled < expected - tol: occluded, no evidence, left out of the denominator
            # - sampled > expected + tol: free-space violation, counted as an outlier
```

**:732-739 (8, HIT):**
```python
            # Signed relative residual over the pixels the ratio counts
            # - occluded pixels are excluded, as in the ratio
            # - near-zero expected depth is dropped: the quotient and the parallax both degenerate
```

**:745-748 (extra).** This drops the per-backbone number:
```python
            # Parallax from ray directions: scale-free, needs no focal length
```

**:775-779 (5, HIT):**
```python
    # ratio is 0 both where no view overlapped and where every view disagreed
    # - judged separates the two for multiview_mask
    # - the ones-denominator only keeps NaN out of the unselected branch
```

**:1054-1083, `BaseFeedforwardCreator` class docstring (8 bullets, HIT).** The per-method contract moves onto the abstract methods (next block):
```python
    """
    Template-method pipeline shared by every feedforward creator.

    - five steps: load_model -> setup_inference -> run_inference -> postprocess -> build_colmap
    - the base owns device choice, CUDA cache release, step state and the COLMAP export
    - subclasses implement the abstract methods; each carries its contract in its docstring

    Attributes:
        camera_model: pycolmap camera model for the export: "PINHOLE" (fx, fy, cx, cy) or
            "SIMPLE_PINHOLE" (f, cx, cy).
    """
```

**:1101-1105, `_lc_collate_outputs` — not touched.** Row 42 deletes the base default in Round 2.


**:1121 (extra).** Delete `# source is the scene's images/ directory, or a legacy eval image dir`. It duplicates Args.

**:1180-1181 (extra).** Delete the two-line comment above `setup_inference`'s docstring. It duplicates the summary.

**:1299-1309, abstract methods.** Add docstrings and **keep each `...` line**. Row 41 later deletes `**kwargs` from `_forward`/`_postprocess`, so that Round 2 commit also drops their `**kwargs:` Args entry:
```python
    @abstractmethod
    def _load_model(self, device: str) -> Any:
        """
        Load the pretrained model onto `device` in eval mode.

        - keep no GPU state outside the returned model

        Args:
            device: torch device string.

        Returns:
            The model.
        """
        ...

    @abstractmethod
    def _preprocess(self, frames: Any, frame_idxs: list[int]) -> tuple[Any, list[Path], np.ndarray]:
        """
        Model input batch from decoded frames.

        Args:
            frames: (N, H, W, 3) uint8, or a list of per-image arrays.
            frame_idxs: source index of each frame.

        Returns:
            (views, image_paths, original_coords): model input batch, frame_{idx:06d} Paths (length N),
            and (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
        """
        ...

    @abstractmethod
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> Any:
        """
        Forward pass under torch.no_grad(); run_inference frees the CUDA cache after.

        Args:
            model: output of _load_model.
            views: output of _preprocess.
            **kwargs: backend-specific forward options.

        Returns:
            Raw outputs in whatever shape _postprocess expects.
        """
        ...

    @abstractmethod
    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """
        Unproject depth, apply confidence filtering and build the result.

        Args:
            raw_outputs: output of _forward.
            **kwargs: backend-specific options.

        Returns:
            The populated FeedforwardResult.
        """
        ...
```

**:1348-1365, `_verify_loop_candidate`.** It takes over the lc_data contract that was deleted from the class docstring:
```python
        """
        Verify a loop closure candidate via the cross-frame attention gate.

        Args:
            frame1: first preprocessed frame, (C, H, W).
            frame2: second preprocessed frame, (C, H, W).
            verify_match_ratio: accept threshold; config value, else the creator's default_verify_match_ratio.
            layer_index: transformer block to tap (-1 = last).
            **kwargs: forwarded to extract_intermediate_features (e.g. minibatch_size=2 for MapAnything).

        Returns:
            (accepted, lc_data): on accept {"poses": (2, 4, 4) w2c float32, "world_points": (2, H, W, 3) | None,
            "conf": (2, H, W) | None}; None on reject or when the backend supplied no poses.
        """
```

- [ ] **Step 4: Targeted run + proof + sanity mutation.**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/test_docstring_contract.py
  cd $WT && $PY $SP/prose_proof.py HEAD collab_splats/pointcloud/feedforward/base.py
  ```
  - `prose_proof` must print `PROSE ONLY`.
  - Sanity mutation, to prove the proof can fail. It works on a scratch copy and restores the file byte-identical:
    ```bash
    cd $WT && F=collab_splats/pointcloud/feedforward/base.py && cp $F $SP/r1-1.bak && echo "_MUTATION = 1" >> $F && $PY $SP/prose_proof.py HEAD $F; cp $SP/r1-1.bak $F && cmp $F $SP/r1-1.bak && $PY $SP/prose_proof.py HEAD $F
    ```
    - Expect `CODE CHANGED: collab_splats/pointcloud/feedforward/base.py`, then silence from `cmp`, then `PROSE ONLY`.
- [ ] **Step 5: Gates.**
  - Contract hits for `feedforward/base.py` → 0, and TOTAL **47**.
  - G′ → 18 failed / 1596 passed / 10 skipped / **18 xfailed / 62 xpassed** / 3 errors.
  - P → `SUMMARY PASS`.
- [ ] **Step 6: Commit.**
  ```bash
  cd $WT && git add collab_splats/pointcloud/feedforward/base.py && git commit --only collab_splats/pointcloud/feedforward/base.py -m "docs(pointcloud): feedforward base docstrings and comment runs to the release contract

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  - Then run `prose_proof.py HEAD~1 collab_splats/pointcloud/feedforward/base.py` → `PROSE ONLY`, and record contract_hits 47.

---

### Task R1-2: backend prose — loger, mapanything, vggt_omega, vggtx  (spec rows: 19, 21, 24-33 prose parts)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/loger.py`
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py`
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`

Pure prose, so steps 1-2 are skipped. Apply bottom-up per file.

The evidence references used below were checked at HEAD:
- `docs/parity.md` :174-187 "Calibration constants that come from VGGT-SPARK": vggtx 10/1.17, omega 13/1.55, mapanything 4/1.46, via a clean-negative sweep.
- `docs/superpowers/specs/2026-08-13-multiview-confidence-measured-report.md`. It is a path, so the banned-word check strips it.

- [ ] **Step 3a: `loger.py`** (20 hits → 2).

**:1-30, module docstring (12 bullets, "measured" ×2):**
```python
"""
LoGeR feedforward backend: patch-aligned resize and creator.

- Pi3 backbone plus a TTT fast-weight memory; sliding-window inference with overlap stitching
- predicts no intrinsics: K is fitted from the camera-frame pointmap
  (geometry.transforms.estimate_intrinsics_from_points)
- vendored and executed: github.com/Junyi42/LoGeR @ 7685b7a (setup/loger.sh -> third_party/LoGeR/)
- prior art, read only, not a dependency: github.com/PolyCam/LoGeR @ 5d7c1a7
- neither fork ships a LICENSE; nothing is copied, and _compute_target_size, whose greedy
  arithmetic matches upstream's, is pinned by a parity test
- citations carry repo + commit + file + line because third_party/ is gitignored
"""
```

**:72-79 (8, HIT, "measured"):**
```python
# Confidence floor for the K fit, below estimate_intrinsics_from_points' 0.1 default
# - LoGeR's conf head is uncalibrated: post-sigmoid values sit in roughly [0.014, 0.117]
# - at 0.1 most pixels fail and a duller scene keeps none; 0.02 rejects only the band floor
# - re-derive if the checkpoint changes
```

**:95-124, `_compute_target_size` docstring** (prose paragraphs, "measured").

It was verified at the pin: `loger/utils/basic.py:11` is the `load_images_as_tensor` signature, :55-61 is the resize rule, and :62-63 is the Target_W/H override.
```python
    """
    Scale to an area budget, then align both axes to whole 14-px patches.

    - must match upstream: the model trains on this preprocessing
      (github.com/Junyi42/LoGeR @ 7685b7a, loger/utils/basic.py:55-61, in load_images_as_tensor at :11)
    - upstream's Target_W/Target_H override (:62-63) is not reimplemented; the size is always computed
    - pinned by test_target_size_matches_the_vendored_loader
    - axes round independently, so aspect stretches a few percent; the PINHOLE K fit absorbs it
    - past ~1300:1 aspect, or a pixel_limit under 196, the result exceeds the budget, as upstream's does

    Args:
        orig_w: source width, px.
        orig_h: source height, px.
        pixel_limit: area budget, px².

    Returns:
        (width, height), both multiples of 14.
    """
```

**:125-129 (5):**
```python
    # Area-budget scale factor, deliberately unguarded against zero area
    # - upstream guards it and falls through to a 14x14 image; not carried over
    # - _preprocess passes frame store dimensions, positive by construction; zero means a corrupt store
    # - a ZeroDivisionError here beats a silent 14x14 tensor the model would consume
```

**:198-201 (history):**
```python
    # Off by default: LoGeR was not in the report's Step D sweep
    # - it inherits the VGGT-family values (rel_thresh is scale-invariant); sweep before trusting them
```

**:203-206 (history "old mv_conf_threshold"):**
```python
    # min_views: "at least K other views agree"
    # - K=1 is inert; K > N-1 empties every judged view, so 2 is the largest count safe on short sequences
```

**:213-218 (6):**
```python
    # se3 unset sentinel, resolved in _load_model from the variant's yaml
    # - declared under model: but consumed as a forward kwarg, so not a constructor kwarg
    # - None means _load_model has not run; False is a valid loaded value, so it cannot be the sentinel
    # - _forward treats None as a contract violation, never defaults it
```

**:235-240 (6):**
```python
        # `or {}` twice: an empty file and a body-less `model:` both parse to None
        # - an empty model_cfg builds Pi3 on constructor defaults, a DIFFERENT architecture
        #   (ttt_inter_multi is 4 in both shipped configs, 2 in the constructor)
        # - that fails 278 state_dict keys later, after a 5 GB download, naming none of this
```

**:251-256 (6):**
```python
        # Vendored tree is not pip-installed, so it is reached via sys.path
        # - the flag keeps the finally idempotent: a nested load must not pop its caller's path
        # - released right after import + construction, before the download/checkpoint load
```

**:300-306 (7):**
```python
        # Frame indices must be strictly increasing
        # - windows and overlap stitching assume temporal order; equal indices collide in frame_{idx:06d}
        # - report the offending pair, not frame_idxs: a 300-frame list buries the bad index
```

**:328-333 (6):**
```python
        # Resize in memory with PIL, not via frames_as_pil_source
        # - frames_as_pil_source patches PIL.Image.open to drive path-based loaders
        # - LoGeR's loader walks a directory with os.listdir, which that patch cannot reach
        #   (github.com/Junyi42/LoGeR @ 7685b7a, loger/utils/basic.py:21)
```

**:335-339 (5, "measured"):**
```python
        # div_ rather than `/ 255.0`: the out-of-place divide holds two float32 copies at once
        # - the second copy is 914 MB at 300 frames of 1080p, on the long-sequence backend
        # - safe in place: `resized` is uint8, so .float() always allocates a fresh tensor
```

**:364-369 (6):**
```python
        # Forward kwargs mirror upstream's build_forward_kwargs
        # - github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:149-164; each popped in Pi3.forward
        #   (github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:584-593)
        # - sim3 stays False: sim3 and se3 are mutually exclusive and raise together (same file, :595-596)
```

**:390 (history "a157421"):**
```python
        # Guard the [0, 1] input range at the source
```

**:401-407 (7, "measured"):**
```python
        # conf_head emits logits, so the sigmoid belongs here
        # - bare LinearPts3d, no output activation (github.com/Junyi42/LoGeR @ 7685b7a, loger/models/pi3.py:172)
        # - upstream activates at its call site too (github.com/PolyCam/LoGeR @ 5d7c1a7, run_loger.py:481)
        # - must precede the K fit: raw logits are all negative, so its conf gate would admit zero pixels
```

**:466-470 (5):**
```python
            # The local K that becomes result.intrinsics, NOT "intrinsics_downsampled"
            # - _forward binds both names to one array, so this is a no-op today
            # - once a real downsampled K lands, the other key would silently disagree with result.intrinsics
```

- [ ] **Step 3b: `mapanything.py`** (4 hits → 0).

**:160-167 (8):**
```python
    # LC verify calibration: layer 4 / 1.46 (clean-negative sweep, see docs/parity.md)
```

**:177-181 (5, history):**
```python
    # K=1 keeps the shipped mask; the VGGT-family creators carry rel=0.01, K=2
    # - evidence: docs/superpowers/specs/2026-08-13-multiview-confidence-measured-report.md
```

**:349-354 (6):**
```python
        # Per-pixel RGB from img_no_norm: [0, 1] at depth resolution
        # - the window's 'img' is ImageNet-normalized, unusable as color
        # - the wrapper prefers these colors over its frame-tensor heuristic
```

**:402-406 (5):**
```python
        # Per-frame masks plus point/color grids in one pass
        # - pred["mask"] already holds non-ambiguous + edge masking
        # - use_multiview_confidence adds the mv keep-mask on top
```

**:1-5, module docstring (extra):**
```python
"""
MapAnything feedforward backend: MapAnythingCreator, metric depth + pose.
"""
```

- [ ] **Step 3c: `vggt_omega.py`** (3 hits → 1; `VGGT_OMEGA_DEFAULT_RESOLUTION` :47 stays for Round 2).

**:1-9, module docstring (Provides):**
```python
"""
VGGT-Omega feedforward backend: center-crop coordinate helper and creator.

- checkpoint defaults: VGGT_OMEGA_HF_REPO / VGGT_OMEGA_DEFAULT_FILENAME (512-res)
- _compute_omega_original_coords maps Omega's center-crop back to source pixels
"""
```

**:131-140 (10, HIT):**
```python
    # LC verify calibration: layer 13 / 1.55 (clean-negative sweep, see docs/parity.md)
    # - token_offset keeps the inherited 5 though Omega has 17 special tokens; 17 scores the same at layer 13
```

**:151-154 (HIT "measured" :152):**
```python
    # Off by default: a complementary tail filter, not a replacement for learned confidence
    # - evidence: docs/superpowers/specs/2026-08-13-multiview-confidence-measured-report.md, Step D
```

**:156-159:**
```python
    # min_views: "at least K other views agree"
    # - K=1 is inert
    # - K > N-1 empties every judged view, so 2 is the largest count safe on short sequences
```

**:164-165:**
```python
    # 0.01, not 0.05: 0.05 removes almost nothing (Step D in the report above)
```

**:232-234 (history):**
```python
        # Decode poses at model resolution only, matching upstream demo_gradio.run_model
```

- [ ] **Step 3d: `vggtx.py`** (3 hits → 1; `VGGTX_IMG_LOAD_RESOLUTION` :42 stays for Round 2).

**:1-7, module docstring (Provides):**
```python
"""
VGGT-X feedforward backend: depth unprojection with confidence filtering, and creator.
"""
```

**:39-41, spec row 19 citation.** Verified at Linketic/VGGT-X @ 26d1b95 (`26d1b956cfda6f83926adda9f5c2c53f75dda749`): `vggt/utils/load_fn.py:211` is `target_size = 518`, the crop resize is at :238-242, and the center crop is at :249-251.
```python
# VGGT-X inference width, px: upstream's crop-mode target_size
# - Linketic/VGGT-X @ 26d1b95, vggt/utils/load_fn.py:211 (target_size = 518)
# - crop mode resizes width to it (:238-242) and center-crops a taller height to it (:249-251)
```

**:120-123, spec row 21 (ROADMAP reference).** Replace all four lines with:
```python
    # Unproject depth to world points in the extrinsics' world frame
```

**:180-185 (6, HIT):**
```python
    # LC verify calibration: layer 10 / 1.17 (clean-negative sweep, see docs/parity.md)
    # - the base layer 20 does not separate loops for VGGT-X
```

**:193-196 (HIT "measured" :194):**
```python
    # Off by default: a complementary tail filter, not a replacement for learned confidence
    # - evidence: docs/superpowers/specs/2026-08-13-multiview-confidence-measured-report.md, Step D
```

**:198-201:**
```python
    # min_views: "at least K other views agree"
    # - K=1 is inert
    # - K > N-1 empties every judged view, so 2 is the largest count safe on short sequences
```

**:206-207:**
```python
    # 0.01, not 0.05: 0.05 removes almost nothing (Step D in the report above)
```

**:303-304 (history):**
```python
        # Decode pose encoding at model resolution; raw["intrinsics"] is model-res K
```

**:323-335, `_postprocess` docstring.**
> Spec deviation: HEAD text is FALSE. There is no global alignment and no semantic-feature lift in this method.

```python
        """
        Unproject depth maps to world-space points and build the FeedforwardResult.

        - optional multiview mask on top of the learned-confidence percentile
        - fills the BA fields: dense world_points, confidence, images

        Args:
            raw_outputs: dict from _forward (depth, depth_conf, extrinsic, intrinsics, images).
            **kwargs: unused.

        Returns:
            FeedforwardResult with points, colors, pixel_indices and BA fields populated.
        """
```

- [ ] **Step 4: Targeted run + proof.**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/test_docstring_contract.py tests/pointcloud/feedforward
  cd $WT && $PY $SP/prose_proof.py HEAD collab_splats/pointcloud/feedforward/{loger,mapanything,vggt_omega,vggtx}.py
  ```
  - `prose_proof` must print `PROSE ONLY`.
- [ ] **Step 5: Gates.**
  - contract_hits TOTAL **21**: loger 2, mapanything 0, omega 1, vggtx 1, plus the untouched files.
  - G′ → 18 / 1596 / 10 / **10 xfailed / 70 xpassed** / 3 errors.
  - P → `SUMMARY PASS`.
- [ ] **Step 6: Commit.**
  ```bash
  cd $WT && git add collab_splats/pointcloud/feedforward/{loger,mapanything,vggt_omega,vggtx}.py && git commit --only collab_splats/pointcloud/feedforward/{loger,mapanything,vggt_omega,vggtx}.py -m "docs(pointcloud): feedforward backend docstrings, citations and comment runs

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  - Then run `prose_proof.py HEAD~1 <same 4 files>` → `PROSE ONLY`, and record contract_hits 21.

---

### Task R1-3: helpers prose — utils, vda, depth_align, base, sfm package  (spec rows: 3, 24-33 prose parts, 71 depth_align bullet; Round 1 false-docstring bullet)

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Modify: `collab_splats/pointcloud/vda.py`
- Modify: `collab_splats/pointcloud/depth_align.py`
- Modify: `collab_splats/pointcloud/base.py`
- Modify: `collab_splats/pointcloud/sfm/__init__.py`

Pure prose, so steps 1-2 are skipped.

- [ ] **Step 3: Replacements.**

**Row 3, path header comments.** Delete line 1 (`# collab_splats/...`) in `sfm/__init__.py` and `depth_align.py` (row 3), and in `utils.py` and `vda.py` (extra, same kind).
- `loger.py:83` matches the same grep but is a sentence, not a header. Leave it.

**`utils.py` :2-10, module docstring.** HEAD omits reprojection and the attention ratio:
```python
"""
Pointcloud utilities: outlier masking, subsampling, plane fit, reprojection, feature lifting.

- also cross_frame_attention_ratio, the loop-closure verify score
- pycolmap input is COLMAP world (Y-down) with OpenCV camera axes (X right, Y down, Z forward)
- reproject_pixels output stays in the caller's frame; nothing here converts to nerfstudio/OpenGL
"""
```

**`utils.py` :61-62 (HIT "measured"):**
```python
    # open3d returns an EMPTY keep set on tiny clouds rather than raising; keep all instead
```

**`utils.py` :275-276 (history):**
```python
    # Whole kernel on the GPU in float32; one .cpu() at the end
```

**`utils.py` :457-459 (history bullets dropped):**
```python
    # Aggregate: mean of top-25% values, port of VGGT-SPARK mean_top_quarter()
```

**`vda.py` :34-40 (7, HIT).** VDA is pinned at `setup.sh:136` (`VDA_COMMIT=4f5ae23…`); `video_depth.py:27` was verified at the pin.
```python
    # VDA lives in a third_party clone (DepthAnything/Video-Depth-Anything @ 4f5ae23), not site-packages
    # - clone root goes on sys.path: video_depth_anything/ sits there
    # - video_depth.py:27 imports a top-level `utils` namespace package from that root
    # - a regular `utils` package elsewhere on sys.path would shadow it
```

**`vda.py` :64-70 (7, HIT, "measured").** Verified: `video_depth.py:135` is `if self.metric:`, and `run.py:45-49` is `model_configs`.
```python
    # metric=True loads the metric head and disables cross-window scale-and-shift chaining
    # - video_depth.py:135: windows stitch on the head's absolute output
    # - constructor values: run.py:45-49 model_configs["vitl"]
```

**`vda.py` :151-155 (5, HIT).** Verified: `video_depth.py:70` is the signature and :162 is the return.
```python
    # Metric inference over the whole sequence at upstream's 518 input resolution
    # - target_fps is only echoed back (video_depth.py:70, :162); any value works
```

**`depth_align.py` :106-107 (history):**
```python
        # Rescale native pixels to the depth grid, then nearest-sample
```

**`depth_align.py` :201-202 (dated bug class):**
```python
    - the depth grid is the result's model resolution: K, images and pixel_indices are scaled to it
```

**`depth_align.py` :207-213, Args.**
> Spec deviation: HEAD says `names` is "in registration order". The code at :218-237 requires `names` to be sorted, all registered, and stem-equal to the sorted registered names.

```python
    Args:
        reconstruction: registered sfm model; its image names are the stems of `names`, in the same order.
        depths: (N, h, w) VDA metric depth.
        images: (N, H, W, 3) uint8 RGB at keyframe resolution.
        names: keyframe filenames, sorted; every one registered (subset to the registered frames first).
        min_obs: minimum valid track-depth pairs a frame needs to get its own scale.
```

**`base.py` :89-105, `PointcloudResult.points`.** Claims verified at `wrapper/reconstructor.py:1167-1171`. The `_tracked_point3d_ids` bullet is dropped here, so row 65's deletion needs no `base.py` edit.
```python
        """
        (P, 3) float32 world XYZ of the reconstruction's 3D points.

        - row order is `reconstruction.points3D` iteration order, NOT sorted by point3D_id
        - pair rows with `list(reconstruction.points3D.keys())`, which iterates in the same order
        - `sorted(reconstruction.points3D)` misaligns the rows and touches the wrong points
        - may hold fewer points than FeedforwardResult.points
        - recomputed on each access, so it reflects the current reconstruction

        Returns:
            (P, 3) float32 array; (0, 3) when the reconstruction has no points.
        """
```

- [ ] **Step 4: Targeted run + proof.**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/test_docstring_contract.py
  cd $WT && $PY $SP/prose_proof.py HEAD collab_splats/pointcloud/{utils,vda,depth_align,base}.py collab_splats/pointcloud/sfm/__init__.py
  ```
  - `prose_proof` must print `PROSE ONLY`.
- [ ] **Step 5: Gates.**
  - contract_hits TOTAL **16** (utils 0, vda 0).
  - G′ → 18 / 1596 / 10 / **7 xfailed / 73 xpassed** / 3 errors.
  - P → `SUMMARY PASS`.
- [ ] **Step 6: Commit.**
  ```bash
  cd $WT && git add collab_splats/pointcloud/{utils,vda,depth_align,base}.py collab_splats/pointcloud/sfm/__init__.py && git commit --only collab_splats/pointcloud/{utils,vda,depth_align,base}.py collab_splats/pointcloud/sfm/__init__.py -m "docs(pointcloud): helper docstrings true to the code, path headers dropped

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  - Then run `prose_proof.py HEAD~1 <same files>` → `PROSE ONLY`, and record contract_hits 16.

---

### Task R1-4: sfm prose — instantsfm, sift_db, `_run_sfm`  (spec rows: 69, 71)

**Files:**
- Modify: `collab_splats/pointcloud/sfm/instantsfm.py`
- Modify: `collab_splats/pointcloud/sfm/sift_db.py`
- Modify: `collab_splats/wrapper/reconstructor.py`

Pure prose, so steps 1-2 are skipped.

All InstantSfM citations were verified at the pin `d3e599e` (`setup.sh:125`, pyproject 0.3.0):
- `track_establishment.py:56`, `:109`
- `scene/defs.py:339`
- `global_positioning.py:229-243`
- `global_mapper.py:25`
- `feature_handler.py:18-56`

- [ ] **Step 3a: `sfm/instantsfm.py`** (10 hits → 0).

**:2 and :4, module docstring.**
> Spec deviation: :2 is FALSE; three backends dispatch now.

```python
InstantSfM global SfM, one of the sfm backends (instantsfm | colmap | hloc).

- drives the upstream python API (cre185/InstantSfM @ d3e599e, 0.3.0), never their CLI
```
Lines :5-7 are unchanged.

**:37-44, `_nudge_edge_keypoints` (scene id GH010229):**
```python
    """
    Pull keypoints sitting exactly on the far image edge (x == width or y == height) inward.

    - upstream sample_depth_at_pixel rejects x / width > 1 but lets == 1 through, then indexes
      depth_map[:, W] -> IndexError; SIFT emits such keypoints rarely
    - coords beyond the edge are left alone, so upstream still marks them depth-unavailable
    """
```

**:226-244, class docstring (8 bullets, HIT).** Field bullets fold into `Attributes:`:
```python
    """
    Global SfM via InstantSfM's python API on a scene directory.

    - upstream github.com/cre185/InstantSfM @ d3e599e (0.3.0, setup.sh); call pattern follows
      instantsfm/scripts/sfm.py::run_sfm
    - reconstruct(data_dir): images/ (+ depth_vda/) -> pycolmap.Reconstruction with stem image names;
      model at colmap/sparse/0, SIFT DB at colmap/instantsfm.db
    - not a BasePointcloudCreator: a scene dir in, a Reconstruction out; the Reconstructor wraps it
    - license CC-BY-NC-4.0 (non-commercial): cleared for research use, revisit before commercial use

    Attributes:
        use_depths: feed depth_vda/ maps into the solve as depth priors.
        retriangulation: GLOMAP-style retriangulate + re-BA after the global solve.
        random_seed: seeds RUNTIME_OPTIONS; None keeps upstream's unseeded init, so runs differ.
        min_num_view_per_track: track-establishment cut; None keeps upstream's 3.
    """
```

**:275-281 (7, HIT):**
```python
        # InstantSfM is nondeterministic unless random_seed is set
        # - unseeded init: cre185/InstantSfM @ d3e599e, instantsfm/processors/global_positioning.py:229-243
        # - random_seed seeds numpy/random/torch/cuda in SolveGlobalMapper (instantsfm/controllers/global_mapper.py:25)
        # - unset by default, here and in upstream's CLI
```

**:285-292 (8, HIT, "measured"):**
```python
        # Track-establishment cut, the memory lever for large image sets
        # - drops tracks with fewer views: cre185/InstantSfM @ d3e599e, instantsfm/processors/track_establishment.py:109
        # - BA normal equations scale with the surviving track count
        # - retriangulation rebuilds from the pre-filter set (TRIANGULATOR_OPTIONS min 2), unaffected
```

**:326-330 (5):**
```python
        # Upstream compat fixes, each documented at its _patch_* definition
        # - track ids vs int32 storage; COLMAP writer output pycolmap cannot read
        # - bae LM.step vs pypose RobustModel.forward(target); PCG 1-D step vs TrustRegion.update
```

**:336-341 (6):**
```python
        # Point ReadData's image dir straight at the keyframe store
        # - ReadData derives every path from data_dir; the keyframe store is already a COLMAP image dir
        # - same PathInfo override as the database and output paths below; no staged JPEG copy
```

**:351-360 (10):**
```python
        # Redirect upstream's flat data_dir/{database.db,sparse} into colmap/
        # - instantsfm.db, not database.db: the verify stage unlinks and rewrites colmap/database.db
        # - the whole stale sparse/ is removed so a leftover cluster dir (sparse/1) cannot trip the warning below
```

**:368-372 (5):**
```python
        # SIFT database: reuse a complete one extracted from THIS image set
        # - makes re-runs idempotent; a partial or stale-selection DB is rebuilt
        # - build failures raise RuntimeError
```

**:393-397 (5):**
```python
        # Global mapping; upstream raises a raw IndexError on several failure paths
        # - every track filtered (numpy-2 empty mask in scene/defs.py filter_by_mask), or no depth priors loaded
        # - log the traceback and re-raise with a pointer to the chained cause
```

- [ ] **Step 3b: `sfm/sift_db.py`** (2 hits → 0).

**:5.**
> Spec deviation: HEAD is FALSE. `hloc.py:18` imports `rename_images_to_stems` and `sfm_image_dir`.

```python
- used by all three sfm creators; not re-exported from sfm/__init__.py
```

**:34 (dated):**
```python
# - both URLs serve the same file; a colmap 3.10 binary loads it
```

**:68-88, `sift_database_valid`.** Bullets move above Args:
```python
    """
    True when the SIFT DB holds extraction and matching output for exactly `image_names`.

    - a crashed colmap run leaves a partial DB; an existence-only check would feed the mapper zero tracks
    - Database.open CREATES a missing file, so the exists() pre-check stays; it raises RuntimeError
      on a file no registered factory can open
    - nothing stages a per-run image copy, so the DB's own image table names the selection it holds

    Args:
        database_path: the scene's SIFT database (colmap/instantsfm.db or colmap/colmap.db).
        image_names: filenames this run is about to reconstruct, in any order.
        params: matching params the DB must have been built with, compared against the
            colmap.db.json sidecar; None skips the check (instantsfm).

    Returns:
        False when the DB is missing, partial, or keyed on a different image set.
    """
```

**:129-151, `build_sift_database` ("measured" ×2, timings, host core count).**
> Spec deviation: the upstream file is 56 lines, so the cite is `:18-56`, not the spec's `:18-57`.

```python
    """
    Build the COLMAP SIFT feature database: extraction + matching under one pairing mode.

    - drives the colmap CLI, not pycolmap: the system binary is the CUDA build, the wheel CPU-only
    - reimplements upstream GenerateDatabase (cre185/InstantSfM @ d3e599e,
      instantsfm/controllers/feature_handler.py:18-56), which forces CPU with no thread cap and
      swallows CalledProcessError, so a colmap crash surfaces only as a later empty-tracks IndexError
    - on failure the partial DB is unlinked so a re-run rebuilds from scratch
    - sequential sets quadratic_overlap 0 so "sequential" means i with i+1..i+N, as hloc's generator does

    Args:
        image_path: directory of images to extract from.
        database_path: SIFT database to write.
        pairing: which image pairs are matched; one of PAIRINGS.
        overlap: sequential neighbors matched per image.
        num_retrieved: vocab-tree neighbors retrieved per image.
        vocab_tree: vocab tree file; required when pairing retrieves (see fetch_vocab_tree).
        num_threads: CPU SIFT thread cap; colmap's default is one thread per host core, whose
            per-thread RAM can exceed the container cap.
    """
```

**:163-166 (history dropped):**
```python
    # One shared SIMPLE_RADIAL camera over the whole set
    # - the sfm path stages frames from a single video/scene, so single_camera holds by construction
    # - per-image cameras would leave every intrinsic solved from one view
```

**:182 (history dropped):**
```python
    # Matcher per pairing; the GPU flag stays right after the DB path
```

- [ ] **Step 3c: `wrapper/reconstructor.py` `_run_sfm` (:1198-1297).**

> Spec deviation: the spec's :1212-1216 / :1218-1222 / :1228-1236 are :1214-1218 / :1222-1225 / :1229-1237 at HEAD.

**:1199-1207, docstring** (Returns bullet becomes a section):
```python
        """
        SfM pointcloud path: staged keyframes -> VDA metric depth -> the configured sfm mapper.

        - reads the scene's images/ in place; nothing is staged
        - caches depth_vda/images/npy/<stem>.npy across runs, runs the sfm creator, then builds the
          dense result rescaled to the COLMAP world and writes pointcloud.zarr with provenance

        Returns:
            The PointcloudResult for the shared tail.
        """
```

**:1214-1218:**
```python
        # The scene's images/ is already the COLMAP image layout, so it is read in place
        # - no JPEG copy is staged; the SIFT DB keys itself on the image set
        # - `names` come from the directory; only their stems reach VDA (depth_vda/images/npy/<stem>.npy)
```

**:1222-1225:**
```python
        # VDA depth is cached by stem; on a gate miss drop depth_vda/ first
        # - generate_vda_depth only adds maps, so a stale stem would keep the gate false forever
```

**:1229-1237 (9 lines, sizes, "875-frame", "left out of this commit"):**
```python
        # Keyframe stack built twice on purpose, freed across the mapper
        # - holding the full-res stack across creator.reconstruct() stacks it on the mapper's peak
        # - the second decode after the solve is the cost; on a depth-cache hit the first build is unused
```

**:1242-1244 (history):**
```python
        # Mapper per backend; writes colmap/<db> + colmap/sparse/0 with stem image names
        # - instantsfm: its three knobs
        # - colmap / hloc: the whole block but min_registered_frac, the floor applied below
```

**:1278 (history):**
```python
        # Provenance per backend
```
Line :1279 is unchanged.

- [ ] **Step 4: Targeted run + proof.**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/test_docstring_contract.py tests/pointcloud/sfm
  cd $WT && $PY $SP/prose_proof.py HEAD collab_splats/pointcloud/sfm/instantsfm.py collab_splats/pointcloud/sfm/sift_db.py collab_splats/wrapper/reconstructor.py
  ```
  - `prose_proof` must print `PROSE ONLY`.
- [ ] **Step 5: Gates.**
  - contract_hits TOTAL **4**, the numeric-constant residue listed in the bookkeeping table.
  - G′ → 18 / 1596 / 10 / **3 xfailed / 77 xpassed** / 3 errors.
  - P is not required (no P path).
- [ ] **Step 6: Commit.**
  ```bash
  cd $WT && git add collab_splats/pointcloud/sfm/instantsfm.py collab_splats/pointcloud/sfm/sift_db.py collab_splats/wrapper/reconstructor.py && git commit --only collab_splats/pointcloud/sfm/instantsfm.py collab_splats/pointcloud/sfm/sift_db.py collab_splats/wrapper/reconstructor.py -m "docs(pointcloud): sfm docstrings and comment runs; pinned InstantSfM citations

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  - Then run `prose_proof.py HEAD~1 <same 3 files>` → `PROSE ONLY`, and record contract_hits 4.

---

### Task R1-5: bundle_adjustment stale reproject pointer  (spec rows: none; Round 1 false-docstring bullet)

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py`

Pure prose, so steps 1-2 are skipped. `creator.reproject(result)` does not exist; the method is `FeedforwardResult.reproject()` (`feedforward/base.py:~316`).

- [ ] **Step 3:**
  - :96 →
    ```python
        - does not reproject points; call result.reproject() afterwards if needed
    ```
  - :218 →
    ```python
            - points, colors and pixel_indices are unchanged; call result.reproject() after
    ```
- [ ] **Step 4: Proof.**
  ```bash
  cd $WT && $PY $SP/prose_proof.py HEAD collab_splats/geometry/bundle_adjustment.py
  ```
  - Must print `PROSE ONLY`. Then run the targeted `tests/test_docstring_contract.py`.
- [ ] **Step 5: Gates.**
  - geometry contract hits stay 0; the pointcloud TOTAL stays **4**.
  - G′ → 18 / 1596 / 10 / 3 / 77 / 3.
  - P is not required. The edit is docstring-only, and `prose_proof` proves no transforms code moved.
- [ ] **Step 6: Commit.**
  ```bash
  cd $WT && git add collab_splats/geometry/bundle_adjustment.py && git commit --only collab_splats/geometry/bundle_adjustment.py -m "docs(geometry): bundle adjustment points at result.reproject()

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task R1-6: sfm config comments + API page  (spec rows: none; Round 1 `configs/base.yaml` and `pointcloud.rst` bullets)

**Files:**
- Modify: `configs/base.yaml`
- Modify: `docs/source/api/pointcloud.rst`

Pure prose, so steps 1-2 are skipped.

> Spec deviation: the spec asks for the `:78-89` instantsfm block as a header plus bullets. Per-key inline comments are used instead, matching the colmap and hloc blocks beside it, and they are shorter.

- [ ] **Step 3a: `configs/base.yaml` :76-107.** Replace the block with the following. Keys, values and order are identical; only comments change.
```yaml
  # InstantSfM global SfM (method: sfm, backend: instantsfm)
  instantsfm:
    retriangulation: false        # GLOMAP-style retriangulate + up to 5 further BA rounds; extra runtime
    random_seed: null             # seeds InstantSfM's RUNTIME_OPTIONS; null = upstream's unseeded init
    min_num_view_per_track: null  # drop thinner tracks before BA, the memory lever; null = upstream's 3
  # COLMAP incremental SfM (method: sfm, backend: colmap): colmap CLI SIFT + pycolmap mapper.
  colmap:
    pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
    overlap: 10              # sequential: pair each frame with the next N
    num_retrieved: 20        # retrieval: vocab-tree neighbors per image (15 MB tree, fetched once)
    num_threads: 8           # CPU SIFT/mapper thread cap; colmap's default is one thread per host core
    min_registered_frac: 0.5 # fail below this share of frames registered; above it, subset
  # hloc incremental SfM (method: sfm, backend: hloc): learned features + pycolmap mapper.
  # Needs the optional `hloc` extra (setup/hloc.sh, then the user-run uv lock + sync).
  hloc:
    pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
    overlap: 10              # sequential: pair each frame with the next N
    num_retrieved: 20        # retrieval: top-k global-descriptor neighbors per image
    retrieval_conf: netvlad  # hloc.extract_features.confs key
    feature_conf: superpoint_max          # hloc.extract_features.confs key
    matcher_conf: superpoint+lightglue    # hloc.match_features.confs key
    num_threads: 8           # pycolmap mapper thread cap
    min_registered_frac: 0.5 # fail below this share of frames registered; above it, subset
```

- [ ] **Step 3b: `docs/source/api/pointcloud.rst`.** Insert after the instantsfm automodule (:32-34):
```rst
.. automodule:: collab_splats.pointcloud.sfm.sift_db
   :members:
   :show-inheritance:
```

- [ ] **Step 4: Proof (yaml is not covered by prose_proof).**
  ```bash
  cd $WT && diff <(git show HEAD:configs/base.yaml | sed 's/[[:space:]]*#.*//' | grep -v '^[[:space:]]*$') <(sed 's/[[:space:]]*#.*//' configs/base.yaml | grep -v '^[[:space:]]*$') && echo KEYS_SAME
  cd $WT && $PY -c "import yaml,subprocess;o=yaml.safe_load(subprocess.run(['git','show','HEAD:configs/base.yaml'],capture_output=True,text=True).stdout);n=yaml.safe_load(open('configs/base.yaml'));print('YAML EQUAL' if o==n else 'YAML DIFF')"
  ```
  - Expect `KEYS_SAME` and `YAML EQUAL`.
- [ ] **Step 5: Gates.** G′ → 18 / 1596 / 10 / 3 / 77 / 3, which covers the config-loading tests in `tests/wrapper`.
- [ ] **Step 6: Commit.**
  ```bash
  cd $WT && git add configs/base.yaml docs/source/api/pointcloud.rst && git commit --only configs/base.yaml docs/source/api/pointcloud.rst -m "docs(pointcloud): sfm config comments, sift_db on the API page

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

## Round 2 — cross-lane helpers H1-H4, then lane A

## Lane A baseline and conventions

- **Baseline is post-Round-1.** R1-1..R1-6 from part0 land first.
  - G′ baseline: **18 failed / 1596 passed / 10 skipped / 3 xfailed / 77 xpassed / 3 errors**.
- **R1 shifts comment lines.** Line numbers below are HEAD numbers, each with anchor text.
  - Lines inside comments/docstrings that R1-3/R1-4 rewrote may have moved.
  - Always locate the anchor text, not the bare number.
- **R1 already covers some spec items.** No lane A step repeats them:
  - base.py:97 `_tracked_point3d_ids` bullet: R1-3 deletes it.
  - instantsfm.py:2 docstring: R1-4 fixes it.
  - The instantsfm comments at :336-341/:351-360/:368-372: R1-4.
  - sift_db :5/:34 and the `sift_database_valid` docstring: R1-4.
- Shell variables used below:
  - `WT=/workspace/collab-splats/.worktrees/pointcloud-release`
  - `SP=/tmp/claude-0/-workspace-collab-splats/e908d41b-1068-4caa-9c92-045ca0ee282c/scratchpad/pc-release`
  - `PY=/opt/venv/reconstruction/bin/python`
- **G′ command (identical everywhere):**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -c "import collab_splats;print(collab_splats.__file__)" && cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors tests/pointcloud tests/geometry tests/wrapper tests/evals tests/localization tests/remote tests/test_docstring_contract.py
  ```
  - The printed path must be under `$WT`.
- **P command:**
  ```bash
  cd $WT && PYTHONPATH=$WT PYTHONUTF8=1 $PY $SP/parity.py --check
  ```
  - It must print `SUMMARY PASS`.
  - On failure: `cd $WT && git revert --no-edit HEAD`, stop, report.
- **Targeted-test command:** `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider <nodeid>`
- **Contract check:** `cd $WT && PYTHONPATH=$WT $PY $SP/contract_hits.py <files>` must print `TOTAL 0` for every file touched.
- **Commit template:**
  ```bash
  cd $WT && git add <paths> && git commit --only <paths> -m "<type>(<scope>): <subject>

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  - Use `git rm` for deletions.
  - Never amend, rebase, reset, or bare stash.
  - Never pipe pytest to `tail`; never use `--tb=no`.
- **Contract test accounting:**
  - A new pointcloud source file adds **+3 passed** (per-file base checks).
  - It also adds **+5 xpassed**: the release checks are `xfail(strict=False)` while pointcloud is unreleased.
  - preproc is RELEASED, so `preproc/frames.py` must stay contract-clean (`TOTAL 0`).
- `tests/utils` and `tests/preproc` are outside G′. Their tests are run as targeted runs only.

### Lane assignment

- Follows the spec: lane A owns `utils.py`, `vda.py`, `depth_align.py`, `sfm/*` (+ `sfm/common.py`), their tests and evals row 84. Lane C drafted no copy of rows 87-99/110-115; its C-3..C-8 cover rows 100-111 and 116 only.
- **H1-H4 land on `clean/pointcloud-release` directly, before `pc-lane-a` is forked.** Lane A tasks start at A-1 in `pc-lane-a`; inside the lane, read `$WT` as the lane worktree and run P with `PARITY_WT=$WT`.

### Helper signatures (lane B relies on these)

| Helper | Location | Signature |
|---|---|---|
| H1 | `collab_splats/utils/torch_utils.py` | `to_numpy(x: torch.Tensor \| np.ndarray) -> np.ndarray` |
| H2 | `collab_splats/utils/torch_utils.py` | `vendored_path(root: Path, hint: str) -> Iterator[None]` (context manager; `ImportError(f"{root} not found — {hint}")`) |
| H3 | `collab_splats/preproc/frames.py` | `frame_name(idx: int) -> str` → `"frame_{idx:06d}"` |
| H4 | `collab_splats/pointcloud/feedforward/base.py` | `full_frame_coords(width: int, height: int, n: int) -> np.ndarray` → (n, 6) float32 |

- Lane B assumed the H2 message is exactly `hint`.
- The actual message is `f"{root} not found — {hint}"`.
- Any lane B `match=` must match a substring of the hint, not the whole message.

---

### Task H1: `to_numpy` helper  (spec rows: cross-lane helper)

**Files:**
- Modify: `collab_splats/utils/torch_utils.py:1-12` (imports), plus a new function after `pytorch_gc` (:20-25) and before the Batching divider (:28)
- Test: `tests/utils/test_torch_utils.py`

- [ ] **Step 1: Write the failing test**

  Change the import line in `tests/utils/test_torch_utils.py` from
  `from collab_splats.utils.torch_utils import batch_iterator, infer_batch_size, pytorch_gc` to:
  ```python
  from collab_splats.utils.torch_utils import batch_iterator, infer_batch_size, pytorch_gc, to_numpy
  ```
  Add `import numpy as np` and `import torch` at the top if they are absent. Append:
  ```python
  def test_to_numpy_casts_bf16_tensor_to_float32():
      out = to_numpy(torch.ones(2, 3, dtype=torch.bfloat16))
      assert isinstance(out, np.ndarray)
      assert out.dtype == np.float32
      assert out.shape == (2, 3)


  def test_to_numpy_passes_ndarray_through():
      arr = np.arange(4, dtype=np.int16)
      assert to_numpy(arr) is arr
  ```

- [ ] **Step 2: Run it, expect FAIL**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/utils/test_torch_utils.py
  ```
  - Expected: collection error `ImportError: cannot import name 'to_numpy'`.

- [ ] **Step 3: Implement**

  Add `import numpy as np` among the third-party imports. Add after `pytorch_gc`:
  ```python
  def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
      """
      Host numpy view of a tensor or array.

      - bf16 has no numpy dtype: cast to float32 first
      - ndarrays pass through uncopied

      Args:
          x: tensor on any device, or an ndarray.

      Returns:
          The data as an ndarray.
      """
      if isinstance(x, torch.Tensor):
          if x.dtype == torch.bfloat16:
              x = x.float()
          return x.detach().cpu().numpy()
      return np.asarray(x)
  ```

- [ ] **Step 4: Run targeted tests, expect PASS** (same command as Step 2).

- [ ] **Step 5: Gates**
  - G′: no change (`tests/utils` is outside G′). Stays at 18 / 1596 / 10 / 3 / 77 / 3.
  - P: required (torch_utils).

- [ ] **Step 6: Commit**
  ```bash
  cd $WT && git add collab_splats/utils/torch_utils.py tests/utils/test_torch_utils.py && git commit --only collab_splats/utils/torch_utils.py tests/utils/test_torch_utils.py -m "feat(utils): to_numpy tensor/array host helper

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task H2: `vendored_path` context manager  (spec rows: cross-lane helper; consumed by row 74)

**Files:**
- Modify: `collab_splats/utils/torch_utils.py` (imports, plus a new function after `to_numpy`)
- Test: `tests/utils/test_torch_utils.py`

- [ ] **Step 1: Write the failing test**

  Extend the import to `..., pytorch_gc, to_numpy, vendored_path`. Add `import sys` and `import pytest` at the top if absent. Append:
  ```python
  def test_vendored_path_inserts_then_removes(tmp_path):
      with vendored_path(tmp_path, "hint"):
          assert sys.path[0] == str(tmp_path)
      assert str(tmp_path) not in sys.path


  def test_vendored_path_keeps_a_preexisting_entry(tmp_path, monkeypatch):
      monkeypatch.syspath_prepend(str(tmp_path))
      with vendored_path(tmp_path, "hint"):
          pass
      assert str(tmp_path) in sys.path


  def test_vendored_path_missing_root_names_the_fix(tmp_path):
      with pytest.raises(ImportError, match="run setup.sh"):
          with vendored_path(tmp_path / "absent", "run setup.sh"):
              pass
  ```

- [ ] **Step 2: Run it, expect FAIL**
  - Same command as H1.
  - Expected: `ImportError: cannot import name 'vendored_path'`.

- [ ] **Step 3: Implement**

  New imports: `import sys`, `from contextlib import contextmanager`, `from pathlib import Path`. Extend `typing` with `Iterator`. Add:
  ```python
  @contextmanager
  def vendored_path(root: Path, hint: str) -> Iterator[None]:
      """
      Vendored checkout placed first on sys.path for the block.

      - missing checkout: ImportError naming the fix, before any import runs
      - exit removes the entry only if this call inserted it

      Args:
          root: checkout directory to import from.
          hint: how to obtain the checkout, appended to the error.

      Yields:
          Nothing; imports inside the block resolve against root.
      """
      if not root.is_dir():
          raise ImportError(f"{root} not found — {hint}")

      # Insert once; leave a caller's pre-existing entry alone
      entry = str(root)
      inserted = entry not in sys.path
      if inserted:
          sys.path.insert(0, entry)

      try:
          yield
      finally:
          if inserted and entry in sys.path:
              sys.path.remove(entry)
  ```

- [ ] **Step 4: Run targeted tests, expect PASS.**

- [ ] **Step 5: Gates**
  - G′: no change.
  - P: required.

- [ ] **Step 6: Commit** `feat(utils): vendored_path context manager for third_party imports`, same two paths as H1.

---

### Task H3: `frame_name` helper  (spec rows: cross-lane helper)

**Files:**
- Modify: `collab_splats/preproc/frames.py`. Insert a new function before `frame_idx_from_path` (:64). Change :133 `path = dir / f"frame_{int(record['frame_idx']):06d}.png"`.
- Test: `tests/preproc/test_frames.py`

- [ ] **Step 1: Write the failing test** (append; the module is already imported as `fr`)
  ```python
  def test_frame_name_round_trips_through_frame_idx_from_path():
      assert fr.frame_name(7) == "frame_000007"
      assert fr.frame_idx_from_path(fr.frame_name(42) + ".png") == 42
  ```

- [ ] **Step 2: Run it, expect FAIL**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/preproc/test_frames.py::test_frame_name_round_trips_through_frame_idx_from_path
  ```
  - Expected: `AttributeError: module 'collab_splats.preproc.frames' has no attribute 'frame_name'`.

- [ ] **Step 3: Implement**
  ```python
  def frame_name(idx: int) -> str:
      """
      Stem of a keyframe file in the images/ store.

      Args:
          idx: source-video frame index.

      Returns:
          frame_ plus the index zero-padded to six digits.
      """
      return f"frame_{int(idx):06d}"
  ```
  - :133 before: `path = dir / f"frame_{int(record['frame_idx']):06d}.png"`
  - :133 after: `path = dir / f"{frame_name(record['frame_idx'])}.png"`

- [ ] **Step 4: Run targeted tests, expect PASS**
  - Run the whole file: `... tests/preproc/test_frames.py`.
  - Also run `tests/test_docstring_contract.py -k preproc`. It is released, so every check must pass.
  - Then `contract_hits.py collab_splats/preproc/frames.py` → `TOTAL 0`.
  - If the numeric-constant check flags `06d`: move the width into the docstring bullet, not a module constant, and rerun.

- [ ] **Step 5: Gates**
  - G′: 0 (tests/preproc outside G′; contract file-level counts unchanged).
  - P: not required.

- [ ] **Step 6: Commit** `feat(preproc): frame_name keyframe stem helper`, with frames.py and test_frames.py.

---

### Task H4: `full_frame_coords` helper  (spec rows: cross-lane helper; consumed by row 64)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`. Insert before the `_STORE_FRAME_NAME` comment block (:1008).
- Test: `tests/pointcloud/feedforward/test_full_frame_coords.py` (new)

- [ ] **Step 1: Write the failing test**
  ```python
  """Tests for full_frame_coords (uncropped crop-box rows)."""

  import numpy as np

  from collab_splats.pointcloud.feedforward.base import full_frame_coords


  def test_full_frame_coords_one_row_per_frame():
      out = full_frame_coords(640, 480, 3)
      assert out.dtype == np.float32
      assert out.shape == (3, 6)
      np.testing.assert_array_equal(out, np.array([[0, 0, 640, 480, 640, 480]] * 3, dtype=np.float32))
  ```

- [ ] **Step 2: Run it, expect FAIL**
  - Expected: `ImportError: cannot import name 'full_frame_coords'`.

- [ ] **Step 3: Implement**
  ```python
  def full_frame_coords(width: int, height: int, n: int) -> np.ndarray:
      """
      Crop-box rows for frames used uncropped at original size.

      - row layout: x0, y0, x1, y1, width, height

      Args:
          width: original image width in pixels.
          height: original image height in pixels.
          n: number of frames.

      Returns:
          (n, 6) float32, one identical row per frame.
      """
      return np.tile(np.array([0, 0, width, height, width, height], dtype=np.float32), (n, 1))
  ```

- [ ] **Step 4: Run targeted tests, expect PASS.**

- [ ] **Step 5: Gates**
  - G′: +1 passed → **1597**.
  - P: required.
  - `contract_hits.py collab_splats/pointcloud/feedforward/base.py`: the count must not rise.

- [ ] **Step 6: Commit** `feat(pointcloud): full_frame_coords helper for uncropped frames`.

---

## Lane A

### Task A-1: shared SfM test fixtures module  (spec rows: 114)

> Spec deviation: row 114 says `conftest.py`. The helpers go in `tests/pointcloud/_stubs.py` instead.
> - `test_depth_align._scene_inputs` builds a recon at module/helper level, outside any fixture.
> - A conftest fixture cannot serve that call site.
> - Precedent: `tests/wrapper/_stubs.py`, imported as `from tests.wrapper._stubs import ...`.
> - `tests/__init__.py` and `tests/pointcloud/__init__.py` exist, so the import resolves.

**Files:**
- Create: `tests/pointcloud/_stubs.py`
- Modify:
  - `tests/pointcloud/sfm/test_colmap.py:19-50`: delete `_recon` and `_scene`.
  - `tests/pointcloud/sfm/test_hloc.py` `_recon` (:34) and `_scene` (:135): delete.
  - `tests/pointcloud/sfm/test_sift_db.py:190-205`: delete `_recon`.
  - `tests/pointcloud/test_depth_align.py:236-251`: delete `_pycolmap_scene`; update the uses at :259 and :340.

- [ ] **Step 3: Implement** (a pure merge, so Steps 1-2 are skipped)
  ```python
  """
  Shared pycolmap/frame-store builders for pointcloud tests.
  """

  from pathlib import Path

  import numpy as np
  import pycolmap

  from collab_splats.preproc import frames as fr


  def make_recon(names: list[str], cam_w: int = 64, cam_h: int = 48) -> pycolmap.Reconstruction:
      """
      One PINHOLE camera, one image per name, one point3D observed in every image.
      """
      recon = pycolmap.Reconstruction()
      cam = pycolmap.Camera(model="PINHOLE", width=cam_w, height=cam_h, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
      recon.add_camera_with_trivial_rig(cam)
      track = pycolmap.Track()
      for i, name in enumerate(names):
          im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
          im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
          pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
          recon.add_image_with_trivial_frame(im, pose)
          track.add_element(i + 1, 0)
      recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
      return recon


  def make_scene(tmp_path: Path, names: list[str]) -> tuple[Path, Path]:
      """
      A backend data dir plus an images/ store holding names as real PNG keyframes.
      """
      images_dir = tmp_path / "images"
      fr.write_frames(
          images_dir,
          [np.zeros((8, 8, 3), np.uint8)] * len(names),
          [{"frame_idx": int(n[6:12]), "blur_score": 1.0} for n in names],
          {"video_path": "x.mp4", "method": "uniform"},
      )
      return tmp_path / "backend", images_dir
  ```

  Call-site changes:
  - test_colmap: `_recon(x)` → `make_recon(x)`; `_scene(tmp_path)` → `make_scene(tmp_path, NAMES)`.
  - test_hloc: the same.
  - Diff test_hloc's `_recon`/`_scene` bodies against test_colmap's first.
    - Where a body differs (for example the data-dir name), pass the difference explicitly.
    - Do not generalize `make_scene`.
    - A test that asserts on the old data-dir name `colmap_backend` must be updated to `backend`.
  - test_sift_db rename test: `make_recon([...])`.
  - test_depth_align `_pycolmap_scene(...)` → `make_recon(names, cam_w=128, cam_h=96)` at :259 and :340. The K params are identical (`[50, 50, 32, 24]`).
  - Add `from tests.pointcloud._stubs import make_recon, make_scene` to each file. Drop imports left unused (`fr` in test_colmap/test_hloc if only `_scene` used it).

- [ ] **Step 4: Run targeted tests, expect PASS** (same counts as before)
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/sfm tests/pointcloud/test_depth_align.py
  ```

- [ ] **Step 5: Gates**
  - G′: 0 (1597).
  - `_stubs.py` is under tests/, so the contract counts are unchanged.

- [ ] **Step 6: Commit** `test(pointcloud): shared make_recon/make_scene builders for sfm tests`, with all five files.

---

### Task A-2: `sfm/common.py`, moving `sfm_image_dir` and `rename_images_to_stems`  (spec rows: 90, 112)

**Files:**
- Create: `collab_splats/pointcloud/sfm/common.py`
- Modify: `collab_splats/pointcloud/sfm/sift_db.py`
  - Delete the Image directory divider and `sfm_image_dir` (:40-59).
  - Delete the Output contract divider and `rename_images_to_stems` (:333-352).
  - Remove the module-docstring output-contract bullet.
- Modify: the colmap/hloc/instantsfm imports, from `.sift_db import ... sfm_image_dir, rename_images_to_stems` to `from .common import ...`.
- Test:
  - Create `tests/pointcloud/sfm/test_common.py`.
  - Move test_sift_db :207-217 (rename) and :236-241 (missing dir) into it.
  - Delete test_sift_db :225-233 `test_sfm_points_at_the_scene_images_dir_and_stages_nothing` (row 112).

- [ ] **Step 3: Implement** (a move, so Steps 1-2 are skipped)
  ```python
  """
  Directory layout and model output shared by every sfm creator.

  - input: the scene images/ store, read in place
  - output: <data_dir>/colmap/sparse/0, image names as frame stems
  """

  import logging
  from pathlib import Path

  import pycolmap

  logger = logging.getLogger(__name__)


  ########################################
  # Image directory
  ########################################


  def sfm_image_dir(images_dir: Path) -> Path:
      """
      Image directory an sfm creator reads.

      Args:
          images_dir: the scene's images/ directory.

      Returns:
          The same directory; the store is the COLMAP image layout.
      """
      images_dir = Path(images_dir)
      if not images_dir.is_dir():
          raise FileNotFoundError(f"sfm expects an image directory at {images_dir} — run preprocess first")
      return images_dir


  ########################################
  # Output contract
  ########################################


  def rename_images_to_stems(recon: pycolmap.Reconstruction, sparse_dir: Path) -> None:
      """
      COLMAP image names cut to filename stems, model rewritten in place.

      - mappers register frame_000000.png; contract is frame_000000
      - pycolmap.Image.name is settable by reference

      Args:
          recon: model whose images are renamed in place.
          sparse_dir: directory the renamed binary model is written to.
      """
      for im in recon.images.values():
          im.name = Path(im.name).stem
      recon.write_binary(str(sparse_dir))
  ```
  - The two docstrings are shortened to fragments. This is an output-neutral prose trim, needed for the release bullet-cap/comment-cap checks.
  - test_common.py header:
    ```python
    """Tests for sfm/common.py: image-dir resolution and model output."""

    from pathlib import Path

    import numpy as np
    import pycolmap
    import pytest

    from collab_splats.pointcloud.sfm.common import rename_images_to_stems, sfm_image_dir
    from tests.pointcloud._stubs import make_recon
    ```
  - The moved tests change `sift_db.` → the direct names.

- [ ] **Step 4: Run targeted tests, expect PASS**
  - `... tests/pointcloud/sfm`
  - `contract_hits.py collab_splats/pointcloud/sfm/common.py collab_splats/pointcloud/sfm/sift_db.py` → `TOTAL 0` for common.py.

- [ ] **Step 5: Gates**
  - G′: +3 contract (new source file) −1 (row 112 delete) → **1599 passed**; xpassed +5 → **82**.

- [ ] **Step 6: Commit** `refactor(pointcloud): sfm/common.py owns image dir and stem-rename output`. Include the new files, sift_db.py, the three creators and both test files.

---

### Task A-3: `prepare_sfm_dirs`, with `images_dir` required  (spec rows: 87, 96, part of 68)

**Files:**
- Modify: `collab_splats/pointcloud/sfm/common.py` (new function)
- Modify: `colmap.py` reconstruct (:54-105)
  - The head that resolves image_dir, makes colmap_dir, clears sparse and lists names.
  - Make `images_dir: Path` required.
- Modify: `hloc.py:105-112`. Replace with `image_dir, colmap_dir, names = prepare_sfm_dirs(data_dir, images_dir)`, `hloc_dir = colmap_dir / "hloc"`, `hloc_dir.mkdir(exist_ok=True)`. Make `images_dir` required.
- Modify: `instantsfm.py`. Make `images_dir` required.
  - :342-345 → `image_dir, colmap_dir, names = prepare_sfm_dirs(data_dir, images_dir)`, `path_info = ReadData(str(data_dir))`, `path_info.image_path = str(image_dir)`.
  - :361-366: the database/output paths from `colmap_dir`.
  - :412 → `str(image_dir)`.
  - :413 → `output_path = colmap_dir / "sparse"`.
- Modify: `evals/scripts/eval.py:357` → `.reconstruct(output_dir, images_dir=staged_dir)`.
- Modify: `tests/evals/test_eval_instantsfm.py` fakes at :64, :100, :122, :141, :163, :189 → `def reconstruct(self, data_dir, images_dir):`
- Test: `tests/pointcloud/sfm/test_common.py`

- [ ] **Step 1: Write the failing test**
  ```python
  from collab_splats.pointcloud.sfm.common import prepare_sfm_dirs  # add to the import line
  from tests.pointcloud._stubs import make_scene


  def test_prepare_sfm_dirs_clears_a_stale_model(tmp_path):
      data_dir, images_dir = make_scene(tmp_path, ["frame_000000.png", "frame_000001.png"])
      stale = data_dir / "colmap" / "sparse" / "0"
      stale.mkdir(parents=True)
      (stale / "cameras.bin").write_bytes(b"old")

      image_dir, colmap_dir, names = prepare_sfm_dirs(data_dir, images_dir)

      assert image_dir == images_dir
      assert colmap_dir == data_dir / "colmap" and colmap_dir.is_dir()
      assert not (colmap_dir / "sparse").exists()
      assert names == ["frame_000000.png", "frame_000001.png"]
  ```

- [ ] **Step 2: Run it, expect FAIL**
  - Expected: `ImportError: cannot import name 'prepare_sfm_dirs'`.

- [ ] **Step 3: Implement**

  Add `import shutil` and `from collab_splats.preproc.frames import frame_paths` to common.py. Add:
  ```python
  def prepare_sfm_dirs(data_dir: Path, images_dir: Path) -> tuple[Path, Path, list[str]]:
      """
      Input dir, fresh colmap/ output dir and frame names for one sfm run.

      - a previous run's sparse/ is removed so no stale model is read back

      Args:
          data_dir: backend output directory.
          images_dir: the scene's images/ store.

      Returns:
          (image_dir, colmap_dir, image filenames in frame order).
      """
      image_dir = sfm_image_dir(images_dir)
      colmap_dir = Path(data_dir) / "colmap"
      colmap_dir.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(colmap_dir / "sparse", ignore_errors=True)
      return image_dir, colmap_dir, [p.name for p in frame_paths(image_dir)]
  ```
  - Each creator's reconstruct head is replaced by the single call.
    - colmap/hloc: the explicit `sfm_image_dir(...)`, `colmap_dir.mkdir`, `rmtree(sparse)` and the names comprehension are deleted.
    - Where a creator named its dir differently, rename it to `colmap_dir`.
  - Before editing, record for each creator whether its HEAD code cleared `sparse/`.
    - If one did NOT, that creator gains a clear of an output dir it recreates anyway. Output-neutral on a fresh dir.
    - List it in the commit body.
  - Signatures:
    ```python
    def reconstruct(self, data_dir: Path, images_dir: Path) -> pycolmap.Reconstruction:
    ```
  - The `images_dir=None` fallback branches (resolving `data_dir / "images"`) are deleted in all three.

- [ ] **Step 4: Run targeted tests, expect PASS**
  - `tests/pointcloud/sfm tests/evals/test_eval_instantsfm.py tests/wrapper/test_sfm_stage.py`
  - The reconstructor already passes `images_dir=`. Confirm with `grep -n "reconstruct(" collab_splats/wrapper/*.py`.
  - If any caller omits it, add it in this commit.

- [ ] **Step 5: Gates**
  - G′: +1 → **1600**.
  - contract_hits common.py → `TOTAL 0`.

- [ ] **Step 6: Commit** `refactor(pointcloud): prepare_sfm_dirs replaces three copies of the sfm dir setup`.

---

### Task A-4: `write_sfm_model`  (spec rows: 88)

**Files:**
- Modify: `common.py` (new function)
- Modify: the colmap tail (after the mapper block :90-96), hloc :172-177, instantsfm :426-429.
  - Each becomes `return write_sfm_model(recon, colmap_dir, "<label>", len(names))`.
  - instantsfm first does `recon = pycolmap.Reconstruction(str(sparse_dst))`.
- Test: `tests/pointcloud/sfm/test_common.py`

- [ ] **Step 1: Write the failing test**
  ```python
  def test_write_sfm_model_persists_stem_names(tmp_path):
      recon = make_recon(["frame_000000.png", "frame_000001.png"])

      out = write_sfm_model(recon, tmp_path, "colmap", 2)

      assert out is recon
      reread = pycolmap.Reconstruction(str(tmp_path / "sparse" / "0"))
      assert sorted(im.name for im in reread.images.values()) == ["frame_000000", "frame_000001"]
  ```
  Extend the import with `write_sfm_model`.

- [ ] **Step 2: Run it, expect FAIL**
  - Expected: `ImportError: cannot import name 'write_sfm_model'`.

- [ ] **Step 3: Implement**
  ```python
  def write_sfm_model(
      recon: pycolmap.Reconstruction, colmap_dir: Path, label: str, n_frames: int
  ) -> pycolmap.Reconstruction:
      """
      Final model written to colmap/sparse/0 under contract names.

      Args:
          recon: the mapper's chosen model.
          colmap_dir: output directory from prepare_sfm_dirs.
          label: backend name for the log line.
          n_frames: frames offered to the mapper.

      Returns:
          recon, renamed in place.
      """
      sparse_dir = colmap_dir / "sparse" / "0"
      sparse_dir.mkdir(parents=True, exist_ok=True)
      rename_images_to_stems(recon, sparse_dir)
      logger.info("%s: %d/%d registered, %d points3D", label, recon.num_reg_images(), n_frames, recon.num_points3D())
      return recon
  ```
  - Before: each creator has its own `sparse_dir.mkdir`, `rename_images_to_stems(...)` and `logger.info(...registered...)` lines.
  - After: one call.
  - Keep each backend's `shutil.rmtree(<scratch>)` that precedes the tail.
  - The per-creator log text changes to the shared format. That is log-only.

- [ ] **Step 4: Run targeted tests, expect PASS** (`tests/pointcloud/sfm`)

- [ ] **Step 5: Gates**
  - G′: +1 → **1601**.
  - contract_hits common.py → `TOTAL 0`.

- [ ] **Step 6: Commit** `refactor(pointcloud): write_sfm_model is the one sfm output path`.

---

### Task A-5: `ensure_sift_database` replaces the sidecar trio  (spec rows: 89, 95)

**Files:**
- Modify: `collab_splats/pointcloud/sfm/sift_db.py`
  - Rename `sift_database_valid` (:67-115) → private `_database_holds`.
  - Delete :229-276 (`database_params_path`, `matching_params`, `write_database_params`).
  - Add `ensure_sift_database`.
- Modify:
  - `colmap.py`: the DB block between prepare and the mapper.
  - `instantsfm.py:373-377`: the DB block.
- Test:
  - `tests/pointcloud/sfm/test_sift_db.py`: the validity tests :93-182 are renamed; :328-367 (7 tests) are replaced.
  - `tests/pointcloud/sfm/test_colmap.py`: `mocked` :52-71; delete :103-111 and :114-119.

> Spec deviation: row 89 says `sift_database_valid` goes away.
> - It becomes the private `_database_holds`.
> - Seven validity tests exercise it directly (wrong names, missing pairs, corrupt file).
> - Folding it into `ensure_sift_database` would force each of them through a mocked build.

- [ ] **Step 1: Write the failing test** (in test_sift_db.py, replacing :328-367)
  ```python
  ENSURE_KW = dict(pairing="exhaustive", overlap=10, num_retrieved=20, vocab_tree=None, num_threads=1)


  def _ensure(tmp_path, monkeypatch, **overrides):
      """
      Run ensure_sift_database with the build stubbed to write a valid DB; return the build count.
      """
      db = tmp_path / "colmap.db"
      calls = []

      def fake_build(image_dir, db_path, **kw):
          calls.append(kw)
          _sift_db(db_path, verified_pair=True)

      monkeypatch.setattr(sift_db, "build_sift_database", fake_build)
      sift_db.ensure_sift_database(tmp_path, db, NAMES, **{**ENSURE_KW, **overrides})
      return db, calls


  def test_ensure_builds_then_reuses(tmp_path, monkeypatch):
      _, first = _ensure(tmp_path, monkeypatch)
      _, second = _ensure(tmp_path, monkeypatch)
      assert len(first) == 1 and second == []


  def test_ensure_rebuilds_when_pairing_changes(tmp_path, monkeypatch):
      _ensure(tmp_path, monkeypatch)
      _, calls = _ensure(tmp_path, monkeypatch, pairing="sequential")
      assert len(calls) == 1


  @pytest.mark.parametrize(
      "pairing, knob, value, rebuilds",
      [
          ("sequential", "overlap", 3, True),
          ("sequential", "num_retrieved", 3, False),
          ("retrieval", "num_retrieved", 3, True),
          ("exhaustive", "overlap", 3, False),
      ],
  )
  def test_ensure_sidecar_holds_only_live_knobs(tmp_path, monkeypatch, pairing, knob, value, rebuilds):
      _ensure(tmp_path, monkeypatch, pairing=pairing)
      _, calls = _ensure(tmp_path, monkeypatch, pairing=pairing, **{knob: value})
      assert (len(calls) == 1) is rebuilds


  def test_ensure_rebuilds_a_db_without_sidecar(tmp_path, monkeypatch):
      _sift_db(tmp_path / "colmap.db", verified_pair=True)
      _, calls = _ensure(tmp_path, monkeypatch)
      assert len(calls) == 1
      assert (tmp_path / "colmap.db.json").is_file()
  ```
  - The count is 7 (1 + 1 + 4 + 1), replacing the 7 deleted tests.
  - `_sift_db(path, verified_pair=True)` is the existing test_sift_db builder. `NAMES` is that file's constant.
  - Confirm `_sift_db` writes rows for every name in `NAMES`. If it takes names, pass `NAMES`.
  - If the `retrieval` case needs a `vocab_tree`, pass `vocab_tree=tmp_path / "vt.bin"`; the build is stubbed.
  - Validity tests :93-182: `sift_db.sift_database_valid(` → `sift_db._database_holds(`.
- test_colmap `mocked` fixture (:52-71):
  - Patch `sift_db.build_sift_database` (instead of the sidecar trio) and `colmap_mod.fetch_vocab_tree` (returns `tmp_path / "vt.bin"`).
  - Patch `sift_db._database_holds` → `lambda *a: False` and `incremental_mapping`.
  - Delete `test_a_valid_db_is_reused...` (:103-111) and `test_sidecar_omits...` (:114-119). The ensure tests above cover both.

- [ ] **Step 2: Run it, expect FAIL**
  - Expected: `AttributeError: module ... sift_db has no attribute 'ensure_sift_database'`.

- [ ] **Step 3: Implement**
  ```python
  def ensure_sift_database(
      image_dir: Path,
      db_path: Path,
      names: list[str],
      *,
      pairing: str,
      overlap: int,
      num_retrieved: int,
      vocab_tree: Path | None,
      num_threads: int,
  ) -> None:
      """
      SIFT database for names, reused when its sidecar and contents still match.

      - sidecar <db>.json holds only the knobs the pairing mode reads
      - mismatch, missing sidecar or incomplete DB: delete and rebuild

      Args:
          image_dir: directory holding names.
          db_path: database file to reuse or build.
          names: image filenames the DB must hold.
          pairing: sequential, retrieval, sequential+retrieval or exhaustive.
          overlap: sequential window, read by sequential modes.
          num_retrieved: neighbors per image, read by retrieval modes.
          vocab_tree: vocab tree file for retrieval modes.
          num_threads: SIFT worker threads.
      """
      # Knobs the pairing mode actually reads, so a dead knob never forces a rebuild
      params = {"pairing": pairing}
      if pairing.startswith("sequential"):
          params["overlap"] = overlap
      if "retrieval" in pairing:
          params["num_retrieved"] = num_retrieved

      sidecar = db_path.with_name(db_path.name + ".json")
      if sidecar.is_file() and json.loads(sidecar.read_text()) == params and _database_holds(db_path, names):
          logger.info("SIFT database %s: reusing", db_path)
          return

      # Rebuild from scratch; the sidecar is written only after a successful build
      db_path.unlink(missing_ok=True)
      sidecar.unlink(missing_ok=True)
      logger.info("SIFT database %s: building (%s)", db_path, pairing)
      build_sift_database(
          image_dir,
          db_path,
          pairing=pairing,
          overlap=overlap,
          num_retrieved=num_retrieved,
          vocab_tree=vocab_tree,
          num_threads=num_threads,
      )
      sidecar.write_text(json.dumps(params, sort_keys=True))
  ```
  - Before editing, check the deleted `matching_params`/`database_params_path` (:229-276).
    - Confirm their key set and sidecar filename match `params` and `<db>.json` exactly.
    - If a key name differs (e.g. `num_neighbors`), keep the HEAD name so existing sidecars still match. Output-neutral.
  - colmap.py after prepare:
    ```python
    db_path = colmap_dir / "colmap.db"
    ensure_sift_database(
        image_dir,
        db_path,
        names,
        pairing=self.pairing,
        overlap=self.overlap,
        num_retrieved=self.num_retrieved,
        vocab_tree=fetch_vocab_tree() if "retrieval" in self.pairing else None,
        num_threads=self.num_threads,
    )
    ```
  - instantsfm.py :373-377 becomes:
    ```python
    ensure_sift_database(
        image_dir, colmap_dir / "instantsfm.db", names,
        pairing="exhaustive", overlap=10, num_retrieved=20, vocab_tree=None, num_threads=8,
    )
    ```
    - Use the HEAD literal values from the replaced `build_sift_database(...)` call at :373-377 if they differ from the above.
  - Remove the now-unused imports in each file (`sift_database_valid`, `write_database_params`, `json` if colmap no longer uses it).

- [ ] **Step 4: Run targeted tests, expect PASS** (`tests/pointcloud/sfm`)

- [ ] **Step 5: Gates**
  - G′: sift_db −7 +7; colmap −2 → **1599**.
  - contract_hits sift_db.py: the count must not rise.

- [ ] **Step 6: Commit** `refactor(pointcloud): ensure_sift_database owns SIFT DB reuse for colmap and InstantSfM`.

---

### Task A-6: inline `largest_model` into colmap  (spec rows: 91)

**Files:**
- Modify: `sift_db.py:355-377`: delete `largest_model`.
- Modify: `colmap.py`: after the mapper call, replace `recon = largest_model(recons)` with the inline block. Drop the import.
- Test: test_sift_db `largest_model` tests :375-382 are deleted (−2). test_colmap :74 and :130 already cover the empty and split cases.

- [ ] **Step 3: Implement**
  ```python
  # Keep the model with most registered images; the mapper returns one per component
  if not recons:
      raise RuntimeError("incremental mapping produced no model — too little overlap between frames")
  recon = max(recons.values(), key=lambda r: r.num_reg_images())
  if len(recons) > 1:
      sizes = sorted((r.num_reg_images() for r in recons.values()), reverse=True)
      logger.warning("incremental mapping split the scene into %d models %s — keeping the largest", len(recons), sizes)
  ```

- [ ] **Step 4: Run targeted tests, expect PASS**
  - `tests/pointcloud/sfm/test_colmap.py tests/pointcloud/sfm/test_sift_db.py`
  - Confirm test_colmap :74/:130 still match the messages.

- [ ] **Step 5: Gates**
  - G′: −2 → **1597**.

- [ ] **Step 6: Commit** `refactor(pointcloud): inline largest_model into its only caller`.

---

### Task A-7: explicit matcher map in `build_sift_database`  (spec rows: 92)

**Files:**
- Modify: `sift_db.py:183`, the matcher-subcommand selection.
- Test: existing argv tests in test_sift_db (`test_exhaustive_argv` etc.).

- [ ] **Step 3: Implement** (a behavior-identical rewrite, so Steps 1-2 are skipped)
  - Before: the conditional chain at :183 choosing the subcommand.
  - After:
    ```python
    matcher = {
        "sequential": "sequential_matcher",
        "sequential+retrieval": "sequential_matcher",
        "retrieval": "vocab_tree_matcher",
        "exhaustive": "exhaustive_matcher",
    }[pairing]
    ```
  - An unknown pairing now raises `KeyError`, not a silent fallback. Check that HEAD validated `pairing` earlier.
    - If HEAD did not validate: add `if pairing not in MATCHERS: raise ValueError(...)` and keep the dict as a module constant `MATCHERS`.
    - That also satisfies the release silent-fallback check.

- [ ] **Step 4:** `tests/pointcloud/sfm/test_sift_db.py` passes with unchanged counts.
- [ ] **Step 5:** G′ 0 (1597).
- [ ] **Step 6: Commit** `refactor(pointcloud): explicit pairing→matcher table`.

---

### Task A-8: `colmap_cli_version` raises instead of returning "unknown"  (spec rows: 93)

**Files:**
- Modify: `sift_db.py:317-330`
- Test: `tests/pointcloud/sfm/test_sift_db.py`

- [ ] **Step 1: Write the failing test**
  ```python
  def test_colmap_cli_version_raises_without_the_binary(monkeypatch):
      def missing(*a, **kw):
          raise FileNotFoundError("colmap")

      monkeypatch.setattr(sift_db.subprocess, "run", missing)
      with pytest.raises(RuntimeError, match="colmap"):
          sift_db.colmap_cli_version()


  def test_colmap_cli_version_raises_on_an_empty_banner(monkeypatch):
      monkeypatch.setattr(
          sift_db.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout="", stderr="")
      )
      with pytest.raises(RuntimeError, match="banner"):
          sift_db.colmap_cli_version()
  ```
  - Add `import subprocess` at the top.
  - Before writing the test, check how :317-330 calls subprocess (`run` vs `check_output`) and match the stub to it.

- [ ] **Step 2: Run it, expect FAIL**
  - The first test: at HEAD it returns `"unknown"`, or `FileNotFoundError` propagates. Either way it is not `RuntimeError`.
  - The second test: it returns `"unknown"`, so `DID NOT RAISE`.

- [ ] **Step 3: Implement**
  ```python
  try:
      out = subprocess.run(["colmap", "help"], capture_output=True, text=True).stdout
  except FileNotFoundError as e:
      raise RuntimeError("colmap binary not on PATH — needed for SIFT extraction/matching") from e
  banner = " ".join(line.strip() for line in out.splitlines()[:2] if line.strip())
  if not banner:
      raise RuntimeError("colmap help printed no version banner")
  return banner
  ```
  - Keep HEAD's exact argv and stream choice. Only the two failure paths change.

- [ ] **Step 4:** targeted PASS.
- [ ] **Step 5:** G′ +2 → **1599**.
- [ ] **Step 6: Commit** `fix(pointcloud): colmap_cli_version fails loudly instead of recording "unknown"`.

---

### Task A-9: `provenance()` on the three creators  (spec rows: 98, lane A half)

**Files:**
- Modify: `colmap.py`, `hloc.py`, `instantsfm.py` (new method each; `import importlib.metadata`)
- Modify: `collab_splats/pointcloud/sfm/__init__.py:5` bullet → `- every creator: reconstruct(data_dir, images_dir) -> pycolmap.Reconstruction, provenance() -> dict`
- Test: `tests/pointcloud/sfm/test_common.py`

- [ ] **Step 1: Write the failing test**
  ```python
  from collab_splats.pointcloud.sfm import colmap as colmap_mod
  from collab_splats.pointcloud.sfm import hloc as hloc_mod
  from collab_splats.pointcloud.sfm import instantsfm as instantsfm_mod


  @pytest.mark.parametrize(
      "creator, keys",
      [
          (lambda: colmap_mod.ColmapCreator(), ["pycolmap_version", "colmap_cli_version"]),
          (lambda: hloc_mod.HlocCreator(), ["pycolmap_version", "hloc_commit"]),
          (lambda: instantsfm_mod.InstantSfMCreator(), ["instantsfm_version"]),
      ],
  )
  def test_provenance_keys(monkeypatch, creator, keys):
      monkeypatch.setattr("importlib.metadata.version", lambda name: f"v-{name}")
      monkeypatch.setattr(colmap_mod, "colmap_cli_version", lambda: "COLMAP 3.10")
      assert list(creator().provenance()) == keys
  ```
  - Before writing, check the three module paths import without instantsfm installed.
    - `instantsfm.py` at HEAD guards the upstream import (tests/pointcloud/sfm/test_instantsfm uses importorskip).
    - If the module import itself needs instantsfm: put that parametrize case in `test_instantsfm.py` behind its existing skip, and drop it from the list here.
    - The delta below assumes all 3 run. If moved, the case skips (+1 skipped, +2 passed).
  - Use each creator's HEAD constructor defaults. If a constructor needs args (e.g. hloc features), pass the HEAD defaults from `configs/`.

- [ ] **Step 2: Run it, expect FAIL**: `AttributeError: 'ColmapCreator' object has no attribute 'provenance'`.

- [ ] **Step 3: Implement**
  ```python
  # colmap.py
  def provenance(self) -> dict:
      """
      Versions that produced this model.

      Returns:
          pycolmap and colmap CLI versions.
      """
      return {
          "pycolmap_version": importlib.metadata.version("pycolmap"),
          "colmap_cli_version": colmap_cli_version(),
      }

  # hloc.py
  def provenance(self) -> dict:
      """
      Versions that produced this model.

      Returns:
          pycolmap version and pinned hloc commit.
      """
      return {"pycolmap_version": importlib.metadata.version("pycolmap"), "hloc_commit": HLOC_PIN}

  # instantsfm.py
  def provenance(self) -> dict:
      """
      Versions that produced this model.

      Returns:
          installed InstantSfM version.
      """
      return {"instantsfm_version": importlib.metadata.version("instantsfm")}
  ```
  - `HLOC_PIN` is the existing pin constant in hloc.py. If it is named differently, use the HEAD name.
  - colmap.py imports `colmap_cli_version` from `.sift_db` at module level, which the test's patch relies on.
  - Before writing the dicts, compare the key names against the reconstructor's provenance block (:1277-1288).
    - Use the SAME keys it writes today, so `sfm` provenance output is unchanged.
    - If they differ from the above, the reconstructor's names win.

- [ ] **Step 4:** targeted PASS.
- [ ] **Step 5:** G′ +3 → **1602**.
- [ ] **Step 6: Commit** `feat(pointcloud): sfm creators report their own provenance`.

---

### Task A-10: InstantSfM filename and cast cleanups  (spec rows: 67, 68)

> Evidence for row 67: upstream pin `cre185/InstantSfM@d3e599e`, `instantsfm/scene/defs.py:153` sets `self.filenames = [""] * num_images` on `Images`. The attribute always exists, so the `getattr` fallback is dead.
> - Checked via `curl https://raw.githubusercontent.com/cre185/InstantSfM/d3e599e/instantsfm/scene/defs.py`.

**Files:**
- Modify: `instantsfm.py:176`: the `getattr(self.images, "filenames", ...)` fallback → `filename = self.images.filenames[idx]`.
- Modify: `instantsfm.py:283`, `:295`: drop the redundant `int()` casts around values that are already Python ints (for example `int(len(...))` or `int(idx)` from `range`).
  - Confirm each operand's type at HEAD before removing a cast.
  - A cast over a numpy scalar that feeds pycolmap stays.

- [ ] **Step 3:** as above (a pure deletion).
- [ ] **Step 4:** `tests/pointcloud/sfm/test_instantsfm.py` (skips without instantsfm).
- [ ] **Step 5:** G′ 0.
- [ ] **Step 6: Commit** `refactor(pointcloud): drop dead InstantSfM filename fallback and casts`.

---

### Task A-11: `get_device()` for device selection  (spec rows: 73)

**Files:**
- Modify: `sift_db.py:159`: `use_gpu = torch.cuda.is_available()` → `use_gpu = get_device() == "cuda"`.
  - Remove `import torch` (:21); add `from collab_splats.utils.torch_utils import get_device`.
- Modify: `pointcloud/utils.py:277`: the inline cuda check → `device = get_device()`.
  - Add the import; drop `torch.cuda` usage there only.

- [ ] **Step 3:** as above.
  - test_sift_db `_capture_argv` still patches `"torch.cuda.is_available"`. `get_device()` reads it at call time, so it is unaffected.
- [ ] **Step 4:** `tests/pointcloud/sfm/test_sift_db.py tests/pointcloud/test_feature_lifting.py`
- [ ] **Step 5:**
  - G′ 0.
  - **P required** (utils.py).
- [ ] **Step 6: Commit** `refactor(pointcloud): get_device() replaces inline cuda probes`.

---

### Task A-12: delete `fit_dominant_plane`  (spec rows: 23)

**Files:**
- Modify: `pointcloud/utils.py:141-177`: delete the plane section.
  - Remove `rotation_align_vectors` from the :12-26 transforms import.
  - Remove "plane fit" from the module docstring (as R1-3 left it).
- Test: `tests/pointcloud/test_pointcloud_utils.py:113-147`: delete the plane tests (2), and drop the import.

- [ ] **Step 3:** first run `grep -rn fit_dominant_plane $WT --include=*.py --include=*.ipynb --include=*.md`.
  - Only utils.py and the test may hit.
  - Any other hit: stop and report.
- [ ] **Step 4:** `tests/pointcloud/test_pointcloud_utils.py`
- [ ] **Step 5:**
  - G′ −2 → **1600**.
  - **P required.**
- [ ] **Step 6: Commit** `refactor(pointcloud): drop unused fit_dominant_plane` (use `git add` on the modified files).

---

### Task A-13: `subsample_points` loses the confidence gate  (spec rows: 58)

> Spec deviation: the spec cites README.md:171. The `conf=conf` call is at **README.md:132** at HEAD.
> - `wrapper.py:501` and `:643` already call without `conf`.

**Files:**
- Modify: `pointcloud/utils.py:102-138`.
  - The signature becomes `subsample_points(points, colors=None, max_points=50_000)`.
  - Delete the conf block :125-130 and the `conf`/`conf_percentile` Args.
  - Summary becomes "Randomly cap a point set to a fixed budget."
- Modify: `README.md:132`: drop `conf=conf`.
- Test: `tests/pointcloud/test_utils_subsample.py`
  - Delete :27-34, :37-41, :77-83 (3 conf tests).
  - Docstring :1 → `"""Tests for subsample_points (random count cap) and confidence_mask."""`.
  - (extra) Move the inline imports at :62 and :71 to the top.

- [ ] **Step 3:** as above. Grep for `subsample_points(` across `$WT` and confirm no caller passes `conf`.
- [ ] **Step 4:** `tests/pointcloud/test_utils_subsample.py`
- [ ] **Step 5:**
  - G′ −3 → **1597**.
  - **P required.**
- [ ] **Step 6: Commit** `refactor(pointcloud): subsample_points is a pure count cap`.

---

### Task A-14: split `cross_frame_attention_ratio` from its reduction  (spec rows: 61, 84)

**Files:**
- Modify: `pointcloud/utils.py:413-465`
  - The signature becomes `cross_frame_attention_ratio(k, q, *, token_offset: int) -> np.ndarray` and returns per-token ratios.
  - Add `mean_top_quarter`.
- Modify: `feedforward/base.py:45`: import `mean_top_quarter` too.
- Modify: `feedforward/base.py:1375` → `ratio = mean_top_quarter(cross_frame_attention_ratio(features["k"], features["q"], token_offset=self._lc_token_offset))`.
- Modify: `evals/scripts/eval_similarity_calibration.py`
  - Delete `mean_top_quarter` :61-65.
  - Delete `_compute_ratio_np` :181-198 with its divider.
  - Add `from collab_splats.pointcloud.utils import cross_frame_attention_ratio, mean_top_quarter` after :49.
  - :255 → `ratio_np = cross_frame_attention_ratio(k, q, token_offset=tok_off)`.
- Test: `tests/pointcloud/test_pointcloud_utils.py`
  - :67-72 `returns_float` → asserts an ndarray.
  - :75-100 wrap in `mean_top_quarter`.
  - :103-110 → `pytest.raises(ValueError, match="no patch tokens")`.
  - Add a `mean_top_quarter` test.

- [ ] **Step 1: Write the failing test**
  ```python
  def test_mean_top_quarter_averages_the_top_quartile():
      assert mean_top_quarter(np.array([0.0, 1.0, 2.0, 3.0])) == 3.0
  ```
  - Plus the three edits above. The existing empty-token test becomes:
  ```python
  with pytest.raises(ValueError, match="no patch tokens"):
      cross_frame_attention_ratio(k, q, token_offset=tokens_per_img)
  ```

- [ ] **Step 2: Run it, expect FAIL**: `ImportError: cannot import name 'mean_top_quarter'`.

- [ ] **Step 3: Implement**
  - In `cross_frame_attention_ratio`: the empty case becomes `raise ValueError(f"token_offset {token_offset} leaves no patch tokens (tokens_per_img {tokens_per_img})")`.
  - The tail becomes `return ratio.cpu().float().numpy().ravel()`.
  - The top-quarter reduction and its SPARK comment move out into:
  ```python
  def mean_top_quarter(ratios: np.ndarray) -> float:
      """
      Mean of the ratios at or above the 75th percentile.

      Args:
          ratios: per-token cross-frame attention ratios.

      Returns:
          The top-quartile mean.
      """
      # SPARK loop gate: top-quartile mean, robust to background tokens
      thresh = float(np.percentile(ratios, 75))
      return float(ratios[ratios >= thresh].mean())
  ```
  - Keep the HEAD SPARK comment text verbatim if it was longer than one line (header + bullets).

- [ ] **Step 4:** `tests/pointcloud/test_pointcloud_utils.py tests/geometry tests/evals`
- [ ] **Step 5:**
  - G′ +1 → **1598**.
  - **P required** (utils + ff/base).
- [ ] **Step 6: Commit** `refactor(pointcloud): cross_frame_attention_ratio returns per-token ratios; mean_top_quarter reduces`.

---

### Task A-15: 4x4 extrinsics and 3-D depth enforced  (spec rows: 22, 59, 60)

**Files:**
- Modify: `pointcloud/utils.py`
  - `reproject_pixels` :352-405:
    - The param `extrinsics_3x4` becomes `extrinsics` (N,4,4).
    - Raise ValueError on `depth.ndim != 3` or `extrinsics.shape[-2:] != (4, 4)`.
    - (extra) `fi, ri, ci = pixel_indices.T`.
    - `z = depth[fi, ri, ci].astype(np.float64)`.
    - `cam2world = invert_poses(extrinsics.astype(np.float64))` replaces the homogenize-then-invert :395-400.
  - `lift_features`:
    - :282-286 → `if result.extrinsics.shape[-2:] != (4, 4): raise ValueError(...)`.
    - :300-303 → `if result.depth.ndim != 3: raise ValueError(...)`.
    - :289-298 → `conf = torch.ones(...) if result.confidence is None else result.confidence.to(device=device, dtype=torch.float32)`.
- Modify: `feedforward/base.py:328-333`: pass `self.extrinsics` (not `[:, :3, :]`).
- Modify: `tests/pointcloud/test_feedforward_reproject.py:45` → `np.testing.assert_array_equal(call_args[2], result.extrinsics)`.
- Test: `tests/pointcloud/test_feature_lifting.py`
  - `_make_extrinsics_intrinsics` :98-105 → 4x4.
  - The depth at :108-114 and :117-124 → 3-D.
  - Add 4 raise tests.

- [ ] **Step 1: Write the failing test**
  ```python
  def test_reproject_pixels_rejects_4d_depth():
      ext, K = _make_extrinsics_intrinsics(1)
      with pytest.raises(ValueError, match="depth"):
          reproject_pixels(np.ones((1, 4, 4, 1), np.float32), np.zeros((1, 3), np.int64), ext, K)


  def test_reproject_pixels_rejects_3x4_extrinsics():
      ext, K = _make_extrinsics_intrinsics(1)
      with pytest.raises(ValueError, match="4, 4"):
          reproject_pixels(np.ones((1, 4, 4), np.float32), np.zeros((1, 3), np.int64), ext[:, :3, :], K)


  def test_lift_features_rejects_4d_depth():
      result = _make_lift_result()
      bad = dataclasses.replace(result, depth=result.depth[..., None])
      with pytest.raises(ValueError, match="depth"):
          lift_features(bad, _feature_maps_for(result))


  def test_lift_features_rejects_3x4_extrinsics():
      result = _make_lift_result()
      bad = dataclasses.replace(result, extrinsics=result.extrinsics[:, :3, :])
      with pytest.raises(ValueError, match="4, 4"):
          lift_features(bad, _feature_maps_for(result))
  ```
  - `_make_lift_result` / `_feature_maps_for` are the file's existing builders. Use their HEAD names and argument lists.
  - If features are built inline in the existing lift tests, copy that construction.
  - Add `import dataclasses`.
  - Error texts:
    - `f"depth must be (N, H, W), got {depth.shape}"`
    - `f"extrinsics must be (N, 4, 4), got {extrinsics.shape}"`

- [ ] **Step 2: Run it, expect FAIL**
  - reproject: no raise, or an IndexError.
  - lift: the HEAD squeeze path accepts 4-D depth and homogenizes 3x4, so `DID NOT RAISE`.

- [ ] **Step 3:** as described above. The asserts at :263-268 stay (see Proposed additions).
- [ ] **Step 4:** `tests/pointcloud/test_feature_lifting.py tests/pointcloud/test_feedforward_reproject.py`
- [ ] **Step 5:**
  - G′ +4 → **1602**.
  - **P required.**
- [ ] **Step 6: Commit** `refactor(pointcloud): reproject/lift take (N,4,4) extrinsics and (N,H,W) depth only`.

---

### Task A-16: one grid_sample path  (spec rows: part of 22)

**Files:**
- Modify: `pointcloud/utils.py:184-233`
  - `_grid_sample_at_pixels(fmap, rows, cols, image_size)` takes torch rows/cols:
    ```python
    grid = torch.stack([(2 * cols + 1) / W - 1, (2 * rows + 1) / H - 1], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(fmap.unsqueeze(0).float(), grid, mode="bilinear", align_corners=False, padding_mode="border")
    return sampled.squeeze(0).squeeze(1).T
    ```
  - `_sample_at_source_pixels` passes `torch.from_numpy(rows.astype(np.float32)).to(fmap.device)` (cols likewise) and appends `.cpu()`.
  - lift :326-331 → `sampled = _grid_sample_at_pixels(fmap, v_safe, u_safe, image_size)`.
- Test: existing test_feature_lifting only.

- [ ] **Step 3:** as above. It is a pure consolidation; the numerics are the same formula.
- [ ] **Step 4:** `tests/pointcloud/test_feature_lifting.py tests/pointcloud/test_pointcloud_utils.py`
- [ ] **Step 5:**
  - G′ 0.
  - **P required.** P decides output neutrality; revert on fail.
- [ ] **Step 6: Commit** `refactor(pointcloud): single grid_sample helper for feature sampling`.

---

### Task A-17: `confidence_mask` warns on uniform confidence  (spec rows: D2)

**Files:**
- Modify: `pointcloud/utils.py:82-99`, the tail:
  ```python
  if above.any():
      return above
  logger.warning("confidence_mask: nothing above p%.1f (uniform confidence) — keeping every point", percentile)
  return np.ones(conf.shape, dtype=bool)
  ```
- Test: extend `test_utils_subsample.py::test_confidence_mask_uniform_confidence_keeps_all` with `caplog` and assert `"uniform confidence" in caplog.text`. No new test.

- [ ] **Step 1:** the caplog assertion.
- [ ] **Step 2:** FAIL `AssertionError` (HEAD is silent).
- [ ] **Step 3:** as above.
- [ ] **Step 4:** targeted PASS.
- [ ] **Step 5:**
  - G′ 0.
  - **P required.**
- [ ] **Step 6: Commit** `fix(pointcloud): confidence_mask logs its keep-all fallback`.

---

### Task A-18: vda imports through `vendored_path`  (spec rows: 74)

> Spec deviation: HEAD checks the `video_depth_anything` subdir; `vendored_path` checks `VDA_ROOT`.
> - A root without the subdir now fails as `ModuleNotFoundError`, not the setup.sh hint.
> - The existing test `test_vda.py:41-52` (match "setup.sh") removes the whole root, so it still passes.
> - The upstream modules' only inline imports are the guarded `xformers`/`decord` try-blocks. Removing the path after import is safe.

**Files:**
- Modify: `vda.py:9` (drop `import sys`) and :41-47:
  ```python
  with vendored_path(VDA_ROOT, "run setup.sh (clones Video-Depth-Anything at 4f5ae23)"):
      from video_depth_anything.video_depth import VideoDepthAnything
  ```

- [ ] **Step 4:** `tests/pointcloud/test_vda.py`
- [ ] **Step 5:** G′ 0.
- [ ] **Step 6: Commit** `refactor(pointcloud): vda imports via vendored_path`.

---

### Task A-19: `_vda_npy_dir`  (spec rows: 80)

**Files:**
- Modify: `vda.py`. Add:
  ```python
  def _vda_npy_dir(out_dir: Path) -> Path:
      """
      Cache directory for per-keyframe VDA depth arrays.
      """
      return Path(out_dir) / "depth_vda" / "images" / "npy"
  ```
- Use it at :100 and :141.

- [ ] **Steps 4-6:**
  - test_vda passes.
  - G′ 0.
  - Commit `refactor(pointcloud): one definition of the VDA npy cache dir`.

---

### Task A-20: `fp32` switch on `generate_vda_depth`  (spec rows: 63)

**Files:**
- Modify: `vda.py`
  - Add the kwarg `fp32: bool = False` plus an `Args:` entry ("run inference in float32 instead of autocast fp16").
  - :157 passes `fp32=fp32` to `infer_video_depth`.
  - Drop `.astype(np.float32)` at :147. The cached arrays were saved float32.
  - (extra) Drop it at :171 as well. Only if `infer_video_depth` returns float32 at HEAD: check the upstream pin `video_depth.py`.
- Test: `tests/pointcloud/test_vda.py`, following the stub pattern at :113-127.

- [ ] **Step 1: Write the failing test**
  ```python
  def test_generate_vda_depth_forwards_fp32(tmp_path, monkeypatch):
      seen = {}

      class Recorder(_StubVDA):
          def infer_video_depth(self, frames, fps, **kw):
              seen.update(kw)
              return super().infer_video_depth(frames, fps, **kw)

      _install_stub(monkeypatch, Recorder)
      vda.generate_vda_depth(*_vda_args(tmp_path), fp32=True)
      assert seen["fp32"] is True
  ```
  - `_StubVDA`, `_install_stub` and `_vda_args` stand for whatever :113-127 uses to stub the model and build the call.
  - Reuse those exact helpers. If the pattern is inline, inline it here too.

- [ ] **Step 2:** FAIL `TypeError: generate_vda_depth() got an unexpected keyword argument 'fp32'`.
- [ ] **Step 3:** as above.
- [ ] **Step 4:** test_vda PASS.
- [ ] **Step 5:** G′ +1 → **1603**.
- [ ] **Step 6: Commit** `feat(pointcloud): generate_vda_depth fp32 switch`.

---

### Task A-21: vda frees the model via `pytorch_gc`  (spec rows: 82)

**Files:**
- Modify: `vda.py:161-163`: the manual `del`/`gc.collect`/`torch.cuda.empty_cache` → `del model` then `pytorch_gc()`. Drop imports that become unused.
- Modify (extra): `evals/scripts/eval.py:352-354`: the same manual cleanup → `pytorch_gc()`.

- [ ] **Steps 4-6:**
  - test_vda and tests/evals pass.
  - G′ 0.
  - Commit `refactor(pointcloud): pytorch_gc for VDA teardown`.

---

### Task A-22: depth_align drops dead guards, names mismatch early  (spec rows: 65)

**Files:**
- Modify: `depth_align.py`
  - Delete `_tracked_point3d_ids` :26-34. At :286 inline:
    ```python
    # Only tracked points: InstantSfM keeps empty-track points3D after retriangulation
    point3d_ids = sorted(pid for pid, p in reconstruction.points3D.items() if len(p.track.elements) > 0)
    ```
    Keep HEAD's comment text if it differs.
  - `_depth_correspondences`: delete the row check :82-84 and the missing check :86-88; keep `name_to_image`.
  - stats :177: remove `"n_fallback"`. The :280 log uses `len(stats["fallback_frames"])`.
  - After `n, h, w = depths.shape` (:241), before the images check :245-246:
    ```python
    if len(stems) != n:
        raise ValueError(f"{len(stems)} image names for {n} depth maps — rows would misalign")
    ```
- Test: `tests/pointcloud/test_depth_align.py`
  - Delete :115-119 (the unregistered-name test, now unreachable).
  - :134 → `assert stats["fallback_frames"] == []`.
  - :137-144 → the new message.
  - Delete :183-194 (the fake-recon tracked-ids test). Replace it with a real-pycolmap test.

- [ ] **Step 1: Write the failing test**
  ```python
  def test_names_depths_mismatch_raises_before_alignment():
      recon, depths, images, names = _scene_inputs(n=2)
      with pytest.raises(ValueError, match="2 image names for 1 depth maps"):
          depth_align.result_from_reconstruction(recon, depths[:1], images, names, min_obs=1)


  def test_empty_track_points_are_ignored():
      recon, depths, images, names = _scene_inputs(n=1)
      recon.add_point3D(np.array([1.0, 1.0, 5.0]), pycolmap.Track(), np.array([1, 2, 3], np.uint8))
      out, _ = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)
      assert out.points.shape == (1, 3)
      assert len(out.pixel_indices) == 1
  ```
  - At HEAD the first raises `"2 frames for 1 depth maps"`, so the `match` fails.
  - `_scene_inputs(n=1)` holds one tracked point, so one output row.
  - Before relying on `out.points.shape == (1, 3)`, check what `out.points` is (the tracked COLMAP points vs dense depth points). If it is dense, assert on the stats' sparse-point count instead.

- [ ] **Step 2:** FAIL (the regex mismatch on test 1).
- [ ] **Step 3:** as above.
- [ ] **Step 4:** `tests/pointcloud/test_depth_align.py`
- [ ] **Step 5:**
  - G′: −1 (the unregistered test); the tracked-ids test is swapped 1→1 → **1602**.
  - **P required.**
- [ ] **Step 6: Commit** `refactor(pointcloud): depth_align fails early on name/depth mismatch, drops unreachable guards`.

---

### Task A-23: depth_align shared transforms and coords  (spec rows: 64)

**Files:**
- Modify: `depth_align.py`
  - Imports :8-16: add `from collab_splats.geometry.transforms import extrinsics_to_homogeneous`. Change the ff import to `from .feedforward.base import FeedforwardResult, full_frame_coords`.
  - :249-251 → `extrinsics = extrinsics_to_homogeneous(np.stack([im.cam_from_world().matrix() for im in images_sorted])).astype(np.float32)`
  - :304 → `original_coords = full_frame_coords(orig_w, orig_h, n)`

- [ ] **Steps 4-6:**
  - test_depth_align passes.
  - G′ 0.
  - **P required.**
  - Commit `refactor(pointcloud): depth_align reuses extrinsics_to_homogeneous and full_frame_coords`.

---

### Task A-24: `Rigid3d * points` in `_depth_correspondences`  (spec rows: 79; own commit)

> Measured: pycolmap 4.0.4 `Rigid3d * (N,3)` differs from the HEAD manual `R @ x + t` by up to 4.4e-16. It is not bit-identical.
> - It gets its own commit so P isolates it.

**Files:**
- Modify: `depth_align.py:102-104` → `d_colmap = (image.cam_from_world() * xyz)[:, 2]`
- Modify: `tests/pointcloud/test_depth_align.py:56-58`. The fake `_Pose` gains:
  ```python
  def __mul__(self_inner, xyz):
      return np.asarray(xyz)
  ```

- [ ] **Step 4:** test_depth_align passes.
- [ ] **Step 5:**
  - G′ 0.
  - **P required.** If P fails: `cd $WT && git revert --no-edit HEAD`, keep the manual transform, and report. See Proposed additions.
- [ ] **Step 6: Commit** `refactor(pointcloud): pycolmap Rigid3d transforms depth-align points`.

---

### Task A-25: InstantSfM tests drop duplicated asserts  (spec rows: 110, lane A part)

**Files:**
- Modify: `tests/pointcloud/sfm/test_instantsfm.py`: delete :189-190 and :211-212 (two tests duplicating the common/ensure coverage).

- [ ] **Steps 4-6:**
  - The file passes or skips.
  - G′ −2 → **1600**.
  - These tests skip without instantsfm. If G′ counts them as skipped, the delta is −2 skipped instead; record the observed delta.
  - Commit `test(pointcloud): drop instantsfm tests covered by sfm/common`.

---

### Task A-26: sift_db test naming and imports  (spec rows: 113)

**Files:**
- Modify: `tests/pointcloud/sfm/test_sift_db.py`
  - :77-81 → rename to `test_exhaustive_argv`.
  - Delete the :29 history comment.
  - (extra) :393 `__import__("hashlib")` → top-level `import hashlib`.

- [ ] **Steps 4-6:**
  - Targeted pass.
  - G′ 0.
  - Commit `test(pointcloud): tidy sift_db test names and imports`.

---

### Task A-27: partial registration reaches depth_align (real pycolmap)  (spec rows: 115)

**Files:**
- Test: `tests/pointcloud/sfm/test_common.py`

- [ ] **Step 1: Write the test**
  ```python
  from collab_splats.pointcloud.depth_align import result_from_reconstruction


  @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="B9: in-memory mapper model keeps deregistered images")
  def test_partial_registration_reaches_depth_align(tmp_path):
      recon = make_recon(["frame_000000", "frame_000001", "frame_000002"])
      recon.deregister_frame(recon.images[3].frame_id)

      model = write_sfm_model(recon, tmp_path / "colmap", "colmap", 3)

      result, _ = result_from_reconstruction(
          model,
          np.full((2, 8, 16), 2.0, np.float32),
          np.zeros((2, 48, 64, 3), np.uint8),
          ["frame_000000.png", "frame_000001.png"],
          min_obs=1,
      )
      assert result.extrinsics.shape == (2, 4, 4)
      assert result.image_paths == [Path("frame_000000"), Path("frame_000001")]
  ```
  - Measured on pycolmap 4.0.4: `deregister_frame` leaves `len(recon.images) == 3` in memory; a write + re-read has 2.
  - `result_from_reconstruction` counts in-memory images, so it raises `RuntimeError` from depth_align :219-225.
  - The xfail is **strict**. When B9 (lane C / reconstructor: re-read the written model, or iterate `reg_image_ids`) lands, this XPASS fails the suite, and the fixer removes the marker.

- [ ] **Step 2: Run it**
  ```bash
  cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/sfm/test_common.py::test_partial_registration_reaches_depth_align
  ```
  - Expect `1 xfailed`.
  - If it reports `XPASS(strict)` or a different exception, stop and report the actual traceback. Do not loosen the marker.

- [ ] **Step 5:** G′ xfailed +1 → **4**.
- [ ] **Step 6: Commit** `test(pointcloud): pin B9 partial-registration gap with a real pycolmap model`.

---

## Lane A running totals

| After | failed | passed | skipped | xfailed | xpassed | errors |
|---|---|---|---|---|---|---|
| baseline (post-R1) | 18 | 1596 | 10 | 3 | 77 | 3 |
| H4 | 18 | 1597 | 10 | 3 | 77 | 3 |
| A-2 | 18 | 1599 | 10 | 3 | 82 | 3 |
| A-3 | 18 | 1600 | 10 | 3 | 82 | 3 |
| A-4 | 18 | 1601 | 10 | 3 | 82 | 3 |
| A-5 | 18 | 1599 | 10 | 3 | 82 | 3 |
| A-6 | 18 | 1597 | 10 | 3 | 82 | 3 |
| A-8 | 18 | 1599 | 10 | 3 | 82 | 3 |
| A-9 | 18 | 1602 | 10 | 3 | 82 | 3 |
| A-12 | 18 | 1600 | 10 | 3 | 82 | 3 |
| A-13 | 18 | 1597 | 10 | 3 | 82 | 3 |
| A-14 | 18 | 1598 | 10 | 3 | 82 | 3 |
| A-15 | 18 | 1602 | 10 | 3 | 82 | 3 |
| A-20 | 18 | 1603 | 10 | 3 | 82 | 3 |
| A-22 | 18 | 1602 | 10 | 3 | 82 | 3 |
| A-25 | 18 | 1600 | 10 | 3 | 82 | 3 |
| **A-27 (final)** | **18** | **1600** | **10** | **4** | **82** | **3** |

- Rows not listed carry no delta.
- **P runs after:** H1, H2, H4, A-11..A-17, A-22..A-24.

---

## Lane B — feedforward creators, registry, LC wrapper, dashboard

Lane worktree: `WT_B=/workspace/collab-splats/.worktrees/pc-lane-b`, forked from `clean/pointcloud-release` after lane A is landed. P in the lane: `cd $WT_B && PARITY_WT=$WT_B PYTHONPATH=$WT_B PYTHONUTF8=1 $PY $SP/parity.py --check`. In every command below `$WT`
means `$WT_B`. Gates run with `PYTHONPATH=$WT_B:$SP/stubs`. P runs after **every** lane-B commit:
every file touched here is on the parity path (`feedforward/*`, `base.py`, `torch_utils`,
`geometry/transforms.py`).

All line numbers cite HEAD `11c5d7c7`. A later task's line numbers shift after an earlier task's
edits, so every edit is anchored by its quoted `before` text, not by the number alone.

## Lane-level decisions

**Helper signatures (H1-H4, checked against their tasks at assembly — they match; H2's message is `f"{root} not found — {hint}"`, and no lane B `match=` depends on it)**
- `collab_splats.utils.torch_utils.to_numpy(x) -> np.ndarray`
  - bf16 is cast to fp32, then `.detach().cpu().numpy()`
- `collab_splats.utils.torch_utils.vendored_path(root: Path, hint: str)`
  - context manager: prepends `root` to `sys.path`, removes it on exit, raises `ImportError(hint)`
- `collab_splats.preproc.frames.frame_name(idx: int) -> str`
  - returns `"frame_{idx:06d}"`
- `collab_splats.pointcloud.feedforward.base.full_frame_coords(W: int, H: int, N: int) -> np.ndarray`
  - returns (N, 6) float32 rows `[0, 0, W, H, W, H]`
- Verified at assembly against H1-H4: the signatures match (H4 is `(width, height, n)`, called positionally).

**Registry (rows 35/72): lane B owns the creator side, and the branch stays green.**
- B-2 makes `BaseFeedforwardCreator` a `RegistryMixin` and decorates the four creators.
- `get_creator` becomes `BaseFeedforwardCreator.get`.
- The reconstructor's `creator_map` (`reconstructor.py:525-530`) and `_FEEDFORWARD_BACKENDS`
  (`:78`) keep working unchanged. They name the same classes, so the two paths agree until lane C
  swaps them to `get_creator`.
- The rows do not have to land together.
- Test ownership:
  - `RegistryMixin.get` raises `ValueError`, not `KeyError`. That breaks
    `test_registry.py:14-17, :41-43, :53-57` and `test_wrapper.py:163-167` in the same commit
    that makes the switch.
  - Lane B therefore takes all four updates in B-2.

> Spec deviation (lanes table): the spec lists `tests/pointcloud/test_registry.py` under lane C.
> The KeyError→ValueError switch happens in B-2, so leaving the test to C would leave B-2 red.
> Lane B edits it. Lane C must not re-edit those three blocks; its reconstructor-side tests are
> separate.

**Row 42: grep proof that only list-returning creators reach `_lc_collate_outputs`.**

```
cd $WT && rtk proxy grep -rn "_lc_collate_outputs" collab_splats
collab_splats/geometry/loop_closure/wrapper.py:297:  raw_lc = self.base._lc_collate_outputs(raw) if isinstance(raw, list) else raw
collab_splats/pointcloud/feedforward/base.py:1100:   def _lc_collate_outputs(self, raw: Any) -> Any:
collab_splats/pointcloud/feedforward/mapanything.py:301: def _lc_collate_outputs(self, raw: list[dict]) -> dict:
```

- The only call site is guarded by `isinstance(raw, list)`.
- Only MapAnything's `_forward` returns a list. VGGT-X, Omega and LoGeR return a dict, so the
  base no-op is unreachable.
- LoGeR is refused for LC at the reconstructor (`test_loop_closure_with_loger_is_refused`).
- The `test_wrapper.py` hits (`:346, 398, 528, 629, 836, 1003, 1200, 1268`) assign
  `base._lc_collate_outputs = lambda r: r` on a mock. They never read the base default.
- Action (B-18): delete the base default and keep the `isinstance` call site. The spec says
  "abstract where LC-capable". An `@abstractmethod` on the base would force LoGeR, which is not
  LC-capable, to stub it. The list-only call site makes a single concrete override on
  MapAnything the whole contract.

> Spec deviation (row 42): "abstract where LC-capable" is realized as "defined only where
> `_forward` returns a list". An abstract base method would force a dead stub on 3 of 4 creators,
> which the reduction directive forbids.

**Row 5 evidence:** the caller confirmed with a scan that no stored zarr carries a `conf` key,
so B-6 deletes the fallback outright.

**Rows that moved to "Proposed additions":**
- row 46
- the vggtx `hasattr(images, "cpu")` guess
- MapAnything `_conf is None`
- the wrapper colors heuristic

Each could change output. Evidence is under the final heading.

---

### Task B-1: package `__init__`s — hard imports, drop re-exports, dividers  (spec rows: 1, 85, 2, 36, 37, 86)

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py:1-84`
- Modify: `collab_splats/pointcloud/feedforward/__init__.py:1-64`
- Modify: `collab_splats/pointcloud/feedforward/base.py:51,337,795,974,1049` (divider style only)
- Modify: `collab_splats/geometry/loop_closure/wrapper.py:25`
- Test: `tests/geometry/loop_closure/test_wrapper.py:97-101`

Pure import and structure work; Steps 1-2 are skipped. The re-export deletions break exactly one
test patch target.

- [ ] **Step 3: Implement**

`feedforward/__init__.py` — replace the whole body after the docstring. Rows 1/85 make Omega and
LoGeR hard imports. Row 36 drops the `_raw_to_world_points` re-export and row 37 the
`unproject_and_filter_points` re-export.

```python
"""
Feedforward pointcloud creators: VGGT-X, MapAnything, VGGT-Omega, LoGeR.

- import from here; the submodule layout is an implementation detail
"""

from __future__ import annotations

########################################################
########## Public types and utilities ##################
########################################################

from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    MultiviewConfidence,
    build_pycolmap_reconstruction,
    compute_multiview_depth_confidence,
    multiview_mask,
)

########################################################
########## Concrete creators ###########################
########################################################

from .loger import LoGeRCreator
from .mapanything import MapAnythingCreator
from .vggt_omega import VGGTOmegaCreator
from .vggtx import VGGTXCreator

__all__ = [
    "BaseFeedforwardCreator",
    "FeedforwardResult",
    "LoGeRCreator",
    "MapAnythingCreator",
    "MultiviewConfidence",
    "VGGTOmegaCreator",
    "VGGTXCreator",
    "build_pycolmap_reconstruction",
    "compute_multiview_depth_confidence",
    "multiview_mask",
]
```

`pointcloud/__init__.py:10-35`:
- rows 1/85 delete the two `try/except ImportError` blocks and both `_*_AVAILABLE` flags
- row 86 deletes `from .sfm import ColmapCreator, HlocCreator` (:12) and the two names in
  `__all__` (:77-78)

What remains after this task:

```python
from .base import BasePointcloudCreator, PointcloudResult
from .feedforward import (
    BaseFeedforwardCreator,
    LoGeRCreator,
    MapAnythingCreator,
    VGGTOmegaCreator,
    VGGTXCreator,
)

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "loger": LoGeRCreator,
    "mapanything": MapAnythingCreator,
    "vggt_omega": VGGTOmegaCreator,
    "vggtx": VGGTXCreator,
}
```

Also in `pointcloud/__init__.py`:
- delete the docstring's line 6 ("the optional backbones register only when their dependencies
  import")
- `get_creator`'s `Args:` becomes `name: mapanything, vggtx, vggt_omega or loger.`
- `get_creator` and `make_creator` are otherwise unchanged; B-2 rewrites them

`ff/base.py` row 2: replace each `# ── Title ───…` line (:51, :337, :795, :974, :1049) with the
three-line `########` block:

```python
########################################################
########## Output type #################################
########################################################
```

The titles are Output type, Geometry helpers, COLMAP reconstruction builders, In-memory frame
decoding and Abstract pipeline. Pad each middle line with `#` to 56 characters.

`geometry/loop_closure/wrapper.py:25`:
```python
# before
from collab_splats.pointcloud.feedforward import FeedforwardResult, _raw_to_world_points
# after
from collab_splats.pointcloud.feedforward.base import FeedforwardResult, _raw_to_world_points
```

`tests/geometry/loop_closure/test_wrapper.py:97-101`:
- the patch target `collab_splats.pointcloud.feedforward.unproject_and_filter_points` no longer
  exists
- it was also vacuous: `VGGTXCreator._postprocess` calls its own module global, and the patch's
  `return_value` is a 2-tuple where the real function returns 3
- delete the `with patch(...)` wrapper and dedent the body:

```python
    # before
    with patch(
        "collab_splats.pointcloud.feedforward.unproject_and_filter_points",
        return_value=(np.zeros((5, 3), dtype=np.float32), np.zeros((5, 3), dtype=np.uint8)),
    ):
        result = creator._postprocess(raw_outputs)
    # after
    result = creator._postprocess(raw_outputs)
```

If `patch` has no other use in the file, remove it from the imports. Check with
`rtk proxy grep -n "patch(" tests/geometry/loop_closure/test_wrapper.py`; it has other uses at
HEAD, so it stays.

- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_registry.py tests/geometry/loop_closure/test_wrapper.py`

  All pass.

- [ ] **Step 5: Gates** — G′ delta 0. P → `SUMMARY PASS`.
- [ ] **Step 6: Commit**

```
cd $WT && git add collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py collab_splats/pointcloud/feedforward/base.py collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_wrapper.py && git commit --only collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/__init__.py collab_splats/pointcloud/feedforward/base.py collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_wrapper.py -m "refactor(pointcloud): hard-import every feedforward backend, drop private re-exports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task B-2: `BaseFeedforwardCreator` is the registry  (spec rows: 35, 72 — creator side)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:1053` (class line), plus a `_registry` classvar
- Modify: `collab_splats/pointcloud/feedforward/{vggtx,vggt_omega,loger,mapanything}.py` (decorators)
- Modify: `collab_splats/pointcloud/__init__.py` (`_REGISTRY` deleted, `get_creator` delegates)
- Test: `tests/pointcloud/test_registry.py:14-17, 41-43, 53-57` (delete :14-17 and :53-57, edit :41-43)
- Test: `tests/geometry/loop_closure/test_wrapper.py:163-167`

- [ ] **Step 1: Write the failing test** — delete the two history guards (spec row 106, moved here from C-8 so they are not rewritten and then deleted): `test_sfm_backends_left_the_feedforward_registry` (2 params, :14-17) and `test_old_keys_removed` (:53-57). Edit the remaining two blocks to expect `ValueError`:

```python
def test_get_creator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown 'nonexistent'"):
        get_creator("nonexistent")

```

Add one test that pins the registry's contents:

```python
def test_registry_holds_exactly_the_feedforward_backends():
    assert set(BaseFeedforwardCreator._registry) == {"loger", "mapanything", "vggt_omega", "vggtx"}
```

It needs `from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator` at the top.

`test_wrapper.py:163-167`:
```python
def test_make_creator_unknown_name():
    from collab_splats.pointcloud import make_creator

    with pytest.raises(ValueError, match="Unknown 'unknown_backend'"):
        make_creator("unknown_backend")
```

- [ ] **Step 2: Run, expect FAIL**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_registry.py tests/geometry/loop_closure/test_wrapper.py::test_make_creator_unknown_name`

  Expected failures:
  - 2 failures with `KeyError: "unknown pointcloud backend ..."`: unknown and make_creator
  - `test_registry_holds_exactly_the_feedforward_backends` fails with `AttributeError: type object 'BaseFeedforwardCreator' has no attribute '_registry'`

- [ ] **Step 3: Implement**

`ff/base.py` imports: add `from collab_splats.utils.torch_utils import RegistryMixin` to the
existing torch_utils import line, if there is one; otherwise add it as a new import.

```python
# before (:1053)
class BaseFeedforwardCreator(BasePointcloudCreator):
# after
class BaseFeedforwardCreator(BasePointcloudCreator, RegistryMixin):
```

Place the classvar first in the class body, after the docstring:

```python
    # Registry of feedforward backends; each creator module registers itself on import
    _registry: ClassVar[dict[str, type["BaseFeedforwardCreator"]]] = {}
```

Each creator class line gets a decorator:

| File | Decorator |
|---|---|
| vggtx.py | `@BaseFeedforwardCreator.register("vggtx")` above `@dataclass` |
| vggt_omega.py | `@BaseFeedforwardCreator.register("vggt_omega")` |
| loger.py | `@BaseFeedforwardCreator.register("loger")` |
| mapanything.py | `@BaseFeedforwardCreator.register("mapanything")` |

The decorator sits outermost, so it registers the dataclass-processed class.

`pointcloud/__init__.py`:
- delete the `_REGISTRY` dict
- drop the four concrete-creator names from the `.feedforward` import unless `__all__` lists
  them; it lists `MapAnythingCreator` and `VGGTXCreator`, so keep those two
- importing `.feedforward` imports all four modules, which registers them

```python
from .base import BasePointcloudCreator, PointcloudResult
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator


def get_creator(name: str) -> type[BaseFeedforwardCreator]:
    """
    Look up a feedforward creator class by registry name.

    Args:
        name: mapanything, vggtx, vggt_omega or loger.

    Returns:
        The creator class.

    Raises:
        ValueError: listing the available names, when the key is unknown.
    """
    return BaseFeedforwardCreator.get(name)
```

- `make_creator` is unchanged. Its `Returns` stays "The creator instance".
- Update the module docstring's line 4-5 bullet to:
  `- feedforward backbones register on BaseFeedforwardCreator; get_creator / make_creator resolve them`
- Keep the rest.

> Compatibility: `reconstructor.py:525-530` (`creator_map`) and `:78` (`_FEEDFORWARD_BACKENDS`)
> are untouched. Lane C (row 35 reconstructor side) swaps them to `get_creator` /
> `set(BaseFeedforwardCreator._registry)`. `test_loger_creator.py:920` asserts
> `"loger" in _FEEDFORWARD_BACKENDS` and stays green either way.

- [ ] **Step 4: Targeted** — the Step 2 command → all pass.
- [ ] **Step 5: Gates** — G′ passed −2 (+1 `test_registry_holds_exactly_the_feedforward_backends`; −3 deleted `test_sfm_backends_left_the_feedforward_registry[colmap]`, `[hloc]`, `test_old_keys_removed`). P → `SUMMARY PASS`.
- [ ] **Step 6: Commit**

```
cd $WT && git add collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/base.py collab_splats/pointcloud/feedforward/vggtx.py collab_splats/pointcloud/feedforward/vggt_omega.py collab_splats/pointcloud/feedforward/loger.py collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_registry.py tests/geometry/loop_closure/test_wrapper.py && git commit --only collab_splats/pointcloud/__init__.py collab_splats/pointcloud/feedforward/base.py collab_splats/pointcloud/feedforward/vggtx.py collab_splats/pointcloud/feedforward/vggt_omega.py collab_splats/pointcloud/feedforward/loger.py collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_registry.py tests/geometry/loop_closure/test_wrapper.py -m "refactor(pointcloud): BaseFeedforwardCreator owns the backend registry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task B-3: `PointcloudResult` K and extrinsics via shared helpers  (spec row: 38)

**Files:**
- Modify: `collab_splats/pointcloud/base.py:141-146` (extrinsics loop body), `:163-172` (intrinsics loop body)
- Test: `tests/pointcloud/test_base.py`

Output-neutral merge; Steps 1-2 are replaced by an API probe. `pycolmap.Camera.calibration_matrix()`
is not yet called anywhere in the tree, so confirm it first:

- [ ] **Step 1: Probe the API against a known-true case**

```
cd $WT && $PY -c "import pycolmap,numpy as np;c=pycolmap.Camera(model='SIMPLE_RADIAL',width=40,height=30,params=[50.,20.,15.,0.1]);print(np.asarray(c.calibration_matrix()))"
```

- Expected: `[[50,0,20],[0,50,15],[0,0,1]]`, where the radial term is not in K.
- If `calibration_matrix` is missing, record a spec deviation: keep the canonical-accessor block
  and delete only the comment at :164-165.

- [ ] **Step 3: Implement**

```python
# before (:141-146)
            R = img.cam_from_world().rotation.matrix()
            t = img.cam_from_world().translation
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3] = t
            result.append(E)
# after
            result.append(extrinsics_to_homogeneous(img.cam_from_world().matrix()).astype(np.float32))
```

```python
# before (:163-172)
            cam = cameras[img.camera_id]
            # Use pycolmap canonical accessors — work for all camera models
            # (SIMPLE_PINHOLE has one focal length; PINHOLE has fx/fy separately)
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            result.append(K)
# after
            # K from pycolmap: correct for every model, distortion excluded
            result.append(np.asarray(cameras[img.camera_id].calibration_matrix(), dtype=np.float32))
```

Add the import `from collab_splats.geometry.transforms import extrinsics_to_homogeneous`.
- Check for a cycle first: `collab_splats.geometry.transforms` imports only numpy/torch.
  Verify with `rtk proxy grep -n "^from\|^import" collab_splats/geometry/transforms.py`.
- `extrinsics_to_homogeneous` must accept a single (3, 4). If it only takes (N, 3, 4), use
  `extrinsics_to_homogeneous(img.cam_from_world().matrix()[None])[0]`.

Add a regression test for the SIMPLE_RADIAL path, which colmap/hloc reach:

```python
def test_intrinsics_simple_radial_drops_distortion():
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=40, height=30, params=[50.0, 20.0, 15.0, 0.1], camera_id=1)
    recon.add_camera(cam)
    img = pycolmap.Image(name="a.png", camera_id=1, image_id=1)
    img.cam_from_world = pycolmap.Rigid3d()
    recon.add_image(img)
    recon.register_image(1)
    result = PointcloudResult(reconstruction=recon, image_paths=[Path("a.png")])
    np.testing.assert_array_equal(result.intrinsics[0], [[50, 0, 20], [0, 50, 15], [0, 0, 1]])
    np.testing.assert_array_equal(result.extrinsics[0], np.eye(4, dtype=np.float32))
```

Mirror the image construction already used in `tests/pointcloud/test_base.py`: it builds recons
via `build_pycolmap_reconstruction`, and the Image/Rigid3d construction style should match
whatever pycolmap version that file uses.

- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_base.py`

- [ ] **Step 5: Gates** — G′ +1. P → PASS.
- [ ] **Step 6: Commit**

```
cd $WT && git add collab_splats/pointcloud/base.py tests/pointcloud/test_base.py && git commit --only collab_splats/pointcloud/base.py tests/pointcloud/test_base.py -m "refactor(pointcloud): PointcloudResult K/extrinsics via calibration_matrix + extrinsics_to_homogeneous

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task B-4: delete `FeedforwardResult.save` / `.load`  (spec row: 4)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:91-145`
- Modify: `tests/integration/test_pipeline_cu121.py:100-120`

Pure deletion, so Steps 1-2 are skipped.

- [ ] **Step 3: Implement**
  - Delete `save` (:91-117) and `load` (:119-145) entirely.
  - In `test_pipeline_cu121.py:100-120`, replace the `result.save(path)` / `FeedforwardResult.load(path)` round-trip with `result.save_zarr(path)` / `load_zarr(path)`. Keep the assertion lines. `load_zarr` is `from collab_splats.pointcloud.feedforward.base import load_zarr`, which is the module-level function at :234.
  - Grep for leftovers: `cd $WT && rtk proxy grep -rn "\.save(\|FeedforwardResult.load\|\.load(" collab_splats/pointcloud tests/pointcloud tests/integration evals | grep -iv "zarr\|np\.\|torch\.\|json\|yaml"`. Expected: empty.
- [ ] **Step 4: Targeted** — `tests/integration/test_pipeline_cu121.py` is outside G′, so run it explicitly:

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/integration/test_pipeline_cu121.py`

  Pass, or skip on missing CUDA exactly as before. Diff the skip count against a pre-edit run.
- [ ] **Step 5: Gates** — G′ delta 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): drop FeedforwardResult.save/load; save_zarr/load_zarr is the one format`, with the paths `collab_splats/pointcloud/feedforward/base.py tests/integration/test_pipeline_cu121.py`, same command shape as B-1.

---

### Task B-5: delete the `features` field and `load_features`  (spec row: 39)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:82` (field)
- Modify: `collab_splats/pointcloud/feedforward/base.py:242` (param), `:259` (docstring), `:279` (read)
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:393`, `vggt_omega.py` `features=None`, `loger.py` `features=None`, `mapanything.py` `features=` line — the `features=None` kwargs
- Modify: `collab_splats/dashboard/app.py:604`, `collab_splats/dashboard/pipeline.py:494`, `collab_splats/dashboard/viewer.py:51`
- Test: `tests/dashboard/test_app.py:393`, `tests/dashboard/test_viewer_lift.py:76`, `tests/pointcloud/feedforward/test_load_zarr_flags.py:43`

Pure deletion, so Steps 1-2 are skipped.

- [ ] **Step 3: Implement**
  - Delete the `features:` field (:82) and its docstring bullet.
  - Delete the `load_features: bool = …` param (:242), its `Args:` entry (:259) and the `features = ...` line (:279). Drop `features=features` from the `FeedforwardResult(...)` return in `load_zarr`.
  - Delete `features=None,` from the four `_postprocess` returns. MapAnything's line is inside its `FeedforwardResult(` call; find it with `rtk proxy grep -n "features=" collab_splats/pointcloud/feedforward/*.py`.
  - Dashboard: at each of the 3 sites, delete the `load_features=...` kwarg from the `load_zarr(...)` call, and any `.features` read that follows. Read ±10 lines at each site first; if a site branches on `result.features is not None`, delete the dead branch.
  - Tests:
    - `test_load_zarr_flags.py:43`: drop the `load_features` argument. If the test only exists to exercise `load_features`, delete it; list its name in the G′ delta.
    - `test_viewer_lift.py:76`: drop the kwarg from the stub's `load_zarr` signature or its assertion.
    - `test_app.py:393`: its test named for `load_features` asserts the dashboard passes the flag, so delete it.
  - Leftover grep: `cd $WT && rtk proxy grep -rn "load_features\|\.features\b\|features=None" collab_splats/pointcloud collab_splats/dashboard tests/pointcloud tests/dashboard`. Expected: empty. `tests/semantics/test_artifact_layout.py:16` is an unrelated name and is not in this path set.
- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/feedforward/test_load_zarr_flags.py tests/dashboard/test_viewer_lift.py tests/dashboard/test_app.py`

  `tests/dashboard` is outside G′. Then run the dashboard smoke gate from memory, `serve.py --smoke`: `cd $WT && PYTHONPATH=$WT $PY -m collab_splats.dashboard.serve --smoke`. Use the exact invocation from `collab_splats/dashboard/serve.py`'s docstring.
- [ ] **Step 5: Gates** — G′: −1 if the `load_features` flag test in `test_load_zarr_flags.py` is deleted (name it), else 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): drop the never-populated FeedforwardResult.features`, covering all files above.

---

### Task B-6: delete the legacy `conf` zarr key fallback  (spec row: 5)

**Files:** Modify `collab_splats/pointcloud/feedforward/base.py:283-285`

Pure deletion; the caller's scan confirmed that no stored zarr has `conf`.

- [ ] **Step 3: Implement**

```python
# before (:283-285)
    # Legacy stores used the "conf" key; honour both under the same flag.
    _conf_key = "confidence" if "confidence" in store else ("conf" if "conf" in store else None)
    confidence = torch.from_numpy(store[_conf_key][:]) if (load_confidence and _conf_key) else None
# after
    confidence = torch.from_numpy(store["confidence"][:]) if (load_confidence and "confidence" in store) else None
```

- [ ] **Step 4: Targeted** — `tests/pointcloud/feedforward/test_load_zarr_flags.py` passes.
  - Also grep `rtk proxy grep -rn '"conf"' tests/pointcloud/feedforward/test_load_zarr_flags.py tests/pointcloud/test_feedforward*.py`.
  - A test that writes a `conf` key and expects it back is a history guard. Delete it and name it in the delta.
- [ ] **Step 5: Gates** — G′ delta 0, or −n naming any legacy-key test. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): load_zarr reads only the confidence key`.

---

### Task B-7: `save_zarr` uses `to_numpy`  (spec row: 75)

**Files:** Modify `collab_splats/pointcloud/feedforward/base.py:200-207` (confidence), `:222-230` (images)

Output-neutral, so Steps 1-2 are skipped. `to_numpy` casts bf16 to fp32 exactly as the inline
blocks do.

- [ ] **Step 3: Implement** — replace each inline `if isinstance(x, torch.Tensor): x = x.detach()…float()…cpu().numpy()` block (read the exact text at :200-207 and :222-230) with `x = to_numpy(x)`.
  - Keep the dtype cast that follows, if present (e.g. `.astype(np.float16)` for storage).
  - Add `to_numpy` to the `collab_splats.utils.torch_utils` import.
  - Before editing, confirm H1 handles `np.ndarray` input as identity: `rtk proxy grep -n "def to_numpy" -A12 collab_splats/utils/torch_utils.py`. If it does not, keep an `isinstance(x, torch.Tensor)` guard at the call site.
- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/feedforward/test_load_zarr_flags.py tests/pointcloud/test_feedforward_preprocess_store.py`

- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): save_zarr tensor conversion via to_numpy`.

---

### Task B-8: delete `_source_paths`  (spec row: 11)

**Files:** Modify `collab_splats/pointcloud/feedforward/base.py:1196-1207`; Test `tests/pointcloud/test_preprocess_frames.py:108`

- [ ] **Step 3: Implement**
  - Delete the staticmethod (:1196-1207).
  - Confirm there are no callers: `cd $WT && rtk proxy grep -rn "_source_paths" collab_splats tests evals`. Expect only `test_preprocess_frames.py:108`.
  - Delete that test function.
  - If `fr` (the `collab_splats.preproc.frames` alias) is now unused in ff/base.py, drop its import. Check with `rtk proxy grep -n "fr\." collab_splats/pointcloud/feedforward/base.py`.
- [ ] **Step 4: Targeted** — `tests/pointcloud/test_preprocess_frames.py` passes.
- [ ] **Step 5: Gates** — G′ −1: `test_preprocess_frames.py::<the test at :108>` (name it from the file). P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): drop the uncalled _source_paths`.

---

### Task B-9: delete creator-side `reproject` / `_reproject`  (spec row: 10)

**Files:**
- Modify: `feedforward/base.py:1392-1426`
- Modify: `vggtx.py:407-434`
- Modify: `vggt_omega.py:326-340`
- Modify: `loger.py:515-531`
- Modify: `mapanything.py:53-94` (`_reproject_mapanything`), `:596-616`
- Delete: `tests/pointcloud/test_mapanything.py` (2 tests, both on `_reproject_mapanything`)
- Test stubs: `tests/geometry/loop_closure/test_feedforward_lc_state.py:51`, `tests/geometry/loop_closure/test_wrapper.py:34, 118-137, 274-290`, `tests/geometry/loop_closure/test_verify_lc_data.py:240`, `tests/pointcloud/test_feedforward_preprocess_store.py:39`, `tests/pointcloud/test_feedforward_shared.py:79`, `tests/test_feedforward_logging.py:47`
- Test reproject tests: `tests/pointcloud/test_vggt_omega_creator.py:488, 601`, `tests/pointcloud/test_loger_creator.py:601, 623, 663`, `tests/pointcloud/feedforward/test_mapanything_creator.py:238, 264, 288`

`FeedforwardResult.reproject()` already covers this path: `evals/scripts/eval.py:417` calls it on
the result. Pure deletion, so Steps 1-2 are skipped.

- [ ] **Step 3: Implement**
  - Delete the base `reproject` and the abstract `_reproject` (:1392-1426), and each creator's `_reproject`, plus `_reproject_mapanything` (mapanything.py:53-94).
  - Drop the imports these leave unused. After editing, check each file with `rtk proxy grep -n "unproject_depth_map_to_point_map\|replace\b"`. Candidates:
    - `dataclasses.replace` in base
    - `unproject_depth_map_to_point_map` in vggtx, which is still used by `unproject_and_filter_points` (keep)
  - Remove the `_reproject` method or assignment from each test stub listed above.
    - `test_wrapper.py:34` sets `_reproject.return_value` on a mock: delete that line.
  - Delete `test_wrapper.py:118-137` (`test_vggtx_has_reproject`, `test_mapanything_has_reproject`) and `:274-290` (the `lc.reproject` test).
  - Delete the per-creator reproject tests at the lines listed. Read each `def test_…` name and put it in the delta.
  - `git rm tests/pointcloud/test_mapanything.py`
  - Leftover grep: `cd $WT && rtk proxy grep -rn "_reproject\b\|\.reproject(\|_reproject_mapanything\|_reproject_ba" collab_splats tests evals`.
    - Allowed hits: `FeedforwardResult.reproject`, its tests, and `eval.py:417`.
    - `geometry/bundle_adjustment.py:96, 218` docstrings mention `creator.reproject`. R1-5 already fixes them; do not edit.
    - `docs/.../bundle_adjustment.ipynb` is not edited (no notebook edits).
  - (extra) Delete the dead `_reproject_ba` from `_StubCreator` in `test_feedforward_shared.py:60-80`.
- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/geometry/loop_closure tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_feedforward_preprocess_store.py tests/pointcloud/test_vggt_omega_creator.py tests/pointcloud/test_loger_creator.py tests/pointcloud/feedforward/test_mapanything_creator.py tests/test_feedforward_logging.py`

- [ ] **Step 5: Gates** — G′ −(2 + 2 + 1 + 2 + 3 + 3) = −13: test_mapanything.py ×2, test_wrapper ×3, omega ×2, loger ×3, mapanything ×3.
  - Name each test from the files.
  - Recount if any listed line is a helper rather than a test.
  - `test_feedforward_logging.py` is outside G′: run it separately (above).
  - P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): drop creator-side reproject; FeedforwardResult.reproject is the one path`. Use `git rm` for `tests/pointcloud/test_mapanything.py`.

---

### Task B-10: `build_pycolmap_reconstruction` — one extrinsics shape, uint8 colors, known models  (spec rows: 14, 44)

**Files:** Modify `feedforward/base.py:798-876`; Test `tests/pointcloud/test_feedforward_shared.py:21-47`

- [ ] **Step 1: Write the failing tests** (append to `test_feedforward_shared.py`):

```python
def test_build_pycolmap_rejects_unknown_camera_model():
    ext, intr, pts, cols, names = _make_inputs()
    with pytest.raises(ValueError, match="camera_model"):
        build_pycolmap_reconstruction(ext, intr, pts, cols, names, camera_model="OPENCV")


def test_build_pycolmap_rejects_float_colors():
    ext, intr, pts, cols, names = _make_inputs()
    with pytest.raises(TypeError, match="uint8"):
        build_pycolmap_reconstruction(ext, intr, pts, cols.astype(np.float32) / 255.0, names)
```

Before writing, read `_make_inputs` (:40-55) and match its real return order and the build
function's real parameter names. The snippet above assumes positional order
`(extrinsics, intrinsics, points, colors, image_names)`; if the names differ, use keywords
exactly as `_make_inputs` feeds them.

- [ ] **Step 2: Run, expect FAIL**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_feedforward_shared.py -k "rejects"`

  Expected: `DID NOT RAISE` ×2.
  - The OPENCV call silently takes the SIMPLE_PINHOLE else-branch.
  - The float colors get scaled.

- [ ] **Step 3: Implement**

```python
# before
    exts = extrinsics[:, :3, :] if extrinsics.shape[1] == 4 else extrinsics
# after
    # (N,3,4) or (N,4,4) w2c: the top three rows are the same either way
    exts = extrinsics[:, :3, :]
```

```python
# before
    colors_u8 = colors if colors.dtype == np.uint8 else (np.clip(colors, 0, 1) * 255).astype(np.uint8)
# after
    if colors.dtype != np.uint8:
        raise TypeError(f"colors must be uint8, got {colors.dtype}")
```

Rename later uses of `colors_u8` to `colors`.

```python
# before
    if camera_model == "PINHOLE":
        params = [fx, fy, cx, cy]
    else:  # SIMPLE_PINHOLE
        params = [(fx + fy) / 2, cx, cy]
# after
    if camera_model == "PINHOLE":
        params = [fx, fy, cx, cy]
    elif camera_model == "SIMPLE_PINHOLE":
        params = [(fx + fy) / 2, cx, cy]
    else:
        raise ValueError(f"camera_model must be PINHOLE or SIMPLE_PINHOLE, got {camera_model!r}")
```

Add a `Raises:` section to the docstring.

> Spec deviation (row 14 "one extrinsics shape"): slicing `[:, :3, :]` unconditionally accepts
> both shapes without a guess. It is a no-op on (N,3,4), and callers pass both:
> `eval_verification.py:75` passes `cam_from_world().matrix()` (3,4), while
> `reconstructor.py:1342` and `FeedforwardResult` pass (N,4,4). Forcing one shape would add
> conversions at 2+ call sites for no output change.

- Every in-tree caller passes uint8. Verify with `rtk proxy grep -rn "build_pycolmap_reconstruction(" collab_splats evals tests` and read each call's colors source:
  - FeedforwardResult colors are uint8
  - `eval_verification` takes `recon` colors
- Any float caller found is fixed at the call site with `(np.clip(c, 0, 1) * 255).astype(np.uint8)` and listed in the commit body.
- `test_accepts_4x4_extrinsics` (:27-33) stays; it pins the slice.

- [ ] **Step 4: Targeted** — `tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_base.py tests/geometry/test_verification.py` pass.
- [ ] **Step 5: Gates** — G′ +2. P → PASS.
- [ ] **Step 6: Commit** — `fix(pointcloud): build_pycolmap_reconstruction raises on unknown camera model / non-uint8 colors`.

---

### Task B-11: `_rescale_reconstruction_to_original_dimensions` — drop dead kwargs, raise on non-pinhole  (spec rows: 12, 45)

**Files:**
- Modify: `feedforward/base.py:879-971` and the `build_colmap` call (~:1285)
- Modify: `wrapper/reconstructor.py:1351-1353` (one argument; cross-lane)
- Test: `tests/pointcloud/test_feedforward_intrinsics.py:191-222`

- [ ] **Step 1: Write the failing test** (append to `test_feedforward_intrinsics.py`):

```python
def test_rescale_rejects_non_pinhole_camera():
    recon = pycolmap.Reconstruction()
    recon.add_camera(pycolmap.Camera(model="OPENCV", width=20, height=10, params=[10, 10, 10, 5, 0, 0, 0, 0], camera_id=1))
    coords = np.array([[0, 0, 40, 20, 40, 20]], dtype=np.float32)
    with pytest.raises(ValueError, match="PINHOLE"):
        _rescale_reconstruction_to_original_dimensions(recon, coords, (20, 10))
```

- [ ] **Step 2: Run, expect FAIL**

  `... tests/pointcloud/test_feedforward_intrinsics.py::test_rescale_rejects_non_pinhole_camera`

  Expected: `TypeError: _rescale_reconstruction_to_original_dimensions() missing 1 required positional argument` (the signature still takes `image_paths`).

- [ ] **Step 3: Implement** — new signature and body. Read :879-971 and keep the existing per-image `points2D` / coords logic verbatim; only the pieces below change.

```python
def _rescale_reconstruction_to_original_dimensions(
    reconstruction: pycolmap.Reconstruction,
    original_image_sizes: np.ndarray,
    image_size: tuple[int, int],
) -> pycolmap.Reconstruction:
```

Changes:
- delete the `image_paths`, `shared_camera`, `shift_point2d_to_original_res` and `verbose` params, their `Args:` entries, and every branch they gate
  - `shared_camera` / `shift_point2d_to_original_res` / `verbose` are False at both callers: `build_colmap` :1285 and `reconstructor.py:1351`
- delete `pyimage.name = image_paths[pyimageid - 1].name`
  - `build_pycolmap_reconstruction` already names the images, so the assignment is idempotent
- delete `pred_params = copy.deepcopy(pycamera.params)`
  - pycolmap `params` returns a copy; write `pred_params = np.asarray(pycamera.params, dtype=np.float64)`
- the model switch becomes:

```python
        if pycamera.model.name == "SIMPLE_PINHOLE":
            pred_params[0] *= max(scale_x, scale_y)
        elif pycamera.model.name == "PINHOLE":
            pred_params[0] *= scale_x
            pred_params[1] *= scale_y
        else:
            raise ValueError(f"rescale supports PINHOLE / SIMPLE_PINHOLE, got {pycamera.model.name}")
        pred_params[-2] *= scale_x
        pred_params[-1] *= scale_y
```

  - Match the existing access idiom: if HEAD reads `pycamera.model_name`, keep `model_name`.
- delete the dead nested `rescale_camera` helper, if present in :879-971
- drop `import copy` (:14); it has no other use (`rtk proxy grep -n "copy\." collab_splats/pointcloud/feedforward/base.py`)

Callers:
```python
# ff/base.py build_colmap — before
    _rescale_reconstruction_to_original_dimensions(recon, o.image_paths, o.original_coords, (o.model_width, o.model_height))
# after
    _rescale_reconstruction_to_original_dimensions(recon, o.original_coords, (o.model_width, o.model_height))
```

```python
# wrapper/reconstructor.py:1351-1353 — before
        recon, ff.image_paths, ff.original_coords, (ff.model_width, ff.model_height)
# after
        recon, ff.original_coords, (ff.model_width, ff.model_height)
```

> Spec deviation (lane boundary): `reconstructor.py` is lane C, but row 12 changes this function's
> signature, and the call must move in the same commit or G′ goes red. One argument is removed,
> nothing else. Lane C rebases over it.

`test_feedforward_intrinsics.py:191-222` `_rescaled_camera_params`: drop the positional
`[Path("0.png")]` argument.

- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_feedforward_intrinsics.py tests/wrapper/test_reconstructor.py -k "rescale or colmap or intrinsics"`

- [ ] **Step 5: Gates** — G′ +1. P → PASS.
- [ ] **Step 6: Commit** — `fix(pointcloud): rescale raises on non-pinhole models; drop always-default kwargs`, with the paths `collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_feedforward_intrinsics.py`.

---

### Task B-12: multiview confidence — no silent CPU fallback  (spec row: 15)

**Files:** Modify `feedforward/base.py:532-541` (signature), `:588`; Test `tests/pointcloud/feedforward/test_mapanything_creator.py` or `tests/pointcloud/test_multiview_confidence.py`. Use whichever file holds the `compute_multiview_depth_confidence` tests: `rtk proxy grep -rln "compute_multiview_depth_confidence" tests`.

- [ ] **Step 1: Write the failing test**

```python
def test_multiview_confidence_raises_on_cuda_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    depth = np.ones((2, 4, 6), dtype=np.float32)
    K = np.tile(np.array([[5, 0, 3], [0, 5, 2], [0, 0, 1]], dtype=np.float32), (2, 1, 1))
    E = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    with pytest.raises(RuntimeError, match="CUDA"):
        compute_multiview_depth_confidence(depth, K, E, device="cuda")
```

- [ ] **Step 2: Run, expect FAIL** — `DID NOT RAISE`; the fallback silently picks CPU.
- [ ] **Step 3: Implement**

```python
# signature before
    device: str = "cuda",
# after
    device: str | None = None,
```

```python
# before (:588)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
# after
    # None picks the best device; an explicit cuda request without CUDA is an error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("compute_multiview_depth_confidence: device='cuda' but CUDA is unavailable")
    dev = torch.device(device) if device is not None else get_device()
```

- Import `get_device` from torch_utils.
- `Args:` for `device` becomes `torch device; None picks cuda when available.`

> Spec deviation (row 15): the default was `"cuda"`, and every caller relies on the default. A
> bare raise would make every CPU-only run and the CPU parity harness fail. Changing the default
> to `None → get_device()` keeps current behavior for callers. Only an explicit `"cuda"` on a
> CUDA-less host raises, which is the silent fallback the row targets.

- [ ] **Step 4: Targeted** — the new test plus the existing mv tests pass.
- [ ] **Step 5: Gates** — G′ +1. P → PASS; the harness runs on CPU with the default.
- [ ] **Step 6: Commit** — `fix(pointcloud): multiview confidence raises instead of silently falling back to CPU`.

---

### Task B-13: `_verify_loop_candidate` — ratio required, no `layer_index`, missing poses raise; drop the wrapper guard  (spec rows: 9, 47)

**Files:**
- Modify: `feedforward/base.py:1341-1390`
- Modify: `geometry/loop_closure/wrapper.py:365-367` (call), `:375-384` (guard), `:404-407` (`.get`)
- Test: `tests/pointcloud/test_feedforward_shared.py:118-200`
- Test: `tests/geometry/loop_closure/test_verify_lc_data.py:212-246, 296, 308`
- Test: `tests/geometry/loop_closure/test_loop_closure_integration.py:94-96`

The spec requires one commit.

- [ ] **Step 1: Write the failing test** — replace `test_verify_accepted_no_poses` (:132) with:

```python
def test_verify_raises_when_backend_supplies_no_poses():
    creator = _make_stub()
    creator.extract_intermediate_features = lambda frames, layer_index, **kw: {
        "q": torch.ones(1, 1, 8, 4),
        "k": torch.ones(1, 1, 8, 4),
    }
    frame = torch.zeros(3, 4, 4)
    with pytest.raises(KeyError, match="poses"):
        creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.0)
```

- Match the q/k shapes and the `_make_stub` model-device plumbing used by the neighbouring `test_verify_with_poses` (:148). Copy its fixture lines, and change only the returned dict and the expectation.
- Delete `test_verify_layer_index_forwarded` (:186). The argument no longer exists; forwarding of `_lc_layer_index` is pinned in the next step.
- Rename and adapt it as:

```python
def test_verify_taps_the_calibrated_layer():
    creator = _make_stub()
    seen = {}

    def _extract(frames, layer_index, **kw):
        seen["layer"] = layer_index
        return {"q": torch.ones(1, 1, 8, 4), "k": torch.ones(1, 1, 8, 4), "poses": np.zeros((2, 4, 4), np.float32)}

    creator.extract_intermediate_features = _extract
    type(creator)._lc_layer_index = 7
    frame = torch.zeros(3, 4, 4)
    creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.0)
    assert seen["layer"] == 7
```

- `type(creator)` is `_StubCreator`, which is local to the test module; setting the classvar there does not leak.

- [ ] **Step 2: Run, expect FAIL**
  - `test_verify_raises_when_backend_supplies_no_poses`: `DID NOT RAISE` (it returns `(True, None)`)
  - `test_verify_taps_the_calibrated_layer`: passes already. It is a characterization pin, stated as such.
- [ ] **Step 3: Implement**

```python
    def _verify_loop_candidate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        verify_match_ratio: float,
        **kwargs: Any,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Gate a loop candidate on cross-frame attention; on accept, return the joint poses.

        Args:
            frame1: first preprocessed frame, (C, H, W).
            frame2: second preprocessed frame, (C, H, W).
            verify_match_ratio: accept threshold on cross_frame_attention_ratio.
            **kwargs: forwarded to extract_intermediate_features.

        Returns:
            (accepted, lc_data): lc_data is None on reject, else {"poses": (2, 4, 4) w2c,
            "world_points": (2, H, W, 3) | None, "conf": (2, H, W) | None}.
        """
        # Capture cross-frame activations at the calibrated layer, on the model's device
        device = next(self.model.parameters()).device
        features = self.extract_intermediate_features(
            torch.stack([frame1, frame2]).to(device), self._lc_layer_index, **kwargs
        )

        # Gate on the cross-frame attention ratio
        ratio = cross_frame_attention_ratio(features["k"], features["q"], token_offset=self._lc_token_offset)
        accepted = ratio >= verify_match_ratio
        logger.info("LC verify: ratio=%.4f threshold=%.4f accepted=%s", ratio, verify_match_ratio, accepted)
        if not accepted:
            return False, None

        # Accepting backends must supply joint poses; geometry is optional
        return True, {
            "poses": features["poses"],
            "world_points": features.get("world_points"),
            "conf": features.get("conf"),
        }
```

> Note: row 47 ("index directly") is applied to `poses` here. `world_points` and `conf` stay
> `.get` until B-32 makes MapAnything always emit them (spec rows 17/57, silent paths
> `:586-593`). B-32's last step then switches both to direct indexing.

`wrapper.py`:
```python
# :365-367 before
            verify_ok, lc_data = self.base._verify_loop_candidate(
                q_frame, d_frame, verify_match_ratio=cfg.verify_match_ratio
            )
# unchanged — the kwarg is still the required ratio
```

Delete :375-384 (the `if verify_ok and lc_data is None:` block with `logger.error` and
`no_joint_poses`). The base now raises, so the branch is unreachable.

- Check whether `"no_joint_poses"` is referenced elsewhere: `rtk proxy grep -rn "no_joint_poses" collab_splats tests`.
- Delete the matching tests at `test_verify_lc_data.py:296` and `:308` (both drive the guard through a stub returning `(True, None)`). Name them in the delta.
- Update the stub at `:212-246` to drop `layer_index` from its `extract_intermediate_features` signature, if it declares it with a default.

`wrapper.py:122-127`: `verify_match_ratio` still resolves from `base.default_verify_match_ratio`. That part is B-14.

`test_loop_closure_integration.py:94-96`:
- the call relied on the 0.85 signature default
- pass `verify_match_ratio=1.46` (the MapAnything calibration it runs against) explicitly

`test_feedforward_shared.py`:
- `test_verify_with_poses` (:148) and `test_verify_is_abstract` (:203): drop any `layer_index=` argument
- the `_StubCreator.extract_intermediate_features` signature becomes `(self, frames, layer_index, **kwargs)`

`evals/scripts/eval.py:735` sets `_lc_layer_index` on the class, which is still honored.

- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_feedforward_shared.py tests/geometry/loop_closure`

- [ ] **Step 5: Gates** — G′ net −2:
  - −1 `test_verify_accepted_no_poses`, −1 `test_verify_layer_index_forwarded`
  - +1 `test_verify_raises_when_backend_supplies_no_poses`, +1 `test_verify_taps_the_calibrated_layer`
  - −2 for the `test_verify_lc_data.py:296, :308` tests (name them)
  - P → PASS.
- [ ] **Step 6: Commit** — `fix(geometry): LC verify requires the ratio and raises on missing poses; drop the wrapper guard`, with the paths `collab_splats/pointcloud/feedforward/base.py collab_splats/geometry/loop_closure/wrapper.py tests/pointcloud/test_feedforward_shared.py tests/geometry/loop_closure/test_verify_lc_data.py tests/geometry/loop_closure/test_loop_closure_integration.py`.

---

### Task B-14: SPARK-era LC classvars default to `None` and raise when unset  (spec row: 8)

**Files:**
- Modify: `feedforward/base.py:1085-1098`
- Modify: `vggtx.py:186`, `vggt_omega.py:138-142`
- Modify: `geometry/loop_closure/wrapper.py:122-127`
- Test: `tests/pointcloud/test_feedforward_shared.py`

- [ ] **Step 1: Write the failing test**

```python
def test_lc_classvars_unset_raise():
    creator = _make_stub()
    frame = torch.zeros(3, 4, 4)
    with pytest.raises(NotImplementedError, match="_lc_layer_index"):
        creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.0)


def test_loop_closure_raises_without_calibrated_ratio():
    base = _make_stub()
    with pytest.raises(NotImplementedError, match="default_verify_match_ratio"):
        LoopClosure(base)
```

- `_StubCreator` must not set the classvars.
- Import `LoopClosure` from `collab_splats.geometry`.
- If `test_verify_taps_the_calibrated_layer` (B-13) sets `_lc_layer_index` on `_StubCreator` itself, change it to set the classvar on a local subclass, so the order of these two tests does not matter:
  `class _S(type(creator)): _lc_layer_index = 7; _lc_token_offset = 0` then `creator.__class__ = _S`.

- [ ] **Step 2: Run, expect FAIL** — `DID NOT RAISE` ×2; base defaults 20 / 0.85 are used.
- [ ] **Step 3: Implement**

```python
# base.py :1085-1098 after
    # LC calibration; each LC-capable creator sets all three (see docs/parity.md)
    # - default_verify_match_ratio: cross_frame_attention_ratio accept threshold
    # - _lc_layer_index: cross-frame block tapped for Q/K
    # - _lc_token_offset: special tokens before the patch tokens (VGGT 5, MapAnything 0)
    default_verify_match_ratio: ClassVar[float | None] = None
    _lc_layer_index: ClassVar[int | None] = None
    _lc_token_offset: ClassVar[int | None] = None
```

At the top of `_verify_loop_candidate`:
```python
        # Refuse an uncalibrated backend
        if self._lc_layer_index is None or self._lc_token_offset is None:
            raise NotImplementedError(f"{type(self).__name__} sets no _lc_layer_index/_lc_token_offset; LC unsupported")
```

`wrapper.py:122-127`:
```python
        # None resolves to the creator's per-model calibration
        if self.config.verify_match_ratio is None:
            if base.default_verify_match_ratio is None:
                raise NotImplementedError(f"{type(base).__name__} sets no default_verify_match_ratio; LC unsupported")
            self.config = dataclasses.replace(self.config, verify_match_ratio=base.default_verify_match_ratio)
```

Creators:
- `vggtx.py:186-187`: add `_lc_token_offset: ClassVar[int] = 5` beside the two existing classvars.
- `vggt_omega.py:138-142`: same. Replace the comment "token_offset=5 (inherited)" with "token_offset=5: VGGT camera + 4 register tokens".
- `mapanything.py:159, 168-169`: unchanged; it sets all three.
- LoGeR sets none. That is correct: it is refused for LC.

> Spec deviation (row 8): the spec says "default None, raise when unset". VGGT-X and Omega
> currently inherit `_lc_token_offset = 5` from the base, so they must now declare it or they
> would start raising. Two one-line additions, output-neutral.

- [ ] **Step 4: Targeted** — `tests/pointcloud/test_feedforward_shared.py tests/geometry/loop_closure tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_vggt_omega_creator.py` pass. `test_wrapper.py` mocks set the attributes explicitly: grep `default_verify_match_ratio` in `test_wrapper.py`.
  - A `MagicMock` base has a truthy mock attribute, not `None`, so it does not raise.
- [ ] **Step 5: Gates** — G′ +2. P → PASS.
- [ ] **Step 6: Commit** — `fix(pointcloud): LC calibration classvars default to None and raise when unset`.

---

### Task B-15: `extract_intermediate_features` — base raises, `layer_index` required, LoGeR stub gone  (spec row: 43)

**Files:** Modify `feedforward/base.py:1314-1339`, `vggtx.py:436`, `vggt_omega.py:342`, `mapanything.py:503`, `loger.py:533-559`; Test `tests/pointcloud/test_loger_creator.py:708`, `test_feedforward_shared.py:203`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_intermediate_features_base_raises():
    creator = _make_stub()
    with pytest.raises(NotImplementedError):
        BaseFeedforwardCreator.extract_intermediate_features(creator, torch.zeros(2, 3, 4, 4), 0)
```

This replaces `test_verify_is_abstract` (:203), which asserts the method is abstract.

- [ ] **Step 2: Run, expect FAIL** — the abstract body `...` returns `None`, so `DID NOT RAISE`.
- [ ] **Step 3: Implement** — base:

```python
    def extract_intermediate_features(self, frames: torch.Tensor, layer_index: int, **kwargs: Any) -> dict[str, Any]:
        """
        Q/K at one cross-frame block, plus joint poses, for LC verification.

        - the hook is removed in a finally block, so it survives a raising forward

        Args:
            frames: (2, C, H, W) preprocessed frames on the model's device.
            layer_index: cross-frame block to tap.
            **kwargs: backend forward kwargs (MapAnything: minibatch_size, memory_efficient_inference).

        Returns:
            {"q", "k": (B, heads, tokens, head_dim), "poses": (2, 4, 4) w2c,
            optional "world_points": (2, H, W, 3), "conf": (2, H, W)}.

        Raises:
            NotImplementedError: the backend does not support loop closure.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support loop closure")
```

- Remove `@abstractmethod`.
- Delete `loger.py:533-559` (the stub that raises). The base now raises the same way.
- `test_loger_creator.py:708` ("refuses") still expects `NotImplementedError`; check that its `match=` still fits the base message and adjust the string.
- In `vggtx.py:436`, `vggt_omega.py:342` and `mapanything.py:503`, change the signature `layer_index: int = -1` to `layer_index: int`.
  - Each creator's `-1` branch is now dead: `rtk proxy grep -n "layer_index == -1\|layer_index < 0\|\[-1\]" …`. If a creator resolves `-1` to the last block, keep negative indexing, since Python list indexing handles it, and delete only an explicit `if layer_index == -1` rewrite.
- [ ] **Step 4: Targeted** — `tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_loger_creator.py tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_vggt_omega_creator.py tests/pointcloud/feedforward/test_mapanything_creator.py` pass.
- [ ] **Step 5: Gates** — G′ 0 (−`test_verify_is_abstract` +`test_extract_intermediate_features_base_raises`). P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): extract_intermediate_features raises on the base; layer_index required`.

---

### Task B-16: `_raw_to_world_points` and MapAnything collate raise on missing keys  (spec row: 6)

**Files:** Modify `feedforward/base.py:373-424` (guard :391-392), `mapanything.py:356-368`; Test `tests/pointcloud/feedforward/test_mapanything_creator.py:843`, `tests/pointcloud/test_loger_creator.py` (`_raw_to_world_points` tests)

Same commit per the spec.

- [ ] **Step 1: Write the failing tests**

```python
def test_raw_to_world_points_raises_on_missing_depth():
    raw = {"extrinsic": np.zeros((1, 3, 4), np.float32), "intrinsics": np.eye(3, dtype=np.float32)[None]}
    with pytest.raises(KeyError, match="depth"):
        _raw_to_world_points(raw)
```

This goes in `test_mapanything_creator.py` beside the `:821` use. Write `"intrinsics_downsampled"` here if B-17 has not landed; B-17 renames it.

Replace `test_lc_collate_outputs_warns_when_depth_z_missing` (:843) with the raising form:

```python
def test_lc_collate_outputs_raises_when_depth_z_missing():
    creator = _make_creator()
    # same fixture lines as the deleted test, with depth_z removed from the processed dicts
    ...
    with pytest.raises(KeyError, match="depth_z"):
        creator._lc_collate_outputs(raw)
```

Copy the deleted test's body verbatim: its fixture construction (:843-880). Keep the `depth_z`
removal and replace the `caplog` assertion with `pytest.raises`. The `...` above marks where those
verbatim lines go; it is not code to write.

- [ ] **Step 2: Run, expect FAIL**
  - the first test gets `(None, None)` back, `DID NOT RAISE`
  - the second gets a warning, `DID NOT RAISE`
- [ ] **Step 3: Implement**
  - base: delete :390-392 (the guard comment and `if not all(...): return None, None`). Direct indexing raises `KeyError` naming the key.
  - `Returns:` drops "or (None, None) if required keys are absent", and the return annotation becomes `tuple[np.ndarray, np.ndarray | None]`.
  - Callers that tested `is not None`: the vggtx, omega and loger `_postprocess` (`if world_pts_flat is not None: … else None`) collapse to `world_points = world_pts_flat.reshape(...)`. B-27 deletes those bodies anyway, so do the collapse here only if B-27 has not landed.
  - wrapper :319 (`wp, wp_conf = _raw_to_world_points(raw_lc)`) is unchanged.
  - mapanything :356-368:

```python
# after
        # Depth and confidence feed the submap's world points; every postprocess variant emits them
        out["depth"] = np.stack([p["depth_z"][0].cpu().float().numpy() for p in processed])
        out["depth_conf"] = np.stack([p["conf"][0].cpu().float().numpy() for p in processed])
        return out
```

  - This replaces the `if all(...) … else: logger.warning(...)` block.
  - The wrapper's `if "depth" in raw_lc and "depth_conf" in raw_lc:` (:334) stays true for every backend now. The guard itself is listed under Proposed additions, because vggtx raw always has both keys.
  - Delete `test_mapanything_creator.py:881` ("warns") as well, if it is a second warn test distinct from :843. Read both names and list each.
- [ ] **Step 4: Targeted** — `tests/pointcloud/feedforward/test_mapanything_creator.py tests/pointcloud/test_loger_creator.py tests/geometry/loop_closure` pass.
- [ ] **Step 5: Gates** — G′ +1, net of the rename of :843 (+1 new raise test; 0 for the replaced one; −1 for each additional warn test deleted, named). P → PASS.
- [ ] **Step 6: Commit** — `fix(pointcloud): _raw_to_world_points and MapAnything LC collate raise on missing depth`.

---

### Task B-17: one intrinsics key — drop the `intrinsics_downsampled` alias  (spec row: 7)

**Files:**
- Modify: `vggtx.py:275, 317, 337`
- Modify: `vggt_omega.py:247, 280`
- Modify: `loger.py:431, 466-470`
- Modify: `mapanything.py:308, 340-348`
- Modify: `base.py:382, 391, 396`

Tests to update:
- `tests/pointcloud/test_vggtx_creator.py:222`
- `tests/pointcloud/test_vggt_omega_creator.py:37, 227, 638`
- `tests/pointcloud/test_loger_creator.py:438`
- `tests/pointcloud/test_feedforward_intrinsics.py:42`
- `tests/pointcloud/feedforward/test_mapanything_creator.py:782, 797, 802`
- `tests/geometry/loop_closure/test_wrapper.py:94, 434, 462`
- `tests/integration/test_pipeline_cu121.py:46-54`

Output-neutral: every producer binds both keys to the same array. The parity harness binds both
too, so it keeps working, and the extra key becomes inert. Steps 1-2 are skipped.

- [ ] **Step 3: Implement**
  - producers: delete the `"intrinsics_downsampled": …` line from each `_forward` return dict and from the mapanything collate dict (:347), along with the alias comments (:342-344, loger :466-470 comment block)
  - consumers:
    - `_raw_to_world_points` reads `raw["intrinsics"]`; docstring :382 updated
    - vggtx :337 `intrinsic = raw_outputs["intrinsics"]`
    - omega :280 `intrinsic=intrinsic`
  - vggtx docstring :275: drop "``intrinsics_downsampled`` (alias of intrinsics),"
  - tests:
    - delete the `"intrinsics_downsampled"` line from each fixture dict
    - `test_vggt_omega_creator.py:227`: drop it from `required_keys`
    - `test_loger_creator.py:438`: delete the alias assertion line
    - `test_mapanything_creator.py:797-802`: delete the alias equality lines; fix the docstring :782
    - `test_wrapper.py:434` docstring: `intrinsics_downsampled` → `intrinsics`
    - `test_pipeline_cu121.py:46-54`: the docstring describes the removed `.get` fallback; rewrite it to one line, "raw outputs carry one `intrinsics` key", and delete the key at :54
  - leftover grep: `cd $WT && rtk proxy grep -rn "intrinsics_downsampled" collab_splats tests evals`. Expected: empty.
- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud tests/geometry/loop_closure tests/integration/test_pipeline_cu121.py`

- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): raw outputs carry one intrinsics key`.

---

### Task B-18: delete the base `_lc_collate_outputs` no-op  (spec row: 42)

**Files:** Modify `feedforward/base.py:1100-1106`

Evidence is the grep in "Lane-level decisions": the only caller is list-guarded, and only
MapAnything returns a list. Pure deletion.

- [ ] **Step 3: Implement**
  - Delete :1100-1106.
  - MapAnything's override (:301) keeps its docstring. Fix its first line to state that it is called only for list outputs.
- [ ] **Step 4: Targeted** — `tests/geometry/loop_closure tests/pointcloud/feedforward/test_mapanything_creator.py tests/pointcloud/feedforward/test_lc_collate_window.py` pass.
- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): drop the unreachable base _lc_collate_outputs`.

---

### Task B-19: `rich.console` → `logging` in feedforward  (spec row: 13)

**Files:** Modify `feedforward/base.py:31, 47` + every `console.log`; `vggtx.py:32`; `mapanything.py:42, 264, 278, 289`; `loger.py`, `vggt_omega.py` (grep); Test `tests/test_feedforward_logging.py:52-139`

- [ ] **Step 1: Rewrite the tests** — each capsys test (:52-139) switches to caplog:

```python
def test_load_model_logs_device(caplog):
    creator = _make_stub()
    with caplog.at_level(logging.INFO, logger="collab_splats.pointcloud.feedforward.base"):
        creator.load_model(device="cpu")
    assert "Loading model (cpu)" in caplog.text
```

- Apply the same transformation to each test in :52-139. Keep the asserted substrings, swap `capsys.readouterr().out` for `caplog.text`, and wrap the call in `caplog.at_level(logging.INFO, logger=<module>)`.
- The patch target at :106 becomes `collab_splats.pointcloud.feedforward.base.build_pycolmap_reconstruction`.
- [ ] **Step 2: Run, expect FAIL**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/test_feedforward_logging.py`

  Expected: `AssertionError` (`caplog.text` is empty; rich writes to stdout).
- [ ] **Step 3: Implement**
  - Replace every `console.log(f"…")` with `logger.info("…%s…", arg)`, using lazy `%` args.
  - Delete `from rich.console import Console` and `console = Console()` in base.
  - Delete the unused `console` imports in vggtx (:32) and mapanything (:42), and in loger / omega if present: `rtk proxy grep -n "console" collab_splats/pointcloud/feedforward/*.py`.
  - The MapAnything `_forward` device logs (:264/:278/:289) become `logger.debug`.
  - `geometry/loop_closure/wrapper.py` `console.log` is out of scope; it is a spec follow-up.
- [ ] **Step 4: Targeted** — the Step 2 command → pass.
- [ ] **Step 5: Gates** — G′ 0 (the logging tests are outside G′). P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): feedforward logs via logging, not rich console`.

---

### Task B-20: frame labels via `frame_name`  (spec rows: 51, 81 — creator side)

**Files:** Modify `vggtx.py:250`, `vggt_omega.py:207`, `loger.py:343`, `mapanything.py:202`, `feedforward/base.py:1000` (`_STORE_FRAME_NAME` if it formats rather than parses)

Output-neutral.

- [ ] **Step 3: Implement**
  - Each `_preprocess` builds `[Path(f"frame_{i:06d}.png") for i in frame_idxs]` (read the exact form at each line). Change it to `[Path(f"{frame_name(i)}.png") for i in frame_idxs]`, or `Path(frame_name(i)).with_suffix(".png")` if H3 returns a bare stem.
  - Import `from collab_splats.preproc.frames import frame_name`.
  - `_STORE_FRAME_NAME` (:1000) is a parse regex. Keep it; `re` stays.
  - `reconstructor.py:1195` belongs to lane C.
- [ ] **Step 4: Targeted** — `tests/pointcloud/test_preprocess_frames.py tests/pointcloud/test_feedforward_preprocess_store.py` pass.
- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): creator frame labels via preproc.frames.frame_name`.

---

### Task B-21: device resolution via `get_device`  (spec row: 73)

**Files:** Modify `feedforward/base.py:1169` (`load_model`); `mapanything.py` extract device loop (:554-558) only if it resolves a device string

- [ ] **Step 3: Implement**

```python
# before (:1169)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
# after
        device = device or str(get_device())
```

Keep `str(...)`: `_load_model(device: str)` takes a string, and `get_device()` returns
`torch.device`. Confirm with `rtk proxy grep -n "def get_device" -A8 collab_splats/utils/torch_utils.py`.
- [ ] **Step 4-5** — G′ 0; P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): load_model resolves its device via get_device`.

---

### Task B-22: `run_inference` frees memory via `pytorch_gc`  (spec row: 82)

**Files:** Modify `feedforward/base.py:1230-1232`

- [ ] **Step 3: Implement**

```python
# before
        # Clear GPU cache after forward pass to free memory before postprocessing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# after
        # Free forward-pass memory before postprocessing
        pytorch_gc()
```

- `pytorch_gc` runs `gc.collect()` + `empty_cache` under `is_available`. Check with `rtk proxy grep -n "def pytorch_gc" -A10 collab_splats/utils/torch_utils.py`.
- The extra `gc.collect()` is output-neutral.
- [ ] **Step 4-5** — G′ 0; P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): run_inference frees memory via pytorch_gc`.

---

### Task B-23: LoGeR vendored import via `vendored_path`  (spec row: 74)

**Files:** Modify `loger.py:37` (`import sys`), `:251-283`

- [ ] **Step 3: Implement** — replace the manual `sys.path.insert` / `try` / `finally: sys.path.remove` block (:251-283) with:

```python
        # Vendored LoGeR model tree (setup/loger.sh); optional heavy dep, imported here
        with vendored_path(_LOGER_ROOT, "run setup/loger.sh to vendor LoGeR into third_party/"):
            from loger.models.pi3 import Pi3
```

- Keep the exact hint string from HEAD's existing `ImportError` message.
- Drop `import sys` if it has no other use.
- The inline import is permitted: it is an optional heavy dep, per CLAUDE.md.
- [ ] **Step 4: Targeted** — `tests/pointcloud/test_loger_creator.py` passes; its `_load_model` tests stub the vendored tree.
- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): LoGeR vendored import via vendored_path`.

---

### Task B-24: whole-frame crop coords via `full_frame_coords`  (spec row: 78)

**Files:** Modify `loger.py:349-351`, plus any creator building `[0, 0, W, H, W, H]` rows by hand (`rtk proxy grep -n "0, 0, " collab_splats/pointcloud/feedforward/*.py`)

- [ ] **Step 3: Implement** — replace the hand-built array at loger :349-351 with `original_coords = full_frame_coords(W, H, len(frames))`. Use the exact W/H/N names at those lines.
- [ ] **Step 4-5** — `tests/pointcloud/test_loger_creator.py`; G′ 0; P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): whole-frame crop coords via full_frame_coords`.

---

### Task B-25: Omega and LoGeR checkpoints via `load_hf_weights`  (spec rows: 55, 83 — creator side)

**Files:** Modify `vggt_omega.py:20, 190-195`, `loger.py:45, 285`; Test `tests/pointcloud/test_vggt_omega_creator.py:328-384`, `tests/pointcloud/test_loger_creator.py` (hf patch targets)

- [ ] **Step 3: Implement** — at each site, replace `hf_hub_download(repo_id=self.model_repo, filename=self.model_filename)` plus the `torch.load` / `safetensors.load_file` that follows (read the exact lines) with `state = load_hf_weights(self.model_repo, self.model_filename)`.
  - Drop the `hf_hub_download` imports.
  - Tests that patch `...vggt_omega.hf_hub_download` switch their target to `...vggt_omega.load_hf_weights`, returning the state dict directly. Apply the same to loger.
  - If `load_hf_weights` handles only one of `.safetensors` / `.pt`, and a creator uses the other format, record a spec deviation and leave that creator unchanged. Check with `rtk proxy grep -n "def load_hf_weights" -A20 collab_splats/utils/torch_utils.py`.
- [ ] **Step 4: Targeted** — omega and loger creator tests pass.
- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): Omega/LoGeR checkpoints via load_hf_weights`.

---

### Task B-26: `unproject_and_filter_points` moves to `ff/base.py`  (spec row: 76)

**Files:**
- Modify: `vggtx.py:91-153` (move out), `vggt_omega.py:37`, `loger.py:61` (import source)
- Modify: `feedforward/base.py` (new home under the Geometry helpers divider)
- Tests (patch targets):
  - `tests/pointcloud/test_vggt_omega_creator.py:282, 312, 501, 592, 614`
  - `tests/integration/test_pipeline_cu121.py:146`
  - `tests/pointcloud/test_feature_lifting.py:7`
  - `tests/pointcloud/test_vggtx_creator.py:177`

Pure move; Steps 1-2 are skipped.

- [ ] **Step 3: Implement**
  - Cut the function verbatim into `ff/base.py`. Its imports come along: `randomly_limit_trues`, `unproject_depth_map_to_point_map` from vggt.
  - ff/base.py then hard-imports `vggt`. That is acceptable: every creator module already does, and row 1 makes them all hard imports.
  - (extra) Delete the ROADMAP comment (:120-123), which describes future work, not code.
  - Omega and LoGeR import from `.base`. vggtx imports it back from `.base` for its own `_postprocess`, until B-27 deletes that.
  - Patch targets:
    - `...vggt_omega.unproject_and_filter_points` keeps working until B-27, because Omega's `_postprocess` resolves the name in its own module
    - after B-27, the name resolves in `ff.base`, so this task rewrites those five patch targets (and `test_pipeline_cu121.py:146`) to `collab_splats.pointcloud.feedforward.base.unproject_and_filter_points` now
    - to keep this commit green, B-26 and B-27 land back-to-back; the targets are updated in B-27 (see there)
    - direct imports (`test_feature_lifting.py:7`, `test_vggtx_creator.py:177`) switch to `from collab_splats.pointcloud.feedforward.base import unproject_and_filter_points`
  - MapAnything:
    - its `_postprocess` open-codes the same mask → `randomly_limit_trues` → gather tail at :463-468
    - folding it onto this function is B-32 via `_mask_to_points`
- [ ] **Step 4: Targeted** — `tests/pointcloud` passes.
- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): unproject_and_filter_points lives on the feedforward base`.

---

### Task B-27: one concrete `_postprocess` on the base for the VGGT family  (spec rows: 16, 77)

**Files:**
- Modify: `feedforward/base.py` (new concrete `_postprocess`, `_multiview`, mv fields)
- Delete from `vggtx.py:322-405`, `vggt_omega.py:252-324`, `loger.py:436-513` (each `_postprocess`)
- Move the mv fields: `vggtx.py:197-208`, `vggt_omega.py:155-166`, `loger.py:202-211` → base
- Modify: `mapanything.py:447-460` (use `_multiview`)
- Tests: patch targets as listed in B-26

The three `_postprocess` bodies are the same recipe. Diffs at HEAD, all output-neutral:
- vggtx and omega read `raw.get("extrinsic_global_4x4", E4)`. Nothing writes that key (`rtk proxy grep -rn "extrinsic_global_4x4" collab_splats` → reads only), so it is dead.
- vggtx `.get("intrinsics_downsampled", …)`. B-17 removed it.
- LoGeR passes `intrinsic` where Omega passes the alias. It is the same array.

The mv fields have identical defaults in all three (`use_multiview_confidence=False`,
`min_views=2`, `mv_conf_abs_thresh=0.0`, `mv_conf_rel_thresh=0.01`). MapAnything overrides them
(True, 1, 0.02, 0.02).

- [ ] **Step 3: Implement** — base fields, after `max_points`:

```python
    # Geometric cross-view depth consistency (compute_multiview_depth_confidence)
    # - min_views: at least K other views agree
    # - abs/rel thresholds: depth tolerance in meters / fraction of expected depth
    use_multiview_confidence: bool = False
    min_views: int = 2
    mv_conf_abs_thresh: float = 0.0
    mv_conf_rel_thresh: float = 0.01
```

- Check the dataclass field ordering: every base field has a default, so subclass fields with defaults can follow.
- The creators' per-field calibration comments (vggtx :198-208, omega :156-166) differ in wording only. Keep the vggtx wording in the base, and delete the per-creator copies and fields.

Base helper and `_postprocess`:

```python
    def _multiview(
        self, depth: np.ndarray, intrinsics: np.ndarray, extrinsics_4x4: np.ndarray, valid: np.ndarray
    ) -> tuple[MultiviewConfidence | None, np.ndarray]:
        """
        Apply the cross-view depth-consistency filter when enabled.

        Args:
            depth: (N, H, W) model-res depth.
            intrinsics: (N, 3, 3) K at depth resolution.
            extrinsics_4x4: (N, 4, 4) w2c.
            valid: (N, H, W) pixels eligible as sources and targets.

        Returns:
            (mv_conf or None when disabled, valid ANDed with the min_views gate).
        """
        if not self.use_multiview_confidence:
            return None, valid
        mv_conf = compute_multiview_depth_confidence(
            depth,
            intrinsics,
            extrinsics_4x4,
            depth_masks=valid,
            abs_thresh=self.mv_conf_abs_thresh,
            rel_thresh=self.mv_conf_rel_thresh,
        )
        return mv_conf, multiview_mask(mv_conf, valid, min_views=self.min_views)

    def _postprocess(self, raw_outputs: dict[str, Any]) -> FeedforwardResult:
        """
        Unproject a depth-head raw dict into the filtered cloud and BA fields.

        - raw keys: images, extrinsic (N,3,4), intrinsics (N,3,3), depth (N,H,W[,1]), depth_conf
        - MapAnything overrides: its raw output is a per-view list

        Args:
            raw_outputs: the dict _forward returned.

        Returns:
            FeedforwardResult at model resolution.
        """
        extrinsic = raw_outputs["extrinsic"]
        intrinsic = raw_outputs["intrinsics"]
        extrinsic_4x4 = extrinsics_to_homogeneous(extrinsic)
        depth = raw_outputs["depth"]
        depth = depth.squeeze(-1) if depth.ndim == 4 else depth
        model_h, model_w = int(depth.shape[1]), int(depth.shape[2])

        # Optional cross-view depth consistency; extra_mask is None when disabled
        mv_conf, mv_valid = self._multiview(depth, intrinsic, extrinsic_4x4, depth > 0)
        mv_mask = mv_valid if mv_conf is not None else None

        # Filtered world-space cloud and per-point colors
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
        )

        # BA fields: dense world-point grid at model resolution
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)

        return FeedforwardResult(
            points=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            extrinsics=extrinsic_4x4,
            intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=raw_outputs["images"],
            confidence=torch.from_numpy(raw_outputs["depth_conf"]),
            world_points=world_pts_flat.reshape(-1, model_h, model_w, 3),
            depth=depth,
            **_mv_result_fields(mv_conf),
        )
```

Notes:
- `postprocess(**kwargs)` (:1236) calls `self._postprocess(self.raw_outputs, **kwargs)`. B-33 drops kwargs; here the base signature keeps `**kwargs: Any` until B-33, to stay compatible with MapAnything's.
- `conf_threshold` is a field on every VGGT-family creator (vggtx 35.0, omega 50.0, loger 50.0). The base does not declare it; the concrete `_postprocess` reads `self.conf_threshold`. Declare `conf_threshold: float = 50.0` on the base and keep the vggtx override (35.0). Omega and LoGeR delete theirs, which is the same value.
  - Their docstring `Attributes:` entries differ. LoGeR's explains percentile-vs-probability (:174), so keep that text in the base docstring.
- `depth_masks=valid` with `valid = depth > 0`: `compute_multiview_depth_confidence` ANDs `depth_masks` with `depth > 0` internally for sources, so passing `depth > 0` equals passing nothing. Verify with `rtk proxy sed -n 590,640p collab_splats/pointcloud/feedforward/base.py`, looking for `src_valid`. If the internal rule differs, pass `depth_masks=None` when `valid` is exactly `depth > 0`: add a `depth_masks: np.ndarray | None` parameter to `_multiview` and have MapAnything pass its mask.
- Delete the three creator `_postprocess` methods, and their now-unused imports: `compute_multiview_depth_confidence`, `multiview_mask`, `_mv_result_fields`, `_raw_to_world_points`, `extrinsics_to_homogeneous` — keep any that the creator's `extract`/`_forward` still uses (vggtx and omega `extract` use `extrinsics_to_homogeneous`); grep each file after deleting.
- MapAnything :447-460:

```python
        # Shared geometric mv_conf filter
        mv_conf, combined_mask = self._multiview(
            np.stack(depth_list), np.stack(intrinsics_list), extrinsics_to_homogeneous(np.stack(extrinsics_list)), combined_mask
        )
```

Patch targets move here:
- the five `test_vggt_omega_creator.py` patches (:282, 312, 501, 592, 614) and `test_pipeline_cu121.py:146` become `collab_splats.pointcloud.feedforward.base.unproject_and_filter_points`
- tests that construct a creator and assert `use_multiview_confidence` defaults (omega :88-98) still pass, since the values are inherited

- [ ] **Step 4: Targeted**

  `cd $WT && PYTHONPATH=$WT:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud tests/geometry/loop_closure tests/integration/test_pipeline_cu121.py`

- [ ] **Step 5: Gates** — G′ 0. P → PASS. P is the binding proof: all four creators plus `depth_align` are compared against the goldens.
- [ ] **Step 6: Commit** — `refactor(pointcloud): one base _postprocess for the VGGT-family depth heads`.

---

### Task B-28: shared hook, pose-decode and `_forward` tails  (spec row: 50)

**Files:** Modify `vggtx.py:463-488`, `vggt_omega.py:368-392`, `mapanything.py` hook block in extract (:530-560), `feedforward/base.py` (new helpers)

Output-neutral.

- [ ] **Step 3: Implement**
  - The three extracts install the same forward hook on the tapped block's `attn.qkv` and split q/k. Each is a copy of the same 15-18 lines: vggtx :463-481, omega :368-385. Add to base:

```python
@contextmanager
def capture_qk(qkv: torch.nn.Module, num_heads: int) -> Iterator[dict[str, torch.Tensor]]:
    """
    Capture q and k from a fused QKV projection for the duration of the block.

    Args:
        qkv: the Linear producing (B, T, 3·heads·head_dim).
        num_heads: attention heads of that block.

    Yields:
        dict filled with "q", "k" as (B, heads, T, head_dim) after the forward runs.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_m: torch.nn.Module, _inp: Any, out: torch.Tensor) -> None:
        B, T, _ = out.shape
        q, k, _v = out.reshape(B, T, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
        captured["q"], captured["k"] = q, k

    handle = qkv.register_forward_hook(_hook)
    try:
        yield captured
    finally:
        handle.remove()
```

  - Before writing it, read vggtx :463-481 and copy its reshape/permute exactly. The snippet above must match HEAD's arithmetic, including any `.detach()` or `.float()`. Apply the same check against omega :368-385 and mapanything's hook. Only the arithmetic common to all three goes in the helper; a creator whose split differs keeps its own hook, stated in the commit body.
  - Pose decode, vggtx :483-488 and omega :387-392 (identical): `pose_encoding_to_extri_intri(...)` → `extrinsics_to_homogeneous(...)` → `.astype(np.float32)`. Make it `def _decode_poses(pose_enc: torch.Tensor, hw: tuple[int, int]) -> np.ndarray` in vggtx.py, and have Omega import it from `.vggtx`. Both already import vggt's `pose_encoding_to_extri_intri`.
  - `_forward` tails: vggtx :305-318 and omega :236-248 both end with the same dict build from `pose_encoding_to_extri_intri` + `.cpu().numpy()`. After B-17 dropped the alias line, if the remaining blocks are line-identical except the variable names, factor them into `_depth_head_outputs(images, preds, hw) -> dict` in vggtx.py, imported by Omega. Diff the two blocks first: `diff <(sed -n 296,318p vggtx.py) <(sed -n 228,248p vggt_omega.py)`. Merge only if the diff shows renames only.
- [ ] **Step 4: Targeted** — the vggtx, omega and mapanything creator tests, plus `tests/geometry/loop_closure` (LC verify).
- [ ] **Step 5: Gates** — G′ 0. P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): shared QK hook and pose decode across VGGT-family extracts`.

---

### Task B-29: VGGT-X leftovers  (spec rows: 52, 53, 19)

**Files:** Modify `vggtx.py:32, 42, 48-85, 221-226, 280-288`; Test `tests/pointcloud/test_feedforward_intrinsics.py:152-180`

All three changes are output-neutral.

- [ ] **Step 3: Implement**
  - **Row 52 (inline the autocast dtype).** Each site becomes one expression, and each site keeps its own condition.
    - The two sites are not the same rule at HEAD. `_load_model` checks `torch.cuda.is_available()` and the default device's capability. `_forward` checks `device_type == "cuda"` and that device's capability.
    - Merging them into one rule changes `_load_model` on a CUDA host asked for `device="cpu"`: it gives bf16 today and would give fp16. That is output-changing, so it is not done here.

    ```python
    # _load_model :221-226 before
            # Choose dtype based on GPU capability: bfloat16 for Ampere+, float16 for older
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
    # after
            # bf16 on Ampere+, fp16 otherwise
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    ```

    ```python
    # _forward :280-288 before (comment run + 5-line expression)
    # after
            # Match the aggregator's dtype: bf16 on Ampere+, fp16 otherwise; it asserts tokens.dtype
            dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    ```

    `device_type` (:279) is then unused and is deleted. Lines over 120 characters wrap inside parentheses (black).

  - **Row 53.**
    - Delete the unused `console` import (:32), if B-19 has not already removed it.
    - Delete the false comment run at :281-283: "model params are fp32 …". `_load_model` :230 does `model.to(device, dtype=dtype)`, so the params are bf16/fp16, not fp32. The one-line comment above replaces it.
  - **Row 19.**
    - Delete `VGGTX_IMG_LOAD_RESOLUTION` (:42) and the `target_size` param of `_compute_vggtx_crop_coords` (:48).
    - Inside the function, write `target_size = 518` with the citation comment `# vggt load_and_preprocess_images crop target (facebookresearch/vggt, vggt/utils/load_fn.py)`.
    - Pin the citation with `git show` against the vendored vggt commit in `third_party/` or `pyproject.toml`, per the attribute-ported-code rule. If the pin cannot be resolved, cite the file only.
    - Callers: `rtk proxy grep -rn "VGGTX_IMG_LOAD_RESOLUTION\|_compute_vggtx_crop_coords" collab_splats tests evals`. Drop the `target_size=` kwarg in `test_feedforward_intrinsics.py:152-180`. If a test there exists only to vary `target_size`, delete it and name it.
  - (extra) Drop the imports B-26/B-27 left unused. Check each one with grep: `randomly_limit_trues`, `unproject_depth_map_to_point_map`, `compute_multiview_depth_confidence`, `multiview_mask`, `_mv_result_fields`, `_raw_to_world_points`.
- [ ] **Step 4-5** — `tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_feedforward_intrinsics.py`; G′ 0, or −n for named target_size tests; P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): VGGT-X dtype one expression, crop target inlined, false comment gone`.

---

### Task B-30: VGGT-Omega leftovers  (spec rows: 20, 54)

**Files:**
- Modify: `vggt_omega.py:45-47` (constants), `:144-146`, `:167-174`, `:186-198`
- Test: `tests/pointcloud/test_vggt_omega_creator.py:88-98, 511-575`

- [ ] **Step 3: Implement**
  - **Row 20.** Delete `VGGT_OMEGA_DEFAULT_RESOLUTION = 512` (:47). It has no reader in collab_splats, tests or evals.
  - **Row 54.** Delete the fields `model_repo`, `model_filename` (:145-146) and `enable_text_alignment` (:167). No config, eval or caller outside the tests sets them; the grep over `collab_splats configs tests evals docs` shows test hits only.
    - `_load_model`:

    ```python
    # before (:186-198)
            else:
                ckpt_path = Path(
                    hf_hub_download(
                        repo_id=self.model_repo,
                        filename=self.model_filename,
                    )
                )

            # Instantiate model, load checkpoint weights, move to device in eval mode (fp32 params)
            model = VGGTOmega(enable_alignment=self.enable_text_alignment)
    # after
            else:
                ckpt_path = Path(hf_hub_download(repo_id=VGGT_OMEGA_HF_REPO, filename=VGGT_OMEGA_DEFAULT_FILENAME))

            # Instantiate model, load checkpoint weights, move to device in eval mode (fp32 params)
            model = VGGTOmega(enable_alignment=False)
    ```

    B-25 later swaps `hf_hub_download` for `load_hf_weights`. If B-25 lands first, apply this edit to its form.
    - (extra) With alignment gone, `resolution=None → 256 if alignment else 512` always resolves to 512. Make the field `resolution: int = 512` and delete that branch of `__post_init__` (:172-174) and its `logger.debug`. The `resize_mode` check stays. Output-neutral: the default path resolved to 512 before too.
  - Tests:
    - `:93-94, :97`: delete the three default assertions for the removed fields.
    - Delete `test_enable_text_alignment_auto_sets_resolution_256` (:515) and `test_enable_text_alignment_auto_sets_resolution_512` (:521).
    - Change the test at :528 to `VGGTOmegaCreator(resolution=768)`.
    - The load test at :549 drops `enable_text_alignment=True`. If it asserts that `enable_alignment=True` reached the model, delete it and name it.
    - Update the section header comment at :511.
- [ ] **Step 4-5** — omega creator tests; G′ −2 (the two `test_enable_text_alignment_*` tests), −1 more if the :549 test goes; P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): VGGT-Omega drops never-set repo/filename/alignment fields`.

---

### Task B-31: LoGeR leftovers  (spec rows: 18, 56)

**Files:**
- Modify: `loger.py:72-80, 87, 184-185, 285, 390-393, 420`
- Test: `tests/pointcloud/test_loger_creator.py:17, 441-452, 800-807`
- Modify: `docs/benchmarks/scripts/decompose_residual.py:21, 39`, `docs/benchmarks/scripts/compare_upstream_focal.py:30, 46` (scripts, not notebooks)

- [ ] **Step 1: Write the failing test** (row 56, bare assert → raise):

```python
def test_forward_raises_on_out_of_range_images():
    creator = LoGeRCreator()
    creator.model = torch.nn.Linear(1, 1)  # parameters() supplies the device
    views = torch.full((1, 3, 14, 14), 255.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        creator._forward(creator.model, views)
```

Match how the existing `_forward` tests near :441 construct the creator and views, and use their
stub model if one exists. The assert fires before any model call, so a `Linear` suffices for
`next(model.parameters()).device`.

- [ ] **Step 2: Run, expect FAIL** — `AssertionError: LoGeR expects RGB in [0, 1]; got [255.0, 255.0]`.
- [ ] **Step 3: Implement**
  - Row 56 assert:

    ```python
    # before (:390-393)
            # Guards the a157421 [0,255] bug class at the source rather than at the mesh.
            assert (
                0.0 <= float(images.min()) and float(images.max()) <= 1.0
            ), f"LoGeR expects RGB in [0, 1]; got [{float(images.min())}, {float(images.max())}]"
    # after
            # Guards the a157421 [0,255] bug class at the source rather than at the mesh
            lo, hi = float(images.min()), float(images.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(f"LoGeR expects RGB in [0, 1]; got [{lo}, {hi}]")
    ```

  - Row 56 `model_repo`:
    - delete the field (:185)
    - `:285` becomes `ckpt = self.model_path or hf_hub_download(repo_id=LOGER_HF_REPO, filename=f"{self.variant}/latest.pt")`, or its `load_hf_weights` form if B-25 landed first
  - Row 18: `LOGER_CONF_THRESHOLD` becomes a kwarg.
    - Delete the module constant (:80). Move its comment run (:72-79) onto a new field placed after `conf_threshold`:

    ```python
        # Confidence floor for the K fit, not estimate_intrinsics_from_points' 0.1 default
        # - LoGeR's conf head is uncalibrated: measured logits span -4.257..-2.019, so the
        #   post-sigmoid band is [0.0140, 0.1172] and 0.1 is its 92nd percentile
        # - at 0.1 only 7.9% of pixels survive (12,217/155,232 on an 8-frame run), and a
        #   marginally duller scene keeps none, raising
        # - 0.02 sits just above the band floor, rejecting only what the model calls junk
        # - the confidence WEIGHTING inside the median is what actually discriminates
        # - re-measure if the checkpoint changes
        k_fit_conf_threshold: float = 0.02
    ```

    - `:420` becomes `estimate_intrinsics_from_points(local_points, depth_conf, self.k_fit_conf_threshold)`.
    - Add an `Attributes:` entry to the class docstring.
    - Test and script readers switch to `LoGeRCreator.k_fit_conf_threshold`. It is a dataclass default, so it is readable on the class:
      - `test_loger_creator.py:17` (import), `:452` (`loger_mod.LOGER_CONF_THRESHOLD`), `:801, :807`
      - both `docs/benchmarks/scripts/*.py` import/use pairs
  - Row 18 `_PATCH`: already module-private at HEAD (`_PATCH = 14`, :87).

  > Spec deviation (row 18 "`_PATCH` → private"): already done at HEAD, so there is no action.

- [ ] **Step 4: Targeted** — `tests/pointcloud/test_loger_creator.py` passes. Also check that the two benchmark scripts parse: `cd $WT && $PY -m py_compile docs/benchmarks/scripts/decompose_residual.py docs/benchmarks/scripts/compare_upstream_focal.py`.
- [ ] **Step 5: Gates** — G′ +1. P → PASS; the default 0.02 is unchanged.
- [ ] **Step 6: Commit** — `refactor(pointcloud): LoGeR K-fit floor is a field; range check raises ValueError`.

---

### Task B-32: MapAnything silent paths and dead branches  (spec rows: 17, 57; row 76 tail)

**Files:**
- Modify: `mapanything.py:193-299` (`_load_model`, `_preprocess`, `_forward`), `:375-390`, `:413-439`, `:463-472`, `:528-529`, `:554-558`, `:580-593`
- Test: `tests/pointcloud/feedforward/test_mapanything_creator.py:314` and the extract tests

The spec row lists: silent paths (including `:586-593`), the dead `Tensor` branch, the device loop ×4, `kwargs.get` and the `pts3d` guards.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_raises_when_pts3d_missing(monkeypatch):
    creator = _make_creator()
    # reuse the fixture of the existing extract test; drop "pts3d" from every processed dict
    ...
    with pytest.raises(KeyError, match="pts3d"):
        creator.extract_intermediate_features(frames, 4)


def test_postprocess_raises_when_conf_missing():
    creator = _make_creator()
    # reuse the fixture of the existing _postprocess test; drop "conf" from every pred dict
    ...
    with pytest.raises(KeyError, match="conf"):
        creator._postprocess(raw)
```

- The `...` lines are the verbatim fixture bodies of the neighbouring tests. Copy them at
  execution; list their source lines in the test docstring. Find them with
  `rtk proxy grep -n "def test_.*extract\|def test_.*postprocess" tests/pointcloud/feedforward/test_mapanything_creator.py`.
- Remove only the named key from the fixture.

- [ ] **Step 2: Run, expect FAIL**
  - the first test hits the warning at :591, `DID NOT RAISE`
  - the second takes the `_conf = None` path at :472, `DID NOT RAISE`
- [ ] **Step 3: Implement**
  - **Silent path, :580-593 (extract geometry).** Replace the two `if all(...)` blocks and the warning with:

    ```python
            # Joint pointmaps + confidence feed LC anchor-scale estimation
            captured["world_points"] = np.stack([p["pts3d"][0].cpu().float().numpy() for p in processed])  # (2, H, W, 3)
            captured["conf"] = np.stack([p["conf"][0].cpu().float().numpy() for p in processed])  # (2, H, W)
    ```

  - **Silent path, :472.** `_conf = torch.stack(conf_list) if conf_list else None` becomes `_conf = torch.stack(conf_list)`. The `pred.get("conf")` at :430 becomes `pred["conf"]`, so a missing key raises there.
  - **Dead `Tensor` branch (:252-264).** `_preprocess` returns a list of view dicts (`:200-238`), so `isinstance(views, torch.Tensor)` is never true. Delete it.
    - `test_mapanything_creator.py:314`: if it drives `_forward` with a Tensor, delete it and name it. If it drives the list path, keep it.
  - **Device loop ×4** (`_forward` :265-278 and :279-289, extract :554-558, and the fourth copy found by `rtk proxy grep -n "\.to(device" collab_splats/pointcloud/feedforward/mapanything.py`). Add a module helper:

    ```python
    def _views_to(views: list[dict[str, Any]], device: torch.device | str) -> list[dict[str, Any]]:
        """
        Move every tensor in a list of MapAnything view dicts to `device`.

        Args:
            views: view dicts as built by _preprocess.
            device: target device.

        Returns:
            New view dicts; non-tensor values are passed through.
        """
        return [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in view.items()} for view in views]
    ```

    - Before replacing, read each loop. If one moves only `"img"` or mutates in place and a later line relies on the mutation, keep that loop's semantics: use `views = _views_to(views, device)` at the same point.
  - **`kwargs.get` (:528-529).** `extract_intermediate_features(self, frames, layer_index, minibatch_size: int = …, memory_efficient_inference: bool = …)`, with the defaults copied from the two `.get(…, default)` calls. Drop `**kwargs`.
    - B-13's `_verify_loop_candidate(**kwargs)` still forwards any caller kwargs by name.
  - **`_raw_list` (:375-380).** Delete it: nothing writes that attribute (`rtk proxy grep -rn "_raw_list" collab_splats tests` → reads only).
  - **Guards (:386-390) and ndim squeezes (:413-439).** Keep each squeeze whose input rank is set by upstream MapAnything (`pts3d` (1,H,W,3), `conf` (1,H,W), `depth_z` (1,H,W,1)), written as a direct `[0]` / `[..., 0]` index.
    - Read each one first. Any `if x.ndim == …` whose both branches are reachable is output-relevant, so leave it and list it in the commit body.
  - **Row 76 tail.** Add to `ff/base.py`, beside `unproject_and_filter_points`:

    ```python
    def _mask_to_points(
        points: np.ndarray, colors: np.ndarray, mask: np.ndarray, max_points: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gather masked pixels into a cloud, randomly capped at max_points.

        Args:
            points: (N, H, W, 3) world points.
            colors: (N, H, W, 3) uint8 colors.
            mask: (N, H, W) bool keep-mask.
            max_points: cap on the returned count.

        Returns:
            (P, 3) float32 points, (P, 3) uint8 colors, (P, 3) int32 (frame, row, col) indices.
        """
        if int(mask.sum()) > max_points:
            mask = randomly_limit_trues(mask, max_points)
        pixel_indices = np.stack(np.where(mask), axis=1).astype(np.int32)
        return points[mask].astype(np.float32), colors[mask], pixel_indices
    ```

    - MapAnything :463-468 becomes `pts3d, colors, pixel_indices = _mask_to_points(stacked_pts3d, stacked_colors, combined_mask, self.max_points)`.
    - `unproject_and_filter_points` calls it for its own tail only if its tail is the same three operations in the same order: same `randomly_limit_trues` call, same `>` test and same gather dtype. Read vggtx :130-153 (now in base). If it differs (e.g. `>=`, or it gathers before capping), leave it and note that in the commit body. P decides; a mismatch here changes the RNG draw.
  - (extra) Add docstrings to `_load_model` (:193) and `_preprocess` (:200), which have none at HEAD.
  - Row 47 completion: in `ff/base.py` `_verify_loop_candidate`, `features.get("world_points")` / `features.get("conf")` become `features["world_points"]` / `features["conf"]`. Every LC-capable extract now emits both: vggtx/omega via `_decode_verify_geometry`, MapAnything above. Confirm with `rtk proxy grep -n 'captured\["world_points"\]\|"world_points":' collab_splats/pointcloud/feedforward/*.py`. Also update `wrapper.py` :404-407 (`lc_data.get(...)` + `is not None` reshapes) to direct indexing and an unconditional reshape. Test mocks that return an `lc_data` dict with only `"poses"` would then `KeyError`. Find them with `rtk proxy grep -rn '"poses":' tests/geometry/loop_closure` and add `"world_points": np.zeros((2, H, W, 3), np.float32), "conf": np.ones((2, H, W), np.float32)` sized to each test's H and W. If a test asserts the scale-1.0 fallback that only a missing `world_points` produces, it is a history guard: delete it and name it in the G′ delta.
- [ ] **Step 4: Targeted** — `tests/pointcloud/feedforward tests/geometry/loop_closure` pass.
- [ ] **Step 5: Gates** — G′ +2, −n for any Tensor-branch test deleted (named). P → PASS.
- [ ] **Step 6: Commit** — `fix(pointcloud): MapAnything raises on missing pts3d/conf; drop dead Tensor branch and device loops`.

---

### Task B-33: drop pass-through `**kwargs` on the template steps  (spec row: 41)

**Files:** Modify `feedforward/base.py` `run_inference` (:1221), `postprocess` (:1236), abstract `_forward` / `_postprocess`; the four creators' `_forward` / `_postprocess` signatures

- [ ] **Step 3: Implement**
  - Find every in-tree caller that passes kwargs: `rtk proxy grep -rn "run_inference(\|postprocess(\|_forward(" collab_splats evals tests | grep -v "def "`.
  - `_postprocess` never reads kwargs in any creator: vggtx's docstring says "Unused". Drop them from `postprocess`, `_postprocess` and all overrides.
  - `run_inference(**kwargs)` → `_forward(model, views, **kwargs)`:
    - MapAnything's `_forward` reads `minibatch_size` / `memory_efficient_inference`
    - the LC wrapper :290 passes `**kwargs` to `_forward`
    - if a caller passes them, keep `**kwargs` on `_forward` only, and make MapAnything's explicit keyword params
    - drop them from `run_inference` if no caller passes any
  - Update the parity harness call shape: it calls `postprocess()` with no kwargs, so there is no change.
- [ ] **Step 4-5** — `tests/pointcloud tests/geometry/loop_closure`; G′ 0; P → PASS.
- [ ] **Step 6: Commit** — `refactor(pointcloud): template steps take no pass-through kwargs`.

---

### Task B-34: annotation and docstring contract for feedforward  (spec row: 48)

**Files:** every file under `collab_splats/pointcloud/feedforward/`, `collab_splats/pointcloud/__init__.py`, `collab_splats/pointcloud/base.py`

- [ ] **Step 1: List the hits** — `cd $WT && PYTHONPATH=$WT $PY $SP/contract_hits.py collab_splats/pointcloud/feedforward collab_splats/pointcloud/__init__.py collab_splats/pointcloud/base.py`. This is the failing list.
- [ ] **Step 3: Implement** — fix each hit:
  - `"""Summary on the opening line` → summary on its own line
  - prose paragraphs → `- ` bullets
  - missing `Args:` / `Returns:`, and types duplicated in `Args:`
  - three-line-plus comment runs → header + bullets
  - (extra) drop over-long docstrings that restate the code: `_verify_loop_candidate`'s old one is replaced by B-13
  - (extra) `Optional` import (:24) → `X | None`, then drop the import
- [ ] **Step 4** — `contract_hits.py` prints no hits for these paths; `tests/test_docstring_contract.py` passes.
- [ ] **Step 5: Gates** — G′: the xpassed/xfailed counts of `test_docstring_contract` for pointcloud can move. Report the before/after counts. P → PASS.
- [ ] **Step 6: Commit** — `docs(pointcloud): feedforward docstrings and annotations meet the contract`.

---

## Lane C, section C bug fixes, end tasks

Conventions (`WT`, `PY`, `SP`, G′, P, commit form) are at the top of the plan. Lane C adds:

- `WT_C=/workspace/collab-splats/.worktrees/pc-lane-c`.
- **G′_C** is G′ with `WT` replaced by `WT_C`:
  `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -c "import collab_splats;print(collab_splats.__file__)" && cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors tests/pointcloud tests/geometry tests/wrapper tests/evals tests/localization tests/remote tests/test_docstring_contract.py`
  - the proof line must print `$WT_C/collab_splats/__init__.py`, or the run tested the main tree
- **Targeted run in lane C:** `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider <nodeid>`
- **P in `WT_C`:** `cd $WT_C && PARITY_WT=$WT_C PYTHONPATH=$WT_C PYTHONUTF8=1 $PY $SP/parity.py --check`. Lane C touches no parity path, so P runs once at C-0 (sanity) and once after the cherry-pick (C-9).
- Lane A and lane B symbol names are anchored by **symbol, at its post-lane-B location**, because their final line numbers are unknown when this part is written. Line numbers below are HEAD `11c5d7c7`. Re-grep a symbol before editing it.
- The G′ deltas below are passed counts relative to the gate at the start of the task. Failed, error and xfail counts must not move unless a step says so.

**Lane C total: −6 passed (C-4 +1, C-5 −1, C-6 −2, C-8 −4).**

---

## Lane C — wrapper (`collab_splats/wrapper/reconstructor.py` + `tests/wrapper`)

### Task C-0: Lane C worktree  (spec rows: —)

**Files:** none tracked.

- [ ] Step 1: Branch from the tip that already holds lanes A and B, then link the gitignored clones (a fresh tree without them would turn guarded failures into SKIPs):
  ```bash
  cd /workspace/collab-splats/.worktrees/pointcloud-release && git worktree add .worktrees/pc-lane-c -b pc-lane-c clean/pointcloud-release
  cd /workspace/collab-splats/.worktrees/pc-lane-c && for d in LoGeR Video-Depth-Anything hloc; do ln -s /workspace/collab-splats/third_party/$d third_party/$d; done && ls -l third_party
  ```
- [ ] Step 2: Record the G′_C baseline. It must equal the G′ recorded at the tip of lane B (same counts, same SKIP count). If the skip count is higher, a `third_party` link is missing: stop and fix it.

---

### Task C-1: Feedforward dispatch through the registry  (spec rows: 35; extra)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:37,78,429-586` (`_run_feedforward`, `_FEEDFORWARD_BACKENDS`)
- Test: `tests/wrapper/test_reconstructor.py:1133-1283`, `tests/wrapper/test_reconstructor_mv_config.py:14-27`

Lane B made `BaseFeedforwardCreator(RegistryMixin)`, and `get_creator` (in `collab_splats.pointcloud`) now reads `BaseFeedforwardCreator._registry` and raises `ValueError`. The wrapper still carries its own `creator_map` and a hand-listed `_FEEDFORWARD_BACKENDS`, so a fifth backend has to be added in three places.

The inline creator imports at :467-472 say they keep "module loads without GPU/model deps". That is false: `reconstructor.py:37` already imports `collab_splats.pointcloud.feedforward.base`, which runs `collab_splats/pointcloud/__init__`, which imports every creator. The loop-closure import at :463-466 is also light, because `collab_splats.pointcloud` already imports `collab_splats.geometry`. Both inline imports move to the top.

- [ ] Step 1: Write the failing test. Retarget the patches to the registry path; before the fix `collab_splats.wrapper.reconstructor.get_creator` and `.LoopClosure` do not exist, so `patch` raises.
  - `tests/wrapper/test_reconstructor.py`, in each of the six tests at :1141, :1170, :1198, :1219, :1241, :1266:
    ```python
    # before
    patch("collab_splats.pointcloud.feedforward.VGGTXCreator", return_value=mock_creator),
    patch("collab_splats.geometry.loop_closure.wrapper.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    # after
    patch("collab_splats.wrapper.reconstructor.get_creator", return_value=MagicMock(return_value=mock_creator)),
    patch("collab_splats.wrapper.reconstructor.LoopClosure", return_value=mock_lc_instance) as mock_lc_cls,
    ```
    - keep each test's own `return_value=` / `as` spelling. Only the target string changes, plus the `MagicMock(return_value=...)` wrap on `get_creator`
    - tests without `return_value=mock_creator` (:1198, :1266) become `patch("collab_splats.wrapper.reconstructor.get_creator")`
    - `patch("collab_splats.viewer.Viewer")` stays; the `Viewer` import stays inline (websocket dep, only with LC + viz)
  - `tests/wrapper/test_reconstructor_mv_config.py:16-17`:
    ```python
    # before
    with patch("collab_splats.pointcloud.feedforward.VGGTOmegaCreator") as mock_cls:
        mock_cls.return_value.outputs = None
    # after
    with patch("collab_splats.wrapper.reconstructor.get_creator") as get:
        mock_cls = get.return_value
        mock_cls.return_value.outputs = None
    ```
- [ ] Step 2: Run it, expect FAIL:
  `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_reconstructor.py -k run_feedforward tests/wrapper/test_reconstructor_mv_config.py`
  - expected: `AttributeError: <module 'collab_splats.wrapper.reconstructor' ...> does not have the attribute 'get_creator'`
- [ ] Step 3: Implement.
  - Imports, top of `reconstructor.py`:
    ```python
    # before (:37)
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    # after
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure, LoopClosureConfig
    from collab_splats.pointcloud import get_creator
    from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator, FeedforwardResult
    ```
    - isort places the `geometry` line after the existing `collab_splats.geometry.transforms` import
    - if lane B re-exports `BaseFeedforwardCreator` from `collab_splats.pointcloud`, import both names from there instead
  - `:78`:
    ```python
    # before
    _FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega", "loger"}
    # after
    _FEEDFORWARD_BACKENDS = set(BaseFeedforwardCreator._registry)
    ```
  - `_run_feedforward` :462-472: delete the `# Heavy dep imports` comment and both inline import blocks.
  - `_run_feedforward` :488-498, LoGeR refusal comment (extra, compress):
    ```python
    # LoGeR refuses loop closure
    # - no LC verify thresholds are calibrated for LoGeR
    # - refused before any filesystem read: a config error must not surface as an IO error
    # - reads the normalized lc_enabled: {"enabled": False} is truthy with falsy intent
    ```
  - :506-511, LoGeR advisory comment (extra, compress):
    ```python
    # LoGeR under Omega's frame ceiling buys nothing: warn, don't change behavior
    # - preproc.max_frames is VGGT-Omega's GPU limit, already applied upstream
    # - None: no ceiling configured, no advice due
    ```
  - :524-544:
    ```python
    # before
    # Select creator class by backend name
    creator_map = {
        "vggtx": VGGTXCreator,
        "mapanything": MapAnythingCreator,
        "vggt_omega": VGGTOmegaCreator,
        "loger": LoGeRCreator,
    }
    # max_points caps the confidence mask during inference — a memory guard, not a preference.
    # ... (7 more lines)
    extra = dict(creator_kwargs or {})
    ...
    creator = creator_map[backend](**explicit, **extra)
    # after
    # Reserved creator kwargs: passed explicitly, so a clash in the backend block is refused by name
    # - otherwise an opaque TypeError naming neither key nor config path
    # - unknown keys are left to the constructor's TypeError, which names them
    extra = dict(creator_kwargs or {})
    explicit = {"max_points": max_points, "use_multiview_confidence": use_multiview_confidence}
    clash = sorted(extra.keys() & explicit.keys())
    if clash:
        raise ValueError(f"pointcloud.{backend}.{clash[0]} is not settable; use pointcloud.{clash[0]}")
    creator = get_creator(backend)(**explicit, **extra)
    ```
  - Docstring :429-461 is prose. Rewrite it to the contract shape: a summary line, bullets, then `Args:` for all ten parameters and `Returns:` (`(PointcloudResult, Viewer | None)`). The wrapper package is outside `PACKAGES`, so this is style only; keep it to one line per argument.
  - Import check: `cd $WT_C && PYTHONPATH=$WT_C $PY -c "import collab_splats.wrapper.reconstructor as R; print(sorted(R._FEEDFORWARD_BACKENDS))"` must print `['loger', 'mapanything', 'vggt_omega', 'vggtx']`.
- [ ] Step 4: Run the Step 2 command, expect PASS. Also run `tests/pointcloud/feedforward/test_loger_creator.py -k FEEDFORWARD_BACKENDS` (its :921 asserts `"loger" in _FEEDFORWARD_BACKENDS`).
- [ ] Step 5: Gates. G′_C, **delta 0**. No P (C-9).
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_mv_config.py && git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_mv_config.py -m "refactor(pointcloud): wrapper dispatches feedforward creators through the registry

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-2: Frame names through `frames.frame_name`  (spec rows: 51, 81; extra)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:1190-1195, 859-865`
- Test: existing (`tests/wrapper/test_sfm_stage.py::test_load_pointcloud_from_disk_reads_the_subset_the_zarr_holds`, `::test_localization_db_pairs_zarr_rows_with_their_own_frames`)

This change is output-neutral: the helper-lane `frames.frame_name(idx)` returns `f"frame_{idx:06d}"`, with no extension. Steps 1-2 are skipped (pure rewrite, pinned by the existing tests).

- [ ] Step 3: Implement.
  - :1190-1195:
    ```python
    # before
    # Rebuild image_paths from the zarr's own rows, so they line up with the per-frame arrays
    # the downstream stages index
    # - COLMAP images are frame_{source_idx:06d} with NO extension
    # - the frame_*.jpg spelling elsewhere is the localization id namespace, not this one
    stored = zarr.open(str(self.pointcloud_zarr), mode="r").attrs["image_paths"]
    image_paths = [Path(f"frame_{frames.frame_idx_from_path(p):06d}") for p in stored]
    # after
    # image_paths from the zarr's own rows, so they line up with the per-frame arrays
    # - COLMAP image names are extensionless stems
    stored = zarr.open(str(self.pointcloud_zarr), mode="r").attrs["image_paths"]
    image_paths = [Path(frames.frame_name(frames.frame_idx_from_path(p))) for p in stored]
    ```
  - :859-865 (extra):
    ```python
    # before
    # The .jpg suffix is deliberate and stays even though the store writes .png. These ids are
    # ... (5 more lines)
    ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
    # after
    # Localization ids keep .jpg though the store writes .png
    # - opaque labels joined on the stem; nothing resolves them to a file
    # - changing the suffix would invalidate every localization DB on disk
    ids = [f"{frames.frame_name(fi)}.jpg" for fi in frame_indices]
    ```
    - `frame_name` must format `int(idx)`. If the helper lane's `frame_name` does not cast, keep `frames.frame_name(int(fi))`
- [ ] Step 4: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_stage.py tests/wrapper/test_localize_stage.py` expect PASS. If `test_localize_stage.py` does not exist, run `-k localiz` over `tests/wrapper`.
- [ ] Step 5: Gates. G′_C, **delta 0**.
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py && git commit --only collab_splats/wrapper/reconstructor.py -m "refactor(pointcloud): wrapper frame names via frames.frame_name

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-3: Direct indexing in `_validate_sfm_block`; validation tests live with the config tests  (spec rows: 102, 111)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:1018-1060` (`_validate_sfm_block`)
- Test: `tests/wrapper/test_sfm_stage.py:237-283` → `tests/wrapper/test_sfm_config.py`

`validate_config` runs after the `base.yaml` deep-merge, so every key exists. `.get` fallbacks there hide a missing default as a `None` bounds error. This is output-neutral for any config that merges `base.yaml`. Steps 1-2 are skipped (a move plus direct indexing, pinned by the moved tests).

- [ ] Step 3: Implement.
  - `_validate_sfm_block`: `block = pc.get(backend) or {}` → `block = pc[backend]`; `block.get("pairing")` → `block["pairing"]`; in the counts loop `block.get(key)` → `block[key]`; `block.get("min_registered_frac")` → `block["min_registered_frac"]`; in the hloc loop `block.get(key)` → `block[key]`.
  - Move `_validated_sfm_config` and the four tests at `test_sfm_stage.py:241-283` verbatim to the end of `tests/wrapper/test_sfm_config.py`, under `# instantsfm knob bounds at config load` + divider. Delete them and their section divider from `test_sfm_stage.py`. `Reconstructor` and `pytest` are already imported in `test_sfm_config.py`. Check with `rtk proxy grep -n "^from\|^import" tests/wrapper/test_sfm_config.py`.
- [ ] Step 4: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_config.py tests/wrapper/test_sfm_stage.py` expect PASS, with the same total count as before the move.
- [ ] Step 5: Gates. G′_C, **delta 0**.
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_stage.py tests/wrapper/test_sfm_config.py && git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_stage.py tests/wrapper/test_sfm_config.py -m "refactor(pointcloud): sfm block validation indexes the merged config directly

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-4: One sfm block validator, keys derived from the creators  (spec rows: 100; fix F-B10)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:81-95` (`_SFM_BLOCK_KEYS`), `:975-998` (`validate_config` instantsfm block), `:1018+` (`_validate_sfm_block`), `configs/README.md` (migration note)
- Test: `tests/wrapper/test_sfm_config.py`

**F-B10.** The instantsfm block has no unknown-key check. A typo such as `retriangulate: true` passes validation and is silently ignored, because `_run_sfm` reads three named keys. After C-5 it would reach `InstantSfMCreator(**block)` as a mid-run `TypeError`. colmap and hloc already refuse unknown keys. Separately, `_SFM_BLOCK_KEYS` restates each creator's dataclass fields by hand, so the two can drift apart.

> Spec deviation: row 100 derives keys for colmap and hloc only. Derive them for all three `SFM_CREATORS`, so that F-B10 is the same check rather than a second one.

**Behavior change (B10):** a published `run_config.yaml` that still carries a retired instantsfm key (`depth_align`, `features`, `single_camera`; see the 2026-09-06 pointcloud-cleanup CHANGELOG entry) now fails at config load, where before it was silently ignored. This is the intended fix. The migration note goes in `configs/README.md`.

**Side effect:** `InstantSfMCreator.use_depths` (field at `sfm/instantsfm.py:246`) becomes a settable key. It is not in `base.yaml`, so default behavior is unchanged. Whether to exclude it is under **Proposed additions**.

- [ ] Step 1: Write the failing test, at the end of the moved block in `tests/wrapper/test_sfm_config.py`:
  ```python
  def test_instantsfm_block_rejects_unknown_keys():
      with pytest.raises(ValueError, match=r"pointcloud.instantsfm has unknown keys \['retriangulate'\]"):
          Reconstructor.validate_config(_sfm_cfg("instantsfm", retriangulate=True))
  ```
  - `_sfm_cfg` is defined later in the file (~:210); move it up next to `_base_config()` so it is defined above its first user
- [ ] Step 2: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_config.py::test_instantsfm_block_rejects_unknown_keys`, expect FAIL with `Failed: DID NOT RAISE <class 'ValueError'>`.
- [ ] Step 3: Implement.
  - :79-95:
    ```python
    # before
    # Every sfm creator ...
    _SFM_BACKENDS = set(SFM_CREATORS)
    # colmap / hloc sub-block keys; anything else is a typo that would TypeError in the creator
    _SFM_BLOCK_KEYS = {
        "colmap": {...},
        "hloc": {...},
    }
    # after
    _SFM_BACKENDS = set(SFM_CREATORS)

    # Every sfm sub-block key: the creator's init fields, plus the wrapper-applied registration floor
    # - instantsfm has no floor: result_from_reconstruction refuses a partial model
    _SFM_BLOCK_KEYS = {
        backend: {f.name for f in dataclasses.fields(creator)} | ({"min_registered_frac"} if backend != "instantsfm" else set())
        for backend, creator in SFM_CREATORS.items()
    }
    ```
    - check before editing: `cd $WT_C && PYTHONPATH=$WT_C $PY -c "import dataclasses; from collab_splats.pointcloud.sfm import SFM_CREATORS as S; print({k: sorted(f.name for f in dataclasses.fields(v)) for k, v in S.items()})"`
    - expected: colmap `[num_retrieved, num_threads, overlap, pairing]`, hloc adds `feature_conf, matcher_conf, retrieval_conf`, instantsfm `[min_num_view_per_track, random_seed, retriangulation, use_depths]`
    - the derived colmap/hloc sets must equal the old literal. Assert it once in a scratch shell before deleting the literal
  - `validate_config` :975-998:
    ```python
    # before
    # InstantSfM sub-block bounds check
    if method == "sfm" and backend == "instantsfm":
        ... (random_seed + min_views checks)
    # colmap / hloc sub-block bounds check
    if method == "sfm" and backend in _SFM_BLOCK_KEYS:
        Reconstructor._validate_sfm_block(pc, backend)
    # after
    # sfm sub-block bounds: every key is read after SIFT or the mapper has started
    if method == "sfm":
        Reconstructor._validate_sfm_block(pc, backend)
    ```
  - `_validate_sfm_block`: keep the unknown-key check first, then add the instantsfm branch and return; the colmap/hloc body follows unchanged:
    ```python
    # instantsfm: seed in np.random.seed's domain; a track needs two views to triangulate
    if backend == "instantsfm":
        random_seed = block["random_seed"]
        if random_seed is not None and not (isinstance(random_seed, int) and 0 <= random_seed < 2**32):
            raise ValueError(
                f"pointcloud.instantsfm.random_seed must be null or an int in [0, 2**32), got {random_seed!r}"
            )
        min_views = block["min_num_view_per_track"]
        if min_views is not None and not (isinstance(min_views, int) and min_views >= 2):
            raise ValueError(
                f"pointcloud.instantsfm.min_num_view_per_track must be null or an int >= 2, got {min_views!r}"
            )
        return
    ```
    - docstring: summary `Bounds-check an sfm sub-block at config load.`, and `backend:` reads `a key of SFM_CREATORS.`
  - `configs/README.md`: add a `2026-09-2x` migration line under the existing migration list: "`pointcloud.instantsfm` now refuses unknown keys at config load. A `run_config.yaml` carrying the retired `depth_align` / `features` / `single_camera` raises `ValueError` naming them; delete them."
- [ ] Step 4: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_config.py tests/wrapper/test_sfm_stage.py` expect PASS, including the four moved instantsfm bounds tests (same messages).
- [ ] Step 5: Gates. G′_C, **delta +1**.
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py configs/README.md tests/wrapper/test_sfm_config.py && git commit --only collab_splats/wrapper/reconstructor.py configs/README.md tests/wrapper/test_sfm_config.py -m "fix(pointcloud): instantsfm block refuses unknown keys; sfm block keys derived from the creators

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-5: Generic sfm creator dispatch  (spec rows: 101, 116, 110 partial)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:38, 1242-1258` (`_run_sfm`)
- Test: `tests/wrapper/test_sfm_stage.py:82-107, 224-234, 287-294`

`_run_sfm` constructs `InstantSfMCreator` from three named keys while colmap and hloc take the whole block. After C-4 every block key is a validated creator field, so one path serves all three backends. The `base.yaml` instantsfm block is exactly those three keys, so the constructor call is identical. The torch sync/empty is `pytorch_gc` (helper lane `utils/torch_utils.py`).

- [ ] Step 1: Write the failing test, by rewriting the patches in `_patched_sfm` (:90-106):
  ```python
  # remove
  "InstantSfMCreator": patch(f"{RECONSTRUCTOR}.InstantSfMCreator"),
  "torch": patch(f"{RECONSTRUCTOR}.torch", SimpleNamespace(cuda=MagicMock(is_available=lambda: False))),
  # change
  "SFM_CREATORS": patch.dict(
      f"{RECONSTRUCTOR}.SFM_CREATORS", {"instantsfm": MagicMock(), "colmap": MagicMock(), "hloc": MagicMock()}
  ),
  # add
  "pytorch_gc": patch(f"{RECONSTRUCTOR}.pytorch_gc"),
  ```
  - `test_run_sfm_forwards_random_seed_from_the_config` (:224-227): `mocks["InstantSfMCreator"]` → `mocks["SFM_CREATORS"]["instantsfm"]`
  - delete `test_run_sfm_random_seed_defaults_to_none` (:230-234; row 110, subsumed: the block is forwarded whole, and the forwarding test pins it)
  - `test_run_sfm_builds_the_creator_from_the_block_minus_the_floor` (:289-294): delete the last line `started["InstantSfMCreator"].assert_not_called()`
- [ ] Step 2: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_stage.py`, expect FAIL: `AttributeError: <module 'collab_splats.wrapper.reconstructor' ...> does not have the attribute 'pytorch_gc'`.
- [ ] Step 3: Implement.
  - Imports: `:38 from collab_splats.pointcloud.sfm import SFM_CREATORS, InstantSfMCreator` → `from collab_splats.pointcloud.sfm import SFM_CREATORS`. Add `from collab_splats.utils.torch_utils import pytorch_gc`. `import torch` stays (it is also used at :668).
  - :1242-1258:
    ```python
    # before
    # Mapper per backend; writes colmap/<db> + colmap/sparse/0 with stem image names
    # - instantsfm: its three knobs, constructed exactly as before
    # - colmap / hloc: the whole block but min_registered_frac, the floor applied below
    if backend == "instantsfm":
        creator = InstantSfMCreator(...)
    else:
        kwargs = {k: v for k, v in pc_cfg[backend].items() if k != "min_registered_frac"}
        creator = SFM_CREATORS[backend](**kwargs)
    recon = creator.reconstruct(backend_dir, images_dir=images_dir)
    del creator
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    # after
    # Mapper: the whole block but the wrapper-applied floor; writes colmap/sparse/0 with stem names
    creator = SFM_CREATORS[backend](**{k: v for k, v in pc_cfg[backend].items() if k != "min_registered_frac"})
    recon = creator.reconstruct(backend_dir, images_dir=images_dir)
    del creator
    pytorch_gc()
    ```
    - check that `pytorch_gc` does `gc.collect()` + `empty_cache` (+ sync) under `cuda.is_available()`. If it lacks `synchronize`, it is still output-neutral (the next op syncs)
- [ ] Step 4: Run the Step 2 command, expect PASS.
- [ ] Step 5: Gates. G′_C, **delta −1**.
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_stage.py && git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_stage.py -m "refactor(pointcloud): one sfm creator dispatch path for every backend

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-6: Provenance from the creator  (spec rows: 98, 110 partial; extra)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:6, 39, 41, 1278-1288`
- Test: `tests/wrapper/test_sfm_stage.py:29, 94-101, 105, 313-333`

Lane A gave each sfm creator `provenance() -> dict`, with the same keys in the same order as today's if/elif (`instantsfm_version`; `pycolmap_version, colmap_cli_version`; `pycolmap_version, hloc_commit`). The wrapper stops knowing per-backend version sources.

- [ ] Step 1: Write the failing test.
  - Add above `_patched_sfm`:
    ```python
    def _creator_cls(provenance):
        """
        Mock sfm creator class whose instances report `provenance`.
        """
        cls = MagicMock()
        cls.return_value.provenance.return_value = provenance
        return cls
    ```
  - In `_patched_sfm`, delete the `"importlib"` patch and its 3-line comment (:95-101) and the `"colmap_cli_version"` patch (:105); rewrite the dict:
    ```python
    # sfm creator classes; patch.dict yields the dict itself
    "SFM_CREATORS": patch.dict(
        f"{RECONSTRUCTOR}.SFM_CREATORS",
        {
            "instantsfm": _creator_cls({"instantsfm_version": "0.0.0"}),
            "colmap": _creator_cls({"pycolmap_version": "0.0.0", "colmap_cli_version": "COLMAP test"}),
            "hloc": _creator_cls({"pycolmap_version": "0.0.0", "hloc_commit": "abc"}),
        },
    ),
    ```
  - Replace `test_run_sfm_stamps_backend_provenance` (:313-319):
    ```python
    def test_run_sfm_stamps_the_creator_provenance(tmp_path):
        started = _run_backend(tmp_path, "colmap", [f"frame_{i:06d}" for i in (0, 9, 30)])
        attrs = started["result_from_reconstruction"].return_value[0].save_zarr.call_args.kwargs["extra_attrs"]
        started["creator"].return_value.provenance.assert_called_once_with()
        assert list(attrs) == ["method", "backend", "pycolmap_version", "colmap_cli_version", "registered_frames", "total_frames"]
        assert attrs["colmap_cli_version"] == "COLMAP test"
    ```
  - Delete `test_run_sfm_hloc_stamps_the_pin` (:322-326; extra: lane A's `test_hloc.py` provenance test pins `HLOC_PIN`) and `test_run_sfm_instantsfm_attrs_are_unchanged` (:329-333; row 110: the key order is the creator's contract, pinned in lane A's instantsfm test). Delete `from collab_splats.pointcloud.sfm.hloc import HLOC_PIN` (:29). Drop `SimpleNamespace` from :17 only if nothing else uses it (`_registered` does, so it stays).
- [ ] Step 2: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_stage.py::test_run_sfm_stamps_the_creator_provenance`, expect FAIL: `AssertionError: Expected 'provenance' to be called once. Called 0 times.` Without the `colmap_cli_version` patch, HEAD shells out to the real CLI, so only the call assertion fails deterministically. It must fail on that line, not on an import.
- [ ] Step 3: Implement.
  - :1278-1288 plus the save:
    ```python
    # before
    # Provenance per backend; instantsfm's keys and order are unchanged
    # - colmap: SIFT from the CLI, mapping from the wheel — two different COLMAPs
    if backend == "instantsfm":
        version_attrs = {"instantsfm_version": importlib.metadata.version("instantsfm")}
    elif backend == "colmap":
        version_attrs = {...}
    else:
        version_attrs = {...}
    ...extra_attrs={"method": "sfm", "backend": backend, **version_attrs, **subset_attrs, **align_attrs},
    # after
    ...extra_attrs={"method": "sfm", "backend": backend, **provenance, **subset_attrs, **align_attrs},
    ```
    - `provenance = creator.provenance()` goes directly after `recon = creator.reconstruct(...)`, before `del creator` (C-5 block)
  - Delete the imports: `:6 import importlib.metadata`, `:39 from collab_splats.pointcloud.sfm.hloc import HLOC_PIN`, `:41 ...sift_db import colmap_cli_version`. Confirm with `rtk proxy grep -n "importlib\|HLOC_PIN\|colmap_cli_version" collab_splats/wrapper/reconstructor.py`, which must print nothing.
- [ ] Step 4: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_stage.py tests/pointcloud/sfm` expect PASS.
- [ ] Step 5: Gates. G′_C, **delta −2** (one test rewritten, two deleted).
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_stage.py && git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_sfm_stage.py -m "refactor(pointcloud): sfm zarr provenance comes from the creator

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-7: `_run_sfm` reads only the frames it keeps  (spec rows: 103, 104; extra)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:1198-1297` (`_run_sfm`)
- Test: existing `tests/wrapper/test_sfm_stage.py::test_run_sfm_subsets_to_registered_frames_in_order` (asserts `keyframes[:, 0, 0, 0] == [0, 30, 57]` against a real frame store)

After the solve, `_run_sfm` decodes the whole store and then slices rows. `frames.read_frames(dir, idxs)` (`preproc/frames.py:153`) takes source indices, so it can decode only the registered frames. This is output-neutral: `read_frames` returns `np.stack(...)`, which is contiguous, so `np.ascontiguousarray` is a no-op, and a subset read equals the sliced full read. Steps 1-2 are skipped (pinned by the existing subset test).

> Spec deviation: row 103 asks to drop `names` from `_run_sfm`. It stays. The VDA cache gate, `generate_vda_depth`, `_registered_rows` and `result_from_reconstruction` all key on it, and it is read from `frames.frame_paths` once.

- [ ] Step 3: Implement. The whole body after C-5, C-6 and C-7:
  ```python
  pc_cfg = self.config["pointcloud"]
  backend = pc_cfg["backend"]
  block = pc_cfg[backend]
  backend_dir = self.backend_dir
  backend_dir.mkdir(parents=True, exist_ok=True)

  # VDA metric depth per keyframe, cached by stem
  # - images/ is read in place: it is already the COLMAP layout
  # - a gate miss drops depth_vda/ first: generate_vda_depth only adds maps
  names = [p.name for p in frames.frame_paths(self.images_dir)]
  if not vda_depth_complete(backend_dir, names):
      shutil.rmtree(backend_dir / "depth_vda", ignore_errors=True)
  depths = generate_vda_depth(frames.read_frames(self.images_dir), backend_dir, names)

  # Mapper: the whole block but the wrapper-applied floor; writes colmap/sparse/0 with stem names
  # - the frame stack is already freed, so its peak never meets the solve's
  creator = SFM_CREATORS[backend](**{k: v for k, v in block.items() if k != "min_registered_frac"})
  recon = creator.reconstruct(backend_dir, images_dir=self.images_dir)
  provenance = creator.provenance()
  del creator
  pytorch_gc()

  # Incremental mappers may drop frames: registered subset above the floor
  # - instantsfm stays strict; result_from_reconstruction refuses a partial model
  subset_attrs = {}
  if backend != "instantsfm":
      rows = _registered_rows(recon, names, block["min_registered_frac"], backend)
      subset_attrs = {"registered_frames": len(rows), "total_frames": len(names)}
      names = [names[row] for row in rows]
      depths = depths[rows]

  # Dense result at VDA resolution in the COLMAP world; decodes only the kept frames
  keyframes = frames.read_frames(self.images_dir, [frames.frame_idx_from_path(n) for n in names])
  outputs, align_attrs = result_from_reconstruction(recon, depths, keyframes, names)
  outputs.save_zarr(
      self.pointcloud_zarr,
      extra_attrs={"method": "sfm", "backend": backend, **provenance, **subset_attrs, **align_attrs},
  )
  logger.info("pointcloud.zarr saved: %s  (%s pts)", self.pointcloud_zarr, f"{len(outputs.points):,}")

  return PointcloudResult(reconstruction=recon, image_paths=outputs.image_paths)
  ```
  - memory: the first stack is a temporary argument to `generate_vda_depth`, so it is freed on return, the same as today's `del keyframes`. Check that `generate_vda_depth` does not stash the frames on a module global: `rtk proxy grep -n "global\|_cache" collab_splats/pointcloud/vda.py`
  - `self.pointcloud_zarr` (:1078) is `backend_dir / "pointcloud.zarr"`, the same path
  - docstring (:1199-1207): summary `SfM path: keyframes -> VDA metric depth -> sfm mapper -> pointcloud.zarr.`, then 2 bullets (images/ read in place; zarr carries alignment + creator provenance), and `Returns:`
- [ ] Step 4: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/wrapper/test_sfm_stage.py` expect PASS. `_run_sfm` must be ≤ 45 lines: `rtk proxy sed -n '/def _run_sfm/,/def refine_poses/p' collab_splats/wrapper/reconstructor.py | wc -l`.
- [ ] Step 5: Gates. G′_C, **delta 0**.
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add collab_splats/wrapper/reconstructor.py && git commit --only collab_splats/wrapper/reconstructor.py -m "refactor(pointcloud): _run_sfm decodes only the registered frames after the solve

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-8: Test suite reduction — registry and sfm config  (spec rows: 106, 107, 108, 109)

**Files:**
- Test: `tests/pointcloud/test_registry.py`, `tests/wrapper/test_sfm_config.py`

This is test-only. Steps 1-3 are the deletions and merges.

- [ ] Step 3: Implement.
  - **Row 106** (`test_registry.py`, −1): delete `test_sfm_creators_maps_every_sfm_backend`. (B-2 already deleted `test_sfm_backends_left_the_feedforward_registry` and `test_old_keys_removed`.) These are history guards: C-4 derives `_SFM_BLOCK_KEYS` from `SFM_CREATORS`, and `get_creator`'s `ValueError` (lane B) covers unknown keys. Delete any sfm import left unused (`rtk proxy grep -n "sfm" tests/pointcloud/test_registry.py`).
  - **Row 107** (`test_sfm_config.py`, 4 → 3, −1): replace `test_sfm_rejects_bundle_adjustment` (:45), `test_sfm_rejects_loop_closure` (:138) and `test_colmap_and_hloc_refuse_ba_and_lc` (:161, 2 params) with:
    ```python
    @pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
    def test_sfm_refuses_bundle_adjustment_and_loop_closure(backend):
        for key, match in (("bundle_adjustment", "bundle_adjustment"), ("loop_closure", "loop_closure")):
            cfg = _sfm_cfg(backend)
            cfg["pointcloud"][key] = True
            with pytest.raises(ValueError, match=match):
                Reconstructor.validate_config(cfg)
    ```
    - copy the `match=` strings from the deleted tests if they are narrower than the key name
  - **Row 108** (3 → 3, 0): replace `test_sfm_instantsfm_config_validates` (:35) and `test_colmap_and_hloc_validate_at_config_load` (:150, 2 params) with:
    ```python
    @pytest.mark.parametrize("backend", ["instantsfm", "colmap", "hloc"])
    def test_sfm_backend_validates_at_config_load(backend):
        cfg = Reconstructor.validate_config(_sfm_cfg(backend))
        assert cfg["pointcloud"]["backend"] == backend
    ```
    - carry over any extra assertion the deleted bodies made (read them first)
  - **Row 109** (−2): delete `test_base_yaml_has_instantsfm_block` (:57), `test_base_yaml_has_colmap_and_hloc_blocks` and `_COLMAP_DEFAULTS`. Row 108 now validates the merged `base.yaml` block for every backend, and C-3 indexes it directly.
- [ ] Step 4: `cd $WT_C && PYTHONPATH=$WT_C:$SP/stubs PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/pointcloud/test_registry.py tests/wrapper/test_sfm_config.py` expect PASS. Mutation check: comment out the `method == "sfm" and lc_enabled` raise in `validate_config`, confirm 3 failures from the row-107 test, then revert the edit with `git checkout -- collab_splats/wrapper/reconstructor.py` (the only uncommitted change).
- [ ] Step 5: Gates. G′_C, **delta −4**.
- [ ] Step 6: Commit:
  ```bash
  cd $WT_C && git add tests/pointcloud/test_registry.py tests/wrapper/test_sfm_config.py && git commit --only tests/pointcloud/test_registry.py tests/wrapper/test_sfm_config.py -m "test(pointcloud): merge sfm config tests over backends; drop history guards

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

### Task C-9: Land lane C  (spec rows: —)

- [ ] Step 1: `cd $WT && git cherry-pick $(git rev-list --reverse HEAD..pc-lane-c)`. On a conflict: `git cherry-pick --abort`, stop, report. Never resolve by hand against another lane.
- [ ] Step 2: Gates in `WT`. G′ must equal the pre-lane-C G′ with **−6 passed** and no other count moved. Run P and expect `SUMMARY PASS`; lane C touched no parity path, so a failure means lane A/B drift (stop, report).
- [ ] Step 3: `cd $WT && git worktree remove ../pc-lane-c && git branch -d pc-lane-c`

---

## Section C — bug fixes (serial in `WT`, after C-9)

All run in `WT`, one commit each. A fix that changes parity follows the parity-change protocol (before committing):

1. `cp -r $SP/parity_baseline $SP/parity_baseline.pre-<ID>`
2. `--check` (expect FAIL). Record the failing fields, and confirm they are only the fields the fix predicts.
3. `--save`, then `--check` (expect `SUMMARY PASS`).
4. Write the delta into the commit body.

An unexpected field in step 2 means: do not save, do not commit; restore the working files with `cd $WT && git checkout -- <the task's files>`, stop, report. If a baseline was already saved, restore it with `rm -r $SP/parity_baseline && cp -r $SP/parity_baseline.pre-<ID> $SP/parity_baseline`.

### Task F-B1: MapAnything `original_coords` records its real crop  (spec rows: B1)

**Files:** Modify: `collab_splats/pointcloud/feedforward/mapanything.py:224-229` (`_preprocess`) / Test: `tests/pointcloud/feedforward/test_mapanything_creator.py`

`_preprocess` writes `[[0, 0, model_w, model_h, W, H]]` for every frame. That is the model grid, not the source crop, so `original_coords` claims the model saw the whole frame at model resolution. Upstream MapAnything resizes (with `force=True`, so it always rescales) and center-crops. Every consumer of `original_coords` (the `_rescale_…` COLMAP export, the eval intrinsics) is wrong for MapAnything.

- [ ] Step 1: Write the failing test:
  ```python
  def test_crop_box_matches_upstream_resize_then_center_crop():
      box = _crop_box(1920, 1080, 518, 294)
      np.testing.assert_allclose(box, [7.346938505622668, 0.0, 1910.2040114618937, 1079.9999603265321, 1920, 1080])
  ```
  - import `_crop_box` from `collab_splats.pointcloud.feedforward.mapanything`
  - known answer: scale `0.27222223` (+1e-8), resized width 522, `left = 2`, top 0
- [ ] Step 2: Targeted run, expect FAIL: `ImportError: cannot import name '_crop_box'`.
- [ ] Step 3: Implement, in `mapanything.py` above the class:
  ```python
  def _crop_box(orig_w: int, orig_h: int, model_w: int, model_h: int) -> list[float]:
      """
      Source-pixel crop box MapAnything's resize-then-center-crop keeps.

      - port: facebookresearch/map-anything @ c845b8f, mapanything/utils/cropping.py:231,240,443-447
      - force=True upstream: always rescales, even when already at size

      Args:
          orig_w: source frame width.
          orig_h: source frame height.
          model_w: model input width.
          model_h: model input height.

      Returns:
          [x0, y0, x1, y1, orig_w, orig_h] in source pixels.
      """
      scale = max(model_w / orig_w, model_h / orig_h) + 1e-8
      left = (int(np.floor(orig_w * scale)) - model_w) // 2
      top = (int(np.floor(orig_h * scale)) - model_h) // 2
      return [left / scale, top / scale, (left + model_w) / scale, (top + model_h) / scale, orig_w, orig_h]
  ```
  - `_preprocess`: replace the `[[0, 0, model_w, model_h, W, H]]` rows with `[_crop_box(W, H, model_w, model_h)]` per frame, and keep the existing dtype and array construction
  - verify the citation at the pin before committing: `curl -s https://raw.githubusercontent.com/facebookresearch/map-anything/c845b8f4f6cde0c20aecd87573656c3f69f5b2b0/mapanything/utils/cropping.py | sed -n '225,245p;440,450p'`
- [ ] Step 4: The targeted test plus `tests/pointcloud/feedforward/test_mapanything_creator.py`, expect PASS.
- [ ] Step 5: Gates. G′ **+1**. Run P and expect `SUMMARY PASS`: the harness sets `original_coords` directly.
- [ ] Step 6: Commit `fix(pointcloud): mapanything original_coords records the resize-then-crop box`.

### Task F-B2: Refine exports COLMAP with the backend's camera model  (spec rows: B2)

**Files:** Modify: `collab_splats/pointcloud/feedforward/base.py` (`build_colmap`, at its post-lane-B location; HEAD :1249-1296), `collab_splats/wrapper/reconstructor.py` (`refine_poses`, HEAD :1299-1390) / Test: `tests/wrapper/test_refine_stage.py`

`refine_poses` calls `build_pycolmap_reconstruction(...)` without `camera_model`, so a vggtx scene (`camera_model = "SIMPLE_PINHOLE"`, `vggtx.py:189`) is re-exported as PINHOLE after BA. The export sequence (build, rescale, write) is also duplicated between `build_colmap` and `refine_poses`.

- [ ] Step 1: Write the failing test in `tests/wrapper/test_refine_stage.py`, by copying `test_refine_poses_refines_and_persists`'s patches:
  ```python
  def test_refine_poses_keeps_the_backend_camera_model(tmp_path):
      recon = Reconstructor(_cfg(tmp_path, backend="vggtx", bundle_adjustment=True, loop_closure=False))
      _write_ff_zarr(recon.backend_dir)
      with (
          patch("collab_splats.geometry.bundle_adjustment.BundleAdjustment.refine", side_effect=fake_refine),
          patch.object(Reconstructor, "_load_pointcloud_from_disk"),
      ):
          recon.refine_poses()
      sparse = recon.backend_dir / "colmap" / "sparse" / "0"
      assert {c.model.name for c in pycolmap.Reconstruction(str(sparse)).cameras.values()} == {"SIMPLE_PINHOLE"}
  ```
  - use the existing test's exact patch targets and `fake_refine`. If `fake_refine` is local to that test, lift it to module level
- [ ] Step 2: Targeted run, expect FAIL: `AssertionError: assert {'PINHOLE'} == {'SIMPLE_PINHOLE'}`.
- [ ] Step 3: Implement. Add to `ff/base.py` next to `build_colmap`:
  ```python
  def write_colmap(result: FeedforwardResult, output_dir: Path, camera_model: str) -> pycolmap.Reconstruction:
      """
      COLMAP model of a feedforward result at original resolution, written to colmap/sparse/0.

      Args:
          result: model-resolution feedforward result.
          output_dir: backend dir; the model lands under colmap/sparse/0.
          camera_model: pycolmap camera model name.

      Returns:
          The written reconstruction.
      """
      recon = build_pycolmap_reconstruction(
          result.points, result.colors, result.extrinsics, result.intrinsics,
          result.model_width, result.model_height, [p.name for p in result.image_paths],
          camera_model=camera_model,
      )
      recon = _rescale_reconstruction_to_original_dimensions(
          recon, result.original_coords, (result.model_width, result.model_height)
      )
      sparse_dir = Path(output_dir) / "colmap" / "sparse" / "0"
      sparse_dir.mkdir(parents=True, exist_ok=True)
      recon.write_binary(str(sparse_dir))
      return recon
  ```
  - match `_rescale_…`'s post-lane-B signature exactly (lane B dropped `image_paths`, `shared_camera`, `shift_point2d` and `verbose`)
  - `build_colmap` body becomes `return write_colmap(self.outputs, output_dir, self.camera_model)`, plus whatever it returns today (keep its return type)
  - `refine_poses`: replace the build + rescale + mkdir + `write_binary` block with
    ```python
    # Re-export COLMAP with the backend's camera model; a block override wins, like the creator's
    camera_model = self.config["pointcloud"].get(backend, {}).get("camera_model", get_creator(backend).camera_model)
    write_colmap(refined, self.backend_dir, camera_model)
    ```
    and delete its inline imports of `build_pycolmap_reconstruction` / `_rescale_…`. Hoist the remaining inline imports (`BundleAdjustment`, `BundleAdjustmentConfig`) to the top: `collab_splats.geometry` is already loaded at import
- [ ] Step 4: `tests/wrapper/test_refine_stage.py` and `tests/pointcloud/feedforward -k colmap`, expect PASS.
- [ ] Step 5: Gates. G′ **+1**. P, expect PASS (the zarr is untouched).
- [ ] Step 6: Commit `fix(pointcloud): refine re-exports COLMAP with the backend camera model`.

### Task F-B3: Omega crop box uses upstream's integer crop  (spec rows: B3)

**Files:** Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py:54-93` (`_compute_omega_original_coords`) / Test: `tests/pointcloud/feedforward/test_vggt_omega_creator.py`

The crop box is computed in float (`tl_x = (W - crop_w) / 2`), while upstream's loader crops at integer offsets. An odd margin lands half a pixel off.

- [ ] Step 1: Write the failing test:
  ```python
  @pytest.mark.parametrize(
      "size, box",
      [((1001, 400), [100, 0, 900, 400, 1001, 400]), ((300, 1001), [0, 200, 300, 800, 300, 1001])],
  )
  def test_omega_crop_box_is_upstream_integer_crop(size, box):
      np.testing.assert_array_equal(_compute_omega_original_coords([size])[0], box)
  ```
  - adapt the call to the function's real signature (read :54-60 first)
- [ ] Step 2: Targeted run, expect FAIL: `Mismatched elements ... 100.5 vs 100`.
- [ ] Step 3: Implement. Mirror `vggt_omega/utils/load_fn.py:68-82 @ 39a0cb8`:
  ```python
  # Center crop to the aspect band, integer offsets as the loader crops
  # - port: vggt-omega @ 39a0cb8, vggt_omega/utils/load_fn.py:68-82
  crop_width = min(width, max(1, int(round(height / 0.5))))
  left = max((width - crop_width) // 2, 0)
  crop_height = min(height, max(1, int(round(width * 2.0))))
  top = max((height - crop_height) // 2, 0)
  ```
  - verify the citation at the pin with a raw fetch of `load_fn.py` at `39a0cb8af88554f15ddcb5354cd52bde588fa014` before committing
- [ ] Step 4: Targeted run plus the file, expect PASS.
- [ ] Step 5: Gates. G′ **+2**. P, expect PASS (the harness sets `original_coords` directly).
- [ ] Step 6: Commit `fix(pointcloud): omega original_coords uses upstream's integer crop`.

### Task F-B4: One `scale_intrinsics_to_original`  (spec rows: B4)

**Files:** Modify: `collab_splats/geometry/transforms.py`, `collab_splats/geometry/metrics.py:239,340-346`, `collab_splats/pointcloud/feedforward/base.py` (`_rescale_reconstruction_to_original_dimensions`), `evals/scripts/eval_splats.py:79-80,98-106` / Test: `tests/geometry/test_transforms.py`, `tests/geometry/test_metrics.py:17,1265-1269`

The model-to-original intrinsics map exists three times: `metrics._scale_intrinsics_to_original` (per image, loop), an inline copy in `eval_splats.py` that ignores the crop offset, and `_rescale_…` in ff/base, which scales the COLMAP camera without the crop offset. Under a crop, the COLMAP sparse camera and the eval intrinsics are shifted by the crop origin.

- [ ] Step 1: Write the failing test. Move `tests/geometry/test_metrics.py:1265-1269` to `tests/geometry/test_transforms.py`:
  ```python
  def test_scale_intrinsics_to_original_undoes_scale_and_crop():
      K = np.array([[20.0, 0, 12], [0, 20, 10], [0, 0, 1]])
      out = scale_intrinsics_to_original(K[None], np.array([[11, 7, 59, 47, 70, 50]]), 12, 8)[0]
      np.testing.assert_allclose(out, [[40, 0, 35], [0, 50, 27], [0, 0, 1]])
  ```
  - keep the moved test's own `K` literal. It must reproduce `[[40,0,35],[0,50,27],[0,0,1]]`
  - remove the `_scale_intrinsics_to_original` import from `test_metrics.py:17`
- [ ] Step 2: Targeted run, expect FAIL: `ImportError: cannot import name 'scale_intrinsics_to_original' from 'collab_splats.geometry.transforms'`.
- [ ] Step 3: Implement in `transforms.py`:
  ```python
  def scale_intrinsics_to_original(
      intrinsics: np.ndarray, original_coords: np.ndarray, model_width: int, model_height: int
  ) -> np.ndarray:
      """
      Model-resolution intrinsics mapped back into source-pixel coordinates.

      - undoes the resize, then shifts the principal point by the crop origin

      Args:
          intrinsics: (N, 3, 3) model-resolution K.
          original_coords: (N, 6) [x0, y0, x1, y1, W, H] crop boxes in source pixels.
          model_width: model input width.
          model_height: model input height.

      Returns:
          (N, 3, 3) float64 K in source pixels.
      """
      tl = original_coords[:, :2].astype(np.float64)
      br = original_coords[:, 2:4].astype(np.float64)
      sx = model_width / (br[:, 0] - tl[:, 0])
      sy = model_height / (br[:, 1] - tl[:, 1])

      K = np.array(intrinsics, dtype=np.float64)
      K[:, 0, 0] /= sx
      K[:, 1, 1] /= sy
      K[:, 0, 2] = K[:, 0, 2] / sx + tl[:, 0]
      K[:, 1, 2] = K[:, 1, 2] / sy + tl[:, 1]
      return K
  ```
  - `metrics.py`: delete `_scale_intrinsics_to_original` (:239) and replace the loop at :340-346 with one call
  - `_rescale_…`, per image:
    ```python
    K = scale_intrinsics_to_original(camera.calibration_matrix()[None], original_coords[row : row + 1], model_w, model_h)[0]
    ```
    - camera params: SIMPLE_PINHOLE `[max(K[0,0], K[1,1]), K[0,2], K[1,2]]`; PINHOLE `[K[0,0], K[1,1], K[0,2], K[1,2]]`; any other model raises `ValueError`
    - width/height from `original_coords[row, 4:6]`
  - `eval_splats.py:98-106`: `intrinsics = scale_intrinsics_to_original(result.intrinsics, result.original_coords, result.model_width, result.model_height).astype(np.float32)`, and correct docstring :79-80 to say the crop offset is applied
- [ ] Step 4: `tests/geometry/test_transforms.py tests/geometry/test_metrics.py tests/pointcloud/feedforward -k "rescale or colmap"` expect PASS.
- [ ] Step 5: Gates. G′ **0** (a moved test). P **required** (`geometry/transforms.py`, `ff/base.py`). Expect PASS: the zarr is unaffected, and only the colmap sparse under a crop changes.
- [ ] Step 6: Commit `fix(pointcloud): one scale_intrinsics_to_original, crop offset applied in COLMAP export and evals`.

### Task F-B5: Depth alignment samples with floor, like its pixel indices  (spec rows: B5) — **parity changes**

**Files:** Modify: `collab_splats/pointcloud/depth_align.py:108-109`, `evals/scripts/eval_verification.py:205` / Test: `tests/pointcloud/test_depth_align.py`

`depth_align.py:108-109` samples VDA depth at `np.rint(xy)`, while `_pixel_indices_from_reconstruction` (:60) uses `int()` (floor). A keypoint at x=3.75 reads column 4 in one path and column 3 in the other.

- [ ] Step 1: Write the failing test:
  ```python
  def test_depth_sample_floors_like_the_pixel_index():
      recon = _fake_reconstruction({"frame_000000.jpg": [(3.75, 4.75, 10.0)]})
      depth = np.ones((1, GRID_H, GRID_W), np.float32)
      depth[0, 4, 3] = 5.0
      depth[0, 5, 4] = 9.0
      _, d_vda = _sample_depths(recon, depth, ["frame_000000.jpg"])
      np.testing.assert_array_equal(d_vda, [5.0])
  ```
  - use the existing helpers/fixture constants in `test_depth_align.py` (GRID 16×32, CAM 64×32, so xy maps exactly), and call the real sampling function that holds :108-109
- [ ] Step 2: Targeted run, expect FAIL: `[9.] != [5.]`.
- [ ] Step 3: `np.rint(...)` → `np.floor(...)` at :108-109. `eval_verification.py:205` → `int(np.floor(px[0] * sx)), int(np.floor(px[1] * sy))`.
- [ ] Step 4: `tests/pointcloud/test_depth_align.py`, expect PASS.
- [ ] Step 5: Gates. G′ **+1**. P, using the parity-change protocol: expected delta only in the `depth_align` case (scale/points). Record `scale` before/after in the commit body.
- [ ] Step 6: Commit `fix(pointcloud): depth alignment samples VDA with floor, matching its pixel indices` with the parity delta in the body.

### Task F-B6: Unproject once per feedforward postprocess  (spec rows: B6) — **parity changes (ulp)**

**Files:** Modify: `collab_splats/pointcloud/feedforward/base.py` (`unproject_and_filter_points` after lane B's move, and the merged `_postprocess` helper), `vggtx.py`/`mapanything.py` callers / Test: `tests/pointcloud/feedforward/test_vggtx_creator.py:174-192`, `tests/pointcloud/test_feature_lifting.py:7,55,81`, `tests/pointcloud/feedforward/test_vggt_omega_creator.py:282+`, `tests/integration/test_pipeline_cu121.py:146`, `tests/geometry/loop_closure/test_wrapper.py:98`

The postprocess unprojects depth twice: once through upstream `unproject_depth_map_to_point_map` inside `unproject_and_filter_points`, and again through `_raw_to_world_points(raw, subsample=1)` (base, HEAD :373-431, float64 invert then float32) for `world_points`. The two grids differ at the ulp level, so `points` and `world_points` are not the same points.

- [ ] Step 1: Write the failing test in `test_vggtx_creator.py`:
  ```python
  def test_points_are_rows_of_the_world_grid():
      out = _postprocessed_outputs()  # new helper: the :174-192 test's setup, returning creator.outputs
      grid = out.world_points.reshape(-1, 3)
      assert {tuple(p) for p in out.points} <= {tuple(p) for p in grid}
  ```
  - `_postprocessed_outputs` is the body of the :174-192 test up to its `_postprocess` call, lifted to a module helper. That test then calls it too
  - set membership is exact float equality, which is the point of the test: before the fix the ulp mismatch makes it fail
- [ ] Step 2: Targeted run, expect FAIL (ulp mismatch).
- [ ] Step 3: `unproject_and_filter_points(points3d, conf, ...)` takes the grid instead of `depth, extrinsic, intrinsic`. The postprocess computes `grid = self._raw_to_world_points(raw, subsample=1)` once, then `pts3d = grid[mask]` and `world_points = grid`. Update every caller listed under **Files**. Patched callers change only their argument names.
- [ ] Step 4: The listed test files, expect PASS.
- [ ] Step 5: Gates. G′ **+1**. P, using the parity-change protocol: expect ulp deltas in `points` for the four feedforward cases only, with max abs diff ≤ 1e-5 m. Record it.
- [ ] Step 6: Commit `fix(pointcloud): feedforward postprocess unprojects depth once`.

### Task F-B7: Multiview confidence inverts poses in closed form  (spec rows: B7) — **parity may change**

**Files:** Modify: `collab_splats/pointcloud/feedforward/base.py:612` (at its post-lane-B location) / Test: `tests/pointcloud/feedforward/test_multiview_confidence.py` (or the file holding the mv tests; `rtk proxy grep -rln "multiview" tests/pointcloud`)

`cam2world = torch.linalg.inv(E)` is a general 4×4 inverse of a rigid transform. The repo's `invert_poses` (`geometry/transforms.py:44-65`, numpy) is the closed form used everywhere else.

- [ ] Step 1: Write the failing test: patch `torch.linalg.inv` to raise, run the mv-confidence function on the existing fixture, and expect no raise.
- [ ] Step 2: Targeted run, expect FAIL: the patched `inv` raises.
- [ ] Step 3: `cam2world = torch.from_numpy(invert_poses(extrinsics.astype(np.float32))).to(dev)`. `invert_poses` is already imported in base, or add it at the top.
- [ ] Step 4: The mv tests, expect PASS.
- [ ] Step 5: Gates. G′ **+1**. P: if `--check` FAILs only on mv-confidence-derived fields, apply the parity-change protocol and record the delta. Otherwise PASS.
- [ ] Step 6: Commit `fix(pointcloud): multiview confidence inverts poses in closed form`.

### Task F-B8: Point cap is seeded and leaves the global RNG alone  (spec rows: B8) — **parity changes**

**Files:** Modify: `collab_splats/pointcloud/feedforward/base.py` (new `_limit_trues`), `vggtx.py:19,145`, `mapanything.py:21,464` / Test: `tests/pointcloud/feedforward/test_base.py` (or the base test file)

`vggt.utils.helper.randomly_limit_trues` draws from the global numpy RNG. The `max_points` cap is therefore nondeterministic across runs unless the caller seeds globally (the parity harness does, via `np.random.seed` per case), and it perturbs every later global-RNG consumer. E-2's base-vs-base run measures how much real-scene noise this causes.

- [ ] Step 1: Write the failing test:
  ```python
  def test_limit_trues_is_deterministic_and_leaves_global_rng():
      mask = np.ones((4, 8, 8), bool)
      state = np.random.get_state()
      a = _limit_trues(mask, 50)
      b = _limit_trues(mask, 50)
      assert a.sum() == 50 and np.array_equal(a, b)
      assert np.array_equal(np.random.get_state()[1], state[1])
  ```
- [ ] Step 2: Targeted run, expect FAIL: `ImportError: cannot import name '_limit_trues'`.
- [ ] Step 3: In base:
  ```python
  def _limit_trues(mask: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
      """
      Mask keeping at most max_points of its True entries, drawn with a private seeded RNG.

      Args:
          mask: boolean mask.
          max_points: cap on True entries.
          seed: RNG seed; the global numpy RNG is never touched.

      Returns:
          Boolean mask of the same shape.
      """
      idx = np.flatnonzero(mask)
      if idx.size <= max_points:
          return mask
      keep = np.zeros(mask.size, bool)
      keep[np.random.default_rng(seed).choice(idx, size=max_points, replace=False)] = True
      return keep.reshape(mask.shape)
  ```
  - replace both `randomly_limit_trues` call sites and delete the `from vggt.utils.helper import randomly_limit_trues` imports
- [ ] Step 4: The base test file and the vggtx/mapanything creator tests, expect PASS.
- [ ] Step 5: Gates. G′ **+1**. P, using the parity-change protocol: a delta is expected in any case whose mask exceeds `max_points`. Record which.
- [ ] Step 6: Commit `fix(pointcloud): max_points cap draws from a seeded private RNG`.

### Task F-B9: Incremental sfm returns the model it wrote  (spec rows: B9)

**Files:** Modify: `collab_splats/pointcloud/sfm/common.py` (`write_sfm_model`, lane A), `sfm/colmap.py` (HEAD :97-105), `sfm/hloc.py` (HEAD :173-177) / Test: `tests/pointcloud/sfm/test_colmap.py` (row 115, lane A)

colmap and hloc return the in-memory mapper model, which still lists deregistered images. `result_from_reconstruction` then refuses a partial model (a probe confirmed: `registered 3/2 frames`). instantsfm already re-reads from disk (:426).

- [ ] Step 1: The test is lane A's row-115 test. If lane A landed it `xfail`, remove the marker; otherwise add:
  ```python
  def test_reconstruct_returns_only_registered_images(tmp_path, mocked, monkeypatch):
      partial = _recon(NAMES)
      partial.deregister_frame(partial.images[3].frame_id)
      monkeypatch.setattr(colmap_mod.pycolmap, "incremental_mapping", lambda *a, **k: {0: partial})
      data_dir, images_dir = _scene(tmp_path)
      recon = ColmapCreator().reconstruct(data_dir, images_dir=images_dir)
      assert sorted(im.name for im in recon.images.values()) == [Path(n).stem for n in NAMES[:2]]
      outputs, _ = result_from_reconstruction(
          recon, np.full((2, 48, 64), 5.0, np.float32), np.zeros((2, 48, 64, 3), np.uint8), NAMES[:2], min_obs=1
      )
      assert len(outputs.image_paths) == 2
  ```
  - pycolmap 4.0.4 has no `deregister_image`; `deregister_frame` is the API
- [ ] Step 2: Targeted run, expect FAIL: `ValueError: ... registered 3/2 frames` (or XFAIL→strict).
- [ ] Step 3: `write_sfm_model` ends with `return pycolmap.Reconstruction(str(sparse_dir))`, and colmap/hloc `reconstruct` return its result. Delete their in-memory return.
- [ ] Step 4: `tests/pointcloud/sfm`, expect PASS.
- [ ] Step 5: Gates. G′ **+1** (or xfail −1 / passed +1). No P path touched, but run P anyway if `common.py` shares a commit with anything else.
- [ ] Step 6: Commit `fix(pointcloud): colmap and hloc return the model re-read from disk`.

(F-B10 landed in C-4.)

---

## End tasks

### Task E-1: `pointcloud` joins the strict docstring contract  (spec rows: —)

**Files:** Modify: `tests/test_docstring_contract.py:194`

- [ ] Step 3: `RELEASED = frozenset({"preproc", "semantics", "geometry"})` → `frozenset({"preproc", "semantics", "geometry", "pointcloud"})`.
- [ ] Step 4: `tests/test_docstring_contract.py`, expect PASS. Every pointcloud XPASS entry becomes a pass, and pointcloud XFAIL must be 0. Any XFAIL names a defect: fix it in a separate `docs(pointcloud):` commit first.
- [ ] Step 5: G′, with the delta recorded (xpassed → passed).
- [ ] Step 6: Commit `test(pointcloud): pointcloud joins the released docstring contract`.

### Task E-2: Real-scene equivalence  (spec rows: —)

Run in **tmux**, one process at a time (46.6 GB cap). semantics, BA and LC are off.

- [ ] Step 1: Build the shared input once: preprocess `data/tutorial/tutorial_example-video.mp4` with `Reconstructor(config).preprocess()` (uniform, `max_frames: 60`), then `cp -r` its `images/` into four fresh output dirs.
- [ ] Step 2: Baseline worktree: `git worktree add --detach /workspace/collab-splats/.worktrees/pc-base a29efaa0`, with the `third_party` symlinks as in C-0. Run `.build_pointcloud()` there **twice** for each backend (`vggt_omega`, `colmap`) under `cd <tree> && PYTHONPATH=<tree>`, then once at `WT` HEAD.
  - instantsfm is skipped (not installed); hloc is not run (optional extra)
- [ ] Step 3: Compare every zarr array and attr (a script in `$SP/e2_compare.py`):
  - base-vs-base establishes the noise floor
  - base-vs-tip may differ only where section C predicts: B5/B6/B8 on vggt_omega, and B4's crop shift on the colmap sparse. It must not differ for colmap's zarr
  - attrs: the key set and order are equal, and provenance values are equal
- [ ] Step 4: `git worktree remove ../pc-base`. Record the table in the E-5 report.

### Task E-3: Docs  (spec rows: —)

**Files:** `docs/superpowers/CHANGELOG.md`, `CLAUDE.md`, `docs/source/api/pointcloud.rst`, `configs/README.md` (already edited in C-4)

- [ ] Step 3:
  - CHANGELOG: prepend a `Recently completed (2026-09-2x): **pointcloud-release** — ...` entry in the geometry-release shape. Cover: gate numbers, parity deltas per fix, the B10 migration, the E-2 table, and deferred notebook callers. Add it with `git add -f`
  - `CLAUDE.md`: add `**pointcloud-release** (date)` at the top of the "Five newest" list and drop `splats-cleanup`. The in-flight list gets no entry (it has none for this effort)
  - `pointcloud.rst`: add `.. automodule:: collab_splats.pointcloud.sfm.common` in "SfM backends"; (extra) add `sfm.sift_db` and `feedforward.vggt_omega`, which are missing today
- [ ] Step 5: No G′ (docs only). Sphinx build if the docs-site env exists; otherwise skip and say so.
- [ ] Step 6: Commit `docs(pointcloud): changelog, API page and CLAUDE.md for pointcloud-release`.

### Task E-4: Graph refresh

- [ ] `cd $WT && graphify update .`. No commit, because `graphify-out/` is untracked.

### Task E-5: Whole-branch review and report

- [ ] Step 1: Dispatch `superpowers:code-reviewer` over `a29efaa0..HEAD`, read-only, against the spec and this plan. Fix findings as new commits, never amends.
- [ ] Step 2: The final report contains:
  - G′ counts at each lane end and at the tip, against the baseline
  - commit list (`git log --oneline a29efaa0..HEAD`)
  - the parity delta table (B5, B6, B7?, B8)
  - `git diff --stat a29efaa0..HEAD -- collab_splats tests`
  - the E-2 table
  - notebook follow-ups for tutorial-rework (every caller of a renamed or removed symbol under `docs/source/tutorials`)
  - the left-alone list
  - disclosure: the shared-venv regression and the `$SP/stubs` nvdiffrast stub used by every gate

---

---

## Assembly notes

### Spec deviations (each carries a `> Spec deviation:` note at its task)

| Where | Spec says | Plan does | Why |
|---|---|---|---|
| Lanes | A/B parallel, cherry-picked A→B→C | A, then B, then C, each forked after the previous lands | A-14/A-15 edit `feedforward/base.py` and `eval_similarity_calibration.py`, which lane B also edits; B-11 edits `reconstructor.py` |
| `test_registry.py` | lane C | lane B (B-2) | the KeyError→ValueError switch lands in B-2 and must be green there |
| Row 12 | lane B file set | B-11 also edits `reconstructor.py` (one argument) | dropping `image_paths` breaks the caller in the same commit otherwise |
| Row 8 | base `_lc_token_offset = None` | VGGT-X/Omega declare `5` explicitly | the base default becomes `None` and raises |
| Row 14 | one extrinsics shape | unconditional `[:, :3, :]` | callers pass both 3x4 and 4x4 |
| Row 15 | raise on no CUDA | default `None` → `get_device()`; only an explicit `"cuda"` without CUDA raises | a bare raise breaks every CPU run and the CPU parity harness |
| Row 18 | `_PATCH` private | nothing | already private at HEAD |
| Row 42 | abstract where LC-capable | defined only on MapAnything | the one call site is `isinstance(raw, list)`-guarded; an abstract method forces dead stubs |
| Row 46 | output-neutral | moved to Proposed additions | float64 inversion and LC's stride-8 dependence change output |
| Row 89 | `sift_database_valid` goes away | kept as private `_database_holds` | seven validity tests use it directly |
| Row 100 | keys for colmap + hloc | keys for all three creators | F-B10 becomes the same check |
| Row 103 | drop `names` from `_run_sfm` | kept | VDA cache gate, `generate_vda_depth`, `_registered_rows`, `result_from_reconstruction` all key on it |
| Row 114 | `conftest.py` | `tests/pointcloud/_stubs.py` | a builder is called outside any fixture |
| Real scene | vggt_omega + instantsfm + colmap | vggt_omega + colmap | `instantsfm` is not importable in the shared venv (checked 2026-09-26: `ModuleNotFoundError`), and installing is forbidden; instantsfm's path stays covered by unit tests only — disclosed in the final report |
| Round 1 offsets | `reconstructor.py:1212-1216…`, README:171, upstream `:18-57` | `:1214-1218…`, README:132, `:18-56` | line drift at HEAD |

### Other execution risks

- **`pycolmap-cuda-docker` (in-flight, another session)** replaces the colmap binary with `pycolmap-cuda12` and rewrites `build_sift_database` / `colmap_cli_version`. A-7 (matcher map) and A-8 (`colmap_cli_version` raises) touch the same code. Before A-7, check `git log clean/final -- collab_splats/pointcloud/sfm/sift_db.py`: if that work has landed, stop and re-draft A-7/A-8 on top of it.
- **B9 handoff:** A-27 lands the real-pycolmap partial-registration test as `xfail(strict=True)`; F-B9 removes the marker in the fix commit.
- **Line numbers are HEAD `11c5d7c7`.** Every edit is anchored by its quoted text; re-grep before editing.
- **Tests outside G′** that a task touches get their own targeted run, named in the task (`tests/utils`, `tests/preproc`, `tests/dashboard` + dashboard `--smoke`, `tests/test_feedforward_logging.py`, `tests/integration/test_pipeline_cu121.py`).
- **Counts-only deltas** (B-5, B-6, B-8, B-9, B-29, B-30, B-32): the executor lists the deleted test names in the commit body.


### Cross-lane notes (lane A)


- **Lane table mismatch** with part0 :87-89: see the header. Deduplicate against lane C's sfm rows.
- **ff/base.py edits in lane A:** :45, :328-333, :1375, and H4 before :1008. Also `tests/pointcloud/test_feedforward_reproject.py:45`. Lane B also edits base.py, so serialize.
- **Row 83** (VDA checkpoint helper) waits on lane B rows 55/83. It is not drafted here.
- **Reconstructor (lane C)** should consume `creator.provenance()` (A-9).
  - Call it BEFORE `reconstruct()`, and certainly before `del creator` (:1255).
  - Replace the if/elif block at :1277-1288.
  - Replace the manual cuda cleanup at :1256-1258 with `pytorch_gc()`.
- **`tests/wrapper/test_sfm_stage.py:105`** patch target → `collab_splats.pointcloud.sfm.colmap.colmap_cli_version` once provenance moves.
- **`evals/scripts/eval_similarity_calibration.py`:** lane B B-15 edits `layer_index` at :252, and A-14 edits :49, :61-65, :181-198, :255. Adjacent hunks, so serialize.
- **High conflict risk: the in-flight `pycolmap-cuda-docker` spec** (docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md) replaces the colmap 3.10 binary with `pycolmap-cuda12`.
  - That rewrites `build_sift_database` and `colmap_cli_version`.
  - A-7 (matcher map) and A-8 (cli version) may become moot. Land them only if that work has not started, or rebase them onto it.
- **B9** (the partial-registration gap) is pinned by A-27's strict xfail. Its fix belongs to whoever owns the reconstructor/depth_align input model.

---

## Proposed additions (need approval — NOT implemented by any task)

### Round 1 (part 0)

These are not implemented by this plan. Each could change output, a signature, or the docs build, or rests on an unverified claim.

1. **`vggt_omega.py:66` crop citation is unverified.** No clone of `facebookresearch/vggt-omega @ 39a0cb8` has been checked. Proposal: fetch the pinned file and fix or confirm the line range. If it cannot be verified, reduce it to repo + commit + function name.
2. **`vggt_omega` on the API page.** Adding `.. automodule:: collab_splats.pointcloud.feedforward.vggt_omega` imports an optional dep at Sphinx build time. It could break the docs build where vggt-omega is not installed. Proposal: add it with `autodoc_mock_imports`, or leave it off.
3. **`generate_vda_depth(frames=None)` on a depth-cache hit.** This would skip the first keyframe decode in `_run_sfm` (`reconstructor.py:1238-1240`), because on a cache hit that stack is used only for `len()`. It is a signature change on a released path. Output is the same, but the call contract changes.
4. **LoGeR `_preprocess` / `_forward` one-line docstrings** (`loger.py:299`, and the same shape elsewhere in the backends). Reshape them to the summary-on-own-line form with Args/Returns. This is prose only, but it widens Round 1 to docstrings with no contract hit. Proposal: fold it into Round 2's per-file tasks where those methods are touched anyway.
5. **`# ── X ──` dividers → `########`** in `feedforward/base.py` and `vggtx.py`. This is cosmetic, reduces nothing, and is left out under the reduction directive unless wanted for consistency.

### Lane A / helpers

Each of these could change output or behavior, so none is in the tasks above.

1. **Lazy vocab-tree fetch.** A-5 calls `fetch_vocab_tree()` before `ensure_sift_database`, so a cached DB still triggers the fetch (a cached file plus sha check).
   - Proposal: pass a thunk, or fetch inside ensure only on rebuild.
2. **Required kwargs on `build_sift_database`.** Its only callers now go through `ensure_sift_database`, so its defaults are dead.
3. **Row 79 inexactness.** If A-24 fails P, keep the manual `R @ x + t` permanently. That needs a decision.
4. **`colmap_cli_version` ordering.** With A-8 it raises, and it is called from `provenance()` after mapping.
   - A missing binary with a cached DB would fail only after a full mapping run.
   - Proposal (lane C): call `creator.provenance()` before `reconstruct()`.
5. **Eval-script empty-token case now raises.** `eval_similarity_calibration.py` records NaN via its pair-loop `except Exception`, where it used to record a silent 0/empty mean.
6. **`.to(fmap.device)` in `_sample_at_source_pixels` (A-16)** also fixes a latent crash when fmap is on GPU. P must confirm the CPU path is unchanged.
7. **`lift_features` asserts :263-268 → ValueError.**
8. **Legacy (N,H,W,1) depth zarrs now raise in `lift_features` (A-15).** Old stores need a re-run or a one-line squeeze at load.
9. **LC verify with `token_offset >= tokens_per_img`** now raises instead of rejecting the loop.
10. **Existing InstantSfM DBs rebuild once.** They have no `<db>.json` sidecar yet, so the first run after A-5 re-extracts.

---

### Lane B

Each item below could change output, so none is folded into a task.

1. **Row 46: merge `_raw_to_world_points`'s own unprojection onto `unproject_depth_map_to_point_map`, and drop its `subsample` parameter.**
   - `_raw_to_world_points` (ff/base.py:373-424) inverts extrinsics in **float64** (`invert_poses(extr_4x4.astype(np.float64)).astype(np.float32)`). vggt's `unproject_depth_map_to_point_map` uses its own float path, so the world points differ at the ~1e-6 level.
   - `FeedforwardResult.world_points` is on the parity path (BA fields), so P would move.
   - Dropping `subsample` changes LC: `geometry/loop_closure/wrapper.py:319` calls it with the default `subsample=8`, and `graph.py:614, 624` rely on that stride-8 point count for anchor/sequential scale.
   - Belongs with bug B6 (double unprojection), which removes the second unprojection in `_postprocess` anyway.
   > Spec deviation (row 46): moved out of the output-neutral set. Evidence: float64 vs float32 inversion, and the stride-8 dependence in `graph.py:614, 624`.

2. **vggtx `unproject_and_filter_points` `hasattr(images, "cpu")` guess (vggtx.py:126-129).**
   - Every caller passes a torch tensor. Replacing the guess with `to_numpy(images)` changes bf16 handling, if any caller ever passes bf16 images.

3. **LC wrapper :334 guard and colors heuristic (`wrapper.py:334-347`).**
   - After B-16, every backend's `raw_lc` carries `depth` and `depth_conf`, so the `if` is always true and could go.
   - The `frames_hw3.max() > 1.0` heuristic guesses the pixel range. Replace it with the known per-backend range: VGGT family [0,1], MapAnything supplies `colors`. That changes behavior for any frame tensor already in [0,255].

4. **`_decode_verify_geometry` (ff/base.py:340-370) onto the shared unprojection.**
   - It is a third copy of depth→world. It should reuse `_raw_to_world_points` once row 46 lands; the numeric caveat is the same as item 1.

5. **vggtx `conf_threshold` semantics vs `pointcloud.utils.confidence_mask`.**
   - `unproject_and_filter_points` reads `>1.0` as a percentile and `≤1.0` as raw. Unifying it with `confidence_mask` changes which pixels survive at the boundary.
   - Left alone per the spec; listed for completeness.

6. **`compute_multiview_depth_confidence` `torch.linalg.inv` at ff/base.py:612 (bug B7).**
   - Switching it to the float64 `invert_poses` changes mv_conf values at the 1e-6 level.
   - Belongs with lane C's Section C bugs; noted here because the function is in lane B's file.

### Lane C / fixes / end

These could change output or behavior, so they are **not** in the tasks above.

1. **`_run_feedforward` raises when a creator has no `outputs`** (reconstructor HEAD :566-575). Today it logs a warning and continues without `pointcloud.zarr`, and every later stage then fails on a missing file. Cost: tests that mock creators with `outputs=None` need a result stub.
2. **`eval_verification.py:213` ground-truth rounding → floor**, for the same reason as B5. This changes eval numbers.
3. **`pytorch_gc()` in `_run_feedforward`** (HEAD :577-584; spec Follow-up), replacing the inline `import torch as _torch` + `empty_cache` + `synchronize`. It adds `gc.collect()`, so the release timing changes; the output does not.
4. **Exclude `use_depths` from the instantsfm block keys** (C-4 makes it settable, because it is a dataclass field). Excluding it keeps the config surface at the three `base.yaml` keys; accepting it exposes a knob nobody has measured.
