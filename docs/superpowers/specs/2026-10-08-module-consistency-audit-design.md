# Module consistency audit — design

- **Status:** designed 2026-10-08; audit not run; fix plan written only after triage
- **Supersedes:** phases 3-7 of [2026-09-26-consistency-design.md](2026-09-26-consistency-design.md)
  - phases 1, 1b, 2 of that spec are landed on `clean/final` and stay as-is
  - its 2026-09-26 audit predates the geometry, pointcloud, mesh, splats, reconstructor,
    localization and dashboard releases; its findings are not carried over unverified
  - phase 4 `SceneLayout` / kind-tagged registry conflict with later rules (no new classes,
    no layout constants) and are dropped unless this audit re-derives them
- **Target:** `collab_splats/` on `clean/final` @ `0ed7dd09`
- **Out of scope:** `evals/`, `tests/`, notebooks, scripts — counted as callers only

## Goal

One implementation per job, one way of doing each job, across `collab_splats/` subpackages.

- no function implemented twice across modules
- copies that disagree on a convention found and fixed (highest bug risk)
- same job done the same way in every package
- one name per meaning in public APIs

## Dimensions

| Dimension | Finds | Fix shape |
|---|---|---|
| Redundant implementation | same logic in 2+ places | keep one, delete the rest |
| Convention divergence | copies disagreeing on pose frame, K resolution, image layout/dtype, NaN policy, units | test-first bug fix, then dedup |
| Pattern divergence | device selection, GC, logging, registry, return kind, error type, zarr codec, config plumbing | adopt the majority / existing shared helper |
| API naming | one meaning, several param or return names | rename to the most-used name |

## Process

Four steps; steps 1-3 are read-only.

### 1. Inventory

Eight read-only agents, one per slice, on the main checkout.

| Agent | Slice |
|---|---|
| 1 | `geometry/` |
| 2 | `semantics/` |
| 3 | `pointcloud/` |
| 4 | `splats/` |
| 5 | `preproc/` + `utils/` |
| 6 | `mesh/` + `localization/` |
| 7 | `dashboard/` + `viewer.py` |
| 8 | `reconstructor.py` + `__main__.py` + `remote.py` |

One JSON row per function or method, private included:

- `file:line`, `qualname`, signature
- `does`: one-line semantic from a shared verb vocabulary (`invert_pose`, `rescale_intrinsics`,
  `read_image`, `open_zarr`, `to_numpy`, `select_device`, ...); agents add verbs when none fit
- `conventions`: pose frame (w2c / c2w, 3x4 / 4x4), K resolution (model / full),
  image layout and dtype, device, NaN policy, units — only those that apply
- `patterns`: logging, device selection, GC, registry, return kind, error type, zarr codec, config plumbing
- `params`: names of semantically loaded params (`conf_*`, `*_dir`, `poses` / `extrinsics`, ...)
- `callers`: call count inside `collab_splats/`

### 2. Cross-join

Done by the orchestrating session over the merged inventory.

- verb synonyms merged first, so `inv_pose` and `invert_extrinsics` land in one group
- redundancy candidates: groups by `does` with more than one row
- convention candidates: rows in one group with conflicting `conventions`
- pattern tally: per package, which pattern each job uses
- naming table: param names grouped by meaning

### 3. Verify

One adversarial agent per batch of candidate groups, reading the code, not the inventory.

- `REAL`: same semantic; copies interchangeable or differ by accident
- `INTENTIONAL`: difference is load-bearing (e.g. PIL-exact preproc matching an upstream loader);
  reason recorded, no fix
- `BUG`: copies disagree on a convention and one is wrong; must name the input giving wrong output
- `DEAD`: copy with zero callers; `main` and `origin/db-*` grepped before the verdict
- unverified candidates are dropped, not reported

Each finding is tagged `conflicts_with` when it touches files changed on an unlanded branch:
`clean/localization`, `feat/rgbd-ba-cf`, `perf/inplace`, `perf/splats-speed`, `clean/rclone-api`, `cleanup/fc-*`.

### 4. Report and triage

- published as an artifact; the same findings table appended to this spec under `## Findings`
- columns: id, dimension, verdict, files, canonical, proposed fix, LOC delta, risk, `conflicts_with`, triage
- the user fills `triage` (keep / drop / defer); the plan covers `keep` rows only

## Canonical choice

For each `REAL` group, in order:

1. a copy already in `utils/` or `geometry/transforms.py` / `geometry/projection.py` wins
2. otherwise the copy with more callers, tests, and correct conventions
3. a new shared home only when 2+ packages need it — a plain function, never a class

## Fix constraints

- behavior-preserving, except `BUG` rows
- `BUG` rows are test-first on a fixture that can see the convention:
  non-identity poses, off-centre principal point, cropped or non-square images;
  a test that passes on the unfixed code is rejected
- losing copies deleted outright: no shims, no re-exports, no wrappers
- tunables stay function keyword defaults; one call per line; absolute imports; docstring contract
- renames take the name with the most call sites
  - a rename that changes a checkpoint key, zarr attr, or config key is flagged and not done blind

## Fix phase

Only after triage; its plan goes in `docs/superpowers/plans/`.

- **Branch:** `clean/consistency-audit`, worktree `.worktrees/consistency-audit`,
  forked from the `clean/final` tip at plan time
- **Commits:** one group per theme, independently revertible
  - `BUG` fixes are `fix(scope):` commits ahead of any dedup touching the same code
- **Deferred:** `conflicts_with` rows wait for their branch to land; listed in the plan, not done
- **Tests:**
  - `cd .worktrees/consistency-audit && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest ...`
  - gated on a printed `collab_splats.__file__`; `third_party/*` symlinked in
  - control run of the same per-package gates on the fork point, recorded first
  - pass/fail counts match control except new `BUG` tests (fail on fork, pass on branch)
  - exit codes checked directly, never through a pipe
- **Behavior proof for dedups:** existing tests, plus a bit-identical array comparison on the
  tutorial scene for any pipeline stage whose code path changed
- **Landing:** squash per theme onto `clean/final` immediately; backup ref under
  `refs/backup/consistency-audit/`; push is the user's call
- **Bookkeeping:** CHANGELOG entry; CLAUDE.md `consistency` line points here, phases 3-7 marked superseded
