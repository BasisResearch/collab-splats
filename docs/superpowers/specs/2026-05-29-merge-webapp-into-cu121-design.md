# Merge feat/webapp → refactor/cu121 Design

**Date:** 2026-05-29
**Author:** Tommy
**Status:** Approved

## Goal

Integrate all `feat/webapp` work (FastAPI app, semantics viz, mesh gen, localize tab) into `refactor/cu121`, which becomes the single integration branch going forward. `feat/webapp` is retired after merge.

## Approach

`git merge feat/webapp --no-ff` on `refactor/cu121`. Conflicts resolved manually per file. Merge commit preserves full history from both branches.

Rejected alternatives:
- **Rebase**: 60 commits × per-commit conflicts = high pain, rewrites SHAs
- **Cherry-pick**: webapp commits interleaved with dashboard fixes, hard to isolate cleanly

## Branch Summary

| Branch | Key changes |
|---|---|
| `feat/webapp` | FastAPI app (preprocess/reconstruct/localize/visualize tabs), semantics 3-panel viz, mesh gen, camera frustums, ground plane, localization cache |
| `refactor/cu121` | frame_sampling overhaul (ffmpeg, rotation-aware), VGGTOmega, SL(4) LC, semantics refactor, dashboard auto-discover, tutorial config unification |

Merge base: `d2c9550` (FastAPI webapp migration plan commit)

## Expected Conflict Zones

- `collab_splats/dashboard/` — both branches heavily modified; resolve per hunk
- `collab_splats/semantics/` — cu121 has semantics refactor squash; webapp adds 3-panel viz + query
- `setup.sh` / pip deps — both may have added packages
- `CLAUDE.md` / worklog docs — likely trivial

**Resolution strategy:** per-conflict manual review. Prefer cu121 for backend/pipeline logic; prefer webapp for net-new UI/viz additions.

## Net-New from webapp (no conflict expected)

- `collab_splats/webapp/` — entire FastAPI app
- Webapp static assets (HTML/CSS/JS)
- `.superpowers/brainstorm/` artifacts

## Post-Merge Verification

- `pytest tests/` — full test suite (env now ready)
- Smoke-check FastAPI app starts cleanly
- No regression on eval pipeline (`evals/eval_gt.py`)
