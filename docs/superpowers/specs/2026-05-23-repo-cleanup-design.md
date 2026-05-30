# Repo Cleanup — patches/ + VGGT-Long + gitignore hygiene

**Date:** 2026-05-23
**Branch:** refactor/cu121
**Status:** approved

## Problem

Three categories of dead weight in the repo:

1. `patches/` — single backup file (`nerfstudio-pyproject.orig.toml`) left from
   an inline-patching approach in `setup_nerfstudio.sh`. The setup script patches
   `/workspace/nerfstudio/pyproject.toml` inline at install time; this backup file
   is never read by any script or test, and the original content is already
   documented by context in the setup script itself.

2. `third_party/VGGT-Long` — git submodule added speculatively. No production
   code imports it. `evals/runners/run_vggt_long.py` was scaffolded alongside it
   but never integrated into the eval harness. `VGGT-SLAM` and `VGGT-X` cover the
   active use cases.

3. `vendor/` and `third_party/` not gitignored — these directories are populated
   by setup scripts (`setup.sh`, `setup_feedforward.sh`, `git submodule update`).
   Any new untracked content (e.g. a new vendor clone) would show up in `git status`
   and risk accidental `git add`.

## Changes (one squash commit)

### Delete `patches/`
- `git rm -r patches/`
- Remove line in `setup_nerfstudio.sh`: `# Original preserved at patches/nerfstudio-pyproject.orig.toml.`

### Remove `third_party/VGGT-Long` submodule
- `git submodule deinit -f third_party/VGGT-Long`
- `git rm third_party/VGGT-Long`
- Remove `[submodule "third_party/VGGT-Long"]` stanza from `.gitmodules` (handled by `git rm` on the submodule path)
- `rm -rf .git/modules/third_party/VGGT-Long`

### Remove associated runner
- `git rm evals/runners/run_vggt_long.py`

### Update `third_party/README.md`
- Remove `VGGT-Long/` row from the "Current entries" table

### Update `.gitignore`
Add two entries under a new "Setup-script-populated dirs" section:
```
# Populated by setup scripts / git submodule update — not tracked directly
vendor/
third_party/
```

> Note: gitignore entries do not affect already-tracked submodule paths
> (`third_party/VGGT-SLAM`, `third_party/VGGT-X`). They only prevent new
> untracked content from surfacing in `git status`.

## Files changed

| File | Change |
|---|---|
| `patches/nerfstudio-pyproject.orig.toml` | deleted |
| `third_party/VGGT-Long/` | submodule deregistered + dir removed |
| `evals/runners/run_vggt_long.py` | deleted |
| `setup_nerfstudio.sh` | remove one comment line |
| `third_party/README.md` | remove VGGT-Long table row |
| `.gitignore` | add `vendor/` and `third_party/` entries |
| `.gitmodules` | VGGT-Long stanza removed by `git rm` |

## Out of scope

- `stage/` — kept as useful reference
- `examples/` — kept as CLI entry points
- `vendor/bae/`, `third_party/xfeat/` — active dependencies
- `collab_splats.egg-info/` — not addressed here

## Commit message

```
chore: drop patches/ backup, VGGT-Long submodule, and gitignore vendor/+third_party/

patches/nerfstudio-pyproject.orig.toml was a reference backup for
setup_nerfstudio.sh inline patching. Not read by any script.

third_party/VGGT-Long was added speculatively; no production code
imports it. run_vggt_long.py removed alongside.

vendor/ and third_party/ added to .gitignore to prevent accidental
tracking of setup-script-populated dirs. Does not affect existing
tracked submodules (VGGT-SLAM, VGGT-X).
```
