# `vendor/`

Locally-cloned, **gitignored** dependencies. Populated at install time by
scripts at the repo root, not by `git clone`.

## Purpose

This directory holds external sources that don't fit the submodule model in
`third_party/`. Typical reasons:

- The dependency needs **local patches** applied after clone (e.g. CUDA
  version compatibility) that can't cleanly live in a pinned submodule.
- The dependency is **not pip-installable** and we need `sys.path` access to
  its source layout.
- The dependency is **optional** — only the install scripts for specific
  pipelines populate it, so contributors who don't need it never download it.

The directory itself is listed in `.gitignore`, so contents are never
committed.

## Populating

Run the install script for the pipeline you need:

```bash
# Loop closure / feedforward
bash setup_feedforward.sh

# Bundle adjustment deps (bae) are installed by setup.sh via pyproject.toml git URL
```

If a `vendor/<name>/` directory is empty, the corresponding setup script
hasn't been run — run it.

## Policy

- Setup scripts own the clone + patch lifecycle. They are idempotent: if the
  target already exists, the script skips re-cloning.
- Local patches applied after clone must be encoded in the install script, not
  applied by hand. Future reclones need to reproduce them.
- Imports from `vendor/` typically rely on a runtime `sys.path` insert (see
  `pyproject.toml` and `collab_splats/semantics/retrieval.py` for the SALAD
  example). New entries should follow that pattern or be installed editable
  by the setup script.

## Current entries

| Path | Upstream | Populated by | Used by |
|---|---|---|---|
| `salad/` | `serizba/salad` | `setup_feedforward.sh` | DINO-SALAD aggregator for loop closure (`collab_splats/semantics/retrieval.py`). Loaded via `sys.path` insert. **Do not delete without an ADR** — see `worklog/history/specs/2026-05-03-feedforward-env-debug.md`. |
| `xfeat/` | `verlab/accelerated_features` | Manual clone (will be in setup script) | Lightweight local feature extractor for camera localization (`collab_splats/localization.py`). Loaded via `sys.path` insert to `modules/`. |

## See also

- `third_party/README.md` — pinned upstreams as submodules, populated by
  `git submodule update`.
