# `third_party/`

Pinned upstream sources, tracked as **git submodules**.

## Purpose

External repositories used as-is. The parent repo pins each entry to a specific
upstream commit via `.gitmodules`, so every checkout of collab-splats resolves
to the same source tree. This matters most for **eval baselines**, where
reproducibility depends on running an identical version of the baseline code.

## Populating

```bash
# At initial clone:
git clone --recursive https://github.com/BasisResearch/collab-splats

# Or after the fact:
git submodule update --init --recursive
```

If a `third_party/<name>/` directory is empty, the submodule hasn't been
initialized — run the command above.

## Policy

- **Do not patch in place.** The submodule SHA is the contract; local edits
  drift silently and break reproducibility. If a dependency needs local
  modifications (e.g. a CUDA-compat patch), put it under `vendor/` instead and
  apply the patch via an install script.
- New entries must be added with `git submodule add` (which updates
  `.gitmodules` and pins a SHA). Don't `git clone` into this directory.

## Current entries

| Path | Upstream | Used by |
|---|---|---|
| `VGGT-SLAM/` | `MIT-SPARK/VGGT-SLAM` | `evals/runners/run_vggt_slam.py` |
| `VGGT-X/` | `Linketic/VGGT-X` | `setup_feedforward.sh` |

## See also

- `vendor/README.md` — gitignored, locally-patched dependencies populated by
  install scripts.
