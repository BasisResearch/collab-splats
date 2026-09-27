# setup.sh prerequisite gating + README install contract

Date: 2026-09-27 · Branch: `clean/final` · Status: design approved, plan pending

## Problem

`setup.sh` is the general installer (Docker builder, bare host, re-run in a built env), but it
demands nvcc unconditionally on line one.

- Runtime Docker image ships no nvcc by design → `setup.sh` exits 1 before doing anything.
- Measured 2026-09-27 in the runtime pod (motivates the fix): `uv sync --locked --all-extras --dry-run` against
  `/opt/venv/reconstruction` builds **none** of bae / gsplat / fused-ssim / nvdiffrast — only
  editable collab-data / collab-splats reinstall + InstantSfM extras prune (setup.sh re-adds).
  The nvcc gate is the only thing failing.
- README Install section is stale in places (see §README).

## Responsibility split (agreed)

| Layer | Owner |
|---|---|
| NVIDIA driver + GPU | user |
| gcc/g++, CUDA 12.1 toolkit (nvcc) | user; `setup.sh` warns if absent, prints recipe if a build fails |
| uv | user |
| venv, locked deps, CUDA extension compiles, CCCL overlay, InstantSfM extras, `third_party/` clones, weights | `setup.sh` |

`setup.sh` never installs system packages (no apt, no micromamba, no `/usr/local/cuda` symlink).
Docker image = the hands-off path. No `setup/cuda_toolkit.sh` helper (rejected: untested, needs
root for the symlink, YAGNI).

## setup.sh changes

Minimal: no dry-run parsing, no version checks. uv already reports whether it must compile.

### 1. nvcc missing → warn, don't exit

- Move the recipe heredoc into a function `cuda_toolkit_help` (text unchanged, plus one line:
  "or use the Docker image; driver + toolkit are yours to provide — README Prerequisites").
- `/usr/local/cuda/bin/nvcc` present → current behavior (CUDA_HOME, PATH, CCCL overlay).
- absent → `HAVE_NVCC=0`, one-line warning ("no nvcc — fine if the CUDA extensions are already
  built"), skip the CCCL overlay block entirely.

### 2. Recipe on failure

- Both `uv sync` calls (deps-only and full) get
  `|| { [ "$HAVE_NVCC" = 0 ] && cuda_toolkit_help; exit 1; }`.
- Built env (runtime Docker pod): nothing compiles → sync succeeds, warning is the only trace.
- Bare host, no toolkit: uv fails on the first CUDA build; recipe prints directly under uv's
  error (uv's error already names the package).

### 3. Header comment

One line documenting `SETUP_DEPS_ONLY=1` (Docker pass 1: lock only, no project install, keeps
the CUDA compile layer cached across source edits). Flag stays — removing it would recompile
gsplat (~2.5 h) on every source edit, and moving it into the Dockerfile duplicates the
nvcc/CCCL/MAX_JOBS env.

### Rejected

- dry-run gate (grep uv's human output): ~15 lines, version-fragile, duplicated flags.
- CUDA 12.1 version warning: torch rejects major mismatch itself; no minor-mismatch incident.
- `setup/cuda_toolkit.sh` helper: untested, needs root for the symlink.

### Out of scope

- Smoke test at the end: unchanged (already works without nvcc — checks AOT `.so` presence).
- `/workspace/setup_profile.sh` (outside repo; `apt install rclone` downgrade) — separate.

## README Install rework

Only the Install section. Uncommitted intro hunk at top of README.md is someone else's WIP —
do not touch, commit Install hunks only.

1. **Prerequisites (you provide)** — moved to first: driver + GPU, gcc/g++, CUDA 12.1 toolkit
   (must match torch cu121; only needed when extensions compile), uv. One line: or use Docker (§5).
2. **Install (`setup.sh` provides)** — list what it does; all four compiled extensions (bae,
   gsplat, fused-ssim, nvdiffrast); idempotent; runs in the Docker image without nvcc because
   nothing compiles.
3. **collab-data** — fix: locked path dependency at `/workspace/collab-data`, required for
   `uv sync`; setup.sh clones it (GitHub auth) if missing. Drop the "best-effort / post-sync /
   `uv pip install git+...`" text (contradicts the `uv pip` warning above it).
4. **rclone** — Linux install via `curl https://rclone.org/install.sh | sudo bash`; note Ubuntu
   apt ships 1.53 (2020) and overwrites a newer binary.
5. **Docker** — add: runtime image has no nvcc by design; `setup.sh` re-run there is safe.

## Testing

- Runtime pod (no nvcc): `bash setup.sh` completes; log shows the no-nvcc warning; smoke test
  passes; InstantSfM extras present after.
- Forced compile without nvcc: `UV_PROJECT_ENVIRONMENT=<scratch empty venv>` (in scratchpad,
  deleted after) → uv fails on a CUDA build → recipe printed under the error, exit 1.
- No Docker rebuild required. Editing setup.sh invalidates the cached CUDA layer — flag to user
  before the next build; builder has nvcc so behavior there is identical.
