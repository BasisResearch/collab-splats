# setup.sh prerequisite gating + README install contract

Date: 2026-09-27 · Branch: `clean/final` · Status: design approved, plan pending

## Problem

`setup.sh` is the general installer (Docker builder, bare host, re-run in a built env), but it
demands nvcc unconditionally on line one.

- Runtime Docker image ships no nvcc by design → `setup.sh` exits 1 before doing anything.
- Measured 2026-09-27 in the runtime pod: `uv sync --locked --all-extras --dry-run` against
  `/opt/venv/reconstruction` builds **none** of bae / gsplat / fused-ssim / nvdiffrast — only
  editable collab-data / collab-splats reinstall + InstantSfM extras prune (setup.sh re-adds).
  The nvcc gate is the only thing failing.
- README Install section is stale in places (see §README).

## Responsibility split (agreed)

| Layer | Owner |
|---|---|
| NVIDIA driver + GPU | user |
| gcc/g++, CUDA 12.1 toolkit (nvcc) | user; `setup.sh` checks only when needed |
| uv | user |
| venv, locked deps, CUDA extension compiles, CCCL overlay, InstantSfM extras, `third_party/` clones, weights | `setup.sh` |

`setup.sh` never installs system packages (no apt, no micromamba, no `/usr/local/cuda` symlink).
Docker image = the hands-off path. No `setup/cuda_toolkit.sh` helper (rejected: untested, needs
root for the symlink, YAGNI).

## setup.sh changes

### 1. Gate toolkit on "will compile"

- Before the nvcc block: `uv sync --locked --all-extras --dry-run` with the same flags the real
  sync will use (`--no-install-project` when `SETUP_DEPS_ONLY=1`).
- Grep the plan for `^ \+ (bae|gsplat|fused-ssim|nvdiffrast) ` → `NEEDS_CUDA_BUILD=1`.
- `NEEDS_CUDA_BUILD=0` → skip nvcc check AND CCCL overlay; log one line saying so.
- `NEEDS_CUDA_BUILD=1` → existing nvcc + CCCL block, unchanged in behavior.
- dry-run itself failing (e.g. stale lock) → let the real `uv sync --locked` report it; treat as
  `NEEDS_CUDA_BUILD=1` so no build ever proceeds without a toolkit.
- Move the collab-data presence check above the dry-run (the resolve needs the path dep).

### 2. Clearer prerequisite errors

nvcc missing (only reachable when a compile is needed) message states:

- which packages need compiling (from the dry-run list)
- the three options: use the Docker image; `apt-get install cuda-nvcc-12-1 cuda-libraries-dev-12-1`;
  the micromamba recipe (kept as-is)
- that the driver/toolkit are the user's responsibility (link README Prerequisites)

nvcc present but release ≠ 12.1 → **warning**, not failure (torch cpp_extension tolerates a
minor-version mismatch; major mismatch it rejects itself).

### Out of scope

- `SETUP_DEPS_ONLY` / Docker two-pass flow: unchanged.
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

- Runtime pod (no nvcc): `bash setup.sh` completes; log shows toolkit skipped; smoke test passes;
  InstantSfM extras present after.
- Gate logic: run the grep against a captured dry-run with and without a `+ gsplat @ git+...`
  line (captured text in plan, no new pytest — shell only).
- Forced compile path without nvcc: `UV_PROJECT_ENVIRONMENT=<scratch empty venv>` dry-run → gate
  trips → new error message printed, exit 1, before any build starts.
- No Docker rebuild required (setup.sh change invalidates the cached CUDA layer — flag to user
  before next build; builder has nvcc so behavior there is identical).
