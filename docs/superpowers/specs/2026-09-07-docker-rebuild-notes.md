# Docker rebuild — notes, not yet executed

Deferred 2026-09-07. Everything below is measured or read off the tree, not planned.
No docker binary exists in this environment, so the rebuild runs on a host that has one.

## Why it is owed

- venv repaired in place 2026-09-06: retired `gsplat-rade` 1.4.0 -> upstream `d2f5c0f`
  (+ `fused-ssim`). The published image predates that repair.
- consequence: a container from the current image does NOT match the working env —
  `gsplat.losses` missing, `tests/splats` a collection error.
- `setup.sh` already carries the whole repair recipe (`fd8094a2`); the Dockerfile
  already calls it. The rebuild is running it, not writing it.

## Does gsplat ship prebuilt binaries — no

- PyPI `gsplat 1.5.3` publishes `gsplat-1.5.3-py3-none-any.whl` + an sdist.
  `py3-none-any` = pure python, zero compiled objects. Verified 2026-09-07.
- `https://docs.gsplat.studio/whl/pt25cu121/gsplat/` -> HTTP 404. No wheel index.
- `gsplat/cuda/_backend.py`: tries `from gsplat import csrc` (AOT `.so`), else falls
  back to JIT `build_and_load_gsplat()`. Someone compiles either way.
- our install is AOT: `site-packages/gsplat/csrc.so`, 89,967,240 bytes,
  built 2026-09-06 04:16. Compile paid once at install, not per process.
  **Keeping it AOT inside the image is the whole point of the rebuild.**
- separately, the PyPI artifact is unusable on content grounds: we pin git `d2f5c0f`
  (main @ 2026-07-09) because the v1.5.3 TAG lacks `gsplat.losses` and `extra_signals`.

## Would gsplat 1.6 / pinning main help — no, and it raises the floor

Recorded in `pyproject.toml` [tool.uv.sources]:

- `4561ac4` (2026-06-01) needs CCCL `cuda::ceil_div` -> CUDA >= 12.3.
- `31f5b3d` (2026-06-08) needs `c10d::wait_tensor` -> torch >= 2.7.

So moving forward means bumping the base image AND torch (we are on 2.5.1+cu121).
It removes no compilation. Treat as its own effort, not part of the rebuild.

## Build environment — the parts that are load-bearing

- **system toolkit required.** Every `nvidia-cuda-nvcc-cu12` wheel (12.1 … 12.8) ships
  only `ptxas`, not a compiler. Measured 2026-09-06. `setup.sh` fails fast with the
  recipe rather than falling back.
  - bare host: `apt-get install -y --no-install-recommends cuda-nvcc-12-1 cuda-libraries-dev-12-1`
  - image: `nvidia/cuda:12.1.1-devel-ubuntu22.04` (what the Dockerfile uses)
- **`nvidia/cuda:12.1.1-devel` alone is NOT enough.** gsplat `d2f5c0f` includes
  `<cuda/std/optional>`, which arrived in CCCL 2.2. CUDA 12.1 ships
  `cuda/std/detail/libcxx/include/optional` but not `cuda/std/optional`.
  Probed: cccl `12.3.101` no, `12.4.127` no, **`12.6.77` yes**.
- **the overlay reaches nvcc ONLY through `NVCC_PREPEND_FLAGS`.** `CPLUS_INCLUDE_PATH`
  and `CPATH` rank below the toolkit's own `-I` on the host preprocessor's search
  order, so the outer header resolves to the overlay while the nested one falls back
  to 12.1's pre-2.2 libcxx: `fatal error: __config: No such file or directory`.
- overlay is **build-time only**. Confirmed on the live env 2026-09-07: gsplat is
  compiled and importable while `/opt/cccl-12.6.77` and
  `/usr/local/cuda/include/cuda/std/optional` are both absent. Runtime stage needs
  no CCCL.
- **`MAX_JOBS` ceiling 6.** torch's `cpp_extension` defaults to one job per CPU; this
  host reports 96 cores and ignores the 46.6 GB cgroup cap. One `cicc` peaks 7.3 GB.
  `setup.sh` defaults 2, Dockerfile `ARG MAX_JOBS=4`.
- debug include problems with a 3-line `.cu` probe, never a 10-minute rebuild.

## Dockerfile — current shape, and the gap

Already right:

- builder stage runs `bash setup.sh`, so the image bakes the AOT `csrc.so`.
- `TORCH_CUDA_ARCH_LIST` cross-compiles; no GPU needed during build.
- runtime stage copies `/opt/venv/reconstruction`, the uv python install, and
  `/workspace/collab-splats` — the editable installs resolve only if that path
  matches the builder's, so do not move it.

Gaps to settle at rebuild:

- base image stays 12.1.1-devel and setup.sh overlays CCCL every build. Alternative
  is a 12.6-devel base + a torch bump, which is the gsplat-1.6 question above.
  Cheapest correct move for now: leave 12.1.1 + overlay.
- Dockerfile `CPATH=/usr/local/cuda/include` is set globally in the builder ENV. It is
  the toolkit's own include, not the overlay, so it does not trip the `__config`
  trap — but do not extend it with the overlay.
- `--build-arg MAX_JOBS=N`: pick from the BUILD host's RAM, not its core count.
- `uv sync` prunes the `--no-deps` installs (instantsfm, pyceres, scikit-sparse,
  easydict). setup.sh reinstalls them after its own sync; any later bare `uv sync`
  silently removes them again.
- `collab-data` and the VDA checkpoint are best-effort in setup.sh — a credential-less
  or network-less build skips them and the image is incomplete by design. Re-run
  setup.sh at deploy.

## Verify after rebuild

1. `docker build --build-arg MAX_JOBS=<host-appropriate> ...` completes.
2. runtime container with `--gpus`: `import torch, bae, gsplat, fused_ssim` (the
   setup.sh smoke test) and the full creator chain, which the build stage can only
   check best-effort.
3. `gsplat.csrc` imports without triggering JIT — `find / -name csrc.so` inside the
   container, and no torch_extensions cache written on first import.
4. installed rev matches the lock:
   `site-packages/gsplat-*.dist-info/direct_url.json` -> `commit_id` vs
   `pyproject.toml` `rev`. Today: `d2f5c0f8eb12190469f92cb408cf033943432532`. OK.
5. `pytest tests/splats tests/mesh tests/wrapper/test_splats_stage.py tests/test_cu121_migration.py`.

## Owed alongside

- drift gate in CI or setup.sh: compare `direct_url.json` `commit_id` against the
  `uv.lock` rev and fail loudly. The 2026-06-02 divergence went unnoticed for months.
