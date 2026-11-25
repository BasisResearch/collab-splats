# Docker image

We build the image with the same `setup.sh`, with every CUDA extension compiled ahead of time. Build it from the collab-splats checkout:

```sh
docker build --platform=linux/amd64 --progress=plain --build-arg MAX_JOBS=4 -t collab-splats:release .
docker run --gpus all -it collab-splats:release bash
```

Push it to Docker Hub. The image has no collab-data; people with access add it with `bash setup.sh` in the container:

```sh
docker tag collab-splats:release tommybotch/collab-splats:release
docker push tommybotch/collab-splats:release
```

- **First build takes ~2.5 h**; later builds take minutes unless `pyproject.toml`, `uv.lock`, `LICENSE.md`, `.python-version`, `setup.sh` or `setup/hloc.sh` change.
- **`MAX_JOBS`**: Docker memory in GB / 8, at most 6.
- **GPUs**: Ampere/Ada (A100, A40, L40, RTX 30xx/40xx) and Hopper (H100). Not supported: Volta/Turing, Blackwell.

**Apple Silicon (Docker Desktop):**

- General: use the **Apple Virtualization framework** with **Rosetta** on (Docker VMM falls back to QEMU, which is far slower).
- Resources: disk limit **≥ 250 GB**; raise the memory limit, then set `MAX_JOBS` from it.
- Docker Engine: raise the build cache so the compiled layer is kept between builds:

  ```json
  "builder": { "gc": { "enabled": true, "defaultKeepStorage": "150GB" } }
  ```
- Never run `docker builder prune`, `docker system prune -a` or "Clean / Purge data": they delete the cached compile.
