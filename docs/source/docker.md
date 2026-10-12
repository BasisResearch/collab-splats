# Docker image

We build the image with the same `setup.sh`, with every CUDA extension compiled ahead of time. Build it from the collab-splats checkout, with `collab-data` cloned beside it (`../collab-data`):

```sh
docker build --platform=linux/amd64 --progress=plain --build-context collab-data=../collab-data --build-arg MAX_JOBS=4 -t collab-splats:release .
docker run --gpus all -it collab-splats:release bash
```

Push it to Docker Hub (keep the repo private: the image contains `collab-data`):

```sh
docker tag collab-splats:release tommybotch/collab-splats:release
docker push tommybotch/collab-splats:release
```

- **First build takes ~2.5 h**; later builds take minutes unless `pyproject.toml`, `uv.lock`, `setup.sh` or `collab-data` change.
- **`MAX_JOBS`**: Docker memory in GB / 8, at most 6.
- **GPUs**: Ampere/Ada (A100, A40, L40, RTX 30xx/40xx) and Hopper (H100). Not supported: Volta/Turing, Blackwell.

**Apple Silicon (Docker Desktop):**

- General: use the **Apple Virtualization framework** with **Rosetta** on (Docker VMM falls back to QEMU, which crashes the build).
- Resources: disk limit **≥ 250 GB**.
- Docker Engine: raise the build cache so the compiled layer is kept between builds:

  ```json
  "builder": { "gc": { "enabled": true, "defaultKeepStorage": "150GB" } }
  ```
- Never run `docker builder prune`, `docker system prune -a` or "Clean / Purge data": they delete the cached compile.
