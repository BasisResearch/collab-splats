# Mesh Feature Transfer — GPU-Accelerated Points → Mesh, with Query Parity

**Date:** 2026-06-04
**Status:** Design — approved, pending spec review
**Scope:** Wire per-point semantic features onto mesh vertices and make the mesh queryable exactly like the pointcloud, with a GPU-accelerated transfer kernel.

## Problem

The feedforward pipeline lifts 2D dense features onto the pointcloud (`lift_features` →
`FeedforwardResult.features`, shape `(P, D)`), and the dashboard queries those point
features (text → per-point similarity heatmap). The **mesh** has no feature channel:
`Open3DTSDFFusion` fuses depth + RGB only. So semantic query works on the pointcloud
but not the mesh.

A points → vertices transfer already exists — `features2vertex` and its wrapper
`transfer_features_to_mesh` (`collab_splats/mesh/utils.py`) — ported from the old
`features2vertex` local-clustering approach (KNN over a KDTree + Gaussian-weighted
aggregation + truncation mask). But:

1. It is **not wired into** the feedforward → mesh pipeline. Nothing populates a
   mesh-side feature array or persists it.
2. The kernel is **CPU-only**: single-threaded `cKDTree.query`, and a Python loop over
   `k` doing `np.add.at` (unbuffered scatter, float64) — the dominant cost on large
   clouds.
3. The viewer's query path scores points only; mesh mode has no equivalent.

## Goals

- GPU-accelerate `features2vertex` while preserving its signature, return contract, and
  numerics (KNN + Gaussian weighting + `sdf_trunc` truncation).
- Populate and persist per-vertex features in the feedforward → mesh pipeline.
- Make mesh mode in the dashboard queryable with the **same** `score_queries` call used
  for points — no new query code.

## Non-Goals (explicit scope guard)

- **No AE-as-color TSDF fusion** (compress to ≤3 dims, smuggle through the RGB channel).
  Evaluated and deferred — 8-bit quantization and 3-channel cap make it a preview path,
  not the primary one.
- **No native multi-channel feature-volume fusion** (custom TSDF accumulation in torch).
  Higher fidelity, far more code — only revisit if KNN transfer quality proves
  insufficient for downstream semantics.
- Single transfer backend. No pluggable backend abstraction.

## Key Insight: Query Parity Is Free

`features2vertex` produces each vertex feature as a **convex combination (weighted mean)
of the same per-point features**. The output `(M, D)` array therefore lives in the
**exact same feature space** as the `(P, D)` point features. Every existing query
primitive that operates on an `(N, D)` array applies verbatim to vertices:

- `BaseQueryableExtractor.score_queries(features, positive, negative)` → `(N,)` sims
  (dashboard text query, `viewer.py:266`)
- `features_to_rgb` (top-3 PCA → RGB display)
- `insid3` cosine-similarity segmentation

**The only contract to preserve:** the viewer scores `self._lifted_normed` — point
features **renormalized** before scoring (`ensure_lifted`). Vertex features must receive
the **same renormalization** before `score_queries`, or similarity magnitudes drift.

## Design

### 1. GPU-accelerated `features2vertex` (`collab_splats/mesh/utils.py`)

Same signature, same return (`(M, D)` `np.ndarray`, dtype matches input features), same
semantics (KNN over KDTree, Gaussian weights, `sdf_trunc` truncation mask, all-far →
zeros).

Hybrid kernel:

- **Spatial query stays on CPU KDTree.** Keep `cKDTree(vertices)` + `tree.query`, but
  pass `workers=-1` for multicore query. KDTree query is `O(N log M)`; a brute-force GPU
  `cdist` would be `O(N·M)` (e.g. `1e6 × 1e5 = 1e11` entries, must tile) — asymptotically
  worse except for tiny meshes. Keep the KDTree.
- **Aggregation moves to GPU.** After the query, move `distances`, `indices`, and the
  truncation-masked `features` to torch float32 on `device = "cuda" if available else
  "cpu"`. Compute the Gaussian weights vectorized, scatter with `index_add_` over the `k`
  neighbours (replacing the Python `np.add.at` loop), accumulate a per-vertex weight sum,
  normalize, single `.cpu().numpy()` at the end. Mirrors the established GPU-kernel
  pattern in `lift_features` (`pointcloud/utils.py:839`).
- **CPU fallback** when CUDA is unavailable — same torch code on the CPU device. Output
  must be `allclose` to the current numpy implementation within float32 tolerance.

`sigma = mean(distances)` is computed over the post-mask distances, as today, to keep the
Gaussian kernel identical.

`normals2vertex` and `transfer_features_to_mesh` are unchanged — they call the faster
`features2vertex` transparently.

### 2. Wire into the feedforward → mesh pipeline

- Add field to `MeshResult` (`collab_splats/mesh/base.py`):
  `vertex_features: np.ndarray | None = None`.
- In the feedforward → mesh orchestration (the step that builds the TSDF mesh after a
  `FeedforwardResult` with `features` populated): load the mesh, call
  `transfer_features_to_mesh(result, mesh)`, store the `(M, D)` array on
  `MeshResult.vertex_features`.
- **Persist** alongside the mesh as `vertex_features.npy` in the mesh `output_dir`, so
  the dashboard can load mesh features without re-running the lift. Index-aligned with
  `mesh.vertices` (Open3D vertex order is stable for a written/read `.ply`).

### 3. Query parity in the viewer (mesh mode)

`viewer.py:score_query` currently scores `self._lifted_normed` (points). In mesh mode:

- Load persisted `vertex_features.npy` (or transfer on demand from the already-lifted
  point features), apply the **same renormalization** `ensure_lifted` applies to point
  features, cache as `self._mesh_lifted_normed`.
- When `self.mode == "mesh"`, `score_query` scores the vertex features instead of the
  point features — same `extractor.score_queries(...)` call — then colors the **mesh**
  right pane (`apply_viridis` → per-vertex RGB on the mesh actor) rather than the point
  cloud.
- Pointcloud mode is unchanged.

## Data Flow

```
FeedforwardResult.features (P, D)            # lift_features, existing
        │
        ├── points (P, 3) ───┐
        │                    ▼
   mesh.vertices (M, 3) ── features2vertex (GPU) ──► vertex_features (M, D)
        │                                                    │
        │                                          persist vertex_features.npy
        ▼                                                    │
   mesh_tsdf.ply                                             ▼
                                          renormalize ──► score_queries ──► (M,) sims
                                                              ▼
                                                   apply_viridis → color mesh
```

## Testing

`tests/mesh/test_utils.py` (extend) and `tests/dashboard/` (query-parity smoke):

- **GPU/CPU parity:** new `features2vertex` output `allclose` to the current numpy
  reference on a fixed small fixture (random points/vertices/features, fixed seed),
  float32 tol. Run the torch kernel on the CPU device so the test is deterministic in CI
  without a GPU.
- **Truncation:** all points beyond `sdf_trunc` → all-zero `(M, D)` output.
- **Shape / dtype:** `(M, D)` out, dtype matches input features, `M == len(vertices)`.
- **Query parity:** `score_queries` on an `(M, D)` vertex array returns `(M,)` and is
  finite — confirms mesh features are a drop-in for the point query path.
- **Pipeline wiring:** `MeshResult.vertex_features` is populated and the `.npy` is
  written when the feedforward → mesh step runs with features present (mock the lift).

## Risks / Open Questions

- **Density mismatch holes.** TSDF voxel size vs point density can leave vertices with no
  point within `sdf_trunc` → zero features (renders as a hole in the query heatmap). The
  current code already truncates this way; this design preserves behaviour. If holes are
  objectionable downstream, a post-transfer Laplacian smoothing pass over mesh edges (the
  old approach's smoothing step) is the natural follow-up — **out of scope here**, noted
  for a later iteration.
- **Vertex-order stability** across `.ply` write/read must hold for the persisted
  `.npy` to stay aligned. Open3D preserves order; the pipeline-wiring test guards it.
```
