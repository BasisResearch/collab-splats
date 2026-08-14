# Archived benchmark tooling

Everything in this directory is **evidence, not package code**. It was written to produce the
numbers in `docs/benchmarks/2026-08-14-loger-vs-omega-gopro.md`, and it is kept because a
report or a docstring elsewhere asserts a figure it produced. Nothing under `collab_splats/`
or `evals/` imports any of it, and nothing here runs in CI — `pyproject.toml` sets
`testpaths = ["./tests"]`, so even the two test modules below are skipped by a default
`pytest` run and must be named explicitly.

The only shipped deliverable from this work is the `loger` feedforward backend itself. The
GoPro reference machinery is deliberately *not* part of the package: it is single-scene,
single-camera measurement scaffolding, and generalising it was never in scope.

## The library (three modules, 19 tests)

| file | what it does |
| --- | --- |
| `gopro_telemetry.py` | GPMF telemetry → pose reference sampled at reconstructed frames |
| `rotation_alignment.py` | fit-free turn-angle invariant + camera-frame offset fit |
| `eval_gopro_reference.py` | writes `gt.tum` + `<method>.tum` for `evals/scripts/eval_compare.py` |
| `test_gopro_telemetry.py` | 10 tests — stream composition order, GPS gaps, frame alignment, the c2w transpose |
| `test_rotation_alignment.py` | 9 tests — invariance under camera offset and world change, offset recovery, the floor |

These reach into the package (`evals/trajectory_io.write_tum`,
`collab_splats.geometry.transforms.invert_poses`) rather than reimplementing it, and they
compute **no** trajectory metric — ATE/RPE/AUC come from the existing
`evals/scripts/eval_compare.py`, unmodified. Imports use an explicit `sys.path` insert
because this directory is not a package and is not on the path.

## The one-off diagnostics

A diagnostic earns a place here when it is the only record of *how* a claim was established.
Each of the four exists because a first answer was wrong.

| script | question it answered | where its conclusion lives now |
| --- | --- | --- |
| `diag_rotation.py` | Does the rotation disagreement live in the poses or in my fit? | `gopro_telemetry.py` module docstring (stabilisation is material); re-measured every run by `eval_gopro_reference.py --orientation auto` |
| `diag_handeye.py` | Why does the closed-form hand-eye fit miss the invariant floor — axis degeneracy, or the wrong transpose? | `rotation_alignment.fit_camera_offset` docstring (the rejected axis-Kabsch numbers) and `gopro_telemetry.reference_rotations_c2w` docstring (the transpose) |
| `decompose_residual.py` | Is the 0.272 % pinhole residual LoGeR's non-pinhole ray field, our K fit, or sharing one K across frames? | `docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md` |
| `compare_upstream_focal.py` | Does upstream LoGeR's own focal estimator beat ours on the same fixture? | same spec |

## Running them

Tests, from the repo root:

```bash
/opt/venv/reconstruction/bin/python -m pytest docs/benchmarks/scripts/ -p no:randomly
```

The reference driver, then the existing metrics runner:

```bash
python docs/benchmarks/scripts/eval_gopro_reference.py \
    --telemetry <telemetry.parquet> \
    --recon omega=<omega>/vggt_omega/colmap/sparse/0 \
    --recon loger=<loger>/loger/colmap/sparse/0 \
    --results-dir evals/results/gopro_GH010230
python evals/scripts/eval_compare.py --results-dir evals/results/gopro_GH010230
```

The two GoPro diagnostics take the telemetry parquet and a COLMAP model:

```bash
python docs/benchmarks/scripts/diag_rotation.py <telemetry.parquet> <colmap/sparse/0>
python docs/benchmarks/scripts/diag_handeye.py  <telemetry.parquet> <colmap/sparse/0>
```

The two LoGeR residual scripts run against the committed tutorial fixture and need a GPU:

```bash
PYTHONPATH=. python docs/benchmarks/scripts/decompose_residual.py
PYTHONPATH=. python docs/benchmarks/scripts/compare_upstream_focal.py
```

Both GoPro diagnostics were re-run on 2026-08-14 and reproduce the figures their docstrings
cite (axis singular values 11.6 / 8.12 / 7.04; transpose residuals 0.592 / 0.762 / 7.395 /
7.093 deg; turn angles 12.02 recon vs 12.29 `iori_cori` vs 15.00 raw CORI). The scene is
gitignored, so reproducing any of this needs `2026_07_15-Goprosplat-GH010230` pulled from
`environments-curated`.

One deliberate difference: `diag_handeye.py` keeps its original seven-seed local search, so
its absolute residuals are slightly pessimistic against `rotation_alignment.fit_camera_offset`,
which sweeps SO(3) first (0.592 → 0.584 deg on this scene). Its *ranking* of the four
transpose conventions — the question it was written to answer — is unaffected, and keeping the
original search is what makes it a faithful record.
