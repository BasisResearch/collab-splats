# `evals/` — ground-truth evaluation

Runs the production pipeline (`Reconstructor`) over sequences with known poses and scores the
result against ground truth. Reference-free quality is not measured here: each stage writes its
own `*_quality_report.json`, and the runner copies those into the cell's `eval_metrics.json`.

```
evals/
  eval.py         grid runner: datasets x conditions -> one Reconstructor run per cell
  gt_metrics.py   ATE / RPE (evo), pairwise AUC, GT depth error
  datasets.py     loaders: 7scenes, tum, co3dv2
  configs/        grid YAMLs
  results/        output (gitignored)
```

## Get data

Paths below match `configs/`; run from the repo root.

```bash
# 7-Scenes: scene zip holds one zip per sequence; unzip exit 1 is a harmless size warning
mkdir -p data/7scenes && cd data/7scenes
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip
unzip -q chess.zip && unzip -q chess/seq-01.zip -d chess     # -> chess/seq-01/frame-*.{color,depth}.png, *.pose.txt
cd -

# TUM RGB-D
mkdir -p data/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz -O data/tum/fr1_desk.tgz
tar -xzf data/tum/fr1_desk.tgz -C data/tum

# CO3Dv2: the downloader must run from inside its package dir
git clone https://github.com/facebookresearch/co3d /tmp/co3d
cd /tmp/co3d/co3d && python download_dataset.py --download_folder "$OLDPWD/data/co3dv2" --download_categories apple
```

## Grid config

```yaml
name: cross_model_chess
output_dir: evals/results/cross_model_chess
base:                        # merged into every cell, over configs/base.yaml
  semantics: {enabled: false}
  mesh: {enabled: false}
  pointcloud:
    loop_closure: {submap_size: 50, lc_retrieval_threshold: 0.0}   # windowed, no loop edges
datasets:
  - {name: chess, type: 7scenes, seq_dir: data/7scenes/chess/seq-01, max_frames: 500}
conditions:                  # label -> override merged over base
  omega:    {pointcloud: {backend: vggt_omega}}
  omega_lc: {pointcloud: {backend: vggt_omega, loop_closure: {lc_retrieval_threshold: 0.95}}}
```

A flag under test (sky masking, BA, a backend) is one more condition, not a script.
With `loop_closure` on, `bundle_adjustment` refines each window (decision 022); without it, a BA
condition runs single-pass and needs a `max_frames` the backend fits in one go.

## Run

```bash
/opt/venv/reconstruction/bin/python -m evals.eval --config evals/configs/cross_model_chess.yaml --dry_run
/opt/venv/reconstruction/bin/python -m evals.eval --config evals/configs/cross_model_chess.yaml   # in tmux
```

Each cell runs in its own subprocess; a cell holding `eval_metrics.json` is skipped on re-run.

## Output

Per cell, `<output_dir>/<dataset>__<condition>/`:

- `eval_metrics.json`: ATE, RPE, AUC@{5,15,30}, GT depth error (7-Scenes), stage reports, config,
  `registered_frames` / `n_frames`
- `trajectories.npz`: GT and predicted camera-to-world poses in GT order, `gt_idx`, per-frame ATE,
  Sim3-aligned predicted camera centers
- `plots/`: `trajectory.png`, `ate_per_frame.png`
- `input/` (frame symlinks), `run/` (the pipeline's own output)

Per grid: `comparison.md` and `comparison.json`, one row per finished cell.

Unregistered frames (an sfm backend may drop some):

- poses match GT by frame name; ATE, RPE and depth score the registered frames only
- AUC counts every pair touching an unregistered frame as a failure (COLMAP benchmark convention)
- `comparison.md` shows `registered` N/M beside every cell
