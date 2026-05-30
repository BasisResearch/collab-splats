# `third_party/`

External dependencies cloned and populated at install time by setup scripts.
Contents are gitignored — only this README is tracked.

## Populating

Run the install script for the pipeline you need:

```bash
bash setup_feedforward.sh   # salad, xfeat, vggt-x, mapanything
bash setup_hloc.sh          # hloc
bash setup.sh               # core env (bae installed via pyproject.toml git URL)
```

Scripts are idempotent: if the target already exists, re-clone is skipped.

## Policy

- Setup scripts own the clone + patch lifecycle.
- Local patches applied after clone must be encoded in the install script, not
  applied by hand. Future reclones need to reproduce them.
- Deps that need `sys.path` access use a runtime insert at import time (see
  `collab_splats/pointcloud/localization.py` for the xfeat pattern).
- New entries: add a row to the table below and wire up in the relevant setup script.

## Current entries

| Path | Upstream | Populated by | Used by |
|---|---|---|---|
| `salad/` | `serizba/salad` | `setup_feedforward.sh` | DINO-SALAD aggregator for loop closure (`collab_splats/semantics/retrieval.py`). `sys.path` insert. |
| `xfeat/` | `verlab/accelerated_features` | `setup_feedforward.sh` | Lightweight local feature extractor for camera localization (`collab_splats/pointcloud/localization.py`). `sys.path` insert to `modules/`. |
| `hloc/` | `cvg/Hierarchical-Localization` | `setup_hloc.sh` | Hierarchical localization pipeline. Installed editable via pip. |
