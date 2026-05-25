# TODO: Reorganize notebook helpers

`collab_splats/utils/notebook.py` was created as a temporary home for notebook display helpers.
Review placement once the set of helpers stabilizes.

## Options to evaluate

**Option A — keep in `utils/notebook.py`**
Pro: minimal churn, single import. Con: mixes display-only code with real utils package.

**Option B — `collab_splats/notebooks/` package**
Pro: clear separation of notebook-only code from core module. Con: another package to maintain.

**Option C — inline helpers inside each notebook**
Pro: no import needed. Con: defeats the purpose of abstracting them.

## Helpers currently in notebook.py

- `clean_and_extract_result` — O3D clean + conf stats + print
- `add_camera_frustums` — PyVista frustum loop
- `feature_viz_row` — PCA+heatmap+masked matplotlib row

## Related

See `worklog/notes/2026-05-23-notebook-abstraction-opportunities.md` for full audit.
