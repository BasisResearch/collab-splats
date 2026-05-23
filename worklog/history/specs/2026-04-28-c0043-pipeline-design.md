# C0043 Full Pipeline Run — Design Spec

**Date:** 2026-04-28  
**Branch:** refactor/core-modules  

## Goal

Run the full collab-splats pipeline on the C0043 fieldwork video from scratch:
preprocess (hloc SfM) → train rade-features (maskclip + dinov2) → mesh (TSDF fusion).
Archive existing output before running so the previous state is preserved for comparison.

## Inputs

| Item | Path |
|------|------|
| Video | `/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4` |
| Dataset config | `docs/splats/configs/datasets/birds_date-02062024_video-C0043.yaml` |
| Base config | `docs/splats/configs/base.yaml` |
| Output (current) | `/workspace/fieldwork-data/birds/2024-02-06/environment/C0043/` |

## Config (resolved from YAML hierarchy)

- `method`: `rade-features`
- `frame_proportion`: 0.25
- `min_frames`: 100
- `preprocess.sfm_tool`: `hloc`
- `mesher_type`: `Open3DTSDFFusion` (base.yaml default)

## Script

**Location:** `examples/run_c0043_pipeline.py`

### Step 1 — Archive

Move existing output dir to a timestamped copy before touching anything:

```
shutil.move(
    src=".../environment/C0043",
    dst=".../environment/C0043_archived_YYYYMMDD_HHMMSS"
)
```

Entire directory moved atomically (same filesystem). Script aborts if src does not exist (nothing to archive — user must confirm intent).

### Step 2 — Load Splatter

```python
splatter = Splatter.from_config_file(
    dataset="birds_date-02062024_video-C0043",
    config_dir=Path(__file__).parent.parent / "docs/splats/configs",
)
```

Picks up all config from YAML hierarchy. No hardcoded paths in the script.

### Step 3 — Staged Pipeline

Each stage:
- Prints a banner with stage name
- Records wall-clock start time
- Calls the splatter method with `overwrite=True`
- Prints elapsed on success
- Catches exceptions, prints traceback + elapsed, exits with code 1

**Stage 1: Preprocess**
`splatter.preprocess(overwrite=True)`  
→ Calls `ns-process-data video --sfm-tool hloc ...`  
→ hloc pipeline: NetVLAD retrieval → SuperPoint features → SuperGlue matching → COLMAP reconstruction  
→ Outputs `preproc/transforms.json`

**Stage 2: Train**
`splatter.extract_features(overwrite=True)`  
→ Calls `ns-train rade-features ...`  
→ FeatureSplattingDataManager extracts maskclip + dinov2 features per frame  
→ rade-gs backbone trains depth/normal-enabled gaussians with feature distillation  
→ Outputs checkpoint under `rade-features/`

**Stage 3: Mesh**
`splatter.mesh(overwrite=True)`  
→ `Open3DTSDFFusion`: renders depth + RGB from trained model, fuses into TSDF, extracts mesh  
→ Outputs `rade-features/mesh/mesh.ply` (+ `mesh_features.pt` if features distilled to mesh)

## Error Handling

- Archive failure → abort (do not run pipeline on unarchived data)
- Stage failure → print exception + elapsed, `sys.exit(1)`
- No silent swallowing; full traceback always shown

## Success Criteria

1. `environment/C0043_archived_YYYYMMDD/` exists and contains original preproc
2. `environment/C0043/preproc/transforms.json` created by hloc SfM
3. `environment/C0043/rade-features/**/config.yml` exists (training complete)
4. `environment/C0043/rade-features/mesh/mesh.ply` exists

## Out of Scope

- PixSFM refinement (commented out in base.yaml, not enabled)
- Mesh querying / semantic segmentation (separate workflow)
- Hyperparameter tuning
