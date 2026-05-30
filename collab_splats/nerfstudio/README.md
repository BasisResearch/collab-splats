# collab_splats/nerfstudio

Extension and reimplementation layer for NerfStudio. All code in this submodule
subclasses or wraps NerfStudio internals. If you are adding a new model,
datamanager, or training config, it belongs here.

## Structure

- **models/** — `RadegsModel` and `RadegsFeaturesModel`, both subclassing
  `SplatfactoModel` from `nerfstudio.models.splatfacto`. `RadegsModel` adds
  depth-normal consistency loss. `RadegsFeaturesModel` extends it to splat
  ANN feature spaces.

- **datamanagers/** — `FeatureSplattingDataManager`, subclassing
  `FullImageDatamanager`. Extracts DINO/CLIP features from images and provides
  them alongside RGB during training.

- **method_configs/** — `MethodSpecification` objects registered with NerfStudio
  via `pyproject.toml` entry points. Each file assembles a full training pipeline
  (datamanager + model + optimizers + scheduler). These are the objects NerfStudio
  discovers when you run `ns-train rade-gs`.

- **trainer_config.py** — Patches `TrainerConfig` to allow `vis=None`, disabling
  the NerfStudio viewer when running programmatically via `Splatter`.

- **model_loading.py** — Thin wrapper around NerfStudio's `eval_setup` for loading
  trained checkpoints outside the training loop.

## Known Tech Debt

`get_outputs` is ~80% duplicated between `RadegsModel` and `RadegsFeaturesModel`.
Extracting a shared base is non-trivial (feature cropping and render output shape
differ). Defer to a dedicated PR with training validation.
