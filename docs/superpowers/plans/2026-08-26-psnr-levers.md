# PSNR Levers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-image affine appearance, decaying loss weights, and a configurable `preproc.search_radius`, per `docs/superpowers/specs/2026-08-26-psnr-levers-design.md`.

**Architecture:** Three independent, minimal changes. Loss decay is one pure function (`loss_weight`) that the existing schedule loop calls. Appearance is one `nn.Embedding(n, 6)` module applied to the rendered rgb before the background blend, with its regulariser riding the loss registry via `render["appearance"]`. Search radius is config plumbing only.

**Tech Stack:** torch, gsplat, pytest (`/opt/venv/reconstruction/bin/python -m pytest`).

---

### Task 1: `loss_weight` decay (losses.py + config validation)

**Files:** Modify `collab_splats/splats/losses.py`, `collab_splats/splats/trainer.py` (from_dict validation, distortion rescale). Test `tests/splats/test_losses.py`, `tests/splats/test_trainer.py`.

- [ ] Tests: `loss_weight` constant / before start / geometric midpoint / after end; `compute_losses` uses decayed weight; `from_dict` rejects `end` without `end_weight`, `end <= start`, non-positive endpoints.
- [ ] Implement:

```python
def loss_weight(step: int, spec: dict | None) -> float:
    if spec is None or step < spec.get("start", 0):
        return 0.0
    weight = spec["weight"]
    end = spec.get("end")
    if end is None:
        return weight
    if step >= end:
        return spec["end_weight"]
    fraction = (step - spec.get("start", 0)) / (end - spec.get("start", 0))
    return weight * (spec["end_weight"] / weight) ** fraction


def loss_active(step, spec):
    return loss_weight(step, spec) > 0
```

`compute_losses` uses `weight = loss_weight(step, spec)`. `from_dict` allows keys `{weight, start, end, end_weight}` and validates decay entries. Distortion `/scale` also divides `end_weight` when present.

- [ ] Run `pytest tests/splats/test_losses.py tests/splats/test_trainer.py -q`; commit `feat(splats): log-linear loss weight decay via {end, end_weight}`.

### Task 2: `AppearanceModule`

**Files:** Create `collab_splats/splats/appearance.py`, `tests/splats/test_appearance.py`. Modify `losses.py` (`appearance_reg`), `trainer.py` (config fields, `make_appearance_module`, train loop, `write_splat_outputs` call), `outputs.py` (apply in `render_all_views`, ckpt key), `configs/base.yaml`, `tests/splats/test_trainer.py` recorder signature.

- [ ] Tests: identity at init; only addressed row gets gradient; `appearance_reg` is mean square of `render["appearance"]`, None when absent; config round-trip `appearance_opt` / `appearance_lr`.
- [ ] Implement:

```python
class AppearanceModule(torch.nn.Module):
    def __init__(self, n_images: int):
        super().__init__()
        self.params = torch.nn.Embedding(n_images, 6)
        torch.nn.init.zeros_(self.params.weight)

    def forward(self, rgb: Tensor, camera_ids: Tensor) -> Tensor:
        params = self.params(camera_ids)
        gain = 1.0 + params[:, None, None, :3]
        bias = params[:, None, None, 3:]
        return rgb * gain + bias
```

Trainer: after `render_view`, `if appearance is not None: render["rgb"] = appearance(render["rgb"], camera_id); render["appearance"] = appearance.params(camera_id)`. `appearance_reg_loss` returns `params.square().mean()` or None. Own Adam at `cfg.appearance_lr` with `ExponentialLR(lr_gamma)`. `write_splat_outputs(cfg, gaussians, pose_refiner, appearance, ...)`, ckpt `"appearance"`, `render_all_views` applies it before clamp. base.yaml: `appearance_opt: false`, `appearance_lr: 1.0e-3`, losses `appearance_reg: {weight: 1.0e-3}` (also in `_default_losses`).

- [ ] Run `pytest tests/splats -q`; commit `feat(splats): per-image affine appearance model (appearance_opt)`.

### Task 3: `preproc.search_radius`

**Files:** Modify `configs/base.yaml`, `collab_splats/wrapper/reconstructor.py` (`extract_frames` param + `preprocess` forwards). Test `tests/preproc/test_sampling.py` (radius capped by spacing, wider window picks sharpest), `tests/wrapper/test_reconstructor_preprocess.py` (forwarded).

- [ ] Tests, implement, run `pytest tests/preproc/test_sampling.py tests/wrapper/test_reconstructor_preprocess.py -q`; commit `feat(preproc): expose search_radius (default 7)`.

### Task 4: Eval

- [ ] Overrides in scratchpad from `gopro_3dgs_c2f_overrides.yaml`: `+appearance`, `+decay`, `+radius7` (rebuild). Sequential tmux via `run_dense.py`. Record in spec §Measured.
