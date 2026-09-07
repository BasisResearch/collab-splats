"""
Both model classes expose one interface, verified structurally rather than by inheritance.

- no shared base class on purpose: `Gaussians` and `Scaffold` have no implementation in common.
- what they do share is a call surface, and that is what this file pins.
"""

import ast
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.splats import rendering, trainer
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.scaffold import Scaffold
from collab_splats.splats.trainer import SplatsConfig

# gsplat's rasterization kernels are registered for CUDA only; every other test here is CPU-only
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")

# The interface, in one place: what `trainer.py` and `rendering.py` read off a model
# - `from_checkpoint` is absent: a classmethod, read off the class, pinned by the round-trip test
INTERFACE_ATTRIBUTES = ["params", "optimizers", "schedulers", "n_primitives", "primitive_unit"]
INTERFACE_METHODS = [
    "render",
    "pre_backward",
    "post_backward",
    "denormalize",
    "export_gaussians",
    "frame_report",
    "checkpoint",
]

# The implementations
# - no list here can guard its own width, so each has a derived-from-source width test below
# - the parametrize cases guard membership, never the set
MODEL_CLASSES = [Gaussians, Scaffold]
MODEL_CLASS_IDS = [model_class.__name__ for model_class in MODEL_CLASSES]


def _model(model_class, device="cpu"):
    """
    Build either model over the same 200-point seed cloud.

    - CPU by default; the render test passes `device="cuda"` — gsplat's kernels are CUDA-only.
    """
    return model_class(_config(model_class), *_seed_cloud(), scene_scale=1.0, n_views=4, device=device)


def _config(model_class):
    """
    The smallest config that selects the given representation.
    """
    overrides = {"representation": "scaffold", "scaffold": {}} if model_class is Scaffold else {}
    return SplatsConfig.from_dict({"max_steps": 100, **overrides})


def _seed_cloud():
    """
    A fixed 200-point cloud spanning the unit cube, with random colors.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (200, 3)).astype(np.uint8)
    return points, colors


def _members_read_off_the_model(path):
    """
    Every attribute the given module reads off a local named ``model``.
    """
    tree = ast.parse(Path(path).read_text())
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "model"
    }


def _classes_declaring(attribute):
    """
    Names of every class under `collab_splats/splats/` whose body assigns the given attribute.
    """
    found = set()
    for path in sorted(Path(trainer.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue

            # Both `x = ...` and the annotated `x: str = ...` form declare a class attribute
            for statement in node.body:
                targets = [statement.target] if isinstance(statement, ast.AnnAssign) else []
                targets += getattr(statement, "targets", [])
                if any(isinstance(target, ast.Name) and target.id == attribute for target in targets):
                    found.add(node.name)
    return found


########################################
# Width of the three lists this file parametrizes over
########################################


def test_the_declared_interface_is_exactly_what_its_consumers_read():
    consumed = _members_read_off_the_model(trainer.__file__) | _members_read_off_the_model(rendering.__file__)

    # Equality, not `consumed <= declared`: a subset passes vacuously if the consumers rename
    # their local away from `model`
    assert consumed == set(INTERFACE_ATTRIBUTES) | set(INTERFACE_METHODS)


def test_every_model_class_in_the_package_is_parametrized_over():
    # `primitive_unit` is the sanctioned discriminator, so declaring it is what makes a model
    assert _classes_declaring("primitive_unit") == set(MODEL_CLASS_IDS)


########################################
# Membership, one case per (member, class)
########################################


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
@pytest.mark.parametrize("attribute", INTERFACE_ATTRIBUTES)
def test_model_exposes_interface_attribute(model_class, attribute):
    model = _model(model_class)

    assert hasattr(model, attribute), f"{model_class.__name__} is missing {attribute}"


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
@pytest.mark.parametrize("method", INTERFACE_METHODS)
def test_model_exposes_interface_method(model_class, method):
    model = _model(model_class)

    assert callable(getattr(model, method, None)), f"{model_class.__name__}.{method} is not callable"


########################################
# The shapes the trainer and the writer assume of those members
########################################


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
def test_model_optimizers_and_schedulers_are_flat_lists(model_class):
    model = _model(model_class)

    # The trainer does `model.optimizers + refine.optimizers`, so these must be lists, not dicts
    assert isinstance(model.optimizers, list)
    assert isinstance(model.schedulers, list)
    assert all(hasattr(optimizer, "step") for optimizer in model.optimizers)
    assert all(hasattr(scheduler, "step") for scheduler in model.schedulers)


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
def test_model_n_primitives_is_an_int(model_class):
    model = _model(model_class)

    assert isinstance(model.n_primitives, int)
    assert model.n_primitives > 0


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
def test_model_primitive_unit_names_what_this_representation_counts(model_class):
    # The writer's log line and the quality report both read this instead of type-checking
    expected = "anchors" if model_class is Scaffold else "gaussians"

    assert _model(model_class).primitive_unit == expected


@cuda
@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
def test_model_render_returns_a_render_dict_and_an_info_dict(model_class):
    model = _model(model_class, device="cuda")
    cam_to_world = torch.eye(4)[None].cuda()
    cam_to_world[:, 2, 3] = -4.0
    intrinsics = torch.tensor([[[40.0, 0.0, 20.0], [0.0, 40.0, 12.0], [0.0, 0.0, 1.0]]]).cuda()

    # render(..., width, height, ...) — width FIRST; 40x24 rather than a square is what fails on
    # a swap, and rgb comes back (1, H, W, 3)
    render, info = model.render(cam_to_world, intrinsics, 40, 24, torch.tensor([0]).cuda(), step=0)

    assert set(render) >= {"rgb", "depth", "alpha"}
    assert render["rgb"].shape == (1, 24, 40, 3)
    assert isinstance(info, dict)


@pytest.mark.parametrize("model_class", MODEL_CLASSES, ids=MODEL_CLASS_IDS)
def test_model_checkpoint_round_trips(model_class):
    model = _model(model_class)

    # `rendering.write_outputs` stores `asdict(cfg)` under `config`; from_checkpoint reads it back
    ckpt = model.checkpoint()
    ckpt["config"] = asdict(_config(model_class))
    restored = model_class.from_checkpoint(ckpt, "cpu")

    assert restored.n_primitives == model.n_primitives
    assert set(restored.params) == set(model.params)
