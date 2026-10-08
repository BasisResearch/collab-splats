import os
import sys

from sphinx.ext.autodoc.mock import _MockObject

sys.path.insert(0, os.path.abspath("../.."))

project = "collab-splats"
copyright = "2026, Basis Research Institute"
author = "Basis Research Institute"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
follow_links = True

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/BasisResearch/collab-splats",
    "navbar_align": "left",
    "show_toc_level": 2,
    "icon_links": [],
}
html_static_path = ["_static"]
html_title = "collab-splats"

# Never re-execute notebooks — use committed outputs
nbsphinx_execute = "never"

# Mock all heavy dependencies so autodoc works in CI without CUDA
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "gsplat",
    "open3d",
    "meshoptimizer",
    "nvdiffrast",
    "mobile_sam",
    "cv2",
    "einops",
    "roma",
    "trimesh",
    "sklearn",
    "skimage",
    "kornia",
    "scipy",
    "PIL",
    "tqdm",
    "rerun",
    "salad",
    "pytorch_metric_learning",
    "vggt",
    "bae",
    "pypose",
    "mapanything",
    "pycolmap",
    "hloc",
    "dataclasses_json",
    "tyro",
    "rich",
    "panel",
    "param",
    "pyvista",
    "gtsam",
    "huggingface_hub",
    "mergedeep",
    "transformers",
    "safetensors",
    "vismatch",
    "meshlib",
    "vggt_omega",
    "nltk",
    "segmentation_models_pytorch",
]

# Mocked types must support `X | None` annotations (PEP 604), which the stock mock rejects
_MockObject.__or__ = lambda self, other: self
_MockObject.__ror__ = lambda self, other: self

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Attributes: sections render as ivars, not duplicate attribute entries next to autodoc members
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
