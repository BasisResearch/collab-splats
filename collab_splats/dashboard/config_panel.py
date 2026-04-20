"""ConfigPanel — param-based widget for per-video Splatter config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import param
import yaml

_METHODS = ["rade-features", "rade-gs", "splatfacto", "feature-splatting"]
_SFM_TOOLS = ["hloc", "colmap"]


class ConfigPanel(param.Parameterized):
    """Reactive config widget backed by YAML files."""

    method = param.Selector(default="rade-features", objects=_METHODS)
    sfm_tool = param.Selector(default="hloc", objects=_SFM_TOOLS)
    frame_proportion = param.Number(default=0.25, bounds=(0.01, 1.0), step=0.01)
    min_frames = param.Integer(default=100, bounds=(10, 1000))

    def __init__(self, configs_dir: Path | str | None = None, **params: Any):
        super().__init__(**params)
        if configs_dir is None:
            configs_dir = Path(__file__).parents[2] / "docs" / "splats" / "configs"
        self._configs_dir = Path(configs_dir)
        from collab_splats.wrapper.config import ConfigLoader
        self._loader = ConfigLoader(self._configs_dir)

    def load_from_yaml(self, yaml_path: Path | None) -> None:
        if yaml_path is not None and yaml_path.exists():
            dataset_name = yaml_path.stem
            try:
                config = self._loader.load(dataset_name)
            except ValueError:
                raw = self._load_raw_yaml(yaml_path)
                base = self._loader.base_config
                config = {**base, **raw}
                if "preprocess" in raw:
                    config["preprocess"] = {
                        **base.get("preprocess", {}), **raw["preprocess"]
                    }
        else:
            config = dict(self._loader.base_config)

        self.method = config.get("method", self.param.method.default)
        self.sfm_tool = config.get("preprocess", {}).get(
            "sfm_tool", self.param.sfm_tool.default
        )
        self.frame_proportion = float(
            config.get("frame_proportion", self.param.frame_proportion.default)
        )
        self.min_frames = int(
            config.get("min_frames", self.param.min_frames.default)
        )

    @staticmethod
    def _load_raw_yaml(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def save_to_yaml(self, yaml_path: Path) -> None:
        base = self._loader.base_config
        override: dict = {}

        if self.method != base.get("method"):
            override["method"] = self.method
        if abs(self.frame_proportion - float(base.get("frame_proportion", 0.25))) > 1e-9:
            override["frame_proportion"] = self.frame_proportion
        if self.min_frames != int(base.get("min_frames", 100)):
            override["min_frames"] = self.min_frames
        base_sfm = base.get("preprocess", {}).get("sfm_tool", "hloc")
        if self.sfm_tool != base_sfm:
            override.setdefault("preprocess", {})["sfm_tool"] = self.sfm_tool

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(override, f, default_flow_style=False)

    def to_splatter_config(
        self, file_path: Path | str, output_path: Path | str
    ) -> dict:
        return {
            "file_path": str(file_path),
            "method": self.method,
            "input_type": "video",
            "output_path": str(output_path),
            "frame_proportion": self.frame_proportion,
            "min_frames": self.min_frames,
        }

    def panel(self) -> "pn.Column":  # noqa: F821
        import panel as pn
        return pn.Column(
            pn.widgets.Select.from_param(self.param.method, name="Method"),
            pn.widgets.Select.from_param(self.param.sfm_tool, name="SfM Tool"),
            pn.widgets.FloatSlider.from_param(
                self.param.frame_proportion, name="Frame proportion"
            ),
            pn.widgets.IntSlider.from_param(self.param.min_frames, name="Min frames"),
        )
