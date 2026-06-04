"""Typed run configuration for the splats dashboard pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

########
# Config
########


@dataclass
class RunConfig:
    """All knobs for one pipeline run; serialised to run_config.yaml for provenance."""

    # Frame sampling
    sampling_method: str = "balanced"  # "balanced" | "optical_flow"
    max_frames: int = 50
    min_disparity: float = 50.0  # optical_flow only

    # Environment (pointcloud) model
    env_model: str = "vggt_omega"  # "vggt_omega" | "vggtx" | "mapanything"
    conf_threshold: float = 50.0

    # Semantics
    semantic_extractor: str = "talk2dino"
    query: str = ""

    # Mesh (TSDF) params
    mesh_voxel_size: float = 0.01
    mesh_sdf_trunc: float = 0.04
    mesh_depth_trunc: float = 10.0
    mesh_clean_repair: bool = True

    # Provenance — filled by the pipeline, not the UI
    frame_indices: list[int] = field(default_factory=list)
    video_ref: str = ""

    def to_yaml(self, path: Path, video_ref: str = "") -> None:
        """Write config (incl. provenance) to a YAML file."""
        data = asdict(self)
        if video_ref:
            data["video_ref"] = video_ref
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        """Load config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
