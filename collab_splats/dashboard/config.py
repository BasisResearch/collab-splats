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

    # Frame sampling — one density knob per method; max_frames is the frame budget
    # (the COUNT for "uniform", a ceiling for "fps"/"optical_flow").
    sampling_method: str = "fps"  # "fps" | "uniform" | "optical_flow"
    fps: float = 1.0  # fps only — samples/second
    max_frames: int = 100
    min_disparity: float = 50.0  # optical_flow only

    # Environment (pointcloud) model
    env_model: str = "vggt_omega"  # "vggt_omega" | "vggtx" | "mapanything"
    conf_threshold: float = 50.0  # vggt_omega native default; UI sets per-model (see _MODEL_CONF_DEFAULTS)

    # Semantics
    semantic_extractor: str = "talk2dino"
    query_positive: str = ""
    query_negative: str = "background, sky"

    # Mesh (TSDF) params
    mesh_voxel_size: float = 0.005
    mesh_sdf_trunc: float = 0.02
    mesh_depth_trunc: float = 1.0
    mesh_clean_repair: bool = False

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


########
# Localization
########


@dataclass
class LocalizationConfig:
    """Knobs for one localization run. UI/call state only — provenance for persisted
    localized frames lives in zarr attrs, so this is never written to run_config.yaml."""

    matcher: str = "loma"  # vismatch model name (LocalMatcher)
    top_k_viz: int = 3  # match-pair figures shown, best-first
    append_to_db: bool = True  # persist successful poses to localized/
    calibration_path: str | None = None  # per-camera K yaml override; None → proportions seed
    max_pairs: int = 200  # line cap per match-pair figure
