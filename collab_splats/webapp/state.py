from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionState:
    """Single-user server-side session state."""

    output_dir: Optional[Path] = None
    video_path: Optional[Path] = None
    creator: str = "vggtx"
    conf: float = 35.0
    extractor: str = "dinov2"
    localize_method: str = ""
    localize_extractor: str = "DISK+LightGlue"


# Module-level singleton — single-user local tool
_session = SessionState()


def get_session() -> SessionState:
    """Return the global session singleton."""
    return _session
