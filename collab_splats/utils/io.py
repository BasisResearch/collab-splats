"""
JSON report IO: one nan-safe, atomic writer for every report the pipeline leaves on disk.

- to_json_safe: numpy -> python, non-finite float -> None, tuples -> lists
- write_json: to_json_safe, then .json.tmp + os.replace, so no half file survives a crash
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

########################################################################
# JSON
########################################################################


def to_json_safe(obj: Any) -> Any:
    """
    Recursively convert a payload into values json.dumps writes as strict JSON.

    - dicts, lists and tuples are walked; tuples become lists, as json.dumps writes them
    - numpy arrays become lists, numpy scalars python scalars
    - non-finite floats (nan, inf) become None: json.dumps would write a bare NaN/Infinity
    - any other value passes through untouched

    Args:
        obj: nested dicts, lists, tuples, numpy values and scalars.

    Returns:
        The same structure built from plain python values.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """
    Write obj as strict JSON, atomically.

    - written to <name>.json.tmp, then os.replace onto path
    - a reader that trusts the file by its existence never sees a half-written one

    Args:
        path: destination JSON file; its parent must exist.
        obj: payload, converted with to_json_safe.
        indent: json.dumps indent.

    Returns:
        The written path.
    """
    path = Path(path)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(to_json_safe(obj), indent=indent))
    os.replace(tmp_path, path)
    return path
