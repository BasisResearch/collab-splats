"""Filesystem scanner for fieldwork video data.

Expected layout:
    {base_dir}/{species}/{date}/SplatsSD/*.MP4

where date is YYYY-MM-DD.

YAML dataset config names use MMDDYYYY format:
    {species}_date-{MMDDYYYY}_video-{stem}
"""

from __future__ import annotations

from pathlib import Path


def discover_videos(base_dir: Path | str) -> dict[str, dict[str, list[Path]]]:
    """Scan base_dir for videos matching the fieldwork layout.

    Args:
        base_dir: Root directory (e.g. /workspace/fieldwork-data).

    Returns:
        Nested dict: {species: {date: [video_paths]}}.
        Empty dict if base_dir doesn't exist or contains no videos.
    """
    base = Path(base_dir)
    if not base.exists():
        return {}

    result: dict[str, dict[str, list[Path]]] = {}

    for species_dir in sorted(base.iterdir()):
        if not species_dir.is_dir():
            continue
        for date_dir in sorted(species_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            splats_dir = date_dir / "SplatsSD"
            if not splats_dir.exists():
                continue
            videos = sorted(
                list(splats_dir.glob("*.MP4")) + list(splats_dir.glob("*.mp4"))
            )
            if not videos:
                continue
            result.setdefault(species_dir.name, {})[date_dir.name] = videos

    return result


def yaml_name_for_video(species: str, date_dir: str, video_stem: str) -> str:
    """Build the dataset YAML name for a video.

    Converts YYYY-MM-DD directory date to MMDDYYYY as used in YAML filenames.

    Args:
        species: e.g. "birds"
        date_dir: Directory name, e.g. "2024-02-06"
        video_stem: Video filename without extension, e.g. "C0043"

    Returns:
        e.g. "birds_date-02062024_video-C0043"
    """
    parts = date_dir.split("-")
    if len(parts) == 3:
        yyyy, mm, dd = parts
        date_str = f"{mm}{dd}{yyyy}"
    else:
        date_str = date_dir.replace("-", "")
    return f"{species}_date-{date_str}_video-{video_stem}"
