"""
COLMAP binary model IO: stem-named, atomically swapped writes.

- callers own the layout: the writer takes the model dir itself
- reading back is plain pycolmap.Reconstruction(str(model_dir))
"""

import shutil
from pathlib import Path

import pycolmap


def write_colmap_reconstruction(
    recon: pycolmap.Reconstruction, model_dir: Path
) -> None:
    """
    Write a binary model to exactly `model_dir`, image names reduced to stems, swapped in whole.

    - stems match the pipeline's frame ids (frame_NNNNNN), whatever extension the mapper saw
    - written to a hidden sibling, the old model moved aside, then the new one renamed in
    - model_dir is always a whole model or absent; a crash's leftover siblings are cleared next write

    Args:
        recon: model to write; its image names are renamed in place.
        model_dir: directory that holds cameras.bin / images.bin / points3D.bin afterwards.
    """
    # Image names -> stems
    for image in recon.images.values():
        image.name = Path(image.name).stem

    # Write a fresh hidden sibling; write_binary needs the dir to exist
    model_dir = Path(model_dir)
    tmp = model_dir.with_name(f".{model_dir.name}.tmp")
    old = model_dir.with_name(f".{model_dir.name}.old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    recon.write_binary(str(tmp))

    # Move the old model aside, rename the new one in, drop the old
    shutil.rmtree(old, ignore_errors=True)
    if model_dir.exists():
        model_dir.rename(old)
    tmp.rename(model_dir)
    shutil.rmtree(old, ignore_errors=True)
