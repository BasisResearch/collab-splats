# tools/patch_tutorial_configs.py
"""
Patch all tutorial notebooks to source tutorial_config.py.

Run from the repo root:
    /opt/conda/envs/nerfstudio/bin/python tools/patch_tutorial_configs.py
"""
import json
import re
from pathlib import Path

TUTORIALS_ROOT = Path("docs/source/tutorials")

NOTEBOOKS = [
    "01_preprocessing/keyframe_extraction.ipynb",
    "02_pointcloud/bundle_adjustment.ipynb",
    "02_pointcloud/colmap_sfm.ipynb",
    "02_pointcloud/feedforward_mesh.ipynb",
    "02_pointcloud/feedforward_methods.ipynb",
    "02_pointcloud/slam_loop_closure.ipynb",
    "03_splats/derive_splats.ipynb",
    "03_splats/visualization.ipynb",
    "04_semantics/feature_extraction.ipynb",
    "04_semantics/maskclip_vs_talk2dino.ipynb",
    "04_semantics/segmentation.ipynb",
    "05_lifting/semantic_lifting.ipynb",
    "06_mesh/create_mesh.ipynb",
    "07_localization/localization.ipynb",
    "evals/ground_truth_evals.ipynb",
]

# Lines to strip from config cells (matched as stripped prefixes)
STRIP_PREFIXES = (
    "DATASET",
    "CACHE_DIR",
    "IMAGES",
)

RUN_LINE = "%run ../tutorial_config.py"


def cell_source(cell: dict) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def set_cell_source(cell: dict, src: str) -> None:
    if isinstance(cell["source"], list):
        cell["source"] = src.splitlines(keepends=True)
    else:
        cell["source"] = src


def is_config_cell(src: str) -> bool:
    # Match any cell with a DATASET = assignment (covers notebooks that don't use get_cache_dir)
    return bool(re.search(r"^DATASET\s*=", src, re.MULTILINE))


def is_imports_cell(src: str) -> bool:
    return "get_cache_dir" in src and "import" in src


def patch_config_cell(src: str) -> str:
    lines = src.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        # Drop DATASET/CACHE_DIR/IMAGES assignments and blank lines they leave behind
        if any(stripped.startswith(p) for p in STRIP_PREFIXES):
            continue
        kept.append(line)
    # Remove leading/trailing blank lines
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    # Prepend %run line
    result_lines = [RUN_LINE] + ([""] if kept else []) + kept
    return "\n".join(result_lines)


def patch_imports_cell(src: str) -> str:
    """Remove 'from collab_splats.utils.paths import get_cache_dir' if present."""
    lines = src.splitlines()
    kept = [l for l in lines if "get_cache_dir" not in l]
    # Remove blank lines left at end
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def patch_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text())
    changed = False
    config_patched = False
    imports_patched = False

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)

        if not config_patched and is_config_cell(src):
            new_src = patch_config_cell(src)
            if new_src != src:
                set_cell_source(cell, new_src)
                changed = True
            config_patched = True
            continue

        if not imports_patched and is_imports_cell(src):
            new_src = patch_imports_cell(src)
            if new_src != src:
                set_cell_source(cell, new_src)
                changed = True
            imports_patched = True

    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"  patched: {path.relative_to(Path('.'))}")
    else:
        print(f"  skip (no change): {path.relative_to(Path('.'))}")
    return changed


def main():
    n_changed = 0
    for rel in NOTEBOOKS:
        path = TUTORIALS_ROOT / rel
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        if patch_notebook(path):
            n_changed += 1
    print(f"\nDone. {n_changed}/{len(NOTEBOOKS)} notebooks patched.")


if __name__ == "__main__":
    main()
