"""Mesh and query C0043_talk2dino — reuses the existing trained talk2dino splat.

Stages:
  1. mesh      — TSDF fusion on the trained C0043_talk2dino run
  2. features  — map per-Gaussian latents to mesh vertices (mesh_features.pt)
  3. query     — text similarity: "feeder" vs "ground / leaves / rocks"
"""
import sys
import time
import traceback
from pathlib import Path

FIELDWORK_ROOT = Path("/workspace/fieldwork-data/birds/2024-02-06")
CONFIG_DIR = Path(__file__).parent.parent / "docs/splats/configs"
DATASET = "birds_date-02062024_video-C0043"
OUTPUT_DIR = FIELDWORK_ROOT / "environment" / "C0043_talk2dino"


def run_stage(name: str, fn, **kwargs):
    print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
    t0 = time.time()
    try:
        result = fn(**kwargs)
        print(f"\n[OK] {name} completed in {time.time() - t0:.1f}s")
        return result
    except Exception:
        traceback.print_exc()
        print(f"\n[FAIL] {name} failed after {time.time() - t0:.1f}s")
        sys.exit(1)


def main() -> None:
    from collab_splats.wrapper.splatter import Splatter

    splatter = Splatter.from_config_file(dataset=DATASET, config_dir=CONFIG_DIR)

    splatter.config["preproc_data_path"] = FIELDWORK_ROOT / "environment" / "C0043" / "preproc"
    splatter.config["output_path"] = OUTPUT_DIR

    mesher_kwargs = {
        "depth_name": "median_depth",
        "depth_trunc": 1.0,
        "voxel_size": 0.005,
        "normals_name": "normals",
        "features_name": "distill_features",
        "sdf_trunc": 0.03,
        "clean_repair": True,
        "align_floor": True,
    }

    run_stage(
        "Mesh (Open3DTSDFFusion)",
        splatter.mesh,
        mesher_type="Open3DTSDFFusion",
        mesher_kwargs=mesher_kwargs,
        overwrite=False,
    )

    # Query: feeder vs background — same as create_mesh notebook
    similarity_feeder = run_stage(
        "Query: feeder",
        splatter.query_mesh,
        positive_queries=["feeder"],
        negative_queries=["ground", "leaves", "rocks"],
        output_fn="query-feeder.ply",
    )

    # Query: tree
    similarity_tree = run_stage(
        "Query: tree",
        splatter.query_mesh,
        positive_queries=["tree"],
        negative_queries=["ground", "feeder", "rocks"],
        output_fn="query-tree.ply",
    )

    import numpy as np
    mesh_dir = splatter.config["mesh_info"]["mesh"].parent
    print("\n=== COMPLETE ===")
    print(f"Mesh     : {splatter.config['mesh_info']['mesh']}")
    print(f"Features : {splatter.config['mesh_info'].get('features')}")
    for name, sim in [("feeder", similarity_feeder), ("tree", similarity_tree)]:
        if sim is not None:
            print(f"Query [{name}]: range [{sim.min():.4f}, {sim.max():.4f}]  → {mesh_dir / f'query-{name}.ply'}")


if __name__ == "__main__":
    main()
