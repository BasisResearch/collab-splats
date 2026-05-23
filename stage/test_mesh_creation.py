"""
Quick script to test mesh creation from existing preprocessed images.
This skips the COLMAP export/rescaling steps and goes straight to mesh creation.
"""

from pathlib import Path
from mapanything_utils import (
    load_mapanything_model,
    load_and_preprocess_images,
    run_mapanything_inference,
    create_mesh_from_mapanything,
)

def test_mesh_from_existing_images(
    image_dir,
    output_path,
    mesh_conf_percentile=60.0,
    mesh_filter_frames="all",
    mesh_as_mesh=True,
    verbose=True,
):
    """Test mesh creation from existing preprocessed images.

    Args:
        image_dir: Directory with preprocessed images (e.g., output/preproc/images/)
        output_path: Where to save the mesh GLB file
        mesh_conf_percentile: Confidence filtering threshold
        mesh_filter_frames: Frame filter ("all", "0:", etc.)
        mesh_as_mesh: True for mesh, False for point cloud
        verbose: Print progress

    Returns:
        Path to created mesh file
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    print(f"Loading images from: {image_dir}")

    # Step 1: Load model (fast, uses cache)
    model = load_mapanything_model(verbose=verbose)

    # Step 2: Load and preprocess images (fast)
    views, image_paths = load_and_preprocess_images(image_dir, verbose=verbose)

    # Step 3: Run inference (this is the main computation, but faster than full pipeline)
    outputs = run_mapanything_inference(model, views, verbose=verbose)

    # Step 4: Create mesh
    mesh_path = create_mesh_from_mapanything(
        outputs=outputs,
        views=views,
        output_path=output_path,
        filter_by_frames=mesh_filter_frames,
        as_mesh=mesh_as_mesh,
        mask_ambiguous=True,
        conf_percentile=mesh_conf_percentile,
        mask_black_bg=True,
        mask_white_bg=False,
        show_cam=True,
        verbose=verbose,
    )

    print(f"\n✓ Mesh created: {mesh_path}")
    return mesh_path


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_mesh_creation.py <image_dir> [output_mesh.glb]")
        print("\nExample:")
        print("  python test_mesh_creation.py output/preproc/images/ test_mesh.glb")
        sys.exit(1)

    image_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_mesh.glb"

    test_mesh_from_existing_images(
        image_dir=image_dir,
        output_path=output_path,
        mesh_conf_percentile=60,
        mesh_filter_frames="all",
        mesh_as_mesh=True,
    )
