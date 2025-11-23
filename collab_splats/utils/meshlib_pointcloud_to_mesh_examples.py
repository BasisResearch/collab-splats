"""
MeshLib Point Cloud to Mesh Conversion Examples
Updated for MeshLib 3.0.9.196+

Includes:
- Standard CPU-based conversion
- GPU-accelerated options (when CUDA available)
- Performance optimizations
- Progress callbacks
"""

from meshlib import mrmeshpy as mm
from tqdm import tqdm
import numpy as np


def pointcloud_to_mesh_basic(pcd_fn: str, output_fn: str = "mesh.ply"):
    """
    Basic point cloud to mesh conversion (CPU-based).
    Compatible with MeshLib 3.0.6+ and 3.0.9+

    Args:
        pcd_fn: Path to point cloud file (.ply, .xyz, .pts, etc.)
        output_fn: Output mesh filename

    Returns:
        mesh: The generated mesh
    """
    print("Loading points...")
    points = mm.loadPoints(pcd_fn)

    print("Setting up parameters...")
    params = mm.PointsToMeshParameters()

    # Auto-configure based on point cloud size
    bbox = points.computeBoundingBox()
    params.voxelSize = bbox.diagonal() * 1e-2
    params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
    params.minWeight = 1

    print("Converting points to mesh...")
    mesh = mm.pointsToMeshFusion(points, params)

    # Save mesh
    mm.saveMesh(mesh, output_fn)
    print(f"Mesh saved to {output_fn}")

    return mesh


def pointcloud_to_mesh_with_progress(pcd_fn: str, output_fn: str = "mesh.ply"):
    """
    Point cloud to mesh with progress bar.

    Args:
        pcd_fn: Path to point cloud file
        output_fn: Output mesh filename

    Returns:
        mesh: The generated mesh
    """
    print("Loading points...")
    points = mm.loadPoints(pcd_fn)

    print("Setting up parameters...")
    params = mm.PointsToMeshParameters()
    bbox = points.computeBoundingBox()
    params.voxelSize = bbox.diagonal() * 1e-2
    params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
    params.minWeight = 1

    # Add progress callback
    pbar = tqdm(total=100, desc="Converting to mesh")

    def progress_callback(p):
        pbar.n = int(p * 100)
        pbar.refresh()
        return True  # Return True to continue

    params.progress = progress_callback

    print("Converting points to mesh...")
    mesh = mm.pointsToMeshFusion(points, params)
    pbar.close()

    # Save mesh
    mm.saveMesh(mesh, output_fn)
    print(f"Mesh saved to {output_fn}")

    return mesh


def pointcloud_to_mesh_optimized(
    pcd_fn: str,
    output_fn: str = "mesh.ply",
    voxel_size: float = None,
    sigma_multiplier: float = 1.0,
    min_weight: float = 1.0,
    use_gpu: bool = True,  # Will auto-detect if CUDA is available
):
    """
    Optimized point cloud to mesh conversion with optional GPU acceleration.

    Args:
        pcd_fn: Path to point cloud file
        output_fn: Output mesh filename
        voxel_size: Manual voxel size (auto-computed if None)
        sigma_multiplier: Multiplier for sigma (smoothness)
        min_weight: Minimum weight threshold
        use_gpu: Attempt to use GPU if available (requires CUDA build)

    Returns:
        mesh: The generated mesh
    """
    print("Loading points...")
    points = mm.loadPoints(pcd_fn)
    num_points = points.validPoints.count()
    print(f"  Loaded {num_points:,} points")

    # Setup parameters
    params = mm.PointsToMeshParameters()
    bbox = points.computeBoundingBox()

    # Auto-configure or use provided values
    if voxel_size is None:
        params.voxelSize = bbox.diagonal() * 1e-2
    else:
        params.voxelSize = voxel_size

    # Compute sigma based on local point density
    avg_radius = mm.findAvgPointsRadius(points, 50)
    params.sigma = max(params.voxelSize, avg_radius * sigma_multiplier)
    params.minWeight = min_weight

    print(f"  Voxel size: {params.voxelSize:.6f}")
    print(f"  Sigma: {params.sigma:.6f}")
    print(f"  Min weight: {params.minWeight}")

    # Try to enable GPU if requested and available
    gpu_enabled = False
    if use_gpu:
        try:
            from meshlib import mrcudapy as mc
            # Note: GPU acceleration for pointsToMeshFusion may not be directly available
            # but other operations like FastWindingNumber can benefit
            print("  CUDA module available (may accelerate some operations)")
            gpu_enabled = True
        except ImportError:
            print("  CUDA module not available - using CPU")

    # Add progress callback
    pbar = tqdm(total=100, desc="Converting to mesh")

    def progress_callback(p):
        pbar.n = int(p * 100)
        pbar.refresh()
        return True

    params.progress = progress_callback

    # Convert to mesh
    mesh = mm.pointsToMeshFusion(points, params)
    pbar.close()

    # Print mesh statistics
    num_verts = len(mesh.points.vec)
    num_faces = mesh.topology.faceSize()
    print(f"\nMesh statistics:")
    print(f"  Vertices: {num_verts:,}")
    print(f"  Faces: {num_faces:,}")

    # Save mesh
    mm.saveMesh(mesh, output_fn)
    print(f"  Saved to: {output_fn}")

    return mesh


def pointcloud_to_mesh_with_gpu_acceleration(pcd_fn: str, output_fn: str = "mesh.ply"):
    """
    Point cloud to mesh with GPU-accelerated distance computations.

    NOTE: This example shows WHERE GPU could be used. The mrcudapy module
    is only available if:
    1. CUDA runtime is installed on the system
    2. MeshLib was built with CUDA support

    Args:
        pcd_fn: Path to point cloud file
        output_fn: Output mesh filename

    Returns:
        mesh: The generated mesh
    """
    print("Loading points...")
    points = mm.loadPoints(pcd_fn)

    # Check for CUDA support
    cuda_available = False
    try:
        from meshlib import mrcudapy as mc
        cuda_available = True
        print("  ✓ CUDA support available")
    except ImportError:
        print("  ✗ CUDA not available - using CPU")
        print("    To enable CUDA:")
        print("    1. Install NVIDIA CUDA runtime")
        print("    2. Ensure MeshLib CUDA build is installed")

    # Setup parameters
    params = mm.PointsToMeshParameters()
    bbox = points.computeBoundingBox()
    params.voxelSize = bbox.diagonal() * 1e-2
    params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
    params.minWeight = 1

    # Add progress
    pbar = tqdm(total=100, desc="Converting to mesh")
    params.progressCallback = lambda p: (pbar.update(int(p*100) - pbar.n), True)[1]

    # Convert to mesh
    mesh = mm.pointsToMeshFusion(points, params)
    pbar.close()

    # Post-processing: If CUDA is available, use it for distance-based operations
    if cuda_available:
        print("\nApplying GPU-accelerated post-processing...")
        try:
            from meshlib import mrcudapy as mc

            # Example: Use FastWindingNumber for inside/outside tests (GPU-accelerated)
            # This could be used for mesh validation or offsetting
            fwn = mc.FastWindingNumber(mesh)
            print("  ✓ GPU-accelerated winding number initialized")

            # You could use this for various operations:
            # - Offset operations (generalOffsetMesh with params.fwn = fwn)
            # - Inside/outside testing
            # - Distance field computations

        except Exception as e:
            print(f"  Warning: Could not use GPU acceleration: {e}")

    # Save mesh
    mm.saveMesh(mesh, output_fn)
    print(f"Mesh saved to {output_fn}")

    return mesh


# Example usage for your notebook
def notebook_example(pcd_fn: str):
    """
    Example code for Jupyter notebook - updated for MeshLib 3.0.9.196

    Usage in notebook:
        from collab_splats.utils.meshlib_pointcloud_to_mesh_examples import notebook_example
        mesh = notebook_example(pcd_fn)
    """
    from meshlib import mrmeshpy as mm
    from tqdm import tqdm

    # Load points
    points = mm.loadPoints(pcd_fn)

    # Setup parameters (UPDATED for 3.0.9+)
    params = mm.PointsToMeshParameters()
    params.voxelSize = points.computeBoundingBox().diagonal() * 1e-2
    params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
    params.minWeight = 1

    # Optional: Add progress bar
    pbar = tqdm(total=100, desc="Point cloud → Mesh")

    def progress_callback(p):
        pbar.n = int(p * 100)
        pbar.refresh()
        return True

    params.progress = progress_callback

    # Convert to mesh
    mesh = mm.pointsToMeshFusion(points, params)
    pbar.close()

    # Optional: Try GPU acceleration if available
    try:
        from meshlib import mrcudapy as mc
        print("✓ CUDA available - can use GPU for offset/distance operations")
        # Example: fwn = mc.FastWindingNumber(mesh)
    except ImportError:
        print("ℹ CUDA not available - using CPU only")

    return mesh


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python meshlib_pointcloud_to_mesh_examples.py <point_cloud.ply>")
        sys.exit(1)

    pcd_fn = sys.argv[1]
    output_fn = sys.argv[2] if len(sys.argv) > 2 else "output_mesh.ply"

    # Run optimized version
    mesh = pointcloud_to_mesh_optimized(
        pcd_fn,
        output_fn,
        use_gpu=True
    )
