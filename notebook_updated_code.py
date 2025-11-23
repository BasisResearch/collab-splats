"""
Updated code for your Jupyter notebook
MeshLib 3.0.9.196 - Ready to copy and paste
"""

# =============================================================================
# OPTION 1: Simple replacement (just add progress bar)
# =============================================================================
from meshlib import mrmeshpy as mm
from tqdm import tqdm

points = mm.loadPoints(pcd_fn)

params = mm.PointsToMeshParameters()
params.voxelSize = points.computeBoundingBox().diagonal() * 1e-2
params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
params.minWeight = 1

# Add progress bar
pbar = tqdm(total=100, desc="Point cloud → Mesh")
params.progress = lambda p: (pbar.update(int(p*100) - pbar.n), True)[1]

mesh = mm.pointsToMeshFusion(points, params)
pbar.close()

print(f"✓ Generated mesh: {len(mesh.points.vec):,} vertices, {mesh.topology.faceSize():,} faces")


# =============================================================================
# OPTION 2: With GPU check and better logging
# =============================================================================
from meshlib import mrmeshpy as mm
from tqdm import tqdm

print("Loading point cloud...")
points = mm.loadPoints(pcd_fn)
num_points = points.validPoints.count()
print(f"  Loaded {num_points:,} points")

print("Configuring parameters...")
params = mm.PointsToMeshParameters()
bbox = points.computeBoundingBox()
params.voxelSize = bbox.diagonal() * 1e-2
params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
params.minWeight = 1

print(f"  Voxel size: {params.voxelSize:.6f}")
print(f"  Sigma: {params.sigma:.6f}")

# Check for CUDA support (for potential post-processing)
try:
    from meshlib import mrcudapy as mc
    print("  ✓ CUDA available for offset/distance operations")
except ImportError:
    print("  ℹ CUDA not available (no impact on point-to-mesh conversion)")

# Add progress bar
pbar = tqdm(total=100, desc="Converting to mesh")
params.progress = lambda p: (pbar.update(int(p*100) - pbar.n), True)[1]

print("Converting to mesh...")
mesh = mm.pointsToMeshFusion(points, params)
pbar.close()

num_verts = len(mesh.points.vec)
num_faces = mesh.topology.faceSize()
print(f"✓ Generated mesh: {num_verts:,} vertices, {num_faces:,} faces")


# =============================================================================
# OPTION 3: Most compact (one-liner callback)
# =============================================================================
from meshlib import mrmeshpy as mm
from tqdm import tqdm

points = mm.loadPoints(pcd_fn)
params = mm.PointsToMeshParameters()
params.voxelSize = points.computeBoundingBox().diagonal() * 1e-2
params.sigma = max(params.voxelSize, mm.findAvgPointsRadius(points, 50))
params.minWeight = 1

pbar = tqdm(total=100, desc="Converting")
params.progress = lambda p: (pbar.update(int(p*100) - pbar.n), True)[1]
mesh = mm.pointsToMeshFusion(points, params)
pbar.close()
