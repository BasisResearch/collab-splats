#!/usr/bin/env python3
"""
Test script to verify MeshLib compatibility and identify breaking changes.

Usage:
    python test_meshlib_versions.py <path_to_mesh.ply>

Or test with generated mesh:
    python test_meshlib_versions.py
"""

import sys
import meshlib.mrmeshpy as mm
import tempfile
import os


def print_version_info():
    """Print MeshLib version information."""
    print("=" * 70)
    print("MeshLib Version Information")
    print("=" * 70)
    try:
        # Try to get version
        version = mm.__version__ if hasattr(mm, '__version__') else "Unknown"
        print(f"Version: {version}")
    except:
        print("Version: Could not determine")

    # Check for key features
    features = {
        "fillHole": hasattr(mm, 'fillHole'),
        "fillHoleNicely": hasattr(mm, 'fillHoleNicely'),
        "fillHoleTrivially": hasattr(mm, 'fillHoleTrivially'),
        "getAllComponents": hasattr(mm, 'getAllComponents'),
        "subdivideMesh": hasattr(mm, 'subdivideMesh'),
        "getUniversalMetric": hasattr(mm, 'getUniversalMetric'),
        "FillHoleNicelySettings": hasattr(mm, 'FillHoleNicelySettings'),
    }

    print("\nAvailable Features:")
    for feature, available in features.items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}")
    print()


def create_test_mesh():
    """Create a simple test mesh with a hole."""
    print("Creating test mesh...")
    mesh = mm.makeTorus(2.0, 1.0, 20, 20)

    # Delete some faces to create a hole
    faces_to_delete = mm.FaceBitSet()
    for i in range(5):
        faces_to_delete.set(mm.FaceId(i), True)

    mm.deleteFaces(mesh, faces_to_delete)

    # Save to temp file
    temp_file = tempfile.mktemp(suffix=".stl")
    mm.saveMesh(mesh, temp_file)
    print(f"  Created test mesh: {temp_file}")
    return temp_file


def test_basic_operations(mesh_path):
    """Test basic mesh operations."""
    print("\n" + "=" * 70)
    print("Test 1: Basic Mesh Operations")
    print("=" * 70)

    try:
        mesh = mm.loadMesh(mesh_path)
        print(f"✓ Load mesh: {len(mesh.points.vec)} vertices, {mesh.topology.faceSize()} faces")
    except Exception as e:
        print(f"✗ Load mesh failed: {e}")
        return False

    try:
        components = mm.getAllComponents(mesh)
        print(f"✓ getAllComponents: {len(components)} components")
    except Exception as e:
        print(f"✗ getAllComponents failed: {e}")
        return False

    try:
        hole_ids = mesh.topology.findHoleRepresentiveEdges()
        print(f"✓ findHoleRepresentiveEdges: {len(hole_ids)} holes")
    except Exception as e:
        print(f"✗ findHoleRepresentiveEdges failed: {e}")
        return False

    if len(hole_ids) > 0:
        try:
            perimeter = mesh.holePerimiter(hole_ids[0])
            print(f"✓ holePerimiter: {perimeter:.4f}")
        except Exception as e:
            print(f"✗ holePerimiter failed: {e}")
            return False

    return True


def test_component_operations(mesh_path):
    """Test component manipulation."""
    print("\n" + "=" * 70)
    print("Test 2: Component Operations")
    print("=" * 70)

    try:
        mesh = mm.loadMesh(mesh_path)
        components = mm.getAllComponents(mesh)

        if len(components) == 0:
            print("✓ No components to test (single component mesh)")
            return True

        sizes = [mask.count() for mask in components]
        largest_idx = max(range(len(sizes)), key=lambda i: sizes[i])

        combined = mm.Mesh()
        combined.addPartByMask(mesh, components[largest_idx])

        print(f"✓ addPartByMask: Created mesh with {len(combined.points.vec)} vertices")
        return True
    except Exception as e:
        print(f"✗ Component operations failed: {e}")
        return False


def test_hole_filling_original(mesh_path):
    """Test hole filling with original method."""
    print("\n" + "=" * 70)
    print("Test 3: Hole Filling (Original Method)")
    print("=" * 70)

    try:
        mesh = mm.loadMesh(mesh_path)
        hole_ids = mesh.topology.findHoleRepresentiveEdges()

        if len(hole_ids) == 0:
            print("✓ No holes to fill")
            return True

        he = hole_ids[0]
        fill_params = mm.FillHoleParams()
        new_faces = mm.FaceBitSet()
        fill_params.outNewFaces = new_faces

        mm.fillHole(mesh, he, fill_params)
        print(f"✓ fillHole: Created {new_faces.count()} new faces")
        return True
    except Exception as e:
        print(f"✗ fillHole failed: {e}")
        print("  This might require the metric parameter in newer versions")
        return False


def test_hole_filling_with_metric(mesh_path):
    """Test hole filling with metric parameter."""
    print("\n" + "=" * 70)
    print("Test 4: Hole Filling (With Metric)")
    print("=" * 70)

    try:
        mesh = mm.loadMesh(mesh_path)
        hole_ids = mesh.topology.findHoleRepresentiveEdges()

        if len(hole_ids) == 0:
            print("✓ No holes to fill")
            return True

        he = hole_ids[0]
        fill_params = mm.FillHoleParams()

        # Try to set metric
        try:
            fill_params.metric = mm.getUniversalMetric(mesh)
            print("  Using metric parameter")
        except:
            print("  Metric parameter not available/required")

        new_faces = mm.FaceBitSet()
        fill_params.outNewFaces = new_faces

        mm.fillHole(mesh, he, fill_params)
        print(f"✓ fillHole with metric: Created {new_faces.count()} new faces")
        return True
    except Exception as e:
        print(f"✗ fillHole with metric failed: {e}")
        return False


def test_hole_filling_nicely(mesh_path):
    """Test hole filling with fillHoleNicely."""
    print("\n" + "=" * 70)
    print("Test 5: Hole Filling (fillHoleNicely)")
    print("=" * 70)

    try:
        mesh = mm.loadMesh(mesh_path)
        hole_ids = mesh.topology.findHoleRepresentiveEdges()

        if len(hole_ids) == 0:
            print("✓ No holes to fill")
            return True

        he = hole_ids[0]
        settings = mm.FillHoleNicelySettings()

        # Try to set metric if available
        try:
            settings.metric = mm.getUniversalMetric(mesh)
        except:
            pass

        mm.fillHoleNicely(mesh, he, settings)

        # Check if hole was filled
        hole_ids_after = mesh.topology.findHoleRepresentiveEdges()
        filled = len(hole_ids_after) < len(hole_ids)

        print(f"✓ fillHoleNicely: Holes before={len(hole_ids)}, after={len(hole_ids_after)}")
        return True
    except Exception as e:
        print(f"✗ fillHoleNicely failed: {e}")
        return False


def test_subdivision(mesh_path):
    """Test mesh subdivision."""
    print("\n" + "=" * 70)
    print("Test 6: Mesh Subdivision")
    print("=" * 70)

    try:
        mesh = mm.loadMesh(mesh_path)

        # Compute average edge length
        avg_edge_length = mesh.averageEdgeLength() if hasattr(mesh, 'averageEdgeLength') else 0.1

        # Create a region to subdivide (all faces)
        all_faces = mm.FaceBitSet()
        for i in range(min(10, mesh.topology.faceSize())):  # Just a few faces
            all_faces.set(mm.FaceId(i), True)

        new_verts = mm.VertBitSet()
        subdiv_settings = mm.SubdivideSettings()
        subdiv_settings.maxEdgeLen = avg_edge_length * 0.5
        subdiv_settings.maxEdgeSplits = 100
        subdiv_settings.region = all_faces
        subdiv_settings.newVerts = new_verts

        # Try to set smoothMode if available
        try:
            subdiv_settings.smoothMode = mm.SubdivideSettings.SmoothMode.Linear
            print("  Using smoothMode parameter")
        except:
            print("  smoothMode parameter not available")

        mm.subdivideMesh(mesh, subdiv_settings)
        print(f"✓ subdivideMesh: Created {new_verts.count()} new vertices")
        return True
    except Exception as e:
        print(f"✗ subdivideMesh failed: {e}")
        return False


def main():
    """Run all tests."""
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
        if not os.path.exists(mesh_path):
            print(f"Error: Mesh file not found: {mesh_path}")
            sys.exit(1)
        temp_file = None
    else:
        temp_file = create_test_mesh()
        mesh_path = temp_file

    print_version_info()

    # Run all tests
    results = []
    results.append(("Basic Operations", test_basic_operations(mesh_path)))
    results.append(("Component Operations", test_component_operations(mesh_path)))
    results.append(("Hole Filling (Original)", test_hole_filling_original(mesh_path)))
    results.append(("Hole Filling (With Metric)", test_hole_filling_with_metric(mesh_path)))
    results.append(("Hole Filling (Nicely)", test_hole_filling_nicely(mesh_path)))
    results.append(("Subdivision", test_subdivision(mesh_path)))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed}/{total}")

    # Cleanup
    if temp_file and os.path.exists(temp_file):
        os.unlink(temp_file)

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
