"""
Updated mesh cleaning functions compatible with MeshLib 3.0.7+

Key changes from original:
1. Added metric parameter to FillHoleParams (required in newer versions)
2. Improved error handling
3. Added alternative hole filling method
4. Updated subdivision settings for newer API
"""

import meshlib.mrmeshpy as mm
from tqdm import tqdm, trange


def clean_repair_mesh_updated(
    mesh_path: str,
    max_hole_size: float = 3.0,
    max_edge_splits: int = 10000,
    use_largest: bool = False,
    use_advanced_fill: bool = True,  # New parameter
):
    """
    Clean and repair mesh compatible with MeshLib 3.0.7+

    Args:
        mesh_path: Path to the mesh file
        max_hole_size: Maximum hole perimeter to fill
        max_edge_splits: Maximum number of edge splits for subdivision
        use_largest: If True, only keep the largest component
        use_advanced_fill: If True, use fillHoleNicely instead of fillHole

    Returns:
        Cleaned and repaired mesh
    """
    # Load mesh
    mesh = mm.loadMesh(mesh_path)

    # Identify all connected components
    components = mm.getAllComponents(mesh)

    # Determine component sizes
    sizes = [mask.count() for mask in components]

    # Always find largest cluster
    largest_idx = max(range(len(sizes)), key=lambda i: sizes[i])

    # Add the largest component
    combined = mm.Mesh()
    mesh_part = mm.MeshPart(mesh, components[largest_idx])
    combined.addMeshPart(mesh_part)

    # Remove the largest component from list of idxs
    n_removed = 0
    if not use_largest:
        idxs = list(range(len(sizes)))
        idxs.remove(largest_idx)

        # Add the remaining components if they fall within the bounds
        combined_bounds = combined.getBoundingBox()

        for idx in tqdm(idxs, desc="Finding components within bounds"):
            _temp = mm.Mesh()
            temp_mesh_part = mm.MeshPart(mesh, components[idx])
            _temp.addMeshPart(temp_mesh_part)

            if combined_bounds.contains(_temp.getBoundingBox()):
                mesh_part = mm.MeshPart(mesh, components[idx])
                combined.addMeshPart(mesh_part)
            else:
                n_removed += 1

    print(f"Removed {n_removed} components")
    mesh = combined

    # Compute average edge length
    avg_edge_length = 0.0
    num_edges = 0

    for i in trange(
        mesh.topology.undirectedEdgeSize(), desc="Calculating average edge length"
    ):
        dir_edge = mm.EdgeId(i * 2)
        org = mesh.topology.org(dir_edge)
        dest = mesh.topology.dest(dir_edge)
        avg_edge_length += (
            mesh.points.vec[dest.get()] - mesh.points.vec[org.get()]
        ).length()
        num_edges += 1
    avg_edge_length /= num_edges

    # Fill holes
    hole_ids = mesh.topology.findHoleRepresentiveEdges()

    # UPDATED: Use fillHoleNicely with settings for newer versions
    if use_advanced_fill:
        print(f"Filling {len(hole_ids)} holes using advanced method...")
        for he in tqdm(hole_ids, desc=f"Filling holes ({len(hole_ids)})"):
            try:
                perimeter = mesh.holePerimiter(he)
                if perimeter < max_hole_size:
                    # Use fillHoleNicely for better results
                    settings = mm.FillHoleNicelySettings()
                    settings.maxEdgeLen = avg_edge_length
                    settings.maxEdgeSplits = max_edge_splits

                    # Optional: Get the metric (required in some newer versions)
                    metric = mm.getUniversalMetric(mesh)
                    settings.metric = metric

                    mm.fillHoleNicely(mesh, he, settings)
                else:
                    print(f"Skipping hole {he} of perimeter {perimeter}")
            except Exception as e:
                print(f"Warning: Failed to fill hole {he}: {e}")
                # Try fallback to simple fill
                try:
                    mm.fillHoleTrivially(mesh, he)
                except Exception as e2:
                    print(f"Warning: Fallback fill also failed: {e2}")
    else:
        # ORIGINAL METHOD with updates for newer API
        fill_params = mm.FillHoleParams()

        # UPDATED: Set metric parameter (required in newer versions)
        try:
            fill_params.metric = mm.getUniversalMetric(mesh)
        except:
            pass  # May not be required in older versions

        for he in tqdm(hole_ids, desc=f"Filling holes ({len(hole_ids)})"):
            try:
                perimeter = mesh.holePerimiter(he)
                if perimeter < max_hole_size:
                    new_faces = mm.FaceBitSet()
                    fill_params.outNewFaces = new_faces
                    mm.fillHole(mesh, he, fill_params)

                    # UPDATED: Subdivision with improved settings
                    new_verts = mm.VertBitSet()
                    subdiv_settings = mm.SubdivideSettings()
                    subdiv_settings.maxEdgeLen = avg_edge_length
                    subdiv_settings.maxEdgeSplits = max_edge_splits
                    subdiv_settings.region = new_faces
                    subdiv_settings.newVerts = new_verts

                    # UPDATED: Set smoothMode if available (newer versions)
                    try:
                        subdiv_settings.smoothMode = mm.SubdivideSettings.SmoothMode.Linear
                    except:
                        pass  # Not available in older versions

                    mm.subdivideMesh(mesh, subdiv_settings)
                    mm.positionVertsSmoothly(mesh, new_verts)
                else:
                    print(f"Skipping hole {he} of perimeter {perimeter}")
            except Exception as e:
                print(f"Warning: Failed to fill hole {he}: {e}")

    return mesh


def clean_repair_mesh_simple(
    mesh_path: str,
    max_hole_size: float = 3.0,
    use_largest: bool = False,
):
    """
    Simplified mesh cleaning using newer API methods.

    This version leverages built-in MeshLib functions for automatic
    hole filling and component cleanup.
    """
    # Load mesh
    mesh = mm.loadMesh(mesh_path)

    # Use built-in hole filling with automatic settings
    hole_ids = mesh.topology.findHoleRepresentiveEdges()

    print(f"Filling {len(hole_ids)} holes...")
    for he in tqdm(hole_ids):
        perimeter = mesh.holePerimiter(he)
        if perimeter < max_hole_size:
            # Use the automatic nice filling method
            settings = mm.FillHoleNicelySettings()
            mm.fillHoleNicely(mesh, he, settings)

    # Remove small components (if decimation is available)
    components = mm.getAllComponents(mesh)
    if len(components) > 1:
        sizes = [mask.count() for mask in components]
        largest_idx = max(range(len(sizes)), key=lambda i: sizes[i])

        if use_largest:
            # Keep only largest
            result = mm.Mesh()
            mesh_part = mm.MeshPart(mesh, components[largest_idx])
            result.addMeshPart(mesh_part)
            return result
        else:
            # Remove small disconnected components
            threshold = sizes[largest_idx] * 0.01  # 1% of largest
            faces_to_remove = mm.FaceBitSet()

            for idx, (comp, size) in enumerate(zip(components, sizes)):
                if idx != largest_idx and size < threshold:
                    faces_to_remove |= comp

            if faces_to_remove.count() > 0:
                mm.deleteFaces(mesh, faces_to_remove)

    return mesh


# Compatibility function - tries updated method, falls back to original
def clean_repair_mesh(
    mesh_path: str,
    max_hole_size: float = 3.0,
    max_edge_splits: int = 10000,
    use_largest: bool = False,
):
    """
    Compatibility wrapper that tries the updated method first,
    falls back to simpler approaches if there are issues.
    """
    try:
        return clean_repair_mesh_updated(
            mesh_path,
            max_hole_size=max_hole_size,
            max_edge_splits=max_edge_splits,
            use_largest=use_largest,
            use_advanced_fill=True
        )
    except Exception as e:
        print(f"Advanced method failed ({e}), trying simple method...")
        try:
            return clean_repair_mesh_simple(
                mesh_path,
                max_hole_size=max_hole_size,
                use_largest=use_largest
            )
        except Exception as e2:
            print(f"Simple method also failed ({e2}), trying original approach...")
            # Fall back to original implementation
            return clean_repair_mesh_updated(
                mesh_path,
                max_hole_size=max_hole_size,
                max_edge_splits=max_edge_splits,
                use_largest=use_largest,
                use_advanced_fill=False
            )
