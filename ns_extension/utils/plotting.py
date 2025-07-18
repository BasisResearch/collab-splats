from typing import Optional, Union
import numpy as np
import pyvista as pv

# Basic loading and plotting
def load_and_plot_ply(mesh_path: str, attribute: Optional[Union[str, np.ndarray]] = None, rgb: bool = True):
    """
    Load a PLY mesh file and display it with basic visualization
    """
    # Load the PLY file
    mesh = pv.read(mesh_path)
    
    # Print basic information about the mesh
    print(f"Number of points: {mesh.n_points}")
    print(f"Number of cells: {mesh.n_cells}")
    print(f"Bounds: {mesh.bounds}")
    
    # Create a plotter and add the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=attribute, rgb=rgb)
    plotter.show_axes()
    plotter.show()