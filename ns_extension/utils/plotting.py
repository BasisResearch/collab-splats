import pyvista as pv
from typing import Optional, Union
import numpy as np
import torch

# Basic loading and plotting
def load_and_plot_ply(mesh_path: str, attribute: Optional[np.ndarray] = None):
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

    if attribute is not None:
        plotter.add_mesh(mesh, scalars=attribute, rgb=False)
    else:
        plotter.add_mesh(mesh, scalars="rgb", rgb=True)
    
    plotter.show_axes()
    plotter.show()
