"""
Visualization module.

Provides Python-native plotting for MRST grids and simulation results
using matplotlib and optionally pyvista for 3D visualization.

Functions:
    - plot_grid: Plot grid structure
    - plot_cell_data: Plot scalar data on grid cells
    - plot_well: Plot well trajectories
"""

from mrst.visualization.viz_module import (
    plot_grid,
    plot_cell_data,
    plot_faces,
    plot_well,
)

__all__ = ["plot_grid", "plot_cell_data", "plot_faces", "plot_well"]
