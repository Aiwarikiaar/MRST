"""
Grid creation and manipulation module.

Provides Pythonic wrappers for MRST grid functions:
    - cart_grid: Create Cartesian grids
    - tensor_grid: Create tensor-product grids
    - compute_geometry: Add geometry information to grids
    - triangulate_grid: Create triangular grids
    - make_layered_grid: Create layered/extruded grids
"""

from mrst.grid.grid_module import GridModule

__all__ = ["GridModule"]
