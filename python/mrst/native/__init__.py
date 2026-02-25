"""
Native Python implementations of MRST functionality.

This subpackage contains pure Python/NumPy/SciPy implementations
that do not require Octave. These are ports of core MRST algorithms.

Modules:
    - grid: Grid data structures and creation (cartGrid, tensorGrid)
    - geometry: Geometry computation (computeGeometry)
"""

from mrst.native.grid import Grid, cart_grid, tensor_grid
from mrst.native.geometry import compute_geometry

__all__ = ["Grid", "cart_grid", "tensor_grid", "compute_geometry"]
