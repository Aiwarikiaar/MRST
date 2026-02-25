"""
Native Python implementations of MRST functionality.

This subpackage contains pure Python/NumPy/SciPy implementations
that do not require Octave. These are ports of core MRST algorithms.

Modules:
    - grid: Grid data structures and creation (cartGrid, tensorGrid)
    - geometry: Geometry computation (computeGeometry)
    - rock: Rock properties (makeRock, poreVolume, permTensor)
    - fluid: Fluid properties (initSimpleFluid, initSimpleADIFluid)
"""

from mrst.native.grid import Grid, cart_grid, tensor_grid
from mrst.native.geometry import compute_geometry
from mrst.native.rock import Rock, make_rock, pore_volume, perm_tensor
from mrst.native.fluid import (
    SimpleFluid, init_simple_fluid,
    ADFluid, init_ad_fluid,
)

__all__ = [
    "Grid", "cart_grid", "tensor_grid", "compute_geometry",
    "Rock", "make_rock", "pore_volume", "perm_tensor",
    "SimpleFluid", "init_simple_fluid",
    "ADFluid", "init_ad_fluid",
]
