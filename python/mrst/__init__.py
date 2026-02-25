"""
mrst-python: Python interface to MRST via GNU Octave.

Provides a Pythonic API to the MATLAB Reservoir Simulation Toolbox (MRST),
running MRST inside GNU Octave and converting results to NumPy arrays.

Usage:
    import mrst
    session = mrst.init("/path/to/MRST")
    G = session.grid.cart_grid([10, 10, 5])
"""

from mrst.session import MRSTSession, init

__version__ = "0.1.0"
__all__ = ["MRSTSession", "init"]
