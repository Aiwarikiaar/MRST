"""
I/O module.

Provides wrappers for MRST file I/O functions:
    - read_eclipse_deck: Read ECLIPSE input decks
    - convert_deck_units: Convert units in a deck
    - init_eclipse_grid: Initialize grid from deck
    - init_eclipse_rock: Initialize rock properties from deck
"""

from mrst.io.io_module import IOModule

__all__ = ["IOModule"]
