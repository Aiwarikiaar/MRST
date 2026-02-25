# mrst-python

Python interface to [MRST](https://www.sintef.no/projectweb/mrst/) (MATLAB Reservoir Simulation Toolbox) via GNU Octave.

This package provides a Pythonic API to MRST's reservoir simulation capabilities, returning results as NumPy arrays and optionally as pandas DataFrames.

## Requirements

- Python >= 3.8
- GNU Octave >= 9.0 (10.x recommended)
- MRST installed and accessible
- `oct2py` Python package

## Installation

```bash
cd python/
pip install -e ".[all]"
```

## Quick Start

```python
import mrst

# Initialize MRST (starts Octave session, runs startup.m)
session = mrst.init()

# Create a simple Cartesian grid
G = session.grid.cart_grid([10, 10, 5], [1000, 1000, 50])

# Compute geometry
G = session.grid.compute_geometry(G)

# Access grid data as NumPy arrays
print(f"Number of cells: {G['cells']['num']}")
print(f"Cell volumes shape: {G['cells']['volumes'].shape}")
print(f"Cell centroids shape: {G['cells']['centroids'].shape}")
```

## Architecture

```
Python Application
    |
    v
mrst-python (Pythonic API)
    |
    v
oct2py (Python <-> Octave bridge)
    |
    v
GNU Octave running MRST
```

The wrapper delegates computation to MRST running inside Octave, converting
MATLAB structs to Python dicts with NumPy arrays. This ensures numerical
correctness — MRST's battle-tested simulation engine does the heavy lifting.
