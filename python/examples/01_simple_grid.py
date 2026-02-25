"""
Example 01: Creating and visualizing a simple Cartesian grid.

Demonstrates:
    - Initializing an MRST session
    - Creating a Cartesian grid
    - Computing geometry
    - Accessing grid data as NumPy arrays
    - Plotting the grid
"""

import numpy as np
import mrst

# Initialize MRST (auto-detects MRST root from package location)
session = mrst.init()

# Create a 10x10x5 Cartesian grid with physical dimensions 1000x1000x50 meters
G = session.grid.cart_grid([10, 10, 5], [1000, 1000, 50])

# Compute geometry (adds volumes, centroids, areas, normals)
G = session.grid.compute_geometry(G)

# Access grid data as NumPy arrays
print(f"Number of cells: {G['cells']['num']}")
print(f"Number of faces: {G['faces']['num']}")
print(f"Number of nodes: {G['nodes']['num']}")
print(f"Cell volumes shape: {G['cells']['volumes'].shape}")
print(f"Cell centroids shape: {G['cells']['centroids'].shape}")
print(f"Total volume: {np.sum(G['cells']['volumes']):.0f} m^3")

# Plot the grid
from mrst.visualization import plot_grid
ax = plot_grid(G)

# You can also create tensor grids with variable spacing
x = np.concatenate([np.linspace(0, 100, 6), np.linspace(150, 1000, 10)])
y = np.linspace(0, 500, 11)
z = np.array([0, 5, 10, 20, 50])

G2 = session.grid.tensor_grid(x, y, z)
G2 = session.grid.compute_geometry(G2)
print(f"\nTensor grid: {G2['cells']['num']} cells")

session.close()
