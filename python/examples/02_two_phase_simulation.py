"""
Example 02: Two-phase oil-water simulation.

Demonstrates:
    - Setting up a simple two-phase model
    - Adding injection and production wells
    - Running a simulation
    - Extracting results as NumPy arrays
    - Plotting pressure and saturation

Requires MRST modules: ad-core, ad-blackoil, ad-props
"""

import numpy as np
import mrst
from mrst.utils.units import *
from mrst.utils.helpers import extract_well_data, states_to_arrays

# Initialize MRST and load required modules
session = mrst.init()
session.load_modules("ad-core", "ad-blackoil", "ad-props", "incomp")

# Create grid: 20x20x5 cells, 1000x1000x50 meters
G = session.grid.cart_grid([20, 20, 5], [1000, 1000, 50])
G = session.grid.compute_geometry(G)

# Define rock properties: 100 mD permeability, 20% porosity
rock = session.io.make_rock(G, 100 * milli * darcy, 0.2)

# Define simple two-phase fluid
fluid = session.io.init_simple_fluid(
    mu=[1 * cp, 5 * cp],         # Water and oil viscosity
    rho=[1000, 700],              # Water and oil density (kg/m^3)
    n=[2, 2],                     # Corey exponents
)

# Add wells: injector in corner cell, producer in opposite corner
# Note: MRST uses 1-based indexing for cells
W = session.simulate.add_well(
    None, G, rock, cells=[1],
    type="rate", val=500 * stb / day,
    radius=0.1, name="Injector", comp_i=[1, 0], sign=1,
)
W = session.simulate.add_well(
    W, G, rock, cells=[G["cells"]["num"]],
    type="bhp", val=100 * barsa,
    radius=0.1, name="Producer", comp_i=[0, 1], sign=-1,
)

# Initial state: uniform pressure, fully oil-saturated
init_state = session.simulate.init_state(
    G,
    pressure=200 * barsa,
    saturation=[0, 1],  # [Sw, So]
)

# Create schedule: 50 timesteps of 30 days each
timesteps = np.ones(50) * 30 * day
schedule = session.simulate.simple_schedule(timesteps, W=W)

# Run simulation
print("Running simulation...")
well_sols, states, report = session.simulate.simulate_schedule(
    init_state, None, schedule, verbose=True,
)

# Extract results
bhp_data = extract_well_data(well_sols, "bhp")
pressures = states_to_arrays(states, "pressure")
saturations = states_to_arrays(states, "s")

print(f"\nSimulation complete!")
print(f"Timesteps: {len(states)}")
print(f"Pressure range at final step: "
      f"{pressures[-1].min() / barsa:.1f} - {pressures[-1].max() / barsa:.1f} bar")

# Plot final pressure distribution
from mrst.visualization import plot_cell_data
ax = plot_cell_data(G, pressures[-1] / barsa, cmap="coolwarm")
ax.set_title("Final Pressure (bar)")

session.close()
