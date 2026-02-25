"""
Simulation module.

Provides wrappers for MRST simulation functions:
    - simulate_schedule: Run a simulation with schedule control
    - setup_model: Configure simulation models (black-oil, compositional, etc.)
    - setup_wells: Configure well models
    - setup_schedule: Build simulation schedules
"""

from mrst.simulate.simulate_module import SimulateModule

__all__ = ["SimulateModule"]
