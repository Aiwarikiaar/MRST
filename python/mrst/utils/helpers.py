"""
Helper functions for data extraction and conversion.
"""

import numpy as np


def extract_well_data(well_sols, field="bhp"):
    """Extract a specific field from well solutions across all timesteps.

    Parameters
    ----------
    well_sols : list
        Well solutions from simulate_schedule (list of timestep results).
    field : str
        Field to extract (e.g., 'bhp', 'qWs', 'qOs', 'qGs').

    Returns
    -------
    dict
        Dictionary mapping well names to NumPy arrays of the field values
        over time.
    """
    result = {}
    for t, ws in enumerate(well_sols):
        if isinstance(ws, dict):
            wells = [ws]
        elif isinstance(ws, (list, np.ndarray)):
            wells = ws
        else:
            continue

        for w in wells:
            if not isinstance(w, dict):
                continue
            name = w.get("name", f"well_{len(result)}")
            val = w.get(field)
            if val is not None:
                if name not in result:
                    result[name] = []
                result[name].append(float(np.asarray(val).ravel()[0]))

    return {name: np.array(vals) for name, vals in result.items()}


def states_to_arrays(states, field="pressure"):
    """Extract a field from simulation states into a 2D array.

    Parameters
    ----------
    states : list of dict
        Reservoir states from simulate_schedule.
    field : str
        State field to extract (e.g., 'pressure', 's' for saturation).

    Returns
    -------
    numpy.ndarray
        Array of shape (n_timesteps, n_cells) for scalar fields,
        or (n_timesteps, n_cells, n_phases) for vector fields like saturation.
    """
    arrays = []
    for state in states:
        if isinstance(state, dict):
            val = state.get(field)
            if val is not None:
                arrays.append(np.asarray(val))

    if not arrays:
        return np.array([])

    return np.stack(arrays, axis=0)
