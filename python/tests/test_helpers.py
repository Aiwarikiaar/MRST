"""Tests for helper utilities."""

import numpy as np
from mrst.utils.helpers import extract_well_data, states_to_arrays


def test_extract_well_data_empty():
    result = extract_well_data([], "bhp")
    assert result == {}


def test_extract_well_data_basic():
    well_sols = [
        [{"name": "I1", "bhp": 200e5}, {"name": "P1", "bhp": 100e5}],
        [{"name": "I1", "bhp": 210e5}, {"name": "P1", "bhp": 95e5}],
    ]
    bhp = extract_well_data(well_sols, "bhp")
    assert "I1" in bhp
    assert "P1" in bhp
    assert len(bhp["I1"]) == 2
    np.testing.assert_allclose(bhp["I1"], [200e5, 210e5])
    np.testing.assert_allclose(bhp["P1"], [100e5, 95e5])


def test_states_to_arrays_pressure():
    states = [
        {"pressure": np.array([1.0, 2.0, 3.0])},
        {"pressure": np.array([1.1, 2.1, 3.1])},
    ]
    result = states_to_arrays(states, "pressure")
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result[1], [1.1, 2.1, 3.1])


def test_states_to_arrays_saturation():
    states = [
        {"s": np.array([[0.2, 0.8], [0.3, 0.7]])},
        {"s": np.array([[0.4, 0.6], [0.5, 0.5]])},
    ]
    result = states_to_arrays(states, "s")
    assert result.shape == (2, 2, 2)


def test_states_to_arrays_empty():
    result = states_to_arrays([], "pressure")
    assert result.size == 0


def test_states_to_arrays_missing_field():
    states = [{"pressure": np.array([1.0])}, {"temperature": np.array([300.0])}]
    result = states_to_arrays(states, "pressure")
    assert result.shape == (1, 1)
