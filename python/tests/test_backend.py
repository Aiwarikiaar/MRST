"""Tests for the Octave backend type conversion."""

import numpy as np
from mrst.backend import _convert_struct


def test_convert_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = _convert_struct(arr)
    np.testing.assert_array_equal(result, arr)


def test_convert_column_vector_to_1d():
    arr = np.array([[1.0], [2.0], [3.0]])
    result = _convert_struct(arr)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_convert_2d_array_preserved():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _convert_struct(arr)
    np.testing.assert_array_equal(result, arr)


def test_convert_dict():
    d = {"a": np.array([1, 2]), "b": "hello"}
    result = _convert_struct(d)
    assert isinstance(result, dict)
    np.testing.assert_array_equal(result["a"], [1, 2])
    assert result["b"] == "hello"


def test_convert_nested_dict():
    d = {"cells": {"num": 100, "volumes": np.array([[1.0], [2.0]])}}
    result = _convert_struct(d)
    assert result["cells"]["num"] == 100
    np.testing.assert_array_equal(result["cells"]["volumes"], [1.0, 2.0])


def test_convert_scalar():
    assert _convert_struct(42) == 42
    assert _convert_struct(3.14) == 3.14
    assert _convert_struct("hello") == "hello"
