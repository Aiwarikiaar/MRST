"""Tests for unit conversion constants."""

import math
from mrst.utils.units import *


def test_length_conversions():
    assert meter == 1.0
    assert ft == 0.3048
    assert inch == 0.0254


def test_time_conversions():
    assert second == 1.0
    assert minute == 60.0
    assert hour == 3600.0
    assert day == 86400.0


def test_pressure_conversions():
    assert barsa == 1e5
    assert math.isclose(atm, 101325.0)
    assert math.isclose(psia, 6894.757293168)


def test_permeability():
    assert math.isclose(darcy, 9.869232667160128e-13)
    # 100 millidarcy
    assert math.isclose(100 * milli * darcy, 100 * 9.869232667160128e-16)


def test_viscosity():
    assert cp == 1e-3  # centipoise = milli Pascal-second


def test_volume():
    assert math.isclose(stb, 0.158987294928)


def test_prefixes():
    assert milli == 1e-3
    assert kilo == 1e3
    assert mega == 1e6
