"""
Unit conversion constants matching MRST's unit system.

MRST uses SI internally. These constants help convert to/from
common petroleum engineering units.

Usage:
    from mrst.utils.units import *
    p = 300 * barsa          # 300 bar in Pa
    k = 100 * milli * darcy  # 100 mD in m^2
    q = 500 * stb / day      # 500 STB/day in m^3/s
"""

# Prefixes
milli = 1e-3
centi = 1e-2
deci = 1e-1
kilo = 1e3
mega = 1e6
giga = 1e9

# Length
meter = 1.0
ft = 0.3048
inch = 0.0254

# Area / Volume
stb = 0.158987294928  # stock tank barrel in m^3
liter = 1e-3

# Time
second = 1.0
minute = 60.0
hour = 3600.0
day = 86400.0
year = 365.2425 * day

# Pressure
Pascal = 1.0
atm = 101325.0
barsa = 1e5
psia = 6894.757293168

# Permeability
darcy = 9.869232667160128e-13

# Viscosity
Pas = 1.0  # Pa*s
cp = milli * Pas  # centipoise

# Density
kg_per_m3 = 1.0

# Gravity
gravity = 9.80665  # m/s^2
