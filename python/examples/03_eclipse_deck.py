"""
Example 03: Reading and simulating an ECLIPSE deck.

Demonstrates:
    - Reading an ECLIPSE input deck
    - Converting units
    - Initializing grid and rock from deck data
    - Setting up a black-oil model from deck

Note: Requires an actual ECLIPSE .DATA file to run.
"""

import mrst

# Initialize MRST with deck format support
session = mrst.init()
session.load_modules("deckformat", "ad-core", "ad-blackoil", "ad-props")

# Read an ECLIPSE deck
# deck = session.io.read_eclipse_deck("/path/to/your/MODEL.DATA")

# Convert to SI units
# deck = session.io.convert_deck_units(deck)

# Initialize grid from deck
# G = session.io.init_eclipse_grid(deck)
# G = session.grid.compute_geometry(G)

# Initialize rock from deck
# rock = session.io.init_eclipse_rock(deck)

# Print summary
# print(f"Grid: {G['cells']['num']} cells")
# print(f"Porosity range: {rock['poro'].min():.3f} - {rock['poro'].max():.3f}")

print("This example requires an ECLIPSE .DATA file.")
print("Uncomment the code above and provide a valid file path to run.")

session.close()
