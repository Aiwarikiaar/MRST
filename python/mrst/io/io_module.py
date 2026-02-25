"""
I/O module: Pythonic wrappers for MRST file I/O functions.

Wraps MRST's model-io/ functions for reading ECLIPSE decks and
other reservoir data formats.
"""

from pathlib import Path

import numpy as np


class IOModule:
    """Provides file I/O functions for reservoir data.

    Parameters
    ----------
    backend : OctaveBackend
        The Octave backend to delegate calls to.
    """

    def __init__(self, backend):
        self._backend = backend

    def _ensure_modules(self):
        """Load required I/O modules."""
        self._backend.load_modules("deckformat")

    def read_eclipse_deck(self, filename):
        """Read an ECLIPSE input deck.

        Wraps MRST's ``readEclipseDeck`` function.

        Parameters
        ----------
        filename : str or Path
            Path to the ECLIPSE .DATA file.

        Returns
        -------
        dict
            Deck structure with sections:
            - RUNSPEC: Basic metadata (dimensions, etc.)
            - GRID: Grid specification (COORD/ZCORN, PERMX, etc.)
            - PROPS: Fluid properties (PVT, relperm)
            - REGIONS: Region partition
            - SOLUTION: Initial conditions
            - SCHEDULE: Timesteps and well controls
        """
        self._ensure_modules()
        filename = str(Path(filename).resolve())
        return self._backend.call("readEclipseDeck", filename)

    def convert_deck_units(self, deck):
        """Convert deck units to SI.

        Wraps MRST's ``convertDeckUnits`` function.

        Parameters
        ----------
        deck : dict
            Deck structure from read_eclipse_deck.

        Returns
        -------
        dict
            Deck with all quantities converted to SI units.
        """
        self._ensure_modules()
        self._backend.push("deck_py", deck)
        self._backend.eval("deck_py = convertDeckUnits(deck_py);")
        return self._backend.pull("deck_py")

    def init_eclipse_grid(self, deck):
        """Initialize a grid from an ECLIPSE deck.

        Wraps MRST's ``initEclipseGrid`` function.

        Parameters
        ----------
        deck : dict
            Deck structure (from read_eclipse_deck).

        Returns
        -------
        dict
            Grid structure.
        """
        self._ensure_modules()
        self._backend.push("deck_py", deck)
        self._backend.eval("G_py = initEclipseGrid(deck_py);")
        return self._backend.pull("G_py")

    def init_eclipse_rock(self, deck):
        """Initialize rock properties from an ECLIPSE deck.

        Wraps MRST's ``initEclipseRock`` function.

        Parameters
        ----------
        deck : dict
            Deck structure (from read_eclipse_deck).

        Returns
        -------
        dict
            Rock structure with fields:
            - perm: Permeability tensor (n x d array, m^2)
            - poro: Porosity (n x 1 array)
        """
        self._ensure_modules()
        self._backend.push("deck_py", deck)
        self._backend.eval("rock_py = initEclipseRock(deck_py);")
        return self._backend.pull("rock_py")

    def make_rock(self, G, perm, poro):
        """Create a rock structure from permeability and porosity.

        Wraps MRST's ``makeRock`` function.

        Parameters
        ----------
        G : dict
            Grid structure.
        perm : float or array_like
            Permeability in m^2. Scalar for isotropic, array for heterogeneous.
        poro : float or array_like
            Porosity (0-1). Scalar for uniform, array for heterogeneous.

        Returns
        -------
        dict
            Rock structure with 'perm' and 'poro' fields.
        """
        self._backend.push("G_py", G)

        if np.isscalar(perm):
            perm_str = str(perm)
        else:
            self._backend.push("perm_py", np.array(perm, dtype=np.float64))
            perm_str = "perm_py"

        if np.isscalar(poro):
            poro_str = str(poro)
        else:
            self._backend.push("poro_py", np.array(poro, dtype=np.float64))
            poro_str = "poro_py"

        self._backend.eval(
            f"rock_py = makeRock(G_py, {perm_str}, {poro_str});"
        )
        return self._backend.pull("rock_py")

    def init_simple_fluid(self, mu, rho, n):
        """Create a simple two-phase fluid.

        Wraps MRST's ``initSimpleFluid`` function.

        Parameters
        ----------
        mu : list of float
            Viscosities [mu_w, mu_o] in Pa*s.
        rho : list of float
            Densities [rho_w, rho_o] in kg/m^3.
        n : list of float
            Corey exponents [n_w, n_o].

        Returns
        -------
        object
            Fluid structure.
        """
        mu = np.array(mu, dtype=np.float64)
        rho = np.array(rho, dtype=np.float64)
        n = np.array(n, dtype=np.float64)
        self._backend.push("mu_py", mu)
        self._backend.push("rho_py", rho)
        self._backend.push("n_py", n)
        self._backend.eval(
            "fluid_py = initSimpleFluid("
            "'mu', mu_py, 'rho', rho_py, 'n', n_py);"
        )
        return self._backend.pull("fluid_py", convert=False)

    def wells_to_dataframe(self, well_sols):
        """Convert well solutions to a pandas DataFrame.

        Parameters
        ----------
        well_sols : list
            Well solutions from simulate_schedule.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns for each well quantity at each timestep.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. "
                "Install with: pip install pandas"
            )

        self._backend.push("ws_py", well_sols)
        self._backend.eval(
            "[qWs, qOs, qGs, bhp] = deal(cell(numel(ws_py), 1));"
        )
        # Extract well data timestep by timestep via Octave
        self._backend.eval("""
            nsteps = numel(ws_py);
            well_data = struct();
            for i = 1:nsteps
                ws = ws_py{i};
                for j = 1:numel(ws)
                    wname = ws(j).name;
                    if ~isfield(well_data, wname)
                        well_data.(wname) = struct('bhp', [], 'qWs', [], 'qOs', [], 'qGs', []);
                    end
                    well_data.(wname).bhp(end+1) = ws(j).bhp;
                    if isfield(ws(j), 'qWs'), well_data.(wname).qWs(end+1) = ws(j).qWs; end
                    if isfield(ws(j), 'qOs'), well_data.(wname).qOs(end+1) = ws(j).qOs; end
                    if isfield(ws(j), 'qGs'), well_data.(wname).qGs(end+1) = ws(j).qGs; end
                end
            end
        """)

        data = self._backend.pull("well_data")
        records = []
        if isinstance(data, dict):
            for well_name, quantities in data.items():
                if isinstance(quantities, dict):
                    for qty_name, values in quantities.items():
                        if hasattr(values, "__len__"):
                            for t, v in enumerate(values):
                                records.append({
                                    "well": well_name,
                                    "timestep": t,
                                    "quantity": qty_name,
                                    "value": float(v),
                                })

        return pd.DataFrame(records)
