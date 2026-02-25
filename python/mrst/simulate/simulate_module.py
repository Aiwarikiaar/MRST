"""
Simulation module: Pythonic wrappers for MRST simulation functions.

Wraps MRST's autodiff/ad-core/ simulation machinery via the Octave backend.
"""

import numpy as np


class SimulateModule:
    """Provides simulation setup and execution functions.

    Parameters
    ----------
    backend : OctaveBackend
        The Octave backend to delegate calls to.
    """

    def __init__(self, backend):
        self._backend = backend

    def _ensure_modules(self):
        """Load the core AD modules if not already loaded."""
        self._backend.load_modules("ad-core", "ad-blackoil", "ad-props")

    def simulate_schedule(self, init_state, model, schedule, **kwargs):
        """Run a simulation using simulateScheduleAD.

        This is the main simulation driver. It takes an initial state,
        a physical model, and a schedule of timesteps/controls.

        Parameters
        ----------
        init_state : dict
            Initial reservoir state (pressure, saturations, etc.).
        model : object
            Physical model (pushed to Octave workspace).
        schedule : dict
            Schedule with 'step' and 'control' fields.
        **kwargs
            Additional keyword arguments passed to simulateScheduleAD:
            - verbose: bool, print convergence info
            - output_ministeps: bool, output all substeps

        Returns
        -------
        well_sols : list of dict
            Well solutions at each timestep.
        states : list of dict
            Reservoir states at each timestep.
        report : dict
            Simulation report with timing and convergence info.
        """
        self._ensure_modules()

        self._backend.push("initState_py", init_state)
        self._backend.push("model_py", model)
        self._backend.push("schedule_py", schedule)

        # Build optional arguments string
        opt_args = []
        if kwargs.get("verbose"):
            opt_args.append("'Verbose', true")
        if kwargs.get("output_ministeps"):
            opt_args.append("'OutputMinisteps', true")

        opt_str = ", ".join(opt_args)
        if opt_str:
            opt_str = ", " + opt_str

        cmd = (
            "[wellSols_py, states_py, report_py] = "
            f"simulateScheduleAD(initState_py, model_py, schedule_py{opt_str});"
        )
        self._backend.eval(cmd)

        well_sols = self._backend.pull("wellSols_py")
        states = self._backend.pull("states_py")
        report = self._backend.pull("report_py")
        return well_sols, states, report

    def water_model(self, G, rock, fluid, **kwargs):
        """Create a single-phase water model.

        Parameters
        ----------
        G : dict
            Grid structure with geometry computed.
        rock : dict
            Rock properties (perm, poro).
        fluid : dict
            Fluid properties.

        Returns
        -------
        object
            MRST WaterModel object (Octave reference).
        """
        self._ensure_modules()
        self._backend.push("G_py", G)
        self._backend.push("rock_py", rock)
        self._backend.push("fluid_py", fluid)
        self._backend.eval(
            "model_py = WaterModel(G_py, rock_py, 'fluid', fluid_py);"
        )
        return self._backend.pull("model_py", convert=False)

    def two_phase_oil_water_model(self, G, rock, fluid, **kwargs):
        """Create a two-phase oil-water model.

        Parameters
        ----------
        G : dict
            Grid structure with geometry computed.
        rock : dict
            Rock properties.
        fluid : dict
            Fluid properties with oil-water PVT data.

        Returns
        -------
        object
            MRST TwoPhaseOilWaterModel (Octave reference).
        """
        self._ensure_modules()
        self._backend.push("G_py", G)
        self._backend.push("rock_py", rock)
        self._backend.push("fluid_py", fluid)
        self._backend.eval(
            "model_py = TwoPhaseOilWaterModel(G_py, rock_py, fluid_py);"
        )
        return self._backend.pull("model_py", convert=False)

    def black_oil_model(self, G, rock, fluid, **kwargs):
        """Create a three-phase black-oil model.

        Parameters
        ----------
        G : dict
            Grid structure with geometry computed.
        rock : dict
            Rock properties.
        fluid : dict
            Fluid properties with black-oil PVT tables.

        Returns
        -------
        object
            MRST GenericBlackOilModel (Octave reference).
        """
        self._ensure_modules()
        self._backend.push("G_py", G)
        self._backend.push("rock_py", rock)
        self._backend.push("fluid_py", fluid)

        # Build optional args
        opt_pairs = []
        if "disgas" in kwargs:
            opt_pairs.append(
                f"'disgas', {'true' if kwargs['disgas'] else 'false'}"
            )
        if "vapoil" in kwargs:
            opt_pairs.append(
                f"'vapoil', {'true' if kwargs['vapoil'] else 'false'}"
            )
        opt_str = ", ".join(opt_pairs)
        if opt_str:
            opt_str = ", " + opt_str

        self._backend.eval(
            f"model_py = GenericBlackOilModel(G_py, rock_py, fluid_py{opt_str});"
        )
        return self._backend.pull("model_py", convert=False)

    def init_state(self, G, **kwargs):
        """Initialize a reservoir state.

        Parameters
        ----------
        G : dict
            Grid structure.
        **kwargs
            State fields:
            - pressure : float or array, initial pressure (Pa)
            - saturation : array, initial saturations, shape (ncells, nphases)

        Returns
        -------
        dict
            Initial state structure.
        """
        self._backend.push("G_py", G)
        ncells_cmd = "G_py.cells.num"

        parts = ["state_py = struct();"]
        if "pressure" in kwargs:
            p = kwargs["pressure"]
            if np.isscalar(p):
                parts.append(
                    f"state_py.pressure = repmat({p}, {ncells_cmd}, 1);"
                )
            else:
                self._backend.push("p_tmp", np.array(p, dtype=np.float64))
                parts.append("state_py.pressure = p_tmp;")

        if "saturation" in kwargs:
            sat = np.array(kwargs["saturation"], dtype=np.float64)
            self._backend.push("sat_tmp", sat)
            if sat.ndim == 1:
                # Single saturation value per phase, replicate for all cells
                nph = sat.size
                parts.append(
                    f"state_py.s = repmat(sat_tmp(:)', {ncells_cmd}, 1);"
                )
            else:
                parts.append("state_py.s = sat_tmp;")

        self._backend.eval(" ".join(parts))
        return self._backend.pull("state_py")

    def add_well(self, W, G, rock, cells, **kwargs):
        """Add a well to a well structure.

        Wraps MRST's ``addWell`` function.

        Parameters
        ----------
        W : object or None
            Existing well structure, or None for first well.
        G : dict
            Grid structure.
        rock : dict
            Rock properties.
        cells : array_like
            Cell indices perforated by the well (1-based).
        **kwargs
            Well parameters:
            - type : str, 'rate' or 'bhp'
            - val : float, control value (m3/s for rate, Pa for bhp)
            - radius : float, wellbore radius (m)
            - name : str, well name
            - comp_i : list, component injection fractions
            - sign : int, 1 for injector, -1 for producer

        Returns
        -------
        object
            Updated well structure.
        """
        self._ensure_modules()
        cells = np.array(cells, dtype=np.float64)

        self._backend.push("G_py", G)
        self._backend.push("rock_py", rock)
        self._backend.push("cells_py", cells)

        if W is None:
            self._backend.eval("W_py = [];")
        else:
            self._backend.push("W_py", W)

        # Build optional arguments
        opt_pairs = []
        for key in ("type", "val", "radius", "name", "sign"):
            if key in kwargs:
                v = kwargs[key]
                if isinstance(v, str):
                    opt_pairs.append(f"'{key}', '{v}'")
                else:
                    opt_pairs.append(f"'{key}', {v}")

        if "comp_i" in kwargs:
            comp = kwargs["comp_i"]
            comp_str = "[" + ", ".join(str(c) for c in comp) + "]"
            opt_pairs.append(f"'comp_i', {comp_str}")

        opt_str = ", ".join(opt_pairs)
        if opt_str:
            opt_str = ", " + opt_str

        self._backend.eval(
            f"W_py = addWell(W_py, G_py, rock_py, cells_py{opt_str});"
        )
        return self._backend.pull("W_py", convert=False)

    def simple_schedule(self, timesteps, W=None, bc=None, src=None):
        """Create a simple simulation schedule.

        Wraps MRST's ``simpleSchedule`` function.

        Parameters
        ----------
        timesteps : array_like
            Timestep sizes in seconds.
        W : object, optional
            Well structure.
        bc : object, optional
            Boundary conditions.
        src : object, optional
            Source terms.

        Returns
        -------
        dict
            Schedule structure with 'step' and 'control' fields.
        """
        self._ensure_modules()
        dt = np.array(timesteps, dtype=np.float64)
        self._backend.push("dt_py", dt)

        opt_parts = []
        if W is not None:
            self._backend.push("W_py", W)
            opt_parts.append("'W', W_py")
        if bc is not None:
            self._backend.push("bc_py", bc)
            opt_parts.append("'bc', bc_py")
        if src is not None:
            self._backend.push("src_py", src)
            opt_parts.append("'src', src_py")

        opt_str = ", ".join(opt_parts)
        if opt_str:
            opt_str = ", " + opt_str

        self._backend.eval(f"schedule_py = simpleSchedule(dt_py{opt_str});")
        return self._backend.pull("schedule_py", convert=False)
