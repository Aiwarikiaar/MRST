"""
MRST session management.

Manages the Octave backend and provides access to MRST submodules
(grid, simulate, io, visualization).
"""

import os
from pathlib import Path

from mrst.backend import OctaveBackend
from mrst.grid import GridModule
from mrst.simulate import SimulateModule
from mrst.io import IOModule


class MRSTSession:
    """A session wrapping an Octave process running MRST.

    Provides access to MRST functionality through submodules:
        - session.grid: Grid creation and manipulation
        - session.simulate: Simulation execution
        - session.io: File I/O (ECLIPSE deck reading, etc.)
        - session.octave: Direct access to the Octave backend

    Parameters
    ----------
    mrst_root : str or Path
        Path to the MRST installation directory (containing startup.m).
    octave_executable : str, optional
        Path to the Octave executable. Defaults to 'octave'.
    """

    def __init__(self, mrst_root, octave_executable="octave"):
        self._mrst_root = Path(mrst_root).resolve()
        if not (self._mrst_root / "startup.m").exists():
            raise FileNotFoundError(
                f"startup.m not found in {self._mrst_root}. "
                f"Is this a valid MRST installation?"
            )

        self._backend = OctaveBackend(
            mrst_root=str(self._mrst_root),
            octave_executable=octave_executable,
        )

        # Initialize submodules
        self.grid = GridModule(self._backend)
        self.simulate = SimulateModule(self._backend)
        self.io = IOModule(self._backend)

    @property
    def octave(self):
        """Direct access to the Octave backend for custom commands."""
        return self._backend

    def load_modules(self, *module_names):
        """Load MRST modules (equivalent to mrstModule('add', ...)).

        Parameters
        ----------
        *module_names : str
            Names of MRST modules to load (e.g., 'ad-core', 'ad-blackoil').
        """
        self._backend.load_modules(*module_names)

    def eval(self, command):
        """Evaluate a raw MATLAB/Octave command string.

        Parameters
        ----------
        command : str
            MATLAB/Octave command to evaluate.

        Returns
        -------
        result
            The result of the evaluation, converted to Python types.
        """
        return self._backend.eval(command)

    def close(self):
        """Close the Octave session."""
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"MRSTSession(mrst_root='{self._mrst_root}')"


def init(mrst_root=None, octave_executable="octave"):
    """Initialize an MRST session.

    Parameters
    ----------
    mrst_root : str or Path, optional
        Path to the MRST installation. If None, attempts to detect
        from the environment variable MRST_ROOT or from the parent
        directory of this package.
    octave_executable : str, optional
        Path to the Octave executable. Defaults to 'octave'.

    Returns
    -------
    MRSTSession
        An initialized MRST session ready for use.
    """
    if mrst_root is None:
        mrst_root = os.environ.get("MRST_ROOT")

    if mrst_root is None:
        # Try to detect from package location (python/ is inside MRST/)
        package_dir = Path(__file__).resolve().parent
        candidate = package_dir.parent.parent  # python/mrst/ -> python/ -> MRST/
        if (candidate / "startup.m").exists():
            mrst_root = candidate

    if mrst_root is None:
        raise RuntimeError(
            "Cannot locate MRST installation. Either pass mrst_root explicitly, "
            "set the MRST_ROOT environment variable, or install this package "
            "from within the MRST directory tree."
        )

    return MRSTSession(mrst_root, octave_executable=octave_executable)
