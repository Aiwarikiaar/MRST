"""
Octave backend for MRST.

Manages the oct2py session, handles MRST initialization, module loading,
and type conversion between MATLAB/Octave and Python.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _convert_struct(obj):
    """Recursively convert oct2py Struct objects to plain Python dicts.

    Numeric arrays become NumPy arrays; scalars become Python ints/floats.
    """
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # Object array — may contain nested structs
        if obj.size == 1:
            return _convert_struct(obj.flat[0])
        return [_convert_struct(item) for item in obj.flat]
    if hasattr(obj, "_fieldnames"):
        # oct2py Struct-like object
        result = {}
        for name in obj._fieldnames:
            result[name] = _convert_struct(getattr(obj, name))
        return result
    if isinstance(obj, dict):
        return {k: _convert_struct(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[1] == 1:
            return obj.ravel()
        return obj
    return obj


class OctaveBackend:
    """Manages an Octave session running MRST.

    Parameters
    ----------
    mrst_root : str
        Path to the MRST installation directory.
    octave_executable : str
        Path to the Octave executable.
    """

    def __init__(self, mrst_root, octave_executable="octave"):
        self._mrst_root = mrst_root
        self._initialized = False
        self._loaded_modules = set()

        try:
            from oct2py import Oct2Py
        except ImportError:
            raise ImportError(
                "oct2py is required but not installed. "
                "Install it with: pip install oct2py"
            )

        logger.info("Starting Octave session...")
        self._oct = Oct2Py(executable=octave_executable)
        self._init_mrst()

    def _init_mrst(self):
        """Run MRST startup.m to initialize the environment."""
        startup_path = str(Path(self._mrst_root) / "startup.m")
        logger.info("Initializing MRST from %s", startup_path)
        self._oct.eval(f"run('{startup_path}')")
        self._initialized = True
        logger.info("MRST initialized successfully")

    def load_modules(self, *module_names):
        """Load MRST modules.

        Parameters
        ----------
        *module_names : str
            Module names (e.g., 'ad-core', 'ad-blackoil', 'deckformat').
        """
        new_modules = [m for m in module_names if m not in self._loaded_modules]
        if not new_modules:
            return

        modules_str = ", ".join(f"'{m}'" for m in new_modules)
        self._oct.eval(f"mrstModule('add', {modules_str})")
        self._loaded_modules.update(new_modules)
        logger.info("Loaded modules: %s", ", ".join(new_modules))

    def call(self, func_name, *args, nout=1, convert=True):
        """Call an MRST/Octave function and return the result.

        Parameters
        ----------
        func_name : str
            Name of the MATLAB/Octave function to call.
        *args
            Arguments to pass to the function. Python lists are converted
            to MATLAB arrays, dicts to structs, etc. by oct2py.
        nout : int
            Number of output arguments expected.
        convert : bool
            If True, convert Octave structs to Python dicts with NumPy arrays.

        Returns
        -------
        result
            The function output, optionally converted to Python types.
        """
        result = self._oct.feval(func_name, *args, nout=nout)

        if convert:
            if nout == 1:
                return _convert_struct(result)
            return tuple(_convert_struct(r) for r in result)

        return result

    def eval(self, command):
        """Evaluate a raw MATLAB/Octave command.

        Parameters
        ----------
        command : str
            Command to evaluate.
        """
        return self._oct.eval(command)

    def push(self, name, value):
        """Push a Python variable into the Octave workspace.

        Parameters
        ----------
        name : str
            Variable name in the Octave workspace.
        value
            The value to assign.
        """
        self._oct.push(name, value)

    def pull(self, name, convert=True):
        """Pull a variable from the Octave workspace into Python.

        Parameters
        ----------
        name : str
            Variable name in the Octave workspace.
        convert : bool
            If True, convert Octave structs to Python dicts.

        Returns
        -------
        result
            The variable value.
        """
        result = self._oct.pull(name)
        if convert:
            return _convert_struct(result)
        return result

    def close(self):
        """Terminate the Octave session."""
        if self._oct is not None:
            self._oct.exit()
            self._oct = None
            logger.info("Octave session closed")

    def __del__(self):
        self.close()
