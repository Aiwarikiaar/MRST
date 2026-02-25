"""
Grid module: Pythonic wrappers for MRST grid functions.

Wraps MRST's core/gridprocessing/ functions via the Octave backend.
"""

import numpy as np


class GridModule:
    """Provides grid creation and manipulation functions.

    Parameters
    ----------
    backend : OctaveBackend
        The Octave backend to delegate calls to.
    """

    def __init__(self, backend):
        self._backend = backend

    def cart_grid(self, cell_dim, phys_dim=None):
        """Create a Cartesian grid.

        Wraps MRST's ``cartGrid`` function.

        Parameters
        ----------
        cell_dim : list of int
            Number of cells in each direction, e.g. [10, 10, 5].
        phys_dim : list of float, optional
            Physical dimensions in meters. Defaults to cell_dim.

        Returns
        -------
        dict
            Grid structure with cells, faces, nodes as nested dicts
            containing NumPy arrays.

        Examples
        --------
        >>> G = session.grid.cart_grid([10, 10, 5])
        >>> G = session.grid.cart_grid([10, 10, 5], [1000, 1000, 50])
        """
        cell_dim = np.array(cell_dim, dtype=np.float64)
        if phys_dim is not None:
            phys_dim = np.array(phys_dim, dtype=np.float64)
            return self._backend.call("cartGrid", cell_dim, phys_dim)
        return self._backend.call("cartGrid", cell_dim)

    def tensor_grid(self, x, y=None, z=None, depthz=None):
        """Create a tensor-product grid with variable cell sizes.

        Wraps MRST's ``tensorGrid`` function.

        Parameters
        ----------
        x : array_like
            Cell vertices along the x-direction.
        y : array_like, optional
            Cell vertices along the y-direction (for 2D/3D grids).
        z : array_like, optional
            Cell vertices along the z-direction (for 3D grids).
        depthz : array_like, optional
            Depth at upper reservoir nodes, shape (len(x), len(y)).

        Returns
        -------
        dict
            Grid structure.

        Examples
        --------
        >>> G = session.grid.tensor_grid([0, 1, 3, 6, 10])
        >>> G = session.grid.tensor_grid([0, 10, 30], [0, 5, 15], [0, 2, 5])
        """
        args = [np.array(x, dtype=np.float64)]
        if y is not None:
            args.append(np.array(y, dtype=np.float64))
        if z is not None:
            args.append(np.array(z, dtype=np.float64))
        if depthz is not None:
            args.extend(["depthz", np.array(depthz, dtype=np.float64)])
        return self._backend.call("tensorGrid", *args)

    def compute_geometry(self, G):
        """Add geometry information (centroids, volumes, areas) to a grid.

        Wraps MRST's ``computeGeometry`` function.

        Parameters
        ----------
        G : dict
            Grid structure (from cart_grid, tensor_grid, etc.).

        Returns
        -------
        dict
            Grid structure with added geometry fields:
            - G['cells']['volumes'] : cell volumes
            - G['cells']['centroids'] : cell center coordinates
            - G['faces']['areas'] : face areas
            - G['faces']['normals'] : face normal vectors
            - G['faces']['centroids'] : face center coordinates
        """
        self._backend.push("G_tmp", G)
        self._backend.eval("G_tmp = computeGeometry(G_tmp);")
        return self._backend.pull("G_tmp")

    def pebi(self, points, edges=None):
        """Create a 2D PEBI (perpendicular bisector) grid.

        Wraps MRST's ``pebi`` function.

        Parameters
        ----------
        points : array_like
            Generating points, shape (n, 2).
        edges : array_like, optional
            Constraint edges.

        Returns
        -------
        dict
            Grid structure.
        """
        points = np.array(points, dtype=np.float64)
        if edges is not None:
            return self._backend.call("pebi", points, np.array(edges))
        return self._backend.call("pebi", points)

    def remove_cells(self, G, cells):
        """Remove cells from a grid.

        Wraps MRST's ``removeCells`` function.

        Parameters
        ----------
        G : dict
            Grid structure.
        cells : array_like
            Indices of cells to remove (1-based, as in MRST).

        Returns
        -------
        dict
            Modified grid with cells removed.
        """
        cells = np.array(cells, dtype=np.float64)
        self._backend.push("G_tmp", G)
        self._backend.push("cells_tmp", cells)
        self._backend.eval("G_tmp = removeCells(G_tmp, cells_tmp);")
        return self._backend.pull("G_tmp")

    def extract_subgrid(self, G, cells):
        """Extract a subgrid from a grid.

        Wraps MRST's ``extractSubgrid`` function.

        Parameters
        ----------
        G : dict
            Grid structure.
        cells : array_like
            Indices of cells to extract (1-based).

        Returns
        -------
        dict
            Subgrid structure.
        """
        cells = np.array(cells, dtype=np.float64)
        self._backend.push("G_tmp", G)
        self._backend.push("cells_tmp", cells)
        self._backend.eval("G_tmp = extractSubgrid(G_tmp, cells_tmp);")
        return self._backend.pull("G_tmp")
