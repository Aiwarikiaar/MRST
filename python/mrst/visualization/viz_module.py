"""
Visualization module: Python-native plotting for MRST grids and data.

Uses matplotlib for 2D plots and pyvista for 3D visualization.
Grid data is extracted from the MRST grid structure (Python dict with
NumPy arrays) and rendered directly in Python — no Octave needed for plotting.
"""

import numpy as np


def _get_cell_faces(G):
    """Extract cell-face connectivity from grid structure.

    Returns a list of lists: cell_faces[i] contains the face indices
    for cell i.
    """
    cells_faces = G["cells"]["faces"]
    if isinstance(cells_faces, dict):
        # Struct with separate fields
        return cells_faces
    # Simple array: columns are [face_index, tag]
    return cells_faces


def _get_face_nodes(G):
    """Extract face-node connectivity from grid structure."""
    return G["faces"]["nodes"]


def plot_grid(G, cells=None, color="yellow", edge_color="black",
              alpha=0.3, ax=None, backend="matplotlib"):
    """Plot a grid structure.

    Parameters
    ----------
    G : dict
        Grid structure with geometry computed (must have centroids).
    cells : array_like, optional
        Subset of cell indices to plot (0-based Python indexing).
        If None, plots all cells.
    color : str or array_like
        Face color.
    edge_color : str
        Edge color.
    alpha : float
        Transparency (0 = transparent, 1 = opaque).
    ax : matplotlib Axes, optional
        Axes to plot on. Created if None.
    backend : str
        'matplotlib' for 2D/3D matplotlib, 'pyvista' for interactive 3D.

    Returns
    -------
    ax or plotter
        The matplotlib axes or pyvista plotter used.
    """
    if backend == "pyvista":
        return _plot_grid_pyvista(G, cells, color, edge_color, alpha)
    return _plot_grid_matplotlib(G, cells, color, edge_color, alpha, ax)


def _plot_grid_matplotlib(G, cells, color, edge_color, alpha, ax):
    """Plot grid using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    coords = G["nodes"]["coords"]
    griddim = int(G.get("griddim", coords.shape[1]))
    is_3d = griddim == 3

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    # Get boundary faces for plotting
    face_centroids = G["faces"].get("centroids")
    node_coords = coords

    # For simple Cartesian grids, plot cell centroids as a scatter
    # For full face rendering, we need face-node connectivity
    centroids = G["cells"].get("centroids")
    if centroids is not None:
        if cells is not None:
            centroids = centroids[cells]

        if is_3d:
            ax.scatter(
                centroids[:, 0], centroids[:, 1], centroids[:, 2],
                c=color, alpha=alpha, s=1,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            ax.scatter(
                centroids[:, 0], centroids[:, 1],
                c=color, alpha=alpha, s=5,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

    ax.set_aspect("equal")
    ax.set_title("MRST Grid")
    return ax


def _plot_grid_pyvista(G, cells, color, edge_color, alpha):
    """Plot grid using pyvista for interactive 3D visualization."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "pyvista is required for 3D visualization. "
            "Install with: pip install pyvista"
        )

    coords = G["nodes"]["coords"]
    centroids = G["cells"].get("centroids")

    if centroids is not None:
        if cells is not None:
            centroids = centroids[cells]

        cloud = pv.PolyData(centroids)
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, color=color, opacity=alpha, point_size=5)
        plotter.show_axes()
        plotter.show()
        return plotter

    return None


def plot_cell_data(G, data, cells=None, cmap="viridis", clim=None,
                   ax=None, backend="matplotlib", colorbar=True):
    """Plot scalar data on grid cells.

    Parameters
    ----------
    G : dict
        Grid structure with geometry.
    data : array_like
        Scalar data, one value per cell (or per selected cell).
    cells : array_like, optional
        Subset of cells to plot (0-based). If None, all cells.
    cmap : str
        Matplotlib colormap name.
    clim : tuple of (float, float), optional
        Color limits (min, max).
    ax : matplotlib Axes, optional
        Axes to plot on.
    backend : str
        'matplotlib' or 'pyvista'.
    colorbar : bool
        Whether to add a colorbar.

    Returns
    -------
    ax or plotter
        The plotting object used.
    """
    data = np.asarray(data).ravel()

    if backend == "pyvista":
        return _plot_cell_data_pyvista(G, data, cells, cmap, clim)
    return _plot_cell_data_matplotlib(G, data, cells, cmap, clim, ax, colorbar)


def _plot_cell_data_matplotlib(G, data, cells, cmap, clim, ax, colorbar):
    """Plot cell data using matplotlib scatter on centroids."""
    import matplotlib.pyplot as plt

    coords = G["nodes"]["coords"]
    centroids = G["cells"]["centroids"]
    griddim = int(G.get("griddim", coords.shape[1]))
    is_3d = griddim == 3

    if cells is not None:
        centroids = centroids[cells]
        data = data[: len(cells)] if len(data) > len(cells) else data

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if clim is None:
        clim = (np.nanmin(data), np.nanmax(data))

    if is_3d:
        sc = ax.scatter(
            centroids[:, 0], centroids[:, 1], centroids[:, 2],
            c=data, cmap=cmap, vmin=clim[0], vmax=clim[1], s=5,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        sc = ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c=data, cmap=cmap, vmin=clim[0], vmax=clim[1], s=10,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    if colorbar:
        plt.colorbar(sc, ax=ax)

    ax.set_aspect("equal")
    return ax


def _plot_cell_data_pyvista(G, data, cells, cmap, clim):
    """Plot cell data using pyvista."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "pyvista is required for 3D visualization. "
            "Install with: pip install pyvista"
        )

    centroids = G["cells"]["centroids"]
    if cells is not None:
        centroids = centroids[cells]
        data = data[: len(cells)] if len(data) > len(cells) else data

    cloud = pv.PolyData(centroids)
    cloud["data"] = data

    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud, scalars="data", cmap=cmap, clim=clim, point_size=5,
    )
    plotter.show_axes()
    plotter.add_scalar_bar()
    plotter.show()
    return plotter


def plot_faces(G, faces=None, color="blue", alpha=0.5, ax=None):
    """Plot grid faces.

    Parameters
    ----------
    G : dict
        Grid structure.
    faces : array_like, optional
        Face indices to plot. If None, plots boundary faces.
    color : str
        Face color.
    alpha : float
        Transparency.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    ax
        Matplotlib axes.
    """
    face_centroids = G["faces"].get("centroids")
    if face_centroids is None:
        raise ValueError("Grid must have geometry computed (run compute_geometry first)")

    import matplotlib.pyplot as plt

    if faces is not None:
        face_centroids = face_centroids[faces]

    is_3d = face_centroids.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if is_3d:
        ax.scatter(
            face_centroids[:, 0], face_centroids[:, 1], face_centroids[:, 2],
            c=color, alpha=alpha, s=2,
        )
    else:
        ax.scatter(
            face_centroids[:, 0], face_centroids[:, 1],
            c=color, alpha=alpha, s=3,
        )

    ax.set_aspect("equal")
    return ax


def plot_well(G, W, ax=None, color="red", marker="o", markersize=8):
    """Plot well locations on a grid.

    Parameters
    ----------
    G : dict
        Grid structure with geometry.
    W : dict or list of dict
        Well structure(s) from add_well.
    ax : matplotlib Axes, optional
        Axes to plot on.
    color : str
        Well marker color.
    marker : str
        Marker style.
    markersize : float
        Marker size.

    Returns
    -------
    ax
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    centroids = G["cells"]["centroids"]
    is_3d = centroids.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if isinstance(W, dict):
        W = [W]

    for well in W:
        cells = np.asarray(well.get("cells", [])).ravel().astype(int) - 1  # 1-based to 0-based
        if len(cells) == 0:
            continue

        well_coords = centroids[cells]
        name = well.get("name", "")

        if is_3d:
            ax.plot(
                well_coords[:, 0], well_coords[:, 1], well_coords[:, 2],
                color=color, marker=marker, markersize=markersize,
                linewidth=2, label=name,
            )
        else:
            ax.plot(
                well_coords[:, 0], well_coords[:, 1],
                color=color, marker=marker, markersize=markersize,
                linewidth=2, label=name,
            )

    if any(w.get("name") for w in W):
        ax.legend()

    return ax
