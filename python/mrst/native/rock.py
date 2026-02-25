"""
Native Python rock property data structures and functions.

Port of MRST's core/params/rock/ module:
  - makeRock.m   -> make_rock()
  - poreVolume.m -> pore_volume()
  - permTensor.m -> perm_tensor()

All arrays use NumPy and are shaped (num_cells, ...).
Works with the native Grid dataclass (0-based indexing).
"""

import warnings

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Rock:
    """Rock property structure for a grid.

    Attributes
    ----------
    perm : np.ndarray
        Permeability tensor, shape (num_cells, ncol) where ncol is:
        - 1: isotropic
        - 2 (2D) or 3 (3D): diagonal
        - 3 (2D) or 6 (3D): full symmetric tensor
        Units: m^2 (SI).
    poro : np.ndarray
        Porosity, shape (num_cells,). Dimensionless, in [0, 1].
    ntg : np.ndarray or None
        Net-to-gross ratio, shape (num_cells,). Dimensionless, in [0, 1].
        If None, treated as 1.0 everywhere.
    """
    perm: np.ndarray
    poro: np.ndarray
    ntg: Optional[np.ndarray] = None


def make_rock(G, perm, poro, ntg=None):
    """Create a Rock structure from permeability and porosity values.

    Port of MRST's makeRock.

    Parameters
    ----------
    G : Grid
        Grid structure (must have cells.num and grid_dim).
    perm : float or array_like
        Permeability in m^2. Accepted shapes:
        - Scalar: uniform isotropic permeability for all cells.
        - 1D array of length num_cells: per-cell isotropic.
        - Row vector of length 1/2/3 (2D) or 1/3/6 (3D): uniform tensor,
          broadcast to all cells.
        - 2D array of shape (num_cells, ncol): per-cell tensor.
    poro : float or array_like
        Porosity (0-1). Scalar or array of length num_cells.
    ntg : float or array_like, optional
        Net-to-gross factor (0-1). Scalar or array of length num_cells.

    Returns
    -------
    Rock
        Rock structure with perm, poro, and optionally ntg fields.

    Raises
    ------
    ValueError
        If perm column count is invalid for the grid dimension,
        or if array sizes don't match the grid.
    """
    nc = G.cells.num
    dim = G.grid_dim

    # --- Expand perm ---
    perm = np.asarray(perm, dtype=np.float64)
    if perm.ndim == 0:
        # Scalar → (nc, 1)
        perm = np.full((nc, 1), perm)
    elif perm.ndim == 1:
        if perm.size == nc:
            # Per-cell isotropic → column vector (nc, 1)
            perm = perm.reshape(-1, 1)
        else:
            # Row vector (tensor components) → (1, ncol) for broadcast
            perm = perm.reshape(1, -1)
    # Now perm is 2D
    if perm.shape[0] == 1:
        perm = np.tile(perm, (nc, 1))
    if perm.shape[0] != nc:
        raise ValueError(
            f"perm must have 1 or {nc} rows, got {perm.shape[0]}"
        )

    # Validate column count against dimension
    ncol = perm.shape[1]
    valid_cols = {1: {1}, 2: {1, 2, 3}, 3: {1, 3, 6}}
    if ncol not in valid_cols.get(dim, set()):
        raise ValueError(
            f"Permeability must have {sorted(valid_cols[dim])} columns "
            f"in {dim}D, got {ncol}"
        )

    # --- Expand poro ---
    poro = np.asarray(poro, dtype=np.float64).ravel()
    if poro.size == 1:
        poro = np.full(nc, poro[0])
    if poro.size != nc:
        raise ValueError(
            f"poro must have 1 or {nc} entries, got {poro.size}"
        )
    if np.any(poro <= 0):
        bad = np.where(poro <= 0)[0]
        warnings.warn(
            f"Zero or negative porosity in {len(bad)} cells: "
            f"{bad[:5].tolist()}{'...' if len(bad) > 5 else ''}"
        )

    # --- Expand ntg ---
    ntg_arr = None
    if ntg is not None:
        ntg_arr = np.asarray(ntg, dtype=np.float64).ravel()
        if ntg_arr.size == 1:
            ntg_arr = np.full(nc, ntg_arr[0])
        if ntg_arr.size != nc:
            raise ValueError(
                f"ntg must have 1 or {nc} entries, got {ntg_arr.size}"
            )

    return Rock(perm=perm, poro=poro, ntg=ntg_arr)


def pore_volume(G, rock):
    """Compute pore volumes of individual cells.

    Port of MRST's poreVolume:
        pv = rock.poro .* G.cells.volumes .* rock.ntg

    Parameters
    ----------
    G : Grid
        Grid with computed geometry (G.cells.volumes must exist).
    rock : Rock
        Rock structure with poro (and optionally ntg).

    Returns
    -------
    np.ndarray
        Pore volume for each cell, shape (num_cells,).

    Raises
    ------
    ValueError
        If G.cells.volumes has not been computed.
    """
    if G.cells.volumes is None:
        raise ValueError(
            "Grid geometry not computed. Call compute_geometry(G) first."
        )
    pv = rock.poro * G.cells.volumes
    if rock.ntg is not None:
        pv = pv * rock.ntg
    return pv


def perm_tensor(rock, dim):
    """Expand permeability to full tensor format.

    Port of MRST's permTensor. Expands compact symmetric storage
    to full row-major tensor form.

    Parameters
    ----------
    rock : Rock
        Rock structure with perm field.
    dim : int
        Number of spatial dimensions (1, 2, or 3).

    Returns
    -------
    K : np.ndarray
        Full permeability tensor, shape (num_cells, dim*dim).
        For dim=2: columns are [K11, K12, K21, K22].
        For dim=3: columns are [K11,K12,K13, K21,K22,K23, K31,K32,K33].
    r : np.ndarray
        Row indices (0-based) for the tensor components.
    c : np.ndarray
        Column indices (0-based) for the tensor components.
    """
    nc, nk = rock.perm.shape

    if dim == 1:
        if nk != 1:
            raise ValueError(f"1D perm must have 1 column, got {nk}")
        K = rock.perm.copy()
        r = np.array([0])
        c = np.array([0])

    elif dim == 2:
        # Expand to 3-column symmetric storage [k11, k12, k22]
        z = np.zeros(nc)
        if nk == 1:
            K_sym = np.column_stack([rock.perm[:, 0], z, rock.perm[:, 0]])
        elif nk == 2:
            K_sym = np.column_stack([rock.perm[:, 0], z, rock.perm[:, 1]])
        elif nk == 3:
            K_sym = rock.perm.copy()
        else:
            raise ValueError(f"{nk}-column perm not supported in 2D")

        # Expand to full 4-column: [k11, k12, k21, k22]
        K = K_sym[:, [0, 1, 1, 2]]
        r = np.array([0, 0, 1, 1])
        c = np.array([0, 1, 0, 1])

    elif dim == 3:
        # Expand to 6-column symmetric storage [k11,k12,k13,k22,k23,k33]
        z = np.zeros(nc)
        if nk == 1:
            K_sym = np.column_stack([
                rock.perm[:, 0], z, z,
                rock.perm[:, 0], z, rock.perm[:, 0]
            ])
        elif nk == 3:
            K_sym = np.column_stack([
                rock.perm[:, 0], z, z,
                rock.perm[:, 1], z, rock.perm[:, 2]
            ])
        elif nk == 6:
            K_sym = rock.perm.copy()
        else:
            raise ValueError(f"{nk}-column perm not supported in 3D")

        # Expand to full 9-column: [k11,k12,k13, k12,k22,k23, k13,k23,k33]
        K = K_sym[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]]
        r = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        c = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    return K, r, c
