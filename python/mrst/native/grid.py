"""
Native Python grid data structures and construction functions.

Port of MRST's core/gridprocessing/tensorGrid.m and cartGrid.m.

IMPORTANT: This module uses 0-based indexing (Python convention),
unlike MRST which uses 1-based indexing (MATLAB convention).
All node, face, and cell indices are 0-based.

The Grid class mirrors MRST's grid_structure with the following layout:

    G.cells.num          - Number of cells
    G.cells.face_pos     - Indirection into cells.faces (0-based CSR format)
    G.cells.faces        - (N, 2) array: [face_index, tag]
    G.cells.index_map    - Maps internal to external cell numbers
    G.cells.volumes      - Cell volumes (after compute_geometry)
    G.cells.centroids    - Cell centroids (after compute_geometry)

    G.faces.num          - Number of faces
    G.faces.node_pos     - Indirection into faces.nodes (0-based CSR format)
    G.faces.nodes        - Node indices for each face
    G.faces.neighbors    - (num_faces, 2) neighbor cells (−1 for boundary)
    G.faces.tag          - Face tags
    G.faces.areas        - Face areas (after compute_geometry)
    G.faces.normals      - Face normals (after compute_geometry)
    G.faces.centroids    - Face centroids (after compute_geometry)

    G.nodes.num          - Number of nodes
    G.nodes.coords       - (num_nodes, dim) node coordinates

    G.grid_dim           - Topological dimension (1, 2, or 3)
    G.cart_dims          - Cartesian dimensions (if applicable)
    G.type               - List of constructor names
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _Cells:
    """Cell properties sub-structure."""
    num: int
    face_pos: np.ndarray  # (num+1,) CSR indirection into faces
    faces: np.ndarray     # (M, 2) array: [face_index, tag]
    index_map: np.ndarray  # (num,) internal-to-external map
    volumes: Optional[np.ndarray] = None     # (num,) after compute_geometry
    centroids: Optional[np.ndarray] = None   # (num, dim) after compute_geometry


@dataclass
class _Faces:
    """Face properties sub-structure."""
    num: int
    node_pos: np.ndarray   # (num+1,) CSR indirection into nodes
    nodes: np.ndarray      # (K,) node indices for each face
    neighbors: np.ndarray  # (num, 2) neighbor cells, -1 for boundary
    tag: np.ndarray        # (num,) face tags
    areas: Optional[np.ndarray] = None       # (num,) after compute_geometry
    normals: Optional[np.ndarray] = None     # (num, dim) after compute_geometry
    centroids: Optional[np.ndarray] = None   # (num, dim) after compute_geometry


@dataclass
class _Nodes:
    """Node (vertex) properties sub-structure."""
    num: int
    coords: np.ndarray  # (num, dim) physical coordinates


@dataclass
class Grid:
    """Unstructured grid representation.

    Mirrors MRST's grid_structure. All indices are 0-based.
    Boundary faces have neighbor value -1 (instead of MRST's 0).
    """
    cells: _Cells
    faces: _Faces
    nodes: _Nodes
    grid_dim: int
    type: list = field(default_factory=list)
    cart_dims: Optional[np.ndarray] = None

    @property
    def num_cells(self):
        return self.cells.num

    @property
    def num_faces(self):
        return self.faces.num

    @property
    def num_nodes(self):
        return self.nodes.num

    @property
    def dim(self):
        return self.nodes.coords.shape[1]

    def boundary_faces(self):
        """Return indices of boundary faces (faces with only one neighbor)."""
        n = self.faces.neighbors
        return np.where((n[:, 0] < 0) | (n[:, 1] < 0))[0]

    def internal_faces(self):
        """Return indices of internal faces (shared by two cells)."""
        n = self.faces.neighbors
        return np.where((n[:, 0] >= 0) & (n[:, 1] >= 0))[0]

    def cell_faces(self, cell_idx):
        """Return face indices for a given cell.

        Parameters
        ----------
        cell_idx : int
            Cell index (0-based).

        Returns
        -------
        numpy.ndarray
            Array of (face_index, tag) pairs for this cell.
        """
        start = self.cells.face_pos[cell_idx]
        end = self.cells.face_pos[cell_idx + 1]
        return self.cells.faces[start:end]

    def face_nodes(self, face_idx):
        """Return node indices for a given face.

        Parameters
        ----------
        face_idx : int
            Face index (0-based).

        Returns
        -------
        numpy.ndarray
            Node indices forming this face.
        """
        start = self.faces.node_pos[face_idx]
        end = self.faces.node_pos[face_idx + 1]
        return self.faces.nodes[start:end]


def _ensure_monotone(x, name="x"):
    """Ensure coordinate vector is strictly increasing, truncating if needed."""
    x = np.asarray(x, dtype=np.float64).ravel()
    dx = np.diff(x)
    if not np.all(dx > 0):
        import warnings
        warnings.warn(f"Nonmonotone {name}-data, truncating")
        while not np.all(dx > 0):
            keep = np.concatenate([[True], dx > 0])
            x = x[keep]
            dx = np.diff(x)
    return x


def tensor_grid(x, y=None, z=None, depthz=None):
    """Construct a tensor-product grid with variable cell sizes.

    Port of MRST's tensorGrid. Creates 1D, 2D, or 3D grids depending
    on which coordinate vectors are provided.

    Parameters
    ----------
    x : array_like
        Cell vertices along x-direction.
    y : array_like, optional
        Cell vertices along y-direction (for 2D/3D).
    z : array_like, optional
        Cell vertices along z-direction (for 3D).
    depthz : array_like, optional
        Depth at upper nodes, shape (len(x), len(y)). Only for 3D.

    Returns
    -------
    Grid
        Grid structure (without geometry - call compute_geometry to add it).

    Examples
    --------
    >>> G = tensor_grid([0, 1, 2, 3])          # 1D, 3 cells
    >>> G = tensor_grid([0, 1, 2], [0, 1, 2])  # 2D, 4 cells
    >>> G = tensor_grid([0, 1], [0, 1], [0, 1])  # 3D, 1 cell
    """
    x = _ensure_monotone(x, "x")

    if y is None and z is None:
        return _tensor_grid_1d(x)
    elif z is None:
        y = _ensure_monotone(y, "y")
        return _tensor_grid_2d(x, y)
    else:
        y = _ensure_monotone(y, "y")
        z = _ensure_monotone(z, "z")
        return _tensor_grid_3d(x, y, z, depthz)


def cart_grid(cell_dim, phys_dim=None):
    """Construct a Cartesian grid.

    Port of MRST's cartGrid. Creates uniform grids by generating
    evenly spaced coordinate vectors and calling tensor_grid.

    Parameters
    ----------
    cell_dim : list of int
        Number of cells in each direction, e.g., [10, 10, 5].
    phys_dim : list of float, optional
        Physical dimensions in meters. Defaults to cell_dim.

    Returns
    -------
    Grid
        Grid structure (without geometry).

    Examples
    --------
    >>> G = cart_grid([10, 10, 5])                    # Unit cube cells
    >>> G = cart_grid([10, 10, 5], [1000, 1000, 50])  # Physical dims
    """
    cell_dim = np.asarray(cell_dim, dtype=int)
    if phys_dim is None:
        phys_dim = cell_dim.astype(float)
    else:
        phys_dim = np.asarray(phys_dim, dtype=float)

    if not np.all(cell_dim > 0):
        raise ValueError("cell_dim must be positive")

    ndim = len(cell_dim)
    coords = [np.linspace(0, phys_dim[i], cell_dim[i] + 1) for i in range(ndim)]

    if ndim == 1:
        G = tensor_grid(coords[0])
    elif ndim == 2:
        G = tensor_grid(coords[0], coords[1])
    elif ndim == 3:
        G = tensor_grid(coords[0], coords[1], coords[2])
    else:
        raise ValueError(f"Cannot create grid with {ndim} dimensions")

    G.type.append("cart_grid")
    return G


def _tensor_grid_1d(x):
    """Construct a 1D tensor grid."""
    sx = len(x) - 1
    num_cells = sx
    num_nodes = sx + 1
    num_faces = num_nodes

    # Coordinates
    coords = x.reshape(-1, 1)

    # Neighbors: face i connects cell i-1 and cell i
    # Boundary: face 0 has left neighbor -1, face sx has right neighbor -1
    neighbors = np.zeros((num_faces, 2), dtype=np.int64)
    neighbors[0, 0] = -1
    neighbors[0, 1] = 0
    for i in range(1, num_faces - 1):
        neighbors[i, 0] = i - 1
        neighbors[i, 1] = i
    neighbors[-1, 0] = sx - 1
    neighbors[-1, 1] = -1

    # Cell faces: each cell has 2 faces (left and right)
    # Cell i has faces i (left, tag=0) and i+1 (right, tag=1)
    cell_faces_list = []
    for i in range(num_cells):
        cell_faces_list.append([i, 0])      # W (left)
        cell_faces_list.append([i + 1, 1])   # E (right)
    cell_faces = np.array(cell_faces_list, dtype=np.int64)

    face_pos = np.arange(0, 2 * num_cells + 1, 2, dtype=np.int64)

    # Face nodes: each face is a single node
    face_nodes = np.arange(num_faces, dtype=np.int64)
    node_pos = np.arange(num_faces + 1, dtype=np.int64)

    cells = _Cells(
        num=num_cells,
        face_pos=face_pos,
        faces=cell_faces,
        index_map=np.arange(num_cells, dtype=np.int64),
    )
    faces = _Faces(
        num=num_faces,
        node_pos=node_pos,
        nodes=face_nodes,
        neighbors=neighbors,
        tag=np.zeros(num_faces, dtype=np.int64),
    )
    nodes = _Nodes(num=num_nodes, coords=coords)

    G = Grid(
        cells=cells, faces=faces, nodes=nodes,
        grid_dim=1, type=["tensor_grid"],
        cart_dims=np.array([sx]),
    )
    return G


def _tensor_grid_2d(x, y):
    """Construct a 2D tensor grid."""
    sx = len(x) - 1
    sy = len(y) - 1

    num_cells = sx * sy
    num_nodes = (sx + 1) * (sy + 1)
    num_fx = (sx + 1) * sy     # faces parallel to y-axis
    num_fy = sx * (sy + 1)     # faces parallel to x-axis
    num_faces = num_fx + num_fy

    # Node coordinates
    xc, yc = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([xc.ravel(), yc.ravel()])

    # Node index matrix (sx+1, sy+1)
    N = np.arange(num_nodes).reshape(sx + 1, sy + 1)

    # X-faces: each has 2 nodes (bottom, top along y)
    nf1 = N[:sx + 1, :sy].ravel()
    nf2 = N[:sx + 1, 1:sy + 1].ravel()
    face_nodes_x = np.column_stack([nf1, nf2]).ravel()

    # Y-faces: nodes ordered so normal points in +x direction
    nf1 = N[:sx, :sy + 1].ravel()
    nf2 = N[1:sx + 1, :sy + 1].ravel()
    face_nodes_y = np.column_stack([nf2, nf1]).ravel()  # reversed for normal direction

    face_nodes = np.concatenate([face_nodes_x, face_nodes_y]).astype(np.int64)
    node_pos = np.arange(0, 2 * num_faces + 1, 2, dtype=np.int64)

    # Face index matrices
    FX = np.arange(num_fx).reshape(sx + 1, sy)
    FY = num_fx + np.arange(num_fy).reshape(sx, sy + 1)

    # Cell faces: W(0), S(2), E(1), N(3) for each cell
    C_idx = np.arange(num_cells).reshape(sx, sy)
    f1 = FX[:sx, :].ravel()      # W
    f2 = FX[1:sx + 1, :].ravel()  # E
    f3 = FY[:, :sy].ravel()       # S
    f4 = FY[:, 1:sy + 1].ravel()  # N

    # Interleave: W, S, E, N per cell (matching MRST's order)
    cell_faces = np.column_stack([
        np.column_stack([f1, f3, f2, f4]).ravel(),
        np.tile([0, 2, 1, 3], num_cells),
    ]).astype(np.int64)

    face_pos = np.arange(0, 4 * num_cells + 1, 4, dtype=np.int64)

    # Neighbors
    C = np.full((sx + 2, sy + 2), -1, dtype=np.int64)
    C[1:sx + 1, 1:sy + 1] = np.arange(num_cells).reshape(sx, sy)

    nx1 = C[:sx + 1, 1:sy + 1].ravel()
    nx2 = C[1:sx + 2, 1:sy + 1].ravel()
    ny1 = C[1:sx + 1, :sy + 1].ravel()
    ny2 = C[1:sx + 1, 1:sy + 2].ravel()

    neighbors = np.vstack([
        np.column_stack([nx1, nx2]),
        np.column_stack([ny1, ny2]),
    ]).astype(np.int64)

    cells = _Cells(
        num=num_cells,
        face_pos=face_pos,
        faces=cell_faces,
        index_map=np.arange(num_cells, dtype=np.int64),
    )
    faces = _Faces(
        num=num_faces,
        node_pos=node_pos,
        nodes=face_nodes,
        neighbors=neighbors,
        tag=np.zeros(num_faces, dtype=np.int64),
    )
    nodes = _Nodes(num=num_nodes, coords=coords)

    G = Grid(
        cells=cells, faces=faces, nodes=nodes,
        grid_dim=2, type=["tensor_grid"],
        cart_dims=np.array([sx, sy]),
    )
    return G


def _tensor_grid_3d(x, y, z, depthz=None):
    """Construct a 3D tensor grid."""
    sx = len(x) - 1
    sy = len(y) - 1
    sz = len(z) - 1

    num_cells = sx * sy * sz
    num_nodes = (sx + 1) * (sy + 1) * (sz + 1)
    num_fx = (sx + 1) * sy * sz       # faces || yz-plane
    num_fy = sx * (sy + 1) * sz       # faces || xz-plane
    num_fz = sx * sy * (sz + 1)       # faces || xy-plane
    num_faces = num_fx + num_fy + num_fz

    # Node coordinates
    xc, yc, zc = np.meshgrid(x, y, z, indexing="ij")
    if depthz is not None:
        depthz = np.asarray(depthz).reshape(sx + 1, sy + 1)
        zc = zc + depthz[:, :, np.newaxis]
    coords = np.column_stack([xc.ravel(), yc.ravel(), zc.ravel()])

    # Node index matrix (sx+1, sy+1, sz+1)
    N = np.arange(num_nodes).reshape(sx + 1, sy + 1, sz + 1)

    # X-faces: 4 nodes each (ordered for correct normal)
    nf1 = N[:sx + 1, :sy, :sz].ravel()
    nf2 = N[:sx + 1, 1:sy + 1, :sz].ravel()
    nf3 = N[:sx + 1, 1:sy + 1, 1:sz + 1].ravel()
    nf4 = N[:sx + 1, :sy, 1:sz + 1].ravel()
    face_nodes_x = np.column_stack([nf1, nf2, nf3, nf4]).ravel()

    # Y-faces
    nf1 = N[:sx, :sy + 1, :sz].ravel()
    nf2 = N[:sx, :sy + 1, 1:sz + 1].ravel()
    nf3 = N[1:sx + 1, :sy + 1, 1:sz + 1].ravel()
    nf4 = N[1:sx + 1, :sy + 1, :sz].ravel()
    face_nodes_y = np.column_stack([nf1, nf2, nf3, nf4]).ravel()

    # Z-faces
    nf1 = N[:sx, :sy, :sz + 1].ravel()
    nf2 = N[1:sx + 1, :sy, :sz + 1].ravel()
    nf3 = N[1:sx + 1, 1:sy + 1, :sz + 1].ravel()
    nf4 = N[:sx, 1:sy + 1, :sz + 1].ravel()
    face_nodes_z = np.column_stack([nf1, nf2, nf3, nf4]).ravel()

    face_nodes = np.concatenate([face_nodes_x, face_nodes_y, face_nodes_z]).astype(np.int64)
    node_pos = np.arange(0, 4 * num_faces + 1, 4, dtype=np.int64)

    # Face index matrices
    foffset = 0
    FX = (foffset + np.arange(num_fx)).reshape(sx + 1, sy, sz)
    foffset += num_fx
    FY = (foffset + np.arange(num_fy)).reshape(sx, sy + 1, sz)
    foffset += num_fy
    FZ = (foffset + np.arange(num_fz)).reshape(sx, sy, sz + 1)

    # Cell faces: W(0), E(1), S(2), N(3), T(4), B(5)
    f1 = FX[:sx, :, :].ravel()       # W
    f2 = FX[1:sx + 1, :, :].ravel()  # E
    f3 = FY[:, :sy, :].ravel()       # S
    f4 = FY[:, 1:sy + 1, :].ravel()  # N
    f5 = FZ[:, :, :sz].ravel()       # T
    f6 = FZ[:, :, 1:sz + 1].ravel()  # B

    cell_faces = np.column_stack([
        np.column_stack([f1, f2, f3, f4, f5, f6]).ravel(),
        np.tile([0, 1, 2, 3, 4, 5], num_cells),
    ]).astype(np.int64)

    face_pos = np.arange(0, 6 * num_cells + 1, 6, dtype=np.int64)

    # Neighbors
    C = np.full((sx + 2, sy + 2, sz + 2), -1, dtype=np.int64)
    C[1:sx + 1, 1:sy + 1, 1:sz + 1] = np.arange(num_cells).reshape(sx, sy, sz)

    nx1 = C[:sx + 1, 1:sy + 1, 1:sz + 1].ravel()
    nx2 = C[1:sx + 2, 1:sy + 1, 1:sz + 1].ravel()
    ny1 = C[1:sx + 1, :sy + 1, 1:sz + 1].ravel()
    ny2 = C[1:sx + 1, 1:sy + 2, 1:sz + 1].ravel()
    nz1 = C[1:sx + 1, 1:sy + 1, :sz + 1].ravel()
    nz2 = C[1:sx + 1, 1:sy + 1, 1:sz + 2].ravel()

    neighbors = np.vstack([
        np.column_stack([nx1, nx2]),
        np.column_stack([ny1, ny2]),
        np.column_stack([nz1, nz2]),
    ]).astype(np.int64)

    cells = _Cells(
        num=num_cells,
        face_pos=face_pos,
        faces=cell_faces,
        index_map=np.arange(num_cells, dtype=np.int64),
    )
    faces = _Faces(
        num=num_faces,
        node_pos=node_pos,
        nodes=face_nodes,
        neighbors=neighbors,
        tag=np.zeros(num_faces, dtype=np.int64),
    )
    nodes = _Nodes(num=num_nodes, coords=coords)

    G = Grid(
        cells=cells, faces=faces, nodes=nodes,
        grid_dim=3, type=["tensor_grid"],
        cart_dims=np.array([sx, sy, sz]),
    )
    return G
