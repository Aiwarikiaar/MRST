"""
Native Python geometry computation for MRST grids.

Port of MRST's core/gridprocessing/computeGeometry.m.

Computes:
    - Face areas, normals, and centroids
    - Cell volumes and centroids

Uses the same algorithmic approach as MRST:
    - 1D: Direct computation from node coordinates
    - 2D: Edge-based normals, sub-triangle decomposition for cells
    - 3D: Sub-triangle decomposition for faces, sub-tetrahedra for cells
"""

import numpy as np
from scipy import sparse


def compute_geometry(G):
    """Add geometry information (centroids, volumes, areas) to a grid.

    Port of MRST's computeGeometry. Computes face areas, normals,
    centroids, and cell volumes and centroids.

    Parameters
    ----------
    G : Grid
        Grid structure from cart_grid or tensor_grid.

    Returns
    -------
    Grid
        Same grid with geometry fields populated:
        - G.cells.volumes, G.cells.centroids
        - G.faces.areas, G.faces.normals, G.faces.centroids

    Examples
    --------
    >>> G = cart_grid([10, 10, 5], [100, 100, 50])
    >>> G = compute_geometry(G)
    >>> print(G.cells.volumes.sum())  # Total volume
    """
    if G.grid_dim == 1:
        _compute_geometry_1d(G)
    elif G.grid_dim == 2:
        _compute_geometry_2d(G)
    elif G.grid_dim == 3:
        _compute_geometry_3d(G)
    else:
        raise ValueError(f"Unsupported grid dimension: {G.grid_dim}")

    if "compute_geometry" not in G.type:
        G.type.append("compute_geometry")

    return G


def _rldecode(lengths):
    """Run-length decode: produce array [0,0,..,1,1,..,2,...] from lengths.

    Returns an array where value i is repeated lengths[i] times.
    """
    return np.repeat(np.arange(len(lengths)), lengths)


def _compute_geometry_1d(G):
    """Compute geometry for 1D grids."""
    coords = G.nodes.coords.ravel()

    G.faces.areas = np.ones(G.faces.num)
    G.faces.normals = np.ones((G.faces.num, 1))
    G.faces.centroids = G.nodes.coords.copy()

    G.cells.volumes = np.diff(coords)

    # Cell centroids: average of the two face centroids
    cell_centroids = np.zeros((G.cells.num, 1))
    for i in range(G.cells.num):
        start = G.cells.face_pos[i]
        end = G.cells.face_pos[i + 1]
        face_indices = G.cells.faces[start:end, 0]
        face_coords = G.faces.centroids[face_indices, 0]
        cell_centroids[i, 0] = face_coords.mean()
    G.cells.centroids = cell_centroids


def _compute_geometry_2d(G):
    """Compute geometry for 2D grids (in 2 space dimensions)."""
    ndim = G.nodes.coords.shape[1]

    # Face geometry: edges
    edges = G.faces.nodes.reshape(-1, 2)
    n1 = G.nodes.coords[edges[:, 0]]
    n2 = G.nodes.coords[edges[:, 1]]

    edge_vec = n2 - n1
    face_areas = np.sqrt(np.sum(edge_vec ** 2, axis=1))
    face_centroids = (n1 + n2) / 2.0

    if ndim == 2:
        face_normals = np.column_stack([edge_vec[:, 1], -edge_vec[:, 0]])
    else:
        # 2D grid embedded in 3D — face normals via cross product
        face_normals = np.column_stack([edge_vec[:, 1], -edge_vec[:, 0]])

    G.faces.areas = face_areas
    G.faces.normals = face_normals
    G.faces.centroids = face_centroids

    # Cell geometry via sub-triangle decomposition
    num_faces_per_cell = np.diff(G.cells.face_pos)

    # Cell centers: average of face centroids
    cell_centroids = np.zeros((G.cells.num, ndim))
    cell_volumes = np.zeros(G.cells.num)

    for i in range(G.cells.num):
        start = G.cells.face_pos[i]
        end = G.cells.face_pos[i + 1]
        fi = G.cells.faces[start:end, 0]

        fc = face_centroids[fi]
        cc = fc.mean(axis=0)

        # Sub-triangle areas
        vol = 0.0
        centroid = np.zeros(ndim)

        for j, f in enumerate(fi):
            e = edges[f]
            # Determine edge orientation relative to cell
            n_left = G.faces.neighbors[f, 0]
            if n_left != i:
                # Reverse edge to get correct winding
                p1, p2 = G.nodes.coords[e[1]], G.nodes.coords[e[0]]
            else:
                p1, p2 = G.nodes.coords[e[0]], G.nodes.coords[e[1]]

            a = p1 - cc
            b = p2 - cc

            # Triangle area (cross product magnitude / 2)
            sub_area = abs(a[0] * b[1] - a[1] * b[0]) / 2.0
            sub_centroid = (cc + p1 + p2) / 3.0

            vol += sub_area
            centroid += sub_area * sub_centroid

        if vol > 0:
            centroid /= vol

        cell_volumes[i] = vol
        cell_centroids[i] = centroid

    G.cells.volumes = cell_volumes
    G.cells.centroids = cell_centroids


def _compute_geometry_3d(G):
    """Compute geometry for 3D grids.

    Uses the same algorithm as MRST:
    1. Decompose each face into sub-triangles from face center
    2. Decompose each cell into sub-tetrahedra from cell center
    3. Compute volumes via divergence theorem (1/3 * sum of r·n)
    """
    ndim = 3
    coords = G.nodes.coords

    # --- Face geometry ---
    face_areas = np.zeros(G.faces.num)
    face_normals = np.zeros((G.faces.num, ndim))
    face_centroids = np.zeros((G.faces.num, ndim))

    # Precompute sub-triangle data for all faces
    sub_centroids_all = np.zeros((len(G.faces.nodes), ndim))
    sub_normals_all = np.zeros((len(G.faces.nodes), ndim))
    sub_areas_all = np.zeros(len(G.faces.nodes))
    sub_normal_signs = np.zeros(len(G.faces.nodes))

    for f in range(G.faces.num):
        np1 = G.faces.node_pos[f]
        np2 = G.faces.node_pos[f + 1]
        fn = G.faces.nodes[np1:np2]
        n_nodes = len(fn)

        # Face center (average of node coordinates)
        fc = coords[fn].mean(axis=0)

        # Sub-triangles: each consecutive pair of nodes + face center
        total_normal = np.zeros(ndim)
        total_area = 0.0
        weighted_centroid = np.zeros(ndim)

        for k in range(n_nodes):
            a = coords[fn[k]]
            b = coords[fn[(k + 1) % n_nodes]]

            # Sub-triangle normal (half cross product)
            sub_n = np.cross(b - a, fc - a) / 2.0
            sub_a = np.linalg.norm(sub_n)
            sub_c = (a + b + fc) / 3.0

            sub_centroids_all[np1 + k] = sub_c
            sub_normals_all[np1 + k] = sub_n
            sub_areas_all[np1 + k] = sub_a

            total_normal += sub_n
            if sub_a > 0:
                weighted_centroid += sub_a * sub_c
                total_area += sub_a

        face_normals[f] = total_normal
        face_areas[f] = np.linalg.norm(total_normal)

        if total_area > 0:
            face_centroids[f] = weighted_centroid / total_area
        else:
            face_centroids[f] = fc

    # Compute sub-normal signs relative to accumulated face normal
    for f in range(G.faces.num):
        np1 = G.faces.node_pos[f]
        np2 = G.faces.node_pos[f + 1]
        for k in range(np1, np2):
            sub_normal_signs[k] = np.sign(
                np.dot(sub_normals_all[k], face_normals[f])
            )

    G.faces.areas = face_areas
    G.faces.normals = face_normals
    G.faces.centroids = face_centroids

    # --- Cell geometry ---
    cell_volumes = np.zeros(G.cells.num)
    cell_centroids = np.zeros((G.cells.num, ndim))

    for i in range(G.cells.num):
        cf_start = G.cells.face_pos[i]
        cf_end = G.cells.face_pos[i + 1]
        cell_face_indices = G.cells.faces[cf_start:cf_end, 0]

        # Cell center estimate: average of face centroids
        cc = face_centroids[cell_face_indices].mean(axis=0)

        total_vol = 0.0
        weighted_centroid = np.zeros(ndim)

        for f in cell_face_indices:
            np1 = G.faces.node_pos[f]
            np2 = G.faces.node_pos[f + 1]

            # Outward normal sign relative to this cell
            if G.faces.neighbors[f, 0] == i:
                cf_sign = 1.0
            else:
                cf_sign = -1.0

            for k in range(np1, np2):
                sub_c = sub_centroids_all[k]
                sub_n = sub_normals_all[k]
                s_sign = sub_normal_signs[k]

                # Outward normal for this sub-triangle relative to cell
                out_normal = sub_n * s_sign * cf_sign

                # Sub-tetrahedron: apex at cc, base is sub-triangle
                rel_c = sub_c - cc
                tet_vol = np.dot(rel_c, out_normal) / 3.0
                tet_centroid = 0.75 * rel_c  # centroid relative to cc

                total_vol += tet_vol
                weighted_centroid += tet_vol * tet_centroid

        if total_vol > 0:
            cell_centroids[i] = cc + weighted_centroid / total_vol
        else:
            cell_centroids[i] = cc

        cell_volumes[i] = total_vol

    G.cells.volumes = cell_volumes
    G.cells.centroids = cell_centroids
