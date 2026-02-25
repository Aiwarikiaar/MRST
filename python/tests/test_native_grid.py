"""Tests for native Python grid implementation.

Validates grid topology, geometry, and consistency against known values.
For Cartesian grids, volumes, centroids, and areas have analytical solutions.
"""

import numpy as np
import pytest
from mrst.native.grid import Grid, cart_grid, tensor_grid
from mrst.native.geometry import compute_geometry


# --- Grid construction tests ---

class TestCartGrid1D:
    def test_basic_counts(self):
        G = cart_grid([5])
        assert G.cells.num == 5
        assert G.nodes.num == 6
        assert G.faces.num == 6
        assert G.grid_dim == 1

    def test_coords(self):
        G = cart_grid([4], [10.0])
        coords = G.nodes.coords.ravel()
        np.testing.assert_allclose(coords, [0, 2.5, 5, 7.5, 10])

    def test_cart_dims(self):
        G = cart_grid([5])
        np.testing.assert_array_equal(G.cart_dims, [5])


class TestCartGrid2D:
    def test_basic_counts(self):
        G = cart_grid([3, 4])
        assert G.cells.num == 12
        assert G.nodes.num == 20  # (3+1) * (4+1)
        assert G.grid_dim == 2

    def test_face_count(self):
        G = cart_grid([3, 4])
        # X-faces: (3+1)*4 = 16, Y-faces: 3*(4+1) = 15
        assert G.faces.num == 31

    def test_index_map(self):
        G = cart_grid([3, 4])
        np.testing.assert_array_equal(G.cells.index_map, np.arange(12))

    def test_boundary_faces(self):
        G = cart_grid([3, 4])
        bf = G.boundary_faces()
        # Boundary: 2*(3+4) = 14 faces
        assert len(bf) == 14

    def test_internal_faces(self):
        G = cart_grid([3, 4])
        ifaces = G.internal_faces()
        # Internal: 3*3 + 2*4 = 17 faces (inner x-faces + inner y-faces)
        assert len(ifaces) == 17

    def test_neighbors_valid(self):
        G = cart_grid([3, 4])
        n = G.faces.neighbors
        # No face should have both neighbors = -1
        assert np.all(np.any(n >= 0, axis=1))
        # All cell indices should be in range
        valid = n[n >= 0]
        assert np.all(valid < G.cells.num)


class TestCartGrid3D:
    def test_basic_counts(self):
        G = cart_grid([2, 3, 4])
        assert G.cells.num == 24
        assert G.nodes.num == 60  # 3*4*5
        assert G.grid_dim == 3

    def test_face_count(self):
        G = cart_grid([2, 3, 4])
        # X: 3*3*4=36, Y: 2*4*4=32, Z: 2*3*5=30
        assert G.faces.num == 98

    def test_faces_per_cell(self):
        G = cart_grid([2, 3, 4])
        # Each cell in 3D has exactly 6 faces
        faces_per_cell = np.diff(G.cells.face_pos)
        np.testing.assert_array_equal(faces_per_cell, 6)

    def test_nodes_per_face(self):
        G = cart_grid([2, 3, 4])
        # Each face in 3D has exactly 4 nodes
        nodes_per_face = np.diff(G.faces.node_pos)
        np.testing.assert_array_equal(nodes_per_face, 4)

    def test_boundary_faces(self):
        G = cart_grid([2, 3, 4])
        bf = G.boundary_faces()
        # Boundary: 2*(2*3 + 2*4 + 3*4) = 2*(6+8+12) = 52
        assert len(bf) == 52

    def test_physical_dimensions(self):
        G = cart_grid([5, 10, 3], [100, 200, 30])
        coords = G.nodes.coords
        np.testing.assert_allclose(coords[:, 0].min(), 0)
        np.testing.assert_allclose(coords[:, 0].max(), 100)
        np.testing.assert_allclose(coords[:, 1].min(), 0)
        np.testing.assert_allclose(coords[:, 1].max(), 200)
        np.testing.assert_allclose(coords[:, 2].min(), 0)
        np.testing.assert_allclose(coords[:, 2].max(), 30)

    def test_neighbors_symmetry(self):
        """If cell A is left neighbor of face f, then f appears in A's faces."""
        G = cart_grid([3, 3, 3])
        for f in range(G.faces.num):
            for side in range(2):
                c = G.faces.neighbors[f, side]
                if c >= 0:
                    cf = G.cell_faces(c)
                    assert f in cf[:, 0], \
                        f"Face {f} claims cell {c} as neighbor, but cell doesn't list face"


class TestTensorGrid:
    def test_nonuniform_spacing(self):
        x = [0, 1, 3, 6, 10]
        y = [0, 2, 5]
        G = tensor_grid(x, y)
        assert G.cells.num == 8  # 4 * 2
        np.testing.assert_allclose(G.nodes.coords[0], [0, 0])

    def test_3d_tensor(self):
        x = [0, 1, 3]
        y = [0, 2]
        z = [0, 5, 10]
        G = tensor_grid(x, y, z)
        assert G.cells.num == 4  # 2 * 1 * 2

    def test_depthz(self):
        x = [0, 1, 2]
        y = [0, 1, 2]
        z = [0, 1]
        depthz = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        G = tensor_grid(x, y, z, depthz=depthz)
        # All z-coordinates should be shifted by 10
        assert G.nodes.coords[:, 2].min() == 10.0
        assert G.nodes.coords[:, 2].max() == 11.0


# --- Geometry tests ---

class TestGeometry1D:
    def test_volumes(self):
        G = cart_grid([5], [10.0])
        G = compute_geometry(G)
        np.testing.assert_allclose(G.cells.volumes, 2.0)
        np.testing.assert_allclose(G.cells.volumes.sum(), 10.0)

    def test_centroids(self):
        G = cart_grid([4], [8.0])
        G = compute_geometry(G)
        expected = np.array([[1.0], [3.0], [5.0], [7.0]])
        np.testing.assert_allclose(G.cells.centroids, expected)


class TestGeometry2D:
    def test_uniform_volumes(self):
        G = cart_grid([4, 5], [8.0, 10.0])
        G = compute_geometry(G)
        # Each cell: 2.0 * 2.0 = 4.0
        np.testing.assert_allclose(G.cells.volumes, 4.0, atol=1e-12)
        np.testing.assert_allclose(G.cells.volumes.sum(), 80.0, atol=1e-10)

    def test_face_areas(self):
        G = cart_grid([3, 4], [9.0, 12.0])
        G = compute_geometry(G)
        # All face areas should be 3.0 (x-faces) or 3.0 (y-faces)
        assert np.all(G.faces.areas > 0)

    def test_centroids_in_cells(self):
        G = cart_grid([5, 5], [10.0, 10.0])
        G = compute_geometry(G)
        # All centroids should be inside [0, 10] x [0, 10]
        assert np.all(G.cells.centroids[:, 0] >= 0)
        assert np.all(G.cells.centroids[:, 0] <= 10)
        assert np.all(G.cells.centroids[:, 1] >= 0)
        assert np.all(G.cells.centroids[:, 1] <= 10)

    def test_nonuniform_volumes(self):
        G = tensor_grid([0, 1, 3], [0, 2, 5])
        G = compute_geometry(G)
        # Cell (0,0): 1*2=2, Cell (1,0): 2*2=4
        # Cell (0,1): 1*3=3, Cell (1,1): 2*3=6
        expected = np.array([2, 4, 3, 6], dtype=float)
        np.testing.assert_allclose(sorted(G.cells.volumes), sorted(expected), atol=1e-12)


class TestGeometry3D:
    def test_uniform_volumes(self):
        G = cart_grid([3, 4, 5], [30, 40, 50])
        G = compute_geometry(G)
        # Each cell: 10 * 10 * 10 = 1000
        np.testing.assert_allclose(G.cells.volumes, 1000.0, atol=1e-8)
        np.testing.assert_allclose(G.cells.volumes.sum(), 60000.0, atol=1e-4)

    def test_total_volume(self):
        G = cart_grid([2, 3, 4], [100, 200, 300])
        G = compute_geometry(G)
        np.testing.assert_allclose(G.cells.volumes.sum(), 6e6, rtol=1e-10)

    def test_face_areas_positive(self):
        G = cart_grid([2, 2, 2], [10, 10, 10])
        G = compute_geometry(G)
        assert np.all(G.faces.areas > 0)

    def test_face_normals_magnitude(self):
        """Face normal magnitudes should equal face areas."""
        G = cart_grid([2, 2, 2], [10, 10, 10])
        G = compute_geometry(G)
        normal_magnitudes = np.sqrt(np.sum(G.faces.normals ** 2, axis=1))
        np.testing.assert_allclose(normal_magnitudes, G.faces.areas, atol=1e-10)

    def test_centroids_in_domain(self):
        G = cart_grid([5, 5, 5], [100, 200, 300])
        G = compute_geometry(G)
        for d, mx in enumerate([100, 200, 300]):
            assert np.all(G.cells.centroids[:, d] >= 0)
            assert np.all(G.cells.centroids[:, d] <= mx)

    def test_cell_volumes_positive(self):
        G = cart_grid([4, 3, 2], [40, 30, 20])
        G = compute_geometry(G)
        assert np.all(G.cells.volumes > 0)

    def test_nonuniform_3d_total_volume(self):
        G = tensor_grid([0, 1, 5], [0, 3], [0, 2, 7])
        G = compute_geometry(G)
        # Total: 5 * 3 * 7 = 105
        np.testing.assert_allclose(G.cells.volumes.sum(), 105.0, atol=1e-10)

    def test_unit_cube(self):
        """Single cell unit cube."""
        G = cart_grid([1, 1, 1])
        G = compute_geometry(G)
        np.testing.assert_allclose(G.cells.volumes, [1.0], atol=1e-14)
        np.testing.assert_allclose(G.cells.centroids, [[0.5, 0.5, 0.5]], atol=1e-14)


# --- Grid utility tests ---

class TestGridUtilities:
    def test_cell_faces_method(self):
        G = cart_grid([2, 2])
        cf = G.cell_faces(0)
        assert len(cf) == 4  # 2D cell has 4 faces

    def test_face_nodes_method(self):
        G = cart_grid([2, 2, 2])
        fn = G.face_nodes(0)
        assert len(fn) == 4  # 3D face has 4 nodes

    def test_type_tracking(self):
        G = cart_grid([3, 3])
        assert "tensor_grid" in G.type
        assert "cart_grid" in G.type
        G = compute_geometry(G)
        assert "compute_geometry" in G.type
