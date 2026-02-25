"""Tests for native Python rock property implementation.

Validates Rock creation, pore volume computation, and permeability
tensor expansion against known analytical values.
"""

import numpy as np
import pytest
from mrst.native.grid import cart_grid
from mrst.native.geometry import compute_geometry
from mrst.native.rock import Rock, make_rock, pore_volume, perm_tensor


# --- make_rock tests ---

class TestMakeRock:
    def setup_method(self):
        self.G3 = cart_grid([3, 4, 5])  # 60 cells, 3D
        self.G2 = cart_grid([3, 4])     # 12 cells, 2D

    def test_scalar_perm_and_poro(self):
        rock = make_rock(self.G3, 1e-13, 0.2)
        assert rock.perm.shape == (60, 1)
        assert rock.poro.shape == (60,)
        np.testing.assert_allclose(rock.perm, 1e-13)
        np.testing.assert_allclose(rock.poro, 0.2)
        assert rock.ntg is None

    def test_vector_poro(self):
        poro = np.linspace(0.1, 0.3, 60)
        rock = make_rock(self.G3, 1e-13, poro)
        np.testing.assert_allclose(rock.poro, poro)

    def test_per_cell_isotropic_perm(self):
        perm = np.random.rand(60) * 1e-13
        rock = make_rock(self.G3, perm, 0.2)
        assert rock.perm.shape == (60, 1)
        np.testing.assert_allclose(rock.perm[:, 0], perm)

    def test_diagonal_perm_3d(self):
        # 3 columns for diagonal tensor in 3D
        perm = np.column_stack([
            np.full(60, 1e-13),
            np.full(60, 1e-13),
            np.full(60, 1e-14),
        ])
        rock = make_rock(self.G3, perm, 0.2)
        assert rock.perm.shape == (60, 3)

    def test_full_tensor_perm_3d(self):
        # 6 columns for full symmetric tensor in 3D
        perm = np.column_stack([
            np.full(60, 1e-13), np.zeros(60), np.zeros(60),
            np.full(60, 1e-13), np.zeros(60), np.full(60, 1e-14),
        ])
        rock = make_rock(self.G3, perm, 0.2)
        assert rock.perm.shape == (60, 6)

    def test_diagonal_perm_2d(self):
        perm = np.column_stack([np.full(12, 1e-13), np.full(12, 1e-14)])
        rock = make_rock(self.G2, perm, 0.2)
        assert rock.perm.shape == (12, 2)

    def test_full_tensor_perm_2d(self):
        perm = np.column_stack([
            np.full(12, 1e-13), np.full(12, 1e-15), np.full(12, 1e-14)
        ])
        rock = make_rock(self.G2, perm, 0.2)
        assert rock.perm.shape == (12, 3)

    def test_row_vector_broadcast(self):
        # Single row with 3 columns broadcast to all cells in 3D
        rock = make_rock(self.G3, [1e-13, 2e-13, 3e-14], 0.2)
        assert rock.perm.shape == (60, 3)
        np.testing.assert_allclose(rock.perm[0], [1e-13, 2e-13, 3e-14])
        np.testing.assert_allclose(rock.perm[59], [1e-13, 2e-13, 3e-14])

    def test_ntg(self):
        rock = make_rock(self.G3, 1e-13, 0.2, ntg=0.8)
        assert rock.ntg is not None
        assert rock.ntg.shape == (60,)
        np.testing.assert_allclose(rock.ntg, 0.8)

    def test_ntg_vector(self):
        ntg = np.linspace(0.5, 1.0, 60)
        rock = make_rock(self.G3, 1e-13, 0.2, ntg=ntg)
        np.testing.assert_allclose(rock.ntg, ntg)

    def test_invalid_perm_cols_raises(self):
        # 2 columns not valid in 3D (valid: 1, 3, 6)
        perm = np.column_stack([np.full(60, 1e-13), np.full(60, 1e-14)])
        with pytest.raises(ValueError, match="columns"):
            make_rock(self.G3, perm, 0.2)

    def test_wrong_row_count_raises(self):
        perm = np.full((30, 1), 1e-13)  # 30 != 60
        with pytest.raises(ValueError, match="rows"):
            make_rock(self.G3, perm, 0.2)

    def test_wrong_poro_count_raises(self):
        poro = np.full(30, 0.2)  # 30 != 60
        with pytest.raises(ValueError, match="poro"):
            make_rock(self.G3, 1e-13, poro)

    def test_zero_porosity_warns(self):
        poro = np.full(60, 0.2)
        poro[0] = 0.0
        with pytest.warns(UserWarning, match="Zero or negative porosity"):
            make_rock(self.G3, 1e-13, poro)


# --- pore_volume tests ---

class TestPoreVolume:
    def test_uniform_grid(self):
        G = compute_geometry(cart_grid([10, 10, 5], [1000, 1000, 50]))
        rock = make_rock(G, 1e-13, 0.2)
        pv = pore_volume(G, rock)
        assert pv.shape == (500,)
        # Total PV = 0.2 * 1000 * 1000 * 50 = 10,000,000
        np.testing.assert_allclose(pv.sum(), 10_000_000.0)

    def test_with_ntg(self):
        G = compute_geometry(cart_grid([10, 10, 5], [1000, 1000, 50]))
        rock = make_rock(G, 1e-13, 0.2, ntg=0.5)
        pv = pore_volume(G, rock)
        # Total PV = 0.2 * 0.5 * 1000 * 1000 * 50 = 5,000,000
        np.testing.assert_allclose(pv.sum(), 5_000_000.0)

    def test_per_cell_poro(self):
        G = compute_geometry(cart_grid([4], [4.0]))
        # 4 cells, each of length 1.0
        poro = np.array([0.1, 0.2, 0.3, 0.4])
        rock = make_rock(G, 1e-13, poro)
        pv = pore_volume(G, rock)
        np.testing.assert_allclose(pv, poro * 1.0)

    def test_no_geometry_raises(self):
        G = cart_grid([5, 5])  # geometry not computed
        rock = make_rock(G, 1e-13, 0.2)
        with pytest.raises(ValueError, match="geometry"):
            pore_volume(G, rock)

    def test_total_pore_volume_2d(self):
        G = compute_geometry(cart_grid([5, 5], [10, 10]))
        rock = make_rock(G, 1e-13, 0.3)
        pv = pore_volume(G, rock)
        # Total PV = 0.3 * 10 * 10 = 30
        np.testing.assert_allclose(pv.sum(), 30.0)


# --- perm_tensor tests ---

class TestPermTensor:
    def test_isotropic_1d(self):
        rock = Rock(perm=np.array([[1e-13]]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 1)
        assert K.shape == (1, 1)
        np.testing.assert_allclose(K[0, 0], 1e-13)

    def test_isotropic_2d(self):
        k = 1e-13
        rock = Rock(perm=np.array([[k]]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 2)
        assert K.shape == (1, 4)
        # [k, 0, 0, k]
        np.testing.assert_allclose(K[0], [k, 0, 0, k])

    def test_diagonal_2d(self):
        rock = Rock(perm=np.array([[1e-13, 2e-13]]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 2)
        # [k1, 0, 0, k2]
        np.testing.assert_allclose(K[0], [1e-13, 0, 0, 2e-13])

    def test_full_symmetric_2d(self):
        # [k11, k12, k22]
        rock = Rock(
            perm=np.array([[1e-13, 5e-15, 2e-13]]),
            poro=np.array([0.2])
        )
        K, r, c = perm_tensor(rock, 2)
        # [k11, k12, k12, k22]
        np.testing.assert_allclose(K[0], [1e-13, 5e-15, 5e-15, 2e-13])

    def test_isotropic_3d(self):
        k = 1e-13
        rock = Rock(perm=np.array([[k]]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 3)
        assert K.shape == (1, 9)
        expected = [k, 0, 0, 0, k, 0, 0, 0, k]
        np.testing.assert_allclose(K[0], expected)

    def test_diagonal_3d(self):
        rock = Rock(
            perm=np.array([[1e-13, 2e-13, 3e-14]]),
            poro=np.array([0.2])
        )
        K, r, c = perm_tensor(rock, 3)
        expected = [1e-13, 0, 0, 0, 2e-13, 0, 0, 0, 3e-14]
        np.testing.assert_allclose(K[0], expected)

    def test_full_symmetric_3d(self):
        # [k11, k12, k13, k22, k23, k33]
        vals = [1e-13, 1e-15, 2e-15, 2e-13, 3e-15, 3e-14]
        rock = Rock(perm=np.array([vals]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 3)
        # Full: [k11,k12,k13, k12,k22,k23, k13,k23,k33]
        expected = [1e-13, 1e-15, 2e-15, 1e-15, 2e-13, 3e-15, 2e-15, 3e-15, 3e-14]
        np.testing.assert_allclose(K[0], expected)

    def test_symmetry_3d(self):
        """Expanded tensor must be symmetric: K[i,j] == K[j,i]."""
        vals = [1e-13, 1e-15, 2e-15, 2e-13, 3e-15, 3e-14]
        rock = Rock(perm=np.array([vals]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 3)
        K_matrix = K[0].reshape(3, 3)
        np.testing.assert_allclose(K_matrix, K_matrix.T)

    def test_symmetry_2d(self):
        vals = [1e-13, 5e-15, 2e-13]
        rock = Rock(perm=np.array([vals]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 2)
        K_matrix = K[0].reshape(2, 2)
        np.testing.assert_allclose(K_matrix, K_matrix.T)

    def test_index_arrays_2d(self):
        rock = Rock(perm=np.array([[1e-13]]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 2)
        np.testing.assert_array_equal(r, [0, 0, 1, 1])
        np.testing.assert_array_equal(c, [0, 1, 0, 1])

    def test_index_arrays_3d(self):
        rock = Rock(perm=np.array([[1e-13]]), poro=np.array([0.2]))
        K, r, c = perm_tensor(rock, 3)
        np.testing.assert_array_equal(r, [0, 0, 0, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(c, [0, 1, 2, 0, 1, 2, 0, 1, 2])

    def test_multiple_cells(self):
        perm = np.array([[1e-13], [2e-13], [3e-13]])
        rock = Rock(perm=perm, poro=np.array([0.2, 0.2, 0.2]))
        K, r, c = perm_tensor(rock, 3)
        assert K.shape == (3, 9)
        # Each cell should be isotropic with its own k
        for i, k in enumerate([1e-13, 2e-13, 3e-13]):
            K_mat = K[i].reshape(3, 3)
            expected = np.diag([k, k, k])
            np.testing.assert_allclose(K_mat, expected)
