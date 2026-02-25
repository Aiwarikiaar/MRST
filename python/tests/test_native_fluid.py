"""Tests for native Python fluid property implementation.

Validates SimpleFluid (incompressible) and ADFluid (compressible)
against known analytical values and derivative correctness.
"""

import numpy as np
import pytest
from mrst.native.fluid import (
    SimpleFluid, init_simple_fluid,
    ADFluid, init_ad_fluid,
)


# --- SimpleFluid tests ---

class TestSimpleFluid:
    def setup_method(self):
        self.fluid = init_simple_fluid(
            mu=[1e-3, 5e-3], rho=[1000, 700], n=[2, 2]
        )

    def test_creation(self):
        assert self.fluid.mu_w == 1e-3
        assert self.fluid.mu_o == 5e-3
        assert self.fluid.rho_w == 1000
        assert self.fluid.rho_o == 700
        assert self.fluid.n_w == 2
        assert self.fluid.n_o == 2

    def test_kr_water_at_endpoints(self):
        kr0, _ = self.fluid.kr_w(0.0)
        kr1, _ = self.fluid.kr_w(1.0)
        np.testing.assert_allclose(kr0, 0.0)
        np.testing.assert_allclose(kr1, 1.0)

    def test_kr_oil_at_endpoints(self):
        kr0, _ = self.fluid.kr_o(0.0)  # so = 0
        kr1, _ = self.fluid.kr_o(1.0)  # so = 1
        np.testing.assert_allclose(kr0, 0.0)
        np.testing.assert_allclose(kr1, 1.0)

    def test_kr_water_corey(self):
        # krW(0.5) = 0.5^2 = 0.25
        kr, _ = self.fluid.kr_w(0.5)
        np.testing.assert_allclose(kr, 0.25)

    def test_kr_oil_corey(self):
        # krO(0.5) = 0.5^2 = 0.25
        kr, _ = self.fluid.kr_o(0.5)
        np.testing.assert_allclose(kr, 0.25)

    def test_kr_derivative_water(self):
        # dkrW/dsw = 2 * sw  at sw=0.5 → 1.0
        _, dkr = self.fluid.kr_w(0.5)
        np.testing.assert_allclose(dkr, 1.0)

    def test_kr_derivative_oil(self):
        # dkrO/dso = 2 * so  at so=0.5 → 1.0
        _, dkr = self.fluid.kr_o(0.5)
        np.testing.assert_allclose(dkr, 1.0)

    def test_relperm_combined(self):
        sw = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        kr, dkr = self.fluid.relperm(sw)
        assert kr.shape == (5, 2)
        assert dkr.shape == (5, 4)
        # At sw=0: krW=0, krO=(1-0)^2=1
        np.testing.assert_allclose(kr[0], [0.0, 1.0])
        # At sw=1: krW=1, krO=(1-1)^2=0
        np.testing.assert_allclose(kr[4], [1.0, 0.0])
        # At sw=0.5: krW=0.25, krO=0.25
        np.testing.assert_allclose(kr[2], [0.25, 0.25])

    def test_relperm_derivative_shape(self):
        sw = np.array([0.3, 0.7])
        _, dkr = self.fluid.relperm(sw)
        assert dkr.shape == (2, 4)
        # Off-diagonal should be zero
        np.testing.assert_allclose(dkr[:, 1], 0.0)
        np.testing.assert_allclose(dkr[:, 2], 0.0)

    def test_relperm_derivative_chain_rule(self):
        """dkrO/dsw should be negative (oil kr decreases as sw increases)."""
        sw = np.array([0.5])
        _, dkr = self.fluid.relperm(sw)
        # dkrO/dsw = -n_o * (1-sw)^(n_o-1) = -2*0.5 = -1.0
        np.testing.assert_allclose(dkr[0, 3], -1.0)

    def test_linear_kr(self):
        """n=1 gives linear relperm."""
        fluid = init_simple_fluid(mu=[1e-3, 1e-3], rho=[1000, 1000], n=[1, 1])
        sw = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        kr, _ = fluid.relperm(sw)
        np.testing.assert_allclose(kr[:, 0], sw)
        np.testing.assert_allclose(kr[:, 1], 1.0 - sw)

    def test_incompressible_bw_bo(self):
        p = np.array([1e5, 1e6, 1e7])
        np.testing.assert_allclose(self.fluid.b_w(p), 1.0)
        np.testing.assert_allclose(self.fluid.b_o(p), 1.0)

    def test_properties(self):
        mu, rho = self.fluid.properties()
        np.testing.assert_allclose(mu, [1e-3, 5e-3])
        np.testing.assert_allclose(rho, [1000, 700])

    def test_vectorized(self):
        sw = np.linspace(0, 1, 100)
        kr, dkr = self.fluid.relperm(sw)
        assert kr.shape == (100, 2)
        # All kr values should be in [0, 1]
        assert np.all(kr >= -1e-15)
        assert np.all(kr <= 1.0 + 1e-15)

    def test_derivative_finite_difference(self):
        """Verify derivatives match finite differences."""
        sw = np.array([0.3, 0.5, 0.7])
        h = 1e-7
        kr_p, _ = self.fluid.kr_w(sw + h)
        kr_m, _ = self.fluid.kr_w(sw - h)
        dkr_fd = (kr_p - kr_m) / (2 * h)
        _, dkr = self.fluid.kr_w(sw)
        np.testing.assert_allclose(dkr, dkr_fd, atol=1e-6)


class TestSimpleFluidValidation:
    def test_wrong_mu_length(self):
        with pytest.raises(ValueError, match="mu"):
            init_simple_fluid(mu=[1e-3], rho=[1000, 700], n=[2, 2])

    def test_wrong_rho_length(self):
        with pytest.raises(ValueError, match="rho"):
            init_simple_fluid(mu=[1e-3, 5e-3], rho=[1000], n=[2, 2])

    def test_wrong_n_length(self):
        with pytest.raises(ValueError, match="n"):
            init_simple_fluid(mu=[1e-3, 5e-3], rho=[1000, 700], n=[2])


# --- ADFluid tests ---

class TestADFluidCreation:
    def test_two_phase_defaults(self):
        fluid = init_ad_fluid(phases='WO')
        assert fluid.num_phases == 2
        assert fluid.phases == 'WO'
        assert fluid.rho['W'] == 1.0
        assert fluid.rho['O'] == 1.0

    def test_three_phase(self):
        fluid = init_ad_fluid(
            phases='WOG',
            mu=[1e-3, 5e-3, 1e-4],
            rho=[1000, 700, 100],
            n=[2, 2, 2],
        )
        assert fluid.num_phases == 3
        assert fluid.mu['G'] == 1e-4
        assert fluid.rho['G'] == 100

    def test_invalid_phases(self):
        with pytest.raises(ValueError, match="subset"):
            init_ad_fluid(phases='WX')

    def test_duplicate_phases(self):
        with pytest.raises(ValueError, match="Duplicate"):
            init_ad_fluid(phases='WW')

    def test_smin_sum_too_large(self):
        with pytest.raises(ValueError, match="Sum"):
            init_ad_fluid(phases='WO', smin=[0.6, 0.5])

    def test_wrong_array_length(self):
        with pytest.raises(ValueError, match="mu"):
            init_ad_fluid(phases='WO', mu=[1e-3])


class TestADFluidFVF:
    def test_incompressible(self):
        fluid = init_ad_fluid(phases='WO')
        p = np.array([1e5, 1e6, 1e7])
        bw, dbw = fluid.bW(p)
        np.testing.assert_allclose(bw, 1.0)
        np.testing.assert_allclose(dbw, 0.0)

    def test_compressible(self):
        c_val = 1e-9  # 1/Pa
        p_ref = 1e7
        fluid = init_ad_fluid(
            phases='WO',
            c=[c_val, c_val],
            p_ref=p_ref,
        )
        p = np.array([p_ref, p_ref + 1e6, p_ref + 1e7])

        bw, dbw = fluid.bW(p)
        # b(p) = 1 * exp(c * (p - p_ref))
        expected_b = np.exp(c_val * (p - p_ref))
        np.testing.assert_allclose(bw, expected_b, rtol=1e-12)

        # db/dp = c * b
        expected_db = c_val * expected_b
        np.testing.assert_allclose(dbw, expected_db, rtol=1e-12)

    def test_compressible_at_pref(self):
        """At reference pressure, b should equal b_ref."""
        fluid = init_ad_fluid(
            phases='WO',
            b=[1.01, 0.99],
            c=[1e-9, 2e-9],
            p_ref=200e5,
        )
        p = np.array([200e5])
        bw, _ = fluid.bW(p)
        bo, _ = fluid.bO(p)
        np.testing.assert_allclose(bw, 1.01)
        np.testing.assert_allclose(bo, 0.99)

    def test_compressible_fvf_derivative_fd(self):
        """Verify FVF derivative with finite differences."""
        c_val = 1e-8
        fluid = init_ad_fluid(phases='WO', c=[c_val, c_val], p_ref=1e7)
        p = np.array([1.5e7, 2e7])
        h = 1.0  # 1 Pa
        bp, _ = fluid.bW(p + h)
        bm, _ = fluid.bW(p - h)
        db_fd = (bp - bm) / (2 * h)
        _, db = fluid.bW(p)
        np.testing.assert_allclose(db, db_fd, rtol=1e-8)


class TestADFluidViscosity:
    def test_constant_viscosity(self):
        fluid = init_ad_fluid(phases='WO', mu=[1e-3, 5e-3])
        p = np.array([1e5, 1e8])
        mu_w, dmu_w = fluid.muW(p)
        np.testing.assert_allclose(mu_w, 1e-3)
        np.testing.assert_allclose(dmu_w, 0.0)


class TestADFluidRelperm:
    def test_linear_kr(self):
        fluid = init_ad_fluid(phases='WO', n=[1, 1])
        sw = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        krw, dkrw = fluid.krW(sw)
        np.testing.assert_allclose(krw, sw)
        np.testing.assert_allclose(dkrw, 1.0)

    def test_quadratic_kr(self):
        fluid = init_ad_fluid(phases='WO', n=[2, 2])
        sw = np.array([0.5])
        krw, _ = fluid.krW(sw)
        np.testing.assert_allclose(krw, 0.25)

    def test_kr_endpoints(self):
        fluid = init_ad_fluid(phases='WO', n=[2, 2])
        krw_0, _ = fluid.krW(np.array([0.0]))
        krw_1, _ = fluid.krW(np.array([1.0]))
        np.testing.assert_allclose(krw_0, 0.0)
        np.testing.assert_allclose(krw_1, 1.0)

    def test_kr_with_smin(self):
        """With connate saturations, effective saturation is scaled."""
        fluid = init_ad_fluid(
            phases='WO', n=[2, 2], smin=[0.2, 0.1]
        )
        # Water range: [smin_w, 1 - smin_o] = [0.2, 0.9]
        # At sw=0.2: s_eff = 0 → kr = 0
        krw, _ = fluid.krW(np.array([0.2]))
        np.testing.assert_allclose(krw, 0.0, atol=1e-15)

        # At sw=0.9: s_eff = 1 → kr = 1
        krw, _ = fluid.krW(np.array([0.9]))
        np.testing.assert_allclose(krw, 1.0, atol=1e-15)

        # At sw=0.55: s_eff = (0.55-0.2)/(0.9-0.2) = 0.5 → kr = 0.25
        krw, _ = fluid.krW(np.array([0.55]))
        np.testing.assert_allclose(krw, 0.25, atol=1e-12)

    def test_kr_derivative_with_smin(self):
        """Derivative should account for saturation scaling."""
        fluid = init_ad_fluid(
            phases='WO', n=[2, 2], smin=[0.2, 0.1]
        )
        # dkr/ds = n * s_eff^(n-1) / (su - sl)
        # At sw=0.55: s_eff=0.5, dkr/ds = 2*0.5/0.7 ≈ 1.4286
        _, dkrw = fluid.krW(np.array([0.55]))
        expected = 2 * 0.5 / 0.7
        np.testing.assert_allclose(dkrw, expected, rtol=1e-10)

    def test_kr_below_smin_is_zero(self):
        fluid = init_ad_fluid(phases='WO', n=[2, 2], smin=[0.2, 0.1])
        krw, dkrw = fluid.krW(np.array([0.1]))  # below smin_w=0.2
        np.testing.assert_allclose(krw, 0.0, atol=1e-15)

    def test_three_phase_kr(self):
        fluid = init_ad_fluid(phases='WOG', n=[2, 2, 2])
        sw = np.array([0.5])
        krw, _ = fluid.krW(sw)
        np.testing.assert_allclose(krw, 0.25)

    def test_kr_derivative_finite_difference(self):
        fluid = init_ad_fluid(phases='WO', n=[3, 2], smin=[0.1, 0.05])
        s = np.array([0.3, 0.5, 0.7])
        h = 1e-7
        kr_p, _ = fluid.krW(s + h)
        kr_m, _ = fluid.krW(s - h)
        dkr_fd = (kr_p - kr_m) / (2 * h)
        _, dkr = fluid.krW(s)
        np.testing.assert_allclose(dkr, dkr_fd, atol=1e-5)


class TestADFluidPVMult:
    def test_no_compressibility(self):
        fluid = init_ad_fluid(phases='WO')
        p = np.array([1e5, 1e7])
        mult, dmult = fluid.pv_mult(p)
        np.testing.assert_allclose(mult, 1.0)
        np.testing.assert_allclose(dmult, 0.0)

    def test_with_compressibility(self):
        c_r = 5e-10
        p_ref = 200e5
        fluid = init_ad_fluid(phases='WO', c_r=c_r, p_ref=p_ref)
        p = np.array([200e5, 300e5, 400e5])
        mult, dmult = fluid.pv_mult(p)
        expected = 1.0 + c_r * (p - p_ref)
        np.testing.assert_allclose(mult, expected)
        np.testing.assert_allclose(dmult, c_r)

    def test_pv_mult_at_pref(self):
        fluid = init_ad_fluid(phases='WO', c_r=1e-9, p_ref=1e7)
        mult, _ = fluid.pv_mult(np.array([1e7]))
        np.testing.assert_allclose(mult, 1.0)
