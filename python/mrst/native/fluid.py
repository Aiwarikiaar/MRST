"""
Native Python fluid property data structures and functions.

Port of MRST's fluid initialization routines:
  - initSimpleFluid.m    -> init_simple_fluid(), SimpleFluid
  - initSimpleADIFluid.m -> init_ad_fluid(), ADFluid

Fluids are represented as dataclass instances with callable methods
for evaluating pressure- and saturation-dependent properties.
Methods return (value, derivative) tuples for future AD integration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Incompressible two-phase fluid
# ---------------------------------------------------------------------------

@dataclass
class SimpleFluid:
    """Incompressible two-phase fluid (water-oil) with Corey relperm.

    All properties are constant (no pressure dependence).
    Relative permeabilities use the Corey power-law model:
        krW(sw) = sw^n_w
        krO(so) = so^n_o    where so = 1 - sw

    Attributes
    ----------
    mu_w : float
        Water viscosity (Pa*s).
    mu_o : float
        Oil viscosity (Pa*s).
    rho_w : float
        Water surface density (kg/m^3).
    rho_o : float
        Oil surface density (kg/m^3).
    n_w : float
        Corey exponent for water relative permeability.
    n_o : float
        Corey exponent for oil relative permeability.
    """
    mu_w: float
    mu_o: float
    rho_w: float
    rho_o: float
    n_w: float
    n_o: float

    def kr_w(self, sw):
        """Water relative permeability and its derivative.

        Parameters
        ----------
        sw : array_like
            Water saturation.

        Returns
        -------
        kr : np.ndarray
            krW values.
        dkr : np.ndarray
            d(krW)/d(sw).
        """
        sw = np.asarray(sw, dtype=np.float64)
        kr = np.power(sw, self.n_w)
        dkr = self.n_w * np.power(sw, self.n_w - 1)
        return kr, dkr

    def kr_o(self, so):
        """Oil relative permeability and its derivative.

        Parameters
        ----------
        so : array_like
            Oil saturation.

        Returns
        -------
        kr : np.ndarray
            krO values.
        dkr : np.ndarray
            d(krO)/d(so).
        """
        so = np.asarray(so, dtype=np.float64)
        kr = np.power(so, self.n_o)
        dkr = self.n_o * np.power(so, self.n_o - 1)
        return kr, dkr

    def relperm(self, sw):
        """Evaluate both relative permeabilities from water saturation.

        Matches MRST's fluid.relperm(s) interface.

        Parameters
        ----------
        sw : array_like
            Water saturation, shape (n,).

        Returns
        -------
        kr : np.ndarray
            Shape (n, 2): columns [krW, krO].
        dkr : np.ndarray
            Shape (n, 4): [dkrW/dsw, 0, 0, dkrO/dsw].
            The off-diagonals are zero (no cross-coupling).
            Note: dkrO/dsw = -n_o * (1-sw)^(n_o-1) (chain rule).
        """
        sw = np.asarray(sw, dtype=np.float64).ravel()
        so = 1.0 - sw
        krw, dkrw = self.kr_w(sw)
        kro, dkro_dso = self.kr_o(so)
        n = len(sw)
        kr = np.column_stack([krw, kro])
        # dkrO/dsw = dkrO/dso * dso/dsw = -dkro_dso
        dkr = np.column_stack([dkrw, np.zeros(n), np.zeros(n), -dkro_dso])
        return kr, dkr

    def properties(self):
        """Return viscosities and densities.

        Returns
        -------
        mu : np.ndarray
            [mu_w, mu_o]
        rho : np.ndarray
            [rho_w, rho_o]
        """
        return (
            np.array([self.mu_w, self.mu_o]),
            np.array([self.rho_w, self.rho_o]),
        )

    def b_w(self, p):
        """Water inverse formation volume factor (always 1, incompressible)."""
        return np.ones_like(np.asarray(p, dtype=np.float64))

    def b_o(self, p):
        """Oil inverse formation volume factor (always 1, incompressible)."""
        return np.ones_like(np.asarray(p, dtype=np.float64))


def init_simple_fluid(mu, rho, n):
    """Create a simple incompressible two-phase fluid.

    Port of MRST's initSimpleFluid.

    Parameters
    ----------
    mu : array_like
        Phase viscosities [mu_w, mu_o] in Pa*s.
    rho : array_like
        Phase densities [rho_w, rho_o] in kg/m^3.
    n : array_like
        Corey exponents [n_w, n_o].

    Returns
    -------
    SimpleFluid
        Incompressible two-phase fluid.

    Examples
    --------
    >>> from mrst.utils.units import cp
    >>> fluid = init_simple_fluid(mu=[1*cp, 5*cp], rho=[1000, 700], n=[2, 2])
    >>> kr, dkr = fluid.relperm(np.array([0.5]))
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n = np.asarray(n, dtype=np.float64).ravel()

    if len(mu) != 2:
        raise ValueError(f"mu must have 2 entries, got {len(mu)}")
    if len(rho) != 2:
        raise ValueError(f"rho must have 2 entries, got {len(rho)}")
    if len(n) != 2:
        raise ValueError(f"n must have 2 entries, got {len(n)}")

    return SimpleFluid(
        mu_w=mu[0], mu_o=mu[1],
        rho_w=rho[0], rho_o=rho[1],
        n_w=n[0], n_o=n[1],
    )


# ---------------------------------------------------------------------------
# AD-compatible multi-phase compressible fluid
# ---------------------------------------------------------------------------

@dataclass
class ADFluid:
    """AD-compatible multi-phase fluid.

    Supports water (W), oil (O), gas (G) phases in any combination.
    Relative permeabilities use Corey model with saturation scaling.
    Formation volume factors support optional compressibility.

    Attributes
    ----------
    phases : str
        Phase string, e.g. 'WO', 'WOG'.
    rho : dict
        Surface densities keyed by phase letter (kg/m^3).
    mu : dict
        Viscosities keyed by phase letter (Pa*s).
    n : dict
        Corey exponents keyed by phase letter.
    b_ref : dict
        Reference inverse FVF keyed by phase letter.
    c : dict or None
        Compressibility per phase (1/Pa). None = incompressible.
    p_ref : float
        Reference pressure for compressibility (Pa).
    c_r : float or None
        Rock compressibility (1/Pa). None = incompressible rock.
    smin : dict
        Minimum (connate) saturations keyed by phase letter.
    """
    phases: str
    rho: dict
    mu: dict
    n: dict
    b_ref: dict
    c: Optional[dict]
    p_ref: float
    c_r: Optional[float]
    smin: dict

    @property
    def num_phases(self):
        """Number of active phases."""
        return len(self.phases)

    # --- Formation volume factors ---

    def b(self, phase, p):
        """Inverse formation volume factor for a phase.

        If compressibility c is set:  b(p) = b_ref * exp(c * (p - p_ref))
        Otherwise: b(p) = b_ref (constant).

        Parameters
        ----------
        phase : str
            Phase letter ('W', 'O', or 'G').
        p : array_like
            Pressure (Pa).

        Returns
        -------
        b_val : np.ndarray
            b(p) values.
        db_dp : np.ndarray
            Derivative db/dp.
        """
        p = np.asarray(p, dtype=np.float64)
        b0 = self.b_ref[phase]
        if self.c is None or self.c.get(phase, 0) == 0:
            return b0 * np.ones_like(p), np.zeros_like(p)
        ci = self.c[phase]
        val = b0 * np.exp(ci * (p - self.p_ref))
        dval = ci * val
        return val, dval

    def bW(self, p):
        """Water inverse FVF. Shorthand for b('W', p)."""
        return self.b('W', p)

    def bO(self, p):
        """Oil inverse FVF. Shorthand for b('O', p)."""
        return self.b('O', p)

    def bG(self, p):
        """Gas inverse FVF. Shorthand for b('G', p)."""
        return self.b('G', p)

    # --- Viscosity ---

    def viscosity(self, phase, p):
        """Viscosity for a phase (constant, pressure-independent).

        Returns
        -------
        mu_val : np.ndarray
            Viscosity values.
        dmu_dp : np.ndarray
            Zero (constant viscosity).
        """
        p = np.asarray(p, dtype=np.float64)
        return self.mu[phase] * np.ones_like(p), np.zeros_like(p)

    def muW(self, p):
        """Water viscosity. Shorthand for viscosity('W', p)."""
        return self.viscosity('W', p)

    def muO(self, p):
        """Oil viscosity. Shorthand for viscosity('O', p)."""
        return self.viscosity('O', p)

    def muG(self, p):
        """Gas viscosity. Shorthand for viscosity('G', p)."""
        return self.viscosity('G', p)

    # --- Relative permeability ---

    def kr(self, phase, s):
        """Relative permeability for a single phase with saturation scaling.

        Applies connate saturation scaling:
            s_eff = clip((s - smin) / (s_upper - smin), 0, 1)
            kr = s_eff ^ n

        where s_upper = 1 - sum(smin of other phases).

        Parameters
        ----------
        phase : str
            Phase letter.
        s : array_like
            Phase saturation.

        Returns
        -------
        kr_val : np.ndarray
            Relative permeability values.
        dkr_ds : np.ndarray
            Derivative dkr/ds.
        """
        s = np.asarray(s, dtype=np.float64)
        sl = self.smin[phase]
        su = 1.0 - sum(self.smin[ph] for ph in self.phases if ph != phase)
        ni = self.n[phase]

        if sl == 0 and su == 1.0:
            s_eff = np.clip(s, 0.0, 1.0)
            ds_eff = np.where((s >= 0) & (s <= 1.0), 1.0, 0.0)
        else:
            denom = su - sl
            s_eff = np.clip((s - sl) / denom, 0.0, 1.0)
            ds_eff = np.where((s > sl) & (s < su), 1.0 / denom, 0.0)

        kr_val = np.power(s_eff, ni)
        dkr_ds_eff = ni * np.power(s_eff, ni - 1)
        dkr_ds = dkr_ds_eff * ds_eff
        return kr_val, dkr_ds

    def krW(self, sw):
        """Water relative permeability. Shorthand for kr('W', sw)."""
        return self.kr('W', sw)

    def krO(self, so):
        """Oil relative permeability. Shorthand for kr('O', so)."""
        return self.kr('O', so)

    def krG(self, sg):
        """Gas relative permeability. Shorthand for kr('G', sg)."""
        return self.kr('G', sg)

    # --- Rock compressibility ---

    def pv_mult(self, p):
        """Pore volume multiplier from rock compressibility.

        pv_mult(p) = 1 + cR * (p - p_ref)

        Parameters
        ----------
        p : array_like
            Pressure (Pa).

        Returns
        -------
        mult : np.ndarray
            Pore volume multiplier.
        dmult_dp : np.ndarray
            Derivative d(mult)/dp = cR.
        """
        p = np.asarray(p, dtype=np.float64)
        if self.c_r is None or self.c_r == 0:
            return np.ones_like(p), np.zeros_like(p)
        mult = 1.0 + self.c_r * (p - self.p_ref)
        dmult = self.c_r * np.ones_like(p)
        return mult, dmult


def init_ad_fluid(
    phases='WO',
    mu=None,
    rho=None,
    n=None,
    b=None,
    c=None,
    p_ref=0.0,
    c_r=None,
    smin=None,
):
    """Create an AD-compatible multi-phase fluid.

    Port of MRST's initSimpleADIFluid.

    Parameters
    ----------
    phases : str
        Phase string, subset of 'WOG'. E.g., 'WO', 'WOG'.
    mu : array_like, optional
        Viscosities, one per phase (Pa*s). Default: [1.0]*nph.
    rho : array_like, optional
        Surface densities, one per phase (kg/m^3). Default: [1.0]*nph.
    n : array_like, optional
        Corey exponents, one per phase. Default: [1.0]*nph (linear).
    b : array_like, optional
        Reference inverse FVF per phase. Default: [1.0]*nph.
    c : array_like, optional
        Compressibility per phase (1/Pa). None = incompressible.
    p_ref : float, optional
        Reference pressure (Pa). Default: 0.
    c_r : float, optional
        Rock compressibility (1/Pa). None = incompressible rock.
    smin : array_like, optional
        Connate saturations per phase. Default: [0]*nph.

    Returns
    -------
    ADFluid
        AD-compatible fluid structure.

    Examples
    --------
    >>> from mrst.utils.units import cp, barsa
    >>> fluid = init_ad_fluid(phases='WO', mu=[1*cp, 5*cp],
    ...                       rho=[1000, 700], n=[2, 2])
    """
    phases = phases.upper()
    nph = len(phases)
    valid = set('WOG')
    if not all(ch in valid for ch in phases):
        raise ValueError(f"phases must be subset of 'WOG', got '{phases}'")
    if len(set(phases)) != nph:
        raise ValueError(f"Duplicate phases in '{phases}'")

    # Defaults
    if mu is None:
        mu = [1.0] * nph
    if rho is None:
        rho = [1.0] * nph
    if n is None:
        n = [1.0] * nph
    if b is None:
        b = [1.0] * nph
    if smin is None:
        smin = [0.0] * nph

    mu_arr = np.asarray(mu, dtype=np.float64).ravel()
    rho_arr = np.asarray(rho, dtype=np.float64).ravel()
    n_arr = np.asarray(n, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    smin_arr = np.asarray(smin, dtype=np.float64).ravel()

    for name, arr in [('mu', mu_arr), ('rho', rho_arr), ('n', n_arr),
                      ('b', b_arr), ('smin', smin_arr)]:
        if len(arr) != nph:
            raise ValueError(
                f"{name} must have {nph} entries for phases '{phases}', "
                f"got {len(arr)}"
            )

    if smin_arr.sum() >= 1.0:
        raise ValueError(
            f"Sum of connate saturations must be < 1, got {smin_arr.sum()}"
        )

    c_dict = None
    if c is not None:
        c_arr = np.asarray(c, dtype=np.float64).ravel()
        if len(c_arr) != nph:
            raise ValueError(
                f"c must have {nph} entries, got {len(c_arr)}"
            )
        c_dict = {ph: float(c_arr[i]) for i, ph in enumerate(phases)}

    return ADFluid(
        phases=phases,
        rho={ph: float(rho_arr[i]) for i, ph in enumerate(phases)},
        mu={ph: float(mu_arr[i]) for i, ph in enumerate(phases)},
        n={ph: float(n_arr[i]) for i, ph in enumerate(phases)},
        b_ref={ph: float(b_arr[i]) for i, ph in enumerate(phases)},
        c=c_dict,
        p_ref=float(p_ref),
        c_r=float(c_r) if c_r is not None else None,
        smin={ph: float(smin_arr[i]) for i, ph in enumerate(phases)},
    )
